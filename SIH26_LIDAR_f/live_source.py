"""
live_source.py - Real-Time Hardware LiDAR Ingestion Layer
SIH26053: Adaptive Variable-Resolution 2.5D LiDAR Mapping Pipeline

Purpose
-------
Everything else in this pipeline (segmentation_engine.py, grid_engine.py,
visualizer.py, dashboard.py) consumes a single, simple object: `LidarFrame`
from data_loader.py. This module is the ONLY place that needs to change to
go from "synthetic demo data" to "an actual physical sensor bolted to a
vehicle/robot."

It defines:
    - LiveLidarSource:  abstract base class for any real-time source.
    - UDPLidarSource:   a working, driver-agnostic UDP receiver. Any process
                         (a ROS node, a vendor SDK wrapper, a microcontroller)
                         that can send a small JSON packet over UDP can feed
                         this dashboard live, with zero code changes on the
                         dashboard side.
    - ROSLidarSource:   a thin, documented stub showing exactly where to plug
                         in `rclpy` + `sensor_msgs/PointCloud2` if/when the
                         team wires up a real ROS-based sensor stack. Not
                         active unless `rclpy` is installed, so it never
                         breaks the rest of the app.

Wiring a REAL sensor (Velodyne / Ouster / Livox / RPLidar / ROS bag, etc.)
---------------------------------------------------------------------------
1. Keep whatever vendor SDK or ROS node you already use to read the sensor.
2. In that process, for every scan, package it as:
       {
         "frame_id": <int>,
         "timestamp": <float, seconds>,
         "points": [[x, y, z], ...],       # sensor frame, meters
         "intensities": [i0, i1, ...],     # optional, 0.0-1.0
         "labels": [c0, c1, ...]           # optional, 0=Noise 1=Ground
       }                                    #          2=Static 3=Dynamic
   and UDP-send it (see `simulate_sensor.py` for a full working example)
   to the host/port shown in the dashboard's "Live Hardware Sensor" panel.
3. That's it. `UDPLidarSource` reassembles it into a `LidarFrame` and the
   existing segmentation + adaptive grid + visualization pipeline runs on
   it exactly as it does on synthetic data - because it's the same object.

If your sensor already publishes ROS2 `sensor_msgs/PointCloud2`, use
`ROSLidarSource` instead (requires `pip install rclpy` + a sourced ROS2
environment) rather than adding a UDP hop.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_loader import LidarFrame

# Max UDP datagram payload we accept. A single dense scan easily exceeds the
# ~64KB practical UDP limit, so real integrations should either (a) downsample
# before sending, (b) send over a length-prefixed TCP stream instead, or
# (c) chunk + reassemble. This reference implementation covers (a)/(b)-scale
# demo traffic; swap `socket.SOCK_DGRAM` for `SOCK_STREAM` for full-density
# production point clouds.
MAX_UDP_PAYLOAD = 65536


@dataclass
class ConnectionStatus:
    connected: bool = False
    frames_received: int = 0
    last_frame_time: float = 0.0
    incoming_rate_hz: float = 0.0
    last_error: str = ""


class LiveLidarSource(ABC):
    """Abstract interface every real-time sensor bridge must implement."""

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def get_latest_frame(self) -> Optional[LidarFrame]:
        """Returns the most recently received frame, or None if none yet."""
        ...

    @abstractmethod
    def status(self) -> ConnectionStatus:
        ...


class UDPLidarSource(LiveLidarSource):
    """
    Driver-agnostic real-time LiDAR ingestion over UDP.

    Any external process - a ROS node, a vendor SDK bridge, a test rig - can
    stream frames into the dashboard by sending UTF-8 JSON datagrams shaped
    like the schema documented at the top of this file.

    This class only ever holds the SINGLE most recent frame (a "latest value"
    buffer, not a queue), which is the correct behavior for a real-time
    dashboard: if the UI is momentarily slower than the sensor, we want the
    newest picture of the world, not a backlog of stale ones.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5599, timeout_s: float = 0.5):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._latest_frame: Optional[LidarFrame] = None
        self._status = ConnectionStatus()
        self._recv_timestamps: list[float] = []

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return  # already running
        self._stop_event.clear()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(self.timeout_s)
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close()
        self._thread = None
        self._sock = None
        with self._lock:
            self._status.connected = False

    # ------------------------------------------------------------------
    def _listen_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload, _addr = self._sock.recvfrom(MAX_UDP_PAYLOAD)
            except socket.timeout:
                with self._lock:
                    # No packet within timeout window -> mark disconnected
                    # if we haven't heard anything for a few consecutive misses.
                    if time.time() - self._status.last_frame_time > max(2.0, self.timeout_s * 4):
                        self._status.connected = False
                continue
            except OSError as exc:
                with self._lock:
                    self._status.last_error = str(exc)
                continue

            try:
                frame = self._parse_packet(payload)
            except Exception as exc:  # noqa: BLE001 - never let a bad packet kill the thread
                with self._lock:
                    self._status.last_error = f"Malformed packet: {exc}"
                continue

            now = time.time()
            with self._lock:
                self._latest_frame = frame
                self._status.connected = True
                self._status.frames_received += 1
                self._status.last_frame_time = now
                self._status.last_error = ""
                self._recv_timestamps.append(now)
                self._recv_timestamps = [t for t in self._recv_timestamps if now - t <= 2.0]
                if len(self._recv_timestamps) >= 2:
                    span = self._recv_timestamps[-1] - self._recv_timestamps[0]
                    self._status.incoming_rate_hz = (
                        (len(self._recv_timestamps) - 1) / span if span > 0 else 0.0
                    )

    @staticmethod
    def _parse_packet(payload: bytes) -> LidarFrame:
        msg = json.loads(payload.decode("utf-8"))
        points = np.asarray(msg["points"], dtype=np.float32).reshape(-1, 3)
        n = len(points)

        labels = msg.get("labels")
        labels = np.asarray(labels, dtype=np.uint8) if labels is not None else np.zeros(n, dtype=np.uint8)

        intensities = msg.get("intensities")
        intensities = (
            np.asarray(intensities, dtype=np.float32) if intensities is not None else np.full(n, 0.5, dtype=np.float32)
        )

        return LidarFrame(
            points=points,
            labels=labels,
            intensities=intensities,
            frame_id=int(msg.get("frame_id", 0)),
            timestamp=float(msg.get("timestamp", time.time())),
        )

    # ------------------------------------------------------------------
    def get_latest_frame(self) -> Optional[LidarFrame]:
        with self._lock:
            return self._latest_frame

    def status(self) -> ConnectionStatus:
        with self._lock:
            return ConnectionStatus(**vars(self._status))


class ROSLidarSource(LiveLidarSource):
    """
    Stub bridge for a real ROS2 `sensor_msgs/PointCloud2` topic.

    Left intentionally minimal: rclpy is a heavyweight, environment-specific
    dependency (needs a sourced ROS2 install) that we don't assume is present
    in every dev/judge-day environment. Install rclpy and fill in `_on_cloud`
    to go live with an actual ROS-based sensor stack without touching any
    other file in this project.
    """

    def __init__(self, topic: str = "/points_raw"):
        self.topic = topic
        self._latest_frame: Optional[LidarFrame] = None
        self._status = ConnectionStatus()
        self._node = None

        try:
            import rclpy  # noqa: F401
            self._rclpy_available = True
        except ImportError:
            self._rclpy_available = False

    def start(self) -> None:
        if not self._rclpy_available:
            self._status.last_error = "rclpy not installed - pip install rclpy in a sourced ROS2 env"
            return
        # --- Real wiring goes here, e.g.: -----------------------------
        # import rclpy
        # from sensor_msgs.msg import PointCloud2
        # import sensor_msgs_py.point_cloud2 as pc2
        #
        # rclpy.init()
        # self._node = rclpy.create_node("sih26053_lidar_bridge")
        # self._node.create_subscription(PointCloud2, self.topic, self._on_cloud, 10)
        # threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True).start()
        # ----------------------------------------------------------------
        raise NotImplementedError("Fill in the ROS2 subscription above for your sensor stack.")

    def _on_cloud(self, msg) -> None:  # pragma: no cover - wired in by integrator
        # Convert msg (sensor_msgs/PointCloud2) -> LidarFrame here.
        raise NotImplementedError

    def stop(self) -> None:
        if self._node is not None:
            self._node.destroy_node()

    def get_latest_frame(self) -> Optional[LidarFrame]:
        return self._latest_frame

    def status(self) -> ConnectionStatus:
        return self._status
