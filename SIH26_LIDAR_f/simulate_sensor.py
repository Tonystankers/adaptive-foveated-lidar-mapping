"""
simulate_sensor.py - Stand-in "Real Sensor" Emitter
SIH26053: Adaptive Variable-Resolution 2.5D LiDAR Mapping Pipeline

Run this in its OWN terminal, separate from the dashboard, to emit UDP
frames exactly like a real sensor bridge would:

    python simulate_sensor.py --host 127.0.0.1 --port 5599 --fps 10

Then, in the dashboard sidebar, choose "Live Hardware Sensor (UDP)" as the
data source and click Connect. You now have a genuinely live, continuously
updating pipeline: this process is a separate program you cannot see inside
Streamlit, pushing frames over the network the same way a real sensor
process would.

Swap the body of `build_frame()` for a call into your actual sensor SDK /
ROS subscriber and every other line in this file (and the dashboard) stays
identical.
"""

from __future__ import annotations

import argparse
import json
import socket
import time

from data_loader import SyntheticLidarGenerator


def build_frame(generator: SyntheticLidarGenerator, frame_id: int) -> dict:
    frame = generator.generate_frame(frame_id=frame_id, timestamp=frame_id * 0.1)
    return {
        "frame_id": frame.frame_id,
        "timestamp": frame.timestamp,
        "points": frame.points.tolist(),
        "labels": frame.labels.tolist(),
        "intensities": frame.intensities.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a real-time LiDAR sensor over UDP.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5599)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    generator = SyntheticLidarGenerator(seed=args.seed)

    print(f"[simulate_sensor] Streaming synthetic frames to {args.host}:{args.port} @ {args.fps} FPS")
    print("[simulate_sensor] Ctrl+C to stop.")

    frame_id = 0
    period = 1.0 / max(args.fps, 0.1)
    try:
        while True:
            msg = build_frame(generator, frame_id)
            payload = json.dumps(msg).encode("utf-8")
            if len(payload) > 60000:
                # Downsample so we stay under the UDP datagram limit - a real
                # bridge sending dense scans should use TCP/length-prefixed
                # framing instead (see live_source.py docstring).
                step = (len(payload) // 60000) + 1
                msg["points"] = msg["points"][::step]
                msg["labels"] = msg["labels"][::step]
                msg["intensities"] = msg["intensities"][::step]
                payload = json.dumps(msg).encode("utf-8")
            sock.sendto(payload, (args.host, args.port))
            frame_id += 1
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[simulate_sensor] Stopped.")


if __name__ == "__main__":
    main()
