"""
data_loader.py - Autonomous Systems LiDAR Data Ingestion & Synthetic Generator
SIH26053: Adaptive Variable Resolution 2.5D LiDAR Mapping Pipeline

Classes & Enums:
    - SemanticClass: Canonical 4-class taxonomy (Noise, Ground, Static, Dynamic).
    - LidarFrame: In-memory dataclass for a single timestamp LiDAR point cloud.
    - SemanticKittiMapper: SemanticKITTI 32-bit to 4-class lookup table mapper.
    - SyntheticLidarGenerator: Physically grounded synthetic LiDAR point cloud generator.
    - SyntheticLidarDataset: Multi-frame temporal synthetic dataset iterator.
    - SemanticKittiDataset: Directory reader for real SemanticKITTI .bin and .label files.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# ==============================================================================
# 1. CANONICAL 4-CLASS TAXONOMY
# ==============================================================================

class SemanticClass(IntEnum):
    """
    Fine-grained 7-class semantic taxonomy optimized for 2.5D grid mapping,
    navigation costmaps, and real-time obstacle avoidance. Splits the old
    coarse STATIC_OBSTACLE / DYNAMIC_OBJECT buckets into concrete object
    categories (walls/buildings, poles, vegetation, vehicles, pedestrians)
    so the model performs genuine object *classification*, not just a
    binary static/dynamic split.
    """
    NOISE = 0                  # Unlabeled, laser dropouts, rain/dust, outlier points
    GROUND = 1                 # Drivable road, sidewalks, parking lots, terrain
    STATIC_STRUCTURE = 2       # Buildings, walls, fences, other large flat structures
    STATIC_POLE = 3            # Poles, traffic signs, lamp posts, thin vertical furniture
    STATIC_VEGETATION = 4      # Trees, trunks, canopies, foliage
    DYNAMIC_VEHICLE = 5        # Cars, trucks, buses, motorcycles, bicycles
    DYNAMIC_PEDESTRIAN = 6     # Pedestrians / cyclists on foot

    @classmethod
    def get_color_map(cls) -> Dict[int, Tuple[float, float, float]]:
        """Normalized RGB colors for visualization."""
        return {
            cls.NOISE: (0.50, 0.50, 0.50),               # Gray
            cls.GROUND: (0.18, 0.80, 0.44),               # Green
            cls.STATIC_STRUCTURE: (0.91, 0.30, 0.24),     # Red (walls/buildings)
            cls.STATIC_POLE: (0.61, 0.35, 0.71),          # Purple (poles)
            cls.STATIC_VEGETATION: (0.42, 0.65, 0.24),    # Olive green (vegetation)
            cls.DYNAMIC_VEHICLE: (0.20, 0.60, 0.86),      # Blue (vehicles)
            cls.DYNAMIC_PEDESTRIAN: (0.95, 0.77, 0.06),   # Amber (pedestrians)
        }

    @classmethod
    def get_hex_color_map(cls) -> Dict[int, str]:
        """Hex colors for matplotlib/plotly/streamlit rendering."""
        return {
            cls.NOISE: "#7f8c8d",
            cls.GROUND: "#2ecc71",
            cls.STATIC_STRUCTURE: "#e74c3c",
            cls.STATIC_POLE: "#9b59b6",
            cls.STATIC_VEGETATION: "#6ab04c",
            cls.DYNAMIC_VEHICLE: "#3498db",
            cls.DYNAMIC_PEDESTRIAN: "#f1c40f",
        }

    @classmethod
    def get_name(cls, class_id: int) -> str:
        names = {
            0: "Noise",
            1: "Ground",
            2: "Static Structure",
            3: "Static Pole",
            4: "Static Vegetation",
            5: "Dynamic Vehicle",
            6: "Dynamic Pedestrian",
        }
        return names.get(int(class_id), "Unknown")

    @classmethod
    def num_classes(cls) -> int:
        return len(cls)

    @classmethod
    def is_drivable(cls, class_id: int) -> bool:
        """Terrain-analysis helper: True for drivable/ground surface."""
        return int(class_id) == int(cls.GROUND)

    @classmethod
    def is_static_obstacle(cls, class_id: int) -> bool:
        return int(class_id) in (int(cls.STATIC_STRUCTURE), int(cls.STATIC_POLE), int(cls.STATIC_VEGETATION))

    @classmethod
    def is_dynamic_object(cls, class_id: int) -> bool:
        return int(class_id) in (int(cls.DYNAMIC_VEHICLE), int(cls.DYNAMIC_PEDESTRIAN))

    @classmethod
    def coarse_group(cls, class_id: int) -> str:
        """Maps a fine-grained class back to the coarse {Ground, Static, Dynamic, Noise} bucket."""
        cid = int(class_id)
        if cid == int(cls.NOISE):
            return "Noise"
        if cid == int(cls.GROUND):
            return "Ground"
        if cls.is_static_obstacle(cid):
            return "Static Obstacle"
        return "Dynamic Object"


# ==============================================================================
# 2. LIDAR FRAME DATA CONTAINER
# ==============================================================================

@dataclass
class LidarFrame:
    """
    Represents a single LiDAR scan in sensor coordinate frame (Right-Handed):
      - X: Forward (meters)
      - Y: Left (meters)
      - Z: Up (meters)
    """
    points: np.ndarray        # Shape (N, 3), float32: [x, y, z]
    labels: np.ndarray        # Shape (N,), uint8: 4-class semantic labels {0, 1, 2, 3}
    intensities: np.ndarray   # Shape (N,), float32: Normalized reflectance [0.0, 1.0]
    frame_id: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        # Validate shapes and enforce contiguous float32 / uint8 types
        self.points = np.ascontiguousarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {self.points.shape}")

        n_points = len(self.points)

        if self.labels is None or len(self.labels) == 0:
            self.labels = np.zeros(n_points, dtype=np.uint8)
        else:
            self.labels = np.ascontiguousarray(self.labels, dtype=np.uint8)

        if self.intensities is None or len(self.intensities) == 0:
            self.intensities = np.full(n_points, 0.5, dtype=np.float32)
        else:
            self.intensities = np.ascontiguousarray(self.intensities, dtype=np.float32)

        if len(self.labels) != n_points:
            raise ValueError(f"labels length ({len(self.labels)}) != points count ({n_points})")
        if len(self.intensities) != n_points:
            raise ValueError(f"intensities length ({len(self.intensities)}) != points count ({n_points})")

    @property
    def num_points(self) -> int:
        return len(self.points)

    def to_xyzi(self) -> np.ndarray:
        """Returns (N, 4) array [x, y, z, intensity]."""
        return np.column_stack([self.points, self.intensities])

    def to_xyzl(self) -> np.ndarray:
        """Returns (N, 4) array [x, y, z, label]."""
        return np.column_stack([self.points, self.labels.astype(np.float32)])

    def filter_roi(
        self,
        x_range: Tuple[float, float] = (-70.0, 70.0),
        y_range: Tuple[float, float] = (-70.0, 70.0),
        z_range: Tuple[float, float] = (-3.0, 10.0),
    ) -> LidarFrame:
        """Crops point cloud within a 3D bounding box."""
        mask = (
            (self.points[:, 0] >= x_range[0]) & (self.points[:, 0] <= x_range[1]) &
            (self.points[:, 1] >= y_range[0]) & (self.points[:, 1] <= y_range[1]) &
            (self.points[:, 2] >= z_range[0]) & (self.points[:, 2] <= z_range[1])
        )
        return LidarFrame(
            points=self.points[mask],
            labels=self.labels[mask],
            intensities=self.intensities[mask],
            frame_id=self.frame_id,
            timestamp=self.timestamp,
        )

    def filter_range(self, min_dist: float = 0.5, max_dist: float = 75.0) -> LidarFrame:
        """Filters points by Euclidean radial distance from sensor origin."""
        dist_sq = self.points[:, 0]**2 + self.points[:, 1]**2 + self.points[:, 2]**2
        mask = (dist_sq >= min_dist**2) & (dist_sq <= max_dist**2)
        return LidarFrame(
            points=self.points[mask],
            labels=self.labels[mask],
            intensities=self.intensities[mask],
            frame_id=self.frame_id,
            timestamp=self.timestamp,
        )

    def get_class_counts(self) -> Dict[str, int]:
        """Returns frequency count of each semantic class in the frame."""
        counts = {}
        for sc in SemanticClass:
            count = int(np.sum(self.labels == sc.value))
            counts[sc.name] = count
        return counts

    def save_to_kitti_format(self, bin_path: str, label_path: Optional[str] = None) -> None:
        """Exports frame into SemanticKITTI binary .bin and .label formats."""
        os.makedirs(os.path.dirname(os.path.abspath(bin_path)), exist_ok=True)
        # KITTI .bin: Nx4 float32 (x, y, z, intensity)
        scan = np.column_stack([self.points, self.intensities]).astype(np.float32)
        scan.tofile(bin_path)

        if label_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(label_path)), exist_ok=True)
            # KITTI .label: uint32 (lower 16 bits = semantic class, upper 16 bits = instance ID)
            labels_uint32 = self.labels.astype(np.uint32)
            labels_uint32.tofile(label_path)


# ==============================================================================
# 3. SEMANTICKITTI LABEL MAPPER
# ==============================================================================

class SemanticKittiMapper:
    """
    High-performance lookup-table mapping raw 16-bit SemanticKITTI labels
    to our canonical 4-class taxonomy:
      0: Noise / Outlier
      1: Ground
      2: Static Obstacle
      3: Dynamic Object
    """
    # Direct mapping definition: raw SemanticKITTI id -> our 7-class taxonomy
    # {0: Noise, 1: Ground, 2: Static Structure, 3: Static Pole,
    #  4: Static Vegetation, 5: Dynamic Vehicle, 6: Dynamic Pedestrian}
    RAW_TO_SIMPLIFIED: Dict[int, int] = {
        0: 0,    # unlabeled -> Noise
        1: 0,    # outlier -> Noise
        10: 5,   # car -> Vehicle
        11: 5,   # bicycle -> Vehicle
        13: 5,   # bus -> Vehicle
        15: 5,   # motorcycle -> Vehicle
        16: 5,   # on-rails -> Vehicle
        18: 5,   # truck -> Vehicle
        20: 5,   # other-vehicle -> Vehicle
        30: 6,   # person -> Pedestrian
        31: 6,   # bicyclist -> Pedestrian
        32: 6,   # motorcyclist -> Pedestrian
        40: 1,   # road -> Ground
        44: 1,   # parking -> Ground
        48: 1,   # sidewalk -> Ground
        49: 1,   # other-ground -> Ground
        50: 2,   # building -> Static Structure
        51: 2,   # fence -> Static Structure
        52: 2,   # other-structure -> Static Structure
        60: 1,   # lane-marking -> Ground
        70: 4,   # vegetation -> Static Vegetation
        71: 4,   # trunk -> Static Vegetation
        72: 1,   # terrain -> Ground
        80: 3,   # pole -> Static Pole
        81: 3,   # traffic-sign -> Static Pole
        99: 2,   # other-object -> Static Structure
        252: 5,  # moving-car -> Vehicle
        253: 6,  # moving-bicyclist -> Pedestrian
        254: 6,  # moving-person -> Pedestrian
        255: 6,  # moving-motorcyclist -> Pedestrian
        256: 5,  # moving-on-rails -> Vehicle
        257: 5,  # moving-bus -> Vehicle
        258: 5,  # moving-truck -> Vehicle
        259: 5,  # moving-other-vehicle -> Vehicle
    }

    _lut: np.ndarray = np.zeros(65536, dtype=np.uint8)

    @classmethod
    def initialize_lut(cls) -> None:
        """Precomputes O(1) vectorized NumPy lookup table for all 16-bit integers."""
        cls._lut.fill(0)
        for raw_id, target_id in cls.RAW_TO_SIMPLIFIED.items():
            if 0 <= raw_id < 65536:
                cls._lut[raw_id] = target_id

    @classmethod
    def map_labels(cls, raw_kitti_labels: np.ndarray) -> np.ndarray:
        """
        Takes raw 32-bit or 16-bit SemanticKITTI labels and maps to {0, 1, 2, 3}.
        Vectorized O(1) memory lookup.
        """
        # Ensure LUT is initialized
        if cls._lut[40] != 1:  # Check if road is mapped
            cls.initialize_lut()

        # SemanticKITTI packs semantic class in lower 16 bits
        semantic_raw = (raw_kitti_labels & 0xFFFF).astype(np.int64)
        return cls._lut[semantic_raw]


# Initialize lookup table on module import
SemanticKittiMapper.initialize_lut()


# ==============================================================================
# 4. SYNTHETIC LIDAR DATA GENERATOR
# ==============================================================================

class SyntheticLidarGenerator:
    """
    Generates realistic, geometrically accurate 3D LiDAR point clouds
    for autonomous driving scenarios without requiring external datasets.

    Simulates:
      - Multi-beam LiDAR (e.g. 64 channels, -25° to +15° vertical FOV)
      - Ego-vehicle at origin (0, 0, sensor_height=1.73m)
      - Asphalt road, curbs, raised sidewalks, and undulating terrain
      - Buildings, fences, tree trunks, and foliage
      - Dynamic vehicles and pedestrians with trajectory offsets
      - Material-dependent LiDAR reflectivity (intensities) and sensor noise
    """

    def __init__(
        self,
        sensor_height: float = 1.73,
        num_beams: int = 64,
        fov_up: float = 15.0,
        fov_down: float = -25.0,
        max_range: float = 75.0,
        seed: Optional[int] = 42,
    ):
        self.sensor_height = sensor_height
        self.num_beams = num_beams
        self.fov_up = np.radians(fov_up)
        self.fov_down = np.radians(fov_down)
        self.max_range = max_range
        self.rng = np.random.default_rng(seed)

    def generate_frame(
        self,
        frame_id: int = 0,
        timestamp: float = 0.0,
        num_cars: int = 8,
        num_pedestrians: int = 5,
        num_trees: int = 14,
        num_buildings: int = 6,
        noise_ratio: float = 0.02,
        road_width: float = 8.0,
        scene_length: float = 120.0,
    ) -> LidarFrame:
        """
        Synthesizes a complete LiDAR frame with dynamic agents evolving over time.
        """
        points_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        intensities_list: List[np.ndarray] = []

        # ----------------------------------------------------------------------
        # 1. Ground Surface (Road + Sidewalks + Undulating Terrain)
        # ----------------------------------------------------------------------
        # Concentric beam-based ground ray sampling for realistic scan-ring geometry
        beam_angles = np.linspace(self.fov_down, np.radians(-1.0), 32)
        ground_pts = []
        ground_intensities = []

        for angle in beam_angles:
            # Ray distance to ground plane: d = sensor_height / -tan(angle)
            if np.abs(np.tan(angle)) > 1e-4:
                d_ground = self.sensor_height / -np.tan(angle)
                if 1.0 <= d_ground <= self.max_range:
                    num_azimuth = int(np.clip(360 * (d_ground / 10.0), 180, 1024))
                    azimuths = np.linspace(0, 2 * np.pi, num_azimuth, endpoint=False)
                    # Add slight azimuth jitter
                    azimuths += self.rng.normal(0, 0.002, size=num_azimuth)

                    r = d_ground * np.cos(angle)
                    gx = r * np.cos(azimuths)
                    gy = r * np.sin(azimuths)
                    gz = -self.sensor_height + self.rng.normal(0, 0.015, size=num_azimuth)

                    # Lateral classification: Road vs Sidewalk vs Terrain
                    for i in range(num_azimuth):
                        y_val = gy[i]
                        x_val = gx[i]

                        # Ground slope / terrain undulation further out
                        if np.abs(y_val) > (road_width / 2.0 + 3.0):
                            # Terrain rolling hills
                            elevation = 0.4 * np.sin(x_val * 0.1) + 0.3 * np.cos(y_val * 0.15)
                            gz[i] += elevation
                            # Terrain intensity
                            intensity = self.rng.uniform(0.15, 0.35)
                        elif (road_width / 2.0) < np.abs(y_val) <= (road_width / 2.0 + 3.0):
                            # Sidewalk curb (raised +0.15m)
                            gz[i] += 0.15
                            intensity = self.rng.uniform(0.35, 0.55)
                        else:
                            # Main asphalt road
                            intensity = self.rng.uniform(0.1, 0.25)
                            # Lane markers have high reflectivity
                            if np.abs(y_val) < 0.15 or np.abs(np.abs(y_val) - (road_width / 4.0)) < 0.12:
                                if int(x_val) % 4 < 2:  # Dashed line
                                    intensity = self.rng.uniform(0.75, 0.95)

                        ground_intensities.append(intensity)

                    pts = np.column_stack([gx, gy, gz])
                    ground_pts.append(pts)

        if ground_pts:
            all_ground = np.vstack(ground_pts)
            points_list.append(all_ground)
            labels_list.append(np.full(len(all_ground), SemanticClass.GROUND, dtype=np.uint8))
            intensities_list.append(np.array(ground_intensities, dtype=np.float32))

        # ----------------------------------------------------------------------
        # 2. Static Obstacles (Buildings, Tree Trunks, Canopies, Poles)
        # ----------------------------------------------------------------------
        # A. Buildings along the perimeter
        bldg_x_locs = np.linspace(-scene_length / 2 + 10, scene_length / 2 - 10, num_buildings)
        for bx in bldg_x_locs:
            for side in [-1, 1]:  # Left and Right sides
                by = side * (road_width / 2.0 + self.rng.uniform(8.0, 14.0))
                b_width = self.rng.uniform(12.0, 18.0)
                b_height = self.rng.uniform(6.0, 12.0)
                b_depth = 10.0

                # Surface points on building facade facing the road
                n_bldg_pts = int(self.rng.integers(300, 700))
                px = self.rng.uniform(bx - b_width / 2, bx + b_width / 2, n_bldg_pts)
                pz = self.rng.uniform(-self.sensor_height, b_height - self.sensor_height, n_bldg_pts)
                py = np.full(n_bldg_pts, by) + self.rng.normal(0, 0.05, n_bldg_pts)

                bldg_pts = np.column_stack([px, py, pz])
                points_list.append(bldg_pts)
                labels_list.append(np.full(n_bldg_pts, SemanticClass.STATIC_STRUCTURE, dtype=np.uint8))
                intensities_list.append(self.rng.uniform(0.4, 0.7, n_bldg_pts).astype(np.float32))

        # B. Trees (Cylinder Trunk + Spherical Canopy)
        tree_x_locs = np.linspace(-scene_length / 2 + 5, scene_length / 2 - 5, num_trees)
        for tx in tree_x_locs:
            for side in [-1, 1]:
                ty = side * (road_width / 2.0 + self.rng.uniform(2.5, 5.5))
                # Trunk
                n_trunk = 80
                t_z = self.rng.uniform(-self.sensor_height, 1.2, n_trunk)
                t_theta = self.rng.uniform(0, 2 * np.pi, n_trunk)
                t_r = 0.2
                t_x = tx + t_r * np.cos(t_theta)
                t_y = ty + t_r * np.sin(t_theta)
                trunk_pts = np.column_stack([t_x, t_y, t_z])

                # Foliage / Canopy
                n_foliage = 220
                canopy_center_z = 2.5
                c_phi = self.rng.uniform(0, np.pi, n_foliage)
                c_theta = self.rng.uniform(0, 2 * np.pi, n_foliage)
                c_radius = self.rng.uniform(0.8, 2.2, n_foliage)

                c_x = tx + c_radius * np.sin(c_phi) * np.cos(c_theta)
                c_y = ty + c_radius * np.sin(c_phi) * np.sin(c_theta)
                c_z = canopy_center_z + c_radius * np.cos(c_phi)
                canopy_pts = np.column_stack([c_x, c_y, c_z])

                tree_pts = np.vstack([trunk_pts, canopy_pts])
                points_list.append(tree_pts)
                labels_list.append(np.full(len(tree_pts), SemanticClass.STATIC_VEGETATION, dtype=np.uint8))
                intensities_list.append(self.rng.uniform(0.2, 0.5, len(tree_pts)).astype(np.float32))

        # C. Street Lights & Utility Poles
        pole_x_locs = np.linspace(-40, 40, 6)
        for px_loc in pole_x_locs:
            for side in [-1, 1]:
                py_loc = side * (road_width / 2.0 + 1.2)
                n_pole = 90
                pz_pole = self.rng.uniform(-self.sensor_height, 4.5, n_pole)
                ptheta = self.rng.uniform(0, 2 * np.pi, n_pole)
                px_p = px_loc + 0.08 * np.cos(ptheta)
                py_p = py_loc + 0.08 * np.sin(ptheta)
                pole_pts = np.column_stack([px_p, py_p, pz_pole])

                points_list.append(pole_pts)
                labels_list.append(np.full(n_pole, SemanticClass.STATIC_POLE, dtype=np.uint8))
                intensities_list.append(self.rng.uniform(0.6, 0.85, n_pole).astype(np.float32))

        # ----------------------------------------------------------------------
        # 3. Dynamic Objects (Vehicles, Pedestrians)
        # ----------------------------------------------------------------------
        # A. Vehicles traveling along lanes
        # Ego lane is right lane (y = -road_width/4), oncoming is left lane (y = +road_width/4)
        car_speeds = [-12.0, 15.0, -10.0, 18.0, -14.0, 8.0, -9.0, 11.0]
        car_base_x = [-45.0, -25.0, -10.0, 8.0, 22.0, 38.0, 52.0, -60.0]
        car_lanes = [1, -1, 1, -1, 1, -1, 1, -1]  # 1 = left lane, -1 = right lane

        for i in range(min(num_cars, len(car_base_x))):
            lane_dir = car_lanes[i]
            lane_y = lane_dir * (road_width / 4.0) + self.rng.normal(0, 0.15)
            # Temporal displacement based on timestamp
            car_x = car_base_x[i] + (car_speeds[i] * timestamp)
            # Wrap around scene boundary
            car_x = ((car_x + scene_length / 2) % scene_length) - scene_length / 2

            # Car 3D Box Dimensions: Length ~4.6m, Width ~1.9m, Height ~1.5m
            c_len, c_wid, c_hgt = 4.6, 1.9, 1.5
            n_car_pts = int(self.rng.integers(250, 600))

            # Distribute points on exterior surfaces (roof, sides, hood)
            sub_pts = []
            # Sides
            n_side = n_car_pts // 3
            sx = self.rng.uniform(-c_len / 2, c_len / 2, n_side)
            sy = self.rng.choice([-c_wid / 2, c_wid / 2], size=n_side)
            sz = self.rng.uniform(-self.sensor_height + 0.3, -self.sensor_height + c_hgt, n_side)
            sub_pts.append(np.column_stack([sx, sy, sz]))

            # Front / Rear
            n_fr = n_car_pts // 4
            fx = self.rng.choice([-c_len / 2, c_len / 2], size=n_fr)
            fy = self.rng.uniform(-c_wid / 2, c_wid / 2, n_fr)
            fz = self.rng.uniform(-self.sensor_height + 0.3, -self.sensor_height + c_hgt, n_fr)
            sub_pts.append(np.column_stack([fx, fy, fz]))

            # Roof
            n_roof = n_car_pts - n_side - n_fr
            rx = self.rng.uniform(-c_len / 3, c_len / 3, n_roof)
            ry = self.rng.uniform(-c_wid / 2, c_wid / 2, n_roof)
            rz = np.full(n_roof, -self.sensor_height + c_hgt)
            sub_pts.append(np.column_stack([rx, ry, rz]))

            car_model_pts = np.vstack(sub_pts)
            car_model_pts[:, 0] += car_x
            car_model_pts[:, 1] += lane_y

            points_list.append(car_model_pts)
            labels_list.append(np.full(len(car_model_pts), SemanticClass.DYNAMIC_VEHICLE, dtype=np.uint8))
            # Highly reflective metallic/glass surfaces + retroreflectors
            car_refl = self.rng.uniform(0.5, 0.95, len(car_model_pts)).astype(np.float32)
            intensities_list.append(car_refl)

        # B. Pedestrians on sidewalks
        ped_base_x = [-18.0, -5.0, 12.0, 28.0, -32.0]
        ped_speeds = [1.3, -1.1, 1.4, -1.2, 0.9]

        for i in range(min(num_pedestrians, len(ped_base_x))):
            ped_side = 1 if (i % 2 == 0) else -1
            ped_y = ped_side * (road_width / 2.0 + 1.6)
            ped_x = ped_base_x[i] + ped_speeds[i] * timestamp

            # Pedestrian Capsule: Height ~1.75m, Radius ~0.3m
            n_ped_pts = int(self.rng.integers(60, 130))
            pz = self.rng.uniform(-self.sensor_height + 0.15, -self.sensor_height + 1.85, n_ped_pts)
            ptheta = self.rng.uniform(0, 2 * np.pi, n_ped_pts)
            pr = self.rng.uniform(0.1, 0.28, n_ped_pts)
            px = ped_x + pr * np.cos(ptheta)
            py = ped_y + pr * np.sin(ptheta)

            ped_pts = np.column_stack([px, py, pz])
            points_list.append(ped_pts)
            labels_list.append(np.full(n_ped_pts, SemanticClass.DYNAMIC_PEDESTRIAN, dtype=np.uint8))
            intensities_list.append(self.rng.uniform(0.2, 0.45, n_ped_pts).astype(np.float32))

        # ----------------------------------------------------------------------
        # 4. Sensor Noise & Outliers
        # ----------------------------------------------------------------------
        all_pts_arr = np.vstack(points_list)
        total_pts = len(all_pts_arr)
        num_noise = int(total_pts * noise_ratio)

        if num_noise > 0:
            noise_r = self.rng.uniform(1.0, self.max_range, num_noise)
            noise_theta = self.rng.uniform(0, 2 * np.pi, num_noise)
            noise_phi = self.rng.uniform(self.fov_down, self.fov_up, num_noise)

            nx = noise_r * np.cos(noise_phi) * np.cos(noise_theta)
            ny = noise_r * np.cos(noise_phi) * np.sin(noise_theta)
            nz = noise_r * np.sin(noise_phi)
            noise_pts = np.column_stack([nx, ny, nz])

            points_list.append(noise_pts)
            labels_list.append(np.full(num_noise, SemanticClass.NOISE, dtype=np.uint8))
            intensities_list.append(self.rng.uniform(0.01, 0.15, num_noise).astype(np.float32))

        # ----------------------------------------------------------------------
        # Assemble Final Consolidated LidarFrame
        # ----------------------------------------------------------------------
        final_points = np.vstack(points_list).astype(np.float32)
        final_labels = np.concatenate(labels_list).astype(np.uint8)
        final_intensities = np.concatenate(intensities_list).astype(np.float32)

        # Apply realistic measurement noise (Gaussian lidar jitter ~1cm standard deviation)
        final_points += self.rng.normal(0, 0.01, size=final_points.shape).astype(np.float32)

        frame = LidarFrame(
            points=final_points,
            labels=final_labels,
            intensities=final_intensities,
            frame_id=frame_id,
            timestamp=timestamp,
        )

        # Filter strictly within operational radial range
        return frame.filter_range(min_dist=0.8, max_dist=self.max_range)


# ==============================================================================
# 5. DATASET LOADERS & ITERATORS
# ==============================================================================

class SyntheticLidarDataset:
    """
    Simulates a streaming sequence of sequential LiDAR frames with smooth
    temporal vehicle/pedestrian dynamics.
    """

    def __init__(
        self,
        num_frames: int = 50,
        fps: float = 10.0,
        sensor_height: float = 1.73,
        seed: int = 42,
    ):
        self.num_frames = num_frames
        self.dt = 1.0 / fps
        self.generator = SyntheticLidarGenerator(sensor_height=sensor_height, seed=seed)

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> LidarFrame:
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_frames}")
        t = idx * self.dt
        return self.generator.generate_frame(frame_id=idx, timestamp=t)


class SemanticKittiDataset:
    """
    Loads real SemanticKITTI sequences from disk:
    Directory structure expected:
      <root_dir>/sequences/<sequence_id>/velodyne/*.bin
      <root_dir>/sequences/<sequence_id>/labels/*.label  (optional)
    """

    def __init__(
        self,
        root_dir: str,
        sequence_id: str = "00",
        load_labels: bool = True,
    ):
        self.root_dir = root_dir
        self.sequence_id = str(sequence_id).zfill(2)
        self.load_labels = load_labels

        seq_dir = os.path.join(root_dir, "sequences", self.sequence_id)
        if not os.path.isdir(seq_dir):
            seq_dir = root_dir  # Try root dir directly if pointed at sequence folder

        self.velodyne_dir = os.path.join(seq_dir, "velodyne")
        self.labels_dir = os.path.join(seq_dir, "labels")

        self.bin_files = sorted(glob.glob(os.path.join(self.velodyne_dir, "*.bin")))
        if not self.bin_files:
            # Also try flat directory
            self.bin_files = sorted(glob.glob(os.path.join(root_dir, "*.bin")))

    def __len__(self) -> int:
        return len(self.bin_files)

    def __getitem__(self, idx: int) -> LidarFrame:
        if idx < 0 or idx >= len(self.bin_files):
            raise IndexError(f"Index {idx} out of range for SemanticKITTI dataset of size {len(self.bin_files)}")

        bin_path = self.bin_files[idx]
        raw_scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        points = raw_scan[:, :3]
        intensities = raw_scan[:, 3]

        labels = None
        if self.load_labels:
            stem = os.path.splitext(os.path.basename(bin_path))[0]
            label_path = os.path.join(self.labels_dir, f"{stem}.label")
            if os.path.exists(label_path):
                raw_labels = np.fromfile(label_path, dtype=np.uint32)
                labels = SemanticKittiMapper.map_labels(raw_labels)
            else:
                labels = np.zeros(len(points), dtype=np.uint8)

        return LidarFrame(
            points=points,
            labels=labels if labels is not None else np.zeros(len(points), dtype=np.uint8),
            intensities=intensities,
            frame_id=idx,
            timestamp=idx * 0.1,  # Standard KITTI 10 Hz
        )


# ==============================================================================
# 6. SELF-TEST AND VERIFICATION
# ==============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("SIH26053: Testing data_loader.py & Synthetic LiDAR Generator")
    print("=" * 70)

    # 1. Initialize Synthetic Generator and Produce Frame 0
    generator = SyntheticLidarGenerator(sensor_height=1.73, seed=123)
    frame = generator.generate_frame(frame_id=0, timestamp=0.0)

    print(f"\n[+] Generated Synthetic Frame 0:")
    print(f"    - Total Points    : {frame.num_points:,}")
    print(f"    - X Extents (m)   : [{frame.points[:, 0].min():.2f}, {frame.points[:, 0].max():.2f}]")
    print(f"    - Y Extents (m)   : [{frame.points[:, 1].min():.2f}, {frame.points[:, 1].max():.2f}]")
    print(f"    - Z Extents (m)   : [{frame.points[:, 2].min():.2f}, {frame.points[:, 2].max():.2f}]")
    print(f"    - Intensity range : [{frame.intensities.min():.2f}, {frame.intensities.max():.2f}]")

    # 2. Inspect Semantic Class Breakdown
    class_counts = frame.get_class_counts()
    print("\n[+] Semantic Class Distribution:")
    for c_name, count in class_counts.items():
        pct = (count / frame.num_points) * 100
        print(f"    - {c_name:<16}: {count:7,d} points ({pct:5.2f}%)")

    # 3. Test Range and ROI Filtering
    cropped = frame.filter_roi(x_range=(0.0, 50.0), y_range=(-20.0, 20.0))
    print(f"\n[+] Forward ROI Filtered Frame (0 to 50m Forward, +/-20m Lateral):")
    print(f"    - Filtered Points : {cropped.num_points:,} / {frame.num_points:,}")

    # 4. Test SemanticKITTI Label Mapping Logic
    test_raw_kitti = np.array([0, 10, 40, 50, 70, 252, 1], dtype=np.uint32)
    mapped = SemanticKittiMapper.map_labels(test_raw_kitti)
    print("\n[+] SemanticKITTI Label Mapping Verification:")
    raw_names = ["unlabeled(0)", "car(10)", "road(40)", "building(50)", "veg(70)", "moving_car(252)", "outlier(1)"]
    for r_name, m_val in zip(raw_names, mapped):
        print(f"    - {r_name:<18} -> Class {m_val} ({SemanticClass.get_name(m_val)})")

    # 5. Test Dataset Stream
    dataset = SyntheticLidarDataset(num_frames=5, fps=10.0)
    print(f"\n[+] Streaming Synthetic Dataset: {len(dataset)} frames")
    for i, f in enumerate(dataset):
        print(f"    - Frame {f.frame_id}: timestamp={f.timestamp:.2f}s, points={f.num_points:,}")

    # 6. Test File Export & Reload (KITTI .bin / .label format round-trip)
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_file = os.path.join(tmpdir, "000000.bin")
        label_file = os.path.join(tmpdir, "000000.label")
        frame.save_to_kitti_format(bin_file, label_file)

        # Verify file sizes
        bin_size = os.path.getsize(bin_file)
        label_size = os.path.getsize(label_file)
        print(f"\n[+] Binary Serialization Round-Trip:")
        print(f"    - Exported .bin ({bin_size:,} bytes) & .label ({label_size:,} bytes)")

        # Reload with raw NumPy
        reloaded_scan = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        reloaded_labels = np.fromfile(label_file, dtype=np.uint32)
        assert np.allclose(frame.points, reloaded_scan[:, :3]), "Points mismatch!"
        assert np.array_equal(frame.labels, reloaded_labels & 0xFFFF), "Labels mismatch!"
        print("    - Verification SUCCESS: Exported and reloaded data identical.")

    print("\n" + "=" * 70)
    print("All data_loader.py unit tests passed successfully!")
    print("=" * 70)
