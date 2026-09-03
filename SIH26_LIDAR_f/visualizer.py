"""
visualizer.py - Multi-Resolution 2.5D Foveated LiDAR Grid Visualizer
SIH26053: Adaptive Variable Resolution 2.5D LiDAR Mapping Pipeline

Features:
    - Top-Down 2D Multi-Resolution Grid Rendering:
        * Explicitly renders individual rectangular cells with variable physical sizes:
            - Near Ring (0-10m)  : 10cm x 10cm cells
            - Mid Ring (10-30m)  : 25cm x 25cm cells
            - Far Ring (30-70m)  : 50cm x 50cm cells
        * Canonical Color Mapping:
            - Green (#2ecc71) : Ground (Drivable road / sidewalks / terrain)
            - Red   (#e74c3c) : Static Obstacles (Buildings, trees, poles, barriers)
            - Blue  (#3498db) : Dynamic Objects (Vehicles, pedestrians, cyclists)
            - Gray  (#7f8c8d) : Noise / Unlabeled / Sensor artifacts
    - Concentric Ring Range Overlays & Ego-Vehicle Footprint.
    - Multi-Panel Dashboard:
        1. Foveated Semantic Occupancy Grid (with visible variable cell sizes)
        2. 2.5D Max Elevation (DEM) Surface
        3. Obstacle Height Difference (Delta Z)
        4. Real-Time Telemetry & Memory Optimization Metrics
    - Interactive Real-Time Playback & High-Res Headless PNG Export.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
import numpy as np
import plotly.graph_objects as go

from data_loader import LidarFrame, SemanticClass, SyntheticLidarDataset, SyntheticLidarGenerator
from grid_engine import AdaptiveGridEngine, FoveatedMap25D, RingGrid


# ==============================================================================
# 1. COLOR PALETTE & STYLING CONSTANTS
# ==============================================================================

COLOR_MAP_HEX = {
    SemanticClass.NOISE: "#7f8c8d",                 # Muted Gray
    SemanticClass.GROUND: "#2ecc71",                # Vibrant Emerald Green
    SemanticClass.STATIC_STRUCTURE: "#e74c3c",      # Vivid Crimson Red (walls/buildings)
    SemanticClass.STATIC_POLE: "#9b59b6",           # Purple (poles/signs)
    SemanticClass.STATIC_VEGETATION: "#6ab04c",     # Olive Green (trees/foliage)
    SemanticClass.DYNAMIC_VEHICLE: "#3498db",       # Electric Azure Blue (vehicles)
    SemanticClass.DYNAMIC_PEDESTRIAN: "#f1c40f",    # Amber (pedestrians)
}

# Normalized RGB for PolyCollection rendering
COLOR_MAP_RGB = {
    SemanticClass.NOISE: np.array([127, 140, 141, 180]) / 255.0,             # Alpha 0.70
    SemanticClass.GROUND: np.array([46, 204, 113, 215]) / 255.0,             # Alpha 0.85
    SemanticClass.STATIC_STRUCTURE: np.array([231, 76, 60, 255]) / 255.0,    # Alpha 1.00
    SemanticClass.STATIC_POLE: np.array([155, 89, 182, 255]) / 255.0,        # Alpha 1.00
    SemanticClass.STATIC_VEGETATION: np.array([106, 176, 76, 255]) / 255.0,  # Alpha 1.00
    SemanticClass.DYNAMIC_VEHICLE: np.array([52, 152, 219, 255]) / 255.0,    # Alpha 1.00
    SemanticClass.DYNAMIC_PEDESTRIAN: np.array([241, 196, 15, 255]) / 255.0, # Alpha 1.00
}


# ==============================================================================
# 2. FOVEATED 2.5D GRID VISUALIZER CLASS
# ==============================================================================

class FoveatedGridVisualizer:
    """
    Renders top-down 2D multi-resolution grids and multi-panel telemetry dashboards.
    """

    def __init__(
        self,
        dark_theme: bool = True,
        figure_size: Tuple[int, int] = (16, 12),
        dpi: int = 120,
    ):
        self.dark_theme = dark_theme
        self.figure_size = figure_size
        self.dpi = dpi

        # Theme color configuration - dark telemetry (slate/cyan) palette
        if dark_theme:
            self.bg_color = "#0B0F17"       # near-black slate
            self.panel_bg = "#141A24"       # panel surface
            self.text_color = "#E6EAF2"     # near-white text
            self.grid_line_color = "#232B3A"
            self.ring_line_color = "#F5A623"  # amber rings
        else:
            self.bg_color = "#FFFFFF"
            self.panel_bg = "#F7FAFC"
            self.text_color = "#1A202C"
            self.grid_line_color = "#E2E8F0"
            self.ring_line_color = "#B5651D"

    def _draw_foveated_semantic_cells(
        self,
        ax: plt.Axes,
        foveated_map: FoveatedMap25D,
        show_cell_borders: bool = True,
    ) -> None:
        """
        Renders each occupied cell as an explicit rectangular polygon with physical
        dimensions corresponding to that zone's resolution.
        """
        ax.set_facecolor(self.panel_bg)

        # Process concentric rings from outermost to innermost
        for ring in reversed(foveated_map.rings):
            cfg = ring.config
            res = cfg.resolution
            extent = cfg.grid_extent

            occ_y, occ_x = np.where(ring.occupied_mask)
            if len(occ_x) == 0:
                continue

            # Convert discrete indices to world coordinates (meters)
            # x_min, x_max, y_min, y_max
            x0 = occ_x * res - extent
            x1 = x0 + res
            y0 = occ_y * res - extent
            y1 = y0 + res

            # Construct 4 vertices for every occupied rectangular cell
            # Shape (N_cells, 4, 2)
            n_cells = len(occ_x)
            verts = np.zeros((n_cells, 4, 2), dtype=np.float32)

            verts[:, 0, 0] = x0
            verts[:, 0, 1] = y0

            verts[:, 1, 0] = x1
            verts[:, 1, 1] = y0

            verts[:, 2, 0] = x1
            verts[:, 2, 1] = y1

            verts[:, 3, 0] = x0
            verts[:, 3, 1] = y1

            # Extract cell labels and map to RGBA colors
            cell_labels = ring.labels[occ_y, occ_x]
            facecolors = np.zeros((n_cells, 4), dtype=np.float32)

            for c_id in [0, 1, 2, 3]:
                mask = (cell_labels == c_id)
                if np.any(mask):
                    facecolors[mask] = COLOR_MAP_RGB[SemanticClass(c_id)]

            edgecolor = (0.05, 0.05, 0.05, 0.4) if show_cell_borders else "none"
            linewidth = 0.4 if res >= 0.25 else 0.2

            # Create fast batch PolyCollection
            poly_coll = PolyCollection(
                verts,
                facecolors=facecolors,
                edgecolors=edgecolor,
                linewidths=linewidth,
                antialiased=True,
            )
            ax.add_collection(poly_coll)

        # Draw Concentric Ring Boundaries & Annotations
        for ring in foveated_map.rings:
            cfg = ring.config
            # Outer range boundary circle
            circle = plt.Circle(
                (0, 0),
                cfg.r_max,
                color=self.ring_line_color,
                fill=False,
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
            )
            ax.add_patch(circle)

            # Label on the positive Y-axis
            label_y = cfg.r_max - (cfg.r_max - cfg.r_min) * 0.15
            ax.text(
                0.8,
                label_y,
                f"{cfg.name}\n[r={cfg.r_min:.0f}-{cfg.r_max:.0f}m | Res={cfg.resolution*100:.0f}cm]",
                color=self.ring_line_color,
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=self.panel_bg, edgecolor=self.ring_line_color, alpha=0.85),
            )

        # Draw Ego-Vehicle representation at origin (0, 0)
        # Vehicle Box: Length 4.5m, Width 2.0m
        ego_box = patches.Rectangle(
            (-1.0, -2.25), 2.0, 4.5,
            facecolor="#f39c12", edgecolor="#ffffff",
            linewidth=1.5, zorder=10, alpha=0.9
        )
        ax.add_patch(ego_box)
        # Heading arrow (Forward = +X in standard sensor frame, let's plot Forward on +Y or +X)
        # For standard driving Top-Down plot: X=Lateral, Y=Forward or X=Forward, Y=Lateral
        # Here: Sensor coordinates X=Forward, Y=Left. We plot X on horizontal, Y on vertical.
        ax.arrow(0, 0, 3.0, 0, head_width=1.0, head_length=1.0, fc="#ffffff", ec="#ffffff", zorder=11)
        ax.text(0, -4.5, "EGO (LIDAR)", color="#f39c12", fontsize=9, fontweight="bold", ha="center", zorder=12)

        max_extent = max(r.config.r_max for r in foveated_map.rings) * 1.05
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Sensor X (Forward / m)", color=self.text_color, fontsize=10)
        ax.set_ylabel("Sensor Y (Left / m)", color=self.text_color, fontsize=10)
        ax.tick_params(colors=self.text_color)
        for spine in ax.spines.values():
            spine.set_color(self.grid_line_color)

    def render_topdown(
        self,
        foveated_map: FoveatedMap25D,
        title: str = "SIH26053: Variable-Resolution 2.5D LiDAR Grid Map",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Renders a focused, standalone high-resolution top-down foveated map.
        """
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi, facecolor=self.bg_color)
        self._draw_foveated_semantic_cells(ax, foveated_map)

        # Custom Legend
        legend_elements = [
            patches.Patch(facecolor=COLOR_MAP_HEX[SemanticClass.GROUND], label="Ground (Drivable / Terrain)"),
            patches.Patch(facecolor=COLOR_MAP_HEX[SemanticClass.STATIC_STRUCTURE], label="Static Structure (Building / Wall)"),
            patches.Patch(facecolor=COLOR_MAP_HEX[SemanticClass.STATIC_POLE], label="Static Pole (Sign / Lamp Post)"),
            patches.Patch(facecolor=COLOR_MAP_HEX[SemanticClass.STATIC_VEGETATION], label="Static Vegetation (Tree / Foliage)"),
            patches.Patch(facecolor=COLOR_MAP_HEX[SemanticClass.DYNAMIC_VEHICLE], label="Dynamic Vehicle (Car / Truck)"),
            patches.Patch(facecolor=COLOR_MAP_HEX[SemanticClass.DYNAMIC_PEDESTRIAN], label="Dynamic Pedestrian"),
            patches.Patch(facecolor=COLOR_MAP_HEX[SemanticClass.NOISE], label="Noise / Outliers"),
        ]
        leg = ax.legend(
            handles=legend_elements,
            loc="upper right",
            facecolor=self.panel_bg,
            edgecolor=self.grid_line_color,
            fontsize=9,
            labelcolor=self.text_color,
            framealpha=0.9,
        )

        ax.set_title(
            f"{title}\nFrame {foveated_map.frame_id} | "
            f"Latency: {foveated_map.processing_time_ms:.1f}ms ({1000.0/max(foveated_map.processing_time_ms, 0.1):.0f} FPS) | "
            f"Occupied: {foveated_map.total_occupied_cells():,} cells",
            color=self.text_color,
            fontsize=12,
            pad=14,
            fontweight="bold",
        )

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, facecolor=self.bg_color, bbox_inches="tight")
            print(f"[+] Top-Down visualization saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def render_dashboard(
        self,
        foveated_map: FoveatedMap25D,
        raw_frame: Optional[LidarFrame] = None,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Renders a complete 4-Panel Autonomous Systems Engineering Dashboard:
          1. Multi-Resolution Foveated Semantic Grid (with visible varying cell sizes)
          2. Digital Elevation Model (DEM: Max Elevation z_max)
          3. Height Difference (Delta Z = z_max - z_min)
          4. Real-time Telemetry & Memory Reduction Profile
        """
        fig = plt.figure(figsize=self.figure_size, dpi=self.dpi, facecolor=self.bg_color)
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.20)

        # ----------------------------------------------------------------------
        # Panel 1: Foveated Semantic Grid (Top-Left)
        # ----------------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_foveated_semantic_cells(ax1, foveated_map)
        ax1.set_title("Foveated Semantic Occupancy Grid (Variable Cell Sizes)", color=self.text_color, fontsize=11, fontweight="bold")

        # ----------------------------------------------------------------------
        # Panel 2: Max Elevation DEM (Top-Right)
        # ----------------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(self.panel_bg)

        # Resample to dense composite for smooth elevation colormap rendering
        composite = foveated_map.to_dense_composite(composite_extent=70.0, composite_resolution=0.50)
        z_max_grid = np.ma.masked_invalid(composite["z_max"])

        extent = composite["extent"]
        im_dem = ax2.imshow(
            z_max_grid,
            origin="lower",
            extent=[-extent, extent, -extent, extent],
            cmap="viridis",
            interpolation="nearest",
        )
        cbar_dem = plt.colorbar(im_dem, ax=ax2, fraction=0.046, pad=0.04)
        cbar_dem.set_label("Max Elevation (Z / meters)", color=self.text_color, fontsize=9)
        cbar_dem.ax.tick_params(colors=self.text_color)

        # Concentric ring outlines
        for ring in foveated_map.rings:
            circle = plt.Circle((0, 0), ring.config.r_max, color=self.ring_line_color, fill=False, linestyle="--", linewidth=1.0, alpha=0.6)
            ax2.add_patch(circle)

        ax2.set_xlim(-extent * 1.05, extent * 1.05)
        ax2.set_ylim(-extent * 1.05, extent * 1.05)
        ax2.set_title("2.5D Digital Elevation Model ($Z_{max}$)", color=self.text_color, fontsize=11, fontweight="bold")
        ax2.set_xlabel("Sensor X (Forward / m)", color=self.text_color, fontsize=9)
        ax2.set_ylabel("Sensor Y (Left / m)", color=self.text_color, fontsize=9)
        ax2.tick_params(colors=self.text_color)
        for spine in ax2.spines.values():
            spine.set_color(self.grid_line_color)

        # ----------------------------------------------------------------------
        # Panel 3: Height Variation / Delta Z (Bottom-Left)
        # ----------------------------------------------------------------------
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(self.panel_bg)

        delta_z_grid = np.ma.masked_where(~composite["occupied"], composite["delta_z"])
        im_dz = ax3.imshow(
            delta_z_grid,
            origin="lower",
            extent=[-extent, extent, -extent, extent],
            cmap="magma",
            interpolation="nearest",
            vmax=3.5,
        )
        cbar_dz = plt.colorbar(im_dz, ax=ax3, fraction=0.046, pad=0.04)
        cbar_dz.set_label("Obstacle Height Delta ($Z_{max} - Z_{min}$ / m)", color=self.text_color, fontsize=9)
        cbar_dz.ax.tick_params(colors=self.text_color)

        for ring in foveated_map.rings:
            circle = plt.Circle((0, 0), ring.config.r_max, color=self.ring_line_color, fill=False, linestyle="--", linewidth=1.0, alpha=0.6)
            ax3.add_patch(circle)

        ax3.set_xlim(-extent * 1.05, extent * 1.05)
        ax3.set_ylim(-extent * 1.05, extent * 1.05)
        ax3.set_title(r"Height Variation / Obstacle Profile ($\Delta Z$)", color=self.text_color, fontsize=11, fontweight="bold")
        ax3.set_xlabel("Sensor X (Forward / m)", color=self.text_color, fontsize=9)
        ax3.set_ylabel("Sensor Y (Left / m)", color=self.text_color, fontsize=9)
        ax3.tick_params(colors=self.text_color)
        for spine in ax3.spines.values():
            spine.set_color(self.grid_line_color)

        # ----------------------------------------------------------------------
        # Panel 4: Telemetry & Memory Profiler (Bottom-Right)
        # ----------------------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(self.panel_bg)
        ax4.axis("off")

        mem_stats = foveated_map.memory_reduction_vs_uniform(uniform_resolution=0.10)
        fps = 1000.0 / max(foveated_map.processing_time_ms, 0.1)

        total_pts = raw_frame.num_points if raw_frame else sum(np.sum(r.point_counts) for r in foveated_map.rings)

        telemetry_text = (
            f"SIH26053 PIPELINE TELEMETRY & SYSTEM HEALTH\n"
            f"{'=' * 48}\n\n"
            f"  * Pipeline Throughput : {fps:5.1f} FPS (Latency: {foveated_map.processing_time_ms:.2f} ms)\n"
            f"  * Total Raw Points    : {total_pts:,} pts\n"
            f"  * Total 2.5D Cells    : {foveated_map.total_cells():,} cells\n"
            f"  * Occupied Grid Cells : {foveated_map.total_occupied_cells():,} cells\n\n"
            f"CONCENTRIC RING BREAKDOWN:\n"
            f"  - Ring 0 (Near / Foveal):  0-10m @ 10cm res ({foveated_map.rings[0].config.grid_dim}x{foveated_map.rings[0].config.grid_dim})\n"
            f"      Occupied: {int(np.sum(foveated_map.rings[0].occupied_mask)):,} cells | RAM: {foveated_map.rings[0].memory_bytes/1024:.1f} KB\n"
            f"  - Ring 1 (Mid-Range)   : 10-30m @ 25cm res ({foveated_map.rings[1].config.grid_dim}x{foveated_map.rings[1].config.grid_dim})\n"
            f"      Occupied: {int(np.sum(foveated_map.rings[1].occupied_mask)):,} cells | RAM: {foveated_map.rings[1].memory_bytes/1024:.1f} KB\n"
            f"  - Ring 2 (Far-Range)   : 30-70m @ 50cm res ({foveated_map.rings[2].config.grid_dim}x{foveated_map.rings[2].config.grid_dim})\n"
            f"      Occupied: {int(np.sum(foveated_map.rings[2].occupied_mask)):,} cells | RAM: {foveated_map.rings[2].memory_bytes/1024:.1f} KB\n\n"
            f"MEMORY SAVINGS ANALYSIS:\n"
            f"  - Uniform 10cm Grid RAM : {mem_stats['uniform_memory_mb']:.2f} MB ({mem_stats['uniform_total_cells']:,} cells)\n"
            f"  - Adaptive 2.5D Map RAM : {mem_stats['adaptive_memory_mb']:.2f} MB ({mem_stats['adaptive_total_cells']:,} cells)\n"
            f"  - RAM REDUCTION        : {mem_stats['savings_percent']:.2f}% (Compression: {mem_stats['compression_ratio']:.1f}x)\n"
        )

        ax4.text(
            0.05, 0.95, telemetry_text,
            transform=ax4.transAxes,
            color=self.text_color,
            fontsize=10,
            fontfamily="monospace",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#12161f", edgecolor=self.ring_line_color, alpha=0.9),
        )

        # Super Title
        fig.suptitle(
            f"SIH26053: Real-Time Adaptive Variable-Resolution 2.5D LiDAR Mapping Dashboard (Frame {foveated_map.frame_id})",
            color=self.text_color,
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, facecolor=self.bg_color, bbox_inches="tight")
            print(f"[+] Full telemetry dashboard saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


# ==============================================================================
# 3. 2.5D ELEVATION & FOVEATED PILLAR GRID BUILDER (PLOTLY 3D)
# ==============================================================================

def create_25d_pillar_grid_figure(
    foveated_map: FoveatedMap25D,
    color_mode: str = "image_gradient",     # "image_gradient", "elevation", "foveated_zones", "semantic"
    height_source: str = "obstacle_profile", # "obstacle_profile", "z_max", "fixed"
    grid_mode: str = "continuous_slab",     # "continuous_slab", "foveated", "composite"
    composite_resolution: float = 0.80,
    show_wireframe: bool = True,
    z_exaggeration: float = 2.4,
    max_range: float = 20.0,
    stride: int = 1,
    dark_theme: bool = True,
    camera_preset: str = "Isometric",       # "Isometric", "Perspective", "Top-Down", "Chase"
    show_rings: bool = False,
    show_ego: bool = False,
    colormap: str = "Turbo",                # "Turbo", "Jet", "Viridis", "Plasma"
    raw_frame: Optional[LidarFrame] = None,
    show_floating_points: bool = True,
) -> go.Figure:
    """
    Renders the 2.5D Elevation & Foveated Occupancy Grid as 3D extruded pillars/voxels,
    reproducing the exact 2.5D Grid Map visualization from the SIH26 LiDAR architecture.

    Features:
        - "continuous_slab": Solid 2.5D grid sheet matching the hackathon slide image,
          where ground cells form a solid base slab and obstacle features (curbs, vehicles,
          trees, buildings) rise up as distinct stepped 3D columns.
        - "image_gradient": Smooth lateral rainbow/turbo scan gradient (Blue -> Cyan -> Green ->
          Yellow -> Orange -> Red) matching the uploaded reference image.
        - Optional floating 3D LiDAR point cloud markers hovering above the 2.5D pillars.
        - Crisp top-face wireframe grid borders outlining each cell block.
    """
    fig = go.Figure()
    bg_color = "#0B0F17" if dark_theme else "#FFFFFF"
    text_color = "#E6EAF2" if dark_theme else "#1A202C"

    if grid_mode == "continuous_slab":
        # Continuous rectangular grid slab matching the uploaded image exactly
        roi_extent = float(min(max_range, 30.0))
        res = float(max(0.4, composite_resolution))
        dim = int(np.ceil((2.0 * roi_extent) / res))

        comp = foveated_map.to_dense_composite(composite_extent=roi_extent, composite_resolution=res)
        occ = comp["occupied"]
        delta_z = comp["delta_z"]
        z_max = comp["z_max"]
        labels = comp["labels"]

        xs = np.linspace(-roi_extent + res / 2.0, roi_extent - res / 2.0, dim)
        ys = np.linspace(-roi_extent + res / 2.0, roi_extent - res / 2.0, dim)
        gx, gy = np.meshgrid(xs, ys)

        # Baseline ground slab height so the floor forms a solid continuous tray
        base_floor = 0.55
        top_h = np.full((dim, dim), base_floor, dtype=np.float32)

        # Obstacles (curbs, cars, trees, structures) extrude significantly above the ground
        obs_mask = occ & ((delta_z > 0.12) | (labels > 1))
        if height_source in ("obstacle_profile", "delta_z"):
            top_h[obs_mask] = base_floor + np.maximum(0.20, delta_z[obs_mask] * z_exaggeration)
        elif height_source == "z_max":
            valid_z = z_max[occ & ~np.isnan(z_max)]
            z_ground = float(np.min(valid_z)) if len(valid_z) > 0 else 0.0
            top_h[occ] = base_floor + np.maximum(0.10, (z_max[occ] - z_ground) * z_exaggeration)
        else:  # fixed
            top_h[obs_mask] = base_floor + 0.80 * z_exaggeration

        # Subtle ground elevation variations for drivable terrain realism
        ground_mask = occ & (~obs_mask)
        if np.any(ground_mask):
            valid_zg = z_max[ground_mask & ~np.isnan(z_max)]
            if len(valid_zg) > 0:
                min_zg = np.min(valid_zg)
                top_h[ground_mask] = base_floor + np.clip((z_max[ground_mask] - min_zg) * 0.30, 0.0, 0.25)

        x0 = (gx - res / 2.0).flatten()
        y0 = (gy - res / 2.0).flatten()
        w = np.full(len(x0), res)
        h = top_h.flatten()
        n_cells = len(x0)

        # Build 8 vertices per pillar
        vx = np.column_stack([x0, x0 + w, x0 + w, x0, x0, x0 + w, x0 + w, x0]).flatten()
        vy = np.column_stack([y0, y0, y0 + w, y0 + w, y0, y0, y0 + w, y0 + w]).flatten()
        vz = np.column_stack([np.zeros(n_cells), np.zeros(n_cells), np.zeros(n_cells), np.zeros(n_cells), h, h, h, h]).flatten()

        cube_triangles = np.array([
            [4, 5, 6], [4, 6, 7],  # Top face
            [0, 1, 5], [0, 5, 4],  # Front face
            [1, 2, 6], [1, 6, 5],  # Right face
            [2, 3, 7], [2, 7, 6],  # Back face
            [3, 0, 4], [3, 4, 7],  # Left face
        ], dtype=np.int32)
        base_offsets = (np.arange(n_cells) * 8)[:, None, None]
        all_triangles = (cube_triangles[None, :, :] + base_offsets).reshape(-1, 3)
        ti = all_triangles[:, 0]
        tj = all_triangles[:, 1]
        tk = all_triangles[:, 2]

        if color_mode == "image_gradient":
            # Lateral scan gradient matching the uploaded reference image (Blue -> Cyan -> Green -> Yellow -> Red)
            norm_y = (y0 - y0.min()) / max(1e-6, (y0.max() - y0.min()))
            intensities = np.repeat(norm_y, 8)
            mesh = go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=ti, j=tj, k=tk,
                intensity=intensities,
                colorscale="Turbo",
                showscale=False,
                lighting=dict(ambient=0.70, diffuse=0.80, specular=0.30, roughness=0.60),
                name="2.5D Elevation Grid",
                hovertemplate="X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Height: %{z:.2f}m<extra></extra>",
            )
            fig.add_trace(mesh)

        elif color_mode == "elevation":
            norm_h = (h - h.min()) / max(1e-6, (h.max() - h.min()))
            intensities = np.repeat(norm_h, 8)
            mesh = go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=ti, j=tj, k=tk,
                intensity=intensities,
                colorscale=colormap,
                colorbar=dict(
                    title=dict(text="Height (m)", font=dict(color=text_color, size=11)),
                    tickfont=dict(color=text_color),
                    len=0.75,
                    thickness=18,
                ),
                lighting=dict(ambient=0.70, diffuse=0.80, specular=0.30, roughness=0.60),
                showscale=True,
                name="2.5D Elevation Grid",
                hovertemplate="X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Height: %{z:.2f}m<extra></extra>",
            )
            fig.add_trace(mesh)

        elif color_mode == "foveated_zones":
            r_flat = np.hypot(x0 + res / 2.0, y0 + res / 2.0)
            z_ids = np.zeros(n_cells, dtype=np.int32)
            for z_idx, ring in enumerate(foveated_map.rings):
                cfg = ring.config
                z_ids[(r_flat >= cfg.r_min) & (r_flat <= cfg.r_max)] = z_idx
            zone_hex = {0: "#2563EB", 1: "#10B981", 2: "#F59E0B"}
            pillar_colors = [zone_hex.get(int(z), "#8B5CF6") for z in z_ids]
            face_colors = np.repeat(pillar_colors, 10)
            mesh = go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=ti, j=tj, k=tk,
                facecolor=face_colors,
                lighting=dict(ambient=0.70, diffuse=0.80, specular=0.30, roughness=0.60),
                showscale=False,
                name="Foveated Zones Grid",
                hovertemplate="X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Height: %{z:.2f}m<extra></extra>",
            )
            fig.add_trace(mesh)

        else:  # semantic
            hex_map = SemanticClass.get_hex_color_map()
            lbl_flat = labels.flatten()
            pillar_colors = [hex_map.get(int(lbl), "#7F8C8D") for lbl in lbl_flat]
            face_colors = np.repeat(pillar_colors, 10)
            mesh = go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=ti, j=tj, k=tk,
                facecolor=face_colors,
                lighting=dict(ambient=0.70, diffuse=0.80, specular=0.30, roughness=0.60),
                showscale=False,
                name="Semantic Grid",
                hovertemplate="X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Height: %{z:.2f}m<extra></extra>",
            )
            fig.add_trace(mesh)

        # Wireframe cell grid lines on the top face of each pillar
        if show_wireframe and n_cells <= 16000:
            wf_color = "#0B0F17" if dark_theme else "rgba(30, 41, 59, 0.70)"
            wx = np.column_stack([x0, x0 + w, x0 + w, x0, x0, np.full(n_cells, np.nan)]).flatten()
            wy = np.column_stack([y0, y0 + w, y0 + w, y0, y0, np.full(n_cells, np.nan)]).flatten()
            wz = np.column_stack([h, h, h, h, h, np.full(n_cells, np.nan)]).flatten()
            fig.add_trace(go.Scatter3d(
                x=wx, y=wy, z=wz,
                mode="lines",
                line=dict(color=wf_color, width=1.5),
                hoverinfo="skip",
                showlegend=False,
                name="Cell Grid Lines",
            ))

        # Floating 3D LiDAR Points hovering above the grid (as seen in the reference slide)
        if show_floating_points and raw_frame is not None and len(raw_frame.points) > 0:
            pts = raw_frame.points
            in_roi = (np.abs(pts[:, 0]) <= roi_extent) & (np.abs(pts[:, 1]) <= roi_extent)
            sub_pts = pts[in_roi][::6]
            if len(sub_pts) > 0:
                pts_z = sub_pts[:, 2] + 1.73 + base_floor
                pts_norm_y = (sub_pts[:, 1] - (-roi_extent)) / (2.0 * roi_extent)
                fig.add_trace(go.Scatter3d(
                    x=sub_pts[:, 0],
                    y=sub_pts[:, 1],
                    z=pts_z,
                    mode="markers",
                    marker=dict(
                        size=2.5,
                        color=pts_norm_y,
                        colorscale="Turbo",
                        opacity=0.80,
                    ),
                    name="Raw LiDAR Points",
                    hoverinfo="skip",
                    showlegend=False,
                ))

    else:
        # Foveated Concentric Multi-Resolution Rings or Sparse Occupied Grid
        x0_list: List[np.ndarray] = []
        y0_list: List[np.ndarray] = []
        w_list: List[np.ndarray] = []
        ztop_list: List[np.ndarray] = []
        dz_list: List[np.ndarray] = []
        zone_list: List[np.ndarray] = []
        lbl_list: List[np.ndarray] = []

        if grid_mode == "foveated":
            for z_idx, ring in enumerate(foveated_map.rings):
                cfg = ring.config
                res = cfg.resolution
                ext = cfg.grid_extent
                occ_y, occ_x = np.where(ring.occupied_mask)
                if len(occ_x) == 0:
                    continue

                xc = (occ_x + 0.5) * res - ext
                yc = (occ_y + 0.5) * res - ext
                r = np.hypot(xc, yc)
                in_range = (r <= max_range)

                indices = np.where(in_range)[0]
                if len(indices) == 0:
                    continue

                effective_stride = stride
                indices = indices[::effective_stride]

                sel_x = occ_x[indices]
                sel_y = occ_y[indices]

                x0_list.append(sel_x * res - ext)
                y0_list.append(sel_y * res - ext)
                w_list.append(np.full(len(indices), res * effective_stride))
                ztop_list.append(ring.z_max[sel_y, sel_x])
                dz_list.append(ring.delta_z[sel_y, sel_x])
                zone_list.append(np.full(len(indices), z_idx, dtype=np.int32))
                lbl_list.append(ring.labels[sel_y, sel_x])
        else:
            # Sparse composite grid
            comp = foveated_map.to_dense_composite(composite_extent=max_range, composite_resolution=composite_resolution)
            occ_y, occ_x = np.where(comp["occupied"])
            if len(occ_x) > 0:
                res = comp["resolution"]
                ext = comp["extent"]
                indices = np.arange(len(occ_x))[::stride]
                sel_x = occ_x[indices]
                sel_y = occ_y[indices]

                xc = (sel_x + 0.5) * res - ext
                yc = (sel_y + 0.5) * res - ext
                r = np.hypot(xc, yc)

                z_ids = np.zeros(len(indices), dtype=np.int32)
                for z_idx, ring in enumerate(foveated_map.rings):
                    cfg = ring.config
                    z_ids[(r >= cfg.r_min) & (r <= cfg.r_max)] = z_idx

                x0_list.append(sel_x * res - ext)
                y0_list.append(sel_y * res - ext)
                w_list.append(np.full(len(indices), res * stride))
                ztop_list.append(comp["z_max"][sel_y, sel_x])
                dz_list.append(comp["delta_z"][sel_y, sel_x])
                zone_list.append(z_ids)
                lbl_list.append(comp["labels"][sel_y, sel_x])

        if len(x0_list) == 0:
            fig.add_annotation(
                text="No occupied cells observed within selected range",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color=text_color, size=14),
            )
            fig.update_layout(
                template="plotly_dark" if dark_theme else "plotly_white",
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color,
            )
            return fig

        x0 = np.concatenate(x0_list)
        y0 = np.concatenate(y0_list)
        w = np.concatenate(w_list)
        z_raw = np.concatenate(ztop_list)
        dz_raw = np.concatenate(dz_list)
        zones = np.concatenate(zone_list)
        labels = np.concatenate(lbl_list)
        n_cells = len(x0)

        valid_z = z_raw[~np.isnan(z_raw)]
        z_ground = float(np.min(valid_z)) if len(valid_z) > 0 else 0.0

        if height_source in ("obstacle_profile", "delta_z"):
            top_h = np.maximum(0.15, dz_raw * z_exaggeration)
        elif height_source == "z_max":
            top_h = np.maximum(0.15, (z_raw - z_ground) * z_exaggeration)
        else:
            top_h = np.full(n_cells, 0.40 * z_exaggeration)

        base_h = np.zeros(n_cells, dtype=np.float32)

        vx = np.column_stack([x0, x0 + w, x0 + w, x0, x0, x0 + w, x0 + w, x0]).flatten()
        vy = np.column_stack([y0, y0, y0 + w, y0 + w, y0, y0, y0 + w, y0 + w]).flatten()
        vz = np.column_stack([base_h, base_h, base_h, base_h, top_h, top_h, top_h, top_h]).flatten()

        cube_triangles = np.array([
            [4, 5, 6], [4, 6, 7],  # Top face
            [0, 1, 5], [0, 5, 4],  # Front face
            [1, 2, 6], [1, 6, 5],  # Right face
            [2, 3, 7], [2, 7, 6],  # Back face
            [3, 0, 4], [3, 4, 7],  # Left face
        ], dtype=np.int32)
        base_offsets = (np.arange(n_cells) * 8)[:, None, None]
        all_triangles = (cube_triangles[None, :, :] + base_offsets).reshape(-1, 3)
        ti = all_triangles[:, 0]
        tj = all_triangles[:, 1]
        tk = all_triangles[:, 2]

        if color_mode in ("image_gradient", "elevation"):
            norm_c = (y0 - y0.min()) / max(1e-6, (y0.max() - y0.min())) if color_mode == "image_gradient" else (z_raw if height_source != "delta_z" else dz_raw)
            intensities = np.repeat(norm_c, 8)
            mesh = go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=ti, j=tj, k=tk,
                intensity=intensities,
                colorscale="Turbo" if color_mode == "image_gradient" else colormap,
                lighting=dict(ambient=0.65, diffuse=0.75, specular=0.25, roughness=0.7),
                showscale=(color_mode == "elevation"),
                name="2.5D Pillar Grid",
                hovertemplate="X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Height: %{z:.2f}m<extra></extra>",
            )
            fig.add_trace(mesh)

        elif color_mode == "foveated_zones":
            zone_hex = {0: "#2563EB", 1: "#10B981", 2: "#F59E0B"}
            pillar_colors = [zone_hex.get(int(z), "#8B5CF6") for z in zones]
            face_colors = np.repeat(pillar_colors, 10)
            mesh = go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=ti, j=tj, k=tk,
                facecolor=face_colors,
                lighting=dict(ambient=0.65, diffuse=0.75, specular=0.25, roughness=0.7),
                showscale=False,
                name="Foveated Zones Grid",
                hovertemplate="X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Height: %{z:.2f}m<extra></extra>",
            )
            fig.add_trace(mesh)

        else:  # semantic
            hex_map = SemanticClass.get_hex_color_map()
            pillar_colors = [hex_map.get(int(lbl), "#7F8C8D") for lbl in labels]
            face_colors = np.repeat(pillar_colors, 10)
            mesh = go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=ti, j=tj, k=tk,
                facecolor=face_colors,
                lighting=dict(ambient=0.65, diffuse=0.75, specular=0.25, roughness=0.7),
                showscale=False,
                name="Semantic 2.5D Grid",
                hovertemplate="X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Height: %{z:.2f}m<extra></extra>",
            )
            fig.add_trace(mesh)

        if show_wireframe and n_cells <= 12000:
            wf_color = "#0B0F17" if dark_theme else "rgba(30, 41, 59, 0.65)"
            wx = np.column_stack([x0, x0 + w, x0 + w, x0, x0, np.full(n_cells, np.nan)]).flatten()
            wy = np.column_stack([y0, y0 + w, y0 + w, y0, y0, np.full(n_cells, np.nan)]).flatten()
            wz = np.column_stack([top_h, top_h, top_h, top_h, top_h, np.full(n_cells, np.nan)]).flatten()
            fig.add_trace(go.Scatter3d(
                x=wx, y=wy, z=wz,
                mode="lines",
                line=dict(color=wf_color, width=1.5),
                hoverinfo="skip",
                showlegend=False,
                name="Cell Grid Lines",
            ))

    # Concentric Ring Boundary Rings in 3D
    if show_rings:
        theta = np.linspace(0, 2 * np.pi, 120)
        for ring in foveated_map.rings:
            cfg = ring.config
            if cfg.r_max <= max_range * 1.05:
                rx = cfg.r_max * np.cos(theta)
                ry = cfg.r_max * np.sin(theta)
                rz = np.full_like(rx, 0.05)
                fig.add_trace(go.Scatter3d(
                    x=rx, y=ry, z=rz,
                    mode="lines",
                    line=dict(color="#F5A623", width=3, dash="dash"),
                    name=f"Ring {cfg.name} ({cfg.r_max:.0f}m)",
                    hoverinfo="name",
                    showlegend=False,
                ))

    # 3D Ego-Vehicle Footprint at Origin
    if show_ego:
        ev_x = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
        ev_y = np.array([-2.25, -2.25, 2.25, 2.25, -2.25, -2.25, -2.25, 2.25, 2.25, -2.25, 2.25, 2.25, 2.25, -2.25, -2.25, 2.25])
        ev_z = np.array([0, 0, 0, 0, 0, 1.5, 1.5, 1.5, 1.5, 1.5, 0, 1.5, 1.5, 0, 1.5, 1.5])
        fig.add_trace(go.Scatter3d(
            x=ev_x, y=ev_y, z=ev_z,
            mode="lines",
            line=dict(color="#F39C12", width=4),
            name="Ego Vehicle",
            hoverinfo="name",
            showlegend=False,
        ))

    # Camera presets tuned to the exact isometric perspective in the slide image
    camera_presets = {
        "Isometric": dict(eye=dict(x=-1.45, y=-1.45, z=1.15)),
        "Perspective": dict(eye=dict(x=1.75, y=0.00, z=0.55)),
        "Top-Down": dict(eye=dict(x=0.01, y=0.01, z=2.60)),
        "Chase": dict(eye=dict(x=-1.80, y=0.00, z=0.65)),
    }
    cam = camera_presets.get(camera_preset, camera_presets["Isometric"])

    fig.update_layout(
        scene=dict(
            xaxis_title="Forward X (m)",
            yaxis_title="Lateral Y (m)",
            zaxis_title=f"Height (x{z_exaggeration:.1f})",
            aspectmode="data",
            camera=cam,
            bgcolor=bg_color,
            xaxis=dict(gridcolor="#1E293B" if dark_theme else "#E2E8F0", zerolinecolor="#334155"),
            yaxis=dict(gridcolor="#1E293B" if dark_theme else "#E2E8F0", zerolinecolor="#334155"),
            zaxis=dict(gridcolor="#1E293B" if dark_theme else "#E2E8F0", zerolinecolor="#334155"),
        ),
        template="plotly_dark" if dark_theme else "plotly_white",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        height=680,
        margin=dict(l=0, r=0, t=10, b=0),
    )

    return fig


def create_3d_pointcloud_figure(
    frame: LidarFrame,
    color_by: str = "elevation",       # "elevation" or "class"
    max_display_points: int = 8000,
    dark_theme: bool = True,
    camera_preset: str = "Isometric",
    colormap: str = "Turbo",
) -> go.Figure:
    """
    Renders raw 3D LiDAR point cloud scatter in 3D, colored by elevation or semantic class,
    for direct side-by-side comparison with the 2.5D Elevation Grid.
    """
    pts = frame.points
    n_pts = len(pts)

    if n_pts == 0:
        fig = go.Figure()
        fig.add_annotation(text="No points in frame", showarrow=False)
        return fig

    # Subsample if necessary for smooth interactive 3D rendering
    downsample = max(1, n_pts // max_display_points)
    sub_pts = pts[::downsample]
    sub_lbl = frame.labels[::downsample]
    z_coords = sub_pts[:, 2]

    bg_color = "#0B0F17" if dark_theme else "#FFFFFF"
    text_color = "#E6EAF2" if dark_theme else "#1A202C"

    fig = go.Figure()

    if color_by == "elevation":
        scatter = go.Scatter3d(
            x=sub_pts[:, 0],
            y=sub_pts[:, 1],
            z=sub_pts[:, 2],
            mode="markers",
            marker=dict(
                size=2.5,
                color=z_coords,
                colorscale=colormap,
                colorbar=dict(
                    title=dict(text="Z (m)", font=dict(color=text_color, size=11)),
                    tickfont=dict(color=text_color),
                    len=0.75,
                    thickness=16,
                ),
                opacity=0.85,
            ),
            name="3D Points",
            hovertemplate="X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>",
        )
        fig.add_trace(scatter)
    else:
        hex_map = SemanticClass.get_hex_color_map()
        colors = [hex_map.get(int(l), "#7F8C8D") for l in sub_lbl]
        scatter = go.Scatter3d(
            x=sub_pts[:, 0],
            y=sub_pts[:, 1],
            z=sub_pts[:, 2],
            mode="markers",
            marker=dict(size=2.5, color=colors, opacity=0.85),
            name="Semantic Points",
            hovertemplate="X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>",
        )
        fig.add_trace(scatter)

    # Ego vehicle puck at origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=8, color="#F5A623", symbol="circle"),
        name="LiDAR Sensor",
        hoverinfo="name",
    ))

    camera_presets = {
        "Isometric": dict(eye=dict(x=1.35, y=1.35, z=1.15)),
        "Perspective": dict(eye=dict(x=1.80, y=0.00, z=0.55)),
        "Top-Down": dict(eye=dict(x=0.01, y=0.01, z=2.60)),
        "Chase": dict(eye=dict(x=-1.80, y=0.00, z=0.65)),
    }
    cam = camera_presets.get(camera_preset, camera_presets["Isometric"])

    fig.update_layout(
        scene=dict(
            xaxis_title="Forward X (m)",
            yaxis_title="Lateral Y (m)",
            zaxis_title="Vertical Z (m)",
            aspectmode="data",
            camera=cam,
            bgcolor=bg_color,
            xaxis=dict(gridcolor="#1E293B" if dark_theme else "#E2E8F0"),
            yaxis=dict(gridcolor="#1E293B" if dark_theme else "#E2E8F0"),
            zaxis=dict(gridcolor="#1E293B" if dark_theme else "#E2E8F0"),
        ),
        template="plotly_dark" if dark_theme else "plotly_white",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        height=680,
        margin=dict(l=0, r=0, t=10, b=0),
    )

    return fig


# ==============================================================================
# 3. VERIFICATION AND DEMO RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("SIH26053: Testing visualizer.py & Multi-Resolution 2D Dashboard")
    print("=" * 75)

    # 1. Generate Synthetic LiDAR Frame
    generator = SyntheticLidarGenerator(seed=42)
    engine = AdaptiveGridEngine()
    visualizer = FoveatedGridVisualizer(dark_theme=True)

    frame = generator.generate_frame(frame_id=0, timestamp=0.0)
    print(f"[+] Ingested Frame 0 with {frame.num_points:,} LiDAR points.")

    # 2. Build Foveated 2.5D Grid
    foveated_map = engine.generate_grid(frame)
    print(f"[+] Generated Foveated 2.5D Map in {foveated_map.processing_time_ms:.2f} ms.")

    # 3. Export Standalone Top-Down Semantic View
    out_topdown_png = "output_foveated_topdown.png"
    visualizer.render_topdown(
        foveated_map=foveated_map,
        title="SIH26053: Foveated 2.5D Semantic LiDAR Grid Map",
        save_path=out_topdown_png,
        show=False,
    )

    # 4. Export Complete 4-Panel Autonomous Dashboard
    out_dashboard_png = "output_foveated_dashboard.png"
    visualizer.render_dashboard(
        foveated_map=foveated_map,
        raw_frame=frame,
        save_path=out_dashboard_png,
        show=False,
    )

    # 5. Export Standalone Interactive 2.5D Pillar Grid (HTML)
    out_pillar_html = "output_25d_pillar_grid.html"
    fig_pillar = create_25d_pillar_grid_figure(
        foveated_map=foveated_map,
        color_mode="elevation",
        height_source="z_max",
        show_wireframe=True,
    )
    fig_pillar.write_html(out_pillar_html)

    # 6. Verify File Generation
    assert os.path.exists(out_topdown_png), "Topdown PNG was not generated!"
    assert os.path.exists(out_dashboard_png), "Dashboard PNG was not generated!"
    assert os.path.exists(out_pillar_html), "Pillar Grid HTML was not generated!"

    topdown_size_kb = os.path.getsize(out_topdown_png) / 1024
    dashboard_size_kb = os.path.getsize(out_dashboard_png) / 1024
    pillar_size_kb = os.path.getsize(out_pillar_html) / 1024

    print(f"\n[+] Visualization Output Artifacts:")
    print(f"    - Standalone Top-Down View    : {out_topdown_png} ({topdown_size_kb:.1f} KB)")
    print(f"    - 4-Panel Telemetry System    : {out_dashboard_png} ({dashboard_size_kb:.1f} KB)")
    print(f"    - Interactive 2.5D Pillar Grid: {out_pillar_html} ({pillar_size_kb:.1f} KB)")

    print("\n" + "=" * 75)
    print("All visualizer.py unit tests & render pipelines passed successfully!")
    print("=" * 75)
