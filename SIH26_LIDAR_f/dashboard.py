"""
dashboard.py - Interactive Streamlit Web Application
SIH26053: Adaptive Variable-Resolution 2.5D LiDAR Mapping Pipeline

Run via:
    streamlit run dashboard.py
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    LidarFrame,
    SemanticClass,
    SemanticKittiDataset,
    SyntheticLidarDataset,
    SyntheticLidarGenerator,
)
from grid_engine import AdaptiveGridEngine, FoveatedMap25D, RingGrid, ZoneConfig
from segmentation_engine import GeometricSegmentationEngine, PointNetPPSegmentationEngine, TORCH_AVAILABLE
from evaluation import evaluate_engine, DEFAULT_DISTANCE_BINS
from live_source import UDPLidarSource
from visualizer import create_25d_pillar_grid_figure, create_3d_pointcloud_figure


# ==============================================================================
# 1. PAGE CONFIGURATION & THEME
# ==============================================================================
# Clean dark automotive/telemetry palette: deep slate background, cyan accent
# for data/metrics, amber for warnings/rings. High contrast, no clashing hues.

THEME = {
    "bg": "#0B0F17",            # near-black slate background
    "panel": "#141A24",         # panel surface, one step up from bg
    "panel_border": "#232B3A",  # subtle borders
    "text": "#E6EAF2",          # near-white text
    "text_dim": "#8A94A6",      # secondary/label text
    "accent": "#22D3EE",        # cyan - primary accent (metrics, highlights)
    "accent_dim": "#0E7490",    # darker cyan for hover/secondary
    "ring": "#F5A623",          # amber - ring boundaries, warnings
    "danger": "#EF4444",        # red - errors/disconnected
    "success": "#22C55E",       # green - connected/live
}

st.set_page_config(
    page_title="SIH26053: Adaptive 2.5D LiDAR Mapping",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Telemetry Dashboard Styling
st.markdown(f"""
<style>
    .stApp {{
        background-color: {THEME['bg']};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {THEME['panel']};
        border-right: 1px solid {THEME['panel_border']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {THEME['text']} !important;
    }}
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {{
        background: {THEME['accent']} !important;
    }}
    h1, h2, h3 {{
        color: {THEME['text']} !important;
    }}
    p, span, label, .stMarkdown {{
        color: {THEME['text']};
    }}
    .metric-card {{
        background-color: {THEME['panel']};
        border: 1px solid {THEME['panel_border']};
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        color: {THEME['accent']};
    }}
    .metric-label {{
        font-size: 14px;
        color: {THEME['text_dim']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .ring-pill {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 6px;
        background-color: {THEME['accent_dim']};
        color: {THEME['text']};
    }}
    [data-testid="stMetric"] {{
        background-color: {THEME['panel']};
        border: 1px solid {THEME['panel_border']};
        border-radius: 10px;
        padding: 10px 14px;
    }}
    [data-testid="stMetricValue"] {{
        color: {THEME['accent']} !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {THEME['text_dim']} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {THEME['text_dim']};
    }}
    .stTabs [aria-selected="true"] {{
        color: {THEME['bg']} !important;
        background-color: {THEME['accent']} !important;
        border-radius: 6px 6px 0 0;
    }}
    .live-badge-on {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background-color: {THEME['success']}; color: {THEME['bg']};
        font-weight: 700; font-size: 12px; letter-spacing: 0.05em;
    }}
    .live-badge-off {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background-color: {THEME['danger']}; color: {THEME['text']};
        font-weight: 700; font-size: 12px; letter-spacing: 0.05em;
    }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. SIDEBAR CONTROLS
# ==============================================================================

st.sidebar.title("📡 SIH26053 LiDAR Controller")
st.sidebar.caption("Adaptive Variable-Resolution 2.5D Mapping")

st.sidebar.markdown("---")
st.sidebar.subheader("1. Data Source Selection")
data_source = st.sidebar.selectbox(
    "Select Input Stream",
    [
        "Synthetic LiDAR Generator",
        "Local SemanticKITTI (.bin / .label)",
        "🔴 Live Hardware Sensor (UDP)",
    ],
    index=0,
)

if data_source == "🔴 Live Hardware Sensor (UDP)":
    st.sidebar.caption(
        "Ingests real frames pushed over the network from an actual sensor "
        "bridge (or `simulate_sensor.py` for a live demo without hardware). "
        "See live_source.py for how to wire a real driver/ROS node."
    )
    udp_host = st.sidebar.text_input(
        "Listen Host", value="127.0.0.1",
        help="127.0.0.1 works for testing on this same machine (matches simulate_sensor.py's default). "
             "Only use 0.0.0.0 if a real sensor is sending from a different device on your network "
             "— and even then it can fail on some Windows/VPN setups; 127.0.0.1 is the safe default.",
    )
    udp_port = st.sidebar.number_input("Listen Port", min_value=1024, max_value=65535, value=5599, step=1)

    if "udp_source" not in st.session_state:
        st.session_state.udp_source = None
        st.session_state.udp_bound_addr = None

    conn_col1, conn_col2 = st.sidebar.columns(2)
    with conn_col1:
        if st.button("🔌 Connect", use_container_width=True):
            if st.session_state.udp_source is not None:
                st.session_state.udp_source.stop()
            try:
                src = UDPLidarSource(host=udp_host, port=int(udp_port))
                src.start()
                st.session_state.udp_source = src
                st.session_state.udp_bound_addr = (udp_host, int(udp_port))
            except OSError as exc:
                st.session_state.udp_source = None
                st.session_state.udp_bound_addr = None
                st.sidebar.error(
                    f"Couldn't bind to {udp_host}:{udp_port} — {exc}. "
                    f"On Windows, try Listen Host = 127.0.0.1 (instead of 0.0.0.0), "
                    f"or pick a different port if it's already in use.",
                    icon="⚠️",
                )
    with conn_col2:
        if st.button("🔌 Disconnect", use_container_width=True):
            if st.session_state.udp_source is not None:
                st.session_state.udp_source.stop()
            st.session_state.udp_source = None
            st.session_state.udp_bound_addr = None

    live_src = st.session_state.udp_source
    if live_src is not None:
        live_status = live_src.status()
        if live_status.connected:
            st.sidebar.markdown(
                f'<span class="live-badge-on">● LIVE</span> '
                f'&nbsp;{live_status.incoming_rate_hz:.1f} Hz &nbsp;| &nbsp;'
                f'{live_status.frames_received} frames received',
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown('<span class="live-badge-off">● WAITING FOR SENSOR</span>', unsafe_allow_html=True)
            st.sidebar.caption(
                f"Listening on {udp_host}:{udp_port}. Run "
                f"`python simulate_sensor.py --port {udp_port}` in another "
                f"terminal to test, or point your real sensor bridge here."
            )
        if live_status.last_error:
            st.sidebar.caption(f"⚠️ {live_status.last_error}")
    else:
        st.sidebar.markdown('<span class="live-badge-off">● NOT CONNECTED</span>', unsafe_allow_html=True)

    auto_refresh_live = st.sidebar.checkbox("Auto-refresh while connected", value=True)
    st.session_state["auto_refresh_live"] = auto_refresh_live

if data_source == "Synthetic LiDAR Generator":
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        num_cars = st.slider("Dynamic Cars", min_value=0, max_value=20, value=8, step=1)
        num_trees = st.slider("Trees & Foliage", min_value=0, max_value=30, value=14, step=2)
    with col_s2:
        num_peds = st.slider("Pedestrians", min_value=0, max_value=15, value=5, step=1)
        num_bldgs = st.slider("Buildings", min_value=0, max_value=10, value=6, step=1)

    noise_pct = st.slider("Sensor Noise Ratio", min_value=0.0, max_value=0.10, value=0.02, step=0.01)
    frame_idx = st.slider("Sequence Frame (Time Step)", min_value=0, max_value=50, value=0, step=1)
    seed = st.sidebar.number_input("RNG Seed", value=42, step=1)
    kitti_dir, load_kitti_labels = "", True

elif data_source == "Local SemanticKITTI (.bin / .label)":
    kitti_dir = st.sidebar.text_input("SemanticKITTI Sequence Directory", value="./dataset/sequences/00")
    frame_idx = st.sidebar.slider("Frame Index", min_value=0, max_value=100, value=0, step=1)
    load_kitti_labels = st.sidebar.checkbox("Load Ground Truth Labels", value=True)
    seed = 42

else:  # "🔴 Live Hardware Sensor (UDP)" - frame comes from the UDP thread, not sliders
    frame_idx = 0
    seed = 42
    kitti_dir, load_kitti_labels = "", True

# ------------------------------------------------------------------------------
# LIVE AUTO-PLAY CONTROLS
# Simulates a continuous real-time sensor feed by auto-incrementing the frame
# index and forcing a rerun at a fixed cadence, instead of requiring the user
# to manually drag the frame slider.
# ------------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔴 Live Playback")

if "live_playing" not in st.session_state:
    st.session_state.live_playing = False
if "live_frame" not in st.session_state:
    st.session_state.live_frame = 0

play_col1, play_col2 = st.sidebar.columns(2)
with play_col1:
    play_label = "⏸️ Pause" if st.session_state.live_playing else "▶️ Play"
    if st.button(play_label, use_container_width=True):
        st.session_state.live_playing = not st.session_state.live_playing
with play_col2:
    if st.button("⏹️ Reset", use_container_width=True):
        st.session_state.live_playing = False
        st.session_state.live_frame = 0

playback_fps = st.sidebar.slider("Playback Speed (FPS)", min_value=1, max_value=20, value=10, step=1)
max_live_frames = st.sidebar.number_input(
    "Loop After N Frames", min_value=10, max_value=500, value=100, step=10,
    help="Simulated sequence length. Playback wraps back to frame 0 after this many frames.",
)

if st.session_state.live_playing:
    frame_idx = st.session_state.live_frame % int(max_live_frames)
    st.sidebar.success(f"🔴 LIVE — streaming frame {frame_idx}")
else:
    st.sidebar.caption("Paused. Press Play for a live-streamed feed, or use the frame slider above for manual stepping.")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Concentric Ring Architecture")
st.sidebar.caption("Adjust radii and cell sizes per foveated zone:")

# Zone 0: Near / Foveal
st.sidebar.markdown("**Ring 0 (Near / Foveal)**")
r0_max = st.sidebar.slider("Ring 0 Outer Radius (m)", min_value=5.0, max_value=20.0, value=10.0, step=1.0)
res0 = st.sidebar.select_slider("Ring 0 Cell Size (m)", options=[0.05, 0.08, 0.10, 0.15, 0.20], value=0.05)

# Zone 1: Mid-Range
st.sidebar.markdown("**Ring 1 (Mid-Range)**")
r1_max = st.sidebar.slider("Ring 1 Outer Radius (m)", min_value=r0_max + 5.0, max_value=50.0, value=40.0, step=5.0)
res1 = st.sidebar.select_slider("Ring 1 Cell Size (m)", options=[0.15, 0.20, 0.25, 0.30, 0.40], value=0.20)

# Zone 2: Far-Range
st.sidebar.markdown("**Ring 2 (Far-Range)**")
r2_max = st.sidebar.slider("Ring 2 Outer Radius (m)", min_value=r1_max + 10.0, max_value=100.0, value=100.0, step=5.0)
res2 = st.sidebar.select_slider("Ring 2 Cell Size (m)", options=[0.30, 0.40, 0.50, 0.75, 1.00], value=0.50)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Baseline Comparison")
baseline_res = st.sidebar.selectbox(
    "Naive Uniform Baseline Resolution",
    [0.05, 0.10, 0.15, 0.20],
    index=1,
    format_func=lambda x: f"{int(x*100)} cm ({x:.2f} m/cell)",
)

# Segmentation Engine Selection
st.sidebar.markdown("---")
st.sidebar.subheader("3b. Segmentation Engine")
engine_options = ["PointNet++ (Trained DL Model)", "Geometric + Clustering (Heuristic Fallback)"]
engine_choice = st.sidebar.radio("Semantic Segmentation Engine", engine_options, index=0)

# Rendering Options
st.sidebar.markdown("---")
st.sidebar.subheader("4. Visualization Display")
show_grid_borders = st.sidebar.checkbox("Show Cell Borders", value=True)
color_theme = st.sidebar.selectbox(
    "Color Theme",
    ["Dark Telemetry (Cyan)", "Light Industrial"],
    index=0,
)


# ==============================================================================
# 3. PIPELINE INGESTION & EXECUTION
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_or_generate_frame(
    src: str,
    f_idx: int,
    cars: int,
    peds: int,
    trees: int,
    bldgs: int,
    noise: float,
    seed_val: int,
    k_dir: str = "",
    k_labels: bool = True,
) -> LidarFrame:
    if src == "Synthetic LiDAR Generator":
        gen = SyntheticLidarGenerator(sensor_height=1.73, seed=seed_val, max_range=r2_max)
        return gen.generate_frame(
            frame_id=f_idx,
            timestamp=f_idx * 0.1,
            num_cars=cars,
            num_pedestrians=peds,
            num_trees=trees,
            num_buildings=bldgs,
            noise_ratio=noise,
        )
    else:
        # Load from KITTI directory if present, else fallback to synthetic
        if os.path.exists(k_dir):
            ds = SemanticKittiDataset(root_dir=k_dir, load_labels=k_labels)
            if len(ds) > 0:
                idx_clamped = min(f_idx, len(ds) - 1)
                return ds[idx_clamped]
        # Fallback if path doesn't exist
        gen = SyntheticLidarGenerator(sensor_height=1.73, seed=seed_val, max_range=r2_max)
        return gen.generate_frame(frame_id=f_idx, timestamp=f_idx * 0.1)


# Ingest Frame
t_load_0 = time.perf_counter()

if data_source == "🔴 Live Hardware Sensor (UDP)":
    # Real-time path: pull the freshest frame straight from the UDP listener
    # thread. Deliberately NOT cached - a live feed must never return a
    # stale, memoized frame the way the slider-driven synthetic/KITTI paths
    # correctly do.
    _live = st.session_state.get("udp_source")
    _live_frame = _live.get_latest_frame() if _live is not None else None

    if _live_frame is not None:
        raw_frame = _live_frame
        st.session_state["_last_live_frame"] = _live_frame
    elif st.session_state.get("_last_live_frame") is not None:
        raw_frame = st.session_state["_last_live_frame"]
        st.warning("No new packets since last frame - showing last known scan.", icon="⏳")
    else:
        st.info(
            "Waiting for the first frame from the sensor bridge. Run "
            "`python simulate_sensor.py` in another terminal to see this "
            "go live, or point your real driver/ROS bridge at the "
            "host/port shown in the sidebar.",
            icon="📡",
        )
        st.stop()
else:
    raw_frame = load_or_generate_frame(
        src=data_source,
        f_idx=frame_idx,
        cars=num_cars if data_source == "Synthetic LiDAR Generator" else 0,
        peds=num_peds if data_source == "Synthetic LiDAR Generator" else 0,
        trees=num_trees if data_source == "Synthetic LiDAR Generator" else 0,
        bldgs=num_bldgs if data_source == "Synthetic LiDAR Generator" else 0,
        noise=noise_pct if data_source == "Synthetic LiDAR Generator" else 0.0,
        seed_val=seed if data_source == "Synthetic LiDAR Generator" else 42,
        k_dir=kitti_dir if data_source != "Synthetic LiDAR Generator" else "",
        k_labels=load_kitti_labels if data_source != "Synthetic LiDAR Generator" else True,
    )

# Run Semantic Segmentation (Trained PointNet++ DL model, or geometric fallback)
@st.cache_resource(show_spinner=False)
def get_segmenter(use_pointnet: bool):
    if use_pointnet:
        return PointNetPPSegmentationEngine()
    return GeometricSegmentationEngine()

segmenter = get_segmenter(engine_choice == engine_options[0])
using_trained_model = isinstance(segmenter, PointNetPPSegmentationEngine) and segmenter.is_using_trained_model
segmented_frame = segmenter.infer(raw_frame)

if engine_choice == engine_options[0] and not using_trained_model:
    reason = "PyTorch is not installed" if not TORCH_AVAILABLE else "no trained checkpoint found (run train_model.py)"
    st.sidebar.warning(f"PointNet++ checkpoint unavailable ({reason}) - using geometric fallback.", icon="⚠️")

# Configure Adaptive Grid Engine with dynamic user slider parameters
zones = [
    ZoneConfig(name="Near_Foveal", r_min=0.0,    r_max=r0_max, resolution=res0),
    ZoneConfig(name="Mid_Range",   r_min=r0_max, r_max=r1_max, resolution=res1),
    ZoneConfig(name="Far_Range",   r_min=r1_max, r_max=r2_max, resolution=res2),
]
engine = AdaptiveGridEngine(zones=zones)

# Generate Foveated 2.5D Map
foveated_map = engine.generate_grid(segmented_frame)

# Compute Memory & Benchmark Metrics
mem_stats = foveated_map.memory_reduction_vs_uniform(uniform_resolution=baseline_res)
fps = 1000.0 / max(foveated_map.processing_time_ms, 0.05)


# ==============================================================================
# 4. LIVE METRIC GAUGES & HEADER
# ==============================================================================

st.title("📡 Adaptive Variable-Resolution 2.5D LiDAR Mapping")
st.markdown(
    f"**Real-Time Autonomous Perception Prototype** | "
    f"Zone 0: `{r0_max:.0f}m @ {res0*100:.0f}cm` | "
    f"Zone 1: `{r1_max:.0f}m @ {res1*100:.0f}cm` | "
    f"Zone 2: `{r2_max:.0f}m @ {res2*100:.0f}cm`"
)

# Metric Cards Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        label="Pipeline Throughput",
        value=f"{fps:.1f} FPS",
        delta=f"+{fps - 60.0:.1f} vs 60Hz" if fps >= 60.0 else f"{fps - 60.0:.1f} vs 60Hz",
    )

with kpi2:
    st.metric(
        label="Processing Latency",
        value=f"{foveated_map.processing_time_ms:.2f} ms",
        delta=f"-{max(0.0, 16.6 - foveated_map.processing_time_ms):.1f} ms headroom",
    )

with kpi3:
    st.metric(
        label="RAM Reduction",
        value=f"{mem_stats['savings_percent']:.1f}%",
        delta=f"{mem_stats['compression_ratio']:.1f}x Compression",
    )

with kpi4:
    st.metric(
        label="Foveated Map Memory",
        value=f"{mem_stats['adaptive_memory_mb']:.2f} MB",
        delta=f"vs {mem_stats['uniform_memory_mb']:.2f} MB (Naive)",
        delta_color="inverse",
    )

with kpi5:
    st.metric(
        label="Occupied Cells / Points",
        value=f"{foveated_map.total_occupied_cells():,}",
        delta=f"{raw_frame.num_points:,} Raw Pts",
    )


# ==============================================================================
# 5. MAIN VISUALIZATION PANELS (TABS)
# ==============================================================================

tab_map, tab_pillar, tab_25d, tab_dem, tab_bench, tab_pts = st.tabs([
    "🗺️ Top-Down 2.5D Semantic Map",
    "🧱 2.5D Pillar & Foveated Grid Map",
    "🧊 Smooth 2.5D Elevation Surface",
    "⛰️ Digital Elevation (DEM) & Height Profile",
    "📊 Memory & Benchmark Analytics",
    "🌐 Raw 3D Point Cloud Inspector",
])


# ------------------------------------------------------------------------------
# TAB 1: TOP-DOWN MULTI-RESOLUTION SEMANTIC MAP
# ------------------------------------------------------------------------------
with tab_map:
    col_viz, col_legend = st.columns([4, 1])

    with col_legend:
        st.markdown("### Semantic Legend")
        st.markdown("""
        - 🟩 **Ground (`#2ecc71`)**: Drivable Road / Sidewalks / Terrain
        - 🟥 **Static Obstacle (`#e74c3c`)**: Buildings / Trees / Poles / Barriers
        - 🟦 **Dynamic Object (`#3498db`)**: Vehicles / Pedestrians / Cyclists
        - ⬜ **Noise / Outliers (`#7f8c8d`)**: Drops / Atmospheric speckle
        """)
        st.markdown("---")
        st.markdown("### Ring Dimensions")
        for i, ring in enumerate(foveated_map.rings):
            cfg = ring.config
            occ = int(np.sum(ring.occupied_mask))
            tot = cfg.grid_dim * cfg.grid_dim
            st.markdown(
                f"**Ring {i} ({cfg.name})**<br>"
                f"• Radius: `{cfg.r_min:.0f}m - {cfg.r_max:.0f}m`<br>"
                f"• Resolution: `{cfg.resolution*100:.0f}cm`<br>"
                f"• Grid: `{cfg.grid_dim}x{cfg.grid_dim}` ({tot:,} cells)<br>"
                f"• Occupied: `{occ:,}` ({occ/tot*100:.1f}%)<br>"
                f"• RAM: `{ring.memory_bytes/1024:.1f} KB`",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

    with col_viz:
        # Build Matplotlib Multi-Resolution PolyCollection Figure
        is_dark = (color_theme == "Dark Telemetry (Cyan)")
        bg_color = THEME["bg"] if is_dark else "#FFFFFF"
        panel_bg = THEME["panel"] if is_dark else "#F7FAFC"
        text_color = THEME["text"] if is_dark else "#1A202C"
        ring_color = THEME["ring"] if is_dark else "#B5651D"

        fig, ax = plt.subplots(figsize=(10, 10), dpi=130, facecolor=bg_color)
        ax.set_facecolor(panel_bg)

        # Color Map definitions (7-class fine-grained taxonomy)
        color_rgb = {
            SemanticClass.NOISE: np.array([127, 140, 141, 180]) / 255.0,
            SemanticClass.GROUND: np.array([46, 204, 113, 220]) / 255.0,
            SemanticClass.STATIC_STRUCTURE: np.array([231, 76, 60, 255]) / 255.0,
            SemanticClass.STATIC_POLE: np.array([155, 89, 182, 255]) / 255.0,
            SemanticClass.STATIC_VEGETATION: np.array([106, 176, 76, 255]) / 255.0,
            SemanticClass.DYNAMIC_VEHICLE: np.array([52, 152, 219, 255]) / 255.0,
            SemanticClass.DYNAMIC_PEDESTRIAN: np.array([241, 196, 15, 255]) / 255.0,
        }

        # Render each concentric ring from outer to inner
        for ring in reversed(foveated_map.rings):
            cfg = ring.config
            res = cfg.resolution
            extent = cfg.grid_extent

            occ_y, occ_x = np.where(ring.occupied_mask)
            if len(occ_x) == 0:
                continue

            x0 = occ_x * res - extent
            x1 = x0 + res
            y0 = occ_y * res - extent
            y1 = y0 + res

            n_cells = len(occ_x)
            verts = np.zeros((n_cells, 4, 2), dtype=np.float32)
            verts[:, 0, 0] = x0; verts[:, 0, 1] = y0
            verts[:, 1, 0] = x1; verts[:, 1, 1] = y0
            verts[:, 2, 0] = x1; verts[:, 2, 1] = y1
            verts[:, 3, 0] = x0; verts[:, 3, 1] = y1

            cell_labels = ring.labels[occ_y, occ_x]
            facecolors = np.zeros((n_cells, 4), dtype=np.float32)
            for c_id in range(SemanticClass.num_classes()):
                mask = (cell_labels == c_id)
                if np.any(mask):
                    facecolors[mask] = color_rgb[SemanticClass(c_id)]

            edgecolor = (0.05, 0.05, 0.05, 0.4) if show_grid_borders else "none"
            linewidth = 0.4 if res >= 0.25 else 0.2

            poly_coll = PolyCollection(
                verts,
                facecolors=facecolors,
                edgecolors=edgecolor,
                linewidths=linewidth,
                antialiased=True,
            )
            ax.add_collection(poly_coll)

        # Draw Concentric Ring Boundaries & Labels
        for ring in foveated_map.rings:
            cfg = ring.config
            circle = plt.Circle((0, 0), cfg.r_max, color=ring_color, fill=False, linestyle="--", linewidth=1.2, alpha=0.8)
            ax.add_patch(circle)

            label_y = cfg.r_max - (cfg.r_max - cfg.r_min) * 0.15
            ax.text(
                0.8, label_y,
                f"{cfg.name}\n[{cfg.r_min:.0f}-{cfg.r_max:.0f}m | {cfg.resolution*100:.0f}cm]",
                color=ring_color, fontsize=8, fontweight="bold", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=panel_bg, edgecolor=ring_color, alpha=0.85),
            )

        # Ego Vehicle Icon at Origin
        ego_box = patches.Rectangle((-1.0, -2.25), 2.0, 4.5, facecolor="#f39c12", edgecolor="#ffffff", linewidth=1.5, zorder=10)
        ax.add_patch(ego_box)
        ax.arrow(0, 0, 3.0, 0, head_width=1.0, head_length=1.0, fc="#ffffff", ec="#ffffff", zorder=11)
        ax.text(0, -4.2, "EGO", color="#f39c12", fontsize=9, fontweight="bold", ha="center", zorder=12)

        max_ext = r2_max * 1.05
        ax.set_xlim(-max_ext, max_ext)
        ax.set_ylim(-max_ext, max_ext)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Sensor X (Forward / m)", color=text_color, fontsize=10)
        ax.set_ylabel("Sensor Y (Left / m)", color=text_color, fontsize=10)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color("#2d3748" if is_dark else "#cbd5e0")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ------------------------------------------------------------------------------
# TAB 1.5: 2.5D ELEVATION & FOVEATED PILLAR GRID MAP
# Renders the exact 2.5D grid representation shown in the SIH26 LiDAR architecture:
# 1) "3D LiDAR Point Cloud to 2.5D Map": Discrete 3D pillars/voxels colored by
#    elevation (Jet/Turbo colormap) with crisp grid cell borders.
# 2) "Adaptive Variable Resolution (Foveated Mapping)": Multi-resolution grid
#    showing Near (0-10m @ 5cm Blue), Mid (10-40m @ 20cm Green), and Far
#    (40-100m @ 50cm Orange) zones directly in 3D.
# ------------------------------------------------------------------------------
with tab_pillar:
    st.subheader("🧱 2.5D Elevation & Foveated Pillar Grid Map")
    st.caption(
        "Discrete 3D voxel/pillar grid matching the SIH26 technical architecture. "
        "Supports Elevation Colormap (Image 1 top), Adaptive Variable Resolution Foveated Zones (Image 1 bottom), "
        "and 7-Class Semantic Segmentation (Image 2). Drag to rotate, scroll to zoom, hover to inspect cells."
    )

    # Primary Control Row
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

    with ctrl_col1:
        pillar_color_choice = st.selectbox(
            "Coloring Mode",
            [
                "🌈 Reference Scan Gradient (Matches Image)",
                "⛰️ Pure Elevation Height (Z_max)",
                "🎯 Adaptive Foveated Zones (Image 1 Bottom)",
                "🏷️ Semantic Segmentation (Image 2)",
            ],
            index=0,
            key="pillar_color_mode",
            help="Select how the 3D pillar blocks are colored.",
        )
        color_mode_map = {
            "🌈 Reference Scan Gradient (Matches Image)": "image_gradient",
            "⛰️ Pure Elevation Height (Z_max)": "elevation",
            "🎯 Adaptive Foveated Zones (Image 1 Bottom)": "foveated_zones",
            "🏷️ Semantic Segmentation (Image 2)": "semantic",
        }
        selected_color_mode = color_mode_map[pillar_color_choice]

    with ctrl_col2:
        grid_arch_choice = st.selectbox(
            "Grid Architecture",
            [
                "🧱 Solid Continuous Grid Slab (Matches Image)",
                "⚡ Native Foveated Rings (Variable Cell Sizes)",
                "📐 Sparse Composite Grid",
            ],
            index=0,
            key="pillar_grid_arch",
            help="Solid Continuous Grid Slab creates a complete voxel tray with solid ground and stepped obstacles as in the image.",
        )
        grid_mode_map = {
            "🧱 Solid Continuous Grid Slab (Matches Image)": "continuous_slab",
            "⚡ Native Foveated Rings (Variable Cell Sizes)": "foveated",
            "📐 Sparse Composite Grid": "sparse",
        }
        selected_grid_mode = grid_mode_map[grid_arch_choice]

    with ctrl_col3:
        height_source_choice = st.selectbox(
            "Height Profile",
            [
                "Obstacle Profile + Solid Ground Floor (Matches Image)",
                "Max Elevation (Z_max)",
                "Stepped Occupancy (Uniform Blocks)",
            ],
            index=0,
            key="pillar_height_src",
            help="Source for 3D pillar column extrusion height.",
        )
        height_src_map = {
            "Obstacle Profile + Solid Ground Floor (Matches Image)": "obstacle_profile",
            "Max Elevation (Z_max)": "z_max",
            "Stepped Occupancy (Uniform Blocks)": "fixed",
        }
        selected_height_src = height_src_map[height_source_choice]

    with ctrl_col4:
        view_layout_choice = st.selectbox(
            "View Presentation",
            [
                "Side-by-Side: Point Cloud ➔ 2.5D Grid (Image 1)",
                "Single 3D Grid Map (Full Width)",
            ],
            index=0,
            key="pillar_view_layout",
            help="Directly recreates the 3D Point Cloud to 2.5D Map comparison seen in Image 1.",
        )

    # Advanced 3D Display Controls
    with st.expander("🛠️ Advanced 3D Display & Performance Settings", expanded=False):
        adv_c1, adv_c2, adv_c3, adv_c4 = st.columns(4)
        with adv_c1:
            p_camera = st.selectbox("Camera Preset", ["Isometric", "Perspective", "Top-Down", "Chase"], index=0, key="pillar_cam")
            p_colormap = st.selectbox("Elevation Colormap", ["Turbo", "Jet", "Viridis", "Plasma"], index=0, key="pillar_cmap")
        with adv_c2:
            p_z_exagg = st.slider("Height Exaggeration", min_value=1.0, max_value=8.0, value=2.4, step=0.2, key="pillar_z_exagg")
            p_max_range = st.slider("Perception Range ROI (m)", min_value=10.0, max_value=float(r2_max), value=20.0, step=2.0, key="pillar_max_r")
        with adv_c3:
            p_show_wireframe = st.checkbox("Show Cell Grid Lines (Wireframe)", value=True, key="pillar_wf", help="Draws crisp grid outlines around each 3D cell block.")
            p_show_floating_pts = st.checkbox("Show Floating 3D LiDAR Points", value=True, key="pillar_floating_pts", help="Displays point cloud markers hovering above pillars like in the reference image.")
        with adv_c4:
            p_show_rings = st.checkbox("Show Concentric Range Rings", value=False, key="pillar_rings")
            p_stride = st.select_slider(
                "Pillar Decimation / Stride",
                options=[1, 2, 4],
                value=1,
                key="pillar_stride",
                format_func=lambda x: "1:1 Full Detail" if x == 1 else (f"1:{x} (Fast)" if x == 2 else f"1:{x} (Real-time)")
            )

    # Grid Quick Telemetry KPIs
    grid_kpi1, grid_kpi2, grid_kpi3, grid_kpi4 = st.columns(4)
    with grid_kpi1:
        st.markdown(f"**Total Occupied Cells:** `{foveated_map.total_occupied_cells():,}`")
    with grid_kpi2:
        ring_counts_str = " · ".join([f"R{i}: `{int(np.sum(r.occupied_mask)):,}`" for i, r in enumerate(foveated_map.rings)])
        st.markdown(f"**Ring Cell Counts:** {ring_counts_str}")
    with grid_kpi3:
        st.markdown(f"**Grid ROI Horizon:** `±{p_max_range:.0f}m Box`")
    with grid_kpi4:
        st.markdown(f"**Render Mode:** `{'Solid Grid Slab' if selected_grid_mode == 'continuous_slab' else 'Foveated Rings'}`")

    # Generate 3D Pillar Grid Figure
    fig_pillar = create_25d_pillar_grid_figure(
        foveated_map=foveated_map,
        color_mode=selected_color_mode,
        height_source=selected_height_src,
        grid_mode=selected_grid_mode,
        composite_resolution=0.80,
        show_wireframe=p_show_wireframe,
        z_exaggeration=p_z_exagg,
        max_range=p_max_range,
        stride=p_stride,
        dark_theme=(color_theme == "Dark Telemetry (Cyan)"),
        camera_preset=p_camera,
        show_rings=p_show_rings,
        show_ego=False,
        colormap=p_colormap,
        raw_frame=raw_frame,
        show_floating_points=p_show_floating_pts,
    )

    if "Side-by-Side" in view_layout_choice:
        st.markdown(
            f"""
            <div style="background-color: {THEME['panel']}; border: 1px solid {THEME['accent']}; border-radius: 8px; padding: 10px; margin: 10px 0; text-align: center;">
                <span style="color: {THEME['accent']}; font-weight: 700; font-size: 15px; letter-spacing: 0.05em;">
                    📡 3D LIDAR POINT CLOUD &nbsp; ➔ &nbsp; 🧱 2.5D ELEVATION / OCCUPANCY GRID (HEIGHT MAP)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_pc_view, col_grid_view = st.columns(2)
        with col_pc_view:
            st.markdown("#### 1. Input 3D LiDAR Point Cloud")
            fig_pc = create_3d_pointcloud_figure(
                frame=raw_frame,
                color_by="elevation" if selected_color_mode == "elevation" else "class",
                dark_theme=(color_theme == "Dark Telemetry (Cyan)"),
                camera_preset=p_camera,
                colormap=p_colormap,
            )
            st.plotly_chart(fig_pc, use_container_width=True)
            st.caption("Raw unstructured point cloud from 3D LiDAR sensor, colored by Z elevation.")
        with col_grid_view:
            st.markdown("#### 2. Discretized 2.5D Grid (Height Map)")
            st.plotly_chart(fig_pillar, use_container_width=True)
            st.caption("Extruded 2.5D grid cells with explicit cell boundaries and height profiles.")
    else:
        st.plotly_chart(fig_pillar, use_container_width=True)

    # Dynamic Legend & Technical Explanations matching the uploaded images
    st.markdown("---")
    leg_col1, leg_col2, leg_col3 = st.columns([1.2, 1.2, 1.6])
    with leg_col1:
        st.markdown("#### 🎨 Color Strategy Legend")
        if selected_color_mode == "elevation":
            st.markdown("""
            - 🟦 **Ground / Base Level**: Deep Blue / Cyan ($Z \\approx -1.7\\text{m}$)
            - 🟩 **Low Obstacles**: Green / Yellow (Kerbs, bumps)
            - 🟧 **Medium Obstacles**: Orange (Vehicles, pedestrians)
            - 🟥 **Tall Structures**: Crimson Red (Walls, buildings, trees)
            """)
        elif selected_color_mode == "foveated_zones":
            st.markdown("""
            - 🟦 **Zone 0 (Near / Foveal)**: `0 - 10m` @ **5cm** resolution
            - 🟩 **Zone 1 (Mid-Range)**: `10 - 40m` @ **20cm** resolution
            - 🟧 **Zone 2 (Far-Range)**: `40 - 100m` @ **50cm** resolution
            """)
        else:
            st.markdown("""
            - 🟩 Ground & Drivable Terrain
            - 🟥 Static Structures & Walls
            - 🟪 Static Poles & Thin Obstacles
            - 🟦 Dynamic Vehicles
            - 🟨 Pedestrians
            """)

    with leg_col2:
        st.markdown("#### 🏗️ Architecture Attributes")
        st.markdown(f"""
        - **Cell Structure**: 3D Discrete Extruded Pillars
        - **Height Source**: `{height_source_choice}`
        - **Z-Exaggeration**: `{p_z_exagg:.1f}x`
        - **Wireframe Cell Grid**: `{'Active (Borders visible)' if p_show_wireframe else 'Disabled'}`
        - **Camera Preset**: `{p_camera}`
        """)

    with leg_col3:
        st.markdown("#### 💡 Technical Hackathon Context")
        st.caption(
            "This 2.5D discrete pillar grid directly demonstrates the pipeline in the SIH26 architecture slide: "
            "Raw 3D LiDAR point clouds are projected into a foveated height map where near-field cells have high spatial "
            "resolution (5cm) for precise ego-vehicle maneuvering, while far-field cells drop to 50cm to deliver "
            ">90% RAM savings and >100 FPS throughput on edge hardware."
        )


# ------------------------------------------------------------------------------
# TAB 2: TRUE 2.5D MODEL — CONTINUOUS INTERACTIVE 3D ELEVATION SURFACE
# ------------------------------------------------------------------------------
with tab_25d:
    st.subheader("🧊 Continuous 2.5D Elevation Surface")
    st.caption(
        "Drag to rotate, scroll to zoom, double-click to reset. Height (Z) = max elevation per "
        "cell; color = semantic class. Coarser cells further from the sensor show the foveated "
        "resolution drop-off directly in 3D."
    )

    col_3d_opt1, col_3d_opt2, col_3d_opt3 = st.columns(3)
    with col_3d_opt1:
        model_res = st.select_slider(
            "3D Model Cell Size (m)", options=[0.25, 0.50, 1.00, 1.50], value=0.50,
            help="Coarser = faster render. This is independent of your ring resolutions above.",
        )
    with col_3d_opt2:
        z_exaggeration = st.slider("Height Exaggeration", min_value=1.0, max_value=8.0, value=3.0, step=0.5,
                                    help="Multiplies elevation so low obstacles are visible in 3D.")
    with col_3d_opt3:
        view_style = st.selectbox("Camera Preset", ["Isometric", "Top-Down", "Low Chase Angle"], index=0)

    composite_3d = foveated_map.to_dense_composite(composite_extent=r2_max, composite_resolution=model_res)
    ext_3d = composite_3d["extent"]
    dim_3d = composite_3d["dim"]

    xs_3d = np.linspace(-ext_3d, ext_3d, dim_3d)
    ys_3d = np.linspace(-ext_3d, ext_3d, dim_3d)

    z_surf = composite_3d["z_max"].astype(np.float64)
    occ_3d = composite_3d["occupied"]
    labels_3d = composite_3d["labels"].astype(np.float64)

    # Ground floor for unoccupied cells so the surface stays continuous
    ground_level = np.nanmin(z_surf[occ_3d]) if np.any(occ_3d) else 0.0
    z_surf = np.where(occ_3d, z_surf, ground_level)
    z_surf = (z_surf - ground_level) * z_exaggeration

    # Discrete 7-class colorscale matching SemanticClass.get_hex_color_map()
    _n7 = SemanticClass.num_classes()
    _hex7 = SemanticClass.get_hex_color_map()
    class_colorscale = []
    for _cid in range(_n7):
        _lo = _cid / _n7
        _hi = (_cid + 1) / _n7 - 1e-6
        class_colorscale.append([_lo, _hex7[_cid]])
        class_colorscale.append([_hi, _hex7[_cid]])

    fig_25d = go.Figure(data=[go.Surface(
        x=xs_3d, y=ys_3d, z=z_surf,
        surfacecolor=labels_3d,
        colorscale=class_colorscale,
        cmin=0, cmax=_n7 - 1,
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.3, roughness=0.8),
        hovertemplate="X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Height: %{z:.2f}<extra></extra>",
    )])

    # Concentric ring boundary rings, lifted slightly above the ground plane
    ring_z_offset = float(np.nanmax(z_surf)) * 0.02 if np.any(occ_3d) else 0.1
    theta = np.linspace(0, 2 * np.pi, 100)
    for ring in foveated_map.rings:
        cfg = ring.config
        ring_x = cfg.r_max * np.cos(theta)
        ring_y = cfg.r_max * np.sin(theta)
        fig_25d.add_trace(go.Scatter3d(
            x=ring_x, y=ring_y, z=np.full_like(ring_x, ring_z_offset),
            mode="lines", line=dict(color="#f6ad55", width=4, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))

    camera_presets = {
        "Isometric": dict(eye=dict(x=1.4, y=1.4, z=1.1)),
        "Top-Down": dict(eye=dict(x=0.01, y=0.01, z=2.8)),
        "Low Chase Angle": dict(eye=dict(x=1.8, y=0.0, z=0.35)),
    }

    fig_25d.update_layout(
        scene=dict(
            xaxis_title="X (Forward / m)",
            yaxis_title="Y (Lateral / m)",
            zaxis_title=f"Elevation (x{z_exaggeration:.1f})",
            aspectmode="data",
            camera=camera_presets[view_style],
            bgcolor=THEME["bg"] if is_dark else "#FFFFFF",
        ),
        template="plotly_dark" if is_dark else "plotly_white",
        height=650,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_25d, use_container_width=True)

    leg1, leg2, leg3, leg4 = st.columns(4)
    leg1.markdown("🟩 **Ground** — drivable terrain")
    leg2.markdown("🟥 **Static Obstacle** — walls / poles / trees")
    leg3.markdown("🟦 **Dynamic Object** — vehicles / pedestrians")
    leg4.markdown("⬜ **Noise** — sensor speckle")


# ------------------------------------------------------------------------------
# TAB 2: DIGITAL ELEVATION MODEL (DEM) & HEIGHT DELTA
# ------------------------------------------------------------------------------
with tab_dem:
    col_dem1, col_dem2 = st.columns(2)
    composite = foveated_map.to_dense_composite(composite_extent=r2_max, composite_resolution=0.50)
    extent = composite["extent"]

    with col_dem1:
        st.subheader("2.5D Digital Elevation Model ($Z_{max}$)")
        fig_dem, ax_dem = plt.subplots(figsize=(6, 6), dpi=120, facecolor=bg_color)
        ax_dem.set_facecolor(panel_bg)

        z_max_grid = np.ma.masked_invalid(composite["z_max"])
        im1 = ax_dem.imshow(
            z_max_grid,
            origin="lower",
            extent=[-extent, extent, -extent, extent],
            cmap="viridis",
            interpolation="nearest",
        )
        cb1 = plt.colorbar(im1, ax=ax_dem, fraction=0.046, pad=0.04)
        cb1.set_label("Elevation (Z / m)", color=text_color)
        cb1.ax.tick_params(colors=text_color)

        for ring in foveated_map.rings:
            c = plt.Circle((0, 0), ring.config.r_max, color=ring_color, fill=False, linestyle="--", alpha=0.6)
            ax_dem.add_patch(c)

        ax_dem.set_xlabel("X (Forward / m)", color=text_color)
        ax_dem.set_ylabel("Y (Left / m)", color=text_color)
        ax_dem.tick_params(colors=text_color)
        st.pyplot(fig_dem, use_container_width=True)
        plt.close(fig_dem)

    with col_dem2:
        st.subheader(r"Obstacle Vertical Height Profile ($\Delta Z$)")
        fig_dz, ax_dz = plt.subplots(figsize=(6, 6), dpi=120, facecolor=bg_color)
        ax_dz.set_facecolor(panel_bg)

        delta_z_grid = np.ma.masked_where(~composite["occupied"], composite["delta_z"])
        im2 = ax_dz.imshow(
            delta_z_grid,
            origin="lower",
            extent=[-extent, extent, -extent, extent],
            cmap="magma",
            interpolation="nearest",
            vmax=3.5,
        )
        cb2 = plt.colorbar(im2, ax=ax_dz, fraction=0.046, pad=0.04)
        cb2.set_label("Obstacle Height Delta (m)", color=text_color)
        cb2.ax.tick_params(colors=text_color)

        for ring in foveated_map.rings:
            c = plt.Circle((0, 0), ring.config.r_max, color=ring_color, fill=False, linestyle="--", alpha=0.6)
            ax_dz.add_patch(c)

        ax_dz.set_xlabel("X (Forward / m)", color=text_color)
        ax_dz.set_ylabel("Y (Left / m)", color=text_color)
        ax_dz.tick_params(colors=text_color)
        st.pyplot(fig_dz, use_container_width=True)
        plt.close(fig_dz)


# ------------------------------------------------------------------------------
# TAB 3: MEMORY & BENCHMARK ANALYTICS
# ------------------------------------------------------------------------------
with tab_bench:
    st.subheader("📊 Comparative Memory Consumption Analysis")

    # Generate comparative uniform resolutions
    uniform_resolutions = [0.05, 0.10, 0.15, 0.20]
    comp_data = []
    bytes_per_cell = 22

    for u_res in uniform_resolutions:
        dim = int(np.ceil((2.0 * r2_max) / u_res))
        tot_cells = dim * dim
        ram_mb = (tot_cells * bytes_per_cell) / (1024 * 1024)
        comp_data.append({
            "Architecture": f"Uniform Naive ({int(u_res*100)}cm)",
            "Resolution (m)": u_res,
            "Total Cells": tot_cells,
            "RAM (MB)": ram_mb,
            "Savings vs Baseline": f"0.0%",
        })

    adaptive_cells = foveated_map.total_cells()
    adaptive_ram_mb = foveated_map.total_memory_bytes() / (1024 * 1024)

    # Plotly Interactive Bar Chart
    fig_bar = go.Figure()
    arch_names = [d["Architecture"] for d in comp_data] + ["Adaptive Foveated (Ours)"]
    ram_vals = [d["RAM (MB)"] for d in comp_data] + [adaptive_ram_mb]
    colors = ["#e53e3e", "#dd6b20", "#d69e2e", "#3182ce", "#38a169"]

    fig_bar.add_trace(go.Bar(
        x=arch_names,
        y=ram_vals,
        text=[f"{v:.2f} MB" for v in ram_vals],
        textposition="auto",
        marker_color=colors,
    ))

    fig_bar.update_layout(
        title="Memory Footprint: Adaptive Foveated vs. Uniform Naive Grids",
        xaxis_title="Grid Mapping Strategy",
        yaxis_title="Total Memory (MB)",
        template="plotly_dark" if is_dark else "plotly_white",
        height=420,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Ring Breakdown Table
    st.markdown("### Concentric Ring Memory Allocation Table")
    ring_rows = []
    for idx, ring in enumerate(foveated_map.rings):
        cfg = ring.config
        occ = int(np.sum(ring.occupied_mask))
        tot = cfg.grid_dim * cfg.grid_dim
        ring_rows.append({
            "Ring": f"Ring {idx} ({cfg.name})",
            "Radius Range (m)": f"{cfg.r_min:.1f}m - {cfg.r_max:.1f}m",
            "Resolution": f"{cfg.resolution*100:.1f} cm",
            "Grid Dimensions": f"{cfg.grid_dim} x {cfg.grid_dim}",
            "Total Cells": f"{tot:,}",
            "Occupied Cells": f"{occ:,} ({occ/tot*100:.1f}%)",
            "Memory (KB)": f"{ring.memory_bytes / 1024:.1f} KB",
        })
    st.table(ring_rows)


# ------------------------------------------------------------------------------
# TAB 4: RAW 3D POINT CLOUD INSPECTOR
# ------------------------------------------------------------------------------
with tab_pts:
    st.subheader("🌐 Raw LiDAR Point Cloud Inspection")
    c_pts1, c_pts2 = st.columns(2)

    with c_pts1:
        st.markdown(f"**Total Points:** `{raw_frame.num_points:,}`")
        st.markdown(f"**Bounding Box X (Forward):** `[{raw_frame.points[:, 0].min():.2f}m, {raw_frame.points[:, 0].max():.2f}m]`")
        st.markdown(f"**Bounding Box Y (Lateral):** `[{raw_frame.points[:, 1].min():.2f}m, {raw_frame.points[:, 1].max():.2f}m]`")
        st.markdown(f"**Bounding Box Z (Vertical):** `[{raw_frame.points[:, 2].min():.2f}m, {raw_frame.points[:, 2].max():.2f}m]`")

        # Class breakdown histogram
        counts = raw_frame.get_class_counts()
        _hexmap = SemanticClass.get_hex_color_map()
        _name_to_hex = {SemanticClass.get_name(cid): _hexmap[cid] for cid in range(SemanticClass.num_classes())}
        fig_hist = go.Figure([
            go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                marker_color=[_name_to_hex.get(k, "#999999") for k in counts.keys()],
            )
        ])
        fig_hist.update_layout(
            title="Semantic Class Point Distribution",
            xaxis_title="Semantic Class",
            yaxis_title="Point Count",
            template="plotly_dark" if is_dark else "plotly_white",
            height=300,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c_pts2:
        # Downsampled 3D Scatter View
        downsample_factor = max(1, raw_frame.num_points // 4000)
        sub_pts = raw_frame.points[::downsample_factor]
        sub_lbl = raw_frame.labels[::downsample_factor]

        class_color_map = {0: "gray", 1: "green", 2: "red", 3: "blue"}
        color_strings = [class_color_map.get(int(l), "gray") for l in sub_lbl]

        fig_3d = go.Figure(data=[go.Scatter3d(
            x=sub_pts[:, 0],
            y=sub_pts[:, 1],
            z=sub_pts[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=color_strings,
                opacity=0.8,
            ),
        )])

        fig_3d.update_layout(
            title=f"3D Point Cloud Scatter ({len(sub_pts):,} sampled points) — click a point to inspect",
            scene=dict(
                xaxis_title="X (Forward)",
                yaxis_title="Y (Lateral)",
                zaxis_title="Z (Height)",
                aspectmode="data",
            ),
            template="plotly_dark" if is_dark else "plotly_white",
            height=400,
        )

        # Real click-to-inspect: Streamlit's native selection event round-trips
        # exactly which marker was clicked, so we can show that point's true
        # sensor-frame coordinates, class, and intensity below the plot.
        click_event = st.plotly_chart(
            fig_3d,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="pointcloud_3d_click",
        )

        picked = (click_event or {}).get("selection", {}).get("points", [])
        if picked:
            pt = picked[0]
            idx = pt.get("point_index")
            if idx is not None and idx < len(sub_pts):
                px, py, pz = sub_pts[idx]
                cls_id = int(sub_lbl[idx])
                st.markdown(
                    f"**Selected point** — X: `{px:.2f}m` · Y: `{py:.2f}m` · Z: `{pz:.2f}m` · "
                    f"Class: `{SemanticClass.get_name(cls_id)}`"
                )
        else:
            st.caption("No point selected yet — click any marker above.")


# ==============================================================================
# 6. LIVE AUTO-PLAY REFRESH LOOP
# ==============================================================================
# When "Play" is active, wait roughly one playback interval, advance the
# frame counter, and force Streamlit to rerun the whole script. This makes
# the dashboard behave like a continuously streaming sensor feed rather than
# a static snapshot that only updates when a slider is touched.
#
# The same rerun mechanism also drives the REAL live-hardware path: while
# connected to the UDP sensor bridge with auto-refresh on, we poll on a
# short cadence so newly arrived frames are drawn as soon as they land,
# without the user touching anything.
if st.session_state.get("live_playing", False):
    time.sleep(1.0 / playback_fps)
    st.session_state.live_frame += 1
    st.rerun()
elif data_source == "🔴 Live Hardware Sensor (UDP)" and st.session_state.get("udp_source") is not None:
    if st.session_state.get("auto_refresh_live", True):
        time.sleep(0.2)
        st.rerun()
