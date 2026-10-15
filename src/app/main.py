"""1D Geomechanics & Hydraulic Fracturing Completion Engine - Engineering Workstation.

A sober, professional analytical dashboard for subsurface engineers, petrophysicists,
and completion specialists. Built without marketing AI slop, designed for technical rigor.
"""

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.components.crossplots import (
    create_rickman_crossplot,
    create_stress_anisotropy_histogram,
)
from src.app.components.well_log_interactive import create_interactive_log_figure
from src.elasticity import CalibrationMethod
from src.engine import GeomechEngine, GeomechResult
from src.petrophysics import ShaleVolumeModel, generate_synthetic_vaca_muerta_well

# Page setup: Wide layout, dark industrial aesthetic
st.set_page_config(
    page_title="1D Geomechanics & Completion Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom sober CSS: Dark slate industrial theme, compact density, monospaced figures
st.markdown(
    """
    <style>
    /* Dark slate theme styling */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    header[data-testid="stHeader"] {
        background-color: #0d1117;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #f0f6fc;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    
    /* Metrics display */
    .metric-container {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 20px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        color: #58a6ff;
        font-weight: 700;
        margin-top: 2px;
    }
    .metric-subtext {
        font-size: 11px;
        color: #3fb950;
        margin-top: 2px;
    }
    
    /* Tables */
    .dataframe {
        font-family: "SFMono-Regular", Consolas, monospace !important;
        font-size: 11px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding-left: 16px;
        padding-right: 16px;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-bottom: none;
        border-radius: 4px 4px 0 0;
        color: #8b949e;
        font-size: 13px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #f0f6fc !important;
        border-top: 2px solid #58a6ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_benchmark_las_path() -> Path:
    """Ensure benchmark LAS file is available."""
    data_dir = PROJECT_ROOT / "data" / "synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)
    las_path = data_dir / "vaca_muerta_benchmark.las"
    if not las_path.exists():
        generate_synthetic_vaca_muerta_well(output_las_path=las_path)
    return las_path


# -------------------------------------------------------------
# SIDEBAR: Engineering Configuration
# -------------------------------------------------------------
st.sidebar.markdown("### Engineering Workstation")
st.sidebar.caption("1D Geomechanics & Completion Optimizer")

st.sidebar.markdown("---")
st.sidebar.markdown("**Well Log Source**")
well_source = st.sidebar.radio(
    label="Select Data Source",
    options=["Vaca Muerta Benchmark (Synthetic)", "Upload Custom .las File"],
    label_visibility="collapsed",
)

uploaded_file = None
if well_source == "Upload Custom .las File":
    uploaded_file = st.sidebar.file_uploader("Upload CWLS .las File", type=["las"])

st.sidebar.markdown("---")
st.sidebar.markdown("**1D MEM Reservoir Parameters**")

biot_alpha = st.sidebar.slider(
    "Biot Coefficient (alpha) [-]",
    min_value=0.70,
    max_value=1.00,
    value=0.85,
    step=0.01,
    help="Biot poroelastic constant. Typically 0.75-0.90 in organic shales.",
)

col_eps1, col_eps2 = st.sidebar.columns(2)
with col_eps1:
    tectonic_x = st.number_input(
        "eps_x [ue]",
        min_value=0,
        max_value=1000,
        value=150,
        step=25,
        help="Min horizontal tectonic strain (microstrain)",
    )
with col_eps2:
    tectonic_y = st.number_input(
        "eps_y [ue]",
        min_value=0,
        max_value=1500,
        value=450,
        step=25,
        help="Max horizontal tectonic strain (microstrain)",
    )

eaton_exp = st.sidebar.slider(
    "Eaton Sonic Exponent (x)",
    min_value=1.0,
    max_value=5.0,
    value=3.0,
    step=0.2,
    help="Empirical exponent for acoustic pore pressure prediction (standard x=3.0)",
)

calib_method = st.sidebar.selectbox(
    "Static Moduli Model",
    options=[
        CalibrationMethod.BRADFORD_SHALE.value,
        CalibrationMethod.LACY.value,
        CalibrationMethod.MORALES.value,
    ],
    index=0,
    help="Dynamic-to-static empirical transformation for unconventional formations.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Completion Optimization Parameters**")

col_comp1, col_comp2 = st.sidebar.columns(2)
with col_comp1:
    stage_len = st.number_input(
        "Stage Length [m]",
        min_value=35.0,
        max_value=120.0,
        value=60.0,
        step=5.0,
    )
with col_comp2:
    clusters_count = st.number_input(
        "Clusters / Stage",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
    )

# -------------------------------------------------------------
# RUN ENGINE WORKFLOW
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def execute_engine(
    file_bytes_or_path,
    is_bytes: bool,
    biot: float,
    eps_x: float,
    eps_y: float,
    eaton_x: float,
    calib: str,
    target_l: float,
    n_clust: int,
) -> GeomechResult:
    engine = GeomechEngine(
        biot_alpha=biot,
        tectonic_strain_x=eps_x * 1e-6,
        tectonic_strain_y=eps_y * 1e-6,
        eaton_exponent=eaton_x,
        static_calibration_method=calib,
        target_stage_length_m=target_l,
        clusters_per_stage=n_clust,
    )

    if is_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tf:
            tf.write(file_bytes_or_path)
            temp_path = tf.name
        res = engine.run_from_las(temp_path)
        Path(temp_path).unlink(missing_ok=True)
        return res
    else:
        return engine.run_from_las(file_bytes_or_path)


with st.spinner("Solving 1D geomechanical equations and optimizing completion..."):
    if uploaded_file is not None:
        result = execute_engine(
            uploaded_file.getvalue(),
            is_bytes=True,
            biot=biot_alpha,
            eps_x=tectonic_x,
            eps_y=tectonic_y,
            eaton_x=eaton_exp,
            calib=calib_method,
            target_l=stage_len,
            n_clust=clusters_count,
        )
    else:
        benchmark_las = get_benchmark_las_path()
        result = execute_engine(
            benchmark_las,
            is_bytes=False,
            biot=biot_alpha,
            eps_x=tectonic_x,
            eps_y=tectonic_y,
            eaton_x=eaton_exp,
            calib=calib_method,
            target_l=stage_len,
            n_clust=clusters_count,
        )

# -------------------------------------------------------------
# HEADER & EXECUTIVE METRICS STRIP
# -------------------------------------------------------------
st.title("1D Geomechanics & Hydraulic Fracturing Completion Engine")
st.caption("Subsurface Geomechanical Profiling, 1D MEM Stress Modeling, and Engineered Perforation Cluster Sweet-Spotting")

rep = result.comparison_report
w_data = result.well_data

m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Logged Interval</div>
            <div class="metric-value">{w_data.start_depth:.0f} - {w_data.stop_depth:.0f} m</div>
            <div class="metric-subtext">Lateral: {w_data.stop_depth - w_data.start_depth:.0f} m</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Total Stages</div>
            <div class="metric-value">{result.engineered_partition.total_stages}</div>
            <div class="metric-subtext">Mean: {result.engineered_partition.mean_stage_length:.1f} m</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Total Clusters</div>
            <div class="metric-value">{rep.total_clusters}</div>
            <div class="metric-subtext">{clusters_count} clusters / stage</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Cluster Fracability (FI)</div>
            <div class="metric-value">{rep.engineered_mean_fi:.3f}</div>
            <div class="metric-subtext">+{rep.fi_improvement_pct}% vs Geometric</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m5:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Intra-Stage Delta Shmin</div>
            <div class="metric-value">{rep.engineered_mean_stress_contrast_psi:.0f} psi</div>
            <div class="metric-subtext">-{rep.stress_contrast_reduction_pct}% Stress Contrast</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m6:
    eliminated = rep.geometric_ductile_clusters - rep.engineered_ductile_clusters
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Ductile Traps Avoided</div>
            <div class="metric-value">{eliminated} Clusters</div>
            <div class="metric-subtext">Redirected to Sweet-Spots</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------
# MAIN WORKSTATION TABS
# -------------------------------------------------------------
tab_log, tab_sched, tab_diagnostics, tab_theory = st.tabs([
    "Composite Well-Log Track (1D MEM)",
    "Engineered Completion Schedule",
    "Rock Physics & Stress Diagnostics",
    "Analytical Formulation & Standards",
])

with tab_log:
    col_ctrl1, col_ctrl2 = st.columns([0.7, 0.3])
    with col_ctrl1:
        st.markdown("**Continuous 7-Track Geomechanical Profile (Techlog / Petrel Standard)**")
        st.caption("Synchronized vertical tracks. Zoom and pan enabled. Green indicators mark Tier 1 sweet-spots.")
    with col_ctrl2:
        zoom_window = st.slider(
            "Depth Zoom Window [m]",
            min_value=float(w_data.start_depth),
            max_value=float(w_data.stop_depth),
            value=(float(w_data.start_depth), float(w_data.stop_depth)),
            step=10.0,
            label_visibility="collapsed",
        )

    log_fig = create_interactive_log_figure(result, depth_range=zoom_window, height=880)
    st.plotly_chart(log_fig, use_container_width=True)

with tab_sched:
    st.markdown("**Perforation Cluster Schedule & Quantitative Design Audit**")
    st.caption("Comparison between blind geometric stage slicing and engineered sweet-spotting.")

    c_left, c_right = st.columns([0.45, 0.55])

    with c_left:
        st.markdown("##### Quantitative Completion Comparison")
        comp_df = pd.DataFrame({
            "Metric": [
                "Mean Perforation Fracability Index (FI)",
                "Intra-Stage Shmin Contrast (Delta Shmin)",
                "Ductile / Screen-Out Traps (Risk Count)",
                "Stages / Total Perforation Clusters",
            ],
            "Geometric Spacing": [
                f"{rep.geometric_mean_fi:.3f}",
                f"{rep.geometric_mean_stress_contrast_psi:.1f} psi",
                f"{rep.geometric_ductile_clusters} clusters",
                f"{result.geometric_partition.total_stages} stages / {rep.total_clusters} clusters",
            ],
            "Engineered Spacing": [
                f"{rep.engineered_mean_fi:.3f} (+{rep.fi_improvement_pct}%)",
                f"{rep.engineered_mean_stress_contrast_psi:.1f} psi (-{rep.stress_contrast_reduction_pct}%)",
                f"{rep.engineered_ductile_clusters} clusters",
                f"{result.engineered_partition.total_stages} stages / {rep.total_clusters} clusters",
            ],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("##### Deliverable Exports")
        csv_buffer = io.StringIO()
        sched_df = result.export_completion_schedule(PROJECT_ROOT / "results" / "temp_sched.csv", design="engineered")
        sched_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="Download Completion Schedule (.csv)",
            data=csv_buffer.getvalue(),
            file_name=f"{w_data.well_name}_completion_schedule.csv",
            mime="text/csv",
        )

        parquet_buffer = io.BytesIO()
        result.to_dataframe().to_parquet(parquet_buffer, index=False)
        st.download_button(
            label="Download Comprehensive 1D Profiles (.parquet)",
            data=parquet_buffer.getvalue(),
            file_name=f"{w_data.well_name}_geomech_profiles.parquet",
            mime="application/octet-stream",
        )

    with c_right:
        st.markdown("##### Stage-by-Stage Engineering Breakdown")
        stage_rows = []
        for p in result.engineered_plans:
            stg = p.stage_info
            stage_rows.append({
                "Stage": f"Stage {stg.stage_id}",
                "Top [m]": f"{stg.start_md:.1f}",
                "Bottom [m]": f"{stg.end_md:.1f}",
                "Length [m]": f"{stg.length_m:.1f}",
                "Mean Shmin [psi]": f"{stg.mean_shmin_psi:.0f}",
                "Delta Shmin [psi]": f"{p.cluster_stress_range_psi:.0f}",
                "Mean FI": f"{p.mean_cluster_fi:.3f}",
            })
        st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)

with tab_diagnostics:
    st.markdown("**Rock Physics & Geomechanical Diagnostics**")
    st.caption("Cross-plot diagnostic evaluation for petrophysical sweet-spotting and stress containment.")

    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        st.markdown("##### Rickman Brittleness Cross-Plot")
        st.caption("Young's Modulus vs. Poisson's Ratio colored by Composite Fracability Index.")
        fig_rickman = create_rickman_crossplot(result)
        st.plotly_chart(fig_rickman, use_container_width=True)

    with diag_col2:
        st.markdown("##### Horizontal Stress Anisotropy (SHmax - Shmin)")
        st.caption("Distribution of differential horizontal stress along the lateral.")
        fig_stress = create_stress_anisotropy_histogram(result)
        st.plotly_chart(fig_stress, use_container_width=True)

with tab_theory:
    st.markdown("**Governing Geomechanical Formulations & Industry Standards**")
    st.caption("Mathematical foundations implemented in the calculation engine.")

    t1, t2 = st.columns(2)

    with t1:
        st.markdown("##### 1. Acoustic Poroelastic Moduli")
        st.latex(r"\nu_{dyn} = \frac{V_p^2 - 2V_s^2}{2(V_p^2 - V_s^2)}")
        st.latex(r"E_{dyn} = \rho_b V_s^2 \left(\frac{3V_p^2 - 4V_s^2}{V_p^2 - V_s^2}\right)")
        st.markdown("##### 2. Static Transformation (Bradford et al., 1998)")
        st.latex(r"E_{stat} = 0.415 \cdot E_{dyn} - 1.059 \quad [\text{GPa}]")
        st.latex(r"\nu_{stat} = 0.70 \cdot \nu_{dyn}")

    with t2:
        st.markdown("##### 3. Poroelastic In-Situ Horizontal Stress (Zoback, 2007)")
        st.latex(r"\sigma_{h,min} = \frac{\nu}{1-\nu}(S_v - \alpha P_p) + \alpha P_p + \frac{E}{1-\nu^2}\epsilon_x + \frac{\nu E}{1-\nu^2}\epsilon_y")
        st.markdown("##### 4. Composite Fracability Index (Jin et al., 2014)")
        st.latex(r"FI = w_1 \cdot BI_{elastic} + w_2 \cdot BI_{mineral} + w_3 \cdot (1 - \sigma_{h,min,norm})")

    st.markdown("---")
    st.markdown(
        """
        **References:**
        - **Zoback, M.D. (2007):** *Reservoir Geomechanics*, Cambridge University Press.
        - **Eaton, B.A. (1975):** *The Equation for Geopressure Prediction from Well Logs*, SPE 5544.
        - **Rickman, R., et al. (2008):** *A Practical Use of Shale Petrophysics for Stimulation Design Optimization*, SPE 115258.
        - **Castagna, J.P., et al. (1985):** *Relationships between compressional-wave and shear-wave velocities in clastic silicate rocks*, Geophysics.
        """
    )
