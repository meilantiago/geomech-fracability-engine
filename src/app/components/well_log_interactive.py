"""Interactive multi-track well log visualizer using Plotly (SPWLA / SPE standard layout).

Provides 7 synchronized tracks sharing an inverted depth axis, allowing seamless zoom,
pan, and tooltip inspection for petroleum geomechanics and completion planning.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...engine import GeomechResult


def create_interactive_log_figure(
    result: GeomechResult,
    depth_range: Optional[tuple[float, float]] = None,
    height: int = 850,
) -> go.Figure:
    """Build an interactive 7-track continuous geomechanical composite log using Plotly.
    
    Args:
        result: GeomechResult object containing computed logs and completion plans.
        depth_range: Optional (min_md, max_md) window.
        height: Figure display height in pixels.
        
    Returns:
        Plotly Figure object.
    """
    df = result.to_dataframe()
    z = df["depth"].values

    if depth_range is not None:
        z_min, z_max = depth_range
        mask = (z >= z_min) & (z <= z_max)
        df_sub = df[mask].copy()
    else:
        df_sub = df.copy()
        z_min, z_max = float(z[0]), float(z[-1])

    depth = df_sub["depth"].values

    # Track column widths: balanced distribution
    column_widths = [0.15, 0.13, 0.14, 0.14, 0.16, 0.14, 0.14]

    fig = make_subplots(
        rows=1,
        cols=7,
        shared_yaxes=True,
        horizontal_spacing=0.015,
        column_widths=column_widths,
        subplot_titles=[
            "Track 1: GR & CALI",
            "Track 2: Resistivity",
            "Track 3: Acoustics",
            "Track 4: Moduli (Stat)",
            "Track 5: 1D MEM Stresses",
            "Track 6: Fracability",
            "Track 7: Completion",
        ],
    )

    # -------------------------------------------------------------
    # TRACK 1: Gamma Ray & Caliper
    # -------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_sub["gr"],
            y=depth,
            name="GR [API]",
            mode="lines",
            line=dict(color="#4caf50", width=1.2),
            hovertemplate="Depth: %{y:.1f} m<br>GR: %{x:.1f} API<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_sub["cali"],
            y=depth,
            name="CALI [in]",
            mode="lines",
            line=dict(color="#78909c", width=1.0, dash="dash"),
            hovertemplate="Depth: %{y:.1f} m<br>CALI: %{x:.2f} in<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # -------------------------------------------------------------
    # TRACK 2: Deep Resistivity (Logarithmic)
    # -------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_sub["rt"],
            y=depth,
            name="RT [ohm.m]",
            mode="lines",
            line=dict(color="#ef5350", width=1.2),
            hovertemplate="Depth: %{y:.1f} m<br>RT: %{x:.1f} ohm.m<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # -------------------------------------------------------------
    # TRACK 3: Acoustic Slowness (DTC & DTS)
    # -------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_sub["dtc"],
            y=depth,
            name="DTC [us/ft]",
            mode="lines",
            line=dict(color="#42a5f5", width=1.2),
            hovertemplate="Depth: %{y:.1f} m<br>DTC: %{x:.1f} us/ft<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=df_sub["dts"],
            y=depth,
            name="DTS [us/ft]",
            mode="lines",
            line=dict(color="#26c6da", width=1.0, dash="dashdot"),
            hovertemplate="Depth: %{y:.1f} m<br>DTS: %{x:.1f} us/ft<extra></extra>",
        ),
        row=1,
        col=3,
    )

    # -------------------------------------------------------------
    # TRACK 4: Static Elastic Moduli (E_stat & nu_stat)
    # -------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_sub["young_stat_gpa"],
            y=depth,
            name="E_stat [GPa]",
            mode="lines",
            line=dict(color="#ec407a", width=1.2),
            hovertemplate="Depth: %{y:.1f} m<br>E_stat: %{x:.1f} GPa<extra></extra>",
        ),
        row=1,
        col=4,
    )
    fig.add_trace(
        go.Scatter(
            x=df_sub["poisson_stat"] * 100.0,  # Scaled for overlay visibility (0.15 -> 15)
            y=depth,
            name="nu_stat x100 [-]",
            mode="lines",
            line=dict(color="#ab47bc", width=1.0, dash="dot"),
            hovertemplate="Depth: %{y:.1f} m<br>nu_stat: %{customdata:.3f}<extra></extra>",
            customdata=df_sub["poisson_stat"],
        ),
        row=1,
        col=4,
    )

    # -------------------------------------------------------------
    # TRACK 5: 1D MEM In-Situ Stress Tensor (psi)
    # -------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_sub["sv_psi"],
            y=depth,
            name="Sv [psi]",
            mode="lines",
            line=dict(color="#bdbdbd", width=1.0),
            hovertemplate="Depth: %{y:.1f} m<br>Sv: %{x:.0f} psi<extra></extra>",
        ),
        row=1,
        col=5,
    )
    fig.add_trace(
        go.Scatter(
            x=df_sub["shmax_psi"],
            y=depth,
            name="SHmax [psi]",
            mode="lines",
            line=dict(color="#90a4ae", width=1.0, dash="dash"),
            hovertemplate="Depth: %{y:.1f} m<br>SHmax: %{x:.0f} psi<extra></extra>",
        ),
        row=1,
        col=5,
    )
    fig.add_trace(
        go.Scatter(
            x=df_sub["shmin_psi"],
            y=depth,
            name="Shmin [psi]",
            mode="lines",
            line=dict(color="#1976d2", width=1.6),
            hovertemplate="Depth: %{y:.1f} m<br>Shmin: %{x:.0f} psi<extra></extra>",
        ),
        row=1,
        col=5,
    )
    fig.add_trace(
        go.Scatter(
            x=df_sub["pp_psi"],
            y=depth,
            name="Pp [psi]",
            mode="lines",
            line=dict(color="#d81b60", width=1.1, dash="dot"),
            hovertemplate="Depth: %{y:.1f} m<br>Pp: %{x:.0f} psi<extra></extra>",
        ),
        row=1,
        col=5,
    )

    # -------------------------------------------------------------
    # TRACK 6: Brittleness & Fracability Index (Sweet-Spots)
    # -------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_sub["bi_elastic"],
            y=depth,
            name="BI_elastic [-]",
            mode="lines",
            line=dict(color="#757575", width=0.8, dash="dot"),
            hovertemplate="Depth: %{y:.1f} m<br>BI: %{x:.3f}<extra></extra>",
        ),
        row=1,
        col=6,
    )
    fig.add_trace(
        go.Scatter(
            x=df_sub["fracability_index"],
            y=depth,
            name="Fracability Index (FI)",
            mode="lines",
            line=dict(color="#ffffff", width=1.5),
            hovertemplate="Depth: %{y:.1f} m<br>FI: %{x:.3f}<extra></extra>",
        ),
        row=1,
        col=6,
    )

    # -------------------------------------------------------------
    # TRACK 7: Engineered Completion Stages & Clusters
    # -------------------------------------------------------------
    # Collect clusters for Track 7
    cluster_x = []
    cluster_y = []
    cluster_text = []
    cluster_colors = []

    for p in result.engineered_plans:
        st = p.stage_info
        if st.end_md < z_min or st.start_md > z_max:
            continue

        # Horizontal stage boundary lines
        fig.add_hline(y=st.start_md, line_width=1.0, line_dash="solid", line_color="#546e7a", row=1, col=7)
        fig.add_hline(y=st.end_md, line_width=1.0, line_dash="solid", line_color="#546e7a", row=1, col=7)

        # Clusters
        for c in p.clusters:
            if z_min <= c.md_m <= z_max:
                cluster_x.append(0.5)
                cluster_y.append(c.md_m)
                color = "#4caf50" if c.is_sweetspot else "#ff9800"
                cluster_colors.append(color)
                cluster_text.append(f"Stage {st.stage_id} | Cluster {c.cluster_id}<br>MD: {c.md_m:.1f} m<br>FI: {c.fracability_index:.3f}<br>Shmin: {c.shmin_psi:.0f} psi")

    fig.add_trace(
        go.Scatter(
            x=cluster_x,
            y=cluster_y,
            mode="markers",
            name="Perforation Clusters",
            marker=dict(
                size=9,
                color=cluster_colors,
                symbol="triangle-left",
                line=dict(width=1, color="#ffffff"),
            ),
            hoverinfo="text",
            hovertext=cluster_text,
        ),
        row=1,
        col=7,
    )

    # -------------------------------------------------------------
    # Axis Configurations & Industrial Dark Theme
    # -------------------------------------------------------------
    # Inverted Depth Axis
    fig.update_yaxes(
        autorange="reversed",
        title_text="Measured Depth [m]",
        showgrid=True,
        gridcolor="#263238",
        zeroline=False,
    )

    # Track 1 axes
    fig.update_xaxes(title_text="GR / CALI", range=[0.0, 200.0], gridcolor="#263238", row=1, col=1)
    # Track 2 axes (Logarithmic RT)
    fig.update_xaxes(title_text="RT [ohm.m]", type="log", range=[0.0, 3.0], gridcolor="#263238", row=1, col=2)
    # Track 3 axes
    fig.update_xaxes(title_text="DTC / DTS [us/ft]", range=[40.0, 220.0], gridcolor="#263238", row=1, col=3)
    # Track 4 axes
    fig.update_xaxes(title_text="E [GPa] / nu*100", range=[0.0, 50.0], gridcolor="#263238", row=1, col=4)
    # Track 5 axes
    fig.update_xaxes(title_text="Stress [psi]", range=[4000.0, 11000.0], gridcolor="#263238", row=1, col=5)
    # Track 6 axes
    fig.update_xaxes(title_text="FI / BI [-]", range=[0.0, 1.0], gridcolor="#263238", row=1, col=6)
    # Track 7 axes
    fig.update_xaxes(title_text="Completion Plan", showticklabels=False, range=[0.0, 1.0], gridcolor="#263238", row=1, col=7)

    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="#121820",
        plot_bgcolor="#0d1117",
        margin=dict(l=60, r=20, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=9, color="#b0bec5"),
        ),
        hovermode="y unified",
    )

    return fig
