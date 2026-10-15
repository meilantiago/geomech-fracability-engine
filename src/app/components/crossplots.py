"""Rock physics and geomechanics cross-plot diagnostic figures for the engineering dashboard."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ...engine import GeomechResult


def create_rickman_crossplot(result: GeomechResult) -> go.Figure:
    """Generate the classical Rickman et al. (SPE 115258) rock brittleness cross-plot.
    
    Plots Static Young's Modulus (E_stat) vs. Static Poisson's Ratio (nu_stat),
    colored by Composite Fracability Index (FI), with rock mechanics quadrants.
    """
    df = result.to_dataframe()

    fig = px.scatter(
        df,
        x="poisson_stat",
        y="young_stat_gpa",
        color="fracability_index",
        color_continuous_scale="Viridis",
        hover_data=["depth", "shmin_psi", "vshale"],
        labels={
            "poisson_stat": "Static Poisson's Ratio (nu_stat) [-]",
            "young_stat_gpa": "Static Young's Modulus (E_stat) [GPa]",
            "fracability_index": "Fracability Index (FI)",
        },
    )

    # Median boundaries for quadrant labeling
    nu_mid = float(np.nanmedian(df["poisson_stat"]))
    e_mid = float(np.nanmedian(df["young_stat_gpa"]))

    # Quadrant dividing lines
    fig.add_vline(x=nu_mid, line_width=1, line_dash="dash", line_color="#78909c")
    fig.add_hline(y=e_mid, line_width=1, line_dash="dash", line_color="#78909c")

    # Quadrant annotations
    fig.add_annotation(
        x=df["poisson_stat"].min() + 0.02,
        y=df["young_stat_gpa"].max() - 2.0,
        text="Brittle & Stiff<br>(Ideal Sweet-Spot)",
        showarrow=False,
        font=dict(size=10, color="#81c784"),
        align="left",
    )
    fig.add_annotation(
        x=df["poisson_stat"].max() - 0.02,
        y=df["young_stat_gpa"].min() + 2.0,
        text="Ductile & Soft<br>(Embedment / Screen-out Risk)",
        showarrow=False,
        font=dict(size=10, color="#e57373"),
        align="right",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121820",
        plot_bgcolor="#0d1117",
        height=450,
        margin=dict(l=40, r=40, t=30, b=30),
        coloraxis_colorbar=dict(title="FI [-]", len=0.8),
    )

    return fig


def create_stress_anisotropy_histogram(result: GeomechResult) -> go.Figure:
    """Generate distribution of horizontal stress anisotropy (SHmax - Shmin)."""
    df = result.to_dataframe()

    fig = px.histogram(
        df,
        x="stress_anisotropy_psi",
        nbins=35,
        labels={"stress_anisotropy_psi": "Horizontal Stress Contrast (SHmax - Shmin) [psi]"},
        color_discrete_sequence=["#29b6f6"],
    )

    # Critical threshold for complex fracture network vs planar fracture
    fig.add_vline(
        x=800.0,
        line_width=1.5,
        line_dash="dot",
        line_color="#ffb74d",
        annotation_text="Planar Fracture Threshold (~800 psi)",
        annotation_position="top right",
        annotation_font=dict(size=9, color="#ffb74d"),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121820",
        plot_bgcolor="#0d1117",
        height=450,
        margin=dict(l=40, r=40, t=30, b=30),
        yaxis_title="Footage Sample Count",
    )

    return fig
