"""Composite Well Log Plotter: SPWLA / SPE publication-grade multi-track visualization.

Renders vertical continuous logs with inverted depth axes, standard petroleum track scales,
lithological shading, in-situ stress profiles, and engineered completion stage overlays.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..engine import GeomechResult


def plot_composite_geomech_log(
    result: GeomechResult,
    depth_range: Optional[tuple[float, float]] = None,
    output_png_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    figsize: tuple[float, float] = (16.0, 11.0),
) -> plt.Figure:
    """Generate a 7-track composite geomechanical well log plot (Techlog / Petrel standard).
    
    Tracks:
        Track 1: Gamma Ray & Caliper (Lithology)
        Track 2: Deep Resistivity (Hydrocarbon saturation)
        Track 3: Acoustic Slowness (DTC & DTS)
        Track 4: Static Elastic Moduli (Young's Modulus & Poisson's Ratio)
        Track 5: 1D MEM Stresses (Sv, SHmax, Shmin, Pp)
        Track 6: Brittleness & Fracability Index (Sweet-Spot flags)
        Track 7: Engineered Completion Stages & Perforation Clusters
        
    Args:
        result: GeomechResult object from engine.
        depth_range: Optional (min_md, max_md) depth window.
        output_png_path: Optional path to save high-resolution figure.
        dpi: Resolution for saved image.
        figsize: Figure dimensions (width, height) in inches.
        
    Returns:
        Matplotlib Figure object.
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

    # Setup 7 synchronized subplots sharing Y-axis
    fig, axes = plt.subplots(
        nrows=1,
        ncols=7,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"wspace": 0.08, "left": 0.06, "right": 0.96, "top": 0.88, "bottom": 0.08},
    )

    ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes

    # Apply clean scientific styling
    for ax in axes:
        ax.set_ylim(z_max, z_min)  # Inverted depth axis (surface top, bottom down)
        ax.grid(True, which="both", color="#e0e0e0", linestyle="--", linewidth=0.5)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="both", which="major", labelsize=8)

    ax1.set_ylabel("Measured Depth [m]", fontsize=10, fontweight="bold")

    # -------------------------------------------------------------
    # TRACK 1: Gamma Ray & Caliper
    # -------------------------------------------------------------
    ax1.set_title("Track 1: GR / CALI", fontsize=9, pad=25, fontweight="bold")
    ax1.plot(df_sub["gr"], depth, color="#2e7d32", linewidth=1.0, label="GR [API]")
    ax1.set_xlim(0.0, 200.0)
    ax1.set_xlabel("GR [API]", color="#2e7d32", fontsize=8)
    ax1.tick_params(axis="x", labelcolor="#2e7d32")

    # Lithology shading: Yellow (clean) to Brown (shale)
    ax1.fill_betweenx(depth, 0.0, df_sub["gr"], where=(df_sub["gr"] < 75.0), color="#fff9c4", alpha=0.5)
    ax1.fill_betweenx(depth, 75.0, df_sub["gr"], where=(df_sub["gr"] >= 75.0), color="#d7ccc8", alpha=0.5)

    # Caliper twin axis
    ax1_cali = ax1.twiny()
    ax1_cali.plot(df_sub["cali"], depth, color="#37474f", linewidth=0.8, linestyle="--", label="CALI [in]")
    ax1_cali.set_xlim(6.0, 12.0)
    ax1_cali.set_xlabel("CALI [in]", color="#37474f", fontsize=8)
    ax1_cali.tick_params(axis="x", labelcolor="#37474f", labelsize=8)

    # -------------------------------------------------------------
    # TRACK 2: Deep Resistivity (Logarithmic)
    # -------------------------------------------------------------
    ax2.set_title("Track 2: Resistivity", fontsize=9, pad=25, fontweight="bold")
    ax2.semilogx(df_sub["rt"], depth, color="#b71c1c", linewidth=1.0)
    ax2.set_xlim(1.0, 500.0)
    ax2.set_xlabel("RT [ohm.m]", color="#b71c1c", fontsize=8)
    ax2.tick_params(axis="x", labelcolor="#b71c1c")

    # -------------------------------------------------------------
    # TRACK 3: Acoustic Slowness (DTC & DTS)
    # -------------------------------------------------------------
    ax3.set_title("Track 3: Acoustics", fontsize=9, pad=25, fontweight="bold")
    ax3.plot(df_sub["dtc"], depth, color="#1565c0", linewidth=1.0, label="DTC")
    ax3.set_xlim(40.0, 120.0)
    ax3.set_xlabel("DTC [us/ft]", color="#1565c0", fontsize=8)
    ax3.tick_params(axis="x", labelcolor="#1565c0")

    ax3_dts = ax3.twiny()
    ax3_dts.plot(df_sub["dts"], depth, color="#00838f", linewidth=0.9, linestyle="-.", label="DTS")
    ax3_dts.set_xlim(70.0, 220.0)
    ax3_dts.set_xlabel("DTS [us/ft]", color="#00838f", fontsize=8)
    ax3_dts.tick_params(axis="x", labelcolor="#00838f", labelsize=8)

    # -------------------------------------------------------------
    # TRACK 4: Static Elastic Moduli (E_stat & nu_stat)
    # -------------------------------------------------------------
    ax4.set_title("Track 4: Moduli (Stat)", fontsize=9, pad=25, fontweight="bold")
    ax4.plot(df_sub["young_stat_gpa"], depth, color="#c2185b", linewidth=1.0)
    ax4.set_xlim(5.0, 45.0)
    ax4.set_xlabel("E_stat [GPa]", color="#c2185b", fontsize=8)
    ax4.tick_params(axis="x", labelcolor="#c2185b")

    ax4_nu = ax4.twiny()
    ax4_nu.plot(df_sub["poisson_stat"], depth, color="#6a1b9a", linewidth=0.9, linestyle=":")
    ax4_nu.set_xlim(0.10, 0.35)
    ax4_nu.set_xlabel("nu_stat [-]", color="#6a1b9a", fontsize=8)
    ax4_nu.tick_params(axis="x", labelcolor="#6a1b9a", labelsize=8)

    # -------------------------------------------------------------
    # TRACK 5: 1D MEM In-Situ Stress Tensor (psi)
    # -------------------------------------------------------------
    ax5.set_title("Track 5: 1D MEM Stresses", fontsize=9, pad=25, fontweight="bold")
    ax5.plot(df_sub["sv_psi"], depth, color="#212121", linewidth=1.0, label="Sv")
    ax5.plot(df_sub["shmax_psi"], depth, color="#757575", linewidth=0.9, linestyle="-", label="SHmax")
    ax5.plot(df_sub["shmin_psi"], depth, color="#0d47a1", linewidth=1.2, linestyle="-", label="Shmin")
    ax5.plot(df_sub["pp_psi"], depth, color="#ad1457", linewidth=0.9, linestyle="--", label="Pp")
    ax5.set_xlim(4000.0, 11000.0)
    ax5.set_xlabel("Stress [psi]", fontsize=8)
    ax5.legend(loc="lower left", fontsize=6, framealpha=0.8)

    # -------------------------------------------------------------
    # TRACK 6: Brittleness & Fracability Index (Sweet-Spot flags)
    # -------------------------------------------------------------
    ax6.set_title("Track 6: Fracability", fontsize=9, pad=25, fontweight="bold")
    ax6.plot(df_sub["bi_elastic"], depth, color="#9e9e9e", linewidth=0.7, linestyle="--", label="BI_el")
    ax6.plot(df_sub["fracability_index"], depth, color="#000000", linewidth=1.2, label="FI")
    ax6.axvline(0.65, color="#2e7d32", linestyle=":", linewidth=0.8)
    ax6.axvline(0.45, color="#c62828", linestyle=":", linewidth=0.8)

    # Highlight Sweet-Spots in Green (FI >= 0.65) and Ductile Barriers in Amber (FI < 0.45)
    ax6.fill_betweenx(depth, 0.65, df_sub["fracability_index"], where=(df_sub["fracability_index"] >= 0.65), color="#a5d6a7", alpha=0.6)
    ax6.fill_betweenx(depth, df_sub["fracability_index"], 0.45, where=(df_sub["fracability_index"] < 0.45), color="#ffccbc", alpha=0.6)
    ax6.set_xlim(0.0, 1.0)
    ax6.set_xlabel("FI / BI [-]", fontsize=8)
    ax6.legend(loc="lower left", fontsize=6, framealpha=0.8)

    # -------------------------------------------------------------
    # TRACK 7: Engineered Completion Stages & Clusters
    # -------------------------------------------------------------
    ax7.set_title("Track 7: Completion", fontsize=9, pad=25, fontweight="bold")
    ax7.set_xlim(0.0, 1.0)
    ax7.set_xticks([])

    # Plot stages and clusters
    for p in result.engineered_plans:
        st = p.stage_info
        if st.end_md < z_min or st.start_md > z_max:
            continue

        # Stage boundary line
        ax7.axhline(st.start_md, color="#212121", linewidth=1.0)
        ax7.axhline(st.end_md, color="#212121", linewidth=1.0)

        # Stage label in middle
        st_mid = 0.5 * (st.start_md + st.end_md)
        if z_min <= st_mid <= z_max:
            ax7.text(
                0.15,
                st_mid,
                f"S{st.stage_id}\n({st.length_m:.0f}m)",
                va="center",
                ha="left",
                fontsize=7,
                fontweight="bold",
                color="#37474f",
            )

        # Plot perforation clusters
        for c in p.clusters:
            if z_min <= c.md_m <= z_max:
                marker_color = "#2e7d32" if c.is_sweetspot else "#e65100"
                ax7.plot(0.80, c.md_m, marker="<", color=marker_color, markersize=5)

    if output_png_path is not None:
        out = Path(output_png_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    return fig
