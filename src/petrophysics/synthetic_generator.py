"""Synthetic unconventional well generator calibrated with Vaca Muerta petrophysical characteristics.

Generates realistic depth sequences with:
- Alternating organic-rich shale (high GR, high RT, low RHOB, high slowness)
- Interbedded tight carbonates / marls (low GR, low RT, high RHOB, low slowness)
- Realistic compaction trends, acoustic waves (DTC and DTS), and export to authentic .las files.
"""

from pathlib import Path
from typing import Optional, Union

import lasio
import numpy as np
import pandas as pd


def generate_synthetic_vaca_muerta_well(
    start_depth: float = 2500.0,
    stop_depth: float = 3000.0,
    step: float = 0.2,
    seed: int = 42,
    output_las_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate a high-fidelity synthetic unconventional well log dataset.
    
    Args:
        start_depth: Measured depth start (meters).
        stop_depth: Measured depth stop (meters).
        step: Depth sampling step (meters).
        seed: Random seed for reproducibility.
        output_las_path: Optional file path to write a CWLS .las file.
        
    Returns:
        DataFrame with standardized curves (depth, gr, rhob, dtc, dts, rt, nphi, cali).
    """
    rng = np.random.RandomState(seed)
    depths = np.arange(start_depth, stop_depth + step / 2.0, step)
    n_samples = len(depths)

    # Base geological cyclicity (interbedded organic shales and tight carbonates)
    # Long wavelength regional cycles + high frequency sequence stratigraphy
    macro_cycle = np.sin(2.0 * np.pi * (depths - start_depth) / 180.0)
    meso_cycle = 0.5 * np.cos(2.0 * np.pi * (depths - start_depth) / 45.0)
    micro_cycle = 0.25 * np.sin(2.0 * np.pi * (depths - start_depth) / 12.0)
    geology_index = macro_cycle + meso_cycle + micro_cycle  # High = organic shale, Low = carbonate streak

    # Organic sweet spots around depth 2700-2850 (e.g. La Cocina member)
    sweetspot_kernel = np.exp(-0.5 * ((depths - 2775.0) / 45.0) ** 2)

    # 1. Gamma Ray (API)
    # Clean carbonates ~ 25 API, Normal shales ~ 80-110 API, Organic hot shales ~ 140-200 API
    gr_base = 90.0 + 35.0 * geology_index + 60.0 * sweetspot_kernel
    gr_noise = rng.normal(0.0, 4.0, n_samples)
    gr = np.clip(gr_base + gr_noise, 20.0, 240.0)

    # 2. Bulk Density RHOB (g/cm3)
    # Carbonates ~ 2.68-2.71, Normal shale ~ 2.55-2.60, Organic shale with kerogen ~ 2.38-2.48
    rhob_base = 2.60 - 0.08 * geology_index - 0.14 * sweetspot_kernel
    rhob_noise = rng.normal(0.0, 0.015, n_samples)
    rhob = np.clip(rhob_base + rhob_noise, 2.30, 2.75)

    # 3. Compressional Slowness DTC (us/ft)
    # Carbonates ~ 48-55 us/ft, Normal shale ~ 68-75 us/ft, Organic rich ~ 82-95 us/ft
    dtc_base = 72.0 + 12.0 * geology_index + 16.0 * sweetspot_kernel
    dtc_noise = rng.normal(0.0, 1.2, n_samples)
    dtc = np.clip(dtc_base + dtc_noise, 46.0, 110.0)

    # 4. Shear Slowness DTS (us/ft)
    # Follows Vp/Vs ratio typical of organic shales (1.65 to 1.85)
    # Vs (km/s) = 304.8 / dtc * (1 / (Vp/Vs))
    vp_km_s = 304.8 / dtc
    vp_vs_ratio = 1.68 + 0.08 * (gr / 150.0) + rng.normal(0.0, 0.02, n_samples)
    vp_vs_ratio = np.clip(vp_vs_ratio, 1.55, 2.05)
    vs_km_s = vp_km_s / vp_vs_ratio
    dts = 304.8 / vs_km_s

    # 5. True Resistivity RT (ohm.m)
    # Water-bearing / tight ~ 5-20 ohm.m, Hydrocarbon saturated organic shale ~ 40-300 ohm.m
    log_rt_base = 1.2 + 0.4 * geology_index + 0.9 * sweetspot_kernel
    rt_noise = rng.normal(0.0, 0.08, n_samples)
    rt = np.power(10.0, log_rt_base + rt_noise)
    rt = np.clip(rt, 1.0, 500.0)

    # 6. Neutron Porosity NPHI (v/v)
    nphi_base = 0.12 + 0.05 * geology_index + 0.08 * sweetspot_kernel
    nphi = np.clip(nphi_base + rng.normal(0.0, 0.01, n_samples), 0.02, 0.35)

    # 7. Caliper CALI (in) - Bit size = 8.5 in
    # Slight washouts in clay-rich swelling zones
    cali = 8.5 + 0.25 * np.maximum(0.0, geology_index) + rng.normal(0.0, 0.05, n_samples)
    cali = np.clip(cali, 8.4, 10.2)

    df = pd.DataFrame({
        "depth": depths,
        "gr": np.round(gr, 2),
        "rhob": np.round(rhob, 3),
        "dtc": np.round(dtc, 2),
        "dts": np.round(dts, 2),
        "rt": np.round(rt, 2),
        "nphi": np.round(nphi, 3),
        "cali": np.round(cali, 2),
    })

    if output_las_path is not None:
        las_file = lasio.LASFile()
        las_file.well.WELL = "VM-HORIZ-BENCH-01"
        las_file.well.FLD = "VACA MUERTA SHALE"
        las_file.well.CNTY = "NEUQUEN"
        las_file.well.CTRY = "ARGENTINA"
        las_file.well.SRVC = "ANTIGRAVITY GEOMECHANICS"
        las_file.well.STRT = start_depth
        las_file.well.STOP = stop_depth
        las_file.well.STEP = step
        las_file.well.NULL = -999.25

        # Add curves
        las_file.append_curve("DEPTH", depths, unit="M", descr="Measured Depth")
        las_file.append_curve("GR", df["gr"].values, unit="GAPI", descr="Gamma Ray")
        las_file.append_curve("RHOB", df["rhob"].values, unit="G/C3", descr="Bulk Density")
        las_file.append_curve("DTC", df["dtc"].values, unit="US/F", descr="Compressional Slowness")
        las_file.append_curve("DTS", df["dts"].values, unit="US/F", descr="Shear Slowness")
        las_file.append_curve("RT", df["rt"].values, unit="OHMM", descr="Deep True Resistivity")
        las_file.append_curve("NPHI", df["nphi"].values, unit="V/V", descr="Neutron Porosity")
        las_file.append_curve("CALI", df["cali"].values, unit="IN", descr="Caliper")

        out_path = Path(output_las_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            las_file.write(f, version=2.0)

    return df
