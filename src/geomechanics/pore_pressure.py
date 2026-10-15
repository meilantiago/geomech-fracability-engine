"""Pore pressure estimation using Eaton's acoustic sonic slowness method.

Equation:
    Pp = Sv - (Sv - Pp_norm) * (dtc_norm / dtc)^x

References:
    - Eaton, B.A. (1975). "The Equation for Geopressure Prediction from Well Logs."
      SPE 5544.
    - Bowers, G.L. (1995). "Pore Pressure Estimation from Inversion of Velocity:
      The Approach, its Benefits and Pitfalls." SPE 28410.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class PorePressureProfile:
    """Container for pore pressure and effective stress profiles.
    
    Attributes:
        pp_mpa: Pore pressure in MPa.
        pp_psi: Pore pressure in psi.
        gradient_mpa_m: Pore pressure gradient in MPa/m.
        gradient_psi_ft: Pore pressure gradient in psi/ft (hydrostatic ~0.433 psi/ft, Vaca Muerta ~0.65-0.85 psi/ft).
        dtc_nct: Normal compaction trendline (NCT) slowness in us/ft.
        is_overpressured: Boolean mask indicating overpressured intervals (gradient > 0.465 psi/ft).
    """
    pp_mpa: np.ndarray
    pp_psi: np.ndarray
    gradient_mpa_m: np.ndarray
    gradient_psi_ft: np.ndarray
    dtc_nct: np.ndarray
    is_overpressured: np.ndarray


def compute_eaton_pore_pressure(
    depth_m: Union[float, np.ndarray, pd.Series],
    dtc_us_ft: Union[float, np.ndarray, pd.Series],
    sv_psi: Union[float, np.ndarray, pd.Series],
    eaton_exponent: float = 3.0,
    hydrostatic_gradient_psi_ft: float = 0.433,
    dtc_surface_us_ft: float = 190.0,
    compaction_decay_m: float = 0.00035,
    custom_dtc_nct: Optional[Union[float, np.ndarray, pd.Series]] = None,
) -> PorePressureProfile:
    """Estimate continuous pore pressure using Eaton's acoustic slowness method.
    
    Args:
        depth_m: True vertical depth array in meters.
        dtc_us_ft: Compressional sonic slowness log in us/ft.
        sv_psi: Overburden vertical stress in psi.
        eaton_exponent: Empirical Eaton exponent (standard x = 3.0 for sonic).
        hydrostatic_gradient_psi_ft: Normal hydrostatic pore pressure gradient (~0.433 psi/ft for freshwater, 0.465 for brine).
        dtc_surface_us_ft: Extrapolated surface mud slowness for normal compaction trendline.
        compaction_decay_m: Exponential depth compaction decay constant (1/m).
        custom_dtc_nct: Optional custom NCT slowness curve to bypass exponential model.
        
    Returns:
        PorePressureProfile dataclass.
    """
    z_m = np.asarray(depth_m, dtype=float)
    dtc = np.asarray(dtc_us_ft, dtype=float)
    sv = np.asarray(sv_psi, dtype=float)

    z_ft = z_m * 3.28084

    # Calculate Normal Compaction Trendline (NCT)
    if custom_dtc_nct is not None:
        dtc_nct = np.asarray(custom_dtc_nct, dtype=float)
    else:
        # Standard exponential compaction: dtc_nct(z) = dtc_0 * exp(-c * z)
        dtc_nct = dtc_surface_us_ft * np.exp(-compaction_decay_m * z_m)

    # Normal hydrostatic pore pressure
    pp_norm_psi = hydrostatic_gradient_psi_ft * z_ft

    # Eaton equation: Pp = Sv - (Sv - Pp_norm) * (dtc_nct / dtc)^x
    with np.errstate(divide="ignore", invalid="ignore"):
        slowness_ratio = dtc_nct / dtc
        # Prevent numerical explosion in washouts or invalid spikes
        slowness_ratio = np.clip(slowness_ratio, 0.2, 2.5)

        pp_psi = sv - (sv - pp_norm_psi) * np.power(slowness_ratio, eaton_exponent)

        # Physical constraints:
        # Pp must be >= hydrostatic (underpressured reservoirs are rare in shale)
        # Pp must not exceed overburden stress Sv (rock would hydrofracture naturally)
        pp_psi = np.clip(pp_psi, pp_norm_psi, 0.98 * sv)

        # Gradients
        grad_psi_ft = pp_psi / z_ft
        PSI_TO_MPA = 1.0 / 145.0377377
        pp_mpa = pp_psi * PSI_TO_MPA
        grad_mpa_m = pp_mpa / z_m

        is_overpressured = grad_psi_ft > 0.465

    return PorePressureProfile(
        pp_mpa=pp_mpa,
        pp_psi=pp_psi,
        gradient_mpa_m=grad_mpa_m,
        gradient_psi_ft=grad_psi_ft,
        dtc_nct=dtc_nct,
        is_overpressured=is_overpressured,
    )
