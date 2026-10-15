"""Overburden vertical stress (Sv) calculation via density log integration and shallow extrapolation.

Equation:
    Sv(z) = g * integral_0^z [ rho_b(z') ] dz'

References:
    - Zoback, M.D. (2007). Reservoir Geomechanics. Cambridge University Press.
    - Plumb, R.A., et al. (2000). "The Mechanical Earth Model Concept." SPE 63013.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class OverburdenStress:
    """Container for overburden stress and gradient profiles.
    
    Attributes:
        sv_mpa: Vertical stress profile in MPa.
        sv_psi: Vertical stress profile in psi.
        gradient_mpa_m: Overburden gradient in MPa/m.
        gradient_psi_ft: Overburden gradient in psi/ft (standard oilfield unit ~1.05 - 1.15 psi/ft).
    """
    sv_mpa: np.ndarray
    sv_psi: np.ndarray
    gradient_mpa_m: np.ndarray
    gradient_psi_ft: np.ndarray


def compute_overburden_stress(
    depth_m: Union[float, np.ndarray, pd.Series],
    rhob_g_cm3: Union[float, np.ndarray, pd.Series],
    surface_density_g_cm3: float = 2.05,
    g_accel: float = 9.80665,
) -> OverburdenStress:
    """Integrate bulk density to calculate overburden vertical stress Sv from surface.
    
    Args:
        depth_m: True vertical depth array in meters (must be sorted ascending).
        rhob_g_cm3: Bulk density log array in g/cm3.
        surface_density_g_cm3: Mean density for shallow unlogged interval (0 to top of log).
        g_accel: Gravitational acceleration in m/s2.
        
    Returns:
        OverburdenStress dataclass.
    """
    z = np.asarray(depth_m, dtype=float)
    rho = np.asarray(rhob_g_cm3, dtype=float)

    if len(z) < 2:
        raise ValueError("Depth array must have at least 2 points for integration")

    # Clean intermediate NaNs by linear interpolation
    df_temp = pd.DataFrame({"z": z, "rho": rho})
    df_temp["rho"] = df_temp["rho"].interpolate(method="linear").bfill().ffill()
    rho_clean = df_temp["rho"].values

    z_start = z[0]

    # Cumulative trapezoidal integration from start of log
    # delta_Sv [MPa] = 1e-3 * g * integral rho [g/cm3] dz [m]
    # (since 1 g/cm3 = 1000 kg/m3 and 1 MPa = 1e6 Pa: 1000 * g / 1e6 = 1e-3 * g)
    factor = 1e-3 * g_accel

    # Surface contribution (0 to z_start)
    sv_surface_mpa = factor * surface_density_g_cm3 * z_start

    # Integration along the logged interval
    dz = np.diff(z)
    rho_mid = 0.5 * (rho_clean[:-1] + rho_clean[1:])
    d_sv = factor * rho_mid * dz
    sv_logged_mpa = np.concatenate([[0.0], np.cumsum(d_sv)])

    sv_total_mpa = sv_surface_mpa + sv_logged_mpa

    # Unit conversions
    MPA_TO_PSI = 145.0377377
    sv_psi = sv_total_mpa * MPA_TO_PSI

    # Gradients
    with np.errstate(divide="ignore", invalid="ignore"):
        grad_mpa_m = sv_total_mpa / z
        # 1 psi/ft = (145.0377 / 3.28084) * (MPa/m) / 1000
        depth_ft = z * 3.28084
        grad_psi_ft = sv_psi / depth_ft

    return OverburdenStress(
        sv_mpa=sv_total_mpa,
        sv_psi=sv_psi,
        gradient_mpa_m=grad_mpa_m,
        gradient_psi_ft=grad_psi_ft,
    )
