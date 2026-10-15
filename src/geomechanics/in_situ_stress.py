"""1D In-Situ Stress Tensor Modeling using the Poroelastic Horizontal Strain Model.

Governing Equations:
    Shmin = [nu / (1 - nu)] * (Sv - alpha * Pp) + alpha * Pp + [E / (1 - nu^2)] * eps_x + [nu * E / (1 - nu^2)] * eps_y
    SHmax = [nu / (1 - nu)] * (Sv - alpha * Pp) + alpha * Pp + [E / (1 - nu^2)] * eps_y + [nu * E / (1 - nu^2)] * eps_x

References:
    - Zoback, M.D. (2007). Reservoir Geomechanics. Cambridge University Press.
    - Blanton, T.L., and Olson, J.E. (1999). "Stress Magnitudes from Logs: Effects of Tectonic
      Strains and Thermal Gradients." SPE 56491.
    - Anderson, E.M. (1951). The Dynamics of Faulting. Oliver and Boyd, Edinburgh.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd


class StressRegime(str, Enum):
    """Andersonian tectonic stress regime classification."""
    NORMAL = "Normal Faulting (Sv > SHmax > Shmin)"
    STRIKE_SLIP = "Strike-Slip (SHmax > Sv > Shmin)"
    REVERSE = "Reverse Faulting (SHmax > Shmin > Sv)"


@dataclass
class StressTensor1D:
    """Container for the complete 1D in-situ stress tensor and orientation indicators.
    
    Attributes:
        sv_psi: Overburden vertical stress in psi.
        pp_psi: Pore pressure in psi.
        shmin_psi: Minimum horizontal in-situ stress in psi.
        shmax_psi: Maximum horizontal in-situ stress in psi.
        shmin_gradient_psi_ft: Minimum horizontal stress gradient in psi/ft (fracture closure gradient).
        stress_anisotropy_psi: Horizontal stress difference (SHmax - Shmin) in psi.
        anisotropy_ratio: Ratio (SHmax - Shmin) / Shmin (dimensionless).
        regime: Dominant Andersonian stress regime per depth point.
    """
    sv_psi: np.ndarray
    pp_psi: np.ndarray
    shmin_psi: np.ndarray
    shmax_psi: np.ndarray
    shmin_gradient_psi_ft: np.ndarray
    stress_anisotropy_psi: np.ndarray
    anisotropy_ratio: np.ndarray
    regime: np.ndarray


def compute_in_situ_stresses(
    depth_m: Union[float, np.ndarray, pd.Series],
    sv_psi: Union[float, np.ndarray, pd.Series],
    pp_psi: Union[float, np.ndarray, pd.Series],
    young_stat_gpa: Union[float, np.ndarray, pd.Series],
    poisson_stat: Union[float, np.ndarray, pd.Series],
    biot_alpha: float = 0.85,
    tectonic_strain_x: float = 0.00015,  # 150 microstrain (minimum horizontal direction)
    tectonic_strain_y: float = 0.00045,  # 450 microstrain (maximum horizontal direction)
) -> StressTensor1D:
    """Compute continuous minimum and maximum horizontal in-situ stresses.
    
    Args:
        depth_m: True vertical depth array in meters.
        sv_psi: Overburden vertical stress in psi.
        pp_psi: Pore pressure in psi.
        young_stat_gpa: Static Young's modulus in GPa.
        poisson_stat: Static Poisson's ratio.
        biot_alpha: Biot's poroelastic coefficient (typically 0.70 to 0.95 in shales).
        tectonic_strain_x: Tectonic strain in min horizontal stress direction (dimensionless).
        tectonic_strain_y: Tectonic strain in max horizontal stress direction (dimensionless).
        
    Returns:
        StressTensor1D dataclass with complete stress tensor profiles.
    """
    z_m = np.asarray(depth_m, dtype=float)
    sv = np.asarray(sv_psi, dtype=float)
    pp = np.asarray(pp_psi, dtype=float)
    e_gpa = np.asarray(young_stat_gpa, dtype=float)
    nu = np.asarray(poisson_stat, dtype=float)

    # Convert Young's Modulus from GPa to psi (1 GPa = 145037.7377 psi)
    GPA_TO_PSI = 145037.7377
    e_psi = e_gpa * GPA_TO_PSI

    with np.errstate(divide="ignore", invalid="ignore"):
        # Effective vertical stress
        effective_sv = sv - biot_alpha * pp

        # Poroelastic term: [nu / (1 - nu)] * (Sv - alpha * Pp) + alpha * Pp
        nu_ratio = nu / (1.0 - nu)
        base_horizontal_stress = nu_ratio * effective_sv + biot_alpha * pp

        # Tectonic strain terms:
        # denom = 1 - nu^2
        denom = 1.0 - np.square(nu)
        e_over_denom = e_psi / denom

        # Shmin (strain x is the lower strain axis)
        tectonic_shmin = e_over_denom * (tectonic_strain_x + nu * tectonic_strain_y)
        shmin = base_horizontal_stress + tectonic_shmin

        # SHmax (strain y is the higher strain axis)
        tectonic_shmax = e_over_denom * (tectonic_strain_y + nu * tectonic_strain_x)
        shmax = base_horizontal_stress + tectonic_shmax

        # Physical safety check: Shmin cannot be less than pore pressure (tensile failure)
        shmin = np.maximum(shmin, pp * 1.02)
        shmax = np.maximum(shmax, shmin)

        # Gradients (psi/ft)
        z_ft = z_m * 3.28084
        shmin_grad = shmin / z_ft

        # Stress anisotropy: Delta_sigma_h = SHmax - Shmin
        stress_anisotropy = shmax - shmin
        anisotropy_ratio = stress_anisotropy / shmin

        # Stress regime classification (Andersonian)
        regimes = np.empty(len(z_m), dtype=object)
        for i in range(len(z_m)):
            if sv[i] >= shmax[i]:
                regimes[i] = StressRegime.NORMAL.value
            elif (shmax[i] > sv[i]) and (sv[i] >= shmin[i]):
                regimes[i] = StressRegime.STRIKE_SLIP.value
            else:
                regimes[i] = StressRegime.REVERSE.value

    return StressTensor1D(
        sv_psi=sv,
        pp_psi=pp,
        shmin_psi=shmin,
        shmax_psi=shmax,
        shmin_gradient_psi_ft=shmin_grad,
        stress_anisotropy_psi=stress_anisotropy,
        anisotropy_ratio=anisotropy_ratio,
        regime=regimes,
    )
