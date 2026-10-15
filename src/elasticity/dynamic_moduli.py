"""Dynamic elastic moduli calculation from acoustic velocities (Vp, Vs) and bulk density (RHOB).

Governing Relations:
    - Poisson's Ratio (nu_dyn): [0.5 * (Vp/Vs)^2 - 1] / [(Vp/Vs)^2 - 1]
    - Shear Modulus (G_dyn): rho_b * Vs^2
    - Young's Modulus (E_dyn): 2 * G_dyn * (1 + nu_dyn)
    - Bulk Modulus (K_dyn): rho_b * (Vp^2 - 4/3 * Vs^2)

References:
    - Mavko, G., Mukerji, T., and Dvorkin, J. (2020). The Rock Physics Handbook.
      Cambridge University Press.
    - Zoback, M.D. (2007). Reservoir Geomechanics. Cambridge University Press.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd


@dataclass
class DynamicElasticModuli:
    """Container for dynamic acoustic elastic properties.
    
    Attributes:
        vp_m_s: Compressional velocity in m/s.
        vs_m_s: Shear velocity in m/s.
        vp_vs_ratio: Ratio Vp/Vs (dimensionless).
        poisson_ratio: Dynamic Poisson's ratio (dimensionless).
        young_gpa: Dynamic Young's modulus in GPa.
        shear_gpa: Dynamic Shear modulus in GPa.
        bulk_gpa: Dynamic Bulk modulus in GPa.
        young_mpsi: Dynamic Young's modulus in Mpsi.
        shear_mpsi: Dynamic Shear modulus in Mpsi.
        bulk_mpsi: Dynamic Bulk modulus in Mpsi.
    """
    vp_m_s: np.ndarray
    vs_m_s: np.ndarray
    vp_vs_ratio: np.ndarray
    poisson_ratio: np.ndarray
    young_gpa: np.ndarray
    shear_gpa: np.ndarray
    bulk_gpa: np.ndarray
    young_mpsi: np.ndarray
    shear_mpsi: np.ndarray
    bulk_mpsi: np.ndarray


def compute_dynamic_moduli(
    rhob_g_cm3: Union[float, np.ndarray, pd.Series],
    dtc_us_ft: Union[float, np.ndarray, pd.Series],
    dts_us_ft: Union[float, np.ndarray, pd.Series],
) -> DynamicElasticModuli:
    """Calculate continuous dynamic elastic moduli from density and acoustic slownesses.
    
    Args:
        rhob_g_cm3: Bulk density in g/cm3.
        dtc_us_ft: Compressional acoustic slowness in us/ft.
        dts_us_ft: Shear acoustic slowness in us/ft.
        
    Returns:
        DynamicElasticModuli dataclass containing all moduli in SI and Oilfield units.
    """
    rhob = np.atleast_1d(np.asarray(rhob_g_cm3, dtype=float))
    dtc = np.atleast_1d(np.asarray(dtc_us_ft, dtype=float))
    dts = np.atleast_1d(np.asarray(dts_us_ft, dtype=float))

    # Convert slownesses (us/ft) to velocities (m/s)
    # 1 ft = 0.3048 m
    # V (m/s) = (0.3048 m) / (dt * 1e-6 s) = 304800.0 / dt
    with np.errstate(divide="ignore", invalid="ignore"):
        vp = 304800.0 / dtc
        vs = 304800.0 / dts

        # Vp/Vs ratio
        vp_vs = vp / vs

        # Dynamic Poisson's Ratio
        # nu = [0.5 * (Vp/Vs)^2 - 1] / [(Vp/Vs)^2 - 1]
        vp_vs_sq = np.square(vp_vs)
        nu_dyn = (0.5 * vp_vs_sq - 1.0) / (vp_vs_sq - 1.0)

        # Bulk density in kg/m3: 1 g/cm3 = 1000 kg/m3
        # Moduli in Pa = rho (kg/m3) * V^2 (m^2/s^2)
        # Moduli in GPa = Pa * 1e-9 = rhob * 1000 * V^2 * 1e-9 = rhob * V^2 * 1e-6
        shear_gpa = rhob * np.square(vs) * 1e-6
        young_gpa = 2.0 * shear_gpa * (1.0 + nu_dyn)
        bulk_gpa = rhob * (np.square(vp) - (4.0 / 3.0) * np.square(vs)) * 1e-6

        # Enforce thermodynamic sanity checks for linear isotropic elasticity
        # 0.0 <= nu < 0.5; E, G, K > 0
        invalid_mask = (
            (nu_dyn < 0.0) | (nu_dyn >= 0.5) | (young_gpa <= 0) | (shear_gpa <= 0) | (bulk_gpa <= 0)
        )
        nu_dyn = np.where(invalid_mask, np.nan, nu_dyn)
        young_gpa = np.where(invalid_mask, np.nan, young_gpa)
        shear_gpa = np.where(invalid_mask, np.nan, shear_gpa)
        bulk_gpa = np.where(invalid_mask, np.nan, bulk_gpa)

        # Conversions to Field Units (1 GPa = 0.1450377377 Mpsi)
        GPA_TO_MPSI = 0.1450377377
        young_mpsi = young_gpa * GPA_TO_MPSI
        shear_mpsi = shear_gpa * GPA_TO_MPSI
        bulk_mpsi = bulk_gpa * GPA_TO_MPSI

    return DynamicElasticModuli(
        vp_m_s=vp,
        vs_m_s=vs,
        vp_vs_ratio=vp_vs,
        poisson_ratio=nu_dyn,
        young_gpa=young_gpa,
        shear_gpa=shear_gpa,
        bulk_gpa=bulk_gpa,
        young_mpsi=young_mpsi,
        shear_mpsi=shear_mpsi,
        bulk_mpsi=bulk_mpsi,
    )
