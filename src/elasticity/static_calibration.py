"""Static elastic moduli calibration from dynamic acoustic measurements.

Dynamic acoustic measurements (high frequency ~10 kHz to 1 MHz, ultra-low strain ~10^-6)
significantly overestimate static rock deformability under reservoir stress conditions
(strain rates ~10^-4 to 10^-2). Dynamic-to-static transformation is mandatory for
reliable hydraulic fracture geometry and in-situ stress predictions.

References:
    - Bradford, I.D.R., et al. (1998). "Dynamic-to-static moduli transitions in soft rocks
      and shales." SPE 47277.
    - Lacy, L.L. (1997). "Dynamic rock mechanics testing for optimized fracture designs."
      SPE 38716.
    - Morales, R.H., and Marcinew, R.P. (1993). "Fracturing of high-permeability formations:
      mechanical properties correlation." SPE 26561.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


class CalibrationMethod(str, Enum):
    """Empirical calibration models for dynamic-to-static transformation."""
    BRADFORD_SHALE = "bradford_shale"  # Standard for clay/organic-rich shales
    LACY = "lacy"                      # Unconventional/tight clastic rocks
    MORALES = "morales"                # High-porosity / semi-consolidated rocks
    CUSTOM = "custom"


@dataclass
class StaticElasticModuli:
    """Container for calibrated static rock mechanical properties.
    
    Attributes:
        young_gpa: Static Young's modulus in GPa.
        poisson_ratio: Static Poisson's ratio (dimensionless).
        young_mpsi: Static Young's modulus in Mpsi.
        method: Method used for dynamic-to-static calibration.
    """
    young_gpa: np.ndarray
    poisson_ratio: np.ndarray
    young_mpsi: np.ndarray
    method: str


def calibrate_static_moduli(
    young_dyn_gpa: Union[float, np.ndarray, pd.Series],
    poisson_dyn: Union[float, np.ndarray, pd.Series],
    method: Union[str, CalibrationMethod] = CalibrationMethod.BRADFORD_SHALE,
    custom_params: Optional[Tuple[float, float, float]] = None,
) -> StaticElasticModuli:
    """Calibrate dynamic acoustic Young's Modulus and Poisson's Ratio to static reservoir conditions.
    
    Args:
        young_dyn_gpa: Dynamic Young's modulus array in GPa.
        poisson_dyn: Dynamic Poisson's ratio array.
        method: Chosen empirical transformation model.
        custom_params: Optional tuple (a, b, nu_mult) for custom model:
                       E_stat = a * E_dyn + b; nu_stat = nu_mult * nu_dyn.
                       
    Returns:
        StaticElasticModuli dataclass.
    """
    e_dyn = np.asarray(young_dyn_gpa, dtype=float)
    nu_dyn = np.asarray(poisson_dyn, dtype=float)
    if isinstance(method, CalibrationMethod):
        method_enum = method
    else:
        val = method.value if hasattr(method, "value") else str(method)
        method_enum = CalibrationMethod(val.lower())

    with np.errstate(invalid="ignore"):
        if method_enum == CalibrationMethod.BRADFORD_SHALE:
            # Bradford et al. (1998) for organic/tight shales:
            # E_stat (GPa) = 0.415 * E_dyn - 1.059
            # nu_stat = 0.70 * nu_dyn
            e_stat = 0.415 * e_dyn - 1.059
            nu_stat = 0.70 * nu_dyn

        elif method_enum == CalibrationMethod.LACY:
            # Lacy (1997) polynomial model:
            # E_stat = 0.0293 * E_dyn^2 + 0.445 * E_dyn
            # nu_stat = 0.75 * nu_dyn
            e_stat = 0.0293 * np.square(e_dyn) + 0.445 * e_dyn
            nu_stat = 0.75 * nu_dyn

        elif method_enum == CalibrationMethod.MORALES:
            # Morales & Marcinew (1993) linear scaling:
            # E_stat = 0.55 * E_dyn
            # nu_stat = 0.80 * nu_dyn
            e_stat = 0.55 * e_dyn
            nu_stat = 0.80 * nu_dyn

        elif method_enum == CalibrationMethod.CUSTOM:
            if custom_params is None:
                raise ValueError("custom_params tuple (a, b, nu_mult) must be provided for CUSTOM method")
            a, b, nu_mult = custom_params
            e_stat = a * e_dyn + b
            nu_stat = nu_mult * nu_dyn

        else:
            raise ValueError(f"Unknown calibration method: {method}")

        # Ensure physical boundaries for static rock mechanics
        # E_stat > 1.0 GPa; 0.05 <= nu_stat <= 0.45
        e_stat = np.where(np.isnan(e_dyn), np.nan, np.maximum(e_stat, 1.0))
        nu_stat = np.where(np.isnan(nu_dyn), np.nan, np.clip(nu_stat, 0.05, 0.45))

        GPA_TO_MPSI = 0.1450377377
        e_stat_mpsi = e_stat * GPA_TO_MPSI

    return StaticElasticModuli(
        young_gpa=e_stat,
        poisson_ratio=nu_stat,
        young_mpsi=e_stat_mpsi,
        method=method_enum.value,
    )
