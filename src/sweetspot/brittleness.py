"""Brittleness Index (BI) calculation combining elastic stiffness and mineralogical proxies.

Formulations:
    - Rickman et al. (2008):
        E_norm = (E - E_min) / (E_max - E_min)
        nu_norm = (nu_max - nu) / (nu_max - nu_min)
        BI_elastic = 0.5 * (E_norm + nu_norm)
    - Jarvie et al. (2007) / Wang & Gale (2009):
        Mineralogical proxy: 1 - Vshale (Quartz + Carbonate fraction)

References:
    - Rickman, R., et al. (2008). "A Practical Use of Shale Petrophysics for Stimulation
      Design Optimization: All Shale Plays Are Not Created Equal." SPE 115258.
    - Jarvie, D.M., et al. (2007). "Unconventional shale-gas systems: The Mississippian Barnett
      Shale of north-central Texas as one model for thermogenic shale-gas assessment."
      AAPG Bulletin, 91(4), 475-499.
    - Wang, F.P., and Gale, J.F. (2009). "Screening criteria for shale-gas systems."
      GCAGS Transactions, 59, 779-793.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class ElasticBrittleness:
    """Container for normalized elastic brittleness properties.
    
    Attributes:
        bi_elastic: Rickman Brittleness Index in range [0.0, 1.0].
        young_norm: Normalized Young's modulus in range [0.0, 1.0].
        poisson_norm: Normalized Poisson's ratio in range [0.0, 1.0].
        e_min: Minimum Young's modulus used for normalization.
        e_max: Maximum Young's modulus used for normalization.
        nu_min: Minimum Poisson's ratio used for normalization.
        nu_max: Maximum Poisson's ratio used for normalization.
    """
    bi_elastic: np.ndarray
    young_norm: np.ndarray
    poisson_norm: np.ndarray
    e_min: float
    e_max: float
    nu_min: float
    nu_max: float


def compute_elastic_brittleness(
    young_modulus: Union[float, np.ndarray, pd.Series],
    poisson_ratio: Union[float, np.ndarray, pd.Series],
    e_bounds: Optional[tuple[float, float]] = None,
    nu_bounds: Optional[tuple[float, float]] = None,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> ElasticBrittleness:
    """Compute continuous Rickman Elastic Brittleness Index (BI_elastic).
    
    Args:
        young_modulus: Young's Modulus array (static or dynamic, in GPa or Mpsi).
        poisson_ratio: Poisson's Ratio array.
        e_bounds: Optional (E_min, E_max) boundary tuple.
        nu_bounds: Optional (nu_min, nu_max) boundary tuple.
        percentile_low: Lower percentile for automatic boundary picking.
        percentile_high: Upper percentile for automatic boundary picking.
        
    Returns:
        ElasticBrittleness dataclass.
    """
    e_arr = np.asarray(young_modulus, dtype=float)
    nu_arr = np.asarray(poisson_ratio, dtype=float)

    valid_mask = ~np.isnan(e_arr) & ~np.isnan(nu_arr)

    # Determine Young's modulus bounds
    if e_bounds is not None:
        e_min, e_max = e_bounds
    else:
        e_min = float(np.percentile(e_arr[valid_mask], percentile_low)) if np.any(valid_mask) else 10.0
        e_max = float(np.percentile(e_arr[valid_mask], percentile_high)) if np.any(valid_mask) else 60.0

    # Determine Poisson's ratio bounds
    if nu_bounds is not None:
        nu_min, nu_max = nu_bounds
    else:
        nu_min = float(np.percentile(nu_arr[valid_mask], percentile_low)) if np.any(valid_mask) else 0.15
        nu_max = float(np.percentile(nu_arr[valid_mask], percentile_high)) if np.any(valid_mask) else 0.35

    if e_max <= e_min:
        e_max = e_min + 1.0
    if nu_max <= nu_min:
        nu_max = nu_min + 0.05

    with np.errstate(invalid="ignore"):
        # E_norm: High E -> High Brittleness (ability to maintain fracture aperture)
        e_norm = (e_arr - e_min) / (e_max - e_min)
        e_norm = np.clip(e_norm, 0.0, 1.0)

        # nu_norm: Low nu -> High Brittleness (elastic failure without plastic creep)
        nu_norm = (nu_max - nu_arr) / (nu_max - nu_min)
        nu_norm = np.clip(nu_norm, 0.0, 1.0)

        # Rickman formulation
        bi_elastic = 0.5 * (e_norm + nu_norm)

    return ElasticBrittleness(
        bi_elastic=bi_elastic,
        young_norm=e_norm,
        poisson_norm=nu_norm,
        e_min=e_min,
        e_max=e_max,
        nu_min=nu_min,
        nu_max=nu_max,
    )


def compute_mineral_brittleness_proxy(
    vshale: Union[float, np.ndarray, pd.Series],
    carbonate_volume: Optional[Union[float, np.ndarray, pd.Series]] = None,
) -> np.ndarray:
    """Compute mineralogical brittleness proxy from petrophysical volume fractions.
    
    In unconventional shales, quartz and brittle carbonates promote fracture complexity,
    while ductile clays promote fracture tip blunting and proppant embedment.
    
    Args:
        vshale: Volume fraction of ductile shale/clay in [0.0, 1.0].
        carbonate_volume: Optional volume fraction of brittle dolomite/limestone.
        
    Returns:
        Array of mineralogical brittleness index in [0.0, 1.0].
    """
    vsh = np.asarray(vshale, dtype=float)
    with np.errstate(invalid="ignore"):
        if carbonate_volume is not None:
            vcarb = np.asarray(carbonate_volume, dtype=float)
            bi_min = 1.0 - vsh + 0.2 * vcarb
            bi_min = np.clip(bi_min, 0.0, 1.0)
        else:
            bi_min = 1.0 - vsh
            bi_min = np.clip(bi_min, 0.0, 1.0)
    return bi_min
