"""Integrated Composite Fracability Index (FI) for hydraulic fracturing sweet-spot identification.

Fracability integrates:
1. Elastic Brittleness (Rickman): Capacity to sustain fracture conductivity without plastic deformation.
2. Mineralogical Brittleness: Low ductile clay content, high quartz/carbonate matrix.
3. In-Situ Stress Containment: Low minimum horizontal stress (Shmin) to promote low initiation pressure
   and avoid stress barrier dissipation.

References:
    - Jin, G., et al. (2014). "A Comprehensive Method for Identifying Fracability in Unconventional
      Shale Reservoirs." SPE 171545.
    - Yuan, J., et al. (2017). "Fracability Evaluation of Tight Oil Reservoirs." SPE 187498.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd


class FracabilityTier(str, Enum):
    """Quality classification of formation fracability."""
    TIER_1_SWEETSPOT = "Tier 1: High Fracability Sweet-Spot (Brittle / Low Stress)"
    TIER_2_MODERATE = "Tier 2: Moderate Fracability (Standard Reservoir)"
    TIER_3_BARRIER = "Tier 3: Poor Fracability / Stress Barrier (Ductile / High Stress)"


@dataclass
class FracabilityProfile:
    """Container for multi-criteria fracability evaluation.
    
    Attributes:
        fracability_index: Composite Fracability Index (FI) in range [0.0, 1.0].
        bi_elastic: Rickman elastic brittleness index.
        bi_mineral: Mineralogical brittleness proxy.
        stress_containment_score: Inverted normalized minimum horizontal stress (1 - Shmin_norm).
        tier: Qualitative category (Tier 1, 2, or 3).
        is_sweetspot: Boolean mask identifying Tier 1 sweet-spots.
    """
    fracability_index: np.ndarray
    bi_elastic: np.ndarray
    bi_mineral: np.ndarray
    stress_containment_score: np.ndarray
    tier: np.ndarray
    is_sweetspot: np.ndarray


def compute_composite_fracability_index(
    bi_elastic: Union[float, np.ndarray, pd.Series],
    bi_mineral: Union[float, np.ndarray, pd.Series],
    shmin_psi: Union[float, np.ndarray, pd.Series],
    weight_elastic: float = 0.45,
    weight_mineral: float = 0.25,
    weight_stress: float = 0.30,
    tier_1_threshold: float = 0.65,
    tier_2_threshold: float = 0.45,
) -> FracabilityProfile:
    """Compute integrated multi-criteria Fracability Index (FI) along the well profile.
    
    Args:
        bi_elastic: Normalized elastic brittleness index in [0.0, 1.0].
        bi_mineral: Normalized mineralogical brittleness proxy in [0.0, 1.0].
        shmin_psi: Minimum horizontal in-situ stress profile in psi.
        weight_elastic: Multi-criteria weight for elastic stiffness.
        weight_mineral: Multi-criteria weight for mineralogy.
        weight_stress: Multi-criteria weight for in-situ stress confinement.
        tier_1_threshold: Threshold above which interval is classified as Tier 1 sweet-spot.
        tier_2_threshold: Threshold separating Tier 2 moderate from Tier 3 ductile barrier.
        
    Returns:
        FracabilityProfile dataclass.
    """
    # Normalize weights to sum to 1.0
    w_sum = weight_elastic + weight_mineral + weight_stress
    w_el = weight_elastic / w_sum
    w_min = weight_mineral / w_sum
    w_str = weight_stress / w_sum

    bie = np.asarray(bi_elastic, dtype=float)
    bim = np.asarray(bi_mineral, dtype=float)
    shmin = np.asarray(shmin_psi, dtype=float)

    # Invert and normalize Shmin: low stress = high fracability score
    valid_shmin = shmin[~np.isnan(shmin)]
    if len(valid_shmin) > 0:
        s_min = float(np.percentile(valid_shmin, 2.0))
        s_max = float(np.percentile(valid_shmin, 98.0))
        if s_max <= s_min:
            s_max = s_min + 100.0
    else:
        s_min, s_max = 5000.0, 9000.0

    with np.errstate(invalid="ignore"):
        shmin_norm = (shmin - s_min) / (s_max - s_min)
        shmin_norm = np.clip(shmin_norm, 0.0, 1.0)
        # Score is inverted: 1.0 for lowest stress, 0.0 for highest stress barrier
        stress_score = 1.0 - shmin_norm

        # Composite multi-criteria formula
        fi = w_el * bie + w_min * bim + w_str * stress_score
        fi = np.clip(fi, 0.0, 1.0)

        # Categorize into Tiers
        tiers = np.empty(len(fi), dtype=object)
        for i in range(len(fi)):
            val = fi[i]
            if np.isnan(val):
                tiers[i] = "Unknown"
            elif val >= tier_1_threshold:
                tiers[i] = FracabilityTier.TIER_1_SWEETSPOT.value
            elif val >= tier_2_threshold:
                tiers[i] = FracabilityTier.TIER_2_MODERATE.value
            else:
                tiers[i] = FracabilityTier.TIER_3_BARRIER.value

        is_sweetspot = (fi >= tier_1_threshold)

    return FracabilityProfile(
        fracability_index=fi,
        bi_elastic=bie,
        bi_mineral=bim,
        stress_containment_score=stress_score,
        tier=tiers,
        is_sweetspot=is_sweetspot,
    )
