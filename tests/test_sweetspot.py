"""Unit tests for Brittleness Index (BI) and Composite Fracability Index (FI)."""

import numpy as np
import pytest

from src.sweetspot import (
    FracabilityTier,
    compute_composite_fracability_index,
    compute_elastic_brittleness,
    compute_mineral_brittleness_proxy,
)


def test_elastic_brittleness_endpoints():
    """Verify Rickman brittleness extremes:
    - High E + Low nu -> High Brittleness (~1.0)
    - Low E + High nu -> Low Brittleness / Ductile (~0.0)
    """
    e_arr = np.array([15.0, 55.0])
    nu_arr = np.array([0.35, 0.16])

    eb = compute_elastic_brittleness(
        young_modulus=e_arr,
        poisson_ratio=nu_arr,
        e_bounds=(15.0, 55.0),
        nu_bounds=(0.16, 0.35),
    )

    # First point: E = 15 (min -> 0), nu = 0.35 (max -> 0) => BI = 0.0
    assert np.isclose(eb.bi_elastic[0], 0.0, atol=1e-3)

    # Second point: E = 55 (max -> 1), nu = 0.16 (min -> 1) => BI = 1.0
    assert np.isclose(eb.bi_elastic[1], 1.0, atol=1e-3)


def test_composite_fracability_tier_classification():
    """Verify composite Fracability Index tiers and sweet-spot detection."""
    # Ideal sweet-spot: high elastic BI (0.9), high mineral BI (0.9), low stress (5500 psi)
    # Poor interval: low elastic BI (0.2), low mineral BI (0.2), high stress (8500 psi)
    bi_elastic = np.array([0.90, 0.20])
    bi_mineral = np.array([0.90, 0.20])
    shmin_psi = np.array([5500.0, 8500.0])

    fi_profile = compute_composite_fracability_index(
        bi_elastic=bi_elastic,
        bi_mineral=bi_mineral,
        shmin_psi=shmin_psi,
        tier_1_threshold=0.65,
        tier_2_threshold=0.45,
    )

    assert fi_profile.fracability_index[0] > 0.70
    assert fi_profile.is_sweetspot[0]
    assert fi_profile.tier[0] == FracabilityTier.TIER_1_SWEETSPOT.value

    assert fi_profile.fracability_index[1] < 0.40
    assert not fi_profile.is_sweetspot[1]
    assert fi_profile.tier[1] == FracabilityTier.TIER_3_BARRIER.value
