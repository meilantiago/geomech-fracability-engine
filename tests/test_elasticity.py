"""Unit tests for dynamic elastic moduli and static geomechanical calibration."""

import numpy as np
import pytest

from src.elasticity import (
    CalibrationMethod,
    calibrate_static_moduli,
    compute_dynamic_moduli,
)


def test_dynamic_moduli_analytical_solution():
    """Verify dynamic moduli calculation against exact analytical rock physics equations.
    
    Given:
        Vp = 4000 m/s
        Vs = 2400 m/s
        RHOB = 2.50 g/cm3
    
    Analytical Derivations:
        Vp/Vs = 5/3 = 1.6667
        (Vp/Vs)^2 = 25/9 = 2.7778
        nu_dyn = (0.5 * 25/9 - 1) / (25/9 - 1) = (7/18) / (16/9) = 7/32 = 0.21875
        G_dyn = 2.5 * 2400^2 * 1e-6 = 14.40 GPa
        E_dyn = 2 * 14.40 * (1 + 0.21875) = 35.10 GPa
        K_dyn = 2.5 * (4000^2 - (4/3)*2400^2) * 1e-6 = 20.80 GPa
    """
    dtc_us_ft = 304800.0 / 4000.0  # 76.2 us/ft
    dts_us_ft = 304800.0 / 2400.0  # 127.0 us/ft
    rhob = 2.50

    moduli = compute_dynamic_moduli(rhob_g_cm3=rhob, dtc_us_ft=dtc_us_ft, dts_us_ft=dts_us_ft)

    assert np.isclose(moduli.poisson_ratio[0], 0.21875, atol=1e-4)
    assert np.isclose(moduli.shear_gpa[0], 14.40, atol=1e-2)
    assert np.isclose(moduli.young_gpa[0], 35.10, atol=1e-2)
    assert np.isclose(moduli.bulk_gpa[0], 20.80, atol=1e-2)

    # Elastic identity check: K = E / [3*(1 - 2*nu)]
    k_check = moduli.young_gpa[0] / (3.0 * (1.0 - 2.0 * moduli.poisson_ratio[0]))
    assert np.isclose(moduli.bulk_gpa[0], k_check, atol=1e-3)


def test_static_calibration_bradford_shale():
    """Verify Bradford dynamic-to-static transformation model."""
    young_dyn = np.array([40.0, 50.0])
    poisson_dyn = np.array([0.25, 0.30])

    stat = calibrate_static_moduli(
        young_dyn_gpa=young_dyn,
        poisson_dyn=poisson_dyn,
        method=CalibrationMethod.BRADFORD_SHALE,
    )

    # E_stat = 0.415 * 40 - 1.059 = 15.541 GPa
    assert np.isclose(stat.young_gpa[0], 15.541, atol=1e-3)
    # nu_stat = 0.70 * 0.25 = 0.175
    assert np.isclose(stat.poisson_ratio[0], 0.175, atol=1e-3)
    # E_stat must be lower than E_dyn (rock is softer statically)
    assert np.all(stat.young_gpa < young_dyn)
    assert np.all(stat.poisson_ratio < poisson_dyn)


def test_static_calibration_boundaries():
    """Verify that static calibration clamps to physical rock boundaries."""
    # Ultra-low or invalid values
    young_dyn = np.array([1.0, 100.0])
    poisson_dyn = np.array([0.02, 0.48])

    stat = calibrate_static_moduli(
        young_dyn_gpa=young_dyn,
        poisson_dyn=poisson_dyn,
        method=CalibrationMethod.BRADFORD_SHALE,
    )

    assert np.all(stat.young_gpa >= 1.0)
    assert np.all(stat.poisson_ratio >= 0.05)
    assert np.all(stat.poisson_ratio <= 0.45)
