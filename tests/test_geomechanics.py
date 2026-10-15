"""Unit tests for 1D Mechanical Earth Model (overburden, pore pressure, stresses, breakdown)."""

import numpy as np
import pytest

from src.geomechanics import (
    StressRegime,
    compute_breakdown_pressure,
    compute_eaton_pore_pressure,
    compute_in_situ_stresses,
    compute_overburden_stress,
)


def test_overburden_constant_density_integration():
    """Verify that constant density yields exact analytical Sv = rho * g * z."""
    depths = np.array([0.0, 1000.0, 2000.0])
    rhob = np.array([2.50, 2.50, 2.50])

    ob = compute_overburden_stress(depth_m=depths, rhob_g_cm3=rhob, surface_density_g_cm3=2.50)

    # Sv [MPa] at 1000m = 2.5 * 9.80665 * 1000 * 1e-3 = 24.5166 MPa
    assert np.isclose(ob.sv_mpa[1], 24.5166, atol=1e-2)
    # Sv [MPa] at 2000m = 49.033 MPa
    assert np.isclose(ob.sv_mpa[2], 49.0332, atol=1e-2)

    # Gradient in psi/ft: (2.5 g/cm3) * 0.4335 psi/ft / (1 g/cm3) approx 1.083 psi/ft
    assert np.isclose(ob.gradient_psi_ft[1], 1.083, atol=1e-2)


def test_eaton_pore_pressure_normal_compaction():
    """Verify that when sonic slowness matches NCT, Pp equals hydrostatic pressure."""
    depth_m = np.array([2000.0, 2500.0])
    sv_psi = np.array([7200.0, 9000.0])
    dtc_nct = np.array([80.0, 75.0])
    dtc_actual = np.array([80.0, 75.0])  # Exactly on trendline

    pp = compute_eaton_pore_pressure(
        depth_m=depth_m,
        dtc_us_ft=dtc_actual,
        sv_psi=sv_psi,
        custom_dtc_nct=dtc_nct,
        hydrostatic_gradient_psi_ft=0.433,
    )

    expected_pp_0 = 0.433 * (2000.0 * 3.28084)
    assert np.isclose(pp.pp_psi[0], expected_pp_0, atol=1e-1)
    assert not pp.is_overpressured[0]


def test_eaton_pore_pressure_overpressure_response():
    """Verify that elevated slowness (undercompaction) triggers overpressure detection."""
    depth_m = np.array([2500.0])
    sv_psi = np.array([9000.0])
    dtc_nct = np.array([70.0])
    dtc_actual = np.array([105.0])  # Severe undercompaction (slowness much higher than normal)

    pp = compute_eaton_pore_pressure(
        depth_m=depth_m,
        dtc_us_ft=dtc_actual,
        sv_psi=sv_psi,
        custom_dtc_nct=dtc_nct,
    )

    hydrostatic = 0.433 * (2500.0 * 3.28084)
    # Pp must be significantly above hydrostatic
    assert pp.pp_psi[0] > hydrostatic * 1.3
    assert pp.is_overpressured[0]
    assert pp.gradient_psi_ft[0] > 0.60


def test_in_situ_stresses_analytical_identity():
    """Verify exact analytical identity for horizontal stress anisotropy:
    
    Delta_sigma_h = SHmax - Shmin = [E / (1 + nu)] * (eps_y - eps_x)
    """
    depth_m = np.array([2500.0])
    sv_psi = np.array([9000.0])
    pp_psi = np.array([5500.0])
    e_gpa = np.array([25.0])
    nu = np.array([0.20])
    eps_x = 0.00010
    eps_y = 0.00040

    stresses = compute_in_situ_stresses(
        depth_m=depth_m,
        sv_psi=sv_psi,
        pp_psi=pp_psi,
        young_stat_gpa=e_gpa,
        poisson_stat=nu,
        biot_alpha=0.80,
        tectonic_strain_x=eps_x,
        tectonic_strain_y=eps_y,
    )

    # Theoretical Delta_sigma_h in psi
    e_psi = 25.0 * 145037.7377
    expected_delta_sh = (e_psi / (1.0 + 0.20)) * (eps_y - eps_x)

    actual_delta_sh = stresses.stress_anisotropy_psi[0]
    assert np.isclose(actual_delta_sh, expected_delta_sh, atol=1e-1)
    # Shmin must be greater than pore pressure
    assert stresses.shmin_psi[0] > pp_psi[0]


def test_hydraulic_breakdown_pressure():
    """Verify breakdown pressure calculations and lower boundary."""
    depth_m = np.array([2500.0])
    shmin = np.array([7200.0])
    shmax = np.array([8100.0])
    pp = np.array([5500.0])
    t0 = 800.0

    breakdown = compute_breakdown_pressure(
        depth_m=depth_m,
        shmin_psi=shmin,
        shmax_psi=shmax,
        pp_psi=pp,
        tensile_strength_psi=t0,
    )

    # Pb = 3*7200 - 8100 - 5500 + 800 = 21600 - 12800 = 8800 psi
    assert np.isclose(breakdown.breakdown_pressure_psi[0], 8800.0, atol=1e-1)
    assert breakdown.closure_pressure_psi[0] == 7200.0
    assert breakdown.breakdown_pressure_psi[0] >= breakdown.closure_pressure_psi[0]
