"""Unit tests for petrophysics ingestion, acoustic synthesis, and shale volume estimation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.petrophysics import (
    CastagnaMudrock,
    LASLoader,
    ShaleVolumeModel,
    calculate_vshale,
    generate_synthetic_vaca_muerta_well,
    synthesize_shear_slowness,
)


def test_vshale_models_boundaries():
    """Verify that all Vshale non-linear models map GR within [0.0, 1.0]."""
    gr = np.linspace(20.0, 200.0, 50)
    for model in ShaleVolumeModel:
        vsh = calculate_vshale(gr, model=model, gr_clean=30.0, gr_shale=180.0)
        assert np.all(vsh >= 0.0)
        assert np.all(vsh <= 1.0)
        # Monotonically non-decreasing
        assert np.all(np.diff(vsh) >= -1e-6)


def test_vshale_clean_and_shale_endpoints():
    """Verify that clean sand gives Vsh ~ 0 and pure shale gives Vsh ~ 1."""
    gr_clean = 25.0
    gr_shale = 150.0
    vsh = calculate_vshale(
        [gr_clean, gr_shale],
        model=ShaleVolumeModel.LARIONOV_OLDER,
        gr_clean=gr_clean,
        gr_shale=gr_shale,
    )
    assert np.isclose(vsh[0], 0.0, atol=1e-3)
    assert np.isclose(vsh[1], 1.0, atol=1e-3)


def test_castagna_dtc_to_vp_conversion():
    """Test acoustic slowness to velocity conversion."""
    # DTC = 100 us/ft -> Vp = 304.8 / 100 = 3.048 km/s
    vp = CastagnaMudrock.dtc_to_vp(100.0, unit="us/ft")
    assert np.isclose(vp, 3.048, atol=1e-3)


def test_castagna_vs_prediction_physical_bounds():
    """Test that predicted Vs satisfies physical rock mechanics limits."""
    vp_km_s = np.array([2.5, 3.5, 4.5, 5.5])
    vs_km_s = CastagnaMudrock.predict_vs(vp_km_s, lithology="shale")
    assert np.all(vs_km_s > 0)
    # Vp/Vs must be > sqrt(2) approx 1.414 for positive Poisson's ratio
    assert np.all(vp_km_s / vs_km_s > 1.414)


def test_synthesize_shear_slowness_patching():
    """Test that synthesize_shear_slowness patches missing NaNs without corrupting valid values."""
    dtc = np.array([75.0, 80.0, 85.0])
    dts_existing = np.array([135.0, np.nan, 150.0])
    dts_filled = synthesize_shear_slowness(dtc, dts_existing=dts_existing)
    assert not np.isnan(dts_filled[1])
    assert dts_filled[0] == 135.0
    assert dts_filled[2] == 150.0


def test_las_generation_and_loading():
    """Test end-to-end LAS creation and loading roundtrip."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        las_path = Path(tmp_dir) / "test_well.las"
        df_gen = generate_synthetic_vaca_muerta_well(
            start_depth=2500.0,
            stop_depth=2550.0,
            step=1.0,
            output_las_path=las_path,
        )
        assert las_path.exists()

        loader = LASLoader()
        well_data = loader.load(las_path)

        assert well_data.well_name == "VM-HORIZ-BENCH-01"
        assert len(well_data.df) == len(df_gen)
        assert "depth" in well_data.df.columns
        assert "gr" in well_data.df.columns
        assert "rhob" in well_data.df.columns
        assert "dtc" in well_data.df.columns
        assert "dts" in well_data.df.columns
