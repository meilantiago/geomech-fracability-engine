"""End-to-end integration tests for GeomechEngine facade."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.engine import GeomechEngine
from src.petrophysics import generate_synthetic_vaca_muerta_well


def test_geomech_engine_end_to_end():
    """Verify that the engine runs seamlessly on a synthetic well and exports all deliverables."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        las_file = tmp_path / "synthetic_test.las"
        parquet_file = tmp_path / "results.parquet"
        csv_file = tmp_path / "schedule.csv"

        # Generate synthetic test log
        generate_synthetic_vaca_muerta_well(
            start_depth=2500.0,
            stop_depth=2750.0,
            step=0.5,
            output_las_path=las_file,
        )

        engine = GeomechEngine(
            target_stage_length_m=50.0,
            clusters_per_stage=4,
        )

        result = engine.run_from_las(las_file)

        # Check result structure
        assert result.well_data.well_name == "VM-HORIZ-BENCH-01"
        assert len(result.dynamic_moduli.young_gpa) == len(result.well_data.df)
        assert len(result.stress_tensor.shmin_psi) == len(result.well_data.df)
        assert len(result.fracability.fracability_index) == len(result.well_data.df)

        # Check export to DataFrame
        df_all = result.to_dataframe()
        assert "shmin_psi" in df_all.columns
        assert "fracability_index" in df_all.columns
        assert "is_sweetspot" in df_all.columns
        assert len(df_all) == len(result.well_data.df)

        # Check export to Parquet
        result.to_parquet(parquet_file)
        assert parquet_file.exists()
        loaded_df = pd.read_parquet(parquet_file)
        assert len(loaded_df) == len(df_all)

        # Check export to CSV schedule
        schedule_df = result.export_completion_schedule(csv_file, design="engineered")
        assert csv_file.exists()
        assert len(schedule_df) == result.engineered_partition.total_stages * 4
        assert "cluster_md_m" in schedule_df.columns
        assert "cluster_shmin_psi" in schedule_df.columns
