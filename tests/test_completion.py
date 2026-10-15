"""Unit tests for completion stage partitioning, cluster sweet-spotting, and design comparison."""

import numpy as np
import pytest

from src.completion import (
    evaluate_completion_designs,
    partition_engineered_stages,
    partition_geometric_stages,
    place_engineered_clusters,
    place_geometric_clusters,
)


@pytest.fixture
def lateral_well_profile():
    """Create a realistic lateral well profile with variable stress and fracability."""
    depths = np.linspace(2500.0, 3000.0, 501)  # 500m lateral, 1m step
    # Stress profile with two distinct lithological blocks (barrier at 2750m)
    shmin = 6500.0 + 800.0 * np.sin(2.0 * np.pi * (depths - 2500.0) / 250.0)
    # Fracability anti-correlated with stress
    fi = 0.55 - 0.25 * np.sin(2.0 * np.pi * (depths - 2500.0) / 250.0)
    return depths, shmin, fi


def test_geometric_partitioning_conservation(lateral_well_profile):
    """Verify that geometric partitioning spans the entire lateral without gaps."""
    depths, shmin, fi = lateral_well_profile
    geom = partition_geometric_stages(depths, shmin, fi, target_length_m=60.0)

    assert geom.stages[0].start_md == 2500.0
    assert geom.stages[-1].end_md == 3000.0
    total_len = sum(s.length_m for s in geom.stages)
    assert np.isclose(total_len, 500.0, atol=1e-1)


def test_engineered_partitioning_reduces_stress_variance(lateral_well_profile):
    """Verify that engineered stage partitioning reduces pooled intra-stage stress variance."""
    depths, shmin, fi = lateral_well_profile
    geom = partition_geometric_stages(depths, shmin, fi, target_length_m=60.0)
    eng = partition_engineered_stages(depths, shmin, fi, target_length_m=60.0, min_length_m=45.0, max_length_m=75.0)

    # Engineered pooled standard deviation must be less than geometric
    assert eng.pooled_stress_std_psi <= geom.pooled_stress_std_psi


def test_engineered_cluster_placement_spacing_constraint(lateral_well_profile):
    """Verify that engineered cluster placement strictly enforces the minimum cluster distance."""
    depths, shmin, fi = lateral_well_profile
    geom = partition_geometric_stages(depths, shmin, fi, target_length_m=60.0)
    first_stage = geom.stages[0]

    plan = place_engineered_clusters(
        stage=first_stage,
        depth_m=depths,
        shmin_psi=shmin,
        fi=fi,
        num_clusters=5,
        min_cluster_spacing_m=8.0,
        boundary_buffer_m=2.0,
    )

    assert len(plan.clusters) == 5
    cluster_mds = [c.md_m for c in plan.clusters]
    diffs = np.diff(cluster_mds)
    # Every adjacent cluster pair must be separated by at least 8.0 meters
    assert np.all(diffs >= 8.0)


def test_completion_evaluation_metrics(lateral_well_profile):
    """Verify evaluation report calculation comparing geometric vs engineered."""
    depths, shmin, fi = lateral_well_profile
    geom_part = partition_geometric_stages(depths, shmin, fi, target_length_m=60.0)
    eng_part = partition_engineered_stages(depths, shmin, fi, target_length_m=60.0)

    geom_plans = [place_geometric_clusters(s, depths, shmin, fi, num_clusters=4) for s in geom_part.stages]
    eng_plans = [place_engineered_clusters(s, depths, shmin, fi, num_clusters=4) for s in eng_part.stages]

    report = evaluate_completion_designs(geom_plans, eng_plans)

    assert report.total_clusters == len(geom_plans) * 4
    assert report.engineered_mean_fi >= report.geometric_mean_fi
    assert report.stress_contrast_reduction_pct >= 0.0
