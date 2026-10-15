"""Perforation cluster placement algorithms: Geometric vs. Engineered Sweet-Spotting.

Geometric: Places clusters at rigid, equal distances across the stage.
Engineered: Locates clusters on local maxima of the Fracability Index (brittle rock, low stress)
subject to anti-collision / stress shadow spacing constraints (d_min >= 8m).
"""

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pandas as pd

from .stage_partitioner import StageBoundary, StagePartitionResult


@dataclass
class PerforationCluster:
    """Individual perforation cluster location and rock mechanical properties.
    
    Attributes:
        cluster_id: Cluster index within the stage (1 to N).
        md_m: Measured depth of the perforation cluster in meters.
        shmin_psi: In-situ minimum horizontal stress at cluster location.
        fracability_index: Fracability Index at cluster location.
        is_sweetspot: True if cluster is located in Tier 1 sweet-spot rock.
    """
    cluster_id: int
    md_m: float
    shmin_psi: float
    fracability_index: float
    is_sweetspot: bool


@dataclass
class StageCompletionPlan:
    """Complete design plan for a single stage including all perforation clusters.
    
    Attributes:
        stage_info: StageBoundary metadata.
        clusters: List of PerforationCluster objects inside this stage.
        cluster_stress_range_psi: Difference between max and min Shmin among clusters (Delta_Shmin).
        mean_cluster_fi: Mean Fracability Index of the selected clusters.
    """
    stage_info: StageBoundary
    clusters: List[PerforationCluster]
    cluster_stress_range_psi: float
    mean_cluster_fi: float


def place_geometric_clusters(
    stage: StageBoundary,
    depth_m: np.ndarray,
    shmin_psi: np.ndarray,
    fi: np.ndarray,
    num_clusters: int = 5,
) -> StageCompletionPlan:
    """Place clusters at rigid, equal spacing within the stage.
    
    Args:
        stage: StageBoundary definition.
        depth_m: Lateral depth array.
        shmin_psi: In-situ stress array.
        fi: Fracability Index array.
        num_clusters: Number of perforation clusters to place.
        
    Returns:
        StageCompletionPlan dataclass.
    """
    st_len = stage.length_m
    spacing = st_len / (num_clusters + 1)

    clusters: List[PerforationCluster] = []
    sh_vals: List[float] = []
    fi_vals: List[float] = []

    for c in range(num_clusters):
        target_md = stage.start_md + (c + 1) * spacing
        idx = int(np.argmin(np.abs(depth_m - target_md)))

        actual_md = float(depth_m[idx])
        c_sh = float(shmin_psi[idx])
        c_fi = float(fi[idx])
        is_sweet = c_fi >= 0.65

        clusters.append(
            PerforationCluster(
                cluster_id=c + 1,
                md_m=round(actual_md, 2),
                shmin_psi=round(c_sh, 1),
                fracability_index=round(c_fi, 3),
                is_sweetspot=is_sweet,
            )
        )
        sh_vals.append(c_sh)
        fi_vals.append(c_fi)

    stress_range = float(np.max(sh_vals) - np.min(sh_vals)) if len(sh_vals) > 0 else 0.0
    mean_fi = float(np.mean(fi_vals)) if len(fi_vals) > 0 else 0.0

    return StageCompletionPlan(
        stage_info=stage,
        clusters=clusters,
        cluster_stress_range_psi=round(stress_range, 1),
        mean_cluster_fi=round(mean_fi, 3),
    )


def place_engineered_clusters(
    stage: StageBoundary,
    depth_m: np.ndarray,
    shmin_psi: np.ndarray,
    fi: np.ndarray,
    num_clusters: int = 5,
    min_cluster_spacing_m: float = 8.0,
    boundary_buffer_m: float = 3.0,
) -> StageCompletionPlan:
    """Place clusters at local sweet-spots (high FI, low stress contrast) within the stage.
    
    Args:
        stage: StageBoundary definition.
        depth_m: Lateral depth array.
        shmin_psi: In-situ stress array.
        fi: Fracability Index array.
        num_clusters: Number of perforation clusters to place.
        min_cluster_spacing_m: Minimum allowed distance between adjacent clusters to avoid severe stress shadow.
        boundary_buffer_m: Buffer zone from stage plugs/boundaries.
        
    Returns:
        StageCompletionPlan dataclass.
    """
    # Usable depth window inside the stage
    win_start = stage.start_md + boundary_buffer_m
    win_end = stage.end_md - boundary_buffer_m
    mask = (depth_m >= win_start) & (depth_m <= win_end)

    stage_depths = depth_m[mask]
    stage_fi = fi[mask]
    stage_sh = shmin_psi[mask]

    if len(stage_depths) < num_clusters:
        # Fallback to geometric if insufficient points
        return place_geometric_clusters(stage, depth_m, shmin_psi, fi, num_clusters)

    # Rank candidate points by highest FI
    sorted_indices = np.argsort(-stage_fi)

    selected_depths: List[float] = []
    selected_sh: List[float] = []
    selected_fi: List[float] = []

    for idx in sorted_indices:
        cand_z = float(stage_depths[idx])
        # Check minimum spacing constraint against already selected clusters
        too_close = any(abs(cand_z - sz) < min_cluster_spacing_m for sz in selected_depths)
        if not too_close:
            selected_depths.append(cand_z)
            selected_sh.append(float(stage_sh[idx]))
            selected_fi.append(float(stage_fi[idx]))
            if len(selected_depths) == num_clusters:
                break

    # If we could not find enough points respecting strict spacing, fill remaining geometrically
    if len(selected_depths) < num_clusters:
        return place_geometric_clusters(stage, depth_m, shmin_psi, fi, num_clusters)

    # Sort selected clusters along measured depth
    order = np.argsort(selected_depths)
    clusters: List[PerforationCluster] = []

    for c_idx, o in enumerate(order):
        md = selected_depths[o]
        c_sh = selected_sh[o]
        c_fi = selected_fi[o]
        clusters.append(
            PerforationCluster(
                cluster_id=c_idx + 1,
                md_m=round(md, 2),
                shmin_psi=round(c_sh, 1),
                fracability_index=round(c_fi, 3),
                is_sweetspot=c_fi >= 0.65,
            )
        )

    stress_range = float(np.max(selected_sh) - np.min(selected_sh))
    mean_fi = float(np.mean(selected_fi))

    return StageCompletionPlan(
        stage_info=stage,
        clusters=clusters,
        cluster_stress_range_psi=round(stress_range, 1),
        mean_cluster_fi=round(mean_fi, 3),
    )
