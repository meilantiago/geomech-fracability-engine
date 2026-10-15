"""Horizontal lateral stage partitioning: Geometric vs. Engineered Stress Homogeneity.

In geometric design, stages are sliced at equal intervals (e.g. 60 m), ignoring stress contrast.
In engineered design, stage boundaries are shifted to align with natural stress barriers,
ensuring that each individual stage contains rock of homogeneous minimum horizontal stress (Shmin).
This prevents the "thief cluster" runaway phenomenon where one low-stress cluster takes 80% of slurry.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class StageBoundary:
    """Individual hydraulic fracturing stage geometry and geomechanical metrics.
    
    Attributes:
        stage_id: Stage index (1 to N, from toe to heel or heel to toe).
        start_md: Top measured depth of stage in meters.
        end_md: Bottom measured depth of stage in meters.
        length_m: Length of stage in meters.
        mean_shmin_psi: Mean minimum horizontal in-situ stress in psi.
        std_shmin_psi: Standard deviation of Shmin within this stage in psi.
        mean_fi: Mean composite Fracability Index within this stage.
    """
    stage_id: int
    start_md: float
    end_md: float
    length_m: float
    mean_shmin_psi: float
    std_shmin_psi: float
    mean_fi: float


@dataclass
class StagePartitionResult:
    """Consolidated stage partitioning results across the lateral.
    
    Attributes:
        stages: List of StageBoundary objects.
        total_stages: Number of stages.
        mean_stage_length: Mean length of stages in meters.
        pooled_stress_std_psi: Pooled within-stage standard deviation of Shmin.
        design_type: Description ('Geometric' or 'Engineered').
    """
    stages: List[StageBoundary]
    total_stages: int
    mean_stage_length: float
    pooled_stress_std_psi: float
    design_type: str


def partition_geometric_stages(
    depth_m: Union[float, np.ndarray, pd.Series],
    shmin_psi: Union[float, np.ndarray, pd.Series],
    fracability_index: Union[float, np.ndarray, pd.Series],
    target_length_m: float = 60.0,
) -> StagePartitionResult:
    """Partition horizontal well into uniform geometric stages of identical length.
    
    Args:
        depth_m: Measured depth array in meters.
        shmin_psi: Minimum horizontal stress in psi.
        fracability_index: Fracability Index in [0.0, 1.0].
        target_length_m: Target geometric stage length in meters.
        
    Returns:
        StagePartitionResult dataclass.
    """
    z = np.asarray(depth_m, dtype=float)
    sh = np.asarray(shmin_psi, dtype=float)
    fi = np.asarray(fracability_index, dtype=float)

    z_start = float(z[0])
    z_end = float(z[-1])
    total_length = z_end - z_start

    num_stages = max(1, int(np.round(total_length / target_length_m)))
    actual_length = total_length / num_stages

    stages: List[StageBoundary] = []
    variances: List[float] = []

    for i in range(num_stages):
        st_start = z_start + i * actual_length
        st_end = z_start + (i + 1) * actual_length
        mask = (z >= st_start) & (z <= st_end)

        sh_stage = sh[mask]
        fi_stage = fi[mask]

        mean_sh = float(np.nanmean(sh_stage)) if len(sh_stage) > 0 else 0.0
        std_sh = float(np.nanstd(sh_stage)) if len(sh_stage) > 1 else 0.0
        mean_f = float(np.nanmean(fi_stage)) if len(fi_stage) > 0 else 0.0

        stages.append(
            StageBoundary(
                stage_id=i + 1,
                start_md=round(st_start, 2),
                end_md=round(st_end, 2),
                length_m=round(actual_length, 2),
                mean_shmin_psi=round(mean_sh, 1),
                std_shmin_psi=round(std_sh, 1),
                mean_fi=round(mean_f, 3),
            )
        )
        variances.append(std_sh ** 2)

    pooled_std = float(np.sqrt(np.mean(variances)))

    return StagePartitionResult(
        stages=stages,
        total_stages=num_stages,
        mean_stage_length=round(actual_length, 2),
        pooled_stress_std_psi=round(pooled_std, 1),
        design_type="Geometric",
    )


def partition_engineered_stages(
    depth_m: Union[float, np.ndarray, pd.Series],
    shmin_psi: Union[float, np.ndarray, pd.Series],
    fracability_index: Union[float, np.ndarray, pd.Series],
    target_length_m: float = 60.0,
    min_length_m: float = 45.0,
    max_length_m: float = 75.0,
) -> StagePartitionResult:
    """Partition lateral into engineered stages that minimize intra-stage stress variance.
    
    Identifies natural stress gradients and inflection points (stress barriers)
    and places stage boundaries at high stress gradients rather than inside uniform pay zones.
    
    Args:
        depth_m: Measured depth array in meters.
        shmin_psi: Minimum horizontal stress in psi.
        fracability_index: Fracability Index in [0.0, 1.0].
        target_length_m: Ideal nominal stage length in meters.
        min_length_m: Minimum allowed stage length for operational safety.
        max_length_m: Maximum allowed stage length.
        
    Returns:
        StagePartitionResult dataclass.
    """
    z = np.asarray(depth_m, dtype=float)
    sh = np.asarray(shmin_psi, dtype=float)
    fi = np.asarray(fracability_index, dtype=float)

    z_start = float(z[0])
    z_end = float(z[-1])
    total_length = z_end - z_start

    # Compute stress gradient (rate of change of Shmin per meter)
    dz = np.gradient(z)
    d_sh = np.gradient(sh)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh_grad = np.abs(d_sh / dz)
        sh_grad = np.nan_to_num(sh_grad, nan=0.0)

    # Estimate number of stages
    num_stages = max(1, int(np.round(total_length / target_length_m)))

    current_start = z_start
    stages: List[StageBoundary] = []
    variances: List[float] = []

    for i in range(num_stages):
        if i == num_stages - 1:
            # Last stage takes whatever remains to end of lateral
            candidate_end = z_end
        else:
            # Nominal end based on target
            nominal_end = current_start + target_length_m
            # Search window for optimal stress barrier: [current_start + min_len, current_start + max_len]
            win_min = current_start + min_length_m
            win_max = min(z_end - (num_stages - 1 - i) * min_length_m, current_start + max_length_m)

            if win_max <= win_min:
                candidate_end = nominal_end
            else:
                window_mask = (z >= win_min) & (z <= win_max)
                win_z = z[window_mask]
                win_grad = sh_grad[window_mask]

                if len(win_z) > 0 and np.max(win_grad) > 0:
                    # Pick boundary at the peak stress contrast (natural barrier)
                    best_idx = np.argmax(win_grad)
                    candidate_end = float(win_z[best_idx])
                else:
                    candidate_end = nominal_end

        stage_mask = (z >= current_start) & (z <= candidate_end)
        sh_stage = sh[stage_mask]
        fi_stage = fi[stage_mask]

        actual_len = candidate_end - current_start
        mean_sh = float(np.nanmean(sh_stage)) if len(sh_stage) > 0 else 0.0
        std_sh = float(np.nanstd(sh_stage)) if len(sh_stage) > 1 else 0.0
        mean_f = float(np.nanmean(fi_stage)) if len(fi_stage) > 0 else 0.0

        stages.append(
            StageBoundary(
                stage_id=i + 1,
                start_md=round(current_start, 2),
                end_md=round(candidate_end, 2),
                length_m=round(actual_len, 2),
                mean_shmin_psi=round(mean_sh, 1),
                std_shmin_psi=round(std_sh, 1),
                mean_fi=round(mean_f, 3),
            )
        )
        variances.append(std_sh ** 2)
        current_start = candidate_end

    pooled_std = float(np.sqrt(np.mean(variances)))

    return StagePartitionResult(
        stages=stages,
        total_stages=len(stages),
        mean_stage_length=round(total_length / len(stages), 2),
        pooled_stress_std_psi=round(pooled_std, 1),
        design_type="Engineered",
    )
