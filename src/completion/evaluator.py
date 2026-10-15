"""Quantitative comparison between Geometric and Engineered completion designs.

Quantifies the engineering return on investment (ROI):
- Reduction of within-stage stress dispersion (lowers risk of heel-bias thief clusters)
- Increase in mean Fracability Index at perforations
- Elimination of clusters placed in high-stress barrier intervals or ductile traps
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from .cluster_placer import StageCompletionPlan


@dataclass
class CompletionComparisonReport:
    """Quantitative performance and risk comparison between geometric and engineered designs.
    
    Attributes:
        geometric_mean_fi: Mean Fracability Index across all geometric clusters.
        engineered_mean_fi: Mean Fracability Index across all engineered clusters.
        fi_improvement_pct: Percentage increase in mean Fracability Index.
        geometric_mean_stress_contrast_psi: Mean intra-stage stress contrast for geometric design.
        engineered_mean_stress_contrast_psi: Mean intra-stage stress contrast for engineered design.
        stress_contrast_reduction_pct: Percentage reduction in intra-stage stress contrast.
        geometric_ductile_clusters: Number of clusters placed in ductile/barrier zones (Tier 3).
        engineered_ductile_clusters: Number of clusters placed in ductile/barrier zones (Tier 3).
        total_clusters: Total number of perforation clusters.
    """
    geometric_mean_fi: float
    engineered_mean_fi: float
    fi_improvement_pct: float
    geometric_mean_stress_contrast_psi: float
    engineered_mean_stress_contrast_psi: float
    stress_contrast_reduction_pct: float
    geometric_ductile_clusters: int
    engineered_ductile_clusters: int
    total_clusters: int


def evaluate_completion_designs(
    geometric_plans: List[StageCompletionPlan],
    engineered_plans: List[StageCompletionPlan],
) -> CompletionComparisonReport:
    """Compare geometric and engineered completion plans quantitatively.
    
    Args:
        geometric_plans: List of StageCompletionPlan from geometric partition.
        engineered_plans: List of StageCompletionPlan from engineered partition.
        
    Returns:
        CompletionComparisonReport dataclass.
    """
    # Geometric metrics
    geom_fi_list = [c.fracability_index for p in geometric_plans for c in p.clusters]
    geom_stress_ranges = [p.cluster_stress_range_psi for p in geometric_plans]
    geom_ductile_count = sum(1 for fi in geom_fi_list if fi < 0.45)

    # Engineered metrics
    eng_fi_list = [c.fracability_index for p in engineered_plans for c in p.clusters]
    eng_stress_ranges = [p.cluster_stress_range_psi for p in engineered_plans]
    eng_ductile_count = sum(1 for fi in eng_fi_list if fi < 0.45)

    geom_m_fi = float(np.mean(geom_fi_list)) if geom_fi_list else 0.0
    eng_m_fi = float(np.mean(eng_fi_list)) if eng_fi_list else 0.0

    fi_gain_pct = ((eng_m_fi - geom_m_fi) / geom_m_fi) * 100.0 if geom_m_fi > 0 else 0.0

    geom_m_contrast = float(np.mean(geom_stress_ranges)) if geom_stress_ranges else 0.0
    eng_m_contrast = float(np.mean(eng_stress_ranges)) if eng_stress_ranges else 0.0

    stress_red_pct = (
        ((geom_m_contrast - eng_m_contrast) / geom_m_contrast) * 100.0 if geom_m_contrast > 0 else 0.0
    )

    return CompletionComparisonReport(
        geometric_mean_fi=round(geom_m_fi, 3),
        engineered_mean_fi=round(eng_m_fi, 3),
        fi_improvement_pct=round(fi_gain_pct, 1),
        geometric_mean_stress_contrast_psi=round(geom_m_contrast, 1),
        engineered_mean_stress_contrast_psi=round(eng_m_contrast, 1),
        stress_contrast_reduction_pct=round(stress_red_pct, 1),
        geometric_ductile_clusters=geom_ductile_count,
        engineered_ductile_clusters=eng_ductile_count,
        total_clusters=len(geom_fi_list),
    )
