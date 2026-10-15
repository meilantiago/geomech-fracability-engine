"""Completion package for engineered stage segmentation and cluster placement."""

from .stage_partitioner import (
    StageBoundary,
    StagePartitionResult,
    partition_engineered_stages,
    partition_geometric_stages,
)
from .cluster_placer import (
    PerforationCluster,
    StageCompletionPlan,
    place_engineered_clusters,
    place_geometric_clusters,
)
from .evaluator import (
    CompletionComparisonReport,
    evaluate_completion_designs,
)

__all__ = [
    "StageBoundary",
    "StagePartitionResult",
    "partition_engineered_stages",
    "partition_geometric_stages",
    "PerforationCluster",
    "StageCompletionPlan",
    "place_engineered_clusters",
    "place_geometric_clusters",
    "CompletionComparisonReport",
    "evaluate_completion_designs",
]
