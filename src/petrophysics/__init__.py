"""Petrophysics package for log ingestion, acoustic synthesis, and shale volume estimation."""

from .las_loader import LASLoader, WellLogData
from .mudrock import CastagnaMudrock, synthesize_shear_slowness
from .shale_volume import calculate_vshale, ShaleVolumeModel
from .synthetic_generator import generate_synthetic_vaca_muerta_well

__all__ = [
    "LASLoader",
    "WellLogData",
    "CastagnaMudrock",
    "synthesize_shear_slowness",
    "calculate_vshale",
    "ShaleVolumeModel",
    "generate_synthetic_vaca_muerta_well",
]
