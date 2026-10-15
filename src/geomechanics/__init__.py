"""1D Mechanical Earth Model (1D MEM) package for overburden, pore pressure, and in-situ stresses."""

from .overburden import OverburdenStress, compute_overburden_stress
from .pore_pressure import PorePressureProfile, compute_eaton_pore_pressure
from .in_situ_stress import (
    StressRegime,
    StressTensor1D,
    compute_in_situ_stresses,
)
from .breakdown_pressure import (
    HydraulicBreakdown,
    compute_breakdown_pressure,
)

__all__ = [
    "OverburdenStress",
    "compute_overburden_stress",
    "PorePressureProfile",
    "compute_eaton_pore_pressure",
    "StressRegime",
    "StressTensor1D",
    "compute_in_situ_stresses",
    "HydraulicBreakdown",
    "compute_breakdown_pressure",
]
