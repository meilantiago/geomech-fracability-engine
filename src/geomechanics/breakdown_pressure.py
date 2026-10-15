"""Hydraulic fracture breakdown and reopening pressure prediction.

Equations:
    - Breakdown Pressure (Haimson-Fairhurst): Pb = 3 * Shmin - SHmax - Pp + T0
    - Reopening Pressure: Pr = 3 * Shmin - SHmax - Pp
    - Closure Pressure: Pc = Shmin

References:
    - Haimson, B., and Fairhurst, C. (1967). "Initiation and Extension of Hydraulic Fractures
      in Rocks." SPE Journal, 7(03), 310-318.
    - Hubbert, M.K., and Willis, D.G. (1957). "Mechanics of Hydraulic Fracturing."
      Transactions of the AIME, 210, 153-168.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd


@dataclass
class HydraulicBreakdown:
    """Container for hydraulic fracturing breakdown, reopening, and closure metrics.
    
    Attributes:
        breakdown_pressure_psi: Total wellbore breakdown pressure (Pb) in psi.
        reopening_pressure_psi: Fracture reopening pressure (Pr) in psi.
        closure_pressure_psi: Fracture closure pressure (Pc = Shmin) in psi.
        breakdown_gradient_psi_ft: Breakdown gradient in psi/ft (FIP / TVD).
        net_breakdown_pressure_psi: Net breakdown overbalance (Pb - Pp) in psi.
    """
    breakdown_pressure_psi: np.ndarray
    reopening_pressure_psi: np.ndarray
    closure_pressure_psi: np.ndarray
    breakdown_gradient_psi_ft: np.ndarray
    net_breakdown_pressure_psi: np.ndarray


def compute_breakdown_pressure(
    depth_m: Union[float, np.ndarray, pd.Series],
    shmin_psi: Union[float, np.ndarray, pd.Series],
    shmax_psi: Union[float, np.ndarray, pd.Series],
    pp_psi: Union[float, np.ndarray, pd.Series],
    tensile_strength_psi: float = 650.0,
) -> HydraulicBreakdown:
    """Compute continuous breakdown pressure and initiation gradients for hydraulic fracturing.
    
    Args:
        depth_m: True vertical depth array in meters.
        shmin_psi: Minimum horizontal in-situ stress in psi.
        shmax_psi: Maximum horizontal in-situ stress in psi.
        pp_psi: Pore pressure in psi.
        tensile_strength_psi: Unconfined tensile strength of the shale (T0) in psi.
        
    Returns:
        HydraulicBreakdown dataclass.
    """
    z_m = np.asarray(depth_m, dtype=float)
    shmin = np.asarray(shmin_psi, dtype=float)
    shmax = np.asarray(shmax_psi, dtype=float)
    pp = np.asarray(pp_psi, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Classical Haimson & Fairhurst (1967) breakdown equation
        pb = 3.0 * shmin - shmax - pp + tensile_strength_psi
        pr = 3.0 * shmin - shmax - pp
        pc = shmin

        # Ensure physical bounds (Pb must be >= Shmin)
        pb = np.maximum(pb, shmin)
        pr = np.maximum(pr, pp)

        z_ft = z_m * 3.28084
        pb_grad = pb / z_ft
        net_pb = pb - pp

    return HydraulicBreakdown(
        breakdown_pressure_psi=pb,
        reopening_pressure_psi=pr,
        closure_pressure_psi=pc,
        breakdown_gradient_psi_ft=pb_grad,
        net_breakdown_pressure_psi=net_pb,
    )
