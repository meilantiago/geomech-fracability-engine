"""Shale volume (Vshale) estimation from Gamma Ray logs using industry standard non-linear models.

References:
    - Larionov, V.V. (1969). Borehole Radiometry: Moscow, U.S.S.R., Nedra.
    - Stieber, S.J. (1970). "Pulsed neutron capture log evaluation in the Louisiana Gulf Coast."
      SPE 2961.
    - Clavier, C., Hoyle, W., and Meunier, D. (1971). "Quantitative interpretation of TDT logs."
      Journal of Petroleum Technology, 23(06), 743-763.
"""

from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd


class ShaleVolumeModel(str, Enum):
    """Supported mathematical models for converting Gamma Ray Index to Vshale."""
    LINEAR = "linear"
    LARIONOV_TERTIARY = "larionov_tertiary"
    LARIONOV_OLDER = "larionov_older"  # Recommended for Mesozoic/Paleozoic shales (e.g. Vaca Muerta)
    STIEBER = "stieber"
    CLAVIER = "clavier"


def calculate_gamma_ray_index(
    gr: Union[float, np.ndarray, pd.Series],
    gr_clean: Optional[float] = None,
    gr_shale: Optional[float] = None,
    percentile_clean: float = 5.0,
    percentile_shale: float = 95.0,
) -> np.ndarray:
    """Calculate normalized Gamma Ray Index (IGR) clamped to [0.0, 1.0].
    
    Args:
        gr: Gamma ray curve array in API units.
        gr_clean: Optional manual baseline for clean (sand/carbonate) formation.
        gr_shale: Optional manual baseline for 100% shale formation.
        percentile_clean: Percentile used to auto-pick gr_clean if not provided.
        percentile_shale: Percentile used to auto-pick gr_shale if not provided.
        
    Returns:
        Array of Gamma Ray Index (IGR) values.
    """
    gr_arr = np.asarray(gr, dtype=float)
    valid_gr = gr_arr[~np.isnan(gr_arr)]

    if len(valid_gr) == 0:
        return np.full_like(gr_arr, np.nan)

    clean = gr_clean if gr_clean is not None else float(np.percentile(valid_gr, percentile_clean))
    shale = gr_shale if gr_shale is not None else float(np.percentile(valid_gr, percentile_shale))

    if shale <= clean:
        raise ValueError(f"gr_shale ({shale:.1f}) must be strictly greater than gr_clean ({clean:.1f})")

    with np.errstate(invalid="ignore"):
        igr = (gr_arr - clean) / (shale - clean)
        igr = np.clip(igr, 0.0, 1.0)

    return igr


def calculate_vshale(
    gr: Union[float, np.ndarray, pd.Series],
    model: Union[str, ShaleVolumeModel] = ShaleVolumeModel.LARIONOV_OLDER,
    gr_clean: Optional[float] = None,
    gr_shale: Optional[float] = None,
) -> np.ndarray:
    """Calculate volume of shale (Vshale) from Gamma Ray using selected non-linear model.
    
    Args:
        gr: Gamma Ray array in API units.
        model: Non-linear correction model.
        gr_clean: Clean sand/carbonate GR baseline.
        gr_shale: Pure shale GR baseline.
        
    Returns:
        Array of Vshale fractions in [0.0, 1.0].
    """
    if isinstance(model, ShaleVolumeModel):
        model_enum = model
    else:
        val = model.value if hasattr(model, "value") else str(model)
        model_enum = ShaleVolumeModel(val.lower())
    igr = calculate_gamma_ray_index(gr, gr_clean=gr_clean, gr_shale=gr_shale)

    with np.errstate(invalid="ignore"):
        if model_enum == ShaleVolumeModel.LINEAR:
            vsh = igr

        elif model_enum == ShaleVolumeModel.LARIONOV_TERTIARY:
            # Vsh = 0.083 * (2^(3.7 * IGR) - 1)
            vsh = 0.083 * (np.power(2.0, 3.7 * igr) - 1.0)

        elif model_enum == ShaleVolumeModel.LARIONOV_OLDER:
            # Vsh = (1/3) * (2^(2.0 * IGR) - 1) (Ideal for Mesozoic Vaca Muerta, exact [0, 1] normalization)
            vsh = (1.0 / 3.0) * (np.power(2.0, 2.0 * igr) - 1.0)

        elif model_enum == ShaleVolumeModel.STIEBER:
            # Vsh = IGR / (3.0 - 2.0 * IGR)
            vsh = igr / (3.0 - 2.0 * igr)

        elif model_enum == ShaleVolumeModel.CLAVIER:
            # Vsh = 1.7 - sqrt(3.38 - (IGR + 0.7)^2)
            arg = np.maximum(0.0, 3.38 - np.power(igr + 0.7, 2.0))
            vsh = 1.7 - np.sqrt(arg)

        else:
            raise ValueError(f"Unknown Vshale model: {model}")

        vsh = np.clip(vsh, 0.0, 1.0)

    return vsh
