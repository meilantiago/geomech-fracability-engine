"""Shear wave acoustic synthesis using Castagna's Mudrock Line and empirical lithology trends.

Reference:
    Castagna, J.P., Batzle, M.L., and Eastwood, R.L. (1985). "Relationships between
    compressional-wave and shear-wave velocities in clastic silicate rocks."
    Geophysics, 50(4), 571-581.
"""

from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


class LithologyClass(str, Enum):
    """Supported lithological classes for Castagna empirical coefficients."""
    SHALE = "shale"
    SANDSTONE = "sandstone"
    LIMESTONE = "limestone"
    DOLOMITE = "dolomite"


class CastagnaMudrock:
    """Computes shear wave velocity (Vs) and slowness (DTS) from compressional wave data."""

    # Coefficients for Vs = a * Vp + b (in km/s)
    # References: Castagna et al. (1985, 1993)
    COEFFICIENTS_KM_S = {
        LithologyClass.SHALE: (0.8621, -1.1724),
        LithologyClass.SANDSTONE: (0.8042, -0.8559),
        LithologyClass.LIMESTONE: (0.5832, -0.0778),
        LithologyClass.DOLOMITE: (0.5832, 0.0),  # Empirical carbonate proxy
    }

    @classmethod
    def dtc_to_vp(cls, dtc: Union[float, np.ndarray, pd.Series], unit: str = "us/ft") -> np.ndarray:
        """Convert compressional slowness (DTC) to velocity (Vp).
        
        Args:
            dtc: Compressional slowness.
            unit: Unit of slowness ('us/ft' or 'us/m').
            
        Returns:
            Compressional velocity Vp in km/s.
        """
        dtc_arr = np.asarray(dtc, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            if unit.lower() in ["us/ft", "usec/ft"]:
                # 1 ft = 0.3048 m = 0.0003048 km
                # Vp (km/s) = (0.3048 * 1e-3 km) / (dtc * 1e-6 s) = 304.8 / dtc
                vp_km_s = 304.8 / dtc_arr
            elif unit.lower() in ["us/m", "usec/m"]:
                # Vp (km/s) = 1e-3 km / (dtc * 1e-6 s) = 1000.0 / dtc
                vp_km_s = 1000.0 / dtc_arr
            else:
                raise ValueError(f"Unsupported acoustic slowness unit: {unit}")
        return vp_km_s

    @classmethod
    def vs_to_dts(cls, vs_km_s: Union[float, np.ndarray, pd.Series], unit: str = "us/ft") -> np.ndarray:
        """Convert shear velocity (Vs in km/s) to slowness (DTS).
        
        Args:
            vs_km_s: Shear velocity in km/s.
            unit: Desired unit ('us/ft' or 'us/m').
            
        Returns:
            Shear slowness DTS in requested unit.
        """
        vs_arr = np.asarray(vs_km_s, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            if unit.lower() in ["us/ft", "usec/ft"]:
                dts = 304.8 / vs_arr
            elif unit.lower() in ["us/m", "usec/m"]:
                dts = 1000.0 / vs_arr
            else:
                raise ValueError(f"Unsupported acoustic slowness unit: {unit}")
        return dts

    @classmethod
    def predict_vs(
        cls,
        vp_km_s: Union[float, np.ndarray, pd.Series],
        lithology: Union[str, LithologyClass] = LithologyClass.SHALE,
        custom_coefficients: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Predict shear wave velocity (Vs in km/s) from compressional velocity (Vp in km/s).
        
        Args:
            vp_km_s: Compressional velocity in km/s.
            lithology: Target lithology class.
            custom_coefficients: Optional (a, b) tuple overriding defaults.
            
        Returns:
            Predicted shear velocity Vs in km/s.
        """
        vp_arr = np.asarray(vp_km_s, dtype=float)
        if custom_coefficients is not None:
            a, b = custom_coefficients
        else:
            if isinstance(lithology, LithologyClass):
                litho_enum = lithology
            else:
                val = lithology.value if hasattr(lithology, "value") else str(lithology)
                litho_enum = LithologyClass(val.lower())
            a, b = cls.COEFFICIENTS_KM_S[litho_enum]

        vs_km_s = a * vp_arr + b

        # Physical constraints: Vs must be positive and Vp/Vs > sqrt(2) for isotropic elastic solids
        # sqrt(2) approx 1.4142 (Poisson's ratio > 0)
        with np.errstate(invalid="ignore"):
            vs_km_s = np.where(vs_km_s <= 0.1, np.nan, vs_km_s)
            max_vs = vp_arr / 1.4143
            vs_km_s = np.minimum(vs_km_s, max_vs)

        return vs_km_s


def synthesize_shear_slowness(
    dtc: Union[float, np.ndarray, pd.Series],
    dts_existing: Optional[Union[float, np.ndarray, pd.Series]] = None,
    lithology: Union[str, LithologyClass] = LithologyClass.SHALE,
    unit: str = "us/ft",
) -> np.ndarray:
    """Synthesize DTS from DTC, patching missing values or generating complete synthetic curve.
    
    Args:
        dtc: Compressional slowness series.
        dts_existing: Optional existing shear slowness (with missing NaNs).
        lithology: Lithology class for empirical parameters.
        unit: Unit of slowness ('us/ft' or 'us/m').
        
    Returns:
        Continuous DTS array with synthesized values where missing.
    """
    dtc_arr = np.asarray(dtc, dtype=float)
    vp_km_s = CastagnaMudrock.dtc_to_vp(dtc_arr, unit=unit)
    vs_km_s = CastagnaMudrock.predict_vs(vp_km_s, lithology=lithology)
    dts_synthetic = CastagnaMudrock.vs_to_dts(vs_km_s, unit=unit)

    if dts_existing is None:
        return dts_synthetic

    dts_orig = np.asarray(dts_existing, dtype=float).copy()
    invalid_mask = np.isnan(dts_orig) | (dts_orig <= 0) | (dts_orig > 500)
    dts_orig[invalid_mask] = dts_synthetic[invalid_mask]
    return dts_orig
