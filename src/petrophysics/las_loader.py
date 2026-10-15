"""Module for loading and standardizing well log files in CWLS Log ASCII Standard (.las) format."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lasio
import numpy as np
import pandas as pd


@dataclass
class WellLogData:
    """Container for standardized well log data and metadata.
    
    Attributes:
        df: DataFrame containing standard curves indexed or columned with canonical names.
        well_name: Name of the well from LAS header.
        field_name: Field or asset name from LAS header.
        start_depth: Top depth of the logged interval.
        stop_depth: Bottom depth of the logged interval.
        step: Depth step interval.
        depth_unit: Unit of depth (m or ft).
        raw_curves: List of original curve mnemonics present in the file.
        mapped_curves: Dictionary of canonical name to original mnemonic.
    """
    df: pd.DataFrame
    well_name: str = "UNKNOWN"
    field_name: str = "UNKNOWN"
    start_depth: float = 0.0
    stop_depth: float = 0.0
    step: float = 0.0
    depth_unit: str = "m"
    raw_curves: List[str] = field(default_factory=list)
    mapped_curves: Dict[str, str] = field(default_factory=dict)


class LASLoader:
    """Robust parser and standardizer for .las well log files using lasio."""

    CANONICAL_ALIASES: Dict[str, List[str]] = {
        "depth": ["DEPTH", "DEPT", "MD", "TVD", "TIME"],
        "gr": ["GR", "GAMMA", "CGR", "SGR", "GRD", "GR_EDTC", "GRR", "HGR"],
        "rhob": ["RHOB", "DEN", "RHOZ", "RHO8", "DEN_COMP", "RHO", "DENS"],
        "dtc": ["DTC", "DT", "DTCO", "AC", "DT_COMP", "D4TP", "DTCX", "DTP"],
        "dts": ["DTS", "DTSM", "DTSX", "DTS_COMP", "DT4S", "DTS1", "DTSS"],
        "rt": ["RT", "ILD", "RD", "AHT90", "LLD", "RES_DEEP", "HDRS", "AT90", "RES"],
        "nphi": ["NPHI", "TNPH", "NEUT", "NPOR", "HNPO", "CNL"],
        "cali": ["CALI", "CAL", "HCAL", "C1", "CALX"],
    }

    NULL_VALUES: List[float] = [-999.25, -999.0, -9999.0, -99999.0, 999.25, 9999.0]

    def __init__(self, alias_dict: Optional[Dict[str, List[str]]] = None):
        """Initialize loader with optional custom alias overrides."""
        self.aliases = alias_dict if alias_dict is not None else self.CANONICAL_ALIASES

    def load(
        self,
        file_path: Union[str, Path],
        clean_nulls: bool = True,
        depth_unit_target: Optional[str] = None,
    ) -> WellLogData:
        """Load a .las file and return standardized WellLogData.
        
        Args:
            file_path: Path to the .las file.
            clean_nulls: If True, replaces standard petroleum null values with np.nan.
            depth_unit_target: Optional target unit ('m' or 'ft') to enforce uniform depth.
            
        Returns:
            WellLogData dataclass with standardized columns.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"LAS file not found at: {path}")

        las = lasio.read(str(path))

        # Extract well metadata
        well_name = str(las.well.WELL.value if "WELL" in las.well else "UNKNOWN")
        field_name = str(las.well.FLD.value if "FLD" in las.well else "UNKNOWN")
        start_depth = float(las.well.STRT.value if "STRT" in las.well else 0.0)
        stop_depth = float(las.well.STOP.value if "STOP" in las.well else 0.0)
        step = float(las.well.STEP.value if "STEP" in las.well else 0.0)

        # Detect depth unit
        depth_unit = "m"
        if "STRT" in las.well and las.well.STRT.unit:
            u = str(las.well.STRT.unit).strip().lower()
            if u in ["f", "ft", "feet"]:
                depth_unit = "ft"
            elif u in ["m", "meter", "metre"]:
                depth_unit = "m"

        raw_df = las.df().reset_index()
        raw_curves = [str(c) for c in raw_df.columns]

        # Map to canonical curves
        std_df = pd.DataFrame()
        mapped_curves: Dict[str, str] = {}

        for canonical_name, aliases in self.aliases.items():
            matched_col = None
            for col in raw_df.columns:
                col_clean = str(col).strip().upper()
                if col_clean in [a.upper() for a in aliases]:
                    matched_col = col
                    break

            if matched_col is not None:
                std_df[canonical_name] = raw_df[matched_col].values.astype(float)
                mapped_curves[canonical_name] = str(matched_col)

        if "depth" not in std_df.columns:
            # First column is usually depth if not explicitly matched
            first_col = raw_df.columns[0]
            std_df["depth"] = raw_df[first_col].values.astype(float)
            mapped_curves["depth"] = str(first_col)

        # Clean null values
        if clean_nulls:
            for col in std_df.columns:
                for null_val in self.NULL_VALUES:
                    mask = np.isclose(std_df[col].values, null_val, atol=1e-2)
                    std_df.loc[mask, col] = np.nan

        # Unit conversion for depth if requested
        if depth_unit_target is not None and depth_unit_target.lower() != depth_unit:
            target = depth_unit_target.lower()
            if depth_unit == "m" and target == "ft":
                std_df["depth"] = std_df["depth"] * 3.28084
                start_depth *= 3.28084
                stop_depth *= 3.28084
                step *= 3.28084
                depth_unit = "ft"
            elif depth_unit == "ft" and target == "m":
                std_df["depth"] = std_df["depth"] / 3.28084
                start_depth /= 3.28084
                stop_depth /= 3.28084
                step /= 3.28084
                depth_unit = "m"

        return WellLogData(
            df=std_df,
            well_name=well_name,
            field_name=field_name,
            start_depth=start_depth,
            stop_depth=stop_depth,
            step=step,
            depth_unit=depth_unit,
            raw_curves=raw_curves,
            mapped_curves=mapped_curves,
        )
