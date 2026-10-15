"""Unified Geomechanics and Fracability Execution Engine.

Connects petrophysics ingestion, poroelasticity, 1D MEM stresses, sweet-spotting,
and completion optimization into a single production-grade Python API.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .completion import (
    CompletionComparisonReport,
    StageCompletionPlan,
    StagePartitionResult,
    evaluate_completion_designs,
    partition_engineered_stages,
    partition_geometric_stages,
    place_engineered_clusters,
    place_geometric_clusters,
)
from .elasticity import (
    CalibrationMethod,
    DynamicElasticModuli,
    StaticElasticModuli,
    calibrate_static_moduli,
    compute_dynamic_moduli,
)
from .geomechanics import (
    HydraulicBreakdown,
    OverburdenStress,
    PorePressureProfile,
    StressTensor1D,
    compute_breakdown_pressure,
    compute_eaton_pore_pressure,
    compute_in_situ_stresses,
    compute_overburden_stress,
)
from .petrophysics import (
    LASLoader,
    ShaleVolumeModel,
    WellLogData,
    calculate_vshale,
    synthesize_shear_slowness,
)
from .sweetspot import (
    ElasticBrittleness,
    FracabilityProfile,
    compute_composite_fracability_index,
    compute_elastic_brittleness,
    compute_mineral_brittleness_proxy,
)


@dataclass
class GeomechResult:
    """Consolidated results of the 1D geomechanical and completion analysis.
    
    Attributes:
        well_data: Standardized WellLogData from ingestion.
        vshale: Volume fraction of shale.
        dynamic_moduli: Dynamic acoustic elastic properties.
        static_moduli: Calibrated static rock mechanics properties.
        overburden: Overburden vertical stress profile.
        pore_pressure: Eaton pore pressure and gradient profiles.
        stress_tensor: In-situ stress tensor (Shmin, SHmax, anisotropy).
        breakdown: Hydraulic fracture initiation and breakdown pressures.
        brittleness: Elastic brittleness indices.
        fracability: Composite Fracability Index (FI) and sweet-spot flags.
        geometric_partition: Geometric stage boundaries.
        geometric_plans: Stage completion plans for geometric spacing.
        engineered_partition: Engineered stage boundaries.
        engineered_plans: Stage completion plans for engineered sweet-spotting.
        comparison_report: Quantitative ROI and performance audit.
    """
    well_data: WellLogData
    vshale: np.ndarray
    dynamic_moduli: DynamicElasticModuli
    static_moduli: StaticElasticModuli
    overburden: OverburdenStress
    pore_pressure: PorePressureProfile
    stress_tensor: StressTensor1D
    breakdown: HydraulicBreakdown
    brittleness: ElasticBrittleness
    fracability: FracabilityProfile
    geometric_partition: StagePartitionResult
    geometric_plans: List[StageCompletionPlan]
    engineered_partition: StagePartitionResult
    engineered_plans: List[StageCompletionPlan]
    comparison_report: CompletionComparisonReport

    def to_dataframe(self) -> pd.DataFrame:
        """Export all continuous 1D profiles into a single consolidated pandas DataFrame."""
        df = self.well_data.df.copy()
        df["vshale"] = np.round(self.vshale, 3)
        df["vp_m_s"] = np.round(self.dynamic_moduli.vp_m_s, 1)
        df["vs_m_s"] = np.round(self.dynamic_moduli.vs_m_s, 1)
        df["vp_vs"] = np.round(self.dynamic_moduli.vp_vs_ratio, 2)
        df["young_dyn_gpa"] = np.round(self.dynamic_moduli.young_gpa, 2)
        df["poisson_dyn"] = np.round(self.dynamic_moduli.poisson_ratio, 3)
        df["young_stat_gpa"] = np.round(self.static_moduli.young_gpa, 2)
        df["poisson_stat"] = np.round(self.static_moduli.poisson_ratio, 3)
        df["sv_psi"] = np.round(self.overburden.sv_psi, 1)
        df["sv_grad_psi_ft"] = np.round(self.overburden.gradient_psi_ft, 3)
        df["pp_psi"] = np.round(self.pore_pressure.pp_psi, 1)
        df["pp_grad_psi_ft"] = np.round(self.pore_pressure.gradient_psi_ft, 3)
        df["shmin_psi"] = np.round(self.stress_tensor.shmin_psi, 1)
        df["shmax_psi"] = np.round(self.stress_tensor.shmax_psi, 1)
        df["shmin_grad_psi_ft"] = np.round(self.stress_tensor.shmin_gradient_psi_ft, 3)
        df["stress_anisotropy_psi"] = np.round(self.stress_tensor.stress_anisotropy_psi, 1)
        df["stress_regime"] = self.stress_tensor.regime
        df["breakdown_pb_psi"] = np.round(self.breakdown.breakdown_pressure_psi, 1)
        df["bi_elastic"] = np.round(self.brittleness.bi_elastic, 3)
        df["fracability_index"] = np.round(self.fracability.fracability_index, 3)
        df["fracability_tier"] = self.fracability.tier
        df["is_sweetspot"] = self.fracability.is_sweetspot
        return df

    def to_parquet(self, output_path: Union[str, Path]) -> None:
        """Save the comprehensive consolidated profile into an optimized Parquet file."""
        df = self.to_dataframe()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)

    def export_completion_schedule(self, output_csv_path: Union[str, Path], design: str = "engineered") -> pd.DataFrame:
        """Export the perforation cluster schedule to CSV for field operations.
        
        Args:
            output_csv_path: Target CSV file path.
            design: 'engineered' or 'geometric'.
            
        Returns:
            DataFrame containing stage-by-stage perforation schedule.
        """
        plans = self.engineered_plans if design.lower() == "engineered" else self.geometric_plans
        rows = []
        for p in plans:
            st = p.stage_info
            for c in p.clusters:
                rows.append({
                    "stage_id": st.stage_id,
                    "stage_top_md": st.start_md,
                    "stage_bottom_md": st.end_md,
                    "stage_length_m": st.length_m,
                    "cluster_id": c.cluster_id,
                    "cluster_md_m": c.md_m,
                    "cluster_shmin_psi": c.shmin_psi,
                    "cluster_fi": c.fracability_index,
                    "is_sweetspot": c.is_sweetspot,
                })
        schedule_df = pd.DataFrame(rows)
        out = Path(output_csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        schedule_df.to_csv(out, index=False)
        return schedule_df


class GeomechEngine:
    """End-to-end 1D Geomechanics & Hydraulic Fracturing Sweet-Spotting Engine."""

    def __init__(
        self,
        shale_model: Union[str, ShaleVolumeModel] = ShaleVolumeModel.LARIONOV_OLDER,
        static_calibration_method: Union[str, CalibrationMethod] = CalibrationMethod.BRADFORD_SHALE,
        biot_alpha: float = 0.85,
        tectonic_strain_x: float = 0.00015,
        tectonic_strain_y: float = 0.00045,
        eaton_exponent: float = 3.0,
        target_stage_length_m: float = 60.0,
        clusters_per_stage: int = 5,
    ):
        """Initialize engine configuration.
        
        Args:
            shale_model: Vshale mathematical model.
            static_calibration_method: Dynamic-to-static moduli conversion model.
            biot_alpha: Biot's poroelastic coefficient.
            tectonic_strain_x: Tectonic strain in min horizontal stress direction.
            tectonic_strain_y: Tectonic strain in max horizontal stress direction.
            eaton_exponent: Pore pressure Eaton exponent.
            target_stage_length_m: Nominal stage length in meters.
            clusters_per_stage: Number of perforation clusters per stage.
        """
        self.shale_model = shale_model
        self.static_method = static_calibration_method
        self.biot_alpha = biot_alpha
        self.tectonic_strain_x = tectonic_strain_x
        self.tectonic_strain_y = tectonic_strain_y
        self.eaton_exponent = eaton_exponent
        self.target_stage_length_m = target_stage_length_m
        self.clusters_per_stage = clusters_per_stage
        self.loader = LASLoader()

    def run_from_las(self, las_file_path: Union[str, Path]) -> GeomechResult:
        """Execute full geomechanical workflow from a .las file."""
        well_data = self.loader.load(las_file_path)
        return self.run(well_data)

    def run(self, well_data: WellLogData) -> GeomechResult:
        """Execute full geomechanical and completion sweet-spotting workflow.
        
        Args:
            well_data: Standardized WellLogData container.
            
        Returns:
            GeomechResult containing full physics and completion deliverables.
        """
        df = well_data.df
        depth_m = df["depth"].values

        # 1. Petrophysics: Vshale
        if "gr" in df.columns:
            vshale = calculate_vshale(df["gr"].values, model=self.shale_model)
        else:
            vshale = np.full(len(depth_m), 0.3)

        # 2. Acoustic synthesis (DTS) if absent
        dtc = df["dtc"].values
        dts_existing = df["dts"].values if "dts" in df.columns else None
        dts = synthesize_shear_slowness(dtc, dts_existing=dts_existing)

        rhob = df["rhob"].values

        # 3. Dynamic Moduli
        dyn_moduli = compute_dynamic_moduli(rhob_g_cm3=rhob, dtc_us_ft=dtc, dts_us_ft=dts)

        # 4. Static Calibration
        stat_moduli = calibrate_static_moduli(
            young_dyn_gpa=dyn_moduli.young_gpa,
            poisson_dyn=dyn_moduli.poisson_ratio,
            method=self.static_method,
        )

        # 5. Overburden Stress Sv
        overburden = compute_overburden_stress(depth_m=depth_m, rhob_g_cm3=rhob)

        # 6. Pore Pressure Pp (Eaton)
        pore_pressure = compute_eaton_pore_pressure(
            depth_m=depth_m,
            dtc_us_ft=dtc,
            sv_psi=overburden.sv_psi,
            eaton_exponent=self.eaton_exponent,
        )

        # 7. In-situ Stress Tensor (Zoback poroelastic strain model)
        stress_tensor = compute_in_situ_stresses(
            depth_m=depth_m,
            sv_psi=overburden.sv_psi,
            pp_psi=pore_pressure.pp_psi,
            young_stat_gpa=stat_moduli.young_gpa,
            poisson_stat=stat_moduli.poisson_ratio,
            biot_alpha=self.biot_alpha,
            tectonic_strain_x=self.tectonic_strain_x,
            tectonic_strain_y=self.tectonic_strain_y,
        )

        # 8. Hydraulic Breakdown Pressure Pb
        breakdown = compute_breakdown_pressure(
            depth_m=depth_m,
            shmin_psi=stress_tensor.shmin_psi,
            shmax_psi=stress_tensor.shmax_psi,
            pp_psi=pore_pressure.pp_psi,
        )

        # 9. Brittleness Indices
        brittleness = compute_elastic_brittleness(
            young_modulus=stat_moduli.young_gpa,
            poisson_ratio=stat_moduli.poisson_ratio,
        )
        mineral_proxy = compute_mineral_brittleness_proxy(vshale=vshale)

        # 10. Composite Fracability Index FI
        fracability = compute_composite_fracability_index(
            bi_elastic=brittleness.bi_elastic,
            bi_mineral=mineral_proxy,
            shmin_psi=stress_tensor.shmin_psi,
        )

        # 11. Completion Optimization: Geometric vs Engineered
        geom_partition = partition_geometric_stages(
            depth_m=depth_m,
            shmin_psi=stress_tensor.shmin_psi,
            fracability_index=fracability.fracability_index,
            target_length_m=self.target_stage_length_m,
        )
        geom_plans = [
            place_geometric_clusters(
                st, depth_m, stress_tensor.shmin_psi, fracability.fracability_index, self.clusters_per_stage
            )
            for st in geom_partition.stages
        ]

        eng_partition = partition_engineered_stages(
            depth_m=depth_m,
            shmin_psi=stress_tensor.shmin_psi,
            fracability_index=fracability.fracability_index,
            target_length_m=self.target_stage_length_m,
        )
        eng_plans = [
            place_engineered_clusters(
                st, depth_m, stress_tensor.shmin_psi, fracability.fracability_index, self.clusters_per_stage
            )
            for st in eng_partition.stages
        ]

        # 12. Comparison Report
        comparison = evaluate_completion_designs(geom_plans, eng_plans)

        return GeomechResult(
            well_data=well_data,
            vshale=vshale,
            dynamic_moduli=dyn_moduli,
            static_moduli=stat_moduli,
            overburden=overburden,
            pore_pressure=pore_pressure,
            stress_tensor=stress_tensor,
            breakdown=breakdown,
            brittleness=brittleness,
            fracability=fracability,
            geometric_partition=geom_partition,
            geometric_plans=geom_plans,
            engineered_partition=eng_partition,
            engineered_plans=eng_plans,
            comparison_report=comparison,
        )
