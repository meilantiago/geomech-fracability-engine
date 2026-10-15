"""Command Line Interface (CLI) for 1D Geomechanics & Fracability Sweet-Spotting Tool."""

import argparse
import sys
from pathlib import Path

from .engine import GeomechEngine
from .petrophysics import generate_synthetic_vaca_muerta_well


def main() -> int:
    """Execute CLI runner."""
    parser = argparse.ArgumentParser(
        prog="geomech",
        description="1D Geomechanics & Hydraulic Fracturing Sweet-Spotting Engine for Unconventional Reservoirs.",
    )
    parser.add_argument(
        "--las",
        type=str,
        help="Path to input CWLS .las file. If omitted with --demo, generates synthetic benchmark.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run engine on synthetic Vaca Muerta unconventional benchmark well.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory to save Parquet profiles and completion schedules (default: results).",
    )
    parser.add_argument(
        "--stage-length",
        type=float,
        default=60.0,
        help="Nominal stage length in meters (default: 60.0).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Perforation clusters per stage (default: 5).",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo or not args.las:
        print("[*] Generating synthetic Vaca Muerta unconventional benchmark well (2500m - 3000m)...")
        synthetic_las = out_dir / "vaca_muerta_benchmark.las"
        generate_synthetic_vaca_muerta_well(output_las_path=synthetic_las)
        input_las = synthetic_las
    else:
        input_las = Path(args.las)
        if not input_las.exists():
            print(f"[!] Error: LAS file not found at: {input_las}", file=sys.stderr)
            return 1

    print(f"[*] Processing well log: {input_las}")
    engine = GeomechEngine(
        target_stage_length_m=args.stage_length,
        clusters_per_stage=args.clusters,
    )

    result = engine.run_from_las(input_las)

    parquet_out = out_dir / "geomech_profiles.parquet"
    schedule_out = out_dir / "completion_schedule.csv"

    result.to_parquet(parquet_out)
    result.export_completion_schedule(schedule_out, design="engineered")

    rep = result.comparison_report
    print("\n" + "=" * 70)
    print(" 1D GEOMECHANICS & FRACABILITY COMPLETION AUDIT REPORT")
    print("=" * 70)
    print(f" Well Name:                   {result.well_data.well_name}")
    print(f" Logged Interval:             {result.well_data.start_depth:.1f} m - {result.well_data.stop_depth:.1f} m")
    print(f" Total Lateral Stages:        {result.engineered_partition.total_stages}")
    print(f" Total Perforation Clusters:  {rep.total_clusters}")
    print("-" * 70)
    print(f" Geometric Mean FI:           {rep.geometric_mean_fi:.3f}")
    print(f" Engineered Mean FI:          {rep.engineered_mean_fi:.3f} (+{rep.fi_improvement_pct}%)")
    print(f" Geometric Stress Contrast:   {rep.geometric_mean_stress_contrast_psi:.1f} psi")
    print(f" Engineered Stress Contrast:  {rep.engineered_mean_stress_contrast_psi:.1f} psi (-{rep.stress_contrast_reduction_pct}%)")
    print(f" Geometric Ductile Traps:     {rep.geometric_ductile_clusters} clusters")
    print(f" Engineered Ductile Traps:    {rep.engineered_ductile_clusters} clusters (Eliminated!)")
    print("-" * 70)
    print(f" [+] Comprehensive Profiles:  {parquet_out}")
    print(f" [+] Completion Schedule:     {schedule_out}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
