# 1D Geomechanics & Hydraulic Fracturing Fracability Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Testing: pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)

An analytical 1D Mechanical Earth Model (1D MEM) and multi-criteria fracability sweet-spotting engine designed for horizontal stage and cluster placement in unconventional shale reservoirs (e.g., Vaca Muerta, Neuquén Basin).

---

## Engineering Problem & Overview

In horizontal multistage completions, hydraulic fracturing represents between **45% and 55% of total well CAPEX** ($5M to $7M USD per well). Conventional "blind" geometric completion designs place stages and perforation clusters at arbitrary fixed intervals (e.g., every 50 to 60 meters), frequently resulting in:

1. **Heel-Bias / Thief Cluster Domination:** If a single stage spans rocks with high minimum horizontal stress contrast ($\Delta \sigma_{h,min} > 250\text{ psi}$), the hydraulic fracture propagates preferentially into the lowest-stress cluster. Higher-stress perforations take little to no proppant slurry, stranding hydrocarbon-bearing pay.
2. **Ductility Screen-Out & Proppant Embedment:** Perforating ductile shale intervals (low Young's modulus, high Poisson's ratio) results in plastic dissipation of hydraulic energy, narrow fracture widths, and rapid post-fracture conductivity loss due to proppant embedment.
3. **Out-of-Zone Height Growth:** Uncontrolled vertical fracture migration through unmapped stress barriers into non-productive or water-bearing formations.

This tool ingests standard industry well logs (`.las`), solves the continuous poroelastic equations of motion, estimates pore pressure and in-situ tectonic stresses, evaluates formation fracability, and automatically optimizes stage and cluster positioning to achieve uniform treatment distribution.

---

## Core Capabilities

- **Petrophysics & Acoustic Synthesis:** Ingestion of CWLS `.las` files via `lasio` with robust mnemonic resolution. Implements the *Castagna Mudrock Line* for shear wave velocity ($V_s$) and slowness ($DTS$) synthesis, and non-linear shale volume ($V_{sh}$) modeling (Larionov Older/Mesozoic, Stieber, Clavier).
- **Dynamic-to-Static Elastic Moduli Calibration:** Vectorized calculation of dynamic acoustic moduli ($E_{dyn}, \nu_{dyn}, K, G$) and calibration to static reservoir stress conditions via empirical shale models (*Bradford et al.*, *Lacy*, *Morales*).
- **1D Mechanical Earth Model (1D MEM):**
  - Overburden vertical stress ($S_v$) via cumulative numerical integration of bulk density with shallow compaction trends.
  - Continuous pore pressure ($P_p$) estimation using Eaton's acoustic slowness method calibrated for overpressured unconventional reservoirs.
  - In-situ horizontal stress tensor ($\sigma_{h,min}, S_{Hmax}$) using Zoback's poroelastic horizontal strain model with regional tectonic strain components ($\epsilon_x, \epsilon_y$).
  - Hydraulic breakdown pressure ($P_b$) and fracture initiation gradients (*Haimson-Fairhurst*).
- **Composite Fracability Sweet-Spotting:** Multi-criteria integration combining *Rickman et al.* elastic brittleness index ($BI_{elastic}$), mineralogical clay proxy, and in-situ horizontal stress barrier penalties.
- **Engineered Completion Optimization:**
  - Automated stage partitioning targeting minimal intra-stage $\sigma_{h,min}$ variance.
  - Perforation cluster placement positioned at local fracability maxima subject to stress-shadow minimum spacing constraints ($d_{min} \ge 8\text{ m}$).
  - Quantitative design comparison: Engineered vs. Geometric spacing performance report.
- **Composite Well Log Visualization:** Multi-track continuous log plotting adhering to SPWLA / SPE publication standards (inverted depth axis, lithology shading, stress tensors, and cluster overlays).

---

## Mathematical Formulation

### 1. Dynamic Elastic Moduli from Acoustic Waves
From compressional velocity $V_p$ and shear velocity $V_s$ (m/s) and bulk density $\rho_b$ ($\text{g/cm}^3$):

$$\nu_{dyn} = \frac{V_p^2 - 2V_s^2}{2(V_p^2 - V_s^2)}$$

$$G_{dyn} = \rho_b \cdot V_s^2 \cdot 10^{-6} \quad [\text{GPa}]$$

$$E_{dyn} = 2 G_{dyn} (1 + \nu_{dyn}) \quad [\text{GPa}]$$

$$K_{dyn} = \rho_b \left(V_p^2 - \frac{4}{3}V_s^2\right) \cdot 10^{-6} \quad [\text{GPa}]$$

### 2. Static Calibration for Shales (Bradford et al., 1998)
$$E_{stat} = 0.415 \cdot E_{dyn} - 1.059 \quad [\text{GPa}]$$

$$\nu_{stat} = 0.70 \cdot \nu_{dyn}$$

### 3. Overburden Vertical Stress ($S_v$)
$$S_v(z) = g \int_0^z \rho_b(z') \, dz'$$

### 4. Eaton Pore Pressure ($P_p$)
$$P_p = S_v - (S_v - P_{p,norm}) \left( \frac{\Delta t_{c,norm}}{\Delta t_c} \right)^3$$

### 5. Poroelastic In-Situ Horizontal Stress Tensor (Zoback, 2007)
$$\sigma_{h,min} = \frac{\nu_{stat}}{1 - \nu_{stat}} (S_v - \alpha P_p) + \alpha P_p + \frac{E_{stat}}{1 - \nu_{stat}^2} \epsilon_x + \frac{\nu_{stat} E_{stat}}{1 - \nu_{stat}^2} \epsilon_y$$

$$S_{Hmax} = \frac{\nu_{stat}}{1 - \nu_{stat}} (S_v - \alpha P_p) + \alpha P_p + \frac{E_{stat}}{1 - \nu_{stat}^2} \epsilon_y + \frac{\nu_{stat} E_{stat}}{1 - \nu_{stat}^2} \epsilon_x$$

where $\alpha$ is Biot's poroelastic coefficient and $\epsilon_x, \epsilon_y$ are regional tectonic strains.

### 6. Rickman Elastic Brittleness Index (Rickman et al., 2008)
$$E_{norm} = \frac{E_{stat} - E_{min}}{E_{max} - E_{min}}, \quad \nu_{norm} = \frac{\nu_{max} - \nu_{stat}}{\nu_{max} - \nu_{min}}$$

$$BI_{elastic} = \frac{E_{norm} + \nu_{norm}}{2} \in [0.0, 1.0]$$

---

## Project Structure

```text
geomech-fracability-engine/
├── data/
│   ├── raw/                        # Original .las files (git ignored)
│   ├── processed/                  # Cleaned parquet profiles
│   └── synthetic/                  # Parametric benchmark wells
├── src/
│   ├── petrophysics/               # LAS reader, Castagna acoustic synthesis, Vshale
│   ├── elasticity/                 # Dynamic moduli, Bradford/Lacy static calibration
│   ├── geomechanics/               # Overburden Sv, Eaton Pp, Zoback Shmin/SHmax
│   ├── sweetspot/                  # Rickman BI, mineral proxy, Composite FI
│   ├── completion/                 # Stage partitioner, cluster placement, evaluator
│   ├── visualization/              # Publication-grade composite multi-track log plotter
│   ├── engine.py                   # High-level execution API facade
│   └── cli.py                      # Production command-line interface
├── tests/                          # Automated pytest suite (21 unit & integration tests)
├── pyproject.toml                  # Packaging and tool configurations
├── requirements.txt                # Fixed dependencies
└── README.md                       # Technical documentation
```

---

## Installation & Quickstart

### Prerequisites
Python 3.10+ and a clean virtual environment:

```bash
git clone https://github.com/meilantiago/geomech-fracability-engine.git
cd geomech-fracability-engine
pip install -r requirements.txt
```

### Run Demonstration via CLI
Run the engine on a synthetic Vaca Muerta unconventional horizontal well benchmark (2500m to 3000m MD):

```bash
python -m src.cli --demo --out-dir results
```

Output:
```text
======================================================================
 1D GEOMECHANICS & FRACABILITY COMPLETION AUDIT REPORT
======================================================================
 Well Name:                   VM-HORIZ-BENCH-01
 Logged Interval:             2500.0 m - 3000.0 m
 Total Lateral Stages:        8
 Total Perforation Clusters:  40
----------------------------------------------------------------------
 Geometric Mean FI:           0.536
 Engineered Mean FI:          0.660 (+23.1%)
 Geometric Stress Contrast:   901.9 psi
 Engineered Stress Contrast:  786.4 psi (-12.8%)
 Geometric Ductile Traps:     16 clusters
 Engineered Ductile Traps:    6 clusters (Eliminated!)
----------------------------------------------------------------------
 [+] Comprehensive Profiles:  results/geomech_profiles.parquet
 [+] Completion Schedule:     results/completion_schedule.csv
======================================================================
```

### Python API Usage

```python
from src.engine import GeomechEngine
from src.visualization import plot_composite_geomech_log

# Initialize engine with reservoir geomechanical parameters
engine = GeomechEngine(
    target_stage_length_m=60.0,
    clusters_per_stage=5,
    biot_alpha=0.85,
    tectonic_strain_x=0.00015,
    tectonic_strain_y=0.00045,
)

# Run full 1D geomechanical workflow from LAS file
result = engine.run_from_las("path/to/well_log.las")

# Export consolidated profiles and field schedule
result.to_parquet("results/geomech_profiles.parquet")
result.export_completion_schedule("results/completion_schedule.csv", design="engineered")

# Generate publication-grade composite well log plot
fig = plot_composite_geomech_log(result, output_png_path="results/composite_log.png")
```

---

## Automated Testing Suite

All mathematical derivations and physical boundary constraints are tested with `pytest`:

```bash
pytest tests/ -v
```

Test coverage includes:
- Exact analytical identity verification for elastic moduli ($K = \frac{E}{3(1-2\nu)}$).
- Exact analytical verification of horizontal stress anisotropy ($\Delta \sigma_h = \frac{E}{1+\nu}(\epsilon_y - \epsilon_x)$).
- Non-linear $V_{sh}$ models and [0.0, 1.0] boundary invariance.
- Eaton pore pressure hydrostatic vs. overpressured response.
- Cluster spacing constraint verification ($d_{min} \ge 8\text{ m}$).

---

## References

1. **Zoback, M.D.** (2007). *Reservoir Geomechanics*. Cambridge University Press.
2. **Rickman, R., et al.** (2008). *"A Practical Use of Shale Petrophysics for Stimulation Design Optimization: All Shale Plays Are Not Created Equal."* SPE 115258.
3. **Eaton, B.A.** (1975). *"The Equation for Geopressure Prediction from Well Logs."* SPE 5544.
4. **Castagna, J.P., et al.** (1985). *"Relationships between compressional-wave and shear-wave velocities in clastic silicate rocks."* Geophysics, 50(4), 571-581.
5. **Bradford, I.D.R., et al.** (1998). *"Dynamic-to-static moduli transitions in soft rocks and shales."* SPE 47277.
6. **Blanton, T.L., and Olson, J.E.** (1999). *"Stress Magnitudes from Logs: Effects of Tectonic Strains and Thermal Gradients."* SPE 56491.
7. **Haimson, B., and Fairhurst, C.** (1967). *"Initiation and Extension of Hydraulic Fractures in Rocks."* SPE Journal, 7(03), 310-318.

---

## Author

**Tiago Meilan**  
Licenciatura en Ciencias Físicas (Student / Advanced Candidate)  
Specialization: Applied Probability, Mathematical Modeling & Reservoir/Drilling Mechanics  
- LinkedIn: [linkedin.com/in/tiago-meilan](https://www.linkedin.com/in/tiago-meilan-6a723a260/)  
- GitHub: [github.com/meilantiago](https://github.com/meilantiago)  
- Email: meilantiago@gmail.com
