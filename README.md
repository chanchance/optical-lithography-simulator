# Optical Lithography Simulator

Python-based optical lithography simulation tool supporting GDS/OAS layout input, Fourier optics (Abbe/Hopkins), FDTD rigorous EM simulation, vector imaging, advanced resist models, and a full S-Litho–style GUI.

**Primary reference:** Pistor, T.V. (2001). *Electromagnetic Simulation and Modeling with Applications in Lithography*. PhD Dissertation, University of California, Berkeley. UCB/ERL M01/19.

---

## Features

| Feature | Reference (Pistor 2001) |
|---------|------------------------|
| FDTD engine (Yee grid, leapfrog, PML) | Ch. 1.2–2.5 |
| Fourier optics — Abbe source integration | Ch. 4.6 |
| Hopkins TCC / SOCS eigendecomposition | Ch. 4.6 |
| Vector imaging (Jones / Mk-matrix, TE/TM/X/Y/Circular) | Ch. 4.5 |
| Köhler illumination (Circular/Annular/Quadrupole/Quasar/Dipole/Freeform) | Ch. 4.2 |
| 37-term Fringe Zernike aberrations (Z1–Z37, OSA/ANSI) | Ch. 4.4 |
| Thin scalar mask (Binary, AttPSM, AltPSM) | Ch. 4.7.2.1 |
| Thick vector mask (EUV absorber) | Ch. 4.7.2.2 |
| RCWA near-field correction (1D, Fourier coefficients) | — |
| EUV multilayer mask (Mo/Si 40-pair, reflectance, flare, shot noise) | Ch. 5 |
| Resist models — Threshold, Dill A/B/C, Chemically Amplified (CA) | — |
| Thin-film stack TMM (Transfer Matrix Method) | — |
| CD / NILS / Contrast / DOF metrics | Ch. 1.3 |
| Advanced metrics — MEEF, Bossung curves, FEM, LER/LWR | — |
| Process window analysis (EL, DOF, PW area) | — |
| Config validation with field-level error messages | — |
| Results export — CSV, HDF5, PNG, text report | — |
| Optional CuPy GPU backend (drop-in NumPy fallback) | — |

---

## GUI — S-Litho Style

6-tab PySide6/PyQt5 application:

| Tab | Description |
|-----|-------------|
| **Layout** | GDS/OAS viewer with layer coloring, LOD rendering, async load |
| **Parameters** | Lithography knobs (λ, NA, σ, defocus, dose, aberrations, resist) |
| **Simulation** | Run/Stop, mode selector, progress bar with ETA, log |
| **Results** | 6-panel GridSpec: aerial image · mask · overlay · wavefront error · cross-section · CD-vs-defocus history |
| **Analysis** | Bossung curves · Focus-Exposure Matrix · Process Window ellipse · MEEF |
| **Stack** | Film stack visual editor (TMM layers) |

### Interactive Gauge Tool
Click two points on the aerial image to draw a cross-section gauge. Multiple gauges overlay on the same cross-section plot with CD annotation. Lock H/V constrains gauge direction.

---

## Theory Background

### FDTD Engine — Chapter 1.2 (TEMPEST)

Yee staggered grid with leapfrog time integration:

```
α = (2ε - σΔt) / (2ε + σΔt)
β = (Δt/Δx) · 2 / (2ε + σΔt)
γ = Δt / (μΔx)
```

PML absorbing boundaries use polynomial-graded conductivity (Berenger): σ ≈ 0 at the inner face, σ_max at the outer boundary. Convergence checked via periodic error metric (Ch. 1.2.2).

Implementation: `core/fdtd_engine.py`

### Fourier Optics / Aerial Image — Chapter 4.4–4.6

**Abbe integration:**
```
I(x,y) = Σ_s w_s · |IFFT[ M(f) · P(f) · exp(jφ_defocus) ]|²
```

**Hopkins TCC / SOCS (optional, `use_hopkins: true`):**
```
TCC(k1,k2) = Σ_s J(s) · P(k1-s) · P*(k2-s)
I(x,y)     = Σ_i λ_i |IFFT[M(k) · φ_i(k)]|²
```
TCC is mask-independent — computed once per source/pupil config. SOCS keeps the top N eigenkernels for fast per-mask evaluation.

**Defocus phase:**
```
φ_defocus = π · λ · defocus · (fx² + fy²)
```

Implementation: `core/fourier_optics.py`, `core/hopkins.py`

### Vector Imaging — Chapter 4.5

Jones matrix formulation with TE/TM decomposition at each source point. Polarization modes: X-linear, Y-linear, TE, TM, Circular-L, Circular-R. Intensity = sum of squared field components.

Implementation: `core/vector_imaging.py`

### Source Model — Chapter 4.2

| Type | Description |
|------|-------------|
| Circular | Uniform disk up to σ·NA |
| Annular | Ring between σ_inner and σ_outer |
| Quadrupole | Four poles at 0°/90°/180°/270°, disc sampling |
| Quasar | Four poles at 45°/135°/225°/315°, disc sampling |
| Dipole | Two annular-arc spots, x or y orientation |
| Freeform | 2D pupil map from expression/file/array |

Implementation: `core/source_model.py`

### Zernike Aberrations — Chapter 4.4

37-term Fringe Zernike (OSA/ANSI) via radial polynomial + normalization. Wavefront phase map displayed in the Results panel (RMS in waves).

Implementation: `core/aberrations.py`

### Photomask Models — Chapter 4.7

| Model | Description |
|-------|-------------|
| `ThinScalarMask` | Binary / AttPSM / AltPSM; Kirchhoff approximation |
| `ThickVectorMask` | Absorber thickness + phase correction; constant scattering |
| RCWA near-field | `RCWAEngine.apply_to_mask_grid()` row-by-row FFT-based correction |

AltPSM alternates phase per connected clear region (scipy label-based).

Implementation: `core/mask_model.py`, `core/rcwa.py`

### Resist Models

| Model | Description |
|-------|-------------|
| Threshold | Binary threshold on normalized intensity |
| Dill A/B/C | Exponential bleaching: M = exp(-A·C·dose·I), PEB diffusion |
| CA (Chemically Amplified) | Acid generation → PEB → logistic deprotection → develop |

Implementation: `core/resist_model.py`

### EUV Lithography — Chapter 5

- `EUVMultilayerMask`: Mo/Si 40-pair reflective stack, reflectance modulation
- `EUVFlare`: uniform background scattered-light fraction added to aerial image
- Shot noise: Poisson sampling at photons/nm² dose level

Implementation: `core/euv_mask.py`

### Thin-Film Stack — Transfer Matrix Method

TMM for multi-layer film stacks (ARC, resist, substrate). Computes reflectance, transmittance, and standing-wave intensity profile.

Implementation: `core/film_stack.py`

### Aerial Image Metrics

| Metric | Formula |
|--------|---------|
| CD | Feature width at intensity threshold |
| NILS | `(w/I) · dI/dx` — Normalized Image Log Slope |
| Contrast | `(I_max - I_min) / (I_max + I_min)` |
| DOF estimate | `0.5 · λ/NA² · min(NILS/2, 2)` |
| MEEF | `(ΔCD_wafer / ΔCD_mask) · M` (4× magnification) |
| EL (%) | Dose range keeping CD within ±tolerance |
| LER 3σ | Edge roughness from 1D PSD |

Implementation: `analysis/aerial_image_analysis.py`, `analysis/advanced_metrics.py`

---

## Project Structure

```
simulation/
├── core/
│   ├── fdtd_engine.py        # FDTD / TEMPEST (Ch. 1.2, 2.5)
│   ├── fourier_optics.py     # Abbe integration, Zernike (Ch. 4.4–4.6)
│   ├── hopkins.py            # Hopkins TCC / SOCS
│   ├── source_model.py       # Illumination sources (Ch. 4.2)
│   ├── imaging_system.py     # Full imaging chain (Ch. 4.1.1)
│   ├── mask_model.py         # Thin/thick mask models (Ch. 4.7)
│   ├── rcwa.py               # RCWA near-field correction
│   ├── vector_imaging.py     # Jones matrix vector imaging (Ch. 4.5)
│   ├── resist_model.py       # Threshold / Dill / CA resist
│   ├── film_stack.py         # TMM thin-film stack
│   ├── euv_mask.py           # EUV multilayer mask + flare + shot noise
│   ├── aberrations.py        # 37-term Fringe Zernike (Z1–Z37)
│   └── gpu_backend.py        # Optional CuPy GPU backend
├── fileio/
│   ├── layout_io.py          # GDS/OAS read via gdstk
│   ├── parameter_io.py       # YAML config load/save/merge
│   ├── config_validator.py   # Field-level validation with severity levels
│   ├── results_exporter.py   # CSV / HDF5 / PNG / text report export
│   └── pdf_parser.py         # Thesis PDF parsing (pdfplumber)
├── analysis/
│   ├── aerial_image_analysis.py  # CD, NILS, Contrast, DOF
│   ├── signal_analysis.py        # Profile extraction, EL from 1D profile
│   ├── process_window.py         # EL/DOF/PW area from pass-fail grid
│   └── advanced_metrics.py       # MEEF, Bossung, FEM, LER/LWR
├── visualization/
│   ├── layout_viewer.py          # GDS polygon viewer (matplotlib)
│   ├── aerial_image_viewer.py    # Aerial image + overlay + cross-section
│   └── field_viewer.py           # EM near-field animation
├── pipeline/
│   ├── simulation_pipeline.py    # End-to-end pipeline (layout→metrics)
│   ├── parameter_sweep.py        # 1D/2D sweep + Bossung convenience
│   └── batch_runner.py           # Parameter sweep batch execution
├── gui/
│   ├── main_window.py            # 6-tab main window
│   ├── theme.py                  # Dark theme constants + QSS
│   ├── gauge_manager.py          # Interactive gauge state + profile extraction
│   ├── qt_compat.py              # PySide6/PyQt5 compatibility shim
│   ├── panels/
│   │   ├── layout_panel.py       # Async GDS viewer + LOD
│   │   ├── parameter_panel.py    # Full lithography parameter controls
│   │   ├── simulation_panel.py   # Run/Stop/Reset + progress + log
│   │   ├── results_panel.py      # 6-panel results + gauge + export
│   │   └── analysis_panel.py     # Bossung / FEM / Process Window
│   └── dialogs/
│       ├── source_dialog.py      # k-space illumination preview + freeform editor
│       ├── stack_dialog.py       # Film stack visual editor
│       └── mask_dialog.py        # Mask type selector
├── config/
│   └── default_config.yaml       # ArF 193nm / EUV 13.5nm presets
├── tests/
│   └── fixtures/
│       ├── make_test_gds.py
│       └── make_test_oas.py
└── setup.py
```

---

## Quick Start

### Prerequisites (Python 3.8+)

```bash
pip install numpy scipy matplotlib pyyaml pdfplumber
pip install gdstk          # GDS/OAS layout I/O
pip install PySide6        # GUI
pip install h5py           # HDF5 export (optional)
pip install cupy-cuda12x   # GPU acceleration (optional)
```

### Run GUI

```bash
cd simulation
python -m gui.main_window
```

### CLI Simulation

```bash
python -m pipeline.simulation_pipeline --gds layout.gds \
    --wavelength 193 --na 0.93 --sigma 0.85 --defocus 0
```

### Python API

```python
from pipeline.simulation_pipeline import SimulationPipeline

config = {
    "lithography": {
        "wavelength_nm": 193.0,
        "NA": 0.93,
        "defocus_nm": 0.0,
        "dose_factor": 1.0,
        "illumination": {"type": "annular", "sigma_outer": 0.85, "sigma_inner": 0.55},
        "aberrations": {"zernike": [0]*37},   # Z1–Z37 in waves
    },
    "simulation": {"grid_size": 256, "domain_size_nm": 2000.0},
    "resist": {"model": "threshold", "threshold": 0.30},
}

result = SimulationPipeline().run(config, "layout.gds")
print(f"CD={result.cd_nm:.1f} nm  NILS={result.nils:.3f}  DOF={result.dof_nm:.0f} nm")
```

### Bossung Sweep

```python
from pipeline.parameter_sweep import ParameterSweep

sweep = ParameterSweep()
data = sweep.bossung_sweep(config, "layout.gds",
                           focus_range_nm=400, n_focus=17,
                           dose_factors=[0.9, 0.95, 1.0, 1.05, 1.1])
# data['focus_nm'], data['cd_by_dose']
```

### MEEF

```python
from analysis.advanced_metrics import MEEF
from pipeline.simulation_pipeline import SimulationPipeline

meef_calc = MEEF.from_pipeline(SimulationPipeline(), nominal_cd_nm=100.0)
result = meef_calc.compute(config, mask_delta_nm=5.0)
print(f"MEEF = {result.meef:.2f}")
```

---

## Simulation Modes

| Mode | Speed | Accuracy | Use case |
|------|-------|----------|----------|
| Fourier Optics (Abbe) | ~1 s | Scalar, thin mask | CD/NILS/DOF optimization |
| Hopkins SOCS | ~1 s | Scalar, mask-independent TCC | Large source grids |
| Vector Imaging | ~2–5 s | Polarized, vector | High-NA, polarization sensitivity |
| FDTD (TEMPEST) | Minutes | Rigorous EM | EUV thick mask, PSM defects |

---

## Supported Configurations

| Technology | λ (nm) | NA | Illumination |
|-----------|--------|-----|-------------|
| ArF Immersion | 193 | 0.93 | Annular σ 0.55/0.85 |
| ArF Dry | 193 | 0.75 | Annular σ 0.50/0.80 |
| KrF | 248 | 0.68 | Annular σ 0.45/0.75 |
| EUV | 13.5 | 0.33 | Annular σ 0.55/0.85 |

---

## Configuration Schema

Key config fields (full schema in `config/default_config.yaml`):

```yaml
lithography:
  wavelength_nm: 193.0
  NA: 0.93
  defocus_nm: 0.0
  dose_factor: 1.0          # Scale aerial image intensity
  use_hopkins: false         # Hopkins TCC/SOCS mode
  n_kernels: 10              # SOCS eigenkernels to keep
  use_vector: false          # Vector imaging engine
  polarization: unpolarized  # x, y, te, tm, circular_l, circular_r
  illumination:
    type: annular            # circular, annular, quadrupole, quasar, dipole, freeform
    sigma_outer: 0.85
    sigma_inner: 0.55
  aberrations:
    zernike: [0, 0, 0, ...]  # 37 coefficients Z1–Z37 in waves
simulation:
  grid_size: 256
  domain_size_nm: 2000.0
  mode: fourier_optics       # fourier_optics | fdtd
resist:
  model: threshold           # threshold | dill | ca
  threshold: 0.30
  dose: 1.0
  A: 0.8                     # Dill bleachable absorption
  C: 1.0                     # Dill exposure rate
  peb_sigma_nm: 30.0
analysis:
  cd_threshold: 0.30
```

---

## Reference

> Pistor, T.V. (2001). *Electromagnetic Simulation and Modeling with Applications in Lithography*. PhD Dissertation, EECS Department, University of California, Berkeley. Memorandum No. UCB/ERL M01/19. Advisor: Prof. Andrew R. Neureuther.
