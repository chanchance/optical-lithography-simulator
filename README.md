# Optical Lithography Simulator

Python-based optical lithography simulation tool supporting GDS/OAS layout input, Fourier optics (Abbe method), and FDTD-based rigorous EM simulation.

**Primary reference:** Pistor, T.V. (2001). *Electromagnetic Simulation and Modeling with Applications in Lithography*. PhD Dissertation, University of California, Berkeley. UCB/ERL M01/19.

---

## Features

| Feature | Reference (Pistor 2001) |
|---------|------------------------|
| FDTD engine (Yee grid, leapfrog) | Ch. 1.2.1 — The FDTD method and the Yee equations |
| PML absorbing boundary | Ch. 1.4.2.1, Ch. 2.5 — Absorbing Boundary Conditions |
| Convergence checking | Ch. 1.2.2 — Convergence Checking |
| Plane wave domain excitation | Ch. 1.2.3 — Domain Excitation |
| Imaging system model | Ch. 4.1.1 — The Imaging System Model (Figure 4-1) |
| Köhler illumination source | Ch. 4.2 — Source and Illumination Optic |
| Source discretization (Circular/Annular/Quadrupole/Quasar) | Ch. 4.2.1 — Discretization of the Source |
| Projection optic & pupil function | Ch. 4.4 — Projection Optic |
| Zernike aberration coefficients | Ch. 4.4 — Projection Optic (W020, W040, W111, W131, W222) |
| Aerial image (Abbe/Hopkins integration) | Ch. 4.6 — The Imaging Equations |
| Thin scalar mask (Binary, AttPSM, AltPSM) | Ch. 4.7.2.1 — Thin, scalar, constant scattering coefficient model |
| Thick vector mask (EUV absorber) | Ch. 4.7.2.2 — Thick, vector, constant scattering coefficient model |
| EUV lithography (13.5 nm) | Ch. 5 — Simulation of Extreme Ultraviolet Lithography |
| Phase shift mask (AltPSM) | Ch. 6 — Phase Shift Mask Inspection |
| CD / NILS / DOF / Contrast metrics | Ch. 1.3 — Optical Imaging Models and Aerial Image Calculation |

---

## Theory Background

### FDTD Engine — Chapter 1.2 (TEMPEST)

Implements the Yee staggered grid algorithm (Ch. 1.2.1) with leapfrog time integration:

```
α = (2ε - σΔt) / (2ε + σΔt)
β = (Δt/Δx) · 2 / (2ε + σΔt)
γ = Δt / (μΔx)
```

Electric (E) and magnetic (H) field components are offset by half a grid cell spatially and half a time step temporally. Convergence is checked via the periodic error metric (Ch. 1.2.2):

```
pterr = |Eamp(cT) - Eamp((c-1)T)| / |Eamp(cT) + Eamp((c-1)T)|
```

Implementation: `core/fdtd_engine.py` — `YeeGrid`, `FDTDSimulator`

### PML Boundary Conditions — Chapter 2.5

Berenger's Perfectly Matched Layer with polynomial-graded conductivity profile. Prevents spurious reflections at the simulation domain boundary.

Implementation: `core/fdtd_engine.py` — `PMLLayer`

### Imaging System Model — Chapter 4.1.1

Follows the full system chain (Figure 4-1, p. 47):

```
Source → Illumination Optic → Photomask → Projection Optic → Film Stack → Aerial Image
```

Implementation: `core/imaging_system.py` — `ImagingSystem`

### Source Model — Chapter 4.2

Implements Köhler illumination with discretized source points in k-space (Ch. 4.2.1). Source density follows:

```
n_2D = π/4 · (Ns · 2σ·NA_w/λ)²
```

Supported illumination schemes:
- **Circular** — uniform disk up to σ·NA
- **Annular** — ring between σ_inner and σ_outer
- **Quadrupole** — four off-axis poles at 0°/90°/180°/270°
- **Quasar** — four poles at 45°/135°/225°/315°

Implementation: `core/source_model.py`

### Fourier Optics / Aerial Image — Chapter 4.4–4.6

The aerial image is computed via Abbe source integration (Ch. 4.6):

```
I(x,y) = Σ_s w_s · |IFFT[ M(f) · P(f) · exp(j·φ_defocus) ]|²
```

where:
- `M(f)` — mask diffraction spectrum
- `P(f)` — coherent transfer function (pupil), NA-limited circular aperture
- `φ_defocus = π·λ·defocus·(fx²+fy²)` — defocus phase (Ch. 4.4)
- Zernike aberrations W020, W040, W111, W131, W222 included

Implementation: `core/fourier_optics.py` — `FourierOpticsEngine`, `ZernikeAberrations`

### Photomask Models — Chapter 4.7

Three mask complexity levels following Ch. 4.7.2:

| Model | Section | Description |
|-------|---------|-------------|
| `ThinScalarMask` | §4.7.2.1 | Binary/AttPSM/AltPSM; scalar transmission; Kirchhoff approximation |
| `ThickVectorMask` | §4.7.2.2 | EUV absorber thickness correction; constant scattering coefficients |

The thin scalar model uses Kirchhoff boundary conditions: transmission = 1 (open), 0 (opaque), or complex (PSM).

Implementation: `core/mask_model.py`

### EUV Lithography — Chapter 5

EUV configuration (λ = 13.5 nm, NA = 0.33) corresponds to Ch. 5.2 mask feature simulations. The thick vector mask captures absorber depth-of-focus degradation due to off-axis imaging and CD dependence on absorber thickness (Ch. 5.2.1.4).

### Aerial Image Metrics — Chapter 1.3

Following the standard lithography merit figures:

| Metric | Formula |
|--------|---------|
| CD | Feature width at intensity threshold |
| NILS | `(w/I) · dI/dx` — Normalized Image Log Slope |
| Contrast | `(I_max - I_min) / (I_max + I_min)` |
| DOF | Defocus range keeping CD within ±10% |

Implementation: `analysis/aerial_image_analysis.py`

---

## Project Structure

```
simulation/
├── core/
│   ├── fdtd_engine.py        # FDTD / TEMPEST (Ch. 1.2, 2.5)
│   ├── fourier_optics.py     # Abbe integration, Zernike (Ch. 4.4–4.6)
│   ├── source_model.py       # Illumination source (Ch. 4.2)
│   ├── imaging_system.py     # Full imaging chain (Ch. 4.1.1)
│   └── mask_model.py         # Thin/thick mask models (Ch. 4.7)
├── fileio/
│   ├── layout_io.py          # GDS/OAS read via gdstk
│   ├── parameter_io.py       # YAML config load/save
│   └── pdf_parser.py         # Thesis PDF parsing (pdfplumber)
├── analysis/
│   ├── aerial_image_analysis.py  # CD, NILS, Contrast, DOF
│   ├── signal_analysis.py        # Profile extraction, peak finding
│   └── process_window.py         # Process window calculation
├── visualization/
│   ├── layout_viewer.py          # GDS polygon viewer (matplotlib)
│   ├── aerial_image_viewer.py    # Aerial image + overlay plots
│   └── field_viewer.py           # EM near-field visualization
├── pipeline/
│   ├── simulation_pipeline.py    # End-to-end pipeline
│   └── batch_runner.py           # Parameter sweep batch execution
├── gui/
│   ├── main_window.py            # PySide6/PyQt5 main window (4 tabs)
│   ├── panels/
│   │   ├── layout_panel.py       # GDS viewer panel
│   │   ├── parameter_panel.py    # Lithography parameter controls
│   │   ├── simulation_panel.py   # Run/Stop/Progress panel
│   │   └── results_panel.py      # 2×2 result plots + metrics table
│   └── dialogs/
│       └── source_dialog.py      # k-space illumination preview
├── config/
│   └── default_config.yaml       # ArF 193nm / EUV 13.5nm presets
└── run.sh                        # CLI launcher
```

---

## Quick Start

### Prerequisites (Linux RHEL 7.9+, Python 3.8+)

```bash
pip install numpy scipy matplotlib pyyaml pdfplumber gdstk PySide6
```

### Run GUI

```bash
cd simulation
./run.sh gui
```

### Run CLI Simulation

```bash
./run.sh sim --gds layout.gds --wavelength 193 --na 0.93 --sigma 0.85 --defocus 0
```

### Parameter Sweep (Batch)

```python
from pipeline.batch_runner import BatchRunner

base_config = {
    "lithography": {
        "wavelength_nm": 193.0, "NA": 0.93,
        "illumination": {"type": "annular", "sigma_outer": 0.85, "sigma_inner": 0.55}
    },
    "simulation": {"grid_size": 256, "domain_size_nm": 2000.0}
}
runner = BatchRunner(base_config)
results = runner.run_sweep({"defocus_nm": [-100, -50, 0, 50, 100]})
runner.save_results(results, "./output")
```

### Parse Reference PDF

```bash
./run.sh parse Pistor_2001_Electromagnetic_Simulation_Lithography.pdf
```

---

## Simulation Modes

| Mode | Speed | Accuracy | Use case |
|------|-------|----------|----------|
| Fourier Optics (Abbe) | Fast | Scalar, thin mask | CD/NILS/DOF optimization |
| FDTD (TEMPEST) | Slow | Rigorous EM, vector | EUV thick mask, PSM defects |

---

## Supported Configurations

| Technology | λ (nm) | NA | σ_outer | σ_inner |
|-----------|--------|-----|---------|---------|
| ArF Immersion | 193 | 0.93 | 0.85 | 0.55 |
| ArF Dry | 193 | 0.75 | 0.80 | 0.50 |
| KrF | 248 | 0.68 | 0.75 | 0.45 |
| EUV | 13.5 | 0.33 | 0.85 | 0.55 |

---

## Reference

> Pistor, T.V. (2001). *Electromagnetic Simulation and Modeling with Applications in Lithography*. PhD Dissertation, EECS Department, University of California, Berkeley. Memorandum No. UCB/ERL M01/19. Advisor: Prof. Andrew R. Neureuther.
