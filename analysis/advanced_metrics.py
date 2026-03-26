"""
Advanced lithography metrics: MEEF, Bossung curves, Focus-Exposure Matrix, LER.
Complements aerial_image_analysis.py with process-window-level analysis.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable


@dataclass
class MEEFResult:
    meef: float            # Mask Error Enhancement Factor
    cd_nominal_nm: float
    cd_plus_nm: float
    cd_minus_nm: float
    mask_delta_nm: float


class MEEF:
    """
    Mask Error Enhancement Factor.
    MEEF = (dCD_wafer / dCD_mask) * M  where M = demagnification (4x)
    """
    magnification: float = 4.0

    def __init__(self, pipeline_fn: Callable, magnification: float = 4.0):
        """
        pipeline_fn: callable(config, mask_delta_nm) → cd_nm
        Runs simulation with ±delta mask CD change, measures wafer CD change.
        """
        self.pipeline_fn = pipeline_fn
        self.magnification = magnification

    def compute(self, config: dict, mask_delta_nm: float = 1.0) -> MEEFResult:
        cd_nominal = self.pipeline_fn(config, 0.0)
        cd_plus = self.pipeline_fn(config, +mask_delta_nm)
        cd_minus = self.pipeline_fn(config, -mask_delta_nm)
        dcd_wafer = (cd_plus - cd_minus) / 2.0
        dcd_mask = mask_delta_nm / self.magnification
        meef = abs(dcd_wafer / dcd_mask) if dcd_mask != 0 else 0.0
        return MEEFResult(meef, cd_nominal, cd_plus, cd_minus, mask_delta_nm)


@dataclass
class BossungPoint:
    focus_nm: float
    dose_factor: float
    cd_nm: float


@dataclass
class BossungCurve:
    dose_factor: float
    focus_points: List[float]   # nm
    cd_points: List[float]      # nm
    best_focus_nm: float
    depth_of_focus_nm: float    # DOF at ±10% CD tolerance

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.focus_points), np.array(self.cd_points)


class BossungAnalyzer:
    """
    Bossung curves: CD vs defocus for multiple dose levels.
    Named after J.W. Bossung who popularized focus-exposure analysis.
    """

    def __init__(self, pipeline_fn: Callable):
        """pipeline_fn: callable(config, defocus_nm, dose_factor) → cd_nm"""
        self.pipeline_fn = pipeline_fn

    def compute(
        self,
        config: dict,
        focus_range_nm: float = 400.0,
        focus_steps: int = 17,
        dose_factors: List[float] = None,
        cd_tolerance_pct: float = 10.0,
    ) -> List[BossungCurve]:
        if dose_factors is None:
            dose_factors = [0.9, 0.95, 1.0, 1.05, 1.1]

        focus_values = np.linspace(-focus_range_nm / 2, focus_range_nm / 2, focus_steps)
        curves = []

        for dose in dose_factors:
            cd_values = []
            for f in focus_values:
                cd = self.pipeline_fn(config, f, dose)
                cd_values.append(cd)

            cd_arr = np.array(cd_values)
            # Best focus = focus with maximum CD symmetry (minimum d²CD/df²)
            # Simplified: focus at peak of parabolic fit
            try:
                coeffs = np.polyfit(focus_values, cd_arr, 2)
                best_focus = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else 0.0
                best_focus = float(np.clip(best_focus, focus_values[0], focus_values[-1]))
            except Exception:
                best_focus = float(focus_values[len(focus_values) // 2])

            # DOF: range of focus where CD stays within ±tolerance of nominal
            cd_nominal = float(cd_arr[len(cd_arr) // 2])
            tol = cd_nominal * cd_tolerance_pct / 100.0
            in_window = np.abs(cd_arr - cd_nominal) <= tol
            dof = float(np.sum(in_window) * (focus_range_nm / focus_steps))

            curves.append(BossungCurve(
                dose_factor=dose,
                focus_points=focus_values.tolist(),
                cd_points=cd_values,
                best_focus_nm=best_focus,
                depth_of_focus_nm=dof,
            ))
        return curves


@dataclass
class FEMPoint:
    focus_nm: float
    dose_factor: float
    cd_nm: float


@dataclass
class FocusExposureMatrix:
    """Focus-Exposure Matrix: CD map over focus × dose grid."""
    focus_values: np.ndarray    # shape (Nf,)
    dose_values: np.ndarray     # shape (Nd,)
    cd_matrix: np.ndarray       # shape (Nf, Nd)

    def process_window(self, cd_target_nm: float, cd_tolerance_pct: float = 10.0) -> np.ndarray:
        """Return boolean mask of (focus, dose) points within process window."""
        tol = cd_target_nm * cd_tolerance_pct / 100.0
        return np.abs(self.cd_matrix - cd_target_nm) <= tol

    def exposure_latitude(self, focus_nm: float = 0.0) -> float:
        """EL at given focus: % dose range keeping CD in spec."""
        fi = np.argmin(np.abs(self.focus_values - focus_nm))
        cd_row = self.cd_matrix[fi]
        cd_ref = float(cd_row[len(cd_row) // 2])
        tol = cd_ref * 0.10
        in_window = np.abs(cd_row - cd_ref) <= tol
        if np.sum(in_window) < 2:
            return 0.0
        dose_in = self.dose_values[in_window]
        return float((dose_in.max() - dose_in.min()) / np.mean(dose_in) * 100)


def build_fem(
    pipeline_fn: Callable,
    config: dict,
    focus_range_nm: float = 400.0,
    focus_steps: int = 9,
    dose_range: Tuple[float, float] = (0.8, 1.2),
    dose_steps: int = 9,
) -> FocusExposureMatrix:
    """Build full Focus-Exposure Matrix by running pipeline_fn on NxM grid."""
    focus_values = np.linspace(-focus_range_nm / 2, focus_range_nm / 2, focus_steps)
    dose_values = np.linspace(dose_range[0], dose_range[1], dose_steps)
    cd_matrix = np.zeros((focus_steps, dose_steps))

    for fi, f in enumerate(focus_values):
        for di, d in enumerate(dose_values):
            cd_matrix[fi, di] = pipeline_fn(config, f, d)

    return FocusExposureMatrix(focus_values, dose_values, cd_matrix)


@dataclass
class LERResult:
    ler_3sigma_nm: float    # LER (3-sigma), line edge roughness
    lwr_3sigma_nm: float    # LWR (3-sigma), line width roughness
    psd: np.ndarray         # Power Spectral Density
    frequencies: np.ndarray # Spatial frequencies [1/nm]


class LERAnalysis:
    """Line Edge Roughness / Line Width Roughness analysis."""

    def compute_ler(self, edge_positions_nm: np.ndarray) -> LERResult:
        """
        edge_positions_nm: 1D array of edge positions along line length
        """
        # Remove mean (global CD offset)
        edge_detrended = edge_positions_nm - np.mean(edge_positions_nm)
        ler_3sigma = 3.0 * float(np.std(edge_detrended))

        # PSD via Welch or simple FFT
        n = len(edge_detrended)
        if n < 4:
            psd = np.zeros(1)
            freqs = np.zeros(1)
        else:
            fft = np.fft.rfft(edge_detrended)
            psd = np.abs(fft) ** 2 / n
            freqs = np.fft.rfftfreq(n)  # normalized [cycles/sample]

        return LERResult(
            ler_3sigma_nm=ler_3sigma,
            lwr_3sigma_nm=ler_3sigma * np.sqrt(2),
            psd=psd,
            frequencies=freqs,
        )
