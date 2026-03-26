"""
Advanced lithography metrics: MEEF, Bossung curves, Focus-Exposure Matrix, LER.
Complements aerial_image_analysis.py with process-window-level analysis.
"""
import copy
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

    Usage via factory (recommended)::

        from pipeline.simulation_pipeline import SimulationPipeline
        meef = MEEF.from_pipeline(SimulationPipeline(), nominal_cd_nm=100.0)
        result = meef.compute(config, mask_delta_nm=10.0)
        # result.meef is the MEEF value; typical ArF range 1.0–3.5

    Usage with custom pipeline_fn::

        def pipeline_fn(cfg, mask_delta_nm):
            cfg2 = copy.deepcopy(cfg)
            # apply mask_delta_nm as dose perturbation proxy
            nominal_cd = cfg2.get('_nominal_cd_nm', 100.0)
            cfg2.setdefault('lithography', {})['dose_factor'] = 1.0 + mask_delta_nm / nominal_cd
            return SimulationPipeline().run(cfg2).cd_nm

        meef = MEEF(pipeline_fn)
        result = meef.compute(config, mask_delta_nm=10.0)
    """
    magnification: float = 4.0

    def __init__(self, pipeline_fn: Callable, magnification: float = 4.0):
        """
        pipeline_fn: callable(config, mask_delta_nm) → cd_nm
        Runs simulation with ±delta mask CD change, measures wafer CD change.
        The callable MUST vary the simulation based on mask_delta_nm (0.0 = nominal).
        Use MEEF.from_pipeline() to get a correctly-wired callable automatically.
        """
        self.pipeline_fn = pipeline_fn
        self.magnification = magnification

    @classmethod
    def from_pipeline(cls, pipeline, nominal_cd_nm: float = 100.0,
                      magnification: float = 4.0) -> 'MEEF':
        """
        Factory: build MEEF from a SimulationPipeline instance.

        Mask CD perturbation is implemented as a dose_factor shift:
            dose_factor = 1.0 + mask_delta_nm / nominal_cd_nm
        This approximates a ±delta mask CD change at the aerial-image level.

        Args:
            pipeline: SimulationPipeline instance (or any object with .run(config)).
            nominal_cd_nm: Reference CD in nm used to convert mask delta to dose
                           fraction. Use the expected target CD for the feature.
            magnification: Mask demagnification factor (default 4x for ArF/EUV).

        Smoke test::

            from pipeline.simulation_pipeline import SimulationPipeline
            meef = MEEF.from_pipeline(SimulationPipeline(), nominal_cd_nm=100.0)
            config = {
                'lithography': {'wavelength_nm': 193.0, 'NA': 0.93, 'defocus_nm': 0.0,
                                 'illumination': {'type': 'annular',
                                                  'sigma_outer': 0.85, 'sigma_inner': 0.55}},
                'simulation': {'grid_size': 32, 'domain_size_nm': 1000.0},
                'resist': {'model': 'threshold', 'threshold': 0.30},
                'analysis': {'cd_threshold': 0.30},
            }
            result = meef.compute(config, mask_delta_nm=10.0)
            assert isinstance(result.meef, float)
            assert result.meef >= 0.0
        """
        ref_cd = float(nominal_cd_nm) if nominal_cd_nm and nominal_cd_nm > 0 else 100.0

        def _pipeline_fn(config: dict, mask_delta_nm: float) -> float:
            cfg = copy.deepcopy(config)
            litho = cfg.setdefault('lithography', {})
            # Translate mask CD offset to a dose_factor perturbation:
            # a larger mask feature (positive delta) increases effective dose.
            base_dose = litho.get('dose_factor', 1.0)
            litho['dose_factor'] = base_dose + mask_delta_nm / ref_cd
            r = pipeline.run(cfg)
            return r.cd_nm

        return cls(_pipeline_fn, magnification=magnification)

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
            # Filter out zero-CD points (unresolved features at extreme defocus)
            valid_mask = cd_arr > 0
            valid_focus = focus_values[valid_mask]
            valid_cd = cd_arr[valid_mask]
            # Best focus = focus with maximum CD symmetry (minimum d²CD/df²)
            # Simplified: focus at peak of parabolic fit
            try:
                if len(valid_focus) >= 3:
                    coeffs = np.polyfit(valid_focus, valid_cd, 2)
                    best_focus = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else 0.0
                    best_focus = float(np.clip(best_focus, focus_values[0], focus_values[-1]))
                elif len(valid_focus) > 0:
                    # Fewer than 3 valid points: use focus at maximum non-zero CD
                    best_focus = float(valid_focus[np.argmax(valid_cd)])
                else:
                    best_focus = float(focus_values[len(focus_values) // 2])
            except Exception:
                best_focus = float(focus_values[len(focus_values) // 2])

            # DOF: range of focus where CD stays within ±tolerance of nominal
            # Use max non-zero CD as nominal to avoid tol=0 when midpoint is unresolved
            if len(valid_cd) > 0:
                cd_nominal = float(np.max(valid_cd))
            else:
                cd_nominal = float(cd_arr[len(cd_arr) // 2])
            if cd_nominal > 0 and len(valid_focus) >= 3:
                tol = cd_nominal * cd_tolerance_pct / 100.0
                in_window = np.abs(cd_arr - cd_nominal) <= tol
                # np.linspace step = range / (n-1), not range / n
                focus_step = focus_range_nm / (focus_steps - 1) if focus_steps > 1 else 0.0
                dof = float(np.sum(in_window) * focus_step)
            else:
                dof = 0.0

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
