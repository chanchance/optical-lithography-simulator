"""
Signal analysis tools for optical lithography.
Extracts intensity profiles, peaks, dose latitude metrics.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import signal as scipy_signal


class SignalAnalyzer:
    """Signal analysis for aerial image intensity profiles."""

    def __init__(self, domain_size_nm: float, grid_size: int):
        self.domain_size_nm = domain_size_nm
        self.grid_size = grid_size
        self.dx_nm = domain_size_nm / grid_size

    def extract_profile(self, intensity_2d: np.ndarray,
                         direction: str = 'x',
                         position_nm: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 1D intensity profile from 2D aerial image.

        Args:
            intensity_2d: 2D intensity array
            direction: 'x' or 'y'
            position_nm: Position along perpendicular axis (None=center)
        Returns:
            (x_nm, intensity_1d) arrays
        """
        N = self.grid_size
        if position_nm is None:
            idx = N // 2
        else:
            idx = int(position_nm / self.dx_nm)
            idx = np.clip(idx, 0, N-1)

        x_nm = np.arange(N) * self.dx_nm
        if direction == 'x':
            profile = intensity_2d[idx, :]   # row = scan along x at y=idx
        else:
            profile = intensity_2d[:, idx]   # column = scan along y at x=idx

        return x_nm, profile

    def find_peaks(self, profile: np.ndarray,
                   min_height: float = 0.3,
                   min_distance_px: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find intensity peaks in profile.
        Returns (positions_nm, amplitudes).
        """
        peaks, props = scipy_signal.find_peaks(
            profile, height=min_height, distance=min_distance_px
        )
        positions_nm = peaks * self.dx_nm
        amplitudes = profile[peaks]
        return positions_nm, amplitudes

    def find_valleys(self, profile: np.ndarray,
                     max_height: float = 0.7,
                     min_distance_px: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find intensity valleys (minima) in profile.
        Returns (positions_nm, amplitudes).
        """
        valleys, props = scipy_signal.find_peaks(
            -profile, height=-max_height, distance=min_distance_px
        )
        positions_nm = valleys * self.dx_nm
        amplitudes = profile[valleys]
        return positions_nm, amplitudes

    def normalize_signal(self, profile: np.ndarray) -> np.ndarray:
        """Normalize profile to [0, 1] range."""
        I_min = np.min(profile)
        I_max = np.max(profile)
        if I_max - I_min < 1e-10:
            return np.zeros_like(profile)
        return (profile - I_min) / (I_max - I_min)

    def compute_dose_latitude(self, profile: np.ndarray,
                               threshold: float = 0.30,
                               cd_target_nm: float = 100.0,
                               cd_tolerance_pct: float = 10.0) -> float:
        """
        Compute dose latitude: range of dose variation for CD within tolerance.
        Returns dose latitude in percent.
        """
        cd_tolerance = cd_target_nm * cd_tolerance_pct / 100.0
        dx = self.dx_nm

        dose_factors = np.linspace(0.5, 2.0, 30)
        cds = []

        for df in dose_factors:
            scaled = profile * df
            # Find threshold crossings
            crossings = []
            for i in range(len(scaled)-1):
                if (scaled[i] - threshold) * (scaled[i+1] - threshold) <= 0:
                    t = (threshold - scaled[i]) / (scaled[i+1] - scaled[i] + 1e-30)
                    crossings.append((i + t) * dx)
            if len(crossings) >= 2:
                cds.append(abs(crossings[1] - crossings[0]))
            else:
                cds.append(0.0)

        cds = np.array(cds)
        in_spec = np.abs(cds - cd_target_nm) <= cd_tolerance

        if not np.any(in_spec):
            return 0.0

        valid_doses = dose_factors[in_spec]
        dose_latitude_pct = (np.max(valid_doses) - np.min(valid_doses)) / np.mean(valid_doses) * 100.0
        return float(dose_latitude_pct)

    def smooth_profile(self, profile: np.ndarray, sigma_px: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing to profile."""
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(profile, sigma=sigma_px)
