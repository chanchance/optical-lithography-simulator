"""
Aerial image analysis tools for optical lithography simulation.
Computes CD, NILS, DOF, contrast, EPE, and process window metrics.

Key metrics:
- CD (Critical Dimension): width of printed feature at threshold
- NILS (Normalized Image Log-Slope): image quality metric at feature edge
  NILS = (w/I) * dI/dx at threshold crossing
- DOF (Depth of Focus): focus range for CD within tolerance
- Contrast = (I_max - I_min) / (I_max + I_min)
"""
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from scipy import ndimage, interpolate


@dataclass
class AerialImageMetrics:
    """Container for aerial image analysis results."""
    cd_nm: float = 0.0          # Critical dimension (nm)
    nils: float = 0.0           # Normalized image log-slope
    contrast: float = 0.0       # Image contrast
    i_max: float = 0.0          # Maximum intensity
    i_min: float = 0.0          # Minimum intensity
    threshold_used: float = 0.30
    dof_nm: Optional[float] = None
    dose_latitude_pct: Optional[float] = None


class AerialImageAnalyzer:
    """
    Analyzes aerial images to extract lithographic performance metrics.
    """

    def __init__(self, domain_size_nm: float, grid_size: int):
        self.domain_size_nm = domain_size_nm
        self.grid_size = grid_size
        self.dx_nm = domain_size_nm / grid_size

    def compute_cd(self, intensity_2d: np.ndarray, threshold: float = 0.30,
                   direction: str = 'x', position: Optional[float] = None) -> float:
        """
        Compute critical dimension at threshold crossing.

        Args:
            intensity_2d: 2D normalized intensity array [0,1]
            threshold: Intensity threshold for CD measurement (default 0.30)
            direction: 'x' or 'y' scan direction
            position: Position along perpendicular axis (nm, None=center)
        Returns:
            CD in nm
        """
        N = self.grid_size

        if position is None:
            idx = N // 2
        else:
            idx = int(position / self.dx_nm)
            idx = np.clip(idx, 0, N-1)

        if direction == 'x':
            profile = intensity_2d[idx, :]   # row = scan along x at y=idx
        else:
            profile = intensity_2d[:, idx]   # column = scan along y at x=idx

        return self._cd_from_profile(profile, threshold)

    def _cd_from_profile(self, profile: np.ndarray, threshold: float) -> float:
        """
        Extract CD from 1D intensity profile by finding threshold crossings.
        """
        N = len(profile)
        x_nm = np.arange(N) * self.dx_nm

        # Find threshold crossings using interpolation
        crossings = []
        for i in range(N - 1):
            if (profile[i] - threshold) * (profile[i+1] - threshold) <= 0:
                # Linear interpolation of crossing
                t = (threshold - profile[i]) / (profile[i+1] - profile[i] + 1e-30)
                x_cross = x_nm[i] + t * self.dx_nm
                crossings.append(x_cross)

        if len(crossings) < 2:
            return 0.0

        # CD = distance between first pair of crossings
        return abs(crossings[1] - crossings[0])

    def compute_nils(self, intensity_1d: np.ndarray, cd_nm: float,
                     threshold: float = 0.30) -> float:
        """
        Compute Normalized Image Log-Slope (NILS).
        NILS = (w/I) * dI/dx evaluated at threshold crossing.

        Args:
            intensity_1d: 1D intensity profile
            cd_nm: Critical dimension in nm (feature width w)
            threshold: Intensity threshold
        Returns:
            NILS value (dimensionless, higher is better)
        """
        N = len(intensity_1d)
        x_nm = np.arange(N) * self.dx_nm

        # Smooth gradient computation
        dI_dx = np.gradient(intensity_1d, self.dx_nm)

        # Find threshold crossings
        crossings_x = []
        crossings_I = []
        crossings_dI = []

        for i in range(N - 1):
            if (intensity_1d[i] - threshold) * (intensity_1d[i+1] - threshold) <= 0:
                t = (threshold - intensity_1d[i]) / (intensity_1d[i+1] - intensity_1d[i] + 1e-30)
                x_c = x_nm[i] + t * self.dx_nm
                I_c = threshold
                dI_c = dI_dx[i] + t * (dI_dx[i+1] - dI_dx[i])
                crossings_x.append(x_c)
                crossings_I.append(I_c)
                crossings_dI.append(dI_c)

        if not crossings_dI or cd_nm <= 0:
            return 0.0

        # NILS = (w/I) * |dI/dx| at threshold
        # Use average over all crossings
        nils_values = []
        for I_c, dI_c in zip(crossings_I, crossings_dI):
            if I_c > 1e-10:
                nils = (cd_nm / I_c) * abs(dI_c)
                nils_values.append(nils)

        return float(np.mean(nils_values)) if nils_values else 0.0

    def compute_contrast(self, intensity_2d: np.ndarray) -> float:
        """
        Compute aerial image contrast.
        Contrast = (I_max - I_min) / (I_max + I_min)
        """
        I_max = float(np.max(intensity_2d))
        I_min = float(np.min(intensity_2d))
        denom = I_max + I_min
        if denom < 1e-10:
            return 0.0
        return (I_max - I_min) / denom

    def compute_dof(self, aerial_images_vs_defocus: List[np.ndarray],
                    defocus_nm_values: List[float],
                    threshold: float = 0.30,
                    cd_target_nm: float = 100.0,
                    cd_tolerance_pct: float = 10.0) -> float:
        """
        Compute Depth of Focus (DOF).
        DOF = focus range over which CD is within tolerance of target.

        Args:
            aerial_images_vs_defocus: List of aerial images at different defocus
            defocus_nm_values: Corresponding defocus values in nm
            threshold: Intensity threshold
            cd_target_nm: Target CD in nm
            cd_tolerance_pct: Allowed CD variation in %
        Returns:
            DOF in nm
        """
        cd_tolerance = cd_target_nm * cd_tolerance_pct / 100.0

        cds = []
        for img in aerial_images_vs_defocus:
            cd = self.analyze(img, threshold).cd_nm
            cds.append(cd)

        cds = np.array(cds)
        defocus = np.array(defocus_nm_values)

        # Find focus range where CD is within tolerance
        in_spec = np.abs(cds - cd_target_nm) <= cd_tolerance

        if not np.any(in_spec):
            return 0.0

        # DOF = span of in-spec defocus values
        in_spec_defocus = defocus[in_spec]
        return float(np.max(in_spec_defocus) - np.min(in_spec_defocus))

    def compute_epe(self, printed_edge_nm: np.ndarray,
                    target_edge_nm: np.ndarray) -> np.ndarray:
        """
        Compute Edge Placement Error (EPE).
        EPE = printed_edge - target_edge (positive = over-exposure)

        Args:
            printed_edge_nm: Printed edge positions (nm)
            target_edge_nm: Target (design) edge positions (nm)
        Returns:
            EPE array in nm
        """
        return printed_edge_nm - target_edge_nm

    def analyze(self, intensity_2d: np.ndarray,
                threshold: float = 0.30) -> AerialImageMetrics:
        """
        Full analysis of aerial image.
        Returns AerialImageMetrics with all computed values.
        """
        I_max = float(np.max(intensity_2d))
        I_min = float(np.min(intensity_2d))
        contrast = self.compute_contrast(intensity_2d)

        # Try both scan directions; use whichever gives more threshold crossings.
        # The line/space test pattern varies along Y, so the column scan wins there,
        # while user-loaded GDS layouts may vary along X.
        n_rows, n_cols = intensity_2d.shape
        profile_x = intensity_2d[n_rows // 2, :]
        profile_y = intensity_2d[:, n_cols // 2]

        def _count_crossings(p):
            return sum(1 for i in range(len(p) - 1)
                       if (p[i] - threshold) * (p[i + 1] - threshold) <= 0)

        profile = profile_y if _count_crossings(profile_y) > _count_crossings(profile_x) else profile_x
        cd = self._cd_from_profile(profile, threshold)

        # Warn when CD=0 despite meaningful contrast — the selected center-line
        # profile never crosses the threshold (all values above threshold).
        profile_min = float(np.min(profile))
        if cd == 0.0 and contrast > 0.1 and profile_min > threshold:
            import warnings
            warnings.warn(
                "CD=0: center-line profile minimum ({:.3f}) is above threshold ({:.2f}). "
                "Features exist but threshold is too low for this pattern; "
                "consider raising cd_threshold or using a larger domain.".format(
                    profile_min, threshold),
                stacklevel=2,
            )

        # NILS
        nils = self.compute_nils(profile, cd, threshold)

        return AerialImageMetrics(
            cd_nm=cd,
            nils=nils,
            contrast=contrast,
            i_max=I_max,
            i_min=I_min,
            threshold_used=threshold
        )

    def process_window_matrix(self, dose_range_pct: List[float],
                               focus_range_nm: List[float],
                               aerial_images: Dict[Tuple, np.ndarray],
                               cd_target_nm: float = 100.0,
                               cd_tolerance_pct: float = 10.0,
                               threshold: float = 0.30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute process window pass/fail matrix.

        Returns:
            (pass_fail_grid, dose_axis, focus_axis)
        """
        n_dose = len(dose_range_pct)
        n_focus = len(focus_range_nm)
        pw_grid = np.zeros((n_dose, n_focus), dtype=bool)
        cd_tolerance = cd_target_nm * cd_tolerance_pct / 100.0

        for i, dose in enumerate(dose_range_pct):
            for j, focus in enumerate(focus_range_nm):
                key = (round(dose, 2), round(focus, 2))
                if key in aerial_images:
                    img = aerial_images[key]
                    # Scale intensity by dose factor
                    dose_factor = 1.0 + dose / 100.0
                    img_dosed = img * dose_factor
                    cd = self.analyze(img_dosed, threshold).cd_nm
                    pw_grid[i, j] = abs(cd - cd_target_nm) <= cd_tolerance

        return pw_grid, np.array(dose_range_pct), np.array(focus_range_nm)
