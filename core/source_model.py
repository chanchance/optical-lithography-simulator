"""
Illumination source models for photolithography simulation.
Based on Pistor 2001 Chapter 4 (Imaging System Modeling).

Implements Köhler illumination with various pupil shapes:
- Circular, Annular, Quadrupole, Quasar, Freeform

Source discretization formula (Eq 4-1):
n_2D = (pi/4) * (Ns * 2*sigma*NA*w / lambda)^2
"""
import os
import numpy as np
from typing import List, Tuple, NamedTuple


class SourcePoint(NamedTuple):
    """A single illumination source point in k-space."""
    kx: float       # Normalized kx (units of NA/lambda)
    ky: float       # Normalized ky
    weight: float   # Integration weight
    polarization: str  # 'TE', 'TM', or 'both'


class BaseSource:
    """Base class for all illumination sources."""

    def __init__(self, NA: float, wavelength_nm: float, N_points: int = 4,
                 polarization: str = 'unpolarized'):
        self.NA = NA
        self.wavelength_nm = wavelength_nm
        self.N_points = N_points
        self.polarization = polarization
        self._points = None

    def get_source_points(self) -> List[SourcePoint]:
        if self._points is None:
            self._points = self._compute_points()
        return self._points

    def _compute_points(self) -> List[SourcePoint]:
        raise NotImplementedError

    def _cartesian_grid(self, kx_range: Tuple, ky_range: Tuple, n: int) -> Tuple:
        """Generate Cartesian grid of source points."""
        kx = np.linspace(kx_range[0], kx_range[1], n)
        ky = np.linspace(ky_range[0], ky_range[1], n)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        return KX.ravel(), KY.ravel()

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1."""
        total = np.sum(weights)
        if total > 0:
            return weights / total
        return weights

    def _te_tm_decomposition(self, kx: float, ky: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose plane wave into TE and TM polarization vectors.
        TE: E perpendicular to plane of incidence
        TM: E in plane of incidence
        """
        k_perp = np.sqrt(kx**2 + ky**2)
        if k_perp < 1e-10:
            # Normal incidence: TE=x, TM=y
            e_te = np.array([1.0, 0.0, 0.0])
            e_tm = np.array([0.0, 1.0, 0.0])
        else:
            # TE perpendicular to k_perp and z
            e_te = np.array([-ky, kx, 0.0]) / k_perp
            # TM: k cross TE / |k|
            kz = np.sqrt(max(0.0, 1.0 - kx**2 - ky**2))
            k_vec = np.array([kx, ky, kz])
            e_tm = np.cross(k_vec, e_te)
            norm = np.linalg.norm(e_tm)
            if norm > 1e-10:
                e_tm /= norm
        return e_te, e_tm


class CircularSource(BaseSource):
    """
    Circular (conventional) illumination.
    Source fills a disk of radius sigma*NA in k-space.
    """

    def __init__(self, NA: float, sigma: float, wavelength_nm: float,
                 N_points: int = 4, polarization: str = 'unpolarized'):
        super().__init__(NA, wavelength_nm, N_points, polarization)
        self.sigma = sigma

    def _compute_points(self) -> List[SourcePoint]:
        # Sample on Cartesian grid within sigma disk
        n = max(2, int(self.N_points * self.sigma * 10))
        kx_arr = np.linspace(-self.sigma, self.sigma, n)
        ky_arr = np.linspace(-self.sigma, self.sigma, n)
        KX, KY = np.meshgrid(kx_arr, ky_arr, indexing='ij')

        # Only keep points inside sigma circle
        mask = (KX**2 + KY**2) <= self.sigma**2
        KX = KX[mask]
        KY = KY[mask]

        n_pts = len(KX)
        if n_pts == 0:
            # Fallback: on-axis only
            return [SourcePoint(0.0, 0.0, 1.0, self.polarization)]

        weight = 1.0 / n_pts
        points = []
        for i in range(n_pts):
            points.append(SourcePoint(float(KX[i]), float(KY[i]), weight, self.polarization))
        return points


class AnnularSource(BaseSource):
    """
    Annular illumination: ring between sigma_inner and sigma_outer.
    Common in advanced lithography for improved CD uniformity.
    """

    def __init__(self, NA: float, sigma_outer: float, sigma_inner: float,
                 wavelength_nm: float, N_points: int = 4, polarization: str = 'unpolarized'):
        super().__init__(NA, wavelength_nm, N_points, polarization)
        self.sigma_outer = sigma_outer
        self.sigma_inner = sigma_inner

    def _compute_points(self) -> List[SourcePoint]:
        n = max(4, int(self.N_points * self.sigma_outer * 12))
        kx_arr = np.linspace(-self.sigma_outer, self.sigma_outer, n)
        ky_arr = np.linspace(-self.sigma_outer, self.sigma_outer, n)
        KX, KY = np.meshgrid(kx_arr, ky_arr, indexing='ij')

        r2 = KX**2 + KY**2
        mask = (r2 <= self.sigma_outer**2) & (r2 >= self.sigma_inner**2)
        KX, KY = KX[mask], KY[mask]

        n_pts = len(KX)
        if n_pts == 0:
            # Single ring at midpoint
            theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
            r_mid = 0.5 * (self.sigma_outer + self.sigma_inner)
            KX = r_mid * np.cos(theta)
            KY = r_mid * np.sin(theta)
            n_pts = len(KX)

        weight = 1.0 / n_pts
        return [SourcePoint(float(KX[i]), float(KY[i]), weight, self.polarization)
                for i in range(n_pts)]


class QuadrupoleSource(BaseSource):
    """
    Quadrupole illumination: 4 poles at ±x and ±y positions.
    Optimized for features oriented along x or y axes.
    """

    def __init__(self, NA: float, sigma_c: float, sigma_r: float,
                 wavelength_nm: float, N_points: int = 4, polarization: str = 'unpolarized'):
        super().__init__(NA, wavelength_nm, N_points, polarization)
        self.sigma_c = sigma_c   # Pole radius
        self.sigma_r = sigma_r   # Pole center distance from origin

    def _compute_points(self) -> List[SourcePoint]:
        # 4 pole centers at N, S, E, W
        pole_centers = [
            (self.sigma_r, 0.0),
            (-self.sigma_r, 0.0),
            (0.0, self.sigma_r),
            (0.0, -self.sigma_r),
        ]

        n = max(2, self.N_points)
        theta = np.linspace(0, 2*np.pi, n*4, endpoint=False)
        r_arr = np.linspace(0, self.sigma_c, max(2, n))

        all_points = []
        for cx, cy in pole_centers:
            for r in r_arr:
                for t in theta:
                    kx = cx + r * np.cos(t)
                    ky = cy + r * np.sin(t)
                    all_points.append((kx, ky))

        n_pts = len(all_points)
        weight = 1.0 / n_pts
        return [SourcePoint(float(p[0]), float(p[1]), weight, self.polarization)
                for p in all_points]


class QuasarSource(BaseSource):
    """
    Quasar illumination: 4 arc-shaped poles rotated by theta_q.
    Used for diagonal feature orientations.
    """

    def __init__(self, NA: float, sigma_c: float, sigma_r: float, theta_q: float,
                 wavelength_nm: float, N_points: int = 4, polarization: str = 'unpolarized'):
        super().__init__(NA, wavelength_nm, N_points, polarization)
        self.sigma_c = sigma_c
        self.sigma_r = sigma_r
        self.theta_q = np.radians(theta_q)   # Opening half-angle

    def _compute_points(self) -> List[SourcePoint]:
        # 4 poles at 45°, 135°, 225°, 315°
        pole_angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]

        n_angular = max(3, self.N_points * 2)
        n_radial = max(2, self.N_points)

        r_arr = np.linspace(self.sigma_r - self.sigma_c,
                            self.sigma_r + self.sigma_c, n_radial)
        r_arr = r_arr[r_arr > 0]

        all_points = []
        for center_angle in pole_angles:
            for r in r_arr:
                for dt in np.linspace(-self.theta_q, self.theta_q, n_angular):
                    angle = center_angle + dt
                    kx = r * np.cos(angle)
                    ky = r * np.sin(angle)
                    all_points.append((kx, ky))

        n_pts = max(1, len(all_points))
        weight = 1.0 / n_pts
        return [SourcePoint(float(p[0]), float(p[1]), weight, self.polarization)
                for p in all_points]


class FreeformSource:
    """
    Freeform pupil-plane illumination source.
    Defined by a 2D intensity map on the normalized pupil coordinate grid.
    Supports: loaded from file (CSV/NPY), manual pixel editing, or mathematical expression.

    Ref: Pistor Ch.4.2 — arbitrary source shapes for source optimization
    """

    def __init__(self, pupil_size: int = 64):
        self.pupil_size = pupil_size
        self._map = np.zeros((pupil_size, pupil_size), dtype=np.float64)
        self._sigma_max = 1.0

    @classmethod
    def from_array(cls, arr: np.ndarray, sigma_max: float = 1.0) -> 'FreeformSource':
        """Create from a 2D numpy array (will be normalized to [0,1])."""
        obj = cls(pupil_size=arr.shape[0])
        obj._map = np.clip(arr.astype(np.float64), 0, None)
        if obj._map.max() > 0:
            obj._map /= obj._map.max()
        obj._sigma_max = sigma_max
        return obj

    @classmethod
    def from_file(cls, path: str, sigma_max: float = 1.0) -> 'FreeformSource':
        """Load from .npy or .csv file."""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            arr = np.load(path)
        elif ext in ('.csv', '.txt'):
            arr = np.loadtxt(path, delimiter=',')
        else:
            raise ValueError("Unsupported format: {}. Use .npy or .csv".format(ext))
        return cls.from_array(arr, sigma_max)

    @classmethod
    def from_expression(cls, expr: str, pupil_size: int = 64,
                         sigma_max: float = 1.0) -> 'FreeformSource':
        """
        Create from a mathematical expression.
        Variables: x, y (normalized pupil coords in [-1, 1]), r (radius)
        Example: "np.exp(-r**2 / 0.1)" or "(r > 0.3) & (r < 0.7)"
        """
        s = np.linspace(-1, 1, pupil_size)
        x, y = np.meshgrid(s, s)
        r = np.sqrt(x**2 + y**2)
        # Safe eval with restricted namespace
        ns = {'np': np, 'x': x, 'y': y, 'r': r,
              'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
              'abs': np.abs, 'sqrt': np.sqrt}
        arr = eval(expr, {"__builtins__": {}}, ns)
        arr = np.asarray(arr, dtype=np.float64)
        # Apply sigma_max circular aperture
        arr[r > sigma_max] = 0.0
        return cls.from_array(arr, sigma_max)

    def get_source_points(self) -> tuple:
        """
        Return (sigma_x, sigma_y, weights) arrays for Abbe integration.
        Only returns points with intensity > threshold.
        """
        N = self.pupil_size
        s = np.linspace(-self._sigma_max, self._sigma_max, N)
        sx, sy = np.meshgrid(s, s)

        # Flatten and threshold
        flat_weights = self._map.ravel()
        flat_sx = sx.ravel()
        flat_sy = sy.ravel()

        mask = flat_weights > 1e-4
        return flat_sx[mask], flat_sy[mask], flat_weights[mask]

    def get_map(self) -> np.ndarray:
        return self._map.copy()

    def set_pixel(self, row: int, col: int, value: float) -> None:
        """Set individual pixel (for interactive painting)."""
        if 0 <= row < self.pupil_size and 0 <= col < self.pupil_size:
            self._map[row, col] = np.clip(value, 0.0, 1.0)

    def save(self, path: str) -> None:
        """Save map to .npy or .csv."""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            np.save(path, self._map)
        else:
            np.savetxt(path, self._map, delimiter=',', fmt='%.6f')

    @property
    def sigma_max(self) -> float:
        return self._sigma_max

    @property
    def n_points(self) -> int:
        return int(np.sum(self._map > 1e-4))


def create_source(config: dict):
    """Factory function to create illumination source from config dict."""
    illum = config.get('illumination', config)
    source_type = illum.get('type', 'circular').lower()
    NA = config.get('NA', 0.93)
    wl = config.get('wavelength_nm', 193.0)
    N = illum.get('N_source_points', 4)
    pol = illum.get('polarization', 'unpolarized')

    if source_type == 'circular':
        return CircularSource(NA, illum.get('sigma_outer', 0.85), wl, N, pol)
    elif source_type == 'annular':
        return AnnularSource(NA, illum.get('sigma_outer', 0.85),
                             illum.get('sigma_inner', 0.55), wl, N, pol)
    elif source_type == 'quadrupole':
        return QuadrupoleSource(NA, illum.get('sigma_c', 0.15),
                                illum.get('sigma_r', 0.30), wl, N, pol)
    elif source_type == 'quasar':
        return QuasarSource(NA, illum.get('sigma_c', 0.15), illum.get('sigma_r', 0.30),
                            illum.get('theta_q', 45.0), wl, N, pol)
    elif source_type == 'freeform':
        expr = illum.get('expression', '')
        path = illum.get('file', '')
        sigma_max = illum.get('sigma_max', 1.0)
        pupil_size = illum.get('pupil_size', 64)
        if path:
            return FreeformSource.from_file(path, sigma_max)
        elif expr:
            return FreeformSource.from_expression(expr, pupil_size, sigma_max)
        else:
            return FreeformSource(pupil_size)
    else:
        raise ValueError("Unknown source type: {}".format(source_type))
