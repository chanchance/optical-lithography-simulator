"""
Thin-film stack simulation using Transfer Matrix Method (TMM).
Pistor 2001 Eq. 4-10/11: film stack factor f(k, z).

Layer convention:
    layers[0]   = incident medium (semi-infinite, e.g. air)
    layers[1..N-2] = thin films
    layers[N-1] = substrate (semi-infinite)
"""
from dataclasses import dataclass, field
import numpy as np
from typing import List

# Built-in material library (n, k) at 193nm
MATERIALS_193NM = {
    'vacuum': (1.0, 0.0),
    'air': (1.0, 0.0),
    'resist': (1.7, 0.02),
    'pr': (1.7, 0.02),
    'arc': (1.8, 0.35),       # anti-reflection coating
    'barc': (1.8, 0.35),
    'si': (0.883, 2.78),      # silicon substrate
    'sio2': (1.56, 0.0),
    'si3n4': (2.02, 0.0),
    'cr': (0.84, 1.65),
}

# EUV 13.5nm material library
MATERIALS_13NM = {
    'vacuum': (1.0, 0.0),
    'air': (1.0, 0.0),
    'mo': (0.923, 0.0064),
    'si': (0.999, 0.0018),
    'tan': (0.93, 0.041),     # TaN absorber
    'ru': (0.886, 0.017),
}


def _get_material_nk(material: str, wavelength_nm: float):
    """Return (n, k) for material name at given wavelength."""
    lib = MATERIALS_13NM if wavelength_nm < 50.0 else MATERIALS_193NM
    key = material.lower()
    if key not in lib:
        raise ValueError(
            f"Unknown material '{material}' for wavelength {wavelength_nm}nm. "
            f"Available: {list(lib.keys())}"
        )
    return lib[key]


@dataclass
class Layer:
    material: str           # key in MATERIALS dict, or 'custom'
    thickness_nm: float     # 0 for semi-infinite (incident medium / substrate)
    n: float = 1.0          # refractive index (set automatically from material)
    k: float = 0.0          # extinction coefficient

    def complex_n(self) -> complex:
        return complex(self.n, -self.k)


@dataclass
class FilmStack:
    """Film stack where layers[0] is the incident medium and layers[-1] is substrate."""
    layers: List[Layer] = field(default_factory=list)
    wavelength_nm: float = 193.0

    def __post_init__(self):
        """Auto-populate n, k from material name if not 'custom'."""
        for layer in self.layers:
            if layer.material.lower() != 'custom':
                try:
                    n, k = _get_material_nk(layer.material, self.wavelength_nm)
                    layer.n = n
                    layer.k = k
                except ValueError:
                    pass  # leave user-supplied values intact

    @classmethod
    def default_193nm(cls) -> 'FilmStack':
        """Typical ArF stack: air / BARC / Resist / Si substrate."""
        wl = 193.0
        spec = [('air', 0.0), ('barc', 77.0), ('resist', 100.0), ('si', 0.0)]
        layers = []
        for mat, t in spec:
            n, k = _get_material_nk(mat, wl)
            layers.append(Layer(material=mat, thickness_nm=t, n=n, k=k))
        return cls(layers=layers, wavelength_nm=wl)

    @classmethod
    def default_euv(cls) -> 'FilmStack':
        """EUV stack: vacuum / Ru capping / Si substrate."""
        wl = 13.5
        spec = [('vacuum', 0.0), ('ru', 2.5), ('si', 0.0)]
        layers = []
        for mat, t in spec:
            n, k = _get_material_nk(mat, wl)
            layers.append(Layer(material=mat, thickness_nm=t, n=n, k=k))
        return cls(layers=layers, wavelength_nm=wl)


def optimize_barc(wavelength_nm: float, n_resist: float, k_resist: float,
                  n_range=(1.4, 2.2), k_range=(0.1, 0.6),
                  thickness_range=(20, 120)) -> dict:
    """
    Minimize substrate reflectance by sweeping BARC n, k, thickness.
    Returns: {'n': optimal_n, 'k': optimal_k, 'thickness_nm': t, 'reflectance': R}
    Uses scipy.optimize.minimize or brute-force grid search.
    """
    engine = TransferMatrixEngine()

    def substrate_reflectance(n_barc, k_barc, t_barc):
        layers = [
            Layer('custom', 0.0, n=1.0, k=0.0),       # air
            Layer('custom', t_barc, n=n_barc, k=k_barc),  # BARC
            Layer('custom', 0.0, n=n_resist, k=k_resist),  # resist as substrate
        ]
        stack = FilmStack(layers=layers, wavelength_nm=wavelength_nm)
        r = engine.reflectance(stack)
        return abs(r) ** 2

    scipy_best_x = None
    scipy_best_R = float('inf')
    try:
        from scipy.optimize import minimize

        def objective(x):
            n_b, k_b, t_b = x
            return substrate_reflectance(n_b, k_b, t_b)

        # Multi-start to avoid local minima
        for n0 in np.linspace(n_range[0], n_range[1], 3):
            for k0 in np.linspace(k_range[0], k_range[1], 3):
                t0 = (thickness_range[0] + thickness_range[1]) / 2.0
                try:
                    res = minimize(
                        objective, x0=[n0, k0, t0],
                        bounds=[n_range, k_range, thickness_range],
                        method='L-BFGS-B',
                    )
                    if res.fun < scipy_best_R:
                        scipy_best_R = res.fun
                        scipy_best_x = res.x
                except Exception:
                    pass
    except ImportError:
        pass  # scipy not available; fall through to brute-force

    if scipy_best_x is not None:
        n_opt, k_opt, t_opt = scipy_best_x
        return {'n': float(n_opt), 'k': float(k_opt),
                'thickness_nm': float(t_opt), 'reflectance': float(scipy_best_R)}

    # Brute-force grid search fallback (no scipy, or all minimize calls failed)
    n_vals = np.linspace(n_range[0], n_range[1], 15)
    k_vals = np.linspace(k_range[0], k_range[1], 15)
    t_vals = np.linspace(thickness_range[0], thickness_range[1], 20)

    best_R = float('inf')
    best = (n_vals[0], k_vals[0], t_vals[0])
    for n_b in n_vals:
        for k_b in k_vals:
            for t_b in t_vals:
                R = substrate_reflectance(n_b, k_b, t_b)
                if R < best_R:
                    best_R = R
                    best = (n_b, k_b, t_b)

    return {'n': float(best[0]), 'k': float(best[1]),
            'thickness_nm': float(best[2]), 'reflectance': float(best_R)}


class TransferMatrixEngine:
    """2×2 Transfer Matrix Method for TE/TM polarization.

    Builds M = T_{0→1} · P_1 · T_{1→2} · P_2 · ... · T_{N-2→N-1}
    where T_{i→i+1} is the interface refraction matrix and P_i is the
    propagation matrix through layer i (films only, not incident/substrate).

    With [E+_0, E-_0]^T = M · [E+_sub, 0]^T:
        r = M[1,0] / M[0,0]
        t = 1 / M[0,0]

    References:
        Pistor 2001, Eq. 4-10/4-11.
        Born & Wolf, Principles of Optics, Ch. 1.6.
    """

    # ------------------------------------------------------------------
    # Low-level matrix helpers
    # ------------------------------------------------------------------

    def _kz(self, n_complex: complex, kx: float, wavelength_nm: float) -> complex:
        """kz = sqrt((n·k0)^2 − kx^2) on the physical forward-propagating branch.

        For propagating waves in absorbing media (Im(n) < 0 with n=n_r-ik):
            the correct root has Re(kz) > 0 (forward direction).
        For evanescent waves (Re(kz)=0): Im(kz) > 0 (decaying away).
        Using Im(kz)>=0 is wrong for lossy media — it selects the backward root.
        """
        k0 = 2.0 * np.pi / wavelength_nm
        kz = np.sqrt((n_complex * k0) ** 2 - kx ** 2 + 0j)
        # Choose root with Re(kz) >= 0; for purely evanescent choose Im(kz) >= 0
        if kz.real < 0 or (kz.real == 0.0 and kz.imag < 0):
            kz = -kz
        return kz

    def _interface_matrix(self, n1: complex, n2: complex,
                          kz1: complex, kz2: complex,
                          polarization: str) -> np.ndarray:
        """2×2 interface (dynamical) matrix T_{1→2}.

        Derived from D_1^{-1} · D_2 with dynamical matrices D_j = [[1,1],[pj,-pj]]:
            T = (1/2p1) · [[p1+p2, p1-p2], [p1-p2, p1+p2]]
        """
        if polarization == 'te':
            p1, p2 = kz1, kz2
        else:                   # TM: matching parameter = kz/n^2
            p1 = kz1 / (n1 ** 2)
            p2 = kz2 / (n2 ** 2)
        factor = 1.0 / (2.0 * p1)
        return factor * np.array([
            [p1 + p2, p1 - p2],
            [p1 - p2, p1 + p2],
        ], dtype=complex)

    def _propagation_matrix(self, kz: complex, thickness_nm: float) -> np.ndarray:
        """2×2 propagation matrix P for a film of given thickness."""
        phi = kz * thickness_nm
        return np.array([
            [np.exp(1j * phi), 0.0],
            [0.0, np.exp(-1j * phi)],
        ], dtype=complex)

    # ------------------------------------------------------------------
    # Core transfer matrix builder
    # ------------------------------------------------------------------

    def _build_transfer_matrix(self, stack: FilmStack, kx: float,
                                polarization: str) -> np.ndarray:
        """Return the 2×2 total transfer matrix M for the stack.

        layers[0]  = incident medium (semi-infinite, no propagation)
        layers[-1] = substrate       (semi-infinite, no propagation)
        layers[1..-2] = films        (propagation included)
        """
        if len(stack.layers) < 2:
            raise ValueError("FilmStack needs at least incident medium + substrate (≥ 2 layers).")

        wl = stack.wavelength_nm
        pol = polarization.lower()
        n_list = [layer.complex_n() for layer in stack.layers]
        kz_list = [self._kz(n, kx, wl) for n in n_list]

        M = np.eye(2, dtype=complex)
        N = len(stack.layers)

        for i in range(1, N):
            # Interface from layer i-1 → layer i
            D = self._interface_matrix(n_list[i - 1], n_list[i],
                                       kz_list[i - 1], kz_list[i], pol)
            M = M @ D
            # Propagation through layer i (skip substrate, i.e. the last layer)
            if i < N - 1:
                t = stack.layers[i].thickness_nm
                if t > 0:
                    P = self._propagation_matrix(kz_list[i], t)
                    M = M @ P

        return M

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reflectance(self, stack: FilmStack, angle_deg: float = 0.0,
                    polarization: str = 'te') -> complex:
        """Complex reflectance r = E-_in / E+_in."""
        n_in = stack.layers[0].complex_n()
        k0 = 2.0 * np.pi / stack.wavelength_nm
        kx = float((n_in * k0 * np.sin(np.deg2rad(angle_deg))).real)
        M = self._build_transfer_matrix(stack, kx, polarization)
        return M[1, 0] / M[0, 0]

    def transmittance(self, stack: FilmStack, angle_deg: float = 0.0,
                      polarization: str = 'te') -> complex:
        """Complex transmittance t = E+_sub / E+_in."""
        n_in = stack.layers[0].complex_n()
        k0 = 2.0 * np.pi / stack.wavelength_nm
        kx = float((n_in * k0 * np.sin(np.deg2rad(angle_deg))).real)
        M = self._build_transfer_matrix(stack, kx, polarization)
        return 1.0 / M[0, 0]

    def standing_wave_intensity(self, stack: FilmStack,
                                z_positions: np.ndarray,
                                angle_deg: float = 0.0,
                                polarization: str = 'te') -> np.ndarray:
        """Normalized intensity |E(z)|^2 vs depth z (nm from top of layer[1]).

        z=0 is at the top surface of the first film (layers[1]).
        Incident intensity is normalized to 1.
        """
        wl = stack.wavelength_nm
        pol = polarization.lower()
        n_in = stack.layers[0].complex_n()
        k0 = 2.0 * np.pi / wl
        kx = float((n_in * k0 * np.sin(np.deg2rad(angle_deg))).real)

        n_list = [layer.complex_n() for layer in stack.layers]
        kz_list = [self._kz(n, kx, wl) for n in n_list]
        N = len(stack.layers)

        # Build cumulative boundary depths (z=0 at top of first film)
        boundaries = [0.0]
        for layer in stack.layers[1:-1]:
            boundaries.append(boundaries[-1] + layer.thickness_nm)

        # Forward/backward amplitudes at each film boundary via partial matrices
        # amplitude_at[i] = [E+, E-] just inside layers[i], at its top surface
        amplitude_at = [None] * N
        # At incident medium entry (layer 0): E+ = 1, E- = r
        r = self.reflectance(stack, angle_deg, polarization)
        amplitude_at[0] = np.array([1.0 + 0j, r], dtype=complex)

        for i in range(1, N):
            # First propagate through layer i-1 to its bottom surface, then
            # cross the interface into layer i.  Layer 0 is semi-infinite so
            # no propagation is needed for i==1.
            amp = amplitude_at[i - 1]
            if i > 1:
                t_prev = stack.layers[i - 1].thickness_nm
                if t_prev > 0:
                    P_prev = self._propagation_matrix(kz_list[i - 1], t_prev)
                    amp = P_prev @ amp

            # Interface matrix from layer i-1 → layer i
            D = self._interface_matrix(n_list[i - 1], n_list[i],
                                       kz_list[i - 1], kz_list[i], pol)
            try:
                ev = np.linalg.solve(D, amp)
            except np.linalg.LinAlgError:
                # Singular interface matrix (degenerate stack); propagate
                # forward amplitude unchanged and zero reflected amplitude.
                ev = np.array([amp[0], 0.0 + 0j], dtype=complex)
            amplitude_at[i] = ev
            # amplitude_at[i] now holds [E+, E-] at the TOP of layer i

        intensity = np.zeros(len(z_positions), dtype=float)
        for iz, z in enumerate(z_positions):
            # Find which film layer z falls in (layers[1..N-2])
            layer_idx = N - 2  # default: deepest film / substrate
            z_in_layer = z - boundaries[-1]
            for li in range(len(boundaries) - 1):
                if boundaries[li] <= z < boundaries[li + 1]:
                    layer_idx = li + 1   # film index in full layers list
                    z_in_layer = z - boundaries[li]
                    break

            # Propagate within the layer
            ev = amplitude_at[layer_idx]
            kz = kz_list[layer_idx]
            E_plus = ev[0] * np.exp(1j * kz * z_in_layer)
            E_minus = ev[1] * np.exp(-1j * kz * z_in_layer)
            intensity[iz] = abs(E_plus + E_minus) ** 2

        return intensity

    def film_factor(self, stack: FilmStack,
                    kx_norm: np.ndarray,
                    ky_norm: np.ndarray,
                    polarization: str = 'te') -> np.ndarray:
        """Pistor Eq. 4-10: f(k) film stack correction factor.

        Parameters
        ----------
        kx_norm, ky_norm : normalized spatial frequencies (dimensionless, k/k0)
        polarization : 'te' or 'tm'

        Returns
        -------
        Complex array, same shape as kx_norm.
        f(k) = 1 + r(k), the total forward-field amplitude at resist top.
        """
        k0 = 2.0 * np.pi / stack.wavelength_nm
        result = np.zeros(kx_norm.shape, dtype=complex)
        kx_flat = kx_norm.ravel()
        ky_flat = ky_norm.ravel()

        for idx in range(len(kx_flat)):
            # Full in-plane wavevector magnitude: k_par = k0 * sqrt(kx_norm² + ky_norm²)
            # TMM is rotationally symmetric; only k_par enters kz = sqrt((nk0)²−k_par²)
            kpar_phys = float(np.sqrt(kx_flat[idx]**2 + ky_flat[idx]**2)) * k0
            try:
                M = self._build_transfer_matrix(stack, kpar_phys, polarization)
                m00 = M[0, 0]
                if m00 == 0 or not np.isfinite(m00):
                    result.ravel()[idx] = 1.0
                else:
                    r_val = M[1, 0] / m00
                    val = 1.0 + r_val
                    # NumPy never raises ZeroDivisionError; check explicitly for
                    # non-finite results that would propagate as nan/inf.
                    result.ravel()[idx] = val if np.isfinite(abs(val)) else 1.0
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                result.ravel()[idx] = 1.0

        return result
