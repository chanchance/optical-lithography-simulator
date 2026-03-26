"""
Fourier optics engine for aerial image computation.
Based on Pistor 2001 Chapter 4: Imaging System Modeling.

Implements Abbe's formulation (Source Integration / Hopkins theory):
  I(x,y) = sum_s w_s * |IFFT[ M(fx,fy) * P(fx,fy) * T(fx+ks_x, fy+ks_y) ]|^2

where:
  M(fx,fy) = mask diffraction spectrum (Fourier transform of mask transmission)
  P(fx,fy) = projection optic pupil function (circ aperture + aberrations)
  T(fx,fy) = film stack transfer function
  (ks_x, ks_y) = source point k-vector
  w_s = source point weight
"""
import numpy as np
from typing import Optional, List, Tuple

from .source_model import BaseSource, SourcePoint
from .aberrations import ZernikeAberration


class _LegacyZernikeAberrations:
    """Legacy 5-term Zernike model using W0xx coefficients (backward compat)."""

    def __init__(self, coefficients: Optional[dict] = None):
        self.coefficients = coefficients or {}

    def compute_wavefront(self, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Compute wavefront error W(rho, phi) in waves.
        rho: normalized pupil radius (0 to 1)
        phi: pupil azimuthal angle
        Returns W in waves (multiply by 2*pi for radians).
        """
        W = np.zeros_like(rho, dtype=np.float64)

        W020 = self.coefficients.get('W020', 0.0)
        W040 = self.coefficients.get('W040', 0.0)
        W111 = self.coefficients.get('W111', 0.0)
        W131 = self.coefficients.get('W131', 0.0)
        W222 = self.coefficients.get('W222', 0.0)

        W += W020 * rho**2                              # Defocus
        W += W040 * rho**4                              # Spherical
        W += W111 * rho * np.cos(phi)                  # Tilt
        W += W131 * rho**3 * np.cos(phi)               # Coma
        W += W222 * rho**2 * np.cos(2*phi)             # Astigmatism

        return W


class ProjectionOptic:
    """
    Projection optic model (Pistor Section 4.4).
    Models the transfer function from mask to wafer.
    """

    def __init__(self, NA: float, wavelength_nm: float,
                 magnification: float = 4.0,
                 aberrations: Optional[dict] = None):
        self.NA = NA
        self.wavelength_nm = wavelength_nm
        self.wavelength = wavelength_nm * 1e-9
        self.magnification = magnification

        self._legacy_aber: Optional[_LegacyZernikeAberrations] = None
        self._zernike_aber: Optional[ZernikeAberration] = None

        if aberrations:
            if any(isinstance(k, str) for k in aberrations):
                # Old W0xx string-key format — keep legacy path
                self._legacy_aber = _LegacyZernikeAberrations(aberrations)
            elif 'zernike' in aberrations:
                # New format: {"zernike": [c1, c2, ..., c37]}
                self._zernike_aber = ZernikeAberration.from_list(
                    aberrations['zernike'])
            else:
                # Integer key format: {1: c1, 2: c2, ...}
                self._zernike_aber = ZernikeAberration(
                    {int(k): v for k, v in aberrations.items()})

    def pupil_function(self, fx: np.ndarray, fy: np.ndarray,
                       defocus_nm: float = 0.0) -> np.ndarray:
        """
        Compute pupil function P(fx, fy).
        P = circ(NA/lambda) * exp(j * 2*pi * W(rho, phi))

        Args:
            fx, fy: spatial frequencies (cycles/nm)
            defocus_nm: defocus in nm
        Returns:
            Complex pupil function array
        """
        # Normalize frequencies to pupil coordinates
        # rho = sqrt(fx^2 + fy^2) / (NA/lambda)
        NA_freq = self.NA / self.wavelength_nm   # NA in cycles/nm

        r = np.sqrt(fx**2 + fy**2)
        rho = r / (NA_freq + 1e-30)
        phi = np.arctan2(fy, fx)

        # Circular aperture (binary)
        circ_mask = (rho <= 1.0).astype(np.float64)

        # Defocus phase — rigorous formula from Hopkins theory:
        #   W_defocus = (dz / lambda) * (sqrt(1 - (rho*NA)^2) - 1)
        # The quadratic approx W ~ dz*NA^2*rho^2/lambda is only valid for
        # small NA. For immersion (NA>1) the sqrt form is essential.
        rho_NA = np.clip(rho * self.NA, 0, 1.0 - 1e-15)
        W_defocus = (defocus_nm / self.wavelength_nm) * (
            np.sqrt(1.0 - rho_NA**2) - 1.0
        )

        # Wavefront from aberration model
        if self._legacy_aber is not None:
            W_aber = self._legacy_aber.compute_wavefront(rho, phi)  # waves
            phase = 2.0 * np.pi * (W_defocus + W_aber)
        elif self._zernike_aber is not None:
            kx_n = fx / (NA_freq + 1e-30)
            ky_n = fy / (NA_freq + 1e-30)
            phase_aber = self._zernike_aber.pupil_phase(kx_n, ky_n)  # radians
            phase = 2.0 * np.pi * W_defocus + phase_aber
        else:
            phase = 2.0 * np.pi * W_defocus

        # Pupil function: P = circ * exp(j * phase)
        P = circ_mask * np.exp(1j * phase)

        return P


class FourierOpticsEngine:
    """
    Aerial image computation via Abbe's source integration formulation.
    Based on Pistor 2001 Chapter 4.

    Algorithm:
    1. For each source point s with k-vector (ks_x, ks_y):
       a. Compute shifted mask spectrum: M_s(fx,fy) = M(fx+ks_x, fy+ks_y)
       b. Apply pupil: M_s * P(fx, fy)
       c. IFFT to get partial image field: E_s(x,y)
       d. Add |E_s|^2 weighted by source weight w_s
    2. I(x,y) = sum_s w_s * |E_s(x,y)|^2
    """

    def __init__(self, config: dict):
        self.config = config
        self.NA = config.get('NA', 0.93)
        self.wavelength_nm = config.get('wavelength_nm', 193.0)
        self.defocus_nm = config.get('defocus_nm', 0.0)
        self.grid_size = config.get('grid_size', 256)
        self.domain_size_nm = config.get('domain_size_nm', 2000.0)
        self.use_vector = config.get('use_vector', False)
        self.polarization = config.get('polarization', 'unpolarized')
        self.use_hopkins = config.get('use_hopkins', False)
        self.n_kernels = config.get('n_kernels', 10)
        self.use_gpu = config.get('use_gpu', False)

        # Pixel size
        self.dx_nm = self.domain_size_nm / self.grid_size

        # Spatial frequency coordinates
        self._setup_frequency_grid()

        # Projection optic
        self.projection_optic = ProjectionOptic(
            self.NA, self.wavelength_nm,
            aberrations=config.get('aberrations'),
        )

    def _setup_frequency_grid(self):
        """Setup spatial frequency (k-space) grid."""
        N = self.grid_size
        dx = self.dx_nm

        # Frequency sampling: df = 1/(N*dx)
        df = 1.0 / (N * dx)   # cycles/nm

        fx_1d = np.fft.fftfreq(N, d=dx)   # cycles/nm
        fy_1d = np.fft.fftfreq(N, d=dx)

        self.FX, self.FY = np.meshgrid(fx_1d, fy_1d, indexing='ij')
        self.df = df

    def compute_aerial_image(self, mask_transmission: np.ndarray,
                              source: BaseSource) -> np.ndarray:
        """
        Compute aerial image using Abbe source integration.

        Args:
            mask_transmission: 2D complex array of mask transmission t(x,y)
            source: Illumination source (CircularSource, AnnularSource, etc.)
        Returns:
            2D float array of intensity I(x,y), normalized to [0,1]
        """
        N = self.grid_size
        if mask_transmission.shape != (N, N):
            raise ValueError(
                "Mask must be {}x{}, got {}".format(N, N, mask_transmission.shape))

        # Delegate to Hopkins TCC/SOCS engine when use_hopkins=True
        if self.use_hopkins:
            from .hopkins import HopkinsTCC
            tcc = HopkinsTCC(self.config)
            tcc.compute(source)
            return tcc.aerial_image(mask_transmission)

        # Delegate to vector imaging engine when use_vector=True
        if self.use_vector:
            from .vector_imaging import VectorImagingEngine, Polarization
            vec_engine = VectorImagingEngine(self.config)
            pol = Polarization(self.polarization)
            mask_fft = np.fft.fft2(mask_transmission)
            source_points = source.get_source_points()
            return vec_engine.compute_aerial_image_vector(mask_fft, source_points, pol)

        # Select compute backend (NumPy or CuPy)
        if self.use_gpu:
            from .gpu_backend import to_gpu, to_cpu, fft2 as _fft2, ifft2 as _ifft2
            xp = None  # not needed directly; use backend functions
        else:
            to_gpu = lambda a: a  # noqa: E731
            to_cpu = lambda a: a  # noqa: E731
            _fft2 = np.fft.fft2
            _ifft2 = np.fft.ifft2

        # Transfer static arrays to device
        mask_gpu = to_gpu(mask_transmission)
        P_np = self.projection_optic.pupil_function(self.FX, self.FY, self.defocus_nm)
        P_gpu = to_gpu(P_np)

        # Accumulate intensity from all source points
        I_total = to_gpu(np.zeros((N, N), dtype=np.float64))

        source_points = source.get_source_points()
        NA_freq = self.NA / self.wavelength_nm  # NA in cycles/nm

        # Pre-compute spatial coordinate grids (constant across source points)
        x_1d = np.arange(N) * self.dx_nm
        y_1d = np.arange(N) * self.dx_nm
        X_np, Y_np = np.meshgrid(x_1d, y_1d, indexing='ij')
        X_gpu = to_gpu(X_np)
        Y_gpu = to_gpu(Y_np)

        for sp in source_points:
            # Source plane wave direction: (ks_x, ks_y) in cycles/nm
            ks_x = sp.kx * NA_freq
            ks_y = sp.ky * NA_freq

            # Shift mask spectrum for oblique illumination
            # M_shifted(fx,fy) = M(fx+ks_x, fy+ks_y)
            # Implement as phase ramp in spatial domain (shift theorem)
            phase_ramp = np.exp(1j * 2.0 * np.pi * (ks_x * X_np + ks_y * Y_np))
            phase_gpu = to_gpu(phase_ramp)
            M_shifted = _fft2(mask_gpu * phase_gpu)

            # Apply pupil function and IFFT
            E_image = _ifft2(M_shifted * P_gpu)

            # Accumulate weighted intensity
            I_total = I_total + sp.weight * (E_image.real**2 + E_image.imag**2)

        # Back to CPU
        I_total = to_cpu(I_total)

        # Normalize
        I_max = np.max(I_total)
        if I_max > 0:
            I_total /= I_max

        return I_total

    def compute_transfer_function(self) -> np.ndarray:
        """
        Compute coherent transfer function H(fx,fy) for on-axis illumination.
        H(fx,fy) = circ(sqrt(fx^2+fy^2) / (NA/lambda))
        """
        NA_freq = self.NA / self.wavelength_nm
        R = np.sqrt(self.FX**2 + self.FY**2)
        H = (R <= NA_freq).astype(np.complex128)
        return H

    def compute_coherent_image(self, mask_transmission: np.ndarray) -> np.ndarray:
        """
        Compute coherent aerial image (single plane wave illumination).
        Simpler but less physical than partial coherent Abbe formulation.
        """
        M = np.fft.fft2(mask_transmission)
        H = self.compute_transfer_function()
        E_image = np.fft.ifft2(M * H)
        I = np.abs(E_image)**2
        I_max = np.max(I)
        if I_max > 0:
            I /= I_max
        return I
