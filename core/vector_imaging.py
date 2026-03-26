"""
Vector imaging engine using Jones matrix / Mk-matrix formalism.
Based on Pistor 2001 Chapter 4.5: Vector Imaging.

In scalar imaging, the field is treated as a single complex amplitude.
Vector imaging accounts for the polarization state of light through the
projection optic, using Jones matrices to track the 2D electric field vector.

The Jones pupil P_J(kx,ky) is a 2x2 matrix mapping input polarization
to output polarization at each pupil point.

Mk-matrix formulation (Sheppard/Torok):
  For each source point s with incident field Jones vector J_in:
    E_out(x,y) = IFFT[ P_J(fx,fy) * J_in * M(fx+ks_x, fy+ks_y) ]
  I(x,y) = sum_s w_s * |E_out_x|^2 + |E_out_y|^2
"""
import numpy as np
from enum import Enum
from typing import Optional, List, Tuple


class Polarization(Enum):
    X = "x"                   # x-linear polarization
    Y = "y"                   # y-linear polarization
    TE = "te"                 # transverse electric (s-pol, perpendicular to plane of incidence)
    TM = "tm"                 # transverse magnetic (p-pol, in plane of incidence)
    CIRCULAR_L = "circular_l" # left circular polarization
    CIRCULAR_R = "circular_r" # right circular polarization
    UNPOLARIZED = "unpolarized"


# Jones vectors for each polarization state (normalized)
JONES_VECTORS = {
    Polarization.X:          np.array([1.0, 0.0], dtype=np.complex128),
    Polarization.Y:          np.array([0.0, 1.0], dtype=np.complex128),
    Polarization.CIRCULAR_L: np.array([1.0,  1j], dtype=np.complex128) / np.sqrt(2),
    Polarization.CIRCULAR_R: np.array([1.0, -1j], dtype=np.complex128) / np.sqrt(2),
}


class VectorImagingEngine:
    """
    Jones matrix / Mk-matrix vector imaging engine.
    Based on Pistor Ch.4.5 vector diffraction theory.

    For high-NA systems (NA > 0.6), scalar imaging underestimates polarization
    effects. The Jones pupil approach tracks the 2x2 polarization transfer
    matrix at each pupil point, coupling input polarization to output field.

    TE/TM decomposition:
        TE (s-pol): electric field perpendicular to the plane of incidence
                    defined by the z-axis and the ray direction
        TM (p-pol): electric field in the plane of incidence
    """

    def __init__(self, params: dict):
        self.NA = params.get('NA', 0.93)
        self.wavelength_nm = params.get('wavelength_nm', 193.0)
        self.defocus_nm = params.get('defocus_nm', 0.0)
        self.grid_size = params.get('grid_size', 256)
        self.domain_size_nm = params.get('domain_size_nm', 2000.0)
        self.dx_nm = self.domain_size_nm / self.grid_size

        # Reuse ZernikeAberrations from fourier_optics if available
        from .fourier_optics import ZernikeAberrations, ProjectionOptic
        self.projection_optic = ProjectionOptic(
            self.NA, self.wavelength_nm,
            aberrations=params.get('aberrations')
        )

        # Frequency grids
        N = self.grid_size
        dx = self.dx_nm
        fx_1d = np.fft.fftfreq(N, d=dx)
        fy_1d = np.fft.fftfreq(N, d=dx)
        self.FX, self.FY = np.meshgrid(fx_1d, fy_1d, indexing='ij')

    def _build_pupil_jones(self, kx_norm: np.ndarray,
                           ky_norm: np.ndarray) -> np.ndarray:
        """
        Build 2x2 Jones pupil matrix at each pupil point.

        The Jones matrix maps (Ex_in, Ey_in) -> (Ex_out, Ey_out) through
        the projection optic, accounting for the TE/TM rotation at each ray angle.

        For a ray with direction (kx, ky) in normalized pupil coordinates:
          - Plane of incidence contains z-axis and ray direction
          - TE direction (s-pol): phi_hat = (-ky, kx) / ||(kx,ky)||
          - TM direction (p-pol): rotated by the refraction angle

        At high NA, TM rays undergo a z-component coupling that reduces
        the in-plane field amplitude (apodization factor cos(theta)).

        Args:
            kx_norm, ky_norm: normalized pupil coordinates (kx/NA, ky/NA)
                              each shape (N,N), values in [-1,1]
        Returns:
            jones: shape (2, 2, N, N) complex array
                   jones[i,j] = Jones matrix element (out_i, in_j)
        """
        N = self.grid_size
        kr = np.sqrt(kx_norm**2 + ky_norm**2)
        # Avoid division by zero at pupil center
        kr_safe = np.where(kr < 1e-10, 1e-10, kr)

        # Azimuthal unit vectors (TE direction = s-pol)
        # phi_hat: perpendicular to plane of incidence in xy-plane
        te_x = -ky_norm / kr_safe   # TE x-component
        te_y =  kx_norm / kr_safe   # TE y-component

        # TM direction in xy-plane (p-pol, in plane of incidence)
        tm_x = kx_norm / kr_safe    # TM x-component
        tm_y = ky_norm / kr_safe    # TM y-component

        # High-NA apodization: TM rays projected onto image plane
        # cos(theta) = sqrt(1 - (NA*kr)^2)  for rays within pupil
        NA_kr = np.clip(self.NA * kr, 0.0, 1.0 - 1e-15)
        cos_theta = np.sqrt(1.0 - NA_kr**2)

        # Within-pupil mask
        in_pupil = (kr <= 1.0).astype(np.float64)

        # Jones matrix construction:
        # For each input polarization component (x=0, y=1):
        #   Project onto TE/TM basis, apply cos(theta) to TM, rotate back
        #
        # J = te_hat * te_hat^T + cos(theta) * tm_hat * tm_hat^T
        #
        # J[out_x, in_x] = te_x*te_x + cos_theta*tm_x*tm_x
        # J[out_x, in_y] = te_x*te_y + cos_theta*tm_x*tm_y
        # J[out_y, in_x] = te_y*te_x + cos_theta*tm_y*tm_x
        # J[out_y, in_y] = te_y*te_y + cos_theta*tm_y*tm_y

        jones = np.zeros((2, 2, N, N), dtype=np.complex128)
        jones[0, 0] = in_pupil * (te_x * te_x + cos_theta * tm_x * tm_x)
        jones[0, 1] = in_pupil * (te_x * te_y + cos_theta * tm_x * tm_y)
        jones[1, 0] = in_pupil * (te_y * te_x + cos_theta * tm_y * tm_x)
        jones[1, 1] = in_pupil * (te_y * te_y + cos_theta * tm_y * tm_y)

        # At pupil center (on-axis), Jones matrix reduces to identity
        on_axis = (kr < 1e-10)
        jones[0, 0][on_axis] = 1.0
        jones[0, 1][on_axis] = 0.0
        jones[1, 0][on_axis] = 0.0
        jones[1, 1][on_axis] = 1.0

        return jones

    def _jones_vector_for_source(self, polarization: Polarization,
                                  kx_norm: float, ky_norm: float) -> np.ndarray:
        """
        Get input Jones vector for a given polarization and source direction.

        For TE/TM, the polarization direction depends on the source k-vector.
        For X/Y/Circular, it's independent of direction.

        Args:
            polarization: Polarization enum value
            kx_norm, ky_norm: source point normalized k-vector
        Returns:
            Jones vector shape (2,) complex
        """
        if polarization in JONES_VECTORS:
            return JONES_VECTORS[polarization]

        kr = np.sqrt(kx_norm**2 + ky_norm**2)
        if kr < 1e-10:
            # On-axis: TE -> x, TM -> y (convention)
            if polarization == Polarization.TE:
                return JONES_VECTORS[Polarization.X]
            else:  # TM
                return JONES_VECTORS[Polarization.Y]

        # TE: perpendicular to plane of incidence (s-pol)
        # phi_hat = (-ky, kx) / kr
        if polarization == Polarization.TE:
            return np.array([-ky_norm / kr, kx_norm / kr], dtype=np.complex128)

        # TM: in plane of incidence (p-pol), projected to xy
        # tm_hat = (kx, ky) / kr
        if polarization == Polarization.TM:
            return np.array([kx_norm / kr, ky_norm / kr], dtype=np.complex128)

        raise ValueError(f"Unknown polarization: {polarization}")

    def compute_aerial_image_vector(self,
                                     mask_fft: np.ndarray,
                                     source_points,
                                     polarization: Polarization = Polarization.UNPOLARIZED
                                     ) -> np.ndarray:
        """
        Compute vectorial aerial image using Jones/Mk-matrix formalism.

        Algorithm (Pistor Ch.4.5 Abbe + vector):
          For each source point s = (kx_s, ky_s, w_s):
            1. Get input Jones vector J_in for polarization + source direction
            2. Shift mask spectrum for oblique illumination (phase ramp)
            3. Apply scalar pupil (aberrations + apodization)
            4. Apply Jones pupil: each component of J_in propagates through
               the corresponding Jones matrix column
            5. IFFT each field component (Ex, Ey) -> image plane
            6. Accumulate I += w_s * (|Ex|^2 + |Ey|^2)

          For UNPOLARIZED: average of X and Y input polarizations.

        Args:
            mask_fft: 2D FFT of mask transmission, shape (N,N) complex
            source_points: list of SourcePoint with .kx, .ky, .weight
                           (kx,ky in normalized pupil coords -1..1)
            polarization: Polarization enum
        Returns:
            I_total: 2D float array, normalized to [0,1]
        """
        N = self.grid_size
        NA_freq = self.NA / self.wavelength_nm  # NA in cycles/nm

        # Normalized pupil coordinates at each frequency grid point
        kx_norm = self.FX / (NA_freq + 1e-30)
        ky_norm = self.FY / (NA_freq + 1e-30)

        # Scalar pupil function (aberrations)
        P_scalar = self.projection_optic.pupil_function(
            self.FX, self.FY, self.defocus_nm)

        # Jones pupil (2x2 per pixel)
        jones = self._build_pupil_jones(kx_norm, ky_norm)

        # Apply scalar pupil phase to Jones pupil
        jones = jones * P_scalar[np.newaxis, np.newaxis, :, :]

        # Spatial coordinate grids for phase ramp
        x_1d = np.arange(N) * self.dx_nm
        y_1d = np.arange(N) * self.dx_nm
        X, Y = np.meshgrid(x_1d, y_1d, indexing='ij')

        # Determine which polarizations to compute
        if polarization == Polarization.UNPOLARIZED:
            pol_list = [Polarization.X, Polarization.Y]
            pol_weights = [0.5, 0.5]
        else:
            pol_list = [polarization]
            pol_weights = [1.0]

        I_total = np.zeros((N, N), dtype=np.float64)

        for pol, pol_w in zip(pol_list, pol_weights):
            for sp in source_points:
                # Source k-vector in cycles/nm
                ks_x = sp.kx * NA_freq
                ks_y = sp.ky * NA_freq

                # Input Jones vector for this polarization + source direction
                J_in = self._jones_vector_for_source(pol, sp.kx, sp.ky)

                # Shift mask spectrum: multiply by phase ramp in spatial domain
                phase_ramp = np.exp(1j * 2.0 * np.pi * (ks_x * X + ks_y * Y))
                # mask_fft already computed; recompute shifted version
                # We store original mask_fft and apply shift each time
                M_shifted = mask_fft * np.exp(
                    1j * 2.0 * np.pi * (ks_x * X + ks_y * Y))
                # Note: mask_fft here is FFT(mask); for shifted illumination
                # we need FFT(mask * phase_ramp). Since mask_fft = FFT(mask),
                # and FFT(mask * e^{i2pi(ksx*x+ksy*y)}) needs spatial domain.
                # Recompute properly via convolution theorem is done below.
                # Actually mask_fft passed in is FFT(mask_transmission),
                # so M_shifted(fx,fy) = FFT(t * e^{i*phi})(fx,fy)
                # = integrate t(x)*e^{i2pi(ksx*x+ksy*y)} * e^{-i2pi(fx*x+fy*y)}
                # = FFT(t)[fx-ksx, fy-ksy]  (shift in freq domain)
                # Implement as: IFFT(M) -> t, multiply phase_ramp, FFT back
                # This is done correctly by passing phase_ramp * mask_spatial
                # We need spatial domain mask. Store it separately.
                # --> See note: caller passes mask_fft = FFT(mask_transmission)
                #     We can recover mask_transmission = IFFT(mask_fft), but
                #     it's better for caller to pass spatial domain mask.
                #     For now: use IFFT to get spatial, apply ramp, FFT back.
                #     This is equivalent to circular shift in freq domain.
                mask_spatial = np.fft.ifft2(mask_fft)
                M_shifted = np.fft.fft2(mask_spatial * phase_ramp)

                # Apply Jones pupil: E_out = J @ J_in * M_shifted
                # J_in = (Jin_x, Jin_y)
                # E_out_x(fx,fy) = (J[0,0]*Jin_x + J[0,1]*Jin_y) * M_shifted
                # E_out_y(fx,fy) = (J[1,0]*Jin_x + J[1,1]*Jin_y) * M_shifted
                Ex_spectrum = (jones[0, 0] * J_in[0] + jones[0, 1] * J_in[1]) * M_shifted
                Ey_spectrum = (jones[1, 0] * J_in[0] + jones[1, 1] * J_in[1]) * M_shifted

                # IFFT to image plane
                Ex = np.fft.ifft2(Ex_spectrum)
                Ey = np.fft.ifft2(Ey_spectrum)

                # Accumulate intensity
                I_total += pol_w * sp.weight * (np.abs(Ex)**2 + np.abs(Ey)**2)

        # Normalize
        I_max = np.max(I_total)
        if I_max > 0:
            I_total /= I_max

        return I_total
