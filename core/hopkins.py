"""
Hopkins TCC/SOCS imaging engine for accelerated aerial image computation.

Hopkins Transmission Cross Coefficient (TCC) formulation:
  I(x,y) = ΣΣ TCC(k1,k2) * M(k1) * M*(k2) * exp(2πi(k1-k2)·r)

where TCC(k1,k2) = Σ_s J(s) * P(k1-s) * P*(k2-s)
      J(s) = source intensity at point s
      P(k) = pupil function
      M(k) = mask diffraction spectrum

SOCS (Sum of Coherent Systems) decomposition:
  TCC = Σ_i λ_i φ_i ⊗ φ_i†    (eigendecomposition)
  I(x,y) = Σ_i λ_i |IFFT[M(k) * φ_i(k)]|²

Advantage over Abbe: TCC is mask-independent — computed once per
(source, pupil) configuration. For large source grids (many points),
SOCS with N_kernels << N_source is much faster than Abbe summation.
"""
import numpy as np
from typing import Optional, Tuple, List

from .source_model import BaseSource, SourcePoint
from .fourier_optics import ProjectionOptic


class HopkinsTCC:
    """
    Precomputes the Hopkins TCC matrix and its SOCS eigendecomposition.

    Usage:
        tcc = HopkinsTCC(params)
        tcc.compute(source)          # once per source/pupil config
        I = tcc.aerial_image(mask)   # fast per-mask evaluation
    """

    def __init__(self, params: dict):
        self.NA = params.get('NA', 0.93)
        self.wavelength_nm = params.get('wavelength_nm', 193.0)
        self.defocus_nm = params.get('defocus_nm', 0.0)
        self.grid_size = params.get('grid_size', 256)
        self.domain_size_nm = params.get('domain_size_nm', 2000.0)
        self.n_kernels = params.get('n_kernels', 10)
        self.dx_nm = self.domain_size_nm / self.grid_size

        # Frequency grid
        N = self.grid_size
        dx = self.dx_nm
        fx_1d = np.fft.fftfreq(N, d=dx)
        fy_1d = np.fft.fftfreq(N, d=dx)
        self.FX, self.FY = np.meshgrid(fx_1d, fy_1d, indexing='ij')

        # Projection optic
        self.projection_optic = ProjectionOptic(
            self.NA, self.wavelength_nm,
            aberrations=params.get('aberrations'),
        )

        # SOCS results (set after compute())
        self.eigenvalues: Optional[np.ndarray] = None   # shape (n_kernels,)
        self.eigenkernels: Optional[np.ndarray] = None  # shape (n_kernels, N, N)

    def compute(self, source: BaseSource) -> None:
        """
        Build TCC matrix via SVD of the source-pupil coupling matrix B,
        then store top n_kernels SOCS eigenpairs.

        TCC = B† @ B  where  B[s, k] = sqrt(w_s) * P(f_k - f_s)

        SVD(B) = U Σ V†  =>  TCC = V Σ² V†
        eigenvalues λ_i = σ_i²,  eigenvectors φ_i = V[:,i]

        Args:
            source: illumination source providing get_source_points()
        """
        source_points = source.get_source_points()
        N = self.grid_size
        NA_freq = self.NA / self.wavelength_nm  # cycles/nm

        n_src = len(source_points)
        n_freq = N * N

        # Build B matrix: shape (n_src, n_freq)
        # B[s, k] = sqrt(w_s) * P(FX[k] - ks_x[s], FY[k] - ks_y[s])
        B = np.zeros((n_src, n_freq), dtype=np.complex128)

        for i, sp in enumerate(source_points):
            ks_x = sp.kx * NA_freq
            ks_y = sp.ky * NA_freq

            # Evaluate pupil at shifted frequencies
            FX_shifted = self.FX - ks_x
            FY_shifted = self.FY - ks_y
            P_s = self.projection_optic.pupil_function(
                FX_shifted, FY_shifted, self.defocus_nm)

            B[i, :] = np.sqrt(sp.weight) * P_s.ravel()

        # SVD: B = U @ diag(sigma) @ Vh
        # TCC = B†B = Vh† diag(sigma²) Vh
        # Eigenvectors of TCC = rows of Vh (= columns of V)
        # Use thin SVD since n_src is typically << n_freq
        n_sv = min(n_src, self.n_kernels)
        # scipy.linalg.svd is more efficient for rectangular matrices,
        # but numpy.linalg.svd works fine here
        try:
            from scipy.linalg import svd as scipy_svd
            _, sigma, Vh = scipy_svd(B, full_matrices=False, check_finite=False)
        except ImportError:
            _, sigma, Vh = np.linalg.svd(B, full_matrices=False)

        # Store all eigenvalues for accurate energy fraction computation.
        self._all_eigenvalues = sigma**2

        # Keep top n_kernels singular values/vectors.
        # SVD already returns sigma in descending order — no sort needed.
        n_keep = min(n_sv, len(sigma))
        sigma_top = sigma[:n_keep]
        Vh_top = Vh[:n_keep, :]  # shape (n_keep, n_freq)

        self.eigenvalues = sigma_top**2                           # λ_i = σ_i²
        self.eigenkernels = Vh_top.reshape(n_keep, N, N)          # φ_i(fx,fy)

    def aerial_image(self, mask_transmission: np.ndarray,
                     normalize: bool = True) -> np.ndarray:
        """
        Compute aerial image from precomputed SOCS kernels.

        I(x,y) = Σ_i λ_i |IFFT[M(k) * φ_i(k)]|²

        Args:
            mask_transmission: 2D complex mask t(x,y), shape (N,N)
            normalize: if True, normalize result to [0,1]
        Returns:
            2D float intensity array, shape (N,N)
        """
        if self.eigenkernels is None:
            raise RuntimeError("Call compute(source) before aerial_image()")

        N = self.grid_size
        if mask_transmission.shape != (N, N):
            raise ValueError(
                "Mask must be {}x{}, got {}".format(N, N, mask_transmission.shape))

        M = np.fft.fft2(mask_transmission)
        I_total = np.zeros((N, N), dtype=np.float64)

        for lam, phi in zip(self.eigenvalues, self.eigenkernels):
            if lam < 1e-12 * self.eigenvalues[0]:
                break  # skip negligible kernels
            E = np.fft.ifft2(M * phi)
            I_total += lam * np.abs(E)**2

        if normalize:
            I_max = np.max(I_total)
            if I_max > 0:
                I_total /= I_max

        return I_total

    @property
    def n_kernels_used(self) -> int:
        """Number of eigenkernels stored after compute()."""
        if self.eigenkernels is None:
            return 0
        return self.eigenkernels.shape[0]

    def eigenvalue_energy(self) -> float:
        """Fraction of TCC energy captured by stored kernels (0-1)."""
        if self.eigenvalues is None:
            return 0.0
        if not hasattr(self, '_all_eigenvalues'):
            return 1.0
        kept_energy = float(np.sum(self.eigenvalues))
        total_energy = float(np.sum(self._all_eigenvalues))
        return kept_energy / total_energy if total_energy > 0 else 1.0
