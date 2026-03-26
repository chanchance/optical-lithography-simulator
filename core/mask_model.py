"""
Mask models for optical lithography simulation.
Based on Pistor 2001 Chapter 4.7: Photomask Models and Scattering Coefficient Calculation.

Three mask models (in increasing complexity):
1. ThinScalarMask: 2D transmission function, scalar, constant scattering (Section 4.7.2.1)
2. ThickVectorMask: vector, constant scattering coefficient (Section 4.7.2.2)
3. (Future) ThickVectorNonConstantMask: angle-dependent scattering (Section 4.7.2.3)
"""
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ScatteringCoefficient:
    """Represents a scattered diffraction order amplitude."""
    order_m: int
    order_n: int
    amplitude_x: complex
    amplitude_y: complex
    amplitude_z: complex


class ThinScalarMask:
    """
    Thin mask model with scalar transmission function (Pistor Section 4.7.2.1).

    Assumes:
    - Mask is infinitely thin (no topography effects)
    - Scalar transmission: t(x,y) is a complex scalar
    - Constant scattering coefficients (angle-independent)

    Mask types:
    - Binary: t = 0 (opaque) or 1 (clear)
    - AttPSM: t = 0 (opaque) or sqrt(att)*exp(j*pi) (attenuated phase shift)
    - AltPSM: t = 1 or -1 (alternating phase shift, phase = 0 or pi)
    """

    def __init__(self, grid_size: int, domain_size_nm: float,
                 mask_type: str = 'binary',
                 attenuated_transmission: float = 0.06,
                 phase_shift_deg: float = 180.0):
        """
        Args:
            grid_size: NxN grid size
            domain_size_nm: Physical domain size in nm
            mask_type: 'binary', 'attPSM', or 'altPSM'
            attenuated_transmission: Transmission of attenuated region (for attPSM)
            phase_shift_deg: Phase shift angle in degrees
        """
        self.grid_size = grid_size
        self.domain_size_nm = domain_size_nm
        self.mask_type = mask_type.lower()
        self.att_transmission = attenuated_transmission
        self.phase_shift = np.radians(phase_shift_deg)

        # Transmission grid (complex)
        self._transmission = np.ones((grid_size, grid_size), dtype=np.complex128)

    @property
    def transmission(self) -> np.ndarray:
        """Get the complex transmission grid."""
        return self._transmission

    def set_binary(self, pattern: np.ndarray) -> None:
        """
        Set transmission from binary pattern.
        Args:
            pattern: 2D array where 1=clear, 0=opaque
        """
        pattern = np.asarray(pattern, dtype=np.float64)
        if self.mask_type == 'binary':
            self._transmission = pattern.astype(np.complex128)
        elif self.mask_type in ('attpsm', 'att_psm'):
            # Opaque regions get attenuated PSM transmission
            att_phasor = np.sqrt(self.att_transmission) * np.exp(1j * self.phase_shift)
            self._transmission = np.where(pattern > 0.5,
                                          1.0 + 0j,
                                          att_phasor)
        elif self.mask_type in ('altpsm', 'alt_psm'):
            # Alternating PSM: clear regions alternate phase
            self._transmission = np.where(pattern > 0.5,
                                          np.exp(1j * self.phase_shift),
                                          0.0 + 0j)
        else:
            self._transmission = pattern.astype(np.complex128)

    def from_gds_polygons(self, polygons: List[np.ndarray],
                          layer_polarity: str = 'dark') -> None:
        """
        Build transmission grid from list of GDS polygon coordinate arrays.

        Args:
            polygons: List of (N,2) coordinate arrays in nm
            layer_polarity: 'dark' (polygons are opaque) or 'clear' (polygons are transparent)
        """
        N = self.grid_size
        dx = self.domain_size_nm / N

        # Start with background
        if layer_polarity == 'dark':
            grid = np.ones((N, N), dtype=np.float64)  # Clear background
        else:
            grid = np.zeros((N, N), dtype=np.float64)  # Dark background

        for poly_coords in polygons:
            if len(poly_coords) < 3:
                continue
            mask_fill = self._rasterize_polygon(poly_coords, N, dx)
            if layer_polarity == 'dark':
                grid[mask_fill] = 0.0  # Opaque polygon
            else:
                grid[mask_fill] = 1.0  # Clear polygon

        self.set_binary(grid)

    def _rasterize_polygon(self, coords: np.ndarray, N: int, dx: float) -> np.ndarray:
        """
        Rasterize a polygon to a binary grid using scanline fill.
        coords: (M,2) array of (x,y) in nm
        Returns boolean mask array.
        """
        # Convert nm coords to grid indices
        ix = (coords[:, 0] / dx).astype(np.float64)
        iy = (coords[:, 1] / dx).astype(np.float64)

        # Clip to grid
        ix = np.clip(ix, 0, N-1)
        iy = np.clip(iy, 0, N-1)

        # Use matplotlib path for robust polygon rasterization
        try:
            from matplotlib.path import Path
            path = Path(np.column_stack([ix, iy]))

            yi_arr, xi_arr = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
            points = np.column_stack([xi_arr.ravel(), yi_arr.ravel()])
            mask = path.contains_points(points).reshape(N, N).T
            return mask
        except Exception:
            # Fallback: bounding box fill
            xmin, xmax = int(np.min(ix)), int(np.max(ix))
            ymin, ymax = int(np.min(iy)), int(np.max(iy))
            mask = np.zeros((N, N), dtype=bool)
            mask[xmin:xmax+1, ymin:ymax+1] = True
            return mask

    def get_diffraction_orders(self, n_orders: int = 10) -> Dict[Tuple[int,int], complex]:
        """
        Compute diffraction orders via 2D FFT of transmission function.
        Returns dict mapping (m, n) -> complex amplitude.
        """
        T = np.fft.fft2(self._transmission) / (self.grid_size ** 2)
        orders = {}
        for m in range(-n_orders, n_orders+1):
            for n in range(-n_orders, n_orders+1):
                orders[(m, n)] = T[m % self.grid_size, n % self.grid_size]
        return orders

    def apply_bias(self, bias_nm: float) -> 'ThinScalarMask':
        """
        Apply CD bias by morphological dilation/erosion.
        Positive bias: expand clear features (erosion of opaque).
        """
        from scipy import ndimage

        binary = np.abs(self._transmission) > 0.5

        if bias_nm != 0:
            n_pixels = abs(bias_nm) / (self.domain_size_nm / self.grid_size)
            n_pixels = max(1, int(round(n_pixels)))
            struct = ndimage.generate_binary_structure(2, 1)

            if bias_nm > 0:
                binary = ndimage.binary_dilation(binary, struct, iterations=n_pixels)
            else:
                binary = ndimage.binary_erosion(binary, struct, iterations=n_pixels)

        new_mask = ThinScalarMask(self.grid_size, self.domain_size_nm,
                                  self.mask_type, self.att_transmission,
                                  np.degrees(self.phase_shift))
        new_mask.set_binary(binary.astype(float))
        return new_mask


class ThickVectorMask:
    """
    Thick mask model with constant vector scattering coefficients.
    (Pistor 2001 Section 4.7.2.2)

    Accounts for mask topography effects by computing scattering matrix
    relating incident plane wave components to scattered diffraction orders.
    For each incident direction (kx_i, ky_i), applies pre-computed or
    analytical scattering coefficients per diffraction order.
    """

    def __init__(self, thin_mask: ThinScalarMask,
                 absorber_thickness_nm: float = 80.0,
                 absorber_n: complex = 0.88 + 1.82j,  # Typical EUV absorber
                 substrate_n: float = 1.5):
        """
        Args:
            thin_mask: Underlying thin mask for geometry
            absorber_thickness_nm: Mask absorber thickness
            absorber_n: Complex refractive index of absorber
            substrate_n: Substrate refractive index
        """
        self.thin_mask = thin_mask
        self.absorber_thickness_nm = absorber_thickness_nm
        self.absorber_n = absorber_n
        self.substrate_n = substrate_n
        self._scattering_cache = {}

    def compute_scattered_field(self, kx_in: float, ky_in: float,
                                wavelength_nm: float) -> Dict[Tuple[int,int], np.ndarray]:
        """
        Compute scattered field orders for incident plane wave.

        Args:
            kx_in, ky_in: Incident k-vector (normalized by NA/lambda)
            wavelength_nm: Wavelength in nm
        Returns:
            Dict mapping diffraction order (m,n) to complex 3-vector [Ex, Ey, Ez]
        """
        cache_key = (round(kx_in, 4), round(ky_in, 4))
        if cache_key in self._scattering_cache:
            return self._scattering_cache[cache_key]

        # Get thin mask diffraction orders as baseline
        orders = self.thin_mask.get_diffraction_orders(n_orders=5)

        # Apply absorption correction for thick mask:
        # A(d) = exp(-2*pi*k_im*d/lambda) where k_im = Im(n_absorber)
        k_im = np.imag(self.absorber_n)
        att_factor = np.exp(-2.0 * np.pi * k_im * self.absorber_thickness_nm / wavelength_nm)

        # Apply phase correction for thick mask:
        # Phase accumulation through absorber thickness
        k_re = np.real(self.absorber_n)
        phase_factor = np.exp(1j * 2.0 * np.pi * k_re * self.absorber_thickness_nm / wavelength_nm)

        scattered = {}
        for (m, n), amp in orders.items():
            # Apply thin-to-thick correction factor
            thick_correction = att_factor * phase_factor

            # Vectorize: assume TE polarization for simplicity
            # Full vector treatment requires RCWA or FDTD
            scattered_amp = amp * thick_correction

            # Convert scalar to 3-vector (TE polarization: E in x for normal incidence)
            e_vec = np.array([scattered_amp, 0.0 + 0j, 0.0 + 0j])
            scattered[(m, n)] = e_vec

        self._scattering_cache[cache_key] = scattered
        return scattered


class MaskFactory:
    """Factory for creating mask models from configuration."""

    @staticmethod
    def create(mask_type: str, grid_size: int, domain_size_nm: float,
               **kwargs) -> ThinScalarMask:
        """
        Create mask model from type string.

        Args:
            mask_type: 'binary', 'attPSM', 'altPSM'
            grid_size: Simulation grid size N
            domain_size_nm: Physical domain size
            **kwargs: Additional mask parameters
        """
        return ThinScalarMask(
            grid_size=grid_size,
            domain_size_nm=domain_size_nm,
            mask_type=mask_type,
            attenuated_transmission=kwargs.get('attenuated_transmission', 0.06),
            phase_shift_deg=kwargs.get('phase_shift_deg', 180.0)
        )

    @staticmethod
    def create_test_pattern(pattern_type: str, grid_size: int,
                             domain_size_nm: float, **kwargs) -> ThinScalarMask:
        """
        Create standard test patterns for simulation.

        pattern_type: 'line_space', 'contact_hole', 'isolated_line', 'checkerboard'
        """
        mask = MaskFactory.create('binary', grid_size, domain_size_nm)
        N = grid_size

        if pattern_type == 'line_space':
            # Alternating lines and spaces
            period = kwargs.get('period_px', N // 4)
            half_period = max(1, period // 2)
            grid = np.zeros((N, N))
            for i in range(N):
                if (i // half_period) % 2 == 0:
                    grid[i, :] = 1.0
            mask.set_binary(grid)

        elif pattern_type == 'contact_hole':
            # Square contact hole in dark field
            grid = np.zeros((N, N))
            size = kwargs.get('hole_size_px', N // 8)
            cx, cy = N // 2, N // 2
            grid[cx-size//2:cx+size//2, cy-size//2:cy+size//2] = 1.0
            mask.set_binary(grid)

        elif pattern_type == 'isolated_line':
            # Single horizontal line
            grid = np.zeros((N, N))
            width = kwargs.get('line_width_px', N // 16)
            cy = N // 2
            grid[:, cy-width//2:cy+width//2] = 1.0
            mask.set_binary(grid)

        elif pattern_type == 'checkerboard':
            # Checkerboard pattern
            period = kwargs.get('period_px', N // 8)
            xi = np.arange(N)
            yi = np.arange(N)
            XI, YI = np.meshgrid(xi, yi, indexing='ij')
            grid = (((XI // period) + (YI // period)) % 2).astype(float)
            mask.set_binary(grid)

        else:
            # Default: fully clear
            mask.set_binary(np.ones((N, N)))

        return mask
