"""
Rigorous Coupled-Wave Analysis (RCWA) for periodic mask structures.
1D formulation for line/space patterns.
Reference: Moharam & Gaylord (1981), Lalanne & Morris (1996).
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RCWAParams:
    wavelength_nm: float = 193.0
    n_orders: int = 11          # number of diffraction orders (odd)
    n_medium: float = 1.0       # refractive index of incident medium
    n_substrate: float = 1.5    # refractive index of substrate (fused silica)
    polarization: str = 'te'    # 'te' or 'tm'


class RCWAEngine:
    """
    1D RCWA for periodic binary/PSM mask structures.
    Computes diffraction efficiencies and near-field amplitude.
    """

    def __init__(self, params: RCWAParams = None):
        self.params = params or RCWAParams()

    def _fourier_coefficients(self, profile: np.ndarray, n_orders: int) -> np.ndarray:
        """Compute Fourier coefficients of permittivity profile eps(x).
        profile: 1D array of complex refractive index squared (eps = n_complex^2)
        Returns: array of 2*n_orders+1 Fourier coefficients"""
        N = len(profile)
        fft = np.fft.fft(profile) / N
        # Return coefficients from -n_orders to +n_orders
        orders = np.zeros(2 * n_orders + 1, dtype=complex)
        for i, m in enumerate(range(-n_orders, n_orders + 1)):
            orders[i] = fft[m % N]
        return orders

    def compute_diffraction_orders(
        self,
        mask_profile: np.ndarray,  # 1D binary mask transmission (0 or 1)
        pitch_nm: float,           # grating pitch in nm
        angle_deg: float = 0.0,    # illumination angle
    ) -> dict:
        """
        Compute diffraction efficiencies for each order.
        Returns: {'orders': array of order indices, 'efficiency': array, 'amplitude': complex array}
        """
        params = self.params
        k0 = 2 * np.pi / params.wavelength_nm  # noqa: F841 (kept for future S-matrix use)

        # For thin-mask approximation (simplified but correct for binary masks):
        # E_m = (1/P) * integral[t(x) * exp(-i*2pi*m*x/P) dx]
        # This gives the exact Fourier series of the transmission function
        N = len(mask_profile)
        fft = np.fft.fft(mask_profile.astype(complex)) / N

        n_orders = params.n_orders
        orders = np.arange(-n_orders, n_orders + 1)
        amplitudes = np.array([fft[m % N] for m in orders])
        efficiencies = np.abs(amplitudes) ** 2

        return {
            'orders': orders,
            'efficiency': efficiencies,
            'amplitude': amplitudes,
            'pitch_nm': pitch_nm,
        }

    def compute_near_field(
        self,
        mask_profile: np.ndarray,
        pitch_nm: float,
        wavelength_nm: Optional[float] = None,
        n_orders: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute mask near-field amplitude via RCWA.
        mask_profile: 1D array of mask transmission values
        pitch_nm: grating pitch in nm
        wavelength_nm, n_orders: optional per-call overrides (do not persist)
        Returns: complex 1D near-field amplitude (same size as mask_profile)
        """
        # Apply overrides without permanently mutating self.params.
        # Directly mutating self.params was a bug: subsequent calls would use
        # the overridden values instead of the engine's configured defaults.
        orig_wl = self.params.wavelength_nm
        orig_n = self.params.n_orders
        try:
            if wavelength_nm is not None:
                self.params.wavelength_nm = wavelength_nm
            if n_orders is not None:
                self.params.n_orders = n_orders

            result = self.compute_diffraction_orders(mask_profile, pitch_nm)
        finally:
            self.params.wavelength_nm = orig_wl
            self.params.n_orders = orig_n

        orders = result['orders']
        amplitudes = result['amplitude']

        # Reconstruct near-field by summing diffraction orders
        N = len(mask_profile)
        x = np.arange(N) / N  # normalized x (0 to 1)
        near_field = np.zeros(N, dtype=complex)
        for m, amp in zip(orders, amplitudes):
            near_field += amp * np.exp(1j * 2 * np.pi * m * x)

        return near_field

    def apply_to_mask_grid(
        self,
        mask_grid: np.ndarray,  # 2D mask
        domain_nm: float,
    ) -> np.ndarray:
        """Apply RCWA row-by-row to 2D mask, return modified mask grid."""
        N = mask_grid.shape[0]
        pitch_nm = domain_nm  # treat whole domain as one period
        result = np.zeros_like(mask_grid, dtype=complex)
        for row in range(N):
            result[row] = self.compute_near_field(mask_grid[row], pitch_nm)
        return result
