"""
EUV lithography mask models:
- EUVMultilayerMask: Mo/Si multilayer reflective mask (13.5nm)
- EUVStochasticModel: shot noise, LER/LWR calculation
- EUVFlare: long-range scattering background
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Mo/Si multilayer constants at 13.5nm
MO_N, MO_K = 0.9234, 0.0064
SI_N, SI_K = 0.9993, 0.0018
TAN_N, TAN_K = 0.9428, 0.0412   # TaN absorber

@dataclass
class EUVMultilayerMask:
    """
    EUV reflective mask with Mo/Si multilayer mirror.
    Standard: 40 bilayer pairs, Mo 2.8nm / Si 4.1nm = 6.9nm pitch.
    """
    n_bilayers: int = 40
    mo_thickness_nm: float = 2.8
    si_thickness_nm: float = 4.1
    wavelength_nm: float = 13.5
    # Absorber on top (TaN or Cr)
    absorber_material: str = 'tan'
    absorber_thickness_nm: float = 56.0  # typical TaN thickness
    # Capping layer (Ru)
    cap_n: float = 0.886
    cap_k: float = 0.017
    cap_thickness_nm: float = 2.5

    def _fresnel_r(self, n1: complex, n2: complex, angle_rad: float = 0.0, pol: str = 'te') -> complex:
        """Fresnel reflection coefficient at interface."""
        cos1 = np.cos(angle_rad)
        sin1 = np.sin(angle_rad)
        # Snell's law
        sin2 = n1 * sin1 / n2
        cos2 = np.sqrt(1 - sin2**2 + 0j)
        if pol == 'te':
            return (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
        else:  # tm
            return (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)

    def multilayer_reflectance(self, angle_deg: float = 6.0) -> float:
        """
        Compute peak reflectance of Mo/Si multilayer stack.
        Uses recursive Fresnel formula (transfer matrix method).
        angle_deg: chief ray incidence angle (typically 6° for EUV tools)
        """
        theta = np.radians(angle_deg)
        lam = self.wavelength_nm

        # Start from substrate (Si), build up through bilayers
        # Use exp(-iωt) convention: n = n_r + i*n_i with n_i > 0 for absorption
        # so that exp(2j*phi) decays (|exp(2j*phi)| < 1) in absorbing media
        n_mo = complex(MO_N, MO_K)
        n_si = complex(SI_N, SI_K)

        # Phase thicknesses
        def phase(n, d, theta_in):
            cos_t = np.sqrt(1 - (np.sin(theta_in) / n)**2 + 0j)
            return 2 * np.pi * n * d * cos_t / lam

        # Initialize reflection from substrate
        r = complex(0, 0)

        # Recursive reflection from bilayer stack
        for _ in range(self.n_bilayers):
            # Si layer
            phi_si = phase(n_si, self.si_thickness_nm, theta)
            r_si_mo = self._fresnel_r(n_si, n_mo, theta)
            r = (r_si_mo + r * np.exp(2j * phi_si)) / (1 + r_si_mo * r * np.exp(2j * phi_si))
            # Mo layer
            phi_mo = phase(n_mo, self.mo_thickness_nm, theta)
            r_mo_si = self._fresnel_r(n_mo, n_si, theta)
            r = (r_mo_si + r * np.exp(2j * phi_mo)) / (1 + r_mo_si * r * np.exp(2j * phi_mo))

        return float(np.abs(r)**2)

    def apply_to_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply EUV mask model to binary mask.
        0 = absorber region (low reflectance ~2%)
        1 = open (multilayer) region (peak reflectance ~70%)
        """
        R_peak = self.multilayer_reflectance()
        # Absorber transmission (two-pass): approximately exp(-4*pi*k*t/lambda)
        k_abs = TAN_K
        T_abs = np.exp(-4 * np.pi * k_abs * self.absorber_thickness_nm / self.wavelength_nm)
        R_absorber = R_peak * T_abs**2

        return R_absorber + (R_peak - R_absorber) * binary_mask


@dataclass
class EUVStochasticModel:
    """
    EUV stochastic effects: photon shot noise → LER/LWR.
    Low photon flux at EUV causes significant shot noise.
    """
    photons_per_nm2: float = 20.0   # typical EUV dose in photons/nm^2
    domain_size_nm: float = 2000.0

    def add_shot_noise(self, aerial_image: np.ndarray, dose_factor: float = 1.0) -> np.ndarray:
        """Add Poisson shot noise to aerial image."""
        N = aerial_image.shape[0]
        nm_per_pixel = self.domain_size_nm / N
        photons_per_pixel = self.photons_per_nm2 * nm_per_pixel**2 * dose_factor
        if photons_per_pixel <= 0:
            return aerial_image.copy()
        # Scale image to photon counts, apply Poisson noise, normalize back
        counts = aerial_image * photons_per_pixel
        noisy_counts = np.random.poisson(np.maximum(0, counts).astype(float))
        return noisy_counts / photons_per_pixel

    def compute_ler(self, edge_positions_nm: np.ndarray) -> float:
        """Compute LER (3-sigma) from edge position array."""
        return 3.0 * float(np.std(edge_positions_nm))

    def extract_edge_positions(self, binary_image: np.ndarray, domain_size_nm: float) -> np.ndarray:
        """Extract left edge positions (nm) row by row from binary image."""
        N = binary_image.shape[0]
        nm_per_pixel = domain_size_nm / N
        edges = []
        for row in binary_image:
            transitions = np.where(np.diff(row.astype(int)) > 0)[0]
            if len(transitions) > 0:
                edges.append(transitions[0] * nm_per_pixel)
        return np.array(edges) if edges else np.array([0.0])


@dataclass
class EUVFlare:
    """Long-range scattering flare in EUV systems."""
    flare_fraction: float = 0.05  # 5% background flare

    def apply(self, aerial_image: np.ndarray) -> np.ndarray:
        """Add uniform background flare."""
        mean_intensity = float(np.mean(aerial_image))
        return aerial_image * (1 - self.flare_fraction) + mean_intensity * self.flare_fraction
