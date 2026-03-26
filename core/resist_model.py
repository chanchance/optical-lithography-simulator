"""
Resist models: Threshold, Dill (A,B,C), Chemically Amplified (CA).
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from abc import ABC, abstractmethod


class BaseResist(ABC):
    @abstractmethod
    def expose(self, aerial_image: np.ndarray, dose: float = 1.0) -> np.ndarray:
        """Return latent image (0=unexposed, 1=fully exposed)."""

    @abstractmethod
    def develop(self, latent_image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary resist pattern (1=resist remaining)."""


@dataclass
class ThresholdResist(BaseResist):
    """Simple threshold model (existing behavior)."""
    threshold: float = 0.30

    def expose(self, aerial_image, dose=1.0):
        return aerial_image * dose

    def develop(self, latent_image, threshold=None):
        t = threshold if threshold is not None else self.threshold
        return (latent_image < t).astype(float)


@dataclass
class DillResist(BaseResist):
    """Dill A, B, C exposure model.
    dM/dE = -A * M * I  (M = PAC concentration, I = intensity)
    A: bleachable absorption [1/mJ/cm2]
    B: non-bleachable absorption [1/cm]
    C: exposure rate constant [cm2/mJ]
    """
    A: float = 0.8    # typical positive resist
    B: float = 0.1
    C: float = 1.0    # normalized exposure rate (dose=1.0 → ~45% bleaching at full intensity)
    peb_sigma_nm: float = 30.0  # PEB diffusion sigma
    domain_size_nm: float = 2000.0

    def expose(self, aerial_image, dose=1.0):
        # M(r) after exposure: M = exp(-A*C*dose*I) approximately
        # Full model: integrate dM/dt = -A*M*I*C over exposure time
        exposure = aerial_image * dose * self.C
        M = np.exp(-self.A * exposure)  # PAC concentration
        return M

    def _peb_diffusion(self, M):
        if self.peb_sigma_nm <= 0:
            return M
        px_per_nm = len(M) / self.domain_size_nm
        sigma_px = self.peb_sigma_nm * px_per_nm
        return gaussian_filter(M, sigma=sigma_px)

    def develop(self, latent_image, threshold=0.5):
        M_peb = self._peb_diffusion(latent_image)
        return (M_peb > threshold).astype(float)


@dataclass
class CAResist(BaseResist):
    """Chemically Amplified Resist model.
    Steps: photon absorption → acid generation → PEB diffusion → deprotection → develop
    """
    quantum_efficiency: float = 0.5   # acid generation per photon
    amplification: float = 50.0       # acid amplification factor
    peb_sigma_nm: float = 25.0        # PEB diffusion sigma
    exposure_threshold: float = 0.3   # development threshold on [H+]
    domain_size_nm: float = 2000.0

    def expose(self, aerial_image, dose=1.0):
        # Acid concentration proportional to aerial image intensity × dose
        acid = aerial_image * dose * self.quantum_efficiency
        return acid

    def develop(self, latent_image, threshold=None):
        # PEB diffusion of acid
        px_per_nm = len(latent_image) / self.domain_size_nm
        sigma_px = self.peb_sigma_nm * px_per_nm
        acid_diffused = gaussian_filter(latent_image, sigma=max(0.1, sigma_px))
        # Deprotection (logistic)
        deprotection = 1.0 / (1.0 + np.exp(-self.amplification * (acid_diffused - 0.3)))
        t = threshold if threshold is not None else self.exposure_threshold
        return (deprotection < t).astype(float)


@dataclass
class ResistProfile3D:
    """Compute 3D resist exposure profile via z-slice aerial images."""
    resist: BaseResist
    n_z_slices: int = 10
    resist_thickness_nm: float = 100.0

    def compute_profile(self, aerial_image_stack: np.ndarray, dose: float = 1.0) -> np.ndarray:
        """
        aerial_image_stack: shape (n_z_slices, N, N) — images at different z depths
        Returns: binary 3D resist pattern (n_z_slices, N, N)
        """
        profiles = []
        for z_img in aerial_image_stack:
            latent = self.resist.expose(z_img, dose)
            binary = self.resist.develop(latent)
            profiles.append(binary)
        return np.stack(profiles, axis=0)


def create_resist(config: dict) -> BaseResist:
    """Factory: create resist from config dict."""
    resist_cfg = config.get('resist', {})
    model = resist_cfg.get('model', 'threshold')
    domain_nm = config.get('simulation', {}).get('domain_size_nm', 2000.0)

    if model == 'dill':
        return DillResist(
            A=resist_cfg.get('A', 0.8),
            B=resist_cfg.get('B', 0.1),
            C=resist_cfg.get('C', 0.01),
            peb_sigma_nm=resist_cfg.get('peb_sigma_nm', 30.0),
            domain_size_nm=domain_nm,
        )
    elif model == 'ca':
        return CAResist(
            quantum_efficiency=resist_cfg.get('quantum_efficiency', 0.5),
            amplification=resist_cfg.get('amplification', 50.0),
            peb_sigma_nm=resist_cfg.get('peb_sigma_nm', 25.0),
            domain_size_nm=domain_nm,
        )
    else:
        return ThresholdResist(threshold=resist_cfg.get('threshold', 0.30))
