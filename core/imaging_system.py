"""
Full optical imaging system model.
Connects source → mask → projection optic → aerial image.
Based on Pistor 2001 Chapter 4 (Figure 4-1, 4-2).
"""
import numpy as np
from typing import Optional, Callable, Dict

from .source_model import create_source, BaseSource
from .fourier_optics import FourierOpticsEngine
from .fdtd_engine import FDTDSimulator


class ImagingSystem:
    """
    Complete lithography imaging system model (Pistor 2001, Section 4.1.1).

    System components (Figure 4-1):
    Source + Illumination Optic → Photomask → Projection Optic → Film Stack → Aerial Image
    """

    def __init__(self, config: Dict):
        self.config = config
        litho = config.get('lithography', config)
        sim_cfg = config.get('simulation', {})

        # Flatten config for engines
        self._params = {
            'wavelength_nm': litho.get('wavelength_nm', 193.0),
            'NA': litho.get('NA', 0.93),
            'defocus_nm': litho.get('defocus_nm', 0.0),
            'grid_size': sim_cfg.get('grid_size', 256),
            'domain_size_nm': sim_cfg.get('domain_size_nm', 2000.0),
            'illumination': litho.get('illumination', {}),
            'aberrations': litho.get('aberrations', {}),
        }

        self._source = None
        self._fourier_engine = None
        self._fdtd_engine = None

    def _get_source(self) -> BaseSource:
        if self._source is None:
            # create_source() expects 'illumination' as a nested dict, plus
            # top-level 'NA' and 'wavelength_nm'.  Pass params directly —
            # they already have the correct structure.
            self._source = create_source(self._params)
        return self._source

    def _get_fourier_engine(self) -> FourierOpticsEngine:
        if self._fourier_engine is None:
            self._fourier_engine = FourierOpticsEngine(self._params)
        return self._fourier_engine

    def compute(self, mask_transmission: np.ndarray,
                mode: str = 'fourier_optics',
                on_progress: Optional[Callable] = None) -> np.ndarray:
        """
        Compute aerial image for given mask transmission.

        Args:
            mask_transmission: 2D array of complex transmission values
            mode: 'fourier_optics' (fast) or 'fdtd' (rigorous)
            on_progress: optional callback(fraction) for progress reporting
        Returns:
            2D float array of normalized intensity
        """
        if mode == 'fourier_optics':
            source = self._get_source()
            engine = self._get_fourier_engine()
            return engine.compute_aerial_image(mask_transmission, source)
        elif mode == 'fdtd':
            return self._run_fdtd(mask_transmission, on_progress)
        else:
            raise ValueError("Unknown simulation mode: {}".format(mode))

    def _run_fdtd(self, mask: np.ndarray,
                  on_progress: Optional[Callable]) -> np.ndarray:
        """Run FDTD simulation for rigorous EM calculation."""
        fdtd_cfg = self.config.get('simulation', {}).get('fdtd', {})
        fdtd_cfg['wavelength_nm'] = self._params['wavelength_nm']

        N = self._params['grid_size']
        sim = FDTDSimulator(fdtd_cfg)

        nz = 32  # Thin 2D simulation
        sim.initialize(N, N, nz)

        # Set material properties from mask
        # Opaque regions: high sigma (absorbing)
        eps_r = np.ones((N, N, nz))
        sigma = np.zeros((N, N, nz))
        for iz in range(nz // 4, nz // 2):
            # Mask layer
            sigma[:, :, iz] = np.clip(1.0 - np.abs(mask), 0.0, 1.0) * 1e6  # Absorber conductivity

        sim.grid.eps_r = eps_r
        sim.grid.sigma = sigma
        sim._precompute_update_coefficients()

        fields = sim.run_simulation(on_progress)

        # Extract intensity at image plane
        I = sim.get_intensity(fields)
        I_slice = I[:, :, nz // 2].copy()
        I_max = np.max(I_slice)
        if I_max > 0:
            I_slice /= I_max
        return I_slice

    def get_source_points_diagram(self) -> np.ndarray:
        """
        Get k-space diagram of source points for visualization.
        Returns (N, 3) array of [kx, ky, weight] per source point.
        """
        source = self._get_source()
        pts = source.get_source_points()
        return np.array([[p.kx, p.ky, p.weight] for p in pts])

    def update_params(self, **kwargs) -> None:
        """Update simulation parameters and reset cached engines."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._source = None
        self._fourier_engine = None
        self._fdtd_engine = None
