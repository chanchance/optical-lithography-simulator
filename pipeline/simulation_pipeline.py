"""
End-to-end simulation pipeline for optical lithography simulator.
Orchestrates: GDS load → mask creation → source setup → aerial image → resist → analysis.
"""
import os
import warnings
import numpy as np
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class SimResult:
    """Container for simulation results."""
    aerial_image: Optional[np.ndarray] = None
    resist_image: Optional[np.ndarray] = None
    mask_grid: Optional[np.ndarray] = None
    source_points: Optional[np.ndarray] = None
    cd_nm: float = 0.0
    nils: float = 0.0
    contrast: float = 0.0
    dof_nm: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict = field(default_factory=dict)
    layout_path: str = ''
    status: str = 'pending'  # pending, running, complete, failed
    error_msg: str = ''
    near_field_applied: bool = False
    euv_mode: bool = False


class SimulationPipeline:
    """
    Full optical simulation pipeline.
    Connects all simulator components end-to-end.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._setup_paths()

    def _setup_paths(self):
        """Add simulation package to Python path."""
        import sys
        sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if sim_dir not in sys.path:
            sys.path.insert(0, sim_dir)

    def run(self, config: Dict, layout_path: Optional[str] = None,
            on_progress: Optional[Callable] = None,
            stop_fn: Optional[Callable[[], bool]] = None) -> SimResult:
        """
        Run full simulation pipeline.

        Args:
            config: Simulation config dict (from ParameterIO)
            layout_path: Path to GDS/OAS file (None=use test pattern)
            on_progress: Callback(step_name, percent) for progress updates
        Returns:
            SimResult with aerial image and metrics
        """
        from fileio.config_validator import ConfigValidator
        ConfigValidator().validate_or_raise(config)

        result = SimResult(config=config, layout_path=layout_path or '')
        result.status = 'running'

        def progress(step, pct):
            if on_progress:
                on_progress(step, pct)

        def is_stopped():
            return stop_fn is not None and stop_fn()

        try:
            # Step 1: Load layout or create test pattern
            progress('Loading layout', 5)
            if is_stopped():
                result.status = 'failed'
                result.error_msg = 'Stopped by user'
                return result
            mask_grid = self._step_load_layout(config, layout_path)

            # Step 1b: Optional RCWA near-field correction
            rcwa_cfg = config.get('rcwa', {})
            if rcwa_cfg.get('enabled', False):
                from core.rcwa import RCWAEngine, RCWAParams
                domain_nm = config.get('simulation', {}).get('domain_size_nm', 2000.0)
                wavelength_nm = config.get('lithography', {}).get('wavelength_nm', 193.0)
                rcwa_params = RCWAParams(
                    wavelength_nm=wavelength_nm,
                    n_orders=rcwa_cfg.get('n_orders', 11),
                )
                engine = RCWAEngine(rcwa_params)
                nf = engine.apply_to_mask_grid(mask_grid, domain_nm)
                mask_grid = nf  # keep complex; aerial image engine accepts complex masks
                result.near_field_applied = True

            # Store magnitude for display (imshow/contour require real arrays)
            result.mask_grid = np.abs(mask_grid) if np.iscomplexobj(mask_grid) else mask_grid
            litho_cfg = config.get('lithography', {})
            wl_check = float(litho_cfg.get('wavelength_nm', 193.0))
            result.euv_mode = abs(wl_check - 13.5) < 0.5
            progress('Layout loaded', 20)

            # Step 2: Create illumination source
            if is_stopped():
                result.status = 'failed'
                result.error_msg = 'Stopped by user'
                return result
            progress('Creating source', 25)
            source = self._step_create_source(config)
            result.source_points = np.array(
                [[p.kx, p.ky, p.weight] for p in source.get_source_points()]
            )
            progress('Source ready', 35)

            # Step 3: Compute aerial image
            if is_stopped():
                result.status = 'failed'
                result.error_msg = 'Stopped by user'
                return result
            progress('Computing aerial image', 40)
            aerial_image = self._step_compute_aerial_image(config, mask_grid, source, on_progress)
            result.aerial_image = aerial_image
            progress('Aerial image done', 75)

            # Step 4: Apply resist model
            if is_stopped():
                result.status = 'failed'
                result.error_msg = 'Stopped by user'
                return result
            progress('Applying resist model', 78)
            resist_image = self._step_apply_resist(config, aerial_image)
            result.resist_image = resist_image
            progress('Resist done', 80)

            # Step 5: Analyze
            if is_stopped():
                result.status = 'failed'
                result.error_msg = 'Stopped by user'
                return result
            progress('Analyzing', 85)
            metrics = self._step_analyze(config, aerial_image)
            result.cd_nm = metrics.get('cd_nm', 0.0)
            result.nils = metrics.get('nils', 0.0)
            result.contrast = metrics.get('contrast', 0.0)
            result.dof_nm = metrics.get('dof_nm', 0.0)
            result.metrics = metrics
            progress('Analysis done', 100)

            result.status = 'complete'

        except Exception as e:
            result.status = 'failed'
            result.error_msg = str(e)
            import logging
            logging.getLogger(__name__).exception("Simulation failed")

        return result

    def _step_load_layout(self, config: Dict,
                          layout_path: Optional[str]) -> np.ndarray:
        """Load GDS layout or create test pattern."""
        sim_cfg = config.get('simulation', {})
        N = sim_cfg.get('grid_size', 256)
        domain_nm = sim_cfg.get('domain_size_nm', 2000.0)

        if layout_path and os.path.exists(layout_path):
            try:
                from fileio.layout_io import read_layout, layout_to_mask_grid
                layout = read_layout(layout_path)
                mask_grid = layout_to_mask_grid(layout, N, domain_nm)
                return mask_grid
            except Exception as e:
                warnings.warn(
                    "Could not read layout '{}': {}. "
                    "Falling back to synthetic test pattern.".format(layout_path, e),
                    stacklevel=2)

        # Default: line/space test pattern
        from core.mask_model import MaskFactory
        mask = MaskFactory.create_test_pattern(
            'line_space', N, domain_nm, period_px=N // 4
        )
        mask_grid = np.abs(mask.transmission).astype(np.float64)

        # Apply EUV multilayer mask model when wavelength is 13.5nm
        litho_cfg = config.get('lithography', {})
        if abs(float(litho_cfg.get('wavelength_nm', 193.0)) - 13.5) < 0.5:
            from core.euv_mask import EUVMultilayerMask, EUVFlare
            euv_mask = EUVMultilayerMask()
            mask_grid = euv_mask.apply_to_mask(mask_grid)
            euv_cfg = config.get('euv', {})
            flare_frac = euv_cfg.get('flare', None)
            if flare_frac is not None:
                flare = EUVFlare(flare_fraction=flare_frac)
                mask_grid = flare.apply(mask_grid)

        return mask_grid

    def _step_create_source(self, config: Dict):
        """Create illumination source from config."""
        from core.source_model import create_source
        litho = config.get('lithography', {})
        params = {
            'NA': litho.get('NA', 0.93),
            'wavelength_nm': litho.get('wavelength_nm', 193.0),
            'illumination': litho.get('illumination', {'type': 'annular',
                                                        'sigma_outer': 0.85,
                                                        'sigma_inner': 0.55}),
        }
        return create_source(params)

    def _step_compute_aerial_image(self, config: Dict, mask_grid: np.ndarray,
                                   source, on_progress) -> np.ndarray:
        """Compute aerial image using Fourier optics."""
        from core.fourier_optics import FourierOpticsEngine

        litho = config.get('lithography', {})
        sim_cfg = config.get('simulation', {})

        engine_params = {
            'wavelength_nm': litho.get('wavelength_nm', 193.0),
            'NA': litho.get('NA', 0.93),
            'defocus_nm': litho.get('defocus_nm', 0.0),
            'grid_size': sim_cfg.get('grid_size', 256),
            'domain_size_nm': sim_cfg.get('domain_size_nm', 2000.0),
            'aberrations': litho.get('aberrations', {}),
            'use_hopkins': litho.get('use_hopkins', False),
            'n_kernels': litho.get('n_kernels', 10),
            'use_vector': litho.get('use_vector', False),
            'polarization': sim_cfg.get('polarization', litho.get('polarization', 'unpolarized')),
        }

        engine = FourierOpticsEngine(engine_params)
        # Convert binary grid to complex transmission
        mask_complex = mask_grid.astype(np.complex128)
        aerial_image = engine.compute_aerial_image(mask_complex, source)
        dose_factor = litho.get('dose_factor', 1.0)
        if dose_factor != 1.0:
            aerial_image = aerial_image * dose_factor
        return aerial_image

    def _step_apply_resist(self, config: Dict, aerial_image: np.ndarray) -> np.ndarray:
        """Apply resist model to aerial image, returning binary resist pattern."""
        from core.resist_model import create_resist
        resist = create_resist(config)
        dose = config.get('resist', {}).get('dose', 1.0)
        latent = resist.expose(aerial_image, dose)
        return resist.develop(latent)

    def _step_analyze(self, config: Dict, aerial_image: np.ndarray) -> Dict:
        """Analyze aerial image for CD, NILS, contrast metrics."""
        from analysis.aerial_image_analysis import AerialImageAnalyzer

        sim_cfg = config.get('simulation', {})
        N = sim_cfg.get('grid_size', 256)
        domain_nm = sim_cfg.get('domain_size_nm', 2000.0)
        threshold = config.get('analysis', {}).get('cd_threshold', 0.30)

        analyzer = AerialImageAnalyzer(domain_nm, N)
        metrics_obj = analyzer.analyze(aerial_image, threshold)

        # Simple NILS-based DOF estimate: DOF ≈ λ/(NA² * NILS) * k2
        # where k2 ≈ 0.5 is a process factor
        litho_cfg2 = config.get('lithography', {})
        wl = float(litho_cfg2.get('wavelength_nm', 193.0))
        na = float(litho_cfg2.get('NA', 0.93))
        nils = metrics_obj.nils
        if nils > 0.1 and na > 0:
            dof_nm = 0.5 * wl / (na ** 2) * min(nils / 2.0, 2.0)
        else:
            dof_nm = 0.0

        return {
            'cd_nm': metrics_obj.cd_nm,
            'nils': metrics_obj.nils,
            'contrast': metrics_obj.contrast,
            'i_max': metrics_obj.i_max,
            'i_min': metrics_obj.i_min,
            'threshold': threshold,
            'dof_nm': dof_nm,
        }

    def run_from_args(self, args: List[str]) -> SimResult:
        """Run simulation from command-line arguments."""
        import argparse
        parser = argparse.ArgumentParser(description='Optical Lithography Simulator')
        parser.add_argument('--gds', help='GDS/OAS layout file')
        parser.add_argument('--config', help='Config YAML file')
        parser.add_argument('--wavelength', type=float, default=193.0)
        parser.add_argument('--na', type=float, default=0.93)
        parser.add_argument('--sigma', type=float, default=0.85)
        parser.add_argument('--defocus', type=float, default=0.0)
        parser.add_argument('--output', default='./results')

        parsed = parser.parse_args(args)

        if parsed.config:
            from fileio.parameter_io import load_config
            config = load_config(parsed.config)
        else:
            config = {
                'lithography': {
                    'wavelength_nm': parsed.wavelength,
                    'NA': parsed.na,
                    'illumination': {'type': 'annular', 'sigma_outer': parsed.sigma,
                                     'sigma_inner': parsed.sigma * 0.6},
                    'defocus_nm': parsed.defocus,
                },
                'simulation': {'grid_size': 256, 'domain_size_nm': 2000.0}
            }

        return self.run(config, parsed.gds,
                        on_progress=lambda s, p: print('[{:.0f}%] {}'.format(p, s)))
