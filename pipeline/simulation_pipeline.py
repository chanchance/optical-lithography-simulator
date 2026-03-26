"""
End-to-end simulation pipeline for optical lithography simulator.
Orchestrates: GDS load → mask creation → source setup → aerial image → analysis.
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
            result.mask_grid = mask_grid
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
            progress('Aerial image done', 80)

            # Step 4: Analyze
            if is_stopped():
                result.status = 'failed'
                result.error_msg = 'Stopped by user'
                return result
            progress('Analyzing', 85)
            metrics = self._step_analyze(config, aerial_image)
            result.cd_nm = metrics.get('cd_nm', 0.0)
            result.nils = metrics.get('nils', 0.0)
            result.contrast = metrics.get('contrast', 0.0)
            result.metrics = metrics
            progress('Analysis done', 100)

            result.status = 'complete'

        except Exception as e:
            result.status = 'failed'
            result.error_msg = str(e)
            import traceback
            traceback.print_exc()

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
        return np.abs(mask.transmission).astype(np.float64)

    def _step_create_source(self, config: Dict):
        """Create illumination source from config."""
        from core.source_model import create_source
        litho = config.get('lithography', config)
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

        litho = config.get('lithography', config)
        sim_cfg = config.get('simulation', {})

        engine_params = {
            'wavelength_nm': litho.get('wavelength_nm', 193.0),
            'NA': litho.get('NA', 0.93),
            'defocus_nm': litho.get('defocus_nm', 0.0),
            'grid_size': sim_cfg.get('grid_size', 256),
            'domain_size_nm': sim_cfg.get('domain_size_nm', 2000.0),
            'aberrations': litho.get('aberrations', {}),
        }

        engine = FourierOpticsEngine(engine_params)
        # Convert binary grid to complex transmission
        mask_complex = mask_grid.astype(np.complex128)
        return engine.compute_aerial_image(mask_complex, source)

    def _step_analyze(self, config: Dict, aerial_image: np.ndarray) -> Dict:
        """Analyze aerial image for CD, NILS, contrast metrics."""
        from analysis.aerial_image_analysis import AerialImageAnalyzer

        sim_cfg = config.get('simulation', {})
        N = sim_cfg.get('grid_size', 256)
        domain_nm = sim_cfg.get('domain_size_nm', 2000.0)
        threshold = config.get('analysis', {}).get('cd_threshold', 0.30)

        analyzer = AerialImageAnalyzer(domain_nm, N)
        metrics_obj = analyzer.analyze(aerial_image, threshold)

        return {
            'cd_nm': metrics_obj.cd_nm,
            'nils': metrics_obj.nils,
            'contrast': metrics_obj.contrast,
            'i_max': metrics_obj.i_max,
            'i_min': metrics_obj.i_min,
            'threshold': threshold,
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
