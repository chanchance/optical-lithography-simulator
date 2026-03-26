"""
Parameter sweep engine for focus-exposure matrix and Bossung analysis.
Wraps SimulationPipeline to run NxM grid of (defocus, dose) combinations.
"""
import copy
import numpy as np
from typing import Callable, Optional


class ParameterSweep:
    def __init__(self, pipeline=None):
        from pipeline.simulation_pipeline import SimulationPipeline
        self.pipeline = pipeline or SimulationPipeline()

    def sweep_1d(
        self,
        base_config: dict,
        layout_path: Optional[str],
        param_path: str,       # e.g. "lithography.defocus_nm"
        values: list,
        on_progress: Optional[Callable] = None,
    ) -> list:
        """Run simulation for each value of a single parameter.
        param_path: dot-separated config key, e.g. "lithography.defocus_nm"
        Returns: list of SimResult objects"""
        results = []
        for i, val in enumerate(values):
            cfg = copy.deepcopy(base_config)
            self._set_nested(cfg, param_path, val)
            if on_progress:
                on_progress(f"Sweep {i+1}/{len(values)}: {param_path}={val}",
                            int(100 * i / len(values)))
            result = self.pipeline.run(cfg, layout_path)
            results.append(result)
        return results

    def sweep_2d(
        self,
        base_config: dict,
        layout_path: Optional[str],
        param1_path: str,   # e.g. "lithography.defocus_nm"
        param1_values: list,
        param2_path: str,   # e.g. "lithography.dose_factor"
        param2_values: list,
        on_progress: Optional[Callable] = None,
    ) -> np.ndarray:
        """Run NxM grid sweep. Returns 2D array of (cd_nm) values."""
        N, M = len(param1_values), len(param2_values)
        cd_matrix = np.zeros((N, M))
        total = N * M
        for i, v1 in enumerate(param1_values):
            for j, v2 in enumerate(param2_values):
                cfg = copy.deepcopy(base_config)
                self._set_nested(cfg, param1_path, v1)
                self._set_nested(cfg, param2_path, v2)
                if on_progress:
                    on_progress(f"FEM {i*M+j+1}/{total}", int(100*(i*M+j)/total))
                result = self.pipeline.run(cfg, layout_path)
                cd_matrix[i, j] = result.cd_nm
        return cd_matrix

    def _set_nested(self, config: dict, path: str, value):
        """Set a nested config key via dot-notation path."""
        keys = path.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def bossung_sweep(self, base_config, layout_path,
                      focus_range_nm=400, n_focus=17,
                      dose_factors=None) -> dict:
        """Convenience: run Bossung sweep. Returns {'focus_nm': arr, 'cd_by_dose': dict}.

        Note: cd_by_dose maps dose_factor -> list[cd_nm]. Values may be 0.0 for
        unresolved features at extreme defocus points. Callers should filter
        zero-CD points before fitting (e.g. parabolic/polyfit)."""
        if dose_factors is None:
            dose_factors = [0.9, 0.95, 1.0, 1.05, 1.1]
        focus_values = np.linspace(-focus_range_nm/2, focus_range_nm/2, n_focus)
        cd_by_dose = {}
        for dose in dose_factors:
            cfg = copy.deepcopy(base_config)
            self._set_nested(cfg, 'lithography.dose_factor', dose)
            results = self.sweep_1d(cfg, layout_path,
                                    'lithography.defocus_nm', focus_values.tolist())
            cd_by_dose[dose] = [r.cd_nm for r in results]
        return {'focus_nm': focus_values, 'cd_by_dose': cd_by_dose}
