"""
Batch simulation runner for parameter sweeps.
Supports parallel execution via multiprocessing.
"""
import os
import csv
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from itertools import product


class BatchRunner:
    """Run parameter sweep simulations in batch mode."""

    def __init__(self, base_config: Dict, n_workers: int = 1):
        self.base_config = base_config
        self.n_workers = n_workers

    def run_sweep(self, sweep_params: Dict[str, List],
                  layout_path: Optional[str] = None,
                  on_progress: Optional[Callable] = None) -> List[Dict]:
        """
        Run simulation sweep over parameter combinations.

        Args:
            sweep_params: {param_name: [value1, value2, ...]}
                Supported: 'defocus_nm', 'NA', 'wavelength_nm', 'sigma_outer'
            layout_path: GDS/OAS file path
            on_progress: Callback(step, total, result)
        Returns:
            List of result dicts with params + metrics
        """
        import copy
        from pipeline.simulation_pipeline import SimulationPipeline

        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        combinations = list(product(*param_values))

        results = []
        pipeline = SimulationPipeline()

        for i, combo in enumerate(combinations):
            config = copy.deepcopy(self.base_config)

            # Apply parameter combination
            param_dict = dict(zip(param_names, combo))
            config = self._apply_params(config, param_dict)

            # Run simulation
            result = pipeline.run(config, layout_path)

            row = dict(param_dict)
            row['cd_nm'] = result.cd_nm
            row['nils'] = result.nils
            row['contrast'] = result.contrast
            row['status'] = result.status
            results.append(row)

            if on_progress:
                on_progress(i + 1, len(combinations), result)

        return results

    def _apply_params(self, config: Dict, params: Dict) -> Dict:
        """Apply flat parameter dict to nested config."""
        litho = config.setdefault('lithography', {})
        illum = litho.setdefault('illumination', {})

        for k, v in params.items():
            if k == 'defocus_nm':
                litho['defocus_nm'] = v
            elif k == 'NA':
                litho['NA'] = v
            elif k == 'wavelength_nm':
                litho['wavelength_nm'] = v
            elif k == 'sigma_outer':
                illum['sigma_outer'] = v
            elif k == 'sigma_inner':
                illum['sigma_inner'] = v

        return config

    def save_results(self, results: List[Dict], output_dir: str) -> None:
        """Save batch results to CSV file."""
        os.makedirs(output_dir, exist_ok=True)

        if not results:
            return

        csv_path = os.path.join(output_dir, 'batch_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

        import logging
        logging.getLogger(__name__).info(
            "Saved %d results to %s", len(results), csv_path)
