"""
Equipment parameter I/O for optical lithography simulator.
Handles loading/saving YAML configuration files with validation.
"""
import os
import copy
from typing import Dict, Any, Optional

import yaml


class ParameterIO:
    """Load and save lithography simulation parameters in YAML format."""

    REQUIRED_KEYS = ['lithography', 'simulation']

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._defaults = None

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load parameters from YAML file. Merges with defaults."""
        filepath = path or self.config_path
        if filepath is None:
            return self._get_defaults()

        if not os.path.exists(filepath):
            raise FileNotFoundError("Config not found: {}".format(filepath))

        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        # Merge with defaults
        defaults = self._get_defaults()
        merged = self._deep_merge(defaults, config)
        self._validate(merged)
        return merged

    def save(self, config: Dict[str, Any], path: Optional[str] = None) -> None:
        """Save parameters to YAML file."""
        filepath = path or self.config_path
        if filepath is None:
            raise ValueError("No output path specified")

        self._validate(config)

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def load_defaults(self) -> Dict[str, Any]:
        """Load default config from package defaults file, or return
        hardcoded fallback if the file doesn't exist."""
        defaults_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'default_config.yaml'
        )
        if os.path.exists(defaults_path):
            with open(defaults_path, 'r') as f:
                return yaml.safe_load(f) or {}
        # Hardcoded fallback — avoids infinite recursion through _get_defaults
        return {
            'lithography': {
                'wavelength_nm': 193.0,
                'NA': 0.93,
                'defocus_nm': 0.0,
                'illumination': {
                    'type': 'annular',
                    'sigma_outer': 0.85,
                    'sigma_inner': 0.55,
                },
                'mask_type': 'binary',
                'aberrations': {},
            },
            'simulation': {
                'grid_size': 256,
                'domain_size_nm': 2000.0,
            },
        }

    def get_lithography_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract flat lithography parameter dict for simulation."""
        litho = config.get('lithography', {})
        illum = litho.get('illumination', {})
        mask = litho.get('mask', {})
        return {
            'wavelength_nm': litho.get('wavelength_nm', 193.0),
            'NA': litho.get('NA', 0.93),
            'illumination_type': illum.get('type', 'circular'),
            'sigma_outer': illum.get('sigma_outer', 0.85),
            'sigma_inner': illum.get('sigma_inner', 0.55),
            'sigma_c': illum.get('sigma_c', 0.15),
            'sigma_r': illum.get('sigma_r', 0.30),
            'theta_q': illum.get('theta_q', 45.0),
            'N_source_points': illum.get('N_source_points', 4),
            'polarization': illum.get('polarization', 'unpolarized'),
            'defocus_nm': litho.get('defocus_nm', 0.0),
            'mask_type': mask.get('type', 'binary'),
            'threshold': config.get('resist', {}).get('threshold', 0.30),
            'grid_size': config.get('simulation', {}).get('grid_size', 256),
            'domain_size_nm': config.get('simulation', {}).get('domain_size_nm', 2000.0),
        }

    def _validate(self, config: Dict[str, Any]) -> None:
        """Basic validation of config structure."""
        litho = config.get('lithography', {})
        wl = litho.get('wavelength_nm', 193.0)
        if not (1.0 <= wl <= 500.0):
            raise ValueError("wavelength_nm must be 1-500nm, got {}".format(wl))
        na = litho.get('NA', 0.93)
        if not (0.01 <= na <= 1.5):
            raise ValueError("NA must be 0.01-1.5, got {}".format(na))

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge override into base dict."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _get_defaults(self) -> Dict[str, Any]:
        if self._defaults is None:
            self._defaults = self.load_defaults()
        return copy.deepcopy(self._defaults)


def load_config(path: str) -> Dict[str, Any]:
    """Convenience function to load a config file."""
    return ParameterIO().load(path)


def save_config(config: Dict[str, Any], path: str) -> None:
    """Convenience function to save a config file."""
    ParameterIO().save(config, path)
