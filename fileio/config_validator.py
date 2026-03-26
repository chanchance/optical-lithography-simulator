"""Config validation for simulation parameters."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationError:
    field: str      # e.g. "lithography.NA"
    message: str    # e.g. "NA must be between 0 and 1.35"
    severity: str   # "error" or "warning"


class ConfigValidator:
    def validate(self, config: dict) -> List[ValidationError]:
        errors = []
        errors.extend(self._validate_lithography(config))
        errors.extend(self._validate_simulation(config))
        errors.extend(self._validate_resist(config))
        errors.extend(self._validate_illumination(config))
        return errors

    def _validate_lithography(self, config: dict) -> List[ValidationError]:
        errors = []
        litho = config.get('lithography', {})

        wl = litho.get('wavelength_nm', 193.0)
        if wl <= 0:
            errors.append(ValidationError('lithography.wavelength_nm',
                'wavelength_nm must be > 0', 'error'))

        na = litho.get('NA', 0.93)
        if not (0 < na <= 1.35):
            errors.append(ValidationError('lithography.NA',
                'NA must be in (0, 1.35]', 'error'))

        defocus = litho.get('defocus_nm', 0.0)
        if not (-2000 <= defocus <= 2000):
            errors.append(ValidationError('lithography.defocus_nm',
                'defocus_nm must be in [-2000, 2000] nm', 'error'))

        # EUV: NA > 0.33 is unusual
        if wl <= 15.0 and na > 0.33:
            errors.append(ValidationError('lithography.NA',
                'NA > 0.33 is unusual for EUV (wavelength <= 15 nm)', 'warning'))

        # EUV 13.5nm: NA > 0.55 is unusually high
        if abs(wl - 13.5) < 0.1 and na > 0.55:
            errors.append(ValidationError('lithography.NA',
                'EUV wavelength with NA > 0.55 is unusually high (typical EUV NA <= 0.33)',
                'warning'))

        # Hopkins TCC: n_kernels must be positive
        if litho.get('use_hopkins', False):
            n_kernels = litho.get('n_kernels', 10)
            if n_kernels <= 0:
                errors.append(ValidationError('lithography.n_kernels',
                    'n_kernels must be > 0 for Hopkins TCC '
                    '(got {}); aerial image will be all-zeros'.format(n_kernels), 'error'))

        # Hopkins TCC and vector imaging are mutually exclusive — Hopkins wins
        if litho.get('use_hopkins', False) and litho.get('use_vector', False):
            errors.append(ValidationError('lithography.use_vector',
                'Hopkins TCC and vector imaging are both enabled; '
                'Hopkins TCC takes priority (vector mode will be ignored)', 'warning'))

        return errors

    def _validate_simulation(self, config: dict) -> List[ValidationError]:
        errors = []
        sim = config.get('simulation', {})

        grid_size = sim.get('grid_size', 256)
        if not (16 <= grid_size <= 4096):
            errors.append(ValidationError('simulation.grid_size',
                'grid_size must be in [16, 4096]', 'error'))
        elif (grid_size & (grid_size - 1)) != 0:
            errors.append(ValidationError('simulation.grid_size',
                'grid_size should be a power of 2 for FFT efficiency', 'warning'))

        domain_nm = sim.get('domain_size_nm', 2000.0)
        if domain_nm <= 0:
            errors.append(ValidationError('simulation.domain_size_nm',
                'domain_size_nm must be > 0', 'error'))
        else:
            wl = config.get('lithography', {}).get('wavelength_nm', 193.0)
            na = config.get('lithography', {}).get('NA', 0.93)
            domain = sim.get('domain_size_nm', 2000.0)
            min_domain = 4 * wl / na  # 4 resolution elements
            if domain < min_domain:
                errors.append(ValidationError('simulation.domain_size_nm',
                    f'Domain ({domain:.0f}nm) may be too small for λ={wl}nm/NA={na:.2f}; '
                    f'recommend ≥ {min_domain:.0f}nm', 'warning'))

            if not (16 <= grid_size <= 4096):
                return errors  # grid_size already flagged; skip pixel-size check
            pixel_size = domain / grid_size
            nyq = wl / (4 * na)
            if pixel_size > nyq:
                errors.append(ValidationError('simulation.grid_size',
                    f'Pixel size {pixel_size:.1f}nm may undersample the PSF '
                    f'(λ/(4·NA)={nyq:.1f}nm)', 'warning'))

        return errors

    def _validate_resist(self, config: dict) -> List[ValidationError]:
        errors = []
        resist = config.get('resist', {})

        model = resist.get('model', 'threshold')
        threshold = resist.get('threshold', 0.30)
        if not (0 < threshold < 1):
            errors.append(ValidationError('resist.threshold',
                'threshold must be in (0, 1)', 'error'))

        if model == 'dill':
            A = resist.get('A', 0.8)
            B = resist.get('B', 0.1)
            C = resist.get('C', 1.0)
            if A <= 0:
                errors.append(ValidationError('resist.A',
                    'Dill A must be > 0', 'error'))
            if B < 0:
                errors.append(ValidationError('resist.B',
                    'Dill B must be >= 0', 'error'))
            if C <= 0:
                errors.append(ValidationError('resist.C',
                    'Dill C must be > 0', 'error'))

        elif model == 'ca':
            qe = resist.get('quantum_efficiency', 0.5)
            amp = resist.get('amplification', 50.0)
            if not (0 < qe < 1):
                errors.append(ValidationError('resist.quantum_efficiency',
                    'quantum_efficiency must be in (0, 1)', 'error'))
            if amp <= 0:
                errors.append(ValidationError('resist.amplification',
                    'amplification must be > 0', 'error'))

        return errors

    def _validate_illumination(self, config: dict) -> List[ValidationError]:
        errors = []
        illum = config.get('lithography', {}).get('illumination', {})

        sigma_outer = illum.get('sigma_outer', 0.85)
        sigma_inner = illum.get('sigma_inner', 0.55)

        if not (0 < sigma_outer <= 1):
            errors.append(ValidationError('lithography.illumination.sigma_outer',
                'sigma_outer must be in (0, 1]', 'error'))

        illum_type = illum.get('type', 'annular')
        if illum_type in ('annular', 'dipole'):
            if sigma_inner >= sigma_outer:
                errors.append(ValidationError('lithography.illumination.sigma_inner',
                    'sigma_inner must be < sigma_outer for {} illumination'.format(illum_type),
                    'error'))
            if sigma_inner < 0:
                errors.append(ValidationError('lithography.illumination.sigma_inner',
                    'sigma_inner must be >= 0', 'error'))

        if illum_type == 'freeform':
            # Use the same default as create_source() (64) so that a freeform
            # config without an explicit pupil_size is not incorrectly blocked.
            pupil_size = illum.get('pupil_size', 64)
            if pupil_size <= 0:
                errors.append(ValidationError('lithography.illumination.pupil_size',
                    'pupil_size must be > 0 for freeform illumination', 'error'))

        return errors

    def validate_or_raise(self, config: dict) -> None:
        """Raise ValueError if any error-severity issues found."""
        errors = [e for e in self.validate(config) if e.severity == 'error']
        if errors:
            msg = '\n'.join('  {}: {}'.format(e.field, e.message) for e in errors)
            raise ValueError("Invalid simulation config:\n" + msg)
