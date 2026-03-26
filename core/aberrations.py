"""
Fringe Zernike polynomial aberrations model (Z1-Z37).
Implements orthonormal Zernike basis on the unit disk.
"""
import numpy as np
from math import factorial

# Fringe Zernike table: (n, m, name) for Z1..Z37
# n = radial order, m = azimuthal order (signed)
# Convention: m > 0 -> cos term, m < 0 -> sin term, m == 0 -> rotationally symmetric
ZERNIKE_TABLE = [
    (0,  0, "Piston"),            # Z1
    (1, -1, "Tilt Y"),            # Z2
    (1,  1, "Tilt X"),            # Z3
    (2, -2, "Astigmatism 45"),    # Z4
    (2,  0, "Defocus"),           # Z5
    (2,  2, "Astigmatism 0"),     # Z6
    (3, -3, "Trefoil Y"),         # Z7
    (3, -1, "Coma Y"),            # Z8
    (3,  1, "Coma X"),            # Z9
    (3,  3, "Trefoil X"),         # Z10
    (4, -4, "Tetrafoil Y"),       # Z11
    (4, -2, "2nd Astig Y"),       # Z12
    (4,  0, "Spherical"),         # Z13
    (4,  2, "2nd Astig X"),       # Z14
    (4,  4, "Tetrafoil X"),       # Z15
    (5, -5, "Pentafoil Y"),       # Z16
    (5, -3, "2nd Trefoil Y"),     # Z17
    (5, -1, "2nd Coma Y"),        # Z18
    (5,  1, "2nd Coma X"),        # Z19
    (5,  3, "2nd Trefoil X"),     # Z20
    (5,  5, "Pentafoil X"),       # Z21
    (6, -6, "Hexafoil Y"),        # Z22
    (6, -4, "2nd Tetrafoil Y"),   # Z23
    (6, -2, "3rd Astig Y"),       # Z24
    (6,  0, "2nd Spherical"),     # Z25
    (6,  2, "3rd Astig X"),       # Z26
    (6,  4, "2nd Tetrafoil X"),   # Z27
    (6,  6, "Hexafoil X"),        # Z28
    (7, -7, "Heptafoil Y"),       # Z29
    (7, -5, "2nd Pentafoil Y"),   # Z30
    (7, -3, "3rd Trefoil Y"),     # Z31
    (7, -1, "3rd Coma Y"),        # Z32
    (7,  1, "3rd Coma X"),        # Z33
    (7,  3, "3rd Trefoil X"),     # Z34
    (7,  5, "2nd Pentafoil X"),   # Z35
    (7,  7, "Heptafoil X"),       # Z36
    (8,  0, "3rd Spherical"),     # Z37
]


def _radial_polynomial(n: int, m_abs: int, rho: np.ndarray) -> np.ndarray:
    """Compute radial Zernike polynomial R_n^{m_abs}(rho)."""
    R = np.zeros_like(rho, dtype=np.float64)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1) ** s * factorial(n - s) /
                 (factorial(s) *
                  factorial((n + m_abs) // 2 - s) *
                  factorial((n - m_abs) // 2 - s)))
        R = R + coeff * rho ** (n - 2 * s)
    return R


def zernike_polynomial(n: int, m: int, rho: np.ndarray,
                       theta: np.ndarray) -> np.ndarray:
    """
    Compute orthonormal Zernike polynomial Z_n^m(rho, theta).

    Normalization (OSA/ANSI):
      N = sqrt(2*(n+1)) for m != 0
      N = sqrt(n+1)     for m == 0
    """
    m_abs = abs(m)
    R = _radial_polynomial(n, m_abs, rho)
    norm = np.sqrt(2.0 * (n + 1)) if m != 0 else np.sqrt(float(n + 1))
    if m > 0:
        angular = np.cos(m * theta)
    elif m < 0:
        angular = np.sin(m_abs * theta)
    else:
        angular = np.ones_like(theta)
    return norm * R * angular


class ZernikeAberration:
    """37-term Fringe Zernike wavefront aberration model."""

    def __init__(self, coefficients: dict):
        """
        Args:
            coefficients: {1: val, 2: val, ...} Z-index 1-based, values in waves
        """
        self.coefficients = coefficients

    @classmethod
    def from_list(cls, coeffs: list) -> 'ZernikeAberration':
        """Create from a list of up to 37 values [Z1, Z2, ..., Z37]."""
        return cls({i + 1: float(v) for i, v in enumerate(coeffs)})

    def pupil_phase(self, kx_norm: np.ndarray,
                    ky_norm: np.ndarray) -> np.ndarray:
        """
        Compute wavefront phase map [radians] on normalized pupil grid.

        Args:
            kx_norm, ky_norm: normalized pupil coordinates (-1..1),
                              where rho = sqrt(kx^2+ky^2) = 1 at pupil edge.
        Returns:
            Phase in radians; zero outside the unit pupil circle.
        """
        rho = np.sqrt(kx_norm ** 2 + ky_norm ** 2)
        theta = np.arctan2(ky_norm, kx_norm)
        inside = rho <= 1.0

        phase_waves = np.zeros_like(rho, dtype=np.float64)
        for idx, (n, m, _) in enumerate(ZERNIKE_TABLE, start=1):
            c = self.coefficients.get(idx, 0.0)
            if c == 0.0:
                continue
            phase_waves = phase_waves + c * zernike_polynomial(n, m, rho, theta)

        return 2.0 * np.pi * phase_waves * inside
