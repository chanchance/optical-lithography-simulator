"""
FDTD Engine based on Pistor 2001 - TEMPEST electromagnetic simulator.
Implements the Yee algorithm for solving Maxwell's equations.

Reference: Pistor, T.V. (2001). Electromagnetic Simulation and Modeling with
Applications in Lithography. UC Berkeley PhD Thesis.
"""
import numpy as np
from typing import Optional, Dict, Tuple, Callable


# Physical constants
C_LIGHT = 2.998e8      # Speed of light (m/s)
EPS_0 = 8.854e-12      # Permittivity of free space (F/m)
MU_0 = 4.0 * np.pi * 1e-7  # Permeability of free space (H/m), exact


class YeeGrid:
    """
    Staggered Yee grid for FDTD simulation.
    Ex, Ey, Ez at integer time steps.
    Hx, Hy, Hz at integer-plus-half time steps.

    Grid layout (2D cross-section shown):
    Each Yee cell contains 6 field components at offset positions.
    """

    def __init__(self, nx: int, ny: int, nz: int, dx: float, dy: float, dz: float):
        """
        Args:
            nx, ny, nz: Grid dimensions (number of cells)
            dx, dy, dz: Cell sizes in meters
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz

        # E field components (at integer time steps n)
        self.Ex = np.zeros((nx, ny+1, nz+1), dtype=np.float64)
        self.Ey = np.zeros((nx+1, ny, nz+1), dtype=np.float64)
        self.Ez = np.zeros((nx+1, ny+1, nz), dtype=np.float64)

        # H field components (at half-integer time steps n+1/2)
        self.Hx = np.zeros((nx+1, ny, nz), dtype=np.float64)
        self.Hy = np.zeros((nx, ny+1, nz), dtype=np.float64)
        self.Hz = np.zeros((nx, ny, nz+1), dtype=np.float64)

        # Material properties (spatially varying)
        self.eps_r = np.ones((nx, ny, nz), dtype=np.float64)   # Relative permittivity
        self.mu_r = np.ones((nx, ny, nz), dtype=np.float64)    # Relative permeability
        self.sigma = np.zeros((nx, ny, nz), dtype=np.float64)  # Conductivity (S/m)

        # Update coefficients (precomputed per cell for efficiency)
        self._alpha = None
        self._beta = None
        self._gamma = None


class PMLLayer:
    """
    Perfectly Matched Layer (PML) absorbing boundary condition.
    Berenger's PML for FDTD - prevents reflections at domain boundaries.
    """

    def __init__(self, thickness: int = 10, sigma_max_factor: float = 4.0):
        self.thickness = thickness
        self.sigma_max_factor = sigma_max_factor

    def compute_sigma_profile(self, n_cells: int, dx: float, freq: float) -> np.ndarray:
        """
        Compute conductivity profile for PML region.
        Uses polynomial grading: sigma(x) = sigma_max * (x/d)^m
        """
        if self.thickness <= 0 or n_cells <= 0:
            return np.zeros(max(0, n_cells))

        sigma_max = self.sigma_max_factor * EPS_0 * C_LIGHT / (self.thickness * dx)
        m = 3  # Grading order

        profile = np.zeros(n_cells)
        # Left PML
        n_left = min(self.thickness, n_cells)
        for i in range(n_left):
            x = (i + 0.5) / self.thickness
            profile[i] = sigma_max * (x ** m)
        # Right PML (only write cells not already covered by left PML)
        for i in range(min(self.thickness, n_cells)):
            j = n_cells - 1 - i
            if j >= n_left:
                x = (i + 0.5) / self.thickness
                profile[j] = sigma_max * (x ** m)

        return profile


class FDTDSimulator:
    """
    Full FDTD simulator implementing the TEMPEST algorithm from Pistor 2001.

    Key references from the thesis:
    - Eq 1-8, 1-9: Yee updating equations
    - Eq 1-13, 1-14: Convergence checking (PTERR)
    - Section 1.2.3: Domain excitation (plane wave)
    - Section 2.5: PML boundary conditions
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Simulation configuration dict with keys:
                dx_nm, dt_factor, max_timesteps, convergence_threshold,
                pml_thickness, wavelength_nm
        """
        self.config = config
        self.wavelength_nm = config.get('wavelength_nm', 193.0)
        self.dx = config.get('dx_nm', 5.0) * 1e-9      # Convert nm to m
        self.dy = self.dx
        self.dz = self.dx

        # Time step via CFL stability condition
        dt_factor = config.get('dt_factor', 0.9)
        self.dt = dt_factor * self.dx / (C_LIGHT * np.sqrt(3.0))

        self.max_timesteps = config.get('max_timesteps', 10000)
        self.mre = config.get('convergence_threshold', 0.01)
        self.pml_thickness = config.get('pml_thickness', 10)

        self.grid = None
        self.pml = PMLLayer(self.pml_thickness)
        self._convergence_history = []

    def initialize(self, nx: int, ny: int, nz: int) -> None:
        """Initialize the simulation grid."""
        self.grid = YeeGrid(nx, ny, nz, self.dx, self.dy, self.dz)
        self._precompute_update_coefficients()

    def _precompute_update_coefficients(self) -> None:
        """
        Precompute Yee update coefficients (Pistor Eq 1-10):
        alpha = (2*eps - sigma*dt) / (2*eps + sigma*dt)
        beta  = dt/dx * 2 / (2*eps + sigma*dt)
        gamma = dt / (mu*dx)
        """
        g = self.grid
        eps = g.eps_r * EPS_0
        mu = g.mu_r * MU_0
        sigma = g.sigma

        g._alpha = (2.0 * eps - sigma * self.dt) / (2.0 * eps + sigma * self.dt)
        g._beta = (self.dt / self.dx) * (2.0 / (2.0 * eps + sigma * self.dt))
        g._gamma = self.dt / (mu * self.dx)

    def _update_E_fields(self) -> None:
        """
        Update E fields using Yee equations (Pistor Eq 1-8).
        E^{n+1} = alpha * E^n + beta * (curl H)
        Vectorized numpy operations for efficiency.

        Yee staggering:
          Ex(nx, ny+1, nz+1) at (i, j+1/2, k+1/2)
          Ey(nx+1, ny, nz+1) at (i+1/2, j, k+1/2)
          Ez(nx+1, ny+1, nz) at (i+1/2, j+1/2, k)
        Coefficients alpha, beta are (nx, ny, nz) at cell centers.
        Map cell-centered coefficients to interior E-field nodes by
        slicing to match the interior dimensions.
        """
        g = self.grid
        nx, ny, nz = g.nx, g.ny, g.nz
        alpha = g._alpha
        beta = g._beta

        # Ex interior: Ex[:, 1:ny, 1:nz] has shape (nx, ny-1, nz-1)
        # Use coefficients alpha[:, :ny-1, :nz-1]
        a_ex = alpha[:, :ny-1, :nz-1]
        b_ex = beta[:, :ny-1, :nz-1]
        g.Ex[:, 1:-1, 1:-1] = (
            a_ex * g.Ex[:, 1:-1, 1:-1] +
            b_ex * (
                (g.Hz[:, 1:, 1:-1] - g.Hz[:, :-1, 1:-1]) / self.dy -
                (g.Hy[:, 1:-1, 1:] - g.Hy[:, 1:-1, :-1]) / self.dz
            )
        )

        # Ey interior: Ey[1:nx, :, 1:nz] has shape (nx-1, ny, nz-1)
        a_ey = alpha[:nx-1, :, :nz-1]
        b_ey = beta[:nx-1, :, :nz-1]
        g.Ey[1:-1, :, 1:-1] = (
            a_ey * g.Ey[1:-1, :, 1:-1] +
            b_ey * (
                (g.Hx[1:-1, :, 1:] - g.Hx[1:-1, :, :-1]) / self.dz -
                (g.Hz[1:, :, 1:-1] - g.Hz[:-1, :, 1:-1]) / self.dx
            )
        )

        # Ez interior: Ez[1:nx, 1:ny, :] has shape (nx-1, ny-1, nz)
        a_ez = alpha[:nx-1, :ny-1, :]
        b_ez = beta[:nx-1, :ny-1, :]
        g.Ez[1:-1, 1:-1, :] = (
            a_ez * g.Ez[1:-1, 1:-1, :] +
            b_ez * (
                (g.Hy[1:, 1:-1, :] - g.Hy[:-1, 1:-1, :]) / self.dx -
                (g.Hx[1:-1, 1:, :] - g.Hx[1:-1, :-1, :]) / self.dy
            )
        )

    def _update_H_fields(self) -> None:
        """
        Update H fields using Yee equations (Pistor Eq 1-9).
        H^{n+1/2} = H^{n-1/2} + gamma * (curl E)

        gamma is (nx, ny, nz) at cell centers.  H-field components live
        at face centres of the dual grid and their shapes differ:
          Hx(nx+1, ny, nz), Hy(nx, ny+1, nz), Hz(nx, ny, nz+1).
        Average gamma to H-positions along the staggered dimension.
        """
        g = self.grid
        gamma = g._gamma  # (nx, ny, nz)

        # Hx at (i+1/2, j, k) — average gamma along x
        # gamma_hx shape: (nx-1, ny, nz) -> use for interior Hx[1:-1,:,:]
        gamma_hx = 0.5 * (gamma[:-1, :, :] + gamma[1:, :, :])
        g.Hx[1:-1, :, :] += gamma_hx * (
            (g.Ey[1:-1, :, 1:] - g.Ey[1:-1, :, :-1]) / self.dz -
            (g.Ez[1:-1, 1:, :] - g.Ez[1:-1, :-1, :]) / self.dy
        )
        # Boundary Hx rows use nearest cell gamma
        g.Hx[0, :, :] += gamma[0, :, :] * (
            (g.Ey[0, :, 1:] - g.Ey[0, :, :-1]) / self.dz -
            (g.Ez[0, 1:, :] - g.Ez[0, :-1, :]) / self.dy
        )
        g.Hx[-1, :, :] += gamma[-1, :, :] * (
            (g.Ey[-1, :, 1:] - g.Ey[-1, :, :-1]) / self.dz -
            (g.Ez[-1, 1:, :] - g.Ez[-1, :-1, :]) / self.dy
        )

        # Hy at (i, j+1/2, k) — average gamma along y
        gamma_hy = 0.5 * (gamma[:, :-1, :] + gamma[:, 1:, :])
        g.Hy[:, 1:-1, :] += gamma_hy * (
            (g.Ez[1:, 1:-1, :] - g.Ez[:-1, 1:-1, :]) / self.dx -
            (g.Ex[:, 1:-1, 1:] - g.Ex[:, 1:-1, :-1]) / self.dz
        )
        g.Hy[:, 0, :] += gamma[:, 0, :] * (
            (g.Ez[1:, 0, :] - g.Ez[:-1, 0, :]) / self.dx -
            (g.Ex[:, 0, 1:] - g.Ex[:, 0, :-1]) / self.dz
        )
        g.Hy[:, -1, :] += gamma[:, -1, :] * (
            (g.Ez[1:, -1, :] - g.Ez[:-1, -1, :]) / self.dx -
            (g.Ex[:, -1, 1:] - g.Ex[:, -1, :-1]) / self.dz
        )

        # Hz at (i, j, k+1/2) — average gamma along z
        gamma_hz = 0.5 * (gamma[:, :, :-1] + gamma[:, :, 1:])
        g.Hz[:, :, 1:-1] += gamma_hz * (
            (g.Ex[:, 1:, 1:-1] - g.Ex[:, :-1, 1:-1]) / self.dy -
            (g.Ey[1:, :, 1:-1] - g.Ey[:-1, :, 1:-1]) / self.dx
        )
        g.Hz[:, :, 0] += gamma[:, :, 0] * (
            (g.Ex[:, 1:, 0] - g.Ex[:, :-1, 0]) / self.dy -
            (g.Ey[1:, :, 0] - g.Ey[:-1, :, 0]) / self.dx
        )
        g.Hz[:, :, -1] += gamma[:, :, -1] * (
            (g.Ex[:, 1:, -1] - g.Ex[:, :-1, -1]) / self.dy -
            (g.Ey[1:, :, -1] - g.Ey[:-1, :, -1]) / self.dx
        )

    def excite_plane_wave(self, timestep: int, polarization: str = 'TE',
                          direction: str = 'z', amplitude: float = 1.0) -> None:
        """
        Plane wave excitation via soft-source injection (Pistor Section 1.2.3).
        Sinusoidal excitation at specified wavelength.
        """
        freq = C_LIGHT / (self.wavelength_nm * 1e-9)
        omega = 2.0 * np.pi * freq
        t = timestep * self.dt

        # Sinusoidal source with smooth turn-on (Gaussian envelope first few cycles)
        n_cycles_rampup = 3.0
        T_period = 1.0 / freq
        if t < n_cycles_rampup * T_period:
            envelope = 0.5 * (1.0 - np.cos(np.pi * t / (n_cycles_rampup * T_period)))
        else:
            envelope = 1.0

        source_value = amplitude * envelope * np.sin(omega * t)

        # Inject at z=pml_thickness plane, clamped to valid range
        z_src = min(self.pml_thickness, self.grid.nz - 1)
        if direction == 'z':
            if polarization == 'TE':
                # TE: E in x-direction
                self.grid.Ex[:, :, z_src] += source_value
            else:
                # TM: E in y-direction
                self.grid.Ey[:, :, z_src] += source_value

    def _check_convergence(self, timestep: int, freq: float) -> float:
        """
        Convergence checking (Pistor Eq 1-13, 1-14).
        Compare field amplitude one period apart.
        pterr = |Eamp(cT) - Eamp((c-1)T)| / |Eamp(cT) + Eamp((c-1)T)|
        Returns max relative error across test points.
        """
        T_period = 1.0 / freq
        steps_per_period = max(1, int(T_period / self.dt))

        if timestep < 2 * steps_per_period:
            return 1.0  # Not enough data yet

        g = self.grid
        nx, ny, nz = g.nx, g.ny, g.nz

        # Sample test points on cubic grid (5 nodes apart, Pistor convention)
        spacing = max(1, min(5, min(nx, ny, nz) // 4))
        ix = np.arange(spacing, nx - spacing, spacing)
        iy = np.arange(spacing, ny - spacing, spacing)
        iz = np.arange(spacing, nz - spacing, spacing)

        if len(ix) == 0 or len(iy) == 0 or len(iz) == 0:
            return 0.0

        # Sample Ez component
        E_now = np.abs(g.Ez[ix[:, None, None], iy[None, :, None], iz[None, None, :]])

        if not hasattr(self, '_E_prev'):
            self._E_prev = E_now.copy()
            return 1.0

        E_prev = self._E_prev

        # Compute pterr for each test point
        denom = np.abs(E_now) + np.abs(E_prev)
        mask = denom > 1e-30  # Avoid division by near-zero

        pterr_array = np.zeros_like(E_now)
        pterr_array[mask] = np.abs(E_now[mask] - E_prev[mask]) / denom[mask]

        max_pterr = float(np.max(pterr_array))
        self._E_prev = E_now.copy()
        self._convergence_history.append(max_pterr)

        return max_pterr

    def run_simulation(self, on_progress: Optional[Callable] = None) -> Dict:
        """
        Main FDTD simulation loop.
        Returns dict with final field arrays and convergence history.
        """
        if self.grid is None:
            raise RuntimeError("Call initialize() before run_simulation()")

        freq = C_LIGHT / (self.wavelength_nm * 1e-9)
        steps_per_period = max(1, int(1.0 / (freq * self.dt)))
        check_interval = steps_per_period  # Check convergence every period

        converged = False
        n = 0
        for n in range(self.max_timesteps):
            # Update H at n+1/2
            self._update_H_fields()

            # Update E at n+1
            self._update_E_fields()

            # Inject source
            self.excite_plane_wave(n, polarization='TE')

            # Convergence check every period
            if n % check_interval == 0 and n > 0:
                max_err = self._check_convergence(n, freq)
                if max_err < self.mre:
                    converged = True
                    break

            if on_progress and n % max(1, self.max_timesteps // 100) == 0:
                on_progress(n / self.max_timesteps)

        return {
            'Ex': self.grid.Ex.copy(),
            'Ey': self.grid.Ey.copy(),
            'Ez': self.grid.Ez.copy(),
            'Hx': self.grid.Hx.copy(),
            'Hy': self.grid.Hy.copy(),
            'Hz': self.grid.Hz.copy(),
            'converged': converged,
            'timesteps': n + 1,
            'convergence_history': self._convergence_history.copy(),
        }

    def get_intensity(self, fields: Dict) -> np.ndarray:
        """
        Compute |E|^2 intensity from field dict.

        Yee grid components live at staggered positions and have different
        shapes: Ex(nx, ny+1, nz+1), Ey(nx+1, ny, nz+1), Ez(nx+1, ny+1, nz).
        Interpolate each to cell centers (nx, ny, nz) before summing.
        """
        Ex = fields['Ex']
        Ey = fields['Ey']
        Ez = fields['Ez']

        # Interpolate to cell centers by averaging over the staggered dims
        Ex_c = 0.5 * (Ex[:, :-1, :-1] + Ex[:, 1:, 1:])
        Ey_c = 0.5 * (Ey[:-1, :, :-1] + Ey[1:, :, 1:])
        Ez_c = 0.5 * (Ez[:-1, :-1, :] + Ez[1:, 1:, :])

        return np.abs(Ex_c)**2 + np.abs(Ey_c)**2 + np.abs(Ez_c)**2
