"""
EM field visualization for FDTD simulation results.
Displays Ex, Ey, Ez, Hx, Hy, Hz components and derived quantities.
"""
import numpy as np
from typing import Optional, Dict, Tuple, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class FieldViewer:
    """Visualization for FDTD electromagnetic field data."""

    def __init__(self, dx_nm: float = 5.0):
        self.dx_nm = dx_nm

    def plot_field_component(self, field_array: np.ndarray,
                              component: str = 'Ez',
                              z_slice: Optional[int] = None,
                              ax=None, cmap: str = 'RdBu') -> plt.Axes:
        """
        Plot a single EM field component.

        Args:
            field_array: 3D or 2D field component array
            component: Field name ('Ex','Ey','Ez','Hx','Hy','Hz')
            z_slice: Z-index for 3D arrays (None=midplane)
            ax: Existing axes
            cmap: Colormap (RdBu for signed fields)
        Returns:
            matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        if field_array.ndim == 3:
            if z_slice is None:
                z_slice = field_array.shape[2] // 2
            data = field_array[:, :, z_slice]
        else:
            data = field_array

        ny, nx = data.shape[:2] if data.ndim >= 2 else (1, len(data))

        vmax = np.max(np.abs(data))
        if vmax < 1e-30:
            vmax = 1.0

        extent_nm = [0, nx * self.dx_nm, 0, ny * self.dx_nm]

        im = ax.imshow(data.T, origin='lower', cmap=cmap,
                        extent=extent_nm, vmin=-vmax, vmax=vmax,
                        interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='{} (V/m)'.format(component))
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_title('Field Component: {}'.format(component))

        return ax

    def plot_intensity(self, fields: Dict[str, np.ndarray],
                        z_slice: Optional[int] = None,
                        ax=None) -> plt.Axes:
        """
        Plot |E|^2 intensity from field dict.
        fields: dict with keys 'Ex', 'Ey', 'Ez'
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        def get_slice(key):
            arr = fields.get(key, np.zeros((1, 1, 1)))
            if arr.ndim == 3:
                sl = arr.shape[2] // 2 if z_slice is None else z_slice
                return arr[:, :, sl]
            return arr

        Ex = get_slice('Ex')
        Ey = get_slice('Ey')
        Ez = get_slice('Ez')

        # Pad to same shape
        shape = (max(Ex.shape[0], Ey.shape[0], Ez.shape[0]),
                 max(Ex.shape[1], Ey.shape[1], Ez.shape[1]))

        def pad_to(a, s):
            out = np.zeros(s)
            out[:a.shape[0], :a.shape[1]] = a[:s[0], :s[1]]
            return out

        intensity = pad_to(np.abs(Ex)**2, shape) + \
                    pad_to(np.abs(Ey)**2, shape) + \
                    pad_to(np.abs(Ez)**2, shape)

        ny, nx = shape
        extent_nm = [0, nx * self.dx_nm, 0, ny * self.dx_nm]

        im = ax.imshow(intensity.T, origin='lower', cmap='hot',
                        extent=extent_nm, vmin=0,
                        interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='|E|² (V²/m²)')
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_title('EM Field Intensity |E|²')

        return ax

    def plot_near_field(self, fields: Dict[str, np.ndarray],
                         z_slice: Optional[int] = None,
                         figsize: Tuple = (15, 5)) -> plt.Figure:
        """
        Plot near-field: all three E components + intensity.
        """
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        components = ['Ex', 'Ey', 'Ez']

        for i, comp in enumerate(components):
            if comp in fields:
                self.plot_field_component(fields[comp], comp, z_slice, axes[i])

        self.plot_intensity(fields, z_slice, axes[3])

        fig.suptitle('Near-Field EM Components', fontsize=14)
        fig.tight_layout()
        return fig

    def animate_field_evolution(self, field_history: List[np.ndarray],
                                 component: str = 'Ez',
                                 interval: int = 50) -> FuncAnimation:
        """
        Create animation of field evolution over time.
        field_history: list of 2D arrays (one per timestep)
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        vmax = max(np.max(np.abs(f)) for f in field_history) + 1e-30

        im = ax.imshow(field_history[0].T, origin='lower', cmap='RdBu',
                        vmin=-vmax, vmax=vmax, animated=True)
        plt.colorbar(im, ax=ax)
        title = ax.set_title('t=0')

        def update(frame):
            im.set_array(field_history[frame].T)
            title.set_text('{}: timestep={}'.format(component, frame))
            return [im, title]

        anim = FuncAnimation(fig, update, frames=len(field_history),
                             interval=interval, blit=True)
        return anim
