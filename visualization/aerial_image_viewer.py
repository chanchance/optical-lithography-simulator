"""
Aerial image visualization: heatmaps, layout overlays, cross-sections.
"""
import numpy as np
from typing import Optional, Tuple, List

import matplotlib
# Do NOT call matplotlib.use('Agg') — it conflicts with the Qt backend
# when this module is imported from the GUI.  Standalone scripts can set
# the backend before importing this module.
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


class AerialImageViewer:
    """Visualization tools for aerial images from optical simulation."""

    def __init__(self, domain_size_nm: float, grid_size: int):
        self.domain_size_nm = domain_size_nm
        self.grid_size = grid_size
        self.dx_nm = domain_size_nm / grid_size

    def _get_extent(self) -> List[float]:
        """Get image extent for imshow [xmin, xmax, ymin, ymax]."""
        return [0, self.domain_size_nm, 0, self.domain_size_nm]

    def plot_aerial_image(self, intensity_2d: np.ndarray,
                           ax=None, colormap: str = 'hot',
                           title: str = 'Aerial Image',
                           show_colorbar: bool = True) -> plt.Axes:
        """
        Plot aerial image as heatmap.

        Args:
            intensity_2d: 2D normalized intensity [0,1]
            ax: Existing axes (None=create new)
            colormap: matplotlib colormap name ('hot', 'inferno', 'viridis')
        Returns:
            matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(intensity_2d, origin='lower', cmap=colormap,
                        extent=self._get_extent(), vmin=0, vmax=1,
                        interpolation='bilinear')

        if show_colorbar:
            plt.colorbar(im, ax=ax, label='Normalized Intensity')

        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_title(title)

        return ax

    def overlay_layout(self, ax, polygons: List[np.ndarray],
                        origin_nm: Tuple[float, float] = (0.0, 0.0),
                        alpha: float = 0.3, color: str = 'cyan') -> plt.Axes:
        """
        Overlay GDS layout polygons on aerial image.

        Args:
            ax: Axes with aerial image
            polygons: List of (N,2) polygon coordinate arrays in nm
            origin_nm: Layout origin offset
            alpha: Polygon transparency
            color: Polygon color
        Returns:
            Same axes with overlay
        """
        mpl_patches = []
        for pts in polygons:
            if len(pts) >= 3:
                pts_shifted = pts - np.array(origin_nm)
                patch = MplPolygon(pts_shifted, closed=True)
                mpl_patches.append(patch)

        if mpl_patches:
            col = PatchCollection(mpl_patches, facecolor=color, alpha=alpha,
                                   edgecolor='white', linewidth=1.0)
            ax.add_collection(col)

        return ax

    def plot_threshold_contour(self, ax, intensity_2d: np.ndarray,
                                threshold: float = 0.30,
                                color: str = 'cyan',
                                linewidth: float = 2.0) -> plt.Axes:
        """
        Plot threshold contour line on aerial image.
        This shows the predicted printed edge at the given exposure threshold.
        """
        extent = self._get_extent()
        x_nm = np.linspace(extent[0], extent[1], self.grid_size)
        y_nm = np.linspace(extent[2], extent[3], self.grid_size)

        ax.contour(x_nm, y_nm, intensity_2d,
                   levels=[threshold], colors=[color], linewidths=linewidth)
        return ax

    def plot_cross_section(self, intensity_2d: np.ndarray,
                            position_nm: Optional[float] = None,
                            direction: str = 'x',
                            ax=None, threshold: float = 0.30) -> plt.Axes:
        """
        Plot 1D cross-section of intensity profile.

        Args:
            intensity_2d: 2D intensity array
            position_nm: Position along perpendicular axis (None=center)
            direction: 'x' or 'y'
            ax: Existing axes
            threshold: Show horizontal threshold line
        Returns:
            matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        N = self.grid_size
        if position_nm is None:
            idx = N // 2
        else:
            idx = int(position_nm / self.dx_nm)
            idx = np.clip(idx, 0, N-1)

        x_nm = np.arange(N) * self.dx_nm
        if direction == 'x':
            profile = intensity_2d[:, idx]
            xlabel = 'X (nm)'
        else:
            profile = intensity_2d[idx, :]
            xlabel = 'Y (nm)'

        ax.plot(x_nm, profile, 'b-', linewidth=2, label='Intensity')
        ax.axhline(y=threshold, color='r', linestyle='--',
                   label='Threshold ({:.2f})'.format(threshold))
        ax.fill_between(x_nm, 0, profile, alpha=0.2, color='blue')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Normalized Intensity')
        ax.set_title('Intensity Cross-Section (direction={})'.format(direction))
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_side_by_side(self, mask_grid: np.ndarray,
                           aerial_image: np.ndarray,
                           threshold: float = 0.30,
                           figsize: Tuple = (16, 7)) -> plt.Figure:
        """
        Create side-by-side comparison: mask | aerial image | overlay.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Panel 1: Mask pattern
        axes[0].imshow(mask_grid, origin='lower', cmap='gray',
                       extent=self._get_extent(), vmin=0, vmax=1)
        axes[0].set_title('Mask Pattern')
        axes[0].set_xlabel('X (nm)')
        axes[0].set_ylabel('Y (nm)')

        # Panel 2: Aerial image
        im = axes[1].imshow(aerial_image, origin='lower', cmap='hot',
                            extent=self._get_extent(), vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[1], label='Intensity')
        axes[1].set_title('Aerial Image')
        axes[1].set_xlabel('X (nm)')

        # Panel 3: Overlay with threshold contour
        axes[2].imshow(aerial_image, origin='lower', cmap='hot',
                       extent=self._get_extent(), vmin=0, vmax=1, alpha=0.7)
        self.plot_threshold_contour(axes[2], aerial_image, threshold, 'cyan')

        # Overlay mask edges
        N = self.grid_size
        x_nm = np.linspace(0, self.domain_size_nm, N)
        y_nm = np.linspace(0, self.domain_size_nm, N)
        axes[2].contour(x_nm, y_nm, mask_grid, levels=[0.5],
                        colors=['white'], linewidths=1.5, linestyles='--')
        axes[2].set_title('Overlay (cyan=print edge, white=mask edge)')
        axes[2].set_xlabel('X (nm)')

        fig.tight_layout()
        return fig
