"""
GDS/OAS layout visualization using matplotlib.
Renders polygons with layer-based coloring, scale bars, zoom.
"""
import numpy as np
from typing import Optional, Dict, List, Tuple

import matplotlib
# Do NOT call matplotlib.use('Agg') — it conflicts with the Qt backend
# when this module is imported from the GUI.
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

try:
    from fileio.layout_io import LayoutData
    HAS_LAYOUT = True
except ImportError:
    HAS_LAYOUT = False

# Default layer colors (layer_num -> RGBA)
DEFAULT_LAYER_COLORS = {
    0:  (0.8, 0.8, 0.2, 0.8),   # Yellow - metal
    1:  (0.2, 0.6, 0.9, 0.8),   # Blue - active
    2:  (0.9, 0.3, 0.3, 0.8),   # Red - poly
    3:  (0.3, 0.9, 0.3, 0.8),   # Green - contact
    4:  (0.7, 0.3, 0.9, 0.8),   # Purple - via
    5:  (0.9, 0.6, 0.2, 0.8),   # Orange - metal2
    6:  (0.4, 0.8, 0.8, 0.8),   # Cyan - metal3
    7:  (0.9, 0.5, 0.5, 0.8),   # Pink - metal4
}


class LayoutViewer:
    """Matplotlib-based GDS/OAS layout viewer."""

    def __init__(self, layer_colors: Optional[Dict[int, Tuple]] = None):
        self.layer_colors = layer_colors or DEFAULT_LAYER_COLORS

    def get_layer_color(self, layer_num: int) -> Tuple:
        if layer_num in self.layer_colors:
            return self.layer_colors[layer_num]
        # Auto-generate deterministic color for unknown layers using a
        # local RNG so we don't corrupt the global numpy random state.
        rng = np.random.RandomState(layer_num)
        r, g, b = rng.rand(3) * 0.7 + 0.2
        return (r, g, b, 0.8)

    def plot_layout(self, layout_data, layers: Optional[List[int]] = None,
                    ax=None, figsize: Tuple = (10, 10)) -> plt.Axes:
        """
        Plot GDS layout polygons.

        Args:
            layout_data: LayoutData from LayoutReader
            layers: List of layer numbers to display (None=all)
            ax: Existing matplotlib axes (None=create new)
            figsize: Figure size tuple
        Returns:
            matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if layers is None and HAS_LAYOUT and hasattr(layout_data, 'get_layer_numbers'):
            layers = layout_data.get_layer_numbers()
        elif layers is None:
            layers = list(layout_data.get('layers', {}).keys())

        for layer_num in layers:
            color = self.get_layer_color(layer_num)

            if HAS_LAYOUT and hasattr(layout_data, 'get_polygons'):
                polys = layout_data.get_polygons(layer_num)
            else:
                polys = layout_data.get('polygons_by_layer', {}).get(layer_num, [])

            mpl_patches = []
            for pts in polys:
                if len(pts) >= 3:
                    patch = MplPolygon(pts, closed=True)
                    mpl_patches.append(patch)

            if mpl_patches:
                col = PatchCollection(mpl_patches,
                                       facecolor=color[:3], alpha=color[3],
                                       edgecolor='k', linewidth=0.5,
                                       label='Layer {}'.format(layer_num))
                ax.add_collection(col)

        # Auto-scale
        if HAS_LAYOUT and hasattr(layout_data, 'bounding_box') and layout_data.bounding_box:
            bb = layout_data.bounding_box
            margin = 0.05 * max(bb.width, bb.height)
            ax.set_xlim(bb.xmin - margin, bb.xmax + margin)
            ax.set_ylim(bb.ymin - margin, bb.ymax + margin)
        else:
            ax.autoscale()

        ax.set_aspect('equal')
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_title('GDS Layout')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_binary_grid(self, grid: np.ndarray, domain_size_nm: float,
                          ax=None, title: str = 'Mask Pattern') -> plt.Axes:
        """Plot binary mask grid."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(grid, origin='lower', cmap='gray',
                  extent=[0, domain_size_nm, 0, domain_size_nm],
                  vmin=0, vmax=1)
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_title(title)
        return ax

    def add_scale_bar(self, ax, length_nm: float, units: str = 'nm',
                       position: str = 'lower right') -> None:
        """Add scale bar to existing axes."""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        margin_x = 0.05 * x_range
        margin_y = 0.05 * y_range

        x0 = xlim[1] - margin_x - length_nm
        y0 = ylim[0] + margin_y

        ax.plot([x0, x0 + length_nm], [y0, y0], 'k-', linewidth=3)
        ax.text(x0 + 0.5 * length_nm, y0 + 0.02 * y_range,
                '{:.0f} {}'.format(length_nm, units),
                ha='center', va='bottom', fontsize=10)

    def zoom_to_region(self, ax, xmin: float, xmax: float,
                        ymin: float, ymax: float) -> None:
        """Zoom axes to specified region."""
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    def export_png(self, fig, filepath: str, dpi: int = 150) -> None:
        """Export figure to PNG file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
