"""
Gauge manager for cross-section analysis on aerial images.
Handles gauge state, axis locking, and profile extraction.
"""
import math
import numpy as np
from typing import Optional

GAUGE_COLORS = [
    '#00ff88', '#ff6b35', '#4ecdc4', '#ffe66d',
    '#e040fb', '#80cbc4', '#f48fb1', '#fff176'
]
MIN_GAUGE_LENGTH = 1.0  # nm (or pixels in fallback mode)


class GaugeManager:
    """Manages interactive gauge drawing and profile extraction."""

    def __init__(self):
        self.gauges: list = []
        self.pending_p1: Optional[tuple] = None
        self.gauge_mode: bool = False
        self.lock_x: bool = False   # vertical gauge (fixed X)
        self.lock_y: bool = False   # horizontal gauge (fixed Y)
        self.extent: Optional[list] = None  # [x0, x1, y0, y1] in data coords
        self._status: str = ""

    def toggle_mode(self, active: bool) -> None:
        """Enable/disable gauge drawing mode."""
        self.gauge_mode = active
        self.pending_p1 = None
        self._status = "Click point 1 on aerial image" if active else ""

    def cancel_pending(self) -> None:
        """Cancel a pending first gauge point."""
        self.pending_p1 = None
        self._status = "Click point 1 on aerial image" if self.gauge_mode else ""

    def on_click(self, x: float, y: float) -> Optional[dict]:
        """
        Handle a click on the aerial image.
        First call: stores p1. Second call: creates gauge.
        Returns completed gauge dict or None.
        """
        if not self.gauge_mode:
            return None

        if self.pending_p1 is None:
            self.pending_p1 = (x, y)
            self._status = "Click point 2 on aerial image"
            return None

        p1 = self.pending_p1
        p2 = (x, y)

        # Apply axis lock
        if self.lock_y:   # horizontal line -> force same Y
            p2 = (p2[0], p1[1])
        elif self.lock_x:  # vertical line -> force same X
            p2 = (p1[0], p2[1])

        # Validate length
        length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if length < MIN_GAUGE_LENGTH:
            self.pending_p1 = None
            self._status = "Gauge too short (< {:.0f} units). Try again.".format(MIN_GAUGE_LENGTH)
            return None

        self.pending_p1 = None
        idx = len(self.gauges) + 1
        gauge = {
            'p1': p1,
            'p2': p2,
            'idx': idx,
            'color': GAUGE_COLORS[(idx - 1) % len(GAUGE_COLORS)],
            'profile': None,
            'distances': None,
            'length': length,
        }
        self.gauges.append(gauge)
        self._status = "{} gauge(s). Click point 1 for next.".format(len(self.gauges))
        return gauge

    def extract_profile(self, image: np.ndarray, p1: tuple, p2: tuple,
                        n_points: int = 256) -> tuple:
        """
        Extract intensity profile along line p1->p2.
        p1, p2 are in data coordinates (nm or pixels).
        Returns (profile: ndarray, distances: ndarray).
        """
        from scipy.ndimage import map_coordinates

        extent = self.extent or [0, image.shape[1], 0, image.shape[0]]
        x0_d, x1_d = extent[0], extent[1]
        y0_d, y1_d = extent[2], extent[3]
        n_rows, n_cols = image.shape

        def to_pixel(xd: float, yd: float):
            col = (xd - x0_d) / (x1_d - x0_d) * n_cols
            row = (yd - y0_d) / (y1_d - y0_d) * n_rows
            return row, col

        r1, c1 = to_pixel(p1[0], p1[1])
        r2, c2 = to_pixel(p2[0], p2[1])

        rows_i = np.clip(np.linspace(r1, r2, n_points), 0, n_rows - 1)
        cols_i = np.clip(np.linspace(c1, c2, n_points), 0, n_cols - 1)

        profile = map_coordinates(image, [rows_i, cols_i], order=1, mode='nearest')

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        total_dist = math.sqrt(dx**2 + dy**2)
        distances = np.linspace(0.0, total_dist, n_points)

        return profile, distances

    def sample_intensity(self, image: np.ndarray, x: float, y: float) -> float:
        """Sample image intensity at data coordinate (x, y)."""
        from scipy.ndimage import map_coordinates
        extent = self.extent or [0, image.shape[1], 0, image.shape[0]]
        x0_d, x1_d = extent[0], extent[1]
        y0_d, y1_d = extent[2], extent[3]
        n_rows, n_cols = image.shape
        col = (x - x0_d) / (x1_d - x0_d) * n_cols
        row = (y - y0_d) / (y1_d - y0_d) * n_rows
        row = float(np.clip(row, 0, n_rows - 1))
        col = float(np.clip(col, 0, n_cols - 1))
        result = map_coordinates(image, [[row], [col]], order=1, mode='nearest')
        return float(result[0])

    def set_extent(self, extent: Optional[list]) -> None:
        self.extent = extent

    def clear(self) -> None:
        self.gauges.clear()
        self.pending_p1 = None
        self._status = ""

    def get_gauges(self) -> list:
        return self.gauges

    def get_pending(self) -> Optional[tuple]:
        return self.pending_p1

    def is_active(self) -> bool:
        return self.gauge_mode

    def status_message(self) -> str:
        return self._status
