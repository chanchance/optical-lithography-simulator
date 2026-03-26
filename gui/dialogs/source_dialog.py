"""
Source dialog: interactive k-space illumination pupil preview.
"""
import math
import numpy as np

from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDoubleSpinBox, QComboBox, QSplitter, QLabel, QGroupBox, Qt,
)
from gui import theme

try:
    from PySide6.QtWidgets import QDialog, QDialogButtonBox
except ImportError:
    from PyQt5.QtWidgets import QDialog, QDialogButtonBox  # type: ignore

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


class SourceDialog(QDialog):
    """Preview illumination pupil in k-space with interactive controls."""

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle("Illumination Source Preview")
        self.resize(720, 520)
        self._build_ui()
        self._update_preview()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # ── Splitter: controls left, canvas right ─────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)

        # Left: controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(10)

        ctrl_group = QGroupBox("Illumination Parameters")
        ctrl = QFormLayout(ctrl_group)
        ctrl.setSpacing(8)
        ctrl.setContentsMargins(10, 14, 10, 10)

        self.illum_combo = QComboBox()
        self.illum_combo.addItems(["Circular", "Annular", "Quadrupole", "Quasar"])
        self.illum_combo.setCurrentText("Annular")
        self.illum_combo.currentTextChanged.connect(self._update_preview)
        ctrl.addRow("Type:", self.illum_combo)

        self.sigma_outer_sb = QDoubleSpinBox()
        self.sigma_outer_sb.setRange(0.01, 1.0)
        self.sigma_outer_sb.setValue(0.85)
        self.sigma_outer_sb.setDecimals(2)
        self.sigma_outer_sb.setSingleStep(0.05)
        self.sigma_outer_sb.valueChanged.connect(self._update_preview)
        ctrl.addRow("σ outer:", self.sigma_outer_sb)

        self.sigma_inner_sb = QDoubleSpinBox()
        self.sigma_inner_sb.setRange(0.0, 0.99)
        self.sigma_inner_sb.setValue(0.55)
        self.sigma_inner_sb.setDecimals(2)
        self.sigma_inner_sb.setSingleStep(0.05)
        self.sigma_inner_sb.valueChanged.connect(self._update_preview)
        ctrl.addRow("σ inner:", self.sigma_inner_sb)

        left_layout.addWidget(ctrl_group)

        # Source point count label
        info_group = QGroupBox("Source Info")
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(10, 14, 10, 10)
        self.point_count_label = QLabel("N=0 source points")
        self.point_count_label.setObjectName("caption")
        info_layout.addWidget(self.point_count_label)
        left_layout.addWidget(info_group)

        # Legend
        legend_group = QGroupBox("Legend")
        legend_layout = QVBoxLayout(legend_group)
        legend_layout.setContentsMargins(10, 14, 10, 10)
        legend_layout.setSpacing(4)

        def _legend_row(color, style, label):
            row = QHBoxLayout()
            swatch = QLabel()
            swatch.setFixedSize(28, 12)
            if style == 'solid':
                swatch.setStyleSheet(
                    "background: {}; border: 1px solid #888;".format(color))
            else:
                swatch.setStyleSheet(
                    "border-top: 2px dashed {}; background: transparent;".format(color))
            lbl = QLabel(label)
            lbl.setObjectName("caption")
            row.addWidget(swatch)
            row.addWidget(lbl)
            row.addStretch()
            return row

        legend_layout.addLayout(_legend_row('#000000', 'solid', 'NA boundary'))
        legend_layout.addLayout(_legend_row('#cc3333', 'dashed', 'σ outer'))
        legend_layout.addLayout(_legend_row('#3366cc', 'dashed', 'σ inner'))
        legend_layout.addLayout(_legend_row('#aaaaaa', 'dashed', 'σ = 0.25/0.5/0.75'))
        left_layout.addWidget(legend_group)
        left_layout.addStretch()

        splitter.addWidget(left_widget)

        # Right: matplotlib canvas (pupil + scatter stacked)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(0)

        self.figure = Figure(figsize=(5, 6), dpi=96)
        self.figure.subplots_adjust(hspace=0.35)
        self.ax = self.figure.add_subplot(211)       # pupil
        self.ax_scatter = self.figure.add_subplot(212)  # scatter
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        splitter.addWidget(right_widget)
        splitter.setSizes([220, 480])

        root_layout.addWidget(splitter, stretch=1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept)
        root_layout.addWidget(bb)

    # ── Grid / reference ring helpers ─────────────────────────────────────────

    def _draw_sigma_grid(self, ax):
        """Draw light dashed reference rings at sigma = 0.25, 0.5, 0.75, 1.0."""
        for sigma, label_str in [(0.25, '0.25'), (0.5, '0.5'), (0.75, '0.75')]:
            ring = mpatches.Circle(
                (0, 0), sigma, fill=False,
                edgecolor='#bbbbbb', linewidth=0.8, linestyle='--'
            )
            ax.add_patch(ring)
            ax.text(
                sigma * 0.707 + 0.02, sigma * 0.707 + 0.02,
                'σ={}'.format(label_str),
                fontsize=7, color='#999999', ha='left', va='bottom'
            )

    def _draw_sigma_refs(self, ax, s_o, s_i, itype):
        """Draw dashed sigma_outer (red) and sigma_inner (blue) reference circles."""
        outer_ring = mpatches.Circle(
            (0, 0), s_o, fill=False,
            edgecolor='#cc3333', linewidth=1.2, linestyle='--', zorder=5
        )
        ax.add_patch(outer_ring)

        if itype in ('annular', 'quadrupole', 'quasar'):
            inner_ring = mpatches.Circle(
                (0, 0), s_i, fill=False,
                edgecolor='#3366cc', linewidth=1.2, linestyle='--', zorder=5
            )
            ax.add_patch(inner_ring)

    # ── Discrete source point sampling ────────────────────────────────────────

    def _sample_source_points(self, itype, s_o, s_i, n_ring=12):
        """Return (kx_arr, ky_arr) arrays of discrete source point centers."""
        pts = []
        # Sample on a polar grid within the illuminated region
        for r_idx in range(1, n_ring + 1):
            r = r_idx / n_ring
            n_pts = max(6, int(2 * math.pi * r * n_ring))
            for a_idx in range(n_pts):
                ang = 2 * math.pi * a_idx / n_pts
                kx = r * math.cos(ang)
                ky = r * math.sin(ang)
                rr = math.sqrt(kx ** 2 + ky ** 2)

                if itype == 'circular':
                    if rr <= s_o:
                        pts.append((kx, ky))

                elif itype == 'annular':
                    if s_i <= rr <= s_o:
                        pts.append((kx, ky))

                elif itype == 'quadrupole':
                    offset = (s_o + s_i) / 2.0
                    rad = (s_o - s_i) / 2.0
                    for cx, cy in [(offset, 0), (-offset, 0),
                                   (0, offset), (0, -offset)]:
                        if (kx - cx) ** 2 + (ky - cy) ** 2 <= rad ** 2:
                            pts.append((kx, ky))
                            break

                elif itype == 'quasar':
                    offset = (s_o + s_i) / 2.0
                    rad = (s_o - s_i) / 2.0
                    for deg in [45, 135, 225, 315]:
                        a = math.radians(deg)
                        cx = offset * math.cos(a)
                        cy = offset * math.sin(a)
                        if (kx - cx) ** 2 + (ky - cy) ** 2 <= rad ** 2:
                            pts.append((kx, ky))
                            break

        if pts:
            kx_arr = np.array([p[0] for p in pts])
            ky_arr = np.array([p[1] for p in pts])
        else:
            kx_arr = np.array([])
            ky_arr = np.array([])
        return kx_arr, ky_arr

    # ── Preview update ────────────────────────────────────────────────────────

    def _update_preview(self):
        itype = self.illum_combo.currentText().lower()
        s_o = self.sigma_outer_sb.value()
        s_i = self.sigma_inner_sb.value()

        # Clamp s_i < s_o
        if s_i >= s_o:
            s_i = max(0.0, s_o - 0.05)

        # Color map per illumination type
        fill_colors = {
            'circular':   '#4e79a7',
            'annular':    '#4e79a7',
            'quadrupole': '#f28e2b',
            'quasar':     '#e15759',
        }
        fill_color = fill_colors.get(itype, '#4e79a7')

        # ── Pupil plot ────────────────────────────────────────────────────────
        self.figure.patch.set_facecolor(theme.BG_PRIMARY)

        ax = self.ax
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("kx / NA", fontsize=theme.MPL_LABEL)
        ax.set_ylabel("ky / NA", fontsize=theme.MPL_LABEL)
        ax.set_title("Illumination Pupil (k-space)", fontsize=theme.MPL_TITLE, fontweight='bold')
        ax.set_facecolor(theme.BG_SECONDARY)
        ax.tick_params(labelsize=theme.MPL_TICK)
        for spine in ax.spines.values():
            spine.set_edgecolor(theme.BORDER)

        # Reference grid
        self._draw_sigma_grid(ax)

        # NA boundary
        ax.add_patch(mpatches.Circle(
            (0, 0), 1.0, fill=False, edgecolor='#222222', linewidth=1.8
        ))

        # Filled illumination region
        if itype == 'circular':
            ax.add_patch(mpatches.Circle(
                (0, 0), s_o, color=fill_color, alpha=0.72))

        elif itype == 'annular':
            ax.add_patch(mpatches.Circle(
                (0, 0), s_o, color=fill_color, alpha=0.72))
            ax.add_patch(mpatches.Circle(
                (0, 0), s_i, color='#f4f6f9', alpha=1.0))

        elif itype == 'quadrupole':
            offset = (s_o + s_i) / 2.0
            r = (s_o - s_i) / 2.0
            for cx, cy in [(offset, 0), (-offset, 0),
                           (0, offset), (0, -offset)]:
                ax.add_patch(mpatches.Circle(
                    (cx, cy), r, color=fill_color, alpha=0.75))

        elif itype == 'quasar':
            offset = (s_o + s_i) / 2.0
            r = (s_o - s_i) / 2.0
            for deg in [45, 135, 225, 315]:
                ang = math.radians(deg)
                ax.add_patch(mpatches.Circle(
                    (offset * math.cos(ang), offset * math.sin(ang)),
                    r, color=fill_color, alpha=0.75))

        # Sigma reference overlays
        self._draw_sigma_refs(ax, s_o, s_i, itype)

        ax.grid(False)

        # ── Scatter plot ──────────────────────────────────────────────────────
        ax_s = self.ax_scatter
        ax_s.clear()
        ax_s.set_aspect('equal')
        ax_s.set_xlim(-1.2, 1.2)
        ax_s.set_ylim(-1.2, 1.2)
        ax_s.set_xlabel("kx / NA", fontsize=theme.MPL_LABEL)
        ax_s.set_ylabel("ky / NA", fontsize=theme.MPL_LABEL)
        ax_s.set_facecolor(theme.BG_SECONDARY)
        ax_s.tick_params(labelsize=theme.MPL_TICK)
        for spine in ax_s.spines.values():
            spine.set_edgecolor(theme.BORDER)

        kx_arr, ky_arr = self._sample_source_points(itype, s_o, s_i)
        n_pts = len(kx_arr)

        if n_pts > 0:
            ax_s.scatter(
                kx_arr, ky_arr,
                s=8, c=fill_color, alpha=0.55, linewidths=0
            )

        # NA boundary on scatter
        ax_s.add_patch(mpatches.Circle(
            (0, 0), 1.0, fill=False, edgecolor='#222222', linewidth=1.5
        ))
        ax_s.set_title(
            "Discrete Source Points  (N={})".format(n_pts),
            fontsize=theme.MPL_TITLE, fontweight='bold'
        )

        self.point_count_label.setText("N={} source points".format(n_pts))

        self.figure.tight_layout(pad=1.2)
        self.canvas.draw()

    def get_illumination_config(self):
        illum_map = {
            "Circular":   "circular",
            "Annular":    "annular",
            "Quadrupole": "quadrupole",
            "Quasar":     "quasar",
        }
        return {
            "type":        illum_map[self.illum_combo.currentText()],
            "sigma_outer": self.sigma_outer_sb.value(),
            "sigma_inner": self.sigma_inner_sb.value(),
        }
