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
    from PySide6.QtWidgets import (
        QDialog, QDialogButtonBox, QLineEdit, QPushButton, QFileDialog,
        QStackedWidget, QMessageBox,
    )
except ImportError:
    from PyQt5.QtWidgets import (  # type: ignore
        QDialog, QDialogButtonBox, QLineEdit, QPushButton, QFileDialog,
        QStackedWidget, QMessageBox,
    )

import matplotlib
# Do NOT call matplotlib.use('Agg') here — it conflicts with the Qt backend
# (backend_qtagg) which is already active when this dialog is opened from the
# main window.  The FigureCanvasQTAgg import below sets the correct backend.
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
        if config:
            self._load_from_config(config)
        self._update_preview()

    def _load_from_config(self, config):
        """Initialize controls from an existing config dict."""
        litho = config.get('lithography', {})
        illum = litho.get('illumination', {})
        illum_type = illum.get('type', 'annular').capitalize()
        # Map config keys to combo items
        type_map = {
            'circular': 'Circular', 'annular': 'Annular',
            'quadrupole': 'Quadrupole', 'quasar': 'Quasar',
            'dipole': 'Dipole', 'freeform': 'Freeform',
        }
        combo_text = type_map.get(illum.get('type', 'annular'), illum_type)
        idx = self.illum_combo.findText(combo_text)
        if idx >= 0:
            self.illum_combo.setCurrentIndex(idx)
        sigma_outer = illum.get('sigma_outer', 0.85)
        sigma_inner = illum.get('sigma_inner', 0.55)
        # Block signals to avoid triggering preview twice during init
        self.sigma_outer_sb.blockSignals(True)
        self.sigma_inner_sb.blockSignals(True)
        self.sigma_outer_sb.setValue(sigma_outer)
        self.sigma_inner_sb.setValue(sigma_inner)
        self.sigma_outer_sb.blockSignals(False)
        self.sigma_inner_sb.blockSignals(False)
        if illum.get('type') == 'dipole':
            sigma_c = illum.get('sigma_center', self._sigma_center_sb.value())
            self._sigma_center_sb.blockSignals(True)
            self._sigma_center_sb.setValue(sigma_c)
            self._sigma_center_sb.blockSignals(False)
            orientation = illum.get('orientation', 'x')
            idx_o = self._orientation_combo.findText(orientation)
            if idx_o >= 0:
                self._orientation_combo.blockSignals(True)
                self._orientation_combo.setCurrentIndex(idx_o)
                self._orientation_combo.blockSignals(False)
        if illum.get('type') == 'freeform':
            sigma_max = illum.get('sigma_max', self._sigma_max_spin.value())
            self._sigma_max_spin.blockSignals(True)
            self._sigma_max_spin.setValue(sigma_max)
            self._sigma_max_spin.blockSignals(False)
            if illum.get('expression'):
                self._freeform_expr.setText(illum['expression'])

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
        self.illum_combo.addItems(["Circular", "Annular", "Quadrupole", "Quasar", "Dipole", "Freeform"])
        self.illum_combo.setCurrentText("Annular")
        self.illum_combo.currentTextChanged.connect(self._on_type_changed)
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

        # Dipole-specific controls
        self._sigma_center_label = QLabel("σ center:")
        self._sigma_center_sb = QDoubleSpinBox()
        self._sigma_center_sb.setRange(0.01, 1.0)
        self._sigma_center_sb.setValue(0.70)
        self._sigma_center_sb.setDecimals(2)
        self._sigma_center_sb.setSingleStep(0.05)
        self._sigma_center_sb.valueChanged.connect(self._update_preview)
        ctrl.addRow(self._sigma_center_label, self._sigma_center_sb)
        self._sigma_center_label.setVisible(False)
        self._sigma_center_sb.setVisible(False)

        self._orientation_label = QLabel("Orientation:")
        self._orientation_combo = QComboBox()
        self._orientation_combo.addItems(["x", "y"])
        self._orientation_combo.currentTextChanged.connect(self._update_preview)
        ctrl.addRow(self._orientation_label, self._orientation_combo)
        self._orientation_label.setVisible(False)
        self._orientation_combo.setVisible(False)

        self._sigma_max_spin = QDoubleSpinBox()
        self._sigma_max_spin.setRange(0.01, 1.0)
        self._sigma_max_spin.setValue(1.0)
        self._sigma_max_spin.setDecimals(2)
        self._sigma_max_spin.setSingleStep(0.05)
        self._sigma_max_spin.valueChanged.connect(self._update_preview)
        self._sigma_max_label = QLabel("σ max (freeform):")
        ctrl.addRow(self._sigma_max_label, self._sigma_max_spin)
        self._sigma_max_label.setVisible(False)
        self._sigma_max_spin.setVisible(False)

        left_layout.addWidget(ctrl_group)

        # ── Freeform editor (hidden unless Freeform selected) ─────────────────
        self._freeform_group = QGroupBox("Freeform Editor")
        ff_layout = QVBoxLayout(self._freeform_group)
        ff_layout.setContentsMargins(10, 14, 10, 10)
        ff_layout.setSpacing(6)

        expr_row = QHBoxLayout()
        expr_lbl = QLabel("Expression:")
        self._freeform_expr = QLineEdit()
        self._freeform_expr.setPlaceholderText("e.g. (r > 0.3) & (r < 0.6)")
        self._freeform_expr.setText("(r > 0.3) & (r < 0.6)")
        apply_btn = QPushButton("Apply")
        apply_btn.setFixedWidth(54)
        apply_btn.setObjectName("secondary")
        apply_btn.clicked.connect(self._apply_freeform_expr)
        expr_row.addWidget(expr_lbl)
        expr_row.addWidget(self._freeform_expr, stretch=1)
        expr_row.addWidget(apply_btn)
        ff_layout.addLayout(expr_row)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load File")
        load_btn.clicked.connect(self._load_freeform_file)
        save_btn = QPushButton("Save Map")
        save_btn.clicked.connect(self._save_freeform_map)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(save_btn)
        btn_row.addStretch()
        ff_layout.addLayout(btn_row)

        left_layout.addWidget(self._freeform_group)
        self._freeform_group.setVisible(False)

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

        theme.apply_mpl_theme()
        self.figure = Figure(figsize=(5, 6), dpi=96)
        self.figure.subplots_adjust(hspace=0.35)
        self.ax = self.figure.add_subplot(211)       # pupil
        self.ax_scatter = self.figure.add_subplot(212)  # scatter
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        right_layout.addWidget(self.canvas)

        splitter.addWidget(right_widget)
        splitter.setSizes([220, 480])

        root_layout.addWidget(splitter, stretch=1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept)
        ok_btn = bb.button(QDialogButtonBox.Ok)
        if ok_btn:
            ok_btn.setProperty("class", "primary")
        root_layout.addWidget(bb)

        # Internal freeform state
        self._freeform_source = None

    # ── Type switch ───────────────────────────────────────────────────────────

    def _on_type_changed(self, text):
        is_freeform = (text == "Freeform")
        is_dipole = (text == "Dipole")
        self._freeform_group.setVisible(is_freeform)
        self._sigma_max_label.setVisible(is_freeform)
        self._sigma_max_spin.setVisible(is_freeform)
        self._sigma_center_label.setVisible(is_dipole)
        self._sigma_center_sb.setVisible(is_dipole)
        self._orientation_label.setVisible(is_dipole)
        self._orientation_combo.setVisible(is_dipole)
        self._update_preview()

    # ── Freeform actions ──────────────────────────────────────────────────────

    def _apply_freeform_expr(self):
        from core.source_model import FreeformSource
        expr = self._freeform_expr.text().strip()
        sigma_max = self._sigma_max_spin.value()
        try:
            self._freeform_source = FreeformSource.from_expression(
                expr, pupil_size=64, sigma_max=sigma_max)
            self._update_preview()
        except Exception as exc:
            QMessageBox.warning(self, "Expression Error", str(exc))

    def _load_freeform_file(self):
        from core.source_model import FreeformSource
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Source Map", "",
            "Source maps (*.npy *.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            self._freeform_source = FreeformSource.from_file(
                path, self._sigma_max_spin.value())
            self._update_preview()
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", str(exc))

    def _save_freeform_map(self):
        if self._freeform_source is None:
            QMessageBox.information(self, "Save Map", "No freeform map to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Source Map", "source_map.npy",
            "NumPy (*.npy);;CSV (*.csv)")
        if not path:
            return
        try:
            self._freeform_source.save(path)
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", str(exc))

    def _on_canvas_click(self, event):
        """Toggle pixel on/off in freeform map when user clicks the pupil axes."""
        if self.illum_combo.currentText() != "Freeform":
            return
        if event.inaxes is not self.ax:
            return
        from core.source_model import FreeformSource
        if self._freeform_source is None:
            self._freeform_source = FreeformSource(pupil_size=64)

        # Map click coordinates (pupil [-1,1]) to pixel indices
        sigma_max = self._freeform_source.sigma_max
        N = self._freeform_source.pupil_size
        # Axes display range is -1.2 to 1.2; map to [-sigma_max, sigma_max]
        cx, cy = event.xdata, event.ydata
        if cx is None or cy is None:
            return
        col = int((cx + sigma_max) / (2 * sigma_max) * N)
        row = int((sigma_max - cy) / (2 * sigma_max) * N)  # y-axis flipped
        if 0 <= row < N and 0 <= col < N:
            current = self._freeform_source.get_map()[row, col]
            self._freeform_source.set_pixel(row, col, 0.0 if current > 0.5 else 1.0)
            self._update_preview()

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

        if itype in ('annular', 'quadrupole', 'quasar', 'dipole'):
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

                elif itype == 'dipole':
                    sigma_c = self._sigma_center_sb.value()
                    orientation = self._orientation_combo.currentText()
                    rad = (s_o - s_i) / 2.0
                    pole_positions = [(sigma_c, 0.0), (-sigma_c, 0.0)] if orientation == 'x' \
                        else [(0.0, sigma_c), (0.0, -sigma_c)]
                    for cx, cy in pole_positions:
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
            'dipole':     '#b07aa1',
            'freeform':   '#59a14f',
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

        elif itype == 'dipole':
            sigma_c = self._sigma_center_sb.value()
            orientation = self._orientation_combo.currentText()
            pole_positions = [(sigma_c, 0.0), (-sigma_c, 0.0)] if orientation == 'x' \
                else [(0.0, sigma_c), (0.0, -sigma_c)]
            r = (s_o - s_i) / 2.0
            for cx, cy in pole_positions:
                ax.add_patch(mpatches.Circle((cx, cy), r, color=fill_color, alpha=0.75))

        elif itype == 'freeform':
            self._draw_freeform_pupil(ax)

        # Sigma reference overlays (skip for freeform)
        if itype != 'freeform':
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

        if itype == 'freeform':
            kx_arr, ky_arr, n_pts = self._sample_freeform_points()
        else:
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

    def _draw_freeform_pupil(self, ax):
        """Render the freeform intensity map as an imshow on the pupil axes."""
        if self._freeform_source is None:
            return
        pupil_map = self._freeform_source.get_map()
        sigma_max = self._freeform_source.sigma_max
        extent = [-sigma_max, sigma_max, -sigma_max, sigma_max]
        ax.imshow(
            pupil_map, origin='lower', extent=extent,
            cmap='viridis', alpha=0.85, vmin=0, vmax=1,
            aspect='equal', interpolation='nearest'
        )

    def _sample_freeform_points(self):
        """Return (kx_arr, ky_arr, n_pts) for the freeform scatter plot."""
        if self._freeform_source is None:
            return np.array([]), np.array([]), 0
        sx, sy, _w = self._freeform_source.get_source_arrays()
        return sx, sy, len(sx)

    def get_illumination_config(self):
        current_type = self.illum_combo.currentText()
        illum_map = {
            "Circular":   "circular",
            "Annular":    "annular",
            "Quadrupole": "quadrupole",
            "Quasar":     "quasar",
            "Dipole":     "dipole",
        }
        if current_type == 'Dipole':
            return {
                'type': 'dipole',
                'sigma_center': self._sigma_center_sb.value(),
                'sigma_outer': self.sigma_outer_sb.value(),
                'sigma_inner': self.sigma_inner_sb.value(),
                'orientation': self._orientation_combo.currentText(),
            }
        if current_type == 'Freeform':
            return {
                'type': 'freeform',
                'expression': self._freeform_expr.text(),
                'sigma_max': self._sigma_max_spin.value(),
                'pupil_size': 64,
            }
        return {
            "type":        illum_map[current_type],
            "sigma_outer": self.sigma_outer_sb.value(),
            "sigma_inner": self.sigma_inner_sb.value(),
        }
