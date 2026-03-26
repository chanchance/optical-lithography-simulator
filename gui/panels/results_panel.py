"""
Results panel: 5-panel GridSpec plot (aerial image, mask, overlay, cross-section, process window)
+ metrics table. Interactive gauge: user draws lines on the aerial image to define cross-section
analysis gauges.
"""
import numpy as np

from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QPushButton, QFileDialog, QMessageBox, QApplication,
    QLabel, QFont, QFrame, Qt, QCheckBox, QComboBox,
)
from gui import theme
from gui.gauge_manager import GaugeManager, GAUGE_COLORS as _GAUGE_COLORS

import matplotlib
# Do NOT call matplotlib.use('Agg') — it conflicts with the Qt backend
# (backend_qtagg) already active in the main window.
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import matplotlib.ticker as ticker

class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._pw_history: list = []
        self._pw_nominal_cd: float = 0.0   # CD from first run (reference for band)
        self._gauge_mgr = GaugeManager()
        self._cb_aerial = None
        self._cb_wf = None
        self._show_resist_edge: bool = True
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ---- Main figure with GridSpec(2, 6) ----
        self.figure = Figure(figsize=(12, 8), dpi=theme.MPL_DPI, constrained_layout=True)
        self.figure.patch.set_facecolor(theme.BG_PRIMARY)

        gs = GridSpec(2, 8, figure=self.figure)
        self.ax_aerial  = self.figure.add_subplot(gs[0, 0:2])
        self.ax_mask    = self.figure.add_subplot(gs[0, 2:4])
        self.ax_overlay = self.figure.add_subplot(gs[0, 4:6])
        self.ax_wf      = self.figure.add_subplot(gs[0, 6:8])
        self.ax_cs      = self.figure.add_subplot(gs[1, 0:4])
        self.ax_pw      = self.figure.add_subplot(gs[1, 4:8])

        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

        canvas_frame = QFrame()
        canvas_frame.setStyleSheet(
            "QFrame { border: 1px solid %s; border-radius: 8px; background: %s; }"
            % (theme.BORDER, theme.BG_PRIMARY)
        )
        frame_layout = QVBoxLayout(canvas_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.canvas)
        layout.addWidget(canvas_frame, stretch=4)

        # ---- Bottom row: table + buttons ----
        bottom = QHBoxLayout()

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(160)
        self.table.setAlternatingRowColors(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        bottom.addWidget(self.table, stretch=2)

        btn_frame = QFrame()
        btn_frame.setStyleSheet(
            "QFrame { background: %s; border-left: 1px solid %s; padding: 8px; }"
            % (theme.BG_SECONDARY, theme.BORDER)
        )
        btn_col = QVBoxLayout(btn_frame)
        btn_col.setSpacing(4)

        self.export_png_btn   = QPushButton("Export PNG")
        self.export_pdf_btn   = QPushButton("Export PDF")
        self.copy_table_btn   = QPushButton("Copy Table")
        self.add_gauge_btn    = QPushButton("Add Gauge")
        self.clear_gauges_btn = QPushButton("Clear Gauges")
        self.gauge_status     = QLabel("")

        self.add_gauge_btn.setCheckable(True)
        self.add_gauge_btn.setToolTip(
            "Enter gauge mode: click two points on the aerial image\n"
            "to define a cross-section analysis line.")

        self.lock_h_chk = QCheckBox("Lock H")
        self.lock_h_chk.setToolTip(
            "Constrain gauge to horizontal direction\n(both points share the same Y).")
        self.lock_v_chk = QCheckBox("Lock V")
        self.lock_v_chk.setToolTip(
            "Constrain gauge to vertical direction\n(both points share the same X).")

        self.show_resist_chk = QCheckBox("Show Resist Edge")
        self.show_resist_chk.setChecked(True)
        self.show_resist_chk.setToolTip(
            "Overlay resist edge contour on the aerial image overlay.")

        self.export_png_btn.clicked.connect(lambda: self._export("png"))
        self.export_pdf_btn.clicked.connect(lambda: self._export("pdf"))
        self.copy_table_btn.clicked.connect(self._copy_table_to_clipboard)
        self.add_gauge_btn.clicked.connect(self._toggle_gauge_mode)
        self.clear_gauges_btn.clicked.connect(self._clear_gauges)
        self.lock_h_chk.toggled.connect(self._on_lock_h_toggled)
        self.lock_v_chk.toggled.connect(self._on_lock_v_toggled)
        self.show_resist_chk.toggled.connect(self._on_resist_edge_toggled)

        pol_lbl = QLabel("Polarization:")
        pol_lbl.setObjectName("caption")
        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems([
            "Scalar", "X-linear", "Y-linear", "TE", "TM", "Circular-L", "Circular-R"])
        self.polarization_combo.setToolTip(
            "Simulation polarization mode.\n"
            "Non-scalar modes use VectorImagingEngine (core.vector_imaging).")

        for w in (self.export_png_btn, self.export_pdf_btn, self.copy_table_btn,
                  self.add_gauge_btn, self.clear_gauges_btn,
                  self.lock_h_chk, self.lock_v_chk,
                  self.show_resist_chk, self.gauge_status,
                  pol_lbl, self.polarization_combo):
            btn_col.addWidget(w)
        btn_col.addStretch()
        bottom.addWidget(btn_frame)

        layout.addLayout(bottom, stretch=1)

        self._clear_plots()
        self._draw_pw_strip()

    # ------------------------------------------------------------------
    # Theme helpers
    # ------------------------------------------------------------------

    def _style_ax(self, ax, title):
        ax.set_facecolor(theme.BG_SECONDARY)
        ax.set_title(title, fontsize=theme.MPL_TITLE, fontweight='600',
                     color=theme.TEXT_PRIMARY, pad=6)
        ax.tick_params(colors=theme.TEXT_TERTIARY, labelsize=theme.MPL_TICK)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(theme.BORDER)
        ax.spines['bottom'].set_color(theme.BORDER)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(theme.TEXT_TERTIARY)

    # ------------------------------------------------------------------
    # Gauge logic
    # ------------------------------------------------------------------

    def _toggle_gauge_mode(self, checked):
        self._gauge_mgr.toggle_mode(checked)
        if checked:
            self.canvas.setCursor(Qt.CrossCursor)
        else:
            self.canvas.unsetCursor()
            if self._result is not None:
                self._redraw_aerial()
                self.canvas.draw()
        self.gauge_status.setText(self._gauge_mgr.status_message())

    def _on_canvas_click(self, event):
        if not self._gauge_mgr.is_active():
            return
        if event.inaxes is not self.ax_aerial:
            return
        if self._result is None or self._result.aerial_image is None:
            return

        x, y = event.xdata, event.ydata
        gauge = self._gauge_mgr.on_click(x, y)
        self.gauge_status.setText(self._gauge_mgr.status_message())

        if gauge is None:
            # First click — redraw with pending marker
            self.ax_aerial.plot(x, y, 'o', color='white', markersize=7,
                                markeredgecolor='black', markeredgewidth=1.2, zorder=12)
            self.canvas.draw_idle()
        else:
            # Second click — extract profile and update plots
            profile, distances = self._gauge_mgr.extract_profile(
                self._result.aerial_image, gauge['p1'], gauge['p2'])
            gauge['profile'] = profile
            gauge['distances'] = distances
            self._redraw_aerial()
            self._draw_cross_section()
            self.canvas.draw()

    def _clear_gauges(self):
        if self._gauge_mgr.is_active():
            self.add_gauge_btn.setChecked(False)
            self._gauge_mgr.toggle_mode(False)
            self.canvas.unsetCursor()
        self._gauge_mgr.clear()
        self.gauge_status.setText("")
        if self._result is not None:
            self._redraw_aerial()
            self._draw_cross_section()
            self.canvas.draw()

    def _on_lock_h_toggled(self, checked):
        self._gauge_mgr.lock_y = checked
        if checked:
            self.lock_v_chk.setChecked(False)

    def _on_lock_v_toggled(self, checked):
        self._gauge_mgr.lock_x = checked
        if checked:
            self.lock_h_chk.setChecked(False)

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _pixel_extent(self, result, shape):
        try:
            domain = result.config["simulation"]["domain_size_nm"]
            half = domain / 2.0
            extent = [-half, half, -half, half]
            return "x (nm)", "y (nm)", extent
        except Exception:
            n = shape[1]
            m = shape[0]
            return "x (px)", "y (px)", [0, n, 0, m]

    def _clear_plots(self):
        for ax, title in [
            (self.ax_aerial,  "Aerial Image"),
            (self.ax_mask,    "Mask"),
            (self.ax_overlay, "Overlay"),
            (self.ax_wf,      "Wavefront Error (waves)"),
            (self.ax_cs,      "Cross-section"),
            (self.ax_pw,      "Process Window"),
        ]:
            ax.clear()
            self._style_ax(ax, title)
            ax.set_axis_off()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Subplot updaters (callers handle canvas.draw except _draw_pw_strip)
    # ------------------------------------------------------------------

    def _redraw_aerial(self):
        """Redraw aerial image (ax_aerial) with gauge overlays. Caller draws canvas."""
        ax = self.ax_aerial

        if self._cb_aerial is not None:
            try:
                self._cb_aerial.remove()
            except Exception:
                pass
            self._cb_aerial = None

        ax.clear()
        self._style_ax(ax, "Aerial Image")

        result = self._result
        if result is None or result.aerial_image is None:
            ax.set_axis_off()
            return

        xl, yl, ext = self._pixel_extent(result, result.aerial_image.shape)
        self._gauge_mgr.set_extent(ext)

        ai = result.aerial_image
        vmax = max(1.0, float(ai.max()))   # dose_factor > 1 can push values above 1
        im = ax.imshow(
            ai, cmap='inferno', origin='lower',
            vmin=0, vmax=vmax, extent=ext, aspect='auto'
        )
        self._cb_aerial = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb_label = "Intensity" if vmax > 1.0 else "Normalized Intensity"
        self._cb_aerial.set_label(cb_label, fontsize=theme.MPL_ANNOT)
        self._cb_aerial.ax.tick_params(labelsize=theme.MPL_ANNOT)
        ax.set_xlabel(xl, fontsize=theme.MPL_LABEL)
        ax.set_ylabel(yl, fontsize=theme.MPL_LABEL)
        ax.tick_params(labelsize=theme.MPL_TICK)

        for g in self._gauge_mgr.get_gauges():
            p1, p2 = g['p1'], g['p2']
            color  = g['color']
            idx    = g['idx']
            dist   = g['distances'][-1]

            ax.annotate(
                '', xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, mutation_scale=12),
                zorder=8
            )
            ax.plot(*p1, 'o', color=color, markersize=5,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=9)
            ax.plot(*p2, 's', color=color, markersize=5,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=9)
            mx = 0.5 * (p1[0] + p2[0])
            my = 0.5 * (p1[1] + p2[1])
            ax.text(mx, my, 'G{}: {:.0f}nm'.format(idx, dist),
                    color=color, fontsize=theme.MPL_ANNOT, fontweight='bold',
                    ha='center', va='bottom', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.15', fc='black',
                              ec='none', alpha=0.55))

        pending = self._gauge_mgr.get_pending()
        if pending is not None:
            ax.plot(*pending, 'o', color='white',
                    markersize=7, markeredgecolor='black',
                    markeredgewidth=1.2, zorder=11)

    def _draw_cross_section(self):
        """Draw cross-section (ax_cs): gauge profiles when available, else center row."""
        ax = self.ax_cs
        ax.clear()
        self._style_ax(ax, "Cross-section")

        result = self._result
        if result is None or result.aerial_image is None:
            ax.set_axis_off()
            return

        threshold = result.metrics.get('threshold', 0.3)

        gauges = self._gauge_mgr.get_gauges()
        if gauges:
            ax.set_title("Cross-section (gauges)", fontsize=theme.MPL_TITLE,
                         fontweight='600', color=theme.TEXT_PRIMARY, pad=6)
            ax.axhline(threshold, color='red', linestyle='--', linewidth=0.9,
                       label='Threshold {:.2f}'.format(threshold), zorder=2)

            for g in gauges:
                distances = g['distances']
                profile   = g['profile']
                color     = g['color']
                idx       = g['idx']

                ax.plot(distances, profile, color=color, linewidth=1.3,
                        label='G{}'.format(idx), zorder=3)

                above = profile > threshold
                crossings = np.where(np.diff(above.astype(int)))[0]
                crossing_xs = []
                for ci in crossings:
                    d0, d1 = distances[ci], distances[ci + 1]
                    y0, y1 = profile[ci], profile[ci + 1]
                    xc = d0 + (threshold - y0) * (d1 - d0) / (y1 - y0) \
                         if abs(y1 - y0) > 1e-10 else d0
                    ax.axvline(xc, color=color, linestyle=':', linewidth=0.8,
                               alpha=0.7, zorder=2)
                    crossing_xs.append(xc)

                if len(crossing_xs) >= 2:
                    cd = crossing_xs[1] - crossing_xs[0]
                    mid_x = 0.5 * (crossing_xs[0] + crossing_xs[1])
                    ax.annotate(
                        'G{} CD={:.0f}nm'.format(idx, cd),
                        xy=(mid_x, threshold),
                        xytext=(0, 8), textcoords='offset points',
                        fontsize=theme.MPL_ANNOT, color=color, ha='center',
                    )

            all_profiles = [g['profile'] for g in gauges]
            y_max = max(1.0, float(max(p.max() for p in all_profiles)))
            ax.set_ylim(0, y_max * 1.05)
            ax.set_xlabel("Distance along gauge (nm)", fontsize=theme.MPL_LABEL)
            ax.set_ylabel("Intensity", fontsize=theme.MPL_LABEL)
            ax.tick_params(labelsize=theme.MPL_TICK)
            ax.legend(fontsize=theme.MPL_LEGEND, loc='upper right', ncol=2)

        else:
            # Choose scan direction with more threshold crossings so the plot
            # is meaningful for both horizontal and vertical line patterns.
            ai = result.aerial_image
            t_val = result.config.get('analysis', {}).get('cd_threshold', 0.30)
            row_p = ai[ai.shape[0] // 2, :]
            col_p = ai[:, ai.shape[1] // 2]

            def _nc(p):
                return sum(1 for i in range(len(p) - 1)
                           if (p[i] - t_val) * (p[i + 1] - t_val) <= 0)

            if _nc(col_p) > _nc(row_p):
                profile = col_p
                scan_label = "Cross-section (center column)"
            else:
                profile = row_p
                scan_label = "Cross-section (center row)"

            ax.set_title(scan_label, fontsize=theme.MPL_TITLE,
                         fontweight='600', color=theme.TEXT_PRIMARY, pad=6)
            nils = result.nils

            try:
                domain = result.config["simulation"]["domain_size_nm"]
                x = np.linspace(-domain / 2.0, domain / 2.0, len(profile))
                x_label = "Position (nm)"
            except Exception:
                x = np.linspace(0, 1, len(profile))
                x_label = "Position (norm.)"

            ax.fill_between(x, 0, threshold, alpha=0.10, color='red',
                            label='Below threshold')
            ax.fill_between(x, threshold, profile,
                            where=(profile > threshold),
                            alpha=0.15, color='blue', label='Above threshold')
            ax.plot(x, profile, color='#1a6bb5', linewidth=1.4, zorder=3)
            ax.axhline(threshold, color='red', linestyle='--', linewidth=0.9,
                       label='Threshold {:.2f}'.format(threshold))

            above = profile > threshold
            crossings = np.where(np.diff(above.astype(int)))[0]
            for ci in crossings:
                x0, x1 = x[ci], x[ci + 1]
                y0, y1 = profile[ci], profile[ci + 1]
                xc = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0) \
                     if abs(y1 - y0) > 1e-10 else x0
                ax.axvline(xc, color='green', linestyle=':', linewidth=0.8, alpha=0.8)

            ax.text(0.02, 0.95, 'NILS={:.2f}'.format(nils),
                    transform=ax.transAxes, fontsize=theme.MPL_TICK, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              ec=theme.BORDER, alpha=0.8))
            ax.set_ylim(0, max(1.0, float(profile.max())) * 1.05)
            ax.set_xlabel(x_label, fontsize=theme.MPL_LABEL)
            ax.set_ylabel("Intensity", fontsize=theme.MPL_LABEL)
            ax.tick_params(labelsize=theme.MPL_TICK)
            ax.legend(fontsize=theme.MPL_LEGEND, loc='upper right')

    def _on_resist_edge_toggled(self, checked):
        self._show_resist_edge = checked
        if self._result is not None:
            self._redraw_overlay(self._result)
            self.canvas.draw_idle()

    def _redraw_overlay(self, result):
        """Redraw ax_overlay: aerial image + mask contour + optional resist edge."""
        ax = self.ax_overlay
        ax.clear()
        self._style_ax(ax, "Overlay")

        if result.aerial_image is None:
            ax.set_axis_off()
            return

        xl, yl, ext = self._pixel_extent(result, result.aerial_image.shape)
        ax.imshow(result.aerial_image, cmap='inferno', origin='lower',
                  alpha=0.7, extent=ext, aspect='auto')

        if result.mask_grid is not None:
            ny, nx = result.mask_grid.shape
            cx = np.linspace(ext[0], ext[1], nx)
            cy = np.linspace(ext[2], ext[3], ny)
            ax.contour(cx, cy, result.mask_grid, levels=[0.5],
                       colors='cyan', linewidths=0.8)

        resist = getattr(result, 'resist_image', None)
        if resist is not None and self._show_resist_edge:
            ry, rx = resist.shape
            rx_arr = np.linspace(ext[0], ext[1], rx)
            ry_arr = np.linspace(ext[2], ext[3], ry)
            ax.contour(rx_arr, ry_arr, resist, levels=[0.5],
                       colors='white', linewidths=1.5)

        ax.set_xlabel(xl, fontsize=theme.MPL_LABEL)
        ax.set_ylabel(yl, fontsize=theme.MPL_LABEL)
        ax.tick_params(labelsize=theme.MPL_TICK)

    def _draw_wavefront(self, result):
        """Draw 2D wavefront error map (ax_wf) from Zernike aberration config."""
        ax = self.ax_wf

        if self._cb_wf is not None:
            try:
                self._cb_wf.remove()
            except Exception:
                pass
            self._cb_wf = None

        ax.clear()
        self._style_ax(ax, "Wavefront Error (waves)")

        try:
            aber = result.config["lithography"]["aberrations"]
        except Exception:
            aber = {}

        zernike_list = aber.get("zernike", []) if isinstance(aber, dict) else []

        # Hide map when all coefficients are zero or absent
        if not zernike_list or all(v == 0.0 for v in zernike_list):
            ax.set_axis_off()
            return

        from core.aberrations import ZernikeAberration
        za = ZernikeAberration.from_list(zernike_list)

        # Build normalized pupil grid
        N = 256
        lin = np.linspace(-1.0, 1.0, N)
        KX, KY = np.meshgrid(lin, lin)
        phase_rad = za.pupil_phase(KX, KY)
        wf_waves = phase_rad / (2.0 * np.pi)   # convert radians → waves

        # Mask outside pupil (leave as NaN so imshow shows background color)
        rho = np.sqrt(KX**2 + KY**2)
        wf_waves[rho > 1.0] = np.nan

        # Symmetric color limits centered on 0
        vmax = np.nanmax(np.abs(wf_waves))
        vmax = max(vmax, 1e-6)

        im = ax.imshow(
            wf_waves, cmap='RdBu_r', origin='lower',
            extent=[-1, 1, -1, 1], aspect='equal',
            vmin=-vmax, vmax=vmax,
        )
        self._cb_wf = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self._cb_wf.set_label("waves", fontsize=theme.MPL_ANNOT)
        self._cb_wf.ax.tick_params(labelsize=theme.MPL_ANNOT)

        # NA boundary circle
        circle = Circle((0, 0), 1.0, fill=False,
                         edgecolor='white', linewidth=1.2, linestyle='--')
        ax.add_patch(circle)

        ax.set_xlabel("Pupil x (norm.)", fontsize=theme.MPL_LABEL)
        ax.set_ylabel("Pupil y (norm.)", fontsize=theme.MPL_LABEL)
        ax.tick_params(labelsize=theme.MPL_TICK)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_axis_on()

        # RMS over pupil (stored for metrics table)
        pupil_vals = wf_waves[rho <= 1.0]
        self._wfe_rms = float(np.sqrt(np.nanmean(pupil_vals**2))) if pupil_vals.size else 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def show_result(self, result):
        self._result = result

        if self._gauge_mgr.is_active():
            self.add_gauge_btn.setChecked(False)
            self._gauge_mgr.toggle_mode(False)
            self.canvas.unsetCursor()
        self._gauge_mgr.clear()
        self.gauge_status.setText("")

        try:
            defocus = result.config["lithography"]["defocus_nm"]
        except Exception:
            defocus = result.metrics.get("defocus_nm", 0.0)
        self._pw_history.append((defocus, result.cd_nm))
        # Store nominal CD from the first run as the stable reference for
        # the +/-10% band — never shift the reference as history grows.
        if self._pw_nominal_cd == 0.0 and result.cd_nm > 0:
            self._pw_nominal_cd = result.cd_nm

        # ---- Aerial image ----
        self._redraw_aerial()

        # ---- Mask ----
        self.ax_mask.clear()
        self._style_ax(self.ax_mask, "Mask")
        if result.mask_grid is not None:
            xl, yl, ext = self._pixel_extent(result, result.mask_grid.shape)
            self.ax_mask.imshow(result.mask_grid, cmap='gray', origin='lower',
                                extent=ext, aspect='auto')
            self.ax_mask.set_xlabel(xl, fontsize=theme.MPL_LABEL)
            self.ax_mask.set_ylabel(yl, fontsize=theme.MPL_LABEL)
            self.ax_mask.tick_params(labelsize=theme.MPL_TICK)
        else:
            self.ax_mask.set_axis_off()

        # ---- Overlay ----
        self._redraw_overlay(result)

        # ---- Wavefront error ----
        self._wfe_rms = 0.0
        self._draw_wavefront(result)

        # ---- Cross-section ----
        self._draw_cross_section()

        self._update_table(result)
        self._draw_pw_strip()

    # ------------------------------------------------------------------
    # Metrics table
    # ------------------------------------------------------------------

    def _update_table(self, result):
        wfe_rms = getattr(self, '_wfe_rms', 0.0)

        # Resist model type from config
        try:
            resist_model = result.config["resist"]["model"]
        except Exception:
            resist_model = None
        resist_label = resist_model.capitalize() if resist_model else "—"

        # Pattern area fraction from resist_image
        resist = getattr(result, 'resist_image', None)
        if resist is not None and resist.size > 0:
            pattern_pct = float(np.mean(resist > 0.5) * 100.0)
            pattern_str = "{:.1f}%".format(pattern_pct)
        else:
            pattern_str = "—"

        rows = [
            ("CD (nm)",       "{:.2f}".format(result.cd_nm)),
            ("NILS",          "{:.3f}".format(result.nils)),
            ("Contrast",      "{:.3f}".format(result.contrast)),
            ("DOF (nm)",      "{:.1f}".format(result.metrics.get('dof_nm', 0.0))),
            ("I_max",         "{:.3f}".format(result.metrics.get('i_max', 0.0))),
            ("I_min",         "{:.3f}".format(result.metrics.get('i_min', 0.0))),
            ("WFE RMS",       "{:.4f} \u03bb".format(wfe_rms)),
            ("Resist",        resist_label),
            ("Pattern Area",  pattern_str),
            ("Status",        result.status),
        ]
        self.table.setRowCount(len(rows))
        bold_font = QFont()
        bold_font.setBold(True)
        for i, (k, v) in enumerate(rows):
            key_item = QTableWidgetItem(k)
            key_item.setFont(bold_font)
            key_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            val_item = QTableWidgetItem(v)
            val_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
            self.table.setItem(i, 0, key_item)
            self.table.setItem(i, 1, val_item)
        self.table.resizeColumnsToContents()

    def _copy_table_to_clipboard(self):
        lines = ["Metric\tValue"]
        for row in range(self.table.rowCount()):
            k = self.table.item(row, 0)
            v = self.table.item(row, 1)
            lines.append("{}\t{}".format(
                k.text() if k else '', v.text() if v else ''))
        QApplication.clipboard().setText("\n".join(lines))

    # ------------------------------------------------------------------
    # Process window (ax_pw)
    # ------------------------------------------------------------------

    def _draw_pw_strip(self):
        ax = self.ax_pw
        ax.clear()
        self._style_ax(ax, "Process Window (CD vs Defocus)")

        if self._pw_history:
            defocuses = [p[0] for p in self._pw_history]
            cds       = [p[1] for p in self._pw_history]
            ax.plot(defocuses, cds, 'o-', color='#c0392b', linewidth=1.4,
                    markersize=5, markerfacecolor='white',
                    markeredgewidth=1.5, zorder=3)
            cd_nom = self._pw_nominal_cd if self._pw_nominal_cd > 0 else (cds[0] if cds else 0)
            if cd_nom > 0:
                band = cd_nom * 0.10
                ax.axhspan(cd_nom - band, cd_nom + band,
                           alpha=0.12, color='green', label='+-10% CD band')
            ax.set_xlabel("Defocus (nm)", fontsize=theme.MPL_LABEL)
            ax.set_ylabel("CD (nm)", fontsize=theme.MPL_LABEL)
            ax.tick_params(labelsize=theme.MPL_TICK)
            ax.legend(fontsize=theme.MPL_LEGEND, loc='upper right')
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        else:
            ax.text(0.5, 0.5, 'Run simulations at different defocus values',
                    ha='center', va='center', fontsize=theme.MPL_LABEL,
                    color=theme.TEXT_TERTIARY, transform=ax.transAxes)
            ax.set_axis_off()

        self.canvas.draw()

    # ------------------------------------------------------------------
    # Export
    def get_polarization(self) -> str:
        """Return the currently selected polarization mode string."""
        return self.polarization_combo.currentText()

    # ------------------------------------------------------------------

    def _export(self, fmt):
        if self._result is None:
            QMessageBox.information(self, "No results", "Run a simulation first.")
            return

        if fmt in ('png', 'pdf'):
            ext_filter = "PNG files (*.png)" if fmt == "png" else "PDF files (*.pdf)"
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "results." + fmt, ext_filter)
            if path:
                try:
                    self.figure.savefig(path, dpi=150, bbox_inches='tight',
                                        facecolor=theme.BG_PRIMARY)
                except Exception as e:
                    QMessageBox.warning(self, "Export Error", str(e))
            return

        filter_str = (
            "CSV files (*.csv);;"
            "HDF5 files (*.h5 *.hdf5);;"
            "PNG image (*.png);;"
            "Text report (*.txt)"
        )
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Results", "results.csv", filter_str)
        if not path:
            return

        if 'CSV' in selected_filter:
            export_fmt = 'csv'
        elif 'HDF5' in selected_filter:
            export_fmt = 'hdf5'
        elif 'PNG' in selected_filter:
            export_fmt = 'png'
        else:
            export_fmt = 'report'

        try:
            from fileio.results_exporter import ResultsExporter
            exp = ResultsExporter()
            if export_fmt == 'csv':
                exp.export_csv(self._result, path)
            elif export_fmt == 'hdf5':
                exp.export_hdf5(self._result, path)
            elif export_fmt == 'png':
                exp.export_png(self._result, path, dpi=150)
            else:
                exp.export_report(self._result, path)
            QMessageBox.information(self, "Export", "Saved to:\n{}".format(path))
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))
