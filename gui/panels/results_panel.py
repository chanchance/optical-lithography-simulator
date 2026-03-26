"""
Results panel: 2x2 plot grid (aerial image, mask, overlay, cross-section) + metrics table
+ process window mini-plot (CD vs defocus, last 5 results).
Interactive gauge: user draws lines on the aerial image to define cross-section analysis gauges.
"""
import os
import numpy as np

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout,
        QTableWidget, QTableWidgetItem, QPushButton,
        QFileDialog, QMessageBox, QApplication, QLabel
    )
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout,
        QTableWidget, QTableWidgetItem, QPushButton,
        QFileDialog, QMessageBox, QApplication, QLabel
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

_FIG_FACE  = '#f0f0f0'
_AXES_FACE = '#fafafa'
# Bright gauge colours that stand out on inferno colormap
_GAUGE_COLORS = ['#00ff88', '#ff6b35', '#4ecdc4', '#ffe66d',
                 '#e040fb', '#80cbc4', '#f48fb1', '#fff176']


class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._pw_history: list = []
        # --- Gauge state ---
        self._gauges: list = []       # list of gauge dicts
        self._gauge_mode: bool = False
        self._pending_p1 = None       # first click in data (nm) coords
        self._extent = None           # current aerial image extent [x0,x1,y0,y1]
        self._cb_aerial = None        # current aerial colorbar (for removal)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ---- Main 2x2 figure ----
        self.figure = Figure(figsize=(10, 8), dpi=90)
        self.figure.patch.set_facecolor(_FIG_FACE)
        self._axes = self.figure.subplots(2, 2)
        for ax in self._axes.flat:
            ax.set_facecolor(_AXES_FACE)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        layout.addWidget(self.canvas, stretch=4)

        # ---- Process window mini-figure ----
        self.pw_figure = Figure(figsize=(10, 2), dpi=90)
        self.pw_figure.patch.set_facecolor(_FIG_FACE)
        self.pw_ax = self.pw_figure.add_subplot(111)
        self.pw_ax.set_facecolor(_AXES_FACE)
        self.pw_canvas = FigureCanvas(self.pw_figure)
        self.pw_canvas.setMaximumHeight(130)
        layout.addWidget(self.pw_canvas, stretch=1)

        # ---- Bottom row: table + buttons ----
        bottom = QHBoxLayout()

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(160)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        bottom.addWidget(self.table, stretch=2)

        btn_col = QVBoxLayout()
        self.export_png_btn  = QPushButton("Export PNG")
        self.export_pdf_btn  = QPushButton("Export PDF")
        self.copy_table_btn  = QPushButton("Copy Table")
        self.add_gauge_btn   = QPushButton("Add Gauge")
        self.clear_gauges_btn = QPushButton("Clear Gauges")
        self.gauge_status    = QLabel("")
        self.gauge_status.setStyleSheet("color: #666; font-size: 10px;")

        self.add_gauge_btn.setCheckable(True)
        self.add_gauge_btn.setToolTip(
            "Enter gauge mode: click two points on the aerial image\n"
            "to define a cross-section analysis line.")

        self.export_png_btn.clicked.connect(lambda: self._export("png"))
        self.export_pdf_btn.clicked.connect(lambda: self._export("pdf"))
        self.copy_table_btn.clicked.connect(self._copy_table_to_clipboard)
        self.add_gauge_btn.clicked.connect(self._toggle_gauge_mode)
        self.clear_gauges_btn.clicked.connect(self._clear_gauges)

        for w in (self.export_png_btn, self.export_pdf_btn, self.copy_table_btn,
                  self.add_gauge_btn, self.clear_gauges_btn, self.gauge_status):
            btn_col.addWidget(w)
        btn_col.addStretch()
        bottom.addLayout(btn_col)

        layout.addLayout(bottom, stretch=1)

        self._clear_plots()
        self._draw_pw_strip()

    # ------------------------------------------------------------------
    # Gauge logic
    # ------------------------------------------------------------------

    def _toggle_gauge_mode(self, checked):
        self._gauge_mode = checked
        if checked:
            self._pending_p1 = None
            self.gauge_status.setText("Click point 1 on aerial image")
            self.canvas.setCursor(Qt.CrossCursor)
        else:
            self._pending_p1 = None
            self.gauge_status.setText("")
            self.canvas.unsetCursor()
            if self._result is not None:
                self._redraw_aerial()
                self.figure.tight_layout(pad=1.2)
                self.canvas.draw()

    def _on_canvas_click(self, event):
        if not self._gauge_mode:
            return
        ax0 = self._axes[0, 0]
        if event.inaxes is not ax0:
            return
        if self._result is None or self._result.aerial_image is None:
            return

        x, y = event.xdata, event.ydata

        if self._pending_p1 is None:
            self._pending_p1 = (x, y)
            self.gauge_status.setText("Click point 2 on aerial image")
            # Show pending marker without full redraw
            ax0.plot(x, y, 'o', color='white', markersize=7,
                     markeredgecolor='black', markeredgewidth=1.2, zorder=12)
            self.canvas.draw_idle()
        else:
            p1 = self._pending_p1
            p2 = (x, y)
            self._pending_p1 = None

            profile, distances = self._extract_gauge_profile(
                self._result.aerial_image, p1, p2, self._extent)

            gauge = {
                'p1': p1, 'p2': p2,
                'profile': profile,
                'distances': distances,
                'color': _GAUGE_COLORS[len(self._gauges) % len(_GAUGE_COLORS)],
                'idx': len(self._gauges) + 1,
            }
            self._gauges.append(gauge)

            self.gauge_status.setText(
                "{} gauge(s). Click point 1 for next gauge.".format(len(self._gauges)))

            self._redraw_aerial()
            self._draw_cross_section()
            self.figure.tight_layout(pad=1.2)
            self.canvas.draw()

    def _extract_gauge_profile(self, image, p1, p2, extent, n_points=256):
        """
        Extract intensity profile along line from p1→p2 (data/nm coords).
        Returns (profile array, distances_nm array).
        """
        from scipy.ndimage import map_coordinates

        if extent is None:
            extent = [0, image.shape[1], 0, image.shape[0]]

        x0_d, x1_d = extent[0], extent[1]
        y0_d, y1_d = extent[2], extent[3]
        n_rows, n_cols = image.shape

        def to_pixel(xd, yd):
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
        total_dist = np.sqrt(dx**2 + dy**2)
        distances = np.linspace(0.0, total_dist, n_points)

        return profile, distances

    def _clear_gauges(self):
        self._gauges.clear()
        self._pending_p1 = None
        self.gauge_status.setText("")
        if self._gauge_mode:
            self.add_gauge_btn.setChecked(False)
            self._gauge_mode = False
            self.canvas.unsetCursor()
        if self._result is not None:
            self._redraw_aerial()
            self._draw_cross_section()
            self.figure.tight_layout(pad=1.2)
            self.canvas.draw()

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
        titles = ["Aerial Image", "Mask", "Overlay", "Cross-section"]
        for ax, title in zip(self._axes.flat, titles):
            ax.clear()
            ax.set_facecolor(_AXES_FACE)
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_axis_off()
        self.figure.tight_layout(pad=1.2)
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Subplot updaters (callers handle tight_layout + canvas.draw)
    # ------------------------------------------------------------------

    def _redraw_aerial(self):
        """Redraw aerial image (ax0) with gauge overlays. Caller draws canvas."""
        ax0 = self._axes[0, 0]

        # Remove previous colorbar to avoid accumulation
        if self._cb_aerial is not None:
            try:
                self._cb_aerial.remove()
            except Exception:
                pass
            self._cb_aerial = None

        ax0.clear()
        ax0.set_facecolor(_AXES_FACE)
        ax0.set_title("Aerial Image", fontsize=9, fontweight='bold')

        result = self._result
        if result is None or result.aerial_image is None:
            ax0.set_axis_off()
            return

        xl, yl, ext = self._pixel_extent(result, result.aerial_image.shape)
        self._extent = ext

        im = ax0.imshow(
            result.aerial_image, cmap='inferno', origin='lower',
            vmin=0, vmax=1, extent=ext, aspect='auto'
        )
        self._cb_aerial = self.figure.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
        self._cb_aerial.set_label("Normalized Intensity", fontsize=7)
        self._cb_aerial.ax.tick_params(labelsize=6)
        ax0.set_xlabel(xl, fontsize=7)
        ax0.set_ylabel(yl, fontsize=7)
        ax0.tick_params(labelsize=6)

        # Draw completed gauges
        for g in self._gauges:
            p1, p2 = g['p1'], g['p2']
            color  = g['color']
            idx    = g['idx']
            dist   = g['distances'][-1]

            ax0.annotate(
                '', xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, mutation_scale=12),
                zorder=8
            )
            ax0.plot(*p1, 'o', color=color, markersize=5,
                     markeredgecolor='white', markeredgewidth=0.8, zorder=9)
            ax0.plot(*p2, 's', color=color, markersize=5,
                     markeredgecolor='white', markeredgewidth=0.8, zorder=9)
            mx = 0.5 * (p1[0] + p2[0])
            my = 0.5 * (p1[1] + p2[1])
            ax0.text(mx, my, 'G{}: {:.0f}nm'.format(idx, dist),
                     color=color, fontsize=6, fontweight='bold',
                     ha='center', va='bottom', zorder=10,
                     bbox=dict(boxstyle='round,pad=0.15', fc='black',
                               ec='none', alpha=0.55))

        # Pending first-click marker
        if self._pending_p1 is not None:
            ax0.plot(*self._pending_p1, 'o', color='white',
                     markersize=7, markeredgecolor='black',
                     markeredgewidth=1.2, zorder=11)

    def _draw_cross_section(self):
        """Draw cross-section (ax3): gauge profiles when available, else center row."""
        ax3 = self._axes[1, 1]
        ax3.clear()
        ax3.set_facecolor(_AXES_FACE)

        result = self._result
        if result is None or result.aerial_image is None:
            ax3.set_title("Cross-section", fontsize=9, fontweight='bold')
            ax3.set_axis_off()
            return

        threshold = result.metrics.get('threshold', 0.3)

        if self._gauges:
            # ---- Gauge profiles ----
            ax3.set_title("Cross-section (gauges)", fontsize=9, fontweight='bold')
            ax3.axhline(threshold, color='red', linestyle='--', linewidth=0.9,
                        label='Threshold {:.2f}'.format(threshold), zorder=2)

            for g in self._gauges:
                distances = g['distances']
                profile   = g['profile']
                color     = g['color']
                idx       = g['idx']

                ax3.plot(distances, profile, color=color, linewidth=1.3,
                         label='G{}'.format(idx), zorder=3)

                # CD crossings for this gauge
                above = profile > threshold
                crossings = np.where(np.diff(above.astype(int)))[0]
                crossing_xs = []
                for ci in crossings:
                    d0, d1 = distances[ci], distances[ci + 1]
                    y0, y1 = profile[ci], profile[ci + 1]
                    xc = d0 + (threshold - y0) * (d1 - d0) / (y1 - y0) \
                         if abs(y1 - y0) > 1e-10 else d0
                    ax3.axvline(xc, color=color, linestyle=':', linewidth=0.8,
                                alpha=0.7, zorder=2)
                    crossing_xs.append(xc)

                if len(crossing_xs) >= 2:
                    cd = crossing_xs[1] - crossing_xs[0]
                    mid_x = 0.5 * (crossing_xs[0] + crossing_xs[1])
                    ax3.annotate(
                        'G{} CD={:.0f}nm'.format(idx, cd),
                        xy=(mid_x, threshold),
                        xytext=(0, 8), textcoords='offset points',
                        fontsize=6, color=color, ha='center',
                    )

            ax3.set_ylim(0, 1)
            ax3.set_xlabel("Distance along gauge (nm)", fontsize=7)
            ax3.set_ylabel("Intensity", fontsize=7)
            ax3.tick_params(labelsize=6)
            ax3.legend(fontsize=6, loc='upper right', ncol=2)

        else:
            # ---- Default: center row cross-section ----
            ax3.set_title("Cross-section (center row)", fontsize=9, fontweight='bold')
            mid = result.aerial_image.shape[0] // 2
            profile = result.aerial_image[mid, :]
            nils = result.nils

            try:
                domain = result.config["simulation"]["domain_size_nm"]
                x = np.linspace(-domain / 2.0, domain / 2.0, len(profile))
                x_label = "Position (nm)"
            except Exception:
                x = np.linspace(0, 1, len(profile))
                x_label = "Position (norm.)"

            ax3.fill_between(x, 0, threshold, alpha=0.10, color='red',
                             label='Below threshold')
            ax3.fill_between(x, threshold, profile,
                             where=(profile > threshold),
                             alpha=0.15, color='blue', label='Above threshold')
            ax3.plot(x, profile, color='#1a6bb5', linewidth=1.4, zorder=3)
            ax3.axhline(threshold, color='red', linestyle='--', linewidth=0.9,
                        label='Threshold {:.2f}'.format(threshold))

            above = profile > threshold
            crossings = np.where(np.diff(above.astype(int)))[0]
            for ci in crossings:
                x0, x1 = x[ci], x[ci + 1]
                y0, y1 = profile[ci], profile[ci + 1]
                xc = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0) \
                     if abs(y1 - y0) > 1e-10 else x0
                ax3.axvline(xc, color='green', linestyle=':', linewidth=0.8, alpha=0.8)

            ax3.text(0.02, 0.95, 'NILS={:.2f}'.format(nils),
                     transform=ax3.transAxes, fontsize=8, va='top',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#aaa', alpha=0.8))
            ax3.set_ylim(0, 1)
            ax3.set_xlabel(x_label, fontsize=7)
            ax3.set_ylabel("Intensity", fontsize=7)
            ax3.tick_params(labelsize=6)
            ax3.legend(fontsize=6, loc='upper right')

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def show_result(self, result):
        self._result = result

        # Clear gauges when a new result arrives (domain may have changed)
        self._gauges.clear()
        self._pending_p1 = None
        if self._gauge_mode:
            self.add_gauge_btn.setChecked(False)
            self._gauge_mode = False
            self.canvas.unsetCursor()
        self.gauge_status.setText("")

        # Process window history
        try:
            defocus = result.config["lithography"]["defocus_nm"]
        except Exception:
            defocus = result.metrics.get("defocus_nm", 0.0)
        self._pw_history.append((defocus, result.cd_nm))
        if len(self._pw_history) > 5:
            self._pw_history.pop(0)

        ax0, ax1, ax2, ax3 = self._axes.flat

        # ---- Aerial image (uses _redraw_aerial for colorbar management) ----
        self._redraw_aerial()

        # ---- Mask ----
        ax1.clear()
        ax1.set_facecolor(_AXES_FACE)
        ax1.set_title("Mask", fontsize=9, fontweight='bold')
        if result.mask_grid is not None:
            xl, yl, ext = self._pixel_extent(result, result.mask_grid.shape)
            ax1.imshow(result.mask_grid, cmap='gray', origin='lower',
                       extent=ext, aspect='auto')
            ax1.set_xlabel(xl, fontsize=7)
            ax1.set_ylabel(yl, fontsize=7)
            ax1.tick_params(labelsize=6)
        else:
            ax1.set_axis_off()

        # ---- Overlay ----
        ax2.clear()
        ax2.set_facecolor(_AXES_FACE)
        ax2.set_title("Overlay", fontsize=9, fontweight='bold')
        if result.aerial_image is not None and result.mask_grid is not None:
            xl, yl, ext = self._pixel_extent(result, result.aerial_image.shape)
            ax2.imshow(result.aerial_image, cmap='inferno', origin='lower',
                       alpha=0.7, extent=ext, aspect='auto')
            ax2.contour(result.mask_grid, levels=[0.5],
                        colors='cyan', linewidths=0.8, extent=ext)
            ax2.set_xlabel(xl, fontsize=7)
            ax2.set_ylabel(yl, fontsize=7)
            ax2.tick_params(labelsize=6)
        else:
            ax2.set_axis_off()

        # ---- Cross-section ----
        self._draw_cross_section()

        self.figure.tight_layout(pad=1.2)
        self.canvas.draw()
        self._update_table(result)
        self._draw_pw_strip()

    # ------------------------------------------------------------------
    # Metrics table
    # ------------------------------------------------------------------

    def _update_table(self, result):
        rows = [
            ("CD (nm)",  "{:.2f}".format(result.cd_nm)),
            ("NILS",     "{:.3f}".format(result.nils)),
            ("Contrast", "{:.3f}".format(result.contrast)),
            ("DOF (nm)", "{:.1f}".format(result.metrics.get('dof_nm', 0.0))),
            ("I_max",    "{:.3f}".format(result.metrics.get('i_max', 0.0))),
            ("I_min",    "{:.3f}".format(result.metrics.get('i_min', 0.0))),
            ("Status",   result.status),
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
    # Process window strip
    # ------------------------------------------------------------------

    def _draw_pw_strip(self):
        ax = self.pw_ax
        ax.clear()
        ax.set_facecolor(_AXES_FACE)
        ax.set_title("Process Window (CD vs Defocus)", fontsize=8, fontweight='bold')

        if self._pw_history:
            defocuses = [p[0] for p in self._pw_history]
            cds       = [p[1] for p in self._pw_history]
            ax.plot(defocuses, cds, 'o-', color='#c0392b', linewidth=1.4,
                    markersize=5, markerfacecolor='white',
                    markeredgewidth=1.5, zorder=3)
            if cds:
                cd_nom = cds[0]
                band = cd_nom * 0.10
                ax.axhspan(cd_nom - band, cd_nom + band,
                           alpha=0.12, color='green', label='±10% CD band')
            ax.set_xlabel("Defocus (nm)", fontsize=7)
            ax.set_ylabel("CD (nm)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6, loc='upper right')
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        else:
            ax.text(0.5, 0.5, 'Run simulations at different defocus values',
                    ha='center', va='center', fontsize=8, color='#888',
                    transform=ax.transAxes)
            ax.set_axis_off()

        self.pw_figure.tight_layout(pad=0.8)
        self.pw_canvas.draw()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self, fmt):
        if self._result is None:
            QMessageBox.information(self, "No results", "Run a simulation first.")
            return
        ext = "PNG files (*.png)" if fmt == "png" else "PDF files (*.pdf)"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "results." + fmt, ext)
        if path:
            try:
                self.figure.savefig(path, dpi=150, bbox_inches='tight',
                                    facecolor=_FIG_FACE)
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))
