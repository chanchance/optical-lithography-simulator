"""
Results panel: 2x2 plot grid (aerial image, mask, overlay, cross-section) + metrics table
+ process window mini-plot (CD vs defocus, last 5 results).
"""
import os
import numpy as np

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout,
        QTableWidget, QTableWidgetItem, QPushButton,
        QFileDialog, QMessageBox, QApplication
    )
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout,
        QTableWidget, QTableWidgetItem, QPushButton,
        QFileDialog, QMessageBox, QApplication
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

# Qt-neutral background colours used for figure faces
_FIG_FACE  = '#f0f0f0'
_AXES_FACE = '#fafafa'


class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        # Store (defocus_nm, cd_nm) tuples for the process-window strip (max 5)
        self._pw_history: list = []
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
        self.export_png_btn = QPushButton("Export PNG")
        self.export_pdf_btn = QPushButton("Export PDF")
        self.copy_table_btn = QPushButton("Copy Table")
        self.export_png_btn.clicked.connect(lambda: self._export("png"))
        self.export_pdf_btn.clicked.connect(lambda: self._export("pdf"))
        self.copy_table_btn.clicked.connect(self._copy_table_to_clipboard)
        btn_col.addWidget(self.export_png_btn)
        btn_col.addWidget(self.export_pdf_btn)
        btn_col.addWidget(self.copy_table_btn)
        btn_col.addStretch()
        bottom.addLayout(btn_col)

        layout.addLayout(bottom, stretch=1)

        self._clear_plots()
        self._draw_pw_strip()

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _clear_plots(self):
        titles = ["Aerial Image", "Mask", "Overlay", "Cross-section"]
        for ax, title in zip(self._axes.flat, titles):
            ax.clear()
            ax.set_facecolor(_AXES_FACE)
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_axis_off()
        self.figure.tight_layout(pad=1.2)
        self.canvas.draw()

    def _pixel_extent(self, result, shape):
        """Return (x_label, y_label, extent) using domain size if available."""
        try:
            domain = result.config["simulation"]["domain_size_nm"]
            half = domain / 2.0
            extent = [-half, half, -half, half]
            return "x (nm)", "y (nm)", extent
        except Exception:
            n = shape[1]
            m = shape[0]
            return "x (px)", "y (px)", [0, n, 0, m]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def show_result(self, result):
        self._result = result

        # Accumulate process-window history
        try:
            defocus = result.config["lithography"]["defocus_nm"]
        except Exception:
            defocus = result.metrics.get("defocus_nm", 0.0)
        self._pw_history.append((defocus, result.cd_nm))
        if len(self._pw_history) > 5:
            self._pw_history.pop(0)

        ax0, ax1, ax2, ax3 = self._axes.flat

        # ---- Aerial image ----
        ax0.clear()
        ax0.set_facecolor(_AXES_FACE)
        ax0.set_title("Aerial Image", fontsize=9, fontweight='bold')
        if result.aerial_image is not None:
            xl, yl, ext = self._pixel_extent(result, result.aerial_image.shape)
            im = ax0.imshow(
                result.aerial_image, cmap='inferno', origin='lower',
                vmin=0, vmax=1, extent=ext, aspect='auto'
            )
            cb = self.figure.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
            cb.set_label("Normalized Intensity", fontsize=7)
            cb.ax.tick_params(labelsize=6)
            ax0.set_xlabel(xl, fontsize=7)
            ax0.set_ylabel(yl, fontsize=7)
            ax0.tick_params(labelsize=6)
        else:
            ax0.set_axis_off()

        # ---- Mask ----
        ax1.clear()
        ax1.set_facecolor(_AXES_FACE)
        ax1.set_title("Mask", fontsize=9, fontweight='bold')
        if result.mask_grid is not None:
            xl, yl, ext = self._pixel_extent(result, result.mask_grid.shape)
            ax1.imshow(
                result.mask_grid, cmap='gray', origin='lower',
                extent=ext, aspect='auto'
            )
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
            ax2.imshow(
                result.aerial_image, cmap='inferno', origin='lower',
                alpha=0.7, extent=ext, aspect='auto'
            )
            ax2.contour(
                result.mask_grid, levels=[0.5],
                colors='cyan', linewidths=0.8,
                extent=ext
            )
            ax2.set_xlabel(xl, fontsize=7)
            ax2.set_ylabel(yl, fontsize=7)
            ax2.tick_params(labelsize=6)
        else:
            ax2.set_axis_off()

        # ---- Cross-section ----
        ax3.clear()
        ax3.set_facecolor(_AXES_FACE)
        ax3.set_title("Cross-section (center row)", fontsize=9, fontweight='bold')
        if result.aerial_image is not None:
            mid = result.aerial_image.shape[0] // 2
            profile = result.aerial_image[mid, :]
            nils = result.nils

            # Physical x axis if possible
            try:
                domain = result.config["simulation"]["domain_size_nm"]
                x = np.linspace(-domain / 2.0, domain / 2.0, len(profile))
                x_label = "Position (nm)"
            except Exception:
                x = np.linspace(0, 1, len(profile))
                x_label = "Position (norm.)"

            threshold = result.metrics.get('threshold', 0.3)

            # Shaded regions
            ax3.fill_between(x, 0, threshold, alpha=0.10, color='red',
                             label='Below threshold')
            ax3.fill_between(x, threshold, profile,
                             where=(profile > threshold),
                             alpha=0.15, color='blue', label='Above threshold')

            # Profile line
            ax3.plot(x, profile, color='#1a6bb5', linewidth=1.4, zorder=3)

            # Threshold line
            ax3.axhline(threshold, color='red', linestyle='--',
                        linewidth=0.9, label=f'Threshold {threshold:.2f}')

            # CD markers: find crossings where profile crosses threshold
            above = profile > threshold
            crossings = np.where(np.diff(above.astype(int)))[0]
            for ci in crossings:
                # Interpolate crossing x position
                x0, x1 = x[ci], x[ci + 1]
                y0, y1 = profile[ci], profile[ci + 1]
                if abs(y1 - y0) > 1e-10:
                    xc = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
                else:
                    xc = x0
                ax3.axvline(xc, color='green', linestyle=':', linewidth=0.8,
                            alpha=0.8)

            # NILS annotation
            ax3.text(
                0.02, 0.95, f'NILS={nils:.2f}',
                transform=ax3.transAxes,
                fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#aaa',
                          alpha=0.8)
            )

            ax3.set_ylim(0, 1)
            ax3.set_xlabel(x_label, fontsize=7)
            ax3.set_ylabel("Intensity", fontsize=7)
            ax3.tick_params(labelsize=6)
            ax3.legend(fontsize=6, loc='upper right')
        else:
            ax3.set_axis_off()

        self.figure.tight_layout(pad=1.2)
        self.canvas.draw()
        self._update_table(result)
        self._draw_pw_strip()

    # ------------------------------------------------------------------
    # Metrics table
    # ------------------------------------------------------------------

    def _update_table(self, result):
        rows = [
            ("CD (nm)",    "{:.2f}".format(result.cd_nm)),
            ("NILS",       "{:.3f}".format(result.nils)),
            ("Contrast",   "{:.3f}".format(result.contrast)),
            ("DOF (nm)",   "{:.1f}".format(result.metrics.get('dof_nm', 0.0))),
            ("I_max",      "{:.3f}".format(result.metrics.get('i_max', 0.0))),
            ("I_min",      "{:.3f}".format(result.metrics.get('i_min', 0.0))),
            ("Status",     result.status),
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
        """Copy metrics table as tab-separated values to the system clipboard."""
        lines = ["Metric\tValue"]
        for row in range(self.table.rowCount()):
            k = self.table.item(row, 0)
            v = self.table.item(row, 1)
            lines.append(f"{k.text() if k else ''}\t{v.text() if v else ''}")
        text = "\n".join(lines)
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

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
            cds = [p[1] for p in self._pw_history]
            ax.plot(defocuses, cds, 'o-', color='#c0392b', linewidth=1.4,
                    markersize=5, markerfacecolor='white',
                    markeredgewidth=1.5, zorder=3)
            # Shade ±10% CD band around the first (nominal) value
            if cds:
                cd_nom = cds[0]
                band = cd_nom * 0.10
                ax.axhspan(cd_nom - band, cd_nom + band,
                           alpha=0.12, color='green',
                           label='±10% CD band')
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
