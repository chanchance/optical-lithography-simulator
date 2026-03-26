"""
Results panel: 2x2 plot grid (aerial image, mask, overlay, cross-section) + metrics table.
"""
import os
import numpy as np

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout,
        QTableWidget, QTableWidgetItem, QPushButton,
        QFileDialog, QMessageBox
    )
    from PySide6.QtCore import Qt
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout,
        QTableWidget, QTableWidgetItem, QPushButton,
        QFileDialog, QMessageBox
    )
    from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.figure = Figure(figsize=(10, 8), dpi=90)
        self._axes = self.figure.subplots(2, 2)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=3)

        bottom = QHBoxLayout()

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(160)
        bottom.addWidget(self.table, stretch=2)

        btn_col = QVBoxLayout()
        self.export_png_btn = QPushButton("Export PNG")
        self.export_pdf_btn = QPushButton("Export PDF")
        self.export_png_btn.clicked.connect(lambda: self._export("png"))
        self.export_pdf_btn.clicked.connect(lambda: self._export("pdf"))
        btn_col.addWidget(self.export_png_btn)
        btn_col.addWidget(self.export_pdf_btn)
        btn_col.addStretch()
        bottom.addLayout(btn_col)

        layout.addLayout(bottom, stretch=1)
        self._clear_plots()

    def _clear_plots(self):
        for ax, title in zip(self._axes.flat,
                             ["Aerial Image", "Mask", "Overlay", "Cross-section"]):
            ax.clear()
            ax.set_title(title, fontsize=9)
            ax.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw()

    def show_result(self, result):
        self._result = result
        ax0, ax1, ax2, ax3 = self._axes.flat

        ax0.clear()
        ax0.set_title("Aerial Image", fontsize=9)
        if result.aerial_image is not None:
            im = ax0.imshow(result.aerial_image, cmap='hot', origin='lower', vmin=0, vmax=1)
            self.figure.colorbar(im, ax=ax0, fraction=0.046)
            ax0.set_xlabel("x (px)", fontsize=7)
        else:
            ax0.set_axis_off()

        ax1.clear()
        ax1.set_title("Mask", fontsize=9)
        if result.mask_grid is not None:
            ax1.imshow(result.mask_grid, cmap='gray', origin='lower')
            ax1.set_xlabel("x (px)", fontsize=7)
        else:
            ax1.set_axis_off()

        ax2.clear()
        ax2.set_title("Overlay", fontsize=9)
        if result.aerial_image is not None and result.mask_grid is not None:
            ax2.imshow(result.aerial_image, cmap='hot', origin='lower', alpha=0.7)
            ax2.contour(result.mask_grid, levels=[0.5], colors='cyan', linewidths=0.8)
            ax2.set_xlabel("x (px)", fontsize=7)
        else:
            ax2.set_axis_off()

        ax3.clear()
        ax3.set_title("Cross-section (center row)", fontsize=9)
        if result.aerial_image is not None:
            mid = result.aerial_image.shape[0] // 2
            profile = result.aerial_image[mid, :]
            x = np.linspace(0, 1, len(profile))
            ax3.plot(x, profile, 'b-', linewidth=1.2)
            threshold = result.metrics.get('threshold', 0.3)
            ax3.axhline(threshold, color='r', linestyle='--', linewidth=0.8, label='Threshold')
            ax3.set_ylim(0, 1)
            ax3.set_xlabel("Position (norm.)", fontsize=7)
            ax3.set_ylabel("Intensity", fontsize=7)
            ax3.legend(fontsize=7)
        else:
            ax3.set_axis_off()

        self.figure.tight_layout()
        self.canvas.draw()
        self._update_table(result)

    def _update_table(self, result):
        rows = [
            ("CD (nm)", "{:.2f}".format(result.cd_nm)),
            ("NILS", "{:.3f}".format(result.nils)),
            ("Contrast", "{:.3f}".format(result.contrast)),
            ("DOF (nm)", "{:.1f}".format(result.metrics.get('dof_nm', 0.0))),
            ("I_max", "{:.3f}".format(result.metrics.get('i_max', 0.0))),
            ("I_min", "{:.3f}".format(result.metrics.get('i_min', 0.0))),
            ("Status", result.status),
        ]
        self.table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(k))
            self.table.setItem(i, 1, QTableWidgetItem(v))

    def _export(self, fmt):
        if self._result is None:
            QMessageBox.information(self, "No results", "Run a simulation first.")
            return
        ext = "PNG files (*.png)" if fmt == "png" else "PDF files (*.pdf)"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "results." + fmt, ext)
        if path:
            try:
                self.figure.savefig(path, dpi=150, bbox_inches='tight')
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))
