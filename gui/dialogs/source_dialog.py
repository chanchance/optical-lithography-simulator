"""
Source dialog: interactive k-space illumination pupil preview.
"""
import math
import numpy as np

try:
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QFormLayout,
        QDoubleSpinBox, QComboBox, QDialogButtonBox
    )
    from PySide6.QtCore import Qt
except ImportError:
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QFormLayout,
        QDoubleSpinBox, QComboBox, QDialogButtonBox
    )
    from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches


class SourceDialog(QDialog):
    """Preview illumination pupil in k-space with interactive controls."""

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle("Illumination Source Preview")
        self.resize(520, 500)
        self._build_ui()
        self._update_preview()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QFormLayout()

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
        ctrl.addRow("sigma outer:", self.sigma_outer_sb)

        self.sigma_inner_sb = QDoubleSpinBox()
        self.sigma_inner_sb.setRange(0.0, 0.99)
        self.sigma_inner_sb.setValue(0.55)
        self.sigma_inner_sb.setDecimals(2)
        self.sigma_inner_sb.setSingleStep(0.05)
        self.sigma_inner_sb.valueChanged.connect(self._update_preview)
        ctrl.addRow("sigma inner:", self.sigma_inner_sb)

        layout.addLayout(ctrl)

        self.figure = Figure(figsize=(4, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept)
        layout.addWidget(bb)

    def _update_preview(self):
        ax = self.ax
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel("kx / NA")
        ax.set_ylabel("ky / NA")
        ax.set_title("Illumination Pupil (k-space)")
        ax.set_facecolor('#f8f8f8')

        # NA boundary
        ax.add_patch(mpatches.Circle((0, 0), 1.0, fill=False,
                                     edgecolor='black', linewidth=1.5))

        itype = self.illum_combo.currentText().lower()
        s_o = self.sigma_outer_sb.value()
        s_i = self.sigma_inner_sb.value()

        if itype == 'circular':
            ax.add_patch(mpatches.Circle((0, 0), s_o, color='#4e79a7', alpha=0.75))

        elif itype == 'annular':
            ax.add_patch(mpatches.Circle((0, 0), s_o, color='#4e79a7', alpha=0.75))
            ax.add_patch(mpatches.Circle((0, 0), s_i, color='#f8f8f8', alpha=1.0))

        elif itype == 'quadrupole':
            offset = (s_o + s_i) / 2.0
            r = (s_o - s_i) / 2.0
            for cx, cy in [(offset, 0), (-offset, 0), (0, offset), (0, -offset)]:
                ax.add_patch(mpatches.Circle((cx, cy), r, color='#f28e2b', alpha=0.75))

        elif itype == 'quasar':
            offset = (s_o + s_i) / 2.0
            r = (s_o - s_i) / 2.0
            for deg in [45, 135, 225, 315]:
                ang = math.radians(deg)
                ax.add_patch(mpatches.Circle(
                    (offset * math.cos(ang), offset * math.sin(ang)),
                    r, color='#e15759', alpha=0.75))

        ax.grid(True, linestyle=':', alpha=0.4)
        self.figure.tight_layout()
        self.canvas.draw()

    def get_illumination_config(self):
        illum_map = {"Circular": "circular", "Annular": "annular",
                     "Quadrupole": "quadrupole", "Quasar": "quasar"}
        return {
            "type": illum_map[self.illum_combo.currentText()],
            "sigma_outer": self.sigma_outer_sb.value(),
            "sigma_inner": self.sigma_inner_sb.value(),
        }
