"""
Layout panel: GDS/OAS polygon viewer with layer visibility controls.
"""
import os
import sys
import numpy as np

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
        QListWidget, QListWidgetItem, QPushButton,
        QLabel, QFileDialog, QMessageBox
    )
    from PySide6.QtCore import Qt, Signal
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
        QListWidget, QListWidgetItem, QPushButton,
        QLabel, QFileDialog, QMessageBox
    )
    from PyQt5.QtCore import Qt, pyqtSignal as Signal

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

_COLORS = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7']


class LayoutPanel(QWidget):
    layout_loaded = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._polys_by_layer = {}
        self._layer_visible = {}
        self._layout_path = None
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)

        tb = QHBoxLayout()
        self.open_btn = QPushButton("Open GDS/OAS")
        self.fit_btn = QPushButton("Fit View")
        self.coord_label = QLabel("x: --   y: --")
        self.open_btn.clicked.connect(self._open_file)
        self.fit_btn.clicked.connect(self._draw)
        tb.addWidget(self.open_btn)
        tb.addWidget(self.fit_btn)
        tb.addStretch()
        tb.addWidget(self.coord_label)
        outer.addLayout(tb)

        splitter = QSplitter(Qt.Horizontal)

        self.layer_list = QListWidget()
        self.layer_list.setMaximumWidth(140)
        self.layer_list.itemChanged.connect(self._on_layer_toggle)
        splitter.addWidget(self.layer_list)

        self.figure = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.set_title("No layout loaded")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 4)
        outer.addWidget(splitter)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Layout", "", "Layout files (*.gds *.gds2 *.oas);;All (*)")
        if path:
            self.load_layout(path)

    def load_layout(self, path):
        sim_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if sim_dir not in sys.path:
            sys.path.insert(0, sim_dir)
        try:
            from fileio.layout_io import LayoutReader
            data = LayoutReader().read(path)
            self._polys_by_layer = {}
            for poly in data.polygons:
                self._polys_by_layer.setdefault(poly.layer, []).append(poly.points)
            self._layout_path = path
            self._refresh_layers()
            self._draw()
            self.layout_loaded.emit(path)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _refresh_layers(self):
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        self._layer_visible = {}
        for i, layer in enumerate(sorted(self._polys_by_layer)):
            item = QListWidgetItem("Layer {}".format(layer))
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, layer)
            self._layer_visible[layer] = True
            self.layer_list.addItem(item)
        self.layer_list.blockSignals(False)

    def _on_layer_toggle(self, item):
        self._layer_visible[item.data(Qt.UserRole)] = (item.checkState() == Qt.Checked)
        self._draw()

    def _draw(self):
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("x (nm)")
        self.ax.set_ylabel("y (nm)")
        self.ax.set_title(os.path.basename(self._layout_path) if self._layout_path else "Layout")

        for i, layer in enumerate(sorted(self._polys_by_layer)):
            if not self._layer_visible.get(layer, True):
                continue
            color = _COLORS[i % len(_COLORS)]
            for pts in self._polys_by_layer[layer]:
                self.ax.add_patch(MplPolygon(pts, closed=True, facecolor=color,
                                             edgecolor='black', alpha=0.5, linewidth=0.5))

        if self._polys_by_layer:
            all_pts = np.concatenate([p for polys in self._polys_by_layer.values() for p in polys])
            xmin, ymin = all_pts.min(axis=0)
            xmax, ymax = all_pts.max(axis=0)
            margin = max(xmax - xmin, ymax - ymin) * 0.05 + 1
            self.ax.set_xlim(xmin - margin, xmax + margin)
            self.ax.set_ylim(ymin - margin, ymax + margin)
        else:
            self.ax.set_xlim(-1000, 1000)
            self.ax.set_ylim(-1000, 1000)

        self.figure.tight_layout()
        self.canvas.draw()

    def _on_mouse_move(self, event):
        if event.inaxes:
            self.coord_label.setText("x: {:.1f} nm   y: {:.1f} nm".format(
                event.xdata, event.ydata))

    def get_layout_path(self):
        return self._layout_path
