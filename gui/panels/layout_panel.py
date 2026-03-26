"""
Layout panel: fast async GDS/OAS viewer with LOD rendering.
"""
import os
import sys
import numpy as np

from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QPushButton,
    QLabel, QFileDialog, QMessageBox, QProgressBar, QFrame,
    Qt, Signal, QThread, QPalette, QColor, QFont,
)
from gui import theme

import matplotlib
# Do NOT call matplotlib.use('Agg') — it conflicts with the Qt backend
# (backend_qtagg) already active in the main window.
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection

_SIM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7']


class LayoutLoaderThread(QThread):
    progress = Signal(str, int)   # (step, percent)
    finished = Signal(object)     # LayoutData
    error = Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        try:
            if _SIM_DIR not in sys.path:
                sys.path.insert(0, _SIM_DIR)
            from fileio.layout_io import LayoutReader
            reader = LayoutReader()
            data = reader.read(
                self._path,
                on_progress=lambda s, p: self.progress.emit(s, p)
            )
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class LoadingOverlay(QWidget):
    """Semi-transparent overlay with progress bar shown during loading."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet(
            "background-color: rgba(25, 31, 40, 180);"
        )

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self._label = QLabel("Loading...")
        self._label.setStyleSheet("color: white; font-size: 13px; background: transparent;")
        self._label.setAlignment(Qt.AlignCenter)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedWidth(300)
        self._bar.setStyleSheet(
            "QProgressBar {{ border: 1px solid {border}; border-radius: 4px; "
            "background: #222; color: white; text-align: center; }}"
            "QProgressBar::chunk {{ background: {accent}; border-radius: 3px; }}".format(
                border=theme.BORDER, accent=theme.ACCENT
            )
        )

        layout.addWidget(self._label)
        layout.addWidget(self._bar, 0, Qt.AlignCenter)

        self.hide()

    def set_progress(self, step: str, percent: int):
        self._label.setText(step)
        self._bar.setValue(percent)

    def show_overlay(self):
        self._bar.setValue(0)
        self._label.setText("Loading...")
        self.show()
        self.raise_()

    def hide_overlay(self):
        self.hide()


class LayoutPanel(QWidget):
    layout_loaded = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._polys_by_layer = {}
        self._layer_visible = {}
        self._layout_path = None
        self._bbox = None          # BoundingBox from LayoutData
        self._layer_info = {}      # layer_num -> LayerInfo
        self._loader_thread = None
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        tb = QHBoxLayout()
        self.open_btn = QPushButton("Open GDS/OAS")
        self.fit_btn = QPushButton("Fit View")
        self.coord_label = QLabel("x: --   y: --")
        self.coord_label.setFont(QFont("Courier", 11))
        self.open_btn.clicked.connect(self._open_file)
        self.fit_btn.clicked.connect(self._draw)
        tb.addWidget(self.open_btn)
        tb.addWidget(self.fit_btn)
        tb.addStretch()

        # Progress bar in toolbar (hidden by default)
        self.toolbar_progress = QProgressBar()
        self.toolbar_progress.setRange(0, 100)
        self.toolbar_progress.setFixedWidth(200)
        self.toolbar_progress.hide()
        tb.addWidget(self.toolbar_progress)

        tb.addWidget(self.coord_label)
        outer.addLayout(tb)

        splitter = QSplitter(Qt.Horizontal)

        self.layer_list = QListWidget()
        self.layer_list.setMaximumWidth(160)
        self.layer_list.itemChanged.connect(self._on_layer_toggle)
        splitter.addWidget(self.layer_list)

        # Canvas container (needed to host overlay as child)
        self._canvas_container = QWidget()
        canvas_layout = QVBoxLayout(self._canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(6, 6), dpi=theme.MPL_DPI)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.set_title("No layout loaded")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        canvas_layout.addWidget(self.canvas)

        # Loading overlay as child of canvas container
        self._overlay = LoadingOverlay(self._canvas_container)

        splitter.addWidget(self._canvas_container)
        splitter.setStretchFactor(1, 4)
        outer.addWidget(splitter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep overlay sized to the canvas container
        if hasattr(self, '_overlay') and hasattr(self, '_canvas_container'):
            self._overlay.setGeometry(self._canvas_container.rect())

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Layout", "", "Layout files (*.gds *.gds2 *.oas);;All (*)")
        if path:
            self.load_layout(path)

    def load_layout(self, path):
        # Cancel any in-flight load
        if self._loader_thread and self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait()

        self._overlay.setGeometry(self._canvas_container.rect())
        self._overlay.show_overlay()
        self.toolbar_progress.setValue(0)
        self.toolbar_progress.show()
        self.open_btn.setEnabled(False)

        self._loader_thread = LayoutLoaderThread(path, self)
        self._loader_thread.progress.connect(self._on_load_progress)
        self._loader_thread.finished.connect(self._on_load_finished)
        self._loader_thread.error.connect(self._on_load_error)
        self._loader_thread.start()

    def _on_load_progress(self, step: str, percent: int):
        self._overlay.set_progress(step, percent)
        self.toolbar_progress.setValue(percent)

    def _on_load_finished(self, data):
        self._overlay.hide_overlay()
        self.toolbar_progress.hide()
        self.open_btn.setEnabled(True)

        self._layout_path = data.filepath
        self._bbox = data.bounding_box
        self._layer_info = data.layers

        # Store display polygons by layer (already sampled by reader)
        self._polys_by_layer = {}
        for layer_num, polys in data.polygons_by_layer.items():
            # Convert each (N,2) array to list-of-arrays usable by PolyCollection
            self._polys_by_layer[layer_num] = polys

        self._refresh_layers()
        self._draw()
        self.layout_loaded.emit(data.filepath)

    def _on_load_error(self, msg: str):
        self._overlay.hide_overlay()
        self.toolbar_progress.hide()
        self.open_btn.setEnabled(True)
        QMessageBox.warning(self, "Load Error", msg)

    def _refresh_layers(self):
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        self._layer_visible = {}
        for i, layer in enumerate(sorted(self._polys_by_layer)):
            info = self._layer_info.get(layer)
            count_str = ""
            if info is not None:
                count_str = "  ({:,} polys)".format(info.n_polygons)
            item = QListWidgetItem("Layer {}{}".format(layer, count_str))
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
        self.ax.set_title(
            os.path.basename(self._layout_path) if self._layout_path else "Layout"
        )

        for i, layer in enumerate(sorted(self._polys_by_layer)):
            if not self._layer_visible.get(layer, True):
                continue
            polys = self._polys_by_layer[layer]
            if not polys:
                continue
            color = _COLORS[i % len(_COLORS)]
            col = PolyCollection(polys, facecolors=color, edgecolors='none', alpha=0.6)
            self.ax.add_collection(col)

        # Use stored bounding box — avoids concatenating all point arrays
        if self._bbox is not None:
            bb = self._bbox
            margin = max(bb.width, bb.height) * 0.05 + 1
            self.ax.set_xlim(bb.xmin - margin, bb.xmax + margin)
            self.ax.set_ylim(bb.ymin - margin, bb.ymax + margin)
        elif self._polys_by_layer:
            flat = [p for polys in self._polys_by_layer.values() for p in polys]
            if not flat:
                self.ax.set_xlim(-1000, 1000)
                self.ax.set_ylim(-1000, 1000)
                self.figure.tight_layout()
                self.canvas.draw()
                return
            all_pts = np.concatenate(flat)
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
