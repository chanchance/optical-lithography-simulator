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
from matplotlib.patches import Rectangle

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
        self._clip_center_nm = None    # (cx, cy) tuple or None
        self._clip_domain_nm = 2000.0
        self._ruler_mode = False
        self._ruler_p1 = None
        self._ruler_line = None
        self._ruler_ann = None
        self._zoom_step = 0.25         # zoom 25% per click
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        tb = QHBoxLayout()
        self.open_btn = QPushButton("Open GDS/OAS")
        self.open_btn.setObjectName("secondary")
        self.fit_btn = QPushButton("Fit View")
        self.fit_btn.setObjectName("secondary")
        self.coord_label = QLabel("x: --   y: --")
        self.coord_label.setFont(QFont("Menlo", 10))
        self.coord_label.setObjectName("caption")
        self.coord_label.setStyleSheet(
            "font-family: 'Menlo','SF Mono','Consolas',monospace; font-size: 11px;"
        )
        self.open_btn.clicked.connect(self._open_file)
        self.fit_btn.clicked.connect(self._draw)
        tb.addWidget(self.open_btn)
        tb.addWidget(self.fit_btn)
        self.zoom_in_btn = QPushButton("＋")
        self.zoom_in_btn.setObjectName("secondary")
        self.zoom_in_btn.setFixedWidth(32)
        self.zoom_in_btn.setToolTip("Zoom in")
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        self.zoom_out_btn = QPushButton("－")
        self.zoom_out_btn.setObjectName("secondary")
        self.zoom_out_btn.setFixedWidth(32)
        self.zoom_out_btn.setToolTip("Zoom out")
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        self.ruler_btn = QPushButton("Ruler")
        self.ruler_btn.setObjectName("secondary")
        self.ruler_btn.setCheckable(True)
        self.ruler_btn.setToolTip("Click two points to measure distance")
        self.ruler_btn.toggled.connect(self._on_ruler_toggled)
        tb.addWidget(self.zoom_in_btn)
        tb.addWidget(self.zoom_out_btn)
        tb.addWidget(self.ruler_btn)
        tb.setSpacing(8)
        tb.setContentsMargins(8, 6, 8, 6)
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
        self.layer_list.setMaximumWidth(180)
        self.layer_list.itemChanged.connect(self._on_layer_toggle)
        splitter.addWidget(self.layer_list)

        # Canvas container (needed to host overlay as child)
        self._canvas_container = QWidget()
        canvas_layout = QVBoxLayout(self._canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(6, 6), dpi=theme.MPL_DPI)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.set_title("No layout — open a GDS/OAS file")
        self.ax.set_facecolor(theme.BG_SECONDARY)
        self.ax.text(0.5, 0.5, "Open GDS/OAS to begin",
                     ha='center', va='center',
                     fontsize=12, color=theme.TEXT_TERTIARY,
                     transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.figure.patch.set_facecolor(theme.BG_PRIMARY)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
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
            color = _COLORS[i % len(_COLORS)]
            item.setForeground(QColor(color))
            self._layer_visible[layer] = True
            self.layer_list.addItem(item)
        self.layer_list.blockSignals(False)

    def _on_layer_toggle(self, item):
        self._layer_visible[item.data(Qt.UserRole)] = (item.checkState() == Qt.Checked)
        self._draw()

    # ------------------------------------------------------------------
    # Zoom helpers
    # ------------------------------------------------------------------

    def _zoom_in(self):
        self._apply_zoom(1.0 - self._zoom_step)

    def _zoom_out(self):
        self._apply_zoom(1.0 + self._zoom_step)

    def _apply_zoom(self, factor: float):
        xl = self.ax.get_xlim()
        yl = self.ax.get_ylim()
        cx = (xl[0] + xl[1]) * 0.5
        cy = (yl[0] + yl[1]) * 0.5
        hw = (xl[1] - xl[0]) * 0.5 * factor
        hh = (yl[1] - yl[0]) * 0.5 * factor
        self.ax.set_xlim(cx - hw, cx + hw)
        self.ax.set_ylim(cy - hh, cy + hh)
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes is not self.ax:
            return
        factor = 1.0 - self._zoom_step if event.button == 'up' else 1.0 + self._zoom_step
        # Zoom toward cursor position
        xl = self.ax.get_xlim()
        yl = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_xl = [xdata + (x - xdata) * factor for x in xl]
        new_yl = [ydata + (y - ydata) * factor for y in yl]
        self.ax.set_xlim(new_xl)
        self.ax.set_ylim(new_yl)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Ruler
    # ------------------------------------------------------------------

    def _on_ruler_toggled(self, checked: bool):
        self._ruler_mode = checked
        if not checked:
            self._ruler_p1 = None
            self._clear_ruler_overlay()
            self.canvas.draw_idle()

    def _clear_ruler_overlay(self):
        if self._ruler_line is not None:
            try:
                self._ruler_line.remove()
            except Exception:
                pass
            self._ruler_line = None
        if self._ruler_ann is not None:
            try:
                self._ruler_ann.remove()
            except Exception:
                pass
            self._ruler_ann = None

    def _on_canvas_click(self, event):
        if not self._ruler_mode or event.inaxes is not self.ax:
            return
        x, y = event.xdata, event.ydata
        if self._ruler_p1 is None:
            self._ruler_p1 = (x, y)
        else:
            x1, y1 = self._ruler_p1
            dist = np.hypot(x - x1, y - y1)
            self._clear_ruler_overlay()
            self._ruler_line, = self.ax.plot(
                [x1, x], [y1, y], color='#FFD600', linewidth=1.5,
                linestyle='--', zorder=10)
            mx, my = (x1 + x) * 0.5, (y1 + y) * 0.5
            self._ruler_ann = self.ax.annotate(
                '{:.1f} nm'.format(dist),
                xy=(mx, my), fontsize=9,
                color='#FFD600', fontweight='bold',
                ha='center', va='bottom', zorder=11,
                bbox=dict(boxstyle='round,pad=0.2', fc='#191F28', ec='none', alpha=0.7),
            )
            self.canvas.draw_idle()
            self._ruler_p1 = None  # reset for next measurement

    # ------------------------------------------------------------------
    # Clip region overlay
    # ------------------------------------------------------------------

    def set_clip_region(self, center_nm, domain_size_nm: float):
        """Update the clip-region rectangle drawn on the layout canvas.
        center_nm: (cx, cy) in nm, or None to hide.
        domain_size_nm: side length of the simulation domain.
        """
        self._clip_center_nm = center_nm
        self._clip_domain_nm = domain_size_nm
        self._redraw_clip_overlay()
        self.canvas.draw_idle()

    def _redraw_clip_overlay(self):
        # Remove existing clip patch (tagged by gid)
        for patch in list(self.ax.patches):
            if getattr(patch, '_omc_clip', False):
                patch.remove()
        if self._clip_center_nm is None:
            return
        cx, cy = self._clip_center_nm
        half = self._clip_domain_nm * 0.5
        rect = Rectangle(
            (cx - half, cy - half), self._clip_domain_nm, self._clip_domain_nm,
            linewidth=1.5, edgecolor='#3182F6', facecolor='#3182F6',
            alpha=0.08, linestyle='--', zorder=5)
        rect._omc_clip = True
        self.ax.add_patch(rect)

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

        self._redraw_clip_overlay()
        self.figure.tight_layout()
        self.canvas.draw()

    def _on_mouse_move(self, event):
        if event.inaxes:
            self.coord_label.setText("x: {:.1f} nm   y: {:.1f} nm".format(
                event.xdata, event.ydata))

    def get_layout_path(self):
        return self._layout_path

    def get_bounding_box(self):
        """Return the loaded layout bounding box, or None."""
        return self._bbox
