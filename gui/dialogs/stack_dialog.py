"""
Film Stack Visual Editor Dialog.
Provides interactive layer editing with TMM reflectance preview.
"""
import numpy as np

from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QComboBox, QPushButton, QListWidget, QLabel, Qt,
)
from gui import theme

try:
    from PySide6.QtWidgets import QDialog, QDialogButtonBox
except ImportError:
    from PyQt5.QtWidgets import QDialog, QDialogButtonBox  # type: ignore

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from core.film_stack import FilmStack, Layer, TransferMatrixEngine, _get_material_nk

_MATERIALS_193 = ['vacuum', 'resist', 'arc', 'barc', 'si', 'sio2', 'si3n4', 'cr', 'custom']
_MATERIALS_EUV = ['vacuum', 'mo', 'si', 'tan', 'ru', 'custom']


class StackDialog(QDialog):
    """Interactive film stack editor with TMM reflectance preview.

    Usage::
        dlg = StackDialog(parent, film_stack=my_stack)
        if dlg.exec() == QDialog.Accepted:
            stack = dlg.get_film_stack()
    """

    def __init__(self, parent=None, film_stack=None):
        super().__init__(parent)
        self.setWindowTitle("Film Stack Editor")
        self.resize(700, 500)
        self._stack = film_stack or FilmStack.default_193nm()
        self._engine = TransferMatrixEngine()
        self._build_ui()
        self._refresh_list()
        if self.layer_list.count() > 0:
            self.layer_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(8)

        content = QHBoxLayout()
        content.setSpacing(12)

        # ── Left panel (1/3): layer list + buttons ─────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        cap = QLabel("Layers (top → bottom)")
        cap.setObjectName("caption")
        left_layout.addWidget(cap)

        self.layer_list = QListWidget()
        self.layer_list.currentRowChanged.connect(self._on_row_changed)
        left_layout.addWidget(self.layer_list)

        self.add_btn = QPushButton("Add Layer")
        self.remove_btn = QPushButton("Remove")
        self.move_up_btn = QPushButton("Move Up")
        self.move_down_btn = QPushButton("Move Down")
        for btn in [self.add_btn, self.remove_btn,
                    self.move_up_btn, self.move_down_btn]:
            btn.setObjectName("secondary")
            left_layout.addWidget(btn)

        self.add_btn.clicked.connect(self._add_layer)
        self.remove_btn.clicked.connect(self._remove_layer)
        self.move_up_btn.clicked.connect(self._move_up)
        self.move_down_btn.clicked.connect(self._move_down)

        content.addWidget(left, 1)

        # ── Right panel (2/3): properties + preview ────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        props = QGroupBox("Layer Properties")
        form = QFormLayout(props)
        form.setContentsMargins(8, 12, 8, 8)
        form.setSpacing(6)

        self.mat_combo = QComboBox()
        self.mat_combo.addItems(_MATERIALS_193)
        self.mat_combo.setToolTip("Material (n, k auto-filled from library; 'custom' = manual)")
        self.mat_combo.currentTextChanged.connect(self._on_material_changed)
        form.addRow("Material:", self.mat_combo)

        self.thick_sb = QDoubleSpinBox()
        self.thick_sb.setRange(0.0, 10000.0)
        self.thick_sb.setValue(100.0)
        self.thick_sb.setSuffix(" nm")
        self.thick_sb.setDecimals(1)
        self.thick_sb.setToolTip("Layer thickness. Use 0 for semi-infinite layers (incident medium / substrate).")
        self.thick_sb.valueChanged.connect(self._on_prop_changed)
        form.addRow("Thickness:", self.thick_sb)

        self.n_sb = QDoubleSpinBox()
        self.n_sb.setRange(0.01, 10.0)
        self.n_sb.setValue(1.0)
        self.n_sb.setDecimals(4)
        self.n_sb.setEnabled(False)
        self.n_sb.setToolTip("Refractive index (enabled for 'custom' material)")
        self.n_sb.valueChanged.connect(self._on_prop_changed)
        form.addRow("n (custom):", self.n_sb)

        self.k_sb = QDoubleSpinBox()
        self.k_sb.setRange(0.0, 10.0)
        self.k_sb.setValue(0.0)
        self.k_sb.setDecimals(4)
        self.k_sb.setEnabled(False)
        self.k_sb.setToolTip("Extinction coefficient (enabled for 'custom' material)")
        self.k_sb.valueChanged.connect(self._on_prop_changed)
        form.addRow("k (custom):", self.k_sb)

        right_layout.addWidget(props)

        if _HAS_MPL:
            theme.apply_mpl_theme()
            self._fig = Figure(figsize=(4, 2.2), dpi=theme.MPL_DPI)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._canvas.setMinimumHeight(160)
            right_layout.addWidget(self._canvas)
            self._draw_empty_preview()
        else:
            self._canvas = None
            right_layout.addWidget(QLabel("(matplotlib unavailable — install it for preview)"))

        self.preview_btn = QPushButton("Update Preview")
        self.preview_btn.setObjectName("secondary")
        self.preview_btn.setToolTip("Recompute TMM reflectance vs angle (0–30°)")
        self.preview_btn.clicked.connect(self._update_preview)
        right_layout.addWidget(self.preview_btn)
        right_layout.addStretch()

        content.addWidget(right, 2)
        root.addLayout(content)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    # ------------------------------------------------------------------
    # Layer list helpers
    # ------------------------------------------------------------------

    def _layer_label(self, layer: Layer) -> str:
        return (f"{layer.material} | {layer.thickness_nm:.1f} nm "
                f"| n={layer.n:.3f} k={layer.k:.3f}")

    def _refresh_list(self):
        cur = self.layer_list.currentRow()
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for layer in self._stack.layers:
            self.layer_list.addItem(self._layer_label(layer))
        self.layer_list.blockSignals(False)
        n = self.layer_list.count()
        self.layer_list.setCurrentRow(min(cur, n - 1) if n > 0 else -1)

    def _on_row_changed(self, row: int):
        n = len(self._stack.layers)
        has = 0 <= row < n
        self.remove_btn.setEnabled(has)
        self.move_up_btn.setEnabled(has and row > 0)
        self.move_down_btn.setEnabled(has and row < n - 1)
        if not has:
            return

        layer = self._stack.layers[row]

        self.mat_combo.blockSignals(True)
        idx = self.mat_combo.findText(layer.material.lower())
        if idx < 0:
            idx = self.mat_combo.findText('custom')
        self.mat_combo.setCurrentIndex(idx)
        self.mat_combo.blockSignals(False)

        self.thick_sb.blockSignals(True)
        self.thick_sb.setValue(layer.thickness_nm)
        self.thick_sb.blockSignals(False)

        is_custom = layer.material.lower() == 'custom'
        self.n_sb.setEnabled(is_custom)
        self.k_sb.setEnabled(is_custom)

        self.n_sb.blockSignals(True)
        self.k_sb.blockSignals(True)
        self.n_sb.setValue(layer.n)
        self.k_sb.setValue(layer.k)
        self.n_sb.blockSignals(False)
        self.k_sb.blockSignals(False)

    def _on_material_changed(self, mat: str):
        row = self.layer_list.currentRow()
        if not (0 <= row < len(self._stack.layers)):
            return
        layer = self._stack.layers[row]
        layer.material = mat
        is_custom = mat.lower() == 'custom'
        self.n_sb.setEnabled(is_custom)
        self.k_sb.setEnabled(is_custom)
        if not is_custom:
            try:
                n, k = _get_material_nk(mat, self._stack.wavelength_nm)
                layer.n = n
                layer.k = k
                self.n_sb.blockSignals(True)
                self.k_sb.blockSignals(True)
                self.n_sb.setValue(n)
                self.k_sb.setValue(k)
                self.n_sb.blockSignals(False)
                self.k_sb.blockSignals(False)
            except ValueError:
                pass
        item = self.layer_list.item(row)
        if item:
            item.setText(self._layer_label(layer))

    def _on_prop_changed(self):
        row = self.layer_list.currentRow()
        if not (0 <= row < len(self._stack.layers)):
            return
        layer = self._stack.layers[row]
        layer.thickness_nm = self.thick_sb.value()
        if layer.material.lower() == 'custom':
            layer.n = self.n_sb.value()
            layer.k = self.k_sb.value()
        item = self.layer_list.item(row)
        if item:
            item.setText(self._layer_label(layer))

    # ------------------------------------------------------------------
    # Layer CRUD / reorder
    # ------------------------------------------------------------------

    def _add_layer(self):
        new = Layer(material='resist', thickness_nm=100.0, n=1.7, k=0.02)
        row = self.layer_list.currentRow()
        insert_at = (row + 1) if row >= 0 else len(self._stack.layers)
        self._stack.layers.insert(insert_at, new)
        self._refresh_list()
        self.layer_list.setCurrentRow(insert_at)

    def _remove_layer(self):
        row = self.layer_list.currentRow()
        if 0 <= row < len(self._stack.layers):
            self._stack.layers.pop(row)
            self._refresh_list()

    def _move_up(self):
        row = self.layer_list.currentRow()
        if row > 0:
            ls = self._stack.layers
            ls[row - 1], ls[row] = ls[row], ls[row - 1]
            self._refresh_list()
            self.layer_list.setCurrentRow(row - 1)

    def _move_down(self):
        row = self.layer_list.currentRow()
        if row < len(self._stack.layers) - 1:
            ls = self._stack.layers
            ls[row], ls[row + 1] = ls[row + 1], ls[row]
            self._refresh_list()
            self.layer_list.setCurrentRow(row + 1)

    # ------------------------------------------------------------------
    # Reflectance preview
    # ------------------------------------------------------------------

    def _draw_empty_preview(self):
        ax = self._ax
        ax.clear()
        ax.set_xlabel("Angle (°)", fontsize=theme.MPL_LABEL)
        ax.set_ylabel("Reflectance", fontsize=theme.MPL_LABEL)
        ax.set_title("Reflectance vs Angle  [click Update Preview]",
                     fontsize=theme.MPL_TITLE)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=theme.MPL_TICK)
        self._fig.tight_layout()
        self._canvas.draw()

    def _update_preview(self):
        if self._canvas is None or len(self._stack.layers) < 2:
            return
        angles = np.linspace(0, 30, 61)
        try:
            R_te = [abs(self._engine.reflectance(self._stack, a, 'te')) ** 2
                    for a in angles]
            R_tm = [abs(self._engine.reflectance(self._stack, a, 'tm')) ** 2
                    for a in angles]
        except Exception:
            return

        ax = self._ax
        ax.clear()
        ax.plot(angles, R_te, color=theme.ACCENT, lw=1.5, label='TE')
        ax.plot(angles, R_tm, color=theme.WARNING, lw=1.5,
                linestyle='--', label='TM')
        ax.set_xlabel("Angle (°)", fontsize=theme.MPL_LABEL)
        ax.set_ylabel("Reflectance", fontsize=theme.MPL_LABEL)
        ax.set_title("Reflectance vs Angle", fontsize=theme.MPL_TITLE)
        ax.legend(fontsize=theme.MPL_LEGEND)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=theme.MPL_TICK)
        self._fig.tight_layout()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_film_stack(self) -> FilmStack:
        """Return the current FilmStack object."""
        return self._stack

    def set_film_stack(self, stack: FilmStack):
        """Load an existing FilmStack into the editor."""
        self._stack = stack
        self._refresh_list()
        if self.layer_list.count() > 0:
            self.layer_list.setCurrentRow(0)
