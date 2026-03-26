"""
Parameter panel for optical lithography simulator.
Controls: preset selector, wavelength, NA, illumination, sigma, defocus, mask type, grid size,
and aberration (Zernike) parameters.
"""
from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QComboBox, QSlider, QPushButton, QFileDialog,
    QMessageBox, QScrollArea, QLabel, QSpinBox, Qt, Signal, QFont,
)

# ---------------------------------------------------------------------------
# Preset definitions: (wavelength_nm, NA, sigma_outer, sigma_inner)
# ---------------------------------------------------------------------------
_PRESETS = {
    "ArF Immersion (193nm, NA=0.93)": (193.0, 0.93, 0.85, 0.55),
    "ArF Dry (193nm, NA=0.75)":       (193.0, 0.75, 0.80, 0.50),
    "KrF (248nm, NA=0.68)":           (248.0, 0.68, 0.75, 0.45),
    "EUV (13.5nm, NA=0.33)":          (13.5,  0.33, 0.70, 0.40),
    "Custom":                          None,
}

class ParameterPanel(QWidget):
    """Panel for setting lithography simulation parameters."""

    params_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._applying_preset = False
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Wrap everything in a scroll area so it works at any window height
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ---- Preset selector ----
        preset_group = QGroupBox("Quick Presets")
        preset_group.setFlat(False)
        preset_layout = QFormLayout(preset_group)
        preset_layout.setContentsMargins(8, 12, 8, 8)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(_PRESETS.keys()))
        self.preset_combo.setToolTip(
            "Select a technology node preset to auto-fill lithography parameters"
        )
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addRow("Preset:", self.preset_combo)
        layout.addWidget(preset_group)

        # ---- Lithography parameters ----
        litho_group = QGroupBox("Lithography Parameters")
        litho_group.setFlat(False)
        form = QFormLayout(litho_group)
        form.setContentsMargins(8, 12, 8, 8)

        self.wavelength_sb = QDoubleSpinBox()
        self.wavelength_sb.setRange(13.5, 365.0)
        self.wavelength_sb.setValue(193.0)
        self.wavelength_sb.setSuffix(" nm")
        self.wavelength_sb.setDecimals(1)
        self.wavelength_sb.setToolTip("Exposure wavelength (ArF=193nm, EUV=13.5nm)")
        self.wavelength_sb.valueChanged.connect(self._on_spinbox_changed)
        form.addRow("Wavelength:", self.wavelength_sb)

        self.na_sb = QDoubleSpinBox()
        self.na_sb.setRange(0.1, 1.35)
        self.na_sb.setValue(0.93)
        self.na_sb.setDecimals(3)
        self.na_sb.setSingleStep(0.01)
        self.na_sb.setToolTip(
            "Numerical Aperture. Immersion max ~1.35, EUV typically 0.33"
        )
        self.na_sb.valueChanged.connect(self._on_spinbox_changed)
        form.addRow("NA:", self.na_sb)

        self.illum_combo = QComboBox()
        self.illum_combo.addItems(["Circular", "Annular", "Quadrupole", "Quasar"])
        self.illum_combo.setCurrentText("Annular")
        self.illum_combo.setToolTip("Illumination pupil shape")
        self.illum_combo.currentTextChanged.connect(self._on_illum_changed)
        form.addRow("Illumination:", self.illum_combo)

        self.sigma_outer_sb = QDoubleSpinBox()
        self.sigma_outer_sb.setRange(0.01, 1.0)
        self.sigma_outer_sb.setValue(0.85)
        self.sigma_outer_sb.setDecimals(2)
        self.sigma_outer_sb.setSingleStep(0.05)
        self.sigma_outer_sb.setToolTip(
            "Partial coherence factor — outer radius (0=coherent, 1=incoherent)"
        )
        self.sigma_outer_sb.valueChanged.connect(self._on_spinbox_changed)
        form.addRow("sigma outer:", self.sigma_outer_sb)

        self.sigma_inner_sb = QDoubleSpinBox()
        self.sigma_inner_sb.setRange(0.0, 1.0)
        self.sigma_inner_sb.setValue(0.55)
        self.sigma_inner_sb.setDecimals(2)
        self.sigma_inner_sb.setSingleStep(0.05)
        self.sigma_inner_sb.setToolTip(
            "Partial coherence factor — inner radius (0=coherent, 1=incoherent)"
        )
        self.sigma_inner_sb.valueChanged.connect(self._on_spinbox_changed)
        form.addRow("sigma inner:", self.sigma_inner_sb)

        layout.addWidget(litho_group)

        # ---- Defocus ----
        defocus_group = QGroupBox("Defocus")
        defocus_group.setFlat(False)

        df_outer = QVBoxLayout(defocus_group)
        df_outer.setContentsMargins(8, 12, 8, 8)

        df_row = QHBoxLayout()

        self.defocus_slider = QSlider(Qt.Horizontal)
        self.defocus_slider.setRange(-500, 500)
        self.defocus_slider.setValue(0)
        self.defocus_slider.setTickInterval(100)
        self.defocus_slider.setTickPosition(QSlider.TicksBelow)
        self.defocus_slider.setToolTip("Focal plane offset from best focus")
        self.defocus_slider.valueChanged.connect(self._slider_to_spinbox)
        df_row.addWidget(self.defocus_slider)

        self.defocus_sb = QDoubleSpinBox()
        self.defocus_sb.setRange(-500.0, 500.0)
        self.defocus_sb.setValue(0.0)
        self.defocus_sb.setSuffix(" nm")
        self.defocus_sb.setDecimals(0)
        self.defocus_sb.setMaximumWidth(90)
        self.defocus_sb.setToolTip("Focal plane offset from best focus")
        self.defocus_sb.valueChanged.connect(self._spinbox_to_slider)
        df_row.addWidget(self.defocus_sb)

        df_outer.addLayout(df_row)

        # Min/max labels under the slider
        minmax_row = QHBoxLayout()
        lbl_min = QLabel("-500 nm")
        lbl_max = QLabel("+500 nm")
        lbl_min.setObjectName("caption")
        lbl_max.setObjectName("caption")
        minmax_row.addWidget(lbl_min)
        minmax_row.addStretch()
        minmax_row.addWidget(lbl_max)
        df_outer.addLayout(minmax_row)

        layout.addWidget(defocus_group)

        # ---- Mask & Grid ----
        mask_group = QGroupBox("Mask & Grid")
        mask_group.setFlat(False)

        mask_form = QFormLayout(mask_group)
        mask_form.setContentsMargins(8, 12, 8, 8)

        self.mask_combo = QComboBox()
        self.mask_combo.addItems(["Binary", "AttPSM", "AltPSM"])
        self.mask_combo.setToolTip("Mask transmission type")
        self.mask_combo.currentTextChanged.connect(self.params_changed)
        mask_form.addRow("Mask type:", self.mask_combo)

        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["64", "128", "256", "512"])
        self.grid_combo.setCurrentText("256")
        self.grid_combo.setToolTip(
            "Simulation grid resolution. Higher = more accurate but slower"
        )
        self.grid_combo.currentTextChanged.connect(self.params_changed)
        mask_form.addRow("Grid size:", self.grid_combo)

        self.domain_sb = QDoubleSpinBox()
        self.domain_sb.setRange(100.0, 10000.0)
        self.domain_sb.setValue(2000.0)
        self.domain_sb.setSuffix(" nm")
        self.domain_sb.setDecimals(0)
        self.domain_sb.setToolTip("Physical size of the simulation domain")
        self.domain_sb.valueChanged.connect(self.params_changed)
        mask_form.addRow("Domain size:", self.domain_sb)

        layout.addWidget(mask_group)

        # ---- Aberrations (Zernike) ----
        aber_group = QGroupBox("Aberrations (Zernike)")
        aber_group.setFlat(False)

        aber_form = QFormLayout(aber_group)
        aber_form.setContentsMargins(8, 12, 8, 8)

        self.w020_sb = QDoubleSpinBox()
        self.w020_sb.setRange(-1.0, 1.0)
        self.w020_sb.setValue(0.0)
        self.w020_sb.setDecimals(3)
        self.w020_sb.setSingleStep(0.01)
        self.w020_sb.setSuffix(" λ")
        self.w020_sb.setToolTip("W020 — defocus aberration (waves)")
        self.w020_sb.valueChanged.connect(self.params_changed)
        aber_form.addRow("W020 (defocus):", self.w020_sb)

        self.w040_sb = QDoubleSpinBox()
        self.w040_sb.setRange(-1.0, 1.0)
        self.w040_sb.setValue(0.0)
        self.w040_sb.setDecimals(3)
        self.w040_sb.setSingleStep(0.01)
        self.w040_sb.setSuffix(" λ")
        self.w040_sb.setToolTip("W040 — spherical aberration (waves)")
        self.w040_sb.valueChanged.connect(self.params_changed)
        aber_form.addRow("W040 (spherical):", self.w040_sb)

        self.w131_sb = QDoubleSpinBox()
        self.w131_sb.setRange(-1.0, 1.0)
        self.w131_sb.setValue(0.0)
        self.w131_sb.setDecimals(3)
        self.w131_sb.setSingleStep(0.01)
        self.w131_sb.setSuffix(" λ")
        self.w131_sb.setToolTip("W131 — coma aberration (waves)")
        self.w131_sb.valueChanged.connect(self.params_changed)
        aber_form.addRow("W131 (coma):", self.w131_sb)

        layout.addWidget(aber_group)

        # ---- Load / Save buttons ----
        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load YAML")
        self.save_btn = QPushButton("Save YAML")
        self.load_btn.setObjectName("secondary")
        self.save_btn.setObjectName("secondary")
        self.load_btn.clicked.connect(self._load_config)
        self.save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)
        layout.addStretch()

        scroll.setWidget(container)
        outer_layout.addWidget(scroll)

    # ------------------------------------------------------------------
    # Preset logic
    # ------------------------------------------------------------------

    def _on_preset_changed(self, text):
        values = _PRESETS.get(text)
        if values is None:
            return  # "Custom" — do nothing
        wl, na, s_out, s_in = values
        self._applying_preset = True
        try:
            self.wavelength_sb.setValue(wl)
            self.na_sb.setValue(na)
            self.sigma_outer_sb.setValue(s_out)
            self.sigma_inner_sb.setValue(s_in)
        finally:
            self._applying_preset = False
        self.params_changed.emit()

    def _on_spinbox_changed(self):
        """Any manual spinbox edit switches preset combo to 'Custom'."""
        if not self._applying_preset:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Custom")
            self.preset_combo.blockSignals(False)
        self.params_changed.emit()

    # ------------------------------------------------------------------
    # Illumination change
    # ------------------------------------------------------------------

    def _on_illum_changed(self, text):
        self.sigma_inner_sb.setEnabled(text in ["Annular", "Quadrupole", "Quasar"])
        self.params_changed.emit()

    # ------------------------------------------------------------------
    # Defocus slider <-> spinbox sync
    # ------------------------------------------------------------------

    def _slider_to_spinbox(self, value):
        self.defocus_sb.blockSignals(True)
        self.defocus_sb.setValue(float(value))
        self.defocus_sb.blockSignals(False)
        self.params_changed.emit()

    def _spinbox_to_slider(self, value):
        self.defocus_slider.blockSignals(True)
        self.defocus_slider.setValue(int(value))
        self.defocus_slider.blockSignals(False)
        self.params_changed.emit()

    # ------------------------------------------------------------------
    # Config get / load
    # ------------------------------------------------------------------

    def get_config(self):
        illum_map = {
            "Circular": "circular", "Annular": "annular",
            "Quadrupole": "quadrupole", "Quasar": "quasar",
        }
        mask_map = {"Binary": "binary", "AttPSM": "att_psm", "AltPSM": "alt_psm"}
        return {
            "lithography": {
                "wavelength_nm": self.wavelength_sb.value(),
                "NA": self.na_sb.value(),
                "defocus_nm": self.defocus_sb.value(),
                "illumination": {
                    "type": illum_map[self.illum_combo.currentText()],
                    "sigma_outer": self.sigma_outer_sb.value(),
                    "sigma_inner": self.sigma_inner_sb.value(),
                },
                "mask_type": mask_map[self.mask_combo.currentText()],
                "aberrations": {
                    "W020": self.w020_sb.value(),
                    "W040": self.w040_sb.value(),
                    "W131": self.w131_sb.value(),
                },
            },
            "simulation": {
                "grid_size": int(self.grid_combo.currentText()),
                "domain_size_nm": self.domain_sb.value(),
            },
        }

    def load_config(self, config):
        litho = config.get("lithography", {})
        sim = config.get("simulation", {})
        illum = litho.get("illumination", {})
        aber = litho.get("aberrations", {})

        self._applying_preset = True
        try:
            self.wavelength_sb.setValue(litho.get("wavelength_nm", 193.0))
            self.na_sb.setValue(litho.get("NA", 0.93))
            self.defocus_sb.setValue(litho.get("defocus_nm", 0.0))
            self.sigma_outer_sb.setValue(illum.get("sigma_outer", 0.85))
            self.sigma_inner_sb.setValue(illum.get("sigma_inner", 0.55))
            self.w020_sb.setValue(aber.get("W020", 0.0))
            self.w040_sb.setValue(aber.get("W040", 0.0))
            self.w131_sb.setValue(aber.get("W131", 0.0))
        finally:
            self._applying_preset = False

        illum_rmap = {
            "circular": "Circular", "annular": "Annular",
            "quadrupole": "Quadrupole", "quasar": "Quasar",
        }
        self.illum_combo.setCurrentText(
            illum_rmap.get(illum.get("type", "annular"), "Annular")
        )
        mask_rmap = {"binary": "Binary", "att_psm": "AttPSM", "alt_psm": "AltPSM"}
        self.mask_combo.setCurrentText(
            mask_rmap.get(litho.get("mask_type", "binary"), "Binary")
        )
        self.grid_combo.setCurrentText(str(sim.get("grid_size", 256)))
        self.domain_sb.setValue(sim.get("domain_size_nm", 2000.0))

        # Mark as custom after loading
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Custom")
        self.preset_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # YAML file I/O
    # ------------------------------------------------------------------

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "", "YAML files (*.yaml *.yml)")
        if not path:
            return
        try:
            import sys, os
            sim_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            if sim_dir not in sys.path:
                sys.path.insert(0, sim_dir)
            from fileio.parameter_io import load_config
            self.load_config(load_config(path))
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "config.yaml", "YAML files (*.yaml *.yml)")
        if not path:
            return
        try:
            import sys, os
            sim_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            if sim_dir not in sys.path:
                sys.path.insert(0, sim_dir)
            from fileio.parameter_io import save_config
            save_config(self.get_config(), path)
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))
