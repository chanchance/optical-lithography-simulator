"""
Parameter panel for optical lithography simulator.
Controls: wavelength, NA, illumination, sigma, defocus, mask type, grid size.
"""
try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
        QDoubleSpinBox, QComboBox, QSlider, QPushButton, QFileDialog, QMessageBox
    )
    from PySide6.QtCore import Qt, Signal
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
        QDoubleSpinBox, QComboBox, QSlider, QPushButton, QFileDialog, QMessageBox
    )
    from PyQt5.QtCore import Qt, pyqtSignal as Signal


class ParameterPanel(QWidget):
    """Panel for setting lithography simulation parameters."""

    params_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Lithography parameters
        litho_group = QGroupBox("Lithography Parameters")
        form = QFormLayout(litho_group)

        self.wavelength_sb = QDoubleSpinBox()
        self.wavelength_sb.setRange(13.5, 365.0)
        self.wavelength_sb.setValue(193.0)
        self.wavelength_sb.setSuffix(" nm")
        self.wavelength_sb.setDecimals(1)
        self.wavelength_sb.valueChanged.connect(self.params_changed)
        form.addRow("Wavelength:", self.wavelength_sb)

        self.na_sb = QDoubleSpinBox()
        self.na_sb.setRange(0.1, 1.35)
        self.na_sb.setValue(0.93)
        self.na_sb.setDecimals(3)
        self.na_sb.setSingleStep(0.01)
        self.na_sb.valueChanged.connect(self.params_changed)
        form.addRow("NA:", self.na_sb)

        self.illum_combo = QComboBox()
        self.illum_combo.addItems(["Circular", "Annular", "Quadrupole", "Quasar"])
        self.illum_combo.setCurrentText("Annular")
        self.illum_combo.currentTextChanged.connect(self._on_illum_changed)
        form.addRow("Illumination:", self.illum_combo)

        self.sigma_outer_sb = QDoubleSpinBox()
        self.sigma_outer_sb.setRange(0.01, 1.0)
        self.sigma_outer_sb.setValue(0.85)
        self.sigma_outer_sb.setDecimals(2)
        self.sigma_outer_sb.setSingleStep(0.05)
        self.sigma_outer_sb.valueChanged.connect(self.params_changed)
        form.addRow("sigma outer:", self.sigma_outer_sb)

        self.sigma_inner_sb = QDoubleSpinBox()
        self.sigma_inner_sb.setRange(0.0, 1.0)
        self.sigma_inner_sb.setValue(0.55)
        self.sigma_inner_sb.setDecimals(2)
        self.sigma_inner_sb.setSingleStep(0.05)
        self.sigma_inner_sb.valueChanged.connect(self.params_changed)
        form.addRow("sigma inner:", self.sigma_inner_sb)

        layout.addWidget(litho_group)

        # Defocus
        defocus_group = QGroupBox("Defocus")
        df_layout = QHBoxLayout(defocus_group)

        self.defocus_slider = QSlider(Qt.Horizontal)
        self.defocus_slider.setRange(-500, 500)
        self.defocus_slider.setValue(0)
        self.defocus_slider.setTickInterval(100)
        self.defocus_slider.setTickPosition(QSlider.TicksBelow)
        self.defocus_slider.valueChanged.connect(self._slider_to_spinbox)
        df_layout.addWidget(self.defocus_slider)

        self.defocus_sb = QDoubleSpinBox()
        self.defocus_sb.setRange(-500.0, 500.0)
        self.defocus_sb.setValue(0.0)
        self.defocus_sb.setSuffix(" nm")
        self.defocus_sb.setDecimals(0)
        self.defocus_sb.setMaximumWidth(90)
        self.defocus_sb.valueChanged.connect(self._spinbox_to_slider)
        df_layout.addWidget(self.defocus_sb)

        layout.addWidget(defocus_group)

        # Mask & Grid
        mask_group = QGroupBox("Mask & Grid")
        mask_form = QFormLayout(mask_group)

        self.mask_combo = QComboBox()
        self.mask_combo.addItems(["Binary", "AttPSM", "AltPSM"])
        self.mask_combo.currentTextChanged.connect(self.params_changed)
        mask_form.addRow("Mask type:", self.mask_combo)

        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["64", "128", "256", "512"])
        self.grid_combo.setCurrentText("256")
        self.grid_combo.currentTextChanged.connect(self.params_changed)
        mask_form.addRow("Grid size:", self.grid_combo)

        self.domain_sb = QDoubleSpinBox()
        self.domain_sb.setRange(100.0, 10000.0)
        self.domain_sb.setValue(2000.0)
        self.domain_sb.setSuffix(" nm")
        self.domain_sb.setDecimals(0)
        self.domain_sb.valueChanged.connect(self.params_changed)
        mask_form.addRow("Domain size:", self.domain_sb)

        layout.addWidget(mask_group)

        # Buttons
        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load YAML")
        self.save_btn = QPushButton("Save YAML")
        self.load_btn.clicked.connect(self._load_config)
        self.save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)
        layout.addStretch()

    def _on_illum_changed(self, text):
        self.sigma_inner_sb.setEnabled(text in ["Annular", "Quadrupole", "Quasar"])
        self.params_changed.emit()

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

    def get_config(self):
        illum_map = {"Circular": "circular", "Annular": "annular",
                     "Quadrupole": "quadrupole", "Quasar": "quasar"}
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
            },
            "simulation": {
                "grid_size": int(self.grid_combo.currentText()),
                "domain_size_nm": self.domain_sb.value(),
            }
        }

    def load_config(self, config):
        litho = config.get("lithography", {})
        sim = config.get("simulation", {})
        illum = litho.get("illumination", {})
        self.wavelength_sb.setValue(litho.get("wavelength_nm", 193.0))
        self.na_sb.setValue(litho.get("NA", 0.93))
        self.defocus_sb.setValue(litho.get("defocus_nm", 0.0))
        self.sigma_outer_sb.setValue(illum.get("sigma_outer", 0.85))
        self.sigma_inner_sb.setValue(illum.get("sigma_inner", 0.55))
        illum_rmap = {"circular": "Circular", "annular": "Annular",
                      "quadrupole": "Quadrupole", "quasar": "Quasar"}
        self.illum_combo.setCurrentText(
            illum_rmap.get(illum.get("type", "annular"), "Annular"))
        mask_rmap = {"binary": "Binary", "att_psm": "AttPSM", "alt_psm": "AltPSM"}
        self.mask_combo.setCurrentText(
            mask_rmap.get(litho.get("mask_type", "binary"), "Binary"))
        self.grid_combo.setCurrentText(str(sim.get("grid_size", 256)))
        self.domain_sb.setValue(sim.get("domain_size_nm", 2000.0))

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "", "YAML files (*.yaml *.yml)")
        if not path:
            return
        try:
            import sys, os
            sim_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
            sim_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if sim_dir not in sys.path:
                sys.path.insert(0, sim_dir)
            from fileio.parameter_io import save_config
            save_config(self.get_config(), path)
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))
