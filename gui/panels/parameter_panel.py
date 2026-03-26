"""
Parameter panel for optical lithography simulator.
Controls: preset selector, wavelength, NA, illumination, sigma, defocus, mask type, grid size,
and aberration (Zernike) parameters.
"""
from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QComboBox, QSlider, QPushButton, QFileDialog,
    QMessageBox, QScrollArea, QLabel, QSpinBox, Qt, Signal, QFont,
    QCheckBox,
)
from core.aberrations import ZERNIKE_TABLE

try:
    from PySide6.QtWidgets import QStackedWidget, QDialog
except ImportError:
    from PyQt5.QtWidgets import QStackedWidget, QDialog  # type: ignore

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
    source_preview_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._applying_preset = False
        # Illumination types not representable in the panel combo (dipole,
        # freeform).  Set by load_config() when such a type arrives; cleared
        # whenever the user touches any illumination control in the panel.
        self._illum_override = None
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
        layout.setSpacing(10)

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
        litho_vbox = QVBoxLayout(litho_group)
        litho_vbox.setContentsMargins(8, 12, 8, 8)
        litho_vbox.setSpacing(6)

        # Quick preset buttons (wavelength + NA shortcuts)
        quick_row = QHBoxLayout()
        quick_row.setSpacing(6)
        for _lbl, _wl, _na in [
            ("ArF 193nm", 193.0, 0.93),
            ("KrF 248nm", 248.0, 0.75),
            ("EUV 13.5nm", 13.5, 0.33),
            ("i-line 365nm", 365.0, 0.65),
        ]:
            _btn = QPushButton(_lbl)
            _btn.setObjectName("secondary")
            _btn.setMaximumHeight(30)
            _btn.setToolTip("Set wavelength={} nm, NA={}".format(_wl, _na))
            _btn.clicked.connect(
                lambda checked=False, w=_wl, n=_na: self._apply_quick_preset(w, n))
            quick_row.addWidget(_btn)
        litho_vbox.addLayout(quick_row)

        form = QFormLayout()
        form.setSpacing(6)
        litho_vbox.addLayout(form)

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
        self.illum_combo.addItems(["Circular", "Annular", "Quadrupole", "Quasar", "Dipole", "Freeform"])
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

        # ---- Illumination-type-specific extra controls ----
        self.illum_stack = QStackedWidget()
        self.illum_stack.setVisible(False)

        # Page 0: placeholder for standard illumination types
        self.illum_stack.addWidget(QWidget())

        # Page 1: Dipole controls
        dipole_page = QWidget()
        dip_form = QFormLayout(dipole_page)
        dip_form.setContentsMargins(0, 4, 0, 4)
        dip_form.setSpacing(6)

        self.dipole_sigma_center_sb = QDoubleSpinBox()
        self.dipole_sigma_center_sb.setRange(0.1, 0.95)
        self.dipole_sigma_center_sb.setValue(0.5)
        self.dipole_sigma_center_sb.setDecimals(2)
        self.dipole_sigma_center_sb.setSingleStep(0.05)
        self.dipole_sigma_center_sb.setToolTip("Dipole pole center sigma")
        self.dipole_sigma_center_sb.valueChanged.connect(self._on_spinbox_changed)
        dip_form.addRow("sigma center:", self.dipole_sigma_center_sb)

        self.dipole_sigma_outer_sb = QDoubleSpinBox()
        self.dipole_sigma_outer_sb.setRange(0.01, 1.0)
        self.dipole_sigma_outer_sb.setValue(0.7)
        self.dipole_sigma_outer_sb.setDecimals(2)
        self.dipole_sigma_outer_sb.setSingleStep(0.05)
        self.dipole_sigma_outer_sb.setToolTip("Dipole outer sigma")
        self.dipole_sigma_outer_sb.valueChanged.connect(self._on_spinbox_changed)
        dip_form.addRow("sigma outer:", self.dipole_sigma_outer_sb)

        self.dipole_sigma_inner_sb = QDoubleSpinBox()
        self.dipole_sigma_inner_sb.setRange(0.0, 1.0)
        self.dipole_sigma_inner_sb.setValue(0.3)
        self.dipole_sigma_inner_sb.setDecimals(2)
        self.dipole_sigma_inner_sb.setSingleStep(0.05)
        self.dipole_sigma_inner_sb.setToolTip("Dipole inner sigma")
        self.dipole_sigma_inner_sb.valueChanged.connect(self._on_spinbox_changed)
        dip_form.addRow("sigma inner:", self.dipole_sigma_inner_sb)

        self.dipole_orientation_combo = QComboBox()
        self.dipole_orientation_combo.addItems(["X", "Y"])
        self.dipole_orientation_combo.setToolTip("Dipole orientation axis")
        self.dipole_orientation_combo.currentTextChanged.connect(self._on_spinbox_changed)
        dip_form.addRow("Orientation:", self.dipole_orientation_combo)

        self.dipole_opening_angle_sb = QDoubleSpinBox()
        self.dipole_opening_angle_sb.setRange(10.0, 90.0)
        self.dipole_opening_angle_sb.setValue(30.0)
        self.dipole_opening_angle_sb.setDecimals(1)
        self.dipole_opening_angle_sb.setSingleStep(5.0)
        self.dipole_opening_angle_sb.setSuffix("\u00b0")
        self.dipole_opening_angle_sb.setToolTip("Dipole opening angle (degrees)")
        self.dipole_opening_angle_sb.valueChanged.connect(self._on_spinbox_changed)
        dip_form.addRow("Opening angle:", self.dipole_opening_angle_sb)

        self.illum_stack.addWidget(dipole_page)

        # Page 2: Freeform label
        freeform_page = QWidget()
        ff_layout = QVBoxLayout(freeform_page)
        ff_layout.setContentsMargins(0, 4, 0, 4)
        ff_lbl = QLabel(
            "Use Source Preview dialog\n(Simulation \u2192 Source Preview)\n"
            "to define freeform illumination."
        )
        ff_lbl.setObjectName("caption")
        ff_lbl.setWordWrap(True)
        ff_layout.addWidget(ff_lbl)
        freeform_btn = QPushButton("Open Source Preview...")
        freeform_btn.setObjectName("secondary")
        freeform_btn.clicked.connect(self._request_source_preview)
        ff_layout.addWidget(freeform_btn)
        self.illum_stack.addWidget(freeform_page)

        form.addRow(self.illum_stack)

        self.dose_factor_sb = QDoubleSpinBox()
        self.dose_factor_sb.setRange(0.5, 2.0)
        self.dose_factor_sb.setValue(1.0)
        self.dose_factor_sb.setDecimals(2)
        self.dose_factor_sb.setSingleStep(0.05)
        self.dose_factor_sb.setToolTip(
            "Scale factor for aerial image intensity (1.0 = nominal dose)"
        )
        self.dose_factor_sb.valueChanged.connect(self._on_spinbox_changed)
        form.addRow("Dose Factor:", self.dose_factor_sb)

        layout.addWidget(litho_group)

        # ---- Defocus ----
        defocus_group = QGroupBox("Defocus")
        defocus_group.setFlat(False)

        df_outer = QVBoxLayout(defocus_group)
        df_outer.setContentsMargins(8, 12, 8, 8)

        df_row = QHBoxLayout()

        self.defocus_slider = QSlider(Qt.Horizontal)
        self.defocus_slider.setRange(-2000, 2000)
        self.defocus_slider.setValue(0)
        self.defocus_slider.setTickInterval(500)
        self.defocus_slider.setTickPosition(QSlider.TicksBelow)
        self.defocus_slider.setToolTip("Focal plane offset from best focus")
        self.defocus_slider.valueChanged.connect(self._slider_to_spinbox)
        df_row.addWidget(self.defocus_slider)

        self.defocus_sb = QDoubleSpinBox()
        self.defocus_sb.setRange(-2000.0, 2000.0)
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
        lbl_min = QLabel("-2000 nm")
        lbl_max = QLabel("+2000 nm")
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

        # ---- Compute Backend ----
        backend_group = QGroupBox("Compute Backend")
        backend_group.setFlat(False)
        backend_form = QFormLayout(backend_group)
        backend_form.setContentsMargins(8, 12, 8, 8)

        from core.gpu_backend import HAS_GPU
        self.gpu_cb = QCheckBox()
        self.gpu_cb.setChecked(False)
        if HAS_GPU:
            self.gpu_cb.setToolTip("Use CuPy GPU backend for FFT-accelerated imaging")
        else:
            self.gpu_cb.setEnabled(False)
            self.gpu_cb.setToolTip("CuPy not installed — GPU acceleration unavailable")
        self.gpu_cb.stateChanged.connect(self.params_changed)
        backend_form.addRow("GPU Acceleration:", self.gpu_cb)

        self.hopkins_cb = QCheckBox()
        self.hopkins_cb.setChecked(False)
        self.hopkins_cb.setToolTip(
            "Use Hopkins TCC (Transmission Cross-Coefficient) partial coherence decomposition"
        )
        self.hopkins_cb.stateChanged.connect(self._on_hopkins_toggled)
        backend_form.addRow("Hopkins TCC:", self.hopkins_cb)

        self.n_kernels_sb = QSpinBox()
        self.n_kernels_sb.setRange(1, 50)
        self.n_kernels_sb.setValue(10)
        self.n_kernels_sb.setToolTip("Number of TCC eigenkernels (more = more accurate, slower)")
        self.n_kernels_sb.setEnabled(False)
        self.n_kernels_sb.valueChanged.connect(self.params_changed)
        backend_form.addRow("TCC kernels:", self.n_kernels_sb)

        self.vector_cb = QCheckBox()
        self.vector_cb.setChecked(False)
        self.vector_cb.setToolTip(
            "Use vector imaging model (polarization-aware; slower but required for high-NA)"
        )
        self.vector_cb.stateChanged.connect(self._on_vector_toggled)
        backend_form.addRow("Vector Imaging:", self.vector_cb)

        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems(
            ["Unpolarized", "X", "Y", "TE", "TM", "Circular L", "Circular R"]
        )
        self.polarization_combo.setToolTip("Polarization state of the illumination")
        self.polarization_combo.setEnabled(False)
        self.polarization_combo.currentTextChanged.connect(self.params_changed)
        backend_form.addRow("Polarization:", self.polarization_combo)

        layout.addWidget(backend_group)

        # ---- Aberrations (Z1-Z37) — collapsible ----
        aber_group = QGroupBox("Aberrations (Z1–Z37)")
        aber_group.setCheckable(True)
        aber_group.setChecked(False)
        aber_group.setFlat(False)

        aber_vbox = QVBoxLayout(aber_group)
        aber_vbox.setContentsMargins(8, 4, 8, 8)
        aber_vbox.setSpacing(0)

        aber_content = QWidget()
        aber_form = QFormLayout(aber_content)
        aber_form.setContentsMargins(0, 4, 0, 0)
        aber_form.setSpacing(2)

        self.zernike_sbs = []
        for idx, (n, m, name) in enumerate(ZERNIKE_TABLE, start=1):
            sb = QDoubleSpinBox()
            sb.setRange(-0.5, 0.5)
            sb.setValue(0.0)
            sb.setDecimals(3)
            sb.setSingleStep(0.01)
            sb.setSuffix(" \u03bb")
            sb.setToolTip("Z{}: {} (n={}, m={})".format(idx, name, n, m))
            sb.valueChanged.connect(self.params_changed)
            aber_form.addRow("Z{} ({}):".format(idx, name), sb)
            self.zernike_sbs.append(sb)

        aber_vbox.addWidget(aber_content)
        aber_content.setVisible(False)
        aber_group.toggled.connect(aber_content.setVisible)

        layout.addWidget(aber_group)

        # ---- RCWA Near-Field Correction ----
        rcwa_group = QGroupBox("RCWA Near-Field Correction")
        rcwa_group.setFlat(False)

        rcwa_form = QFormLayout(rcwa_group)
        rcwa_form.setContentsMargins(8, 12, 8, 8)

        self.rcwa_enabled_cb = QCheckBox()
        self.rcwa_enabled_cb.setChecked(False)
        self.rcwa_enabled_cb.setToolTip(
            "Apply RCWA near-field correction to mask after loading"
        )
        self.rcwa_enabled_cb.stateChanged.connect(self.params_changed)
        rcwa_form.addRow("Enable RCWA:", self.rcwa_enabled_cb)

        self.rcwa_n_orders_sb = QSpinBox()
        self.rcwa_n_orders_sb.setRange(1, 51)
        self.rcwa_n_orders_sb.setValue(11)
        self.rcwa_n_orders_sb.setSingleStep(2)
        self.rcwa_n_orders_sb.setToolTip("Number of diffraction orders (odd)")
        self.rcwa_n_orders_sb.valueChanged.connect(self.params_changed)
        rcwa_form.addRow("Diffraction orders:", self.rcwa_n_orders_sb)

        layout.addWidget(rcwa_group)

        # ---- Resist Model ----
        resist_group = QGroupBox("Resist Model")
        resist_group.setFlat(False)
        resist_group.setCheckable(True)
        resist_group.setChecked(True)
        self.resist_group = resist_group
        resist_outer = QVBoxLayout(resist_group)
        resist_outer.setContentsMargins(8, 12, 8, 8)
        resist_outer.setSpacing(6)

        resist_top_form = QFormLayout()
        resist_top_form.setSpacing(6)

        self.resist_combo = QComboBox()
        self.resist_combo.addItems(["Threshold", "Dill ABC", "Chemically Amplified (CA)"])
        self.resist_combo.setToolTip("Select resist exposure/development model")
        self.resist_combo.currentIndexChanged.connect(self._on_resist_model_changed)
        resist_top_form.addRow("Model:", self.resist_combo)
        resist_outer.addLayout(resist_top_form)

        self.resist_stack = QStackedWidget()

        # Page 0: Threshold
        thresh_page = QWidget()
        thresh_form = QFormLayout(thresh_page)
        thresh_form.setContentsMargins(0, 0, 0, 0)
        thresh_form.setSpacing(6)
        self.resist_threshold_sb = QDoubleSpinBox()
        self.resist_threshold_sb.setRange(0.1, 0.9)
        self.resist_threshold_sb.setValue(0.30)
        self.resist_threshold_sb.setDecimals(2)
        self.resist_threshold_sb.setSingleStep(0.05)
        self.resist_threshold_sb.setToolTip("Development threshold on normalized intensity")
        self.resist_threshold_sb.valueChanged.connect(self.params_changed)
        thresh_form.addRow("Threshold:", self.resist_threshold_sb)
        self.resist_stack.addWidget(thresh_page)

        # Page 1: Dill ABC
        dill_page = QWidget()
        dill_form = QFormLayout(dill_page)
        dill_form.setContentsMargins(0, 0, 0, 0)
        dill_form.setSpacing(6)
        self.dill_A_sb = QDoubleSpinBox()
        self.dill_A_sb.setRange(0.001, 5.0)   # validator requires A > 0
        self.dill_A_sb.setValue(0.8)
        self.dill_A_sb.setDecimals(3)
        self.dill_A_sb.setToolTip("Dill A: bleachable absorption coefficient [1/mJ·cm⁻²]")
        self.dill_A_sb.valueChanged.connect(self.params_changed)
        dill_form.addRow("A (bleach):", self.dill_A_sb)

        self.dill_B_sb = QDoubleSpinBox()
        self.dill_B_sb.setRange(0.0, 5.0)
        self.dill_B_sb.setValue(0.0)
        self.dill_B_sb.setDecimals(3)
        self.dill_B_sb.setToolTip("Dill B: non-bleachable absorption [1/cm]")
        self.dill_B_sb.valueChanged.connect(self.params_changed)
        dill_form.addRow("B (non-bleach):", self.dill_B_sb)

        self.dill_C_sb = QDoubleSpinBox()
        self.dill_C_sb.setRange(0.001, 10.0)
        self.dill_C_sb.setValue(1.0)
        self.dill_C_sb.setDecimals(3)
        self.dill_C_sb.setSingleStep(0.1)
        self.dill_C_sb.setToolTip("Dill C: exposure rate constant [cm²/mJ]")
        self.dill_C_sb.valueChanged.connect(self.params_changed)
        dill_form.addRow("C (rate):", self.dill_C_sb)

        self.dill_peb_sb = QDoubleSpinBox()
        self.dill_peb_sb.setRange(0.0, 200.0)
        self.dill_peb_sb.setValue(30.0)
        self.dill_peb_sb.setSuffix(" nm")
        self.dill_peb_sb.setDecimals(1)
        self.dill_peb_sb.setToolTip("Post-exposure bake diffusion sigma (nm)")
        self.dill_peb_sb.valueChanged.connect(self.params_changed)
        dill_form.addRow("PEB σ:", self.dill_peb_sb)
        self.resist_stack.addWidget(dill_page)

        # Page 2: Chemically Amplified (CA)
        ca_page = QWidget()
        ca_form = QFormLayout(ca_page)
        ca_form.setContentsMargins(0, 0, 0, 0)
        ca_form.setSpacing(6)
        self.ca_qe_sb = QDoubleSpinBox()
        self.ca_qe_sb.setRange(0.001, 0.999)   # validator requires QE in (0, 1)
        self.ca_qe_sb.setValue(0.5)
        self.ca_qe_sb.setDecimals(3)
        self.ca_qe_sb.setToolTip("Quantum efficiency: acid molecules generated per absorbed photon")
        self.ca_qe_sb.valueChanged.connect(self.params_changed)
        ca_form.addRow("Quantum eff.:", self.ca_qe_sb)

        self.ca_amp_sb = QDoubleSpinBox()
        self.ca_amp_sb.setRange(1.0, 500.0)
        self.ca_amp_sb.setValue(50.0)
        self.ca_amp_sb.setDecimals(1)
        self.ca_amp_sb.setToolTip("Acid amplification (catalytic chain length)")
        self.ca_amp_sb.valueChanged.connect(self.params_changed)
        ca_form.addRow("Amplification:", self.ca_amp_sb)

        self.ca_peb_sb = QDoubleSpinBox()
        self.ca_peb_sb.setRange(0.0, 200.0)
        self.ca_peb_sb.setValue(25.0)
        self.ca_peb_sb.setSuffix(" nm")
        self.ca_peb_sb.setDecimals(1)
        self.ca_peb_sb.setToolTip("Post-exposure bake diffusion sigma (nm)")
        self.ca_peb_sb.valueChanged.connect(self.params_changed)
        ca_form.addRow("PEB σ:", self.ca_peb_sb)

        self.ca_exposure_threshold_sb = QDoubleSpinBox()
        self.ca_exposure_threshold_sb.setRange(0.01, 1.0)
        self.ca_exposure_threshold_sb.setValue(0.3)
        self.ca_exposure_threshold_sb.setDecimals(3)
        self.ca_exposure_threshold_sb.setSingleStep(0.05)
        self.ca_exposure_threshold_sb.setToolTip("Development threshold on deprotection level [0,1]")
        self.ca_exposure_threshold_sb.valueChanged.connect(self.params_changed)
        ca_form.addRow("Exposure Threshold:", self.ca_exposure_threshold_sb)
        self.resist_stack.addWidget(ca_page)

        resist_outer.addWidget(self.resist_stack)

        self.stack_editor_btn = QPushButton("Stack Editor...")
        self.stack_editor_btn.setObjectName("secondary")
        self.stack_editor_btn.setToolTip("Open film stack visual editor (TMM reflectance preview)")
        self.stack_editor_btn.clicked.connect(self._open_stack_editor)
        resist_outer.addWidget(self.stack_editor_btn)

        self._film_stack = None
        layout.addWidget(resist_group)

        # ---- Analysis ----
        analysis_group = QGroupBox("Analysis")
        analysis_group.setFlat(False)
        analysis_form = QFormLayout(analysis_group)
        analysis_form.setContentsMargins(8, 12, 8, 8)

        self.cd_threshold_sb = QDoubleSpinBox()
        self.cd_threshold_sb.setRange(0.01, 0.99)
        self.cd_threshold_sb.setValue(0.30)
        self.cd_threshold_sb.setDecimals(2)
        self.cd_threshold_sb.setSingleStep(0.01)
        self.cd_threshold_sb.setToolTip("Intensity threshold used for CD measurement")
        self.cd_threshold_sb.valueChanged.connect(self.params_changed)
        analysis_form.addRow("CD Threshold:", self.cd_threshold_sb)

        layout.addWidget(analysis_group)

        # ---- Load / Save buttons ----
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.setContentsMargins(8, 4, 8, 8)
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

    def _apply_quick_preset(self, wavelength_nm, na):
        """Apply a quick wavelength/NA preset without changing sigma values."""
        self._applying_preset = True
        try:
            self.wavelength_sb.setValue(wavelength_nm)
            self.na_sb.setValue(na)
        finally:
            self._applying_preset = False
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Custom")
        self.preset_combo.blockSignals(False)
        self.params_changed.emit()

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
        """Any manual spinbox edit switches preset combo to 'Custom'.
        Also enforces sigma_inner < sigma_outer constraint."""
        if self._applying_preset:
            return
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Custom")
        self.preset_combo.blockSignals(False)
        # User is explicitly editing illumination params — drop any
        # override from SourceDialog so the panel values take effect.
        self._illum_override = None
        # Enforce sigma_inner < sigma_outer for annular-type sources
        if self.sigma_inner_sb.isEnabled():
            if self.sigma_inner_sb.value() >= self.sigma_outer_sb.value():
                self.sigma_inner_sb.blockSignals(True)
                self.sigma_inner_sb.setValue(
                    max(0.0, self.sigma_outer_sb.value() - 0.05))
                self.sigma_inner_sb.blockSignals(False)
        self.params_changed.emit()

    # ------------------------------------------------------------------
    # Illumination change
    # ------------------------------------------------------------------

    def _on_illum_changed(self, text):
        standard = ["Circular", "Annular", "Quadrupole", "Quasar"]
        self.sigma_outer_sb.setEnabled(text in standard)
        self.sigma_inner_sb.setEnabled(text in ["Annular", "Quadrupole", "Quasar"])
        if text == "Dipole":
            self.illum_stack.setCurrentIndex(1)
            self.illum_stack.setVisible(True)
        elif text == "Freeform":
            self.illum_stack.setCurrentIndex(2)
            self.illum_stack.setVisible(True)
        else:
            self.illum_stack.setCurrentIndex(0)
            self.illum_stack.setVisible(False)
        # Keep freeform override when user selects "Freeform" from combo
        if not (text == "Freeform" and self._illum_override is not None
                and self._illum_override.get("type") == "freeform"):
            self._illum_override = None
        self.params_changed.emit()

    def _on_hopkins_toggled(self, state):
        self.n_kernels_sb.setEnabled(bool(state))
        self.params_changed.emit()

    def _on_vector_toggled(self, state):
        self.polarization_combo.setEnabled(bool(state))
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

    def get_polarization(self) -> str:
        """Return the selected polarization mode."""
        return self.polarization_combo.currentText()

    def get_config(self):
        illum_map = {
            "Circular": "circular", "Annular": "annular",
            "Quadrupole": "quadrupole", "Quasar": "quasar",
        }
        mask_map = {"Binary": "binary", "AttPSM": "att_psm", "AltPSM": "alt_psm"}
        current_illum = self.illum_combo.currentText()
        if self._illum_override is not None:
            illum_cfg = self._illum_override
        elif current_illum == "Dipole":
            illum_cfg = {
                "type": "dipole",
                "sigma_center": self.dipole_sigma_center_sb.value(),
                "sigma_outer": self.dipole_sigma_outer_sb.value(),
                "sigma_inner": self.dipole_sigma_inner_sb.value(),
                "orientation": self.dipole_orientation_combo.currentText().lower(),
                "opening_angle_deg": self.dipole_opening_angle_sb.value(),
            }
        elif current_illum == "Freeform":
            illum_cfg = {"type": "freeform"}
        else:
            illum_cfg = {
                "type": illum_map[current_illum],
                "sigma_outer": self.sigma_outer_sb.value(),
                "sigma_inner": self.sigma_inner_sb.value(),
            }
        return {
            "lithography": {
                "wavelength_nm": self.wavelength_sb.value(),
                "NA": self.na_sb.value(),
                "defocus_nm": self.defocus_sb.value(),
                "dose_factor": self.dose_factor_sb.value(),
                "illumination": illum_cfg,
                "mask_type": mask_map[self.mask_combo.currentText()],
                "aberrations": {
                    "zernike": [sb.value() for sb in self.zernike_sbs],
                },
                "use_hopkins": self.hopkins_cb.isChecked(),
                "n_kernels": self.n_kernels_sb.value(),
                "use_vector": self.vector_cb.isChecked(),
                "polarization": self.polarization_combo.currentText().lower().replace(" ", "_"),
            },
            "simulation": {
                "grid_size": int(self.grid_combo.currentText()),
                "domain_size_nm": self.domain_sb.value(),
                "use_gpu": self.gpu_cb.isChecked(),
            },
            "rcwa": {
                "enabled": self.rcwa_enabled_cb.isChecked(),
                "n_orders": self.rcwa_n_orders_sb.value(),
            },
            "resist": self._get_resist_config(),
            "analysis": {
                "cd_threshold": self.cd_threshold_sb.value(),
            },
        }

    def _get_resist_config(self) -> dict:
        if self.resist_group.isCheckable() and not self.resist_group.isChecked():
            return {'model': 'threshold', 'threshold': 1.0, 'dose': 1.0}
        idx = self.resist_combo.currentIndex()
        if idx == 1:  # Dill ABC
            return {
                "model": "dill",
                "A": self.dill_A_sb.value(),
                "B": self.dill_B_sb.value(),
                "C": self.dill_C_sb.value(),
                "peb_sigma_nm": self.dill_peb_sb.value(),
            }
        if idx == 2:  # CA
            return {
                "model": "ca",
                "quantum_efficiency": self.ca_qe_sb.value(),
                "amplification": self.ca_amp_sb.value(),
                "peb_sigma_nm": self.ca_peb_sb.value(),
                "exposure_threshold": self.ca_exposure_threshold_sb.value(),
            }
        return {"model": "threshold", "threshold": self.resist_threshold_sb.value()}

    def load_config(self, config):
        litho = config.get("lithography", {})
        sim = config.get("simulation", {})
        illum = litho.get("illumination", {})
        aber = litho.get("aberrations", {})

        illum_rmap = {
            "circular": "Circular", "annular": "Annular",
            "quadrupole": "Quadrupole", "quasar": "Quasar",
            "dipole": "Dipole", "freeform": "Freeform",
        }
        mask_rmap = {"binary": "Binary", "att_psm": "AttPSM", "alt_psm": "AltPSM"}

        self._applying_preset = True
        try:
            self.wavelength_sb.setValue(litho.get("wavelength_nm", 193.0))
            self.na_sb.setValue(litho.get("NA", 0.93))
            self.defocus_sb.setValue(litho.get("defocus_nm", 0.0))
            self.dose_factor_sb.setValue(litho.get("dose_factor", 1.0))
            self.sigma_outer_sb.setValue(illum.get("sigma_outer", 0.85))
            self.sigma_inner_sb.setValue(illum.get("sigma_inner", 0.55))
            illum_type = illum.get("type", "annular")
            if illum_type in illum_rmap:
                self.illum_combo.setCurrentText(illum_rmap[illum_type])
                if illum_type == "dipole":
                    self._illum_override = None
                    self.dipole_sigma_center_sb.setValue(illum.get("sigma_center", 0.5))
                    self.dipole_sigma_outer_sb.setValue(illum.get("sigma_outer", 0.7))
                    self.dipole_sigma_inner_sb.setValue(illum.get("sigma_inner", 0.3))
                    self.dipole_orientation_combo.setCurrentText(
                        illum.get("orientation", "x").upper()
                    )
                    self.dipole_opening_angle_sb.setValue(illum.get("opening_angle_deg", 30.0))
                elif illum_type == "freeform":
                    # Store full freeform config (from SourceDialog) as override
                    self._illum_override = dict(illum) if len(illum) > 1 else None
                else:
                    self._illum_override = None
            else:
                self._illum_override = dict(illum)
            self.mask_combo.setCurrentText(
                mask_rmap.get(litho.get("mask_type", "binary"), "Binary")
            )
            zernike_coeffs = aber.get("zernike", [0.0] * 37)
            for i, sb in enumerate(self.zernike_sbs):
                sb.setValue(zernike_coeffs[i] if i < len(zernike_coeffs) else 0.0)
        finally:
            self._applying_preset = False

        self.grid_combo.setCurrentText(str(sim.get("grid_size", 256)))
        self.domain_sb.setValue(sim.get("domain_size_nm", 2000.0))
        if self.gpu_cb.isEnabled():
            self.gpu_cb.setChecked(sim.get("use_gpu", False))

        rcwa = config.get("rcwa", {})
        self.rcwa_enabled_cb.setChecked(rcwa.get("enabled", False))
        self.rcwa_n_orders_sb.setValue(rcwa.get("n_orders", 11))

        resist_cfg = config.get("resist", {})
        model = resist_cfg.get("model", "threshold")
        model_idx = {"threshold": 0, "dill": 1, "ca": 2}.get(model, 0)
        self.resist_combo.setCurrentIndex(model_idx)
        self.resist_threshold_sb.setValue(resist_cfg.get("threshold", 0.30))
        self.dill_A_sb.setValue(resist_cfg.get("A", 0.8))
        self.dill_B_sb.setValue(resist_cfg.get("B", 0.0))
        self.dill_C_sb.setValue(resist_cfg.get("C", 1.0))
        self.dill_peb_sb.setValue(resist_cfg.get("peb_sigma_nm", 30.0))
        self.ca_qe_sb.setValue(resist_cfg.get("quantum_efficiency", 0.5))
        self.ca_amp_sb.setValue(resist_cfg.get("amplification", 50.0))
        self.ca_peb_sb.setValue(resist_cfg.get("peb_sigma_nm", 25.0))
        self.ca_exposure_threshold_sb.setValue(resist_cfg.get("exposure_threshold", 0.3))

        analysis_cfg = config.get("analysis", {})
        self.cd_threshold_sb.setValue(analysis_cfg.get("cd_threshold", 0.30))

        self.hopkins_cb.setChecked(litho.get("use_hopkins", False))
        self.n_kernels_sb.setValue(litho.get("n_kernels", 10))
        self.n_kernels_sb.setEnabled(litho.get("use_hopkins", False))
        self.vector_cb.setChecked(litho.get("use_vector", False))
        pol_rmap = {
            "unpolarized": "Unpolarized", "x": "X", "y": "Y",
            "te": "TE", "tm": "TM", "circular_l": "Circular L", "circular_r": "Circular R",
        }
        self.polarization_combo.setCurrentText(
            pol_rmap.get(litho.get("polarization", "unpolarized"), "Unpolarized")
        )
        self.polarization_combo.setEnabled(litho.get("use_vector", False))

        # Mark as custom after loading
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Custom")
        self.preset_combo.blockSignals(False)
        self.params_changed.emit()

    # ------------------------------------------------------------------
    # Resist model
    # ------------------------------------------------------------------

    def _on_resist_model_changed(self, idx: int):
        self.resist_stack.setCurrentIndex(idx)
        self.params_changed.emit()

    def _request_source_preview(self):
        self.source_preview_requested.emit()

    def _open_stack_editor(self):
        from gui.dialogs.stack_dialog import StackDialog
        dlg = StackDialog(self, film_stack=self._film_stack)
        if dlg.exec() == QDialog.Accepted:
            self._film_stack = dlg.get_film_stack()

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
