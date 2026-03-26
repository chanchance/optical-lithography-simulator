"""Mask settings dialog."""
try:
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QLabel, QComboBox,
        QFormLayout, QDialogButtonBox
    )
    from PySide6.QtCore import Qt
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QLabel, QComboBox,
            QFormLayout, QDialogButtonBox
        )
        from PyQt5.QtCore import Qt
    except ImportError:
        QDialog = object
        Qt = None

from gui import theme

_MASK_DESCRIPTIONS = {
    0: "Binary mask — fully opaque/transparent (Cr/MoSi absorber)",
    1: "Attenuated PSM — 6% transmittance, 180° phase shift",
    2: "Alternating PSM — adjacent apertures alternated 180° phase",
}


class MaskDialog(QDialog if QDialog != object else object):
    """Dialog for configuring mask type and bias settings."""

    def __init__(self, parent=None, mask_type='binary'):
        if QDialog is object:
            return
        super().__init__(parent)
        self.setStyleSheet(theme.get_qss())
        self.setWindowTitle('Mask Settings')
        self.setMinimumWidth(380)
        self._mask_type = mask_type
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)

        form = QFormLayout()

        self.type_combo = QComboBox()
        self.type_combo.addItems(['Binary', 'AttPSM (6%)', 'AltPSM'])
        idx = {'binary': 0, 'att_psm': 1, 'attpsm': 1, 'alt_psm': 2, 'altpsm': 2}.get(
            self._mask_type.lower(), 0)
        self.type_combo.setCurrentIndex(idx)
        form.addRow('Mask type:', self.type_combo)

        layout.addLayout(form)

        self.desc_label = QLabel(_MASK_DESCRIPTIONS[idx])
        self.desc_label.setWordWrap(True)
        self.desc_label.setMaximumHeight(54)  # ~3 lines at 18px line height
        self.desc_label.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: 11px;"
        )
        layout.addWidget(self.desc_label)

        self.type_combo.currentIndexChanged.connect(self._on_type_changed)

        layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_btn = buttons.button(QDialogButtonBox.Ok)
        ok_btn.setProperty("class", "primary")
        cancel_btn = buttons.button(QDialogButtonBox.Cancel)
        cancel_btn.setObjectName("secondary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_type_changed(self, index: int):
        self.desc_label.setText(_MASK_DESCRIPTIONS.get(index, ""))

    def get_mask_type(self) -> str:
        """Return selected mask type key (lowercase, consistent with mask_model)."""
        map_ = {0: 'binary', 1: 'att_psm', 2: 'alt_psm'}
        return map_.get(self.type_combo.currentIndex(), 'binary')
