"""Mask settings dialog (stub)."""
try:
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QLabel, QComboBox,
        QFormLayout, QDialogButtonBox
    )
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QLabel, QComboBox,
            QFormLayout, QDialogButtonBox
        )
    except ImportError:
        QDialog = object


class MaskDialog(QDialog if QDialog != object else object):
    """Dialog for configuring mask type and bias settings."""

    def __init__(self, parent=None, mask_type='binary'):
        if QDialog is object:
            return
        super().__init__(parent)
        self.setWindowTitle('Mask Settings')
        self._mask_type = mask_type
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.type_combo = QComboBox()
        self.type_combo.addItems(['Binary', 'AttPSM (6%)', 'AltPSM'])
        idx = {'binary': 0, 'att_psm': 1, 'attpsm': 1, 'alt_psm': 2, 'altpsm': 2}.get(
            self._mask_type.lower(), 0)
        self.type_combo.setCurrentIndex(idx)
        form.addRow('Mask type:', self.type_combo)

        layout.addLayout(form)
        layout.addWidget(QLabel('Additional mask settings coming soon.'))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_mask_type(self) -> str:
        """Return selected mask type key (lowercase, consistent with mask_model)."""
        map_ = {0: 'binary', 1: 'att_psm', 2: 'alt_psm'}
        return map_.get(self.type_combo.currentIndex(), 'binary')
