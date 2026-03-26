"""
Simulation control panel: mode selection, run/stop, progress bar, log.
"""
try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
        QRadioButton, QPushButton, QProgressBar, QTextEdit, QLabel
    )
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QTextCursor
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
        QRadioButton, QPushButton, QProgressBar, QTextEdit, QLabel
    )
    from PyQt5.QtCore import Qt, pyqtSignal as Signal
    from PyQt5.QtGui import QTextCursor


class SimulationPanel(QWidget):
    """Panel for controlling simulation execution."""

    run_requested = Signal(str)   # emits mode string
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        mode_group = QGroupBox("Simulation Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.fourier_radio = QRadioButton("Fourier Optics  (fast, scalar)")
        self.fdtd_radio = QRadioButton("FDTD  (rigorous EM, slow)")
        self.fourier_radio.setChecked(True)
        mode_layout.addWidget(self.fourier_radio)
        mode_layout.addWidget(self.fdtd_radio)
        layout.addWidget(mode_group)

        ctrl_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.stop_btn = QPushButton("Stop")
        self.reset_btn = QPushButton("Reset")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        self.reset_btn.clicked.connect(self._on_reset)
        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addWidget(self.reset_btn)
        layout.addLayout(ctrl_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Ready")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFontFamily("Monospace")
        self.log_edit.setMinimumHeight(120)
        log_layout.addWidget(self.log_edit)
        layout.addWidget(log_group)
        layout.addStretch()

    def _on_run(self):
        mode = 'fdtd' if self.fdtd_radio.isChecked() else 'fourier_optics'
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_edit.clear()
        self.run_requested.emit(mode)

    def _on_stop(self):
        self.stop_requested.emit()

    def _on_reset(self):
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.log_edit.clear()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def set_progress(self, step_name, percent):
        self.progress_bar.setValue(int(percent))
        self.status_label.setText(step_name)

    def append_log(self, text):
        self.log_edit.append(text)
        self.log_edit.moveCursor(QTextCursor.End)

    def on_simulation_done(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_simulation_error(self, msg):
        self.append_log("[ERROR] " + msg)
        self.status_label.setText("Failed")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
