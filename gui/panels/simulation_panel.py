"""
Simulation control panel: mode selection, run/stop, progress bar, log.
"""
from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QProgressBar, QLabel,
    QFrame, QSizePolicy, Qt, Signal, QElapsedTimer,
)
from gui import theme

try:
    from PySide6.QtWidgets import QRadioButton, QButtonGroup, QTextEdit
    from PySide6.QtGui import QTextCursor
except ImportError:
    from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QTextEdit  # type: ignore
    from PyQt5.QtGui import QTextCursor  # type: ignore


_MODE_DESCRIPTIONS = {
    'fourier_optics': "Fast scalar simulation. Best for CD/NILS/DOF sweeps. ~1s per point.",
    'fdtd':           "Rigorous EM solver (TEMPEST). Accurate for thick masks/EUV. ~minutes per point.",
}


class SimulationPanel(QWidget):
    """Panel for controlling simulation execution."""

    run_requested = Signal(str)   # emits mode string
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._elapsed = QElapsedTimer()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # ── Mode selection ────────────────────────────────────────────────────
        mode_group = QGroupBox("Simulation Mode")
        mode_outer = QVBoxLayout(mode_group)
        mode_outer.setSpacing(6)

        radio_frame = QFrame()
        radio_frame.setFrameShape(QFrame.StyledPanel)
        radio_layout = QVBoxLayout(radio_frame)
        radio_layout.setSpacing(4)
        radio_layout.setContentsMargins(10, 8, 10, 8)

        self._mode_group = QButtonGroup(self)
        self.fourier_radio = QRadioButton("Fourier Optics  (fast, scalar)")
        self.fdtd_radio = QRadioButton("FDTD  (rigorous EM, slow)")
        self.fourier_radio.setChecked(True)
        self._mode_group.addButton(self.fourier_radio, 0)
        self._mode_group.addButton(self.fdtd_radio, 1)

        radio_layout.addWidget(self.fourier_radio)
        radio_layout.addWidget(self.fdtd_radio)
        mode_outer.addWidget(radio_frame)

        self.mode_desc_label = QLabel(_MODE_DESCRIPTIONS['fourier_optics'])
        self.mode_desc_label.setWordWrap(True)
        self.mode_desc_label.setObjectName("caption")
        mode_outer.addWidget(self.mode_desc_label)

        self._mode_group.buttonClicked.connect(self._on_mode_changed)
        layout.addWidget(mode_group)

        # ── Run / Stop / Reset buttons ────────────────────────────────────────
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(8)

        self.run_btn = QPushButton("Run")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.run_btn.setObjectName("success")

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setObjectName("danger")

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setMinimumHeight(36)

        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        self.reset_btn.clicked.connect(self._on_reset)

        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addWidget(self.reset_btn)
        layout.addLayout(ctrl_layout)

        # ── Progress bar + ETA ────────────────────────────────────────────────
        progress_group = QGroupBox("Progress")
        progress_outer = QVBoxLayout(progress_group)
        progress_outer.setSpacing(4)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(18)
        progress_outer.addWidget(self.progress_bar)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)

        # LED indicator
        self.led_label = QLabel()
        self.led_label.setFixedSize(14, 14)
        self._set_led('gray')
        status_row.addWidget(self.led_label)

        self.status_label = QLabel("Ready")
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        status_row.addWidget(self.status_label)

        self.eta_label = QLabel("")
        self.eta_label.setObjectName("caption")
        self.eta_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_row.addWidget(self.eta_label)

        progress_outer.addLayout(status_row)
        layout.addWidget(progress_group)

        # ── Log ───────────────────────────────────────────────────────────────
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setSpacing(4)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFontFamily("Monospace")
        self.log_edit.setMinimumHeight(140)

        log_header = QHBoxLayout()
        log_header.addStretch()
        self.clear_log_btn = QPushButton("Clear")
        self.clear_log_btn.setFixedHeight(22)
        self.clear_log_btn.clicked.connect(self.log_edit.clear)
        log_header.addWidget(self.clear_log_btn)
        log_layout.addLayout(log_header)

        log_layout.addWidget(self.log_edit)

        layout.addWidget(log_group)
        layout.addStretch()

    # ── LED helper ────────────────────────────────────────────────────────────

    def _set_led(self, color):
        """Set LED indicator color: 'gray', 'yellow', 'green', 'red'."""
        colors = {
            'gray':   theme.TEXT_TERTIARY,
            'yellow': theme.WARNING,
            'green':  theme.SUCCESS,
            'red':    theme.DANGER,
        }
        bg = colors.get(color, theme.TEXT_TERTIARY)
        self.led_label.setStyleSheet(
            "QLabel {{ background-color: {bg}; border-radius: 7px; }}".format(bg=bg)
        )

    # ── Mode change ───────────────────────────────────────────────────────────

    def _on_mode_changed(self, btn):
        mode = 'fdtd' if self.fdtd_radio.isChecked() else 'fourier_optics'
        self.mode_desc_label.setText(_MODE_DESCRIPTIONS[mode])

    # ── Run / Stop / Reset ────────────────────────────────────────────────────

    def _on_run(self):
        mode = 'fdtd' if self.fdtd_radio.isChecked() else 'fourier_optics'
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.eta_label.setText("")
        self.log_edit.clear()
        self._elapsed.start()
        self._set_led('yellow')
        self.status_label.setText("Running...")
        self.run_requested.emit(mode)

    def _on_stop(self):
        self.stop_requested.emit()

    def _on_reset(self):
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.eta_label.setText("")
        self.log_edit.clear()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_led('gray')

    # ── Public API ────────────────────────────────────────────────────────────

    def set_progress(self, step_name, percent):
        self.progress_bar.setValue(int(percent))
        self.status_label.setText(step_name)

        # ETA estimation
        if self._elapsed.isValid():
            elapsed_s = self._elapsed.elapsed() / 1000.0
            if percent > 0:
                eta_s = elapsed_s * (100 - percent) / percent
                self.eta_label.setText("ETA: {:.0f}s".format(eta_s))
            else:
                self.eta_label.setText("")

    def append_log(self, text):
        # Escape HTML special characters
        safe = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        if '[ERROR]' in text:
            html = '<span style="color:#cc3333; font-family:monospace;">{}</span>'.format(safe)
        elif 'complete' in text.lower() or '100%' in text:
            html = '<span style="color:#2d8a4e; font-family:monospace;">{}</span>'.format(safe)
        else:
            html = '<span style="color:#333333; font-family:monospace;">{}</span>'.format(safe)
        self.log_edit.append(html)
        self.log_edit.moveCursor(QTextCursor.End)

    def on_simulation_done(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_led('green')
        self.eta_label.setText("")

    def on_simulation_error(self, msg):
        self.append_log("[ERROR] " + msg)
        self.status_label.setText("Failed")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_led('red')
        self.eta_label.setText("")
