"""
Main application window for optical lithography simulator.
Tabs: Layout | Parameters | Simulation | Results
"""
import os
import sys

from gui.qt_compat import (
    QMainWindow, QTabWidget, QStatusBar, QProgressBar,
    QLabel, QFileDialog, QMessageBox, QApplication,
    QToolBar, QStyle, Qt, QThread, Signal, QObject, QAction,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame,
)
from gui import theme

# Ensure simulation package is importable
_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)


# ── Worker thread ──────────────────────────────────────────────────────────────

class _SimWorker(QObject):
    progress = Signal(str, int)    # (step_name, percent)
    finished = Signal(object)      # SimResult
    error = Signal(str)

    def __init__(self, config, layout_path, mode):
        super().__init__()
        self._config = config
        self._layout_path = layout_path
        self._mode = mode
        self._stop = False

    def run(self):
        try:
            from pipeline.simulation_pipeline import SimulationPipeline
            pipeline = SimulationPipeline()

            def on_progress(step, pct):
                self.progress.emit(step, int(pct))

            # Override mode in config (deep-copy simulation sub-dict so
            # the caller's original config is never mutated)
            import copy
            cfg = copy.deepcopy(self._config)
            cfg.setdefault('simulation', {})['mode'] = self._mode

            result = pipeline.run(cfg, self._layout_path, on_progress,
                                  stop_fn=lambda: self._stop)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit("{}\n{}".format(e, traceback.format_exc()))


class SimWorkerThread(QThread):
    progress = Signal(str, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, config, layout_path, mode, parent=None):
        super().__init__(parent)
        self._worker = _SimWorker(config, layout_path, mode)
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self.finished)
        self._worker.error.connect(self.error)

    def run(self):
        self._worker.run()

    def request_stop(self):
        self._worker._stop = True


# ── Stack panel (non-modal embedded film stack editor) ──────────────────────

class _StackPanel(QWidget):
    """Embeds StackDialog content as an in-window tab (non-modal)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            from gui.dialogs.stack_dialog import StackDialog
            self._dialog = StackDialog(self)
            self._dialog.setWindowFlags(Qt.Widget)
            layout.addWidget(self._dialog)
        except Exception as exc:
            err = QLabel("Film Stack editor unavailable: {}".format(exc))
            err.setObjectName("caption")
            layout.addWidget(err)
            self._dialog = None

    def get_film_stack(self):
        return self._dialog.get_film_stack() if self._dialog else None


# ── Workflow step indicator bar ────────────────────────────────────────────────

class _WorkflowBar(QWidget):
    step_clicked = Signal(int)

    STEPS = ["1. Layout", "2. Parameters", "3. Simulation", "4. Results", "5. Analysis"]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(0)
        self._btns = []
        for i, label in enumerate(self.STEPS):
            btn = QPushButton(label)
            btn.setFlat(True)
            btn.setObjectName("workflow_step")
            btn.clicked.connect(lambda checked, idx=i: self.step_clicked.emit(idx))
            self._btns.append(btn)
            layout.addWidget(btn)
            if i < len(self.STEPS) - 1:
                arrow = QLabel("→")
                arrow.setObjectName("caption")
                layout.addWidget(arrow)
        layout.addStretch()
        self.set_current(0)

    def set_current(self, idx):
        for i, btn in enumerate(self._btns):
            if i == idx:
                btn.setStyleSheet(
                    "font-weight: 700; color: {ACCENT}; background: transparent; border: none;".format(
                        ACCENT=theme.ACCENT))
            else:
                btn.setStyleSheet(
                    "font-weight: 400; color: {TEXT_SECONDARY}; background: transparent; border: none;".format(
                        TEXT_SECONDARY=theme.TEXT_SECONDARY))


# ── Main window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optical Lithography Simulator")
        self.resize(1100, 780)
        self._sim_thread = None
        theme.apply_mpl_theme()
        self._build_ui()
        self._build_menu()
        self._build_statusbar()
        self._build_toolbar()
        self._apply_stylesheet()
        self._build_shortcuts()
        self._update_sim_info()

    def _build_ui(self):
        from gui.panels.layout_panel import LayoutPanel
        from gui.panels.parameter_panel import ParameterPanel
        from gui.panels.simulation_panel import SimulationPanel
        from gui.panels.results_panel import ResultsPanel
        from gui.panels.analysis_panel import AnalysisPanel

        self.setMinimumSize(900, 650)
        self.tabs = QTabWidget()

        self.layout_panel = LayoutPanel()
        self.param_panel = ParameterPanel()
        self.sim_panel = SimulationPanel()
        self.results_panel = ResultsPanel()
        self.analysis_panel = AnalysisPanel()

        self.stack_panel = _StackPanel()

        self.tabs.addTab(self.layout_panel, "Layout")
        self.tabs.addTab(self.param_panel, "Parameters")
        self.tabs.addTab(self.sim_panel, "Simulation")
        self.tabs.addTab(self.results_panel, "Results")
        self.tabs.addTab(self.analysis_panel, "Analysis")
        self.tabs.addTab(self.stack_panel, "Stack")

        # Workflow bar + separator above tabs
        self._workflow_bar = _WorkflowBar()

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: {};".format(theme.BORDER))

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(self._workflow_bar)
        vbox.addWidget(sep)
        vbox.addWidget(self.tabs)
        self.setCentralWidget(container)

        # Wire signals
        self.sim_panel.run_requested.connect(self._run_simulation)
        self.sim_panel.stop_requested.connect(self._stop_simulation)
        self.layout_panel.layout_loaded.connect(self._on_layout_loaded)
        self.tabs.currentChanged.connect(self._workflow_bar.set_current)
        self._workflow_bar.step_clicked.connect(self.tabs.setCurrentIndex)
        self.analysis_panel.navigate_requested.connect(self.tabs.setCurrentIndex)
        self.param_panel.source_preview_requested.connect(self._show_source_dialog)
        self._results_stale = False
        self.param_panel.params_changed.connect(self._mark_stale)

    def _on_layout_loaded(self, path):
        basename = os.path.basename(path)
        self.setWindowTitle("Optical Lithography Simulator — {}".format(basename))
        self._status("Layout loaded: {}".format(basename))
        if hasattr(self, 'layout_label'):
            self.layout_label.setText(basename)
        # Hint user to move to Parameters next
        self._workflow_bar.set_current(1)

    def _build_menu(self):
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("&File")
        act_open = QAction("Open GDS/OAS...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_layout)
        file_menu.addAction(act_open)

        act_load_params = QAction("Load Parameters...", self)
        act_load_params.triggered.connect(self.param_panel._load_config)
        file_menu.addAction(act_load_params)

        act_save_params = QAction("Save Parameters...", self)
        act_save_params.triggered.connect(self.param_panel._save_config)
        file_menu.addAction(act_save_params)

        file_menu.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Simulation menu
        sim_menu = mb.addMenu("&Simulation")
        act_run = QAction("Run", self)
        act_run.setShortcut("F5")
        act_run.triggered.connect(lambda: (self.tabs.setCurrentIndex(2), self.sim_panel._on_run()))
        sim_menu.addAction(act_run)

        act_stop = QAction("Stop", self)
        act_stop.triggered.connect(self._stop_simulation)
        sim_menu.addAction(act_stop)

        act_source = QAction("Source Preview...", self)
        act_source.triggered.connect(self._show_source_dialog)
        sim_menu.addAction(act_source)

        # Analysis menu
        analysis_menu = mb.addMenu("&Analysis")
        act_export = QAction("Export Results...", self)
        act_export.triggered.connect(lambda: self.results_panel._export("png"))
        analysis_menu.addAction(act_export)

        # Help menu
        help_menu = mb.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _build_statusbar(self):
        self.status_label = QLabel("Ready")
        self.status_progress = QProgressBar()
        self.status_progress.setMaximumWidth(200)
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.setVisible(False)
        self.sim_info_label = QLabel("Mode: Fourier  |  Grid: 256×256  |  Points: 65,536")
        self.sim_info_label.setObjectName("caption")
        self._stale_label = QLabel("⚠ Results may be stale")
        self._stale_label.setStyleSheet("color: orange; font-weight: bold; padding: 0 6px;")
        self._stale_label.setVisible(False)
        sb = self.statusBar()
        self.layout_label = QLabel("No layout loaded")
        self.layout_label.setObjectName("caption")
        sb.addWidget(self.status_label, 1)
        sb.addPermanentWidget(self._stale_label)
        sb.addPermanentWidget(self.layout_label)
        sb.addPermanentWidget(self.sim_info_label)
        sb.addPermanentWidget(self.status_progress)

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)

        act_open = QAction("Open Layout", self)
        act_open.triggered.connect(self._open_layout)
        act_open.setToolTip("Open GDS/OAS layout (Ctrl+O)")
        tb.addAction(act_open)

        tb.addSeparator()

        act_run = QAction("Run ▶", self)
        act_run.triggered.connect(
            lambda: (self.tabs.setCurrentIndex(2), self.sim_panel._on_run())
        )
        act_run.setToolTip("Run simulation (Ctrl+R)")
        tb.addAction(act_run)

        act_stop = QAction("Stop ■", self)
        act_stop.triggered.connect(self._stop_simulation)
        act_stop.setToolTip("Stop simulation (Esc)")
        tb.addAction(act_stop)

        tb.addSeparator()

        act_source = QAction("Source...", self)
        act_source.triggered.connect(self._show_source_dialog)
        act_source.setToolTip("Illumination source (k-space) preview / settings")
        tb.addAction(act_source)

    def _build_shortcuts(self):
        try:
            from PySide6.QtGui import QShortcut, QKeySequence
        except ImportError:
            try:
                from PyQt5.QtWidgets import QShortcut   # type: ignore
                from PyQt5.QtGui import QKeySequence    # type: ignore
            except ImportError:
                return

        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(
            lambda: (self.tabs.setCurrentIndex(2), self.sim_panel._on_run()))
        QShortcut(QKeySequence("Escape"), self).activated.connect(self._stop_simulation)
        QShortcut(QKeySequence("F1"), self).activated.connect(self._show_about)
        for i in range(6):
            QShortcut(QKeySequence("Ctrl+{}".format(i + 1)), self).activated.connect(
                lambda j=i: self.tabs.setCurrentIndex(j))

    def _update_sim_info(self, mode=None):
        try:
            cfg = self.param_panel.get_config()
            grid = cfg.get('simulation', {}).get('grid_size', 256)
        except Exception:
            grid = 256
        mode_str = "Fourier" if (mode is None or mode == "fourier") else "FDTD"
        n_pts = grid * grid
        self.sim_info_label.setText(
            "Mode: {}  |  Grid: {}×{}  |  Points: {:,}".format(
                mode_str, grid, grid, n_pts))

    def _apply_stylesheet(self):
        app = QApplication.instance()
        if app:
            app.setStyleSheet(theme.get_qss())

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _status(self, msg):
        self.status_label.setText(msg)

    def _mark_stale(self):
        if not self._results_stale:
            self._results_stale = True
            self._stale_label.setVisible(True)

    def _open_layout(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Layout", "",
            "Layout files (*.gds *.gds2 *.oas);;All (*)")
        if path:
            self.layout_panel.load_layout(path)
            self.tabs.setCurrentIndex(0)

    def _show_source_dialog(self):
        from gui.dialogs.source_dialog import SourceDialog
        dlg = SourceDialog(self, self.param_panel.get_config())
        if dlg.exec():
            illum_cfg = dlg.get_illumination_config()
            # Apply back to parameter panel
            cfg = self.param_panel.get_config()
            cfg['lithography']['illumination'] = illum_cfg
            self.param_panel.load_config(cfg)

    def _run_simulation(self, mode):
        if self._sim_thread and self._sim_thread.isRunning():
            return
        config = self.param_panel.get_config()
        layout_path = self.layout_panel.get_layout_path()

        # Inject film stack into config if available
        film_stack = self.stack_panel.get_film_stack()
        if film_stack is not None:
            config.setdefault('_film_stack', film_stack)

        # Inject polarization into config if non-scalar is selected
        pol_text = self.param_panel.get_polarization()
        if pol_text != "Unpolarized":
            try:
                import core.vector_imaging  # availability check  # noqa: F401
                pol_map = {
                    "X": "x", "Y": "y",
                    "TE": "te", "TM": "tm",
                    "Circular L": "circular_l", "Circular R": "circular_r",
                }
                pol_val = pol_map.get(pol_text, 'unpolarized')
                config.setdefault('simulation', {})['polarization'] = pol_val
                # Selecting a vector polarization from the results panel must
                # also activate the vector imaging engine — otherwise the
                # polarization key has no effect because use_vector stays False.
                config.setdefault('lithography', {})['use_vector'] = True
            except ImportError:
                pass  # VectorImagingEngine unavailable — keep scalar

        self._update_sim_info(mode)

        # Validate config and show errors/warnings before launching
        from fileio.config_validator import ConfigValidator
        validation_results = ConfigValidator().validate(config)
        errors = [e for e in validation_results if e.severity == 'error']
        warnings = [e for e in validation_results if e.severity == 'warning']
        if errors or warnings:
            lines = []
            if errors:
                lines.append('<span style="color:red;font-weight:bold;">Errors (simulation blocked):</span>')
                for e in errors:
                    lines.append('<span style="color:red;">&bull; {}: {}</span>'.format(e.field, e.message))
            if warnings:
                lines.append('<span style="color:#b8860b;font-weight:bold;">Warnings:</span>')
                for w in warnings:
                    lines.append('<span style="color:#b8860b;">&bull; {}: {}</span>'.format(w.field, w.message))
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle('Config Validation')
            msg_box.setTextFormat(1)  # Qt.RichText
            msg_box.setText('<br>'.join(lines))
            if errors:
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec()
                return
            else:
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                msg_box.setDefaultButton(QMessageBox.Ok)
                if msg_box.exec() != QMessageBox.Ok:
                    return

        if not layout_path:
            answer = QMessageBox.question(
                self, "No Layout Loaded",
                "No layout file is loaded. Run simulation with a synthetic test pattern?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self._sim_thread = SimWorkerThread(config, layout_path, mode, self)
        self._sim_thread.progress.connect(self._on_progress)
        self._sim_thread.finished.connect(self._on_sim_finished)
        self._sim_thread.error.connect(self._on_sim_error)

        self.status_progress.setVisible(True)
        self.status_progress.setValue(0)
        self._status("Running simulation ({})...".format(mode))
        self._sim_thread.start()

    def _stop_simulation(self):
        if self._sim_thread and self._sim_thread.isRunning():
            self._sim_thread.request_stop()
            self._sim_thread.wait(3000)
            self._status("Simulation stopped.")
            self.sim_panel.on_simulation_stopped()
            self.status_progress.setVisible(False)

    def _on_progress(self, step, pct):
        self.sim_panel.set_progress(step, pct)
        self.sim_panel.append_log("[{:3d}%] {}".format(pct, step))
        self.status_progress.setValue(pct)
        self._status(step)

    def _on_sim_finished(self, result):
        self.sim_panel.on_simulation_done()
        self.status_progress.setVisible(False)
        self._results_stale = False
        self._stale_label.setVisible(False)
        if result.status == 'complete':
            self._status("Simulation complete — CD={:.1f} nm, NILS={:.3f}".format(
                result.cd_nm, result.nils))
            self.results_panel.show_result(result)
            self.analysis_panel.set_result(result)
            self.tabs.setCurrentIndex(3)
        else:
            self._status("Simulation failed: " + result.error_msg)
            self.sim_panel.on_simulation_error(result.error_msg)

    def _on_sim_error(self, msg):
        self.sim_panel.on_simulation_error(msg)
        self.status_progress.setVisible(False)
        self._status("Error during simulation.")

    def _show_about(self):
        QMessageBox.about(self, "About",
            "Optical Lithography Simulator  v1.0\n\n"
            "Based on Pistor (2001) PhD Dissertation, UCB\n"
            "Fourier Optics (Abbe/Hopkins SOCS) + FDTD\n\n"
            "Supported:\n"
            "  \u2022 ArF 193nm / KrF 248nm / EUV 13.5nm\n"
            "  \u2022 Scalar / Vector (Jones matrix) imaging\n"
            "  \u2022 Threshold / Dill ABC / CA resist models\n"
            "  \u2022 RCWA near-field, TMM film stack\n\n"
            "Shortcuts:  Ctrl+R = Run   Esc = Stop\n"
            "            Ctrl+1~6 = Tab navigation\n\n"
            "Python + PySide6 + NumPy + SciPy + gdstk")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    try:
        app = QApplication.instance() or QApplication(sys.argv)
    except RuntimeError:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
