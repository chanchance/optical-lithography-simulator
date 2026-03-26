"""
Main application window for optical lithography simulator.
Tabs: Layout | Parameters | Simulation | Results
"""
import os
import sys

try:
    from PySide6.QtWidgets import (
        QMainWindow, QTabWidget, QStatusBar, QProgressBar,
        QLabel, QFileDialog, QMessageBox, QApplication,
        QToolBar, QStyle
    )
    from PySide6.QtCore import Qt, QThread, Signal, QObject
    from PySide6.QtGui import QAction
    _QT = 'PySide6'
except ImportError:
    from PyQt5.QtWidgets import (
        QMainWindow, QTabWidget, QStatusBar, QProgressBar,
        QLabel, QFileDialog, QMessageBox, QApplication,
        QToolBar, QStyle
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal, QObject
    from PyQt5.QtWidgets import QAction
    _QT = 'PyQt5'

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

            # Override mode in config
            cfg = dict(self._config)
            cfg.setdefault('simulation', {})['mode'] = self._mode

            result = pipeline.run(cfg, self._layout_path, on_progress)
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


# ── Main window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optical Lithography Simulator")
        self.resize(1100, 780)
        self._sim_thread = None
        self._build_ui()
        self._build_menu()
        self._build_statusbar()
        self._build_toolbar()
        self._apply_stylesheet()

    def _build_ui(self):
        from gui.panels.layout_panel import LayoutPanel
        from gui.panels.parameter_panel import ParameterPanel
        from gui.panels.simulation_panel import SimulationPanel
        from gui.panels.results_panel import ResultsPanel

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.layout_panel = LayoutPanel()
        self.param_panel = ParameterPanel()
        self.sim_panel = SimulationPanel()
        self.results_panel = ResultsPanel()

        self.tabs.addTab(self.layout_panel, "Layout")
        self.tabs.addTab(self.param_panel, "Parameters")
        self.tabs.addTab(self.sim_panel, "Simulation")
        self.tabs.addTab(self.results_panel, "Results")

        # Wire signals
        self.sim_panel.run_requested.connect(self._run_simulation)
        self.sim_panel.stop_requested.connect(self._stop_simulation)
        self.layout_panel.layout_loaded.connect(
            lambda p: self._status("Layout loaded: " + os.path.basename(p)))

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
        act_run.triggered.connect(lambda: self.tabs.setCurrentIndex(2) or self.sim_panel._on_run())
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
        sb = self.statusBar()
        sb.addWidget(self.status_label, 1)
        sb.addPermanentWidget(self.status_progress)

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        app_style = QApplication.instance().style()

        # Open action
        act_open = QAction(
            app_style.standardIcon(QStyle.SP_DirOpenIcon),
            "Open GDS/OAS", self
        )
        act_open.triggered.connect(self._open_layout)
        tb.addAction(act_open)

        tb.addSeparator()

        # Run action
        act_run = QAction(
            app_style.standardIcon(QStyle.SP_MediaPlay),
            "Run", self
        )
        act_run.triggered.connect(
            lambda: (self.tabs.setCurrentIndex(2), self.sim_panel._on_run())
        )
        tb.addAction(act_run)

        # Stop action
        act_stop = QAction(
            app_style.standardIcon(QStyle.SP_MediaStop),
            "Stop", self
        )
        act_stop.triggered.connect(self._stop_simulation)
        tb.addAction(act_stop)

        tb.addSeparator()

        # Source preview action
        act_source = QAction(
            app_style.standardIcon(QStyle.SP_FileDialogDetailedView),
            "Source", self
        )
        act_source.triggered.connect(self._show_source_dialog)
        tb.addAction(act_source)

    def _apply_stylesheet(self):
        style = """
        QMainWindow { background: #f5f5f5; }
        QTabWidget::pane { border: 1px solid #cccccc; background: #ffffff; }
        QTabBar::tab {
            background: #e0e0e0; padding: 6px 16px; margin-right: 2px;
            border: 1px solid #cccccc; border-bottom: none;
            border-radius: 3px 3px 0 0;
        }
        QTabBar::tab:selected {
            background: #ffffff; border-bottom: 1px solid #ffffff;
            font-weight: bold;
        }
        QTabBar::tab:hover { background: #ebebeb; }
        QGroupBox {
            font-weight: bold; border: 1px solid #cccccc; border-radius: 4px;
            margin-top: 8px; padding-top: 4px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        QPushButton {
            background: #e8e8e8; border: 1px solid #bbbbbb; padding: 5px 12px;
            border-radius: 3px;
        }
        QPushButton:hover { background: #d8d8d8; }
        QPushButton:pressed { background: #c8c8c8; }
        QProgressBar {
            border: 1px solid #bbbbbb; border-radius: 3px; text-align: center;
        }
        QProgressBar::chunk { background: #4e79a7; border-radius: 2px; }
        QStatusBar { background: #e8e8e8; border-top: 1px solid #cccccc; }
        QMenuBar { background: #f0f0f0; border-bottom: 1px solid #cccccc; }
        QMenuBar::item:selected { background: #d0d0d0; }
        QMenu::item:selected { background: #4e79a7; color: white; }
        QListWidget { border: 1px solid #cccccc; border-radius: 3px; }
        QListWidget::item:selected { background: #4e79a7; color: white; }
        QTextEdit { border: 1px solid #cccccc; border-radius: 3px; background: #fafafa; }
        QDoubleSpinBox, QSpinBox, QComboBox {
            border: 1px solid #bbbbbb; border-radius: 3px;
            padding: 2px 4px; background: white;
        }
        QToolBar {
            background: #f0f0f0; border-bottom: 1px solid #cccccc;
            spacing: 4px; padding: 2px 4px;
        }
        QToolBar::separator { width: 1px; background: #cccccc; margin: 4px 2px; }
        QToolButton {
            background: transparent; border: 1px solid transparent;
            border-radius: 3px; padding: 3px 6px;
        }
        QToolButton:hover { background: #dde4ed; border-color: #b0bece; }
        QToolButton:pressed { background: #c8d4e0; }
        QSplitter::handle { background: #dddddd; }
        QSplitter::handle:horizontal { width: 4px; }
        """
        app = QApplication.instance()
        if app:
            app.setStyleSheet(style)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _status(self, msg):
        self.status_label.setText(msg)

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
        self.sim_panel.on_simulation_done()
        self.status_progress.setVisible(False)

    def _on_progress(self, step, pct):
        self.sim_panel.set_progress(step, pct)
        self.sim_panel.append_log("[{:3d}%] {}".format(pct, step))
        self.status_progress.setValue(pct)
        self._status(step)

    def _on_sim_finished(self, result):
        self.sim_panel.on_simulation_done()
        self.status_progress.setVisible(False)
        if result.status == 'complete':
            self._status("Simulation complete — CD={:.1f} nm, NILS={:.3f}".format(
                result.cd_nm, result.nils))
            self.results_panel.show_result(result)
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
            "Optical Lithography Simulator\n"
            "Based on Pistor (2001) — FDTD + Fourier Optics\n\n"
            "Supports GDS/OAS layout input, ArF/EUV wavelengths,\n"
            "Fourier optics (Abbe) and FDTD simulation modes.\n\n"
            "Python + PySide6 + NumPy + SciPy")


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
