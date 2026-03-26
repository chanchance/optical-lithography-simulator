"""
Analysis Panel: Bossung curves, Focus-Exposure Matrix, MEEF display.
S-Litho style professional analysis viewer.
"""
import copy
import numpy as np

from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton, QLabel,
    QDoubleSpinBox, QGroupBox, QTableWidget, QTableWidgetItem,
    QFormLayout, QFont, Qt, QThread, Signal, QObject, QSizePolicy, QFrame,
)
from gui import theme

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import matplotlib.cm as cm


# ── Background worker threads ───────────────────────────────────────────────

class _BossungWorker(QObject):
    finished = Signal(list)
    progress = Signal(str, int)
    error = Signal(str)

    def __init__(self, config, layout_path, focus_range, dose_min, dose_max, cd_tol):
        super().__init__()
        self._config = config
        self._layout_path = layout_path
        self._focus_range = focus_range
        self._dose_min = dose_min
        self._dose_max = dose_max
        self._cd_tol = cd_tol

    def run(self):
        try:
            from pipeline.parameter_sweep import ParameterSweep
            from analysis.advanced_metrics import BossungCurve

            sweep = ParameterSweep()
            dose_factors = np.linspace(self._dose_min, self._dose_max, 5).tolist()

            def on_progress(msg, pct):
                self.progress.emit(msg, pct)

            bossung_data = sweep.bossung_sweep(
                self._config,
                self._layout_path,
                focus_range_nm=self._focus_range,
                n_focus=17,
                dose_factors=dose_factors,
                on_progress=on_progress,
            )
            focus_arr = bossung_data['focus_nm']
            cd_by_dose = bossung_data['cd_by_dose']

            from analysis.advanced_metrics import _fit_bossung_curve
            curves = []
            for dose, cd_vals in cd_by_dose.items():
                cd_arr = np.array(cd_vals)
                best_focus, dof = _fit_bossung_curve(focus_arr, cd_arr, self._cd_tol)
                curves.append(BossungCurve(
                    dose_factor=dose,
                    focus_points=focus_arr.tolist(),
                    cd_points=cd_vals,
                    best_focus_nm=best_focus,
                    depth_of_focus_nm=dof,
                ))
            self.finished.emit(curves)
        except Exception as e:
            import traceback
            self.error.emit("{}\n{}".format(e, traceback.format_exc()))


class _BossungThread(QThread):
    finished = Signal(list)
    progress = Signal(str, int)
    error = Signal(str)

    def __init__(self, config, layout_path, focus_range, dose_min, dose_max, cd_tol, parent=None):
        super().__init__(parent)
        self._worker = _BossungWorker(config, layout_path, focus_range, dose_min, dose_max, cd_tol)
        self._worker.finished.connect(self.finished)
        self._worker.progress.connect(self.progress)
        self._worker.error.connect(self.error)

    def run(self):
        self._worker.run()


class _FEMWorker(QObject):
    finished = Signal(object)
    progress = Signal(str, int)
    error = Signal(str)

    def __init__(self, config, layout_path, focus_range, dose_min, dose_max):
        super().__init__()
        self._config = config
        self._layout_path = layout_path
        self._focus_range = focus_range
        self._dose_min = dose_min
        self._dose_max = dose_max

    def run(self):
        try:
            from pipeline.parameter_sweep import ParameterSweep
            from analysis.advanced_metrics import FocusExposureMatrix

            focus_values = np.linspace(-self._focus_range / 2, self._focus_range / 2, 7)
            dose_values = np.linspace(self._dose_min, self._dose_max, 7)

            def on_progress(msg, pct):
                self.progress.emit(msg, pct)

            sweep = ParameterSweep()
            cd_matrix = sweep.sweep_2d(
                self._config,
                self._layout_path,
                'lithography.defocus_nm', focus_values.tolist(),
                'lithography.dose_factor', dose_values.tolist(),
                on_progress=on_progress,
            )
            fem = FocusExposureMatrix(focus_values, dose_values, cd_matrix)
            self.finished.emit(fem)
        except Exception as e:
            import traceback
            self.error.emit("{}\n{}".format(e, traceback.format_exc()))


class _FEMThread(QThread):
    finished = Signal(object)
    progress = Signal(str, int)
    error = Signal(str)

    def __init__(self, config, layout_path, focus_range, dose_min, dose_max, parent=None):
        super().__init__(parent)
        self._worker = _FEMWorker(config, layout_path, focus_range, dose_min, dose_max)
        self._worker.finished.connect(self.finished)
        self._worker.progress.connect(self.progress)
        self._worker.error.connect(self.error)

    def run(self):
        self._worker.run()


# ── Analysis Panel ──────────────────────────────────────────────────────────

class AnalysisPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._config = None
        self._layout_path = None
        self._bossung_thread = None
        self._fem_thread = None
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left controls ────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(230)
        left.setMaximumWidth(280)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(theme.SP_MD, theme.SP_MD, theme.SP_MD, theme.SP_MD)
        lv.setSpacing(theme.SP_SM)

        run_group = QGroupBox("Analysis Controls")
        run_layout = QVBoxLayout(run_group)
        run_layout.setSpacing(10)

        self.bossung_btn = QPushButton("Run Bossung")
        self.bossung_btn.setObjectName("secondary")
        self.bossung_btn.setMinimumHeight(40)
        self.fem_btn = QPushButton("Run FEM")
        self.fem_btn.setObjectName("secondary")
        self.fem_btn.setMinimumHeight(36)
        self.bossung_btn.clicked.connect(self._run_bossung)
        self.fem_btn.clicked.connect(self._run_fem)
        run_layout.addWidget(self.bossung_btn)
        run_layout.addWidget(self.fem_btn)
        lv.addWidget(run_group)

        param_group = QGroupBox("Parameters")
        form = QFormLayout(param_group)
        form.setSpacing(10)

        self.focus_range_sb = QDoubleSpinBox()
        self.focus_range_sb.setRange(50, 2000)
        self.focus_range_sb.setValue(400)
        self.focus_range_sb.setSuffix(" nm")
        form.addRow("Focus range:", self.focus_range_sb)

        self.dose_min_sb = QDoubleSpinBox()
        self.dose_min_sb.setRange(0.5, 2.0)
        self.dose_min_sb.setValue(0.8)
        self.dose_min_sb.setSingleStep(0.05)
        form.addRow("Dose min:", self.dose_min_sb)

        self.dose_max_sb = QDoubleSpinBox()
        self.dose_max_sb.setRange(0.5, 2.0)
        self.dose_max_sb.setValue(1.2)
        self.dose_max_sb.setSingleStep(0.05)
        form.addRow("Dose max:", self.dose_max_sb)

        self.cd_tol_sb = QDoubleSpinBox()
        self.cd_tol_sb.setRange(1, 50)
        self.cd_tol_sb.setValue(10)
        self.cd_tol_sb.setSuffix(" %")
        form.addRow("CD tolerance:", self.cd_tol_sb)

        self.cd_target_sb = QDoubleSpinBox()
        self.cd_target_sb.setRange(1, 10000)
        self.cd_target_sb.setValue(100)
        self.cd_target_sb.setSuffix(" nm")
        form.addRow("CD target:", self.cd_target_sb)

        lv.addWidget(param_group)

        self.status_label = QLabel("No simulation result loaded.")
        self.status_label.setObjectName("caption")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding: 4px 6px; border-radius: 4px;")
        lv.addWidget(self.status_label)
        lv.addStretch()

        metrics_group = QGroupBox("Metrics")
        mv = QVBoxLayout(metrics_group)
        mv.setContentsMargins(0, theme.SP_SM, 0, 0)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setSelectionMode(QTableWidget.NoSelection)
        self.metrics_table.setMinimumHeight(200)
        self.metrics_table.setAlternatingRowColors(True)
        mv.addWidget(self.metrics_table)
        lv.addWidget(metrics_group)

        splitter.addWidget(left)

        # ── Right: matplotlib figure ─────────────────────────────────
        self.figure = Figure(figsize=(12, 8), dpi=theme.MPL_DPI, constrained_layout=True)
        self.figure.patch.set_facecolor(theme.BG_PRIMARY)

        gs = GridSpec(2, 2, figure=self.figure)
        self.ax_bossung = self.figure.add_subplot(gs[0, 0])
        self.ax_fem     = self.figure.add_subplot(gs[0, 1])
        self.ax_pw      = self.figure.add_subplot(gs[1, 0])
        self.ax_metrics = self.figure.add_subplot(gs[1, 1])

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        canvas_frame = QFrame()
        canvas_frame.setStyleSheet(
            "QFrame { border: 1px solid %s; border-radius: 8px; background: %s; }"
            % (theme.BORDER, theme.BG_PRIMARY)
        )
        cf_layout = QVBoxLayout(canvas_frame)
        cf_layout.setContentsMargins(0, 0, 0, 0)
        cf_layout.addWidget(self.canvas)

        splitter.addWidget(canvas_frame)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)
        self._init_plots()

    # ── Theme helper ───────────────────────────────────────────────────

    def _style_ax(self, ax, title, image_panel=False):
        ax.set_facecolor(theme.BG_SECONDARY)
        ax.set_title(title, fontsize=theme.MPL_TITLE, fontweight='600',
                     color=theme.TEXT_PRIMARY, pad=6)
        ax.tick_params(colors=theme.TEXT_TERTIARY, labelsize=theme.MPL_TICK)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(theme.BORDER)
        ax.spines['bottom'].set_color(theme.BORDER)
        if image_panel:
            ax.grid(False)
        else:
            ax.grid(True, color=theme.MPL_GRID, linewidth=0.5, alpha=0.7, zorder=0)

    def _init_plots(self):
        for ax, title, is_img in [
            (self.ax_bossung, "Bossung Curves",        False),
            (self.ax_fem,     "Focus-Exposure Matrix", True),
            (self.ax_pw,      "Process Window",        False),
            (self.ax_metrics, "Key Metrics",           False),
        ]:
            ax.clear()
            self._style_ax(ax, title, image_panel=is_img)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    fontsize=theme.MPL_LABEL, color=theme.TEXT_TERTIARY,
                    transform=ax.transAxes)
            ax.set_axis_off()
        self.canvas.draw()

    # ── Public API ─────────────────────────────────────────────────────

    def set_result(self, sim_result):
        """Store simulation result and sync CD target from result."""
        self._result = sim_result
        self._config = getattr(sim_result, 'config', None)
        self._layout_path = getattr(sim_result, 'layout_path', None)
        if sim_result.cd_nm > 0:
            self.cd_target_sb.setValue(sim_result.cd_nm)
        self.status_label.setText(
            "Result loaded. CD={:.1f} nm, NILS={:.3f}".format(
                sim_result.cd_nm, sim_result.nils))
        self._update_metrics_table(
            cd_target=sim_result.cd_nm,
            nils=sim_result.nils,
            meef=sim_result.metrics.get('meef', None),
            best_focus=sim_result.metrics.get('best_focus_nm',
                       sim_result.metrics.get('defocus_nm', 0.0)),
            dof=sim_result.metrics.get('dof_nm', 0.0),
            el=sim_result.metrics.get('el_pct', 0.0),
        )

    # ── Bossung ────────────────────────────────────────────────────────

    def _run_bossung(self):
        if self._result is None or self._config is None:
            self.status_label.setText("Run a simulation first.")
            return
        if self._bossung_thread and self._bossung_thread.isRunning():
            return
        if self.dose_min_sb.value() >= self.dose_max_sb.value():
            self.status_label.setText("Dose min must be less than dose max.")
            return
        self.bossung_btn.setEnabled(False)
        self.status_label.setText("Computing Bossung curves...")
        self._bossung_thread = _BossungThread(
            config=self._config,
            layout_path=self._layout_path,
            focus_range=self.focus_range_sb.value(),
            dose_min=self.dose_min_sb.value(),
            dose_max=self.dose_max_sb.value(),
            cd_tol=self.cd_tol_sb.value(),
            parent=self,
        )
        self._bossung_thread.finished.connect(self._on_bossung_done)
        self._bossung_thread.progress.connect(lambda msg, _pct: self.status_label.setText(msg))
        self._bossung_thread.error.connect(
            lambda msg: self._on_analysis_error(msg, self.bossung_btn))
        self._bossung_thread.start()

    def _on_bossung_done(self, curves):
        self.bossung_btn.setEnabled(True)
        self.status_label.setText("Bossung done ({} curves).".format(len(curves)))
        self._plot_bossung(curves)

    def _plot_bossung(self, curves):
        ax = self.ax_bossung
        ax.clear()
        self._style_ax(ax, "Bossung Curves")

        colors = theme.BOSSUNG_COLORS
        for i, curve in enumerate(curves):
            focus, cd = curve.to_arrays()
            color = colors[i % len(colors)]
            is_nominal = abs(curve.dose_factor - 1.0) < 0.01
            lw = 2.2 if is_nominal else 1.4
            alpha = 1.0 if is_nominal else 0.85
            label = "D={:.2f}× ★".format(curve.dose_factor) if is_nominal else "D={:.2f}×".format(curve.dose_factor)
            ax.plot(focus, cd, '-o', color=color, linewidth=lw, alpha=alpha,
                    markersize=3 if not is_nominal else 4, label=label,
                    zorder=4 if is_nominal else 3)
            ax.axvline(curve.best_focus_nm, color=color, linestyle=':',
                       linewidth=0.8, alpha=0.3)

        cd_target = self.cd_target_sb.value()
        tol = self.cd_tol_sb.value() / 100.0
        ax.axhspan(cd_target * (1 - tol), cd_target * (1 + tol),
                   alpha=0.10, color=theme.SUCCESS,
                   label='±{:.0f}% band'.format(self.cd_tol_sb.value()))
        ax.axhline(cd_target, color=theme.SUCCESS, linestyle='--',
                   linewidth=0.9, alpha=0.7)

        ax.set_xlabel("Defocus (nm)", fontsize=theme.MPL_LABEL)
        ax.set_ylabel("CD (nm)", fontsize=theme.MPL_LABEL)
        ax.legend(fontsize=theme.MPL_LEGEND, loc='upper right', ncol=2)
        self.canvas.draw_idle()
        self._plot_pw_from_bossung(curves)

    def _plot_pw_from_bossung(self, curves):
        """Draw process window from Bossung curve data using real pass/fail grid."""
        from analysis.process_window import ProcessWindow

        ax = self.ax_pw
        ax.clear()
        self._style_ax(ax, "Process Window")

        if not curves or len(curves) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    fontsize=theme.MPL_LABEL, color=theme.TEXT_TERTIARY,
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        focus_nm = np.array(curves[0].focus_points)
        dose_axis = np.array([c.dose_factor for c in curves])

        # Build cd_matrix [n_dose, n_focus] by interpolating each curve onto
        # the shared focus_nm grid
        cd_matrix = np.array([
            np.interp(focus_nm, c.focus_points, c.cd_points) for c in curves
        ])

        # Nominal CD: dose closest to 1.0, focus closest to 0
        ni = min(range(len(curves)), key=lambda i: abs(curves[i].dose_factor - 1.0))
        fj = int(np.argmin(np.abs(focus_nm)))
        nominal_cd = cd_matrix[ni, fj]

        if nominal_cd <= 0:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    fontsize=theme.MPL_LABEL, color=theme.TEXT_TERTIARY,
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        cd_tol_frac = self.cd_tol_sb.value() / 100.0
        pass_fail = np.abs(cd_matrix - nominal_cd) / (nominal_cd + 1e-9) <= cd_tol_frac

        pw = ProcessWindow(2000.0, 256)
        pw_result = pw.compute_from_grid(pass_fail, dose_axis * 100.0, focus_nm)

        dof = pw_result.dof_nm
        el_pct = pw_result.el_pct
        best_focus = pw_result.best_focus_nm

        ax.pcolormesh(focus_nm, dose_axis * 100.0, pass_fail.astype(float),
                      cmap='RdYlGn', alpha=0.45, vmin=0, vmax=1)
        if np.any(pass_fail):
            ax.contour(focus_nm, dose_axis * 100.0, pass_fail.astype(float),
                       levels=[0.5], colors=[theme.ACCENT], linewidths=1.5)
            ax.plot(best_focus, pw_result.nominal_dose_pct, '+',
                    color=theme.ACCENT, markersize=12, markeredgewidth=1.8, zorder=3)

        ax.set_xlabel("Defocus (nm)", fontsize=theme.MPL_LABEL)
        ax.set_ylabel("Dose (%)", fontsize=theme.MPL_LABEL)
        ax.set_title("Process Window  DOF={:.0f} nm  EL={:.1f}%".format(dof, el_pct),
                     fontsize=theme.MPL_TITLE, fontweight='600', color=theme.TEXT_PRIMARY, pad=6)
        ax.text(0.05, 0.95,
                "DOF={:.0f} nm\nEL={:.1f}%".format(dof, el_pct),
                transform=ax.transAxes, fontsize=theme.MPL_TICK,
                va='top', color=theme.TEXT_SECONDARY,
                bbox=dict(boxstyle='round,pad=0.3', fc=theme.BG_PRIMARY,
                          ec=theme.BORDER, alpha=0.9))

        self.canvas.draw_idle()
        self._update_metrics_table(
            cd_target=self.cd_target_sb.value(),
            nils=self._result.nils if self._result else 0.0,
            meef=None,
            best_focus=best_focus,
            dof=dof,
            el=el_pct,
        )

    # ── FEM ────────────────────────────────────────────────────────────

    def _run_fem(self):
        if self._result is None or self._config is None:
            self.status_label.setText("Run a simulation first.")
            return
        if self._fem_thread and self._fem_thread.isRunning():
            return
        if self.dose_min_sb.value() >= self.dose_max_sb.value():
            self.status_label.setText("Dose min must be less than dose max.")
            return
        self.fem_btn.setEnabled(False)
        self.status_label.setText("Computing Focus-Exposure Matrix...")
        self._fem_thread = _FEMThread(
            config=self._config,
            layout_path=self._layout_path,
            focus_range=self.focus_range_sb.value(),
            dose_min=self.dose_min_sb.value(),
            dose_max=self.dose_max_sb.value(),
            parent=self,
        )
        self._fem_thread.finished.connect(self._on_fem_done)
        self._fem_thread.progress.connect(lambda msg, _pct: self.status_label.setText(msg))
        self._fem_thread.error.connect(
            lambda msg: self._on_analysis_error(msg, self.fem_btn))
        self._fem_thread.start()

    def _on_fem_done(self, fem):
        self.fem_btn.setEnabled(True)
        self.status_label.setText("FEM done.")
        self._plot_fem(fem)

    def _plot_fem(self, fem):
        ax = self.ax_fem
        ax.clear()
        self._style_ax(ax, "Focus-Exposure Matrix", image_panel=True)

        cd_target = self.cd_target_sb.value()
        cd_tol_pct = self.cd_tol_sb.value()

        im = ax.imshow(
            fem.cd_matrix,
            cmap='plasma',
            origin='lower',
            aspect='auto',
            extent=[
                fem.dose_values[0], fem.dose_values[-1],
                fem.focus_values[0], fem.focus_values[-1],
            ],
            vmin=cd_target * (1 - cd_tol_pct / 100),
            vmax=cd_target * (1 + cd_tol_pct / 100),
        )
        cb = self.figure.colorbar(im, ax=ax, fraction=0.035, shrink=0.85, pad=0.03)
        cb.set_label("CD (nm)", fontsize=theme.MPL_ANNOT)
        cb.ax.tick_params(labelsize=theme.MPL_ANNOT)

        # Process window contour overlay
        pw_mask = fem.process_window(cd_target, cd_tol_pct)
        dose_grid = fem.dose_values
        focus_grid = fem.focus_values
        if pw_mask.shape[0] > 1 and pw_mask.shape[1] > 1:
            ax.contour(dose_grid, focus_grid, pw_mask.astype(float),
                       levels=[0.5], colors=[theme.ACCENT], linewidths=2.0)

        # CD value labels on sparse grid
        n_focus, n_dose = fem.cd_matrix.shape
        step_f = max(1, n_focus // 4)
        step_d = max(1, n_dose // 4)
        for fi in range(0, n_focus, step_f):
            for di in range(0, n_dose, step_d):
                ax.text(
                    fem.dose_values[di], fem.focus_values[fi],
                    "{:.0f}".format(fem.cd_matrix[fi, di]),
                    ha='center', va='center',
                    fontsize=theme.MPL_ANNOT, color='white', fontweight='600',
                    bbox=dict(boxstyle='round,pad=0.1', fc='black', ec='none', alpha=0.45),
                )

        ax.set_xlabel("Dose factor", fontsize=theme.MPL_LABEL)
        ax.set_ylabel("Defocus (nm)", fontsize=theme.MPL_LABEL)
        self.canvas.draw_idle()
        self._plot_pw_from_fem(fem, cd_target, cd_tol_pct)

    def _plot_pw_from_fem(self, fem, cd_target, cd_tol_pct):
        """Update process window ellipse using FEM pass/fail data."""
        pw_mask = fem.process_window(cd_target, cd_tol_pct)

        focus_has_window = np.any(pw_mask, axis=1)
        if np.any(focus_has_window):
            focus_pass = fem.focus_values[focus_has_window]
            dof = float(focus_pass[-1] - focus_pass[0])
            best_focus = float(np.mean(focus_pass))
        else:
            dof = 0.0
            best_focus = 0.0

        el_pct = fem.exposure_latitude(best_focus, cd_tol_pct)

        ax = self.ax_pw
        ax.clear()
        self._style_ax(ax, "Process Window")

        if dof > 0 and el_pct > 0:
            ellipse = Ellipse(
                xy=(best_focus, 100.0),
                width=max(dof, 1.0),
                height=max(el_pct, 0.1),
                facecolor=theme.ACCENT,
                alpha=0.20,
                edgecolor=theme.ACCENT,
                linewidth=1.8,
                zorder=2,
            )
            ax.add_patch(ellipse)
            ax.plot(best_focus, 100.0, '+',
                    color=theme.ACCENT, markersize=12, markeredgewidth=1.8, zorder=3)
            ax.axhline(100.0, color=theme.TEXT_TERTIARY, linestyle=':', linewidth=0.8, alpha=0.4, zorder=1)
            ax.axvline(best_focus, color=theme.TEXT_TERTIARY, linestyle=':', linewidth=0.8, alpha=0.4, zorder=1)
            margin = max(dof, 30) * 1.6
            ax.set_xlim(best_focus - margin, best_focus + margin)
            ax.set_ylim(100.0 - el_pct * 1.8 - 1, 100.0 + el_pct * 1.8 + 1)
            ax.set_xlabel("Defocus (nm)", fontsize=theme.MPL_LABEL)
            ax.set_ylabel("Dose (%)", fontsize=theme.MPL_LABEL)
            ax.set_title("Process Window  DOF={:.0f} nm  EL={:.1f}%".format(dof, el_pct),
                         fontsize=theme.MPL_TITLE, fontweight='600', color=theme.TEXT_PRIMARY, pad=6)
            ax.text(0.05, 0.95,
                    "DOF={:.0f} nm\nEL={:.1f}%".format(dof, el_pct),
                    transform=ax.transAxes, fontsize=theme.MPL_TICK,
                    va='top', color=theme.TEXT_SECONDARY,
                    bbox=dict(boxstyle='round,pad=0.3', fc=theme.BG_PRIMARY,
                              ec=theme.BORDER, alpha=0.9))
        else:
            ax.text(0.5, 0.5, 'No process window found', ha='center', va='center',
                    fontsize=theme.MPL_LABEL, color=theme.TEXT_TERTIARY,
                    transform=ax.transAxes)
            ax.set_axis_off()

        self.canvas.draw_idle()
        self._update_metrics_table(
            cd_target=cd_target,
            nils=self._result.nils if self._result else 0.0,
            meef=None,
            best_focus=best_focus,
            dof=dof,
            el=el_pct,
        )

    # ── Metrics ────────────────────────────────────────────────────────

    def _update_metrics_table(self, cd_target=0.0, nils=0.0, meef=None,
                               best_focus=0.0, dof=0.0, el=0.0):
        rows = [
            ("CD target (nm)",  "{:.2f}".format(cd_target)),
            ("NILS",            "{:.3f}".format(nils)),
            ("MEEF",            "{:.3f}".format(meef) if meef is not None else "—"),
            ("Best focus (nm)", "{:.1f}".format(best_focus)),
            ("DOF (nm)",        "{:.1f}".format(dof)),
            ("EL (%)",          "{:.1f}".format(el)),
        ]
        self.metrics_table.setRowCount(len(rows))
        bold_font = QFont()
        bold_font.setBold(True)
        for i, (k, v) in enumerate(rows):
            ki = QTableWidgetItem(k)
            ki.setFont(bold_font)
            ki.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            vi = QTableWidgetItem(v)
            vi.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
            self.metrics_table.setItem(i, 0, ki)
            self.metrics_table.setItem(i, 1, vi)
        self.metrics_table.resizeColumnsToContents()
        self._draw_metrics_ax(rows)

    def _draw_metrics_ax(self, rows):
        ax = self.ax_metrics
        ax.clear()
        self._style_ax(ax, "Key Metrics")
        ax.set_axis_off()

        # Header label
        ax.text(0.05, 0.97, "KEY METRICS",
                transform=ax.transAxes,
                fontsize=theme.MPL_ANNOT, color=theme.TEXT_TERTIARY,
                fontweight='700', va='top')

        y = 0.86
        dy = 0.13
        for k, v in rows:
            is_dash = v.strip() == '\u2014'
            val_color = theme.TEXT_TERTIARY if is_dash else theme.TEXT_PRIMARY
            ax.text(0.05, y, k,
                    transform=ax.transAxes,
                    fontsize=theme.MPL_LABEL, color=theme.TEXT_SECONDARY, va='top')
            ax.text(0.95, y, v,
                    transform=ax.transAxes,
                    fontsize=theme.MPL_LABEL, color=val_color,
                    fontweight='700', va='top', ha='right')
            divider_y = y - dy * 0.75
            ax.plot([0.05, 0.95], [divider_y, divider_y],
                    transform=ax.transAxes,
                    color=theme.BORDER, linewidth=0.5, solid_capstyle='butt')
            y -= dy

        self.canvas.draw_idle()

    # ── Error handler ──────────────────────────────────────────────────

    def _on_analysis_error(self, msg, button=None):
        if button is not None:
            button.setEnabled(True)
        else:
            self.bossung_btn.setEnabled(True)
            self.fem_btn.setEnabled(True)
        self.status_label.setText("Error: " + msg.split('\n')[0])
