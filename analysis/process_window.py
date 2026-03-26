"""
Process window analysis for optical lithography.
Computes Exposure Latitude (EL) and Depth of Focus (DOF).
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ProcessWindowResult:
    """Results from process window analysis."""
    el_pct: float = 0.0      # Exposure Latitude (%)
    dof_nm: float = 0.0      # Depth of Focus (nm)
    pw_area: float = 0.0     # Process window area (% * nm)
    best_focus_nm: float = 0.0
    nominal_dose_pct: float = 0.0
    pass_fail_grid: Optional[np.ndarray] = None
    dose_axis: Optional[np.ndarray] = None
    focus_axis: Optional[np.ndarray] = None


class ProcessWindow:
    """Compute and analyze lithographic process windows."""

    def __init__(self, domain_size_nm: float, grid_size: int):
        self.domain_size_nm = domain_size_nm
        self.grid_size = grid_size

    def compute_from_grid(self, pass_fail_grid: np.ndarray,
                           dose_axis_pct: np.ndarray,
                           focus_axis_nm: np.ndarray) -> ProcessWindowResult:
        """
        Compute process window metrics from pass/fail grid.

        Args:
            pass_fail_grid: Boolean array [n_dose, n_focus]
            dose_axis_pct: Dose values (% relative to nominal)
            focus_axis_nm: Focus values (nm)
        Returns:
            ProcessWindowResult with EL, DOF, area
        """
        if not np.any(pass_fail_grid):
            return ProcessWindowResult()

        # EL per focus: dose range that passes at each focus
        el_per_focus = []
        for j in range(len(focus_axis_nm)):
            col = pass_fail_grid[:, j]
            if np.any(col):
                passing_doses = dose_axis_pct[col]
                el = float(np.max(passing_doses) - np.min(passing_doses))
            else:
                el = 0.0
            el_per_focus.append(el)

        # DOF: focus range where EL > 0
        el_per_focus = np.array(el_per_focus)
        has_window = el_per_focus > 0
        if np.any(has_window):
            if len(focus_axis_nm) >= 2:
                focus_step = focus_axis_nm[1] - focus_axis_nm[0]
            else:
                focus_step = 0.0
            dof = float(np.sum(has_window) * focus_step)
            best_focus_idx = np.argmax(el_per_focus)
            best_focus = float(focus_axis_nm[best_focus_idx])
        else:
            dof = 0.0
            best_focus = 0.0

        # Maximum EL
        el_max = float(np.max(el_per_focus))

        # Process window area (EL * DOF)
        pw_area = el_max * dof

        # Nominal dose: center of passing dose range at best focus
        best_j = np.argmax(el_per_focus) if np.any(el_per_focus > 0) else len(focus_axis_nm) // 2
        col_best = pass_fail_grid[:, best_j]
        if np.any(col_best):
            passing_doses = dose_axis_pct[col_best]
            nominal_dose = float(0.5 * (np.max(passing_doses) + np.min(passing_doses)))
        else:
            nominal_dose = 0.0

        return ProcessWindowResult(
            el_pct=el_max,
            dof_nm=dof,
            pw_area=pw_area,
            best_focus_nm=best_focus,
            nominal_dose_pct=nominal_dose,
            pass_fail_grid=pass_fail_grid,
            dose_axis=dose_axis_pct,
            focus_axis=focus_axis_nm
        )
