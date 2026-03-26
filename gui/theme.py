"""Toss-style design system for the optical lithography simulator GUI."""

# ── Color Palette ──────────────────────────────────────────────────────────
BG_PRIMARY     = "#FFFFFF"
BG_SECONDARY   = "#F9FAFB"
BG_TERTIARY    = "#F2F4F6"
TEXT_PRIMARY   = "#191F28"
TEXT_SECONDARY = "#4E5968"
TEXT_TERTIARY  = "#8B95A1"
BORDER         = "#E5E8EB"
ACCENT         = "#3182F6"
ACCENT_HOVER   = "#1B64DA"
SUCCESS        = "#00C853"
DANGER         = "#F44336"
WARNING        = "#FF9800"

# ── Bossung/FEM color palette (5 dose levels) ──────────────────────────────
BOSSUNG_COLORS = ['#3182F6', '#00C853', '#FF9800', '#F44336', '#9C27B0']

# ── Matplotlib ─────────────────────────────────────────────────────────────
MPL_DPI    = 96
MPL_TITLE  = 11
MPL_LABEL  = 9
MPL_TICK   = 8
MPL_ANNOT  = 7
MPL_LEGEND = 7

# ── Spacing (px) ───────────────────────────────────────────────────────────
SP_XS = 4
SP_SM = 8
SP_MD = 12
SP_LG = 16
SP_XL = 24


def get_qss() -> str:
    """Return complete Toss-style Qt stylesheet."""
    return f"""
    /* === Global === */
    * {{ font-family: -apple-system, "SF Pro Text", "Segoe UI", "Pretendard", sans-serif; }}
    QWidget {{ background: {BG_PRIMARY}; color: {TEXT_PRIMARY}; font-size: 13px; }}

    /* === Main Window === */
    QMainWindow {{ background: {BG_TERTIARY}; }}

    /* === Tab Widget === */
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        background: {BG_PRIMARY};
        border-radius: 0 0 8px 8px;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {TEXT_SECONDARY};
        padding: 8px 20px;
        min-height: 36px;
        margin-right: 2px;
        border: none;
        border-bottom: 2px solid transparent;
        font-size: 13px;
        font-weight: 500;
    }}
    QTabBar::tab:selected {{
        color: {ACCENT};
        border-bottom: 3px solid {ACCENT};
        font-weight: 600;
    }}
    QTabBar::tab:hover:!selected {{ color: {TEXT_PRIMARY}; background: {BG_TERTIARY}; }}

    /* === Buttons === */
    QPushButton {{
        background: {BG_TERTIARY};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 13px;
        font-weight: 500;
        min-height: 32px;
    }}
    QPushButton:hover {{ background: #EAECEF; border-color: #CDD1D7; }}
    QPushButton:pressed {{ background: #DDE0E4; }}
    QPushButton:checked {{ background: {ACCENT}; color: white; border-color: {ACCENT}; }}
    QPushButton:disabled {{ color: {TEXT_TERTIARY}; background: {BG_TERTIARY}; border-color: {BORDER}; }}

    QPushButton[class="primary"] {{
        background: {ACCENT}; color: white; border: none;
        min-height: 40px; font-weight: 600;
    }}
    QPushButton[class="primary"]:hover {{ background: {ACCENT_HOVER}; }}

    QPushButton#run_btn, QPushButton[objectName="success"] {{
        background: {SUCCESS}; color: white; border: none; font-weight: 600; min-height: 40px;
    }}
    QPushButton#run_btn:hover, QPushButton[objectName="success"]:hover {{ background: #00A844; }}

    QPushButton#stop_btn, QPushButton[objectName="danger"] {{
        background: {DANGER}; color: white; border: none; font-weight: 600; min-height: 40px;
    }}
    QPushButton#stop_btn:hover, QPushButton[objectName="danger"]:hover {{ background: #D32F2F; }}

    QPushButton[objectName="secondary"] {{
        background: {BG_PRIMARY}; color: {ACCENT}; border: 1px solid {ACCENT};
    }}
    QPushButton[objectName="secondary"]:hover {{ background: #EBF2FF; }}

    /* === GroupBox === */
    QGroupBox {{
        font-weight: 600;
        font-size: 13px;
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 8px;
        margin-top: 10px;
        padding-top: 8px;
        background: {BG_PRIMARY};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: {TEXT_PRIMARY};
    }}

    /* === Inputs === */
    QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 4px 8px;
        background: {BG_PRIMARY};
        color: {TEXT_PRIMARY};
        font-size: 13px;
        min-height: 30px;
        selection-background-color: {ACCENT};
    }}
    QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
        border: 2px solid {ACCENT};
    }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox::down-arrow {{ width: 10px; height: 10px; }}
    QComboBox QAbstractItemView {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        background: {BG_PRIMARY};
        selection-background-color: #EBF2FF;
        selection-color: {ACCENT};
    }}

    /* === Table === */
    QTableWidget {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        background: {BG_PRIMARY};
        alternate-background-color: {BG_SECONDARY};
        gridline-color: {BORDER};
        font-size: 12px;
    }}
    QTableWidget::item {{ padding: 4px 8px; border-bottom: 1px solid {BORDER}; }}
    QTableWidget::item:selected {{ background: #EBF2FF; color: {ACCENT}; }}
    QHeaderView::section {{
        background: {BG_TERTIARY};
        color: {TEXT_SECONDARY};
        font-weight: 700;
        font-size: 11px;
        padding: 6px 8px;
        border: none;
        border-bottom: 1px solid {BORDER};
        border-right: 1px solid {BORDER};
    }}

    /* === List Widget === */
    QListWidget {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        background: {BG_PRIMARY};
        font-size: 12px;
        outline: none;
    }}
    QListWidget::item {{ padding: 6px 8px; border-bottom: 1px solid {BORDER}; }}
    QListWidget::item:selected {{ background: #EBF2FF; color: {ACCENT}; }}
    QListWidget::item:hover:!selected {{ background: {BG_TERTIARY}; }}

    /* === Text Edit === */
    QTextEdit {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        background: {BG_PRIMARY};
        color: {TEXT_PRIMARY};
        font-family: "Menlo", "Consolas", "Courier New", monospace;
        font-size: 12px;
        padding: 4px;
    }}

    /* === Progress Bar === */
    QProgressBar {{
        border: none;
        border-radius: 4px;
        background: {BG_TERTIARY};
        text-align: center;
        font-size: 11px;
        color: {TEXT_SECONDARY};
        min-height: 6px;
        max-height: 6px;
    }}
    QProgressBar::chunk {{
        background: {ACCENT};
        border-radius: 4px;
    }}

    /* === Toolbar === */
    QToolBar {{
        background: {BG_TERTIARY};
        border-bottom: 1px solid {BORDER};
        spacing: 4px;
        padding: 4px 8px;
    }}
    QToolBar::separator {{ width: 1px; background: {BORDER}; margin: 4px 2px; }}
    QToolButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 5px 10px;
        color: {TEXT_PRIMARY};
        font-size: 13px;
        font-weight: 500;
    }}
    QToolButton:hover {{ background: #EAECEF; border-color: {BORDER}; }}
    QToolButton:pressed {{ background: #DDE0E4; }}

    /* === Status Bar === */
    QStatusBar {{
        background: {BG_TERTIARY};
        border-top: 1px solid {BORDER};
        color: {TEXT_SECONDARY};
        font-size: 12px;
    }}

    /* === Menu === */
    QMenuBar {{
        background: {BG_TERTIARY};
        border-bottom: 1px solid {BORDER};
        color: {TEXT_PRIMARY};
        font-size: 13px;
    }}
    QMenuBar::item {{ padding: 6px 10px; background: transparent; }}
    QMenuBar::item:selected {{ background: #EAECEF; border-radius: 4px; }}
    QMenu {{
        background: {BG_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 4px;
    }}
    QMenu::item {{ padding: 8px 16px; border-radius: 4px; color: {TEXT_PRIMARY}; }}
    QMenu::item:selected {{ background: #EBF2FF; color: {ACCENT}; }}
    QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}

    /* === Splitter === */
    QSplitter::handle {{ background: {BORDER}; }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical {{ height: 1px; }}

    /* === Scroll Area === */
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{
        width: 6px; background: transparent; border: none;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER}; border-radius: 3px; min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        height: 6px; background: transparent; border: none;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER}; border-radius: 3px; min-width: 20px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

    /* === Frame === */
    QFrame[frameShape="4"], QFrame[frameShape="5"] {{
        color: {BORDER};
    }}

    /* === Label === */
    QLabel {{ background: transparent; color: {TEXT_PRIMARY}; }}
    QLabel[objectName="caption"] {{ color: {TEXT_SECONDARY}; font-size: 11px; }}
    QLabel[objectName="led_idle"] {{ background: {TEXT_TERTIARY}; border-radius: 6px; }}
    QLabel[objectName="led_running"] {{ background: {WARNING}; border-radius: 6px; }}
    QLabel[objectName="led_done"] {{ background: {SUCCESS}; border-radius: 6px; }}
    QLabel[objectName="led_error"] {{ background: {DANGER}; border-radius: 6px; }}
    """


def apply_mpl_theme() -> None:
    """Apply Toss-style defaults to matplotlib."""
    try:
        import matplotlib as mpl
        mpl.rcParams.update({
            "figure.facecolor":  BG_PRIMARY,
            "axes.facecolor":    BG_SECONDARY,
            "axes.edgecolor":    BORDER,
            "axes.labelcolor":   TEXT_SECONDARY,
            "xtick.color":       TEXT_TERTIARY,
            "ytick.color":       TEXT_TERTIARY,
            "text.color":        TEXT_PRIMARY,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "axes.titlesize":    MPL_TITLE,
            "axes.labelsize":    MPL_LABEL,
            "xtick.labelsize":   MPL_TICK,
            "ytick.labelsize":   MPL_TICK,
            "legend.fontsize":   MPL_LEGEND,
            "figure.dpi":        MPL_DPI,
            "savefig.dpi":       MPL_DPI,
            "font.family":       ["DejaVu Sans", "sans-serif"],
        })
    except Exception:
        pass
