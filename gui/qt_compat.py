"""PySide6/PyQt5 compatibility shim — import Qt classes from here."""

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QListWidget,
        QListWidgetItem, QPushButton, QLabel, QFileDialog, QMessageBox,
        QProgressBar, QFrame, QMainWindow, QTabWidget, QStatusBar,
        QApplication, QToolBar, QStyle, QTableWidget, QTableWidgetItem,
        QDoubleSpinBox, QSpinBox, QComboBox, QScrollArea, QGroupBox,
        QCheckBox, QSlider, QGridLayout, QFormLayout, QSizePolicy,
        QLineEdit,
    )
    from PySide6.QtCore import Qt, Signal, QThread, QObject, QElapsedTimer
    from PySide6.QtGui import QAction, QColor, QPalette, QFont

except ImportError:
    from PyQt5.QtWidgets import (  # type: ignore[no-redef]
        QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QListWidget,
        QListWidgetItem, QPushButton, QLabel, QFileDialog, QMessageBox,
        QProgressBar, QFrame, QMainWindow, QTabWidget, QStatusBar,
        QApplication, QToolBar, QStyle, QTableWidget, QTableWidgetItem,
        QDoubleSpinBox, QSpinBox, QComboBox, QScrollArea, QGroupBox,
        QCheckBox, QSlider, QGridLayout, QFormLayout, QSizePolicy,
        QLineEdit, QAction,
    )
    from PyQt5.QtCore import Qt, pyqtSignal as Signal, QThread, QObject, QElapsedTimer  # type: ignore[no-redef]
    from PyQt5.QtGui import QColor, QPalette, QFont  # type: ignore[no-redef]

__all__ = [
    "QWidget", "QVBoxLayout", "QHBoxLayout", "QSplitter", "QListWidget",
    "QListWidgetItem", "QPushButton", "QLabel", "QFileDialog", "QMessageBox",
    "QProgressBar", "QFrame", "QMainWindow", "QTabWidget", "QStatusBar",
    "QApplication", "QToolBar", "QStyle", "QTableWidget", "QTableWidgetItem",
    "QFont", "Qt", "Signal", "QThread", "QObject", "QAction", "QColor",
    "QPalette", "QElapsedTimer", "QDoubleSpinBox", "QSpinBox", "QComboBox",
    "QScrollArea", "QGroupBox", "QCheckBox", "QSlider", "QGridLayout",
    "QFormLayout", "QSizePolicy", "QLineEdit",
]
