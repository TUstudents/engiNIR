"""WelcomeWindow — launcher screen shown when enginir-viewer starts without arguments."""

from __future__ import annotations

import importlib.resources
import os
import time

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .._logging import get_logger
from . import _is_project_file
from ._jobs import BusyStatusWidget
from ._recent import load as _load_recent
from ._theme import secondary_text_style

logger = get_logger(__name__)

_FILTER = "Hyperspectral files (*.eiff *.hdr *.h5 *.spc *.spa *.jdx *.csv *.xlsx);;All files (*)"

__all__ = ["WelcomeWindow"]


class WelcomeWindow(QMainWindow):
    """Launcher window: open file button, dark/white reference section, recent list."""

    #: Emitted when the user picks a file. Args: (path, dark_path | None, white_path | None)
    file_selected = pyqtSignal(str, object, object)
    #: Emitted when the user opens an existing project file. Args: (path,)
    project_selected = pyqtSignal(str)
    #: Emitted when the user creates a new project file. Args: (path,)
    project_created = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("engiNIR — HSI Viewer")
        self.setFixedSize(680, 500)

        try:
            icon_file = importlib.resources.files("enginir._resources").joinpath("enginir.ico")
            with importlib.resources.as_file(icon_file) as icon_path:
                self.setWindowIcon(QIcon(str(icon_path)))
        except Exception as exc:
            logger.debug("WelcomeWindow icon load failed: {}", exc)

        self._dark_path: str | None = None
        self._white_path: str | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("enginir")
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        subtitle = QLabel(
            "Open a dataset for exploration, or continue inside one persisted scientific workspace."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(secondary_text_style())
        layout.addWidget(subtitle)

        quick_start = QGroupBox("Quick Start")
        quick_layout = QVBoxLayout(quick_start)
        quick_layout.setSpacing(6)

        self._btn_open = QPushButton("Open Dataset...")
        self._btn_open.clicked.connect(self._on_open_clicked)
        quick_layout.addWidget(self._btn_open)

        project_row = QHBoxLayout()
        self._btn_new_project = QPushButton("Create Project...")
        self._btn_new_project.clicked.connect(self._on_new_project_clicked)
        self._btn_open_project = QPushButton("Open Project...")
        self._btn_open_project.clicked.connect(self._on_open_project_clicked)
        project_row.addWidget(self._btn_new_project)
        project_row.addWidget(self._btn_open_project)
        quick_layout.addLayout(project_row)

        launcher_hint = QLabel(
            "Open a dataset for EDA, or open/create a project to continue modeling and prediction."
        )
        launcher_hint.setWordWrap(True)
        launcher_hint.setStyleSheet(secondary_text_style())
        quick_layout.addWidget(launcher_hint)
        layout.addWidget(quick_start)

        refs_box = QGroupBox("Optional References")
        refs_layout = QVBoxLayout(refs_box)
        refs_layout.setSpacing(6)
        dark_row = QHBoxLayout()
        self._cb_dark = QCheckBox("Include dark reference")
        self._lbl_dark = QLabel("(none)")
        self._btn_dark = QPushButton("Browse...")
        self._btn_dark.setEnabled(False)
        self._btn_dark.clicked.connect(self._on_browse_dark)
        self._cb_dark.toggled.connect(self._btn_dark.setEnabled)
        dark_row.addWidget(self._cb_dark)
        dark_row.addWidget(self._lbl_dark)
        dark_row.addWidget(self._btn_dark)
        refs_layout.addLayout(dark_row)

        white_row = QHBoxLayout()
        self._cb_white = QCheckBox("Include white reference")
        self._lbl_white = QLabel("(none)")
        self._btn_white = QPushButton("Browse...")
        self._btn_white.setEnabled(False)
        self._btn_white.clicked.connect(self._on_browse_white)
        self._cb_white.toggled.connect(self._btn_white.setEnabled)
        white_row.addWidget(self._cb_white)
        white_row.addWidget(self._lbl_white)
        white_row.addWidget(self._btn_white)
        refs_layout.addLayout(white_row)
        layout.addWidget(refs_box)

        recent_box = QGroupBox("Recent Workspaces")
        recent_layout = QVBoxLayout(recent_box)
        recent_layout.setSpacing(6)
        recent_label = QLabel("Recent Files")
        recent_layout.addWidget(recent_label)
        self._recent_list = QListWidget()
        self._populate_recent()
        self._recent_list.itemActivated.connect(self._on_recent_double_clicked)
        recent_layout.addWidget(self._recent_list)
        layout.addWidget(recent_box, 1)

        self._loading_status = BusyStatusWidget(idle_text="Ready", parent=self)
        layout.addWidget(self._loading_status)

    def loading_status_widget(self) -> BusyStatusWidget:
        """Return the launcher busy-state widget for startup orchestration."""
        return self._loading_status

    def loading_controls(self) -> list[QWidget]:
        """Return controls that should be disabled while startup loading runs."""
        return [
            self._btn_open,
            self._btn_new_project,
            self._btn_open_project,
            self._cb_dark,
            self._cb_white,
            self._btn_dark,
            self._btn_white,
            self._recent_list,
        ]

    def loading_snapshot(self) -> tuple[str, bool]:
        """Return the current loading label and progress visibility for tests."""
        return self._loading_status.snapshot()

    def _populate_recent(self) -> None:
        self._recent_list.clear()
        for path in _load_recent():
            if os.path.exists(path):
                self._recent_list.addItem(path)

    def _on_open_clicked(self) -> None:
        dialog_started = time.perf_counter()
        path, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", _FILTER)
        logger.debug(
            "Startup timing: dataset file dialog returned in {:.3f}s (selected={})",
            time.perf_counter() - dialog_started,
            bool(path),
        )
        if not path:
            return
        dark = self._dark_path if self._cb_dark.isChecked() else None
        white = self._white_path if self._cb_white.isChecked() else None
        self.file_selected.emit(path, dark, white)

    def _on_open_project_clicked(self) -> None:
        dialog_started = time.perf_counter()
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "HDF5 (*.h5 *.hdf5)")
        logger.debug(
            "Startup timing: project file dialog returned in {:.3f}s (selected={})",
            time.perf_counter() - dialog_started,
            bool(path),
        )
        if path:
            self.project_selected.emit(path)

    def _on_new_project_clicked(self) -> None:
        dialog_started = time.perf_counter()
        path, _ = QFileDialog.getSaveFileName(self, "Create Project", "", "HDF5 (*.h5 *.hdf5)")
        logger.debug(
            "Startup timing: create-project dialog returned in {:.3f}s (selected={})",
            time.perf_counter() - dialog_started,
            bool(path),
        )
        if path:
            self.project_created.emit(path)

    def _on_browse_dark(self) -> None:
        dialog_started = time.perf_counter()
        path, _ = QFileDialog.getOpenFileName(self, "Dark Reference", "", _FILTER)
        logger.debug(
            "Startup timing: dark-reference dialog returned in {:.3f}s (selected={})",
            time.perf_counter() - dialog_started,
            bool(path),
        )
        if path:
            self._dark_path = path
            self._lbl_dark.setText(os.path.basename(path))

    def _on_browse_white(self) -> None:
        dialog_started = time.perf_counter()
        path, _ = QFileDialog.getOpenFileName(self, "White Reference", "", _FILTER)
        logger.debug(
            "Startup timing: white-reference dialog returned in {:.3f}s (selected={})",
            time.perf_counter() - dialog_started,
            bool(path),
        )
        if path:
            self._white_path = path
            self._lbl_white.setText(os.path.basename(path))

    def _on_recent_double_clicked(self, item: QListWidgetItem) -> None:
        path = os.path.abspath(item.text())
        logger.debug("Startup timing: recent item activated: {}", path)
        if _is_project_file(path):
            self.project_selected.emit(path)
            return
        dark = self._dark_path if self._cb_dark.isChecked() else None
        white = self._white_path if self._cb_white.isChecked() else None
        self.file_selected.emit(path, dark, white)
