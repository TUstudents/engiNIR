"""WelcomeWindow — launcher screen shown when enginir-viewer starts without arguments."""

from __future__ import annotations

import os
import time

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
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
from ._app import app_icon
from ._icons import icon
from ._jobs import BusyStatusWidget
from ._recent import load as _load_recent
from ._theme import secondary_text_style
from ._ui_helpers import PROJECT_OPEN_FILTER, PROJECT_SAVE_FILTER, ask_save_path

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

    #: Emitted when the user asks for the example project (created on first use).
    example_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("engiNIR")
        self.setFixedSize(760, 600)

        self._dark_path: str | None = None
        self._white_path: str | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 16)

        heading = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(40, 40))
        heading.addWidget(logo)
        title = QLabel("engiNIR")
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        title.setFont(font)
        heading.addWidget(title)
        heading.addStretch(1)
        layout.addLayout(heading)

        subtitle = QLabel("Calibration models for NIR spectra and hyperspectral images.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(secondary_text_style())
        layout.addWidget(subtitle)

        quick_start = QGroupBox("Start")
        quick_layout = QGridLayout(quick_start)
        quick_layout.setHorizontalSpacing(10)
        quick_layout.setVerticalSpacing(4)
        self._btn_example = QPushButton("Open Example Project")
        self._btn_example.setIcon(icon("wand", "accent"))
        self._btn_example.setDefault(True)
        self._btn_example.clicked.connect(self.example_requested.emit)
        self._btn_new_project = QPushButton("New Project...")
        self._btn_new_project.setIcon(icon("file-plus"))
        self._btn_new_project.clicked.connect(self._on_new_project_clicked)
        self._btn_open_project = QPushButton("Open Project...")
        self._btn_open_project.setIcon(icon("folder-open"))
        self._btn_open_project.clicked.connect(self._on_open_project_clicked)
        hints = (
            "Five tablet scans with a fitted model: the fastest way to see how it works.",
            "Choose your scans and their references, then name what to predict.",
            "Continue a project; scans that moved are found again.",
        )
        for column, (button, hint) in enumerate(zip(
                (self._btn_example, self._btn_new_project, self._btn_open_project), hints)):
            button.setMinimumHeight(34)
            quick_layout.addWidget(button, 0, column)
            label = QLabel(hint)
            label.setWordWrap(True)
            label.setStyleSheet(secondary_text_style())
            quick_layout.addWidget(label, 1, column)
        layout.addWidget(quick_start)

        recent_box = QGroupBox("Recent")
        recent_layout = QVBoxLayout(recent_box)
        recent_layout.setSpacing(6)
        self._recent_list = QListWidget()
        self._populate_recent()
        self._recent_list.itemActivated.connect(self._on_recent_double_clicked)
        recent_layout.addWidget(self._recent_list)
        layout.addWidget(recent_box, 1)

        refs_box = QGroupBox("Explore a dataset without a project")
        refs_layout = QVBoxLayout(refs_box)
        refs_layout.setSpacing(6)
        self._btn_open = QPushButton("Open Dataset...")
        self._btn_open.clicked.connect(self._on_open_clicked)
        dark_row = QHBoxLayout()
        self._cb_dark = QCheckBox("Dark reference")
        self._lbl_dark = QLabel("(none)")
        self._btn_dark = QPushButton("Browse...")
        self._btn_dark.setEnabled(False)
        self._btn_dark.clicked.connect(self._on_browse_dark)
        self._cb_dark.toggled.connect(self._btn_dark.setEnabled)
        dark_row.addWidget(self._btn_open)
        dark_row.addSpacing(12)
        dark_row.addWidget(self._cb_dark)
        dark_row.addWidget(self._lbl_dark)
        dark_row.addWidget(self._btn_dark)
        white_row = QHBoxLayout()
        self._cb_white = QCheckBox("White reference")
        self._lbl_white = QLabel("(none)")
        self._btn_white = QPushButton("Browse...")
        self._btn_white.setEnabled(False)
        self._btn_white.clicked.connect(self._on_browse_white)
        self._cb_white.toggled.connect(self._btn_white.setEnabled)
        dark_row.addWidget(self._cb_white)
        dark_row.addWidget(self._lbl_white)
        dark_row.addWidget(self._btn_white)
        refs_layout.addLayout(dark_row)
        refs_layout.addLayout(white_row)
        layout.addWidget(refs_box)

        self._loading_status = BusyStatusWidget(idle_text="Ready", parent=self)
        layout.addWidget(self._loading_status)

    def loading_status_widget(self) -> BusyStatusWidget:
        """Return the launcher busy-state widget for startup orchestration."""
        return self._loading_status

    def loading_controls(self) -> list[QWidget]:
        """Return controls that should be disabled while startup loading runs."""
        return [
            self._btn_example,
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
        """Recent files by name, with the folder after it (the whole path in the tooltip)."""
        self._recent_list.clear()
        for path in _load_recent():
            if os.path.exists(path):
                folder = os.path.dirname(path)
                kind = "project" if _is_project_file(path) else "dataset"
                item = QListWidgetItem(icon("file-text" if kind == "project" else "photo"),
                                       f"{os.path.basename(path)}    {_short_folder(folder)}")
                item.setData(Qt.ItemDataRole.UserRole, path)
                item.setToolTip(path)
                self._recent_list.addItem(item)

    def _on_open_clicked(self) -> None:
        dialog_started = time.perf_counter()
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset", "", _FILTER)
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
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", PROJECT_OPEN_FILTER)
        logger.debug(
            "Startup timing: project file dialog returned in {:.3f}s (selected={})",
            time.perf_counter() - dialog_started,
            bool(path),
        )
        if path:
            self.project_selected.emit(path)

    def _on_new_project_clicked(self) -> None:
        dialog_started = time.perf_counter()
        path = ask_save_path(self, "Create Project", PROJECT_SAVE_FILTER, ".h5", (".hdf5",))
        logger.debug(
            "Startup timing: create-project dialog returned in {:.3f}s (selected={})",
            time.perf_counter() - dialog_started,
            bool(path),
        )
        if path:
            self.project_created.emit(path)

    def _on_browse_dark(self) -> None:
        dialog_started = time.perf_counter()
        path, _ = QFileDialog.getOpenFileName(self, "Dark reference", "", _FILTER)
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
        path, _ = QFileDialog.getOpenFileName(self, "White reference", "", _FILTER)
        logger.debug(
            "Startup timing: white-reference dialog returned in {:.3f}s (selected={})",
            time.perf_counter() - dialog_started,
            bool(path),
        )
        if path:
            self._white_path = path
            self._lbl_white.setText(os.path.basename(path))

    def _on_recent_double_clicked(self, item: QListWidgetItem) -> None:
        path = os.path.abspath(str(item.data(Qt.ItemDataRole.UserRole) or item.text()))
        logger.debug("Startup timing: recent item activated: {}", path)
        if _is_project_file(path):
            self.project_selected.emit(path)
            return
        dark = self._dark_path if self._cb_dark.isChecked() else None
        white = self._white_path if self._cb_white.isChecked() else None
        self.file_selected.emit(path, dark, white)


def _short_folder(folder: str, limit: int = 48) -> str:
    """The end of a folder path, the part that tells recent files apart."""
    return folder if len(folder) <= limit else "…" + folder[-(limit - 1):]
