"""WelcomeWindow — launcher screen shown when enginir-viewer starts without arguments."""

from __future__ import annotations

import os
import time

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
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
from ._files import _is_project_file
from ._app import app_icon
from ._windows import keep_open
from ._icons import icon
from ._jobs import BusyStatusWidget, start_background_job_soon
from ._preferences import start_folder
from . import _recent
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
        # Deleted by Qt when it closes, after the hand-off to the first viewer window.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        keep_open(self)

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

        subtitle = QLabel("A universal engineering tool for near-infrared spectra and "
                          "hyperspectral imaging")
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
        """Recent files by name, with the folder after it (the whole path in the tooltip).

        Drawn from the recent-files list alone: no recent path is touched on the UI
        thread, since it may be on a slow or unreachable network drive (plan
        startup-speed-1.7.1, D4). A background job then checks which files exist,
        grays out the missing ones and learns the kind of entries that have none.
        """
        self._recent_list.clear()
        recent = _recent.entries()
        for entry in recent:
            item = QListWidgetItem(f"{os.path.basename(entry.path)}    "
                                   f"{_short_folder(os.path.dirname(entry.path))}")
            item.setData(Qt.ItemDataRole.UserRole, entry.path)
            item.setToolTip(entry.path)
            self._recent_list.addItem(item)
            _show_kind(item, entry.kind)
        self._recent_checked = False
        if recent:
            # From the event loop: no thread starts while the window is built or shown.
            start_background_job_soon(owner=self, fn=lambda: _recent.check(recent),
                                 on_success=self._apply_recent_check,
                                 on_error=lambda message: logger.warning(
                                     "Recent files could not be checked: {}", message),
                                 on_finished=lambda: None)

    def _apply_recent_check(self, result: object) -> None:
        """Gray out the recent files that do not exist; show and store the kinds learned."""
        if not isinstance(result, dict):
            return
        for row in range(self._recent_list.count()):
            item = self._recent_list.item(row)
            if item is None:
                continue
            path = str(item.data(Qt.ItemDataRole.UserRole))
            exists, kind = result.get(path, (True, None))
            if not exists:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip(f"{path}\n(not found)")
            elif kind is not None and item.data(_KIND_ROLE) != kind:
                _show_kind(item, kind)
                _recent.set_kind(path, kind)
        self._recent_checked = True

    def _on_open_clicked(self) -> None:
        dialog_started = time.perf_counter()
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset", start_folder("dataset"), _FILTER)
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
        path, _ = QFileDialog.getOpenFileName(self, "Open project", start_folder("project"),
                                              PROJECT_OPEN_FILTER)
        logger.debug(
            "Startup timing: project file dialog returned in {:.3f}s (selected={})",
            time.perf_counter() - dialog_started,
            bool(path),
        )
        if path:
            self.project_selected.emit(path)

    def _on_new_project_clicked(self) -> None:
        dialog_started = time.perf_counter()
        path = ask_save_path(self, "Create Project", PROJECT_SAVE_FILTER, ".h5", (".hdf5",),
                             directory=start_folder("project"))
        logger.debug(
            "Startup timing: create-project dialog returned in {:.3f}s (selected={})",
            time.perf_counter() - dialog_started,
            bool(path),
        )
        if path:
            self.project_created.emit(path)

    def _on_browse_dark(self) -> None:
        dialog_started = time.perf_counter()
        path, _ = QFileDialog.getOpenFileName(self, "Dark reference", start_folder("dataset"),
                                              _FILTER)
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
        path, _ = QFileDialog.getOpenFileName(self, "White reference", start_folder("dataset"),
                                              _FILTER)
        logger.debug(
            "Startup timing: white-reference dialog returned in {:.3f}s (selected={})",
            time.perf_counter() - dialog_started,
            bool(path),
        )
        if path:
            self._white_path = path
            self._lbl_white.setText(os.path.basename(path))

    def _on_recent_double_clicked(self, item: QListWidgetItem) -> None:
        if not item.flags() & Qt.ItemFlag.ItemIsEnabled:
            return  # a missing file, grayed out by the background check
        path = os.path.abspath(str(item.data(Qt.ItemDataRole.UserRole) or item.text()))
        logger.debug("Startup timing: recent item activated: {}", path)
        kind = item.data(_KIND_ROLE)
        if kind == "project" or (kind is None and _is_project_file(path)):
            self.project_selected.emit(path)
            return
        dark = self._dark_path if self._cb_dark.isChecked() else None
        white = self._white_path if self._cb_white.isChecked() else None
        self.file_selected.emit(path, dark, white)


#: The item data holding a recent entry's kind ("project", "dataset" or None).
_KIND_ROLE = Qt.ItemDataRole.UserRole + 1


def _show_kind(item: QListWidgetItem, kind: str | None) -> None:
    item.setData(_KIND_ROLE, kind)
    item.setIcon(icon("file-text" if kind == "project" else "photo") if kind else QIcon())


def _short_folder(folder: str, limit: int = 48) -> str:
    """The end of a folder path, the part that tells recent files apart."""
    return folder if len(folder) <= limit else "…" + folder[-(limit - 1):]
