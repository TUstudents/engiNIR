"""Preferences of the viewer: Tools > Preferences.

Stored per user in ``preferences.json`` next to the other viewer settings
(``config_dir()``), never in a project: a project must not depend on one
machine's preferences. Fitting and prediction defaults only pre-fill the
settings docks.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from collections.abc import Sequence
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QKeySequence, QPalette
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .._logging import get_logger
from .._env import BLAS_THREADS_ENV
from .._levels import DEFAULT_BLACK_LEVEL, DEFAULT_WHITE_LEVEL
from ._icons import icon
from ._recent import config_dir
from ._theme import HEAT_STOPS, SERIES, secondary_text_style, value_colormap

logger = get_logger(__name__)

_FILE = "preferences.json"
#: Interface scales offered (Qt's QT_SCALE_FACTOR; applied at the next start).
SCALES = (0.8, 0.9, 1.0, 1.1, 1.25)


@dataclass(frozen=True)
class Preferences:
    """Everything Tools > Preferences sets."""

    # Appearance
    theme: str = "dark"                 # dark, light, system
    interface_scale: float = 1.0        # Qt scale factor, from the next start
    raise_checks: bool = True           # bring Checks to the front when a new check opens
    plot_help: bool = True              # the "?" button of each plot
    # Data and paths
    project_folder: str = ""            # first folder of New/Open Project ("": last used)
    dataset_folder: str = ""
    # Detector levels a new dataset starts with (Add Dataset; a file's own levels win)
    black_level: float = float(DEFAULT_BLACK_LEVEL)
    white_level: float = float(DEFAULT_WHITE_LEVEL)
    # Fitting defaults (pre-fill the Fit settings)
    model_type: str = "PLS"
    n_components: int = 6
    auto_components: bool = False
    hold_out: str = "roi"               # roi, sample, target
    x_scaling: str = "center"
    y_scaling: str = "uv"
    # Prediction defaults (pre-fill the Prediction settings)
    exclude_dmodx: bool = False
    dmodx_level: float = 0.95  # LIMIT_LEVEL, SIMCA's default
    # Plots and export
    export_dpi: int = 300
    # Performance
    blas_threads: int = 0               # 0: half the CPUs
    # Logging
    log_level: str = "INFO"             # the log file's level, from the next start


def _path():
    return config_dir() / _FILE


def load() -> Preferences:
    """The stored preferences; defaults for anything missing or unreadable."""
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Preferences()
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable preferences {}: {}", _path(), exc)
        return Preferences()
    if not isinstance(data, dict):
        return Preferences()
    defaults = Preferences()
    values: dict[str, Any] = {}
    for item in fields(Preferences):
        if item.name in data:
            default = getattr(defaults, item.name)
            try:
                values[item.name] = type(default)(data[item.name])
            except (TypeError, ValueError):
                logger.warning("Ignoring preference {}={!r}", item.name, data[item.name])
    loaded = replace(defaults, **values)
    if not loaded.white_level > loaded.black_level:
        logger.warning("Ignoring detector levels {} / {}: the white level must be above the "
                       "black level", loaded.black_level, loaded.white_level)
        loaded = replace(loaded, black_level=defaults.black_level,
                         white_level=defaults.white_level)
    return loaded


def save(preferences: Preferences) -> None:
    """Store ``preferences`` (logged, never raised, when the folder is not writable)."""
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(preferences), indent=1), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not save preferences {}: {}", path, exc)


def drive_mapping_text(*, windows: bool | None = None, wsl_root: str | None = None) -> str:
    """The drive mapping this platform applies to stored paths, as Preferences shows it."""
    windows = os.name == "nt" if windows is None else windows
    if windows:
        return "/mnt/z/… and /z/…  →  Z:\\…  (every drive letter)"
    from ..project._data_paths import wsl_mount_root  # the library, only when needed

    root = wsl_mount_root() if wsl_root is None else wsl_root
    return f"Z:\\…  →  {root}z/…  (every drive letter)"


def start_folder(kind: str) -> str:
    """The folder a ``"project"`` or ``"dataset"`` file dialog opens in ("": Qt's default).

    A preferred folder that does not exist (another computer, a share not
    mounted) is ignored.
    """
    folder = load().project_folder if kind == "project" else load().dataset_folder
    return folder if folder and os.path.isdir(folder) else ""


def apply_environment(preferences: Preferences) -> None:
    """Settings that act through the environment: BLAS threads now, the scale at the next start.

    An environment variable the user set before starting wins.
    """
    if preferences.blas_threads > 0:
        os.environ[BLAS_THREADS_ENV] = str(preferences.blas_threads)
    elif os.environ.get("_ENGINIR_BLAS_FROM_PREFERENCES"):
        os.environ.pop(BLAS_THREADS_ENV, None)
    if preferences.blas_threads > 0:
        os.environ["_ENGINIR_BLAS_FROM_PREFERENCES"] = "1"


def startup_scale() -> None:
    """Set Qt's scale factor from the preferences before the application starts."""
    if "QT_SCALE_FACTOR" in os.environ:
        return
    scale = load().interface_scale
    if scale and abs(scale - 1.0) > 1e-6:
        os.environ["QT_SCALE_FACTOR"] = f"{scale:g}"



def shortcut_table(shortcuts: Sequence[tuple[str, str]]) -> QTableWidget:
    """A read-only table of ``(menu item, shortcut)`` (Preferences > Keyboard and
    Help > Keyboard Shortcuts)."""
    table = QTableWidget(len(shortcuts), 2)
    table.setHorizontalHeaderLabels(["Menu item", "Shortcut"])
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    vertical = table.verticalHeader()
    if vertical is not None:
        vertical.setVisible(False)
    header = table.horizontalHeader()
    if header is not None:
        header.setStretchLastSection(True)
    for row, (name, keys) in enumerate(shortcuts):
        table.setItem(row, 0, QTableWidgetItem(name))
        table.setItem(row, 1, QTableWidgetItem(
            QKeySequence(keys).toString(QKeySequence.SequenceFormat.NativeText)))
    table.resizeColumnToContents(0)
    return table

class PreferencesDialog(QDialog):
    """Tools > Preferences: pages on the left, the settings of the page on the right."""

    PAGES = ("Appearance", "Data and paths", "Fitting defaults", "Prediction defaults",
             "Plots and export", "Performance", "Logging", "Keyboard")
    _ICONS = ("adjustments-horizontal", "folders", "player-play", "wand", "chart-line",
              "settings", "file-text", "list-details")

    def __init__(self, preferences: Preferences, *, shortcuts: Sequence[tuple[str, str]] = (),
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(820, 520)
        self._start = preferences
        root = QVBoxLayout(self)
        body = QHBoxLayout()
        self.pages = QListWidget()
        self.pages.setFixedWidth(190)
        self.stack = QStackedWidget()
        body.addWidget(self.pages)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)
        self._build(preferences, list(shortcuts))
        for name, icon_name in zip(self.PAGES, self._ICONS):
            self.pages.addItem(QListWidgetItem(icon(icon_name), name))
        self.pages.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.pages.setCurrentRow(0)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel
                                   | QDialogButtonBox.StandardButton.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        restore = buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        if restore is not None:
            restore.clicked.connect(lambda: self.set_values(Preferences()))
        root.addWidget(buttons)

    # -- pages ----------------------------------------------------------------------

    def _page(self, title: str) -> QFormLayout:
        page = QWidget()
        form = QFormLayout(page)
        heading = QLabel(title)
        heading.setObjectName("EmptyTitle")
        form.addRow(heading)
        self.stack.addWidget(page)
        return form

    def _note(self, form: QFormLayout, text: str) -> None:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(secondary_text_style())
        form.addRow("", label)

    def _build(self, p: Preferences, shortcuts: list[tuple[str, str]]) -> None:
        form = self._page("Appearance")
        self.theme = QComboBox()
        for label, value in (("Dark", "dark"), ("Light", "light"),
                             ("Follow the system", "system")):
            self.theme.addItem(label, value)
        form.addRow("Theme:", self.theme)
        self.scale = QComboBox()
        for scale in SCALES:
            self.scale.addItem(f"{scale:.0%}".replace("%", " %"), scale)
        form.addRow("Interface scale:", self.scale)
        self._note(form, "For screens smaller than 1536 × 864 choose 90 % or 80 %. "
                         "The scale takes effect at the next start.")
        swatches = QHBoxLayout()
        for colors in (SERIES[True], HEAT_STOPS):
            for color in colors:
                swatch = QLabel()
                swatch.setFixedSize(20, 12)
                swatch.setAutoFillBackground(True)
                pal = swatch.palette()
                pal.setColor(QPalette.ColorRole.Window, QColor(color))
                swatch.setPalette(pal)
                swatches.addWidget(swatch)
            swatches.addSpacing(12)
        swatches.addStretch(1)
        form.addRow("Colors:", swatches)
        self._note(form, f"Data: six cool colors; values: {value_colormap().name}; "
                         "maps of distances: the icon's heat scale.")
        self.raise_checks = QCheckBox("Bring Checks to the front when a new check opens")
        self.plot_help = QCheckBox('Show the "?" help on plots')
        for box in (self.raise_checks, self.plot_help):
            form.addRow("", box)

        form = self._page("Data and paths")
        self.project_folder = QLineEdit()
        self.project_folder.setPlaceholderText("not set: the file dialog's own choice")
        self.dataset_folder = QLineEdit()
        self.dataset_folder.setPlaceholderText("not set: the file dialog's own choice")
        form.addRow("Projects:", self.project_folder)
        form.addRow("Datasets:", self.dataset_folder)
        self._note(form, "The folders Open, New Project and Add Dataset start in.")
        levels = QHBoxLayout()
        self.black_level = QDoubleSpinBox()
        self.white_level = QDoubleSpinBox()
        for label, spin in (("Black:", self.black_level), ("White:", self.white_level)):
            spin.setRange(0.0, 65535.0)
            spin.setDecimals(1)
            levels.addWidget(QLabel(label))
            levels.addWidget(spin)
        levels.addStretch(1)
        form.addRow("Detector levels:", levels)
        self._note(form, "The levels Add Dataset starts with for a camera that corrects its "
                         "counts with its own dark and white reference (128 and 3100: the EVK "
                         "Helios white balance). A file that states its levels keeps them.")
        self.drive_mapping = QLabel(drive_mapping_text())
        self.drive_mapping.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Drive mapping:", self.drive_mapping)
        self._note(form, "A dataset's file is looked for next to the project, at its stored "
                         "path, and at that path mapped between a Windows drive and its WSL "
                         "mount. The mapping is automatic; the WSL mount root comes from "
                         "/etc/wsl.conf ([automount] root).")

        form = self._page("Fitting defaults")
        self.model_type = QComboBox()
        self.model_type.addItems(["PLS", "PCR"])
        self.n_components = QSpinBox()
        self.n_components.setRange(1, 128)  # as the Fit settings
        self.auto_components = QCheckBox("Choose the number by the rule (one SE)")
        self.hold_out = QComboBox()
        for label, value in (("Each ROI", "roi"), ("ROIs with the same Sample label", "sample"),
                             ("ROIs with the same target value", "target")):
            self.hold_out.addItem(label, value)
        self.x_scaling, self.y_scaling = QComboBox(), QComboBox()
        for combo in (self.x_scaling, self.y_scaling):
            for label, value in (("Center", "center"), ("Unit variance", "uv"),
                                 ("Pareto", "pareto")):
                combo.addItem(label, value)
        form.addRow("Model:", self.model_type)
        form.addRow("Components:", self.n_components)
        form.addRow("", self.auto_components)
        form.addRow("Hold out together:", self.hold_out)
        form.addRow("X scaling:", self.x_scaling)
        form.addRow("Y scaling:", self.y_scaling)
        self._note(form, "These fill the Fit settings when the viewer starts; a model "
                         "stores what it was fitted with, whatever the defaults.")

        form = self._page("Prediction defaults")
        self.exclude_dmodx = QCheckBox("Exclude pixels above the DModXPS limit")
        self.dmodx_level = QDoubleSpinBox()
        self.dmodx_level.setRange(0.5, 0.999)
        self.dmodx_level.setDecimals(3)
        self.dmodx_level.setSingleStep(0.01)
        form.addRow("", self.exclude_dmodx)
        form.addRow("Level:", self.dmodx_level)

        form = self._page("Plots and export")
        self.export_dpi = QSpinBox()
        self.export_dpi.setRange(72, 1200)
        self.export_dpi.setSuffix(" dpi")
        form.addRow("PNG resolution:", self.export_dpi)
        self._note(form, "Exported images always use the light colors.")

        form = self._page("Performance")
        self.blas_threads = QSpinBox()
        self.blas_threads.setRange(0, 256)
        self.blas_threads.setSpecialValueText(f"automatic ({max(1, (os.cpu_count() or 2) // 2)})")
        form.addRow("Threads of the fit:", self.blas_threads)
        self._note(form, f"The numerical library's threads for fits and band proposals. "
                         f"${BLAS_THREADS_ENV}, when set, wins.")

        form = self._page("Logging")
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING"])
        form.addRow("Log file level:", self.log_level)
        open_log = QPushButton("Open Log File")
        open_log.clicked.connect(self._open_log)
        form.addRow("", open_log)
        self._note(form, "The level takes effect at the next start. The log file is what "
                         "to send with a problem report.")

        form = self._page("Keyboard")
        form.addRow(shortcut_table(shortcuts))
        self.set_values(p)

    def _open_log(self) -> None:
        from .._logging import log_file_path
        from ._ui_helpers import reveal_in_file_manager

        path = log_file_path()
        if path is not None:
            reveal_in_file_manager(path)

    # -- values ---------------------------------------------------------------------

    def set_values(self, p: Preferences) -> None:
        """Show ``p`` in every page."""
        def select(combo: QComboBox, value: Any) -> None:
            index = combo.findData(value)
            if index < 0:
                index = combo.findText(str(value))
            combo.setCurrentIndex(max(0, index))

        select(self.theme, p.theme)
        select(self.scale, p.interface_scale)
        self.raise_checks.setChecked(p.raise_checks)
        self.plot_help.setChecked(p.plot_help)
        self.project_folder.setText(p.project_folder)
        self.dataset_folder.setText(p.dataset_folder)
        select(self.model_type, p.model_type)
        self.n_components.setValue(p.n_components)
        self.auto_components.setChecked(p.auto_components)
        select(self.hold_out, p.hold_out)
        select(self.x_scaling, p.x_scaling)
        select(self.y_scaling, p.y_scaling)
        self.exclude_dmodx.setChecked(p.exclude_dmodx)
        self.dmodx_level.setValue(p.dmodx_level)
        self.export_dpi.setValue(p.export_dpi)
        self.blas_threads.setValue(p.blas_threads)
        select(self.log_level, p.log_level)
        self.black_level.setValue(p.black_level)
        self.white_level.setValue(p.white_level)

    def values(self) -> Preferences:
        """The preferences the dialog shows."""
        return Preferences(
            theme=str(self.theme.currentData()),
            interface_scale=float(self.scale.currentData()),
            raise_checks=self.raise_checks.isChecked(),
            plot_help=self.plot_help.isChecked(),
            project_folder=self.project_folder.text().strip(),
            dataset_folder=self.dataset_folder.text().strip(),
            model_type=self.model_type.currentText(),
            n_components=self.n_components.value(),
            auto_components=self.auto_components.isChecked(),
            hold_out=str(self.hold_out.currentData()),
            x_scaling=str(self.x_scaling.currentData()),
            y_scaling=str(self.y_scaling.currentData()),
            exclude_dmodx=self.exclude_dmodx.isChecked(),
            dmodx_level=self.dmodx_level.value(),
            export_dpi=self.export_dpi.value(),
            blas_threads=self.blas_threads.value(),
            log_level=self.log_level.currentText(),
            black_level=self.black_level.value(),
            white_level=self.white_level.value(),
        )


__all__ = ["Preferences", "PreferencesDialog", "apply_environment", "load", "save",
           "start_folder", "startup_scale"]
