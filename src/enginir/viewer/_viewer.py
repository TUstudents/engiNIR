"""HSIViewer QMainWindow shell for Explore, Model, and Predict workspaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar
from collections import Counter
import os
import time

import numpy as np
from PyQt6 import sip
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QMainWindow,
    QWidget,
    QDockWidget,
    QFormLayout,
    QGroupBox,
    QInputDialog,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
    QPlainTextEdit,
    QSizePolicy,
    QFrame,
    QLabel,
    QLayout,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QHBoxLayout,
    QPushButton,
    QToolButton,
    QTabWidget,
    QAbstractButton,
    QAbstractItemView,
    QMenu,
    QMenuBar,
)
from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (QAction, QActionGroup, QCloseEvent, QDesktopServices,
                         QFontDatabase, QKeySequence, QShortcut)

from .._checks import log_and_raise
from .._workflow import own_reference_frames, read_header
from ..project import ReferenceCorrection, viewable_signal_units
from .._logging import add_log_sink, get_logger, log_file_path, remove_log_sink
from ..io import load_spectra

from ._eda import EDAView
from ._attachment import confirm_project_dataset_binding
from ._browser import ProjectBrowser
from ..io import file_format
from ..io import is_band_numbers
from ..prediction import spectral_axis_in_nm
from ._inspector import ProjectInspector, binning_text
from ._data_files import AddDataFileDialog, SetWavelengthsDialog
from ._project_rois import ProjectRoisView
from ._roi_table import RoiTableWidget
from ._bottom_panels import ChecksPanel, ModelsPanel, verdict_mark
from ._icons import icon
from ._workflow_strip import STEPS, WorkflowStrip
from . import _preferences as preferences_store
from ._preferences import Preferences, PreferencesDialog
from .._guidance import Check, Fix, prediction_roi_checks
from ._jobs import start_background_job, start_background_job_soon
from ._loading import ViewerBusyContext
from ._model_diagnostics import ModelDiagnosticsWidget
from . import _layout_settings as layout_settings
from ._project_settings import load_last_data_file, save_last_data_file
from ._plot_grid import PlotPage, PlotPageWidget, PlotSlot
from ._plot_panel import LARGE_EXPORT_VALUES, PlotFrame, PlotPanel
from ._modeling import ModelingView, NewROIDialog, TargetSelectionDialog
from ._predict_dialogs import ROI_RESULTS
from ._prediction import PREDICTION_HISTOGRAM, PredictionMapWidget, PredictionView
from ._windows import keep_open
from ._state import (
    DEFAULT_SOURCE_MODALITY,
    ProjectState,
    ViewerState,
    look_up_data_files,
    signal_label,
)
from ._panels import (
    _PanelBase as _ExplorePanel,
    HistogramPanel,
    SensorPanel,
    SpatialProfilePanel,
    SpectrumPanel,
)
from ._theme import apply_qt_theme, is_dark, panel_colors, secondary_text_style
from ._ui_helpers import reveal_in_file_manager, wrap_tooltip

logger = get_logger(__name__)
_QtObject = TypeVar("_QtObject")
_COMPACT_LEFT_DOCK_WIDTH = 260
_CONTROLS_DOCK_TARGET_HEIGHT = 230
#: Width of the Settings dock and height of the bottom panel per screen-size
#: class: "full" from 1800 px wide, "compact" below.
_SETTINGS_DOCK_WIDTH = {"full": 340, "compact": 304}
_BOTTOM_PANEL_HEIGHT = {"full": 200, "compact": 160}
#: The share of the window's height the bottom panel takes while the ROI table is its
#: front tab: a table of all ROIs needs more rows than a list of checks.
_ROI_TABLE_HEIGHT_SHARE = 0.42
_FULL_WIDTH = 1800
#: Settings dock title per workspace.
_SETTINGS_TITLES = ("View settings", "Fit settings", "Prediction settings")
_SPECTRUM_PANEL_HEIGHT = 220
_UNLIMITED_HEIGHT = 16777215  # QWIDGETSIZE_MAX
_SAVED_MESSAGE_MS = 8000
_ADD_DATA_FILE = "__add_data_file__"


def _default_levels() -> tuple[float, float]:
    """The detector levels of Tools > Preferences (default 128 / 3100)."""
    preferences = preferences_store.load()
    return float(preferences.black_level), float(preferences.white_level)


def _reference_correction_view(reference_correction: ReferenceCorrection | None,
                      meta: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """``ViewerState.load_dataset`` keywords and the signal unit for a data file's reference correction."""
    black, white = _default_levels()
    levels: dict[str, Any] = {
        "black_level": float(meta.get("black_level", black)),
        "white_level": float(meta.get("white_level", white)),
    }
    if reference_correction is None:
        return levels, "DN"
    if reference_correction.dark_ref_data is not None and reference_correction.white_ref_data is not None:
        levels["dark_ref"] = np.asarray(reference_correction.dark_ref_data, dtype=np.float32)
        levels["white_ref"] = np.asarray(reference_correction.white_ref_data, dtype=np.float32)
    elif reference_correction.black_level is not None and reference_correction.white_level is not None:
        levels["black_level"] = float(reference_correction.black_level)
        levels["white_level"] = float(reference_correction.white_level)
    return levels, reference_correction.output_signal_unit


def _require_qt_object(value: _QtObject | None, context: str) -> _QtObject:
    """Return a Qt object that PyQt stubs mark optional, or fail with telemetry."""
    if value is None:
        log_and_raise("Qt did not create {}", context, error=RuntimeError)
    return value


def _add_required_menu(parent: QMenuBar | QMenu, title: str) -> QMenu:
    return _require_qt_object(parent.addMenu(title), f"{title} menu")


def _add_required_action(menu: QMenu, text: str) -> QAction:
    return _require_qt_object(menu.addAction(text), f"{text} action")


__all__ = [
    "HSIViewer",
    "LogPanel",
    "NewROIDialog",
    "TargetSelectionDialog",
]


# ------------------------------------------------------------------
# LogPanel - dock widget for loguru output
# ------------------------------------------------------------------


#: Where engiNIR lives, and its concept DOI (every version; README, Citation).
GITHUB_URL = "https://github.com/TUstudents/engiNIR"
#: The online documentation (Help > User Guide).
USER_GUIDE_URL = "https://enginir.readthedocs.io/en/stable/"
#: Where a problem is reported (Help > Report a Problem).
ISSUES_URL = f"{GITHUB_URL}/issues/new"
CONCEPT_DOI = "10.5281/zenodo.23235907"


def about_text(version: str) -> str:
    """The About box: version, tagline, the citation of the README and the links."""
    doi = f"https://doi.org/{CONCEPT_DOI}"
    return (f"<b>engiNIR</b> {version}<br>"
            "A universal engineering tool for near-infrared spectra and hyperspectral "
            "imaging.<br><br>"
            "If you use engiNIR in your work, please cite:<br>"
            "Johannes Poms. (2026). TUstudents/engiNIR: A universal engineering tool for "
            f"near-infrared spectra and hyperspectral imaging. Zenodo. "
            f'<a href="{doi}">{doi}</a><br><br>'
            f'Source, issues and releases: <a href="{GITHUB_URL}">{GITHUB_URL}</a><br>'
            "Licensed under CC BY 4.0.")


def _shown_axis(dataset: Any, data: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """The axis and metadata a registered file is shown with: the dataset's stored axis.

    The same as the file's, except for a file without wavelengths that a
    wavelength table gave them (``dataset.wavelength_table``).
    """
    if not getattr(dataset, "wavelength_table", ""):
        return data.wavelengths, data.meta
    return (np.asarray(dataset.wavelengths), {**data.meta, "x_units": "nm",
                                               "wavelength_table": dataset.wavelength_table})


def _button_of(button: QAbstractButton | Callable[[], QAbstractButton]) -> QAbstractButton:
    """The button itself, or the one a function returns (a page built on first use)."""
    return button if isinstance(button, QAbstractButton) else button()


def _narrow_combos(widget: QWidget) -> None:
    """Long choices ("Auto (7 round-robin folds, ...)") must not widen the settings column."""
    for combo in widget.findChildren(QComboBox):
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(8)


def _apply_fit_defaults(form: Any, preferences: Any) -> None:
    """The user's defaults (Tools > Preferences) in a new Fit settings form."""
    form._model_type_combo.setCurrentText(preferences.model_type)
    form._components_spin.setValue(preferences.n_components)
    form._auto_check.setChecked(preferences.auto_components)
    for combo, value in ((form._group_by_combo, preferences.hold_out),
                         (form._x_scaling_combo, preferences.x_scaling),
                         (form._y_scaling_combo, preferences.y_scaling)):
        if combo.findData(value) >= 0:
            combo.setCurrentIndex(combo.findData(value))


def _apply_predict_defaults(page: Any, preferences: Any) -> None:
    """The user's defaults (Tools > Preferences) in a new Predict page."""
    page._exclusion_level.setValue(preferences.dmodx_level)
    page._exclusion_check.setChecked(preferences.exclude_dmodx)


class _ColumnScrollArea(QScrollArea):
    """Scrolls up and down only: the content always gets the viewport's width.

    The Settings dock's groups are narrow but tall; the content is held to the
    dock's width so rows wrap (flow layouts) and drop-downs shrink instead of
    running out of the dock.
    """

    def resizeEvent(self, a0: Any) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(a0)
        content, viewport = self.widget(), self.viewport()
        if content is not None and viewport is not None:
            content.setFixedWidth(viewport.width())


class LogPanel(QWidget):
    """Scrollable read-only log viewer.  Receives loguru messages via a
    custom sink connected through a thread-safe Qt signal."""

    _append = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(1000)
        self._font_set = False  # the fixed-width font comes with the first look (use_font)
        self.restyle_theme()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(80)

        self._append.connect(self._text.appendPlainText)

    def use_font(self) -> None:
        """Set the fixed-width font, once: when the Log tab is first selected.

        On Windows the first font family other than the UI font makes Qt load its
        whole font database (0.3 s warm, 0.94 s in a user's log, 2026-11-19); the
        Log tab is behind the Checks tab when a window opens. Not on show: Qt
        shows the hidden tabs of a dock area too (off-screen). The system's
        fixed-width font, since a family name such as "Monospace" exists on Linux
        only.
        """
        if not self._font_set:
            self._font_set = True
            font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
            font.setPointSize(9)
            self._text.setFont(font)

    def sink(self, message) -> None:
        """Loguru sink safe to call from any thread."""
        try:
            self._append.emit(str(message).rstrip())
        except RuntimeError:
            # Qt may destroy the widget before the sink is removed.
            pass

    def restyle_theme(self) -> None:
        """Apply the current theme to the log panel surface."""
        colors = panel_colors()
        self._text.setStyleSheet(
            "QPlainTextEdit { "
            f"background: {colors['ax_bg']}; "
            f"color: {colors['fg']}; "
            "border-radius: 8px; "
            "padding: 6px; "
            "}"
        )


class HSIViewer(QMainWindow):
    """5-panel interactive HSI viewer built on QMainWindow + pyqtgraph.

    Parameters
    ----------
    cube:
        Hyperspectral image cube (n_frames, height, n_bands), uint16.
    wavelengths:
        Wavelength axis in nm, shape (n_bands,).
    meta:
        Metadata dict from the file loader (optional).
    dark:
        Optional dark-reference cube or image.  If 3-D, the mean across
        axis 0 is taken.
    white:
        Optional white-reference cube or image.  Same convention as *dark*.
    """

    #: The checks and verdicts were recomputed.
    checks_refreshed = pyqtSignal()

    def __init__(
        self,
        cube: np.ndarray,
        wavelengths: np.ndarray,
        meta: dict | None = None,
        *,
        dark: np.ndarray | None = None,
        white: np.ndarray | None = None,
    ) -> None:
        init_started = time.perf_counter()
        step_started = init_started

        def _mark_init_step(label: str) -> None:
            nonlocal step_started
            now = time.perf_counter()
            logger.debug(
                "HSIViewer init step '{}': {:.3f}s ({:.3f}s cumulative)",
                label,
                now - step_started,
                now - init_started,
            )
            step_started = now

        super().__init__()
        self.setWindowTitle("engiNIR - HSI Viewer")

        _mark_init_step("window")

        _meta = meta or {}

        default_black, default_white = _default_levels()
        black_level: float = float(_meta.get("black_level", default_black))
        white_level: float = float(_meta.get("white_level", default_white))

        self.state = ViewerState(
            cube,
            wavelengths,
            _meta,
            black_level=black_level,
            white_level=white_level,
        )
        _mark_init_step("state")

        self.project_state = ProjectState()
        # Deleted by Qt when it closes; the registry keeps it until then and drops its
        # data afterwards (viewer-workspace-ui contract, Window lifetime).
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        keep_open(self, on_deleted=self.state.release_data)
        self._cube_path: str | None = None
        self._workspace_refresh_pending = False
        self._bottom_dock_resize_pending = False
        self._controls_dock_height = _CONTROLS_DOCK_TARGET_HEIGHT

        if dark is not None:
            ref = dark.mean(axis=0) if dark.ndim == 3 else dark
            self.state.set_dark_ref(ref.astype(np.float32, copy=False))
        if white is not None:
            ref = white.mean(axis=0) if white.ndim == 3 else white
            self.state.set_white_ref(ref.astype(np.float32, copy=False))
        _mark_init_step("references")

        self._log_sink_id: int | None = None
        self._build_canvas()
        _mark_init_step("canvas")
        self._build_dock_widgets()
        _mark_init_step("docks")
        self._build_controls_panel()
        _mark_init_step("controls")
        self._build_log_panel()
        _mark_init_step("log")
        self._wire_signals()
        _mark_init_step("signals")
        self._build_menu_bar()
        self._build_workflow_strip()
        _mark_init_step("menu")
        self.refresh_theme()
        _mark_init_step("theme")
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)
        logger.info(
            "HSIViewer ready: {} frames x {} rows x {} bands",
            self.state.n_frames,
            self.state.height,
            self.state.n_bands,
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_canvas(self) -> None:
        """Create the named central image view with a reset control."""
        from ._canvas import FalseColorCanvas

        self._canvas = FalseColorCanvas()
        self._spectrum = SpectrumPanel(self.state)
        self._data_file_page = QWidget()
        central_layout = QVBoxLayout(self._data_file_page)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        self._canvas_frame = PlotFrame("explore.image", "Spatial image", self._canvas,
                                       self._canvas, provenance=self._explore_provenance)
        self._data_file_job: Any = None
        self._data_file_combo = QComboBox()
        self._data_file_combo.setToolTip(wrap_tooltip("Dataset of the project shown in the image"))
        self._data_file_combo.setMinimumWidth(180)
        self._data_file_combo.activated.connect(self._on_data_file_chosen)
        self._canvas_frame.add_header_widget(self._data_file_combo)
        self._btn_reset_image_view = QToolButton()
        self._btn_reset_image_view.setText("Reset View")
        self._btn_reset_image_view.setAutoRaise(True)
        self._btn_reset_image_view.clicked.connect(self._reset_image_view)
        self._canvas_frame.add_header_widget(self._btn_reset_image_view)
        self._canvas_frame.maximize_toggled.connect(
            lambda: self._toggle_explore_maximized("image"))
        # The rows of the scan, read across: the Sensor frame (bands x rows) and the
        # Spatial profile (value x row) beside the image share its row axis, so one
        # row lines up in all three (user decision 2026-11-16; they were docks
        # between the page and the settings before).
        self._sensor = SensorPanel(self.state)
        self._spatial = SpatialProfilePanel(self.state)
        self._sensor.link_rows(self._canvas.view_box)
        self._spatial.link_rows(self._canvas.view_box)
        for panel in (self._sensor, self._spatial):
            # As small as the image: the page, not a dock minimum, sets their height,
            # so the bottom panel can still grow for the ROI table.
            panel.setMinimumSize(120, 60)
        self._sensor_frame = self._explore_frame(self._sensor, "explore.sensor")
        self._spatial_frame = self._explore_frame(self._spatial, "explore.spatial")
        self._sensor_frame.maximize_toggled.connect(
            lambda: self._toggle_explore_maximized("sensor"))
        self._spatial_frame.maximize_toggled.connect(
            lambda: self._toggle_explore_maximized("spatial"))
        self._sensor_frame.add_close_button(
            lambda: self._on_page_plot_visibility_action("sensor", False))
        self._spatial_frame.add_close_button(
            lambda: self._on_page_plot_visibility_action("spatial", False))
        self._rows_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._rows_splitter.setObjectName("RowsSplitter")
        self._rows_splitter.setChildrenCollapsible(False)
        for index, (frame, stretch) in enumerate(((self._canvas_frame, 5),
                                                  (self._sensor_frame, 2),
                                                  (self._spatial_frame, 1))):
            self._rows_splitter.addWidget(frame)
            self._rows_splitter.setStretchFactor(index, stretch)
        # The profile's header (title and three buttons) must fit: never under 180 px.
        self._spatial_frame.setMinimumWidth(180)
        self._rows_splitter.setSizes([600, 260, 180])
        central_layout.addWidget(self._rows_splitter, 1)

        self._spectrum_panel = self._explore_frame(self._spectrum, "explore.spectrum")
        self._spectrum_panel.setObjectName("SpectrumPanelContainer")
        self._spectrum_panel.maximize_toggled.connect(
            lambda: self._toggle_explore_maximized("spectrum"))
        central_layout.addWidget(self._spectrum_panel)
        self._spectrum_panel.setFixedHeight(_SPECTRUM_PANEL_HEIGHT)
        self._explore_maximized: str | None = None
        self._explore_escape = QShortcut(QKeySequence(Qt.Key.Key_Escape),
                                         self._data_file_page)
        self._explore_escape.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._explore_escape.activated.connect(self._on_explore_escape)
        self._explore_escape.setEnabled(False)

        self._spectrum_toggle_action = QAction("Spectral Profile", self)
        self._spectrum_toggle_action.setCheckable(True)
        self._spectrum_toggle_action.setChecked(True)
        self._spectrum_toggle_action.triggered.connect(
            self._on_spectrum_panel_visibility_action
        )
        #: View menu switches of the Dataset page's plots (spectrum, sensor, spatial).
        self._page_plot_actions: dict[str, QAction] = {"spectrum": self._spectrum_toggle_action}
        for name, title in (("sensor", "Sensor Frame"), ("spatial", "Spatial Profile")):
            action = QAction(title, self)
            action.setCheckable(True)
            action.setChecked(True)
            action.triggered.connect(
                lambda visible, name=name: self._on_page_plot_visibility_action(name, visible))
            self._page_plot_actions[name] = action

        self._model_diagnostics_widget = None
        self._predict_map_widget = None
        self._model_content_page = self._build_deferred_workspace_page("Model diagnostics")
        self._predict_content_page = self._build_deferred_workspace_page("Prediction results")

        self._workspace_content_stack = QStackedWidget()
        self._workspace_content_stack.setObjectName("WorkspaceContentStack")
        # Explore: the shown data file, or all ROIs of the project (filled in later).
        self._explore_tabs = QTabWidget()
        self._explore_tabs.setDocumentMode(True)
        _require_qt_object(self._explore_tabs.tabBar(), "Explore tab bar").setDrawBase(False)
        self._explore_tabs.addTab(self._data_file_page, "Dataset")
        self._explore_content_page = self._explore_tabs
        self._project_rois: ProjectRoisView | None = None
        self._pending_roi_selection: str | None = None
        self._workspace_content_stack.addWidget(self._explore_content_page)
        self._workspace_content_stack.addWidget(self._model_content_page)
        self._workspace_content_stack.addWidget(self._predict_content_page)
        # No bar above the pages: the Checks tab says what a step needs, and a bar
        # repeating its first line only cost height (user decision, 2026-11-16).
        self.setCentralWidget(self._workspace_content_stack)

        self._refresh_canvas_image()

    @staticmethod
    def _build_deferred_workspace_page(title: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        label = QLabel(title)
        label.setStyleSheet("font-weight: 600;")
        hint = QLabel("Preparing workspace...")
        hint.setStyleSheet(secondary_text_style())
        layout.addWidget(label)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _replace_workspace_content_page(self, old_page: QWidget, new_page: QWidget) -> None:
        index = self._workspace_content_stack.indexOf(old_page)
        if index < 0:
            return
        self._workspace_content_stack.removeWidget(old_page)
        self._workspace_content_stack.insertWidget(index, new_page)
        old_page.deleteLater()

    def _apply_plot_help(self) -> None:
        """Show the plots' help buttons as the preferences say (also on pages built later)."""
        preferences = getattr(self, "_preferences", None)
        if preferences is None:
            return
        for frame in self.findChildren(PlotFrame):
            button = getattr(frame, "_help_button", None)
            if button is not None:
                button.setVisible(preferences.plot_help)

    def _ensure_model_content_page(self) -> None:
        if self._model_diagnostics_widget is not None:
            return
        old_page = self._model_content_page
        self._model_content_page = self._build_model_content_page()
        self._replace_workspace_content_page(old_page, self._model_content_page)
        self._apply_plot_help()
        if self._model_diagnostics_widget is not None:
            self._model_diagnostics_widget.roi_selected.connect(self._on_canvas_roi_selected)
            grid = self._model_diagnostics_widget.grid
            saved = layout_settings.load_maximized().get("Model")
            if saved in grid.plot_ids:
                grid.maximize(saved)
            grid.maximized_changed.connect(
                lambda plot_id: layout_settings.save_maximized("Model", plot_id))

    def _ensure_predict_content_page(self) -> None:
        if self._predict_map_widget is not None:
            return
        old_page = self._predict_content_page
        self._predict_content_page = self._build_predict_content_page()
        self._replace_workspace_content_page(old_page, self._predict_content_page)
        self._apply_plot_help()
        saved = layout_settings.load_maximized().get("Predict")
        if saved in self._predict_graph_panel.panels:
            self._predict_graph_panel.maximize(saved)
        self._predict_graph_panel.maximized_changed.connect(
            lambda plot_id: layout_settings.save_maximized("Predict", plot_id))

    def _build_model_content_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._model_content_title = QLabel("Model Diagnostics")
        self._model_workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._model_workspace_splitter.setChildrenCollapsible(False)

        self._model_diagnostics_widget = ModelDiagnosticsWidget()

        self._model_text_panel = QWidget()
        self._model_text_panel.setObjectName("ModelDiagnosticsTextPanel")
        self._model_text_panel.setMinimumWidth(260)
        self._model_text_panel.setMaximumWidth(360)
        text_layout = QVBoxLayout(self._model_text_panel)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(8)

        summary_box = QGroupBox("Fit summary")
        summary_layout = QFormLayout(summary_box)
        summary_layout.setContentsMargins(8, 8, 8, 8)
        summary_layout.setSpacing(4)

        self._model_card_target_value = QLabel("none")
        self._model_card_sample_set_value = QLabel("none")
        self._model_card_model_value = QLabel("none")
        self._model_card_components_value = QLabel("-")
        self._model_card_cv_value = QLabel("-")
        self._model_card_signal_value = QLabel("-")
        self._model_card_roi_count_value = QLabel("0")
        self._model_card_dataset_count_value = QLabel("0")
        self._model_card_pixel_count_value = QLabel("0")

        for label in (self._model_card_target_value, self._model_card_sample_set_value,
                      self._model_card_model_value, self._model_card_cv_value,
                      self._model_card_signal_value):
            label.setWordWrap(True)  # long values wrap instead of being cut off
        # The library's verdict of the model shown (the Checks tab has the reasons).
        verdict_row = QWidget()
        verdict_layout = QHBoxLayout(verdict_row)
        verdict_layout.setContentsMargins(0, 0, 0, 0)
        verdict_layout.setSpacing(5)
        self._model_verdict_icon = QLabel()
        self._model_verdict_value = QLabel("–")
        self._model_verdict_value.setObjectName("VerdictText")
        verdict_layout.addWidget(self._model_verdict_icon)
        verdict_layout.addWidget(self._model_verdict_value, 1)
        self._model_verdict_level = ""
        summary_layout.addRow("Verdict", verdict_row)
        summary_layout.addRow("Target", self._model_card_target_value)
        summary_layout.addRow("Calibration set", self._model_card_sample_set_value)
        summary_layout.addRow("Model", self._model_card_model_value)
        summary_layout.addRow("Components", self._model_card_components_value)
        summary_layout.addRow("Validation", self._model_card_cv_value)
        summary_layout.addRow("Signal, preprocessing", self._model_card_signal_value)
        summary_layout.addRow("ROIs", self._model_card_roi_count_value)
        summary_layout.addRow("Datasets", self._model_card_dataset_count_value)
        summary_layout.addRow("Pixels", self._model_card_pixel_count_value)

        # Two levels (model-workspace-diagnostics contract): ROI means, single pixels.
        metrics_box = QGroupBox("Metrics per ROI")
        metrics_box.setToolTip(wrap_tooltip("Over ROI means: the model's quality. The target, the folds "
                               "and the reported metrics are per ROI."))
        self._model_metrics_layout = QFormLayout(metrics_box)
        self._model_metrics_layout.setContentsMargins(8, 8, 8, 8)
        self._model_metrics_layout.setSpacing(4)
        pixel_box = QGroupBox("Metrics per pixel")
        pixel_box.setToolTip(wrap_tooltip("Over single pixels (what the model is fitted on). Single "
                             "pixels are noisy, so these are worse than per ROI."))
        self._model_pixel_metrics_layout = QFormLayout(pixel_box)
        self._model_pixel_metrics_layout.setContentsMargins(8, 8, 8, 8)
        self._model_pixel_metrics_layout.setSpacing(4)
        self._model_pixel_metrics_box = pixel_box
        self._model_metric_r2_value = QLabel("-")
        self._model_metric_rmsecv_value = QLabel("-")
        self._model_metric_bias_value = QLabel("-")
        self._model_metric_rows: list[str] = []

        layout.addWidget(self._model_content_title)
        text_layout.addWidget(summary_box)
        text_layout.addWidget(metrics_box)
        text_layout.addWidget(pixel_box)
        text_layout.addStretch(1)

        self._model_workspace_splitter.addWidget(self._model_diagnostics_widget)
        self._model_workspace_splitter.addWidget(self._model_text_panel)
        self._model_workspace_splitter.setStretchFactor(0, 5)
        self._model_workspace_splitter.setStretchFactor(1, 1)
        self._model_workspace_splitter.setSizes([900, 300])
        layout.addWidget(self._model_workspace_splitter, 1)
        return page

    def _build_predict_content_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._predict_content_title = QLabel("Prediction Results")
        # One headline names what is shown (data file, model); the groups below never
        # repeat it. Progress is in the Controls dock.
        self._predict_headline = QLabel("No prediction yet.")
        self._predict_headline.setWordWrap(True)
        self._predict_workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._predict_workspace_splitter.setChildrenCollapsible(False)

        self._predict_map_widget = PredictionMapWidget()
        self._predict_map_frame = PlotFrame(
            "predict.map", self._predict_map_widget.plot_title, self._predict_map_widget,
            self._predict_map_widget, provenance=self._prediction_view.workspace_provenance)
        for control in self._predict_map_widget.header_widgets():
            self._predict_map_frame.add_header_widget(control)
        self._predict_histogram = PlotPanel(PREDICTION_HISTOGRAM)
        self._predict_roi_results = PlotPanel(ROI_RESULTS)
        self._predict_roi_results.roi_selected.connect(self._on_canvas_roi_selected)
        self._predict_map_frame.add_menu_action("Color Limits...", self._ask_colour_limits)
        self._predict_shown_map: dict[str, Any] | None = None
        # Hovering a map pixel or a tablet's dot fills the window's status line.
        self._predict_map_widget.hovered.connect(self._show_hover_status)
        self._predict_roi_results.roi_hovered.connect(
            lambda roi_id: self._show_hover_status(self._prediction_view.roi_hover_text(roi_id)))
        self._predict_map_widget.displayed_map_changed.connect(self._on_predict_map_shown)
        self._prediction_view.display_changed.connect(self._on_predict_display_changed)
        self._predict_graph_panel = PlotPageWidget(
            PlotPage("Prediction", (PlotSlot("predict.map", 0, 0, row_span=2, column_span=3),
                                    PlotSlot("predict.roi_results", 2, 0, column_span=2),
                                    PlotSlot("predict.histogram", 2, 2))),
            {"predict.map": self._predict_map_frame,
             "predict.roi_results": self._predict_roi_results,
             "predict.histogram": self._predict_histogram})

        self._predict_text_panel = QWidget()
        self._predict_text_panel.setObjectName("PredictionResultsTextPanel")
        self._predict_text_panel.setMinimumWidth(260)
        self._predict_text_panel.setMaximumWidth(360)
        predict_text_layout = QVBoxLayout(self._predict_text_panel)
        predict_text_layout.setContentsMargins(0, 0, 0, 0)
        predict_text_layout.setSpacing(8)

        # Three groups (viewer contract §3): the chain, the scan, the applicability; ROIs.
        def wrapped(text: str = "-") -> QLabel:
            label = QLabel(text)
            label.setWordWrap(True)
            return label

        def group(title: str) -> tuple[QGroupBox, QFormLayout]:
            box = QGroupBox(title)
            form = QFormLayout(box)
            form.setContentsMargins(8, 8, 8, 8)
            form.setSpacing(4)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            return box, form

        summary_box, summary_layout = group("Chain")
        self._predict_contract_signal_value = wrapped()
        self._predict_card_target_value = wrapped("none")
        self._predict_contract_bands_value = wrapped()
        self._predict_contract_format_value = wrapped()
        summary_layout.addRow("Signal, recipe", self._predict_contract_signal_value)
        summary_layout.addRow("Target", self._predict_card_target_value)
        summary_layout.addRow("Bands", self._predict_contract_bands_value)
        summary_layout.addRow("Input", self._predict_contract_format_value)

        compatibility_box, compatibility_layout = group("Scan")
        self._predict_card_correction_value = wrapped("none")
        self._predict_map_shape_value = wrapped()
        self._predict_map_range_value = wrapped()
        self._predict_clipping_value = wrapped()
        compatibility_layout.addRow("Reference correction", self._predict_card_correction_value)
        compatibility_layout.addRow("Map shape", self._predict_map_shape_value)
        compatibility_layout.addRow("Map range", self._predict_map_range_value)
        compatibility_layout.addRow("Clipping", self._predict_clipping_value)

        applicability_box, applicability_layout = group("Applicability")
        self._predict_diagnostics_value = wrapped()
        self._predict_wavelengths_value = wrapped()
        applicability_layout.addRow("Pixels", self._predict_diagnostics_value)
        applicability_layout.addRow("Wavelengths", self._predict_wavelengths_value)

        rois_box, rois_layout = group("Prediction ROIs")
        self._predict_rois_value = wrapped("none")
        rois_layout.addRow(self._predict_rois_value)

        layout.addWidget(self._predict_content_title)
        predict_text_layout.addWidget(self._predict_headline)
        predict_text_layout.addWidget(rois_box)
        predict_text_layout.addWidget(summary_box)
        predict_text_layout.addWidget(compatibility_box)
        predict_text_layout.addWidget(applicability_box)
        predict_text_layout.addStretch(1)

        self._predict_workspace_splitter.addWidget(self._predict_graph_panel)
        self._predict_workspace_splitter.addWidget(self._predict_text_panel)
        self._predict_workspace_splitter.setStretchFactor(0, 5)
        self._predict_workspace_splitter.setStretchFactor(1, 1)
        self._predict_workspace_splitter.setSizes([900, 300])
        # Before the first prediction of the session the page invites one instead of
        # showing empty axes.
        self._predict_page_stack = QStackedWidget()
        self._predict_empty = self._empty_state(
            "wand", "Predict your samples",
            "A model predicts the prediction ROIs from the project file alone, one result "
            "per ROI, with a check that each one is inside the model; or every pixel of a "
            "dataset as a map.",
            (("Predict ROIs...", self._prediction_view._btn_predict_rois),
             ("Predict Map...", self._prediction_view._btn_predict)))
        self._predict_page_stack.addWidget(self._predict_empty)
        self._predict_page_stack.addWidget(self._predict_workspace_splitter)
        layout.addWidget(self._predict_page_stack, 1)
        return page

    @staticmethod
    def _empty_state(icon_name: str, title: str, text: str,
                     buttons: tuple[tuple[str, Any], ...]) -> QWidget:
        """A page with one sentence and the buttons of the next action."""
        page = QWidget()
        page.setObjectName("EmptyPage")
        column = QVBoxLayout(page)
        column.addStretch(1)
        mark = QLabel()
        mark.setPixmap(icon(icon_name, "accent").pixmap(48, 48))
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(mark)
        heading = QLabel(title)
        heading.setObjectName("EmptyTitle")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(heading)
        body = QLabel(text)
        body.setObjectName("EmptyState")
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.setMaximumWidth(520)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(body)
        row.addStretch(1)
        column.addLayout(row)
        actions = QHBoxLayout()
        actions.addStretch(1)
        for index, (label, target) in enumerate(buttons):
            button = QPushButton(label)
            button.setDefault(index == 0)
            button.clicked.connect(lambda _checked=False, target=target: target.click())
            actions.addWidget(button)
        actions.addStretch(1)
        column.addLayout(actions)
        column.addStretch(2)
        return page

    def _on_predict_display_changed(self) -> None:
        """Colour scale, overlays or histogram source changed in the Predict controls."""
        widget = self._predict_map_widget
        if widget is None:
            return
        if widget.user_limits != self._prediction_view.colour_limits:
            widget.user_limits = self._prediction_view.colour_limits
        self.schedule_workspace_content_refresh()

    def _show_hover_status(self, text: str) -> None:
        """A hover readout in the status line; ``""`` restores the idle line."""
        if text:
            self._status_label.setText(text)
        else:
            self._update_status_strip()

    def _on_predict_map_shown(self, shown: dict[str, Any] | None) -> None:
        self._predict_shown_map = shown
        self._update_predict_histogram()

    def _update_predict_histogram(self) -> None:
        """The ROI pixels with ROI results (following the exclusion), else the shown map."""
        if self._predict_map_widget is None:
            return
        self._predict_histogram.update_data(
            self._prediction_view.histogram_payload(self._predict_shown_map),
            provenance=self._prediction_view.workspace_provenance())

    def _reset_image_view(self) -> None:
        """Reset the central image view to the full image bounds."""
        self._canvas.reset_view()

    def _false_colour_rgb(self) -> np.ndarray:
        """Return the current false-color preview image as float RGB."""
        return self.state.raw_false_colour_rgb()

    def _build_dock_widgets(self) -> None:
        s = self.state
        self._hist = HistogramPanel(s)
        self._project_browser = ProjectBrowser(self.project_state)
        self._inspector = ProjectInspector(self.project_state)

        self._project_browser_dock = QDockWidget("Project", self)
        self._project_browser_dock.setObjectName("ProjectBrowserDock")
        self._project_browser_dock.setWidget(self._project_browser)
        self._configure_compact_left_dock(self._project_browser_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._project_browser_dock)

        self._inspector_dock = QDockWidget("Properties", self)
        self._inspector_dock.setObjectName("InspectorDock")
        self._inspector_dock.setWidget(self._inspector)
        self._configure_compact_left_dock(self._inspector_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._inspector_dock)
        self.splitDockWidget(
            self._project_browser_dock,
            self._inspector_dock,
            Qt.Orientation.Vertical,
        )
        # Left docks - Properties above the collapsed Histogram dock
        self._dock_frames: dict[QDockWidget, PlotFrame] = {}
        self._maximized_dock: QDockWidget | None = None
        self._hist_dock = self._plot_dock("Histogram", "HistogramDock", self._hist,
                                          "explore.histogram")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._hist_dock)
        self.splitDockWidget(self._inspector_dock, self._hist_dock, Qt.Orientation.Vertical)

        self._hist_dock.hide()

        # ROI table: target values of all ROIs; available in every workspace.
        self._roi_table = RoiTableWidget(self.project_state)
        self._roi_table.roi_selected.connect(self._on_canvas_roi_selected)
        self._roi_table_dock = QDockWidget("ROI table", self)
        self._roi_table_dock.setObjectName("RoiTableDock")
        self._roi_table_dock.setWidget(self._roi_table)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._roi_table_dock)
        self._roi_table_dock.hide()
        #: ROI table per Explore tab: shown with Project ROIs, not on the Data file tab;
        #: the user's choice on each tab is kept. Model and Predict leave it as it is.
        self._default_roi_table_visibility = {"data_file": True, "project_rois": True}
        self._roi_table_visibility = dict(self._default_roi_table_visibility)
        self._roi_table_dock.visibilityChanged.connect(self._on_roi_table_visibility_changed)
        # Taller while the table is the front tab, back to the panel's height after.
        # (A bound method, not a lambda: Qt disconnects it when the window is destroyed;
        # a lambda was called for a closing window and crashed.)
        self._roi_table_dock.visibilityChanged.connect(self._schedule_bottom_dock_resize)

        #: Docks that show the data file at the cursor: only on Explore's Data file tab.
        self._data_file_docks = {"histogram": self._hist_dock}
        self._default_explore_plot_visibility = {
            "histogram": False,
            "sensor": True,
            "spatial": True,
            "spectrum": True,
        }
        self._explore_plot_visibility = dict(self._default_explore_plot_visibility)

    # ------------------------------------------------------------------
    # Explore plot frames (Plot panel policy)
    # ------------------------------------------------------------------

    def _explore_frame(self, panel: _ExplorePanel, plot_id: str) -> PlotFrame:
        frame = PlotFrame(plot_id, panel.plot_title, panel, panel,
                          provenance=self._explore_provenance)
        return frame

    def _plot_dock(self, title: str, object_name: str, panel: _ExplorePanel,
                   plot_id: str) -> QDockWidget:
        """A dock whose title bar is the plot header (title, maximize, export, close)."""
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        frame = self._explore_frame(panel, plot_id)
        dock.setWidget(frame)
        header = frame.take_header()
        frame.add_close_button(dock.hide)
        dock.setTitleBarWidget(header)
        frame.maximize_toggled.connect(lambda: self._toggle_dock_maximized(dock))
        escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), dock)
        escape.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        escape.activated.connect(lambda: self._restore_dock(dock))
        dock.visibilityChanged.connect(
            lambda visible: None if visible else self._restore_dock(dock))
        self._dock_frames[dock] = frame
        return dock

    def _explore_provenance(self) -> dict[str, str]:
        meta = self.project_state.project_meta
        provenance = {"project_id": str(meta.get("project_id", "none")),
                      "data_file": self._cube_path or "none"}
        dataset_id = self.project_state.dataset_id_of(self._cube_path)
        if dataset_id is not None:
            provenance["dataset_id"] = dataset_id
        return provenance

    def _toggle_dock_maximized(self, dock: QDockWidget) -> None:
        """Float the dock over the main window, or put it back."""
        if self._dock_frames[dock].is_maximized:
            self._restore_dock(dock)
            return
        self._restore_maximized_plots()
        dock.setFeatures(dock.features() | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dock.setFloating(True)
        dock.setGeometry(self.geometry())
        dock.raise_()
        self._dock_frames[dock].set_maximized(True)
        self._maximized_dock = dock

    def _restore_dock(self, dock: QDockWidget) -> None:
        frame = self._dock_frames[dock]
        if not frame.is_maximized:
            return
        dock.setFloating(False)
        frame.set_maximized(False)
        self._maximized_dock = None

    def _toggle_explore_maximized(self, which: str) -> None:
        """Maximize the spatial image or the spectral profile within the Explore page."""
        maximize = self._explore_maximized != which
        self._restore_maximized_plots()
        if maximize:
            self._set_explore_maximized(which)

    def _set_explore_maximized(self, which: str | None) -> None:
        self._explore_maximized = which
        self._apply_explore_page_layout()
        layout_settings.save_maximized(
            "Explore", None if which is None else f"explore.{which}")

    def _restore_maximized_plots(self) -> None:
        if self._maximized_dock is not None:
            self._restore_dock(self._maximized_dock)
        if self._explore_maximized is not None:
            self._set_explore_maximized(None)

    def _on_explore_escape(self) -> None:
        if self._canvas.is_drawing_roi:
            self._canvas.cancel_roi_draw_mode()
        else:
            self._restore_maximized_plots()

    def _apply_explore_page_layout(self) -> None:
        maximized = self._explore_maximized
        shown = self._explore_plot_visibility
        self._rows_splitter.setVisible(maximized != "spectrum")
        self._canvas_frame.setVisible(maximized in (None, "image"))
        self._canvas_frame.set_maximized(maximized == "image")
        for name, frame in (("sensor", self._sensor_frame), ("spatial", self._spatial_frame)):
            frame.setVisible(shown[name] and maximized in (None, name))
            frame.set_maximized(maximized == name)
        self._spectrum_panel.set_maximized(maximized == "spectrum")
        if maximized == "spectrum":
            self._spectrum_panel.setMinimumHeight(0)
            self._spectrum_panel.setMaximumHeight(_UNLIMITED_HEIGHT)
        else:
            self._spectrum_panel.setMinimumHeight(_SPECTRUM_PANEL_HEIGHT)
            self._spectrum_panel.setMaximumHeight(_SPECTRUM_PANEL_HEIGHT)
        explore = self._workspace_tabs.currentWidget() is self._eda_view
        self._spectrum_panel.setVisible(
            explore and shown["spectrum"] and maximized in (None, "spectrum"))
        self._explore_escape.setEnabled(maximized is not None)

    @staticmethod
    def _configure_compact_left_dock(dock: QDockWidget) -> None:
        dock.setMinimumWidth(_COMPACT_LEFT_DOCK_WIDTH)
        dock.setMaximumWidth(_COMPACT_LEFT_DOCK_WIDTH)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)

    def size_class(self) -> str:
        """``"full"`` for windows from 1800 px wide (Full HD), else ``"compact"`` (plan D4)."""
        screen = self.screen()
        width = self.width() if self.isVisible() else (
            screen.availableGeometry().width() if screen is not None else self.width())
        return "full" if width >= _FULL_WIDTH else "compact"

    def _configure_controls_dock_height(self) -> None:
        """The Settings dock: a fixed width per size class, the full height."""
        width = _SETTINGS_DOCK_WIDTH[self.size_class()]
        self._controls_dock.setMinimumWidth(width)
        self._controls_dock.setMaximumWidth(width)
        self._controls_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)

    def _build_controls_panel(self) -> None:
        # The Model page's view holds the current target (Explore and the ROI table use
        # it), so it exists from the start; its Fit settings form and the Predict page
        # are built when first shown.
        self._modeling_view = ModelingView(self.state, self.project_state, parent=self)
        self._prediction_page: PredictionView | None = None
        #: The Predict tab's widget: a placeholder until the page is built, then the page.
        self._predict_tab: QWidget = QWidget()
        self._eda_view = EDAView(
            self.state,
            self.project_state,
            canvas=self._canvas,
            cube_path_provider=lambda: self._cube_path,
            dataset_attached_provider=self.has_attached_dataset,
            current_target_provider=self._modeling_view.current_target,
            selected_roi_id_provider=self._selected_roi_id,
            parent=self,
        )


        # The workspaces: Explore, Model, Predict. The workflow strip switches
        # them, so the tab bar is hidden.
        self._workspace_tabs = QTabWidget()
        self._workspace_tabs.setObjectName("WorkspaceTabs")
        self._workspace_tabs.addTab(self._eda_view, "Explore")
        self._workspace_tabs.addTab(self._modeling_view, "Model")
        self._workspace_tabs.addTab(self._predict_tab, "Predict")
        _require_qt_object(self._workspace_tabs.tabBar(), "workspace tab bar").hide()
        self._workspace_tabs.setDocumentMode(True)

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(6)
        outer.addWidget(self._workspace_tabs)

        # One column of settings; it scrolls vertically when the window is low.
        _narrow_combos(container)
        scroll = _ColumnScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._controls_dock = QDockWidget(_SETTINGS_TITLES[0], self)
        self._controls_dock.setObjectName("SettingsDock")
        column = QWidget()
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(0)
        column_layout.addWidget(scroll, 1)
        # Fit Model stays in view under the scrolling settings (Model step only); the
        # row comes with the Fit settings form (_ensure_fit_settings).
        self._settings_column = column_layout
        self._fit_footer: QWidget | None = None
        self._workspace_tabs.currentChanged.connect(self._sync_fit_footer)
        self._controls_dock.setWidget(column)
        self._configure_controls_dock_height()
        # The Settings column has the right side to itself; no plot covers it.
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._controls_dock)
        self._refresh_workspace_content_pages()

    @property
    def _prediction_view(self) -> PredictionView:
        """The Predict page, built on first use."""
        page = self._prediction_page
        return self._build_prediction_page() if page is None else page

    def _build_prediction_page(self) -> PredictionView:
        started = time.perf_counter()
        page = PredictionView(self.project_state, parent=self)
        self._prediction_page = page
        # The page takes the placeholder's tab, without a tab change (the shown step stays).
        tabs, placeholder = self._workspace_tabs, self._predict_tab
        index, current = tabs.indexOf(placeholder), tabs.currentIndex()
        tabs.blockSignals(True)
        try:
            tabs.removeTab(index)
            tabs.insertTab(index, page, "Predict")
            tabs.setCurrentIndex(current)
        finally:
            tabs.blockSignals(False)
        self._predict_tab = page
        placeholder.deleteLater()
        _narrow_combos(page)
        page.results_changed.connect(self.schedule_workspace_content_refresh)
        page.results_changed.connect(self._update_predict_histogram)
        page.set_data_file_adder(self._add_prediction_data_file)
        page._service_status.status_changed.connect(self.schedule_workspace_content_refresh)
        preferences = getattr(self, "_preferences", None)
        if preferences is not None:
            _apply_predict_defaults(page, preferences)
        logger.debug("Predict page built in {:.3f}s", time.perf_counter() - started)
        return page

    def _ensure_fit_settings(self) -> None:
        """Build the Model page's Fit settings form, once."""
        view = self._modeling_view
        if view.fit_form is not None:
            return
        started = time.perf_counter()
        preferences = getattr(self, "_preferences", None)
        view.embed_fit_settings(
            prepare=None if preferences is None
            else lambda form: _apply_fit_defaults(form, preferences))
        _narrow_combos(view)
        form = view.fit_form
        if form is not None:
            form.setVisible(True)  # Qt shows no child added to a visible widget by itself
        self._fit_footer = view.take_fit_footer()
        if self._fit_footer is not None:
            self._settings_column.addWidget(self._fit_footer)
        self._sync_fit_footer()
        logger.debug("Fit settings built in {:.3f}s", time.perf_counter() - started)

    def _sync_fit_footer(self, *_args: Any) -> None:
        """The Fit Model row shows with the Model step."""
        if self._fit_footer is not None:
            self._fit_footer.setVisible(
                self._workspace_tabs.currentWidget() is self._modeling_view)

    def _build_log_panel(self) -> None:
        """The bottom panel: Checks, ROI table, Models and Log as tabs."""
        self._log_panel = LogPanel()
        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setObjectName("LogDock")
        self._log_dock.setWidget(self._log_panel)
        self._log_dock.visibilityChanged.connect(
            lambda visible: self._log_panel.use_font() if visible else None)
        self._checks_panel = ChecksPanel()
        #: Counts refreshes, so a prediction check that ends after a newer refresh is dropped.
        self._checks_generation = 0
        self._checks_dock = QDockWidget("Checks", self)
        self._checks_dock.setObjectName("ChecksDock")
        self._checks_dock.setWidget(self._checks_panel)
        self._models_panel = ModelsPanel()
        self._models_dock = QDockWidget("Models", self)
        self._models_dock.setObjectName("ModelsDock")
        self._models_dock.setWidget(self._models_panel)
        self.removeDockWidget(self._roi_table_dock)
        bottom = Qt.DockWidgetArea.BottomDockWidgetArea
        self.addDockWidget(bottom, self._checks_dock)
        for dock in (self._roi_table_dock, self._models_dock, self._log_dock):
            self.addDockWidget(bottom, dock)
            self.tabifyDockWidget(self._checks_dock, dock)
        for dock in (self._checks_dock, self._roi_table_dock, self._models_dock, self._log_dock,
                     self._project_browser_dock, self._inspector_dock):
            dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
            dock.show()
        for dock in (self._checks_dock, self._roi_table_dock, self._models_dock, self._log_dock):
            # The tab names the panel; a title bar above it would repeat the name.
            dock.setTitleBarWidget(QWidget(dock))
        self.setTabPosition(bottom, QTabWidget.TabPosition.North)
        self._checks_dock.raise_()
        self._checks_panel.fix_requested.connect(self.run_fix)
        self._checks_panel.check_activated.connect(self._show_check)
        self._models_panel.model_selected.connect(self.select_model)
        self._restore_bottom_dock_sizes()
        self._attach_log_sink(preferences_store.load().log_level)

    def _attach_log_sink(self, level: str) -> None:
        """Send engiNIR's messages of ``level`` and above to the Log panel.

        The level of Tools > Preferences (default INFO), so the panel never shows
        DEBUG lines at INFO; a change applies at once.
        """
        self._teardown_log_sink()
        self._log_sink_level = level
        self._log_sink_id = add_log_sink(
            self._log_panel.sink,
            level=level,
            format="{time:HH:mm:ss} | {level:<7} | {message}",
            filter="enginir",
        )

    def _teardown_log_sink(self, *_args) -> None:
        """Remove the loguru sink once the window or log panel is torn down."""
        if self._log_sink_id is None:
            return
        try:
            remove_log_sink(self._log_sink_id)
        except ValueError:
            pass
        self._log_sink_id = None

    def _open_project_btn_clicked_with_path(self, path: str) -> None:
        """Testable core of the Open flow - accepts path directly."""
        self._eda_view.open_project_btn_clicked_with_path(path)

    # ------------------------------------------------------------------
    # Canvas <-> State wiring
    # ------------------------------------------------------------------

    def _on_canvas_click(self, frame: int, row: int) -> None:
        """Handle canvas click - clip to valid bounds and update state."""
        s = self.state
        if not s.has_dataset:
            return
        frame = int(np.clip(frame, 0, s.n_frames - 1))
        row = int(np.clip(row, 0, s.height - 1))
        s.set_frame(frame)
        s.set_row(row)

    def has_attached_dataset(self) -> bool:
        """Return True when the window has a real dataset source path attached."""
        return bool(self._cube_path)

    def _on_crosshair_update(self, _: int = 0) -> None:
        """Sync canvas crosshair with current state position."""
        self._canvas.set_crosshair(self.state.frame, self.state.row)

    def _update_status_strip(self, *_args) -> None:
        """What the shown step works on; the cursor position only where there is a cursor."""
        project_label = "none"
        if self.project_state.project_path:
            project_label = os.path.basename(self.project_state.project_path)
        tabs = getattr(self, "_workspace_tabs", None)
        current = None if tabs is None else tabs.currentWidget()
        if tabs is not None and current is self._predict_tab:
            # Predict: what the shown map is (the hover readout replaces it on the map).
            self._status_label.setText(
                f"Project: {project_label} | {self._prediction_view.status_text()}")
            return
        if tabs is not None and current is self._modeling_view:
            summary = self._modeling_view.workspace_summary()
            self._status_label.setText(
                f"Project: {project_label} | Model: {summary['model']} | "
                f"Target: {summary['target']} | Calibration set: {summary['sample_set']}")
            return

        dataset_label = "none"
        if self._cube_path:
            shown = self._eda_view.shown_dataset_id()
            dataset_label = (self.project_state.data_file_label(shown) if shown is not None
                             else os.path.basename(self._cube_path))
        parts = [f"Project: {project_label}", f"Dataset: {dataset_label}",
                 f"Signal: {signal_label(self.state.y_unit_effective)}"]
        shown_record = (None if not self._cube_path else self.project_state.datasets.get(
            self._eda_view.shown_dataset_id() or ""))
        if shown_record is not None and shown_record.binning > 1:
            parts.insert(2, f"Binned {binning_text(shown_record.binning)}")
        if not hasattr(self, "_explore_tabs") or self._on_data_file_page():
            # Frame, row and band are the crosshair of the dataset's image.
            has = self.state.has_dataset
            parts += [f"Frame: {self.state.frame if has else '-'}",
                      f"Row: {self.state.row if has else '-'}",
                      f"Band: {self.state.band if has else '-'}"]
        self._status_label.setText(" | ".join(parts))

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        s = self.state
        # Panel -> state
        self._sensor.row_pick.connect(s.set_row)
        self._sensor.band_pick.connect(s.set_band)
        self._spatial.row_pick.connect(s.set_row)
        self._spectrum.band_pick.connect(s.set_band)

        # Canvas -> state
        self._canvas.clicked.connect(self._on_canvas_click)

        # State -> canvas crosshair
        s.frame_changed.connect(self._on_crosshair_update)
        s.row_changed.connect(self._on_crosshair_update)
        s.frame_changed.connect(self._update_status_strip)
        s.row_changed.connect(self._update_status_strip)
        s.band_changed.connect(self._update_status_strip)
        s.units_changed.connect(self._update_status_strip)
        self.project_state.project_loaded.connect(self._update_status_strip)
        self.project_state.project_loaded.connect(self.schedule_workspace_content_refresh)
        self.project_state.rois_changed.connect(self.schedule_workspace_content_refresh)
        self.project_state.targets_changed.connect(self.schedule_workspace_content_refresh)
        self.project_state.target_assignments_changed.connect(
            self.schedule_workspace_content_refresh
        )
        self.project_state.sample_sets_changed.connect(self.schedule_workspace_content_refresh)
        self.project_state.model_runs_changed.connect(self.schedule_workspace_content_refresh)
        self.project_state.predictions_changed.connect(self.schedule_workspace_content_refresh)
        self.project_state.active_model_changed.connect(self.schedule_workspace_content_refresh)
        self._modeling_view.view_state_changed.connect(self.schedule_workspace_content_refresh)
        self._project_browser.itemSelectionChanged.connect(
            self._on_project_browser_selection_changed
        )
        self._workspace_tabs.currentChanged.connect(self._on_workspace_tab_changed)
        self._canvas.roi_selected.connect(self._on_canvas_roi_selected)
        self._canvas.roi_selection_cleared.connect(self._on_canvas_roi_selection_cleared)
        self._modeling_view._service_status.status_changed.connect(
            self.schedule_workspace_content_refresh
        )
        for name, dock in self._data_file_docks.items():
            dock.visibilityChanged.connect(
                lambda _visible, name=name: self._on_data_file_dock_visibility_changed(name))

        self.project_state.project_loaded.connect(self._refresh_data_file_choices)
        self.project_state.datasets_changed.connect(self._refresh_data_file_choices)
        self.project_state.data_files_checked.connect(self._refresh_data_file_choices)
        # The data files are looked up in the background.
        self.project_state.project_loaded.connect(self._look_up_data_files)
        self.project_state.datasets_changed.connect(self._look_up_data_files)
        self._project_browser.itemDoubleClicked.connect(self._on_browser_item_activated)
        self._eda_view.roi_table_requested.connect(self.show_roi_table)
        self._eda_view.dataset_needed.connect(self.add_rois_on)
        self._project_rois = ProjectRoisView(
            self.state, self.project_state,
            current_target_id=lambda: (None if (target := self._modeling_view.current_target())
                                       is None else target.target_id),
            provenance=self._explore_provenance)
        self._explore_tabs.addTab(self._project_rois, "Project ROIs")
        delete = getattr(self, "_delete_roi_action", None)
        if delete is not None:
            self._project_rois.mosaic.addAction(delete)
        self._project_rois.roi_selected.connect(self._on_canvas_roi_selected)
        self._project_rois.roi_opened.connect(self.open_roi)
        self._project_rois.add_rois_requested.connect(self.add_rois_on)
        self._modeling_view.view_state_changed.connect(self._project_rois.current_target_changed)
        self.project_state.rois_changed.connect(self._update_project_rois_tab)
        self.project_state.project_loaded.connect(self._update_project_rois_tab)
        self.project_state.project_loaded.connect(self._choose_explore_tab)
        self._explore_tabs.currentChanged.connect(self._show_last_data_file)
        self.project_state.data_files_checked.connect(self._show_last_data_file)
        self._explore_tabs.currentChanged.connect(self._on_explore_tab_changed)
        self._project_rois.data_file_filter_changed.connect(self._follow_roi_table_filter)
        self.project_state.project_loaded.connect(self._sync_roi_table)
        self._update_project_rois_tab()
        self._choose_explore_tab()
        self._modeling_view.roi_table_requested.connect(self.show_roi_table)
        self._refresh_data_file_choices()
        saved = layout_settings.load_maximized().get("Explore", "")
        if saved in ("explore.image", "explore.spectrum"):
            self._explore_maximized = saved.removeprefix("explore.")
        self._sync_workspace_content_host()

    def open_roi(self, roi_id: str) -> None:
        """Show the data file of an ROI with the ROI selected (Data file tab)."""
        roi = self.project_state.rois.get(roi_id)
        if roi is None:
            return
        self._explore_tabs.setCurrentWidget(self._data_file_page)
        if self._eda_view.shown_dataset_id() == roi.dataset_id:
            self._canvas.select_roi(roi_id)
            self._on_canvas_roi_selected(roi_id)
            return
        self._pending_roi_selection = roi_id
        self.show_data_file(roi.dataset_id)

    def _update_project_rois_tab(self, *_args: Any) -> None:
        if self._project_rois is not None:
            self._explore_tabs.setTabText(self._explore_tabs.indexOf(self._project_rois),
                                          f"Project ROIs ({len(self.project_state.rois)})")

    def _choose_explore_tab(self, *_args: Any) -> None:
        """A project with ROIs and no data file shown opens on its ROIs (attachment contract)."""
        if self._project_rois is not None and self.project_state.rois \
                and not self.has_attached_dataset():
            self._explore_tabs.setCurrentWidget(self._project_rois)
        else:
            self._show_last_data_file()

    def _look_up_data_files(self) -> None:
        """Look up the data files not looked up yet, in a background job."""
        state = self.project_state
        project, unknown = state.project, state.data_files_unknown()
        if project is None or not unknown or getattr(self, "_data_file_lookup", False):
            return
        self._data_file_lookup = True
        logger.debug("Looking up the files of {} dataset(s) in the background", len(unknown))

        def done() -> None:
            self._data_file_lookup = False
            if state.project is project and state.data_files_unknown():
                self._look_up_data_files()  # datasets added while it ran

        # From the event loop: opening a project calls this before the window is shown,
        # and no thread may start during show() (a hang on macOS, 2026-11-19).
        start_background_job_soon(
            owner=self, fn=lambda: look_up_data_files(project, unknown),
            on_success=lambda ids: state.data_files_looked_up(project, list(ids)),
            on_error=lambda message: logger.warning("The files of the datasets could not be looked up: {}",
                                                    message),
            on_finished=done)

    def _show_last_data_file(self, *_args: Any) -> None:
        """Load the project's last shown data file when the Data file tab shows nothing.

        The last file is user settings (by project id); a missing one is not
        loaded, it stays marked "(not found)" in the list.
        """
        if self._explore_tabs.currentWidget() is not self._data_file_page \
                or self.has_attached_dataset() or self._data_file_job is not None:
            return
        project_id = self.project_state.project_meta.get("project_id")
        dataset_id = None if not project_id else load_last_data_file(str(project_id))
        dataset = self.project_state.datasets.get(dataset_id or "")
        if dataset_id is None or dataset is None:
            return
        if not self.project_state.data_file_known(dataset_id):
            return  # shown when the background lookup has answered (data_files_checked)
        if self.project_state.data_file(dataset_id) is not None:
            self.show_data_file(dataset_id)

    def show_roi_table(self) -> None:
        """Show the ROI table on the ROIs of the current page and the target selected in Model.

        The table is a tab of the bottom panel, beside Checks, Models and Log;
        while it is in front the panel takes about 40 % of the window's height.
        """
        self._roi_table.show_data_file(self._roi_table_data_file())
        target = self._modeling_view.current_target()
        self._roi_table.focus_target(None if target is None else target.target_id)
        self._roi_table_dock.show()
        self._roi_table_dock.raise_()

    def _on_data_file_page(self) -> bool:
        return (self._workspace_tabs.currentWidget() is self._eda_view
                and self._explore_tabs.currentWidget() is self._data_file_page)

    def _roi_table_data_file(self) -> str | None:
        """The data file the ROI table shows: the page's (``None``: all files)."""
        if self._on_data_file_page():
            return self._eda_view.shown_dataset_id()
        if self._workspace_tabs.currentWidget() is self._eda_view \
                and self._project_rois is not None:
            return self._project_rois.shown_dataset_id()
        return None

    def _follow_roi_table_filter(self, *_args: Any) -> None:
        if not self._roi_table_dock.isHidden():
            self._roi_table.show_data_file(self._roi_table_data_file())

    def _on_explore_tab_changed(self, *_args: Any) -> None:
        self._sync_data_file_docks()
        self._follow_roi_table_filter()

    def _explore_page_key(self) -> str | None:
        """``"data_file"`` or ``"project_rois"`` in Explore, ``None`` in Model and Predict."""
        if self._workspace_tabs.currentWidget() is not self._eda_view:
            return None
        return "data_file" if self._on_data_file_page() else "project_rois"

    def _sync_roi_table(self, *_args: Any) -> None:
        """Show or hide the ROI table as last chosen on the current Explore tab."""
        key = self._explore_page_key()
        if key is None or self.project_state.project_path is None:
            return
        wanted = self._roi_table_visibility[key]
        if wanted == (not self._roi_table_dock.isHidden()):
            return
        self._syncing_roi_table = True
        try:
            if wanted:
                self.show_roi_table()
            else:
                self._roi_table_dock.hide()
        finally:
            self._syncing_roi_table = False

    def _on_roi_table_visibility_changed(self, _visible: bool) -> None:
        """Remember the user opening or closing the ROI table on an Explore tab."""
        key = self._explore_page_key()
        if key is None or getattr(self, "_syncing_roi_table", False) \
                or getattr(self, "_syncing_data_file_docks", False):
            return
        self._roi_table_visibility[key] = not self._roi_table_dock.isHidden()

    def _on_project_browser_selection_changed(self) -> None:
        payload = self._selected_project_browser_payload()
        if payload is not None and payload.get("kind") == "roi":
            self._canvas.select_roi(payload.get("id"))
            self._roi_table.select_roi(str(payload.get("id")))
            if self._project_rois is not None:
                self._project_rois.select_roi(str(payload.get("id")))
        else:
            self._canvas.select_roi(None)
        if payload is not None and payload.get("kind") == "model_run":
            self._modeling_view.select_model_run(payload.get("id"))
        elif payload is None or payload.get("kind") != "roi":
            self._modeling_view.select_model_run(None)
        if payload is not None and payload.get("kind") == "prediction":
            self._prediction_view.select_prediction_run(payload.get("id"))
        elif self._prediction_page is not None:  # an unbuilt page has nothing selected
            self._prediction_page.select_prediction_run(None)
        self._eda_view.refresh_attachment_state()
        self._refresh_workspace_content_pages()

    def _on_canvas_roi_selected(self, roi_id: str) -> None:
        self._project_browser.select_payload({"kind": "roi", "id": roi_id})

    def _on_canvas_roi_selection_cleared(self) -> None:
        payload = self._selected_project_browser_payload()
        if payload is not None and payload.get("kind") == "roi":
            self._project_browser.clearSelection()

    def _refresh_workspace_content_pages(self, *_args) -> None:
        current = self._workspace_tabs.currentWidget()
        if current is self._modeling_view:
            self._refresh_model_workspace_content()
        elif current is self._predict_tab:
            self._refresh_prediction_workspace_content()
        self._refresh_inspector_context()
        self._sync_workspace_content_host()

    def schedule_workspace_content_refresh(self, *_args) -> None:
        """Coalesce repeated shell refreshes into one event-loop update."""
        if self._workspace_refresh_pending:
            return
        self._workspace_refresh_pending = True
        QTimer.singleShot(0, self._flush_workspace_content_refresh)

    def _flush_workspace_content_refresh(self) -> None:
        self._workspace_refresh_pending = False
        self._refresh_workspace_content_pages()

    def _on_workspace_tab_changed(self, _index: int) -> None:
        current = self._workspace_tabs.currentWidget()
        if current is not self._eda_view and self._roi_table_in_front():
            # The tall ROI table belongs to Values: Model gets its Models tab,
            # Predict the Checks, both with the page's height back.
            (self._models_dock if current is self._modeling_view else self._checks_dock).raise_()
        self._sync_workflow_step()
        self._refresh_context()
        self._update_status_strip()
        self._sync_workspace_content_host()
        self._refresh_workspace_content_pages()

    def _sync_workspace_content_host(self) -> None:
        current = self._workspace_tabs.currentWidget()
        is_explore = current is self._eda_view
        is_model = current is self._modeling_view
        if is_model:
            self._ensure_fit_settings()
            self._ensure_model_content_page()
        elif not is_explore:
            self._ensure_predict_content_page()

        if is_explore:
            self._workspace_content_stack.setCurrentWidget(self._explore_content_page)
        elif is_model:
            self._workspace_content_stack.setCurrentWidget(self._model_content_page)
        else:
            self._workspace_content_stack.setCurrentWidget(self._predict_content_page)

        self._sync_data_file_docks()
        self._follow_roi_table_filter()
        self._apply_explore_page_layout()
        for name, action in self._page_plot_actions.items():
            action.setEnabled(is_explore)
            action.blockSignals(True)
            action.setChecked(self._explore_plot_visibility[name])
            action.blockSignals(False)
        self._restore_bottom_dock_sizes()
        self._schedule_bottom_dock_resize()

    def _schedule_bottom_dock_resize(self, *_args: Any) -> None:
        if self._bottom_dock_resize_pending:
            return
        self._bottom_dock_resize_pending = True
        QTimer.singleShot(0, self._restore_bottom_dock_sizes)

    def _roi_table_in_front(self) -> bool:
        """Whether the ROI table is the front tab of the bottom panel.

        ``isVisible`` is true for a dock behind another tab too; a widget that
        is really on screen has a visible region.
        """
        return not self._roi_table.visibleRegion().isEmpty()

    def _restore_bottom_dock_sizes(self) -> None:
        """The bottom panel's height and the Settings width for the window's size class."""
        self._bottom_dock_resize_pending = False
        if not hasattr(self, "_checks_dock"):
            return
        size = self.size_class()
        self._configure_controls_dock_height()
        height = _BOTTOM_PANEL_HEIGHT[size]
        front = self._checks_dock
        if self._roi_table_in_front():
            height = max(height, int(_ROI_TABLE_HEIGHT_SHARE * self.height()))
            front = self._roi_table_dock
        self.resizeDocks([front], [height], Qt.Orientation.Vertical)

    def _refresh_model_workspace_content(self) -> None:
        self._ensure_model_content_page()
        model_diagnostics_widget = self._model_diagnostics_widget
        if model_diagnostics_widget is None:
            return
        model_summary = self._modeling_view.workspace_summary()
        self._model_content_title.setText(model_summary["title"])
        self._model_card_target_value.setText(model_summary["target"])
        self._model_card_sample_set_value.setText(model_summary["sample_set"])
        self._model_card_model_value.setText(model_summary["model"])
        self._model_card_components_value.setText(model_summary["components"])
        self._model_card_cv_value.setText(model_summary["cv"])
        self._model_card_signal_value.setText(model_summary["signal"])
        self._model_card_roi_count_value.setText(model_summary["roi_count"])
        self._model_card_dataset_count_value.setText(model_summary["dataset_count"])
        self._model_card_pixel_count_value.setText(model_summary["pixel_count"])
        self._model_metric_r2_value.setText(model_summary["r2"])
        self._model_metric_rmsecv_value.setText(model_summary["rmsecv"])
        self._model_metric_bias_value.setText(model_summary["bias"])
        self._set_metric_rows(model_summary)
        self._update_verdict_row()
        model_diagnostics_widget.update_data(
            self._modeling_view.workspace_diagnostics_payload(),
            empty_reason=model_summary["readiness"],
            provenance=self._modeling_view.workspace_diagnostics_provenance(),
        )

    def _refresh_prediction_workspace_content(self) -> None:
        self._ensure_predict_content_page()
        predict_map_widget = self._predict_map_widget
        if predict_map_widget is None:
            return
        prediction_summary = self._prediction_view.workspace_summary()
        has_results = bool(self.project_state.predictions or self.project_state.roi_results)
        self._predict_page_stack.setCurrentWidget(
            self._predict_workspace_splitter if has_results else self._predict_empty)
        self._predict_content_title.setText(prediction_summary["title"])
        self._predict_headline.setText(prediction_summary["headline"])
        self._predict_card_correction_value.setText(prediction_summary["reference_correction"])
        self._predict_card_target_value.setText(prediction_summary["target"])
        self._predict_contract_format_value.setText(prediction_summary["format"])
        self._predict_contract_signal_value.setText(prediction_summary["signal"])
        self._predict_contract_bands_value.setText(prediction_summary["bands"])
        self._predict_map_shape_value.setText(prediction_summary["map_shape"])
        self._predict_map_range_value.setText(prediction_summary["map_range"])
        self._predict_clipping_value.setText(prediction_summary["clipping"])
        self._predict_diagnostics_value.setText(prediction_summary["diagnostics"])
        self._predict_wavelengths_value.setText(prediction_summary["wavelengths"])
        self._predict_rois_value.setText(prediction_summary.get("rois", "none"))
        self._predict_roi_results.update_data(
            self._prediction_view.roi_results_payload(),
            provenance=self._prediction_view.workspace_provenance())
        self._update_predict_histogram()
        if not predict_map_widget.hover_text:
            self._update_status_strip()
        predict_map_widget.update_data(
            self._prediction_view.workspace_visual_payload(),
            empty_reason=prediction_summary["readiness"],
        )

    def _set_metric_rows(self, model_summary: dict[str, Any]) -> None:
        """Fill the *per ROI* and *per pixel* groups.

        The fold warnings of the run are a check (Checks tab, guide bar), not a
        line of this summary.
        """
        roi_rows = list(model_summary.get("roi_metrics") or [])
        if not roi_rows:  # no run yet: the required three, empty
            roi_rows = [("Q² (CV R²)", model_summary["r2"]), ("RMSECV (per ROI)", model_summary["rmsecv"]),
                        ("Bias", model_summary["bias"])]
        rows = [f"{label}: {text}" for label, text in roi_rows]
        pixel_rows = list(model_summary.get("pixel_metrics") or [])
        shown = (rows, pixel_rows)
        if shown == getattr(self, "_shown_metric_rows", None):
            return
        self._shown_metric_rows = shown
        self._model_metric_rows, self._pixel_rows = rows, pixel_rows
        for layout, entries in ((self._model_metrics_layout, roi_rows),
                                (self._model_pixel_metrics_layout, pixel_rows)):
            while layout.rowCount():
                layout.removeRow(0)
            for label, text in entries:
                layout.addRow(label, QLabel(text))
        self._model_pixel_metrics_box.setVisible(bool(pixel_rows))

    def _on_explore_plot_visibility_changed(self, plot_name: str, visible: bool) -> None:
        if self._workspace_tabs.currentWidget() is not self._eda_view:
            return
        self._explore_plot_visibility[plot_name] = visible
        self._schedule_bottom_dock_resize()

    def _sync_data_file_docks(self) -> None:
        """Show the data-file docks (as last chosen) on the Data file tab, hide them elsewhere."""
        on_page = self._on_data_file_page()
        maximized = self._maximized_dock
        if not on_page and maximized is not None and maximized in self._data_file_docks.values():
            self._restore_dock(maximized)  # a floating dock would cover the page
        self._syncing_data_file_docks = True
        try:
            for name, dock in self._data_file_docks.items():
                dock.setVisible(on_page and self._explore_plot_visibility[name])
                _require_qt_object(dock.toggleViewAction(),
                                   f"{dock.windowTitle()} toggle action").setEnabled(on_page)
        finally:
            self._syncing_data_file_docks = False
        self._sync_roi_table()

    def _on_data_file_dock_visibility_changed(self, name: str) -> None:
        """Remember a dock the user shows or closes on the Data file tab.

        ``isHidden`` rather than the signal's flag: a dock behind another dock's
        tab is not visible but still shown.
        """
        if getattr(self, "_syncing_data_file_docks", False) or not self._on_data_file_page():
            return
        self._explore_plot_visibility[name] = not self._data_file_docks[name].isHidden()
        self._schedule_bottom_dock_resize()

    def _on_spectrum_panel_visibility_action(self, visible: bool) -> None:
        self._on_page_plot_visibility_action("spectrum", visible)

    def _on_page_plot_visibility_action(self, name: str, visible: bool) -> None:
        """A View menu switch of a Dataset page plot (or its header's close button)."""
        self._on_explore_plot_visibility_changed(name, visible)
        action = getattr(self, "_page_plot_actions", {}).get(name)
        if action is not None and action.isChecked() != visible:
            action.blockSignals(True)
            action.setChecked(visible)
            action.blockSignals(False)
        if not visible and self._explore_maximized == name:
            self._restore_maximized_plots()
        self._apply_explore_page_layout()

    def _refresh_inspector_context(self) -> None:
        payload = self._selected_project_browser_payload()
        if payload is not None:
            self._inspector.show_item(payload)
            return

        current = self._workspace_tabs.currentWidget()
        if current is self._modeling_view:
            self._inspector.show_workspace_summary(
                self._model_content_title.text(),
                [
                    f"Target: {self._model_card_target_value.text()}",
                    f"Calibration set: {self._model_card_sample_set_value.text()}",
                    f"Model: {self._model_card_model_value.text()}",
                    f"Components: {self._model_card_components_value.text()}",
                    f"Validation: {self._model_card_cv_value.text()}",
                    f"Signal, preprocessing: {self._model_card_signal_value.text()}",
                    f"ROIs: {self._model_card_roi_count_value.text()}",
                    f"Datasets: {self._model_card_dataset_count_value.text()}",
                    f"Pixels: {self._model_card_pixel_count_value.text()}",
                ],
            )
            return
        if current is self._predict_tab:
            self._inspector.show_workspace_summary(
                self._predict_content_title.text(),
                [
                    self._predict_headline.text(),
                    f"Prediction ROIs: {self._predict_rois_value.text()}",
                    f"Reference correction: {self._predict_card_correction_value.text()}",
                    f"Target: {self._predict_card_target_value.text()}",
                    f"Input contract: {self._predict_contract_format_value.text()}",
                    f"Signal: {self._predict_contract_signal_value.text()}",
                    f"Bands: {self._predict_contract_bands_value.text()}",
                    f"Map shape: {self._predict_map_shape_value.text()}",
                    f"Map range: {self._predict_map_range_value.text()}",
                    f"Reflectance clipping: {self._predict_clipping_value.text()}",
                    f"Applicability: {self._predict_diagnostics_value.text()}",
                    f"Wavelengths: {self._predict_wavelengths_value.text()}",
                ],
            )
            return
        self._inspector.show_default()

    def _selected_project_browser_payload(self) -> dict[str, str] | None:
        selected_items = self._project_browser.selectedItems()
        item = selected_items[0] if selected_items else None
        payload = None if item is None else item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(payload, dict):
            return payload
        return None

    def edit_grid_defaults(self) -> bool:
        """Tools > ROI Grid Defaults: the geometry every new ROI grid starts with."""
        from ._grid_settings import load_grid_defaults
        from ._roi_grid import GridDefaultsDialog

        return GridDefaultsDialog(load_grid_defaults(), parent=self).exec() \
            == QDialog.DialogCode.Accepted

    def _selected_roi_id(self) -> str | None:
        payload = self._selected_project_browser_payload()
        if payload is None or payload.get("kind") != "roi":
            return None
        roi_id = payload.get("id")
        return None if roi_id is None else str(roi_id)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        mb = _require_qt_object(self.menuBar(), "main menu bar")
        # Every action of the viewer is a menu item (no toolbar).
        self._menu_file = _add_required_menu(mb, "File")
        self._menu_edit = _add_required_menu(mb, "Edit")
        self._menu_view = _add_required_menu(mb, "View")
        self._menu_project = _add_required_menu(mb, "Project")
        self._menu_roi = _add_required_menu(mb, "ROI")
        self._menu_model = _add_required_menu(mb, "Model")
        self._menu_predict = _add_required_menu(mb, "Predict")
        self._menu_help = _add_required_menu(mb, "Help")
        self._populate_file_menu()
        self._populate_edit_menu()
        self._populate_view_menu()
        self._populate_workflow_menus()
        self._populate_help_menu()
        self._default_layout_state = self.saveState()
        self._release_minimum_size()
        self._status_label = QLabel()
        self._saved_label = QLabel()
        self._saved_label.setObjectName("SavedMessage")
        self._saved_clear = QTimer(self)
        self._saved_clear.setSingleShot(True)
        self._saved_clear.timeout.connect(self._saved_label.clear)
        status_bar = _require_qt_object(self.statusBar(), "main status bar")
        status_bar.addPermanentWidget(self._status_label, 1)
        status_bar.addPermanentWidget(self._saved_label)
        self.project_state.saved.connect(self._show_saved_message)
        self._update_status_strip()

    # ------------------------------------------------------------------
    # Menus of the workflow
    # ------------------------------------------------------------------

    def _menu_button_action(self, menu: QMenu, text: str,
                            button: QAbstractButton | Callable[[], QAbstractButton], *,
                            icon_name: str = "", shortcut: str = "",
                            tip: str = "", before: Callable[[], None] | None = None
                            ) -> QAction:
        """A menu item that presses ``button`` and is enabled when the button is.

        ``button`` may be a function that returns it (a page built on first use);
        ``before`` runs before the press (e.g. builds the form the button needs).
        """
        action = _add_required_action(menu, text)
        if icon_name:
            action.setIcon(icon(icon_name))
        if shortcut:
            action.setShortcut(shortcut)
        if tip:
            action.setToolTip(wrap_tooltip(tip))
        def press(_checked: bool = False) -> None:
            if before is not None:
                before()
            _button_of(button).click()

        action.triggered.connect(press)
        self._button_actions.append((action, button))
        return action

    def _sync_button_actions(self) -> None:
        """Enable each menu item like its button (the menu shown builds its page)."""
        menu = self.sender()
        for action, button in self._button_actions:
            if callable(button) and isinstance(menu, QMenu) and action not in menu.actions():
                continue  # a page not built for another menu's sake
            action.setEnabled(_button_of(button).isEnabled())

    def _populate_edit_menu(self) -> None:
        m = self._menu_edit
        act = _add_required_action(m, "Copy")
        act.setShortcut("Ctrl+C")
        act.triggered.connect(self._copy_focused)
        act = _add_required_action(m, "Paste")
        act.setShortcut("Ctrl+V")
        act.triggered.connect(self._paste_focused)
        m.addSeparator()
        act = _add_required_action(m, "Preferences...")
        act.setIcon(icon("settings"))
        act.setShortcut("Ctrl+,")
        act.triggered.connect(self.edit_preferences)

    def _populate_workflow_menus(self) -> None:
        self._button_actions: list[
            tuple[QAction, QAbstractButton | Callable[[], QAbstractButton]]] = []
        eda, model = self._eda_view, self._modeling_view
        m = self._menu_project
        act = _add_required_action(m, "Add Dataset...")
        act.setIcon(icon("photo-plus"))
        act.triggered.connect(lambda: self.add_data_file())
        act.setShortcut("Ctrl+D")
        act = _add_required_action(m, "Rename Dataset...")
        act.setToolTip(wrap_tooltip("A short name for a dataset, shown instead of its file name"))
        act.triggered.connect(lambda: self.name_dataset())
        act = _add_required_action(m, "Set Wavelengths...")
        act.setToolTip(wrap_tooltip("Wavelengths from a table for datasets whose files have none"))
        act.triggered.connect(lambda: self.set_wavelengths())
        act = _add_required_action(m, "Locate Moved Datasets...")
        act.setToolTip(wrap_tooltip("Find the files of datasets that are not at their stored place"))
        act.triggered.connect(self._locate_first_missing)
        act = _add_required_action(m, "Remove Dataset...")
        act.setToolTip(wrap_tooltip("Remove a dataset without ROIs from the project; the file "
                                    "is not touched"))
        act.triggered.connect(lambda: self.remove_dataset())
        m.addSeparator()
        self._menu_button_action(m, "Targets...", model._btn_target, icon_name="target")
        # Both open the ROI table. Only ROI > ROI Table holds Ctrl+T: two actions with one
        # shortcut would be ambiguous in Qt and neither would fire; this item shows the
        # key after a tab, which Qt draws in the shortcut column (plan menu-polish, P1).
        act = _add_required_action(m, "Target Values...\tCtrl+T")
        act.setIcon(icon("table"))
        act.setToolTip(wrap_tooltip("The ROI table, where target values are typed"))
        act.triggered.connect(lambda: model._btn_assign_target.click())

        m = self._menu_roi
        self._menu_button_action(m, "Add ROI", eda._btn_roi_training, icon_name="square-plus",
                                 shortcut="Ctrl+R")
        self._menu_button_action(m, "Add ROI Grid...", eda._btn_roi_grid, icon_name="grid-dots",
                                 shortcut="Ctrl+G")
        act = _add_required_action(m, "ROI Grid Defaults...")
        act.triggered.connect(self.edit_grid_defaults)
        delete = self._menu_button_action(m, "Delete ROI", eda._btn_roi_delete,
                                          icon_name="trash")
        # Delete deletes the selected ROI only where ROIs are selected (the image and the
        # mosaic), never while typing in the ROI table (plan menu-polish, R2).
        delete.setShortcut(QKeySequence(QKeySequence.StandardKey.Delete))
        delete.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._canvas.addAction(delete)
        self._delete_roi_action = delete  # the mosaic gets it when it is built
        act = _add_required_action(m, "Find ROI...")
        act.setShortcut("Ctrl+F")
        act.triggered.connect(self._find_roi)
        m.addSeparator()
        act = _add_required_action(m, "ROI Table")
        act.setIcon(icon("table"))
        act.setShortcut("Ctrl+T")
        act.triggered.connect(self.show_roi_table)
        act = _add_required_action(m, "Export ROI Spectra (CSV)...")
        act.triggered.connect(eda.export_csv_btn_clicked)

        m = self._menu_model
        self._menu_button_action(m, "Fit Model", model._btn_fit_model, icon_name="player-play",
                                 shortcut="F5", before=self._ensure_fit_settings)
        self._menu_button_action(m, "Load Model Settings", model._btn_fit_again,
                                 icon_name="refresh",
                                 tip="The selected model's settings in the Fit settings",
                                 before=self._ensure_fit_settings)
        act = _add_required_action(m, "Propose Bands...")
        act.triggered.connect(lambda: self._with_fit_form("_propose_bands_clicked"))
        act = _add_required_action(m, "Select Bands on Plot...")
        act.triggered.connect(lambda: self._with_fit_form("_select_bands_on_plot"))
        m.addSeparator()
        act = _add_required_action(m, "Rename Model...")
        act.triggered.connect(lambda: self.rename_model())
        act = _add_required_action(m, "Retire Model...")
        act.setToolTip(wrap_tooltip("Keep the model as a record that predicts no more; its "
                                    "per-pixel diagnostics are deleted"))
        act.triggered.connect(lambda: self.retire_model())

        m = self._menu_predict
        self._menu_button_action(m, "Predict ROIs...",
                                 lambda: self._prediction_view._btn_predict_rois,
                                 icon_name="wand", shortcut="F6")
        self._menu_button_action(m, "Predict Map...", lambda: self._prediction_view._btn_predict,
                                 icon_name="map", shortcut="F7")
        self._menu_button_action(m, "Cancel Prediction",
                                 lambda: self._prediction_view._btn_cancel,
                                 icon_name="player-stop")
        for menu in (self._menu_project, self._menu_roi, self._menu_model, self._menu_predict):
            menu.aboutToShow.connect(self._sync_button_actions)

    def _with_fit_form(self, method: str) -> None:
        """Call a method of the Fit settings form, building it first."""
        self._ensure_fit_settings()
        form = self._modeling_view.fit_form
        if form is not None:
            getattr(form, method)()

    def _copy_focused(self) -> None:
        """Copy the selection of the focused table or text to the clipboard."""
        widget = QApplication.focusWidget()
        copy = getattr(widget, "copy", None)
        if callable(copy):
            copy()
            return
        if isinstance(widget, QAbstractItemView):
            rows = sorted({(i.row(), i.column()) for i in widget.selectedIndexes()})
            model = widget.model()
            if model is None or not rows:
                return
            lines: dict[int, list[str]] = {}
            for row, column in rows:
                lines.setdefault(row, []).append(str(model.index(row, column).data() or ""))
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText("\n".join("\t".join(cells) for cells in lines.values()))

    def _find_roi(self) -> None:
        """Ask for an ROI name and select it."""
        from PyQt6.QtWidgets import QInputDialog

        names = sorted(roi.name for roi in self.project_state.rois.values())
        if not names:
            QMessageBox.information(self, "Find ROI", "The project has no ROI.")
            return
        name, ok = QInputDialog.getItem(self, "Find ROI", "ROI:", names, 0, True)
        if ok:
            found = [roi_id for roi_id, roi in self.project_state.rois.items()
                     if roi.name == name]
            if found:
                self.show_roi_table()
                self._on_canvas_roi_selected(found[0])

    def _locate_first_missing(self) -> None:
        missing = [dataset_id for dataset_id in self.project_state.datasets
                   if self.project_state.data_file(dataset_id) is None]
        if not missing:
            QMessageBox.information(self, "Locate moved datasets",
                                    "The files of all datasets were found.")
            return
        self.locate_data_file(missing[0])

    # ------------------------------------------------------------------
    # Workflow strip and checks
    # ------------------------------------------------------------------

    def _build_workflow_strip(self) -> None:
        from PyQt6.QtWidgets import QToolBar

        self._workflow = WorkflowStrip()
        self._workflow.step_clicked.connect(self.show_step)
        bar = QToolBar("Workflow", self)
        bar.setObjectName("WorkflowBar")
        bar.setMovable(False)
        bar.setFloatable(False)
        toggle = bar.toggleViewAction()
        if toggle is not None:
            toggle.setVisible(False)
        bar.addWidget(self._workflow)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)
        self._workflow_bar = bar
        self._checks_timer = QTimer(self)
        self._checks_timer.setSingleShot(True)
        self._checks_timer.setInterval(150)
        self._checks_timer.timeout.connect(self.refresh_checks)
        state = self.project_state
        for signal in (state.project_loaded, state.datasets_changed, state.rois_changed,
                       state.targets_changed, state.target_assignments_changed,
                       state.model_runs_changed, state.sample_sets_changed,
                       state.data_files_checked):
            signal.connect(self._checks_timer.start)
        self._explore_tabs.currentChanged.connect(self._sync_workflow_step)
        self._explore_tabs.currentChanged.connect(self._update_status_strip)
        self._modeling_view.view_state_changed.connect(self._update_status_strip)
        self._modeling_view.view_state_changed.connect(self._refresh_context)
        self.project_state.active_model_changed.connect(self._refresh_context)
        self._sync_workflow_step()
        self.apply_preferences(preferences_store.load(), startup=True)
        self.refresh_checks()

    def show_step(self, step: str) -> None:
        """Open the workspace of a workflow step (``data`` … ``predict``)."""
        if step in ("data", "rois") and self._workflow.current() == "values" \
                and self._roi_table_in_front():
            # The Values step brought the (tall) ROI table to the front; the next step
            # gets the Checks and its page's height back (Model and Predict: on the
            # workspace tab change).
            self._checks_dock.raise_()
        if step in ("data", "rois", "values"):
            self._workspace_tabs.setCurrentWidget(self._eda_view)
            if step == "data":
                self._explore_tabs.setCurrentWidget(self._data_file_page)
            elif self._project_rois is not None:
                self._explore_tabs.setCurrentWidget(self._project_rois)
            if step == "values":
                self.show_roi_table()
        elif step == "model":
            self._workspace_tabs.setCurrentWidget(self._modeling_view)
        elif step == "predict":
            self._workspace_tabs.setCurrentWidget(self._predict_tab)
        self._workflow.set_current(step)
        self._sync_workflow_step()
        self._refresh_context()

    def _sync_workflow_step(self, *_args: Any) -> None:
        """Mark the step of the shown workspace; title the Settings dock after it."""
        if not hasattr(self, "_workflow"):
            return
        current = self._workspace_tabs.currentWidget()
        index = self._workspace_tabs.currentIndex()
        self._controls_dock.setWindowTitle(_SETTINGS_TITLES[max(0, index)])
        if current is self._modeling_view:
            step = "model"
        elif current is self._predict_tab:
            step = "predict"
        elif self._on_data_file_page():
            step = "data"
        else:
            step = "values" if self._workflow.current() == "values" else "rois"
        self._workflow.set_current(step)
        self._refresh_context()

    def refresh_checks(self) -> None:
        """Ask the library for the project's checks and verdicts and show them.

        The checks that need no prediction are shown at once; the prediction-ROI
        checks (``predict.rois_outside``) predict the prediction ROIs and come from
        a background job, which replaces the Predict step's checks when it ends.
        """
        self._checks_generation += 1
        project = self.project_state.project
        if project is None:
            self._checks_panel.set_checks([])
            self._models_panel.set_models([])
            self._workflow.set_states({})
            self._checks_dock.setWindowTitle("Checks")
            return
        try:
            # The file check only once the background lookup has answered: looking up
            # here would wait for every data file's drive on the UI thread.
            checks = project.checks(files=not self.project_state.data_files_unknown(),
                                    predict=False)
        except Exception as exc:  # a check must never take the window down
            logger.exception("Checks failed: {}", exc)
            checks = []
        predicting = any(check.check_id == "predict.rois_ready" for check in checks)
        self._show_checks(project, checks, keep_predict=predicting)
        if predicting:
            generation = self._checks_generation
            start_background_job(
                owner=self, fn=lambda: prediction_roi_checks(project),
                on_success=lambda result: self._apply_prediction_checks(
                    generation, project, checks, result),
                on_error=lambda message: logger.warning("Prediction ROI checks failed: {}",
                                                        message),
                on_finished=lambda: None)

    def _apply_prediction_checks(self, generation: int, project: Any, checks: list[Check],
                                 result: list[Check]) -> None:
        """Show the background prediction-ROI checks, unless a newer refresh came first."""
        if generation != self._checks_generation or self.project_state.project is not project:
            logger.debug("Prediction ROI checks of an earlier refresh dropped")
            return
        self._show_checks(project, [check for check in checks if check.step != "predict"]
                          + list(result))

    def _show_checks(self, project: Any, checks: list[Check], *,
                     keep_predict: bool = False) -> None:
        """Show ``checks``: the Checks panel, the workflow strip and the Models panel.

        ``keep_predict``: the Predict step's checks are still being computed (a
        background job), so its open checks seen before stay "seen"; otherwise
        each refresh would find them new again and bring the Checks tab forward.
        """
        names = {run_id: (run.name or run_id) for run_id, run in
                 self.project_state.model_runs.items()}
        before = getattr(self, "_seen_open_checks", set())
        self._checks_panel.set_checks(checks, names)
        open_count = self._checks_panel.open_count()
        opened = {check.check_id for check in checks if check.is_open}
        new_open = opened - before
        self._seen_open_checks = opened | (
            {check_id for check_id in before if check_id.startswith("predict.")}
            if keep_predict else set())
        if new_open and hasattr(self, "_checks") and getattr(
                self, "_preferences", Preferences()).raise_checks:
            self.show_checks()
        self._checks_dock.setWindowTitle(f"Checks ({open_count})" if open_count else "Checks")
        states: dict[str, tuple[str, int]] = {}
        for step in ("data", "rois", "values", "model", "predict"):
            mine = [check for check in checks if check.step == step]
            blocking = sum(1 for check in mine if check.severity == "blocking")
            opened = sum(1 for check in mine if check.is_open)
            if blocking:
                states[step] = ("blocked", opened)
            elif opened:
                states[step] = ("open", opened)
            elif any(check.severity == "ok" for check in mine):
                states[step] = ("done", 0)
        if self.project_state.predictions:
            states["predict"] = states.get("predict", ("done", 0))
            if states["predict"][0] == "idle":
                states["predict"] = ("done", 0)
        self._workflow.set_states(states)
        self._refresh_models_panel(project)
        self._checks = checks
        self._refresh_context()
        self.checks_refreshed.emit()

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def menu_shortcuts(self) -> list[tuple[str, str]]:
        """``(menu item, shortcut)`` of every menu item with a shortcut, in menu order
        (submenus included; an item in two menus once)."""
        found: list[tuple[str, str]] = []
        seen: set[int] = set()

        def walk(menu: QMenu, path: str) -> None:
            for action in menu.actions():
                sub = action.menu()
                if sub is not None:
                    walk(sub, f"{path} > {action.text().replace('&', '')}")
                elif not action.shortcut().isEmpty() and id(action) not in seen:
                    seen.add(id(action))
                    found.append((f"{path} > {action.text().replace('&', '')}",
                                  action.shortcut().toString()))

        menu_bar = _require_qt_object(self.menuBar(), "main menu bar")
        for top in menu_bar.actions():
            menu = top.menu()
            if isinstance(menu, QMenu):
                walk(menu, top.text().replace("&", ""))
        return found

    def edit_preferences(self) -> bool:
        """Edit > Preferences; ``True`` when the user applied new preferences."""
        dialog = PreferencesDialog(self._preferences, shortcuts=self.menu_shortcuts(),
                                   parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        preferences = dialog.values()
        preferences_store.save(preferences)
        self.apply_preferences(preferences)
        return True

    def apply_preferences(self, preferences: Any, *, startup: bool = False) -> None:
        """Make ``preferences`` take effect; defaults pre-fill the docks only at startup."""
        self._preferences = preferences
        preferences_store.apply_environment(preferences)
        if getattr(self, "_log_sink_level", None) != preferences.log_level:
            self._attach_log_sink(preferences.log_level)
        if not startup:
            dark = preferences.theme == "dark"
            if preferences.theme == "system":
                app = QApplication.instance()
                hints = app.styleHints() if isinstance(app, QApplication) else None
                dark = hints is not None and hints.colorScheme() == Qt.ColorScheme.Dark
            if dark != is_dark():
                self._set_theme_mode(dark)
        self._apply_plot_help()
        if startup:  # pages built later get the defaults when they are built
            form = self._modeling_view.fit_form
            if form is not None:
                _apply_fit_defaults(form, preferences)
            if self._prediction_page is not None:
                _apply_predict_defaults(self._prediction_page, preferences)

    def show_checks(self) -> None:
        """Bring the Checks tab of the bottom panel to the front."""
        self._checks_dock.show()
        self._checks_dock.raise_()

    def _update_strip_context(self) -> None:
        """The project and the model shown, at the right end of the workflow strip."""
        path = self.project_state.project_path
        run = None if path is None else self._modeling_view._resolved_workspace_model_run()
        from ._modeling import model_label

        self._workflow.set_context("" if path is None else os.path.basename(path),
                                   "" if run is None else model_label(run))

    def _refresh_context(self, *_args: Any) -> None:
        """What the window works on: the strip's project and model, the verdict of the model."""
        if not hasattr(self, "_workflow"):
            return
        self._update_strip_context()
        self._update_verdict_row()

    def _update_verdict_row(self) -> None:
        """The verdict line of the Fit summary: the library's level of the model shown."""
        if not hasattr(self, "_model_verdict_value"):
            return  # the Model page is built when first shown
        run = self._modeling_view._resolved_workspace_model_run()
        verdict = None if run is None else self.model_verdict(run.model_run_id)
        self._model_verdict_level = "" if verdict is None else verdict.level
        if verdict is None:
            self._model_verdict_icon.clear()
            self._model_verdict_value.setText("–")
            self._model_verdict_value.setToolTip(wrap_tooltip(""))
            return
        mark, text = verdict_mark(verdict.level)
        self._model_verdict_icon.setPixmap(mark.pixmap(16, 16))
        self._model_verdict_value.setText(text)
        self._model_verdict_value.setToolTip(wrap_tooltip(verdict.summary))

    def _refresh_models_panel(self, project: Any) -> None:
        from .._workflow import Model

        rows = []
        for run in self.project_state.model_runs.values():
            if run.retired_at:  # in the Project tree's Retired models, not here
                continue
            model = Model(project, run)
            try:
                verdict = model.verdict()
            except Exception as exc:
                logger.exception("Verdict of model '{}' failed: {}", model.name, exc)
                verdict = None
            spec = run.fit_spec.get("folds", {}) if isinstance(run.fit_spec, dict) else {}
            validation = (f"{spec.get('rule', '')}".replace("_", " ")
                          + (f", by {spec.get('group_by')}" if spec.get("group_by") else ""))
            rows.append({"id": run.model_run_id, "name": model.name, "type": run.model_type,
                         "components": run.hyperparameters.get("n_components"),
                         "metrics": run.metrics, "validation": validation.strip(", "),
                         "verdict": verdict, "fitted": str(run.created_at)[:16].replace("T", " ")})
        self._models_panel.set_models(rows)

    def model_verdict(self, model_run_id: str) -> Any:
        """The library's verdict of a model (for the verdict bar), or ``None``."""
        project = self.project_state.project
        if project is None or model_run_id not in self.project_state.model_runs:
            return None
        from .._workflow import Model

        try:
            return Model(project, self.project_state.model_runs[model_run_id]).verdict()
        except Exception as exc:
            logger.exception("Verdict failed: {}", exc)
            return None

    def select_model(self, model_run_id: str) -> None:
        """Show a model in the Model workspace."""
        self.show_step("model")
        self._modeling_view.select_model_run(model_run_id)
        self.schedule_workspace_content_refresh()
        self._refresh_context()

    def _show_check(self, check: Check) -> None:
        """Show what a check refers to: its model, ROI or step."""
        if check.model_run_id:
            self.select_model(check.model_run_id)
        elif check.roi_ids:
            self.show_step("values")
            self._on_canvas_roi_selected(check.roi_ids[0])
        else:
            self.show_step(check.step)

    def run_fix(self, fix: Fix) -> None:
        """Run the action a check offers (the viewer's own actions, pre-filled)."""
        args = dict(fix.arguments)
        action = fix.action
        logger.info("Fix: {} {}", action, args)
        if action == "add_dataset":
            self.add_data_file()
        elif action == "locate_dataset":
            self.locate_data_file(str(args["dataset_id"]))
        elif action == "set_wavelengths":
            self.set_wavelengths(tuple(args.get("dataset_ids") or ()), args.get("table"))
        elif action == "add_roi":
            self.add_rois_on("add_roi")  # shows a dataset, then starts drawing
        elif action == "add_target":
            self.show_step("model")
            self._modeling_view.select_target_btn_clicked()
        elif action in ("edit_values", "edit_samples"):
            self.show_step("values")
            roi_ids = args.get("roi_ids") or ()
            if roi_ids:
                self._on_canvas_roi_selected(str(roi_ids[0]))
        elif action == "select_roi":
            self.show_step("values")
            self._on_canvas_roi_selected(str(args["roi_id"]))
        elif action == "fit":
            self.show_step("model")
            self._ensure_fit_settings()
            form = self._modeling_view.fit_form
            if form is not None and args.get("target_id"):
                form.set_targets(self._modeling_view._target_choices(), str(args["target_id"]))
        elif action in ("fit_again", "refit"):
            run_id = str(args["model_run_id"])
            run = self.project_state.model_runs.get(run_id)
            if run is None:
                return
            self.select_model(run_id)
            self._ensure_fit_settings()
            self._modeling_view.load_model_settings(run)
            form = self._modeling_view.fit_form
            if form is not None and args.get("group_by"):
                combo = form._group_by_combo
                combo.setCurrentIndex(max(0, combo.findData(args["group_by"])))
        elif action == "show_page":
            self.select_model(str(args["model_run_id"]))
            self._ensure_model_content_page()
            widget = self._model_diagnostics_widget
            if widget is not None and args.get("page") in widget.grid.page_names:
                widget.grid.set_current_page(str(args["page"]))
        elif action == "predict_rois":
            self.show_step("predict")
            self._prediction_view._btn_predict_rois.click()
        else:
            logger.warning("Unknown fix action '{}'", action)

    def _show_saved_message(self, message: str) -> None:
        """Confirm a project write in the status bar for a few seconds."""
        self._saved_label.setText(f"✓ {message}")
        self._saved_clear.start(_SAVED_MESSAGE_MS)

    def _viewer_busy_context(self, text: str) -> ViewerBusyContext:
        """Return a status-bar backed wait cursor context for blocking UI work."""
        status_bar = _require_qt_object(self.statusBar(), "main status bar")
        return ViewerBusyContext(
            text,
            status_setter=lambda message: status_bar.showMessage(message),
            status_clearer=status_bar.clearMessage,
        )

    def _populate_file_menu(self) -> None:
        m = self._menu_file

        act = _add_required_action(m, "New Project...")
        act.setShortcut("Ctrl+N")
        act.triggered.connect(self._eda_view.new_project_btn_clicked)

        act = _add_required_action(m, "Open Project...")
        act.setShortcut("Ctrl+O")
        act.triggered.connect(self._eda_view.open_project_btn_clicked)

        act = _add_required_action(m, "Open Example Project")
        act.setToolTip(wrap_tooltip("Five synthetic tablet scans with a fitted model, created on first use"))
        act.triggered.connect(self.open_example_project)

        self._menu_open_recent = _add_required_menu(m, "Open Recent")
        self._menu_open_recent.aboutToShow.connect(self._populate_open_recent)

        m.addSeparator()

        act = _add_required_action(m, "Export ROI Spectra (CSV)...")
        act.triggered.connect(self._eda_view.export_csv_btn_clicked)
        act = _add_required_action(m, "Export Plots...")
        act.setToolTip(wrap_tooltip("Save every plot of the current workspace as PNG and CSV"))
        act.triggered.connect(self._action_export_plots)

        m.addSeparator()
        act = _add_required_action(m, "Compact Project...")
        act.setToolTip(wrap_tooltip("Give back the space of retired models' diagnostics and "
                                    "removed datasets and ROIs: the project file shrinks"))
        act.triggered.connect(lambda: self.compact_project())

        m.addSeparator()

        act = _add_required_action(m, "Quit")
        act.setShortcut("Ctrl+Q")
        act.triggered.connect(self._quit_application)

    def open_example_project(self) -> None:
        """Create the example project (first use) and open it in this window."""
        from .._example import create_example_project
        from ._recent import config_dir

        busy = self._viewer_busy_context("Preparing the example project")
        busy.begin()
        try:
            path = create_example_project(config_dir() / "example")
        except (OSError, ValueError) as exc:
            logger.error("Cannot create the example project: {}", exc)
            QMessageBox.warning(self, "Example project", str(exc))
            return
        finally:
            busy.end()
        self._open_project_btn_clicked_with_path(str(path))

    def _open_data_path(self, path: str) -> None:
        if self.project_state.project_path is not None:
            # A project window adds the file to its project (attachment contract).
            self.add_data_file(path)
            return
        from enginir.viewer import open_viewer

        busy = self._viewer_busy_context(f"Opening dataset: {path}")
        busy.begin()
        try:
            child = open_viewer(path)
        except Exception as exc:
            busy.end()
            logger.error("Failed to open {}: {}", path, exc)
            QMessageBox.critical(self, "Cannot open file", str(exc))
            return
        finally:
            busy.end()
        logger.debug("Opened {} in a new window {}", path, type(child).__name__)

    # ------------------------------------------------------------------
    # Data files of the project (attachment contract, Data Files Of A Project)
    # ------------------------------------------------------------------

    def add_data_file(self, path: str | None = None, *,
                      trained_with: str | None = None, binning: int = 1) -> str | None:
        """Add a data file to the project with its reference correction, then show it.

        A file that is already registered is shown directly. Returns the
        dataset id, or ``None`` when nothing was added.
        """
        if self.project_state.project_path is None:
            QMessageBox.warning(self, "No project open",
                                "Create or open a project before adding datasets.")
            return None
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Add dataset", preferences_store.start_folder("dataset"),
                "Hyperspectral files (*.eiff *.hdr *.h5 *.hdf5 *.csv *.spa);;All files (*.*)")
            if not path:
                return None
        registered = self._registered_dataset_id(path)
        if registered is not None:
            self.show_data_file(registered)
            return registered
        try:
            header = read_header(os.path.abspath(path))
        except (OSError, ValueError, KeyError) as exc:
            logger.error("Cannot read dataset {}: {}", path, exc)
            QMessageBox.critical(self, "Cannot open file", str(exc))
            return None
        state = self.project_state
        dialog = AddDataFileDialog(
            path, header, state.datasets,
            {dataset_id: reference_correction for dataset_id in state.datasets
             if (reference_correction := state.active_reference_correction(dataset_id)) is not None},
            trained_with=trained_with, binning=binning, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.request is None:
            return None
        request = dialog.request
        try:
            dataset_id = state.add_data_file(request.path, **request.add_dataset_kwargs())
        except (OSError, ValueError, KeyError) as exc:
            logger.error("Cannot add dataset {}: {}", path, exc)
            QMessageBox.critical(self, "Cannot add dataset", str(exc))
            return None
        self.show_data_file(dataset_id)
        return dataset_id

    def add_rois_on(self, action: str, dataset_id: str | None = None) -> None:
        """Add ROI (``"add_roi"``) or Add ROI Grid (``"add_roi_grid"``) from anywhere.

        Shows the Data file tab and a dataset first when none is shown, or when
        ``dataset_id`` is another one: ``dataset_id``, else the last shown, else
        the only one, else the one the user picks; with none, Add Dataset. Then
        starts drawing or opens the grid dialog.
        """
        if action not in ("add_roi", "add_roi_grid"):
            log_and_raise("Unknown ROI action {!r}", action, error=ValueError)
        if self.project_state.project is None:
            return
        self.show_step("data")
        shown = self._eda_view.shown_dataset_id()
        if shown is not None and dataset_id in (None, shown):
            self._start_roi_action(action)
            return
        target = dataset_id or self._dataset_for_rois()
        if target is None and not self.project_state.datasets:
            self._after_shown = ("", action)  # the dataset Add Dataset adds and shows
            added = self.add_data_file()
            if added is None:
                self._after_shown = None
            elif self._after_shown is not None:
                self._after_shown = (added, action)
            return
        if target is None:
            return  # the user chose no dataset
        self._after_shown = (target, action)
        self.show_data_file(target)

    def _start_roi_action(self, action: str) -> None:
        if not self.has_attached_dataset():
            return
        if action == "add_roi":
            self._eda_view._btn_roi_training.setChecked(True)
        else:
            self._eda_view.open_roi_grid_dialog()

    def _dataset_for_rois(self) -> str | None:
        """The dataset to draw ROIs on: the last shown, the only one, or the user's choice."""
        state = self.project_state
        project_id = state.project_meta.get("project_id")
        last = None if not project_id else load_last_data_file(str(project_id))
        if last in state.datasets:
            return last
        if len(state.datasets) == 1:
            return next(iter(state.datasets))
        if not state.datasets:
            return None
        ids = sorted(state.datasets, key=lambda d: state.data_file_label(d).casefold())
        labels = [state.data_file_label(d) for d in ids]
        choice, ok = QInputDialog.getItem(self, "Add ROIs", "Dataset to draw ROIs on:",
                                          labels, 0, False)
        return ids[labels.index(choice)] if ok and choice in labels else None

    def show_data_file(self, dataset_id: str) -> None:
        """Load a registered data file in the background and show it with its reference correction."""
        dataset = self.project_state.datasets.get(dataset_id)
        if dataset is None:
            log_and_raise("Unknown dataset '{}'", dataset_id, error=KeyError)
        path = self.project_state.data_file(dataset_id)
        if path is None:
            box = QMessageBox(QMessageBox.Icon.Warning, "Dataset file not found",
                              f"{dataset.source_path} was not found, also not next to the "
                              "project or on the mapped drive. Its ROIs, values and models stay "
                              "usable; the image cannot be shown.\n\nIf the file was moved or "
                              "renamed, locate it: it is accepted if its pixels equal the "
                              "stored ROI pixels.", parent=self)
            locate = box.addButton("Locate File...", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is locate:
                self.locate_data_file(dataset_id)
            else:
                self._after_shown = None  # Add ROI waited for this dataset
            return
        if self._data_file_job is not None:
            return
        busy = self._viewer_busy_context(f"Loading dataset: {os.path.basename(path)}")
        busy.begin()

        def finished() -> None:
            busy.end()
            self._data_file_job = None

        record = self.project_state.datasets.get(dataset_id)
        binning = 1 if record is None else record.binning
        self._data_file_job = start_background_job(
            owner=self, fn=lambda: load_spectra(path, binning=binning),
            on_success=lambda data: self._apply_data_file(dataset_id, data),
            on_error=self._data_file_failed,
            on_finished=finished)

    def locate_data_file(self, dataset_id: str, path: str | None = None) -> bool:
        """Point a moved data file at its new location, then show it.

        Without ``path`` a file dialog asks for it. The file is accepted only
        if it is the same data file (shape, wavelengths and the pixels of
        every ROI); otherwise nothing changes.
        """
        dataset = self.project_state.datasets.get(dataset_id)
        if dataset is None:
            log_and_raise("Unknown dataset '{}'", dataset_id, error=KeyError)
        old_path = dataset.source_path
        if path is None:
            name = os.path.basename(old_path)
            folder = os.path.dirname(old_path)
            if not os.path.isdir(folder):  # e.g. a path of the other platform
                folder = os.path.dirname(str(self.project_state.project_path or ""))
            path, _ = QFileDialog.getOpenFileName(
                self, f"Locate {name}", folder,
                f"{name} (*{os.path.splitext(name)[1]});;All files (*)")
            if not path:
                return False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.project_state.locate_data_file(dataset_id, path)
        except (OSError, ValueError, KeyError) as exc:
            logger.error("Dataset file not located at {}: {}", path, exc)
            QMessageBox.warning(self, "Not this dataset's file", str(exc))
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self._locate_moved_data_files(old_path, path)
        self.show_data_file(dataset_id)
        return True

    def _locate_moved_data_files(self, old_path: str, new_path: str) -> None:
        """Offer the folder change of one Locate for the other data files not found."""
        moved = self.project_state.moved_data_files(old_path, new_path)
        if not moved:
            return
        answer = QMessageBox.question(
            self, "Locate other dataset files",
            f"{len(moved)} more dataset file(s) that were not found are in the same new place "
            f"(for example {next(iter(moved.values()))}).\n\nLocate them too? Each file is "
            "checked against its stored ROI pixels.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        failed: list[str] = []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for dataset_id, candidate in moved.items():
                try:
                    self.project_state.locate_data_file(dataset_id, candidate)
                except (OSError, ValueError, KeyError) as exc:
                    logger.warning("Dataset file not located at {}: {}", candidate, exc)
                    failed.append(f"{self.project_state.data_file_label(dataset_id)}: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
        if failed:
            QMessageBox.warning(
                self, "Some dataset files not located",
                f"{len(moved) - len(failed)} of {len(moved)} located. Not located:\n"
                + "\n".join(failed))

    def _data_file_failed(self, message: str) -> None:
        self._after_shown = None
        logger.error("Cannot load dataset: {}", message)
        QMessageBox.critical(self, "Cannot open file", message)

    def wait_for_data_file(self, timeout_s: float = 30.0) -> None:
        """Process events until a data file load has finished (for scripts and tests).

        Also waits for the background lookup of the project's data files, which
        decides whether the last shown file is loaded and which files are found.
        """
        deadline = time.monotonic() + timeout_s
        app = QApplication.instance()
        while (self._data_file_job is not None or getattr(self, "_data_file_lookup", False)) \
                and time.monotonic() < deadline:
            if app is not None:
                app.processEvents()
            time.sleep(0.005)

    def _apply_data_file(self, dataset_id: str, data: Any) -> None:
        # Where the file was found this session (possibly not its stored path).
        path = self.project_state.data_file(dataset_id) or \
            self.project_state.datasets[dataset_id].source_path
        X = np.asarray(data.X)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        if not confirm_project_dataset_binding(
            self,
            project_path=str(self.project_state.project_path),
            project_state=self.project_state,
            dataset_path=path,
            source_format=file_format(path),
            source_modality=str(data.meta.get("source_modality") or DEFAULT_SOURCE_MODALITY),
            cube_shape=(int(X.shape[0]), int(X.shape[1]), int(X.shape[2])),
            wavelengths=spectral_axis_in_nm(data.wavelengths, data.meta)[0],
        ):
            return
        self.state.load_dataset(X, *_shown_axis(self.project_state.datasets[dataset_id], data))
        self._cube_path = os.path.abspath(path)
        self.apply_shown_reference_correction()
        self._refresh_canvas_image()
        self._eda_view.refresh_roi_overlays()
        self._eda_view.refresh_attachment_state()
        self._update_status_strip()
        self._refresh_data_file_choices()
        if self._pending_roi_selection is not None:
            pending, self._pending_roi_selection = self._pending_roi_selection, None
            self._canvas.select_roi(pending)  # also when the browser already selects it
            self._on_canvas_roi_selected(pending)
        else:
            self._centre_on_first_roi(dataset_id)
        from ._recent import add as _recent_add

        _recent_add(path, "dataset")
        project_id = self.project_state.project_meta.get("project_id")
        if project_id:
            save_last_data_file(str(project_id), dataset_id)
        logger.info("Showing dataset {} ({})", path, self.state.y_unit)
        pending, self._after_shown = getattr(self, "_after_shown", None), None
        if pending is not None and pending[0] == dataset_id:
            QTimer.singleShot(0, lambda: self._start_roi_action(pending[1]))

    def apply_shown_reference_correction(self) -> None:
        """Convert the shown data file with its stored reference correction (references or levels).

        The Explore signal stays the one the user chose when this file can
        give it; otherwise (or before any choice) it is the reference correction's
        working signal. Only the signals the file can give are enabled.
        """
        dataset_id = self._eda_view.shown_dataset_id()
        if dataset_id is None:
            self._eda_view.set_signal_choices(viewable_signal_units(None))
            # No dataset shown (Project ROIs): start with the working signal of the
            # stored ROIs' reference corrections, not with the raw counts
            # (dataset-reference-correction contract, Signal units of a reference correction).
            unit = self._stored_rois_working_signal()
            if unit is not None and self._eda_view.chosen_signal is None:
                self.state.set_y_unit(unit)
            return
        reference_correction = self.project_state.active_reference_correction(dataset_id)
        levels, unit = _reference_correction_view(reference_correction, self.state.meta)
        allowed = viewable_signal_units(reference_correction)
        if self._eda_view.chosen_signal in allowed:
            unit = str(self._eda_view.chosen_signal)
        self.state.set_reference_correction(
            **levels, reference_frames=own_reference_frames(reference_correction, self._cube_path))
        self.state.set_y_unit(unit)
        self._eda_view.set_signal_choices(allowed)

    def _stored_rois_working_signal(self) -> str | None:
        """The working signal most of the stored ROIs' reference corrections give, if any."""
        state = self.project_state
        units = Counter(
            reference_correction.output_signal_unit for roi in state.rois.values()
            if (reference_correction := state.reference_corrections.get(roi.reference_correction_id)) is not None)
        return units.most_common(1)[0][0] if units else None

    def use_registered_axis(self) -> None:
        """Show the file with its dataset's stored axis when a wavelength table gave it one.

        For a file shown before its project was attached, and after Set
        Wavelengths; a file loaded through :meth:`show_data_file` has it already.
        """
        if not self._cube_path or self.state.cube is None:
            return
        dataset_id = self._registered_dataset_id(self._cube_path)
        dataset = None if dataset_id is None else self.project_state.datasets.get(dataset_id)
        if dataset is None or not dataset.wavelength_table:
            return
        if np.allclose(self.state.wavelengths, dataset.wavelengths, atol=1e-6, rtol=0.0):
            return
        logger.info("Showing {} with the wavelengths of table '{}'", self._cube_path,
                    dataset.wavelength_table)
        self.state.use_axis(dataset.wavelengths, {**self.state.meta, "x_units": "nm",
                                                  "wavelength_table": dataset.wavelength_table})
        self._refresh_canvas_image()
        self._update_status_strip()

    def _registered_dataset_id(self, path: str) -> str | None:
        return self.project_state.dataset_id_of(path)

    def _refresh_data_file_choices(self, *_args) -> None:
        combo = self._data_file_combo
        combo.blockSignals(True)
        combo.clear()
        shown = self._eda_view.shown_dataset_id()
        state = self.project_state
        datasets = sorted(state.datasets.items(),
                          key=lambda item: state.data_file_label(item[0]).lower())
        for dataset_id, dataset in datasets:
            name = state.data_file_label(dataset_id)
            # The tooltip is the path the file is read from; how it was found is logged only.
            if not state.data_file_known(dataset_id):
                combo.addItem(f"{name} (checking…)", dataset_id)
                combo.setItemData(combo.count() - 1, f"{dataset.source_path} (looking for it)",
                                  Qt.ItemDataRole.ToolTipRole)
                continue
            found = state.find_data_file(dataset_id)
            if found is None:
                name += " (not found)"
                tooltip = f"{dataset.source_path} (not found)"
            else:
                tooltip = found.path
            combo.addItem(name, dataset_id)
            combo.setItemData(combo.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole)
        if shown is None:
            label = ("Dataset: none" if not self._cube_path
                     else f"{os.path.basename(self._cube_path)} (not in project)")
            combo.insertItem(0, label, None)
        combo.insertSeparator(combo.count())
        combo.addItem("Add Dataset...", _ADD_DATA_FILE)
        combo.setCurrentIndex(max(0, combo.findData(shown)) if shown is not None else 0)
        combo.setEnabled(self.project_state.project_path is not None)
        combo.blockSignals(False)

    def _on_data_file_chosen(self, index: int) -> None:
        choice = self._data_file_combo.itemData(index)
        if choice == _ADD_DATA_FILE:
            self._refresh_data_file_choices()  # back to the shown file until one is added
            self.add_data_file()
        elif isinstance(choice, str) and choice != self._eda_view.shown_dataset_id():
            self.show_data_file(choice)

    def _on_browser_item_activated(self, item: Any, _column: int = 0) -> None:
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(payload, dict) and payload.get("kind") == "dataset":
            self.show_dataset(str(payload.get("id")))

    def show_dataset(self, dataset_id: str) -> None:
        """Show a dataset's image: load it and bring the Data step to the front (review A5)."""
        self.show_data_file(dataset_id)  # first, so the tab switch does not load another file
        self.show_step("data")

    def _centre_on_first_roi(self, dataset_id: str) -> None:
        """Put the crosshair on the dataset's first ROI (not on a reference frame)."""
        rois = [roi for roi in self.project_state.rois.values() if roi.dataset_id == dataset_id]
        if rois and self.state.has_dataset:
            f0, f1, r0, r1 = rois[0].spatial_bounds
            self.state.set_frame((f0 + f1) // 2)
            self.state.set_row((r0 + r1) // 2)

    def set_wavelengths(self, dataset_ids: tuple[str, ...] = (),
                        table: str | None = None) -> bool:
        """Project > Set Wavelengths: a wavelength table for datasets with band numbers.

        ``dataset_ids`` and ``table`` preselect the dialog (the check's fix);
        ``True`` when wavelengths were set.
        """
        state = self.project_state
        if state.project is None:
            return False
        if not any(is_band_numbers(ref.wavelengths) for ref in state.datasets.values()):
            QMessageBox.information(self, "Set wavelengths",
                                    "Every dataset of the project has wavelengths.")
            return False
        dialog = SetWavelengthsDialog(state.datasets, state.data_file_labels,
                                      preselected=dataset_ids, table=table, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.choice is None:
            return False
        name, chosen = dialog.choice
        try:
            state.set_dataset_wavelengths(chosen, name)
        except (OSError, ValueError, KeyError) as exc:
            logger.error("Cannot set the wavelengths of {}: {}", chosen, exc)
            QMessageBox.critical(self, "Cannot set wavelengths", str(exc))
            return False
        self.use_registered_axis()
        return True

    def _choose_model(self, title: str) -> str | None:
        """The model to act on: the one selected on the Model page, else asked for."""
        state = self.project_state
        runs = [run for run in state.model_runs.values() if not run.retired_at]
        if not runs:
            QMessageBox.information(self, title, "The project has no model.")
            return None
        from PyQt6.QtWidgets import QInputDialog

        labels = [run.name or run.model_run_id for run in runs]
        ids = [run.model_run_id for run in runs]
        selected = self._modeling_view.selected_model_run_id
        current = ids.index(selected) if selected in ids else 0
        label, ok = QInputDialog.getItem(self, title, "Model:", labels, current, False)
        return ids[labels.index(label)] if ok else None

    def rename_model(self, model_run_id: str | None = None) -> bool:
        """Model > Rename Model: give a model another name (the model is its id)."""
        from PyQt6.QtWidgets import QInputDialog

        model_run_id = model_run_id or self._choose_model("Rename model")
        if model_run_id is None:
            return False
        run = self.project_state.model_runs[model_run_id]
        name, ok = QInputDialog.getText(self, "Rename model", "New name:", text=run.name or "")
        if not ok:
            return False
        try:
            self.project_state.rename_model(model_run_id, name)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot rename model", str(exc))
            return False
        return True

    def retire_model(self, model_run_id: str | None = None) -> bool:
        """Model > Retire Model: the model stays as a record but predicts no more (D1-D5)."""
        from PyQt6.QtWidgets import QInputDialog

        model_run_id = model_run_id or self._choose_model("Retire model")
        if model_run_id is None:
            return False
        name = self.project_state.model_runs[model_run_id].name or model_run_id
        reason, ok = QInputDialog.getText(
            self, "Retire model",
            f"Retire '{name}'?\n\nIt stays in the project with its metrics and predictive "
            "core, under Retired models, but predicts no more. Its per-pixel diagnostics "
            "(scores, T², DModX of every calibration pixel) are deleted. This cannot be "
            "undone; refitting gives the same model again as a new model.\n\n"
            "Reason (optional):")
        if not ok:
            return False
        try:
            self.project_state.retire_model(model_run_id, reason)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot retire model", str(exc))
            return False
        return True

    def compact_project(self) -> bool:
        """File > Compact Project: rewrite the project file without its free space."""
        if self.project_state.project_path is None:
            QMessageBox.information(self, "Compact project", "No project is open.")
            return False
        try:
            before, after = self.project_state.compact_project()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot compact project", str(exc))
            return False
        QMessageBox.information(self, "Compact project",
                                f"The project file is {after / 1e6:.1f} MB (was "
                                f"{before / 1e6:.1f} MB).")
        return True

    def remove_dataset(self, dataset_id: str | None = None) -> bool:
        """Project > Remove Dataset: remove a dataset without ROIs; the file stays on disk."""
        from PyQt6.QtWidgets import QInputDialog

        state = self.project_state
        if not state.datasets:
            QMessageBox.information(self, "Remove dataset", "The project has no dataset.")
            return False
        if dataset_id is None:
            choices = list(state.datasets)
            labels = [state.data_file_label(d) for d in choices]
            shown = self._eda_view.shown_dataset_id()
            current = choices.index(shown) if shown in choices else 0
            label, ok = QInputDialog.getItem(self, "Remove dataset", "Dataset:", labels,
                                             current, False)
            if not ok:
                return False
            dataset_id = choices[labels.index(label)]
        path = state.datasets[dataset_id].source_path
        answer = QMessageBox.question(
            self, "Remove dataset",
            f"Remove {state.data_file_label(dataset_id)} ({path}) and its reference "
            "corrections from the project?\n\nThe file is not touched; it can be added "
            "again.")
        if answer != QMessageBox.StandardButton.Yes:
            return False
        try:
            state.remove_dataset(dataset_id)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot remove dataset", str(exc))
            return False
        return True

    def name_dataset(self, dataset_id: str | None = None) -> bool:
        """Project > Rename Dataset: give a dataset a short display name ("Layer 7")."""
        from PyQt6.QtWidgets import QInputDialog

        state = self.project_state
        if not state.datasets:
            QMessageBox.information(self, "Name dataset", "The project has no dataset.")
            return False
        if dataset_id is None:
            shown = self._eda_view.shown_dataset_id()
            choices = list(state.datasets)
            labels = [state.data_file_label(d) for d in choices]
            current = choices.index(shown) if shown in choices else 0
            label, ok = QInputDialog.getItem(self, "Name dataset", "Dataset:", labels,
                                             current, False)
            if not ok:
                return False
            dataset_id = choices[labels.index(label)]
        ref = state.datasets[dataset_id]
        name, ok = QInputDialog.getText(
            self, "Name dataset",
            f"Name of {os.path.basename(ref.source_path)} (empty: the file name):",
            text=ref.label)
        if not ok:
            return False
        state.set_dataset_label(dataset_id, name)
        return True

    def _refresh_canvas_image(self) -> None:
        """Rebuild the central false-color preview from the current viewer state."""
        s = self.state
        if not s.has_dataset:
            self._canvas.clear_image()
            return
        try:
            self._canvas.set_image(self._false_colour_rgb())
        except Exception:
            # The rest of the viewer stays usable; the log keeps the reason the canvas is empty.
            logger.opt(exception=True).warning("False-color preview could not be drawn")
            self._canvas.clear_image()
        self._canvas.set_crosshair(s.frame, s.row)

    def _populate_open_recent(self) -> None:
        """The recent files, drawn from the list alone (no file access on the UI thread).

        A file the last background check found missing is grayed out; each opening of
        the menu checks again in the background and updates the entries still shown.
        """
        from . import _recent

        menu = self._menu_open_recent
        menu.clear()
        recent = _recent.entries()
        if not recent:
            act = _add_required_action(menu, "(No recent files)")
            act.setEnabled(False)
            return
        known = _recent.last_check()
        actions = {}
        projects = [entry for entry in recent if entry.kind == "project"]
        others = [entry for entry in recent if entry.kind != "project"]
        for entry in [*projects, *others]:
            if entry is (others[0] if others else None) and projects:
                menu.addSeparator()  # recent projects above, recent data files below
            act = _add_required_action(menu, entry.path)
            act.setEnabled(known.get(entry.path, (True, None))[0])
            act.triggered.connect(lambda checked, p=entry.path: self._open_data_path(p))
            actions[entry.path] = act

        def checked(result: object) -> None:
            if not isinstance(result, dict) or sip.isdeleted(menu):
                return
            for path, act in actions.items():
                if not sip.isdeleted(act) and act in menu.actions():
                    act.setEnabled(bool(result.get(path, (True, None))[0]))

        # After the menu has popped up, not while it does.
        start_background_job_soon(owner=menu, fn=lambda: _recent.check(recent),
                                  on_success=checked,
                                  on_error=lambda message: logger.warning(
                                      "Recent files could not be checked: {}", message),
                                  on_finished=lambda: None)

    def _populate_view_menu(self) -> None:
        m = self._menu_view
        for dock, text in (
            (self._project_browser_dock, "Project"),
            (self._inspector_dock, "Properties"),
            (self._controls_dock, "Settings"),
            (self._log_dock, "Log"),
            (self._hist_dock, "Histogram"),
            (self._roi_table_dock, "ROI Table"),
        ):
            m.addAction(self._dock_toggle(dock, text))
        for action in self._page_plot_actions.values():
            m.addAction(action)
        m.addSeparator()
        go_to = _add_required_menu(m, "Go To")
        for number, step in enumerate(STEPS, 1):
            act = _add_required_action(go_to, step.label)
            act.setShortcut(f"Ctrl+{number}")
            act.setToolTip(wrap_tooltip(step.tooltip))
            act.triggered.connect(lambda _checked=False, key=step.key: self.show_step(key))
        m.addSeparator()
        theme_menu = _add_required_menu(m, "Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        self._theme_dark_action = _add_required_action(theme_menu, "Dark")
        self._theme_dark_action.setCheckable(True)
        self._theme_light_action = _add_required_action(theme_menu, "Light")
        self._theme_light_action.setCheckable(True)
        theme_group.addAction(self._theme_dark_action)
        theme_group.addAction(self._theme_light_action)
        self._theme_dark_action.setChecked(is_dark())
        self._theme_light_action.setChecked(not is_dark())
        self._theme_dark_action.triggered.connect(
            lambda checked: self._set_theme_mode(True) if checked else None
        )
        self._theme_light_action.triggered.connect(
            lambda checked: self._set_theme_mode(False) if checked else None
        )

        m.addSeparator()
        act = _add_required_action(m, "Reset View")
        act.triggered.connect(self._reset_image_view)
        act = _add_required_action(m, "Reset Layout")
        act.triggered.connect(self.reset_layout)

    def _dock_toggle(self, dock: QDockWidget, text: str) -> QAction:
        """A View item for a dock, in Title Case whatever the dock's title (plan
        menu-polish, V1): it follows the dock's own toggle (checked, enabled)."""
        source = _require_qt_object(dock.toggleViewAction(), f"{text} toggle action")
        action = QAction(text, self)
        action.setCheckable(True)

        def follow() -> None:
            action.setChecked(source.isChecked())
            action.setEnabled(source.isEnabled())

        follow()
        source.changed.connect(follow)
        action.triggered.connect(
            lambda checked: source.trigger() if source.isChecked() != checked else None)
        return action

    def _paste_focused(self) -> None:
        """Paste into the focused text or table: the ROI table pastes a column of values."""
        widget = QApplication.focusWidget()
        paste = getattr(widget, "paste", None)
        if callable(paste):
            paste()
            return
        if widget is not None and self._roi_table.isAncestorOf(widget):
            self._roi_table.paste_clipboard()

    def _populate_help_menu(self) -> None:
        m = self._menu_help

        act = _add_required_action(m, "User Guide")
        act.setToolTip(wrap_tooltip(f"The online documentation: {USER_GUIDE_URL}"))
        act.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(USER_GUIDE_URL)))
        act = _add_required_action(m, "Keyboard Shortcuts")
        act.triggered.connect(self._action_shortcuts)
        act = _add_required_action(m, "Report a Problem...")
        act.triggered.connect(self._action_report_problem)

        m.addSeparator()

        act = _add_required_action(m, "View Log")
        act.triggered.connect(self._action_view_log)

        act = _add_required_action(m, "Log File...")
        act.triggered.connect(self._action_log_file)

        m.addSeparator()

        act = _add_required_action(m, "About engiNIR")
        act.triggered.connect(self._action_about)

    def _action_shortcuts(self) -> None:
        """Every menu shortcut, read from the menus (so the list cannot go stale)."""
        from ._preferences import shortcut_table

        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard shortcuts")
        layout = QVBoxLayout(dialog)
        layout.addWidget(shortcut_table(self.menu_shortcuts()))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(480, 520)
        dialog.exec()

    def _action_report_problem(self) -> None:
        """Say what to attach, then open a new GitHub issue."""
        log_file = log_file_path()
        box = QMessageBox(self)
        box.setWindowTitle("Report a problem")
        attach = ("" if log_file is None else
                  f"<br><br>Please attach the log file:<br><b>{log_file}</b>")
        box.setText("Describe what you did and what happened in a new issue on GitHub."
                    + attach)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        open_button = box.addButton("Open GitHub Issue", QMessageBox.ButtonRole.AcceptRole)
        copy_button = (None if log_file is None else
                       box.addButton("Copy Log Path", QMessageBox.ButtonRole.ActionRole))
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is open_button:
            QDesktopServices.openUrl(QUrl(ISSUES_URL))
        elif copy_button is not None and clicked is copy_button:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(str(log_file))

    def _action_view_log(self) -> None:
        self._log_dock.setVisible(True)
        self._log_dock.raise_()

    def _action_log_file(self) -> None:
        """Show the log file path, with buttons to copy it and to open its location."""
        log_file = log_file_path()
        box = QMessageBox(self)
        box.setWindowTitle("Log file")
        if log_file is None:
            box.setText("No log file is active.")
            box.exec()
            return
        box.setText(f"engiNIR writes its log to:<br><br><b>{log_file}</b>")
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        copy_button = box.addButton("Copy Path", QMessageBox.ButtonRole.ActionRole)
        open_button = box.addButton("Open Location", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        clicked = box.clickedButton()
        if clicked is copy_button:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(str(log_file))
        elif clicked is open_button and not reveal_in_file_manager(log_file):
            failed = QMessageBox(QMessageBox.Icon.Warning, "Log File",
                                 f"Could not open the folder of {log_file}.", parent=self)
            failed.exec()

    def _action_about(self) -> None:
        from .. import __version__

        box = QMessageBox(self)
        box.setWindowTitle("About engiNIR")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(about_text(__version__))
        # Selectable (the citation is copied from here) and with links that open.
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        box.exec()

    def workspace_plot_frames(self) -> list[PlotFrame]:
        """The plots of the current workspace, including Model pages not shown yet."""
        current = self._workspace_tabs.currentWidget()
        if current is self._modeling_view:
            self._ensure_model_content_page()
            widget = self._model_diagnostics_widget
            return [] if widget is None else list(widget.grid.all_panels().values())
        if current is self._predict_tab:
            self._ensure_predict_content_page()
            return list(self._predict_graph_panel.panels.values())
        frames: list[PlotFrame] = [self._canvas_frame, self._sensor_frame, self._spatial_frame,
                                   self._spectrum_panel, *self._dock_frames.values()]
        if self._project_rois is not None:
            frames += [self._project_rois.mosaic_frame, self._project_rois.means,
                       self._project_rois.pixels]
        return frames

    def export_workspace_plots(self, folder: str | os.PathLike[str]) -> tuple[list[str], list[str]]:
        """Write PNG and CSV of every plot of the current workspace that shows data.

        Files are named by plot id (``model.pred_vs_ref.png``, ...) and
        overwrite files of the same name. Returns the written file names and
        the ids of the plots skipped because they show no data.
        """
        target = os.fspath(folder)
        if not os.path.isdir(target):
            log_and_raise("Export folder does not exist: {}", target)
        written: list[str] = []
        skipped: list[str] = []
        for frame in self.workspace_plot_frames():
            if frame.export_reason():
                skipped.append(frame.plot_id)
                continue
            for suffix, export in ((".png", frame.export_image), (".csv", frame.export_csv)):
                path = os.path.join(target, f"{frame.plot_id}{suffix}")
                export(path)
                written.append(os.path.basename(path))
        logger.info("Exported {} plot files to {}; skipped without data: {}", len(written),
                    target, skipped)
        return written, skipped

    def _action_export_plots(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export plots to folder")
        if not folder:
            return
        frames = [frame for frame in self.workspace_plot_frames() if not frame.export_reason()]
        if not frames:
            QMessageBox.information(self, "Export Plots", "No plot of this workspace shows data.")
            return
        n_values = sum(table.n_values for frame in frames
                       if (table := frame.table()) is not None)
        existing = [f"{frame.plot_id}{suffix}" for frame in frames for suffix in (".png", ".csv")
                    if os.path.exists(os.path.join(folder, f"{frame.plot_id}{suffix}"))]
        questions = []
        if existing:
            questions.append(f"{len(existing)} file(s) in this folder will be replaced.")
        if n_values > LARGE_EXPORT_VALUES:
            questions.append(f"The CSV files hold {n_values:,} values.")
        if questions:
            answer = QMessageBox.question(self, "Export Plots",
                                          " ".join(questions) + " Export anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            written, skipped = self.export_workspace_plots(folder)
        except (OSError, ValueError) as exc:
            logger.error("Export of the workspace plots failed: {}", exc)
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        message = f"Wrote {len(written)} files to {folder}."
        if skipped:
            message += f"\n\nSkipped (no data): {', '.join(skipped)}"
        QMessageBox.information(self, "Export Plots", message)

    def _ask_colour_limits(self) -> None:
        """The prediction map's colour limits: two numbers, or empty for the training range."""
        widget = self._predict_map_widget
        if widget is None:
            return
        current = "" if widget.user_limits is None else (
            f"{widget.user_limits[0]:g}, {widget.user_limits[1]:g}")
        text, ok = QInputDialog.getText(
            self, "Color limits", "Low, high (empty: the calibration range):",
            text=current)
        if not ok:
            return
        parts = [part for part in text.replace(";", ",").split(",") if part.strip()]
        try:
            limits = None if not parts else (float(parts[0]), float(parts[1]))
        except (ValueError, IndexError):
            QMessageBox.warning(self, "Color limits", f"{text!r} is not two numbers.")
            return
        if limits is not None and not limits[1] > limits[0]:
            QMessageBox.warning(self, "Color limits", "The high limit must be above the low.")
            return
        self._prediction_view.set_colour_limits(limits)  # the Controls dock shows them too

    def _add_prediction_data_file(self, model_run_id: str) -> str | None:
        """Add a scan for prediction with the Add data file dialog, noting the model's training."""
        model_run = self.project_state.model_runs[model_run_id]
        note = self._prediction_view._training_reference_correction(model_run)
        domain = self.project_state.signal_domains.get(model_run.signal_domain_id)
        return self.add_data_file(trained_with=note,
                                  binning=1 if domain is None else domain.binning)

    def _release_minimum_size(self) -> None:
        """Let the compositor choose the window size, whatever the docks' minimums.

        Qt makes a top-level window's minimum size its layout's minimum. When
        that exceeds the size a Wayland compositor configures (maximized, or a
        tiled half screen), the window draws a larger buffer and WSLg's
        compositor ends the connection ("xdg_surface buffer does not match the
        configured maximized state"). Without the constraint the docks are
        squeezed instead; the scrolling Controls dock keeps them usable.
        """
        layout = self.layout()
        if layout is not None:
            layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.setMinimumSize(1, 1)

    def reset_layout(self) -> None:
        """Restore default dock layout (positions, splits, visibility) and plot layout."""
        self._restore_maximized_plots()
        if self._model_diagnostics_widget is not None:
            for page in self._model_diagnostics_widget.grid.page_names:
                self._model_diagnostics_widget.grid.restore_page(page)
        if self._predict_map_widget is not None:
            self._predict_graph_panel.restore()
        layout_settings.clear()
        self._explore_plot_visibility = dict(self._default_explore_plot_visibility)
        self._roi_table_visibility = dict(self._default_roi_table_visibility)
        self.restoreState(self._default_layout_state)
        self._sync_workspace_content_host()

    def _set_theme_mode(self, dark: bool) -> None:
        """Switch the application theme and restyle all open viewer surfaces."""
        app = QApplication.instance()
        if not isinstance(app, QApplication):
            return
        apply_qt_theme(app, dark=dark)
        for widget in app.topLevelWidgets():
            refresh = getattr(widget, "refresh_theme", None)
            if callable(refresh):
                refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Leave the project refresh and remove the log sink; Qt deletes the window next.

        The data are released when Qt deletes the window (:func:`keep_open`), not
        here: a refresh already queued may still draw before the deletion.
        """
        self.project_state.close()
        self._teardown_log_sink()
        super().closeEvent(a0)

    def refresh_theme(self) -> None:
        """Refresh non-Qt themed surfaces after one global theme switch."""
        if hasattr(self, "_theme_dark_action"):
            self._theme_dark_action.setChecked(is_dark())
        if hasattr(self, "_theme_light_action"):
            self._theme_light_action.setChecked(not is_dark())
        self._refresh_panel_theme()

    def export_csv(self, path: str) -> None:
        """Export persisted training ROIs from the active project to one CSV file."""
        self._eda_view.export_csv(path)

    def _quit_application(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        app.quit()

    def _refresh_panel_theme(self) -> None:
        """Restyle matplotlib/pyqtgraph surfaces that are outside Qt palette control."""
        self._canvas.restyle_theme()
        self._hist.restyle_theme()
        self._sensor.restyle_theme()
        self._spatial.restyle_theme()
        self._spectrum.restyle_theme()
        if self._model_diagnostics_widget is not None:
            self._model_diagnostics_widget.restyle_theme()
        if self._predict_map_widget is not None:
            self._predict_map_widget.restyle_theme()
            self._predict_histogram.restyle_theme()
        self._log_panel.restyle_theme()
        if self._project_rois is not None:
            self._project_rois.mosaic.restyle_theme()
            self._project_rois.means.restyle_theme()
            self._project_rois.pixels.restyle_theme()
