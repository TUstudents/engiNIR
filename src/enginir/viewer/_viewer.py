"""HSIViewer QMainWindow shell for Explore, Model, and Predict workspaces."""

from __future__ import annotations

from typing import Any, TypeVar
import os
import time
from functools import partial

import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QMainWindow,
    QWidget,
    QDockWidget,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
    QPlainTextEdit,
    QSizePolicy,
    QLabel,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QTabWidget,
    QMenu,
    QMenuBar,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QFont, QCloseEvent, QKeySequence, QShortcut

from .._checks import log_and_raise
from .._workflow import read_header
from ..project import CalibrationSpec, viewable_signal_units
from .._logging import add_log_sink, get_logger, log_file_path, remove_log_sink
from ..io import load_spectra

from ._eda import EDAView
from ._attachment import confirm_project_dataset_binding
from ._browser import ProjectBrowser
from ..io import file_format
from ._inspector import ProjectInspector
from ._data_files import AddDataFileDialog
from ._project_rois import ProjectRoisView
from ._roi_table import RoiTableWidget
from ._jobs import BusyStatusWidget, start_background_job
from ._loading import ViewerBusyContext
from ._model_diagnostics import ModelDiagnosticsWidget
from . import _layout_settings as layout_settings
from ._project_settings import load_last_data_file, save_last_data_file
from ._plot_grid import PlotPage, PlotPageWidget, PlotSlot
from ._plot_panel import LARGE_EXPORT_VALUES, PlotFrame, PlotPanel
from ._modeling import ModelingView, NewROIDialog, TargetSelectionDialog
from ._prediction import PREDICTION_HISTOGRAM, PredictionMapWidget, PredictionView
from ._state import (
    DEFAULT_BLACK_LEVEL,
    DEFAULT_SOURCE_MODALITY,
    DEFAULT_WHITE_LEVEL,
    ProjectState,
    ViewerState,
)
from ._panels import (
    _PanelBase as _ExplorePanel,
    HistogramPanel,
    DC3Panel,
    SensorPanel,
    SpatialProfilePanel,
    SpectrumPanel,
)
from ._theme import apply_qt_theme, is_dark, panel_colors, secondary_text_style
from ._ui_helpers import reveal_in_file_manager

logger = get_logger(__name__)
_QtObject = TypeVar("_QtObject")
_COMPACT_LEFT_DOCK_WIDTH = 260
_CONTROLS_DOCK_TARGET_HEIGHT = 230
_SPECTRUM_PANEL_HEIGHT = 220
_UNLIMITED_HEIGHT = 16777215  # QWIDGETSIZE_MAX
_SAVED_MESSAGE_MS = 8000
_ADD_DATA_FILE = "__add_data_file__"


def _calibration_view(calibration: CalibrationSpec | None,
                      meta: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """``ViewerState.load_dataset`` keywords and the signal unit for a data file's calibration."""
    levels: dict[str, Any] = {
        "black_level": float(meta.get("black_level", DEFAULT_BLACK_LEVEL)),
        "white_level": float(meta.get("white_level", DEFAULT_WHITE_LEVEL)),
    }
    if calibration is None:
        return levels, "DN"
    if calibration.dark_ref_data is not None and calibration.white_ref_data is not None:
        levels["dark_ref"] = np.asarray(calibration.dark_ref_data, dtype=np.float32)
        levels["white_ref"] = np.asarray(calibration.white_ref_data, dtype=np.float32)
    elif calibration.black_level is not None and calibration.white_level is not None:
        levels["black_level"] = float(calibration.black_level)
        levels["white_level"] = float(calibration.white_level)
    return levels, calibration.output_signal_unit


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


class LogPanel(QWidget):
    """Scrollable read-only log viewer.  Receives loguru messages via a
    custom sink connected through a thread-safe Qt signal."""

    _append = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(1000)
        self._text.setFont(QFont("Monospace", 9))
        self.restyle_theme()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(80)

        self._append.connect(self._text.appendPlainText)

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

        black_level: float = float(_meta.get("black_level", DEFAULT_BLACK_LEVEL))
        white_level: float = float(_meta.get("white_level", DEFAULT_WHITE_LEVEL))

        self.state = ViewerState(
            cube,
            wavelengths,
            _meta,
            black_level=black_level,
            white_level=white_level,
        )
        _mark_init_step("state")

        self.project_state = ProjectState()
        self._cube_path: str | None = None
        self._child_viewers: list["HSIViewer"] = []
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
        from ._canvas import FalseColourCanvas

        self._canvas = FalseColourCanvas()
        self._spectrum = SpectrumPanel(self.state)
        self._data_file_page = QWidget()
        central_layout = QVBoxLayout(self._data_file_page)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        self._canvas_frame = PlotFrame("explore.image", "Spatial Image", self._canvas,
                                       self._canvas, provenance=self._explore_provenance)
        self._data_file_job: Any = None
        self._data_file_combo = QComboBox()
        self._data_file_combo.setToolTip("Data file of the project shown in the image")
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
        central_layout.addWidget(self._canvas_frame, 1)

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

        self._model_diagnostics_widget = None
        self._predict_map_widget = None
        self._model_content_page = self._build_deferred_workspace_page("Model Diagnostics")
        self._predict_content_page = self._build_deferred_workspace_page("Prediction Results")

        self._workspace_content_stack = QStackedWidget()
        self._workspace_content_stack.setObjectName("WorkspaceContentStack")
        # Explore: the shown data file, or all ROIs of the project (filled in later).
        self._explore_tabs = QTabWidget()
        self._explore_tabs.setDocumentMode(True)
        self._explore_tabs.addTab(self._data_file_page, "Data file")
        self._explore_content_page = self._explore_tabs
        self._project_rois: ProjectRoisView | None = None
        self._pending_roi_selection: str | None = None
        self._workspace_content_stack.addWidget(self._explore_content_page)
        self._workspace_content_stack.addWidget(self._model_content_page)
        self._workspace_content_stack.addWidget(self._predict_content_page)
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

    def _ensure_model_content_page(self) -> None:
        if self._model_diagnostics_widget is not None:
            return
        old_page = self._model_content_page
        self._model_content_page = self._build_model_content_page()
        self._replace_workspace_content_page(old_page, self._model_content_page)
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

        summary_box = QGroupBox("Fit Summary")
        summary_layout = QFormLayout(summary_box)
        summary_layout.setContentsMargins(8, 8, 8, 8)
        summary_layout.setSpacing(4)

        self._model_card_target_value = QLabel("none")
        self._model_card_sample_set_value = QLabel("none")
        self._model_card_model_value = QLabel("none")
        self._model_card_components_value = QLabel("-")
        self._model_card_cv_value = QLabel("-")
        self._model_card_roi_count_value = QLabel("0")
        self._model_card_pixel_count_value = QLabel("0")

        summary_layout.addRow("Target", self._model_card_target_value)
        summary_layout.addRow("Sample set", self._model_card_sample_set_value)
        summary_layout.addRow("Model", self._model_card_model_value)
        summary_layout.addRow("Components", self._model_card_components_value)
        summary_layout.addRow("Validation", self._model_card_cv_value)
        summary_layout.addRow("ROIs", self._model_card_roi_count_value)
        summary_layout.addRow("Pixels", self._model_card_pixel_count_value)

        metrics_box = QGroupBox("Metrics")
        metrics_layout = QFormLayout(metrics_box)
        metrics_layout.setContentsMargins(8, 8, 8, 8)
        metrics_layout.setSpacing(4)

        self._model_metric_r2_value = QLabel("-")
        self._model_metric_rmsecv_value = QLabel("-")
        self._model_metric_bias_value = QLabel("-")

        metrics_layout.addRow("R2", self._model_metric_r2_value)
        metrics_layout.addRow("RMSECV", self._model_metric_rmsecv_value)
        metrics_layout.addRow("Bias", self._model_metric_bias_value)

        readiness_box = QGroupBox("Readiness")
        readiness_layout = QVBoxLayout(readiness_box)
        readiness_layout.setContentsMargins(8, 8, 8, 8)
        readiness_layout.setSpacing(4)
        self._model_readiness_body = QLabel("Ready: define or select a target.")
        self._model_readiness_body.setWordWrap(True)
        readiness_layout.addWidget(self._model_readiness_body)
        self._model_summary_service = BusyStatusWidget(idle_text="Service: idle", parent=page)

        layout.addWidget(self._model_content_title)
        text_layout.addWidget(self._model_summary_service)
        text_layout.addWidget(summary_box)
        text_layout.addWidget(metrics_box)
        text_layout.addWidget(readiness_box)
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
        self._predict_summary_status = QLabel("Prediction: none")
        self._predict_summary_readiness = QLabel("Ready: fit or load a model run.")
        self._predict_summary_readiness.setWordWrap(True)
        self._predict_summary_service = BusyStatusWidget(idle_text="Service: idle", parent=page)
        self._predict_workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._predict_workspace_splitter.setChildrenCollapsible(False)

        self._predict_map_widget = PredictionMapWidget()
        self._predict_map_frame = PlotFrame(
            "predict.map", self._predict_map_widget.plot_title, self._predict_map_widget,
            self._predict_map_widget, provenance=self._prediction_view.workspace_provenance)
        for control in self._predict_map_widget.header_widgets():
            self._predict_map_frame.add_header_widget(control)
        self._predict_histogram = PlotPanel(PREDICTION_HISTOGRAM)
        self._predict_map_widget.displayed_map_changed.connect(
            lambda shown: self._predict_histogram.update_data(
                shown, provenance=self._prediction_view.workspace_provenance()))
        self._predict_graph_panel = PlotPageWidget(
            PlotPage("Prediction", (PlotSlot("predict.map", 0, 0, row_span=2),
                                    PlotSlot("predict.histogram", 2, 0))),
            {"predict.map": self._predict_map_frame,
             "predict.histogram": self._predict_histogram})

        self._predict_text_panel = QWidget()
        self._predict_text_panel.setObjectName("PredictionResultsTextPanel")
        self._predict_text_panel.setMinimumWidth(260)
        self._predict_text_panel.setMaximumWidth(360)
        predict_text_layout = QVBoxLayout(self._predict_text_panel)
        predict_text_layout.setContentsMargins(0, 0, 0, 0)
        predict_text_layout.setSpacing(8)

        summary_box = QGroupBox("Result Summary")
        summary_layout = QFormLayout(summary_box)
        summary_layout.setContentsMargins(8, 8, 8, 8)
        summary_layout.setSpacing(4)
        self._predict_card_prediction_value = QLabel("none")
        self._predict_card_model_value = QLabel("none")
        self._predict_card_dataset_value = QLabel("none")
        self._predict_card_calibration_value = QLabel("none")
        self._predict_card_target_value = QLabel("none")
        summary_layout.addRow("Prediction", self._predict_card_prediction_value)
        summary_layout.addRow("Model", self._predict_card_model_value)
        summary_layout.addRow("Dataset", self._predict_card_dataset_value)
        summary_layout.addRow("Calibration", self._predict_card_calibration_value)
        summary_layout.addRow("Target", self._predict_card_target_value)

        compatibility_box = QGroupBox("Compatibility")
        compatibility_layout = QFormLayout(compatibility_box)
        compatibility_layout.setContentsMargins(8, 8, 8, 8)
        compatibility_layout.setSpacing(4)
        self._predict_contract_format_value = QLabel("-")
        self._predict_contract_signal_value = QLabel("-")
        self._predict_contract_bands_value = QLabel("-")
        self._predict_map_shape_value = QLabel("-")
        self._predict_map_range_value = QLabel("-")
        self._predict_clipping_value = QLabel("-")
        self._predict_diagnostics_value = QLabel("-")
        self._predict_diagnostics_value.setWordWrap(True)
        self._predict_wavelengths_value = QLabel("-")
        self._predict_wavelengths_value.setWordWrap(True)
        compatibility_layout.addRow("Input contract", self._predict_contract_format_value)
        compatibility_layout.addRow("Signal contract", self._predict_contract_signal_value)
        compatibility_layout.addRow("Bands", self._predict_contract_bands_value)
        compatibility_layout.addRow("Map shape", self._predict_map_shape_value)
        compatibility_layout.addRow("Map range", self._predict_map_range_value)
        compatibility_layout.addRow("Calibration clipping", self._predict_clipping_value)
        compatibility_layout.addRow("Applicability", self._predict_diagnostics_value)
        compatibility_layout.addRow("Wavelengths", self._predict_wavelengths_value)

        layout.addWidget(self._predict_content_title)
        predict_text_layout.addWidget(self._predict_summary_status)
        predict_text_layout.addWidget(self._predict_summary_readiness)
        predict_text_layout.addWidget(self._predict_summary_service)
        predict_text_layout.addWidget(summary_box)
        predict_text_layout.addWidget(compatibility_box)
        predict_text_layout.addStretch(1)

        self._predict_workspace_splitter.addWidget(self._predict_graph_panel)
        self._predict_workspace_splitter.addWidget(self._predict_text_panel)
        self._predict_workspace_splitter.setStretchFactor(0, 5)
        self._predict_workspace_splitter.setStretchFactor(1, 1)
        self._predict_workspace_splitter.setSizes([900, 300])
        layout.addWidget(self._predict_workspace_splitter, 1)
        return page

    def _reset_image_view(self) -> None:
        """Reset the central image view to the full image bounds."""
        self._canvas.reset_view()

    def _false_colour_rgb(self) -> np.ndarray:
        """Return the current false-color preview image as float RGB."""
        return self.state.raw_false_colour_rgb()

    def _build_dock_widgets(self) -> None:
        s = self.state
        self._hist = HistogramPanel(s)
        self._dc3 = DC3Panel(s)
        self._sensor = SensorPanel(s)
        self._spatial = SpatialProfilePanel(s)
        self._project_browser = ProjectBrowser(self.project_state)
        self._inspector = ProjectInspector(self.project_state)

        self._project_browser_dock = QDockWidget("Project Browser", self)
        self._project_browser_dock.setObjectName("ProjectBrowserDock")
        self._project_browser_dock.setWidget(self._project_browser)
        self._configure_compact_left_dock(self._project_browser_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._project_browser_dock)

        self._inspector_dock = QDockWidget("Inspector", self)
        self._inspector_dock.setObjectName("InspectorDock")
        self._inspector_dock.setWidget(self._inspector)
        self._configure_compact_left_dock(self._inspector_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._inspector_dock)
        self.splitDockWidget(
            self._project_browser_dock,
            self._inspector_dock,
            Qt.Orientation.Vertical,
        )
        # Left docks - Inspector above collapsed Histogram/RAW utilities
        self._dock_frames: dict[QDockWidget, PlotFrame] = {}
        self._maximized_dock: QDockWidget | None = None
        self._hist_dock = self._plot_dock("Histogram", "HistogramDock", self._hist,
                                          "explore.histogram")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._hist_dock)
        self.splitDockWidget(self._inspector_dock, self._hist_dock, Qt.Orientation.Vertical)

        self._dc3_dock = self._plot_dock("RAW", "DC3Dock", self._dc3, "explore.raw")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dc3_dock)
        self.splitDockWidget(self._hist_dock, self._dc3_dock, Qt.Orientation.Vertical)
        self._hist_dock.hide()
        self._dc3_dock.hide()

        # Right docks - Sensor Frame above Spatial Profile
        self._sensor_dock = self._plot_dock("Sensor Frame", "SensorDock", self._sensor,
                                            "explore.sensor")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._sensor_dock)

        self._spatial_dock = self._plot_dock("Spatial Profile", "SpatialProfileDock",
                                             self._spatial, "explore.spatial")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._spatial_dock)
        self.splitDockWidget(self._sensor_dock, self._spatial_dock, Qt.Orientation.Vertical)

        # ROI table: target values of all ROIs; available in every workspace.
        self._roi_table = RoiTableWidget(self.project_state)
        self._roi_table.roi_selected.connect(self._on_canvas_roi_selected)
        self._roi_table_dock = QDockWidget("ROI Table", self)
        self._roi_table_dock.setObjectName("RoiTableDock")
        self._roi_table_dock.setWidget(self._roi_table)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._roi_table_dock)
        self._roi_table_dock.hide()
        #: ROI table per Explore tab: shown with Project ROIs, not on the Data file tab;
        #: the user's choice on each tab is kept. Model and Predict leave it as it is.
        self._default_roi_table_visibility = {"data_file": False, "project_rois": True}
        self._roi_table_visibility = dict(self._default_roi_table_visibility)
        self._roi_table_dock.visibilityChanged.connect(self._on_roi_table_visibility_changed)

        #: Docks that show the data file at the cursor: only on Explore's Data file tab.
        self._data_file_docks = {"histogram": self._hist_dock, "raw": self._dc3_dock,
                                 "sensor": self._sensor_dock, "spatial": self._spatial_dock}
        self._default_explore_plot_visibility = {
            "histogram": False,
            "raw": False,
            "sensor": True,
            "spatial": True,
            "spectrum": True,
        }
        self._explore_plot_visibility = dict(self._default_explore_plot_visibility)

    # ------------------------------------------------------------------
    # Explore plot frames (Plot Panel Policy)
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
        for dataset in self.project_state.datasets.values():
            if self._cube_path and os.path.abspath(dataset.source_path) == os.path.abspath(
                    self._cube_path):
                provenance["dataset_id"] = dataset.dataset_id
                break
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
        self._canvas_frame.setVisible(maximized != "spectrum")
        self._canvas_frame.set_maximized(maximized == "image")
        self._spectrum_panel.set_maximized(maximized == "spectrum")
        if maximized == "spectrum":
            self._spectrum_panel.setMinimumHeight(0)
            self._spectrum_panel.setMaximumHeight(_UNLIMITED_HEIGHT)
        else:
            self._spectrum_panel.setMinimumHeight(_SPECTRUM_PANEL_HEIGHT)
            self._spectrum_panel.setMaximumHeight(_SPECTRUM_PANEL_HEIGHT)
        explore = self._workspace_tabs.currentWidget() is self._eda_view
        self._spectrum_panel.setVisible(
            explore and self._explore_plot_visibility["spectrum"] and maximized != "image")
        self._explore_escape.setEnabled(maximized is not None)

    @staticmethod
    def _configure_compact_left_dock(dock: QDockWidget) -> None:
        dock.setMinimumWidth(_COMPACT_LEFT_DOCK_WIDTH)
        dock.setMaximumWidth(_COMPACT_LEFT_DOCK_WIDTH)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)

    def _configure_controls_dock_height(self) -> None:
        self._controls_dock_height = _CONTROLS_DOCK_TARGET_HEIGHT
        self._controls_dock.setMinimumHeight(self._controls_dock_height)
        self._controls_dock.setMaximumHeight(self._controls_dock_height)

    def _build_controls_panel(self) -> None:
        self._modeling_view = ModelingView(self.state, self.project_state, parent=self)
        self._prediction_view = PredictionView(self.project_state, parent=self)
        self._eda_view = EDAView(
            self.state,
            self.project_state,
            dc3_panel=self._dc3,
            canvas=self._canvas,
            cube_path_provider=lambda: self._cube_path,
            dataset_attached_provider=self.has_attached_dataset,
            current_target_provider=self._modeling_view.current_target,
            selected_roi_id_provider=self._selected_roi_id,
            parent=self,
        )

        self._project_status = self._eda_view.project_status_label
        self._target_status = self._modeling_view.target_status_label

        self._workspace_tabs = QTabWidget()
        self._workspace_tabs.setObjectName("WorkspaceTabs")
        self._workspace_tabs.addTab(self._eda_view, "Explore")
        self._workspace_tabs.addTab(self._modeling_view, "Model")
        self._workspace_tabs.addTab(self._prediction_view, "Predict")
        self._workspace_tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Ignored,
        )

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        outer = QHBoxLayout(container)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(6)
        outer.addWidget(self._workspace_tabs)

        self._controls_dock = QDockWidget("Controls", self)
        self._controls_dock.setObjectName("ControlsDock")
        self._controls_dock.setWidget(container)
        self._configure_controls_dock_height()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._controls_dock)
        self._refresh_workspace_content_pages()

    def _build_log_panel(self) -> None:
        self._log_panel = LogPanel()
        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setObjectName("LogDock")
        self._log_dock.setWidget(self._log_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)
        self.splitDockWidget(self._controls_dock, self._log_dock, Qt.Orientation.Vertical)
        self._log_dock.hide()
        self._restore_bottom_dock_sizes()
        self._log_sink_id = add_log_sink(
            self._log_panel.sink,
            level="DEBUG",
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
        project_label = "none"
        if self.project_state.project_path:
            project_label = os.path.basename(self.project_state.project_path)

        dataset_label = "none"
        if self._cube_path:
            shown = self._eda_view.shown_dataset_id()
            dataset_label = (self.project_state.data_file_label(shown) if shown is not None
                             else os.path.basename(self._cube_path))

        frame_label = "-" if not self.state.has_dataset else str(self.state.frame)
        row_label = "-" if not self.state.has_dataset else str(self.state.row)
        band_label = "-" if not self.state.has_dataset else str(self.state.band)
        self._status_label.setText(
            f"Project: {project_label} | "
            f"Dataset: {dataset_label} | "
            f"Signal: {self.state.y_unit_effective} | "
            f"Frame: {frame_label} | "
            f"Row: {row_label} | "
            f"Band: {band_label}"
        )

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        s = self.state
        # Panel -> state
        self._dc3.frame_pick.connect(s.set_frame)
        self._dc3.row_pick.connect(s.set_row)
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
        self._prediction_view._service_status.status_changed.connect(
            self.schedule_workspace_content_refresh
        )
        for name, dock in self._data_file_docks.items():
            dock.visibilityChanged.connect(
                lambda _visible, name=name: self._on_data_file_dock_visibility_changed(name))

        self.project_state.project_loaded.connect(self._refresh_data_file_choices)
        self.project_state.datasets_changed.connect(self._refresh_data_file_choices)
        self._project_browser.itemDoubleClicked.connect(self._on_browser_item_activated)
        self._eda_view.roi_table_requested.connect(self.show_roi_table)
        self._project_rois = ProjectRoisView(
            self.state, self.project_state,
            current_target_id=lambda: (None if (target := self._modeling_view.current_target())
                                       is None else target.target_id),
            provenance=self._explore_provenance)
        self._explore_tabs.addTab(self._project_rois, "Project ROIs")
        self._project_rois.roi_selected.connect(self._on_canvas_roi_selected)
        self._project_rois.roi_opened.connect(self.open_roi)
        self._modeling_view.view_state_changed.connect(self._project_rois.current_target_changed)
        self.project_state.rois_changed.connect(self._update_project_rois_tab)
        self.project_state.project_loaded.connect(self._update_project_rois_tab)
        self.project_state.project_loaded.connect(self._choose_explore_tab)
        self._explore_tabs.currentChanged.connect(self._show_last_data_file)
        self._explore_tabs.currentChanged.connect(self._on_explore_tab_changed)
        self._project_rois.roi_table_requested.connect(self.show_roi_table)
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

    def show_project_rois(self) -> None:
        """Show the Project ROIs tab of Explore."""
        self._workspace_tabs.setCurrentWidget(self._eda_view)
        if self._project_rois is not None:
            self._explore_tabs.setCurrentWidget(self._project_rois)

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
        if dataset_id is not None and dataset is not None \
                and os.path.exists(dataset.source_path):
            self.show_data_file(dataset_id)

    def show_roi_table(self) -> None:
        """Show the ROI table on the ROIs of the current page and the target selected in Model.

        The table takes the right dock column: alone on Project ROIs, Model and
        Predict (the data-file docks are hidden there), as a tab next to the
        Sensor Frame on the Data file tab.
        """
        self._roi_table.show_data_file(self._roi_table_data_file())
        target = self._modeling_view.current_target()
        self._roi_table.focus_target(None if target is None else target.target_id)
        was_hidden = self._roi_table_dock.isHidden()
        self._roi_table_dock.show()
        self._place_roi_table()
        self._roi_table_dock.raise_()
        if was_hidden and not self._roi_table_dock.isFloating():
            self.resizeDocks([self._roi_table_dock], [460], Qt.Orientation.Horizontal)

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

    def _place_roi_table(self) -> None:
        """On the Data file tab, tab a docked ROI table with the Sensor Frame (full slot)."""
        table, sensor = self._roi_table_dock, self._sensor_dock
        if (not self._on_data_file_page() or table.isHidden() or table.isFloating()
                or sensor.isHidden() or sensor.isFloating()
                or table in self.tabifiedDockWidgets(sensor)):
            return
        self.tabifyDockWidget(sensor, table)
        table.raise_()

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
        else:
            self._prediction_view.select_prediction_run(None)
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
        elif current is self._prediction_view:
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
        self._sync_workspace_content_host()
        self._refresh_workspace_content_pages()

    def _sync_workspace_content_host(self) -> None:
        current = self._workspace_tabs.currentWidget()
        is_explore = current is self._eda_view
        is_model = current is self._modeling_view
        if is_model:
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
        self._spectrum_toggle_action.setEnabled(is_explore)
        self._spectrum_toggle_action.blockSignals(True)
        self._spectrum_toggle_action.setChecked(self._explore_plot_visibility["spectrum"])
        self._spectrum_toggle_action.blockSignals(False)
        self._restore_bottom_dock_sizes()
        self._schedule_bottom_dock_resize()

    def _schedule_bottom_dock_resize(self) -> None:
        if self._bottom_dock_resize_pending:
            return
        self._bottom_dock_resize_pending = True
        QTimer.singleShot(0, self._restore_bottom_dock_sizes)

    def _restore_bottom_dock_sizes(self) -> None:
        self._bottom_dock_resize_pending = False
        if not hasattr(self, "_controls_dock"):
            return
        docks = [self._controls_dock]
        sizes = [self._controls_dock_height]
        if hasattr(self, "_log_dock") and not self._log_dock.isHidden():
            docks.append(self._log_dock)
            sizes.append(_CONTROLS_DOCK_TARGET_HEIGHT)
        self.resizeDocks(docks, sizes, Qt.Orientation.Vertical)

    def _refresh_model_workspace_content(self) -> None:
        self._ensure_model_content_page()
        model_diagnostics_widget = self._model_diagnostics_widget
        if model_diagnostics_widget is None:
            return
        model_summary = self._modeling_view.workspace_summary()
        service_text, service_busy = self._modeling_view.service_status_snapshot()
        self._model_content_title.setText(model_summary["title"])
        self._model_card_target_value.setText(model_summary["target"])
        self._model_card_sample_set_value.setText(model_summary["sample_set"])
        self._model_card_model_value.setText(model_summary["model"])
        self._model_card_components_value.setText(model_summary["components"])
        self._model_card_cv_value.setText(model_summary["cv"])
        self._model_card_roi_count_value.setText(model_summary["roi_count"])
        self._model_card_pixel_count_value.setText(model_summary["pixel_count"])
        self._model_metric_r2_value.setText(model_summary["r2"])
        self._model_metric_rmsecv_value.setText(model_summary["rmsecv"])
        self._model_metric_bias_value.setText(model_summary["bias"])
        self._model_readiness_body.setText(model_summary["readiness"])
        self._set_busy_status_widget(
            self._model_summary_service, text=service_text, busy=service_busy
        )
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
        service_text, service_busy = self._prediction_view.service_status_snapshot()
        self._predict_content_title.setText(prediction_summary["title"])
        self._predict_summary_status.setText(prediction_summary["status"])
        self._predict_summary_readiness.setText(prediction_summary["readiness"])
        self._set_busy_status_widget(
            self._predict_summary_service, text=service_text, busy=service_busy
        )
        self._predict_card_prediction_value.setText(prediction_summary["prediction"])
        self._predict_card_model_value.setText(prediction_summary["model"])
        self._predict_card_dataset_value.setText(prediction_summary["dataset"])
        self._predict_card_calibration_value.setText(prediction_summary["calibration"])
        self._predict_card_target_value.setText(prediction_summary["target"])
        self._predict_contract_format_value.setText(prediction_summary["format"])
        self._predict_contract_signal_value.setText(prediction_summary["signal"])
        self._predict_contract_bands_value.setText(prediction_summary["bands"])
        self._predict_map_shape_value.setText(prediction_summary["map_shape"])
        self._predict_map_range_value.setText(prediction_summary["map_range"])
        self._predict_clipping_value.setText(prediction_summary["clipping"])
        self._predict_diagnostics_value.setText(prediction_summary["diagnostics"])
        self._predict_wavelengths_value.setText(prediction_summary["wavelengths"])
        predict_map_widget.update_data(
            self._prediction_view.workspace_visual_payload(),
            empty_reason=prediction_summary["readiness"],
        )

    @staticmethod
    def _set_busy_status_widget(widget: BusyStatusWidget, *, text: str, busy: bool) -> None:
        current_text = widget.label.text()
        current_busy = widget.progress.isVisible()
        if current_text == text and current_busy == busy:
            return
        if busy:
            widget.set_busy(text)
        else:
            widget.set_idle(text)

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
        self._place_roi_table()

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
        self._on_explore_plot_visibility_changed("spectrum", visible)
        if not visible and self._explore_maximized == "spectrum":
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
                    f"Sample set: {self._model_card_sample_set_value.text()}",
                    f"Model: {self._model_card_model_value.text()}",
                    f"Components: {self._model_card_components_value.text()}",
                    f"Validation: {self._model_card_cv_value.text()}",
                    f"ROIs: {self._model_card_roi_count_value.text()}",
                    f"Pixels: {self._model_card_pixel_count_value.text()}",
                    f"R2: {self._model_metric_r2_value.text()}",
                    f"RMSECV: {self._model_metric_rmsecv_value.text()}",
                    f"Bias: {self._model_metric_bias_value.text()}",
                    self._model_readiness_body.text(),
                ],
            )
            return
        if current is self._prediction_view:
            self._inspector.show_workspace_summary(
                self._predict_content_title.text(),
                [
                    self._predict_summary_status.text(),
                    self._predict_summary_readiness.text(),
                    self._predict_summary_service.label.text(),
                    f"Prediction: {self._predict_card_prediction_value.text()}",
                    f"Model: {self._predict_card_model_value.text()}",
                    f"Dataset: {self._predict_card_dataset_value.text()}",
                    f"Calibration: {self._predict_card_calibration_value.text()}",
                    f"Target: {self._predict_card_target_value.text()}",
                    f"Input contract: {self._predict_contract_format_value.text()}",
                    f"Signal contract: {self._predict_contract_signal_value.text()}",
                    f"Bands: {self._predict_contract_bands_value.text()}",
                    f"Map shape: {self._predict_map_shape_value.text()}",
                    f"Map range: {self._predict_map_range_value.text()}",
                    f"Calibration clipping: {self._predict_clipping_value.text()}",
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
        """Options > ROI Grid Defaults: the geometry every new ROI grid starts with."""
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
        self._menu_file = _add_required_menu(mb, "File")
        self._menu_view = _add_required_menu(mb, "View")
        self._menu_options = _add_required_menu(mb, "Options")
        self._menu_help = _add_required_menu(mb, "Help")
        self._populate_file_menu()
        self._populate_view_menu()
        grid_defaults = QAction("ROI Grid Defaults...", self)
        grid_defaults.triggered.connect(self.edit_grid_defaults)
        self._menu_options.addAction(grid_defaults)
        self._populate_help_menu()
        self._default_layout_state = self.saveState()
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

        act = _add_required_action(m, "Open Data...")
        act.setShortcut("Ctrl+O")
        act.triggered.connect(self._open_data)

        act = _add_required_action(m, "Add Data File...")
        act.setToolTip("Add a data file with its calibration to the open project")
        act.triggered.connect(lambda: self.add_data_file())

        self._menu_open_recent = _add_required_menu(m, "Open Recent")
        self._menu_open_recent.aboutToShow.connect(self._populate_open_recent)

        m.addSeparator()

        act = _add_required_action(m, "New Project...")
        act.setShortcut("Ctrl+N")
        act.triggered.connect(self._eda_view.new_project_btn_clicked)

        act = _add_required_action(m, "Open Project...")
        act.setShortcut("Ctrl+Shift+O")
        act.triggered.connect(self._eda_view.open_project_btn_clicked)

        m.addSeparator()

        act = _add_required_action(m, "Export CSV...")
        act.triggered.connect(self._eda_view.export_csv_btn_clicked)
        act = _add_required_action(m, "Export Plots...")
        act.setToolTip("Save every plot of the current workspace as PNG and CSV")
        act.triggered.connect(self._action_export_plots)

        m.addSeparator()

        act = _add_required_action(m, "Close Window")
        act.setShortcut("Ctrl+W")
        act.triggered.connect(self.close)

        act = _add_required_action(m, "Quit")
        act.setShortcut("Ctrl+Q")
        act.triggered.connect(self._quit_application)

    def _open_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Data",
            "",
            "Hyperspectral files (*.eiff *.hdr *.h5 *.hdf5 *.csv *.spa);;All files (*.*)",
        )
        if path:
            self._open_data_path(path)

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
        self._track_child_viewer(child)

    # ------------------------------------------------------------------
    # Data files of the project (attachment contract, Data Files Of A Project)
    # ------------------------------------------------------------------

    def add_data_file(self, path: str | None = None) -> str | None:
        """Add a data file to the project with its calibration, then show it.

        A file that is already registered is shown directly. Returns the
        dataset id, or ``None`` when nothing was added.
        """
        if self.project_state.project_path is None:
            QMessageBox.warning(self, "No project open",
                                "Create or open a project before adding data files.")
            return None
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Add data file", "",
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
            logger.error("Cannot read data file {}: {}", path, exc)
            QMessageBox.critical(self, "Cannot open file", str(exc))
            return None
        state = self.project_state
        dialog = AddDataFileDialog(
            path, header, state.datasets,
            {dataset_id: calibration for dataset_id in state.datasets
             if (calibration := state.active_calibration(dataset_id)) is not None},
            parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.request is None:
            return None
        request = dialog.request
        try:
            dataset_id = state.add_data_file(request.path, **request.add_dataset_kwargs())
        except (OSError, ValueError, KeyError) as exc:
            logger.error("Cannot add data file {}: {}", path, exc)
            QMessageBox.critical(self, "Cannot add data file", str(exc))
            return None
        self.show_data_file(dataset_id)
        return dataset_id

    def show_data_file(self, dataset_id: str) -> None:
        """Load a registered data file in the background and show it with its calibration."""
        dataset = self.project_state.datasets.get(dataset_id)
        if dataset is None:
            log_and_raise("Unknown dataset '{}'", dataset_id, error=KeyError)
        path = dataset.source_path
        if not os.path.exists(path):
            box = QMessageBox(QMessageBox.Icon.Warning, "Data file not found",
                              f"{path} does not exist. Its ROIs, values and models stay "
                              "usable; the image cannot be shown.\n\nIf the file was moved or "
                              "renamed, locate it: it is accepted if its pixels equal the "
                              "stored ROI pixels.", parent=self)
            locate = box.addButton("Locate file...", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is locate:
                self.locate_data_file(dataset_id)
            return
        if self._data_file_job is not None:
            return
        busy = self._viewer_busy_context(f"Loading data file: {os.path.basename(path)}")
        busy.begin()

        def finished() -> None:
            busy.end()
            self._data_file_job = None

        self._data_file_job = start_background_job(
            owner=self, fn=lambda: load_spectra(path),
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
        if path is None:
            name = os.path.basename(dataset.source_path)
            path, _ = QFileDialog.getOpenFileName(
                self, f"Locate {name}", os.path.dirname(dataset.source_path),
                f"{name} (*{os.path.splitext(name)[1]});;All files (*)")
            if not path:
                return False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.project_state.locate_data_file(dataset_id, path)
        except (OSError, ValueError, KeyError) as exc:
            logger.error("Data file not located at {}: {}", path, exc)
            QMessageBox.warning(self, "Not this data file", str(exc))
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self.show_data_file(dataset_id)
        return True

    def _data_file_failed(self, message: str) -> None:
        logger.error("Cannot load data file: {}", message)
        QMessageBox.critical(self, "Cannot open file", message)

    def wait_for_data_file(self, timeout_s: float = 30.0) -> None:
        """Process events until a data file load has finished (for scripts and tests)."""
        deadline = time.monotonic() + timeout_s
        app = QApplication.instance()
        while self._data_file_job is not None and time.monotonic() < deadline:
            if app is not None:
                app.processEvents()
            time.sleep(0.005)

    def _apply_data_file(self, dataset_id: str, data: Any) -> None:
        dataset = self.project_state.datasets[dataset_id]
        X = np.asarray(data.X)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        if not confirm_project_dataset_binding(
            self,
            project_path=str(self.project_state.project_path),
            project_state=self.project_state,
            dataset_path=dataset.source_path,
            source_format=file_format(dataset.source_path),
            source_modality=str(data.meta.get("source_modality") or DEFAULT_SOURCE_MODALITY),
            cube_shape=(int(X.shape[0]), int(X.shape[1]), int(X.shape[2])),
            wavelengths=np.asarray(data.wavelengths, dtype=np.float64),
        ):
            return
        self.state.load_dataset(X, data.wavelengths, data.meta)
        self._cube_path = os.path.abspath(dataset.source_path)
        self.apply_shown_calibration()
        self._refresh_canvas_image()
        self._eda_view.refresh_roi_overlays()
        self._eda_view.refresh_attachment_state()
        self._eda_view.update_project_status()
        self._update_status_strip()
        self._refresh_data_file_choices()
        if self._pending_roi_selection is not None:
            pending, self._pending_roi_selection = self._pending_roi_selection, None
            self._canvas.select_roi(pending)  # also when the browser already selects it
            self._on_canvas_roi_selected(pending)
        from ._recent import add as _recent_add

        _recent_add(dataset.source_path)
        project_id = self.project_state.project_meta.get("project_id")
        if project_id:
            save_last_data_file(str(project_id), dataset_id)
        logger.info("Showing data file {} ({})", dataset.source_path, self.state.y_unit)

    def apply_shown_calibration(self) -> None:
        """Convert the shown data file with its stored calibration (references or levels).

        The Explore signal stays the one the user chose when this file can
        give it; otherwise (or before any choice) it is the calibration's
        working signal. Only the signals the file can give are enabled.
        """
        dataset_id = self._eda_view.shown_dataset_id()
        if dataset_id is None:
            self._eda_view.set_signal_choices(viewable_signal_units(None))
            return
        calibration = self.project_state.active_calibration(dataset_id)
        levels, unit = _calibration_view(calibration, self.state.meta)
        allowed = viewable_signal_units(calibration)
        if self._eda_view.chosen_signal in allowed:
            unit = str(self._eda_view.chosen_signal)
        self.state.set_calibration(**levels)
        self.state.set_y_unit(unit)
        self._eda_view.set_signal_choices(allowed)

    def _registered_dataset_id(self, path: str) -> str | None:
        target = os.path.normcase(os.path.abspath(path))
        return next((dataset_id for dataset_id, dataset in self.project_state.datasets.items()
                     if os.path.normcase(os.path.abspath(dataset.source_path)) == target), None)

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
            if not os.path.exists(dataset.source_path):
                name += " (not found)"
            combo.addItem(name, dataset_id)
            combo.setItemData(combo.count() - 1, dataset.source_path,
                              Qt.ItemDataRole.ToolTipRole)
        if shown is None:
            label = ("Data file: none" if not self._cube_path
                     else f"{os.path.basename(self._cube_path)} (not in project)")
            combo.insertItem(0, label, None)
        combo.insertSeparator(combo.count())
        combo.addItem("Add data file...", _ADD_DATA_FILE)
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
            self.show_data_file(str(payload.get("id")))

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
            logger.opt(exception=True).warning("False-colour preview could not be drawn")
            self._canvas.clear_image()
        self._canvas.set_crosshair(s.frame, s.row)

    def _populate_open_recent(self) -> None:
        from ._recent import load as _recent_load

        self._menu_open_recent.clear()
        paths = [p for p in _recent_load() if os.path.isfile(p)]
        if not paths:
            act = _add_required_action(self._menu_open_recent, "(No recent files)")
            act.setEnabled(False)
            return
        for path in paths:
            act = _add_required_action(self._menu_open_recent, path)
            act.triggered.connect(lambda checked, p=path: self._open_data_path(p))

    def _populate_view_menu(self) -> None:
        m = self._menu_view
        for dock in (
            self._project_browser_dock,
            self._inspector_dock,
            self._controls_dock,
            self._log_dock,
            self._hist_dock,
            self._dc3_dock,
            self._sensor_dock,
            self._spatial_dock,
            self._roi_table_dock,
        ):
            action = _require_qt_object(
                dock.toggleViewAction(), f"{dock.windowTitle()} toggle action"
            )
            m.addAction(action)
        m.addAction(self._spectrum_toggle_action)
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
        act = _add_required_action(m, "Reset Image View")
        act.triggered.connect(self._reset_image_view)
        act = _add_required_action(m, "Reset Layout")
        act.triggered.connect(self.reset_layout)

    def _populate_help_menu(self) -> None:
        m = self._menu_help

        act = _add_required_action(m, "View Log")
        act.triggered.connect(self._action_view_log)

        act = _add_required_action(m, "Log File...")
        act.triggered.connect(self._action_log_file)

        m.addSeparator()

        act = _add_required_action(m, "About engiNIR")
        act.triggered.connect(self._action_about)

    def _action_view_log(self) -> None:
        self._log_dock.setVisible(True)
        self._log_dock.raise_()

    def _action_log_file(self) -> None:
        """Show the log file path, with buttons to copy it and to open its location."""
        log_file = log_file_path()
        box = QMessageBox(self)
        box.setWindowTitle("Log File")
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
        box.setText(f"<b>engiNIR</b> v{__version__}<br><br>"
                    "Hyperspectral NIR spectroscopy library and viewer.")
        box.exec()

    def workspace_plot_frames(self) -> list[PlotFrame]:
        """The plots of the current workspace, including Model pages not shown yet."""
        current = self._workspace_tabs.currentWidget()
        if current is self._modeling_view:
            self._ensure_model_content_page()
            widget = self._model_diagnostics_widget
            return [] if widget is None else list(widget.grid.all_panels().values())
        if current is self._prediction_view:
            self._ensure_predict_content_page()
            return list(self._predict_graph_panel.panels.values())
        frames: list[PlotFrame] = [self._canvas_frame, self._spectrum_panel,
                                   *self._dock_frames.values()]
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
        """Clean up loguru sink when the window closes."""
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

    def _track_child_viewer(self, child: "HSIViewer") -> None:
        self._child_viewers.append(child)
        child.destroyed.connect(partial(self._detach_child_viewer, child))

    def _detach_child_viewer(self, child: "HSIViewer", *_args) -> None:
        if child in self._child_viewers:
            self._child_viewers.remove(child)

    def _refresh_panel_theme(self) -> None:
        """Restyle matplotlib/pyqtgraph surfaces that are outside Qt palette control."""
        self._canvas.restyle_theme()
        self._hist.restyle_theme()
        self._dc3.restyle_theme()
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
