"""HSIViewer QMainWindow shell for Explore, Model, and Predict workspaces."""

from __future__ import annotations

from typing import TypeVar
import os
import time
from functools import partial

import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
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
    QComboBox,
    QMenu,
    QMenuBar,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QFont, QIcon, QCloseEvent, QStandardItemModel
import importlib.resources

from .._checks import log_and_raise
from .._logging import add_log_sink, get_logger, log_file_path, remove_log_sink
from ..io import load_spectra

from ._eda import EDAView
from ._attachment import confirm_project_dataset_binding
from ._browser import ProjectBrowser
from ..io import file_format
from ._inspector import ProjectInspector
from ._jobs import BusyStatusWidget
from ._loading import ViewerBusyContext
from ._model_diagnostics import ModelDiagnosticsWidget
from ._modeling import AssignTargetValueDialog, ModelingView, NewROIDialog, TargetSelectionDialog
from ._prediction import PredictionMapWidget, PredictionView
from ._state import (
    DEFAULT_BLACK_LEVEL,
    DEFAULT_SOURCE_MODALITY,
    DEFAULT_WHITE_LEVEL,
    ProjectState,
    ViewerState,
)
from ._panels import (
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
    "AssignTargetValueDialog",
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

        try:
            icon_file = importlib.resources.files("enginir._resources").joinpath("enginir.ico")
            with importlib.resources.as_file(icon_file) as icon_path:
                self.setWindowIcon(QIcon(str(icon_path)))
        except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
            logger.debug("Window icon unavailable: {}", exc)
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
        self._model_spectral_mode = "coefficients"

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
        self._explore_content_page = QWidget()
        central_layout = QVBoxLayout(self._explore_content_page)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(8)
        self._image_view_label = QLabel("Spatial Image")
        self._btn_reset_image_view = QToolButton()
        self._btn_reset_image_view.setText("Reset View")
        self._btn_reset_image_view.clicked.connect(self._reset_image_view)
        header_layout.addWidget(self._image_view_label)
        header_layout.addStretch()
        header_layout.addWidget(self._btn_reset_image_view)

        central_layout.addWidget(header)
        central_layout.addWidget(self._canvas, 1)

        self._spectrum_panel = QGroupBox("Spectral Profile")
        self._spectrum_panel.setObjectName("SpectrumPanelContainer")
        self._spectrum_panel.setMinimumHeight(_SPECTRUM_PANEL_HEIGHT)
        self._spectrum_panel.setMaximumHeight(_SPECTRUM_PANEL_HEIGHT)
        spectrum_layout = QVBoxLayout(self._spectrum_panel)
        spectrum_layout.setContentsMargins(8, 8, 8, 8)
        spectrum_layout.setSpacing(6)
        spectrum_layout.addWidget(self._spectrum)
        central_layout.addWidget(self._spectrum_panel)

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

    def _ensure_predict_content_page(self) -> None:
        if self._predict_map_widget is not None:
            return
        old_page = self._predict_content_page
        self._predict_content_page = self._build_predict_content_page()
        self._replace_workspace_content_page(old_page, self._predict_content_page)

    def _build_model_content_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._model_content_title = QLabel("Model Diagnostics")
        self._model_workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._model_workspace_splitter.setChildrenCollapsible(False)

        self._model_graph_panel = QGroupBox("Diagnostic Graphs")
        graph_layout = QVBoxLayout(self._model_graph_panel)
        graph_layout.setContentsMargins(8, 8, 8, 8)
        graph_layout.setSpacing(6)
        self._model_diagnostics_widget = ModelDiagnosticsWidget()
        self._model_diagnostics_widget.set_spectral_mode(self._model_spectral_mode)
        graph_layout.addWidget(self._model_diagnostics_widget, 1)

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

        spectral_box = QGroupBox("Spectral Plot")
        spectral_layout = QVBoxLayout(spectral_box)
        spectral_layout.setContentsMargins(8, 8, 8, 8)
        spectral_layout.setSpacing(4)
        self._model_spectral_combo = QComboBox()
        self._model_spectral_combo.addItem("Regression coefficients", "coefficients")
        self._model_spectral_combo.addItem("Latent loadings (LV1)", "loadings")
        self._model_spectral_combo.setToolTip(
            "Choose what the bottom diagnostics spectral plot displays."
        )
        self._model_spectral_combo.setEnabled(False)
        self._model_spectral_combo.currentIndexChanged.connect(
            self._on_model_spectral_mode_changed
        )
        self._model_spectral_hint = QLabel("No coefficients or loadings are available yet.")
        self._model_spectral_hint.setStyleSheet(secondary_text_style())
        self._model_spectral_hint.setWordWrap(True)
        spectral_layout.addWidget(self._model_spectral_combo)
        spectral_layout.addWidget(self._model_spectral_hint)

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
        text_layout.addWidget(spectral_box)
        text_layout.addWidget(readiness_box)
        text_layout.addStretch(1)

        self._model_workspace_splitter.addWidget(self._model_graph_panel)
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

        self._predict_graph_panel = QGroupBox("Prediction Map")
        predict_graph_layout = QVBoxLayout(self._predict_graph_panel)
        predict_graph_layout.setContentsMargins(8, 8, 8, 8)
        predict_graph_layout.setSpacing(6)
        self._predict_map_widget = PredictionMapWidget()
        predict_graph_layout.addWidget(self._predict_map_widget, 1)

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
        compatibility_layout.addRow("Input contract", self._predict_contract_format_value)
        compatibility_layout.addRow("Signal contract", self._predict_contract_signal_value)
        compatibility_layout.addRow("Bands", self._predict_contract_bands_value)
        compatibility_layout.addRow("Map shape", self._predict_map_shape_value)
        compatibility_layout.addRow("Map range", self._predict_map_range_value)
        compatibility_layout.addRow("Calibration clipping", self._predict_clipping_value)

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
        self._hist_dock = QDockWidget("Histogram", self)
        self._hist_dock.setObjectName("HistogramDock")
        self._hist_dock.setWidget(self._hist)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._hist_dock)
        self.splitDockWidget(self._inspector_dock, self._hist_dock, Qt.Orientation.Vertical)

        self._dc3_dock = QDockWidget("RAW", self)
        self._dc3_dock.setObjectName("DC3Dock")
        self._dc3_dock.setWidget(self._dc3)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dc3_dock)
        self.splitDockWidget(self._hist_dock, self._dc3_dock, Qt.Orientation.Vertical)
        self._hist_dock.hide()
        self._dc3_dock.hide()

        # Right docks - Sensor Frame above Spatial Profile
        self._sensor_dock = QDockWidget("Sensor Frame", self)
        self._sensor_dock.setObjectName("SensorDock")
        self._sensor_dock.setWidget(self._sensor)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._sensor_dock)

        self._spatial_dock = QDockWidget("Spatial Profile", self)
        self._spatial_dock.setObjectName("SpatialProfileDock")
        self._spatial_dock.setWidget(self._spatial)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._spatial_dock)
        self.splitDockWidget(self._sensor_dock, self._spatial_dock, Qt.Orientation.Vertical)

        self._default_explore_plot_visibility = {
            "sensor": True,
            "spatial": True,
            "spectrum": True,
        }
        self._explore_plot_visibility = dict(self._default_explore_plot_visibility)

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
            dataset_label = os.path.basename(self._cube_path)

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
        self._sensor_dock.visibilityChanged.connect(self._on_sensor_dock_visibility_changed)
        self._spatial_dock.visibilityChanged.connect(self._on_spatial_dock_visibility_changed)

        self._sync_workspace_content_host()

    def _on_project_browser_selection_changed(self) -> None:
        payload = self._selected_project_browser_payload()
        if payload is not None and payload.get("kind") == "roi":
            self._canvas.select_roi(payload.get("id"))
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

        if is_explore:
            self._sensor_dock.setVisible(self._explore_plot_visibility["sensor"])
            self._spatial_dock.setVisible(self._explore_plot_visibility["spatial"])
            self._spectrum_panel.setVisible(self._explore_plot_visibility["spectrum"])
        else:
            self._sensor_dock.hide()
            self._spatial_dock.hide()
            self._spectrum_panel.hide()
        _require_qt_object(
            self._sensor_dock.toggleViewAction(), "sensor dock toggle action"
        ).setEnabled(is_explore)
        _require_qt_object(
            self._spatial_dock.toggleViewAction(), "spatial dock toggle action"
        ).setEnabled(is_explore)
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
        diagnostics_payload = self._modeling_view.workspace_diagnostics_payload()
        self._sync_model_spectral_options(diagnostics_payload)
        model_diagnostics_widget.set_spectral_mode(self._model_spectral_mode)
        model_diagnostics_widget.update_data(
            diagnostics_payload,
            empty_reason=model_summary["readiness"],
        )

    def _sync_model_spectral_options(self, payload: dict[str, object] | None) -> None:
        has_coefficients = bool(
            payload is not None and np.asarray(payload.get("coefficients", [])).size
        )
        has_loadings = bool(payload is not None and np.asarray(payload.get("loadings", [])).size)
        self._set_model_spectral_option_enabled("coefficients", has_coefficients)
        self._set_model_spectral_option_enabled("loadings", has_loadings)
        any_available = has_coefficients or has_loadings
        self._model_spectral_combo.setEnabled(any_available)
        if any_available:
            if not self._is_current_model_spectral_mode_available():
                self._model_spectral_mode = "coefficients" if has_coefficients else "loadings"
                index = self._model_spectral_combo.findData(self._model_spectral_mode)
                if index >= 0:
                    self._model_spectral_combo.setCurrentIndex(index)
            self._model_spectral_hint.setText(self._model_spectral_hint_text())
        else:
            self._model_spectral_hint.setText("No coefficients or loadings are available yet.")

    def _set_model_spectral_option_enabled(self, value: str, enabled: bool) -> None:
        index = self._model_spectral_combo.findData(value)
        if index < 0:
            return
        model = self._model_spectral_combo.model()
        if not isinstance(model, QStandardItemModel):
            return
        item = model.item(index)
        if item is not None:
            item.setEnabled(enabled)

    def _is_current_model_spectral_mode_available(self) -> bool:
        index = self._model_spectral_combo.currentIndex()
        if index < 0:
            return False
        model = self._model_spectral_combo.model()
        if not isinstance(model, QStandardItemModel):
            return False
        item = model.item(index)
        return bool(item is not None and item.isEnabled())

    def _on_model_spectral_mode_changed(self, _index: int) -> None:
        self._model_spectral_mode = str(
            self._model_spectral_combo.currentData() or "coefficients"
        )
        self._model_spectral_hint.setText(self._model_spectral_hint_text())
        if self._model_diagnostics_widget is not None:
            self._model_diagnostics_widget.set_spectral_mode(self._model_spectral_mode)

    def _model_spectral_hint_text(self) -> str:
        if self._model_spectral_mode == "loadings":
            return "Loadings show the first latent variable's spectral structure."
        return "Coefficients show how each band contributes to the model prediction."

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

    def _on_sensor_dock_visibility_changed(self, visible: bool) -> None:
        self._on_explore_plot_visibility_changed("sensor", visible)

    def _on_spatial_dock_visibility_changed(self, visible: bool) -> None:
        self._on_explore_plot_visibility_changed("spatial", visible)

    def _on_spectrum_panel_visibility_action(self, visible: bool) -> None:
        self._on_explore_plot_visibility_changed("spectrum", visible)
        self._spectrum_panel.setVisible(visible)

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
        self._menu_help = _add_required_menu(mb, "Help")
        self._populate_file_menu()
        self._populate_view_menu()
        self._populate_help_menu()
        self._default_layout_state = self.saveState()
        self._status_label = QLabel()
        status_bar = _require_qt_object(self.statusBar(), "main status bar")
        status_bar.addPermanentWidget(self._status_label, 1)
        self._update_status_strip()

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
        if self.project_state.project_path is not None and not self.has_attached_dataset():
            self._attach_dataset_path(path)
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

    def _attach_dataset_path(
        self,
        path: str,
        *,
        dark: str | None = None,
        white: str | None = None,
    ) -> None:
        busy = self._viewer_busy_context(f"Attaching dataset: {path}")
        busy.begin()
        try:
            X, wavelengths, meta = load_spectra(str(path))
            if X.ndim == 2:
                X = X[:, np.newaxis, :]
            normalized_path = os.path.abspath(str(path))

            if self.project_state.project_path is not None:
                if not confirm_project_dataset_binding(
                    self,
                    project_path=self.project_state.project_path,
                    project_state=self.project_state,
                    dataset_path=normalized_path,
                    source_format=file_format(normalized_path),
                    source_modality=str(meta.get("source_modality") or DEFAULT_SOURCE_MODALITY),
                    cube_shape=(int(X.shape[0]), int(X.shape[1]), int(X.shape[2])),
                    wavelengths=np.asarray(wavelengths, dtype=np.float64),
                ):
                    return

            dark_ref = None
            if dark is not None:
                dark_X, _, _ = load_spectra(str(dark))
                dark_ref = dark_X.mean(axis=0) if dark_X.ndim == 3 else dark_X

            white_ref = None
            if white is not None:
                white_X, _, _ = load_spectra(str(white))
                white_ref = white_X.mean(axis=0) if white_X.ndim == 3 else white_X

            black_level = float(meta.get("black_level", DEFAULT_BLACK_LEVEL))
            white_level = float(meta.get("white_level", DEFAULT_WHITE_LEVEL))
            self.state.load_dataset(
                X,
                wavelengths,
                meta,
                    black_level=black_level,
                white_level=white_level,
                dark_ref=None if dark_ref is None else np.asarray(dark_ref, dtype=np.float32),
                white_ref=None if white_ref is None else np.asarray(white_ref, dtype=np.float32),
            )
            self._cube_path = normalized_path
            self._refresh_canvas_image()
            self._eda_view.refresh_attachment_state()
            self._eda_view.update_project_status()
            self._update_status_strip()
            from ._recent import add as _recent_add

            _recent_add(str(path))
        except (OSError, ValueError) as exc:
            busy.end()
            logger.error("Failed to attach dataset {}: {}", path, exc)
            QMessageBox.critical(self, "Cannot open file", str(exc))
        finally:
            busy.end()

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

    def reset_layout(self) -> None:
        """Restore default dock layout (positions, splits, visibility)."""
        self._explore_plot_visibility = dict(self._default_explore_plot_visibility)
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
        self._log_panel.restyle_theme()
