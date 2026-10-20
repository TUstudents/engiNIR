"""Prediction view widgets for project-backed model inference."""

from __future__ import annotations

from threading import Event
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .._checks import log_and_raise
from .._workflow import resolve_dataset, resolve_prediction_calibration
from .._logging import get_logger
from ..io import load_spectra
from ..prediction import PredictionProgressEvent, predict_with_model_run, preflight_prediction
from ..project import (
    CalibrationSpec,
    PredictionRun,
    check_calibration_signal_domain_compatibility,
)
from ._jobs import BackgroundWorker, BusyStatusWidget, start_background_job
from ._state import ProjectState
from ._theme import panel_colors, secondary_text_style
from ..project._ids import now_iso
from ._ui_helpers import maybe_tight_layout, style_axes

logger = get_logger(__name__)


def _array_payload_fingerprint(value: Any) -> tuple[Any, ...]:
    """Return a stable-enough cache token for array-like prediction payloads."""
    if value is None:
        return ("none",)
    if not isinstance(value, np.ndarray):
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None or dtype is not None:
            shape_token = tuple(int(dim) for dim in shape) if shape is not None else ()
            return ("array_like", type(value).__name__, shape_token, str(dtype))
        return ("object", type(value).__name__, repr(value))
    arr = value
    return (
        "array",
        int(arr.ctypes.data),
        tuple(int(dim) for dim in arr.shape),
        tuple(int(stride) for stride in arr.strides),
        str(arr.dtype),
    )


def _load_prediction_metadata(data_path: str) -> Any | None:
    """Load dataset metadata cheaply, or return None when only a full load works.

    Formats without header-only support defer the full load, dataset/calibration
    resolution, and preflight to the background prediction job so the UI thread
    never blocks on a large cube read.
    """
    try:
        return load_spectra(data_path, header_only=True)
    except TypeError:
        logger.warning(
            "Loader for {} does not support header-only metadata; deferring full "
            "load and preflight to the prediction job",
            data_path,
        )
        return None


def _resolve_prediction_target(
    context: Any,
    *,
    data_path: str,
    spectral_data: Any,
    dataset_id: str | None,
    calibration: CalibrationSpec | None,
    default_calibration: CalibrationSpec | None,
) -> tuple[Any, CalibrationSpec, bool]:
    """Resolve the prediction dataset and applied calibration from one snapshot.

    The dataset's active calibration is used unless ``calibration`` is given;
    a dataset without one gets ``default_calibration``.
    """
    dataset, _ = resolve_dataset(context.datasets, data_path, spectral_data,
                                 dataset_id=dataset_id)
    calibration_input = calibration
    if calibration_input is None:
        active_id = context.active_calibration_ids.get(dataset.dataset_id)
        if not active_id or active_id not in context.calibrations:
            calibration_input = default_calibration
    resolved_calibration, calibration_is_new = resolve_prediction_calibration(
        dataset, context.calibrations, context.active_calibration_ids, calibration_input)
    return dataset, resolved_calibration, calibration_is_new


def _persist_new_prediction_context(
    *,
    project_path: str,
    dataset: Any,
    calibration: CalibrationSpec,
    dataset_is_new: bool,
    calibration_is_new: bool,
) -> None:
    """Persist new dataset/calibration specs into the job's project file.

    Running this inside the worker keeps every file write of one prediction job
    targeting the project that was open at submission, so the persisted run
    never references specs that were written into a different project.
    """
    from ..io.project_store import save_calibration_spec, save_dataset_ref

    if dataset_is_new:
        payload = dataset.to_dict()
        del payload["dataset_id"]
        save_dataset_ref(project_path, dataset.dataset_id, payload)
    if calibration_is_new:
        payload = calibration.to_dict()
        del payload["calibration_id"]
        del payload["dataset_id"]
        save_calibration_spec(
            project_path,
            calibration.dataset_id,
            calibration.calibration_id,
            payload,
        )


class _PredictionProgressRelay(QObject):
    progressed = pyqtSignal(object)


class PredictionRunDialog(QDialog):
    """Dialog for selecting one model run and dataset for prediction."""

    def __init__(
        self,
        model_run_ids: list[str],
        *,
        default_model_run_id: str | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run Prediction")

        layout = QFormLayout(self)

        self._model_combo = QComboBox()
        for model_run_id in model_run_ids:
            self._model_combo.addItem(model_run_id, model_run_id)
        layout.addRow("Model run:", self._model_combo)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("path/to/dataset")
        btn_browse = QToolButton()
        btn_browse.setText("Browse...")
        btn_browse.clicked.connect(self._browse)
        path_row.addWidget(self._path_edit)
        path_row.addWidget(btn_browse)
        layout.addRow("Dataset:", path_row)

        self._dataset_id_edit = QLineEdit()
        self._dataset_id_edit.setPlaceholderText("optional")
        layout.addRow("Dataset id:", self._dataset_id_edit)

        self._explicit_calibration = QCheckBox("Override existing calibration")
        self._explicit_calibration.toggled.connect(self._sync_calibration_fields)
        layout.addRow(self._explicit_calibration)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["dark_white_reference", "precalibrated_import", "raw_dn"])
        self._mode_combo.currentTextChanged.connect(self._sync_calibration_fields)
        layout.addRow("Mode:", self._mode_combo)

        self._output_unit_combo = QComboBox()
        self._output_unit_combo.addItems(["Reflectance", "Absorbance", "DN"])
        layout.addRow("Output unit:", self._output_unit_combo)

        self._black_level_spin = QDoubleSpinBox()
        self._black_level_spin.setRange(-1e9, 1e9)
        self._black_level_spin.setValue(128.0)
        layout.addRow("Black level:", self._black_level_spin)

        self._white_level_spin = QDoubleSpinBox()
        self._white_level_spin.setRange(-1e9, 1e9)
        self._white_level_spin.setValue(3500.0)
        layout.addRow("White level:", self._white_level_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        if default_model_run_id is not None:
            current_index = self._model_combo.findData(default_model_run_id)
            if current_index >= 0:
                self._model_combo.setCurrentIndex(current_index)
        self._sync_calibration_fields()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Prediction Dataset", "", "All files (*.*)"
        )
        if path:
            self._path_edit.setText(path)

    def _sync_calibration_fields(self, _text: str | None = None) -> None:
        mode = self._mode_combo.currentText()
        for widget in (self._mode_combo, self._output_unit_combo):
            widget.setEnabled(True)
        level_enabled = mode == "dark_white_reference"
        self._black_level_spin.setEnabled(level_enabled)
        self._white_level_spin.setEnabled(level_enabled)

    def _on_accept(self) -> None:
        if not self.data_path:
            QMessageBox.warning(self, "Validation", "Dataset path is required.")
            return
        mode = self._mode_combo.currentText()
        output_unit = self._output_unit_combo.currentText()
        if (
            mode == "dark_white_reference"
            and self._white_level_spin.value() == self._black_level_spin.value()
        ):
            QMessageBox.warning(self, "Validation", "Black and white levels must differ.")
            return
        if mode == "raw_dn" and output_unit != "DN":
            QMessageBox.warning(self, "Validation", "Mode 'raw_dn' requires output unit 'DN'.")
            return
        if mode in {"dark_white_reference", "precalibrated_import"} and output_unit == "DN":
            QMessageBox.warning(
                self,
                "Validation",
                f"Mode '{mode}' requires output unit 'Reflectance' or 'Absorbance'.",
            )
            return
        self.accept()

    @property
    def model_run_id(self) -> str:
        """Return the selected model-run id."""
        return str(self._model_combo.currentData())

    @property
    def data_path(self) -> str:
        """Return the prediction dataset path entered by the user."""
        return self._path_edit.text().strip()

    @property
    def dataset_id(self) -> str | None:
        """Return the optional dataset id override for the prediction dataset."""
        text = self._dataset_id_edit.text().strip()
        return text or None

    @property
    def calibration(self) -> CalibrationSpec | None:
        """Return an explicit calibration override, or ``None`` to reuse defaults."""
        if not self._explicit_calibration.isChecked():
            return None
        return self.default_calibration

    @property
    def default_calibration(self) -> CalibrationSpec:
        """Return a provisional calibration built from the dialog fields."""
        return self._calibration_from_fields()

    def _calibration_from_fields(self) -> CalibrationSpec:
        mode = self._mode_combo.currentText()
        output_unit = self._output_unit_combo.currentText()
        black_level = None
        white_level = None
        if mode == "dark_white_reference":
            black_level = float(self._black_level_spin.value())
            white_level = float(self._white_level_spin.value())
        return CalibrationSpec(
            calibration_id="",
            dataset_id="",
            mode=mode,
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=black_level,
            white_level=white_level,
            output_signal_unit=output_unit,
            created_at=now_iso(),
        )


class PredictionMapWidget(QWidget):
    """Read-only spatial prediction map surface for the Predict workspace."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fig = Figure(figsize=(8, 5), facecolor=panel_colors()["bg"])
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._colorbar: Colorbar | None = None
        self._last_payload: dict[str, Any] | None = None
        self._last_empty_reason: str | None = None
        self._last_layout_size: tuple[int, int] | None = None
        self._current_display_map: np.ndarray | None = None
        self._rendered_output_index = 0
        self._placeholder_label = QLabel("")
        self._placeholder_label.setStyleSheet(secondary_text_style())
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(secondary_text_style())
        self._cursor_label = QLabel("")
        self._cursor_label.setStyleSheet(secondary_text_style())
        self._output_label = QLabel("Output")
        self._output_label.setStyleSheet(secondary_text_style())
        self._output_combo = QComboBox()
        self._output_combo.currentIndexChanged.connect(self._on_output_changed)
        self._output_label.setVisible(False)
        self._output_combo.setVisible(False)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self._output_label)
        output_layout.addWidget(self._output_combo)
        output_layout.addStretch(1)
        layout.addWidget(self._toolbar)
        layout.addLayout(output_layout)
        layout.addWidget(self._canvas)
        layout.addWidget(self._stats_label)
        layout.addWidget(self._cursor_label)
        layout.addWidget(self._placeholder_label)

        self.clear()

    def update_data(self, payload: dict[str, Any] | None, *, empty_reason: str = "") -> None:
        """Render one persisted spatial prediction map when available."""
        if payload is None:
            self.clear(empty_reason=empty_reason)
            return
        output_index = self._selected_output_index(payload.get("spatial_map"))
        if (
            payload is self._last_payload
            and empty_reason == self._last_empty_reason
            and output_index == self._rendered_output_index
        ):
            return

        spatial_map = payload.get("spatial_map")
        if spatial_map is None:
            self.clear(
                empty_reason=empty_reason or "No spatial prediction map stored for this run."
            )
            return

        self._configure_output_selector(spatial_map, payload=payload)
        output_index = self._selected_output_index(spatial_map)
        arr = self._display_spatial_map(spatial_map, output_index=output_index)
        if arr is None:
            self.clear(
                empty_reason=empty_reason or "No spatial prediction map stored for this run."
            )
            return

        colors = panel_colors()
        self._fig.set_facecolor(colors["bg"])
        self._remove_colorbar()
        self._ax.clear()
        style_axes(self._ax)
        image = self._ax.imshow(
            arr,
            cmap="viridis",
            origin="upper",
            aspect="auto",
            interpolation="nearest",
        )
        self._current_display_map = arr
        self._rendered_output_index = output_index
        units = str(payload.get("target_unit", ""))
        unit_suffix = f" ({units})" if units else ""
        self._ax.set_title(f"Prediction Map{unit_suffix}", color=colors["fg"], fontsize=9)
        self._ax.set_xlabel("Frame", color=colors["fg"])
        self._ax.set_ylabel("Row", color=colors["fg"])
        self._colorbar = self._fig.colorbar(image, ax=self._ax, fraction=0.046, pad=0.04)
        self._style_colorbar(unit_suffix=unit_suffix)

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            self._stats_label.setText("Map stats: no finite prediction values.")
        else:
            self._stats_label.setText(
                "Map stats: "
                f"shape={arr.shape[0]} x {arr.shape[1]} | "
                f"min={float(np.min(finite)):.4g} | "
                f"mean={float(np.mean(finite)):.4g} | "
                f"max={float(np.max(finite)):.4g}"
            )
        self._placeholder_label.setText("")
        self._cursor_label.setText("")
        self._last_layout_size = maybe_tight_layout(
            self._fig,
            self._canvas,
            pad=1.0,
            last_size=self._last_layout_size,
        )
        self._canvas.draw_idle()
        self._last_payload = payload
        self._last_empty_reason = empty_reason

    def clear(self, *, empty_reason: str = "") -> None:
        """Show an empty prediction-map placeholder."""
        if self._last_payload is None and empty_reason == self._last_empty_reason:
            return
        colors = panel_colors()
        self._fig.set_facecolor(colors["bg"])
        self._remove_colorbar()
        self._ax.clear()
        self._current_display_map = None
        self._output_label.setVisible(False)
        self._output_combo.setVisible(False)
        style_axes(self._ax)
        self._ax.set_title("Prediction Map", color=colors["fg"], fontsize=9)
        self._ax.set_xlabel("Frame", color=colors["fg"])
        self._ax.set_ylabel("Row", color=colors["fg"])
        self._stats_label.setText("")
        self._cursor_label.setText("")
        self._placeholder_label.setText(empty_reason or "No saved prediction results yet.")
        self._last_layout_size = maybe_tight_layout(
            self._fig,
            self._canvas,
            pad=1.0,
            last_size=self._last_layout_size,
        )
        self._canvas.draw_idle()
        self._last_payload = None
        self._last_empty_reason = empty_reason

    @staticmethod
    def _display_spatial_map(spatial_map: Any, *, output_index: int = 0) -> np.ndarray | None:
        """Return a prediction map in the same row/frame orientation as the canvas."""
        arr = np.asarray(spatial_map, dtype=np.float32)
        if arr.ndim < 2:
            return None
        if arr.ndim > 2:
            if output_index < 0 or output_index >= arr.shape[-1]:
                return None
            arr = arr[..., output_index]
        return arr.T

    def _selected_output_index(self, spatial_map: Any) -> int:
        arr = np.asarray(spatial_map)
        if arr.ndim <= 2:
            return 0
        return max(0, min(self._output_combo.currentIndex(), int(arr.shape[-1]) - 1))

    def _configure_output_selector(self, spatial_map: Any, *, payload: dict[str, Any]) -> None:
        arr = np.asarray(spatial_map)
        n_outputs = int(arr.shape[-1]) if arr.ndim > 2 else 1
        show_selector = n_outputs > 1
        self._output_label.setVisible(show_selector)
        self._output_combo.setVisible(show_selector)
        if not show_selector:
            return
        names = payload.get("output_names")
        if not isinstance(names, (list, tuple)) or len(names) != n_outputs:
            names = [f"Output {idx + 1}" for idx in range(n_outputs)]
        current = self._selected_output_index(spatial_map)
        self._output_combo.blockSignals(True)
        try:
            self._output_combo.clear()
            self._output_combo.addItems([str(name) for name in names])
            self._output_combo.setCurrentIndex(current)
        finally:
            self._output_combo.blockSignals(False)

    def _on_output_changed(self, _index: int) -> None:
        payload = self._last_payload
        empty_reason = self._last_empty_reason or ""
        if payload is None:
            return
        self._last_payload = None
        self.update_data(payload, empty_reason=empty_reason)

    def _on_motion(self, event: Any) -> None:
        arr = self._current_display_map
        if arr is None or event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            self._cursor_label.setText("")
            return
        frame = int(round(float(event.xdata)))
        row = int(round(float(event.ydata)))
        if row < 0 or frame < 0 or row >= arr.shape[0] or frame >= arr.shape[1]:
            self._cursor_label.setText("")
            return
        value = float(arr[row, frame])
        self._cursor_label.setText(f"Frame {frame} | Row {row} | Value {value:.4g}")

    def _remove_colorbar(self) -> None:
        if self._colorbar is None:
            return
        self._colorbar.remove()
        self._colorbar = None

    def _style_colorbar(self, *, unit_suffix: str) -> None:
        if self._colorbar is None:
            return
        colors = panel_colors()
        label = f"Prediction{unit_suffix}"
        self._colorbar.set_label(label, color=colors["fg"], fontsize=8)
        self._colorbar.ax.tick_params(colors=colors["fg"], labelsize=8)

    def restyle_theme(self) -> None:
        """Repaint the widget for the current theme."""
        self._last_layout_size = None
        payload = self._last_payload
        empty_reason = self._last_empty_reason or ""
        if payload is None:
            self._last_empty_reason = None
            self.clear(empty_reason=empty_reason)
            return
        self._last_payload = None
        self.update_data(payload, empty_reason=empty_reason)


class PredictionView(QWidget):
    """Prediction controls for running saved model runs on new datasets."""

    def __init__(
        self,
        project_state: ProjectState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project_state = project_state
        self._selected_prediction_run_id: str | None = None
        self._job_thread: QThread | None = None
        self._job_worker: BackgroundWorker | None = None
        self._job_relay: object | None = None
        self._job_progress_relay: _PredictionProgressRelay | None = None
        self._job_cancel_event: Event | None = None
        self._job_project_path: str | None = None
        self._last_prediction_phase: str | None = None
        self._cached_workspace_visual_token: tuple[Any, ...] | None = None
        self._cached_workspace_visual_payload: dict[str, Any] | None = None
        self._cached_summary_map_token: tuple[Any, ...] | None = None
        self._cached_summary_map_fields: tuple[str, str] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_group())

        self.project_state.active_model_changed.connect(self._update_status)
        self.project_state.model_runs_changed.connect(self._update_status)
        self.project_state.predictions_changed.connect(self._update_status)
        self.project_state.predictions_changed.connect(self._invalidate_prediction_workspace_cache)
        self.project_state.prediction_completed.connect(self._on_prediction_completed)
        self._update_status()

    def _build_group(self) -> QGroupBox:
        box = QGroupBox("Prediction")
        layout = QVBoxLayout(box)
        layout.setSpacing(2)

        self._prediction_status = QLabel("Prediction: none")
        self._prediction_status.setStyleSheet(secondary_text_style())
        layout.addWidget(self._prediction_status)

        self._readiness_status = QLabel("Ready: fit or load a model run.")
        self._readiness_status.setStyleSheet(secondary_text_style())
        self._readiness_status.setWordWrap(True)
        layout.addWidget(self._readiness_status)

        button_row = QHBoxLayout()
        self._btn_predict = QToolButton()
        self._btn_predict.setText("Predict...")
        self._btn_predict.clicked.connect(self.predict_btn_clicked)
        button_row.addWidget(self._btn_predict)
        self._btn_cancel = QToolButton()
        self._btn_cancel.setText("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_prediction_clicked)
        button_row.addWidget(self._btn_cancel)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self._service_status = BusyStatusWidget(idle_text="Service: idle", parent=self)
        layout.addWidget(self._service_status)
        return box

    def predict_btn_clicked(self) -> None:
        if self._job_thread is not None:
            return
        if self.project_state.project_path is None:
            QMessageBox.warning(
                self, "No project open", "Open a project before running prediction."
            )
            return
        if not self.project_state.model_runs:
            QMessageBox.information(self, "No models", "Fit or load a model run before prediction.")
            return
        dlg = PredictionRunDialog(
            list(self.project_state.model_runs),
            default_model_run_id=self.project_state.active_model_name,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            context = self.project_state.build_prediction_job_context(model_run_id=dlg.model_run_id)
        except Exception as exc:
            logger.error("Prediction failed while building job context: {}", exc)
            QMessageBox.warning(self, "Prediction failed", str(exc))
            return

        data_path = dlg.data_path
        calibration = dlg.calibration
        default_calibration = dlg.default_calibration
        dataset_id = dlg.dataset_id
        dataset: Any = None
        resolved_calibration: CalibrationSpec | None = None
        calibration_is_new = False
        dataset_is_new = False
        try:
            if calibration is not None:
                check_calibration_signal_domain_compatibility(
                    calibration=calibration,
                    signal_domain=context.signal_domain,
                )
            spectral_data = _load_prediction_metadata(data_path)
            if spectral_data is not None:
                dataset, resolved_calibration, calibration_is_new = _resolve_prediction_target(
                    context,
                    data_path=data_path,
                    spectral_data=spectral_data,
                    dataset_id=dataset_id,
                    calibration=calibration,
                    default_calibration=default_calibration,
                )
                dataset_is_new = dataset.dataset_id not in context.datasets
                preflight_prediction(
                    project_path=context.project_path,
                    model_run=context.model_run,
                    signal_domain=context.signal_domain,
                    recipe=context.recipe,
                    dataset=dataset,
                    calibration=resolved_calibration,
                    model_registry=context.model_registry,
                )
        except Exception as exc:
            logger.error("Prediction failed during preflight: {}", exc)
            QMessageBox.warning(self, "Prediction failed", str(exc))
            return

        cancel_event = Event()
        self._job_cancel_event = cancel_event
        progress_relay = _PredictionProgressRelay(self)
        progress_relay.progressed.connect(self._on_prediction_progress)
        self._job_progress_relay = progress_relay

        def emit_progress(event: PredictionProgressEvent) -> None:
            progress_relay.progressed.emit(event)

        def job() -> tuple[Any, Any, Any, bool, bool]:
            local_spectral = spectral_data
            local_dataset = dataset
            local_calibration = resolved_calibration
            local_calibration_is_new = calibration_is_new
            local_dataset_is_new = dataset_is_new
            if local_spectral is None:
                local_spectral = load_spectra(data_path)
                local_dataset, local_calibration, local_calibration_is_new = (
                    _resolve_prediction_target(
                        context,
                        data_path=data_path,
                        spectral_data=local_spectral,
                        dataset_id=dataset_id,
                        calibration=calibration,
                        default_calibration=default_calibration,
                    )
                )
                local_dataset_is_new = local_dataset.dataset_id not in context.datasets
            if local_dataset is None or local_calibration is None:
                log_and_raise("Prediction target resolution produced no dataset/calibration")
            _persist_new_prediction_context(
                project_path=context.project_path,
                dataset=local_dataset,
                calibration=local_calibration,
                dataset_is_new=local_dataset_is_new,
                calibration_is_new=local_calibration_is_new,
            )
            prediction_result = predict_with_model_run(
                project_path=context.project_path,
                model_run=context.model_run,
                signal_domain=context.signal_domain,
                recipe=context.recipe,
                dataset=local_dataset,
                calibration=local_calibration,
                model_registry=context.model_registry,
                data_path=data_path,
                spectral_data=local_spectral,
                preprocessing_recipe_id=context.preprocessing_recipe_id,
                progress_callback=emit_progress,
                cancel_callback=cancel_event.is_set,
                persist_artifacts=True,
                return_arrays=False,
            )
            return (
                prediction_result,
                local_dataset,
                local_calibration,
                local_dataset_is_new,
                local_calibration_is_new,
            )

        self._job_project_path = context.project_path
        self._set_busy(True, "Running prediction...")
        self._job_thread, self._job_worker, self._job_relay = start_background_job(
            owner=self,
            fn=job,
            on_success=self._on_prediction_success,
            on_error=lambda msg: self._on_prediction_error("Prediction failed", msg),
            on_finished=self._clear_job,
        )

    def _update_status(self, _value: str | None = None) -> None:
        active_model = self.project_state.active_model_name
        if active_model is None and self.project_state.model_runs:
            active_model = next(reversed(self.project_state.model_runs))
        if active_model is None:
            self._prediction_status.setText("Prediction: none")
            self._readiness_status.setText("Ready: fit or load a model run.")
            return
        self._prediction_status.setText(f"Ready: model {active_model}")
        self._readiness_status.setText("Ready: choose a dataset and run prediction.")

    def select_prediction_run(self, prediction_run_id: str | None) -> None:
        selected_prediction_run_id = None if prediction_run_id is None else str(prediction_run_id)
        if selected_prediction_run_id not in self.project_state.predictions:
            selected_prediction_run_id = None
        self._selected_prediction_run_id = selected_prediction_run_id

    def workspace_summary(self) -> dict[str, str]:
        prediction = self._resolved_workspace_prediction_run()
        dataset = (
            None if prediction is None else self.project_state.datasets.get(prediction.dataset_id)
        )
        calibration = (
            None
            if prediction is None
            else self.project_state.calibrations.get(prediction.calibration_id)
        )
        model_run = (
            None
            if prediction is None
            else self.project_state.model_runs.get(prediction.model_run_id)
        )
        signal_domain = None
        if prediction is not None:
            signal_domain = self.project_state.signal_domains.get(prediction.signal_domain_id)

        target_text = "none"
        if model_run is not None:
            sample_set = self.project_state.sample_sets.get(model_run.sample_set_id)
            if sample_set is not None:
                target = self.project_state.targets.get(sample_set.target_id)
                if target is not None:
                    target_text = (
                        target.name if not target.unit else f"{target.name} ({target.unit})"
                    )

        model_text = "none"
        if model_run is not None:
            model_text = f"{model_run.model_type} [{model_run.model_run_id}]"

        dataset_text = "none"
        if dataset is not None:
            path_name = Path(dataset.source_path).name if dataset.source_path else "none"
            dataset_text = f"{dataset.dataset_id} | {path_name}"

        calibration_text = "none"
        if calibration is not None:
            calibration_text = f"{calibration.calibration_id} | {calibration.mode} -> {calibration.output_signal_unit}"

        format_text = "-"
        signal_text = "-"
        bands_text = "-"
        if signal_domain is not None:
            format_text = f"{signal_domain.input_format} | {signal_domain.source_modality}"
            signal_text = (
                f"{signal_domain.required_signal_unit} | {signal_domain.required_calibration_mode}"
            )
            bands_text = str(signal_domain.n_bands)

        map_shape_text = "-"
        map_range_text = "-"
        clipping_text = "-"
        if prediction is not None:
            map_shape_text, map_range_text = self._prediction_map_summary_fields(prediction)
            clipping_text = self._prediction_clipping_summary(prediction)

        status_text = self._prediction_status.text()
        readiness_text = self._readiness_status.text()
        if prediction is not None:
            status_text = f"Prediction: {prediction.prediction_run_id}"
            readiness_text = "Ready: run another prediction or inspect results."

        return {
            "title": "Prediction Results",
            "status": status_text,
            "readiness": readiness_text,
            "service": self._service_status.label.text(),
            "prediction": "none" if prediction is None else prediction.prediction_run_id,
            "model": model_text,
            "dataset": dataset_text,
            "calibration": calibration_text,
            "target": target_text,
            "format": format_text,
            "signal": signal_text,
            "bands": bands_text,
            "map_shape": map_shape_text,
            "map_range": map_range_text,
            "clipping": clipping_text,
        }

    def workspace_visual_payload(self) -> dict[str, Any] | None:
        """Return one stable visual payload for the selected Predict workspace run.

        The returned object is cached so repeated shell refreshes do not rebuild
        identical prediction-map payloads or trigger redundant redraws.
        """
        prediction = self._resolved_workspace_prediction_run()
        if prediction is None:
            self._cached_workspace_visual_token = None
            self._cached_workspace_visual_payload = None
            return None
        model_run = self.project_state.model_runs.get(prediction.model_run_id)
        target_unit = ""
        if model_run is not None:
            sample_set = self.project_state.sample_sets.get(model_run.sample_set_id)
            if sample_set is not None:
                target = self.project_state.targets.get(sample_set.target_id)
                if target is not None:
                    target_unit = target.unit
        payload: dict[str, Any] = {"target_unit": target_unit}
        artifact_refs = prediction.artifact_refs
        for key in ("output_names", "output_axis", "output_index", "n_outputs"):
            if key in artifact_refs:
                payload[key] = artifact_refs[key]
        base_token = (
            prediction.prediction_run_id,
            target_unit,
            self.project_state.prediction_artifact_version(prediction.prediction_run_id),
            _array_payload_fingerprint(artifact_refs.get("spatial_map_shape")),
            _array_payload_fingerprint(artifact_refs.get("prediction_shape")),
            tuple(str(name) for name in artifact_refs.get("output_names", [])),
        )
        if self._cached_workspace_visual_token == base_token:
            return self._cached_workspace_visual_payload
        try:
            payload["spatial_map"] = self.project_state.load_prediction_spatial_map_preview(
                prediction.prediction_run_id
            )
        except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
            logger.warning(
                "Could not load prediction '{}' spatial-map preview: {}",
                prediction.prediction_run_id,
                exc,
            )
        self._cached_workspace_visual_token = base_token
        self._cached_workspace_visual_payload = payload
        return payload

    def _prediction_map_summary_fields(self, prediction: PredictionRun) -> tuple[str, str]:
        """Return cached map-shape/map-range summary strings for one prediction run."""
        artifact_refs = prediction.artifact_refs
        metadata_token = (
            prediction.prediction_run_id,
            self.project_state.prediction_artifact_version(prediction.prediction_run_id),
            _array_payload_fingerprint(artifact_refs.get("spatial_shape")),
            artifact_refs.get("prediction_min"),
            artifact_refs.get("prediction_max"),
        )
        if (
            self._cached_summary_map_token == metadata_token
            and self._cached_summary_map_fields is not None
        ):
            return self._cached_summary_map_fields
        spatial_shape = artifact_refs.get("spatial_shape")
        prediction_min = artifact_refs.get("prediction_min")
        prediction_max = artifact_refs.get("prediction_max")
        if spatial_shape is not None and prediction_min is not None and prediction_max is not None:
            shape = tuple(int(dim) for dim in np.asarray(spatial_shape).tolist())
            if len(shape) >= 2:
                fields = (
                    f"{shape[1]} x {shape[0]}",
                    f"{float(prediction_min):.4g} .. {float(prediction_max):.4g}",
                )
                self._cached_summary_map_token = metadata_token
                self._cached_summary_map_fields = fields
                return fields

        map_shape_text = "-"
        map_range_text = "-"
        spatial_shape = artifact_refs.get("spatial_shape")
        if spatial_shape is None:
            spatial_shape = artifact_refs.get("spatial_map_shape")
        if spatial_shape is not None:
            shape = tuple(int(dim) for dim in np.asarray(spatial_shape).tolist())
            if len(shape) >= 2:
                map_shape_text = f"{shape[1]} x {shape[0]}"
        self._cached_summary_map_token = metadata_token
        self._cached_summary_map_fields = (map_shape_text, map_range_text)
        return self._cached_summary_map_fields

    @staticmethod
    def _prediction_clipping_summary(prediction: PredictionRun) -> str:
        """Return reflectance clipping telemetry from prediction metadata."""
        artifact_refs = prediction.artifact_refs
        keys = ("clip_low_count", "clip_high_count", "near_zero_denominator_count")
        if not any(key in artifact_refs for key in keys):
            return "-"
        low = int(artifact_refs.get("clip_low_count", 0) or 0)
        high = int(artifact_refs.get("clip_high_count", 0) or 0)
        near_zero = int(artifact_refs.get("near_zero_denominator_count", 0) or 0)
        return f"low={low} | high={high} | near-zero denom={near_zero}"

    def _invalidate_prediction_workspace_cache(self, *_args) -> None:
        """Drop cached Predict workspace payloads after prediction-state changes."""
        self._cached_workspace_visual_token = None
        self._cached_workspace_visual_payload = None
        self._cached_summary_map_token = None
        self._cached_summary_map_fields = None

    def _on_prediction_completed(self, prediction_run_id: str) -> None:
        logger.info("Prediction completed -> {}", prediction_run_id)
        prediction = self.project_state.predictions.get(prediction_run_id)
        if prediction is None:
            self._prediction_status.setText(f"Prediction: {prediction_run_id}")
            self._readiness_status.setText("Ready: run another prediction or inspect results.")
            return
        self._prediction_status.setText(f"Prediction: {prediction.prediction_run_id}")
        self._readiness_status.setText("Ready: run another prediction or inspect results.")

    def _on_prediction_success(self, payload: tuple[Any, Any, Any, bool, bool]) -> None:
        prediction_result, dataset, calibration, dataset_is_new, calibration_is_new = payload
        if self.project_state.project_path != self._job_project_path:
            msg = (
                "Prediction finished for project '{}', but project '{}' is now open. "
                "The run was saved into the original project file; reopen it to inspect."
            )
            logger.warning(msg, self._job_project_path, self.project_state.project_path)
            self._service_status.set_idle("Service: idle")
            QMessageBox.information(
                self,
                "Prediction saved to previous project",
                msg.format(self._job_project_path, self.project_state.project_path),
            )
            return
        try:
            self.project_state.finalize_prediction_result(
                prediction_result=prediction_result,
                dataset=dataset,
                calibration=calibration,
                dataset_is_new=dataset_is_new,
                calibration_is_new=calibration_is_new,
            )
        except Exception as exc:
            logger.exception("Finalizing the completed prediction run failed")
            self._on_prediction_error("Prediction failed", str(exc))
            return
        self._service_status.set_idle(
            f"Service: prediction {prediction_result.prediction_run.prediction_run_id}"
        )
        self._prediction_status.setText(
            f"Prediction: {prediction_result.prediction_run.prediction_run_id} | {Path(dataset.source_path).name}"
        )
        self._readiness_status.setText("Ready: run another prediction or inspect results.")

    def _on_prediction_progress(self, event: object) -> None:
        if not isinstance(event, PredictionProgressEvent):
            return
        self._last_prediction_phase = event.phase
        phase = event.phase.replace("_", " ").title()
        if event.total > 0 and not event.indeterminate:
            text = f"Service: {phase} {event.completed}/{event.total} {event.unit}".rstrip()
        elif event.total > 0:
            text = f"Service: {phase} {event.completed}/{event.total} {event.unit}".rstrip()
        else:
            text = f"Service: {phase}"
        if event.message:
            text = f"{text} | {event.message}"
        self._service_status.set_busy(text)

    def _on_prediction_error(self, title: str, message: str) -> None:
        logger.error(
            "{} during phase '{}': {}",
            title,
            self._last_prediction_phase or "unknown",
            message,
        )
        if "cancelled" in message.lower():
            self._service_status.set_idle("Service: prediction cancelled")
            self._update_status()
            return
        if self._last_prediction_phase:
            phase = self._last_prediction_phase.replace("_", " ").title()
            self._service_status.set_idle(f"Service: failed during {phase}")
        else:
            self._service_status.set_idle("Service: idle")
        self._update_status()
        QMessageBox.warning(self, title, message)

    def service_status_snapshot(self) -> tuple[str, bool]:
        """Return the current service label text and busy/progress visibility."""
        return self._service_status.label.text(), self._service_status.progress.isVisible()

    def _set_busy(self, busy: bool, text: str | None = None) -> None:
        self._btn_predict.setEnabled(not busy)
        self._btn_cancel.setEnabled(busy)
        if busy:
            self._service_status.set_busy(text or "Working...")
        elif text is not None:
            self._service_status.set_idle(text)
        else:
            self._service_status.set_idle()

    def _cancel_prediction_clicked(self) -> None:
        if self._job_cancel_event is None:
            return
        self._job_cancel_event.set()
        self._btn_cancel.setEnabled(False)
        self._service_status.set_busy("Service: cancelling after current chunk...")

    def _clear_job(self) -> None:
        self._job_thread = None
        self._job_worker = None
        self._job_relay = None
        self._job_progress_relay = None
        self._job_cancel_event = None
        self._last_prediction_phase = None
        self._btn_predict.setEnabled(True)
        self._btn_cancel.setEnabled(False)

    def _selected_workspace_prediction_run(self) -> PredictionRun | None:
        if self._selected_prediction_run_id is None:
            return None
        prediction = self.project_state.predictions.get(self._selected_prediction_run_id)
        if prediction is None:
            self._selected_prediction_run_id = None
            return None
        return prediction

    def _resolved_workspace_prediction_run(self) -> PredictionRun | None:
        selected_prediction = self._selected_workspace_prediction_run()
        if selected_prediction is not None:
            return selected_prediction
        if not self.project_state.predictions:
            return None
        latest_prediction_run_id = next(reversed(self.project_state.predictions))
        return self.project_state.predictions.get(latest_prediction_run_id)


__all__ = ["PredictionMapWidget", "PredictionRunDialog", "PredictionView"]
