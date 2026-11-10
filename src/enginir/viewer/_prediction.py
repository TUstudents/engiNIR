"""Prediction view widgets for project-backed model inference."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colorbar import Colorbar
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .._logging import get_logger
from ..io import load_spectra
from ..prediction import (
    PredictionProgressEvent,
    PredictionResult,
    RoiPrediction,
    RoiPredictionSummary,
    predict_rois,
    predict_with_model_run,
    preflight_prediction,
    roi_prediction_statistics,
)
from ..preprocessing import PreprocessingRecipe
from ..project import CalibrationSpec
from ._predict_dialogs import ModelChoice, PredictMapDialog, PredictRoisDialog
from ._jobs import BackgroundWorker, BusyStatusWidget, start_background_job
from ._state import ProjectState
from ._theme import GUIDE_COLOR, panel_colors, secondary_text_style
from ._plot_panel import PlotColumn, PlotSpec, PlotTable, draw_placeholder, print_copy
from ._plot_registry import register_plot
from ._ui_helpers import LayoutState, maybe_tight_layout, style_axes

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


#: Colours of predictions below and above the map's colour range.
_BELOW_RANGE, _ABOVE_RANGE = "#4cc3d9", "#ff5f7e"


def _ordinal(number: int) -> str:
    """1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st ..."""
    if 10 <= number % 100 <= 20:
        return f"{number}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def describe_calibration(calibration: CalibrationSpec) -> str:
    """A calibration in words, e.g. "detector levels 128 / 3500" or "dark/white references"."""
    if calibration.mode == "raw_dn":
        return "raw counts"
    if calibration.mode == "precalibrated_import":
        return f"calibrated {calibration.output_signal_unit.lower()} values"
    if calibration.dark_ref_data is not None or calibration.dark_ref_source is not None:
        return "dark/white references"
    if calibration.black_level is not None and calibration.white_level is not None:
        return f"detector levels {calibration.black_level:g} / {calibration.white_level:g}"
    return "references"


class _PredictionProgressRelay(QObject):
    progressed = pyqtSignal(object)


#: Pixels of the prediction-map preview (subsampled with one step on both axes).
_MAP_PREVIEW_PIXELS = 512 * 512


#: Map layers of the Predict workspace; the diagnostics come with the saved model's
#: projection data (prediction-runtime contract, Applicability Diagnostics).
MAP_LAYERS = ("Prediction", "DModX", "T²")
_LAYER_LIMITS = {"DModX": ("dcrit_099", "n_above_dcrit", "share_above_dcrit", "DCrit"),
                 "T²": ("t2_crit_099", "n_above_t2crit", "share_above_t2crit", "T²Crit")}
_NO_DIAGNOSTICS = ("No {} map: this model has no projection data (fitted before engiNIR 1.3). "
                   "Refit it to get the DModX and T² maps.")


def _map_previews(result: PredictionResult) -> dict[str, Any]:
    """Subsampled map layers of an in-memory prediction, with the step and full shape."""
    from ..io.project_store import _preview_step

    spatial_map = result.spatial_map
    if spatial_map is None or spatial_map.ndim < 2:
        return {}
    full = tuple(int(n) for n in spatial_map.shape[:2])
    step = _preview_step(full, _MAP_PREVIEW_PIXELS)
    layers = {"DModX": result.diagnostic_map(result.dmodx), "T²": result.diagnostic_map(result.t2)}
    return {"spatial_map": np.ascontiguousarray(spatial_map[::step, ::step, ...]),
            "layers": {name: np.ascontiguousarray(values[::step, ::step])
                       for name, values in layers.items() if values is not None},
            "map_step": step, "full_map_shape": full}


class PredictionMapWidget(QWidget):
    """Read-only spatial prediction map surface for the Predict workspace.

    A :class:`~enginir.viewer._plot_panel.PlotSource`: the viewer shows it in a
    plot frame and puts :meth:`header_widgets` (layer and output selectors,
    pan, zoom, reset view) into the frame header. The layer is the prediction
    or one of the applicability diagnostics (DModX, T²), whose pixels above
    the 0.99 limit are overlaid. Large maps are shown as a subsampled
    preview; the axes, the hover readout and the CSV use the true frame and
    row of each shown pixel (``map_step`` in the payload).
    """

    #: The header title changed (it carries the target unit).
    title_changed = pyqtSignal(str)
    #: The shown map changed: a payload for the prediction histogram, or ``None``.
    displayed_map_changed = pyqtSignal(object)
    plot_title = "Prediction Map"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fig = Figure(figsize=(8, 5), facecolor=panel_colors()["bg"])
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._colorbar: Colorbar | None = None
        self._last_payload: dict[str, Any] | None = None
        self._last_empty_reason: str | None = None
        self._last_layout_size: LayoutState | None = None
        self._current_display_map: np.ndarray | None = None
        self._step = 1
        self._rendered_output_index = 0
        self._rendered_layer = "Prediction"
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(secondary_text_style())
        self._cursor_label = QLabel("")
        self._cursor_label.setStyleSheet(secondary_text_style())
        self._output_combo = QComboBox()
        self._output_combo.setToolTip("Prediction output shown in the map")
        self._output_combo.currentIndexChanged.connect(self._on_output_changed)
        self._output_combo.setVisible(False)
        self._layer_combo = QComboBox()
        self._layer_combo.addItems(list(MAP_LAYERS))
        self._layer_combo.setToolTip("Map layer: the prediction, or the distance to the model "
                                     "(DModX) or within it (T²) with the pixels above the "
                                     "0.99 limit marked")
        self._layer_combo.currentIndexChanged.connect(self._on_output_changed)
        self._overlay: Any = None
        #: Colour limits the user set (⋯ menu); ``None``: the training range of the target.
        self.user_limits: tuple[float, float] | None = None
        self._shown_limits: tuple[float, float] | None = None
        # Drives pan and zoom from the plot header; the toolbar itself is not shown.
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._toolbar.hide()
        self._pan_button = self._mode_button("Pan", "Pan the map (drag)", self._toolbar.pan)
        self._zoom_button = self._mode_button("Zoom", "Zoom to a rectangle (drag)",
                                              self._toolbar.zoom)
        self._reset_button = QToolButton()
        self._reset_button.setText("Reset View")
        self._reset_button.setAutoRaise(True)
        self._reset_button.setToolTip("Show the whole map")
        self._reset_button.clicked.connect(self.reset_view)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("resize_event", self._on_canvas_resize)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._canvas, 1)
        info = QHBoxLayout()
        info.setContentsMargins(6, 0, 6, 2)
        info.addWidget(self._stats_label, 1)
        info.addWidget(self._cursor_label)
        layout.addLayout(info)

        self.clear()

    def header_widgets(self) -> list[QWidget]:
        """Controls for the plot header: layer and output selectors, pan, zoom, reset view."""
        return [self._layer_combo, self._output_combo, self._pan_button, self._zoom_button,
                self._reset_button]

    @property
    def layer(self) -> str:
        """The shown layer, one of :data:`MAP_LAYERS`."""
        return self._layer_combo.currentText()

    def set_layer(self, layer: str) -> None:
        """Show ``layer`` (one of :data:`MAP_LAYERS`)."""
        self._layer_combo.setCurrentIndex(MAP_LAYERS.index(layer))

    def _sync_layer_items(self, payload: dict[str, Any]) -> None:
        """Disable the diagnostics layers a prediction does not have."""
        layers = payload.get("layers") or {}
        model = self._layer_combo.model()
        if not isinstance(model, QStandardItemModel):
            return
        for index, name in enumerate(MAP_LAYERS):
            item = model.item(index)
            if item is None or index == 0:
                continue
            available = name in layers
            item.setEnabled(available)
            item.setToolTip("" if available else _NO_DIAGNOSTICS.format(name))

    def _mode_button(self, text: str, tip: str, toggle: Any) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setAutoRaise(True)
        button.setCheckable(True)
        button.clicked.connect(lambda _checked=False: (toggle(), self._sync_mode_buttons()))
        return button

    def _sync_mode_buttons(self) -> None:
        mode = getattr(getattr(self._toolbar, "mode", None), "name", "")  # NONE, PAN or ZOOM
        self._pan_button.setChecked(mode == "PAN")
        self._zoom_button.setChecked(mode == "ZOOM")

    def reset_view(self) -> None:
        """Show the whole map again after pan or zoom."""
        self._toolbar.home()

    @property
    def figure(self) -> Figure:
        return self._fig

    def update_data(self, payload: dict[str, Any] | None, *, empty_reason: str = "") -> None:
        """Render the selected layer of one prediction (``None`` clears the map)."""
        if payload is None:
            self.clear(empty_reason=empty_reason)
            return
        layer = self.layer
        output_index = self._selected_output_index(payload.get("spatial_map"))
        if (
            payload is self._last_payload
            and empty_reason == self._last_empty_reason
            and output_index == self._rendered_output_index
            and layer == self._rendered_layer
        ):
            return

        spatial_map = payload.get("spatial_map")
        if spatial_map is None:
            self.clear(empty_reason=empty_reason or "No spatial prediction map for this run.")
            return

        self._sync_layer_items(payload)
        if layer == "Prediction":
            self._configure_output_selector(spatial_map, payload=payload)
            output_index = self._selected_output_index(spatial_map)
            arr = self._display_spatial_map(spatial_map, output_index=output_index)
        else:
            self._output_combo.setVisible(False)
            values = (payload.get("layers") or {}).get(layer)
            arr = None if values is None else self._display_spatial_map(values)
            if values is None:
                self._show_missing_layer(payload, _NO_DIAGNOSTICS.format(layer))
                return
        if arr is None:
            self.clear(empty_reason=empty_reason or "No spatial prediction map for this run.")
            return

        colors = panel_colors()
        step = max(1, int(payload.get("map_step", 1)))
        self._fig.set_facecolor(colors["bg"])
        self._remove_colorbar()
        self._ax.clear()
        style_axes(self._ax)
        # Each shown pixel covers ``step`` true frames and rows, starting at its index * step.
        extent = (-0.5, arr.shape[1] * step - 0.5, arr.shape[0] * step - 0.5, -0.5)
        limits = None
        cmap: Any = "viridis" if layer == "Prediction" else "magma"
        if layer == "Prediction":
            # The training range (or the user's limits); values outside get their own colours.
            limits = self.user_limits or payload.get("colour_range")
            if limits is not None and limits[1] > limits[0]:
                cmap = colormaps["viridis"].with_extremes(under=_BELOW_RANGE, over=_ABOVE_RANGE)
            else:
                limits = None
        self._shown_limits = limits
        image = self._ax.imshow(
            arr, cmap=cmap, origin="upper", aspect="auto", interpolation="nearest",
            extent=extent, vmin=None if limits is None else float(limits[0]),
            vmax=None if limits is None else float(limits[1]))
        removed_limit = payload.get("removed_limit")
        dmodx = (payload.get("layers") or {}).get("DModX")
        if layer == "Prediction" and removed_limit is not None and dmodx is not None:
            removed = self._display_spatial_map(dmodx)
            if removed is not None and removed.shape == arr.shape:
                self._ax.imshow(np.ma.masked_where(~(removed > removed_limit),
                                                   np.ones_like(arr)),
                                cmap=ListedColormap([GUIDE_COLOR]), alpha=0.85, origin="upper",
                                aspect="auto", interpolation="nearest", extent=extent)
        for (f0, f1, r0, r1), label in payload.get("roi_overlays") or ():
            self._ax.add_patch(Rectangle((f0 - 0.5, r0 - 0.5), f1 - f0 + 1, r1 - r0 + 1,
                                         fill=False, edgecolor=colors["fg"], linewidth=1.0,
                                         linestyle="--"))
            self._ax.text(f0, r0 - 1, label, fontsize=7, color=colors["fg"], va="bottom",
                          clip_on=True)
        self._current_display_map = arr
        self._step = step
        self._rendered_output_index = output_index
        self._rendered_layer = layer
        units = str(payload.get("target_unit", "")) if layer == "Prediction" else ""
        unit_suffix = f" ({units})" if units else ""
        limit = self._layer_limit(payload, layer)
        if limit is not None:
            above = np.ma.masked_where(~(arr > limit), np.ones_like(arr))
            self._overlay = self._ax.imshow(
                above, cmap=ListedColormap([panel_colors()["accent"]]), alpha=0.9,
                origin="upper", aspect="auto", interpolation="nearest",
                extent=(-0.5, arr.shape[1] * step - 0.5, arr.shape[0] * step - 0.5, -0.5))
        self._set_title(f"{layer} Map{unit_suffix}")
        self._ax.set_xlabel("Frame", color=colors["fg"])
        self._ax.set_ylabel("Row", color=colors["fg"])
        self._colorbar = self._fig.colorbar(image, ax=self._ax, fraction=0.046, pad=0.04,
                                            extend="both" if limits is not None else "neither")
        self._style_colorbar(unit_suffix=unit_suffix)

        finite = arr[np.isfinite(arr)]
        preview = f" | preview: every {_ordinal(step)} frame and row" if step > 1 else ""
        if finite.size == 0:
            self._stats_label.setText(f"Map stats: no finite values.{preview}")
        else:
            self._stats_label.setText(
                ("Map stats (preview): " if step > 1 else "Map stats: ")
                + f"{arr.shape[1]} frames x {arr.shape[0]} rows shown | "
                f"min={float(np.min(finite)):.4g} | "
                f"mean={float(np.mean(finite)):.4g} | "
                f"max={float(np.max(finite)):.4g}{self._limit_text(payload, layer)}{preview}"
            )
        self._cursor_label.setText("")
        self._last_layout_size = maybe_tight_layout(
            self._fig,
            self._canvas,
            pad=1.0,
            last_size=self._last_layout_size,
        )
        self._canvas.draw_idle()
        self._toolbar.update()
        self._toolbar.push_current()  # the whole new map is the home view
        self._last_payload = payload
        self._last_empty_reason = empty_reason
        self.displayed_map_changed.emit({
            "values": arr, "target_unit": units, "output": self._output_name(),
            "quantity": layer, "limit": limit,
            "map_step": step, "provenance": self.plot_table_metadata()})

    @staticmethod
    def _layer_limit(payload: dict[str, Any], layer: str) -> float | None:
        """The 0.99 limit of a diagnostics layer, or ``None``."""
        if layer not in _LAYER_LIMITS:
            return None
        value = (payload.get("diagnostics") or {}).get(_LAYER_LIMITS[layer][0])
        return None if value is None or not np.isfinite(value) else float(value)

    @staticmethod
    def _limit_text(payload: dict[str, Any], layer: str) -> str:
        """`` | above DCrit 2.1: 40 pixels (1.3 %)`` over the whole scan, or ``""``."""
        summary = payload.get("diagnostics") or {}
        if layer not in _LAYER_LIMITS or summary.get(_LAYER_LIMITS[layer][0]) is None:
            return ""
        key, count, share, name = _LAYER_LIMITS[layer]
        return (f" | above {name} {float(summary[key]):.3g}: {int(summary[count])} pixels "
                f"({100 * float(summary[share]):.2g} %)")

    def _show_missing_layer(self, payload: dict[str, Any], reason: str) -> None:
        """Placeholder for a diagnostics layer the prediction does not have."""
        self._last_payload = None
        self.clear(empty_reason=reason)
        self._layer_combo.setVisible(True)
        self._last_payload = payload
        self._rendered_layer = self.layer

    def clear(self, *, empty_reason: str = "") -> None:
        """Show an empty prediction-map placeholder."""
        if self._last_payload is None and empty_reason == self._last_empty_reason:
            return
        colors = panel_colors()
        self._fig.set_facecolor(colors["bg"])
        self._remove_colorbar()
        self._ax.clear()
        self._current_display_map = None
        self._overlay = None
        self._output_combo.setVisible(False)
        style_axes(self._ax)
        self._set_title(f"{self.layer} Map")
        self._ax.set_xlabel("Frame", color=colors["fg"])
        self._ax.set_ylabel("Row", color=colors["fg"])
        draw_placeholder(self._ax, empty_reason or "No prediction in this session yet.", colors)
        self._stats_label.setText("")
        self._cursor_label.setText("")
        self._last_layout_size = maybe_tight_layout(
            self._fig,
            self._canvas,
            pad=1.0,
            last_size=self._last_layout_size,
        )
        self._canvas.draw_idle()
        self._last_payload = None
        self._last_empty_reason = empty_reason
        self.displayed_map_changed.emit(None)

    # ------------------------------------------------------------------
    # Plot source
    # ------------------------------------------------------------------

    def plot_table(self) -> PlotTable | None:
        """The shown map as a row x frame matrix, labelled with the true rows and frames."""
        arr = self._current_display_map
        if arr is None:
            return None
        return PlotTable(matrix=arr, row_label="row", column_label="frame",
                         row_index=np.arange(arr.shape[0]) * self._step,
                         column_index=np.arange(arr.shape[1]) * self._step,
                         metadata=self.plot_table_metadata())

    def set_user_limits(self, limits: tuple[float, float] | None) -> None:
        """Colour limits of the prediction layer; ``None`` returns to the training range."""
        self.user_limits = limits
        payload, self._last_payload = self._last_payload, None  # force a redraw
        if payload is not None:
            self.update_data(payload, empty_reason=self._last_empty_reason or "")

    def plot_table_metadata(self) -> dict[str, Any]:
        payload = self._last_payload or {}
        metadata: dict[str, Any] = {"layer": self.layer, "output": self._output_name(),
                                    "target_unit": str(payload.get("target_unit", ""))}
        limit = self._layer_limit(payload, self.layer)
        if limit is not None:
            metadata["limit_099"] = f"{limit:.6g}"
        if self.layer == "Prediction" and self._shown_limits is not None:
            low, high = self._shown_limits
            metadata["colour_scale"] = (f"{low:.6g} .. {high:.6g} "
                                        f"({'user limits' if self.user_limits else 'training range'})")
        if self._step > 1:
            metadata["preview_step"] = self._step
            full_shape = payload.get("full_map_shape")
            if full_shape is not None:
                metadata["full_map_shape"] = (f"{int(full_shape[0])} frames x "
                                              f"{int(full_shape[1])} rows")
        return metadata

    def print_figure(self) -> Figure:
        """The current view in the print colors."""
        return print_copy(self._fig)

    def _output_name(self) -> str:
        if self._output_combo.count() > 1:
            return self._output_combo.currentText()
        return "prediction"

    def _set_title(self, title: str) -> None:
        if title != self.plot_title:
            self.plot_title = title
            self.title_changed.emit(title)

    def _on_canvas_resize(self, _event: Any) -> None:
        self._last_layout_size = maybe_tight_layout(
            self._fig, self._canvas, pad=1.0, last_size=self._last_layout_size)

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

    def _pixel_at(self, x: float, y: float) -> tuple[int, int] | None:
        """(row, column) of the shown pixel under data coordinates ``x``, ``y``."""
        arr = self._current_display_map
        if arr is None:
            return None
        column = int(np.floor((x + 0.5) / self._step))
        row = int(np.floor((y + 0.5) / self._step))
        if row < 0 or column < 0 or row >= arr.shape[0] or column >= arr.shape[1]:
            return None
        return row, column

    def _on_motion(self, event: Any) -> None:
        arr = self._current_display_map
        if arr is None or event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            self._cursor_label.setText("")
            return
        pixel = self._pixel_at(float(event.xdata), float(event.ydata))
        if pixel is None:
            self._cursor_label.setText("")
            return
        row, column = pixel
        value = float(arr[row, column])
        self._cursor_label.setText(
            f"Frame {column * self._step} | Row {row * self._step} | Value {value:.4g}")

    def _remove_colorbar(self) -> None:
        if self._colorbar is None:
            return
        self._colorbar.remove()
        self._colorbar = None

    def _style_colorbar(self, *, unit_suffix: str) -> None:
        if self._colorbar is None:
            return
        colors = panel_colors()
        label = f"{self.layer}{unit_suffix}"
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


def _histogram_render(fig: Figure, payload: Any, colors: Any) -> None:
    ax = fig.add_subplot(111)
    style_axes(ax, colors)
    counts, edges = _prediction_histogram(payload)
    if counts is None or edges is None:
        draw_placeholder(ax, "No finite prediction values.", colors)
        return
    ax.stairs(counts, edges, fill=True, color=colors["accent"], alpha=0.85)
    unit = str(payload.get("target_unit", ""))
    quantity = str(payload.get("quantity", "Prediction"))
    if payload.get("limit") is not None:
        ax.axvline(float(payload["limit"]), color=colors["fg"], linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"{quantity} ({unit})" if unit else quantity, color=colors["fg"])
    ax.set_ylabel("Pixels", color=colors["fg"])


def _histogram_table(payload: Any) -> PlotTable:
    counts, edges = _prediction_histogram(payload)
    if counts is None or edges is None:
        return PlotTable()
    unit = str(payload.get("target_unit", ""))
    metadata = dict(payload.get("provenance", {}))
    metadata["pixels"] = "shown map" + (
        f", every {payload['map_step']}th frame and row" if payload.get("map_step", 1) > 1
        else "")
    return PlotTable(columns=(PlotColumn("bin_start", edges[:-1], unit),
                              PlotColumn("bin_end", edges[1:], unit),
                              PlotColumn("pixel_count", counts)), metadata=metadata)


def _prediction_histogram(payload: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    values = np.asarray(payload.get("values", []), dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    return np.histogram(finite, bins=50)


#: Distribution of the shown prediction values; fed by
#: :attr:`PredictionMapWidget.displayed_map_changed`.
PREDICTION_HISTOGRAM = register_plot(PlotSpec(
    "predict.histogram", "Prediction Distribution", _histogram_render, _histogram_table,
    empty_text="No prediction in this session yet."))


class PredictionView(QWidget):
    """Predict ROIs and Predict map: saved models on prediction ROIs and project data files."""

    #: The ROI results changed (a prediction, or another exclusion).
    results_changed = pyqtSignal()

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
        self._cached_workspace_visual_token: Any = None
        self._cached_workspace_visual_payload: dict[str, Any] | None = None
        self._roi_predictions: list[RoiPrediction] = []
        self._roi_summaries: list[RoiPredictionSummary] = []
        self._roi_scope = ""
        self._results_revision = 0
        self._roi_payload_cache: tuple[Any, dict[str, Any]] | None = None
        self._data_file_adder: Callable[[str], str | None] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_group())

        self.project_state.active_model_changed.connect(self._update_status)
        self.project_state.model_runs_changed.connect(self._update_status)
        self.project_state.predictions_changed.connect(self._update_status)
        self.project_state.predictions_changed.connect(self._invalidate_prediction_workspace_cache)
        self.project_state.prediction_completed.connect(self._on_prediction_completed)
        # A changed reference value changes residuals and RMSEP.
        self.project_state.target_assignments_changed.connect(self._recompute_if_results)
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
        self._btn_predict_rois = QToolButton()
        self._btn_predict_rois.setText("Predict ROIs...")
        self._btn_predict_rois.setToolTip(
            "Predict the prediction ROIs (tablets) with one or more models, from the project "
            "file alone; one result per ROI")
        self._btn_predict_rois.clicked.connect(self.predict_rois_clicked)
        button_row.addWidget(self._btn_predict_rois)
        self._btn_predict = QToolButton()
        self._btn_predict.setText("Predict map...")
        self._btn_predict.setToolTip("Predict every pixel of a project data file with one model, "
                                     "with the results of its prediction ROIs")
        self._btn_predict.clicked.connect(self.predict_btn_clicked)
        button_row.addWidget(self._btn_predict)
        self._btn_cancel = QToolButton()
        self._btn_cancel.setText("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_prediction_clicked)
        button_row.addWidget(self._btn_cancel)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        exclusion_row = QHBoxLayout()
        self._exclusion_check = QCheckBox("Remove pixels above the DModX limit")
        self._exclusion_check.setToolTip(
            "Leave pixels outside the model (glare, exposed substrate, direct reflections) "
            "out of every ROI statistic; recalculated without predicting again")
        self._exclusion_level = QDoubleSpinBox()
        self._exclusion_level.setRange(0.5, 0.999)
        self._exclusion_level.setDecimals(3)
        self._exclusion_level.setSingleStep(0.01)
        self._exclusion_level.setValue(0.99)
        self._exclusion_level.setPrefix("level ")
        self._exclusion_check.toggled.connect(self._recompute_roi_results)
        self._exclusion_level.valueChanged.connect(self._recompute_roi_results)
        exclusion_row.addWidget(self._exclusion_check)
        exclusion_row.addWidget(self._exclusion_level)
        exclusion_row.addStretch(1)
        layout.addLayout(exclusion_row)

        self._service_status = BusyStatusWidget(idle_text="Service: idle", parent=self)
        layout.addWidget(self._service_status)
        return box

    # -- choices ----------------------------------------------------------

    def _model_choices(self) -> list[ModelChoice]:
        """The project's models by name, each with the chain a prediction applies."""
        state = self.project_state
        choices = []
        for model_run_id, run in state.model_runs.items():
            name = run.name or f"{run.model_type} model"
            sample_set = state.sample_sets.get(run.sample_set_id)
            domain = state.signal_domains.get(run.signal_domain_id)
            recipe = "none"
            if sample_set is not None and sample_set.preprocessing_recipe_id in state.recipes:
                recipe = PreprocessingRecipe.from_dict(
                    state.recipes[sample_set.preprocessing_recipe_id]).describe()
            unit = "?" if domain is None else domain.required_signal_unit
            chain = (f"the data file's calibration → {unit} → {recipe} → {name}; "
                     f"trained with {self._training_calibration(run)}")
            choices.append(ModelChoice(model_run_id, name, chain))
        return choices

    def _training_calibration(self, run: Any) -> str:
        """How the model's training ROIs were calibrated, in words."""
        state = self.project_state
        sample_set = state.sample_sets.get(run.sample_set_id)
        texts = {describe_calibration(state.calibrations[state.rois[roi_id].calibration_id])
                 for roi_id in (sample_set.roi_ids if sample_set is not None else ())
                 if roi_id in state.rois
                 and state.rois[roi_id].calibration_id in state.calibrations}
        return " / ".join(sorted(texts)) or "unknown calibration"

    def _default_model_id(self) -> str | None:
        active = self.project_state.active_model_name
        if active in self.project_state.model_runs:
            return active
        return next(reversed(self.project_state.model_runs), None)

    def _require_models(self) -> bool:
        if self._job_thread is not None:
            return False
        if self.project_state.project_path is None:
            QMessageBox.warning(self, "No project open", "Open a project before predicting.")
            return False
        if not self.project_state.model_runs:
            QMessageBox.information(self, "No models", "Fit a model before predicting.")
            return False
        return True

    # -- Predict ROIs -------------------------------------------------------

    def predict_rois_clicked(self) -> None:
        """Predict the prediction ROIs with the chosen models (contract, Prediction ROIs)."""
        if not self._require_models():
            return
        state = self.project_state
        counts: dict[str, int] = {}
        for roi in state.rois.values():
            if roi.use == "prediction":
                counts[roi.dataset_id] = counts.get(roi.dataset_id, 0) + 1
        files = sorted(((dataset_id, state.data_file_label(dataset_id), count)
                        for dataset_id, count in counts.items()),
                       key=lambda item: item[1].lower())
        dialog = PredictRoisDialog(self._model_choices(), files,
                                   default_model_id=self._default_model_id(), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        model_ids, dataset_id = dialog.model_ids, dialog.dataset_id
        rois = [roi for roi in state.rois.values() if roi.use == "prediction"
                and (dataset_id is None or roi.dataset_id == dataset_id)]
        self._start_roi_job(model_ids, rois, scope=(
            "all data files" if dataset_id is None else state.data_file_label(dataset_id)))

    def _start_roi_job(self, model_ids: list[str], rois: list[Any], *, scope: str) -> None:
        state = self.project_state
        project_path = str(state.project_path)
        cancel_event = Event()
        self._job_cancel_event = cancel_event
        snapshot = {"datasets": dict(state.datasets), "calibrations": dict(state.calibrations),
                    "signal_domains": dict(state.signal_domains),
                    "recipes": dict(state.recipes), "sample_sets": dict(state.sample_sets)}
        runs = [(state.model_runs[model_id], self._model_name(model_id))
                for model_id in model_ids]

        def job() -> list[RoiPrediction]:
            return [predict_rois(project_path=project_path, model_run=run, model_name=name,
                                 rois=rois, cancelled=cancel_event.is_set, **snapshot)
                    for run, name in runs]

        def done(predictions: list[RoiPrediction]) -> None:
            if cancel_event.is_set():
                self._service_status.set_idle("Service: ROI prediction cancelled")
                return
            self._set_roi_predictions(predictions, scope=scope)
            refused = sum(len(prediction.refused) for prediction in predictions)
            self._service_status.set_idle(
                f"Service: {len(rois)} ROI(s) predicted with {len(predictions)} model(s) "
                f"(not stored){f'; {refused} refused' if refused else ''}")
            for prediction in predictions:
                logger.info("Predicted {} ROI(s) of {} with model '{}' ({}); {} refused",
                            len(prediction.pixels), project_path, prediction.model_name,
                            prediction.model_run_id, len(prediction.refused))

        self._job_project_path = project_path
        self._set_busy(True, f"Predicting {len(rois)} ROI(s)...")
        self._job_thread, self._job_worker, self._job_relay = start_background_job(
            owner=self, fn=job, on_success=done,
            on_error=lambda msg: self._on_prediction_error("ROI prediction failed", msg),
            on_finished=self._clear_job)

    def _set_roi_predictions(self, predictions: list[RoiPrediction], *, scope: str) -> None:
        """Keep the ROI predictions of this session; the exclusion starts at the model's own."""
        self._roi_predictions = predictions
        self._roi_scope = scope
        default = next((p.default_exclusion for p in predictions
                        if p.default_exclusion is not None), None)
        has_projection = any(p.limit_args is not None for p in predictions)
        for widget in (self._exclusion_check, self._exclusion_level):
            widget.blockSignals(True)
        self._exclusion_check.setChecked(default is not None)
        self._exclusion_level.setValue(0.99 if default is None else default)
        self._exclusion_check.setEnabled(has_projection)
        self._exclusion_check.setToolTip(
            self._exclusion_check.toolTip() if has_projection
            else "This model has no projection data (fitted before 1.3): refit it to remove "
                 "pixels by DModX")
        for widget in (self._exclusion_check, self._exclusion_level):
            widget.blockSignals(False)
        self._recompute_roi_results()

    def _recompute_roi_results(self, *_args: Any) -> None:
        """Per-ROI statistics for the current exclusion, without predicting again."""
        level = self._exclusion_level.value() if self._exclusion_check.isChecked() else None
        self._exclusion_level.setEnabled(self._exclusion_check.isChecked())
        self._roi_summaries = [roi_prediction_statistics(
            prediction, target_assignments=self.project_state.target_assignments,
            exclude_dmodx=level) for prediction in self._roi_predictions]
        self._results_revision += 1
        self.results_changed.emit()

    def _recompute_if_results(self, *_args: Any) -> None:
        if self._roi_predictions:
            self._recompute_roi_results()

    def _model_name(self, model_run_id: str) -> str:
        run = self.project_state.model_runs.get(model_run_id)
        return model_run_id if run is None else (run.name or f"{run.model_type} model")

    def roi_results_payload(self) -> dict[str, Any]:
        """The per-ROI plot's payload: one row per ROI and model (contract, Prediction ROIs).

        The same object until the results, the exclusion or names change, so a refresh
        of the workspace does not redraw the plot.
        """
        state = self.project_state
        token = (self._results_revision, tuple((roi_id, roi.name)
                                               for roi_id, roi in state.rois.items()))
        if self._roi_payload_cache is not None and self._roi_payload_cache[0] == token:
            return self._roi_payload_cache[1]
        payload = self._build_roi_results_payload()
        self._roi_payload_cache = (token, payload)
        return payload

    def _build_roi_results_payload(self) -> dict[str, Any]:
        state = self.project_state
        rows: list[dict[str, Any]] = []
        rmsep: dict[str, str] = {}
        for prediction, summary in zip(self._roi_predictions, self._roi_summaries, strict=True):
            for row in summary.rois:
                roi = state.rois.get(row.roi_id)
                rows.append({
                    "roi": row.roi_id if roi is None else roi.name, "roi_id": row.roi_id,
                    "data_file": "" if roi is None else state.data_file_label(roi.dataset_id),
                    "model": prediction.model_name, "model_run_id": prediction.model_run_id,
                    "pixels": row.n_pixels, "removed": row.n_removed, "mean": row.mean,
                    "median": row.median, "sd": row.sd, "p05": row.p05, "p95": row.p95,
                    "mean_dmodx": row.mean_dmodx, "mean_t2": row.mean_t2,
                    "share_above_dcrit": row.share_above_dcrit,
                    "share_above_t2crit": row.share_above_t2crit,
                    "reference": row.reference, "residual": row.residual})
            if summary.rmsep is not None:
                rmsep[prediction.model_name] = (f"RMSEP {summary.rmsep:.4g}, bias "
                                                f"{summary.bias:.4g}, n {summary.n_reference}")
        level = self._exclusion_level.value() if self._exclusion_check.isChecked() else None
        first = self._roi_predictions[0] if self._roi_predictions else None
        unit = ""
        if first is not None and first.target_id in state.targets:
            unit = state.targets[first.target_id].unit
        return {"roi_rows": rows, "target_unit": unit, "rmsep": rmsep,
                "training_share": None if first is None else first.training_share_range,
                "exclusion_text": ("" if not rows else "all pixels" if level is None
                                   else f"pixels above DModX {level:g} removed"),
                "message": (None if self._roi_predictions or not state.model_runs else
                            "Predict ROIs to see one result per tablet.")}

    def roi_summary_lines(self) -> list[str]:
        """Plain lines for the summary panel: scope, refused ROIs, RMSEP per model."""
        if not self._roi_predictions:
            return []
        lines = [f"{sum(len(p.pixels) for p in self._roi_predictions[:1])} ROI(s) of "
                 f"{self._roi_scope}"]
        for prediction, summary in zip(self._roi_predictions, self._roi_summaries, strict=True):
            text = prediction.model_name
            if summary.rmsep is not None:
                text += (f": RMSEP {summary.rmsep:.3g}, bias {summary.bias:.3g} "
                         f"(n {summary.n_reference})")
            if prediction.refused:
                text += f"; {len(prediction.refused)} ROI(s) refused"
            lines.append(text)
        for prediction in self._roi_predictions[:1]:
            lines += [f"Wavelengths of {self.project_state.data_file_label(d)}: {w}"
                      for d, w in prediction.wavelength_warnings.items()]
        return lines

    # -- Predict map --------------------------------------------------------

    def set_data_file_adder(self, adder: Callable[[str], str | None]) -> None:
        """How Predict map adds a data file (the window's Add data file dialog)."""
        self._data_file_adder = adder

    def _map_check(self, model_run_id: str, dataset_id: str) -> str | None:
        """Why ``dataset_id`` cannot be predicted with the model, or ``None``."""
        state = self.project_state
        dataset = state.datasets.get(dataset_id)
        calibration = state.active_calibration(dataset_id)
        if dataset is None or calibration is None:
            return "the data file has no calibration in the project"
        if not Path(dataset.source_path).exists():
            return f"{dataset.source_path} was not found (Locate file... in Explore)"
        try:
            context = state.build_prediction_job_context(model_run_id=model_run_id)
            preflight_prediction(project_path=context.project_path, model_run=context.model_run,
                                 signal_domain=context.signal_domain, recipe=context.recipe,
                                 dataset=dataset, calibration=calibration,
                                 model_registry=context.model_registry, prefer_saved_model=True)
        except Exception as exc:  # every refusal is shown, not raised
            return str(exc)
        return None

    def predict_btn_clicked(self) -> None:
        """Predict every pixel of a project data file, and its prediction ROIs."""
        if not self._require_models():
            return
        state = self.project_state
        files = sorted(((dataset_id, state.data_file_label(dataset_id))
                        for dataset_id in state.datasets), key=lambda item: item[1].lower())
        adder = self._data_file_adder
        dialog = PredictMapDialog(
            self._model_choices(), files, check=self._map_check,
            add_data_file=None if adder is None else adder, file_label=state.data_file_label,
            default_model_id=self._default_model_id(), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.dataset_id is None:
            return
        self._start_map_job(dialog.model_id, dialog.dataset_id)

    def _start_map_job(self, model_run_id: str, dataset_id: str) -> None:
        state = self.project_state
        try:
            context = state.build_prediction_job_context(model_run_id=model_run_id)
        except Exception as exc:
            logger.error("Prediction failed while building job context: {}", exc)
            QMessageBox.warning(self, "Prediction failed", str(exc))
            return
        dataset = state.datasets[dataset_id]
        calibration = state.active_calibration(dataset_id)
        if calibration is None:
            QMessageBox.warning(self, "Prediction failed", "The data file has no calibration.")
            return
        rois = [roi for roi in state.rois.values()
                if roi.use == "prediction" and roi.dataset_id == dataset_id]
        name = self._model_name(model_run_id)
        snapshot = {"datasets": dict(state.datasets), "calibrations": dict(state.calibrations),
                    "signal_domains": dict(state.signal_domains),
                    "recipes": dict(state.recipes), "sample_sets": dict(state.sample_sets)}
        cancel_event = Event()
        self._job_cancel_event = cancel_event
        progress_relay = _PredictionProgressRelay(self)
        progress_relay.progressed.connect(self._on_prediction_progress)
        self._job_progress_relay = progress_relay

        def emit_progress(event: PredictionProgressEvent) -> None:
            progress_relay.progressed.emit(event)

        def job() -> tuple[PredictionResult, RoiPrediction | None]:
            # Nothing is written: the prediction stays in memory (Result Handling).
            spectral_data = load_spectra(dataset.source_path)
            result = predict_with_model_run(
                project_path=context.project_path, model_run=context.model_run,
                signal_domain=context.signal_domain, recipe=context.recipe, dataset=dataset,
                calibration=calibration, model_registry=context.model_registry,
                data_path=dataset.source_path, spectral_data=spectral_data,
                preprocessing_recipe_id=context.preprocessing_recipe_id,
                progress_callback=emit_progress, cancel_callback=cancel_event.is_set,
                prefer_saved_model=True)
            # The ROI results come from the ROIs' stored pixels, as with Predict ROIs.
            roi_prediction = None if not rois else predict_rois(
                project_path=context.project_path, model_run=context.model_run,
                model_name=name, rois=rois, **snapshot)
            return result, roi_prediction

        def done(outcome: tuple[PredictionResult, RoiPrediction | None]) -> None:
            result, roi_prediction = outcome
            self._on_prediction_success(result)
            if roi_prediction is not None:
                self._set_roi_predictions([roi_prediction],
                                          scope=state.data_file_label(dataset_id))

        self._job_project_path = context.project_path
        self._set_busy(True, "Running prediction...")
        self._job_thread, self._job_worker, self._job_relay = start_background_job(
            owner=self, fn=job, on_success=done,
            on_error=lambda msg: self._on_prediction_error("Prediction failed", msg),
            on_finished=self._clear_job)

    def _update_status(self, _value: str | None = None) -> None:
        active_model = self.project_state.active_model_name
        if active_model is None and self.project_state.model_runs:
            active_model = next(reversed(self.project_state.model_runs))
        if active_model is None:
            self._prediction_status.setText("Prediction: none")
            self._readiness_status.setText("Ready: fit or load a model run.")
            return
        run = self.project_state.model_runs.get(str(active_model))
        name = active_model if run is None else (run.name or f"{run.model_type} model")
        self._prediction_status.setText(f"Ready: model {name}")
        self._readiness_status.setText("Ready: choose a dataset and run prediction.")

    def select_prediction_run(self, prediction_run_id: str | None) -> None:
        selected_prediction_run_id = None if prediction_run_id is None else str(prediction_run_id)
        if selected_prediction_run_id not in self.project_state.predictions:
            selected_prediction_run_id = None
        self._selected_prediction_run_id = selected_prediction_run_id

    def workspace_summary(self) -> dict[str, str]:
        result = self._resolved_workspace_prediction()
        run = None if result is None else result.prediction_run
        model_run = None if run is None else self.project_state.model_runs.get(run.model_run_id)
        signal_domain = (None if run is None
                         else self.project_state.signal_domains.get(run.signal_domain_id))

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
        recipe_text = "-"
        if model_run is not None:
            model_text = self._model_name(model_run.model_run_id)
            sample_set = self.project_state.sample_sets.get(model_run.sample_set_id)
            if sample_set is not None and sample_set.preprocessing_recipe_id in \
                    self.project_state.recipes:
                recipe_text = PreprocessingRecipe.from_dict(
                    self.project_state.recipes[sample_set.preprocessing_recipe_id]).describe()

        dataset_text = "none"
        calibration_text = "none"
        if result is not None:
            dataset_text = (self.project_state.data_file_label(result.dataset.dataset_id)
                            if result.dataset.dataset_id in self.project_state.datasets
                            else Path(result.dataset.source_path).name or "none")
            calibration_text = describe_calibration(result.calibration)

        format_text = "-"
        signal_text = "-"
        bands_text = "-"
        if signal_domain is not None:
            format_text = f"{signal_domain.input_format} | {signal_domain.source_modality}"
            signal_text = f"{signal_domain.required_signal_unit}, {recipe_text}"
            bands_text = str(signal_domain.n_bands)

        map_shape_text = "-"
        map_range_text = "-"
        clipping_text = "-"
        diagnostics_text = "-"
        wavelengths_text = "-"
        if result is not None:
            warning = result.provenance.get("wavelength_warning")
            wavelengths_text = f"Warning: {warning}" if warning else "match the model"
            map_shape_text, map_range_text = self._prediction_map_summary_fields(result)
            clipping_text = self._prediction_clipping_summary(result)
            diagnostics_text = self._diagnostics_summary(result)

        status_text = self._prediction_status.text()
        readiness_text = self._readiness_status.text()
        if result is not None:
            status_text = f"Prediction: {dataset_text} (this session, not stored)"
            readiness_text = "Ready: run another prediction or inspect results."

        return {
            "title": "Prediction Results",
            "status": status_text,
            "readiness": readiness_text,
            "service": self._service_status.label.text(),
            "prediction": "none" if result is None else f"map of {dataset_text}",
            "rois": "\n".join(self.roi_summary_lines()) or "none",
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
            "diagnostics": diagnostics_text,
            "wavelengths": wavelengths_text,
        }

    def workspace_provenance(self) -> dict[str, str]:
        """Provenance lines for exported prediction plots (Plot Panel Policy)."""
        provenance = {"project_id": str(self.project_state.project_meta.get("project_id", ""))}
        result = self._resolved_workspace_prediction()
        if result is None:
            return provenance
        run = result.prediction_run
        provenance.update(model_run_id=run.model_run_id,
                          signal_domain_id=run.signal_domain_id,
                          signal_unit=result.calibration.output_signal_unit,
                          calibration_mode=result.calibration.mode,
                          data_file=result.dataset.source_path,
                          predicted_at=str(result.provenance.get("created_at", "")))
        model_run = self.project_state.model_runs.get(run.model_run_id)
        if model_run is not None and model_run.name:
            provenance["model_name"] = model_run.name
        return provenance

    def workspace_visual_payload(self) -> dict[str, Any] | None:
        """Return one stable visual payload for the selected prediction of this session.

        The returned object is cached so repeated shell refreshes do not rebuild
        identical prediction-map payloads or trigger redundant redraws.
        """
        result = self._resolved_workspace_prediction()
        if result is None:
            self._cached_workspace_visual_token = None
            self._cached_workspace_visual_payload = None
            return None
        run = result.prediction_run
        token = (run.prediction_run_id, self._results_revision)
        if self._cached_workspace_visual_token == token:
            return self._cached_workspace_visual_payload
        model_run = self.project_state.model_runs.get(run.model_run_id)
        target_unit = ""
        if model_run is not None:
            sample_set = self.project_state.sample_sets.get(model_run.sample_set_id)
            if sample_set is not None:
                target = self.project_state.targets.get(sample_set.target_id)
                if target is not None:
                    target_unit = target.unit
        payload: dict[str, Any] = {"target_unit": target_unit,
                                   "diagnostics": result.diagnostics or {}}
        for key in ("output_names", "output_axis", "output_index", "n_outputs"):
            if key in run.artifact_refs:
                payload[key] = run.artifact_refs[key]
        payload.update(_map_previews(result))
        payload.update(self._map_extras(run.model_run_id, result.dataset.dataset_id))
        self._cached_workspace_visual_token = token
        self._cached_workspace_visual_payload = payload
        return payload

    def _map_extras(self, model_run_id: str, dataset_id: str) -> dict[str, Any]:
        """Colour range (the training values), ROI outlines with their means, removal limit."""
        state = self.project_state
        extras: dict[str, Any] = {}
        run = state.model_runs.get(model_run_id)
        sample_set = None if run is None else state.sample_sets.get(run.sample_set_id)
        if sample_set is not None:
            values = [state.target_assignments[(roi_id, sample_set.target_id)].value
                      for roi_id in sample_set.roi_ids
                      if (roi_id, sample_set.target_id) in state.target_assignments]
            if values:
                extras["colour_range"] = (float(min(values)), float(max(values)))
        means: dict[str, float] = {}
        for prediction, summary in zip(self._roi_predictions, self._roi_summaries, strict=True):
            if prediction.model_run_id == model_run_id:
                means = {row.roi_id: row.mean for row in summary.rois}
                if self._exclusion_check.isChecked():
                    extras["removed_limit"] = prediction.dmodx_limit_at(
                        self._exclusion_level.value())
        from ._project_rois import _short_names

        rois = [roi for roi in state.rois.values()
                if roi.dataset_id == dataset_id and roi.use == "prediction"]
        short = _short_names({roi.roi_id: roi.name for roi in rois}, {})
        extras["roi_overlays"] = [
            (roi.spatial_bounds, short[roi.roi_id] if roi.roi_id not in means
             else f"{short[roi.roi_id]}  {means[roi.roi_id]:.3g}")
            for roi in rois]
        return extras

    @staticmethod
    def _prediction_map_summary_fields(result: PredictionResult) -> tuple[str, str]:
        """Map shape and value range of one prediction."""
        refs = result.prediction_run.artifact_refs
        shape = refs.get("spatial_shape", refs.get("spatial_map_shape"))
        shape_text = "-"
        if shape is not None:
            dims = tuple(int(dim) for dim in np.asarray(shape).tolist())
            if len(dims) >= 2:
                shape_text = f"{dims[1]} x {dims[0]}"
        low, high = refs.get("prediction_min"), refs.get("prediction_max")
        range_text = "-" if low is None or high is None else f"{float(low):.4g} .. {float(high):.4g}"
        return shape_text, range_text

    @staticmethod
    def _prediction_clipping_summary(result: PredictionResult) -> str:
        """Return reflectance clipping telemetry of one prediction."""
        artifact_refs = result.prediction_run.artifact_refs
        keys = ("clip_low_count", "clip_high_count", "near_zero_denominator_count")
        if not any(key in artifact_refs for key in keys):
            return "-"
        low = int(artifact_refs.get("clip_low_count", 0) or 0)
        high = int(artifact_refs.get("clip_high_count", 0) or 0)
        near_zero = int(artifact_refs.get("near_zero_denominator_count", 0) or 0)
        return f"low={low} | high={high} | near-zero denom={near_zero}"

    @staticmethod
    def _diagnostics_summary(result: PredictionResult) -> str:
        """Shares above the 0.99 limits and the mean prediction within and outside them."""
        summary = result.diagnostics
        if not summary:
            return "none (refit the model to get DModX and T² maps)"
        parts = []
        for key, share, name in (("dcrit_099", "share_above_dcrit", "DCrit"),
                                 ("t2_crit_099", "share_above_t2crit", "T²Crit")):
            if summary.get(key) is not None:
                parts.append(f"above {name}: {100 * float(summary[share]):.2g} %")
        inside = summary.get("mean_prediction_within_limits")
        outside = summary.get("mean_prediction_outside_limits")
        if inside is not None and np.isfinite(inside):
            parts.append(f"mean within limits {float(inside):.4g}")
        if outside is not None and np.isfinite(outside):
            parts.append(f"outside {float(outside):.4g}")
        return " | ".join(parts) or "-"

    def _invalidate_prediction_workspace_cache(self, *_args) -> None:
        """Drop cached Predict workspace payloads after prediction-state changes."""
        self._cached_workspace_visual_token = None
        self._cached_workspace_visual_payload = None

    def _on_prediction_completed(self, prediction_run_id: str) -> None:
        logger.info("Prediction completed -> {}", prediction_run_id)
        self._selected_prediction_run_id = prediction_run_id
        result = self.project_state.predictions.get(prediction_run_id)
        name = "" if result is None else Path(result.dataset.source_path).name
        self._prediction_status.setText(f"Prediction: {name or prediction_run_id}")
        self._readiness_status.setText("Ready: run another prediction or inspect results.")

    def _on_prediction_success(self, result: PredictionResult) -> None:
        if self.project_state.project_path != self._job_project_path:
            msg = ("Prediction finished for project '{}', but project '{}' is now open. "
                   "The result is not shown; predict again in this project.")
            logger.warning(msg, self._job_project_path, self.project_state.project_path)
            self._service_status.set_idle("Service: idle")
            QMessageBox.information(
                self, "Prediction discarded",
                msg.format(self._job_project_path, self.project_state.project_path))
            return
        self.project_state.add_prediction(result)
        self._service_status.set_idle(
            f"Service: prediction of {Path(result.dataset.source_path).name} done (not stored)")

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
        self._btn_predict_rois.setEnabled(not busy)
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
        self._btn_predict_rois.setEnabled(True)
        self._btn_cancel.setEnabled(False)

    def _resolved_workspace_prediction(self) -> PredictionResult | None:
        """The selected prediction of this session, or the latest one."""
        predictions = self.project_state.predictions
        selected = self._selected_prediction_run_id
        if selected is not None and selected in predictions:
            return predictions[selected]
        self._selected_prediction_run_id = None
        return predictions[next(reversed(predictions))] if predictions else None


__all__ = ["PredictionMapWidget", "PredictionView", "describe_calibration"]
