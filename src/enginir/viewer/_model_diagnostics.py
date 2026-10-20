"""Model workspace diagnostics widgets."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..visualization.plots import plot_prediction
from ._theme import panel_colors
from ._ui_helpers import maybe_tight_layout, style_axes

_EMPTY = "No fitted model diagnostics yet."


class ModelDiagnosticsWidget(QWidget):
    """Primary diagnostics surface for the Model workspace."""

    roi_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fig = Figure(figsize=(8, 6), facecolor=panel_colors()["bg"])
        self._canvas = FigureCanvas(self._fig)
        self._canvas.mpl_connect("button_press_event", self._on_canvas_click)
        grid = self._fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0])
        self._pred_ax = self._fig.add_subplot(grid[0, 0])
        self._resid_ax = self._fig.add_subplot(grid[0, 1])
        self._target_dist_ax = self._fig.add_subplot(grid[1, 0])
        self._scores_ax = self._fig.add_subplot(grid[1, 1])
        self._spectral_ax = self._fig.add_subplot(grid[2, :])
        self._spectral_payload: dict[str, Any] | None = None
        self._spectral_mode = "coefficients"
        self._roi_point_payloads: dict[object, tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = {}
        self._last_render_payload: dict[str, Any] | None = None
        self._last_empty_reason: str | None = None
        self._last_layout_size: tuple[int, int] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._canvas)

        self.clear()

    def set_spectral_mode(self, mode: str) -> None:
        """Select which spectral diagnostic is rendered in the bottom plot."""
        normalized = str(mode).strip().lower()
        if normalized not in {"coefficients", "loadings"}:
            normalized = "coefficients"
        if normalized == self._spectral_mode:
            return
        self._spectral_mode = normalized
        self._render_spectral()
        self._last_layout_size = maybe_tight_layout(
            self._fig,
            self._canvas,
            pad=1.2,
            last_size=self._last_layout_size,
        )
        self._canvas.draw_idle()

    def update_data(self, payload: dict[str, Any] | None, *, empty_reason: str = "") -> None:
        """Render ROI-level prediction and residual diagnostics."""
        if payload is None:
            self.clear(empty_reason=empty_reason)
            return
        if payload is self._last_render_payload and empty_reason == self._last_empty_reason:
            return

        y_true = np.asarray(payload.get("y_true_roi", []), dtype=float)
        y_pred = np.asarray(payload.get("y_pred_roi", []), dtype=float)
        scores = payload.get("scores_roi")
        self._spectral_payload = payload
        roi_ids = self._coerce_roi_ids(payload.get("roi_ids"), expected_size=y_true.size)
        units = str(payload.get("target_unit", ""))
        if y_true.size == 0 or y_pred.size == 0:
            self.clear()
            return

        colors = panel_colors()
        self._fig.set_facecolor(colors["bg"])
        self._pred_ax.clear()
        self._resid_ax.clear()
        self._target_dist_ax.clear()
        self._scores_ax.clear()
        self._spectral_ax.clear()
        style_axes(self._pred_ax)
        style_axes(self._resid_ax)
        style_axes(self._target_dist_ax)
        style_axes(self._scores_ax)
        style_axes(self._spectral_ax)

        plot_prediction(
            y_true,
            y_pred,
            units=units,
            title="Predicted vs Reference",
            ax=self._pred_ax,
        )
        style_axes(self._pred_ax)
        self._register_roi_points(self._pred_ax, x=y_true, y=y_pred, roi_ids=roi_ids)

        residuals = y_pred - y_true
        self._resid_ax.scatter(y_true, residuals, s=30, alpha=0.75, edgecolors="none")
        self._resid_ax.axhline(0.0, color="#cccccc", linewidth=0.8, linestyle="--")
        unit_suffix = f" ({units})" if units else ""
        self._resid_ax.set_xlabel(f"Reference{unit_suffix}", color=colors["fg"])
        self._resid_ax.set_ylabel(f"Residual{unit_suffix}", color=colors["fg"])
        self._resid_ax.set_title("Residuals", color=colors["fg"], fontsize=9)
        style_axes(self._resid_ax)
        self._register_roi_points(self._resid_ax, x=y_true, y=residuals, roi_ids=roi_ids)
        self._render_target_distribution(y_true=y_true, units=units)
        scores_arr = self._render_scores(scores=scores, color_by=y_true)
        if scores_arr is not None:
            self._register_roi_points(
                self._scores_ax,
                x=scores_arr[:, 0],
                y=scores_arr[:, 1],
                roi_ids=roi_ids,
            )
        else:
            self._roi_point_payloads.pop(self._scores_ax, None)
        self._render_spectral()
        self._last_layout_size = maybe_tight_layout(
            self._fig,
            self._canvas,
            pad=1.2,
            last_size=self._last_layout_size,
        )
        self._canvas.draw_idle()
        self._last_render_payload = payload
        self._last_empty_reason = empty_reason

    def clear(self, *, empty_reason: str = "") -> None:
        """Show empty diagnostics placeholders."""
        if (
            self._spectral_payload is None
            and self._last_render_payload is None
            and empty_reason == self._last_empty_reason
        ):
            return
        colors = panel_colors()
        self._fig.set_facecolor(colors["bg"])
        self._pred_ax.clear()
        self._resid_ax.clear()
        self._target_dist_ax.clear()
        self._scores_ax.clear()
        self._spectral_ax.clear()
        style_axes(self._pred_ax)
        style_axes(self._resid_ax)
        style_axes(self._target_dist_ax)
        style_axes(self._scores_ax)
        style_axes(self._spectral_ax)
        self._pred_ax.set_title("Predicted vs Reference", color=colors["fg"], fontsize=9)
        self._resid_ax.set_title("Residuals", color=colors["fg"], fontsize=9)
        self._pred_ax.set_xlabel("Reference", color=colors["fg"])
        self._pred_ax.set_ylabel("Predicted", color=colors["fg"])
        self._resid_ax.set_xlabel("Reference", color=colors["fg"])
        self._resid_ax.set_ylabel("Residual", color=colors["fg"])
        self._target_dist_ax.set_title("Target Distribution", color=colors["fg"], fontsize=9)
        self._target_dist_ax.set_xlabel("Reference", color=colors["fg"])
        self._target_dist_ax.set_ylabel("ROI count", color=colors["fg"])
        self._scores_ax.set_title("Scores (LV1 vs LV2)", color=colors["fg"], fontsize=9)
        self._scores_ax.set_xlabel("LV1 score", color=colors["fg"])
        self._scores_ax.set_ylabel("LV2 score", color=colors["fg"])
        title, ylabel = self._spectral_labels()
        self._spectral_ax.set_title(title, color=colors["fg"], fontsize=9)
        self._spectral_ax.set_xlabel("Wavelength", color=colors["fg"])
        self._spectral_ax.set_ylabel(ylabel, color=colors["fg"])
        self._placeholder(self._pred_ax, empty_reason or _EMPTY)
        self._placeholder(self._resid_ax, "No residual diagnostics yet.")
        self._placeholder(self._target_dist_ax, "No target distribution diagnostics yet.")
        self._placeholder(self._scores_ax, "No latent scores diagnostics yet.")
        self._placeholder(self._spectral_ax, "No spectral diagnostics yet.")
        self._spectral_payload = None
        self._roi_point_payloads.clear()
        self._last_layout_size = maybe_tight_layout(
            self._fig,
            self._canvas,
            pad=1.2,
            last_size=self._last_layout_size,
        )
        self._canvas.draw_idle()
        self._last_render_payload = None
        self._last_empty_reason = empty_reason

    def _render_target_distribution(self, *, y_true: np.ndarray, units: str) -> None:
        colors = panel_colors()
        y_true_arr = np.asarray(y_true, dtype=float)
        if y_true_arr.ndim != 1 or y_true_arr.size == 0:
            self._target_dist_ax.set_title("Target Distribution", color=colors["fg"], fontsize=9)
            self._placeholder(self._target_dist_ax, "No target distribution diagnostics yet.")
            return
        n_unique = int(np.unique(y_true_arr).size)
        n_bins = max(1, min(10, n_unique if n_unique > 1 else 1))
        self._target_dist_ax.hist(
            y_true_arr,
            bins=n_bins,
            color=colors["accent"],
            alpha=0.85,
            edgecolor=colors["ax_bg"],
        )
        unit_suffix = f" ({units})" if units else ""
        self._target_dist_ax.set_xlabel(f"Reference{unit_suffix}", color=colors["fg"])
        self._target_dist_ax.set_ylabel("ROI count", color=colors["fg"])
        self._target_dist_ax.set_title("Target Distribution", color=colors["fg"], fontsize=9)
        style_axes(self._target_dist_ax)

    def _render_scores(self, *, scores: Any, color_by: np.ndarray) -> np.ndarray | None:
        colors = panel_colors()
        if scores is None:
            self._scores_ax.set_title("Scores (LV1 vs LV2)", color=colors["fg"], fontsize=9)
            self._placeholder(self._scores_ax, "No latent scores diagnostics yet.")
            return None
        scores_arr = np.asarray(scores, dtype=float)
        if scores_arr.ndim != 2 or scores_arr.shape[0] == 0 or scores_arr.shape[1] < 2:
            self._scores_ax.set_title("Scores (LV1 vs LV2)", color=colors["fg"], fontsize=9)
            self._placeholder(self._scores_ax, "No latent scores diagnostics yet.")
            return None
        self._scores_ax.scatter(
            scores_arr[:, 0],
            scores_arr[:, 1],
            c=np.asarray(color_by, dtype=float),
            s=36,
            alpha=0.8,
            edgecolors="none",
        )
        self._scores_ax.axhline(0.0, color="#cccccc", linewidth=0.6, linestyle="--")
        self._scores_ax.axvline(0.0, color="#cccccc", linewidth=0.6, linestyle="--")
        self._scores_ax.set_xlabel("LV1 score", color=colors["fg"])
        self._scores_ax.set_ylabel("LV2 score", color=colors["fg"])
        self._scores_ax.set_title("Scores (LV1 vs LV2)", color=colors["fg"], fontsize=9)
        style_axes(self._scores_ax)
        return scores_arr

    def _render_spectral(self) -> None:
        colors = panel_colors()
        self._spectral_ax.clear()
        style_axes(self._spectral_ax)
        mode = self._spectral_mode
        title, ylabel = self._spectral_labels()
        payload = self._spectral_payload
        if payload is None:
            self._spectral_ax.set_title(title, color=colors["fg"], fontsize=9)
            self._placeholder(self._spectral_ax, "No spectral diagnostics yet.")
            return

        values = payload.get(mode)
        if values is None:
            self._spectral_ax.set_title(title, color=colors["fg"], fontsize=9)
            self._placeholder(self._spectral_ax, "No spectral diagnostics yet.")
            return
        y = np.asarray(values, dtype=float).squeeze()
        if y.ndim != 1 or y.size == 0:
            self._spectral_ax.set_title(title, color=colors["fg"], fontsize=9)
            self._placeholder(self._spectral_ax, "No spectral diagnostics yet.")
            return

        axis = payload.get("spectral_axis")
        if axis is None:
            x = np.arange(y.size)
            xlabel = "Band"
        else:
            x = np.asarray(axis, dtype=float).squeeze()
            if x.ndim != 1 or x.size != y.size:
                x = np.arange(y.size)
                xlabel = "Band"
            else:
                xlabel = "Wavelength"

        self._spectral_ax.plot(x, y, linewidth=1.2, color=colors["accent"])
        self._spectral_ax.axhline(0.0, color="#cccccc", linewidth=0.6, linestyle="--")
        self._spectral_ax.set_xlabel(xlabel, color=colors["fg"])
        self._spectral_ax.set_ylabel(ylabel, color=colors["fg"])
        self._spectral_ax.set_title(title, color=colors["fg"], fontsize=9)
        style_axes(self._spectral_ax)

    @staticmethod
    def _placeholder(ax, message: str) -> None:
        """Write an empty-state message in the middle of ``ax``."""
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
                color=panel_colors()["fg"], fontsize=8, wrap=True)

    def _spectral_labels(self) -> tuple[str, str]:
        if self._spectral_mode == "loadings":
            return "Latent Loadings (LV1)", "Loading"
        return "Regression Coefficients", "Coefficient"

    def restyle_theme(self) -> None:
        """Repaint the diagnostics surface for the current theme."""
        self._last_layout_size = None
        payload = self._last_render_payload
        empty_reason = self._last_empty_reason or ""
        if payload is None:
            self._last_empty_reason = None
            self.clear(empty_reason=empty_reason)
            return
        self._last_render_payload = None
        self.update_data(payload, empty_reason=empty_reason)

    def _coerce_roi_ids(self, roi_ids: Any, *, expected_size: int) -> tuple[str, ...] | None:
        if roi_ids is None:
            return None
        roi_ids_arr = np.asarray(roi_ids, dtype=object).reshape(-1)
        if roi_ids_arr.size != expected_size:
            return None
        return tuple(str(roi_id) for roi_id in roi_ids_arr.tolist())

    def _register_roi_points(
        self,
        ax,
        *,
        x: np.ndarray,
        y: np.ndarray,
        roi_ids: tuple[str, ...] | None,
    ) -> None:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if (
            roi_ids is None
            or x_arr.size == 0
            or y_arr.size != x_arr.size
            or len(roi_ids) != x_arr.size
        ):
            self._roi_point_payloads.pop(ax, None)
            return
        self._roi_point_payloads[ax] = (x_arr, y_arr, roi_ids)

    def _on_canvas_click(self, event) -> None:
        if event.inaxes not in self._roi_point_payloads:
            return
        if event.x is None or event.y is None:
            return
        x, y, roi_ids = self._roi_point_payloads[event.inaxes]
        if x.size == 0:
            return
        point_positions = event.inaxes.transData.transform(np.column_stack([x, y]))
        event_position = np.asarray([event.x, event.y], dtype=float)
        distances = np.linalg.norm(point_positions - event_position, axis=1)
        nearest_index = int(np.argmin(distances))
        if not np.isfinite(distances[nearest_index]) or distances[nearest_index] > 12.0:
            return
        self.roi_selected.emit(roi_ids[nearest_index])

