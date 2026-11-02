"""Model workspace diagnostics: the plots of one model run on the `Fit` and `Spectral` pages.

Every diagnostic is a :class:`~enginir.viewer._plot_panel.PlotSpec`
registered for the ``"Model"`` workspace
(``model-workspace-diagnostics-contract_v3.md``). All plots receive the same
diagnostics payload from
:meth:`~enginir.viewer._modeling.ModelingView.workspace_diagnostics_payload`;
each one draws and exports its own part of it. Points are per ROI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..validation import bias, r2_score, rmsecv
from ._plot_grid import PlotGrid, PlotPage, PlotSlot
from ._plot_panel import PlotColumn, PlotSpec, PlotTable, RoiPoints, draw_placeholder
from ._plot_registry import PLOT_SPECS, WORKSPACE_PAGES, register_pages, register_plot
from ._ui_helpers import style_axes

_EMPTY = "No fitted model diagnostics yet."
_WORKSPACE = "Model"
_GUIDE = "#9aa3ad"  # reference lines, readable on light and dark backgrounds


# ----------------------------------------------------------------------
# Payload access
# ----------------------------------------------------------------------


def _cv_pairs(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """ROI-level reference values and cross-validated predictions."""
    y_true = np.asarray(payload.get("y_true_roi", []), dtype=float).reshape(-1)
    y_pred = np.asarray(payload.get("y_pred_roi", []), dtype=float).reshape(-1)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        return None
    return y_true, y_pred


def _roi_ids(payload: Mapping[str, Any], n: int) -> tuple[str, ...] | None:
    roi_ids = payload.get("roi_ids")
    if roi_ids is None:
        return None
    ids = np.asarray(roi_ids, dtype=object).reshape(-1)
    return tuple(str(roi_id) for roi_id in ids) if ids.size == n else None


def _roi_columns(payload: Mapping[str, Any], n: int) -> tuple[PlotColumn, ...]:
    ids = _roi_ids(payload, n)
    if ids is None:
        return ()
    names = payload.get("roi_names")
    names = list(names) if names is not None and len(names) == n else list(ids)
    return PlotColumn("roi_name", names), PlotColumn("roi_id", list(ids))


def _scores(payload: Mapping[str, Any], n: int) -> np.ndarray | None:
    scores = payload.get("scores_roi")
    if scores is None:
        return None
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != n or scores.shape[1] < 2:
        return None
    return scores


def _spectrum(payload: Mapping[str, Any], key: str) -> tuple[np.ndarray, np.ndarray, str] | None:
    """(x, y, x label) of a spectral diagnostic; band index without a wavelength axis."""
    values = payload.get(key)
    if values is None:
        return None
    y = np.asarray(values, dtype=float).squeeze()
    if y.ndim != 1 or y.size == 0:
        return None
    axis = payload.get("spectral_axis")
    if axis is not None:
        x = np.asarray(axis, dtype=float).squeeze()
        if x.ndim == 1 and x.size == y.size:
            return x, y, "Wavelength"
    return np.arange(y.size, dtype=float), y, "Band"


def _unit(payload: Mapping[str, Any]) -> str:
    return str(payload.get("target_unit", ""))


def _suffix(unit: str) -> str:
    return f" ({unit})" if unit else ""


def _target_histogram(y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_unique = int(np.unique(y_true).size)
    return np.histogram(y_true, bins=max(1, min(10, n_unique)))


def _axes(fig: Figure, colors: Mapping[str, str]) -> Any:
    ax = fig.add_subplot(111)
    style_axes(ax, colors)
    return ax


def _label(ax: Any, xlabel: str, ylabel: str, colors: Mapping[str, str]) -> None:
    ax.set_xlabel(xlabel, color=colors["fg"])
    ax.set_ylabel(ylabel, color=colors["fg"])


# ----------------------------------------------------------------------
# Fit page
# ----------------------------------------------------------------------


def _render_pred_vs_ref(fig: Figure, payload: Any, colors: Mapping[str, str]
                        ) -> Sequence[RoiPoints] | None:
    ax = _axes(fig, colors)
    pairs = _cv_pairs(payload)
    if pairs is None:
        draw_placeholder(ax, _EMPTY, colors)
        return None
    y_true, y_pred = pairs
    ax.scatter(y_true, y_pred, s=30, alpha=0.75, edgecolors="none", color=colors["accent"])
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (high - low) if high > low else 1.0
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color=_GUIDE, linewidth=1.0,
            linestyle="--")
    unit = _unit(payload)
    value_unit = f" {unit}" if unit else ""
    ax.text(0.03, 0.97,
            f"$R^2$ = {r2_score(y_true, y_pred):.4f}\nRMSECV = {rmsecv(y_true, y_pred):.4g}"
            f"{value_unit}\nBias = {bias(y_true, y_pred):.4g}{value_unit}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=colors["fg"])
    _label(ax, f"Reference{_suffix(unit)}", f"Predicted (CV){_suffix(unit)}", colors)
    ids = _roi_ids(payload, y_true.size)
    return None if ids is None else [RoiPoints(ax, y_true, y_pred, ids)]


def _table_pred_vs_ref(payload: Any) -> PlotTable:
    pairs = _cv_pairs(payload)
    if pairs is None:
        return PlotTable()
    y_true, y_pred = pairs
    unit = _unit(payload)
    return PlotTable(columns=(*_roi_columns(payload, y_true.size),
                              PlotColumn("reference", y_true, unit),
                              PlotColumn("predicted_cv", y_pred, unit)),
                     metadata={"r2": f"{r2_score(y_true, y_pred):.6g}",
                               "rmsecv": f"{rmsecv(y_true, y_pred):.6g}",
                               "bias": f"{bias(y_true, y_pred):.6g}"})


def _render_residuals(fig: Figure, payload: Any, colors: Mapping[str, str]
                      ) -> Sequence[RoiPoints] | None:
    ax = _axes(fig, colors)
    pairs = _cv_pairs(payload)
    if pairs is None:
        draw_placeholder(ax, "No residual diagnostics yet.", colors)
        return None
    y_true, y_pred = pairs
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, s=30, alpha=0.75, edgecolors="none", color=colors["accent"])
    ax.axhline(0.0, color=_GUIDE, linewidth=0.8, linestyle="--")
    unit = _unit(payload)
    _label(ax, f"Reference{_suffix(unit)}", f"Residual{_suffix(unit)}", colors)
    ids = _roi_ids(payload, y_true.size)
    return None if ids is None else [RoiPoints(ax, y_true, residuals, ids)]


def _table_residuals(payload: Any) -> PlotTable:
    pairs = _cv_pairs(payload)
    if pairs is None:
        return PlotTable()
    y_true, y_pred = pairs
    unit = _unit(payload)
    return PlotTable(columns=(*_roi_columns(payload, y_true.size),
                              PlotColumn("reference", y_true, unit),
                              PlotColumn("residual", y_pred - y_true, unit)),
                     metadata={"residual": "predicted_cv - reference"})


def _render_target_distribution(fig: Figure, payload: Any, colors: Mapping[str, str]
                                ) -> Sequence[RoiPoints] | None:
    ax = _axes(fig, colors)
    pairs = _cv_pairs(payload)
    if pairs is None:
        draw_placeholder(ax, "No target distribution diagnostics yet.", colors)
        return None
    counts, edges = _target_histogram(pairs[0])
    ax.stairs(counts, edges, fill=True, color=colors["accent"], alpha=0.85)
    _label(ax, f"Reference{_suffix(_unit(payload))}", "ROI count", colors)
    return None


def _table_target_distribution(payload: Any) -> PlotTable:
    pairs = _cv_pairs(payload)
    if pairs is None:
        return PlotTable()
    counts, edges = _target_histogram(pairs[0])
    unit = _unit(payload)
    return PlotTable(columns=(PlotColumn("bin_start", edges[:-1], unit),
                              PlotColumn("bin_end", edges[1:], unit),
                              PlotColumn("roi_count", counts)))


def _render_scores(fig: Figure, payload: Any, colors: Mapping[str, str]
                   ) -> Sequence[RoiPoints] | None:
    ax = _axes(fig, colors)
    pairs = _cv_pairs(payload)
    scores = None if pairs is None else _scores(payload, pairs[0].size)
    if pairs is None or scores is None:
        draw_placeholder(ax, "No latent scores diagnostics yet.", colors)
        return None
    ax.scatter(scores[:, 0], scores[:, 1], c=pairs[0], s=36, alpha=0.8, edgecolors="none")
    ax.axhline(0.0, color=_GUIDE, linewidth=0.6, linestyle="--")
    ax.axvline(0.0, color=_GUIDE, linewidth=0.6, linestyle="--")
    _label(ax, "LV1 training score", "LV2 training score", colors)
    ids = _roi_ids(payload, pairs[0].size)
    return None if ids is None else [RoiPoints(ax, scores[:, 0], scores[:, 1], ids)]


def _table_scores(payload: Any) -> PlotTable:
    pairs = _cv_pairs(payload)
    scores = None if pairs is None else _scores(payload, pairs[0].size)
    if pairs is None or scores is None:
        return PlotTable()
    return PlotTable(columns=(*_roi_columns(payload, scores.shape[0]),
                              PlotColumn("lv1_training_score", scores[:, 0]),
                              PlotColumn("lv2_training_score", scores[:, 1]),
                              PlotColumn("reference", pairs[0], _unit(payload))),
                     metadata={"scores": "ROI mean of the training pixel scores"})


# ----------------------------------------------------------------------
# Spectral page
# ----------------------------------------------------------------------


def _spectral_spec(key: str, plot_id: str, title: str, ylabel: str, empty: str) -> PlotSpec:
    def render(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
        ax = _axes(fig, colors)
        spectrum = _spectrum(payload, key)
        if spectrum is None:
            draw_placeholder(ax, empty, colors)
            return
        x, y, xlabel = spectrum
        ax.plot(x, y, linewidth=1.2, color=colors["accent"])
        ax.axhline(0.0, color=_GUIDE, linewidth=0.6, linestyle="--")
        _label(ax, xlabel, ylabel, colors)

    def table(payload: Any) -> PlotTable:
        spectrum = _spectrum(payload, key)
        if spectrum is None:
            return PlotTable()
        x, y, xlabel = spectrum
        return PlotTable(columns=(PlotColumn(xlabel.lower(), x), PlotColumn(key, y)))

    return PlotSpec(plot_id, title, render, table, empty_text=_EMPTY)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------

_SPECS = (
    PlotSpec("model.pred_vs_ref", "Predicted vs Reference", _render_pred_vs_ref,
             _table_pred_vs_ref, empty_text=_EMPTY),
    PlotSpec("model.residuals", "Residuals", _render_residuals, _table_residuals,
             empty_text=_EMPTY),
    PlotSpec("model.target_distribution", "Target Distribution", _render_target_distribution,
             _table_target_distribution, empty_text=_EMPTY),
    PlotSpec("model.scores", "Training Scores (LV1 vs LV2)", _render_scores, _table_scores,
             empty_text=_EMPTY),
    _spectral_spec("coefficients", "model.coefficients", "Regression Coefficients",
                   "Coefficient", "No regression coefficients yet."),
    _spectral_spec("loadings", "model.loadings", "Latent Loadings (LV1)", "Loading",
                   "No loadings yet."),
)

_PAGES = (
    PlotPage("Fit", (PlotSlot("model.pred_vs_ref", 0, 0), PlotSlot("model.residuals", 0, 1),
                     PlotSlot("model.target_distribution", 1, 0),
                     PlotSlot("model.scores", 1, 1))),
    PlotPage("Spectral", (PlotSlot("model.coefficients", 0, 0),
                          PlotSlot("model.loadings", 1, 0))),
)

for _spec in _SPECS:
    register_plot(_spec)
register_pages(_WORKSPACE, _PAGES)


class ModelDiagnosticsWidget(QWidget):
    """Primary diagnostics surface for the Model workspace: the `Fit` and `Spectral` pages."""

    roi_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.grid = PlotGrid(WORKSPACE_PAGES[_WORKSPACE], PLOT_SPECS)
        self.grid.roi_selected.connect(self.roi_selected.emit)
        self._payload: Any = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.grid)
        self.clear()

    def update_data(
        self,
        payload: Mapping[str, Any] | None,
        *,
        empty_reason: str = "",
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Show the diagnostics of one model run; ``None`` shows ``empty_reason``.

        A new run opens the `Fit` page. The same payload object is a no-op.
        """
        if payload is not None and payload is not self._payload:
            self.grid.set_current_page(self.grid.page_names[0])
        self._payload = payload
        payloads = {} if payload is None else {spec.plot_id: payload for spec in _SPECS}
        self.grid.update_data(payloads, empty_reason=empty_reason or _EMPTY,
                              provenance=provenance)

    def clear(self, *, empty_reason: str = "") -> None:
        """Show the empty state on every plot."""
        self.update_data(None, empty_reason=empty_reason)

    def restyle_theme(self) -> None:
        """Repaint the diagnostics for the current theme."""
        self.grid.restyle_theme()


__all__ = ["ModelDiagnosticsWidget"]
