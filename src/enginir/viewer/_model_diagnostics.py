"""Model workspace diagnostics: the plots of one model run on the `Fit`, `Spectral`,
`Components` and `Distance` pages.

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
    statistics = payload.get("fit_statistics") or {}
    extra = "".join(f"\n{label} = {statistics[key]:.4g}{suffix}" for key, label, suffix in
                    (("rmsee", "RMSEE", value_unit), ("r2_adj", "$R^2_{adj}$", ""))
                    if key in statistics)
    ax.text(0.03, 0.97,
            f"$R^2$ = {r2_score(y_true, y_pred):.4f}\nRMSECV = {rmsecv(y_true, y_pred):.4g}"
            f"{value_unit}\nBias = {bias(y_true, y_pred):.4g}{value_unit}{extra}",
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
    variances = payload.get("score_variances")
    if variances is not None:  # Hotelling T² ellipses of the training pixels (§7.5)
        angle = np.linspace(0.0, 2.0 * np.pi, 200)
        for key, style in (("t2_crit_2_095", ":"), ("t2_crit_2_099", "--")):
            limit = payload.get(key)
            if limit is not None:
                radii = np.sqrt(float(limit) * np.asarray(variances, dtype=float))
                ax.plot(radii[0] * np.cos(angle), radii[1] * np.sin(angle), color=_GUIDE,
                        linewidth=0.9, linestyle=style)
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


def _curve(payload: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    curve = payload.get("component_curve")
    if not isinstance(curve, Mapping) or curve.get("rmsecv") is None:
        return None
    return {key: np.asarray(value, dtype=float).reshape(-1) if np.ndim(value) else value
            for key, value in curve.items()}


_RULE_LABELS = (("suggested_one_se", "one SE"), ("suggested_min_rmsecv", "lowest RMSECV"),
                ("suggested_q2_limit", "Q² limit"))


def _render_components(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = _axes(fig, colors)
    curve = _curve(payload)
    if curve is None:
        draw_placeholder(ax, "No component curve (refit a model from before 1.3).", colors)
        return
    a = np.arange(1, curve["rmsecv"].size + 1)
    ax.errorbar(a, curve["rmsecv"], yerr=curve.get("rmsecv_se"), color=colors["accent"],
                marker="o", markersize=4, linewidth=1.2, capsize=3)
    chosen = payload.get("n_components")
    if isinstance(chosen, (int, np.integer)) and 1 <= int(chosen) <= a.size:
        ax.axvline(int(chosen), color=colors["fg"], linewidth=1.0, alpha=0.6)
    lines = [f"model: {chosen}"] + [f"{label}: {int(curve[key])}" for key, label in _RULE_LABELS
                                     if key in curve]
    ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=colors["fg"])
    ax.set_xticks(a)
    _label(ax, "Number of components", f"RMSECV{_suffix(_unit(payload))}", colors)


def _table_components(payload: Any) -> PlotTable:
    curve = _curve(payload)
    if curve is None:
        return PlotTable()
    columns = [PlotColumn("n_components", np.arange(1, curve["rmsecv"].size + 1)),
               PlotColumn("rmsecv", curve["rmsecv"], _unit(payload))]
    if "rmsecv_se" in curve:  # the ends of the drawn error bars
        columns += [PlotColumn("rmsecv_minus_se", curve["rmsecv"] - curve["rmsecv_se"]),
                    PlotColumn("rmsecv_plus_se", curve["rmsecv"] + curve["rmsecv_se"])]
    return PlotTable(columns=tuple(columns),
                     metadata={label: str(int(curve[key])) for key, label in _RULE_LABELS
                               if key in curve})


def _table_r2_q2(payload: Any) -> PlotTable:
    curve = _curve(payload)
    if curve is None:
        return PlotTable()
    return PlotTable(columns=(PlotColumn("n_components", np.arange(1, curve["rmsecv"].size + 1)),
                              *(PlotColumn(key, curve[key]) for key in
                                ("r2y_cum", "q2_cum", "r2x_cum") if key in curve)))


def _render_r2_q2(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = _axes(fig, colors)
    curve = _curve(payload)
    if curve is None:
        draw_placeholder(ax, "No component curve (refit a model from before 1.3).", colors)
        return
    a = np.arange(1, curve["rmsecv"].size + 1)
    for key, label, style in (("r2y_cum", "R²Y(cum)", "-"), ("q2_cum", "Q²(cum)", "--"),
                              ("r2x_cum", "R²X(cum)", ":")):
        if key in curve:
            ax.plot(a, curve[key], linestyle=style, marker="o", markersize=3, label=label)
    ax.set_xticks(a)
    ax.legend(fontsize=7, loc="lower right")
    _label(ax, "Number of components", "Fraction", colors)


def _render_loadings(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = _axes(fig, colors)
    matrix = payload.get("loadings_matrix")
    axis = payload.get("spectral_axis")
    if matrix is None:
        draw_placeholder(ax, "No loadings yet.", colors)
        return
    matrix = np.asarray(matrix, dtype=float)
    x = (np.asarray(axis, dtype=float) if axis is not None and np.size(axis) == matrix.shape[0]
         else np.arange(matrix.shape[0], dtype=float))
    for c in range(matrix.shape[1]):
        ax.plot(x, matrix[:, c], linewidth=1.1, label=f"LV{c + 1}")
    ax.axhline(0.0, color=_GUIDE, linewidth=0.6, linestyle="--")
    ax.legend(fontsize=7)
    _label(ax, "Wavelength" if axis is not None else "Band", "Loading", colors)


def _table_loadings(payload: Any) -> PlotTable:
    matrix = payload.get("loadings_matrix")
    if matrix is None:
        return PlotTable()
    matrix = np.asarray(matrix, dtype=float)
    axis = payload.get("spectral_axis")
    x = (np.asarray(axis, dtype=float) if axis is not None and np.size(axis) == matrix.shape[0]
         else np.arange(matrix.shape[0], dtype=float))
    return PlotTable(columns=(PlotColumn("wavelength" if axis is not None else "band", x),
                              *(PlotColumn(f"lv{c + 1}_loading", matrix[:, c])
                                for c in range(matrix.shape[1]))))


def _histogram(values: Any) -> tuple[np.ndarray, np.ndarray]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return np.histogram(finite, bins=40) if finite.size else (np.zeros(1), np.array([0.0, 1.0]))


def _distance_spec(key: str, limit_key: str, plot_id: str, title: str, xlabel: str) -> PlotSpec:
    def render(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
        ax = _axes(fig, colors)
        values = payload.get(f"{key}_pixels")
        if values is None:
            draw_placeholder(ax, "No training distances (refit a model from before 1.3).",
                             colors)
            return
        counts, edges = _histogram(values)
        ax.stairs(counts, edges, fill=True, color=colors["accent"], alpha=0.85)
        values = np.asarray(values, dtype=float)
        limit = payload.get(limit_key)
        if limit is not None:
            ax.axvline(float(limit), color=colors["fg"], linewidth=1.0, linestyle="--")
            share = float(np.mean(values > float(limit)))
            ax.text(0.97, 0.97, f"limit (0.99) = {float(limit):.3g}\nabove: {share:.1%}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color=colors["fg"])
        _label(ax, xlabel, "Training pixels", colors)

    def table(payload: Any) -> PlotTable:
        values = payload.get(f"{key}_pixels")
        if values is None:
            return PlotTable()
        counts, edges = _histogram(values)
        limit = payload.get(limit_key)
        return PlotTable(columns=(PlotColumn("bin_start", edges[:-1]),
                                  PlotColumn("bin_end", edges[1:]),
                                  PlotColumn("pixel_count", counts)),
                         metadata={} if limit is None else {"limit_099": f"{float(limit):.6g}"})

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
    PlotSpec("model.loadings", "Latent Loadings (LV1-LV3)", _render_loadings, _table_loadings,
             empty_text=_EMPTY),
    _spectral_spec("coef_cs", "model.coef_cs", "Scaled Coefficients (CoeffCS)", "CoeffCS",
                   "No scaled coefficients (refit a model from before 1.3)."),
    PlotSpec("model.components", "RMSECV vs Components", _render_components, _table_components,
             empty_text=_EMPTY),
    PlotSpec("model.r2_q2", "R² and Q² vs Components", _render_r2_q2, _table_r2_q2,
             empty_text=_EMPTY),
    _distance_spec("dmodx", "dcrit_099", "model.dmodx", "Distance to Model (DModX)",
                   "Normalised DModX"),
    _distance_spec("t2", "t2_crit_099", "model.t2", "Hotelling T²", "T²"),
)

_PAGES = (
    PlotPage("Fit", (PlotSlot("model.pred_vs_ref", 0, 0), PlotSlot("model.residuals", 0, 1),
                     PlotSlot("model.target_distribution", 1, 0),
                     PlotSlot("model.scores", 1, 1))),
    PlotPage("Spectral", (PlotSlot("model.coefficients", 0, 0),
                          PlotSlot("model.coef_cs", 0, 1),
                          PlotSlot("model.loadings", 1, 0))),
    PlotPage("Components", (PlotSlot("model.components", 0, 0),
                            PlotSlot("model.r2_q2", 0, 1))),
    PlotPage("Distance", (PlotSlot("model.dmodx", 0, 0), PlotSlot("model.t2", 0, 1))),
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
