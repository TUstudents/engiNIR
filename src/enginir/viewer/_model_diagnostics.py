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
from dataclasses import replace
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from scipy.stats import f as f_distribution
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..validation import bias, r2_score, rmsecv
from ._plot_grid import PlotGrid, PlotPage, PlotSlot
from ._plot_help import MODEL_PLOT_HELP
from ._plot_panel import PlotColumn, PlotSpec, PlotTable, RoiPoints, draw_placeholder
from ._plot_registry import PLOT_SPECS, WORKSPACE_PAGES, register_pages, register_plot
from ._theme import GUIDE_COLOR
from ._ui_helpers import is_band_index_axis, style_axes

_EMPTY = "No fitted model diagnostics yet."
_WORKSPACE = "Model"


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
            # A file without wavelengths carries band numbers 0..n-1 as its axis.
            return x, y, "Band" if is_band_index_axis(x) else "Wavelength"
    return np.arange(y.size, dtype=float), y, "Band"


def _legend(ax: Any, colors: Mapping[str, str], **kwargs: Any) -> None:
    """A legend in the plot's colours (matplotlib's default box is white in the dark theme)."""
    legend = ax.legend(fontsize=7, frameon=True, labelcolor=colors["fg"], framealpha=0.85,
                       **kwargs)
    legend.get_frame().set_facecolor(colors.get("ax_bg", colors["bg"]))
    legend.get_frame().set_edgecolor("none")


def _gapped(x: np.ndarray, *ys: np.ndarray) -> tuple[np.ndarray, ...]:
    """Insert NaN where a band selection leaves out bands, so lines do not bridge the gap.

    A gap is a step larger than 1.5 times the median step of ``x``.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return (x, *(np.asarray(y, dtype=float) for y in ys))
    steps = np.diff(x)
    typical = float(np.median(np.abs(steps)))
    gaps = np.flatnonzero(np.abs(steps) > 1.5 * typical) + 1 if typical > 0 else np.array([], int)
    if gaps.size == 0:
        return (x, *(np.asarray(y, dtype=float) for y in ys))
    xs = np.insert(x, gaps, np.nan)
    return (xs, *(np.insert(np.asarray(y, dtype=float), gaps, np.nan, axis=0) for y in ys))


def _unit(payload: Mapping[str, Any]) -> str:
    return str(payload.get("target_unit", ""))


def _suffix(unit: str) -> str:
    return f" ({unit})" if unit else ""


#: Up to this many distinct reference values are counted per value, not binned.
_DISCRETE_VALUES = 12


def _target_histogram(y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_unique = int(np.unique(y_true).size)
    return np.histogram(y_true, bins=max(1, min(10, n_unique)))


def _target_counts(y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Each distinct value and its ROI count, when there are few (levels such as 1, 4, 7,
    10); ``None`` for continuous values, which are binned. Equal-width bins over a few
    levels join into one flat block."""
    values, counts = np.unique(np.asarray(y_true, dtype=float), return_counts=True)
    return (values, counts) if values.size <= _DISCRETE_VALUES else None


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
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color=GUIDE_COLOR, linewidth=1.0,
            linestyle="--")
    unit = _unit(payload)
    value_unit = f" {unit}" if unit not in ("", "1") else ""  # "1": dimensionless
    statistics = payload.get("fit_statistics") or {}
    extra = "".join(f"\n{label} = {statistics[key]:.4g}{suffix}" for key, label, suffix in
                    (("rmsee", "RMSEE (per ROI)", value_unit), ("r2_adj", "$R^2_{adj}$", ""))
                    if key in statistics)
    ax.text(0.03, 0.97,
            f"per ROI\n$Q^2$ (CV $R^2$) = {r2_score(y_true, y_pred):.4f}"
            f"\nRMSECV (per ROI) = {rmsecv(y_true, y_pred):.4g}"
            f"{value_unit}\nBias = {bias(y_true, y_pred):.4g}{value_unit}{extra}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=colors["fg"])
    _label(ax, f"Target value{_suffix(unit)}", f"Predicted (CV){_suffix(unit)}", colors)
    ids = _roi_ids(payload, y_true.size)
    return None if ids is None else [RoiPoints(ax, y_true, y_pred, ids)]


def _table_pred_vs_ref(payload: Any) -> PlotTable:
    pairs = _cv_pairs(payload)
    if pairs is None:
        return PlotTable()
    y_true, y_pred = pairs
    unit = _unit(payload)
    return PlotTable(columns=(*_roi_columns(payload, y_true.size),
                              PlotColumn("target_value", y_true, unit),
                              PlotColumn("predicted_cv", y_pred, unit)),
                     metadata={"level": "per ROI (ROI means)",
                               "r2": f"{r2_score(y_true, y_pred):.6g}",
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
    ax.axhline(0.0, color=GUIDE_COLOR, linewidth=0.8, linestyle="--")
    unit = _unit(payload)
    _label(ax, f"Target value{_suffix(unit)}", f"Residual{_suffix(unit)}", colors)
    ids = _roi_ids(payload, y_true.size)
    return None if ids is None else [RoiPoints(ax, y_true, residuals, ids)]


def _table_residuals(payload: Any) -> PlotTable:
    pairs = _cv_pairs(payload)
    if pairs is None:
        return PlotTable()
    y_true, y_pred = pairs
    unit = _unit(payload)
    return PlotTable(columns=(*_roi_columns(payload, y_true.size),
                              PlotColumn("target_value", y_true, unit),
                              PlotColumn("residual", y_pred - y_true, unit)),
                     metadata={"residual": "predicted_cv - target_value"})


def _render_target_distribution(fig: Figure, payload: Any, colors: Mapping[str, str]
                                ) -> Sequence[RoiPoints] | None:
    ax = _axes(fig, colors)
    pairs = _cv_pairs(payload)
    if pairs is None:
        draw_placeholder(ax, "No target distribution diagnostics yet.", colors)
        return None
    levels = _target_counts(pairs[0])
    if levels is not None:
        values, counts = levels
        # Thick vertical lines, as the sorted VIP: the drawn data are the values and counts.
        ax.vlines(values, 0, counts, colors=colors["accent"], linewidth=14, alpha=0.85)
        span = float(values[-1] - values[0]) or 1.0
        ax.set_xlim(values[0] - 0.12 * span - 0.5, values[-1] + 0.12 * span + 0.5)
        ax.set_ylim(0, float(counts.max()) * 1.08)
        ax.set_xticks(values)
        ax.set_xticklabels([f"{v:g}" for v in values])
    else:
        counts, edges = _target_histogram(pairs[0])
        ax.stairs(counts, edges, fill=True, color=colors["accent"], alpha=0.85)
    ax.yaxis.get_major_locator().set_params(integer=True)
    _label(ax, f"Target value{_suffix(_unit(payload))}", "ROI count", colors)
    return None


def _table_target_distribution(payload: Any) -> PlotTable:
    pairs = _cv_pairs(payload)
    if pairs is None:
        return PlotTable()
    unit = _unit(payload)
    levels = _target_counts(pairs[0])
    if levels is not None:
        return PlotTable(columns=(PlotColumn("value", levels[0], unit),
                                  PlotColumn("roi_count", levels[1])))
    counts, edges = _target_histogram(pairs[0])
    return PlotTable(columns=(PlotColumn("bin_start", edges[:-1], unit),
                              PlotColumn("bin_end", edges[1:], unit),
                              PlotColumn("roi_count", counts)))


def roi_t2_ellipse(scores: np.ndarray, level: float, n_points: int = 200
                   ) -> np.ndarray | None:
    """The ROI-level Hotelling T² ellipse of two score columns, as ``(n_points, 2)``.

    The ROIs are the observations: centred on their mean scores, shaped by
    their covariance (ROI means of PLS scores need not be uncorrelated), with
    the limit ``2 (n − 1) / (n − 2) · F(level; 2, n − 2)`` for ``n`` ROIs.
    ``None`` with fewer than 4 ROIs or no spread. An ROI outside it is unusual
    among the ROIs; the pixel ellipse is the limit for single pixels.
    """
    finite = np.asarray(scores, dtype=float)[:, :2]
    finite = finite[np.all(np.isfinite(finite), axis=1)]
    n = finite.shape[0]
    if n < 4:
        return None
    variances, axes = np.linalg.eigh(np.cov(finite, rowvar=False))
    if not np.all(np.isfinite(variances)) or variances.min() <= 1e-12 * max(variances.max(), 0):
        return None
    limit = 2.0 * (n - 1) / (n - 2) * float(f_distribution.ppf(level, 2, n - 2))
    angle = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    return (finite.mean(axis=0)[:, None]
            + axes @ (np.sqrt(limit * variances)[:, None] * circle)).T


def _render_scores(fig: Figure, payload: Any, colors: Mapping[str, str]
                   ) -> Sequence[RoiPoints] | None:
    ax = _axes(fig, colors)
    pairs = _cv_pairs(payload)
    scores = None if pairs is None else _scores(payload, pairs[0].size)
    if pairs is None or scores is None:
        draw_placeholder(ax, payload.get("scores_message") or "No latent scores diagnostics yet.",
                         colors)
        return None
    missing = int(payload.get("scores_without_pixels", 0))
    if missing:  # every pixel of these ROIs was excluded: no point (§2)
        ax.text(0.01, 0.97, f"{missing} ROI(s) without fitted pixels not shown",
                transform=ax.transAxes, va="top", fontsize=8, color=colors["fg"])
    ax.scatter(scores[:, 0], scores[:, 1], c=pairs[0], s=36, alpha=0.8, edgecolors="none")
    ax.axhline(0.0, color=GUIDE_COLOR, linewidth=0.6, linestyle="--")
    ax.axvline(0.0, color=GUIDE_COLOR, linewidth=0.6, linestyle="--")
    variances = payload.get("score_variances")
    if variances is not None:  # Hotelling T² ellipses of the training pixels (§7.5)
        angle = np.linspace(0.0, 2.0 * np.pi, 200)
        for key, style, level in (("t2_crit_2_095", ":", 95), ("t2_crit_2_099", "--", 99)):
            limit = payload.get(key)
            if limit is not None:
                radii = np.sqrt(float(limit) * np.asarray(variances, dtype=float))
                ax.plot(radii[0] * np.cos(angle), radii[1] * np.sin(angle), color=GUIDE_COLOR,
                        linewidth=0.9, linestyle=style, label=f"T² {level} % pixels")
    for level, style in ((0.95, ":"), (0.99, "--")):  # the ROIs as observations
        ellipse = roi_t2_ellipse(scores, level)
        if ellipse is not None:
            ax.plot(ellipse[:, 0], ellipse[:, 1], color=colors["accent"], linewidth=1.1,
                    linestyle=style, label=f"T² {level:.0%} ROIs".replace("%", " %"))
    if ax.get_legend_handles_labels()[0]:
        _legend(ax, colors, loc="upper right", ncols=2, handlelength=2.2)
    _label(ax, "LV1 calibration score", "LV2 calibration score", colors)
    ids = _roi_ids(payload, pairs[0].size)
    return None if ids is None else [RoiPoints(ax, scores[:, 0], scores[:, 1], ids)]


def _table_scores(payload: Any) -> PlotTable:
    pairs = _cv_pairs(payload)
    scores = None if pairs is None else _scores(payload, pairs[0].size)
    if pairs is None or scores is None:
        return PlotTable()
    return PlotTable(columns=(*_roi_columns(payload, scores.shape[0]),
                              PlotColumn("lv1_calibration_score", scores[:, 0]),
                              PlotColumn("lv2_calibration_score", scores[:, 1]),
                              PlotColumn("target_value", pairs[0], _unit(payload))),
                     metadata={"scores": "ROI mean of the calibration pixel scores"})


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
        ax.plot(*_gapped(x, y), linewidth=1.2, color=colors["accent"])
        ax.axhline(0.0, color=GUIDE_COLOR, linewidth=0.6, linestyle="--")
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


_RULE_LABELS = (("suggested_one_se", "one SE"), ("suggested_min_rmsecv", "lowest RMSECV (per ROI)"),
                ("suggested_q2_limit", "Q² limit"), ("suggested_f_test", "F-test"),
                ("suggested_q2_lookahead", "Q² look-ahead"))
_RULE_KEYS = {"one_se": "suggested_one_se", "min_rmsecv": "suggested_min_rmsecv",
              "q2_limit": "suggested_q2_limit", "f_test": "suggested_f_test",
              "q2_lookahead": "suggested_q2_lookahead"}
_NO_CURVE = "No component curve available for this model."


def _n_groups(payload: Mapping[str, Any]) -> int:
    choice = payload.get("components_choice") or {}
    groups = choice.get("n_groups")
    return int(groups) if isinstance(groups, (int, np.integer)) and groups > 0 else 20


def _rule_reasons(curve: Mapping[str, Any], n_groups: int) -> list[str]:
    """Every rule's suggestion with the number that decided it (contract §7.4)."""
    from ..validation.components import F_TEST_ALPHA, q2_limit

    rmsecv, se, q2 = curve["rmsecv"], curve.get("rmsecv_se"), curve.get("q2")
    best = int(np.argmin(rmsecv))
    limit = q2_limit(n_groups)
    lines = []
    for key, label in _RULE_LABELS:
        if key not in curve:
            continue
        chosen = int(curve[key])
        reason = ""
        if key == "suggested_one_se" and se is not None:
            reason = f"RMSECV (per ROI) ≤ {rmsecv[best] + se[best]:.3g}"
        elif key == "suggested_min_rmsecv":
            reason = f"{rmsecv[best]:.3g}"
        elif key == "suggested_q2_limit" and q2 is not None:
            failing = np.flatnonzero(~(np.nan_to_num(q2, nan=-np.inf) > limit))
            reason = (f"Q²{_subscript(int(failing[0]) + 1)} = {q2[failing[0]]:.2f} ≤ {limit:g}"
                      if failing.size else f"all Q²ₐ > {limit:g}")
        elif key == "suggested_f_test":
            from scipy.stats import f as f_distribution

            critical = f_distribution.ppf(1 - F_TEST_ALPHA, n_groups, n_groups)
            reason = f"PRESS ratio < {critical:.2f}"
        lines.append(f"{label}: {chosen}" + (f" ({reason})" if reason else ""))
    return lines


def _subscript(number: int) -> str:
    return str(number).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def _render_components(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = _axes(fig, colors)
    curve = _curve(payload)
    if curve is None:
        draw_placeholder(ax, _NO_CURVE, colors)
        return
    a = np.arange(1, curve["rmsecv"].size + 1)
    ax.errorbar(a, curve["rmsecv"], yerr=curve.get("rmsecv_se"), color=colors["accent"],
                marker="o", markersize=4, linewidth=1.2, capsize=3, label="RMSECV (per ROI) ± SE")
    if "rmsee_roi" in curve:
        ax.plot(a, curve["rmsee_roi"], color=GUIDE_COLOR, marker=".", linewidth=1.0,
                linestyle="--", label="RMSEE (per ROI), fit")
    chosen = payload.get("n_components")
    if isinstance(chosen, (int, np.integer)) and 1 <= int(chosen) <= a.size:
        ax.axvline(int(chosen), color=colors["fg"], linewidth=1.0, alpha=0.6)
    lines = [f"model: {chosen}"]
    choice = payload.get("components_choice") or {}
    rule_choice = choice.get("rule_choice")
    if choice.get("noise_guard") == "on" and rule_choice not in (None, chosen):
        lines.append(f"noise guard: {_RULE_LABEL_OF.get(str(choice.get('rule')), 'rule')} "
                     f"{rule_choice} → {chosen}")
    lines += _rule_reasons(curve, _n_groups(payload))
    ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=colors["fg"], zorder=5,
            bbox={"facecolor": colors.get("ax_bg", colors["bg"]), "alpha": 0.85,
                  "edgecolor": "none", "pad": 2})
    ax.set_xticks(a)
    _legend(ax, colors, loc="lower left")
    _label(ax, "Number of components", f"Error per ROI{_suffix(_unit(payload))}", colors)


_RULE_LABEL_OF = {"one_se": "one SE", "min_rmsecv": "lowest RMSECV (per ROI)", "q2_limit": "Q² limit",
                  "f_test": "F-test", "q2_lookahead": "Q² look-ahead"}


def _table_components(payload: Any) -> PlotTable:
    curve = _curve(payload)
    if curve is None:
        return PlotTable()
    columns = [PlotColumn("n_components", np.arange(1, curve["rmsecv"].size + 1)),
               PlotColumn("rmsecv", curve["rmsecv"], _unit(payload))]
    if "rmsecv_se" in curve:  # the ends of the drawn error bars
        columns += [PlotColumn("rmsecv_minus_se", curve["rmsecv"] - curve["rmsecv_se"]),
                    PlotColumn("rmsecv_plus_se", curve["rmsecv"] + curve["rmsecv_se"])]
    if "rmsee_roi" in curve:
        columns.append(PlotColumn("rmsee", curve["rmsee_roi"], _unit(payload)))
    return PlotTable(columns=tuple(columns),
                     metadata={"level": "per ROI (ROI means)",
                               **{label: str(int(curve[key])) for key, label in _RULE_LABELS
                                  if key in curve}})


def _table_r2_q2(payload: Any) -> PlotTable:
    curve = _curve(payload)
    if curve is None:
        return PlotTable()
    return PlotTable(columns=(PlotColumn("n_components", np.arange(1, curve["rmsecv"].size + 1)),
                              *(PlotColumn(key, curve[key]) for key in
                                ("r2y_cum_roi", "q2_cum") if key in curve)),
                     metadata={"level": "per ROI (ROI means)"})


def _render_r2_q2(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = _axes(fig, colors)
    curve = _curve(payload)
    if curve is None:
        draw_placeholder(ax, _NO_CURVE, colors)
        return
    a = np.arange(1, curve["rmsecv"].size + 1)
    r2y, q2 = curve.get("r2y_cum_roi"), curve.get("q2_cum")
    if r2y is not None:
        ax.plot(a, r2y, marker="o", markersize=3, color=colors["accent"], label="R²Y(cum), fit")
    if q2 is not None:
        ax.plot(a, q2, marker="o", markersize=3, linestyle="--", color="#e3a33b",
                label="Q²(cum), cross-validated")
    if r2y is not None and q2 is not None:  # the gap: what the fit gains over prediction
        ax.fill_between(a, q2, r2y, color=GUIDE_COLOR, alpha=0.18, linewidth=0,
                        label="gap (over-fit)")
    ax.set_xticks(a)
    _legend(ax, colors, loc="lower right")
    _label(ax, "Number of components", "Fraction of the target, per ROI", colors)


def _per_component(payload: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    curve = _curve(payload)
    if curve is None or "q2" not in curve or "r2x_cum" not in curve:
        return None
    from ..validation.components import delta_r2x, noise_components

    out = {"a": np.arange(1, curve["rmsecv"].size + 1, dtype=float), "q2": curve["q2"],
           "delta_r2x": delta_r2x(curve),
           "noise": np.asarray(curve["noise"], dtype=bool) if "noise" in curve
           else noise_components(curve, n_samples=_n_groups(payload))}
    if "dw_weights" in curve:
        out["dw"] = curve["dw_weights"]
    return out


def _render_per_component(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = _axes(fig, colors)
    each = _per_component(payload)
    if each is None:
        draw_placeholder(ax, _NO_CURVE, colors)
        return
    from ..validation.components import NOISE_R2X, q2_limit

    a, noise = each["a"], each["noise"]
    q2_colors = np.where(noise, GUIDE_COLOR, colors["accent"]).tolist()
    ax.vlines(a - 0.17, 0.0, each["q2"], colors=q2_colors, linewidth=7,
              label="Q²ₐ, per ROI")
    ax.vlines(a + 0.17, 0.0, each["delta_r2x"], colors=np.where(
        noise, GUIDE_COLOR, "#3fb27f").tolist(), linewidth=7, label="ΔR²X, per pixel")
    ax.axhline(q2_limit(_n_groups(payload)), color=colors["fg"], linewidth=0.8,
               linestyle="--", label="Q² limit L")
    ax.axhline(NOISE_R2X, color="#3fb27f", linewidth=0.7, linestyle=":",
               label="ΔR²X 1 %")
    ax.axhline(0.0, color=GUIDE_COLOR, linewidth=0.6)
    if "dw" in each:
        rough = ax.twinx()
        rough.plot(a, each["dw"], color="#e3a33b", marker="o", markersize=3, linewidth=1.0,
                   label="Durbin–Watson of the weights")
        rough.set_ylabel("Durbin–Watson (0 smooth, 2 noise)", color=colors["fg"], fontsize=8)
        rough.tick_params(colors=colors["fg"], labelsize=7)
        for side in ("top", "left", "bottom"):
            rough.spines[side].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        more, more_labels = rough.get_legend_handles_labels()
        legend = rough.legend(handles + more, labels + more_labels, fontsize=7,
                              loc="upper center", frameon=True, labelcolor=colors["fg"],
                              framealpha=0.85)
        legend.get_frame().set_facecolor(colors.get("ax_bg", colors["bg"]))
        legend.get_frame().set_edgecolor("none")
    else:
        _legend(ax, colors, loc="upper right")
    ax.set_xticks(a)
    _label(ax, "Component (gray: noise)", "Q²ₐ and ΔR²X", colors)


def _table_per_component(payload: Any) -> PlotTable:
    each = _per_component(payload)
    if each is None:
        return PlotTable()
    columns = [PlotColumn("component", each["a"]),
               PlotColumn("q2_component_roi", each["q2"]),
               PlotColumn("delta_r2x_pixel", each["delta_r2x"]),
               PlotColumn("noise", each["noise"].astype(int))]
    if "dw" in each:
        columns.append(PlotColumn("durbin_watson_weights", each["dw"]))
    return PlotTable(columns=tuple(columns),
                     metadata={"noise": "ΔR²X < 0.01 and Q²ₐ ≤ L (contract §7.4)"})


def _render_r2_pixels(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = _axes(fig, colors)
    curve = _curve(payload)
    if curve is None:
        draw_placeholder(ax, _NO_CURVE, colors)
        return
    a = np.arange(1, curve["rmsecv"].size + 1)
    for key, label, style, color in (("r2x_cum", "R²X(cum), spectra", ":", "#3fb27f"),
                                     ("r2y_cum", "R²Y(cum), target", "-", colors["accent"])):
        if key in curve:
            ax.plot(a, curve[key], linestyle=style, marker="o", markersize=3, color=color,
                    label=label)
    ax.text(0.97, 0.05, "single pixels are noisy:\nR²Y per pixel stays below the ROI value",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=colors["fg"])
    ax.set_xticks(a)
    _legend(ax, colors, loc="center right")
    _label(ax, "Number of components", "Fraction, per pixel", colors)


def _table_r2_pixels(payload: Any) -> PlotTable:
    curve = _curve(payload)
    if curve is None:
        return PlotTable()
    return PlotTable(columns=(PlotColumn("n_components", np.arange(1, curve["rmsecv"].size + 1)),
                              *(PlotColumn(key, curve[key]) for key in ("r2x_cum", "r2y_cum")
                                if key in curve)),
                     metadata={"level": "per pixel (single spectra)"})


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
    gapped_x, gapped = _gapped(x, matrix)
    for c in range(matrix.shape[1]):
        ax.plot(gapped_x, gapped[:, c], linewidth=1.1, label=f"LV{c + 1}")
    ax.axhline(0.0, color=GUIDE_COLOR, linewidth=0.6, linestyle="--")
    _legend(ax, colors)
    band_axis = axis is None or is_band_index_axis(x)
    _label(ax, "Band" if band_axis else "Wavelength", "Loading", colors)


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


def _mean_behind(ax: Any, payload: Mapping[str, Any], x: np.ndarray,
                 colors: Mapping[str, str]) -> None:
    """Shade the model's mean spectrum behind a band-importance plot (own, hidden y scale)."""
    mean = payload.get("mean_spectrum")
    if mean is None or np.size(mean) != x.size:
        return
    behind = ax.twinx()
    # The twin axes lie below the plot and carry its background (matplotlib's is white).
    behind.set_facecolor(colors.get("ax_bg", colors["bg"]))
    gx, gy = _gapped(x, np.asarray(mean, dtype=float).reshape(-1))
    behind.fill_between(gx, np.nanmin(gy), gy, color=GUIDE_COLOR, alpha=0.18, linewidth=0)
    behind.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        behind.spines[side].set_visible(False)
    ax.set_zorder(behind.get_zorder() + 1)
    ax.patch.set_visible(False)


def _render_vip(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = _axes(fig, colors)
    spectrum = _spectrum(payload, "vip")
    if spectrum is None:
        draw_placeholder(ax, "VIP exists for PLS models.", colors)
        return
    x, y, xlabel = spectrum
    _mean_behind(ax, payload, x, colors)
    ax.plot(*_gapped(x, y), linewidth=1.2, color=colors["accent"])
    ax.axhline(1.0, color=colors["fg"], linewidth=0.8, alpha=0.7)
    ax.axhline(0.8, color=GUIDE_COLOR, linewidth=0.8, linestyle="--")
    above = float(np.mean(y > 1.0))
    ax.text(0.97, 0.97, f"VIP > 1: {above:.0%} of bands", transform=ax.transAxes, ha="right",
            va="top", fontsize=8, color=colors["fg"])
    _label(ax, xlabel, "VIP", colors)


def _table_vip(payload: Any) -> PlotTable:
    spectrum = _spectrum(payload, "vip")
    if spectrum is None:
        return PlotTable()
    x, y, xlabel = spectrum
    columns = [PlotColumn(xlabel.lower(), x), PlotColumn("vip", y)]
    mean = payload.get("mean_spectrum")
    if mean is not None and np.size(mean) == x.size:
        columns.append(PlotColumn("mean_spectrum", np.asarray(mean, dtype=float).reshape(-1)))
    return PlotTable(columns=tuple(columns))


def _sorted_vip(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, str] | None:
    spectrum = _spectrum(payload, "vip")
    if spectrum is None:
        return None
    x, y, xlabel = spectrum
    order = np.argsort(-y, kind="stable")
    return x[order], y[order], xlabel


def _render_vip_sorted(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = _axes(fig, colors)
    ranked = _sorted_vip(payload)
    if ranked is None:
        draw_placeholder(ax, "VIP exists for PLS models.", colors)
        return
    x, y, xlabel = ranked
    rank = np.arange(1, y.size + 1)
    width = max(0.6, min(4.0, 300.0 / y.size))
    ax.vlines(rank, 0.0, y, colors=np.where(y > 1.0, colors["accent"], GUIDE_COLOR).tolist(),
              linewidth=width)
    # The five largest, listed where the short bars leave room (labels on the bars overlap).
    top = "\n".join(f"{band:.0f} nm  {value:.2f}" if xlabel == "Wavelength"
                     else f"band {band:g}  {value:.2f}" for band, value in zip(x[:5], y[:5],
                                                                              strict=True))
    ax.text(0.97, 0.97, f"Largest VIP\n{top}", transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=colors["fg"], family="monospace")
    ax.axhline(1.0, color=colors["fg"], linewidth=0.8, alpha=0.7)
    ax.axhline(0.8, color=GUIDE_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xlim(0.5, y.size + 0.5)
    _label(ax, "Bands, from the largest VIP", "VIP", colors)


def _table_vip_sorted(payload: Any) -> PlotTable:
    ranked = _sorted_vip(payload)
    if ranked is None:
        return PlotTable()
    x, y, xlabel = ranked
    # The bands of each rank as metadata: the plot draws ranks, not wavelengths.
    return PlotTable(columns=(PlotColumn("rank", np.arange(1, y.size + 1)),
                              PlotColumn("vip", y)),
                     metadata={f"{xlabel.lower()}_by_rank": ", ".join(f"{v:g}" for v in x)})


def _render_selectivity(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = _axes(fig, colors)
    spectrum = _spectrum(payload, "selectivity_ratio")
    if spectrum is None:
        draw_placeholder(ax, "No selectivity ratio available for this model.", colors)
        return
    x, y, xlabel = spectrum
    _mean_behind(ax, payload, x, colors)
    ax.plot(*_gapped(x, y), linewidth=1.2, color=colors["accent"])
    _label(ax, xlabel, "Selectivity ratio", colors)


def _table_selectivity(payload: Any) -> PlotTable:
    spectrum = _spectrum(payload, "selectivity_ratio")
    if spectrum is None:
        return PlotTable()
    x, y, xlabel = spectrum
    return PlotTable(columns=(PlotColumn(xlabel.lower(), x), PlotColumn("selectivity_ratio", y)))


#: Share of the pixels the distance histograms show bin by bin; the rest is one overflow bar.
_SHOWN_QUANTILE = 99.5
#: The share of training pixels expected above a 0.99 limit.
_EXPECTED_ABOVE = 0.01


def _histogram(values: Any) -> tuple[np.ndarray, np.ndarray, int, float]:
    """40 bins up to the 99.5th percentile, the count above it and the largest value.

    Pixel distances have long tails (a few glints far out); bins over the whole
    range put almost every pixel into one bar.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return np.zeros(1), np.array([0.0, 1.0]), 0, 0.0
    top = float(np.percentile(finite, _SHOWN_QUANTILE))
    if top <= float(finite.min()):
        top = float(finite.max())
    counts, edges = np.histogram(finite[finite <= top], bins=40,
                                 range=(float(finite.min()), top))
    return counts, edges, int(np.sum(finite > top)), float(finite.max())


def _overflow_bin(edges: np.ndarray) -> tuple[float, float]:
    """Where the overflow bar is drawn: one bin width right of the last bin."""
    width = float(edges[-1] - edges[-2]) if edges.size > 1 else 1.0
    return float(edges[-1]), float(edges[-1]) + width


def _distance_spec(key: str, limit_key: str, plot_id: str, title: str, xlabel: str) -> PlotSpec:
    def render(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
        ax = _axes(fig, colors)
        values = payload.get(f"{key}_pixels")
        if values is None:
            draw_placeholder(ax, "No calibration distances available for this model.",
                             colors)
            return
        counts, edges, overflow, largest = _histogram(values)
        ax.stairs(counts, edges, fill=True, color=colors["accent"], alpha=0.85)
        lines = []
        if overflow:
            start, end = _overflow_bin(edges)
            ax.stairs([overflow], [start, end], fill=True, color=GUIDE_COLOR, alpha=0.85)
            lines.append(f"last bar: {overflow} pixels > {start:.3g} (max {largest:.3g})")
        values = np.asarray(values, dtype=float)
        limit = payload.get(limit_key)
        if limit is not None:
            ax.axvline(float(limit), color=colors["fg"], linewidth=1.0, linestyle="--")
            share = float(np.mean(values > float(limit)))
            lines[:0] = [f"limit (0.99) = {float(limit):.3g}",
                         f"above: {share:.1%} (expected {_EXPECTED_ABOVE:.0%})"]
        if lines:
            ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right",
                    va="top", fontsize=8, color=colors["fg"])
        _label(ax, xlabel, "Calibration pixels", colors)

    def table(payload: Any) -> PlotTable:
        values = payload.get(f"{key}_pixels")
        if values is None:
            return PlotTable()
        counts, edges, overflow, largest = _histogram(values)
        starts, ends, pixel_counts = list(edges[:-1]), list(edges[1:]), list(counts)
        metadata: dict[str, str] = {}
        if overflow:  # the drawn overflow bar; its values reach up to the largest one
            start, end = _overflow_bin(edges)
            starts, ends, pixel_counts = starts + [start], ends + [end], pixel_counts + [overflow]
            metadata["last_row"] = f"pixels above {start:.6g}, up to {largest:.6g}"
        limit = payload.get(limit_key)
        if limit is not None:
            metadata["limit_099"] = f"{float(limit):.6g}"
        return PlotTable(columns=(PlotColumn("bin_start", np.asarray(starts)),
                                  PlotColumn("bin_end", np.asarray(ends)),
                                  PlotColumn("pixel_count", np.asarray(pixel_counts))),
                         metadata=metadata)

    return PlotSpec(plot_id, title, render, table, empty_text=_EMPTY)


def _roi_shares(payload: Mapping[str, Any]
                ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, tuple[str, ...]] | None:
    """Per ROI, sorted from the largest DModX share: (DModX share, T² share or None,
    reference values, ROI ids)."""
    pairs = _cv_pairs(payload)
    dmodx = payload.get("roi_share_above_dcrit")
    if pairs is None or dmodx is None:
        return None
    dmodx = np.asarray(dmodx, dtype=float).reshape(-1)
    ids = _roi_ids(payload, pairs[0].size)
    if dmodx.size != pairs[0].size or ids is None:
        return None
    t2 = payload.get("roi_share_above_t2crit")
    t2 = None if t2 is None or np.size(t2) != dmodx.size else np.asarray(t2, float).reshape(-1)
    order = np.argsort(-dmodx, kind="stable")
    return (dmodx[order], None if t2 is None else t2[order], pairs[0][order],
            tuple(ids[i] for i in order))


def _roi_labels(payload: Mapping[str, Any], ids: tuple[str, ...]) -> list[str]:
    """Short ROI names (the shared prefix dropped, as in the ROI mosaic)."""
    from ._project_rois import _short_names

    names: Sequence[Any] = payload.get("roi_names") or ()
    all_ids = _roi_ids(payload, len(names)) or ()
    by_id = {str(roi_id): str(name) for roi_id, name in zip(all_ids, names, strict=False)}
    short = _short_names({roi_id: by_id.get(roi_id, roi_id) for roi_id in ids}, {})
    return [short[roi_id] for roi_id in ids]


def _render_roi_distance(fig: Figure, payload: Any, colors: Mapping[str, str]
                         ) -> Sequence[RoiPoints] | None:
    ax = _axes(fig, colors)
    shares = _roi_shares(payload)
    if shares is None:
        draw_placeholder(ax, "No per-ROI distance shares available for this model.",
                         colors)
        return None
    dmodx, t2, reference, ids = shares
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    x = np.arange(1, dmodx.size + 1, dtype=float)
    low, high = float(np.min(reference)), float(np.max(reference))
    norm = Normalize(low - 1, high + 1) if low == high else Normalize(low, high)
    width = max(2.0, min(14.0, 320.0 / dmodx.size))
    ax.vlines(x, 0.0, dmodx * 100.0, colors=[colormaps["viridis"](norm(v)) for v in reference],
              linewidth=width, label="DModX > limit")
    if t2 is not None:
        ax.scatter(x, t2 * 100.0, marker="D", s=18, color=colors["fg"], zorder=3,
                   label="T² > limit")
    ax.axhline(_EXPECTED_ABOVE * 100.0, color=GUIDE_COLOR, linewidth=0.9, linestyle="--",
               label="expected 1 %")
    ax.set_xticks(x)
    ax.set_xticklabels(_roi_labels(payload, ids), rotation=60, ha="right", fontsize=7)
    ax.set_xlim(0.3, dmodx.size + 0.7)
    _legend(ax, colors, loc="upper right")
    _label(ax, "ROI, from the largest DModX share (bar color: target value)",
           "Pixels above the 0.99 limit (%)", colors)
    return [RoiPoints(ax, x, dmodx * 100.0, ids)]


def _table_roi_distance(payload: Any) -> PlotTable:
    shares = _roi_shares(payload)
    if shares is None:
        return PlotTable()
    dmodx, t2, _reference, ids = shares
    columns = [PlotColumn("roi_id", list(ids)),
               PlotColumn("rank", np.arange(1, dmodx.size + 1)),
               PlotColumn("dmodx_share_above_percent", dmodx * 100.0)]
    if t2 is not None:
        columns.append(PlotColumn("t2_share_above_percent", t2 * 100.0))
    return PlotTable(columns=tuple(columns),
                     metadata={"limits": "0.99 limits of the calibration pixels",
                               "expected_percent": "1",
                               "bar_colour": "target value of the ROI"})


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------

_SPECS = (
    PlotSpec("model.pred_vs_ref", "Predicted vs target value, per ROI", _render_pred_vs_ref,
             _table_pred_vs_ref, empty_text=_EMPTY),
    PlotSpec("model.residuals", "Residuals, per ROI", _render_residuals, _table_residuals,
             empty_text=_EMPTY),
    PlotSpec("model.target_distribution", "Target value distribution, per ROI",
             _render_target_distribution,
             _table_target_distribution, empty_text=_EMPTY),
    PlotSpec("model.scores", "Calibration scores (LV1 vs LV2), per ROI", _render_scores,
             _table_scores,
             empty_text=_EMPTY),
    _spectral_spec("coefficients", "model.coefficients", "Regression coefficients",
                   "Coefficient", "No regression coefficients yet."),
    PlotSpec("model.loadings", "Latent loadings (LV1-LV3)", _render_loadings, _table_loadings,
             empty_text=_EMPTY),
    _spectral_spec("coef_cs", "model.coef_cs", "Scaled coefficients (CoeffCS)", "CoeffCS",
                   "No scaled coefficients available for this model."),
    PlotSpec("model.vip", "VIP", _render_vip, _table_vip, empty_text=_EMPTY),
    PlotSpec("model.vip_sorted", "VIP, sorted", _render_vip_sorted, _table_vip_sorted,
             empty_text=_EMPTY),
    PlotSpec("model.selectivity_ratio", "Selectivity ratio", _render_selectivity,
             _table_selectivity, empty_text=_EMPTY),
    PlotSpec("model.components", "RMSECV (per ROI) and RMSEE (per ROI) vs components", _render_components,
             _table_components, empty_text=_EMPTY),
    PlotSpec("model.r2_q2", "R²Y and Q² vs components, per ROI", _render_r2_q2, _table_r2_q2,
             empty_text=_EMPTY),
    PlotSpec("model.per_component", "Each component: Q²ₐ, ΔR²X, roughness",
             _render_per_component, _table_per_component, empty_text=_EMPTY),
    PlotSpec("model.r2_pixels", "R²X and R²Y vs components, per pixel", _render_r2_pixels,
             _table_r2_pixels, empty_text=_EMPTY),
    _distance_spec("dmodx", "dcrit_099", "model.dmodx", "Distance to the model (DModX), per pixel",
                   "Normalized DModX"),
    _distance_spec("t2", "t2_crit_099", "model.t2", "Hotelling's T², per pixel", "T²"),
    PlotSpec("model.roi_distance", "Pixels above the limits, per ROI", _render_roi_distance,
             _table_roi_distance, empty_text=_EMPTY),
)

_PAGES = (
    PlotPage("Fit", (PlotSlot("model.pred_vs_ref", 0, 0), PlotSlot("model.residuals", 0, 1),
                     PlotSlot("model.target_distribution", 1, 0),
                     PlotSlot("model.scores", 1, 1))),
    PlotPage("Spectral", (PlotSlot("model.coefficients", 0, 0),
                          PlotSlot("model.coef_cs", 0, 1),
                          PlotSlot("model.loadings", 0, 2),
                          PlotSlot("model.vip", 1, 0),
                          PlotSlot("model.vip_sorted", 1, 1),
                          PlotSlot("model.selectivity_ratio", 1, 2))),
    PlotPage("Components", (PlotSlot("model.components", 0, 0),
                            PlotSlot("model.r2_q2", 0, 1),
                            PlotSlot("model.per_component", 1, 0),
                            PlotSlot("model.r2_pixels", 1, 1))),
    PlotPage("Distance", (PlotSlot("model.dmodx", 0, 0), PlotSlot("model.t2", 0, 1),
                          PlotSlot("model.roi_distance", 1, 0, column_span=2))),
)

_SPECS = tuple(replace(spec, help=MODEL_PLOT_HELP.get(spec.plot_id, "")) for spec in _SPECS)
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


__all__ = ["ModelDiagnosticsWidget", "roi_t2_ellipse"]
