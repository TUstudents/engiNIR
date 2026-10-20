"""Matplotlib figures for spectra, calibration results and prediction maps.

Every function draws into ``ax`` (or a new figure when ``ax`` is ``None``),
returns the :class:`~matplotlib.axes.Axes`, and passes extra keyword arguments
to the matplotlib call that draws the data (``plot``, ``scatter``, ``bar`` or
``imshow``), so colors, markers and line styles can be adjusted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from enginir._checks import check_integer, log_and_raise
from enginir._logging import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = get_logger(__name__)

_MAX_LEGEND_ENTRIES = 20
_MAX_POINT_LABELS = 30


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _axes(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize)[1]


def _numeric(value: Any, name: str, *, ndim: int | None = None, allow_nan: bool = False
             ) -> np.ndarray:
    """Return a non-empty float array, optionally of fixed dimensionality."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        log_and_raise("{} must be numeric", name, cause=exc, depth=2)
    if ndim is not None and array.ndim != ndim:
        log_and_raise("{} must be {}-D, got shape {}", name, ndim, array.shape, depth=2)
    if array.size == 0:
        log_and_raise("{} must not be empty", name, depth=2)
    finite = np.isfinite(array) | (np.isnan(array) if allow_nan else False)
    if not np.all(finite):
        log_and_raise("{} must contain only finite values", name, depth=2)
    return array


def _spectral_axis(wavelengths: Any, n_points: int, xlabel: str | None
                   ) -> tuple[np.ndarray, str]:
    """Return the x values and x label: wavelengths in nm, or band indices."""
    if wavelengths is None:
        return np.arange(n_points), xlabel or "Band index"
    axis = _numeric(wavelengths, "wavelengths", ndim=1)
    if axis.size != n_points:
        log_and_raise("wavelengths has {} values but the data have {} points", axis.size,
                      n_points, depth=2)
    return axis, xlabel or "Wavelength (nm)"


def _check_labels(labels: list[str] | None, n_samples: int) -> None:
    if labels is not None and len(labels) != n_samples:
        log_and_raise("labels has {} entries but there are {} samples", len(labels), n_samples,
                      depth=2)


def _target_vector(value: Any, name: str) -> np.ndarray:
    """Return a 1-D numeric vector; ``(n, 1)`` columns are accepted."""
    array = _numeric(value, name)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        log_and_raise("{} must be 1-D (plot one target at a time), got shape {}", name,
                      array.shape, depth=2)
    return array


# ---------------------------------------------------------------------------
# Spectra and model vectors
# ---------------------------------------------------------------------------


def plot_spectra(
    X: np.ndarray,
    wavelengths: np.ndarray | None = None,
    labels: list[str] | None = None,
    color_by: np.ndarray | None = None,
    title: str = "Spectra",
    ax: Axes | None = None,
    *,
    xlabel: str | None = None,
    ylabel: str = "Intensity",
    colorbar_label: str = "",
    cmap: str = "viridis",
    **kwargs: Any,
) -> Axes:
    """Plot one line per spectrum.

    Parameters
    ----------
    X:
        Spectra, shape ``(n_samples, n_wavelengths)``.
    wavelengths:
        Wavelengths in nm; band indices are used if ``None``.
    labels:
        Sample names. A legend is drawn when there are at most 20 samples.
    color_by:
        Optional value per sample (e.g. moisture content); spectra are colored
        by it on the ``cmap`` scale with a colorbar, and ``labels`` and a
        ``color`` keyword are ignored.
    title:
        Axes title.
    ax:
        Axes to draw on; a new figure is created if ``None``.
    xlabel, ylabel:
        Axis labels. ``xlabel`` defaults to ``"Wavelength (nm)"`` or
        ``"Band index"``; set ``ylabel`` to the signal (e.g. ``"Absorbance"``).
    colorbar_label:
        Label of the ``color_by`` colorbar.
    cmap:
        Matplotlib colormap name for ``color_by``.
    **kwargs:
        Passed to :meth:`~matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    X = _numeric(X, "X", ndim=2)
    x_axis, xlabel = _spectral_axis(wavelengths, X.shape[1], xlabel)
    ax = _axes(ax, (9, 4))
    style = {"linewidth": 0.8, "alpha": 0.8, **kwargs}
    if color_by is not None:
        values = _target_vector(color_by, "color_by")
        if values.size != X.shape[0]:
            log_and_raise("color_by has {} values but X has {} samples", values.size, X.shape[0])
        low, high = values.min(), values.max()
        norm = Normalize(low - 1, high + 1) if low == high else Normalize(low, high)
        # One collection for all spectra: colored by value and the colorbar's source.
        style.pop("color", None)
        segments = list(np.stack(np.broadcast_arrays(x_axis, X), axis=-1))  # views
        lines = LineCollection(segments, cmap=cmap,
                               norm=norm, **{"linewidths": style.pop("linewidth"), **style})
        lines.set_array(values)
        ax.add_collection(lines)
        ax.autoscale_view()
        ax.figure.colorbar(lines, ax=ax, label=colorbar_label)
    else:
        _check_labels(labels, X.shape[0])
        for index, spectrum in enumerate(X):
            line_style = style if labels is None else {**style, "label": labels[index]}
            ax.plot(x_axis, spectrum, **line_style)
        if labels is not None and len(labels) <= _MAX_LEGEND_ENTRIES:
            ax.legend(fontsize=7, loc="upper right")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return ax


def plot_loadings(
    loadings: np.ndarray,
    wavelengths: np.ndarray | None = None,
    components: list[int] | None = None,
    title: str = "Loadings",
    ax: Axes | None = None,
    *,
    component_label: str = "PC",
    xlabel: str | None = None,
    ylabel: str = "Loading",
    **kwargs: Any,
) -> Axes:
    """Plot loading (or weight, or coefficient) vectors over the spectral axis.

    Parameters
    ----------
    loadings:
        Loadings, shape ``(n_wavelengths, n_components)``, or one vector of
        shape ``(n_wavelengths,)`` such as the PLS1 ``coef_``.
    wavelengths:
        Wavelengths in nm; band indices are used if ``None``.
    components:
        0-based components to plot. Default: the first three.
    title:
        Axes title.
    ax:
        Axes to draw on; a new figure is created if ``None``.
    component_label:
        Legend prefix, ``"PC"`` (default) for PCA or ``"LV"`` for PLS.
    xlabel:
        x-axis label (default from ``wavelengths``).
    ylabel:
        y-axis label, e.g. ``"Regression coefficient"``.
    **kwargs:
        Passed to :meth:`~matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    loadings = _numeric(loadings, "loadings")
    if loadings.ndim == 1:
        loadings = loadings[:, np.newaxis]
    if loadings.ndim != 2:
        log_and_raise("loadings must be 1-D or 2-D, got shape {}", loadings.shape)
    n_components = loadings.shape[1]
    x_axis, xlabel = _spectral_axis(wavelengths, loadings.shape[0], xlabel)
    if components is None:
        components = list(range(min(3, n_components)))
    components = [check_integer(c, "component", minimum=0) for c in components]
    if not components or len(set(components)) != len(components) or max(components) >= n_components:
        log_and_raise("components must be distinct and below {}, got {}", n_components,
                      components)
    ax = _axes(ax, (9, 4))
    for component in components:
        ax.plot(x_axis, loadings[:, component],
                **{"linewidth": 1.0, **kwargs, "label": f"{component_label} {component + 1}"})
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize=8)
    return ax


def plot_vip(
    vip_scores: np.ndarray,
    wavelengths: np.ndarray | None = None,
    threshold: float = 1.0,
    title: str = "VIP Scores",
    ax: Axes | None = None,
    *,
    xlabel: str | None = None,
    **kwargs: Any,
) -> Axes:
    """Bar chart of VIP scores; bars at or above ``threshold`` are highlighted.

    Parameters
    ----------
    vip_scores:
        VIP score per wavelength, e.g. from
        :meth:`enginir.models.PLSRegression.vip_scores`.
    wavelengths:
        Wavelengths in nm; band indices are used if ``None``.
    threshold:
        Importance threshold drawn as a dashed line. Default ``1.0``.
    title:
        Axes title.
    ax:
        Axes to draw on; a new figure is created if ``None``.
    xlabel:
        x-axis label (default from ``wavelengths``).
    **kwargs:
        Passed to :meth:`~matplotlib.axes.Axes.bar`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    vip_scores = _numeric(vip_scores, "vip_scores", ndim=1)
    x_axis, xlabel = _spectral_axis(wavelengths, vip_scores.size, xlabel)
    spacing = float(np.median(np.abs(np.diff(x_axis)))) if x_axis.size > 1 else 1.0
    colors = np.where(vip_scores >= threshold, "steelblue", "lightsteelblue")
    ax = _axes(ax, (9, 4))
    ax.bar(x_axis, vip_scores, **{"color": colors, "width": 0.9 * spacing, **kwargs})
    ax.axhline(threshold, color="red", linewidth=1.0, linestyle="--",
               label=f"threshold = {threshold:g}")
    ax.set(xlabel=xlabel, ylabel="VIP score", title=title)
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def plot_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    units: str = "",
    title: str = "Reference vs Predicted",
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Scatter predicted against reference values with a dashed 1:1 line and R².

    Parameters
    ----------
    y_true:
        Reference values, shape ``(n_samples,)``.
    y_pred:
        Predictions, same shape. For several targets, call once per target.
    units:
        Unit appended to the axis labels, e.g. ``"%"``.
    title:
        Axes title.
    ax:
        Axes to draw on; a new figure is created if ``None``.
    **kwargs:
        Passed to :meth:`~matplotlib.axes.Axes.scatter`.

    Returns
    -------
    matplotlib.axes.Axes
        R² is computed with :func:`enginir.validation.r2_score`.
    """
    from enginir.validation import r2_score

    y_true = _target_vector(y_true, "y_true")
    y_pred = _target_vector(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        log_and_raise("y_true has {} values but y_pred has {}", y_true.size, y_pred.size)
    ax = _axes(ax, (5, 5))
    ax.scatter(y_true, y_pred, **{"s": 30, "alpha": 0.7, "edgecolors": "none", **kwargs})
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    pad = 0.05 * (high - low)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="black", linewidth=1.0,
            linestyle="--")
    ax.annotate(f"$R^2$ = {r2_score(y_true, y_pred):.4f}", xy=(0.05, 0.92),
                xycoords="axes fraction", fontsize=9)
    unit_suffix = f" ({units})" if units else ""
    ax.set(xlabel=f"Reference{unit_suffix}", ylabel=f"Predicted{unit_suffix}", title=title)
    return ax


def plot_scores(
    scores: np.ndarray,
    y: np.ndarray | None = None,
    labels: list[str] | None = None,
    pc_x: int = 0,
    pc_y: int = 1,
    title: str = "Score Plot",
    ax: Axes | None = None,
    *,
    component_label: str = "PC",
    colorbar_label: str = "",
    **kwargs: Any,
) -> Axes:
    """Scatter two score columns against each other.

    Parameters
    ----------
    scores:
        Scores, shape ``(n_samples, n_components)``.
    y:
        Optional value per sample. Numeric values color the points on a
        viridis scale with a colorbar; class labels (strings or booleans)
        give one color per class and a legend.
    labels:
        Sample names, written next to the points when there are at most 30.
    pc_x, pc_y:
        Different 0-based components for the x and y axes.
    title:
        Axes title.
    ax:
        Axes to draw on; a new figure is created if ``None``.
    component_label:
        Axis label prefix, ``"PC"`` (default) for PCA or ``"LV"`` for PLS.
    colorbar_label:
        Label of the colorbar for numeric ``y``.
    **kwargs:
        Passed to :meth:`~matplotlib.axes.Axes.scatter`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    scores = _numeric(scores, "scores", ndim=2)
    n_samples, n_components = scores.shape
    pc_x = check_integer(pc_x, "pc_x", minimum=0)
    pc_y = check_integer(pc_y, "pc_y", minimum=0)
    if pc_x == pc_y or max(pc_x, pc_y) >= n_components:
        log_and_raise("pc_x and pc_y must differ and be below {}, got {} and {}", n_components,
                      pc_x, pc_y)
    _check_labels(labels, n_samples)
    tx, ty = scores[:, pc_x], scores[:, pc_y]
    ax = _axes(ax, (6, 5))
    style = {"s": 40, "edgecolors": "none", "alpha": 0.85, **kwargs}
    if y is None:
        ax.scatter(tx, ty, **style)
    else:
        classes = np.asarray(y)
        if classes.ndim == 2 and classes.shape[1] == 1:
            classes = classes[:, 0]
        if classes.shape != (n_samples,):
            log_and_raise("y must have shape ({},), got {}", n_samples, classes.shape)
        if classes.dtype.kind in "iuf":
            points = ax.scatter(tx, ty, **{"cmap": "viridis", **style,
                                           "c": _numeric(classes, "y")})
            ax.figure.colorbar(points, ax=ax, label=colorbar_label)
        else:
            names = classes.astype(str)  # also handles None and mixed labels
            for name in np.unique(names):
                members = names == name
                ax.scatter(tx[members], ty[members], **{**style, "label": name})
            ax.legend(fontsize=8)
    if labels is not None and n_samples <= _MAX_POINT_LABELS:
        for name, x, y_value in zip(labels, tx, ty):
            ax.annotate(name, (x, y_value), fontsize=6, textcoords="offset points",
                        xytext=(3, 3))
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set(xlabel=f"{component_label} {pc_x + 1}", ylabel=f"{component_label} {pc_y + 1}",
           title=title)
    return ax


def plot_leverage(
    leverage: np.ndarray,
    q_residuals: np.ndarray,
    leverage_crit: float | None = None,
    q_crit: float | None = None,
    labels: list[str] | None = None,
    title: str = "Leverage vs Q-Residuals",
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Influence plot: Q residual against leverage, outliers in red.

    Inputs match :func:`enginir.validation.leverage_outliers` and
    :func:`enginir.validation.spectral_outliers`.

    Parameters
    ----------
    leverage:
        Leverage of every sample, shape ``(n_samples,)``.
    q_residuals:
        Q residual of every sample, same shape.
    leverage_crit, q_crit:
        Optional limits, drawn as dashed lines; samples beyond either are
        outliers.
    labels:
        Sample names, written next to the outliers (next to all samples when
        no limit is given).
    title:
        Axes title.
    ax:
        Axes to draw on; a new figure is created if ``None``.
    **kwargs:
        Passed to :meth:`~matplotlib.axes.Axes.scatter`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    leverage = _numeric(leverage, "leverage", ndim=1)
    q_residuals = _numeric(q_residuals, "q_residuals", ndim=1)
    if leverage.shape != q_residuals.shape:
        log_and_raise("leverage has {} values but q_residuals has {}", leverage.size,
                      q_residuals.size)
    _check_labels(labels, leverage.size)
    has_limits = leverage_crit is not None or q_crit is not None
    is_outlier = np.zeros(leverage.size, dtype=bool)
    if leverage_crit is not None:
        is_outlier |= leverage > leverage_crit
    if q_crit is not None:
        is_outlier |= q_residuals > q_crit

    ax = _axes(ax, (6, 5))
    colors = np.where(is_outlier, "tomato", "steelblue")
    ax.scatter(leverage, q_residuals,
               **{"c": colors, "s": 40, "edgecolors": "none", "alpha": 0.85, **kwargs})
    if leverage_crit is not None:
        ax.axvline(leverage_crit, color="black", linewidth=1.0, linestyle="--",
                   label=f"leverage limit = {leverage_crit:.3g}")
    if q_crit is not None:
        ax.axhline(q_crit, color="dimgrey", linewidth=1.0, linestyle="--",
                   label=f"Q limit = {q_crit:.3g}")
    if labels is not None:
        for name, x, y, annotate in zip(labels, leverage, q_residuals,
                                        is_outlier if has_limits else np.ones_like(is_outlier)):
            if annotate:
                ax.annotate(name, (x, y), fontsize=6, textcoords="offset points", xytext=(3, 3))
    if has_limits:
        ax.legend(fontsize=8)
    ax.set(xlabel="Leverage", ylabel="Q residual", title=title)
    return ax


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def plot_image_map(
    y_map: np.ndarray,
    background_rgb: np.ndarray | None = None,
    cmap: str = "viridis",
    title: str = "Predicted Property Map",
    ax: Axes | None = None,
    *,
    colorbar_label: str = "",
    **kwargs: Any,
) -> Axes:
    """Show a map of predicted values, optionally over an RGB image.

    Parameters
    ----------
    y_map:
        Predicted values, shape ``(rows, cols)``. ``NaN`` marks pixels without
        a prediction (e.g. background) and is drawn transparent.
    background_rgb:
        Optional RGB composite of the same scene, shape ``(rows, cols, 3)``,
        drawn underneath at reduced opacity. Values outside ``[0, 1]`` are
        rescaled to that range.
    cmap:
        Colormap of the predictions.
    title:
        Axes title.
    ax:
        Axes to draw on; a new figure is created if ``None``.
    colorbar_label:
        Label of the colorbar (e.g. the predicted property and unit).
    **kwargs:
        Passed to :meth:`~matplotlib.axes.Axes.imshow` for ``y_map``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    y_map = _numeric(y_map, "y_map", ndim=2, allow_nan=True)
    if np.all(np.isnan(y_map)):
        log_and_raise("y_map contains no finite values")
    ax = _axes(ax, (6, 5))
    if background_rgb is not None:
        rgb = _numeric(background_rgb, "background_rgb", ndim=3)
        if rgb.shape != (*y_map.shape, 3):
            log_and_raise("background_rgb must have shape {}, got {}", (*y_map.shape, 3),
                          rgb.shape)
        if rgb.min() < 0.0 or rgb.max() > 1.0:
            span = float(np.ptp(rgb))
            if span == 0.0:
                logger.warning("background_rgb is constant; drawing it black")
                rgb = np.zeros_like(rgb)
            else:
                rgb = (rgb - rgb.min()) / span
        ax.imshow(rgb, origin="upper", alpha=0.4)
    image = ax.imshow(y_map, **{"cmap": cmap, "origin": "upper",
                                "alpha": 0.85 if background_rgb is not None else 1.0, **kwargs})
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    ax.set(xlabel="Column", ylabel="Row", title=title)
    return ax


__all__ = [
    "plot_image_map",
    "plot_leverage",
    "plot_loadings",
    "plot_prediction",
    "plot_scores",
    "plot_spectra",
    "plot_vip",
]
