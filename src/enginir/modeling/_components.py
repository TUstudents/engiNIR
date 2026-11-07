"""Component curve and training diagnostics (sample-set-model-run contract §5, §7.4, §7.5).

The component curve shows, for 1 … A_max components, how cross-validated
prediction (RMSECV, Q²) and fit (R²Y, R²X) change; the rules of
:func:`enginir.validation.choose_components` pick a number from it. The
training diagnostics project every drawn pixel onto the final model: T², the
distance to the model (DModX) and their limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import f
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline

from .._checks import log_and_raise
from .._logging import get_logger
from ..models import PCR, PLSRegression
from ..validation import choose_components, cross_validate
from ..validation.components import COMPONENT_CHOICE_RULES
from ..validation.outliers import dmodx_limit

logger = get_logger(__name__)

_EPS = 1e-10
_CHUNK_ROWS = 20_000


def component_predictions(fitted: Any, X: np.ndarray) -> np.ndarray:
    """Predictions with 1 … *A* components of a fitted model (or pipeline ending in one)."""
    if isinstance(fitted, Pipeline):
        final: Any = fitted[-1]
        return final.predict_components(fitted[:-1].transform(X))
    return fitted.predict_components(X)


class AllComponents(RegressorMixin, BaseEstimator):
    """A model whose ``predict`` returns the predictions for every number of components.

    Used to cross-validate the component curve in one pass: each fold fits
    A_max components once.
    """

    def __init__(self, estimator: Any = None) -> None:
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> AllComponents:
        model: Any = clone(self.estimator)
        self.estimator_ = model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return component_predictions(self.estimator_, X)


@dataclass
class Projection:
    """Drawn pixels projected onto a fitted model (all in the model's scaled units)."""

    scores: np.ndarray            # (N, A)
    residual_ss: np.ndarray       # (N,) row sums of squared residuals after A components
    cumulative_rss: np.ndarray    # (A,) total squared residual after 1 … A components
    total_ss: float               # total sum of squares of the scaled, centred X
    n_variables: int              # K, bands seen by the model


def _model_space(model: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    """(scale-and-centre function, rotations R, loadings P) of a PLS or PCR model."""
    if isinstance(model, PLSRegression):
        mean, std = np.asarray(model.x_mean_), np.asarray(model.x_std_)
        return ((lambda X: (np.asarray(X, dtype=np.float64) - mean) / std),
                np.asarray(model.x_rotations_, dtype=np.float64),
                np.asarray(model.x_loadings_, dtype=np.float64))
    if isinstance(model, PCR):
        components = np.asarray(model.pca_.components_, dtype=np.float64)
        pca_mean = np.asarray(model.pca_.mean_, dtype=np.float64)
        return ((lambda X: np.asarray(model._scaled(X), dtype=np.float64) - pca_mean),
                components.T, components.T)
    log_and_raise("Component diagnostics support PLS and PCR models, got {}",
                  type(model).__name__)


def project(fitted: Any, X: np.ndarray) -> Projection:
    """Project every row of ``X`` onto a fitted model, chunk by chunk (no full float64 copy)."""
    steps = fitted[:-1] if isinstance(fitted, Pipeline) else None
    model = fitted[-1] if isinstance(fitted, Pipeline) else fitted
    centre, rotations, loadings = _model_space(model)
    n_components = rotations.shape[1]
    scores = np.empty((X.shape[0], n_components))
    residual_ss = np.empty(X.shape[0])
    cumulative = np.zeros(n_components)
    total = 0.0
    n_variables = 0
    for start in range(0, X.shape[0], _CHUNK_ROWS):
        block = X[start:start + _CHUNK_ROWS]
        scaled = centre(steps.transform(block) if steps is not None else block)
        n_variables = scaled.shape[1]
        t = scaled @ rotations
        scores[start:start + len(block)] = t
        total += float(np.einsum("ij,ij->", scaled, scaled))
        for a in range(n_components):  # deflate in place: scaled becomes the residual
            scaled -= np.outer(t[:, a], loadings[:, a])
            cumulative[a] += float(np.einsum("ij,ij->", scaled, scaled))
        residual_ss[start:start + len(block)] = np.einsum("ij,ij->i", scaled, scaled)
    return Projection(scores, residual_ss, cumulative, total, n_variables)


def _roi_means(values: np.ndarray, roi_index: np.ndarray, n_rois: int) -> np.ndarray:
    """Mean of ``values`` (N,) or (N, C) per ROI (``roi_index`` in 0 … n_rois - 1)."""
    counts = np.bincount(roi_index, minlength=n_rois).astype(np.float64)
    if values.ndim == 1:
        return np.bincount(roi_index, weights=values, minlength=n_rois) / counts
    return np.stack([np.bincount(roi_index, weights=values[:, c], minlength=n_rois) / counts
                     for c in range(values.shape[1])], axis=1)


def rank_limit(n_rows: list[int], n_variables: int) -> int:
    """The largest number of components every training set can fit: min(n - 1, K)."""
    return max(1, min(min(n - 1 for n in n_rows), n_variables))


def default_max(n_given: int, n_groups: int, limit: int) -> int:
    """A_max of the curve: max(n, min(15, max(1, G - 2), M)) (contract §7)."""
    return max(n_given, min(15, max(1, n_groups - 2), limit))


def _ratio(numerator: np.ndarray, denominator: np.ndarray, what: str) -> np.ndarray:
    """``numerator / denominator``, NaN (with a warning) where the denominator is ~0."""
    small = np.abs(denominator) < _EPS
    if np.any(small):
        logger.warning("Component curve: {} is (almost) zero for {} component count(s); the "
                       "value is NaN there", what, int(small.sum()))
    out = np.full(np.shape(numerator), np.nan)
    np.divide(numerator, denominator, out=out, where=~small)
    return out


def component_curve(model: Any, X: np.ndarray, y: np.ndarray, *, splitter: Any,
                    cv_groups: np.ndarray, roi_index: np.ndarray, n_samples: int
                    ) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    """The component curve of ``model`` (unfitted, with A_max components; contract §7.4).

    Returns the curve (arrays over *a* = 1 … A_max and every rule's
    suggestion), the out-of-fold pixel predictions (N, A_max) and the
    cross-validation scheme.
    """
    cv = cross_validate(AllComponents(model), X, y, cv=splitter, groups=cv_groups, scoring=[])
    oof = np.asarray(cv["y_pred"], dtype=np.float64)
    if oof.ndim != 2:
        log_and_raise("The component curve supports one target, got predictions of shape {}",
                      oof.shape)
    n_rois = int(roi_index.max()) + 1
    y_roi = _roi_means(np.asarray(y, dtype=np.float64), roi_index, n_rois)
    ss = float(((y_roi - y_roi.mean()) ** 2).sum())
    errors = _roi_means(oof, roi_index, n_rois) - y_roi[:, np.newaxis]   # (R, A)
    press = (errors ** 2).sum(axis=0)
    rmsecv = np.sqrt(press / n_rois)
    spread = np.std(errors ** 2, axis=0, ddof=1) if n_rois > 1 else np.zeros_like(press)
    se = np.zeros_like(rmsecv)
    ok = rmsecv >= _EPS
    se[ok] = spread[ok] / (2.0 * rmsecv[ok] * np.sqrt(n_rois))

    full: Any = clone(model)
    full.fit(X, y)
    fitted = component_predictions(full, X)
    rss_roi = ((_roi_means(fitted, roi_index, n_rois) - y_roi[:, np.newaxis]) ** 2).sum(axis=0)
    y_pixels = np.asarray(y, dtype=np.float64)
    ss_pixels = float(((y_pixels - y_pixels.mean()) ** 2).sum())
    rss_pixels = ((fitted - y_pixels[:, np.newaxis]) ** 2).sum(axis=0)
    previous = np.concatenate([[ss], rss_roi[:-1]])
    projection = project(full, X)
    curve: dict[str, Any] = {
        "n_components": np.arange(1, oof.shape[1] + 1, dtype=np.int64),
        "rmsecv": rmsecv,
        "rmsecv_se": se,
        "q2": 1.0 - _ratio(press, previous, "RSS of the previous component count"),
        "q2_cum": 1.0 - _ratio(press, np.full_like(press, ss), "the reference spread"),
        "r2y_cum": 1.0 - _ratio(rss_pixels, np.full_like(rss_pixels, ss_pixels),
                                "the pixel reference spread"),
        "r2y_cum_roi": 1.0 - _ratio(rss_roi, np.full_like(rss_roi, ss), "the reference spread"),
        "r2x_cum": 1.0 - _ratio(projection.cumulative_rss,
                                np.full_like(projection.cumulative_rss, projection.total_ss),
                                "the spectral spread"),
    }
    for rule in COMPONENT_CHOICE_RULES:
        curve[f"suggested_{rule}"] = choose_components(curve, rule, n_samples=n_samples)
    return curve, oof, cv["cv_scheme"]


def training_diagnostics(fitted: Any, X: np.ndarray, y: np.ndarray, kept: np.ndarray,
                         roi_index: np.ndarray, *, y_divisor: float) -> tuple[
                             dict[str, Any], dict[str, Any]]:
    """Fit statistics (metrics) and per-pixel diagnostics of the final model (§5, §7.5).

    ``kept`` marks the pixels the model was fitted on (all, unless DModX
    excluded some); excluded pixels get the same formulas.
    """
    model: Any = fitted[-1] if isinstance(fitted, Pipeline) else fitted
    projection = project(fitted, X)
    n = int(kept.sum())
    a = projection.scores.shape[1]
    k = projection.n_variables
    variances = np.var(projection.scores[kept], axis=0, ddof=1)
    variances = np.where(variances < _EPS, np.nan, variances)
    t2 = (projection.scores ** 2 / variances).sum(axis=1)
    diagnostics: dict[str, Any] = {"t2": t2.astype(np.float32)}

    def t2_crit(components: int, level: float) -> float:
        return float(components * (n - 1) / (n - components) * f.ppf(level, components,
                                                                      n - components))

    if n - a >= 1:
        diagnostics["t2_crit_095"] = t2_crit(a, 0.95)
        diagnostics["t2_crit_099"] = t2_crit(a, 0.99)
    if a >= 2 and n - 2 >= 1:
        diagnostics["t2_crit_2_095"] = t2_crit(2, 0.95)
        diagnostics["t2_crit_2_099"] = t2_crit(2, 0.99)
    if k - a >= 1 and n - a - 1 >= 1:
        s0 = float(np.sqrt(projection.residual_ss[kept].sum() / ((n - a - 1) * (k - a))))
        v = np.sqrt(n / (n - a - 1))
        distance = (np.sqrt(projection.residual_ss / (k - a)) * v / s0 if s0 >= _EPS
                    else np.zeros(X.shape[0]))
        diagnostics["dmodx"] = distance.astype(np.float32)
        diagnostics["dcrit_099"] = dmodx_limit(n, k, a, 0.99)

    n_rois = int(roi_index.max()) + 1
    counts = np.bincount(roi_index, minlength=n_rois)
    if "dmodx" in diagnostics:
        above = diagnostics["dmodx"] > diagnostics["dcrit_099"]
        diagnostics["roi_share_above_dcrit"] = (
            np.bincount(roi_index, weights=above, minlength=n_rois) / counts).astype(np.float32)
    if "t2_crit_099" in diagnostics:
        above = t2 > diagnostics["t2_crit_099"]
        diagnostics["roi_share_above_t2crit"] = (
            np.bincount(roi_index, weights=above, minlength=n_rois) / counts).astype(np.float32)
    coef = np.asarray(model.coef_, dtype=np.float64)
    diagnostics["coef_cs"] = (coef * np.asarray(model.x_std_) / y_divisor).astype(np.float32)

    fitted_values = np.asarray(fitted.predict(X), dtype=np.float64).reshape(X.shape[0])
    y = np.asarray(y, dtype=np.float64)
    metrics: dict[str, Any] = {}
    rss = float(((fitted_values[kept] - y[kept]) ** 2).sum())
    tss = float(((y[kept] - y[kept].mean()) ** 2).sum())
    if n - 1 - a >= 1:
        metrics["pixel_rmsee"] = float(np.sqrt(rss / (n - 1 - a)))
        if tss >= _EPS:
            metrics["pixel_r2_adj"] = float(1 - (rss / (n - 1 - a)) / (tss / (n - 1)))
    kept_index = roi_index[kept]
    fitted_roi = _roi_means(fitted_values[kept], kept_index, n_rois)
    y_roi = _roi_means(y[kept], kept_index, n_rois)
    if n_rois - 1 - a >= 1:
        rss_roi = float(((fitted_roi - y_roi) ** 2).sum())
        tss_roi = float(((y_roi - y_roi.mean()) ** 2).sum())
        metrics["rmsee"] = float(np.sqrt(rss_roi / (n_rois - 1 - a)))
        if tss_roi >= _EPS:
            metrics["r2_adj"] = float(1 - (rss_roi / (n_rois - 1 - a)) / (tss_roi / (n_rois - 1)))
    return metrics, diagnostics


__all__ = ["AllComponents", "Projection", "component_curve", "component_predictions",
           "default_max", "project", "rank_limit", "training_diagnostics"]
