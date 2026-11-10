"""Prediction intervals by the leverage formula (prediction-runtime contract, Prediction Intervals).

An interval is ŷ ± q · s · √(1 + h) (ASTM E1655, ISO 12099), with s the
model's RMSEE, q the two-sided Student t quantile with R − A − 1 degrees of
freedom and h the leverage. A pixel's leverage comes from its T²PS with the
ROIs as the independent samples, h = 1/R + T²PS / (R − 1); a ROI mean's from its mean scores against the
training ROI scores, h̄ = 1/R′ + d² / (R′ − 1). Outside the model (above
the 0.99 limits) there is no interval (NaN). Nothing here is stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import f, t

from .._checks import log_and_raise
from .._logging import get_logger

logger = get_logger(__name__)

DEFAULT_LEVEL = 0.95


@dataclass(frozen=True)
class RoiScoreStatistics:
    """Mean, inverse covariance and ROI-level T² limit of the training ROI scores."""

    mean: np.ndarray
    cov_inv: np.ndarray
    n_rois: int
    t2_crit: float


@dataclass(frozen=True)
class IntervalInputs:
    """What a model run stores for its prediction intervals (``None`` parts: no such interval).

    ``n_rois`` is the metric R (degrees of freedom R − A − 1); ``roi_scores``
    the statistics of the training ROI scores for mean intervals.
    """

    n_components: int
    n_train: int
    n_rois: int
    pixel_rmsee: float | None
    rmsee: float | None
    t2_crit: float | None
    dcrit: float | None
    roi_scores: RoiScoreStatistics | None = None

    def quantile(self, level: float) -> float | None:
        """t(1 − (1 − level)/2; R − A − 1), or ``None`` when R − A − 1 < 1."""
        check_level(level)
        dof = self.n_rois - self.n_components - 1
        if dof < 1:
            return None
        return float(t.ppf(1.0 - (1.0 - level) / 2.0, dof))

    @property
    def has_pixel_interval(self) -> bool:
        return self.pixel_rmsee is not None and self.n_rois - self.n_components - 1 >= 1


def check_level(level: float) -> None:
    if not 0.0 < float(level) < 1.0:
        log_and_raise("The interval level must be between 0 and 1, got {}", level, depth=2)


def _number(values: dict[str, Any], key: str) -> float | None:
    value = values.get(key)
    if value is None:
        return None
    value = float(np.asarray(value))
    return value if np.isfinite(value) else None


def roi_score_statistics(roi_scores: np.ndarray | None,
                         n_components: int) -> RoiScoreStatistics | None:
    """Statistics of the stored ``roi_scores`` (rows without a fitted pixel are NaN, skipped)."""
    if roi_scores is None:
        return None
    scores = np.asarray(roi_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != n_components:
        return None
    scores = scores[np.isfinite(scores).all(axis=1)]
    r, a = scores.shape
    if r - a < 1 or r < 2:
        return None
    cov = np.cov(scores, rowvar=False, ddof=1).reshape(a, a)
    if np.linalg.matrix_rank(cov) < a:
        logger.warning("The training ROI scores have rank below {} components; mean intervals "
                       "use the pseudo-inverse of their covariance", a)
    return RoiScoreStatistics(
        mean=scores.mean(axis=0), cov_inv=np.linalg.pinv(cov), n_rois=r,
        t2_crit=float(a * (r - 1) / (r - a) * f.ppf(0.99, a, r - a)))


def interval_inputs(metrics: dict[str, Any] | None, model_blob: dict[str, Any] | None,
                    roi_scores: np.ndarray | None = None) -> IntervalInputs | None:
    """The interval inputs of a model run, or ``None`` when it has none (contract)."""
    metrics, blob = metrics or {}, model_blob or {}
    n_components, n_train, n_rois = (blob.get("n_components"), blob.get("n_train"),
                                     metrics.get("n_rois"))
    if n_components is None or n_train is None or n_rois is None:
        return None
    if len(blob.get("target_names") or [None]) != 1:
        return None  # intervals are for single-target regression runs
    a = int(np.asarray(n_components))
    inputs = IntervalInputs(
        n_components=a, n_train=int(np.asarray(n_train)), n_rois=int(n_rois),
        pixel_rmsee=_number(metrics, "pixel_rmsee"), rmsee=_number(metrics, "rmsee"),
        t2_crit=_number(blob, "t2_crit_099"), dcrit=_number(blob, "dcrit_099"),
        roi_scores=roi_score_statistics(roi_scores, a))
    if inputs.pixel_rmsee is None and inputs.rmsee is None:
        return None
    return inputs


def pixel_half_width(t2: np.ndarray, dmodx: np.ndarray, inputs: IntervalInputs | None, *,
                     level: float = DEFAULT_LEVEL,
                     y: np.ndarray | None = None) -> np.ndarray | None:
    """Half-width q · s · √(1 + h) of each pixel's interval, float32; NaN outside the model.

    h = 1/R + T²PS / (R − 1): the leverage of the standards with the R
    training ROIs, not the N pixels, as the independent samples (the
    reference values come from the ROIs). NaN where T²PS (or ``y``) is not finite, T²PS >
    T²Crit or DModXPS > DCrit. ``None`` when the run has no pixel interval.
    """
    check_level(level)
    if inputs is None or inputs.pixel_rmsee is None:
        return None
    q = inputs.quantile(level)
    if q is None:
        return None
    t2 = np.asarray(t2, dtype=np.float64)
    leverage = 1.0 / inputs.n_rois + t2 / (inputs.n_rois - 1)
    with np.errstate(invalid="ignore"):
        half = q * inputs.pixel_rmsee * np.sqrt(1.0 + leverage)
        outside = ~np.isfinite(t2)
        if inputs.t2_crit is not None:
            outside |= t2 > inputs.t2_crit
        if inputs.dcrit is not None:
            outside |= np.asarray(dmodx, dtype=np.float64) > inputs.dcrit
        if y is not None:
            outside |= ~np.isfinite(np.asarray(y, dtype=np.float64))
    return np.where(outside, np.nan, half).astype(np.float32)


def mean_half_width(mean_scores: np.ndarray | None, inputs: IntervalInputs | None, *,
                    level: float = DEFAULT_LEVEL) -> float:
    """Half-width of a ROI mean's interval; NaN without one (contract, Prediction Intervals)."""
    check_level(level)
    if inputs is None or inputs.rmsee is None or inputs.roi_scores is None \
            or mean_scores is None:
        return float("nan")
    q = inputs.quantile(level)
    scores = np.asarray(mean_scores, dtype=np.float64).reshape(-1)
    if q is None or not np.isfinite(scores).all():
        return float("nan")
    stats = inputs.roi_scores
    centred = scores - stats.mean
    d2 = float(centred @ stats.cov_inv @ centred)
    if d2 > stats.t2_crit:
        return float("nan")
    leverage = 1.0 / stats.n_rois + d2 / (stats.n_rois - 1)
    return float(q * inputs.rmsee * np.sqrt(1.0 + leverage))


def interval_coverage(inputs: IntervalInputs, *, pixel_y_true: np.ndarray,
                      pixel_y_cv: np.ndarray, pixel_t2: np.ndarray | None,
                      pixel_dmodx: np.ndarray | None, roi_y_true: np.ndarray,
                      roi_y_cv: np.ndarray, roi_scores: np.ndarray | None,
                      level: float = DEFAULT_LEVEL) -> dict[str, Any]:
    """Share of held-out predictions inside their interval (model-run contract §5).

    Each held-out prediction gets the interval around itself with the
    leverage of its observation in the final model; observations without an
    interval are left out. Returns the coverage metrics that have at least one
    interval.
    """
    metrics: dict[str, Any] = {}

    def record(name: str, error: np.ndarray, half: np.ndarray) -> None:
        counted = np.isfinite(error) & np.isfinite(half)
        if counted.any():
            metrics[name] = float(np.mean(np.abs(error[counted]) <= half[counted]))
            metrics[f"{name}_n"] = int(counted.sum())

    if pixel_t2 is not None:
        dmodx = (np.full(np.shape(pixel_t2), np.nan) if pixel_dmodx is None
                 else np.asarray(pixel_dmodx))
        half = pixel_half_width(pixel_t2, dmodx, inputs, level=level, y=pixel_y_cv)
        if half is not None:
            record("interval_coverage_pixel",
                   np.asarray(pixel_y_cv, np.float64) - np.asarray(pixel_y_true, np.float64),
                   half.astype(np.float64))
    if roi_scores is not None and inputs.roi_scores is not None:
        half = np.array([mean_half_width(row, inputs, level=level)
                         for row in np.asarray(roi_scores, dtype=np.float64)])
        record("interval_coverage_roi",
               np.asarray(roi_y_cv, np.float64) - np.asarray(roi_y_true, np.float64), half)
    return metrics


__all__ = ["DEFAULT_LEVEL", "IntervalInputs", "RoiScoreStatistics", "check_level",
           "interval_coverage",
           "interval_inputs", "mean_half_width", "pixel_half_width", "roi_score_statistics"]
