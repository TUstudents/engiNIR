"""The fit of a model run on sufficient statistics (plan ``plans/kernel-pls.md``, Phase 2).

For a recipe without MSC or EMSC and a ``kernel`` (PLS) or ``gram`` (PCR)
model, :func:`enginir.modeling.fit_model_run` builds one
:class:`~enginir.models._gram.GramStats` with a block per fold group (folds
never split one) and works on it:

- every fold's model is fitted on the sum of its training blocks;
- held-out ROI predictions are the predictions of the ROI mean spectra (the
  model is linear in the preprocessed spectrum, contract §5), so the
  component curve needs no pass over the pixels;
- the pixel residual sums of the curve come from the cross products;
- the DModX exclusion takes excluded pixels out of their block (a downdate)
  instead of refitting on a copy of the kept rows.

What is per pixel (the held-out prediction of every pixel, T², DModX) takes
one chunked pass. Results equal the pipeline path (NIPALS or SVD, or these
engines on the rows) within floating-point precision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from .._checks import log_and_raise
from .._logging import get_logger
from ..models import PCR, PLSRegression
from ..models._gram import CHUNK_ROWS, GramStats, blas_threads
from ..validation.outliers import dmodx_limit
from ._components import _model_space, component_predictions, curve_from, projection_data

logger = get_logger(__name__)

_EPS = 1e-10


@dataclass
class Layout:
    """Where the ROIs and fold groups of a calibration set are in its pixel matrix."""

    ranges: list[tuple[int, int, int, int]]   # (start, stop, block, roi) per ROI
    block_of_row: np.ndarray                   # (N,) block (fold group) of every row
    roi_of_row: np.ndarray                     # (N,) ROI position of every row
    n_blocks: int
    n_rois: int


def layout(roi_index: np.ndarray, cv_groups: np.ndarray) -> Layout:
    """The row ranges of the ROIs (each ROI is one contiguous range) and their blocks."""
    roi_index = np.asarray(roi_index)
    n = roi_index.size
    change = np.flatnonzero(np.diff(roi_index)) + 1
    starts, stops = np.r_[0, change], np.r_[change, n]
    _values, block_of_row = np.unique(np.asarray(cv_groups), return_inverse=True)
    ranges = [(int(a), int(b), int(block_of_row[a]), int(roi_index[a]))
              for a, b in zip(starts, stops, strict=True)]
    if len({roi for *_rest, roi in ranges}) != len(ranges):
        log_and_raise("The pixels of an ROI are not one row range of the calibration set")
    return Layout(ranges, block_of_row.astype(np.int64), roi_index,
                  int(_values.size), int(roi_index.max()) + 1)


def build_stats(X: np.ndarray, y: np.ndarray, lay: Layout) -> GramStats:
    """The sums per fold group and per ROI, in one chunked pass."""
    return GramStats.from_ranges(X, y, lay.ranges, n_blocks=lay.n_blocks, n_rois=lay.n_rois)


@dataclass
class Fold:
    train_blocks: np.ndarray
    test_rois: np.ndarray


def folds(splitter: Any, cv_groups: np.ndarray, lay: Layout
          ) -> tuple[list[Fold], dict[str, Any]]:
    """The folds of ``splitter`` as block and ROI sets, and the cross-validation scheme."""
    splits = LeaveOneGroupOut() if isinstance(splitter, str) else splitter
    rows = np.empty((cv_groups.size, 0), dtype=np.float32)  # only the row count is used
    result = []
    for train, test in splits.split(rows, None, cv_groups):
        result.append(Fold(np.unique(lay.block_of_row[train]),
                           np.unique(lay.roi_of_row[test])))
    scheme = {"type": type(splits).__name__,
              "n_splits": int(splits.get_n_splits(rows, None, cv_groups))}
    return result, scheme


def _fit(make: Callable[[int], Any], n_components: int, stats: GramStats,
         blocks: np.ndarray | None) -> Any:
    return make(n_components).fit_stats(stats, blocks, single_target=True)


def coefficients_by_count(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Raw-unit coefficients (K, A) and intercepts (A,) with 1 … A components (one target)."""
    if isinstance(model, PLSRegression):
        rotations = np.asarray(model.x_rotations_, dtype=np.float64)
        q = np.asarray(model.y_loadings_, dtype=np.float64)[0]
        scaled = np.cumsum(rotations * q[None, :], axis=1)
        coef = scaled / np.asarray(model.x_std_)[:, None] * float(np.asarray(model.y_std_)[0])
        intercept = float(np.asarray(model.y_mean_)[0]) - np.asarray(model.x_mean_) @ coef
        return coef, intercept
    if isinstance(model, PCR):
        loadings = np.asarray(model.loadings_, dtype=np.float64)
        c = np.atleast_2d(np.asarray(model.reg_.coef_, dtype=np.float64))[0]
        coef = np.cumsum(loadings * c[None, :], axis=1) / np.asarray(model.x_std_)[:, None]
        offset = np.asarray(model.x_mean_) + np.asarray(model.pca_.mean_) * np.asarray(model.x_std_)
        intercept = float(np.atleast_1d(model.reg_.intercept_)[0]) - offset @ coef
        return coef, intercept
    log_and_raise("Fits on statistics support PLS and PCR, got {}", type(model).__name__)


def x_residual_sums(model: Any, stats: GramStats) -> tuple[np.ndarray, float]:
    """Σ‖z − Σⱼ≤ₐ tⱼpⱼᵀ‖² for a = 1 … A and Σ‖z‖², from the cross products (no pass).

    z are the rows in the model's scaled, centred space; with T = Z R,
    ZᵀT = G R and TᵀT = Rᵀ G R for G = ZᵀZ.
    """
    data = projection_data(model)
    moments = stats.moments()
    std = data["x_std"]
    gram = moments.cxx / np.outer(std, std)
    rotations, loadings = data["rotations"], data["loadings"]
    gr = gram @ rotations
    cross = np.einsum("ka,ka->a", loadings, gr)                 # pⱼᵀ G rⱼ
    joint = (loadings.T @ loadings) * (rotations.T @ gr)        # (pⱼᵀpₖ)(rⱼᵀ G rₖ)
    prefix = np.cumsum(np.cumsum(joint, axis=0), axis=1)
    total = float(np.trace(gram))
    return total - 2.0 * np.cumsum(cross) + np.diag(prefix), total


@dataclass
class CurveResult:
    curve: dict[str, Any]
    fold_models: list[Any]
    full: Any


def component_curve(make: Callable[[int], Any], stats: GramStats, fold_list: list[Fold], *,
                    a_max: int, n_samples: int) -> CurveResult:
    """The component curve (contract §7.4) from the statistics: one engine call per fold."""
    roi_x, roi_y = stats.roi_means()
    y_roi = roi_y[:, 0]
    oof = np.empty((stats.roi_n.size, a_max))
    fold_models = []
    for fold in fold_list:
        model = _fit(make, a_max, stats, fold.train_blocks)
        oof[fold.test_rois] = component_predictions(model, roi_x[fold.test_rois])
        fold_models.append(model)
    full = _fit(make, a_max, stats, None)
    fitted_roi = component_predictions(full, roi_x)
    rss_roi = ((fitted_roi - y_roi[:, None]) ** 2).sum(axis=0)
    moments = stats.moments()
    coef, _intercept = coefficients_by_count(full)  # the model passes through the means
    cyy, cxy = float(moments.cyy[0, 0]), moments.cxy[:, 0]
    rss_pixels = cyy - 2.0 * (coef.T @ cxy) + np.einsum("ka,kl,la->a", coef, moments.cxx, coef)
    cumulative, total = x_residual_sums(full, stats)
    curve = curve_from(errors=oof - y_roi[:, None], y_roi=y_roi, rss_roi=rss_roi,
                       rss_pixels=np.clip(rss_pixels, 0.0, None), ss_pixels=cyy,
                       cumulative_rss=np.clip(cumulative, 0.0, None), total_ss=total,
                       full=full, n_samples=n_samples)
    return CurveResult(curve, fold_models, full)


@dataclass
class GramExclusion:
    """The DModX exclusion of a fit on statistics (the attributes of ``DModXExclusion``)."""

    estimator_: Any
    excluded_: np.ndarray       # (N,) rows excluded from this fit
    dcrit_: float
    flagged_rois_: np.ndarray   # ROI positions with more pixels above DCrit than budget


def _residual_rows(model: Any, X: np.ndarray, rows: slice, weights: np.ndarray | None
                   ) -> np.ndarray:
    """Σₖ (wₖ eᵢₖ)² of every row of ``X[rows]`` in the model's scaled space."""
    centre, rotations, loadings = _model_space(model)
    z = centre(X[rows])
    e = z - (z @ rotations) @ loadings.T
    if weights is not None:
        e *= weights
    return np.einsum("ik,ik->i", e, e)


def _modelling_power(model: Any, X: np.ndarray, kept_ranges: list[tuple[slice, np.ndarray]],
                     n: int, a: int) -> np.ndarray:
    """MPow of every band over the kept rows (as :func:`~enginir.validation.modelling_power`)."""
    centre, rotations, loadings = _model_space(model)
    e2 = z2 = 0.0
    for rows, keep in kept_ranges:
        z = centre(X[rows])[keep]
        e = z - (z @ rotations) @ loadings.T
        e2 = e2 + np.einsum("ik,ik->k", e, e)
        z2 = z2 + np.einsum("ik,ik->k", z, z)
    if n - a - 1 < 1:
        log_and_raise("MPow needs N - A - A0 >= 1, got N={}, A={}, A0={}", n, a, 1)
    residual_sd = np.sqrt(np.asarray(e2) / (n - a - 1))
    initial_sd = np.sqrt(np.asarray(z2) / (n - 1))
    constant = initial_sd < _EPS
    if np.any(constant):
        logger.warning("MPow: {} constant band(s) get modeling power 0", int(constant.sum()))
    ratio = np.divide(residual_sd, initial_sd, out=np.ones_like(residual_sd), where=~constant)
    return np.clip(1.0 - ratio, 0.0, None)


def exclude(make: Callable[[int], Any], n_components: int, stats: GramStats, X: np.ndarray,
            y: np.ndarray, lay: Layout, blocks: np.ndarray | None, *, level: float,
            rounds: int, max_fraction: float, formula: str) -> GramExclusion:
    """DModX exclusion on the rows of ``blocks`` (contract §7.2), as ``DModXExclusion.fit``.

    Fit, compute every kept row's normalised DModX and DCrit, exclude the rows
    above DCrit within each ROI's budget (⌊max_fraction × n⌋ over all rounds),
    take them out of their block and refit; per round.
    """
    if formula not in ("dmodx-1", "dmodx-2"):
        log_and_raise("Unknown DModX formula {!r}", formula)
    chosen = (set(range(lay.n_blocks)) if blocks is None
              else {int(b) for b in np.asarray(blocks)})
    ranges = [r for r in lay.ranges if r[2] in chosen]
    ranges.sort(key=lambda r: r[0])
    budget = {roi: int(np.floor(max_fraction * (stop - start)))
              for start, stop, _b, roi in ranges}
    kept = np.ones(X.shape[0], dtype=bool)
    flagged: set[int] = set()
    work = stats.copy()
    model = _fit(make, n_components, work, blocks)
    dcrit = float("nan")
    for _round in range(int(rounds)):
        a = int(np.asarray(projection_data(model)["rotations"]).shape[1])
        k = int(np.asarray(projection_data(model)["rotations"]).shape[0])
        pieces = [(slice(lo, min(lo + CHUNK_ROWS, stop)),
                   kept[lo:min(lo + CHUNK_ROWS, stop)])
                  for start, stop, _b, _r in ranges for lo in range(start, stop, CHUNK_ROWS)]
        n = int(sum(int(keep.sum()) for _rows, keep in pieces))
        with blas_threads(n * k):
            weights = (_modelling_power(model, X, pieces, n, a) if formula == "dmodx-2"
                       else None)
            ss = np.concatenate([_residual_rows(model, X, rows, weights)[keep]
                                 for rows, keep in pieces])
        kept_rows = np.concatenate([np.arange(rows.start, rows.stop)[keep]
                                    for rows, keep in pieces])
        if n - a - 1 < 1 or k - a < 1:
            log_and_raise("DModX needs N - A - A0 >= 1 and K - A >= 1, got N={}, K={}, A={}, "
                          "A0={}", n, k, a, 1)
        s0 = float(np.sqrt(ss.sum() / ((n - a - 1) * (k - a))))
        if s0 < _EPS:
            logger.warning("DModX: the residuals are (almost) zero (s0={}); DModX is set to 0",
                           s0)
            distance = np.zeros(n)
        else:
            distance = np.sqrt(ss / (k - a)) * np.sqrt(n / (n - a - 1)) / s0
        dcrit = dmodx_limit(n, k, a, level)
        above = distance > dcrit
        roi_of_kept = lay.roi_of_row[kept_rows]
        newly: list[np.ndarray] = []
        for roi in np.unique(roi_of_kept[above]):
            in_roi = above & (roi_of_kept == roi)
            candidates = kept_rows[in_roi][np.argsort(-distance[in_roi], kind="stable")]
            if candidates.size > budget[int(roi)]:
                flagged.add(int(roi))
            take = candidates[: budget[int(roi)]]
            budget[int(roi)] -= take.size
            newly.append(take)
        excluded_now = np.concatenate(newly) if newly else np.empty(0, dtype=np.int64)
        if excluded_now.size == 0:
            break  # nothing above DCrit within the budgets: the model stays
        kept[excluded_now] = False
        work.remove_rows(X, y, excluded_now, lay.block_of_row, lay.roi_of_row)
        model = _fit(make, n_components, work, blocks)
    excluded = ~kept
    logger.debug("DModX exclusion: {} pixels excluded, DCrit {:.4g}", int(excluded.sum()), dcrit)
    return GramExclusion(model, excluded, dcrit, np.array(sorted(flagged), dtype=np.int64))


def held_out_predictions(X: np.ndarray, lay: Layout, fold_list: list[Fold],
                         coefficients: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Every pixel predicted by its fold's model (one chunked pass), float64."""
    fold_of_roi = np.full(lay.n_rois, -1, dtype=np.int64)
    for f, fold in enumerate(fold_list):
        fold_of_roi[fold.test_rois] = f
    out = np.empty(X.shape[0])
    with blas_threads(X.size):
        for start, stop, _block, roi in lay.ranges:
            coef, intercept = coefficients[int(fold_of_roi[roi])]
            for lo in range(start, stop, CHUNK_ROWS):
                hi = min(lo + CHUNK_ROWS, stop)
                out[lo:hi] = np.asarray(X[lo:hi], dtype=np.float64) @ coef + intercept
    return out


def model_coefficients(model: Any, n_components: int | None = None) -> tuple[np.ndarray, float]:
    """Raw-unit coefficients and intercept of ``model`` with ``n_components`` (default all)."""
    coef, intercept = coefficients_by_count(model)
    a = coef.shape[1] if n_components is None else int(n_components)
    return coef[:, a - 1], float(intercept[a - 1])


__all__ = ["CurveResult", "Fold", "GramExclusion", "Layout", "build_stats",
           "coefficients_by_count", "component_curve", "exclude", "folds",
           "held_out_predictions", "layout", "model_coefficients", "x_residual_sums"]
