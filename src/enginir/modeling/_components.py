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
from ..validation.components import (
    COMPONENT_CHOICE_RULES,
    noise_components,
    structure_ceiling,
)
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
    errors = _roi_means(oof, roi_index, n_rois) - y_roi[:, np.newaxis]   # (R, A)

    full: Any = clone(model)
    full.fit(X, y)
    fitted = component_predictions(full, X)
    rss_roi = ((_roi_means(fitted, roi_index, n_rois) - y_roi[:, np.newaxis]) ** 2).sum(axis=0)
    y_pixels = np.asarray(y, dtype=np.float64)
    ss_pixels = float(((y_pixels - y_pixels.mean()) ** 2).sum())
    rss_pixels = ((fitted - y_pixels[:, np.newaxis]) ** 2).sum(axis=0)
    projection = project(full, X)
    curve = curve_from(errors=errors, y_roi=y_roi, rss_roi=rss_roi, rss_pixels=rss_pixels,
                       ss_pixels=ss_pixels, cumulative_rss=projection.cumulative_rss,
                       total_ss=projection.total_ss, full=full, n_samples=n_samples)
    return curve, oof, cv["cv_scheme"]


def curve_from(*, errors: np.ndarray, y_roi: np.ndarray, rss_roi: np.ndarray,
               rss_pixels: np.ndarray, ss_pixels: float, cumulative_rss: np.ndarray,
               total_ss: float, full: Any, n_samples: int) -> dict[str, Any]:
    """The component curve from its sums (contract §7.4), for every fit path.

    ``errors`` are the held-out ROI errors (R, A); ``rss_roi`` and
    ``rss_pixels`` the residual sums of the full model per component count;
    ``cumulative_rss`` and ``total_ss`` the X residuals of the full model.
    """
    n_rois = int(errors.shape[0])
    ss = float(((y_roi - y_roi.mean()) ** 2).sum())
    press = (errors ** 2).sum(axis=0)
    rmsecv = np.sqrt(press / n_rois)
    spread = np.std(errors ** 2, axis=0, ddof=1) if n_rois > 1 else np.zeros_like(press)
    se = np.zeros_like(rmsecv)
    ok = rmsecv >= _EPS
    se[ok] = spread[ok] / (2.0 * rmsecv[ok] * np.sqrt(n_rois))
    previous = np.concatenate([[ss], rss_roi[:-1]])
    curve: dict[str, Any] = {
        "n_components": np.arange(1, errors.shape[1] + 1, dtype=np.int64),
        "rmsecv": rmsecv,
        "rmsecv_se": se,
        "q2": 1.0 - _ratio(press, previous, "RSS of the previous component count"),
        "q2_cum": 1.0 - _ratio(press, np.full_like(press, ss), "the reference spread"),
        "r2y_cum": 1.0 - _ratio(rss_pixels, np.full_like(rss_pixels, ss_pixels),
                                "the pixel reference spread"),
        "r2y_cum_roi": 1.0 - _ratio(rss_roi, np.full_like(rss_roi, ss), "the reference spread"),
        "r2x_cum": 1.0 - _ratio(cumulative_rss, np.full_like(cumulative_rss, total_ss),
                                "the spectral spread"),
    }
    dof = n_rois - curve["n_components"] - 1  # RMSEE degrees of freedom (§7.5), per ROI
    with np.errstate(invalid="ignore", divide="ignore"):
        curve["rmsee_roi"] = np.where(dof >= 1, np.sqrt(rss_roi / np.maximum(dof, 1)), np.nan)
    curve["dw_weights"] = weight_roughness(full)
    curve["noise"] = noise_components(curve, n_samples=n_samples)
    curve["structure_ceiling"] = structure_ceiling(curve)
    for rule in COMPONENT_CHOICE_RULES:
        curve[f"suggested_{rule}"] = choose_components(curve, rule, n_samples=n_samples)
    return curve


def weight_roughness(fitted: Any) -> np.ndarray:
    """Durbin–Watson value Σ (wₖ₊₁ − wₖ)² / Σ wₖ² of each component's weight vector (§7.4).

    PLS: the X weights; PCR: the principal components. Near 0 for a smooth,
    spectrum-like vector, near 2 for white noise. NaN for a zero vector.
    """
    model: Any = fitted[-1] if isinstance(fitted, Pipeline) else fitted
    if isinstance(model, PLSRegression):
        weights = np.asarray(model.x_weights_, dtype=np.float64)
    elif isinstance(model, PCR):
        weights = np.asarray(model.pca_.components_, dtype=np.float64).T
    else:
        log_and_raise("Weight roughness exists for PLS and PCR models, got {}",
                      type(model).__name__)
    total = np.sum(weights ** 2, axis=0)
    out = np.full(total.shape, np.nan)
    if weights.shape[0] < 2:
        return out
    rough = np.sum(np.diff(weights, axis=0) ** 2, axis=0)
    np.divide(rough, total, out=out, where=total >= _EPS)
    return out


def projection_data(model: Any) -> dict[str, np.ndarray]:
    """``x_mean``, ``x_std``, ``rotations``, ``loadings`` of a PLS or PCR model.

    Scores are ((x - x_mean) / x_std) @ rotations for a preprocessed spectrum
    x, for both model types (PCR's PCA mean is folded into ``x_mean``).
    """
    if isinstance(model, PLSRegression):
        return {"x_mean": np.asarray(model.x_mean_, dtype=np.float64),
                "x_std": np.asarray(model.x_std_, dtype=np.float64),
                "rotations": np.asarray(model.x_rotations_, dtype=np.float64),
                "loadings": np.asarray(model.x_loadings_, dtype=np.float64)}
    if isinstance(model, PCR):
        std = np.asarray(model.x_std_, dtype=np.float64)
        components = np.asarray(model.pca_.components_, dtype=np.float64).T
        return {"x_mean": np.asarray(model.x_mean_, dtype=np.float64)
                + np.asarray(model.pca_.mean_, dtype=np.float64) * std,
                "x_std": std, "rotations": components, "loadings": components}
    log_and_raise("Projection data exist for PLS and PCR models, got {}", type(model).__name__)


def band_importance(fitted: Any, X: np.ndarray, kept: np.ndarray) -> dict[str, np.ndarray]:
    """Mean spectrum, VIP (PLS) and selectivity ratio of a fitted model (contract §7.5).

    The selectivity ratio (Rajalahti et al., *Chemometrics and Intelligent
    Laboratory Systems* 95 (2009) 35-48) uses the target projection in the
    model's scaled, centred X space Z over the fitted pixels: w = b / ‖b‖ with
    b = coef · x_std, t = Z w, p = Zᵀt / tᵀt, and per band
    SRⱼ = pⱼ² tᵀt / (Σᵢ Zᵢⱼ² − pⱼ² tᵀt). Computed chunk by chunk in one pass.
    The selectivity ratio only for one target; VIP for every PLS model.
    """
    steps = fitted[:-1] if isinstance(fitted, Pipeline) else None
    model: Any = fitted[-1] if isinstance(fitted, Pipeline) else fitted
    result: dict[str, np.ndarray] = {
        "mean_spectrum": projection_data(model)["x_mean"].astype(np.float32)}
    if isinstance(model, PLSRegression):
        result["vip"] = np.asarray(model.vip_scores(), dtype=np.float32)
    coef = np.asarray(model.coef_, dtype=np.float64)
    if coef.ndim != 1 and not (coef.ndim == 2 and coef.shape[1] == 1):
        logger.debug("The selectivity ratio is computed for one target only")
        return result
    centre, _rotations, _loadings = _model_space(model)
    b = coef.reshape(-1) * np.asarray(model.x_std_, dtype=np.float64)
    norm = float(np.linalg.norm(b))
    if norm < _EPS:
        logger.warning("Selectivity ratio undefined: the regression vector is zero")
        return result
    w = b / norm
    rows = np.flatnonzero(kept)
    tt = 0.0
    zt = np.zeros(b.size)
    zz = np.zeros(b.size)
    for start in range(0, rows.size, _CHUNK_ROWS):
        block = X[rows[start:start + _CHUNK_ROWS]]
        z = centre(steps.transform(block) if steps is not None else block)
        t = z @ w
        tt += float(t @ t)
        zt += z.T @ t
        zz += np.einsum("ij,ij->j", z, z)
    if tt < _EPS:
        logger.warning("Selectivity ratio undefined: the target-projection scores are zero")
        return result
    p = zt / tt
    explained = p ** 2 * tt
    residual = zz - explained
    zero = residual <= _EPS * np.maximum(zz, 1.0)
    if np.any(zero):
        logger.warning("Selectivity ratio of {} band(s) is infinite (no residual variance); "
                       "stored as NaN", int(zero.sum()))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(zero, np.nan, explained / np.where(zero, 1.0, residual))
    result["selectivity_ratio"] = ratio.astype(np.float32)
    return result


def training_diagnostics(fitted: Any, X: np.ndarray, y: np.ndarray, kept: np.ndarray,
                         roi_index: np.ndarray, *, y_divisor: float,
                         with_scores: bool = False) -> Any:
    """Fit statistics (metrics), per-pixel diagnostics and projection data (§5, §7.5).

    ``kept`` marks the pixels the model was fitted on (all, unless DModX
    excluded some); excluded pixels get the same formulas. The projection
    data let predictions compute DModXPS and T²PS (prediction-runtime
    contract, Applicability diagnostics). With ``with_scores`` the scores
    of every pixel (N, A) are returned as a fourth element.
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
    projection_fields: dict[str, Any] = {**projection_data(model),
                                         "score_variances": np.asarray(variances),
                                         "n_train": n, "n_components": a}

    def t2_crit(components: int, level: float) -> float:
        return float(components * (n - 1) / (n - components) * f.ppf(level, components,
                                                                      n - components))

    # Undefined limits and distances are NaN, never left out (project-schema
    # contract, Core and display data).
    nan = float("nan")
    diagnostics.update(t2_crit_095=nan, t2_crit_099=nan, t2_crit_2_095=nan, t2_crit_2_099=nan,
                       dcrit_099=nan, dmodx=np.full(X.shape[0], np.nan, dtype=np.float32))
    projection_fields.update(t2_crit_099=nan, s0=nan, dcrit_099=nan)
    if n - a >= 1:
        diagnostics["t2_crit_095"] = t2_crit(a, 0.95)
        diagnostics["t2_crit_099"] = t2_crit(a, 0.99)
        projection_fields["t2_crit_099"] = diagnostics["t2_crit_099"]
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
        projection_fields.update(s0=s0, dcrit_099=diagnostics["dcrit_099"])

    n_rois = int(roi_index.max()) + 1
    counts = np.bincount(roi_index, minlength=n_rois)
    for key, values, limit in (("roi_share_above_dcrit", diagnostics["dmodx"], "dcrit_099"),
                               ("roi_share_above_t2crit", t2, "t2_crit_099")):
        if np.isfinite(diagnostics[limit]):
            above = values > diagnostics[limit]
            diagnostics[key] = (np.bincount(roi_index, weights=above, minlength=n_rois)
                                / counts).astype(np.float32)
        else:
            diagnostics[key] = np.full(n_rois, np.nan, dtype=np.float32)
    # ROI scores of the score plot: mean over each ROI's fitted pixels (§7.5).
    fitted_counts = np.bincount(roi_index[kept], minlength=n_rois)
    sums = np.stack([np.bincount(roi_index[kept], weights=projection.scores[kept, c],
                                 minlength=n_rois) for c in range(a)], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        diagnostics["roi_scores"] = np.where(fitted_counts[:, None] > 0,
                                             sums / fitted_counts[:, None],
                                             np.nan).astype(np.float32)
    diagnostics["roi_score_pixels"] = fitted_counts.astype(np.int64)
    coef = np.asarray(model.coef_, dtype=np.float64)
    diagnostics["coef_cs"] = (coef * np.asarray(model.x_std_) / y_divisor).astype(np.float32)
    diagnostics.update(band_importance(fitted, X, kept))

    fitted_values = np.asarray(fitted.predict(X), dtype=np.float64).reshape(X.shape[0])
    y = np.asarray(y, dtype=np.float64)
    metrics: dict[str, Any] = {key: float("nan") for key in
                               ("pixel_rmsee", "pixel_r2_adj", "rmsee", "r2_adj")}
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
    if with_scores:
        return metrics, diagnostics, projection_fields, projection.scores
    return metrics, diagnostics, projection_fields


__all__ = ["AllComponents", "Projection", "band_importance", "component_curve",
           "component_predictions",
           "default_max", "project", "projection_data", "rank_limit", "training_diagnostics"]
