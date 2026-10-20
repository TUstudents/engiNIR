"""Outlier diagnostics: leverage, Hotelling's T² and Q residuals.

These diagnostics are usually computed on the scores (T²) and residuals (Q)
of a fitted PCA or PLS model, or on its scores (leverage). Raw spectra with
more wavelengths than samples carry no leverage information (every sample has
the same leverage); pass model scores instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import beta, chi2, f

from enginir._checks import as_float_matrix, check_real, log_and_raise
from enginir._logging import get_logger

logger = get_logger(__name__)


def _as_float64_matrix(value: Any, name: str) -> np.ndarray:
    return as_float_matrix(value, name).astype(np.float64, copy=False)


def _check_alpha(alpha: float) -> float:
    alpha = check_real(alpha, "alpha", positive=True)
    if alpha >= 1.0:
        log_and_raise("alpha must be in (0, 1), got {}", alpha, depth=2)
    return alpha


def leverage_outliers(
    X: np.ndarray,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the hat-matrix leverage of every sample and flag high-leverage samples.

    For a model with intercept, the leverage of sample ``i`` is

    .. math:: h_i = \\frac{1}{n} + t_i^\\top (T^\\top T)^{-1} t_i

    where ``T`` is the mean-centred ``X`` restricted to its ``r`` non-zero
    principal directions. The leverages sum to ``r + 1``.

    Parameters
    ----------
    X:
        Model scores (recommended) or spectra, shape ``(n_samples, n_features)``.
    threshold:
        Samples with leverage above this value are flagged. Default:
        ``3 (r + 1) / n``, three times the mean leverage.

    Returns
    -------
    leverage : numpy.ndarray, shape (n_samples,)
        Leverage ``h_i`` in ``[1/n, 1]``.
    outlier_mask : numpy.ndarray of bool, shape (n_samples,)
        ``True`` where the leverage exceeds the threshold.
    """
    X = _as_float64_matrix(X, "X")
    n_samples = X.shape[0]
    U, singular_values, _ = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    tolerance = singular_values[0] * max(X.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank >= n_samples - 1:
        logger.warning(
            "leverage_outliers: X has rank {} with {} samples, so every sample has "
            "leverage 1; pass model scores instead of raw spectra",
            rank,
            n_samples,
        )
    leverage = 1.0 / n_samples + np.sum(U[:, :rank] ** 2, axis=1)
    if threshold is None:
        threshold = 3.0 * (rank + 1) / n_samples
    threshold = check_real(threshold, "threshold")
    return leverage, leverage > threshold


def hotelling_t2(
    scores: np.ndarray,
    alpha: float = 0.05,
    *,
    calibration_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return Hotelling's T² of every sample, its critical value and the outlier mask.

    ``T² = (t - mean)^T S^{-1} (t - mean)``, with the mean and sample
    covariance ``S`` of the calibration scores. The critical value at
    significance ``alpha`` depends on whether the samples are part of the
    calibration set:

    * calibration samples (``calibration_scores=None``):
      ``(n - 1)^2 / n * Beta_{1-alpha}(A / 2, (n - A - 1) / 2)``
    * new samples (``calibration_scores`` given):
      ``A (n - 1)(n + 1) / (n (n - A)) * F_{1-alpha}(A, n - A)``

    where ``n`` is the number of calibration samples and ``A`` the number of
    components.

    Parameters
    ----------
    scores:
        Scores of the samples to test, shape ``(n_samples, n_components)``.
    alpha:
        Significance level, ``0 < alpha < 1``. Default ``0.05``.
    calibration_scores:
        Scores of the calibration samples, shape ``(n_calibration,
        n_components)``. If ``None`` (default), ``scores`` are the
        calibration samples.

    Returns
    -------
    t2 : numpy.ndarray, shape (n_samples,)
    t2_crit : float
    outlier_mask : numpy.ndarray of bool, shape (n_samples,)

    References
    ----------
    N. D. Tracy, J. C. Young, R. L. Mason, "Multivariate control charts for
    individual observations", *Journal of Quality Technology* 24 (1992) 88-95.
    """
    alpha = _check_alpha(alpha)
    scores = _as_float64_matrix(scores, "scores")
    is_calibration = calibration_scores is None
    reference = scores if is_calibration else _as_float64_matrix(
        calibration_scores, "calibration_scores"
    )
    n, n_components = reference.shape
    if scores.shape[1] != n_components:
        log_and_raise(
            "scores have {} components but calibration_scores have {}",
            scores.shape[1],
            n_components,
        )
    if n <= n_components + 1:
        log_and_raise(
            "T² needs more than n_components + 1 = {} calibration samples, got {}",
            n_components + 1,
            n,
        )
    centred = scores - reference.mean(axis=0)
    covariance = np.atleast_2d(np.cov(reference, rowvar=False))
    t2 = np.einsum("ij,ij->i", centred @ np.linalg.pinv(covariance), centred)
    if is_calibration:
        quantile = beta.ppf(1.0 - alpha, n_components / 2.0, (n - n_components - 1) / 2.0)
        t2_crit = (n - 1) ** 2 / n * quantile
    else:
        quantile = f.ppf(1.0 - alpha, n_components, n - n_components)
        t2_crit = n_components * (n - 1) * (n + 1) / (n * (n - n_components)) * quantile
    return t2, float(t2_crit), t2 > t2_crit


def spectral_outliers(
    X: np.ndarray,
    X_reconstructed: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return the Q residual of every spectrum, its critical value and the outlier mask.

    ``Q_i = sum_k (x_ik - x_hat_ik)^2`` is the squared reconstruction error
    (also called SPE). The critical value approximates the distribution of Q
    by ``g chi^2(h)`` with the moments of the observed Q values,
    ``g = var(Q) / (2 mean(Q))`` and ``h = 2 mean(Q)^2 / var(Q)``.

    Parameters
    ----------
    X:
        Spectra, shape ``(n_samples, n_wavelengths)``.
    X_reconstructed:
        Model reconstruction of ``X`` (e.g. ``scores @ loadings.T + mean``),
        same shape.
    alpha:
        Significance level, ``0 < alpha < 1``. Default ``0.05``.

    Returns
    -------
    q_residuals : numpy.ndarray, shape (n_samples,)
    q_crit : float
        Critical value; the largest Q if the Q values have no spread.
    outlier_mask : numpy.ndarray of bool, shape (n_samples,)

    References
    ----------
    P. Nomikos, J. F. MacGregor, "Multivariate SPC charts for monitoring batch
    processes", *Technometrics* 37 (1995) 41-59.
    """
    alpha = _check_alpha(alpha)
    X = as_float_matrix(X, "X")
    X_reconstructed = as_float_matrix(X_reconstructed, "X_reconstructed")
    if X.shape != X_reconstructed.shape:
        log_and_raise(
            "X shape {} does not match X_reconstructed shape {}", X.shape, X_reconstructed.shape
        )
    if X.shape[0] < 2:
        log_and_raise("spectral_outliers needs at least 2 samples, got {}", X.shape[0])
    # One float64 residual buffer; float32 spectra are not converted as a whole.
    residuals = np.subtract(X, X_reconstructed, dtype=np.float64)
    q = np.einsum("ij,ij->i", residuals, residuals)
    q_mean, q_var = float(q.mean()), float(q.var(ddof=1))
    if q_var <= np.finfo(np.float64).eps * q_mean**2:
        logger.warning(
            "spectral_outliers: Q residuals have no spread (mean {:.3g}); using max(Q) as limit",
            q_mean,
        )
        q_crit = float(q.max())
    else:
        g = q_var / (2.0 * q_mean)
        h = 2.0 * q_mean**2 / q_var
        q_crit = float(g * chi2.ppf(1.0 - alpha, h))
    return q, q_crit, q > q_crit


__all__ = ["hotelling_t2", "leverage_outliers", "spectral_outliers"]
