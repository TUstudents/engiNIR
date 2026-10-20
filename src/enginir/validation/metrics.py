"""Figures of merit for quantitative calibrations.

All functions take reference values ``y_true`` and predictions ``y_pred`` as
1-D arrays of equal length. With residuals ``e_i = y_pred_i - y_true_i``:

=========  ==============================================================
RMSE       ``sqrt(mean(e^2))`` - reported as RMSEC (calibration), RMSECV
           (cross-validation) or RMSEP (independent test set); no
           degrees-of-freedom correction is applied
bias       ``mean(e)``; positive when the model over-predicts
SEP        ``sqrt(sum((e - bias)^2) / (n - 1))``, the bias-corrected
           standard error of prediction
R²         ``1 - sum(e^2) / sum((y_true - mean(y_true))^2)``, the
           coefficient of determination (negative for predictions worse
           than the mean)
RPD        ``SD(y_true) / RMSE`` with the sample standard deviation
RPIQ       ``(Q3 - Q1)(y_true) / RMSE``
=========  ==============================================================

``RMSE^2 = bias^2 + SEP^2 (n - 1) / n`` links the error measures. RPD and
RPIQ are defined with RMSE in the denominator; some authors use SEP
instead, which gives the same value for unbiased predictions and large ``n``.

References
----------
T. Fearn, "Assessing calibrations: SEP, RPD, RER and R2", *NIR news* 13
(2002) 12-14.

V. Bellon-Maurel, E. Fernandez-Ahumada, B. Palagos, J.-M. Roger,
A. McBratney, "Critical review of chemometric indicators commonly used for
assessing the quality of the prediction of soil attributes by NIR
spectroscopy", *TrAC Trends in Analytical Chemistry* 29 (2010) 1073-1081.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from enginir._checks import log_and_raise
from enginir._logging import get_logger

logger = get_logger(__name__)


def _validated_pair(y_true: Any, y_pred: Any, *, min_samples: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return ``y_true`` and ``y_pred`` as finite 1-D float64 arrays of equal length."""
    try:
        true = np.asarray(y_true, dtype=np.float64)
        pred = np.asarray(y_pred, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        log_and_raise("y_true and y_pred must be numeric", cause=exc, depth=2)
    if true.ndim != 1 or pred.ndim != 1:
        log_and_raise(
            "y_true and y_pred must be 1-D (compute multi-target metrics per column), "
            "got shapes {} and {}",
            true.shape,
            pred.shape,
            depth=2,
        )
    if true.shape != pred.shape:
        log_and_raise("y_true has {} values but y_pred has {}", true.size, pred.size, depth=2)
    if true.size < min_samples:
        log_and_raise("at least {} sample(s) are required, got {}", min_samples, true.size,
                      depth=2)
    if not (np.all(np.isfinite(true)) and np.all(np.isfinite(pred))):
        log_and_raise("y_true and y_pred must contain only finite values", depth=2)
    return true, pred


def _rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _ratio_to_rmse(spread: float, true: np.ndarray, pred: np.ndarray, label: str) -> float:
    rmse = _rmse(true, pred)
    if rmse == 0.0:
        logger.warning("{}: predictions are exact (RMSE = 0); returning inf", label)
        return float("inf")
    return spread / rmse


def rmsec(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean square error of calibration (predictions on the training samples)."""
    return _rmse(*_validated_pair(y_true, y_pred))


def rmsecv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean square error of cross-validation (out-of-fold predictions)."""
    return _rmse(*_validated_pair(y_true, y_pred))


def rmsep(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean square error of prediction (independent test samples)."""
    return _rmse(*_validated_pair(y_true, y_pred))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean prediction error ``mean(y_pred - y_true)``; positive means over-prediction."""
    true, pred = _validated_pair(y_true, y_pred)
    return float(np.mean(pred - true))


def sep(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Bias-corrected standard error of prediction.

    ``SEP = sqrt(sum((e - mean(e))^2) / (n - 1))`` with ``e = y_pred - y_true``,
    i.e. the sample standard deviation of the prediction errors. Requires at
    least two samples.
    """
    true, pred = _validated_pair(y_true, y_pred, min_samples=2)
    return float(np.std(pred - true, ddof=1))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot``.

    Returns ``nan`` (with a warning) if ``y_true`` is constant.
    """
    true, pred = _validated_pair(y_true, y_pred)
    if np.ptp(true) == 0.0:
        logger.warning("r2_score: y_true is constant; returning nan")
        return float("nan")
    total = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - float(np.sum((pred - true) ** 2)) / total


def rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Ratio of performance to deviation, ``SD(y_true) / RMSE``.

    ``SD`` is the sample standard deviation (``n - 1``). Requires at least
    two samples; returns ``inf`` for exact predictions.
    """
    true, pred = _validated_pair(y_true, y_pred, min_samples=2)
    return _ratio_to_rmse(float(np.std(true, ddof=1)), true, pred, "rpd")


def rpiq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Ratio of performance to interquartile range, ``(Q3 - Q1) / RMSE``.

    Quartiles use NumPy's default linear interpolation. Requires at least two
    samples; returns ``inf`` for exact predictions.
    """
    true, pred = _validated_pair(y_true, y_pred, min_samples=2)
    q75, q25 = np.percentile(true, [75, 25])
    return _ratio_to_rmse(float(q75 - q25), true, pred, "rpiq")


def metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_cv: np.ndarray | None = None,
    y_true_test: np.ndarray | None = None,
    y_pred_test: np.ndarray | None = None,
    as_dataframe: bool = False,
) -> Any:
    """Collect the figures of merit of a calibration in one record.

    Parameters
    ----------
    y_true:
        Reference values of the calibration samples.
    y_pred:
        Predictions for the calibration samples by the model fitted on them.
    y_pred_cv:
        Optional cross-validated predictions for the calibration samples.
    y_true_test, y_pred_test:
        Optional reference values and predictions of an independent test set.
        Both must be given.
    as_dataframe:
        If ``True``, return a one-row :class:`pandas.DataFrame` instead of a
        dict.

    Returns
    -------
    dict or pandas.DataFrame
        Always ``"RMSEC"`` and ``"R2_cal"``; ``"RMSECV"`` and ``"R2_cv"`` if
        ``y_pred_cv`` is given; ``"RMSEP"``, ``"bias"``, ``"SEP"``, ``"R2"``,
        ``"RPD"`` and ``"RPIQ"`` if a test set is given.

    Raises
    ------
    ValueError
        If only one of ``y_true_test`` and ``y_pred_test`` is given.
    """
    if (y_true_test is None) != (y_pred_test is None):
        log_and_raise("y_true_test and y_pred_test must be given together")
    report: dict[str, float] = {
        "RMSEC": rmsec(y_true, y_pred),
        "R2_cal": r2_score(y_true, y_pred),
    }
    if y_pred_cv is not None:
        report["RMSECV"] = rmsecv(y_true, y_pred_cv)
        report["R2_cv"] = r2_score(y_true, y_pred_cv)
    if y_true_test is not None and y_pred_test is not None:
        report.update(
            RMSEP=rmsep(y_true_test, y_pred_test),
            bias=bias(y_true_test, y_pred_test),
            SEP=sep(y_true_test, y_pred_test),
            R2=r2_score(y_true_test, y_pred_test),
            RPD=rpd(y_true_test, y_pred_test),
            RPIQ=rpiq(y_true_test, y_pred_test),
        )
    if as_dataframe:
        import pandas as pd

        return pd.DataFrame([report])
    return report


__all__ = [
    "bias",
    "metrics_report",
    "r2_score",
    "rmsec",
    "rmsecv",
    "rmsep",
    "rpd",
    "rpiq",
    "sep",
]
