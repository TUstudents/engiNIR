"""Model validation: cross-validation, sample selection, outliers and figures of merit.

Cross-validation and sample selection
    :func:`cross_validate` (grouped, k-fold, leave-one-group-out and
    leave-one-out), :func:`kennard_stone`, and
    :func:`aggregate_regression_predictions` for group (ROI) level reporting.
Outlier diagnostics
    :func:`leverage_outliers`, :func:`hotelling_t2`, :func:`spectral_outliers`
Figures of merit
    :func:`rmsec`, :func:`rmsecv`, :func:`rmsep`, :func:`bias`, :func:`sep`,
    :func:`r2_score`, :func:`rpd`, :func:`rpiq` and :func:`metrics_report`;
    definitions in :mod:`enginir.validation.metrics`.
"""

from .cross_val import (
    DEFAULT_GROUPED_N_SPLITS,
    aggregate_regression_predictions,
    cross_validate,
    kennard_stone,
)
from .metrics import bias, metrics_report, r2_score, rmsec, rmsecv, rmsep, rpd, rpiq, sep
from .outliers import hotelling_t2, leverage_outliers, spectral_outliers

__all__ = [
    "DEFAULT_GROUPED_N_SPLITS",
    "aggregate_regression_predictions",
    "bias",
    "cross_validate",
    "hotelling_t2",
    "kennard_stone",
    "leverage_outliers",
    "metrics_report",
    "r2_score",
    "rmsec",
    "rmsecv",
    "rmsep",
    "rpd",
    "rpiq",
    "sep",
    "spectral_outliers",
]
