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
    RoundRobinGroups,
    cross_validate,
    kennard_stone,
)
from .components import (
    COMPONENT_CHOICE_RULES,
    choose_components,
    guarded_choice,
    noise_components,
    structure_ceiling,
)
from .metrics import bias, metrics_report, r2_score, rmsec, rmsecv, rmsep, rpd, rpiq, sep
from .outliers import (
    dmodx,
    dmodx_limit,
    hotelling_t2,
    leverage_outliers,
    modelling_power,
    spectral_outliers,
)

__all__ = [
    "DEFAULT_GROUPED_N_SPLITS",
    "aggregate_regression_predictions",
    "bias",
    "RoundRobinGroups",
    "cross_validate",
    "hotelling_t2",
    "kennard_stone",
    "leverage_outliers",
    "dmodx",
    "dmodx_limit",
    "choose_components",
    "guarded_choice",
    "noise_components",
    "structure_ceiling",
    "COMPONENT_CHOICE_RULES",
    "modelling_power",
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
