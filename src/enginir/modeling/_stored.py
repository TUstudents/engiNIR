"""Plain facts of a stored model that need no fitting: its model type, the algorithm of
its hyperparameters and the words of its fold warnings.

A light module (no scikit-learn, no SciPy): the viewer's main window reads stored models
with it before the modelling libraries are loaded. :mod:`enginir.modeling` re-exports it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._checks import log_and_raise

#: The model types a model run can have.
MODEL_TYPES = ("PLS", "PCR")

#: The algorithm a stored model without one was fitted with.
STORED_ALGORITHM_DEFAULTS = {"PLS": "nipals", "PCR": "svd"}

#: Plain words of the fold warnings (contract §7.4).
_FOLD_WARNING_TEXT = {
    "few_groups": "fewer than 7 independent samples (fold groups)",
    "one_level_per_file": "the ROIs of one dataset share one value but are held out in "
                          "different folds, so the model can learn the dataset; give "
                          "them one sample label",
}


def _model_type(model_type: str) -> str:
    normalized = str(model_type).strip().upper()
    if normalized not in MODEL_TYPES:
        log_and_raise("Unsupported model type {!r}; use 'PLS' or 'PCR'", model_type, depth=2)
    return normalized


def stored_hyperparameters(model_type: str, hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
    """A stored model's hyperparameters with its algorithm (missing reads nipals or svd).

    A refit and "Fit Again" use them, so they run the algorithm the model was
    fitted with (sample-set-model-run contract §7).
    """
    values = dict(hyperparameters)
    default = STORED_ALGORITHM_DEFAULTS.get(_model_type(model_type))
    if default is not None:
        values.setdefault("algorithm", default)
    return values


def fold_warning_text(codes: Any) -> list[str]:
    """Plain words for stored or predicted fold warning codes (unknown codes as they are)."""
    if isinstance(codes, bytes):
        codes = codes.decode()
    if isinstance(codes, str):
        codes = [code for code in codes.split(",") if code]
    return [_FOLD_WARNING_TEXT.get(str(code), str(code)) for code in codes]
