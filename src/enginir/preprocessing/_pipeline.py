"""Preprocessing helpers shared by model fitting and prediction.

Recipes are turned into transformers by
:meth:`enginir.preprocessing.PreprocessingRecipe.build_steps`. This module
composes those steps with an estimator, applies them to matrices, and saves and
restores the fitted state of the two steps that learn from training data (MSC
and EMSC) so that saved models replay exactly the training-time preprocessing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from enginir._checks import as_float_matrix, as_float_vector, check_choice, check_integer
from enginir._checks import log_and_raise
from enginir._logging import get_logger

from .recipe import PreprocessingRecipe
from .scatter import EMSC, MSC

logger = get_logger(__name__)
_WORKING_DTYPE = np.float32


def make_model_pipeline(
    *,
    recipe: PreprocessingRecipe,
    wavelengths: np.ndarray,
    estimator: Any,
) -> Pipeline:
    """Return an unfitted pipeline of the recipe steps followed by ``estimator``.

    The estimator is the final step, named ``"model"``.
    """
    return Pipeline([*recipe.build_steps(wavelengths), ("model", estimator)])


def apply_preprocessing_recipe(
    X: np.ndarray,
    *,
    recipe: PreprocessingRecipe,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """Fit and apply every recipe step to ``X`` in ``float32``.

    Only leakage-free for cross-validation when ``recipe.requires_fitting`` is
    ``False``; MSC and EMSC must otherwise be fitted inside each training fold.
    """
    X_out = np.asarray(X, dtype=_WORKING_DTYPE)
    for _, transformer in recipe.build_steps(wavelengths):
        X_out = np.asarray(transformer.fit_transform(X_out), dtype=_WORKING_DTYPE)
    return X_out


def transform_preprocessing_steps(
    X: np.ndarray,
    steps: list[tuple[str, Any]],
) -> np.ndarray:
    """Apply already fitted (or stateless) steps to ``X`` in ``float32``."""
    X_out = np.asarray(X, dtype=_WORKING_DTYPE)
    for _, transformer in steps:
        X_out = np.asarray(transformer.transform(X_out), dtype=_WORKING_DTYPE)
    return X_out


def fitted_preprocessing_state(
    predictor: Any,
    *,
    recipe: PreprocessingRecipe,
) -> dict[str, Any]:
    """Extract the fitted MSC/EMSC parameters of a trained pipeline.

    Parameters
    ----------
    predictor:
        Fitted pipeline built by :func:`make_model_pipeline`.
    recipe:
        Recipe the pipeline was built from.

    Returns
    -------
    dict
        Empty if the recipe has no stateful step. Otherwise contains
        ``"msc": {"reference"}`` and/or
        ``"emsc": {"method", "order", "reference", "weights", "interferences"}``,
        the inputs :func:`restore_fitted_preprocessing_steps` needs.
    """
    if not recipe.requires_fitting:
        return {}
    named_steps = getattr(predictor, "named_steps", None)
    if not isinstance(named_steps, Mapping):
        log_and_raise("Stateful preprocessing requires a fitted pipeline with named_steps")
    state: dict[str, Any] = {}
    if recipe.msc:
        msc = named_steps["msc"]
        state["msc"] = {"reference": np.asarray(msc.reference_, dtype=np.float64)}
    if recipe.emsc:
        emsc = named_steps["emsc"]
        interferences = emsc.interferences_
        state["emsc"] = {
            "method": emsc.method,
            "order": int(emsc.order_),
            "reference": np.asarray(emsc.reference_, dtype=np.float64),
            "weights": np.asarray(emsc.weights_, dtype=np.float64),
            "interferences": (
                None if interferences is None else np.asarray(interferences, dtype=np.float64)
            ),
        }
    return state


def restore_fitted_preprocessing_steps(
    steps: list[tuple[str, Any]],
    *,
    recipe: PreprocessingRecipe,
    state: Mapping[str, Any] | None,
) -> list[tuple[str, Any]]:
    """Replace the unfitted MSC/EMSC steps by transformers rebuilt from ``state``.

    Parameters
    ----------
    steps:
        Unfitted steps from :meth:`PreprocessingRecipe.build_steps`.
    recipe:
        Recipe the steps were built from.
    state:
        Mapping produced by :func:`fitted_preprocessing_state`.

    Returns
    -------
    list of (str, transformer)
        ``steps`` with the stateful transformers fitted to the saved state.
        Persisted parameters are validated by the transformers' own ``fit``.
    """
    if not recipe.requires_fitting:
        return steps
    if not isinstance(state, Mapping):
        log_and_raise("Model artifact with MSC/EMSC preprocessing is missing its fitted state")
    restorers = {"msc": _restore_msc, "emsc": _restore_emsc}
    restored: list[tuple[str, Any]] = []
    for name, transformer in steps:
        if name in restorers:
            step_state = state.get(name)
            if not isinstance(step_state, Mapping):
                log_and_raise("Model artifact is missing the fitted {} state", name.upper())
            transformer = restorers[name](step_state)
        restored.append((name, transformer))
    return restored


def _restore_msc(state: Mapping[str, Any]) -> MSC:
    reference = as_float_vector(state.get("reference"), "persisted MSC reference")
    return MSC(reference=reference).fit(reference[np.newaxis, :])


def _restore_emsc(state: Mapping[str, Any]) -> EMSC:
    reference = as_float_vector(state.get("reference"), "persisted EMSC reference")
    interferences = state.get("interferences")
    emsc = EMSC(
        method=check_choice(state.get("method"), "persisted EMSC method", ("mean", "median")),
        order=check_integer(state.get("order"), "persisted EMSC order", minimum=0),
        reference=reference,
        interferences=(
            None if interferences is None
            else as_float_matrix(interferences, "persisted EMSC interferences")
        ),
        weights=as_float_vector(state.get("weights"), "persisted EMSC weights"),
    )
    return emsc.fit(reference[np.newaxis, :])


__all__ = [
    "apply_preprocessing_recipe",
    "fitted_preprocessing_state",
    "make_model_pipeline",
    "restore_fitted_preprocessing_steps",
    "transform_preprocessing_steps",
]
