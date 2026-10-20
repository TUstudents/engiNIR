"""Cross-validation and sample-set selection utilities."""

from __future__ import annotations

from typing import Any

from sklearn.base import clone, is_classifier
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    StratifiedKFold,
)
import numpy as np
from scipy.spatial.distance import cdist

from enginir._checks import as_float_matrix, check_choice, check_integer, log_and_raise
from enginir._logging import get_logger

from .metrics import r2_score, rmsec, rmsecv, rmsep, rpd, rpiq

logger = get_logger(__name__)

DEFAULT_GROUPED_N_SPLITS = 7
_AUTO_CV = "auto"
_SUPPORTED_STRING_CV = {_AUTO_CV, "loo", "logo"}
_METRICS = {"rmsecv": rmsecv, "r2": r2_score, "rpd": rpd, "rpiq": rpiq}
_KS_BLOCK_ELEMENTS = 2**22  # distances per block (32 MiB of float64)


def cross_validate(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int | str = _AUTO_CV,
    groups: np.ndarray | None = None,
    scoring: list[str] | None = None,
    return_train_score: bool = False,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Return out-of-fold predictions and figures of merit of ``estimator``.

    Each fold fits a fresh clone of ``estimator`` on the training part and
    predicts the held-out part, so every sample is predicted exactly once by a
    model that did not see it. Preprocessing that learns from data (MSC, EMSC,
    scaling) must be part of ``estimator`` (e.g. a
    :class:`sklearn.pipeline.Pipeline`) to avoid leaking test information into
    training.

    Parameters
    ----------
    estimator:
        scikit-learn compatible regressor or classifier (fitted or not; it
        is cloned).
    X:
        Spectra, shape ``(n_samples, n_wavelengths)``.
    y:
        Reference values, shape ``(n_samples,)`` or ``(n_samples, n_targets)``,
        or class labels, shape ``(n_samples,)``, for classifiers.
    cv:
        Splitting strategy:

        * ``"auto"`` (default): with ``groups``, 7-fold
          :class:`~sklearn.model_selection.GroupKFold` if there are at least 7
          groups and leave-one-group-out otherwise; without ``groups``,
          shuffled 7-fold (stratified for classifiers).
        * an integer ``k >= 2``: k-fold, grouped if ``groups`` is given,
          otherwise shuffled (stratified for classifiers).
        * ``"logo"``: leave-one-group-out (requires ``groups``).
        * ``"loo"``: leave-one-sample-out (ignores ``groups`` with a warning).
    groups:
        Group label of every sample, shape ``(n_samples,)``. Samples of one
        group (e.g. pixels of one ROI or replicate spectra of one physical
        sample) never appear in both the training and the test part of a fold.
    scoring:
        Regression metrics to compute, any of ``"rmsecv"``, ``"r2"``,
        ``"rpd"`` and ``"rpiq"`` (see :mod:`enginir.validation.metrics`).
        Default: all of them. Ignored for classifiers.
    return_train_score:
        If ``True``, also report per-fold training and test errors
        (regression only).
    random_state:
        Seed for shuffling the samples in ungrouped k-fold splitting.
        Default ``0`` (reproducible splits).

    Returns
    -------
    dict
        ``"y_true"`` and ``"y_pred"`` (out-of-fold predictions in input order);
        ``"cv_scheme"``, the splitter used, e.g.
        ``{"type": "GroupKFold", "n_splits": 7}``;
        for regressors, one entry per requested metric (a float, or an array
        of shape ``(n_targets,)`` for 2-D ``y``); and, for regressors with
        ``return_train_score``, ``"fold_results"``, a list of
        ``{"train_rmsec", "test_rmsep"}`` dicts.

    Raises
    ------
    ValueError
        If the inputs are inconsistent or the splitting strategy cannot be
        applied to the data.
    """
    X = as_float_matrix(X, "X")
    is_regression = not is_classifier(estimator)
    y = _validated_targets(y, n_samples=X.shape[0], is_regression=is_regression)
    if isinstance(scoring, str):
        log_and_raise("scoring must be a list of metric names, got {!r}", scoring)
    metric_names = list(_METRICS) if scoring is None else list(scoring)
    for name in metric_names:
        check_choice(name, "scoring metric", _METRICS)
    groups_array = _validate_groups(groups, n_samples=X.shape[0])
    splitter = _make_splitter(
        cv=cv,
        n_samples=X.shape[0],
        y=y,
        groups=groups_array,
        is_regression=is_regression,
        random_state=random_state,
    )

    predictions: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    fold_results: list[dict[str, Any]] = []
    split_groups = None if cv == "loo" else groups_array
    for train_idx, test_idx in splitter.split(X, y, split_groups):
        model: Any = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        fold_prediction = np.asarray(model.predict(X[test_idx]))
        predictions.append(fold_prediction)
        test_indices.append(test_idx)
        if return_train_score and is_regression:
            fold_results.append(
                {
                    "train_rmsec": _per_target(rmsec, y[train_idx], model.predict(X[train_idx])),
                    "test_rmsep": _per_target(rmsep, y[test_idx], fold_prediction),
                }
            )

    order = np.argsort(np.concatenate(test_indices))
    y_pred = np.concatenate(predictions)[order]
    result: dict[str, Any] = {
        "y_true": y,
        "y_pred": y_pred,
        "cv_scheme": {
            "type": type(splitter).__name__,
            "n_splits": int(splitter.get_n_splits(X, y, split_groups)),
        },
    }
    if is_regression:
        for name in metric_names:
            result[name] = _per_target(_METRICS[name], y, y_pred)
    if return_train_score and is_regression:
        result["fold_results"] = fold_results
    return result


def _validated_targets(y: Any, *, n_samples: int, is_regression: bool) -> np.ndarray:
    y = np.asarray(y)
    if y.shape[:1] != (n_samples,):
        log_and_raise("y has shape {} but X has {} samples", y.shape, n_samples, depth=2)
    if not is_regression:
        if y.ndim != 1:
            log_and_raise("class labels must be 1-D, got shape {}", y.shape, depth=2)
        return y
    if y.ndim not in (1, 2):
        log_and_raise("y must be 1-D or 2-D, got shape {}", y.shape, depth=2)
    try:
        y = y.astype(np.float64)
    except (TypeError, ValueError) as exc:
        log_and_raise("y must be numeric for regression", cause=exc, depth=2)
    if not np.all(np.isfinite(y)):
        log_and_raise("y must contain only finite values", depth=2)
    return y


def _per_target(metric: Any, y_true: np.ndarray, y_pred: np.ndarray) -> float | np.ndarray:
    """Apply a 1-D metric to every target column of 2-D outputs."""
    if y_true.ndim == 1:
        return float(metric(y_true, np.ravel(y_pred)))
    y_pred = np.asarray(y_pred).reshape(y_true.shape)
    return np.array([metric(y_true[:, k], y_pred[:, k]) for k in range(y_true.shape[1])])


def _validate_groups(groups: np.ndarray | None, *, n_samples: int) -> np.ndarray | None:
    if groups is None:
        return None
    groups_array = np.asarray(groups)
    if groups_array.shape != (n_samples,):
        log_and_raise("groups must have shape ({},), got {}", n_samples, groups_array.shape)
    return groups_array


def _make_splitter(
    *,
    cv: int | str,
    n_samples: int,
    y: np.ndarray,
    groups: np.ndarray | None,
    is_regression: bool,
    random_state: int | None,
) -> Any:
    if isinstance(cv, str):
        check_choice(cv, "cv", _SUPPORTED_STRING_CV)
        n_splits = DEFAULT_GROUPED_N_SPLITS
    else:
        n_splits = check_integer(cv, "cv", minimum=2)

    if cv == "loo":
        if n_samples < 2:
            log_and_raise("cv='loo' requires at least 2 samples")
        if groups is not None:
            logger.warning("cv='loo' was requested explicitly and ignores groups")
        return LeaveOneOut()

    if cv == "logo":
        if groups is None:
            log_and_raise("cv='logo' requires groups")
        _check_group_count(groups, minimum=2, label="cv='logo'")
        return LeaveOneGroupOut()

    if groups is not None:
        n_groups = _check_group_count(groups, minimum=2, label="grouped cross-validation")
        if cv == _AUTO_CV:
            return LeaveOneGroupOut() if n_groups < n_splits else GroupKFold(n_splits=n_splits)
        _check_group_count(groups, minimum=n_splits, label=f"{n_splits}-fold grouped validation")
        return GroupKFold(n_splits=n_splits)

    if n_samples < n_splits:
        log_and_raise(
            "{}-fold validation requires at least {} samples, got {}", n_splits, n_splits,
            n_samples
        )
    if is_regression:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    _check_class_counts(y, n_splits=n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _check_class_counts(y: np.ndarray, *, n_splits: int) -> None:
    """Raise if no class can fill every fold; warn if some class cannot."""
    labels, counts = np.unique(y, return_counts=True)
    if counts.max() < n_splits:
        log_and_raise(
            "{}-fold stratified validation needs a class with at least {} samples; "
            "the largest class has {}", n_splits, n_splits, int(counts.max())
        )
    small = labels[counts < n_splits]
    if small.size:
        logger.warning(
            "{}-fold stratified validation: {} class(es) have fewer than {} samples ({}); "
            "some folds will not contain them", n_splits, small.size, n_splits,
            ", ".join(map(str, small)),
        )


def _check_group_count(groups: np.ndarray, *, minimum: int, label: str) -> int:
    n_groups = int(np.unique(groups).size)
    if n_groups < minimum:
        log_and_raise("{} requires at least {} groups, got {}", label, minimum, n_groups)
    return n_groups


def aggregate_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    aggregation: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average observation-level regression results per group.

    Used to report ROI-level figures of merit from pixel-level predictions.

    Parameters
    ----------
    y_true, y_pred:
        Observation-level reference values and predictions, shape ``(n,)``.
    groups:
        Group label of every observation, shape ``(n,)``.
    aggregation:
        Aggregation rule; only ``"mean"`` is supported.

    Returns
    -------
    group_ids : numpy.ndarray of object, shape (n_groups,)
        Group labels in order of first appearance.
    y_true_grouped, y_pred_grouped : numpy.ndarray, shape (n_groups,)
        Group means of ``y_true`` and ``y_pred``.
    """
    check_choice(aggregation, "aggregation", ("mean",))
    try:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        log_and_raise("y_true and y_pred must be numeric", cause=exc)
    groups = np.asarray(groups).reshape(-1)
    if not y_true.size == y_pred.size == groups.size:
        log_and_raise(
            "y_true, y_pred and groups must have the same length, got {}, {} and {}",
            y_true.size,
            y_pred.size,
            groups.size,
        )
    if y_true.size == 0:
        log_and_raise("at least one observation is required")
    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
        log_and_raise("y_true and y_pred must contain only finite values")

    unique_groups, first_index, group_of_sample = np.unique(
        groups, return_index=True, return_inverse=True
    )
    order = np.argsort(first_index)  # sorted-unique position -> first-seen rank
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    group_index = rank[group_of_sample]
    counts = np.bincount(group_index)
    return (
        np.asarray(unique_groups[order], dtype=object),
        np.bincount(group_index, weights=y_true) / counts,
        np.bincount(group_index, weights=y_pred) / counts,
    )


def kennard_stone(
    X: np.ndarray,
    n_select: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Select a calibration subset that spans ``X`` uniformly (Kennard-Stone).

    The two most distant samples are selected first. Then, repeatedly, the
    sample whose distance to its nearest selected sample is largest is added,
    until ``n_select`` samples are selected.

    Parameters
    ----------
    X:
        Spectra or scores, shape ``(n_samples, n_features)``.
    n_select:
        Number of samples to select, ``2 <= n_select < n_samples``.
    metric:
        ``"euclidean"`` (default) or ``"mahalanobis"``. The Mahalanobis
        distance is computed in the space of the non-zero principal
        components of ``X``; it requires ``X`` to have rank below
        ``n_samples - 1``, so pass PCA scores rather than raw spectra with
        more wavelengths than samples.

    Returns
    -------
    calibration_indices : numpy.ndarray, shape (n_select,)
        Selected samples in the order they were selected.
    test_indices : numpy.ndarray, shape (n_samples - n_select,)
        Remaining samples in ascending order.

    Notes
    -----
    Distances are computed one row at a time rather than as the full
    ``n x n`` matrix, so memory is linear in ``n_samples``; run time is
    quadratic because of the initial search for the most distant pair.

    References
    ----------
    R. W. Kennard, L. A. Stone, "Computer aided design of experiments",
    *Technometrics* 11 (1969) 137-148.
    """
    check_choice(metric, "metric", ("euclidean", "mahalanobis"))
    X = as_float_matrix(X, "X").astype(np.float64, copy=False)
    n_samples = X.shape[0]
    n_select = check_integer(n_select, "n_select", minimum=2)
    if n_select >= n_samples:
        log_and_raise("n_select must be smaller than n_samples={}, got {}", n_samples, n_select)
    points = _whitened_scores(X) if metric == "mahalanobis" else X

    # Distances are computed one row at a time, never as the full n x n matrix.
    def distances_to(index: int) -> np.ndarray:
        return cdist(points[index:index + 1], points)[0]

    first, second = _farthest_pair(points)
    selected = [first, second]
    distance_to_selected = np.minimum(distances_to(first), distances_to(second))
    distance_to_selected[selected] = -np.inf
    while len(selected) < n_select:
        best = int(np.argmax(distance_to_selected))
        selected.append(best)
        np.minimum(distance_to_selected, distances_to(best), out=distance_to_selected)
        distance_to_selected[best] = -np.inf
    remaining = np.setdiff1d(np.arange(n_samples), selected)
    return np.array(selected), remaining


def _farthest_pair(points: np.ndarray) -> tuple[int, int]:
    """Indices of the two most distant points, scanning the distances in row blocks."""
    block_rows = max(1, _KS_BLOCK_ELEMENTS // points.shape[0])
    best_distance, best_pair = -1.0, (0, 0)
    for start in range(0, points.shape[0], block_rows):
        block = cdist(points[start:start + block_rows], points)
        row, column = np.unravel_index(int(np.argmax(block)), block.shape)
        if block[row, column] > best_distance:  # strict: keeps the first maximum
            best_distance, best_pair = float(block[row, column]), (start + int(row), int(column))
    return best_pair


def _whitened_scores(X: np.ndarray) -> np.ndarray:
    """Return PCA scores scaled to unit variance; their Euclidean distance is Mahalanobis."""
    U, singular_values, _ = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    tolerance = singular_values[0] * max(X.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank >= X.shape[0] - 1:
        log_and_raise(
            "Mahalanobis distances are degenerate: X has rank {} with {} samples; "
            "use PCA scores with fewer components",
            rank,
            X.shape[0],
            depth=2,
        )
    return U[:, :rank] * np.sqrt(X.shape[0] - 1)


__all__ = [
    "DEFAULT_GROUPED_N_SPLITS",
    "aggregate_regression_predictions",
    "cross_validate",
    "kennard_stone",
]
