"""Input checks shared by the estimators in :mod:`enginir.models`."""

from __future__ import annotations

from typing import Any

import numpy as np

from enginir._checks import as_float_array, check_integer, log_and_raise


def as_target_array(y: Any, *, n_samples: int) -> np.ndarray:
    """Return finite regression targets of shape ``(n_samples,)`` or ``(n_samples, n_targets)``."""
    y = as_float_array(y, "y")
    if y.ndim not in (1, 2):
        log_and_raise("y must be 1-D or 2-D, got shape {}", y.shape, depth=2)
    if y.shape[0] != n_samples:
        log_and_raise("y has {} samples but X has {}", y.shape[0], n_samples, depth=2)
    if not np.all(np.isfinite(y)):
        log_and_raise("y must contain only finite values", depth=2)
    return y


def as_class_labels(y: Any, *, n_samples: int) -> np.ndarray:
    """Return a 1-D array of class labels with one label per sample."""
    labels = np.asarray(y)
    if labels.ndim != 1:
        log_and_raise("class labels must be 1-D, got shape {}", labels.shape, depth=2)
    if labels.shape[0] != n_samples:
        log_and_raise("y has {} labels but X has {} samples", labels.shape[0], n_samples, depth=2)
    return labels


def max_components(X: np.ndarray) -> int:
    """Return the largest usable number of latent components for ``X``.

    Mean-centred ``X`` has rank at most ``min(n_samples - 1, n_wavelengths)``.
    Raises if ``X`` has fewer than two samples.
    """
    if X.shape[0] < 2:
        log_and_raise("at least 2 samples are required, got {}", X.shape[0], depth=2)
    return min(X.shape[0] - 1, X.shape[1])


def check_n_components(value: Any, name: str, *, maximum: int | None = None) -> int:
    """Return ``value`` as an int in ``[1, maximum]`` (no upper bound if ``None``) or raise."""
    value = check_integer(value, name, minimum=1)
    if maximum is not None and value > maximum:
        log_and_raise("{} must be at most {}, got {}", name, maximum, value, depth=2)
    return value
