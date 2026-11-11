"""Input checks shared by the estimators in :mod:`enginir.models`."""

from __future__ import annotations

from typing import Any

import numpy as np

from enginir._checks import as_float_array, check_integer, log_and_raise
from enginir._logging import get_logger

logger = get_logger(__name__)

#: Scaling methods of X and Y (sample-set-model-run contract §7); every one centres.
SCALINGS = ("center", "uv", "pareto")
_EPS = 1e-10


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


def check_scaling(value: Any, name: str) -> str:
    """Return ``value`` if it is one of :data:`SCALINGS`, else raise."""
    if not isinstance(value, str) or value not in SCALINGS:
        log_and_raise("{} must be one of {}, got {!r}", name, list(SCALINGS), value, depth=2)
    return value


def resolve_scalings(scale: bool, x_scaling: str | None,
                     y_scaling: str | None) -> tuple[str, str]:
    """The X and Y scaling of an estimator: from ``x_scaling``/``y_scaling``, else ``scale``.

    ``scale=True`` (the default) means ``uv``, ``scale=False`` ``center``. A
    scaling left as ``None`` takes the value of ``scale``; ``scale=False``
    together with an explicit scaling is ambiguous and refused.
    """
    if not isinstance(scale, (bool, np.bool_)):
        log_and_raise("scale must be True or False, got {!r}", scale, depth=2)
    if not scale and (x_scaling is not None or y_scaling is not None):
        log_and_raise("Give either scale=False or x_scaling/y_scaling, not both", depth=2)
    implied = "uv" if scale else "center"
    return (check_scaling(implied if x_scaling is None else x_scaling, "x_scaling"),
            check_scaling(implied if y_scaling is None else y_scaling, "y_scaling"))


def scaling_statistics(values: np.ndarray, scaling: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean and divisor of every column of ``values`` for ``scaling`` (float64).

    ``center`` divides by 1, ``uv`` by the standard deviation with N - 1,
    ``pareto`` by its square root. A standard deviation below ``1e-10`` (a
    constant column) is replaced by 1, with a warning.
    """
    matrix = values.reshape(values.shape[0], -1)
    mean = matrix.mean(axis=0, dtype=np.float64)
    if scaling == "center":
        return mean, np.ones_like(mean)
    std = matrix.std(axis=0, ddof=1, dtype=np.float64)
    constant = std < _EPS
    if np.any(constant):
        logger.warning("{} constant column(s) are only centered, not scaled ({})",
                       int(constant.sum()), scaling)
        std = np.where(constant, 1.0, std)
    return mean, (std if scaling == "uv" else np.sqrt(std))
