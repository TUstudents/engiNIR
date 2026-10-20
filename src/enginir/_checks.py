"""Shared argument and array validation helpers.

engiNIR logs every error before raising it so that the persisted log is a
complete error report (see :mod:`enginir._logging`). The helpers in this module
implement that rule once: :func:`log_and_raise` writes the ERROR record and
raises, and the ``check_*`` / ``as_*`` helpers use it to validate common
parameter and array shapes.

Messages use :meth:`str.format` placeholders (``{}``, ``{!r}``), which is also
the placeholder style Loguru uses, so the logged and raised texts are identical.
"""

from __future__ import annotations

from collections.abc import Collection
from numbers import Integral, Real
from typing import Any, NoReturn

import numpy as np
from loguru import logger

__all__ = [
    "as_float_array",
    "as_float_matrix",
    "as_float_vector",
    "check_choice",
    "check_integer",
    "check_n_features",
    "check_real",
    "log_and_raise",
]


def log_and_raise(
    message: str,
    *args: Any,
    error: type[Exception] = ValueError,
    cause: BaseException | None = None,
    depth: int = 1,
) -> NoReturn:
    """Log ``message`` at ERROR level, then raise ``error`` with the same text.

    Parameters
    ----------
    message:
        Message template with ``{}`` placeholders filled from ``args``.
    *args:
        Values substituted into ``message``.
    error:
        Exception class to raise. Defaults to :class:`ValueError`.
    cause:
        Optional exception to chain with ``raise ... from cause``.
    depth:
        Stack depth used to attribute the log record. The default attributes
        it to the direct caller; helpers that validate on behalf of their own
        caller pass ``depth=2``.
    """
    logger.opt(depth=depth).error(message, *args)
    text = message.format(*args) if args else message
    if cause is None:
        raise error(text)
    raise error(text) from cause


def check_integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    """Return ``value`` as ``int`` after checking it is an integer ``>= minimum``.

    Booleans are rejected even though ``bool`` subclasses ``int``.
    """
    bound = "" if minimum is None else f" >= {minimum}"
    if isinstance(value, bool) or not isinstance(value, Integral):
        log_and_raise("{} must be an integer{}, got {!r}", name, bound, value, depth=2)
    result = int(value)
    if minimum is not None and result < minimum:
        log_and_raise("{} must be an integer{}, got {!r}", name, bound, value, depth=2)
    return result


def check_real(value: Any, name: str, *, positive: bool = False) -> float:
    """Return ``value`` as a finite ``float``, optionally requiring ``> 0``.

    Booleans are rejected even though ``bool`` subclasses ``int``.
    """
    requirement = "a positive real number" if positive else "a finite real number"
    if isinstance(value, bool) or not isinstance(value, Real):
        log_and_raise("{} must be {}, got {!r}", name, requirement, value, depth=2)
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        log_and_raise("{} must be {}, got {!r}", name, requirement, value, depth=2)
    return result


def check_choice(value: Any, name: str, choices: Collection[str]) -> str:
    """Return ``value`` after checking it is one of ``choices``."""
    try:
        valid = value in choices
    except TypeError:  # unhashable value tested against a set or dict
        valid = False
    if not valid:
        log_and_raise("{} must be one of {}, got {!r}", name, sorted(choices), value, depth=2)
    return value


def as_float_array(value: Any, name: str) -> np.ndarray:
    """Return ``value`` as a floating-point array.

    ``float32`` and ``float64`` are kept so that large hyperspectral cubes are
    not silently doubled in memory. Integer, boolean and ``float16`` inputs
    are converted to ``float32``, which represents raw sensor counts
    (``uint16``) exactly. Complex, string, date and object arrays (e.g. lists
    containing ``None``) are rejected.
    """
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as exc:
        log_and_raise("{} must be a numeric array", name, cause=exc, depth=2)
    if arr.dtype.kind not in "biuf":
        log_and_raise("{} must be a real numeric array, got dtype {}", name, arr.dtype, depth=2)
    if arr.dtype.kind != "f" or arr.dtype.itemsize < 4:
        arr = arr.astype(np.float32)
    return arr


def as_float_matrix(value: Any, name: str) -> np.ndarray:
    """Return a non-empty, finite, 2-D floating array (see :func:`as_float_array`)."""
    arr = as_float_array(value, name)
    if arr.ndim != 2:
        log_and_raise("{} must be 2-D, got shape {}", name, arr.shape, depth=2)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        log_and_raise(
            "{} must contain at least one sample and one feature, got shape {}",
            name,
            arr.shape,
            depth=2,
        )
    if not np.all(np.isfinite(arr)):
        log_and_raise("{} must contain only finite values", name, depth=2)
    return arr


def as_float_vector(value: Any, name: str, *, size: int | None = None) -> np.ndarray:
    """Return a finite 1-D floating array, optionally of length ``size``.

    The result never aliases ``value``, so callers may store it as fitted state.
    """
    arr = as_float_array(value, name)
    if arr.ndim != 1:
        log_and_raise("{} must be 1-D, got shape {}", name, arr.shape, depth=2)
    if size is not None and arr.shape[0] != size:
        log_and_raise("{} has {} values but X has {} features", name, arr.shape[0], size, depth=2)
    if not np.all(np.isfinite(arr)):
        log_and_raise("{} must contain only finite values", name, depth=2)
    return arr.copy()


def check_n_features(X: np.ndarray, n_features_in: int, owner: str) -> None:
    """Check that ``X`` has the number of columns ``owner`` was fitted with."""
    if X.shape[1] != n_features_in:
        log_and_raise(
            "X has {} features, but {} was fitted with {} features",
            X.shape[1],
            owner,
            n_features_in,
            depth=2,
        )
