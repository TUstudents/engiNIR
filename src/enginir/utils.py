"""Reshaping between image cubes and spectra matrices."""

from __future__ import annotations

from typing import Any

import numpy as np

from enginir._checks import check_integer, log_and_raise


def flatten_cube(X_3d: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Reshape an image cube into a matrix with one spectrum per pixel.

    Parameters
    ----------
    X_3d:
        Image cube, shape ``(rows, cols, n_wavelengths)``.

    Returns
    -------
    X_2d : numpy.ndarray, shape (rows * cols, n_wavelengths)
        Pixels in row-major order. This is a view of ``X_3d`` when possible,
        so modifying it in place also modifies the cube; copy it first if
        that is not wanted.
    spatial_shape : tuple of int
        ``(rows, cols)``, for :func:`unflatten_cube`.

    Raises
    ------
    ValueError
        If ``X_3d`` is not a non-empty 3-D array.

    Examples
    --------
    >>> import numpy as np
    >>> cube = np.zeros((4, 5, 10))
    >>> X, shape = flatten_cube(cube)
    >>> X.shape, shape
    ((20, 10), (4, 5))
    >>> unflatten_cube(X, shape).shape
    (4, 5, 10)
    """
    array = _non_empty(X_3d, "X_3d")
    if array.ndim != 3:
        log_and_raise("X_3d must have shape (rows, cols, n_wavelengths), got {}", array.shape)
    rows, cols, n_wavelengths = array.shape
    return array.reshape(rows * cols, n_wavelengths), (rows, cols)


def unflatten_cube(X_2d: np.ndarray, spatial_shape: tuple[int, int]) -> np.ndarray:
    """Reshape a matrix with one row per pixel back into an image cube.

    Parameters
    ----------
    X_2d:
        Pixel spectra (or per-pixel predictions), shape
        ``(rows * cols, n_values)``, or one value per pixel, shape
        ``(rows * cols,)`` (e.g. the output of a PLS1 ``predict``).
    spatial_shape:
        ``(rows, cols)`` as returned by :func:`flatten_cube`.

    Returns
    -------
    numpy.ndarray, shape (rows, cols, n_values) or (rows, cols)
        A view of ``X_2d`` when possible; 2-D for 1-D input, ready for
        :func:`enginir.visualization.plot_image_map`.

    Raises
    ------
    ValueError
        If ``X_2d`` is not a non-empty 1-D or 2-D array, ``spatial_shape`` is
        not a pair of positive integers, or ``rows * cols`` differs from the
        number of rows of ``X_2d``.
    """
    array = _non_empty(X_2d, "X_2d")
    if array.ndim not in (1, 2):
        log_and_raise("X_2d must be 1-D or 2-D, got shape {}", array.shape)
    try:
        rows, cols = spatial_shape
    except (TypeError, ValueError) as exc:
        log_and_raise("spatial_shape must be a (rows, cols) pair, got {!r}", spatial_shape,
                      cause=exc)
    rows = check_integer(rows, "rows", minimum=1)
    cols = check_integer(cols, "cols", minimum=1)
    if array.shape[0] != rows * cols:
        log_and_raise("X_2d has {} rows but spatial_shape {} needs {}", array.shape[0],
                      (rows, cols), rows * cols)
    return array.reshape(rows, cols, *array.shape[1:])


def _non_empty(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        log_and_raise("{} must not be empty", name, depth=2)
    return array


__all__ = ["flatten_cube", "unflatten_cube"]
