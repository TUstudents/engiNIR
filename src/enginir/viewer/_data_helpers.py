"""Shared data-shape helpers for viewer-side validation."""

from __future__ import annotations

import numpy as np

from .._logging import get_logger

logger = get_logger(__name__)

WAVELENGTH_ATOL_NM = 1e-6


def wavelength_axes_match(left: np.ndarray, right: np.ndarray) -> bool:
    """Return ``True`` when two wavelength axes have matching finite values.

    Non-1-D or non-finite arrays are treated as incompatible axes instead of
    raising because callers use this helper as a compatibility predicate.
    """
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.ndim != 1 or right_arr.ndim != 1:
        logger.warning(
            "Wavelength axis compatibility check expected 1-D arrays, got {} and {}",
            left_arr.shape,
            right_arr.shape,
        )
        return False
    if not np.all(np.isfinite(left_arr)) or not np.all(np.isfinite(right_arr)):
        logger.warning("Wavelength axis compatibility check received non-finite values")
        return False
    return left_arr.shape == right_arr.shape and np.allclose(
        left_arr,
        right_arr,
        atol=WAVELENGTH_ATOL_NM,
        rtol=0.0,
    )


__all__ = ["WAVELENGTH_ATOL_NM", "wavelength_axes_match"]
