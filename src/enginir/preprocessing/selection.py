"""Band selection: keep a fixed subset of the spectral bands (columns)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from enginir._checks import log_and_raise
from enginir._logging import get_logger

logger = get_logger(__name__)


class BandSelection(TransformerMixin, BaseEstimator):
    """Keep the bands (columns) at ``indices``, in increasing order.

    Stateless: ``fit`` only checks the input. The last step of a
    :class:`~enginir.preprocessing.PreprocessingRecipe` with ``bands_keep`` or
    ``bands_exclude``; the recipe computes the indices from its ranges.

    Parameters
    ----------
    indices:
        Column indices to keep, strictly increasing, at least one.
    """

    def __init__(self, indices: Any = None) -> None:
        self.indices = indices

    def _columns(self, n_bands: int) -> np.ndarray:
        columns = np.asarray(self.indices if self.indices is not None else [], dtype=np.int64)
        if columns.ndim != 1 or columns.size == 0:
            log_and_raise("BandSelection needs at least one band index", depth=3)
        if np.any(np.diff(columns) <= 0):
            log_and_raise("BandSelection indices must be strictly increasing", depth=3)
        if columns[0] < 0 or columns[-1] >= n_bands:
            log_and_raise("BandSelection index {} is outside the {} bands of the input",
                          int(columns[-1] if columns[-1] >= n_bands else columns[0]),
                          n_bands, depth=3)
        return columns

    def fit(self, X: Any, y: Any = None) -> BandSelection:
        """Check that the indices fit ``X``; nothing is learned."""
        self._columns(np.shape(X)[-1])
        self.n_features_in_ = int(np.shape(X)[-1])
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Stateless: usable without ``fit`` (e.g. at the end of a fitted pipeline)."""
        return True

    def transform(self, X: Any) -> np.ndarray:
        """Return the selected columns of ``X`` (dtype kept)."""
        X = np.asarray(X)
        if X.ndim != 2:
            log_and_raise("BandSelection expects a 2-D array, got shape {}", X.shape)
        return X[:, self._columns(X.shape[1])]


__all__ = ["BandSelection"]
