"""Savitzky-Golay smoothing and derivative transformer."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from .._estimator import EnginirEstimator
from sklearn.base import TransformerMixin

from enginir._checks import as_float_matrix, check_integer, check_real, log_and_raise
from enginir._logging import get_logger

logger = get_logger(__name__)


class SavitzkyGolay(TransformerMixin, EnginirEstimator):
    """Savitzky-Golay smoothing and derivative filter.

    A polynomial of degree ``polyorder`` is fitted by least squares to every
    window of ``window_length`` consecutive wavelengths, and the value (or
    derivative) of that polynomial at the window centre replaces the centre
    point. The filter is applied to every spectrum independently using
    :func:`scipy.signal.savgol_filter` with its default ``mode="interp"``
    edge handling.

    Parameters
    ----------
    window_length:
        Number of wavelengths in the filter window. Must be a positive odd
        integer no larger than the number of wavelengths. Default ``11``.
    polyorder:
        Degree of the fitted polynomial, ``< window_length``. Default ``2``.
    deriv:
        Derivative order, ``0`` (smoothing only) to ``polyorder``.
        Default ``1``.
    delta:
        Spacing between adjacent wavelengths; the filter assumes evenly
        spaced bands. Derivatives are expressed per
        unit of ``delta``: the default ``1.0`` gives derivatives per band;
        passing the wavelength step in nm gives derivatives per nm. Ignored
        when ``deriv == 0``.

    References
    ----------
    A. Savitzky, M. J. E. Golay, "Smoothing and differentiation of data by
    simplified least squares procedures", *Analytical Chemistry* 36 (1964)
    1627-1639.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.tile(np.arange(15.0) ** 2, (2, 1))
    >>> d1 = SavitzkyGolay(window_length=5, polyorder=2, deriv=1).fit_transform(X)
    >>> np.allclose(d1, 2 * np.arange(15.0))
    True
    """

    def __init__(
        self,
        window_length: int = 11,
        polyorder: int = 2,
        deriv: int = 1,
        delta: float = 1.0,
    ) -> None:
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta

    def fit(self, X: np.ndarray, y: object = None) -> SavitzkyGolay:
        """Return ``self`` unchanged; the filter has no fitted state."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Filter every spectrum (row) of ``X``.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.

        Returns
        -------
        numpy.ndarray
            Smoothed spectra or derivatives with the same shape and floating
            dtype as ``X``.

        Raises
        ------
        ValueError
            If the filter parameters are inconsistent with each other or with
            the number of wavelengths.
        """
        X = as_float_matrix(X, "X")
        window_length = check_integer(self.window_length, "window_length", minimum=1)
        polyorder = check_integer(self.polyorder, "polyorder", minimum=0)
        deriv = check_integer(self.deriv, "deriv", minimum=0)
        delta = check_real(self.delta, "delta", positive=True)
        if window_length % 2 == 0:
            log_and_raise("window_length must be odd, got {}", window_length)
        if window_length > X.shape[1]:
            log_and_raise(
                "window_length={} exceeds the number of wavelengths ({})",
                window_length,
                X.shape[1],
            )
        if polyorder >= window_length:
            log_and_raise(
                "polyorder={} must be smaller than window_length={}", polyorder, window_length
            )
        if deriv > polyorder:
            log_and_raise(
                "deriv={} must not exceed polyorder={} (higher derivatives are zero)",
                deriv,
                polyorder,
            )
        filtered = savgol_filter(
            X,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
            axis=1,
        )
        return np.asarray(filtered, dtype=X.dtype)


__all__ = ["SavitzkyGolay"]
