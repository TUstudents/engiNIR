"""Baseline-removal and smoothing transformers: Detrend and Smoother."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import uniform_filter1d
from .._estimator import EnginirEstimator
from sklearn.base import TransformerMixin

from enginir._checks import (
    as_float_matrix,
    as_float_vector,
    check_choice,
    check_integer,
    check_real,
    log_and_raise,
)
from enginir._logging import get_logger

logger = get_logger(__name__)

_SMOOTHER_METHODS = ("moving_average", "whittaker")
_WHITTAKER_BLOCK_ROWS = 1024  # rows solved per float64 block


class Detrend(TransformerMixin, EnginirEstimator):
    """Polynomial baseline removal.

    A polynomial of degree ``order`` is fitted to every spectrum by least
    squares and subtracted, removing offsets (``order=0``), tilted
    (``order=1``) or curved (``order>=2``) baselines.

    Parameters
    ----------
    order:
        Polynomial degree of the baseline. Default ``1`` (linear).
    wavelengths:
        Optional abscissa, shape ``(n_wavelengths,)``, for the polynomial.
        If ``None`` (default), equally spaced band indices are used. The
        result only differs from the default for unevenly spaced wavelengths.

    Notes
    -----
    The abscissa is rescaled to ``[-1, 1]`` before fitting. This does not
    change the fitted baseline, but keeps the least-squares problem well
    conditioned for raw wavelength values and ``float32`` input. The fit runs
    in the dtype of ``X``; no higher-precision copy of the spectra is made.

    References
    ----------
    R. J. Barnes, M. S. Dhanoa, S. J. Lister, "Standard normal variate
    transformation and de-trending of near-infrared diffuse reflectance
    spectra", *Applied Spectroscopy* 43 (1989) 772-777.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0, 3.0, 4.0]])
    >>> np.allclose(Detrend(order=1).fit_transform(X), 0.0)
    True
    """

    def __init__(self, order: int = 1, wavelengths: np.ndarray | None = None) -> None:
        self.order = order
        self.wavelengths = wavelengths

    def fit(self, X: np.ndarray, y: object = None) -> Detrend:
        """Return ``self`` unchanged; detrending has no fitted state."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Subtract the fitted polynomial baseline from every spectrum (row) of ``X``.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.

        Returns
        -------
        numpy.ndarray
            Detrended spectra with the same shape and floating dtype as ``X``.
        """
        X = as_float_matrix(X, "X")
        order = check_integer(self.order, "order", minimum=0)
        n_wavelengths = X.shape[1]
        if order >= n_wavelengths:
            log_and_raise(
                "order={} must be smaller than the number of wavelengths ({})",
                order,
                n_wavelengths,
            )
        if self.wavelengths is None:
            abscissa = np.arange(n_wavelengths, dtype=np.float64)
        else:
            abscissa = as_float_vector(self.wavelengths, "wavelengths", size=n_wavelengths)
            abscissa = abscissa.astype(np.float64, copy=False)
        span = np.ptp(abscissa)
        if span > 0:
            abscissa = 2.0 * (abscissa - abscissa.min()) / span - 1.0
        elif order > 0:
            log_and_raise("wavelengths must not all be equal for order={}", order)
        # Least squares through the small projector pinv(V), computed in float64; the
        # baseline buffer becomes the output, so X is never copied as a whole.
        vandermonde = np.vander(abscissa, N=order + 1)
        projector = np.linalg.pinv(vandermonde).astype(X.dtype)
        out = (X @ projector.T) @ vandermonde.T.astype(X.dtype)
        np.subtract(X, out, out=out)
        return out


class Smoother(TransformerMixin, EnginirEstimator):
    """Moving-average or Whittaker smoothing of spectra.

    Parameters
    ----------
    method:
        ``"moving_average"`` (default) replaces each point by the mean of a
        centred window of ``window_length`` points, repeating the edge values
        at the spectrum ends. ``"whittaker"`` solves the penalised least-squares
        problem ``min_z |x - z|^2 + lam |D_d z|^2``, where ``D_d`` is the
        ``order``-th difference operator.
    window_length:
        Odd window size for ``"moving_average"``. Default ``5``.
    lam:
        Positive smoothing strength for ``"whittaker"``; larger is smoother.
        Default ``1e4``.
    order:
        Difference order ``d >= 1`` of the Whittaker penalty. Default ``2``.

    References
    ----------
    P. H. C. Eilers, "A perfect smoother", *Analytical Chemistry* 75 (2003)
    3631-3636.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).normal(size=(3, 50))
    >>> Smoother(method="whittaker", lam=1e3).fit_transform(X).shape
    (3, 50)
    """

    def __init__(
        self,
        method: str = "moving_average",
        window_length: int = 5,
        lam: float = 1e4,
        order: int = 2,
    ) -> None:
        self.method = method
        self.window_length = window_length
        self.lam = lam
        self.order = order

    def fit(self, X: np.ndarray, y: object = None) -> Smoother:
        """Return ``self`` unchanged; smoothing has no fitted state."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Smooth every spectrum (row) of ``X``.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.

        Returns
        -------
        numpy.ndarray
            Smoothed spectra with the same shape and floating dtype as ``X``.
        """
        check_choice(self.method, "method", _SMOOTHER_METHODS)
        X = as_float_matrix(X, "X")
        if self.method == "moving_average":
            return self._moving_average(X)
        return self._whittaker(X)

    def _moving_average(self, X: np.ndarray) -> np.ndarray:
        window_length = check_integer(self.window_length, "window_length", minimum=1)
        if window_length % 2 == 0:
            log_and_raise("window_length must be odd, got {}", window_length)
        return uniform_filter1d(X, size=window_length, axis=1, mode="nearest")

    def _whittaker(self, X: np.ndarray) -> np.ndarray:
        lam = check_real(self.lam, "lam", positive=True)
        order = check_integer(self.order, "order", minimum=1)
        n_wavelengths = X.shape[1]
        if order >= n_wavelengths:
            log_and_raise(
                "order={} must be smaller than the number of wavelengths ({})",
                order,
                n_wavelengths,
            )
        difference = sp.eye(n_wavelengths, format="csr")  # CSR: row slicing below
        for _ in range(order):
            difference = difference[1:] - difference[:-1]  # type: ignore[index]
        system = sp.eye(n_wavelengths, format="csc") + lam * (difference.T @ difference)
        # The system has a condition number of about 16 * lam, far too large for
        # float32. Factorise and solve in float64, but only one block of rows at a
        # time, so that X is never copied to float64 as a whole.
        solver = spla.splu(system.tocsc())
        out = np.empty_like(X)
        for start in range(0, X.shape[0], _WHITTAKER_BLOCK_ROWS):
            block = X[start:start + _WHITTAKER_BLOCK_ROWS]
            out[start:start + block.shape[0]] = solver.solve(block.T.astype(np.float64)).T
        return out


__all__ = ["Detrend", "Smoother"]
