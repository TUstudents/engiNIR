"""Baseline-removal and smoothing transformers: AsLSBaseline, Detrend and Smoother."""

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
_ASLS_BLOCK_ROWS = 4096  # spectra solved together, in float64
_ASLS_DIRECTIONS = ("up", "down")


def _second_difference_bands(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Main, first and second sub-diagonals of DᵀD for the second difference D (n points).

    ``sub1[i]`` is entry ``(i, i - 1)`` and ``sub2[i]`` entry ``(i, i - 2)``
    (zero where undefined).
    """
    if n < 8:  # few bands: the edge rows overlap, take the bands of the dense product
        D = np.diff(np.eye(n), n=2, axis=0)
        dtd = D.T @ D
        return np.diag(dtd).copy(), np.r_[0.0, np.diag(dtd, -1)], np.r_[0.0, 0.0, np.diag(dtd, -2)]
    main = np.full(n, 6.0)
    main[[0, -1]], main[[1, -2]] = 1.0, 5.0
    sub1 = np.full(n, -4.0)
    sub1[0], sub1[[1, -1]] = 0.0, -2.0
    sub2 = np.ones(n)
    sub2[:2] = 0.0
    return main, sub1, sub2


def _solve_pentadiagonal(diag: np.ndarray, sub1: np.ndarray, sub2: np.ndarray,
                         rhs: np.ndarray) -> np.ndarray:
    """Solve symmetric positive-definite pentadiagonal systems by LDLᵀ, one per column.

    ``diag`` and ``rhs`` are ``(n, columns)`` (band-major, so that every step
    reads contiguous rows); ``sub1``/``sub2`` (shared by all systems) are the
    first and second sub-diagonals as in :func:`_second_difference_bands`.
    O(n) per system, vectorized over the systems.
    """
    n = diag.shape[0]
    d = np.empty_like(diag)
    l1 = np.zeros_like(diag)
    l2 = np.zeros_like(diag)
    y = np.empty_like(rhs)
    for i in range(n):
        value = diag[i].copy()
        right = rhs[i].copy()
        if i >= 2:
            np.divide(sub2[i], d[i - 2], out=l2[i])
            value -= l2[i] * l2[i] * d[i - 2]
            right -= l2[i] * y[i - 2]
        if i >= 1:
            coupling = sub1[i] - l2[i] * d[i - 2] * l1[i - 1] if i >= 2 else sub1[i]
            np.divide(coupling, d[i - 1], out=l1[i])
            value -= l1[i] * l1[i] * d[i - 1]
            right -= l1[i] * y[i - 1]
        d[i] = value
        y[i] = right
    x = y
    x /= d
    for i in range(n - 2, -1, -1):
        x[i] -= l1[i + 1] * x[i + 1]
        if i + 2 < n:
            x[i] -= l2[i + 2] * x[i + 2]
    return x


class AsLSBaseline(TransformerMixin, EnginirEstimator):
    """Asymmetric least-squares (AsLS) baseline correction.

    For every spectrum z a smooth baseline b is fitted by minimizing
    Σ wᵢ (zᵢ − bᵢ)² + λ Σ (Δ²b)ᵢ, where Δ² is the second difference. The
    weights start at 1 and are then set from the previous baseline:
    with ``direction="up"`` (peaks point up, e.g. absorbance) wᵢ = p where
    zᵢ > bᵢ and 1 − p otherwise, so the baseline lies under the spectrum;
    with ``direction="down"`` (absorption bands are dips, e.g. reflectance)
    the step is AsLS of −z, negated (wᵢ = p where zᵢ < bᵢ, 1 − p otherwise),
    and the baseline lies over the spectrum. The transform returns z − b.
    Exactly ``n_iter`` solves are done; the solver works in float64.

    Parameters
    ----------
    lam:
        Stiffness λ of the baseline, positive. Default ``1e4``. A camera
        with more bands needs a larger λ for the same baseline (roughly with
        the fourth power of the band count); check the baseline in a plot.
    p:
        Asymmetry, ``0 < p < 0.5``. Default ``0.001``.
    n_iter:
        Number of weighted solves, ``>= 1``. Default ``10``.
    direction:
        ``"up"`` (default) or ``"down"``, see above.

    References
    ----------
    P. H. C. Eilers, H. F. M. Boelens, "Baseline correction with asymmetric
    least squares smoothing", Leiden University Medical Centre report (2005).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 200)
    >>> z = 0.5 + x + np.exp(-((x - 0.5) / 0.02) ** 2)
    >>> corrected = AsLSBaseline(lam=1e5).fit_transform(z[None, :])
    >>> bool(abs(corrected[0, 0]) < 0.05 and corrected[0, 100] > 0.9)
    True
    """

    def __init__(self, lam: float = 1e4, p: float = 0.001, n_iter: int = 10,
                 direction: str = "up") -> None:
        self.lam = lam
        self.p = p
        self.n_iter = n_iter
        self.direction = direction

    def fit(self, X: np.ndarray, y: object = None) -> AsLSBaseline:
        """Return ``self`` unchanged; the baseline is fitted per spectrum in ``transform``."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Subtract the AsLS baseline of every spectrum (row) of ``X``.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.

        Returns
        -------
        numpy.ndarray
            Baseline-corrected spectra, same shape and floating dtype as ``X``.
        """
        lam = check_real(self.lam, "lam", positive=True)
        p = check_real(self.p, "p")
        if not 0.0 < p < 0.5:
            log_and_raise("p must be between 0 and 0.5, got {}", p)
        n_iter = check_integer(self.n_iter, "n_iter", minimum=1)
        check_choice(self.direction, "direction", _ASLS_DIRECTIONS)
        X = as_float_matrix(X, "X")
        n = X.shape[1]
        if n < 3:
            log_and_raise("AsLS needs at least 3 wavelengths, got {}", n)
        main, sub1, sub2 = _second_difference_bands(n)
        sub1, sub2 = lam * sub1, lam * sub2
        sign = 1.0 if self.direction == "up" else -1.0
        out = np.empty_like(X)
        stiffness = lam * main[:, None]
        for start in range(0, X.shape[0], _ASLS_BLOCK_ROWS):
            # band-major float64 block: one system per column
            z = np.ascontiguousarray(X[start:start + _ASLS_BLOCK_ROWS].T, dtype=np.float64)
            weights = np.ones_like(z)
            baseline = z
            for iteration in range(n_iter):
                baseline = _solve_pentadiagonal(weights + stiffness, sub1, sub2, weights * z)
                if iteration < n_iter - 1:
                    weights = np.where(sign * (z - baseline) > 0, p, 1.0 - p)
            out[start:start + z.shape[1]] = (z - baseline).T
        return out


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


__all__ = ["AsLSBaseline", "Detrend", "Smoother"]
