"""Scatter-correction transformers: SNV, MSC and EMSC.

All three remove multiplicative and additive light-scattering effects from
diffuse-reflectance spectra. SNV works on each spectrum alone; MSC and EMSC
correct each spectrum against a reference spectrum learned in ``fit``.
"""

from __future__ import annotations

import numpy as np
from .._estimator import EnginirEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from enginir._checks import (
    as_float_matrix,
    as_float_vector,
    check_choice,
    check_integer,
    check_n_features,
    log_and_raise,
)
from enginir._logging import get_logger

logger = get_logger(__name__)

_EPS = 1e-10
_EMSC_METHODS = ("mean", "median")


def _clamp_near_zero(values: np.ndarray, label: str) -> np.ndarray:
    """Replace entries with ``|value| < _EPS`` by ``+/-_EPS`` (keeping the sign).

    Used for multiplicative scatter terms that are divided by. One aggregated
    warning is emitted so that large image cubes do not flood the log.
    """
    near_zero = np.abs(values) < _EPS
    n_near_zero = int(np.count_nonzero(near_zero))
    if n_near_zero == 0:
        return values
    logger.warning(
        "{}: {} spectrum(a) have a near-zero multiplicative term (|b| < {}); "
        "clamping to +/-{} to avoid division by zero",
        label,
        n_near_zero,
        _EPS,
        _EPS,
    )
    return np.where(near_zero, np.where(values >= 0, _EPS, -_EPS), values).astype(
        values.dtype, copy=False
    )


class SNV(TransformerMixin, EnginirEstimator):
    """Standard Normal Variate scatter correction.

    Each spectrum is centred on its own mean and divided by its own sample
    standard deviation:

    .. math:: x_{\\mathrm{snv}} = \\frac{x - \\bar{x}}{s_x},
              \\qquad s_x = \\sqrt{\\tfrac{1}{p-1} \\sum_k (x_k - \\bar{x})^2}

    where ``p`` is the number of wavelengths. SNV is a per-spectrum operation,
    so :meth:`fit` learns nothing. Constant spectra (standard deviation at the
    rounding level of their mean, or below ``1e-10``) are set to zero and a
    warning is logged.

    References
    ----------
    R. J. Barnes, M. S. Dhanoa, S. J. Lister, "Standard normal variate
    transformation and de-trending of near-infrared diffuse reflectance
    spectra", *Applied Spectroscopy* 43 (1989) 772-777.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    >>> SNV().fit_transform(X)
    array([[-1.,  0.,  1.],
           [-1.,  0.,  1.]])
    """

    def fit(self, X: np.ndarray, y: object = None) -> SNV:
        """Return ``self`` unchanged; SNV has no fitted state."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply SNV to every spectrum (row) of ``X``.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)`` with
            ``n_wavelengths >= 2``.

        Returns
        -------
        numpy.ndarray
            Corrected spectra with the same shape and floating dtype as ``X``.
        """
        X = as_float_matrix(X, "X")
        if X.shape[1] < 2:
            log_and_raise("SNV needs at least 2 wavelengths, got {}", X.shape[1])
        mean = X.mean(axis=1, keepdims=True)
        out = X - mean  # the single working copy; everything below is in place
        # Sample standard deviation from the centred rows, without a second pass over X.
        std = np.sqrt(np.einsum("ij,ij->i", out, out) / (X.shape[1] - 1))[:, np.newaxis]
        # A constant spectrum has a computed std of the order of eps * |mean|, not 0.
        rounding_level = np.finfo(X.dtype).eps * np.sqrt(X.shape[1]) * np.abs(mean)
        constant = std <= np.maximum(rounding_level, _EPS)
        n_constant = int(np.count_nonzero(constant))
        if n_constant:
            logger.warning("SNV: {} constant spectrum(a) set to zero", n_constant)
            std[constant] = 1.0
        out /= std
        if n_constant:
            out[constant[:, 0]] = 0.0
        return out


class MSC(TransformerMixin, EnginirEstimator):
    """Multiplicative Scatter Correction.

    Each spectrum ``x`` is regressed on a reference spectrum ``r`` by ordinary
    least squares, ``x = a + b r + e``, and replaced by ``(x - a) / b``. The
    reference is the mean training spectrum unless one is supplied.

    Parameters
    ----------
    reference:
        Optional reference spectrum, shape ``(n_wavelengths,)``. If ``None``
        (default), the mean of the training spectra is used.

    Attributes
    ----------
    reference_ : numpy.ndarray, shape (n_wavelengths,)
        Reference spectrum used by :meth:`transform`.
    n_features_in_ : int
        Number of wavelengths seen during :meth:`fit`.

    Notes
    -----
    Spectra whose slope ``b`` is (near-)zero are divided by ``+/-1e-10``
    instead and a warning is logged.

    References
    ----------
    P. Geladi, D. MacDougall, H. Martens, "Linearization and scatter-correction
    for near-infrared reflectance spectra of meat", *Applied Spectroscopy* 39
    (1985) 491-500.
    """

    def __init__(self, reference: np.ndarray | None = None) -> None:
        self.reference = reference

    def fit(self, X: np.ndarray, y: object = None) -> MSC:
        """Determine the reference spectrum.

        Parameters
        ----------
        X:
            Training spectra, shape ``(n_samples, n_wavelengths)``.
        y:
            Ignored.

        Raises
        ------
        ValueError
            If the reference spectrum is constant, which makes the slope ``b``
            undefined.
        """
        X = as_float_matrix(X, "X")
        if self.reference is None:
            reference = X.mean(axis=0)
        else:
            reference = as_float_vector(self.reference, "reference", size=X.shape[1])
        if np.ptp(reference) < _EPS:
            log_and_raise("MSC reference spectrum must not be constant")
        self.reference_ = reference
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply MSC to every spectrum (row) of ``X``.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.

        Returns
        -------
        numpy.ndarray
            Corrected spectra with the same shape and floating dtype as ``X``.
        """
        check_is_fitted(self, "reference_")
        X = as_float_matrix(X, "X")
        check_n_features(X, self.n_features_in_, "MSC")
        # Closed-form simple linear regression of every row of X on r.
        reference = self.reference_.astype(X.dtype, copy=False)
        r_centred = reference - reference.mean()
        b = (X @ r_centred) / np.dot(r_centred, r_centred)
        a = X.mean(axis=1) - b * reference.mean()
        b = _clamp_near_zero(b, "MSC")
        out = X.copy()  # the single working copy; everything below is in place
        out -= a[:, np.newaxis]
        out /= b[:, np.newaxis]
        return out


class EMSC(TransformerMixin, EnginirEstimator):
    """Extended Multiplicative Signal Correction.

    Every spectrum is modelled as a polynomial baseline, a multiple of a
    reference spectrum and, optionally, known interference spectra:

    .. math:: x = \\sum_{i=0}^{d} c_i \\, u^i + b \\, r + \\sum_j g_j z_j + e

    where ``u`` is the wavelength *index* rescaled to ``[-1, 1]``. The
    coefficients are found by least squares and the corrected spectrum is
    ``(x - sum_i c_i u^i - sum_j g_j z_j) / b``.

    Parameters
    ----------
    method:
        How to estimate the reference spectrum when ``reference`` is ``None``:
        ``"mean"`` (default) or ``"median"`` of the training spectra.
    order:
        Polynomial baseline degree ``d``. ``0`` fits a constant offset, ``1``
        adds a linear slope, ``2`` (default) adds curvature.
    reference:
        Optional reference spectrum, shape ``(n_wavelengths,)``. Overrides
        ``method``.
    interferences:
        Optional interference spectra ``z_j`` to remove, shape
        ``(n_interferences, n_wavelengths)``.
    weights:
        Optional non-negative wavelength weights, shape ``(n_wavelengths,)``.
        Both the design matrix rows and the spectrum are multiplied by the
        weights before the least-squares solve, so the minimised quantity is
        ``sum_k (w_k e_k)^2``. Binary 0/1 weights therefore exclude wavelength
        regions from the fit; the correction itself is applied to all
        wavelengths. At least one weight must be positive.

    Attributes
    ----------
    reference_ : numpy.ndarray, shape (n_wavelengths,)
        Reference spectrum ``r``.
    interferences_ : numpy.ndarray or None, shape (n_interferences, n_wavelengths)
        Validated interference spectra.
    weights_ : numpy.ndarray, shape (n_wavelengths,)
        Wavelength weights (all ones when ``weights`` is ``None``).
    design_matrix_ : numpy.ndarray, shape (n_wavelengths, order + 2 + n_interferences)
        Columns are the polynomial terms ``u^0 ... u^d``, then ``r``, then the
        interferences. Must have full column rank: a reference that is itself
        a polynomial of degree ``<= order``, or collinear interferences, are
        rejected in :meth:`fit`.
    order_ : int
        Polynomial degree the design matrix was built with.
    n_features_in_ : int
        Number of wavelengths seen during :meth:`fit`.

    Notes
    -----
    Spectra whose multiplicative coefficient ``b`` is (near-)zero are divided
    by ``+/-1e-10`` instead and a warning is logged.

    References
    ----------
    H. Martens, E. Stark, "Extended multiplicative signal correction and
    spectral interference subtraction: new preprocessing methods for near
    infrared spectroscopy", *Journal of Pharmaceutical and Biomedical Analysis*
    9 (1991) 625-635.

    N. K. Afseth, A. Kohler, "Extended multiplicative signal correction in
    vibrational spectroscopy, a tutorial", *Chemometrics and Intelligent
    Laboratory Systems* 117 (2012) 92-99.
    """

    def __init__(
        self,
        method: str = "mean",
        order: int = 2,
        reference: np.ndarray | None = None,
        interferences: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        self.method = method
        self.order = order
        self.reference = reference
        self.interferences = interferences
        self.weights = weights

    def fit(self, X: np.ndarray, y: object = None) -> EMSC:
        """Determine the reference spectrum, weights and design matrix.

        Parameters
        ----------
        X:
            Training spectra, shape ``(n_samples, n_wavelengths)``.
        y:
            Ignored.
        """
        check_choice(self.method, "method", _EMSC_METHODS)
        order = check_integer(self.order, "order", minimum=0)
        X = as_float_matrix(X, "X")
        n_features = X.shape[1]

        if self.reference is not None:
            self.reference_ = as_float_vector(self.reference, "reference", size=n_features)
        elif self.method == "mean":
            self.reference_ = X.mean(axis=0)
        else:
            self.reference_ = np.median(X, axis=0)

        if self.weights is None:
            self.weights_ = np.ones(n_features, dtype=X.dtype)
        else:
            self.weights_ = as_float_vector(self.weights, "weights", size=n_features)
            if np.any(self.weights_ < 0.0):
                log_and_raise("weights must be non-negative")
            if not np.any(self.weights_ > 0.0):
                log_and_raise("weights must contain at least one positive value")

        self.interferences_ = None
        columns = [
            np.vander(np.linspace(-1.0, 1.0, n_features), N=order + 1, increasing=True),
            self.reference_[:, np.newaxis],
        ]
        if self.interferences is not None:
            self.interferences_ = as_float_matrix(self.interferences, "interferences")
            if self.interferences_.shape[1] != n_features:
                log_and_raise(
                    "interferences have {} features but X has {}",
                    self.interferences_.shape[1],
                    n_features,
                )
            columns.append(self.interferences_.T)
        design_matrix = np.hstack(columns).astype(np.float64)  # small; float64 so that
        # a model restored from its persisted float64 state replays bit for bit
        rank = np.linalg.matrix_rank(design_matrix * self.weights_[:, np.newaxis])
        if rank < design_matrix.shape[1]:
            log_and_raise(
                "EMSC design matrix has rank {} < {} columns: the reference is a polynomial "
                "of degree <= {} or the interferences are collinear with the other terms "
                "(on the positively weighted wavelengths)",
                rank, design_matrix.shape[1], order,
            )
        self.design_matrix_ = design_matrix
        self.order_ = order
        self.n_features_in_ = n_features
        return self

    def compute_coefficients(self, X: np.ndarray) -> np.ndarray:
        """Return the least-squares EMSC coefficients of every spectrum.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.

        Returns
        -------
        numpy.ndarray, shape (n_samples, order + 2 + n_interferences)
            Coefficients in the column order of :attr:`design_matrix_`:
            polynomial terms, the multiplicative term ``b``, interferences.
        """
        return self._coefficients(as_float_matrix(X, "X"))

    def _coefficients(self, X: np.ndarray) -> np.ndarray:
        """Coefficients of an already validated ``X``."""
        check_is_fitted(self, "design_matrix_")
        check_n_features(X, self.n_features_in_, "EMSC")
        # Weighted least squares through the small (n_terms, n_wavelengths) projector
        # pinv(W D) W, computed in float64; applying it costs no full-size copy of X.
        weights = self.weights_.astype(np.float64)
        projector = np.linalg.pinv(self.design_matrix_ * weights[:, np.newaxis]) * weights
        return X @ projector.T.astype(X.dtype)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply EMSC to every spectrum (row) of ``X``.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.

        Returns
        -------
        numpy.ndarray
            Corrected spectra with the same shape and floating dtype as ``X``.
        """
        X = as_float_matrix(X, "X")
        coefficients = self._coefficients(X)
        b_column = self.order_ + 1
        b = _clamp_near_zero(coefficients[:, b_column], "EMSC")
        additive = coefficients.copy()
        additive[:, b_column] = 0.0
        # The correction buffer is reused as the output (one full-size allocation).
        out = additive @ self.design_matrix_.astype(X.dtype, copy=False).T
        np.subtract(X, out, out=out)
        out /= b[:, np.newaxis]
        return out


__all__ = ["EMSC", "MSC", "SNV"]
