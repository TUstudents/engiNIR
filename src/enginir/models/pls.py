"""Partial Least Squares regression (PLS1 and PLS2)."""

from __future__ import annotations

import numpy as np
from .._estimator import EnginirEstimator
from sklearn.base import RegressorMixin
from sklearn.cross_decomposition import PLSRegression as _SklearnPLSRegression
from sklearn.utils.validation import check_is_fitted

from enginir._checks import as_float_matrix, check_n_features, log_and_raise
from enginir._logging import get_logger

from ._common import as_target_array, check_n_components, max_components

logger = get_logger(__name__)


class PLSRegression(RegressorMixin, EnginirEstimator):
    """Partial Least Squares regression for one (PLS1) or several (PLS2) targets.

    The model is fitted with the NIPALS algorithm of
    :class:`sklearn.cross_decomposition.PLSRegression` and exposes the results
    in chemometric conventions: regression coefficients ``B`` with one row per
    wavelength, an intercept in raw data units, and VIP scores.

    Parameters
    ----------
    n_components:
        Number of latent variables, ``1 <= n_components <= min(n_samples - 1,
        n_wavelengths)`` (the rank limit of mean-centred X). Default ``6``.
    scale:
        If ``True`` (default), X and Y are autoscaled (unit standard deviation)
        before fitting; if ``False`` they are only mean-centred.
    max_iter:
        Maximum number of NIPALS iterations per component (only used for
        PLS2). Default ``500``.
    tol:
        NIPALS convergence tolerance (only used for PLS2). Default ``1e-6``.

    Attributes
    ----------
    coef_ : numpy.ndarray, shape (n_wavelengths,) or (n_wavelengths, n_targets)
        Regression coefficients ``B`` in raw data units, so that
        ``y = X @ coef_ + intercept_``. One-dimensional for PLS1.
    intercept_ : float or numpy.ndarray, shape (n_targets,)
        Intercept in raw data units (a float for PLS1).
    x_mean_, x_std_ : numpy.ndarray, shape (n_wavelengths,)
        Centring and scaling applied to X (``x_std_`` is all ones when
        ``scale=False``).
    y_mean_, y_std_ : numpy.ndarray, shape (n_targets,)
        Centring and scaling applied to Y.
    x_weights_ : numpy.ndarray, shape (n_wavelengths, n_components)
        X weights ``W`` (unit-norm columns).
    x_loadings_ : numpy.ndarray, shape (n_wavelengths, n_components)
        X loadings ``P``.
    y_loadings_ : numpy.ndarray, shape (n_targets, n_components)
        Y loadings ``Q``.
    x_scores_ : numpy.ndarray, shape (n_samples, n_components)
        Training X scores ``T``.
    y_scores_ : numpy.ndarray, shape (n_samples, n_components)
        Training Y scores ``U``.
    x_rotations_ : numpy.ndarray, shape (n_wavelengths, n_components)
        Projection ``W (P^T W)^-1`` that maps scaled spectra to scores, used
        by :meth:`transform`.
    n_features_in_ : int
        Number of wavelengths seen during :meth:`fit`.

    References
    ----------
    S. Wold, M. Sjöström, L. Eriksson, "PLS-regression: a basic tool of
    chemometrics", *Chemometrics and Intelligent Laboratory Systems* 58 (2001)
    109-130.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(30, 50))
    >>> y = X[:, 3] - 2 * X[:, 7]
    >>> model = PLSRegression(n_components=3).fit(X, y)
    >>> np.allclose(model.predict(X), X @ model.coef_ + model.intercept_)
    True
    """

    def __init__(
        self,
        n_components: int = 6,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> PLSRegression:
        """Fit the PLS model.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.
        y:
            Reference values, shape ``(n_samples,)`` for PLS1 or
            ``(n_samples, n_targets)`` for PLS2.

        Raises
        ------
        ValueError
            If the inputs are malformed or ``n_components`` is out of range.
        """
        X = as_float_matrix(X, "X")
        y = as_target_array(y, n_samples=X.shape[0])
        n_components = check_n_components(self.n_components, "n_components",
                                          maximum=max_components(X))

        backend = _SklearnPLSRegression(
            n_components=n_components,
            scale=self.scale,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        backend.fit(X, y)
        is_pls1 = y.ndim == 1

        # scikit-learn predicts (X - x_mean) @ coef.T + y_mean; fold the centring
        # into the intercept so that y = X @ coef_ + intercept_.
        coef = backend.coef_.T  # (n_wavelengths, n_targets)
        intercept = backend.intercept_ - backend._x_mean @ coef
        self.coef_ = coef[:, 0] if is_pls1 else coef
        self.intercept_ = float(intercept[0]) if is_pls1 else intercept
        self.x_mean_ = backend._x_mean.copy()
        self.x_std_ = backend._x_std.copy()
        self.y_mean_ = backend._y_mean.copy()
        self.y_std_ = backend._y_std.copy()
        self.x_weights_ = backend.x_weights_
        self.x_loadings_ = backend.x_loadings_
        self.y_loadings_ = backend.y_loadings_
        self.x_scores_ = backend.x_scores_
        self.y_scores_ = backend.y_scores_
        self.x_rotations_ = backend.x_rotations_
        self.n_features_in_ = X.shape[1]
        logger.info("PLSRegression fitted: n_components={}, X={}, y={}", n_components, X.shape,
                    y.shape)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict reference values.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.

        Returns
        -------
        numpy.ndarray, shape (n_samples,) or (n_samples, n_targets)
            Predictions in the floating dtype of ``X``.
        """
        check_is_fitted(self, "coef_")
        X = as_float_matrix(X, "X")
        check_n_features(X, self.n_features_in_, "PLSRegression")
        coef = np.asarray(self.coef_, dtype=X.dtype)
        return X @ coef + np.asarray(self.intercept_, dtype=X.dtype)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the X scores ``T`` of new spectra.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_components)
        """
        check_is_fitted(self, "coef_")
        X = as_float_matrix(X, "X")
        check_n_features(X, self.n_features_in_, "PLSRegression")
        scaled = X - self.x_mean_.astype(X.dtype)  # the one working copy, scaled in place
        scaled /= self.x_std_.astype(X.dtype)
        return scaled @ self.x_rotations_.astype(X.dtype)

    def vip_scores(self) -> np.ndarray:
        """Return the Variable Importance in Projection (VIP) of every wavelength.

        .. math::

            \\mathrm{VIP}_j = \\sqrt{\\frac{p \\sum_a SS_a \\, (w_{ja} / \\lVert w_a \\rVert)^2}
                                      {\\sum_a SS_a}},
            \\qquad SS_a = t_a^\\top t_a \\sum_k q_{ka}^2

        where ``p`` is the number of wavelengths, ``w_a`` the X weights,
        ``t_a`` the X scores and ``q_a`` the Y loadings of component ``a``.
        ``SS_a`` is the (scaled) Y variance explained by component ``a``;
        for PLS2 it is summed over the targets. The mean of the squared VIP
        scores is 1, so ``VIP > 1`` marks wavelengths of above-average
        importance.

        Returns
        -------
        numpy.ndarray, shape (n_wavelengths,)

        References
        ----------
        I.-G. Chong, C.-H. Jun, "Performance of some variable selection
        methods when multicollinearity is present", *Chemometrics and
        Intelligent Laboratory Systems* 78 (2005) 103-112.
        """
        check_is_fitted(self, "x_weights_")
        weights = self.x_weights_ / np.linalg.norm(self.x_weights_, axis=0)
        explained = np.sum(self.x_scores_**2, axis=0) * np.sum(self.y_loadings_**2, axis=0)
        total = explained.sum()
        if total <= 0.0:
            log_and_raise("VIP is undefined: the PLS components explain no Y variance")
        n_wavelengths = weights.shape[0]
        return np.sqrt(n_wavelengths * (weights**2 @ explained) / total)


__all__ = ["PLSRegression"]
