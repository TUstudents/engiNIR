"""Principal Component Regression (PCR)."""

from __future__ import annotations

import numpy as np
from .._estimator import EnginirEstimator
from sklearn.base import RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from enginir._checks import as_float_matrix, check_n_features
from enginir._logging import get_logger

from ._common import as_target_array, check_n_components, max_components

logger = get_logger(__name__)


class PCR(RegressorMixin, EnginirEstimator):
    """Principal Component Regression.

    The spectra are (optionally) autoscaled, compressed to their first
    ``n_components`` principal components, and the reference values are
    regressed on the PCA scores by ordinary least squares.

    Parameters
    ----------
    n_components:
        Number of principal components, ``1 <= n_components <=
        min(n_samples - 1, n_wavelengths)`` (the rank limit of mean-centred
        X), or ``None`` to keep that maximum.
        Default ``10``.
    scale:
        If ``True`` (default), every wavelength is scaled to unit variance
        before PCA (constant wavelengths are left unscaled); if ``False`` the
        spectra are only mean-centred.

    Attributes
    ----------
    coef_ : numpy.ndarray, shape (n_wavelengths,) or (n_wavelengths, n_targets)
        Regression coefficients in raw data units, so that
        ``y = X @ coef_ + intercept_``.
    intercept_ : float or numpy.ndarray, shape (n_targets,)
        Intercept in raw data units (a float for a single target).
    scaler_ : sklearn.preprocessing.StandardScaler or None
        Fitted scaler, ``None`` when ``scale=False``.
    pca_ : sklearn.decomposition.PCA
        Fitted PCA of the (scaled) spectra.
    reg_ : sklearn.linear_model.LinearRegression
        Fitted regression of the targets on the PCA scores.
    loadings_ : numpy.ndarray, shape (n_wavelengths, n_components)
        PCA loadings, ``pca_.components_.T``.
    scores_ : numpy.ndarray, shape (n_samples, n_components)
        PCA scores of the training spectra.
    explained_variance_ratio_ : numpy.ndarray, shape (n_components,)
        Fraction of the (scaled) spectral variance explained by each component.
    n_features_in_ : int
        Number of wavelengths seen during :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(30, 50))
    >>> y = X[:, 3] - 2 * X[:, 7]
    >>> model = PCR(n_components=5).fit(X, y)
    >>> np.allclose(model.predict(X), X @ model.coef_ + model.intercept_)
    True
    """

    def __init__(self, n_components: int | None = 10, scale: bool = True) -> None:
        self.n_components = n_components
        self.scale = scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> PCR:
        """Fit the PCA and the score regression.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.
        y:
            Reference values, shape ``(n_samples,)`` or
            ``(n_samples, n_targets)``.

        Raises
        ------
        ValueError
            If the inputs are malformed or ``n_components`` is out of range.
        """
        X = as_float_matrix(X, "X")
        y = as_target_array(y, n_samples=X.shape[0])
        maximum = max_components(X)
        n_components = (
            maximum
            if self.n_components is None
            else check_n_components(self.n_components, "n_components", maximum=maximum)
        )

        self.scaler_ = StandardScaler().fit(X) if self.scale else None
        X_scaled = X if self.scaler_ is None else self.scaler_.transform(X)
        self.pca_ = PCA(n_components=n_components).fit(X_scaled)
        self.scores_ = self.pca_.transform(X_scaled)
        self.reg_ = LinearRegression().fit(self.scores_, y)
        self.loadings_ = np.asarray(self.pca_.components_).T
        self.explained_variance_ratio_ = np.asarray(self.pca_.explained_variance_ratio_)

        # Compose scaling, centring, projection and regression into one affine
        # map y = X @ coef_ + intercept_.
        coef = self.loadings_ @ np.atleast_2d(self.reg_.coef_).T  # (n_wavelengths, n_targets)
        offset = np.asarray(self.pca_.mean_)
        if self.scaler_ is not None:
            scale = np.asarray(self.scaler_.scale_)
            coef = coef / scale[:, np.newaxis]
            offset = np.asarray(self.scaler_.mean_) + offset * scale
        intercept = np.atleast_1d(self.reg_.intercept_) - offset @ coef
        single_target = y.ndim == 1
        self.coef_ = coef[:, 0] if single_target else coef
        self.intercept_ = float(intercept[0]) if single_target else intercept
        self.n_features_in_ = X.shape[1]
        logger.info("PCR fitted: n_components={}, X={}, y={}", n_components, X.shape, y.shape)
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
        check_n_features(X, self.n_features_in_, "PCR")
        coef = np.asarray(self.coef_, dtype=X.dtype)
        return X @ coef + np.asarray(self.intercept_, dtype=X.dtype)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the PCA scores of new spectra.

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
        check_n_features(X, self.n_features_in_, "PCR")
        X_scaled = X if self.scaler_ is None else self.scaler_.transform(X)
        return self.pca_.transform(X_scaled)


__all__ = ["PCR"]
