"""Principal Component Regression (PCR)."""

from __future__ import annotations

import numpy as np
from .._estimator import EnginirEstimator
from sklearn.base import RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from enginir._checks import as_float_matrix, check_n_features, log_and_raise
from enginir._logging import get_logger

from ._common import (
    as_target_array,
    check_n_components,
    check_scaling,
    max_components,
    scaling_statistics,
)

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
        Used when ``x_scaling`` is ``None`` (models stored before engiNIR
        1.2): ``True`` (default) scales every wavelength to unit variance with
        scikit-learn's ``StandardScaler`` (standard deviation with N;
        constant wavelengths are left unscaled), ``False`` only centres.
    x_scaling:
        Scaling of the spectra: ``"center"``, ``"uv"`` (standard deviation
        with N - 1) or ``"pareto"`` (its square root); ``None`` (default)
        follows ``scale``. Model runs use ``"center"``
        (sample-set-model-run contract §7).

    Attributes
    ----------
    coef_ : numpy.ndarray, shape (n_wavelengths,) or (n_wavelengths, n_targets)
        Regression coefficients in raw data units, so that
        ``y = X @ coef_ + intercept_``.
    intercept_ : float or numpy.ndarray, shape (n_targets,)
        Intercept in raw data units (a float for a single target).
    scaler_ : sklearn.preprocessing.StandardScaler or None
        Fitted scaler of ``scale=True`` without ``x_scaling``, else ``None``.
    x_mean_, x_std_ : numpy.ndarray, shape (n_wavelengths,)
        Centring and divisor applied before the PCA. For centring only both
        are neutral (zeros and ones): the PCA centres the spectra itself.
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

    def __init__(self, n_components: int | None = 10, scale: bool = True,
                 x_scaling: str | None = None) -> None:
        self.n_components = n_components
        self.scale = scale
        self.x_scaling = x_scaling

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

        if self.x_scaling is not None and not self.scale:
            log_and_raise("Give either scale=False or x_scaling, not both")
        if self.x_scaling is None:  # models stored before engiNIR 1.2
            self.scaler_ = StandardScaler().fit(X) if self.scale else None
            if self.scaler_ is None:
                self.x_mean_, self.x_std_ = np.zeros(X.shape[1]), np.ones(X.shape[1])
            else:
                self.x_mean_ = np.asarray(self.scaler_.mean_, dtype=np.float64)
                self.x_std_ = np.asarray(self.scaler_.scale_, dtype=np.float64)
        elif check_scaling(self.x_scaling, "x_scaling") == "center":
            self.scaler_ = None
            self.x_mean_, self.x_std_ = np.zeros(X.shape[1]), np.ones(X.shape[1])
        else:
            self.scaler_ = None
            self.x_mean_, self.x_std_ = scaling_statistics(X, self.x_scaling)
        self.x_scaling_ = self.x_scaling or ("uv" if self.scale else "center")
        X_scaled = self._scaled(X)
        # A full SVD: reproducible, and the first a components are those of an
        # a-component fit (scikit-learn may otherwise choose a randomised SVD).
        self.pca_ = PCA(n_components=n_components, svd_solver="full").fit(X_scaled)
        self.scores_ = self.pca_.transform(X_scaled)
        self.reg_ = LinearRegression().fit(self.scores_, y)
        self.loadings_ = np.asarray(self.pca_.components_).T
        self.explained_variance_ratio_ = np.asarray(self.pca_.explained_variance_ratio_)

        # Compose scaling, centring, projection and regression into one affine
        # map y = X @ coef_ + intercept_.
        coef = self.loadings_ @ np.atleast_2d(self.reg_.coef_).T  # (n_wavelengths, n_targets)
        offset = np.asarray(self.pca_.mean_)
        coef = coef / self.x_std_[:, np.newaxis]
        offset = self.x_mean_ + offset * self.x_std_
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
        return self.pca_.transform(self._scaled(X))

    def predict_components(self, X: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Predictions of the models with 1, 2, ... ``n_components`` components, from this fit.

        The training PCA scores are centred and orthogonal, so the regression
        on the first *a* scores has the first *a* coefficients of the full
        regression: every smaller model's predictions come from one fit.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, n_components)`` for one target, ``(n_samples,
            n_targets, n_components)`` for several; column *a* - 1 holds the
            predictions with *a* components.
        """
        check_is_fitted(self, "coef_")
        n_fitted = int(self.loadings_.shape[1])
        n = n_fitted if n_components is None else check_n_components(
            n_components, "n_components", maximum=n_fitted)
        scores = np.asarray(self.transform(X), dtype=np.float64)[:, :n]
        coef = np.atleast_2d(np.asarray(self.reg_.coef_, dtype=np.float64))[:, :n]
        intercept = np.atleast_1d(np.asarray(self.reg_.intercept_, dtype=np.float64))
        contributions = scores[:, np.newaxis, :] * coef[np.newaxis, :, :]
        predictions = np.cumsum(contributions, axis=2) + intercept[np.newaxis, :, np.newaxis]
        if np.ndim(self.coef_) == 1:
            return predictions[:, 0, :]
        return predictions

    def _scaled(self, X: np.ndarray) -> np.ndarray:
        """``X`` centred and scaled as in :meth:`fit` (centring only is left to the PCA)."""
        if self.scaler_ is not None:
            return np.asarray(self.scaler_.transform(X))
        if self.x_scaling_ == "center":
            return X  # the PCA centres; this keeps the pre-1.2 numbers exactly
        scaled = X - self.x_mean_.astype(X.dtype)
        scaled /= self.x_std_.astype(X.dtype)
        return scaled


__all__ = ["PCR"]
