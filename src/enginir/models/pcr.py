"""Principal Component Regression (PCR)."""

from __future__ import annotations

import numpy as np
from .._estimator import EnginirEstimator
from sklearn.base import RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from dataclasses import dataclass

from enginir._checks import as_float_matrix, check_n_features, log_and_raise
from enginir._logging import get_logger

from ._common import (
    as_target_array,
    check_n_components,
    check_scaling,
    max_components,
    max_components_of,
    scaling_statistics,
)
from ._gram import (
    CHUNK_ROWS,
    GRAM_CONDITION_LIMIT,
    GramStats,
    PCAFactors,
    blas_threads,
    gram_condition,
    gram_pca,
)

logger = get_logger(__name__)

#: The algorithms of :class:`PCR` (sample-set-model-run contract §7); a fit stores
#: ``gram`` or ``svd``, ``auto`` chooses by the conditioning guard.
PCR_ALGORITHMS = ("auto", "gram", "svd")


@dataclass
class GramPCA:
    """The principal components of a ``gram`` fit, with the attributes of scikit-learn's PCA
    that engiNIR reads (``components_``, ``mean_``, the explained variances)."""

    components_: np.ndarray
    mean_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray

    @property
    def n_components_(self) -> int:
        return int(self.components_.shape[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scores in the floating dtype of ``X``, as scikit-learn's PCA."""
        X = np.asarray(X)
        dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64
        return (X - self.mean_.astype(dtype)) @ self.components_.T.astype(dtype)


@dataclass
class ScoreRegression:
    """The least-squares regression of y on the training scores (``coef_``, ``intercept_``)."""

    coef_: np.ndarray
    intercept_: float | np.ndarray

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores) @ np.asarray(self.coef_).T + self.intercept_


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
    x_scaling:
        Scaling of the spectra: ``"center"`` (default, as for model runs,
        sample-set-model-run contract §7), ``"uv"`` (standard deviation with
        N - 1) or ``"pareto"`` (its square root).
    algorithm:
        ``"auto"`` (default): ``"gram"`` when σ_A / σ₁ ≥ 10⁻⁶ for the
        ``n_components`` fitted, else ``"svd"``; ``"gram"``: eigendecomposition
        of XᵀX built in one chunked pass; ``"svd"``: full SVD of X (kept until
        2.0). Both give the same components to floating-point precision.

    Attributes
    ----------
    coef_ : numpy.ndarray, shape (n_wavelengths,) or (n_wavelengths, n_targets)
        Regression coefficients in raw data units, so that
        ``y = X @ coef_ + intercept_``.
    intercept_ : float or numpy.ndarray, shape (n_targets,)
        Intercept in raw data units (a float for a single target).
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
    algorithm_ : str
        The algorithm used, ``"gram"`` or ``"svd"``.
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

    def __init__(self, n_components: int | None = 10, x_scaling: str = "center",
                 algorithm: str = "auto") -> None:
        self.n_components = n_components
        self.x_scaling = x_scaling
        self.algorithm = algorithm

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
        if self._algorithm() != "svd":
            stats = GramStats.from_rows(X, y)
            if self._resolve(stats) == "gram":
                self.fit_stats(stats, single_target=y.ndim == 1)
                self.scores_ = self._scores(X)
                return self
        maximum = max_components(X)
        n_components = (
            maximum
            if self.n_components is None
            else check_n_components(self.n_components, "n_components", maximum=maximum)
        )

        if check_scaling(self.x_scaling, "x_scaling") == "center":
            self.x_mean_, self.x_std_ = np.zeros(X.shape[1]), np.ones(X.shape[1])
        else:
            self.x_mean_, self.x_std_ = scaling_statistics(X, self.x_scaling)
        self.x_scaling_ = self.x_scaling
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
        self.algorithm_ = "svd"
        logger.info("PCR fitted (SVD): n_components={}, X={}, y={}", n_components, X.shape,
                    y.shape)
        return self

    def _algorithm(self) -> str:
        if self.algorithm not in PCR_ALGORITHMS:
            log_and_raise("algorithm must be one of {}, got {!r}", list(PCR_ALGORITHMS),
                          self.algorithm)
        return str(self.algorithm)

    def _n_components(self, stats: GramStats, blocks: np.ndarray | None = None,
                      n_features: int | None = None) -> int:
        n = float(stats.n[slice(None) if blocks is None else blocks].sum())
        maximum = max_components_of(n, stats.n_features if n_features is None else n_features)
        return (maximum if self.n_components is None
                else check_n_components(self.n_components, "n_components", maximum=maximum))

    def _resolve(self, stats: GramStats) -> str:
        """``gram`` or ``svd``: ``auto`` applies the conditioning guard (contract §7)."""
        algorithm = self._algorithm()
        if algorithm != "auto":
            return algorithm
        x_scaling = check_scaling(self.x_scaling, "x_scaling")
        condition = gram_condition(stats.moments(), self._n_components(stats), x_scaling)
        if condition >= GRAM_CONDITION_LIMIT:
            return "gram"
        logger.info("PCR uses the full SVD: σ_A/σ₁ = {:.2g} is below {:g}", condition,
                    GRAM_CONDITION_LIMIT)
        return "svd"

    def fit_stats(self, stats: GramStats, blocks: np.ndarray | None = None, *,
                  single_target: bool | None = None, columns: np.ndarray | None = None) -> PCR:
        """Fit from sufficient statistics by the eigendecomposition of XᵀX (``gram``).

        ``blocks`` selects the blocks (default all), ``columns`` the bands
        (default all). ``auto`` fits ``gram`` here (no rows for an SVD); a
        model run decides the guard once on the whole set. The training
        scores are not computed: ``scores_`` is not set.
        """
        if self._algorithm() == "svd":
            log_and_raise("PCR with algorithm 'svd' needs the rows of X, not their statistics")
        x_scaling = check_scaling(self.x_scaling, "x_scaling")
        moments = stats.moments(blocks, columns)
        n_components = self._n_components(stats, blocks, int(moments.x_mean.shape[0]))
        factors = gram_pca(moments, n_components, x_scaling=x_scaling)
        self._set_factors(factors, moments.n, x_scaling,
                          single_target=(stats.n_targets == 1 if single_target is None
                                         else single_target), y_mean=moments.y_mean)
        logger.debug("PCR fitted (gram): n_components={}, {} rows, {} bands", n_components,
                     int(moments.n), stats.n_features)
        return self

    def _set_factors(self, factors: PCAFactors, n: float, x_scaling: str, *,
                     single_target: bool, y_mean: np.ndarray) -> None:
        a = factors.components.shape[0]
        if x_scaling == "center":  # the PCA centers, as in the SVD path
            self.x_mean_ = np.zeros_like(factors.x_mean)
            self.x_std_ = np.ones_like(factors.x_mean)
            pca_mean = factors.x_mean
        else:
            self.x_mean_, self.x_std_ = factors.x_mean, factors.x_std
            pca_mean = np.zeros_like(factors.x_mean)
        values = factors.eigenvalues[:a]
        self.x_scaling_ = x_scaling
        self.pca_ = GramPCA(components_=factors.components, mean_=pca_mean,
                            explained_variance_=values / max(n - 1.0, 1.0),
                            explained_variance_ratio_=values / factors.total_ss)
        with np.errstate(divide="ignore", invalid="ignore"):
            score_coef = (factors.components @ factors.gram_y) / values[:, None]   # (A, T)
        score_coef = np.where(values[:, None] > 0, score_coef, 0.0)
        self.reg_ = ScoreRegression(coef_=score_coef[:, 0] if single_target else score_coef.T,
                                    intercept_=float(y_mean[0]) if single_target else y_mean)
        self.loadings_ = factors.components.T
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        coef = self.loadings_ @ score_coef / self.x_std_[:, np.newaxis]
        offset = self.x_mean_ + pca_mean * self.x_std_
        intercept = y_mean - offset @ coef
        self.coef_ = coef[:, 0] if single_target else coef
        self.intercept_ = float(intercept[0]) if single_target else intercept
        self.n_features_in_ = int(factors.x_mean.shape[0])
        self.algorithm_ = "gram"

    def _scores(self, X: np.ndarray) -> np.ndarray:
        """Training scores in float64, chunk by chunk (no full working copy of X)."""
        components = np.asarray(self.pca_.components_, dtype=np.float64)
        mean = np.asarray(self.pca_.mean_, dtype=np.float64)
        scores = np.empty((X.shape[0], components.shape[0]))
        with blas_threads(X.size):
            for lo in range(0, X.shape[0], CHUNK_ROWS):
                block = np.asarray(X[lo:lo + CHUNK_ROWS], dtype=np.float64)
                if self.x_scaling_ != "center":
                    block = (block - self.x_mean_) / self.x_std_
                scores[lo:lo + CHUNK_ROWS] = (block - mean) @ components.T
        return scores

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
        if self.x_scaling_ == "center":
            return X  # the PCA centers
        scaled = X - self.x_mean_.astype(X.dtype)
        scaled /= self.x_std_.astype(X.dtype)
        return scaled


__all__ = ["PCR", "PCR_ALGORITHMS", "GramPCA", "ScoreRegression"]
