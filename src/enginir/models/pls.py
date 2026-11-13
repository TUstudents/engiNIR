"""Partial Least Squares regression (PLS1 and PLS2)."""

from __future__ import annotations

import numpy as np
from .._estimator import EnginirEstimator
from sklearn.base import RegressorMixin
from sklearn.cross_decomposition import PLSRegression as _SklearnPLSRegression
from sklearn.utils.validation import check_is_fitted

from enginir._checks import as_float_matrix, check_n_features, log_and_raise
from enginir._logging import get_logger

from ._common import (
    as_target_array,
    check_n_components,
    max_components,
    max_components_of,
    resolve_scalings,
    scaling_statistics,
)
from ._gram import CHUNK_ROWS, GramStats, PLSFactors, blas_threads, kernel_pls

#: The algorithms of :class:`PLSRegression` (sample-set-model-run contract §7).
PLS_ALGORITHMS = ("kernel", "nipals")

logger = get_logger(__name__)


class PLSRegression(RegressorMixin, EnginirEstimator):
    """Partial Least Squares regression for one (PLS1) or several (PLS2) targets.

    The model is fitted by improved kernel PLS on XᵀX (``algorithm="kernel"``,
    Dayal and MacGregor 1997) or by the NIPALS algorithm of
    :class:`sklearn.cross_decomposition.PLSRegression` (``"nipals"``); both
    give the same model to floating-point precision. The results follow
    chemometric conventions: regression coefficients ``B`` with one row per
    wavelength, an intercept in raw data units, and VIP scores.

    Parameters
    ----------
    n_components:
        Number of latent variables, ``1 <= n_components <= min(n_samples - 1,
        n_wavelengths)`` (the rank limit of mean-centred X). Default ``6``.
    x_scaling, y_scaling:
        Scaling of X and of Y: ``"center"`` (mean-centered), ``"uv"`` (also
        divided by the standard deviation, N - 1) or ``"pareto"`` (divided by
        its square root). Defaults ``x_scaling="center"``, ``y_scaling="uv"``,
        as for model runs (sample-set-model-run contract §7). For one target,
        the Y scaling does not change the predictions or the coefficients in
        raw units.
    max_iter:
        Maximum number of NIPALS iterations per component (only used for
        PLS2). Default ``500``.
    tol:
        NIPALS convergence tolerance (only used for PLS2). Default ``1e-6``.
    algorithm:
        ``"kernel"`` (default) or ``"nipals"`` (kept until 2.0). The kernel
        fit builds XᵀX in one chunked pass (float64 sums of float32 rows) and
        makes no full copy of X.

    Attributes
    ----------
    coef_ : numpy.ndarray, shape (n_wavelengths,) or (n_wavelengths, n_targets)
        Regression coefficients ``B`` in raw data units, so that
        ``y = X @ coef_ + intercept_``. One-dimensional for PLS1.
    intercept_ : float or numpy.ndarray, shape (n_targets,)
        Intercept in raw data units (a float for PLS1).
    x_mean_, x_std_ : numpy.ndarray, shape (n_wavelengths,)
        Centring and divisor applied to X (``x_std_`` is all ones for
        ``center``, the square root of the standard deviation for ``pareto``).
    y_mean_, y_std_ : numpy.ndarray, shape (n_targets,)
        Centring and divisor applied to Y.
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
    x_score_ss_ : numpy.ndarray, shape (n_components,)
        Sum of squares of every column of the training X scores, ``t_a^T t_a``.
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
        max_iter: int = 500,
        tol: float = 1e-6,
        x_scaling: str = "center",
        y_scaling: str = "uv",
        algorithm: str = "kernel",
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.x_scaling = x_scaling
        self.y_scaling = y_scaling
        self.algorithm = algorithm

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
        if self._algorithm() == "kernel":
            self.fit_stats(GramStats.from_rows(X, y), single_target=y.ndim == 1)
            self.x_scores_ = self._scores(X, np.float64)  # float64, as NIPALS stores them
            self.y_scores_ = self._y_scores(y)
            return self
        n_components = check_n_components(self.n_components, "n_components",
                                          maximum=max_components(X))
        x_scaling, y_scaling = resolve_scalings(self.x_scaling, self.y_scaling)

        # engiNIR centres and scales; scikit-learn gets the scaled data (one
        # float64 working copy, which scikit-learn would make anyway) and only
        # re-centres it.
        is_pls1 = y.ndim == 1
        x_mean, x_std = scaling_statistics(X, x_scaling)
        y_mean, y_std = scaling_statistics(y, y_scaling)
        X_scaled = np.asarray(X, dtype=np.float64) - x_mean
        X_scaled /= x_std
        y_scaled = (np.asarray(y, dtype=np.float64).reshape(y.shape[0], -1) - y_mean) / y_std
        backend = _SklearnPLSRegression(
            n_components=n_components,
            scale=False,
            max_iter=self.max_iter,
            tol=self.tol,
            copy=False,
        )
        backend.fit(X_scaled, y_scaled[:, 0] if is_pls1 else y_scaled)
        del X_scaled

        # Coefficients in raw units: y = X @ coef_ + intercept_, the model
        # passes through the means.
        coef = backend.coef_.T / x_std[:, np.newaxis] * y_std[np.newaxis, :]
        intercept = y_mean - x_mean @ coef
        self.coef_ = coef[:, 0] if is_pls1 else coef
        self.intercept_ = float(intercept[0]) if is_pls1 else intercept
        self.x_mean_ = x_mean + backend._x_mean * x_std
        self.x_std_ = x_std
        self.y_mean_ = y_mean + backend._y_mean * y_std
        self.y_std_ = y_std
        self.x_scaling_, self.y_scaling_ = x_scaling, y_scaling
        self.x_weights_ = backend.x_weights_
        self.x_loadings_ = backend.x_loadings_
        self.y_loadings_ = backend.y_loadings_
        self.x_scores_ = backend.x_scores_
        self.y_scores_ = backend.y_scores_
        self.x_rotations_ = backend.x_rotations_
        self.x_score_ss_ = np.sum(np.asarray(backend.x_scores_) ** 2, axis=0)
        self.n_features_in_ = X.shape[1]
        logger.info("PLSRegression fitted (NIPALS): n_components={}, X={}, y={}", n_components,
                    X.shape, y.shape)
        return self

    def _algorithm(self) -> str:
        if self.algorithm not in PLS_ALGORITHMS:
            log_and_raise("algorithm must be one of {}, got {!r}", list(PLS_ALGORITHMS),
                          self.algorithm)
        return str(self.algorithm)

    def fit_stats(self, stats: GramStats, blocks: np.ndarray | None = None, *,
                  single_target: bool | None = None,
                  columns: np.ndarray | None = None) -> PLSRegression:
        """Fit by kernel PLS from sufficient statistics (:class:`~enginir.models._gram.GramStats`).

        ``blocks`` selects the blocks to fit on (default all), e.g. the
        calibration part of a fold; ``columns`` the bands (default all, e.g.
        a candidate of the band proposal). The training scores are not computed
        (no rows are seen): ``x_scores_`` and ``y_scores_`` are not set,
        ``x_score_ss_`` is.
        """
        moments = stats.moments(blocks, columns)
        maximum = max_components_of(moments.n, int(moments.x_mean.shape[0]))
        n_components = check_n_components(self.n_components, "n_components", maximum=maximum)
        x_scaling, y_scaling = resolve_scalings(self.x_scaling, self.y_scaling)
        self._set_factors(kernel_pls(moments, n_components, x_scaling=x_scaling,
                                     y_scaling=y_scaling),
                          single_target=(stats.n_targets == 1 if single_target is None
                                         else single_target))
        self.x_scaling_, self.y_scaling_ = x_scaling, y_scaling
        logger.debug("PLSRegression fitted (kernel): n_components={}, {} rows, {} bands",
                     n_components, int(moments.n), int(moments.x_mean.shape[0]))
        return self

    def _set_factors(self, factors: PLSFactors, *, single_target: bool) -> None:
        coef, intercept = factors.coefficients()
        self.coef_ = coef[:, 0] if single_target else coef
        self.intercept_ = float(intercept[0]) if single_target else intercept
        self.x_mean_, self.x_std_ = factors.x_mean, factors.x_std
        self.y_mean_, self.y_std_ = factors.y_mean, factors.y_std
        self.x_weights_ = factors.weights
        self.x_loadings_ = factors.loadings
        self.y_loadings_ = factors.y_loadings
        self.x_rotations_ = factors.rotations
        self.x_score_ss_ = factors.score_ss
        self.n_features_in_ = int(factors.x_mean.shape[0])

    def _y_scores(self, y: np.ndarray) -> np.ndarray:
        """Training Y scores U as scikit-learn's NIPALS defines them (regression mode)."""
        ys = (np.asarray(y, dtype=np.float64).reshape(len(y), -1) - self.y_mean_) / self.y_std_
        t, q = np.asarray(self.x_scores_), np.asarray(self.y_loadings_)
        u = np.empty_like(t)
        for a in range(t.shape[1]):
            residual = ys - t[:, :a] @ q[:, :a].T
            u[:, a] = residual @ q[:, a] / float(q[:, a] @ q[:, a])
        return u

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
        return self._scores(X, X.dtype)

    def _scores(self, X: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
        """Scores of ``X`` in ``dtype``, chunk by chunk (no full working copy of X)."""
        mean, std = self.x_mean_.astype(dtype), self.x_std_.astype(dtype)
        rotations = np.asarray(self.x_rotations_).astype(dtype)
        n_components = int(np.shape(rotations)[-1])
        scores = np.empty((X.shape[0], n_components), dtype=dtype)
        with blas_threads(X.size):
            for lo in range(0, X.shape[0], CHUNK_ROWS):
                block = np.asarray(X[lo:lo + CHUNK_ROWS], dtype=dtype) - mean
                block /= std
                scores[lo:lo + CHUNK_ROWS] = block @ rotations
        return scores

    def predict_components(self, X: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Predictions of the models with 1, 2, ... ``n_components`` components, from this fit.

        The first *a* columns of the rotations and Y loadings are exactly the
        *a*-component PLS model (NIPALS), so every smaller model's predictions
        come from one fit.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.
        n_components:
            Largest number of components (default: all of this fit).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, n_components)`` for one target, ``(n_samples,
            n_targets, n_components)`` for several; column *a* - 1 holds the
            predictions with *a* components.
        """
        check_is_fitted(self, "coef_")
        rotations = np.asarray(self.x_rotations_)
        n = rotations.shape[-1] if n_components is None else check_n_components(
            n_components, "n_components", maximum=rotations.shape[-1])
        scores = np.asarray(self.transform(X), dtype=np.float64)[:, :n]
        q = np.asarray(self.y_loadings_, dtype=np.float64)[:, :n]          # (targets, n)
        contributions = scores[:, np.newaxis, :] * q[np.newaxis, :, :]     # (samples, targets, n)
        predictions = (np.cumsum(contributions, axis=2) * self.y_std_[np.newaxis, :, np.newaxis]
                       + self.y_mean_[np.newaxis, :, np.newaxis])
        if np.ndim(self.coef_) == 1:
            return predictions[:, 0, :]
        return predictions

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
        explained = np.asarray(self.x_score_ss_) * np.sum(self.y_loadings_**2, axis=0)
        total = explained.sum()
        if total <= 0.0:
            log_and_raise("VIP is undefined: the PLS components explain no Y variance")
        n_wavelengths = weights.shape[0]
        return np.sqrt(n_wavelengths * (weights**2 @ explained) / total)


__all__ = ["PLS_ALGORITHMS", "PLSRegression"]
