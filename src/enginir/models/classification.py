"""Classification models: SIMCA and PCA-LDA."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import norm
from .._estimator import EnginirEstimator
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from enginir._checks import as_float_matrix, check_n_features, check_real, log_and_raise
from enginir._logging import get_logger

from ._common import as_class_labels, check_n_components

logger = get_logger(__name__)


def jackson_mudholkar_limit(discarded_eigenvalues: np.ndarray, alpha: float) -> float:
    """Return the ``1 - alpha`` upper confidence limit of the Q statistic.

    Parameters
    ----------
    discarded_eigenvalues:
        Covariance eigenvalues of the principal components *not* retained in
        the model.
    alpha:
        Significance level, e.g. ``0.05`` for a 95 % limit.

    Returns
    -------
    float
        ``Q_alpha = theta_1 [c_alpha sqrt(2 theta_2 h_0^2) / theta_1 + 1 +
        theta_2 h_0 (h_0 - 1) / theta_1^2]^(1 / h_0)``, with
        ``theta_i = sum(lambda^i)`` and ``h_0 = 1 - 2 theta_1 theta_3 / (3 theta_2^2)``.

    References
    ----------
    J. E. Jackson, G. S. Mudholkar, "Control procedures for residuals
    associated with principal component analysis", *Technometrics* 21 (1979)
    341-349.
    """
    eigenvalues = np.asarray(discarded_eigenvalues, dtype=np.float64)
    theta1, theta2, theta3 = (float(np.sum(eigenvalues**power)) for power in (1, 2, 3))
    h0 = 1.0 - 2.0 * theta1 * theta3 / (3.0 * theta2**2)
    if h0 <= 0.0:
        logger.warning("Jackson-Mudholkar: h0={:.3g} <= 0; using h0=1e-6", h0)
        h0 = 1e-6
    c_alpha = norm.ppf(1.0 - alpha)
    bracket = (
        c_alpha * np.sqrt(2.0 * theta2 * h0**2) / theta1
        + 1.0
        + theta2 * h0 * (h0 - 1.0) / theta1**2
    )
    return float(theta1 * bracket ** (1.0 / h0))


def _rejection_label(classes: np.ndarray) -> tuple[Any, np.dtype]:
    """Return the label for rejected samples and an output dtype that can hold it."""
    kind = classes.dtype.kind
    if kind == "i":
        return -1, np.result_type(classes.dtype, np.int8)
    if kind == "u":
        return -1, np.dtype(np.int64)
    if kind == "O" and all(isinstance(label, str) for label in classes):
        kind = "U"
        classes = classes.astype(str)
    if kind == "U":
        width = max(classes.dtype.itemsize // 4, len("unknown"))
        return "unknown", np.dtype(f"<U{width}")
    if kind == "f":
        return np.nan, classes.dtype
    return None, np.dtype(object)


class SIMCA(ClassifierMixin, EnginirEstimator):
    """Soft Independent Modelling of Class Analogy.

    One PCA model is fitted per class. A sample is accepted by a class if its
    Q residual (squared distance to the class PCA subspace, in the class's
    scaled units) does not exceed the class's Jackson-Mudholkar limit at
    significance level ``alpha``. Every sample is assigned to the accepting
    class with the smallest Q residual; samples accepted by no class are
    rejected (see :meth:`predict`). Only Q residuals are used; Hotelling's
    T² is not part of the decision rule.

    Parameters
    ----------
    n_components:
        Number of principal components per class model: one integer for all
        classes, or a mapping ``{class_label: n_components}`` that covers every
        class. Reduced (with a warning) to ``rank - 1`` for classes whose
        centred data have rank ``<= n_components``, so that a residual
        subspace remains. Default ``5``.
    alpha:
        Significance level of the acceptance limit, ``0 < alpha < 1``.
        Default ``0.05`` (95 % confidence).
    scale:
        If ``True`` (default), each class block is autoscaled before PCA.

    Attributes
    ----------
    classes_ : numpy.ndarray, shape (n_classes,)
        Sorted class labels.
    n_components_ : dict
        Number of components actually used for each class.
    scalers_ : dict
        ``{class_label: StandardScaler}`` (empty when ``scale=False``).
    pca_models_ : dict
        ``{class_label: PCA}`` fitted to each (scaled) class block.
    boundaries_ : dict
        ``{class_label: float}`` Q-residual acceptance limits.
    n_features_in_ : int
        Number of wavelengths seen during :meth:`fit`.

    References
    ----------
    S. Wold, "Pattern recognition by means of disjoint principal components
    models", *Pattern Recognition* 8 (1976) 127-139.
    """

    def __init__(
        self,
        n_components: int | Mapping[Any, int] = 5,
        alpha: float = 0.05,
        scale: bool = True,
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.scale = scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> SIMCA:
        """Fit one PCA model and acceptance limit per class.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.
        y:
            Class labels, shape ``(n_samples,)``.

        Raises
        ------
        ValueError
            If a class has fewer than 3 samples, ``n_components`` does not
            cover every class, or ``alpha`` is not in ``(0, 1)``.
        """
        X = as_float_matrix(X, "X")
        y = as_class_labels(y, n_samples=X.shape[0])
        alpha = check_real(self.alpha, "alpha", positive=True)
        if alpha >= 1.0:
            log_and_raise("alpha must be in (0, 1), got {}", alpha)
        self.classes_ = np.unique(y)
        self.n_components_: dict[Any, int] = {}
        self.scalers_: dict[Any, StandardScaler] = {}
        self.pca_models_: dict[Any, PCA] = {}
        self.boundaries_: dict[Any, float] = {}

        for label in self.classes_:
            X_class = X[y == label]
            if self.scale:
                self.scalers_[label] = StandardScaler().fit(X_class)
                X_class = np.asarray(self.scalers_[label].transform(X_class))
            eigenvalues = _covariance_eigenvalues(X_class)
            n_components = self._resolve_n_components(label, rank=eigenvalues.size)
            self.n_components_[label] = n_components
            self.pca_models_[label] = PCA(n_components=n_components).fit(X_class)
            self.boundaries_[label] = jackson_mudholkar_limit(eigenvalues[n_components:], alpha)

        self.n_features_in_ = X.shape[1]
        logger.info("SIMCA fitted: classes={}, n_components={}", self.classes_.tolist(),
                    self.n_components_)
        return self

    def _resolve_n_components(self, label: Any, *, rank: int) -> int:
        if rank < 2:
            log_and_raise(
                "Class {!r} needs at least 3 linearly independent samples for SIMCA", label
            )
        if isinstance(self.n_components, Mapping):
            if label not in self.n_components:
                log_and_raise("n_components has no entry for class {!r}", label)
            requested = self.n_components[label]
        else:
            requested = self.n_components
        requested = check_n_components(requested, "n_components")
        if requested >= rank:
            logger.warning(
                "SIMCA: class {!r} has rank {}; reducing n_components from {} to {}",
                label,
                rank,
                requested,
                rank - 1,
            )
            return rank - 1
        return requested

    def q_residuals(self, X: np.ndarray) -> np.ndarray:
        """Return the Q residual of every sample with respect to every class model.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_classes)
            Squared distance of each (class-scaled) spectrum to each class PCA
            subspace, in the column order of :attr:`classes_`.
        """
        check_is_fitted(self, "pca_models_")
        X = as_float_matrix(X, "X")
        check_n_features(X, self.n_features_in_, "SIMCA")
        residuals = np.empty((X.shape[0], self.classes_.size))
        for column, label in enumerate(self.classes_):
            pca = self.pca_models_[label]
            if self.scale:
                residual = self.scalers_[label].transform(X)  # a fresh copy, centred in place
                residual -= pca.mean_
            else:
                residual = X - pca.mean_
            # Remove the projection onto the class subspace in place. The residual is
            # formed explicitly: ||x||^2 - ||t||^2 would cancel catastrophically in
            # float32 when the model explains most of the variance.
            components = np.asarray(pca.components_)
            residual -= (residual @ components.T) @ components
            residuals[:, column] = np.einsum("ij,ij->i", residual, residual)
        return residuals

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign every sample to the accepting class with the smallest Q residual.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Class labels. Samples accepted by no class get ``-1`` for integer
            labels, ``"unknown"`` for string labels (also when they arrive as
            an object array, e.g. a pandas Series), ``nan`` for float labels
            and ``None`` (object dtype) otherwise.
        """
        residuals = self.q_residuals(X)
        limits = np.array([self.boundaries_[label] for label in self.classes_])
        accepted = np.where(residuals <= limits, residuals, np.inf)
        best = np.argmin(accepted, axis=1)
        rejected = np.isinf(accepted[np.arange(best.size), best])
        rejection_label, dtype = _rejection_label(self.classes_)
        labels = self.classes_[best].astype(dtype)
        labels[rejected] = rejection_label
        return labels


def _covariance_eigenvalues(X: np.ndarray) -> np.ndarray:
    """Return the non-zero covariance eigenvalues of ``X`` in descending order."""
    singular_values = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    tolerance = singular_values[0] * max(X.shape) * np.finfo(singular_values.dtype).eps
    singular_values = singular_values[singular_values > tolerance]
    return singular_values**2 / (X.shape[0] - 1)


class LDAClassifier(ClassifierMixin, EnginirEstimator):
    """Linear Discriminant Analysis with optional PCA compression (PCA-LDA).

    Spectra usually have far more (collinear) wavelengths than samples, which
    makes the LDA within-class covariance singular. By default the spectra are
    therefore autoscaled and compressed to ``n_pca_components`` principal
    components before LDA.

    Parameters
    ----------
    n_pca_components:
        Number of principal components fed to LDA. Reduced (with a warning)
        to ``min(n_samples - 1, n_wavelengths)`` if larger. ``None`` applies
        LDA directly to the unscaled spectra. Default ``20``.
    solver:
        :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
        solver: ``"svd"`` (default), ``"lsqr"`` or ``"eigen"``.

    Attributes
    ----------
    classes_ : numpy.ndarray, shape (n_classes,)
        Sorted class labels.
    scaler_ : sklearn.preprocessing.StandardScaler or None
        Autoscaling applied before PCA (``None`` without PCA).
    pca_ : sklearn.decomposition.PCA or None
        PCA compression (``None`` without PCA).
    lda_ : sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        LDA fitted on the PCA scores (or on the spectra without PCA). Its
        ``coef_`` therefore refers to PCA scores, not to wavelengths, unless
        ``n_pca_components=None``.
    n_features_in_ : int
        Number of wavelengths seen during :meth:`fit`.
    """

    def __init__(self, n_pca_components: int | None = 20, solver: str = "svd") -> None:
        self.n_pca_components = n_pca_components
        self.solver = solver

    def fit(self, X: np.ndarray, y: np.ndarray) -> LDAClassifier:
        """Fit the optional PCA compression and the LDA.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.
        y:
            Class labels, shape ``(n_samples,)``.
        """
        X = as_float_matrix(X, "X")
        y = as_class_labels(y, n_samples=X.shape[0])
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        if self.n_pca_components is not None:
            maximum = max(1, min(X.shape[0] - 1, X.shape[1]))
            requested = check_n_components(self.n_pca_components, "n_pca_components")
            if requested > maximum:
                logger.warning(
                    "LDAClassifier: reducing n_pca_components from {} to {}", requested, maximum
                )
            self.scaler_ = StandardScaler().fit(X)
            self.pca_ = PCA(n_components=min(requested, maximum)).fit(self.scaler_.transform(X))
        self.n_features_in_ = X.shape[1]
        self.lda_ = LinearDiscriminantAnalysis(solver=self.solver).fit(self._project(X), y)
        self.classes_ = self.lda_.classes_
        logger.info("LDAClassifier fitted: classes={}, X={}", self.classes_.tolist(), X.shape)
        return self

    def _project(self, X: np.ndarray) -> np.ndarray:
        if self.pca_ is None or self.scaler_ is None:
            return X
        return self.pca_.transform(self.scaler_.transform(X))

    def _checked_projection(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "lda_")
        X = as_float_matrix(X, "X")
        check_n_features(X, self.n_features_in_, "LDAClassifier")
        return self._project(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels, shape ``(n_samples,)``."""
        scores = self._checked_projection(X)
        return np.asarray(self.lda_.predict(scores))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior class probabilities, shape ``(n_samples, n_classes)``."""
        scores = self._checked_projection(X)
        return self.lda_.predict_proba(scores)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project spectra onto the discriminant axes.

        Not available for ``solver="lsqr"``.

        Returns
        -------
        numpy.ndarray, shape (n_samples, min(n_classes - 1, n_components))
        """
        if self.solver == "lsqr":
            log_and_raise(
                "transform is not available for solver='lsqr'; use 'svd' or 'eigen'",
                error=NotImplementedError,
            )
        scores = self._checked_projection(X)
        return self.lda_.transform(scores)


__all__ = ["LDAClassifier", "SIMCA", "jackson_mudholkar_limit"]
