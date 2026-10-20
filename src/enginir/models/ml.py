"""Machine-learning model wrappers: SVR, random forest and multilayer perceptron.

The wrappers give scikit-learn's general-purpose learners the same input
validation, logging and defaults as the chemometric models in this package.
They do not scale the spectra; put a scaler or scatter correction in a
:class:`sklearn.pipeline.Pipeline` in front of them where the learner needs it
(SVR with an RBF kernel and MLPs usually do).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from .._estimator import EnginirEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR
from sklearn.utils import ClassifierTags, RegressorTags
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from enginir._checks import as_float_matrix, check_choice, check_n_features
from enginir._logging import get_logger

from ._common import as_class_labels, as_target_array

logger = get_logger(__name__)

_TASKS = ("regression", "classification")


class _DelegatingModel(EnginirEstimator):
    """Base class that fits and delegates to one scikit-learn estimator.

    Subclasses implement :meth:`_build_estimator`. Subclasses with a ``task``
    parameter act as a classifier when ``task == "classification"`` and as a
    regressor otherwise, including the scikit-learn estimator tags.
    """

    def _is_classifier(self) -> bool:
        return getattr(self, "task", "regression") == "classification"

    def _build_estimator(self) -> Any:
        raise NotImplementedError

    def __sklearn_tags__(self):  # type: ignore[override]
        tags = super().__sklearn_tags__()
        if self._is_classifier():
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags()
            tags.regressor_tags = None
        else:
            tags.estimator_type = "regressor"
            tags.regressor_tags = RegressorTags()
            tags.classifier_tags = None
        return tags

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Fit the wrapped estimator.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_wavelengths)``.
        y:
            Reference values, shape ``(n_samples,)`` (or
            ``(n_samples, n_targets)`` where the learner supports it), or
            class labels, shape ``(n_samples,)``, when ``task="classification"``.

        Returns
        -------
        self
        """
        check_choice(getattr(self, "task", "regression"), "task", _TASKS)
        X = as_float_matrix(X, "X")
        if self._is_classifier():
            y = as_class_labels(y, n_samples=X.shape[0])
        else:
            y = as_target_array(y, n_samples=X.shape[0])
        self.estimator_ = self._build_estimator().fit(X, y)
        if self._is_classifier():
            self.classes_ = self.estimator_.classes_
        self.n_features_in_ = X.shape[1]
        logger.info("{} fitted: X={}, y={}", type(self).__name__, X.shape, y.shape)
        return self

    def _checked_input(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        X = as_float_matrix(X, "X")
        check_n_features(X, self.n_features_in_, type(self).__name__)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict reference values or class labels for spectra of shape ``(n_samples, n_wavelengths)``."""
        X = self._checked_input(X)
        return self.estimator_.predict(X)

    @available_if(lambda self: self._is_classifier())
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape ``(n_samples, n_classes)`` (classification only)."""
        X = self._checked_input(X)
        return self.estimator_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: Any = None) -> float:
        """Return accuracy for classification and the coefficient of determination R² otherwise."""
        metric = accuracy_score if self._is_classifier() else r2_score
        return float(metric(y, self.predict(X), sample_weight=sample_weight))


class SVRModel(_DelegatingModel):
    """Epsilon-support vector regression (wraps :class:`sklearn.svm.SVR`).

    Parameters
    ----------
    kernel:
        ``"rbf"`` (default), ``"linear"``, ``"poly"`` or ``"sigmoid"``.
    C:
        Regularisation strength; larger values fit the training data more
        closely. Default ``1.0``.
    epsilon:
        Half-width of the loss-free tube around the predictions, in units of
        ``y``. Default ``0.1``.
    gamma:
        Kernel coefficient: ``"scale"`` (default, ``1 / (n_wavelengths *
        X.var())``), ``"auto"`` or a positive float.

    Attributes
    ----------
    estimator_ : sklearn.svm.SVR
        Fitted SVR.
    n_features_in_ : int
        Number of wavelengths seen during :meth:`fit`.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: str | float = "scale",
    ) -> None:
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma

    def _build_estimator(self) -> SVR:
        return SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma=self.gamma)  # type: ignore[arg-type]


class RandomForestModel(_DelegatingModel):
    """Random-forest regression or classification.

    Wraps :class:`sklearn.ensemble.RandomForestRegressor` or
    :class:`sklearn.ensemble.RandomForestClassifier`, depending on ``task``.

    Parameters
    ----------
    n_estimators:
        Number of trees. Default ``200``.
    max_features:
        Number (int), fraction (float) or rule (``"sqrt"``, ``"log2"``) of
        wavelengths considered at each split. Default ``"sqrt"``.
    task:
        ``"regression"`` (default) or ``"classification"``.
    random_state:
        Seed for bootstrapping and feature sampling. Default ``42``.

    Attributes
    ----------
    estimator_ : RandomForestRegressor or RandomForestClassifier
        Fitted forest.
    classes_ : numpy.ndarray, shape (n_classes,)
        Class labels (classification only).
    n_features_in_ : int
        Number of wavelengths seen during :meth:`fit`.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_features: str | int | float = "sqrt",
        task: str = "regression",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.task = task
        self.random_state = random_state

    def _build_estimator(self) -> RandomForestRegressor | RandomForestClassifier:
        forest = RandomForestClassifier if self._is_classifier() else RandomForestRegressor
        return forest(
            n_estimators=self.n_estimators,
            max_features=self.max_features,  # type: ignore[arg-type]
            random_state=self.random_state,
        )

    @property
    def feature_importances_(self) -> np.ndarray:
        """Impurity-based importance of every wavelength, shape ``(n_wavelengths,)``.

        Mean decrease in squared error (regression) or Gini impurity
        (classification), normalised to sum to 1.
        """
        check_is_fitted(self, "estimator_")
        return self.estimator_.feature_importances_


class MLPModel(_DelegatingModel):
    """Multilayer-perceptron regression or classification.

    Wraps :class:`sklearn.neural_network.MLPRegressor` or
    :class:`sklearn.neural_network.MLPClassifier` (Adam optimiser), depending
    on ``task``.

    Parameters
    ----------
    hidden_layer_sizes:
        Number of neurons in each hidden layer. Default ``(128, 64)``.
    activation:
        Hidden-layer activation: ``"relu"`` (default), ``"tanh"``,
        ``"logistic"`` or ``"identity"``.
    max_iter:
        Maximum number of training epochs. Default ``500``.
    learning_rate_init:
        Initial Adam learning rate. Default ``1e-3``.
    task:
        ``"regression"`` (default) or ``"classification"``.
    random_state:
        Seed for weight initialisation and batch shuffling. Default ``42``.

    Attributes
    ----------
    estimator_ : MLPRegressor or MLPClassifier
        Fitted network.
    classes_ : numpy.ndarray, shape (n_classes,)
        Class labels (classification only).
    n_features_in_ : int
        Number of wavelengths seen during :meth:`fit`.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        activation: str = "relu",
        max_iter: int = 500,
        learning_rate_init: float = 1e-3,
        task: str = "regression",
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.task = task
        self.random_state = random_state

    def _build_estimator(self) -> MLPRegressor | MLPClassifier:
        network = MLPClassifier if self._is_classifier() else MLPRegressor
        return network(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,  # type: ignore[arg-type]
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            random_state=self.random_state,
        )


__all__ = ["MLPModel", "RandomForestModel", "SVRModel"]
