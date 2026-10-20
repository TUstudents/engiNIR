"""Chemometric and machine-learning estimators with the scikit-learn API.

All estimators take spectra of shape ``(n_samples, n_wavelengths)``, validate
their input, and can be used inside :class:`sklearn.pipeline.Pipeline`,
:func:`enginir.validation.cross_validate` and scikit-learn model selection.

Linear regression (``y = X @ coef_ + intercept_``)
    :class:`PLSRegression` (PLS1/PLS2 with VIP scores), :class:`PCR`
Non-linear regression and classification
    :class:`SVRModel`, :class:`RandomForestModel`, :class:`MLPModel`
Classification
    :class:`SIMCA` (class modelling with a reject option),
    :class:`LDAClassifier` (PCA-LDA)
"""

from .pls import PLSRegression
from .pcr import PCR
from .ml import SVRModel, RandomForestModel, MLPModel
from .classification import SIMCA, LDAClassifier

__all__ = [
    "PLSRegression",
    "PCR",
    "SVRModel",
    "RandomForestModel",
    "MLPModel",
    "SIMCA",
    "LDAClassifier",
]
