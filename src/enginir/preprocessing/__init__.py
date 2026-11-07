"""Spectral preprocessing transformers.

All transformers follow the scikit-learn API (``fit`` / ``transform`` /
``fit_transform``) and can be combined in a :class:`sklearn.pipeline.Pipeline`.
Inputs are 2-D arrays of shape ``(n_samples, n_wavelengths)``; floating dtypes
(including ``float32``) are preserved.

Scatter correction
    :class:`SNV`, :class:`MSC`, :class:`EMSC`
Smoothing and derivatives
    :class:`SavitzkyGolay`, :class:`Smoother`
Baseline removal
    :class:`AsLSBaseline`, :class:`Detrend`
Serializable pipelines
    :class:`PreprocessingRecipe` describes a preprocessing pipeline as a value
    object that is stored in project files and rebuilt for model fitting and
    prediction.
"""

from .baseline import AsLSBaseline, Detrend, Smoother
from .derivatives import SavitzkyGolay
from .recipe import PreprocessingRecipe
from .scatter import EMSC, MSC, SNV

__all__ = [
    "AsLSBaseline",
    "Detrend",
    "EMSC",
    "MSC",
    "PreprocessingRecipe",
    "SNV",
    "SavitzkyGolay",
    "Smoother",
]
