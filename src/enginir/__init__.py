"""
engiNIR top-level package.

`engiNIR` is the human-facing product name. The Python package name remains
``enginir``.

The package provides chemometric tools and project-backed workflows for
near-infrared spectra and hyperspectral imaging. The imports exposed here are
the high-level convenience surface for common loading, preprocessing, modeling,
and validation tasks.
"""

from importlib.metadata import PackageNotFoundError, version as _metadata_version

from ._logging import disable_logging, enable_logging
from ._workflow import Project
from .io import load_spectra
from .models import LDAClassifier, PCR, PLSRegression, SIMCA
from .preprocessing import (Detrend, EMSC, MSC, PreprocessingRecipe, SNV, SavitzkyGolay,
                            Smoother)
from .validation import cross_validate, kennard_stone, metrics_report


try:
    __version__ = _metadata_version("enginir")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    # Project workflow
    "Project",
    "PreprocessingRecipe",
    # Logging
    "enable_logging",
    "disable_logging",
    # I/O
    "load_spectra",
    # Preprocessing
    "SNV",
    "MSC",
    "EMSC",
    "SavitzkyGolay",
    "Detrend",
    "Smoother",
    # Models
    "PLSRegression",
    "PCR",
    "SIMCA",
    "LDAClassifier",
    # Validation
    "cross_validate",
    "kennard_stone",
    "metrics_report",
]
