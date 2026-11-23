"""
engiNIR top-level package.

`engiNIR` is the human-facing product name. The Python package name remains
``enginir``.

The package provides chemometric tools and project-backed workflows for
near-infrared spectra and hyperspectral imaging. The imports exposed here are
the high-level convenience surface for common loading, preprocessing, modeling,
and validation tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._lazy import lazy_names
from ._logging import disable_logging, enable_logging

if TYPE_CHECKING:  # the names below load on first use (PEP 562), see __getattr__
    __version__: str
    from ._workflow import Project
    from .io import load_spectra
    from .models import LDAClassifier, PCR, PLSRegression, SIMCA
    from .preprocessing import (Detrend, EMSC, MSC, PreprocessingRecipe, SNV, SavitzkyGolay,
                                Smoother)
    from .validation import cross_validate, kennard_stone, metrics_report

#: Public name -> the module it lives in. ``import enginir`` imports none of them, so a
#: program that only needs a few (the viewer's launcher) does not load scikit-learn,
#: SciPy, pandas or h5py before it needs them.
_LAZY = {
    "Project": "._workflow",
    "load_spectra": ".io",
    **{name: ".models" for name in ("LDAClassifier", "PCR", "PLSRegression", "SIMCA")},
    **{name: ".preprocessing" for name in ("Detrend", "EMSC", "MSC", "PreprocessingRecipe",
                                           "SNV", "SavitzkyGolay", "Smoother")},
    **{name: ".validation" for name in ("cross_validate", "kennard_stone", "metrics_report")},
}


_lazy_getattr, _lazy_dir = lazy_names(__name__, globals(), _LAZY)


def _read_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("enginir")
    except PackageNotFoundError:  # running from a source tree that was never installed
        return "0+unknown"


def __getattr__(name: str) -> Any:
    # The version is read on first use: importlib.metadata costs a cold start about 80
    # modules before the viewer's welcome window (plans/fast-start.md).
    if name == "__version__":
        value = globals()["__version__"] = _read_version()
        return value
    return _lazy_getattr(name)


def __dir__() -> list[str]:
    return sorted({*_lazy_dir(), "__version__"})

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
