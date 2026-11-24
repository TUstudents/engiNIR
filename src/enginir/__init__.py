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

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _metadata_version
from typing import TYPE_CHECKING, Any

from ._logging import disable_logging, enable_logging

if TYPE_CHECKING:  # the names below load on first use (PEP 562), see __getattr__
    from ._workflow import GridCell, Project, read_frames, read_header, roi_grid
    from .io import load_spectra
    from .models import LDAClassifier, PCR, PLSRegression, SIMCA
    from .preprocessing import (Detrend, EMSC, MSC, PreprocessingRecipe, SNV, SavitzkyGolay,
                                Smoother)
    from .validation import cross_validate, kennard_stone, metrics_report

#: Public name -> the module it lives in. ``import enginir`` imports none of them, so a
#: program that only needs a few (the viewer's launcher) does not load scikit-learn,
#: SciPy, pandas or h5py before it needs them.
_LAZY = {
    **{name: "._workflow" for name in ("GridCell", "Project", "read_frames",
                                         "read_header", "roi_grid")},
    "load_spectra": ".io",
    **{name: ".models" for name in ("LDAClassifier", "PCR", "PLSRegression", "SIMCA")},
    **{name: ".preprocessing" for name in ("Detrend", "EMSC", "MSC", "PreprocessingRecipe",
                                           "SNV", "SavitzkyGolay", "Smoother")},
    **{name: ".validation" for name in ("cross_validate", "kennard_stone", "metrics_report")},
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'enginir' has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value  # resolved once
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})

try:
    __version__ = _metadata_version("enginir")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    # Project workflow
    "Project",
    "GridCell",
    "roi_grid",
    "read_header",
    "read_frames",
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
