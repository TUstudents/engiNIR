"""Base class of engiNIR estimators and transformers: version-checked pickles."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from sklearn.base import BaseEstimator

from ._checks import log_and_raise

__all__ = ["EnginirEstimator"]


def _enginir_version() -> str:
    try:
        return version("enginir")
    except PackageNotFoundError:  # a source tree that was never installed
        return "0+unknown"


class EnginirEstimator(BaseEstimator):
    """A scikit-learn estimator whose pickles record the engiNIR version.

    Fitted state of engiNIR 1.0 is not compatible with earlier releases (for
    example, the PLS intercept now applies to uncentred spectra). A pickle
    without a recorded version therefore comes from before 1.0 and is refused
    instead of silently giving wrong results: fit the model again.
    """

    def __getstate__(self) -> dict[str, Any]:
        base = super().__getstate__()
        state: dict[str, Any] = dict(base) if isinstance(base, dict) else {}
        state["_enginir_version"] = _enginir_version()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if "_enginir_version" not in state:
            log_and_raise("This {} was saved by engiNIR before 1.0, whose fitted state means "
                          "something else; fit it again with engiNIR {}",
                          type(self).__name__, _enginir_version())
        super().__setstate__(state)
