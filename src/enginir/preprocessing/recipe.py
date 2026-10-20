"""Serializable preprocessing recipe shared by the viewer, model fitting and prediction.

A :class:`PreprocessingRecipe` is the single definition of how a recipe turns
into scikit-learn transformers (:meth:`PreprocessingRecipe.build_steps`), so the
viewer preview, model training and prediction replay always apply identical
preprocessing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin

from enginir._checks import check_choice, check_integer, log_and_raise
from enginir._logging import get_logger

from .baseline import Detrend
from .derivatives import SavitzkyGolay
from .scatter import EMSC, MSC, SNV

logger = get_logger(__name__)

#: Savitzky-Golay filter modes and the derivative order each one computes.
SG_FILTERS: dict[str, int] = {"sg_smooth": 0, "sg_d1": 1, "sg_d2": 2}
_FILTERS = ("none", *SG_FILTERS)
_BOOLEAN_FIELDS = ("snv", "msc", "emsc", "detrend")


@dataclass(frozen=True)
class PreprocessingRecipe:
    """Immutable description of a preprocessing pipeline.

    Enabled steps are always applied in this fixed order:

    1. Savitzky-Golay smoothing or derivative (``filter``)
    2. SNV
    3. MSC
    4. EMSC (mean reference, quadratic baseline)
    5. Detrend (polynomial of degree ``detrend_order``)

    Parameters
    ----------
    filter:
        ``"none"``, ``"sg_smooth"`` (Savitzky-Golay smoothing), ``"sg_d1"``
        (first derivative) or ``"sg_d2"`` (second derivative). Derivatives are
        computed per wavelength band (``delta=1``).
    sg_window:
        Savitzky-Golay window length, an odd integer ``>= 3``.
    sg_polyorder:
        Savitzky-Golay polynomial degree, ``>= 1``, ``< sg_window`` and at
        least the derivative order of ``filter``.
    snv, msc, emsc, detrend:
        Whether each scatter-correction or baseline step is enabled.
    detrend_order:
        Polynomial degree for Detrend, ``>= 0``.

    Raises
    ------
    ValueError
        If a field has the wrong type or the fields are inconsistent.
    """

    filter: str = "none"
    sg_window: int = 11
    sg_polyorder: int = 2
    snv: bool = False
    msc: bool = False
    emsc: bool = False
    detrend: bool = False
    detrend_order: int = 1

    def __post_init__(self) -> None:
        check_choice(self.filter, "filter", _FILTERS)
        for name in _BOOLEAN_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                log_and_raise("{} must be a bool, got {!r}", name, value)
            object.__setattr__(self, name, bool(value))
        minimums = {"sg_window": 3, "sg_polyorder": 1, "detrend_order": 0}
        for name, minimum in minimums.items():
            value = check_integer(getattr(self, name), name, minimum=minimum)
            object.__setattr__(self, name, value)
        sg_window, sg_polyorder = self.sg_window, self.sg_polyorder
        if sg_window % 2 == 0:
            log_and_raise("sg_window must be odd, got {}", sg_window)
        if sg_polyorder >= sg_window:
            log_and_raise(
                "sg_polyorder must be smaller than sg_window, got {} >= {}",
                sg_polyorder,
                sg_window,
            )
        deriv = SG_FILTERS.get(self.filter, 0)
        if sg_polyorder < deriv:
            log_and_raise(
                "filter {!r} needs sg_polyorder >= {}, got {}", self.filter, deriv, sg_polyorder
            )

    @property
    def requires_fitting(self) -> bool:
        """``True`` if a step (MSC, EMSC) learns a reference from training data."""
        return self.msc or self.emsc

    def build_steps(self, wavelengths: np.ndarray) -> list[tuple[str, TransformerMixin]]:
        """Return new, unfitted ``(name, transformer)`` pairs for a pipeline.

        Parameters
        ----------
        wavelengths:
            Wavelength axis, shape ``(n_wavelengths,)``. Used as the Detrend
            abscissa.

        Returns
        -------
        list of (str, transformer)
            Steps named ``"sg"``, ``"snv"``, ``"msc"``, ``"emsc"`` and
            ``"detrend"``, in application order, suitable for
            :class:`sklearn.pipeline.Pipeline`.

        Raises
        ------
        ValueError
            If ``sg_window`` exceeds the number of wavelengths.
        """
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        if wavelengths.ndim != 1:
            log_and_raise("wavelengths must be 1-D, got shape {}", wavelengths.shape)
        steps: list[tuple[str, TransformerMixin]] = []
        if self.filter in SG_FILTERS:
            if self.sg_window > wavelengths.size:
                log_and_raise(
                    "sg_window={} exceeds the {} available wavelengths",
                    self.sg_window,
                    wavelengths.size,
                )
            sg = SavitzkyGolay(
                window_length=self.sg_window,
                polyorder=self.sg_polyorder,
                deriv=SG_FILTERS[self.filter],
            )
            steps.append(("sg", sg))
        if self.snv:
            steps.append(("snv", SNV()))
        if self.msc:
            steps.append(("msc", MSC()))
        if self.emsc:
            steps.append(("emsc", EMSC()))
        if self.detrend:
            steps.append(("detrend", Detrend(order=self.detrend_order, wavelengths=wavelengths)))
        return steps

    def describe(self) -> str:
        """Return a one-line human-readable summary, e.g. ``"SG d1 (w=11, p=2) > SNV"``."""
        labels = {"sg_smooth": "SG smooth", "sg_d1": "SG d1", "sg_d2": "SG d2"}
        parts: list[str] = []
        if self.filter in SG_FILTERS:
            parts.append(f"{labels[self.filter]} (w={self.sg_window}, p={self.sg_polyorder})")
        parts.extend(name.upper() for name in ("snv", "msc", "emsc") if getattr(self, name))
        if self.detrend:
            parts.append(f"Detrend (order={self.detrend_order})")
        return " > ".join(parts) if parts else "none"

    def to_dict(self) -> dict[str, Any]:
        """Return the recipe as a plain dict (the persisted HDF5 representation)."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        location: str = "preprocessing recipe",
    ) -> PreprocessingRecipe:
        """Construct a recipe from a complete dict as produced by :meth:`to_dict`.

        Every field is required: a persisted recipe defines how a model was
        trained, so missing values are never filled from defaults. Unknown
        keys are ignored with a warning.

        Parameters
        ----------
        payload:
            Mapping with all recipe fields.
        location:
            Where the payload came from (e.g. an HDF5 path), used in messages.

        Raises
        ------
        ValueError
            If ``payload`` is not a mapping, misses fields or has invalid values.
        """
        if not isinstance(payload, Mapping):
            log_and_raise("{} must be a mapping, got {}", location, type(payload).__name__)
        names = [field.name for field in fields(cls)]
        missing = [name for name in names if name not in payload]
        if missing:
            log_and_raise("{} is missing required field(s): {}", location, missing)
        extras = sorted(set(payload) - set(names))
        if extras:
            logger.warning("Ignoring unknown field(s) in {}: {}", location, extras)
        return cls(**{name: payload[name] for name in names})


__all__ = ["PreprocessingRecipe", "SG_FILTERS"]
