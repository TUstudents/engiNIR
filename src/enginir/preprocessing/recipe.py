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

from enginir._checks import check_choice, check_integer, check_real, log_and_raise
from enginir._logging import get_logger

from .baseline import AsLSBaseline, Detrend
from .derivatives import SG_EDGES, SavitzkyGolay
from .scatter import EMSC, MSC, SNV

logger = get_logger(__name__)

#: Savitzky-Golay filter modes and the derivative order each one computes.
SG_FILTERS: dict[str, int] = {"sg_smooth": 0, "sg_d1": 1, "sg_d2": 2}
_FILTERS = ("none", *SG_FILTERS)
_BOOLEAN_FIELDS = ("snv", "msc", "emsc", "detrend", "asls")
#: The fields of every stored recipe (project-schema contract §4).
_REQUIRED = ("filter", "sg_window", "sg_polyorder", "snv", "msc", "emsc", "detrend",
             "detrend_order")
_ASLS_FIELDS = ("asls_lambda", "asls_p", "asls_iterations")
_ASLS_DEFAULTS = {"asls_lambda": 1e4, "asls_p": 0.001, "asls_iterations": 10}
#: Signal units whose peaks point up (AsLS as is); other units are flipped.
_PEAKS_UP = ("absorbance",)


@dataclass(frozen=True)
class PreprocessingRecipe:
    """Immutable description of a preprocessing pipeline.

    Enabled steps are always applied in this fixed order:

    1. AsLS baseline correction (``asls``)
    2. Savitzky-Golay smoothing or derivative (``filter``)
    3. SNV
    4. MSC
    5. EMSC (mean reference, quadratic baseline)
    6. Detrend (polynomial of degree ``detrend_order``)

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
    sg_edges:
        ``"drop"`` (default) leaves out the first and last ``sg_window // 2``
        bands of a Savitzky-Golay filter, as SIMCA does, so the spectra get
        fewer bands (:meth:`output_wavelengths`); ``"interp"`` extrapolates
        them (recipes stored before engiNIR 1.3). No effect without a
        Savitzky-Golay filter.
    asls:
        Asymmetric least-squares baseline correction
        (:class:`~enginir.preprocessing.AsLSBaseline`), applied first. In
        absorbance the baseline lies under the spectrum; in reflectance and
        DN, where absorption bands are dips, over it (the direction comes
        from the model's signal unit).
    asls_lambda, asls_p, asls_iterations:
        AsLS stiffness λ (``> 0``, default ``1e4``; more bands need a larger
        λ), asymmetry p (``0 < p < 0.5``, default ``0.001``) and number of
        weighted solves (``>= 1``, default ``10``). No effect without
        ``asls``.

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
    sg_edges: str = "drop"
    asls: bool = False
    asls_lambda: float = 1e4
    asls_p: float = 0.001
    asls_iterations: int = 10

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
        check_choice(self.sg_edges, "sg_edges", SG_EDGES)
        if self.filter not in SG_FILTERS:
            object.__setattr__(self, "sg_edges", "drop")  # no effect: one normal form
        if not self.asls:
            for name, value in _ASLS_DEFAULTS.items():  # no effect: one normal form
                object.__setattr__(self, name, value)
            return
        lam = check_real(self.asls_lambda, "asls_lambda", positive=True)
        p = check_real(self.asls_p, "asls_p")
        if not 0.0 < p < 0.5:
            log_and_raise("asls_p must be between 0 and 0.5, got {}", p)
        object.__setattr__(self, "asls_lambda", float(lam))
        object.__setattr__(self, "asls_p", float(p))
        object.__setattr__(self, "asls_iterations",
                           check_integer(self.asls_iterations, "asls_iterations", minimum=1))

    @property
    def sg_trim(self) -> int:
        """Bands left out at each end by the Savitzky-Golay step (0 unless ``sg_edges="drop"``)."""
        if self.filter in SG_FILTERS and self.sg_edges == "drop":
            return self.sg_window // 2
        return 0

    def output_wavelengths(self, wavelengths: np.ndarray) -> np.ndarray:
        """The wavelength axis of the preprocessed spectra (shorter with ``sg_edges="drop"``)."""
        wavelengths = np.asarray(wavelengths)
        trim = self.sg_trim
        if trim and wavelengths.shape[-1] <= 2 * trim:
            log_and_raise("sg_window={} leaves no band of the {} available", self.sg_window,
                          wavelengths.shape[-1])
        return wavelengths[..., trim:wavelengths.shape[-1] - trim] if trim else wavelengths

    @property
    def requires_fitting(self) -> bool:
        """``True`` if a step (MSC, EMSC) learns a reference from training data."""
        return self.msc or self.emsc

    def build_steps(self, wavelengths: np.ndarray, *,
                    signal_unit: str | None = None) -> list[tuple[str, TransformerMixin]]:
        """Return new, unfitted ``(name, transformer)`` pairs for a pipeline.

        Parameters
        ----------
        wavelengths:
            Wavelength axis of the input spectra, shape ``(n_wavelengths,)``.
            Detrend uses the axis after the Savitzky-Golay step.
        signal_unit:
            Signal unit of the input spectra (``"Absorbance"``,
            ``"Reflectance"`` or ``"DN"``); required with ``asls``, whose
            direction depends on it.

        Returns
        -------
        list of (str, transformer)
            Steps named ``"asls"``, ``"sg"``, ``"snv"``, ``"msc"``, ``"emsc"``
            and ``"detrend"``, in application order, suitable for
            :class:`sklearn.pipeline.Pipeline`.

        Raises
        ------
        ValueError
            If ``sg_window`` exceeds the number of wavelengths, or ``asls``
            is enabled without a signal unit.
        """
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        if wavelengths.ndim != 1:
            log_and_raise("wavelengths must be 1-D, got shape {}", wavelengths.shape)
        steps: list[tuple[str, TransformerMixin]] = []
        if self.asls:
            steps.append(("asls", AsLSBaseline(lam=self.asls_lambda, p=self.asls_p,
                                               n_iter=self.asls_iterations,
                                               direction=self.asls_direction(signal_unit))))
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
                edges=self.sg_edges,
            )
            steps.append(("sg", sg))
            wavelengths = self.output_wavelengths(wavelengths)
        if self.snv:
            steps.append(("snv", SNV()))
        if self.msc:
            steps.append(("msc", MSC()))
        if self.emsc:
            steps.append(("emsc", EMSC()))
        if self.detrend:
            steps.append(("detrend", Detrend(order=self.detrend_order, wavelengths=wavelengths)))
        return steps

    def asls_direction(self, signal_unit: str | None) -> str:
        """``"up"`` for absorbance, ``"down"`` for reflectance and DN (see ``asls``)."""
        if signal_unit is None or not str(signal_unit).strip():
            log_and_raise("AsLS baseline correction needs the signal unit of the spectra "
                          "(its direction depends on it)", depth=2)
        return "up" if str(signal_unit).strip().lower() in _PEAKS_UP else "down"

    def describe(self) -> str:
        """Return a one-line human-readable summary, e.g. ``"SG d1 (w=11, p=2) > SNV"``."""
        labels = {"sg_smooth": "SG smooth", "sg_d1": "SG d1", "sg_d2": "SG d2"}
        parts: list[str] = []
        if self.asls:
            parts.append(f"AsLS (λ={self.asls_lambda:g}, p={self.asls_p:g})")
        if self.filter in SG_FILTERS:
            edges = "" if self.sg_edges == "drop" else ", edges kept"
            parts.append(f"{labels[self.filter]} (w={self.sg_window}, p={self.sg_polyorder}"
                         f"{edges})")
        parts.extend(name.upper() for name in ("snv", "msc", "emsc") if getattr(self, name))
        if self.detrend:
            parts.append(f"Detrend (order={self.detrend_order})")
        return " > ".join(parts) if parts else "none"

    def to_dict(self) -> dict[str, Any]:
        """Return the recipe as a plain dict (the persisted HDF5 representation).

        The fields added in engiNIR 1.3 are included only when they change
        what the recipe does (``sg_edges="drop"`` with a Savitzky-Golay
        filter; the AsLS fields with ``asls``), so recipes stored before keep
        their content and id.
        """
        payload = {name: value for name, value in asdict(self).items() if name in _REQUIRED}
        if self.filter in SG_FILTERS and self.sg_edges == "drop":
            payload["sg_edges"] = "drop"
        if self.asls:
            payload["asls"] = True
            payload.update({name: getattr(self, name) for name in _ASLS_FIELDS})
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        location: str = "preprocessing recipe",
    ) -> PreprocessingRecipe:
        """Construct a recipe from a dict as produced by :meth:`to_dict`.

        The eight fields of every stored recipe are required: a persisted
        recipe defines how a model was trained, so missing values are never
        filled from defaults. The fields added in 1.3 mean what a recipe
        stored before does when absent: no ``sg_edges`` is ``"interp"``, no
        ``asls`` is ``False``; with ``asls`` the three AsLS parameters are
        required. Unknown keys are ignored with a warning.

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
        required = list(_REQUIRED)
        if _truthy(payload.get("asls", False)):
            required += list(_ASLS_FIELDS)
        missing = [name for name in required if name not in payload]
        if missing:
            log_and_raise("{} is missing required field(s): {}", location, missing)
        extras = sorted(set(payload) - set(names))
        if extras:
            logger.warning("Ignoring unknown field(s) in {}: {}", location, extras)
        values = {name: payload[name] for name in names if name in payload}
        values.setdefault("sg_edges", "interp")
        values.setdefault("asls", False)
        return cls(**values)


def _truthy(value: Any) -> bool:
    """A stored boolean (``bool`` or ``numpy.bool_``); anything else is left to validation."""
    return isinstance(value, (bool, np.bool_)) and bool(value)


__all__ = ["PreprocessingRecipe", "SG_FILTERS"]
