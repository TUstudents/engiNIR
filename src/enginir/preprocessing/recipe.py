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
from .selection import BandSelection

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
_BAND_FIELDS = ("bands_keep", "bands_exclude")
#: A band lies in a range when low - tau <= wavelength <= high + tau, with tau
#: this share of the smallest band spacing (contract §4: typed bounds match
#: float32 wavelengths), or _SINGLE_BAND_TOL for a one-band axis.
_RANGE_TOL_SHARE = 1e-3
_SINGLE_BAND_TOL = 1e-6
#: Ranges already reported as selecting nothing (one warning per range and axis).
_WARNED_EMPTY: set[tuple[str, tuple[float, float], int, float, float]] = set()


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
    7. Band selection (``bands_keep``, ``bands_exclude``)

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
    bands_keep, bands_exclude:
        Inclusive ``(low, high)`` ranges on the axis after the Savitzky-Golay
        step (wavelengths in nm; band numbers 0 … K − 1 for data without
        wavelengths). A band is kept when it lies in a keep range (or no keep
        range is given) and in no exclude range, e.g. ``bands_exclude=[(1400,
        1500)]`` leaves out a water band. The selection is the last step: SNV,
        MSC, EMSC and Detrend see every band, the model only the selected
        ones (:meth:`output_wavelengths`). Normalized: each pair ordered, the
        ranges sorted, overlapping or touching ranges merged.

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
    bands_keep: tuple[tuple[float, float], ...] = ()
    bands_exclude: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        for name in _BAND_FIELDS:
            object.__setattr__(self, name, _normalize_ranges(getattr(self, name), name))
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

    @property
    def has_band_selection(self) -> bool:
        """``True`` if ``bands_keep`` or ``bands_exclude`` holds a range."""
        return bool(self.bands_keep or self.bands_exclude)

    def _trimmed(self, wavelengths: np.ndarray) -> np.ndarray:
        """The axis after the Savitzky-Golay step (shorter with ``sg_edges="drop"``)."""
        trim = self.sg_trim
        if trim and wavelengths.shape[-1] <= 2 * trim:
            log_and_raise("sg_window={} leaves no band of the {} available", self.sg_window,
                          wavelengths.shape[-1], depth=3)
        return wavelengths[..., trim:wavelengths.shape[-1] - trim] if trim else wavelengths

    def selected_columns(self, axis: np.ndarray) -> np.ndarray:
        """Indices into ``axis`` (the axis after Savitzky-Golay) of the selected bands.

        Every band without a band selection. Raises ``ValueError`` when the
        ranges select no band; a range that selects nothing is logged once.
        """
        axis = np.asarray(axis, dtype=np.float64).reshape(-1)
        if not self.has_band_selection:
            return np.arange(axis.size)
        spacing = np.abs(np.diff(axis))
        spacing = spacing[spacing > 0]
        tol = _RANGE_TOL_SHARE * float(spacing.min()) if spacing.size else _SINGLE_BAND_TOL
        keep = np.ones(axis.size, dtype=bool)
        if self.bands_keep:
            keep = np.zeros(axis.size, dtype=bool)
            for low, high in self.bands_keep:
                keep |= self._in_range(axis, low, high, tol, "bands_keep")
        for low, high in self.bands_exclude:
            keep &= ~self._in_range(axis, low, high, tol, "bands_exclude")
        if not keep.any():
            log_and_raise("The band selection (keep {}, exclude {}) selects none of the {} bands "
                          "({:g} to {:g})", list(self.bands_keep), list(self.bands_exclude),
                          axis.size, float(axis.min()) if axis.size else 0.0,
                          float(axis.max()) if axis.size else 0.0, depth=2)
        return np.flatnonzero(keep)

    @staticmethod
    def _in_range(axis: np.ndarray, low: float, high: float, tol: float,
                  name: str) -> np.ndarray:
        inside = (axis >= low - tol) & (axis <= high + tol)
        if not inside.any() and axis.size:
            key = (name, (low, high), axis.size, float(axis[0]), float(axis[-1]))
            if key not in _WARNED_EMPTY:
                _WARNED_EMPTY.add(key)
                logger.warning("{} range {:g}-{:g} selects none of the {} bands ({:g} to {:g})",
                               name, low, high, axis.size, float(axis.min()), float(axis.max()))
        return inside

    def output_band_indices(self, wavelengths: np.ndarray) -> np.ndarray:
        """Indices into the input axis of the bands the preprocessed spectra keep.

        Without Savitzky-Golay edge dropping and band selection, every band.
        """
        wavelengths = np.asarray(wavelengths).reshape(-1)
        trimmed = self._trimmed(np.arange(wavelengths.size))
        if not self.has_band_selection:
            return trimmed
        return trimmed[self.selected_columns(wavelengths[trimmed])]

    def output_wavelengths(self, wavelengths: np.ndarray) -> np.ndarray:
        """The wavelength axis of the preprocessed spectra.

        Shorter with ``sg_edges="drop"``, and only the selected bands (with
        gaps) with a band selection.
        """
        wavelengths = np.asarray(wavelengths)
        trimmed = self._trimmed(wavelengths)
        if not self.has_band_selection:
            return trimmed
        return trimmed[..., self.selected_columns(trimmed)]

    def n_bands_before_selection(self, n_bands: int) -> int:
        """Bands after the Savitzky-Golay step (what MSC, EMSC and Detrend see)."""
        return int(n_bands) - 2 * self.sg_trim

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
            Steps named ``"asls"``, ``"sg"``, ``"snv"``, ``"msc"``, ``"emsc"``,
            ``"detrend"`` and ``"bands"``, in application order, suitable for
            :class:`sklearn.pipeline.Pipeline`.

        Raises
        ------
        ValueError
            If ``sg_window`` exceeds the number of wavelengths, ``asls``
            is enabled without a signal unit, or the band selection selects
            no band.
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
        if self.has_band_selection:
            steps.append(("bands", BandSelection(indices=self.selected_columns(wavelengths))))
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
        if self.has_band_selection:
            parts.append(f"Bands ({self.describe_bands()})")
        return " > ".join(parts) if parts else "none"

    def describe_bands(self) -> str:
        """The band selection as text, e.g. ``"keep 1100-1350, 1500-2300; exclude 1400-1500"``."""
        parts = [f"{name} {format_ranges(ranges)}" for name, ranges in
                 (("keep", self.bands_keep), ("exclude", self.bands_exclude)) if ranges]
        return "; ".join(parts) or "all"

    def to_dict(self) -> dict[str, Any]:
        """Return the recipe as a plain dict (the persisted HDF5 representation).

        The fields added in engiNIR 1.3 and 1.4 are included only when they
        change what the recipe does (``sg_edges="drop"`` with a Savitzky-Golay
        filter; the AsLS fields with ``asls``; band ranges when given), so
        recipes stored before keep their content and id.
        """
        payload = {name: value for name, value in asdict(self).items() if name in _REQUIRED}
        if self.filter in SG_FILTERS and self.sg_edges == "drop":
            payload["sg_edges"] = "drop"
        if self.asls:
            payload["asls"] = True
            payload.update({name: getattr(self, name) for name in _ASLS_FIELDS})
        for name in _BAND_FIELDS:
            if getattr(self, name):
                payload[name] = [list(pair) for pair in getattr(self, name)]
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
        required. No ``bands_keep``/``bands_exclude`` selects every band.
        Unknown keys are ignored with a warning.

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


def _normalize_ranges(value: Any, name: str) -> tuple[tuple[float, float], ...]:
    """Ranges as sorted, merged ``(low, high)`` float pairs (project-schema contract §4)."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)):
        log_and_raise("{} must be a list of (low, high) pairs, got {!r}", name, value, depth=3)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        log_and_raise("{} must be a list of (low, high) number pairs, got {!r}", name, value,
                      depth=3)
    if array.size == 0:
        return ()
    if array.ndim == 1 and array.size == 2:
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        log_and_raise("{} must be a list of (low, high) pairs, got shape {}", name, array.shape,
                      depth=3)
    if not np.all(np.isfinite(array)):
        log_and_raise("{} must hold finite numbers, got {}", name, array.tolist(), depth=3)
    pairs = sorted((float(min(a, b)), float(max(a, b))) for a, b in array)
    merged: list[list[float]] = [list(pairs[0])]
    for low, high in pairs[1:]:
        if low <= merged[-1][1]:  # overlapping or touching
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return tuple((low, high) for low, high in merged)


def format_ranges(ranges: Any) -> str:
    """Ranges as text, ``"1100-1350, 1500-2300"`` (the viewer's input format)."""
    return ", ".join(f"{low:g}-{high:g}" for low, high in ranges)


def parse_ranges(text: str, name: str = "ranges") -> tuple[tuple[float, float], ...]:
    """Parse ``"1100-1350, 1500-2300"`` (a dash or en dash between the numbers).

    Raises ``ValueError`` for anything else; an empty text is no range.
    """
    ranges: list[tuple[float, float]] = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        bounds = part.replace("–", "-").replace("—", "-")
        # A minus sign may start a number (band axes are not negative, but be exact).
        split = bounds.find("-", 1)
        if split <= 0:
            log_and_raise("{}: '{}' is not a range like 1100-1350", name, part, depth=2)
        try:
            ranges.append((float(bounds[:split]), float(bounds[split + 1:])))
        except ValueError:
            log_and_raise("{}: '{}' is not a range like 1100-1350", name, part, depth=2)
    return _normalize_ranges(ranges, name)


def _truthy(value: Any) -> bool:
    """A stored boolean (``bool`` or ``numpy.bool_``); anything else is left to validation."""
    return isinstance(value, (bool, np.bool_)) and bool(value)


__all__ = ["PreprocessingRecipe", "SG_FILTERS", "format_ranges", "parse_ranges"]
