"""The fit specification of a model run (``sample-set-model-run-contract_v3.md`` §7).

A :class:`FitSpec` records how a model was fitted: the fold rule, the pixel
sample and the DModX exclusion. Every new model run stores its complete
specification as the nested mapping of :meth:`FitSpec.to_dict`; a run without
one is a legacy run (:meth:`FitSpec.legacy`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .._checks import log_and_raise
from .._logging import get_logger

logger = get_logger(__name__)

FOLD_RULES = ("round_robin", "leave_one_group_out", "group_kfold")
GROUP_BY = ("roi", "sample", "target")
COMPONENT_RULES = ("fixed", "one_se", "min_rmsecv", "q2_limit", "f_test", "q2_lookahead")
#: ``"default"`` (requests only) resolves to ``"on"`` with ``n_components="auto"``, else ``"off"``.
NOISE_GUARD = ("default", "on", "off")
EXCLUSION_METHODS = ("none", "dmodx")
DMODX_FORMULAS = ("dmodx-1", "dmodx-2")
ALL_PIXELS = "all"
DEFAULT_N_FOLDS = 7
_SEED_LIMIT = 2**63

_KEYS = {
    "folds": ("rule", "n_folds", "group_by", "groups"),
    "pixels": ("per_roi", "seed", "checksum"),
    "exclusion": ("method", "level", "rounds", "max_fraction", "formula"),
    "components": ("rule", "max", "selected", "noise_guard", "rule_choice"),
}


@dataclass(frozen=True, slots=True)
class FitSpec:
    """How a model run is fitted and cross-validated.

    Attributes
    ----------
    fold_rule : str
        ``"round_robin"`` (the fold group at position *j* is held out in fold
        *j* mod *k*, *k* = min(``n_folds``, number of groups)),
        ``"leave_one_group_out"``, or ``"group_kfold"`` (legacy runs only).
    n_folds : int
        Requested number of folds *F*, at least 2.
    group_by : str
        Which ROIs are held out together: ``"roi"`` (each ROI alone),
        ``"sample"`` (ROIs with the same sample label) or ``"target"`` (ROIs
        with the same target value).
    groups : tuple of int
        Fold-group index of every ROI of the sample set (in ``roi_ids``
        order), set by the fit; empty before the fit.
    pixels_per_roi : int or None
        Pixels drawn from every ROI; ``None`` uses all pixels.
    seed : int
        Seed of the pixel subsample, ``0 <= seed < 2**63``.
    checksum : str
        SHA-256 (hex) of the drawn pixels, set by the fit; ``""`` before the
        fit and for all pixels.
    exclusion : str
        ``"none"`` or ``"dmodx"``.
    exclusion_level, exclusion_rounds, exclusion_max_fraction, exclusion_formula
        Settings of the DModX exclusion (contract §7.2).
    components_rule : str
        How the number of components is chosen (§7.4): ``"fixed"`` (the given
        number; with ``n_components="auto"`` it means the default rule
        ``"one_se"``), ``"one_se"``, ``"min_rmsecv"`` or ``"q2_limit"``.
    components_max : int or None
        Largest number of components of the component curve; ``None`` in a
        request takes the default.
    components_selected : int
        Number of components of the run, set by the fit; 0 before the fit.
    components_noise_guard : str
        ``"on"``: after an autofit rule, drop trailing noise components and
        apply the structure ceiling (§7.4); ``"off"``: use the rule's number;
        ``"default"`` (a request) becomes ``"on"`` with ``n_components="auto"``
        and ``"off"`` with a number. A stored run without the key reads
        ``"off"``.
    components_rule_choice : int
        The rule's number before the guard and the ceiling, set by the fit.
    """

    fold_rule: str = "round_robin"
    n_folds: int = DEFAULT_N_FOLDS
    group_by: str = "roi"
    groups: tuple[int, ...] = ()
    pixels_per_roi: int | None = None
    seed: int = 0
    checksum: str = ""
    exclusion: str = "none"
    exclusion_level: float = 0.99
    exclusion_rounds: int = 1
    exclusion_max_fraction: float = 0.2
    exclusion_formula: str = "dmodx-1"
    components_rule: str = "fixed"
    components_max: int | None = None
    components_selected: int = 0
    components_noise_guard: str = "default"
    components_rule_choice: int = 0

    def __post_init__(self) -> None:
        _check_choice(self.fold_rule, "folds/rule", FOLD_RULES)
        _check_int(self.n_folds, "folds/n_folds", minimum=2)
        _check_choice(self.group_by, "folds/group_by", GROUP_BY)
        if self.fold_rule == "group_kfold" and self.group_by != "roi":
            log_and_raise("folds/rule group_kfold works only with folds/group_by roi, got {!r}",
                          self.group_by)
        groups = np.asarray(self.groups)
        if groups.ndim != 1 or (groups.size and groups.dtype.kind not in "iu"):
            log_and_raise("folds/groups must be a list of integers, got {!r}", self.groups)
        if groups.size and int(groups.min()) < 0:
            log_and_raise("folds/groups must not be negative, got {!r}", self.groups)
        object.__setattr__(self, "groups", tuple(int(g) for g in groups))
        if self.pixels_per_roi is not None:
            _check_int(self.pixels_per_roi, "pixels/per_roi", minimum=1)
        _check_int(self.seed, "pixels/seed", minimum=0)
        if int(self.seed) >= _SEED_LIMIT:
            log_and_raise("pixels/seed must be below 2**63, got {}", self.seed)
        if not isinstance(self.checksum, str):
            log_and_raise("pixels/checksum must be a string, got {!r}", self.checksum)
        _check_choice(self.exclusion, "exclusion/method", EXCLUSION_METHODS)
        level = _check_float(self.exclusion_level, "exclusion/level")
        if not 0.5 < level < 1.0:
            log_and_raise("exclusion/level must be between 0.5 and 1, got {}", level)
        if int(_check_int(self.exclusion_rounds, "exclusion/rounds", minimum=1)) not in (1, 2):
            log_and_raise("exclusion/rounds must be 1 or 2, got {}", self.exclusion_rounds)
        fraction = _check_float(self.exclusion_max_fraction, "exclusion/max_fraction")
        if not 0.0 < fraction < 1.0:
            log_and_raise("exclusion/max_fraction must be between 0 and 1, got {}", fraction)
        _check_choice(self.exclusion_formula, "exclusion/formula", DMODX_FORMULAS)
        _check_choice(self.components_rule, "components/rule", COMPONENT_RULES)
        if self.components_max is not None:
            object.__setattr__(self, "components_max",
                               _check_int(self.components_max, "components/max", minimum=1))
        object.__setattr__(self, "components_selected",
                           _check_int(self.components_selected, "components/selected",
                                      minimum=0))
        _check_choice(self.components_noise_guard, "components/noise_guard", NOISE_GUARD)
        object.__setattr__(self, "components_rule_choice",
                           _check_int(self.components_rule_choice, "components/rule_choice",
                                      minimum=0))
        # Stored values come back as NumPy scalars; keep plain Python values.
        for name, kind in (("n_folds", int), ("seed", int), ("exclusion_rounds", int),
                           ("exclusion_level", float), ("exclusion_max_fraction", float)):
            object.__setattr__(self, name, kind(getattr(self, name)))
        if self.pixels_per_roi is not None:
            object.__setattr__(self, "pixels_per_roi", int(self.pixels_per_roi))

    @classmethod
    def from_cv(cls, cv: int | str | None) -> FitSpec:
        """The specification of the short ``cv`` argument (contract §7).

        ``None`` and ``"auto"`` give round-robin folds with *F* = 7, an integer
        *F* round-robin folds with that *F*, ``"logo"`` leave one ROI out.
        """
        if cv is None or (isinstance(cv, str) and cv == "auto"):
            return cls()
        if isinstance(cv, str) and cv == "logo":
            return cls(fold_rule="leave_one_group_out")
        if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
            return cls(n_folds=int(cv))
        log_and_raise("cv must be 'auto', 'logo' or a number of folds, got {!r}", cv)

    @classmethod
    def legacy(cls, requested: Any) -> FitSpec:
        """The specification of a run stored without one (contract §7.3).

        ``requested`` is its ``cv_spec/requested`` value.
        """
        if isinstance(requested, bytes):
            requested = requested.decode()
        if requested is None or requested == "auto":
            return cls(fold_rule="group_kfold")
        if requested == "logo":
            return cls(fold_rule="leave_one_group_out")
        if isinstance(requested, (int, np.integer)) and not isinstance(requested, bool):
            return cls(fold_rule="group_kfold", n_folds=int(requested))
        log_and_raise("A legacy model has an unknown cv_spec/requested value {!r}", requested)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FitSpec:
        """Read the nested mapping of :meth:`to_dict`; left-out keys take their defaults."""
        if not isinstance(payload, Mapping):
            log_and_raise("fit_spec must be a mapping, got {}", type(payload).__name__)
        unknown = sorted(set(payload) - set(_KEYS))
        if unknown:
            log_and_raise("fit_spec has unknown part(s): {}", unknown)
        values: dict[str, Any] = {}
        for part, keys in _KEYS.items():
            group = payload.get(part, {})
            if not isinstance(group, Mapping):
                log_and_raise("fit_spec/{} must be a mapping, got {!r}", part, group)
            extra = sorted(set(group) - set(keys))
            if extra:
                log_and_raise("fit_spec/{} has unknown key(s): {}", part, extra)
            for key in keys:
                if key in group:
                    values[_FIELD[part, key]] = _decoded(group[key])
        if values.get("components_max") == 0:  # "not set" in the stored form
            values["components_max"] = None
        if "components" in payload and "components_noise_guard" not in values:
            values["components_noise_guard"] = "off"  # stored before 1.4 (contract §7.4)
        per_roi = values.get("pixels_per_roi")
        if per_roi == ALL_PIXELS:
            values["pixels_per_roi"] = None
        elif isinstance(per_roi, str):
            log_and_raise("pixels/per_roi must be 'all' or a number of pixels, got {!r}", per_roi)
        return cls(**values)

    @classmethod
    def coerce(cls, value: FitSpec | Mapping[str, Any] | None) -> FitSpec:
        """A :class:`FitSpec`, built from a mapping if needed (``None`` gives the defaults)."""
        if value is None:
            return cls()
        return value if isinstance(value, FitSpec) else cls.from_dict(value)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """The stored form: every key of contract §7, also those without effect."""
        return {
            "folds": {"rule": self.fold_rule, "n_folds": self.n_folds,
                      "group_by": self.group_by,
                      "groups": np.asarray(self.groups, dtype=np.int64)},
            "pixels": {"per_roi": ALL_PIXELS if self.pixels_per_roi is None
                       else self.pixels_per_roi,
                       "seed": self.seed, "checksum": self.checksum},
            "exclusion": {"method": self.exclusion, "level": self.exclusion_level,
                          "rounds": self.exclusion_rounds,
                          "max_fraction": self.exclusion_max_fraction,
                          "formula": self.exclusion_formula},
            "components": {"rule": self.components_rule,
                           "max": self.components_max or 0,
                           "selected": self.components_selected,
                           "noise_guard": self.components_noise_guard,
                           "rule_choice": self.components_rule_choice},
        }

    def with_checksum(self, checksum: str) -> FitSpec:
        """A copy with the checksum of the drawn pixels."""
        return replace(self, checksum=checksum)

    def with_groups(self, groups: Any) -> FitSpec:
        """A copy with the fold-group index of every ROI, as used by the fit."""
        return replace(self, groups=tuple(int(g) for g in groups))

    @property
    def sets_folds(self) -> bool:
        """``True`` if the folds differ from the default round-robin over 7 folds."""
        return ((self.fold_rule, self.n_folds, self.group_by)
                != ("round_robin", DEFAULT_N_FOLDS, "roi"))


def fold_groups_problem(groups: Any, group_by: str, n_rois: int) -> str | None:
    """Why ``groups`` are not well-formed fold groups of ``n_rois`` ROIs, or ``None`` (§7).

    Well-formed: one entry per ROI, values exactly 0 .. G - 1 with every index
    used; 0 .. R - 1 for ``group_by = "roi"``; for ``"roi"`` and ``"sample"``
    the indices first appear in increasing order.
    """
    values = [int(g) for g in groups]
    if len(values) != n_rois:
        return "it has {} entries for {} ROIs".format(len(values), n_rois)
    if sorted(set(values)) != list(range(len(set(values)))):
        return "its values are not 0 .. G - 1 with every index used"
    if group_by == "roi" and values != list(range(n_rois)):
        return "group_by roi needs the groups 0 .. R - 1"
    if group_by in ("roi", "sample"):
        first = list(dict.fromkeys(values))
        if first != sorted(first):
            return "its groups do not first appear in increasing order"
    return None


_FIELD = {
    ("folds", "rule"): "fold_rule", ("folds", "n_folds"): "n_folds",
    ("folds", "group_by"): "group_by", ("folds", "groups"): "groups",
    ("pixels", "per_roi"): "pixels_per_roi", ("pixels", "seed"): "seed",
    ("pixels", "checksum"): "checksum",
    ("exclusion", "method"): "exclusion", ("exclusion", "level"): "exclusion_level",
    ("exclusion", "rounds"): "exclusion_rounds",
    ("exclusion", "max_fraction"): "exclusion_max_fraction",
    ("exclusion", "formula"): "exclusion_formula",
    ("components", "rule"): "components_rule", ("components", "max"): "components_max",
    ("components", "selected"): "components_selected",
    ("components", "noise_guard"): "components_noise_guard",
    ("components", "rule_choice"): "components_rule_choice",
}


def _decoded(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _check_choice(value: Any, name: str, choices: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in choices:
        log_and_raise("{} must be one of {}, got {!r}", name, list(choices), value, depth=2)


def _check_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        log_and_raise("{} must be an integer, got {!r}", name, value, depth=2)
    if int(value) < minimum:
        log_and_raise("{} must be at least {}, got {}", name, minimum, value, depth=2)
    return int(value)


def _check_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        log_and_raise("{} must be a number, got {!r}", name, value, depth=2)
    return float(value)


__all__ = ["ALL_PIXELS", "COMPONENT_RULES", "DMODX_FORMULAS", "EXCLUSION_METHODS",
           "FOLD_RULES", "GROUP_BY", "NOISE_GUARD", "FitSpec", "fold_groups_problem"]
