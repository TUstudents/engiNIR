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

FOLD_RULES = ("round_robin", "leave_one_roi_out", "group_kfold")
EXCLUSION_METHODS = ("none", "dmodx")
DMODX_FORMULAS = ("dmodx-1",)
ALL_PIXELS = "all"
DEFAULT_N_FOLDS = 7
_SEED_LIMIT = 2**63

_KEYS = {
    "folds": ("rule", "n_folds"),
    "pixels": ("per_roi", "seed", "checksum"),
    "exclusion": ("method", "level", "rounds", "max_fraction", "formula"),
}


@dataclass(frozen=True, slots=True)
class FitSpec:
    """How a model run is fitted and cross-validated.

    Attributes
    ----------
    fold_rule : str
        ``"round_robin"`` (the ROI at position *i* of the sample set is held
        out in fold *i* mod *k*, *k* = min(``n_folds``, number of ROIs)),
        ``"leave_one_roi_out"``, or ``"group_kfold"`` (legacy runs only).
    n_folds : int
        Requested number of folds *F*, at least 2.
    pixels_per_roi : int or None
        Pixels drawn from every ROI; ``None`` uses all pixels.
    seed : int
        Seed of the pixel sample, ``0 <= seed < 2**63``.
    checksum : str
        SHA-256 (hex) of the drawn pixels, set by the fit; ``""`` before the
        fit and for all pixels.
    exclusion : str
        ``"none"`` or ``"dmodx"``.
    exclusion_level, exclusion_rounds, exclusion_max_fraction, exclusion_formula
        Settings of the DModX exclusion (contract §7.2).
    """

    fold_rule: str = "round_robin"
    n_folds: int = DEFAULT_N_FOLDS
    pixels_per_roi: int | None = None
    seed: int = 0
    checksum: str = ""
    exclusion: str = "none"
    exclusion_level: float = 0.99
    exclusion_rounds: int = 1
    exclusion_max_fraction: float = 0.2
    exclusion_formula: str = "dmodx-1"

    def __post_init__(self) -> None:
        _check_choice(self.fold_rule, "folds/rule", FOLD_RULES)
        _check_int(self.n_folds, "folds/n_folds", minimum=2)
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
            return cls(fold_rule="leave_one_roi_out")
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
            return cls(fold_rule="leave_one_roi_out")
        if isinstance(requested, (int, np.integer)) and not isinstance(requested, bool):
            return cls(fold_rule="group_kfold", n_folds=int(requested))
        log_and_raise("A legacy model run has an unknown cv_spec/requested value {!r}", requested)

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
            "folds": {"rule": self.fold_rule, "n_folds": self.n_folds},
            "pixels": {"per_roi": ALL_PIXELS if self.pixels_per_roi is None
                       else self.pixels_per_roi,
                       "seed": self.seed, "checksum": self.checksum},
            "exclusion": {"method": self.exclusion, "level": self.exclusion_level,
                          "rounds": self.exclusion_rounds,
                          "max_fraction": self.exclusion_max_fraction,
                          "formula": self.exclusion_formula},
        }

    def with_checksum(self, checksum: str) -> FitSpec:
        """A copy with the checksum of the drawn pixels."""
        return replace(self, checksum=checksum)

    @property
    def sets_folds(self) -> bool:
        """``True`` if the folds differ from the default round-robin over 7 folds."""
        return (self.fold_rule, self.n_folds) != ("round_robin", DEFAULT_N_FOLDS)


_FIELD = {
    ("folds", "rule"): "fold_rule", ("folds", "n_folds"): "n_folds",
    ("pixels", "per_roi"): "pixels_per_roi", ("pixels", "seed"): "seed",
    ("pixels", "checksum"): "checksum",
    ("exclusion", "method"): "exclusion", ("exclusion", "level"): "exclusion_level",
    ("exclusion", "rounds"): "exclusion_rounds",
    ("exclusion", "max_fraction"): "exclusion_max_fraction",
    ("exclusion", "formula"): "exclusion_formula",
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


__all__ = ["ALL_PIXELS", "DMODX_FORMULAS", "EXCLUSION_METHODS", "FOLD_RULES", "FitSpec"]
