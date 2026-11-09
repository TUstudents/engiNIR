"""Choosing the number of components from a component curve (contract §7.4)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from enginir._checks import check_choice, check_integer, log_and_raise
from enginir._logging import get_logger

logger = get_logger(__name__)

#: Rules of :func:`choose_components`.
COMPONENT_CHOICE_RULES = ("one_se", "min_rmsecv", "q2_limit", "f_test", "q2_lookahead")
#: Significance level of the ``f_test`` rule (Haaland and Thomas 1988).
F_TEST_ALPHA = 0.25
#: A component explaining less of the spectra than this is a noise candidate (SIMCA N4).
NOISE_R2X = 0.01
#: Components explaining at least this much of the spectra count as structure.
STRUCTURE_R2X = 0.005


def q2_limit(n_samples: int) -> float:
    """SIMCA's R1 limit *L*: 0.05 for at most 100 fold groups, else 0."""
    return 0.05 if n_samples <= 100 else 0.0


def delta_r2x(curve: Mapping[str, Any]) -> np.ndarray:
    """ΔR²X(*a*): the share of the spectra component *a* explains (from ``r2x_cum``)."""
    cumulative = np.asarray(curve["r2x_cum"], dtype=np.float64)
    return np.diff(np.concatenate([[0.0], cumulative]))


def noise_components(curve: Mapping[str, Any], *, n_samples: int) -> np.ndarray:
    """Per component: ΔR²X < 0.01 and Q²ₐ ≤ *L* (a NaN Q² counts as ≤ *L*), contract §7.4.

    Such a component explains under 1 % of the spectra and does not improve
    the prediction (SIMCA's rule N4 combined with R1).
    """
    q2 = np.nan_to_num(np.asarray(curve["q2"], dtype=np.float64), nan=-np.inf)
    small = delta_r2x(curve) < NOISE_R2X
    if small.shape != q2.shape:
        log_and_raise("r2x_cum and q2 must have one value per component")
    return small & (q2 <= q2_limit(n_samples))


def structure_ceiling(curve: Mapping[str, Any]) -> int:
    """*C* = 1 + the number of components with ΔR²X ≥ 0.005, at most A_max (contract §7.4)."""
    gains = delta_r2x(curve)
    return int(min(gains.size, 1 + int(np.sum(gains >= STRUCTURE_R2X))))


def guarded_choice(curve: Mapping[str, Any], chosen: int, *, n_samples: int) -> int:
    """A rule's number after the structure ceiling and the noise guard (contract §7.4).

    At most :func:`structure_ceiling`; then, while the number is above 1 and
    its component is a noise component, one less.

    Examples
    --------
    >>> curve = {"r2x_cum": [0.7, 0.85, 0.9, 0.903, 0.905], "q2": [0.5, 0.3, 0.2, 0.1, -0.1]}
    >>> guarded_choice(curve, 5, n_samples=20)  # 5 is noise, 4 adds 0.3 % but predicts
    4
    """
    noise = noise_components(curve, n_samples=n_samples)
    number = check_integer(chosen, "chosen", minimum=1)
    if number > noise.size:
        log_and_raise("chosen={} exceeds the {} components of the curve", number, noise.size)
    number = min(number, structure_ceiling(curve))  # the ceiling first: never end on noise
    while number > 1 and noise[number - 1]:
        number -= 1
    return number


def choose_components(curve: Mapping[str, Any], rule: str = "one_se", *,
                      n_samples: int) -> int:
    """The number of components chosen from a component curve by ``rule``.

    Parameters
    ----------
    curve:
        Arrays over *a* = 1 … A_max: ``rmsecv`` (ROI-level RMSECV),
        ``rmsecv_se`` (its standard error) and ``q2`` (Q² per component).
    rule:
        - ``"min_rmsecv"``: the smallest *a* with the lowest RMSECV
        - ``"one_se"`` (default): the smallest *a* whose RMSECV is at most the
          lowest RMSECV plus its standard error (the simplest model that is
          not clearly worse than the best)
        - ``"q2_limit"``: the largest *a* such that Q² of every component up
          to *a* exceeds *L* = 0.05 for ``n_samples`` ≤ 100, else 0 (SIMCA's
          Autofit rule R1); 1 if the first does not; a NaN Q² fails
        - ``"f_test"`` (Haaland and Thomas 1988): the smallest *a* whose PRESS
          is below F(0.75; *G*, *G*) times the lowest PRESS, *G* = ``n_samples``
        - ``"q2_lookahead"``: as ``"q2_limit"``, but a component that fails
          is kept when the next one passes (both are taken)
    n_samples:
        Number of independent samples the curve was cross-validated over
        (the fold groups: ROIs, or samples when ROIs are grouped).

    Returns
    -------
    int
        The chosen number of components, from 1 to A_max.

    Examples
    --------
    >>> curve = {"rmsecv": [0.9, 0.5, 0.41, 0.40], "rmsecv_se": [0.1, 0.05, 0.03, 0.03],
    ...          "q2": [0.5, 0.6, 0.3, 0.02]}
    >>> choose_components(curve, "min_rmsecv", n_samples=20), choose_components(curve, n_samples=20)
    (4, 3)
    >>> choose_components(curve, "q2_limit", n_samples=20)
    3
    """
    check_choice(rule, "rule", COMPONENT_CHOICE_RULES)
    n_samples = check_integer(n_samples, "n_samples", minimum=1)
    rmsecv = np.asarray(curve["rmsecv"], dtype=np.float64)
    if rmsecv.ndim != 1 or rmsecv.size < 1 or not np.all(np.isfinite(rmsecv)):
        log_and_raise("rmsecv must be a non-empty 1-D array of finite values")
    best = int(np.argmin(rmsecv))  # the first (smallest a) of equal minima
    if rule == "min_rmsecv":
        return best + 1
    if rule == "f_test":
        from scipy.stats import f as f_distribution

        critical = float(f_distribution.ppf(1.0 - F_TEST_ALPHA, n_samples, n_samples))
        # PRESS(a) / PRESS(a*) = (RMSECV(a) / RMSECV(a*))²; a* itself always passes.
        if rmsecv[best] ** 2 < 1e-10:
            logger.warning("f_test: the lowest PRESS is (almost) zero; choosing {} components",
                           best + 1)
            return best + 1
        ratios = (rmsecv / rmsecv[best]) ** 2
        return int(np.flatnonzero((ratios < critical) | (np.arange(rmsecv.size) == best))[0]) + 1
    if rule == "one_se":
        se = np.asarray(curve["rmsecv_se"], dtype=np.float64)
        if se.shape != rmsecv.shape or not np.isfinite(se[best]) or se[best] < 0:
            log_and_raise("rmsecv_se must match rmsecv and be finite and non-negative")
        threshold = rmsecv[best] + se[best]
        return int(np.flatnonzero(rmsecv <= threshold)[0]) + 1
    q2 = np.asarray(curve["q2"], dtype=np.float64)
    if q2.shape != rmsecv.shape:
        log_and_raise("q2 must match rmsecv")
    passing = np.nan_to_num(q2, nan=-np.inf) > q2_limit(n_samples)  # a NaN Q² fails
    if rule == "q2_lookahead":
        chosen, b = 1, 0
        while b < passing.size:
            if passing[b]:
                chosen, b = b + 1, b + 1
            elif b + 1 < passing.size and passing[b + 1]:
                chosen, b = b + 2, b + 2
            else:
                break
        return chosen
    if not passing[0]:
        return 1
    failing = np.flatnonzero(~passing)
    return int(failing[0]) if failing.size else int(q2.size)


__all__ = ["COMPONENT_CHOICE_RULES", "F_TEST_ALPHA", "NOISE_R2X", "STRUCTURE_R2X",
           "choose_components", "delta_r2x", "guarded_choice", "noise_components",
           "q2_limit", "structure_ceiling"]
