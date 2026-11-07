"""Choosing the number of components from a component curve (contract §7.4)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from enginir._checks import check_choice, check_integer, log_and_raise
from enginir._logging import get_logger

logger = get_logger(__name__)

#: Rules of :func:`choose_components`.
COMPONENT_CHOICE_RULES = ("one_se", "min_rmsecv", "q2_limit")


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
    if rule == "one_se":
        se = np.asarray(curve["rmsecv_se"], dtype=np.float64)
        if se.shape != rmsecv.shape or not np.isfinite(se[best]) or se[best] < 0:
            log_and_raise("rmsecv_se must match rmsecv and be finite and non-negative")
        threshold = rmsecv[best] + se[best]
        return int(np.flatnonzero(rmsecv <= threshold)[0]) + 1
    q2 = np.asarray(curve["q2"], dtype=np.float64)
    if q2.shape != rmsecv.shape:
        log_and_raise("q2 must match rmsecv")
    limit = 0.05 if n_samples <= 100 else 0.0
    passing = np.nan_to_num(q2, nan=-np.inf) > limit  # a NaN Q² fails
    if not passing[0]:
        return 1
    failing = np.flatnonzero(~passing)
    return int(failing[0]) if failing.size else int(q2.size)


__all__ = ["COMPONENT_CHOICE_RULES", "choose_components"]
