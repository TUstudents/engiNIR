"""Pixel exclusion by DModX (``sample-set-model-run-contract_v3.md`` §7.2).

:class:`DModXExclusion` wraps a model (a PLS or PCR estimator, or a pipeline
ending in one). Its ``fit`` fits the model, computes the normalised DModX of
every training pixel and DCrit, leaves out the pixels above DCrit (at most a
share of each ROI), and refits. Cross-validation fits a clone in every fold
on that fold's training pixels only, so held-out pixels are never excluded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from .._checks import log_and_raise
from .._logging import get_logger
from ..models import PCR, PLSRegression
from ..validation.outliers import dmodx, dmodx_limit, modelling_power

logger = get_logger(__name__)


def x_residuals(model: Any, X: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """X residuals of ``X`` after the model's components, *A*, and the centred X.

    Residuals and centred X are in the scaled units of the model.

    ``model`` is a fitted :class:`~enginir.models.PLSRegression` or
    :class:`~enginir.models.PCR`, or a fitted pipeline ending in one (its
    earlier steps transform ``X`` first).
    """
    if isinstance(model, Pipeline):
        X = model[:-1].transform(X)
        model = model[-1]
    X = np.asarray(X, dtype=np.float64)
    if isinstance(model, PLSRegression):
        rotations = np.asarray(model.x_rotations_)
        scaled = (X - model.x_mean_) / model.x_std_
        scores = scaled @ rotations
        residuals = scaled - scores @ np.asarray(model.x_loadings_).T
        return residuals, int(rotations.shape[-1]), scaled
    if isinstance(model, PCR):
        centred = np.asarray(model._scaled(X), dtype=np.float64) - model.pca_.mean_
        components = np.asarray(model.pca_.components_)
        residuals = centred - (centred @ components.T) @ components
        return residuals, int(components.shape[0]), centred
    log_and_raise("DModX exclusion supports PLS and PCR models, got {}", type(model).__name__)


class DModXExclusion(BaseEstimator):
    """A model fitted with DModX pixel exclusion (contract §7.2).

    Parameters
    ----------
    estimator:
        The model to fit (unfitted; cloned).
    roi_index:
        ROI of every row of the full pixel matrix the samples are taken from
        (an integer per row). :meth:`fit` gets the rows it is trained on as
        ``sample_index``; the exclusion budget is counted per ROI.
    level:
        Probability of the DCrit limit.
    rounds:
        Exclusion rounds, 1 or 2.
    max_fraction:
        Largest share of an ROI's training pixels that may be excluded over
        all rounds (⌊max_fraction × n⌋ pixels).
    formula:
        ``"dmodx-1"`` (unweighted residuals) or ``"dmodx-2"`` (weighted by
        the modelling power of every band, SIMCA's default).

    Attributes
    ----------
    estimator_ :
        The model refitted on the kept pixels.
    excluded_ : numpy.ndarray of bool
        Which training rows were excluded (union over all rounds).
    dcrit_ : float
        DCrit of the last round.
    flagged_rois_ : numpy.ndarray of int
        ROIs (values of ``roi_index``) with more pixels above DCrit than
        their remaining budget in some round.
    """

    #: :func:`enginir.validation.cross_validate` passes the training rows.
    uses_sample_index = True

    def __init__(self, estimator: Any = None, roi_index: Any = None, level: float = 0.99,
                 rounds: int = 1, max_fraction: float = 0.2,
                 formula: str = "dmodx-1") -> None:
        self.estimator = estimator
        self.roi_index = roi_index
        self.level = level
        self.rounds = rounds
        self.max_fraction = max_fraction
        self.formula = formula

    def fit(self, X: np.ndarray, y: np.ndarray, sample_index: Any = None) -> DModXExclusion:
        """Fit, exclude the pixels above DCrit within each ROI's budget, refit (per round)."""
        rows = np.arange(X.shape[0]) if sample_index is None else np.asarray(sample_index)
        if self.roi_index is None:
            log_and_raise("DModXExclusion needs roi_index")
        if self.formula not in ("dmodx-1", "dmodx-2"):
            log_and_raise("Unknown DModX formula {!r}", self.formula)
        roi = np.asarray(self.roi_index)[rows]
        if roi.shape[0] != X.shape[0]:
            log_and_raise("sample_index has {} rows, X has {}", roi.shape[0], X.shape[0])
        rois, roi_of_row = np.unique(roi, return_inverse=True)
        budget = np.floor(self.max_fraction * np.bincount(roi_of_row)).astype(np.int64)
        kept = np.ones(X.shape[0], dtype=bool)
        flagged: set[int] = set()
        model: Any = clone(self.estimator)
        model.fit(X, y)
        dcrit = float("nan")
        for _round in range(int(self.rounds)):
            residuals, n_components, centred = x_residuals(model, X[kept])
            weights = (modelling_power(residuals, centred, n_components)
                       if self.formula == "dmodx-2" else None)
            distance, _s0 = dmodx(residuals, n_components, weights=weights)
            dcrit = dmodx_limit(int(kept.sum()), residuals.shape[1], n_components, self.level)
            kept_rows = np.flatnonzero(kept)
            above = distance > dcrit
            newly: list[np.ndarray] = []
            for r in np.unique(roi_of_row[kept_rows[above]]):
                in_roi = above & (roi_of_row[kept_rows] == r)
                candidates = kept_rows[in_roi][np.argsort(-distance[in_roi], kind="stable")]
                if candidates.size > budget[r]:
                    flagged.add(int(rois[r]))
                take = candidates[: budget[r]]
                budget[r] -= take.size
                newly.append(take)
            excluded_now = np.concatenate(newly) if newly else np.empty(0, dtype=np.int64)
            if excluded_now.size == 0:
                break  # nothing above DCrit within the budgets: the model stays
            kept[excluded_now] = False
            model = clone(self.estimator)
            model.fit(X[kept], y[kept])
        self.estimator_ = model
        self.excluded_ = ~kept
        self.dcrit_ = dcrit
        self.flagged_rois_ = np.array(sorted(flagged), dtype=np.int64)
        logger.debug("DModX exclusion: {} of {} pixels excluded, DCrit {:.4g}",
                     int((~kept).sum()), X.shape[0], dcrit)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the model refitted on the kept pixels."""
        return self.estimator_.predict(X)


__all__ = ["DModXExclusion", "x_residuals"]
