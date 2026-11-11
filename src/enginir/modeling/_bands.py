"""Band proposal by forward interval PLS with nested validation (headless contract).

:func:`propose_bands` prepares the training data like a fit (signal domain,
pixel subsample, fold groups) and searches for band ranges on the recipe's axis
after Savitzky-Golay: forward selection of contiguous intervals, the number
of components of every candidate chosen by the component rule, stopped by
the one-SE idea. The whole search is repeated inside every outer fold to
estimate the error of the procedure (``nested_rmsecv``). Nothing is stored.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from .._checks import check_integer, log_and_raise
from .._logging import get_logger
from ..preprocessing import PreprocessingRecipe
from ..preprocessing._pipeline import apply_preprocessing_recipe
from ..project import (
    CalibrationSpec,
    DatasetRef,
    SampleSet,
    SignalDomainContract,
    TargetAssignment,
    TrainingROI,
)
from ..project._fit_spec import FitSpec
from ..validation import choose_components
from ._components import default_max, rank_limit
from ._service import _ESTIMATORS, _model_type, _splitter, _training_data, _with_default_scaling

logger = get_logger(__name__)

_EPS = 1e-10
_WORKING_DTYPE = np.float32


class BandProposalCancelledError(RuntimeError):
    """The band proposal was cancelled through its ``cancel`` callback."""


@dataclass(frozen=True)
class BandStep:
    """One step of the forward search: the intervals chosen so far and their RMSECV."""

    intervals: tuple[int, ...]
    rmsecv: float
    rmsecv_se: float
    n_components: int  # 0 for step 0, the mean predictor


@dataclass(frozen=True)
class BandProposal:
    """The result of :func:`propose_bands` (not stored in the project).

    Attributes
    ----------
    ranges:
        Proposed keep ranges ``(low, high)`` on the recipe's axis after
        Savitzky-Golay, ready for ``PreprocessingRecipe(bands_keep=...)``.
    steps:
        The forward search on all ROIs, step 0 (the mean predictor) first.
    intervals:
        ``(low, high)`` of every interval the axis was split into.
    nested_rmsecv:
        RMSECV of the whole procedure, the search repeated inside every outer
        fold (selection included).
    all_bands_rmsecv:
        RMSECV of the recipe with every band (same folds, rule, no exclusion).
    n_components:
        Components chosen at the last step (for a fit with ``ranges``).
    """

    ranges: tuple[tuple[float, float], ...]
    steps: tuple[BandStep, ...]
    intervals: tuple[tuple[float, float], ...]
    nested_rmsecv: float
    all_bands_rmsecv: float
    n_components: int


# -- the search on one set of ROIs ---------------------------------------------


@dataclass
class _Folds:
    """Preprocessed training and test matrices of every fold, plus the whole set."""

    train: list[np.ndarray]          # row indices into the set
    test: list[np.ndarray]
    X_train: list[np.ndarray]        # preprocessed, all bands after Savitzky-Golay
    X_test: list[np.ndarray]
    X_all: np.ndarray                # preprocessed on the whole set (for Q²)


def _split(splitter: Any, X: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, ...]]:
    folds = LeaveOneGroupOut() if isinstance(splitter, str) else splitter
    return [(np.asarray(train), np.asarray(test)) for train, test in
            folds.split(X, None, groups)]


def _preprocessed_folds(X: np.ndarray, splits: list[tuple[np.ndarray, ...]],
                        recipe: PreprocessingRecipe, wavelengths: np.ndarray,
                        signal_unit: str | None) -> _Folds:
    """Stateless recipes are applied once; MSC/EMSC are fitted per training part."""
    if not recipe.requires_fitting:
        X_all = apply_preprocessing_recipe(X, recipe=recipe, wavelengths=wavelengths,
                                           signal_unit=signal_unit)
        return _Folds([s[0] for s in splits], [s[1] for s in splits],
                      [X_all[s[0]] for s in splits], [X_all[s[1]] for s in splits], X_all)
    X_train, X_test = [], []
    for train, test in splits:
        steps = recipe.build_steps(wavelengths, signal_unit=signal_unit)
        block = np.asarray(X[train], dtype=_WORKING_DTYPE)
        held = np.asarray(X[test], dtype=_WORKING_DTYPE)
        for _name, step in steps:
            transformer: Any = step
            block = np.asarray(transformer.fit_transform(block), dtype=_WORKING_DTYPE)
            held = np.asarray(transformer.transform(held), dtype=_WORKING_DTYPE)
        X_train.append(block)
        X_test.append(held)
    X_all = apply_preprocessing_recipe(X, recipe=recipe, wavelengths=wavelengths,
                                       signal_unit=signal_unit)
    return _Folds([s[0] for s in splits], [s[1] for s in splits], X_train, X_test, X_all)


def _roi_means(values: np.ndarray, roi_index: np.ndarray, n_rois: int) -> np.ndarray:
    counts = np.bincount(roi_index, minlength=n_rois).astype(np.float64)
    if values.ndim == 1:
        return np.bincount(roi_index, weights=values, minlength=n_rois) / counts
    return np.stack([np.bincount(roi_index, weights=values[:, c], minlength=n_rois) / counts
                     for c in range(values.shape[1])], axis=1)


def _curve_values(oof: np.ndarray, y: np.ndarray, roi_index: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROI-level (RMSECV, SE, PRESS) over the columns of ``oof`` (contract §7.4)."""
    n_rois = int(roi_index.max()) + 1
    y_roi = _roi_means(y, roi_index, n_rois)
    errors = _roi_means(oof, roi_index, n_rois) - y_roi[:, np.newaxis]
    press = (errors ** 2).sum(axis=0)
    rmsecv = np.sqrt(press / n_rois)
    spread = np.std(errors ** 2, axis=0, ddof=1) if n_rois > 1 else np.zeros_like(press)
    se = np.zeros_like(rmsecv)
    ok = rmsecv >= _EPS
    se[ok] = spread[ok] / (2.0 * rmsecv[ok] * np.sqrt(n_rois))
    return rmsecv, se, press


@dataclass
class _Search:
    """The search problem on one set of ROIs (all, or an outer training part)."""

    folds: _Folds
    y: np.ndarray
    roi_index: np.ndarray            # ROI position of every row, 0 … R − 1
    n_groups: int
    model_type: str
    params: dict[str, Any]           # estimator parameters without n_components
    rule: str
    a_max: int | None                # fit_spec components max, or None
    fixed: int | None                # a given number of components (rule "fixed")
    tick: Callable[[], None]

    def baseline(self) -> BandStep:
        oof = np.empty(self.y.size)
        for train, test in zip(self.folds.train, self.folds.test, strict=True):
            oof[test] = self.y[train].mean()
        rmsecv, se, _press = _curve_values(oof[:, None], self.y, self.roi_index)
        return BandStep((), float(rmsecv[0]), float(se[0]), 0)

    def evaluate(self, columns: np.ndarray) -> tuple[float, float, int]:
        """RMSECV, SE and number of components of a candidate (its band columns)."""
        self.tick()
        sizes = [t.size for t in self.folds.train] + [self.y.size]
        limit = rank_limit(sizes, columns.size)
        if self.fixed is not None:
            a_max = min(self.fixed, limit)
        else:
            a_max = min(self.a_max or default_max(0, self.n_groups, limit), limit)
        estimator = _ESTIMATORS[self.model_type]
        oof = np.empty((self.y.size, a_max))
        for train, test, X_train, X_test in zip(self.folds.train, self.folds.test,
                                                self.folds.X_train, self.folds.X_test,
                                                strict=True):
            model = estimator(n_components=a_max, **self.params).fit(X_train[:, columns],
                                                                     self.y[train])
            oof[test] = model.predict_components(X_test[:, columns])
        rmsecv, se, press = _curve_values(oof, self.y, self.roi_index)
        if self.fixed is not None:
            return float(rmsecv[-1]), float(se[-1]), a_max
        full = estimator(n_components=a_max, **self.params).fit(self.folds.X_all[:, columns],
                                                                self.y)
        fitted = full.predict_components(self.folds.X_all[:, columns])
        n_rois = int(self.roi_index.max()) + 1
        y_roi = _roi_means(self.y, self.roi_index, n_rois)
        rss_roi = ((_roi_means(fitted, self.roi_index, n_rois) - y_roi[:, None]) ** 2).sum(axis=0)
        ss = float(((y_roi - y_roi.mean()) ** 2).sum())
        previous = np.concatenate([[ss], rss_roi[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            q2 = np.where(previous > _EPS, 1.0 - press / previous, np.nan)
        chosen = choose_components({"rmsecv": rmsecv, "rmsecv_se": se, "q2": q2}, self.rule,
                                   n_samples=self.n_groups)
        return float(rmsecv[chosen - 1]), float(se[chosen - 1]), chosen


def _forward(search: _Search, intervals: list[np.ndarray], max_steps: int
             ) -> list[BandStep]:
    """Forward selection of intervals, stopped by the one-SE idea (headless contract)."""
    steps = [search.baseline()]
    chosen: list[int] = []
    while len(chosen) < max_steps and len(chosen) < len(intervals):
        best: tuple[float, float, int, int] | None = None
        for index in range(len(intervals)):
            if index in chosen:
                continue
            columns = np.sort(np.concatenate([intervals[i] for i in [*chosen, index]]))
            rmsecv, se, a = search.evaluate(columns)
            if best is None or rmsecv < best[0]:  # ties keep the lower index
                best = (rmsecv, se, a, index)
        if best is None:
            break
        current = steps[-1]
        if best[0] >= current.rmsecv - current.rmsecv_se:
            break
        chosen.append(best[3])
        steps.append(BandStep(tuple(sorted(chosen)), best[0], best[1], best[2]))
    return steps


def _ranges(chosen: tuple[int, ...], intervals: list[np.ndarray], axis: np.ndarray
            ) -> tuple[tuple[float, float], ...]:
    """Merge adjacent chosen intervals into ``[first band, last band]`` ranges."""
    ranges: list[tuple[float, float]] = []
    run: list[int] = []
    for index in sorted(chosen):
        if run and index != run[-1] + 1:
            ranges.append((float(axis[intervals[run[0]][0]]), float(axis[intervals[run[-1]][-1]])))
            run = []
        run.append(index)
    if run:
        ranges.append((float(axis[intervals[run[0]][0]]), float(axis[intervals[run[-1]][-1]])))
    return tuple(ranges)


# -- the public function ----------------------------------------------------------


def propose_bands(
    *,
    project_path: str,
    sample_set: SampleSet,
    datasets: dict[str, DatasetRef],
    calibrations: dict[str, CalibrationSpec],
    rois: dict[str, TrainingROI],
    target_assignments: dict[tuple[str, str], TargetAssignment],
    recipes: dict[str, dict[str, Any]],
    signal_domains: dict[str, SignalDomainContract],
    signal_domain_id: str,
    model_type: str = "PLS",
    hyperparameters: dict[str, Any] | None = None,
    n_intervals: int = 20,
    max_steps: int | None = None,
    fit_spec: FitSpec | Mapping[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> BandProposal:
    """Propose band ranges by forward interval PLS with nested validation.

    The search uses the sample set's recipe without its band selection, the
    fit specification's folds, fold groups, pixel subsample and component rule
    (``headless-workflow-contract_v3.md``, Band proposal). ``progress(done,
    total)`` counts candidate evaluations (``total`` is an upper bound);
    ``cancel()`` returning ``True`` raises :class:`BandProposalCancelledError`.

    Raises
    ------
    ValueError
        If there are fewer than 3 fold groups, ``n_intervals`` is not between
        2 and the band count, or the inputs are inconsistent.
    """
    model_type = _model_type(model_type)
    prepared = _training_data(
        project_path=project_path, sample_set=sample_set, datasets=datasets,
        calibrations=calibrations, rois=rois, target_assignments=target_assignments,
        recipes=recipes, signal_domains=signal_domains, signal_domain_id=signal_domain_id,
        cv=None, fit_spec=fit_spec)
    spec = prepared.spec
    if prepared.n_fold_groups < 3:
        log_and_raise("Proposing bands needs at least 3 fold groups (nested validation), got {}",
                      prepared.n_fold_groups)
    stored = PreprocessingRecipe.from_dict(recipes[sample_set.preprocessing_recipe_id])
    recipe = replace(stored, bands_keep=(), bands_exclude=())
    wavelengths = np.asarray(prepared.data.wavelengths, dtype=np.float64)
    axis = recipe.output_wavelengths(wavelengths)
    n_intervals = check_integer(n_intervals, "n_intervals", minimum=2)
    if n_intervals > axis.size:
        log_and_raise("n_intervals={} exceeds the {} bands after Savitzky-Golay", n_intervals,
                      axis.size)
    max_steps = n_intervals if max_steps is None else check_integer(max_steps, "max_steps",
                                                                    minimum=1)
    intervals = np.array_split(np.arange(axis.size), n_intervals)
    params = _with_default_scaling(model_type, dict(hyperparameters or {}))
    requested = params.pop("n_components", "auto")
    fixed = None
    if spec.components_rule == "fixed" and not isinstance(requested, str):
        fixed = check_integer(requested, "n_components", minimum=1)
    rule = "one_se" if spec.components_rule == "fixed" else spec.components_rule
    signal_unit = prepared.signal_domain.required_signal_unit
    X, y = prepared.X_signal, np.asarray(prepared.data.y, dtype=np.float64)
    groups, roi_index = prepared.cv_groups, prepared.roi_index
    outer = _split(prepared.splitter, X, groups)
    per_search = sum(n_intervals - s for s in range(min(max_steps, n_intervals)))
    total = per_search * (len(outer) + 1) + 1
    done = [0]
    capped = [False]

    def tick() -> None:
        if cancel is not None and cancel():
            raise BandProposalCancelledError("Band proposal canceled")
        done[0] += 1
        if progress is not None:
            progress(done[0], total)

    def search_on(rows: np.ndarray) -> _Search:
        sub_groups = groups[rows]
        n_groups = int(np.unique(sub_groups).size)
        _positions, sub_roi = np.unique(roi_index[rows], return_inverse=True)
        folds = _preprocessed_folds(
            X[rows], _split(_splitter(spec, n_groups), X[rows], sub_groups), recipe,
            wavelengths, signal_unit)
        if fixed is not None and not capped[0] and fixed > axis.size:
            capped[0] = True
            logger.warning("n_components={} is capped at the band count of each candidate",
                           fixed)
        return _Search(folds, y[rows], sub_roi, n_groups, model_type, params, rule,
                       spec.components_max, fixed, tick)

    everything = search_on(np.arange(y.size))
    all_rmsecv, _se, _a = everything.evaluate(np.arange(axis.size))
    steps = _forward(everything, intervals, max_steps)

    # Nested: the whole search inside every outer fold, judged on its held-out ROIs.
    held_out = np.empty(y.size)
    estimator = _ESTIMATORS[model_type]
    for train, test in outer:
        inner = _forward(search_on(train), intervals, max_steps)
        last = inner[-1]
        if not last.intervals:  # no interval beat the mean: predict the training mean
            held_out[test] = y[train].mean()
            continue
        columns = np.sort(np.concatenate([intervals[i] for i in last.intervals]))
        fold = _preprocessed_folds(X, [(train, test)], recipe, wavelengths,
                                   signal_unit) if recipe.requires_fitting else None
        if fold is not None:
            X_train, X_test = fold.X_train[0], fold.X_test[0]
        else:
            X_all = everything.folds.X_all
            X_train, X_test = X_all[train], X_all[test]
        model = estimator(n_components=last.n_components, **params).fit(X_train[:, columns],
                                                                        y[train])
        held_out[test] = np.asarray(model.predict(X_test[:, columns])).reshape(-1)
    nested, _nested_se, _press = _curve_values(held_out[:, None], y, roi_index)

    proposal = BandProposal(
        ranges=_ranges(steps[-1].intervals, intervals, axis),
        steps=tuple(steps),
        intervals=tuple((float(axis[i[0]]), float(axis[i[-1]])) for i in intervals),
        nested_rmsecv=float(nested[0]),
        all_bands_rmsecv=float(all_rmsecv),
        n_components=steps[-1].n_components,
    )
    logger.info("Band proposal for calibration set '{}': ranges {}, RMSECV (per ROI) {:.4g} after {} step(s), "
                "nested RMSECV (per ROI) {:.4g}, all bands {:.4g}", sample_set.sample_set_id,
                list(proposal.ranges), steps[-1].rmsecv, len(steps) - 1,
                proposal.nested_rmsecv, proposal.all_bands_rmsecv)
    return proposal


__all__ = ["BandProposal", "BandProposalCancelledError", "BandStep", "propose_bands"]
