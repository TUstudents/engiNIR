"""Checks of the id references between project objects."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import Any, NoReturn

from enginir._checks import log_and_raise
from enginir._logging import get_logger

from ._models import (
    CalibrationSpec,
    DatasetRef,
    ModelRun,
    PredictionRun,
    SampleSet,
    SignalDomainContract,
    TargetAssignment,
    TargetDefinition,
    TrainingROI,
)

logger = get_logger(__name__)


class ProjectRelationshipError(ValueError):
    """An object references a missing object, or one belonging to another dataset."""


def validate_project_relationships(
    *,
    datasets: Mapping[str, DatasetRef],
    calibrations: Mapping[str, CalibrationSpec],
    rois: Mapping[str, TrainingROI],
    targets: Mapping[str, TargetDefinition],
    target_assignments: Iterable[TargetAssignment],
    active_calibration_ids: Mapping[str, str] | None = None,
    recipes: Collection[str],
    signal_domains: Mapping[str, SignalDomainContract],
    sample_sets: Mapping[str, SampleSet],
    model_runs: Mapping[str, ModelRun],
    predictions: Mapping[str, PredictionRun],
    allow_stale_sample_set_rois: bool = False,
) -> None:
    """Check that every id referenced by an object exists and fits together.

    Parameters
    ----------
    datasets, calibrations, rois, targets, signal_domains, sample_sets, model_runs, predictions:
        Objects keyed by their id. Pass only the objects to check together
        with everything they reference (e.g. one new sample set and empty
        ``model_runs``).
    target_assignments:
        Target assignments.
    active_calibration_ids:
        Optional ``{dataset_id: calibration_id}``.
    recipes:
        Ids of the stored preprocessing recipes (a mapping keyed by id works).
    allow_stale_sample_set_rois:
        If ``True``, sample sets may reference deleted ROIs (logged as a
        warning); their model runs remain usable for prediction. Default
        ``False`` (strict, for new or changed objects);
        :func:`hydrate_project_store` uses ``True`` so that projects with
        deleted ROIs still open.

    Raises
    ------
    ProjectRelationshipError
        On the first broken reference: a missing object, a calibration of
        another dataset, a dictionary key differing from the object's id, two
        assignments of one target to one ROI, or a sample set whose ROIs come
        from datasets it does not list.
    """
    for label, objects, id_field in (
        ("dataset", datasets, "dataset_id"),
        ("calibration", calibrations, "calibration_id"),
        ("roi", rois, "roi_id"),
        ("target", targets, "target_id"),
        ("signal domain", signal_domains, "signal_domain_id"),
        ("sample set", sample_sets, "sample_set_id"),
        ("model run", model_runs, "model_run_id"),
        ("prediction run", predictions, "prediction_run_id"),
    ):
        for key, item in objects.items():
            if getattr(item, id_field) != key:
                _fail("{} key '{}' differs from its {} '{}'", label, key, id_field,
                      getattr(item, id_field))

    for calibration in calibrations.values():
        _require(calibration.dataset_id, datasets, "Calibration", calibration.calibration_id,
                 "dataset_id")
    for dataset_id, calibration_id in (active_calibration_ids or {}).items():
        _require(dataset_id, datasets, "Active calibration entry", dataset_id, "dataset_id")
        _require_owned_calibration("Dataset", dataset_id, dataset_id, calibration_id,
                                   calibrations)
    for roi in rois.values():
        _require(roi.dataset_id, datasets, "ROI", roi.roi_id, "dataset_id")
        _require_owned_calibration("ROI", roi.roi_id, roi.dataset_id, roi.calibration_id,
                                   calibrations)
    assigned: set[tuple[str, str]] = set()
    for assignment in target_assignments:
        _require(assignment.roi_id, rois, "Target assignment", assignment.target_id, "roi_id")
        _require(assignment.target_id, targets, "Target assignment", assignment.roi_id,
                 "target_id")
        key = (assignment.roi_id, assignment.target_id)
        if key in assigned:
            _fail("Duplicate target assignment for roi_id '{}' and target_id '{}'", *key)
        assigned.add(key)
    for sample_set in sample_sets.values():
        _check_sample_set(sample_set, datasets=datasets, rois=rois, targets=targets,
                          recipes=recipes, allow_stale_rois=allow_stale_sample_set_rois)
    for model_run in model_runs.values():
        _require(model_run.sample_set_id, sample_sets, "ModelRun", model_run.model_run_id,
                 "sample_set_id")
        _require(model_run.signal_domain_id, signal_domains, "ModelRun", model_run.model_run_id,
                 "signal_domain_id")
    for prediction in predictions.values():
        owner = prediction.prediction_run_id
        _require(prediction.dataset_id, datasets, "PredictionRun", owner, "dataset_id")
        _require_owned_calibration("PredictionRun", owner, prediction.dataset_id,
                                   prediction.calibration_id, calibrations)
        _require(prediction.model_run_id, model_runs, "PredictionRun", owner, "model_run_id")
        _require(prediction.signal_domain_id, signal_domains, "PredictionRun", owner,
                 "signal_domain_id")
        _require(prediction.preprocessing_recipe_id, recipes, "PredictionRun", owner,
                 "preprocessing_recipe_id")


def _check_sample_set(
    sample_set: SampleSet,
    *,
    datasets: Mapping[str, DatasetRef],
    rois: Mapping[str, TrainingROI],
    targets: Mapping[str, TargetDefinition],
    recipes: Collection[str],
    allow_stale_rois: bool,
) -> None:
    owner = sample_set.sample_set_id
    _require(sample_set.target_id, targets, "SampleSet", owner, "target_id")
    _require(sample_set.preprocessing_recipe_id, recipes, "SampleSet", owner,
             "preprocessing_recipe_id")
    for dataset_id in sample_set.dataset_ids:
        _require(dataset_id, datasets, "SampleSet", owner, "dataset_id")
    deleted = [roi_id for roi_id in sample_set.roi_ids if roi_id not in rois]
    if deleted:
        if not allow_stale_rois:
            _fail("SampleSet '{}' references missing roi_id(s) {}", owner, deleted)
        logger.warning(
            "SampleSet '{}' references {} deleted ROI(s) {}; its model runs remain usable for "
            "prediction, but it cannot be changed or refitted until it is recreated",
            owner, len(deleted), deleted,
        )
    for roi_id in sample_set.roi_ids:
        if roi_id not in rois:
            continue
        if rois[roi_id].dataset_id not in sample_set.dataset_ids:
            _fail("SampleSet '{}' does not list dataset_id '{}' of ROI '{}'", owner,
                  rois[roi_id].dataset_id, roi_id)


def _require(value: str, existing: Collection[str], owner_kind: str, owner_id: str,
             field: str) -> None:
    if value not in existing:
        _fail("{} '{}' references missing {} '{}'", owner_kind, owner_id, field, value)


def _require_owned_calibration(owner_kind: str, owner_id: str, dataset_id: str,
                               calibration_id: str,
                               calibrations: Mapping[str, CalibrationSpec]) -> None:
    _require(calibration_id, calibrations, owner_kind, owner_id, "calibration_id")
    owner_dataset = calibrations[calibration_id].dataset_id
    if owner_dataset != dataset_id:
        _fail("{} '{}' uses calibration '{}' of dataset '{}' instead of dataset '{}'",
              owner_kind, owner_id, calibration_id, owner_dataset, dataset_id)


def _fail(message: str, *args: Any) -> NoReturn:
    log_and_raise(message, *args, error=ProjectRelationshipError, depth=3)


__all__ = ["ProjectRelationshipError", "validate_project_relationships"]
