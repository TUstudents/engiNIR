"""Build typed workspace objects from the dicts read by :func:`~enginir.io.load_project_store`."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, TypeVar

from enginir._checks import log_and_raise
from enginir.io.project_store import PROJECT_SCHEMA_VERSION, PROJECT_TOP_LEVEL_GROUPS
from enginir._logging import get_logger
from enginir.preprocessing import PreprocessingRecipe

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
    validate_project_meta,
)
from ._validation import validate_project_relationships

logger = get_logger(__name__)

_Model = TypeVar("_Model")
_PROJECT_META_FIELDS = ("project_id", "schema_version", "created_at", "updated_at")
_CALIBRATION_REFERENCE_FIELDS = ("dark_ref_source", "white_ref_source", "dark_ref_data",
                                 "white_ref_data", "black_level", "white_level")


@dataclass(slots=True)
class ProjectWorkspace:
    """All objects of one project, validated and keyed by stable id.

    Attributes
    ----------
    project_meta : dict
        ``project_id``, ``schema_version``, ``created_at`` and ``updated_at``.
    datasets, calibrations, rois, targets, signal_domains, sample_sets, model_runs,
    predictions : dict
        Typed objects keyed by their id.
    active_calibration_ids : dict
        ``{dataset_id: calibration_id}`` of every dataset with calibrations.
    target_assignments : dict
        ``{(roi_id, target_id): TargetAssignment}``.
    recipes : dict
        ``{recipe_id: recipe dict}`` (see :class:`~enginir.preprocessing.PreprocessingRecipe`).
    """

    project_meta: dict[str, Any]
    datasets: dict[str, DatasetRef]
    active_calibration_ids: dict[str, str]
    calibrations: dict[str, CalibrationSpec]
    rois: dict[str, TrainingROI]
    targets: dict[str, TargetDefinition]
    target_assignments: dict[tuple[str, str], TargetAssignment]
    recipes: dict[str, dict[str, Any]]
    signal_domains: dict[str, SignalDomainContract]
    sample_sets: dict[str, SampleSet]
    model_runs: dict[str, ModelRun]
    predictions: dict[str, PredictionRun]


def hydrate_project_store(
    store: dict[str, Any],
    *,
    allow_stale_sample_set_rois: bool = True,
) -> ProjectWorkspace:
    """Validate a raw project store and build the typed workspace.

    Parameters
    ----------
    store:
        Result of :func:`enginir.io.load_project_store`.
    allow_stale_sample_set_rois:
        If ``True`` (default), sample sets may reference deleted ROIs (with a
        warning), so that their model runs stay usable for prediction.

    Returns
    -------
    ProjectWorkspace

    Raises
    ------
    ValueError
        If an object misses required fields or has invalid values; the
        message names the object's location, e.g. ``/rois/roi_1``.
    ProjectRelationshipError
        If ids reference missing or foreign objects.

    Notes
    -----
    Unknown fields are ignored with a warning. Every dataset with
    calibrations must name one of them as ``active_calibration_id``.
    """
    store = _known_fields(store, PROJECT_TOP_LEVEL_GROUPS, location="/")
    project_meta = validate_project_meta(
        _known_fields(store.get("project_meta", {}), _PROJECT_META_FIELDS,
                      location="/project_meta", required=_PROJECT_META_FIELDS),
        schema_version=PROJECT_SCHEMA_VERSION,
    )

    datasets: dict[str, DatasetRef] = {}
    calibrations: dict[str, CalibrationSpec] = {}
    active_calibration_ids: dict[str, str] = {}
    for dataset_id, payload in store.get("datasets", {}).items():
        location = f"/datasets/{dataset_id}"
        datasets[dataset_id] = _build(DatasetRef, location, payload, {"dataset_id": dataset_id},
                                      ignored={"calibrations", "active_calibration_id"})
        stored_calibrations = payload.get("calibrations", {})
        for calibration_id, calibration_payload in stored_calibrations.items():
            if calibration_id in calibrations:
                log_and_raise(
                    "Calibration id '{}' of dataset '{}' is already used by dataset '{}'; "
                    "calibration ids must be unique in a project",
                    calibration_id, dataset_id, calibrations[calibration_id].dataset_id,
                )
            calibrations[calibration_id] = _build(
                CalibrationSpec, f"{location}/calibrations/{calibration_id}", calibration_payload,
                {"calibration_id": calibration_id, "dataset_id": dataset_id},
                optional=_CALIBRATION_REFERENCE_FIELDS,
            )
        if stored_calibrations or "active_calibration_id" in payload:
            active = payload.get("active_calibration_id")
            if not isinstance(active, str) or not active:
                log_and_raise("{} has calibrations but no active_calibration_id", location)
            active_calibration_ids[dataset_id] = active

    targets = {
        target_id: _build(TargetDefinition, f"/targets/{target_id}", payload,
                          {"target_id": target_id})
        for target_id, payload in store.get("targets", {}).items()
    }
    target_assignments = {
        (roi_id, target_id): _build(TargetAssignment, f"/target_assignments/{target_id}/{roi_id}",
                                    payload, {"roi_id": roi_id, "target_id": target_id})
        for target_id, assignments in store.get("target_assignments", {}).items()
        for roi_id, payload in assignments.items()
    }
    rois = {
        roi_id: _build(TrainingROI, f"/rois/{roi_id}", payload, {"roi_id": roi_id},
                       ignored={"spectra", "calibration_refs"})
        for roi_id, payload in store.get("rois", {}).items()
    }
    recipes = {
        recipe_id: PreprocessingRecipe.from_dict(payload, location=f"/recipes/{recipe_id}")
        .to_dict()
        for recipe_id, payload in store.get("recipes", {}).items()
    }
    signal_domains = _build_all(SignalDomainContract, "signal_domains", "signal_domain_id", store)
    sample_sets = _build_all(SampleSet, "sample_sets", "sample_set_id", store)
    model_runs = _build_all(ModelRun, "model_runs", "model_run_id", store, optional=("name",))
    predictions = _build_all(PredictionRun, "predictions", "prediction_run_id", store)

    validate_project_relationships(
        datasets=datasets,
        calibrations=calibrations,
        rois=rois,
        targets=targets,
        target_assignments=target_assignments.values(),
        active_calibration_ids=active_calibration_ids,
        recipes=recipes,
        signal_domains=signal_domains,
        sample_sets=sample_sets,
        model_runs=model_runs,
        predictions=predictions,
        allow_stale_sample_set_rois=allow_stale_sample_set_rois,
    )
    logger.info("Hydrated project: {} datasets, {} ROIs, {} model runs, {} predictions",
                len(datasets), len(rois), len(model_runs), len(predictions))
    return ProjectWorkspace(
        project_meta=project_meta,
        datasets=datasets,
        active_calibration_ids=active_calibration_ids,
        calibrations=calibrations,
        rois=rois,
        targets=targets,
        target_assignments=target_assignments,
        recipes=recipes,
        signal_domains=signal_domains,
        sample_sets=sample_sets,
        model_runs=model_runs,
        predictions=predictions,
    )


def _build_all(
    cls: type[_Model], collection: str, id_field: str, store: dict[str, Any],
    *, optional: tuple[str, ...] = (),
) -> dict[str, _Model]:
    return {
        object_id: _build(cls, f"/{collection}/{object_id}", payload, {id_field: object_id},
                          optional=optional)
        for object_id, payload in store.get(collection, {}).items()
    }


def _build(
    cls: type[_Model],
    location: str,
    payload: dict[str, Any],
    ids: dict[str, str],
    *,
    optional: tuple[str, ...] = (),
    ignored: set[str] | frozenset[str] = frozenset(),
    strict: bool = False,
) -> _Model:
    """Construct ``cls`` from a stored payload; the ids come from the HDF5 path.

    Every dataclass field except ``ids`` and ``optional`` is required.
    Stored values are passed unchanged so that the class validates them. A
    stored id field is accepted when it repeats the path id and rejected when
    it contradicts it.
    """
    for name, value in ids.items():
        if name in payload and payload[name] != value:
            log_and_raise("{} stores {}={!r}, but its location gives {!r}", location, name,
                          payload[name], value)
    names = [field.name for field in fields(cls) if field.name not in ids]  # type: ignore[arg-type]
    values = _known_fields(payload, names, location=location,
                           required=[name for name in names if name not in optional],
                           ignored=set(ignored) | set(ids), strict=strict)
    try:
        return cls(**ids, **{name: values.get(name) for name in names})
    except (TypeError, ValueError) as exc:
        log_and_raise("Invalid {} at {}: {}", cls.__name__, location, exc, cause=exc)


def _known_fields(
    payload: Any,
    names: tuple[str, ...] | list[str],
    *,
    location: str,
    required: tuple[str, ...] | list[str] = (),
    ignored: set[str] | frozenset[str] = frozenset(),
    strict: bool = False,
) -> dict[str, Any]:
    """Return the entries of ``payload`` named in ``names``.

    Unknown entries are warned about and ignored, or rejected if ``strict``.
    """
    if not isinstance(payload, dict):
        log_and_raise("{} must be a group of fields, got {}", location, type(payload).__name__,
                      depth=2)
    missing = [name for name in required if name not in payload]
    if missing:
        log_and_raise("{} is missing required field(s): {}", location, missing, depth=2)
    unknown = sorted(set(payload) - set(names) - set(ignored))
    if unknown and strict:
        log_and_raise("{} has unknown field(s): {}", location, unknown, depth=2)
    if unknown:
        logger.warning("Ignoring unknown field(s) at {}: {}", location, unknown)
    return {key: value for key, value in payload.items() if key in names}


_WRITABLE_MODELS: dict[str, type] = {
    "DatasetRef": DatasetRef,
    "CalibrationSpec": CalibrationSpec,
    "TargetDefinition": TargetDefinition,
    "TargetAssignment": TargetAssignment,
    "TrainingROI": TrainingROI,
    "SignalDomainContract": SignalDomainContract,
    "SampleSet": SampleSet,
    "ModelRun": ModelRun,
    "PredictionRun": PredictionRun,
}
_WRITE_OPTIONS: dict[str, dict[str, Any]] = {
    "DatasetRef": {"ignored": {"calibrations", "active_calibration_id"}},
    "CalibrationSpec": {"optional": _CALIBRATION_REFERENCE_FIELDS},
    "TrainingROI": {"ignored": {"spectra", "calibration_refs"}},
    "ModelRun": {"optional": ("name",)},
}


def validate_object_payload(label: str, location: str, payload: dict[str, Any],
                            ids: dict[str, str]) -> None:
    """Check a payload about to be written exactly as hydration will read it back.

    Used by the :mod:`enginir.io.project_store` writers so that a direct
    library write cannot store an object the project would refuse to open.
    Unlike hydration, unknown fields are rejected rather than ignored.
    """
    _build(_WRITABLE_MODELS[label], location, payload, ids, strict=True,
           **_WRITE_OPTIONS.get(label, {}))


__all__ = ["ProjectWorkspace", "hydrate_project_store", "validate_object_payload"]
