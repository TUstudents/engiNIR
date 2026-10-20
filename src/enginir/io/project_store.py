"""Reading and writing engiNIR project files (schema ``viewer-v3``).

A project file is an HDF5 file holding the scientific workspace: datasets and
their calibrations, ROIs with their spectra, targets and target assignments,
preprocessing recipes, signal-domain contracts, sample sets, model runs and
prediction runs. Every object is stored as one group under a top-level
collection, keyed by its stable id:

====================================================  =========================
Group                                                 Object
====================================================  =========================
``/project_meta`` (attributes)                        project id, schema version
``/datasets/{dataset_id}``                            dataset reference
``/datasets/{dataset_id}/calibrations/{id}``          calibration
``/targets/{target_id}``                              target definition
``/target_assignments/{target_id}/{roi_id}``          target value of one ROI
``/recipes/{recipe_id}``                              preprocessing recipe
``/signal_domains/{signal_domain_id}``                signal-domain contract
``/rois/{roi_id}``                                    ROI metadata and spectra
``/sample_sets/{sample_set_id}``                      sample set
``/model_runs/{model_run_id}``                        model run and artifacts
``/predictions/{prediction_run_id}``                  prediction run and maps
====================================================  =========================

Stable ids are immutable: saving an existing id with different content is
refused (only display names may change). Loading is metadata-first; ROI spectra
and large model/prediction arrays are read on request. The normative schema is
``contracts/project-schema-contract_v3.md``; ``docs/hdf5_format.md`` describes
the layout.

The functions here work on plain dicts; :mod:`enginir.project` converts them to
typed objects.
"""
# pyright: reportIndexIssue=false, reportAttributeAccessIssue=false
# pyright: reportArgumentType=false, reportOperatorIssue=false
# h5py's stubs do not describe Group/Dataset item access.

from __future__ import annotations

import datetime as dt
import os
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from enginir._checks import log_and_raise
from enginir._logging import get_logger

from ._hdf5_groups import (
    MAX_ATTRIBUTE_VECTOR,
    child_group,
    decode,
    is_scalar_like,
    read_mapping,
    replacement_group,
    values_equal,
    write_mapping,
)

logger = get_logger(__name__)

PROJECT_SCHEMA_VERSION = "viewer-v3"
PROJECT_TOP_LEVEL_GROUPS = (
    "project_meta",
    "datasets",
    "targets",
    "target_assignments",
    "recipes",
    "signal_domains",
    "rois",
    "sample_sets",
    "model_runs",
    "predictions",
)
_PREDICTION_RUN_FIELDS = frozenset({
    "dataset_id",
    "calibration_id",
    "model_run_id",
    "signal_domain_id",
    "preprocessing_recipe_id",
    "created_at",
    "artifact_refs",
})
_PREDICTION_ARRAYS = ("y_pred", "spatial_map")
_PREVIEW_BLOCK_VALUES = 2**22  # values read per block for a map preview

PathLike = str | os.PathLike[str]


# ---------------------------------------------------------------------------
# Project files
# ---------------------------------------------------------------------------


def is_project_file(handle: h5py.File) -> bool:
    """Return ``True`` if the open HDF5 file is an engiNIR v3 project."""
    project_meta = handle.get("project_meta")
    return (
        isinstance(project_meta, h5py.Group)
        and decode(project_meta.attrs.get("schema_version", "")) == PROJECT_SCHEMA_VERSION
    )


def create_project(path: PathLike, *, project_id: str | None = None) -> None:
    """Create an empty project file (overwriting ``path``).

    Parameters
    ----------
    path:
        File to create.
    project_id:
        Stable project id; a random UUID by default.
    """
    now = _now_iso()
    with h5py.File(path, "w") as f:
        project_meta = f.create_group("project_meta")
        project_meta.attrs.update(
            project_id=project_id or str(uuid.uuid4()),
            schema_version=PROJECT_SCHEMA_VERSION,
            created_at=now,
            updated_at=now,
        )
        for name in PROJECT_TOP_LEVEL_GROUPS[1:]:
            f.create_group(name)
    logger.info("Created project {}", path)


def load_project_store(
    path: PathLike,
    *,
    include_roi_spectra: bool = False,
    include_artifact_arrays: bool = False,
) -> dict[str, Any]:
    """Read all objects of a project into nested dicts keyed by stable id.

    This is the raw reader: it checks the file structure (schema version,
    top-level groups, interrupted writes) and decodes values, but does not
    validate field values, types or references. Pass the result to
    :func:`enginir.project.hydrate_project_store` for the fully validated,
    typed workspace; only that step guarantees a contract-conforming project.

    Parameters
    ----------
    path:
        Project file.
    include_roi_spectra:
        Also read ROI spectra and calibration reference arrays (large).
        Default ``False``: they stay on disk, see :func:`load_roi_spectra`.
    include_artifact_arrays:
        Also read the arrays under ``artifact_refs`` of model and prediction
        runs (cross-validation outputs, prediction maps). Default ``False``:
        only their scalar attributes are read.

    Returns
    -------
    dict
        ``{"project_meta": {...}, "datasets": {id: {...}}, ...}`` with one
        entry per top-level group; target assignments are nested as
        ``{target_id: {roi_id: {...}}}``.

    Raises
    ------
    ValueError
        If the file is not a valid v3 project or holds an interrupted
        prediction write.
    """
    roi_options: dict[str, Any] = {}
    if not include_roi_spectra:
        roi_options = {"skip_datasets": ("spectra",), "metadata_only_groups": ("calibration_refs",)}
    artifact_options: dict[str, Any] = {}
    if not include_artifact_arrays:
        artifact_options = {"metadata_only_groups": ("artifact_refs",)}

    with _open_project(path) as f:
        def read_all(collection: str, **options: Any) -> dict[str, dict[str, Any]]:
            return {key: read_mapping(group, **options) for key, group in f[collection].items()}

        for roi_group in f["rois"].values():
            _check_roi_arrays(roi_group)
        for prediction_group in f["predictions"].values():
            _check_prediction_complete(prediction_group)
        predictions = read_all("predictions", **artifact_options)
        if not include_artifact_arrays:
            predictions = {
                key: {field: value for field, value in run.items()
                      if field in _PREDICTION_RUN_FIELDS}
                for key, run in predictions.items()
            }
        for key, group in f["predictions"].items():
            if isinstance(group.get("artifact_refs"), h5py.Group):
                predictions[key].setdefault("artifact_refs", {})
        return {
            "project_meta": {key: decode(value) for key, value in f["project_meta"].attrs.items()},
            "datasets": read_all("datasets"),
            "targets": read_all("targets"),
            "target_assignments": {
                target_id: {roi_id: read_mapping(group) for roi_id, group in assignments.items()}
                for target_id, assignments in f["target_assignments"].items()
            },
            "recipes": read_all("recipes"),
            "signal_domains": read_all("signal_domains"),
            "rois": read_all("rois", **roi_options),
            "sample_sets": read_all("sample_sets"),
            "model_runs": read_all("model_runs", **artifact_options),
            "predictions": predictions,
        }


# ---------------------------------------------------------------------------
# Saving objects
# ---------------------------------------------------------------------------


def save_dataset_ref(
    path: PathLike, dataset_id: str, dataset_dict: dict[str, Any], compression: str = "gzip"
) -> None:
    """Save a dataset reference to ``/datasets/{dataset_id}``."""
    _save_object(path, "datasets", dataset_id, dataset_dict, "DatasetRef", compression,
                 child_members=frozenset({"calibrations", "active_calibration_id"}))


def save_calibration_spec(
    path: PathLike,
    dataset_id: str,
    calibration_id: str,
    calibration_dict: dict[str, Any],
    compression: str = "gzip",
) -> None:
    """Save a calibration of a dataset and make it the dataset's active calibration.

    Written to ``/datasets/{dataset_id}/calibrations/{calibration_id}``; the
    dataset's ``active_calibration_id`` attribute is set to ``calibration_id``.
    """
    _validate_payload("CalibrationSpec", f"/datasets/{dataset_id}/calibrations/{calibration_id}",
                      calibration_dict, {"calibration_id": calibration_id,
                                         "dataset_id": dataset_id})
    with _open_project(path, writable=True) as f:
        dataset_group = child_group(f["datasets"], dataset_id)
        calibrations = child_group(dataset_group, "calibrations", create=True)
        with replacement_group(calibrations, calibration_id) as group:
            write_mapping(group, calibration_dict, compression)
        dataset_group.attrs["active_calibration_id"] = calibration_id
    logger.info("Saved calibration '{}' of dataset '{}' to {}", calibration_id, dataset_id, path)


def save_target_definition(
    path: PathLike, target_id: str, target_dict: dict[str, Any], compression: str = "gzip"
) -> None:
    """Save a target definition to ``/targets/{target_id}`` (its ``name`` may change)."""
    _save_object(path, "targets", target_id, target_dict, "TargetDefinition", compression,
                 mutable_fields=frozenset({"name"}))


def save_target_assignment(
    path: PathLike,
    target_id: str,
    roi_id: str,
    assignment_dict: dict[str, Any],
    compression: str = "gzip",
) -> None:
    """Save (or replace) the value of one target for one ROI.

    Written to ``/target_assignments/{target_id}/{roi_id}``.
    """
    _validate_payload("TargetAssignment", f"/target_assignments/{target_id}/{roi_id}",
                      assignment_dict, {"target_id": target_id, "roi_id": roi_id})
    with _open_project(path, writable=True) as f:
        target_group = child_group(f["target_assignments"], target_id, create=True)
        with replacement_group(target_group, roi_id) as group:
            write_mapping(group, assignment_dict, compression)
    logger.info("Saved target assignment target='{}' roi='{}' to {}", target_id, roi_id, path)


def save_preprocessing_recipe(
    path: PathLike, recipe_id: str, recipe_dict: dict[str, Any], compression: str = "gzip"
) -> None:
    """Save a preprocessing recipe to ``/recipes/{recipe_id}`` after validating it.

    Saving the same recipe again under its id is a no-op.
    """
    from enginir.preprocessing import PreprocessingRecipe

    recipe_id = str(recipe_id).strip()
    if not recipe_id:
        log_and_raise("recipe_id must be a non-empty string")
    payload = PreprocessingRecipe.from_dict(recipe_dict, location=f"/recipes/{recipe_id}").to_dict()
    _save_object(path, "recipes", recipe_id, payload, "PreprocessingRecipe", compression)


def save_signal_domain_contract(
    path: PathLike,
    signal_domain_id: str,
    signal_domain_dict: dict[str, Any],
    compression: str = "gzip",
) -> None:
    """Save a signal-domain contract to ``/signal_domains/{signal_domain_id}``."""
    _save_object(path, "signal_domains", signal_domain_id, signal_domain_dict,
                 "SignalDomainContract", compression)


def save_training_roi(
    path: PathLike,
    roi_id: str,
    roi_dict: dict[str, Any],
    spectra: np.ndarray,
    *,
    dark_ref_spectra: np.ndarray | None = None,
    white_ref_spectra: np.ndarray | None = None,
    compression: str = "gzip",
) -> None:
    """Save an ROI with its spectra to ``/rois/{roi_id}`` (its ``name`` may change).

    Parameters
    ----------
    path:
        Project file.
    roi_id:
        Stable ROI id.
    roi_dict:
        ROI metadata; ``n_pixels`` and ``wavelengths``, if present, must match
        ``spectra``.
    spectra:
        ROI pixel spectra, shape ``(n_pixels, n_bands)``. Floating spectra
        are stored as float32, everything else as uint16 (raw counts).
    dark_ref_spectra, white_ref_spectra:
        Optional per-pixel dark and white references (both or neither), same
        shape as ``spectra``, stored under ``calibration_refs``.
    compression:
        HDF5 compression filter.
    """
    spectra = as_roi_storage_array(spectra)
    if spectra.ndim != 2:
        log_and_raise("ROI spectra must be 2-D (n_pixels, n_bands), got shape {}", spectra.shape)
    n_pixels, n_bands = spectra.shape
    declared_pixels = roi_dict.get("n_pixels")
    if declared_pixels is not None and int(declared_pixels) != n_pixels:
        log_and_raise("ROI n_pixels={} does not match {} spectra rows", declared_pixels, n_pixels)
    declared_wavelengths = roi_dict.get("wavelengths")
    if declared_wavelengths is not None and np.shape(declared_wavelengths) != (n_bands,):
        log_and_raise("ROI wavelengths shape {} does not match {} bands",
                      np.shape(declared_wavelengths), n_bands)
    payload = {**roi_dict, "spectra": spectra}
    if (dark_ref_spectra is None) != (white_ref_spectra is None):
        log_and_raise("ROI calibration references need both dark_ref_spectra and white_ref_spectra")
    if dark_ref_spectra is not None and white_ref_spectra is not None:
        dark = np.asarray(dark_ref_spectra, dtype=np.float32)
        white = np.asarray(white_ref_spectra, dtype=np.float32)
        if dark.shape != spectra.shape or white.shape != spectra.shape:
            log_and_raise("ROI calibration references must have shape {}, got dark={} white={}",
                          spectra.shape, dark.shape, white.shape)
        payload["calibration_refs"] = {"dark": dark, "white": white}
    _save_object(path, "rois", roi_id, payload, "TrainingROI", compression,
                 mutable_fields=frozenset({"name"}))


def save_sample_set(
    path: PathLike, sample_set_id: str, sample_set_dict: dict[str, Any], compression: str = "gzip"
) -> None:
    """Save a sample set to ``/sample_sets/{sample_set_id}``."""
    _save_object(path, "sample_sets", sample_set_id, sample_set_dict, "SampleSet", compression)


def save_model_run(
    path: PathLike, model_run_id: str, model_run_dict: dict[str, Any], compression: str = "gzip"
) -> None:
    """Save a model run, including its ``artifact_refs``, to ``/model_runs/{model_run_id}``.

    Its ``name`` may change.
    """
    _save_object(path, "model_runs", model_run_id, model_run_dict, "ModelRun", compression,
                 mutable_fields=frozenset({"name"}))


def save_prediction_run(
    path: PathLike,
    prediction_run_id: str,
    prediction_run_dict: dict[str, Any],
    compression: str = "gzip",
) -> None:
    """Save a complete prediction run to ``/predictions/{prediction_run_id}``.

    Large predictions are better written incrementally with
    :func:`initialize_prediction_run_artifacts`.
    """
    _save_object(path, "predictions", prediction_run_id, prediction_run_dict, "PredictionRun",
                 compression)


def remove_roi_from_project(path: PathLike, roi_id: str) -> None:
    """Delete an ROI and all target values assigned to it."""
    with _open_project(path, writable=True) as f:
        rois = f["rois"]
        if roi_id not in rois:
            log_and_raise("ROI '{}' not found in project {}", roi_id, path, error=KeyError)
        del rois[roi_id]
        for assignments in f["target_assignments"].values():
            if roi_id in assignments:
                del assignments[roi_id]
    logger.info("Removed ROI '{}' from {}", roi_id, path)


# ---------------------------------------------------------------------------
# Incremental prediction artifacts
# ---------------------------------------------------------------------------


def initialize_prediction_run_artifacts(
    path: PathLike,
    prediction_run_id: str,
    prediction_run_dict: dict[str, Any],
    *,
    y_pred_shape: tuple[int, ...],
    spatial_map_shape: tuple[int, ...] | None,
    compression: str = "gzip",
) -> None:
    """Create a prediction run whose ``y_pred`` (and ``spatial_map``) are filled later.

    The run is marked ``artifact_write_status = "writing"`` until
    :func:`finalize_prediction_run_artifacts`; a project containing such a run
    does not load, so an interrupted prediction must be removed with
    :func:`discard_prediction_run_artifacts`. An existing run with the same id
    is never replaced.

    Parameters
    ----------
    path:
        Project file.
    prediction_run_id:
        Stable id of the new run.
    prediction_run_dict:
        Run metadata; array entries in its ``artifact_refs`` are ignored.
    y_pred_shape:
        Shape of the per-pixel predictions, ``(n_pixels,)`` or
        ``(n_pixels, n_targets)``.
    spatial_map_shape:
        Shape of the prediction image, e.g. ``(n_frames, height)``, or
        ``None`` for non-image data.
    compression:
        HDF5 compression filter.
    """
    artifact_refs = {key: value for key, value in prediction_run_dict.get("artifact_refs", {}).items()
                     if key not in _PREDICTION_ARRAYS}
    artifact_refs["artifact_write_status"] = "writing"
    payload = {**prediction_run_dict, "artifact_refs": artifact_refs}
    shapes = {"y_pred": y_pred_shape, "spatial_map": spatial_map_shape}
    with _open_project(path, writable=True) as f:
        if prediction_run_id in f["predictions"]:
            log_and_raise("Prediction run '{}' already exists in {}; use a new id",
                          prediction_run_id, path)
        with replacement_group(f["predictions"], prediction_run_id) as group:
            write_mapping(group, payload, compression)
            artifact_group = child_group(group, "artifact_refs", create=True)
            for name, shape in shapes.items():
                if shape is None:
                    continue
                shape = tuple(int(dim) for dim in shape)
                artifact_group.create_dataset(name, shape=shape, dtype=np.float32,
                                              compression=compression, chunks=_chunk_shape(shape))
    logger.info("Initialized prediction artifacts '{}' in {}", prediction_run_id, path)


@dataclass(slots=True)
class PredictionArtifactWriter:
    """Writes prediction chunks into the datasets created by
    :func:`initialize_prediction_run_artifacts`; see
    :func:`open_prediction_run_artifact_writer`."""

    prediction_run_id: str
    y_dataset: h5py.Dataset
    spatial_dataset: h5py.Dataset | None

    def write_chunk(
        self,
        *,
        row_start: int,
        y_pred_chunk: np.ndarray,
        frame_start: int | None = None,
        spatial_map_chunk: np.ndarray | None = None,
    ) -> None:
        """Store predictions for rows ``row_start:`` and, optionally, frames ``frame_start:``."""
        y_chunk = np.asarray(y_pred_chunk, dtype=np.float32)
        self.y_dataset[row_start : row_start + y_chunk.shape[0], ...] = y_chunk
        if spatial_map_chunk is None:
            return
        if self.spatial_dataset is None or frame_start is None:
            log_and_raise(
                "PredictionRun '{}': a spatial-map chunk needs a spatial_map dataset and "
                "frame_start",
                self.prediction_run_id,
            )
        spatial_chunk = np.asarray(spatial_map_chunk, dtype=np.float32)
        self.spatial_dataset[frame_start : frame_start + spatial_chunk.shape[0], ...] = (
            spatial_chunk
        )


@contextmanager
def open_prediction_run_artifact_writer(
    path: PathLike, prediction_run_id: str
) -> Iterator[PredictionArtifactWriter]:
    """Keep the project open while all chunks of a prediction are written.

    Yields
    ------
    PredictionArtifactWriter
    """
    with _open_project(path, writable=True) as f:
        artifact_group = child_group(_object_group(f, "predictions", prediction_run_id),
                                     "artifact_refs")
        y_dataset = artifact_group.get("y_pred")
        if not isinstance(y_dataset, h5py.Dataset):
            log_and_raise("PredictionRun '{}' has no y_pred dataset", prediction_run_id,
                          error=KeyError)
        spatial_dataset = artifact_group.get("spatial_map")
        yield PredictionArtifactWriter(
            prediction_run_id,
            y_dataset,
            spatial_dataset if isinstance(spatial_dataset, h5py.Dataset) else None,
        )


def finalize_prediction_run_artifacts(
    path: PathLike, prediction_run_id: str, artifact_metadata: dict[str, Any]
) -> None:
    """Store summary metadata of an incrementally written run and mark it complete.

    Scalars and short numeric vectors of ``artifact_metadata`` become
    attributes of ``artifact_refs``; arrays and ``None`` values are skipped.
    """
    with _open_project(path, writable=True) as f:
        artifact_group = child_group(_object_group(f, "predictions", prediction_run_id),
                                     "artifact_refs")
        for key, value in artifact_metadata.items():
            if key in _PREDICTION_ARRAYS or value is None:
                continue
            if is_scalar_like(value):
                artifact_group.attrs[key] = value
                continue
            array = np.asarray(value)
            if array.ndim == 0:
                artifact_group.attrs[key] = array.item()
            elif (array.ndim == 1 and array.size <= MAX_ATTRIBUTE_VECTOR
                  and array.dtype.kind not in "UO"):
                artifact_group.attrs[key] = array
        artifact_group.attrs["artifact_write_status"] = "complete"
    logger.info("Finalized prediction artifacts '{}' in {}", prediction_run_id, path)


def discard_prediction_run_artifacts(path: PathLike, prediction_run_id: str) -> None:
    """Delete a staged (``"writing"``) prediction run if it exists.

    Completed runs are refused, so that cleaning up a failed attempt can never
    remove finished results.
    """
    with _open_project(path, writable=True) as f:
        group = f["predictions"].get(prediction_run_id)
        if group is None:
            return
        artifact_group = group.get("artifact_refs")
        status = (decode(artifact_group.attrs.get("artifact_write_status", "complete"))
                  if isinstance(artifact_group, h5py.Group) else "complete")
        if status != "writing":
            log_and_raise("Prediction run '{}' is {!r}, not a staged run; refusing to discard "
                          "it", prediction_run_id, status)
        del f["predictions"][prediction_run_id]
    logger.warning("Discarded prediction artifacts '{}' in {}", prediction_run_id, path)


def has_prediction_run(path: PathLike, prediction_run_id: str) -> bool:
    """Return whether the project already contains a prediction run with this id."""
    with _open_project(path) as f:
        return prediction_run_id in f["predictions"]


# ---------------------------------------------------------------------------
# On-demand loading
# ---------------------------------------------------------------------------


def load_roi_spectra(path: PathLike, hdf5_key: str) -> np.ndarray:
    """Return the ROI spectra stored at ``hdf5_key`` (e.g. ``"/rois/{id}/spectra"``).

    Raw spectra are returned as uint16, precalibrated spectra as float32.
    """
    return load_roi_spectra_batch(path, [hdf5_key])[hdf5_key]


def load_roi_spectra_batch(path: PathLike, hdf5_keys: Iterable[str]) -> dict[str, np.ndarray]:
    """Return ``{hdf5_key: spectra}`` for several ROIs, opening the file once."""
    spectra: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in map(str, hdf5_keys):
            if key not in f:
                log_and_raise("HDF5 key '{}' not found in {}", key, path, error=KeyError)
            spectra[key] = as_roi_storage_array(f[key][()])
    return spectra


def load_roi_spectra_matrix(
    path: PathLike, hdf5_keys: Iterable[str], *, n_bands: int
) -> tuple[np.ndarray, dict[str, slice]]:
    """Read several ROIs directly into one stacked matrix.

    Parameters
    ----------
    path:
        Project file.
    hdf5_keys:
        Spectra datasets to stack, in order.
    n_bands:
        Expected number of bands of every dataset.

    Returns
    -------
    matrix : numpy.ndarray, shape (total_pixels, n_bands)
        uint16, or float32 if any ROI stores floating spectra.
    rows : dict
        ``{hdf5_key: slice}`` with the rows of every ROI in ``matrix``.
    """
    keys = [str(key) for key in hdf5_keys]
    with h5py.File(path, "r") as f:
        datasets = []
        for key in keys:
            dataset = f.get(key)
            if not isinstance(dataset, h5py.Dataset):
                log_and_raise("ROI spectra dataset '{}' not found in {}", key, path,
                              error=KeyError)
            if dataset.ndim != 2 or dataset.shape[1] != n_bands:
                log_and_raise("ROI spectra '{}' must have shape (n_pixels, {}), got {}", key,
                              n_bands, dataset.shape)
            datasets.append(dataset)
        dtype = np.float32 if any(d.dtype.kind == "f" for d in datasets) else np.uint16
        matrix = np.empty((sum(d.shape[0] for d in datasets), n_bands), dtype=dtype)
        rows: dict[str, slice] = {}
        start = 0
        for key, dataset in zip(keys, datasets):
            rows[key] = slice(start, start + dataset.shape[0])
            dataset.read_direct(matrix, dest_sel=np.s_[rows[key], :])
            start = rows[key].stop
    return matrix, rows


def iter_roi_calibration_refs(
    path: PathLike, roi_ids: Iterable[str]
) -> Iterator[tuple[str, tuple[np.ndarray, np.ndarray] | None]]:
    """Yield ``(roi_id, (dark, white))`` per ROI, or ``(roi_id, None)`` without references."""
    with _open_project(path) as f:
        for roi_id in map(str, roi_ids):
            refs = _object_group(f, "rois", roi_id).get("calibration_refs")
            if refs is None:
                yield roi_id, None
                continue
            if not isinstance(refs, h5py.Group) or "dark" not in refs or "white" not in refs:
                log_and_raise("ROI '{}' calibration_refs needs dark and white datasets", roi_id,
                              error=KeyError)
            yield roi_id, (np.asarray(refs["dark"][()], np.float32),
                           np.asarray(refs["white"][()], np.float32))


def load_model_run_artifact_refs(
    path: PathLike, model_run_id: str, keys: Iterable[str] | None = None
) -> dict[str, Any]:
    """Read ``/model_runs/{model_run_id}/artifact_refs``, optionally only ``keys``."""
    with _open_project(path) as f:
        artifact_group = _object_group(f, "model_runs", model_run_id).get("artifact_refs")
        if not isinstance(artifact_group, h5py.Group):
            return {}
        return read_mapping(artifact_group, keys=None if keys is None else set(map(str, keys)))


def load_model_run_prediction_artifact(path: PathLike, model_run_id: str) -> dict[str, Any]:
    """Return the ``prediction_model`` artifact needed to apply a saved model run.

    Raises
    ------
    KeyError
        If the model run has no ``prediction_model`` artifact.
    """
    artifact = load_model_run_artifact_refs(path, model_run_id, keys=("prediction_model",))
    prediction_model = artifact.get("prediction_model")
    if not isinstance(prediction_model, dict):
        log_and_raise("ModelRun '{}' has no prediction_model artifact", model_run_id,
                      error=KeyError)
    return prediction_model


def load_prediction_run_artifact_refs(
    path: PathLike, prediction_run_id: str, keys: Iterable[str] | None = None
) -> dict[str, Any]:
    """Read the artifacts of a prediction run, optionally only ``keys``.

    If no ``spatial_map`` dataset is stored but ``spatial_map_shape`` is,
    ``"spatial_map"`` is reconstructed by reshaping ``y_pred``.
    """
    wanted = None if keys is None else set(map(str, keys))
    with _open_project(path) as f:
        group = _object_group(f, "predictions", prediction_run_id)
        # Read the run's own fields without the artifact arrays; those follow below.
        names = (set(group.attrs) | set(group)) - {"artifact_refs"}
        refs = {key: value
                for key, value in read_mapping(group, keys=names if wanted is None
                                               else names & wanted).items()
                if key not in _PREDICTION_RUN_FIELDS}
        artifact_group = group.get("artifact_refs")
        if isinstance(artifact_group, h5py.Group):
            refs.update(read_mapping(artifact_group, keys=wanted))
            wants_map = wanted is None or "spatial_map" in wanted
            y_dataset = artifact_group.get("y_pred")
            shape = _spatial_map_shape(artifact_group)
            if (wants_map and "spatial_map" not in refs and shape is not None
                    and isinstance(y_dataset, h5py.Dataset)
                    and int(np.prod(shape)) == int(np.prod(y_dataset.shape))):
                y_pred = refs["y_pred"] if "y_pred" in refs else y_dataset[()]
                refs["spatial_map"] = np.asarray(y_pred).reshape(shape)
    return refs


def load_prediction_spatial_map_preview(
    path: PathLike, prediction_run_id: str, *, max_pixels: int = 512 * 512
) -> np.ndarray:
    """Return a subsampled prediction map with at most about ``max_pixels`` pixels.

    Both spatial axes are subsampled with the same step; only the selected
    values are read from disk.
    """
    with _open_project(path) as f:
        artifact_group = child_group(_object_group(f, "predictions", prediction_run_id),
                                     "artifact_refs")
        spatial = artifact_group.get("spatial_map")
        if isinstance(spatial, h5py.Dataset):
            shape = tuple(int(dim) for dim in spatial.shape)
            if len(shape) < 2:
                return np.asarray(spatial[()], dtype=np.float32)
            step = _preview_step(shape, max_pixels)
            return np.asarray(spatial[::step, ::step, ...], dtype=np.float32)

        y_dataset = artifact_group.get("y_pred")
        shape = _spatial_map_shape(artifact_group)
        if not isinstance(y_dataset, h5py.Dataset) or shape is None:
            log_and_raise("PredictionRun '{}' has no spatial prediction artifact",
                          prediction_run_id, error=KeyError)
        if int(np.prod(shape)) != int(np.prod(y_dataset.shape)):
            log_and_raise("PredictionRun '{}' spatial_map_shape does not match y_pred",
                          prediction_run_id)
        if len(shape) < 2:
            return np.asarray(y_dataset[()], dtype=np.float32).reshape(shape)
        step = _preview_step(shape, max_pixels)
        # Read whole frames in contiguous blocks and subsample in memory; h5py point
        # selection of scattered pixels is orders of magnitude slower.
        frame_rows = y_dataset.shape[0] // shape[0]  # y_pred rows per frame
        frames_per_block = step * max(1, _PREVIEW_BLOCK_VALUES // (step * frame_rows))
        blocks = []
        for start in range(0, shape[0], frames_per_block):
            stop = min(start + frames_per_block, shape[0])
            block = y_dataset[start * frame_rows:stop * frame_rows]
            blocks.append(block.reshape(stop - start, *shape[1:])[::step, ::step])
        return np.asarray(np.concatenate(blocks), dtype=np.float32)


def as_roi_storage_array(value: Any) -> np.ndarray:
    """Return ROI spectra in their storage dtype: float32 if floating, else uint16.

    Raises
    ------
    ValueError
        If integer spectra do not fit into uint16 or the dtype is not numeric.
    """
    array = np.asarray(value)
    if array.dtype.kind == "f":
        return array.astype(np.float32, copy=False)
    if array.dtype.kind not in "iu":
        log_and_raise("ROI spectra must be numeric, got dtype {}", array.dtype)
    if array.dtype != np.uint16 and array.size and (array.min() < 0 or array.max() > 65535):
        log_and_raise("Integer ROI spectra must fit into uint16 (0..65535), got values from {} "
                      "to {}", array.min(), array.max())
    return array.astype(np.uint16, copy=False)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_ID_FIELDS = {
    "datasets": "dataset_id", "targets": "target_id", "recipes": "recipe_id",
    "signal_domains": "signal_domain_id", "rois": "roi_id", "sample_sets": "sample_set_id",
    "model_runs": "model_run_id", "predictions": "prediction_run_id",
}


def _validate_payload(label: str, location: str, payload: dict[str, Any],
                      ids: dict[str, str]) -> None:
    """Reject a payload that hydration would refuse, before anything is written."""
    # Imported here: enginir.project imports this module.
    from enginir.project._hydration import validate_object_payload

    validate_object_payload(label, location, payload, ids)


@contextmanager
def _open_project(path: PathLike, *, writable: bool = False) -> Iterator[h5py.File]:
    """Open and validate a project; after a successful write, update ``updated_at``."""
    with h5py.File(path, "r+" if writable else "r") as f:
        _check_project_file(f, path)
        yield f
        if writable:
            f["project_meta"].attrs["updated_at"] = _now_iso()


def _check_project_file(f: h5py.File, path: PathLike) -> None:
    if not is_project_file(f):
        version = decode(f["project_meta"].attrs.get("schema_version", "")) \
            if "project_meta" in f else None
        log_and_raise("{} is not an engiNIR {} project file (schema_version={!r})", path,
                      PROJECT_SCHEMA_VERSION, version, depth=3)
    missing = [name for name in PROJECT_TOP_LEVEL_GROUPS if name not in f]
    if missing:
        log_and_raise("Project {} is missing top-level group(s) {}", path, missing, depth=3)
    extra = tuple(sorted(set(f) - set(PROJECT_TOP_LEVEL_GROUPS)))
    if extra:
        _warn_extra_groups(str(Path(path).resolve()), extra)


@lru_cache(maxsize=256)
def _warn_extra_groups(path: str, names: tuple[str, ...]) -> None:
    logger.warning("Project {} has unknown top-level group(s) {}; ignoring them", path,
                   list(names))


def _object_group(f: h5py.File, collection: str, object_id: str) -> h5py.Group:
    group = f[collection].get(object_id)
    if not isinstance(group, h5py.Group):
        log_and_raise("'{}' not found in /{} of {}", object_id, collection, f.filename,
                      error=KeyError, depth=2)
    return group


def _save_object(
    path: PathLike,
    collection: str,
    object_id: str,
    payload: dict[str, Any],
    label: str,
    compression: str,
    *,
    mutable_fields: frozenset[str] = frozenset(),
    child_members: frozenset[str] = frozenset(),
) -> None:
    """Validate ``payload`` and write it to ``/{collection}/{object_id}``.

    The payload is checked as hydration will read it back (see
    :func:`_validate_payload`); a stable object is never changed.

    Re-saving identical content is a no-op; only ``mutable_fields`` (scalar
    labels) may change, and are updated in place. ``child_members`` are
    written by other functions (e.g. a dataset's calibrations) and are
    neither compared nor touched.
    """
    if label != "PreprocessingRecipe":  # validated by its writer
        _validate_payload(label, f"/{collection}/{object_id}", payload,
                          {_ID_FIELDS[collection]: object_id})
    payload = {key: value for key, value in payload.items() if value is not None}
    with _open_project(path, writable=True) as f:
        parent = f[collection]
        existing = parent.get(object_id)
        if existing is None:
            with replacement_group(parent, object_id) as group:
                write_mapping(group, payload, compression)
        else:
            if not isinstance(existing, h5py.Group):
                log_and_raise("{} '{}' in {} is not a group", label, object_id, path)
            stored = {key: value for key, value in read_mapping(existing).items()
                      if key not in child_members}
            fixed = set(stored) | set(payload)
            fixed -= mutable_fields
            if not values_equal({k: stored.get(k) for k in fixed},
                                {k: payload.get(k) for k in fixed}):
                log_and_raise(
                    "Refusing to overwrite {} '{}' with different content; stable ids are "
                    "immutable, use a new id",
                    label,
                    object_id,
                )
            for key in mutable_fields & set(payload):
                existing.attrs[key] = payload[key]
    logger.info("Saved {} '{}' to {}", label, object_id, path)


def _check_roi_arrays(group: h5py.Group) -> None:
    """Validate stored ROI array shapes without reading the arrays."""
    spectra = group.get("spectra")
    if not isinstance(spectra, h5py.Dataset) or spectra.ndim != 2 or spectra.dtype.kind not in "iuf":
        log_and_raise("ROI group '{}' must store a numeric 2-D spectra dataset", group.name)
    n_pixels = group.attrs.get("n_pixels")
    if n_pixels is not None and int(n_pixels) != spectra.shape[0]:
        log_and_raise("ROI group '{}' n_pixels={} does not match {} stored rows", group.name,
                      int(n_pixels), spectra.shape[0])
    wavelengths = group.get("wavelengths")
    if isinstance(wavelengths, h5py.Dataset) and wavelengths.shape != (spectra.shape[1],):
        log_and_raise("ROI group '{}' wavelengths shape {} does not match {} bands", group.name,
                      wavelengths.shape, spectra.shape[1])
    refs = group.get("calibration_refs")
    if isinstance(refs, h5py.Group):
        for key in ("dark", "white"):
            ref = refs.get(key)
            if not isinstance(ref, h5py.Dataset) or ref.shape != spectra.shape:
                log_and_raise(
                    "ROI group '{}' calibration_refs/{} shape {} does not match spectra shape {}",
                    group.name, key, getattr(ref, "shape", None), spectra.shape,
                )


def _check_prediction_complete(group: h5py.Group) -> None:
    artifact_group = group.get("artifact_refs")
    if not isinstance(artifact_group, h5py.Group):
        return
    status = decode(artifact_group.attrs.get("artifact_write_status", "complete"))
    if status == "writing":
        log_and_raise("Prediction run '{}' was not written completely; discard it", group.name)
    if status != "complete":
        log_and_raise("Prediction run '{}' has unknown artifact_write_status {!r}", group.name,
                      status)


def _spatial_map_shape(artifact_group: h5py.Group) -> tuple[int, ...] | None:
    value = artifact_group.attrs.get("spatial_map_shape")
    if value is None:
        return None
    return tuple(int(dim) for dim in np.ravel(value)) or None


def _preview_step(shape: tuple[int, ...], max_pixels: int) -> int:
    return max(1, int(np.ceil(np.sqrt(shape[0] * shape[1] / max(1, max_pixels)))))


def _chunk_shape(shape: tuple[int, ...]) -> tuple[int, ...] | None:
    """HDF5 chunks of up to 65536 values (1-D) or 32 frames (n-D)."""
    if not shape:
        return None
    if len(shape) == 1:
        return (min(max(shape[0], 1), 65536),)
    return (min(max(shape[0], 1), 32), *shape[1:])


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


__all__ = [
    "PROJECT_SCHEMA_VERSION",
    "PROJECT_TOP_LEVEL_GROUPS",
    "PredictionArtifactWriter",
    "as_roi_storage_array",
    "create_project",
    "discard_prediction_run_artifacts",
    "has_prediction_run",
    "finalize_prediction_run_artifacts",
    "initialize_prediction_run_artifacts",
    "is_project_file",
    "iter_roi_calibration_refs",
    "load_model_run_artifact_refs",
    "load_model_run_prediction_artifact",
    "load_prediction_run_artifact_refs",
    "load_prediction_spatial_map_preview",
    "load_project_store",
    "load_roi_spectra",
    "load_roi_spectra_batch",
    "load_roi_spectra_matrix",
    "open_prediction_run_artifact_writer",
    "remove_roi_from_project",
    "save_calibration_spec",
    "save_dataset_ref",
    "save_model_run",
    "save_prediction_run",
    "save_preprocessing_recipe",
    "save_sample_set",
    "save_signal_domain_contract",
    "save_target_assignment",
    "save_target_definition",
    "save_training_roi",
]
