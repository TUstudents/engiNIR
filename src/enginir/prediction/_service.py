"""Applying a saved model run to a dataset.

:func:`predict_with_model_run` checks that the dataset and its calibration can
produce the spectra the model was trained on, then streams the dataset in
chunks of frames. Every chunk is converted to the model's signal domain
(:func:`enginir.project.transform_spectra_to_signal_domain`), preprocessed with
the model's recipe and predicted. Predictions are returned as arrays and/or
written chunk by chunk into the project file, so whole cubes never need to fit
in memory. Results equal one-shot prediction up to float32 rounding (blocked
matrix products differ in the last bits).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import numpy as np

from enginir._checks import check_integer, log_and_raise
from enginir._logging import get_logger

from ..io import (
    SpectralData,
    file_format,
    iter_eiff_frame_chunks,
    iter_envi_frame_chunks,
    iter_hdf5_frame_chunks,
    load_spectra,
)
from ..io.project_store import (
    PredictionArtifactWriter,
    discard_prediction_run_artifacts,
    finalize_prediction_run_artifacts,
    has_prediction_run,
    initialize_prediction_run_artifacts,
    load_model_run_prediction_artifact,
    open_prediction_run_artifact_writer,
)
from ..preprocessing import PreprocessingRecipe
from ..preprocessing._pipeline import (
    restore_fitted_preprocessing_steps,
    transform_preprocessing_steps,
)
from ..project import (
    CalibrationSpec,
    CalibrationStats,
    DatasetRef,
    ModelRun,
    PredictionRun,
    SignalDomainContract,
    check_signal_domain_compatibility,
    transform_spectra_to_signal_domain,
)
from ..project._ids import new_id, now_iso

logger = get_logger(__name__)

_WORKING_DTYPE = np.float32
#: Default number of frames per prediction chunk.
DEFAULT_CHUNK_SIZE = 512
# Used when neither the caller nor the loader metadata names a modality.
_DEFAULT_SOURCE_MODALITY = "NIR"


class PredictionCancelledError(RuntimeError):
    """The prediction was cancelled through its ``cancel_callback``."""


@dataclass(frozen=True, slots=True)
class PredictionProgressEvent:
    """Progress report of a running prediction.

    Attributes
    ----------
    phase : str
        ``"preflight"``, ``"loading"``, ``"calibrating"``, ``"preprocessing"``,
        ``"predicting"``, ``"writing"`` or ``"finalizing"``. Every chunk passes
        through ``"loading"`` to ``"writing"``, so an error can be attributed
        to the phase of the last event.
    message : str
        Human-readable description.
    completed, total : int
        Chunks done and total (``total`` 0 while unknown).
    unit : str
        Unit of ``completed``/``total`` (``"chunk"``, or empty while unknown).
    indeterminate : bool
        ``True`` while progress cannot be quantified.
    """

    phase: str
    message: str
    completed: int = 0
    total: int = 0
    unit: str = ""
    indeterminate: bool = True


PredictionProgressCallback = Callable[[PredictionProgressEvent], None]
PredictionCancelCallback = Callable[[], bool]


@dataclass(slots=True)
class PredictionPreflight:
    """Runtime state resolved once per prediction by :func:`preflight_prediction`.

    Attributes
    ----------
    predictor : object or None
        In-memory predictor (includes preprocessing), or ``None`` to replay
        the saved model.
    model_blob : dict or None
        Saved ``prediction_model`` artifact when ``predictor`` is ``None``.
    preprocessing_steps : list
        Fitted recipe steps applied before a saved model.
    wavelengths : numpy.ndarray
        Wavelength axis of the model.
    n_chunks, chunk_size : int
        Number of chunks and frames per chunk.
    spatial_shape : tuple of int
        ``(n_frames, height)`` of the dataset (``height`` is 1 for spectra
        matrices).
    n_rows : int
        Number of predicted spectra, ``n_frames * height``.
    """

    predictor: Any | None
    model_blob: dict[str, Any] | None
    preprocessing_steps: list[tuple[str, Any]]
    wavelengths: np.ndarray
    n_chunks: int
    chunk_size: int
    spatial_shape: tuple[int, int]
    n_rows: int


@dataclass(slots=True)
class PredictionResult:
    """Outcome of :func:`predict_with_model_run`.

    Attributes
    ----------
    dataset, calibration : DatasetRef, CalibrationSpec
        The predicted dataset and the calibration used.
    prediction_run : PredictionRun
        Metadata of the run (stored in the project when
        ``artifacts_persisted``).
    y_pred : numpy.ndarray of float32 or None
        Prediction per spectrum, ``(n_rows,)`` or ``(n_rows, n_outputs)``;
        ``None`` unless ``return_arrays`` was set.
    spatial_map : numpy.ndarray of float32 or None
        ``y_pred`` viewed as ``(n_frames, height[, n_outputs])``.
    artifacts_persisted : bool
        Whether the predictions were written into the project file.
    """

    dataset: DatasetRef
    calibration: CalibrationSpec
    prediction_run: PredictionRun
    y_pred: np.ndarray | None
    spatial_map: np.ndarray | None
    artifacts_persisted: bool = False


@dataclass(frozen=True, slots=True)
class _InputChunk:
    raw_matrix: np.ndarray
    spatial_shape: tuple[int, int] | None
    row_start: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preflight_prediction(
    *,
    project_path: str,
    model_run: ModelRun,
    signal_domain: SignalDomainContract,
    recipe: PreprocessingRecipe,
    dataset: DatasetRef,
    calibration: CalibrationSpec,
    model_registry: dict[str, Any],
    chunk_size: int | None = None,
    prediction_run_id: str | None = None,
) -> PredictionPreflight:
    """Check that a prediction can run and resolve the state shared by all chunks.

    Everything that can be checked without reading the data is checked here:
    signal-domain compatibility, that the signal domain is the model run's, an
    in-memory predictor's recipe steps, the saved model artifact and its
    fitted preprocessing state, and that ``prediction_run_id`` is unused.

    Parameters
    ----------
    project_path:
        Project file holding the model run.
    model_run, signal_domain, recipe:
        The model and the signal domain and preprocessing it was trained with.
    dataset, calibration:
        Dataset to predict and its calibration.
    model_registry:
        In-memory predictors by model-run id (e.g. models fitted in this
        session); other model runs are replayed from the saved artifact.
    chunk_size:
        Frames per chunk; default ``min(n_frames, 512)``.
    prediction_run_id:
        Id the new prediction run will get; must not exist in the project.

    Raises
    ------
    SignalDomainCompatibilityError
        If the dataset or calibration does not fit the model.
    ValueError
        If ``chunk_size`` is not positive, the prediction run id is taken, an
        in-memory predictor does not apply the recipe's preprocessing, or the
        saved model artifact is incomplete or does not fit the band count.
    KeyError
        If the saved model artifact is missing.
    """
    if signal_domain.signal_domain_id != model_run.signal_domain_id:
        log_and_raise("Signal domain '{}' is not the one of model run '{}' ('{}')",
                      signal_domain.signal_domain_id, model_run.model_run_id,
                      model_run.signal_domain_id)
    check_signal_domain_compatibility(dataset=dataset, calibration=calibration,
                                      signal_domain=signal_domain)
    if prediction_run_id is not None and has_prediction_run(project_path, prediction_run_id):
        log_and_raise("Prediction run '{}' already exists in {}; use a new id",
                      prediction_run_id, project_path)
    n_frames, height, _ = dataset.cube_shape
    if chunk_size is None:
        chunk_size = max(1, min(n_frames, DEFAULT_CHUNK_SIZE))
    chunk_size = check_integer(chunk_size, "chunk_size", minimum=1)
    n_chunks = max(1, -(-n_frames // chunk_size))

    wavelengths = np.asarray(signal_domain.wavelengths, dtype=_WORKING_DTYPE)
    steps = recipe.build_steps(wavelengths)
    predictor = model_registry.get(model_run.model_run_id)
    model_blob = None
    if predictor is not None:
        if not hasattr(predictor, "predict"):
            log_and_raise("In-memory predictor of model run '{}' has no predict()",
                          model_run.model_run_id)
        _check_recipe_aware(predictor, steps, model_run.model_run_id)
    else:
        model_blob = load_model_run_prediction_artifact(project_path, model_run.model_run_id)
        _check_model_blob(model_blob, signal_domain.n_bands, model_run.model_run_id)
        if recipe.requires_fitting:
            steps = restore_fitted_preprocessing_steps(
                steps, recipe=recipe, state=model_blob.get("preprocessing_state")
            )
            for name, transformer in steps:
                n_features = getattr(transformer, "n_features_in_", signal_domain.n_bands)
                if n_features != signal_domain.n_bands:
                    log_and_raise("Saved {} state of model run '{}' has {} bands, the model "
                                  "needs {}", name.upper(), model_run.model_run_id, n_features,
                                  signal_domain.n_bands)
    logger.info("Prediction preflight: {} chunks of {} frames, cube shape {}", n_chunks,
                chunk_size, dataset.cube_shape)
    return PredictionPreflight(
        predictor=predictor,
        model_blob=model_blob,
        preprocessing_steps=steps,
        wavelengths=wavelengths,
        n_chunks=n_chunks,
        chunk_size=chunk_size,
        spatial_shape=(n_frames, height),
        n_rows=n_frames * height,
    )


def _check_recipe_aware(predictor: Any, steps: list[tuple[str, Any]], model_run_id: str) -> None:
    """Reject in-memory predictors that would skip the recipe's preprocessing."""
    if not steps or getattr(predictor, "recipe_aware", False):
        return
    named_steps = getattr(predictor, "named_steps", None)
    if named_steps is None:
        log_and_raise("In-memory predictor of model run '{}' is not recipe-aware; use a "
                      "Pipeline or an adapter that applies the recipe's preprocessing",
                      model_run_id, depth=2)
    missing = [name for name, _ in steps if name not in named_steps]
    if missing:
        log_and_raise("In-memory pipeline of model run '{}' lacks the recipe step(s) {}",
                      model_run_id, missing, depth=2)


def _check_model_blob(blob: dict[str, Any], n_bands: int, model_run_id: str) -> None:
    """Check that a saved prediction model has coefficients for ``n_bands`` bands."""
    missing = [key for key in ("coef_", "intercept_") if key not in blob]
    if missing:
        log_and_raise("Saved prediction model of model run '{}' lacks {}", model_run_id,
                      missing, depth=2)
    coef_shape = np.shape(blob["coef_"])
    if len(coef_shape) not in (1, 2) or coef_shape[0] != n_bands:
        log_and_raise("Saved prediction model of model run '{}' has coefficients of shape {} "
                      "for {} bands", model_run_id, coef_shape, n_bands, depth=2)


def predict_with_model_run(
    *,
    project_path: str,
    model_run: ModelRun,
    signal_domain: SignalDomainContract,
    recipe: PreprocessingRecipe,
    dataset: DatasetRef,
    calibration: CalibrationSpec,
    model_registry: dict[str, Any],
    data_path: str | None = None,
    spectral_data: SpectralData | None = None,
    preprocessing_recipe_id: str,
    prediction_run_id: str | None = None,
    created_at: str | None = None,
    loader_kwargs: dict[str, Any] | None = None,
    progress_callback: PredictionProgressCallback | None = None,
    cancel_callback: PredictionCancelCallback | None = None,
    chunk_size: int | None = None,
    persist_artifacts: bool = False,
    return_arrays: bool = True,
) -> PredictionResult:
    """Predict a whole dataset with a model run.

    Parameters
    ----------
    project_path, model_run, signal_domain, recipe, dataset, calibration, model_registry:
        As for :func:`preflight_prediction`.
    data_path:
        Data file to read; default ``dataset.source_path``.
    spectral_data:
        Already loaded data. A header-only load (empty ``X``) only supplies
        metadata; the data are then streamed from ``data_path``.
    preprocessing_recipe_id:
        Id of ``recipe`` in the project.
    prediction_run_id, created_at:
        Id and timestamp of the new prediction run (generated by default).
    loader_kwargs:
        Passed to the data reader (e.g. ``frames=`` or ``spectra_key=``).
    progress_callback:
        Called with :class:`PredictionProgressEvent` objects.
    cancel_callback:
        Polled after every progress step; returning ``True`` cancels.
    chunk_size:
        Frames per chunk; see :func:`preflight_prediction`.
    persist_artifacts:
        Write the run and its predictions into the project file, chunk by
        chunk. Nothing remains in the project if the prediction fails or is
        cancelled.
    return_arrays:
        Keep the predictions in memory and return them.

    Returns
    -------
    PredictionResult

    Raises
    ------
    PredictionCancelledError
        If ``cancel_callback`` returned ``True``.
    SignalDomainCompatibilityError, ValueError, KeyError
        See :func:`preflight_prediction`.

    Notes
    -----
    Contiguous frame ranges of EIFF and HDF5 files are read chunk by chunk.
    Other formats, and frame selections with a step, are read completely
    before chunking (logged as a warning).
    """
    loader_kwargs = dict(loader_kwargs or {})
    data_path = data_path or dataset.source_path
    run_id = prediction_run_id or new_id()
    progress = _Progress(progress_callback, cancel_callback)
    progress.report("preflight", "Validating prediction inputs")
    preflight = preflight_prediction(
        project_path=project_path,
        model_run=model_run,
        signal_domain=signal_domain,
        recipe=recipe,
        dataset=dataset,
        calibration=calibration,
        model_registry=model_registry,
        chunk_size=chunk_size,
        prediction_run_id=run_id if persist_artifacts else None,
    )
    progress.total = preflight.n_chunks
    run_payload = {
        "dataset_id": dataset.dataset_id,
        "calibration_id": calibration.calibration_id,
        "model_run_id": model_run.model_run_id,
        "signal_domain_id": model_run.signal_domain_id,
        "preprocessing_recipe_id": preprocessing_recipe_id,
        "created_at": created_at or now_iso(),
    }
    stats = CalibrationStats()
    with _PredictionOutputs(
        project_path=project_path if persist_artifacts else None,
        prediction_run_id=run_id,
        run_payload=run_payload,
        preflight=preflight,
        keep_arrays=return_arrays,
        base_metadata=_artifact_metadata(project_path, data_path, preflight, dataset,
                                         model_run),
    ) as outputs:
        chunks = _iter_input_chunks(data_path, spectral_data, dataset, loader_kwargs,
                                    preflight.chunk_size)
        while True:
            step = f"chunk {progress.completed + 1} of {preflight.n_chunks}"
            if progress.completed < preflight.n_chunks:
                progress.report("loading", f"Loading {step}")
            chunk = next(chunks, None)
            if chunk is None:
                break
            progress.report("calibrating", f"Calibrating {step}")
            signal = transform_spectra_to_signal_domain(
                chunk.raw_matrix,
                dataset=dataset,
                calibration=calibration,
                signal_domain=signal_domain,
                spatial_shape=chunk.spatial_shape,
                stats=stats,
                check_compatibility=False,
            )
            row_start = chunk.row_start
            del chunk  # the raw block is not needed after calibration
            progress.report("preprocessing", f"Preprocessing {step}")
            X_ready = _preprocess_chunk(signal, preflight)
            del signal  # one working chunk at a time (X_ready may be signal itself)
            progress.report("predicting", f"Predicting {step}")
            y_chunk = _predict_chunk(X_ready, preflight)
            del X_ready
            progress.report("writing", f"Writing {step}")
            outputs.add(row_start, y_chunk)
            progress.completed += 1
            del y_chunk
        progress.report("finalizing", "Finalizing prediction artifacts")
        artifact_refs = outputs.finish(stats)
    prediction_run = PredictionRun(
        prediction_run_id=run_id, artifact_refs=artifact_refs, **run_payload
    )
    progress.report("finalizing", f"Finalized prediction {run_id}", check_cancel=False)
    return PredictionResult(
        dataset=dataset,
        calibration=calibration,
        prediction_run=prediction_run,
        y_pred=outputs.y_pred,
        spatial_map=outputs.spatial_map,
        artifacts_persisted=persist_artifacts,
    )


def build_dataset_ref_from_spectral_data(
    *,
    data_path: str,
    spectral_data: SpectralData,
    dataset_id: str | None = None,
    source_format: str | None = None,
    source_modality: str | None = None,
    import_timestamp: str | None = None,
) -> DatasetRef:
    """Describe a newly loaded dataset as a :class:`~enginir.project.DatasetRef`.

    The format defaults to the loader's ``meta["format"]`` (or the file
    extension), the modality to ``meta["source_modality"]`` or ``"NIR"``.
    Wavelengths that do not match the band count are replaced by band
    indices (with a warning).
    """
    cube_shape = spectral_data.cube_shape
    n_bands = cube_shape[2]
    wavelengths = np.asarray(spectral_data.wavelengths, dtype=np.float32)
    if wavelengths.shape != (n_bands,):
        logger.warning("{}: {} wavelengths for {} bands; using band indices", data_path,
                       wavelengths.size, n_bands)
        wavelengths = np.arange(n_bands, dtype=np.float32)
    meta = spectral_data.meta
    return DatasetRef(
        dataset_id=dataset_id or new_id(),
        source_path=str(data_path),
        source_format=source_format or str(meta.get("format") or file_format(data_path)),
        source_modality=source_modality
        or str(meta.get("source_modality") or "")
        or _DEFAULT_SOURCE_MODALITY,
        cube_shape=cube_shape,
        n_bands=n_bands,
        wavelengths=wavelengths,
        loader_metadata=dict(meta),
        import_timestamp=import_timestamp or now_iso(),
    )


# ---------------------------------------------------------------------------
# Chunked input
# ---------------------------------------------------------------------------


def _iter_input_chunks(
    data_path: str,
    spectral_data: SpectralData | None,
    dataset: DatasetRef,
    loader_kwargs: dict[str, Any],
    chunk_size: int,
) -> Iterator[_InputChunk]:
    """Yield the dataset in chunks of ``chunk_size`` frames (rows of a spectra matrix).

    Image files are streamed; chunks may arrive in any order and are placed
    by their ``row_start``.
    """
    height = dataset.cube_shape[1]
    blocks: Iterator[tuple[int, np.ndarray]]
    source_format = dataset.source_format.lower()
    if spectral_data is not None and np.shape(spectral_data.X)[0] > 0:
        blocks = _blocks_of_array(np.asarray(spectral_data.X), chunk_size, height)
    elif source_format == "hdf5":
        blocks = iter_hdf5_frame_chunks(
            data_path, chunk_size=chunk_size, frames=loader_kwargs.get("frames"),
            spectra_key=str(loader_kwargs.get("spectra_key", "spectra")),
        )
    elif source_format in ("eiff", "eiff_raw"):
        blocks = iter_eiff_frame_chunks(data_path, chunk_size=chunk_size,
                                        frames=loader_kwargs.get("frames"))
    elif source_format == "envi":
        blocks = iter_envi_frame_chunks(data_path, chunk_size=chunk_size)
    else:
        # Point-spectrum formats (CSV, SPC, SPA, JCAMP) hold no image cube.
        X = np.asarray(load_spectra(data_path, **loader_kwargs).X)
        blocks = _blocks_of_array(X, chunk_size, height)
    for offset, block in blocks:
        rows_per_frame = height if block.ndim == 3 else 1
        chunk = _flat_chunk(block, offset * rows_per_frame, height)
        del block
        yield chunk


def _blocks_of_array(X: np.ndarray, chunk_size: int,
                     height: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(first row, view)`` of ``chunk_size`` frames; a flat matrix is cut at whole frames."""
    rows_per_chunk = chunk_size * height if X.ndim == 2 else chunk_size
    for first in range(0, X.shape[0], rows_per_chunk):
        yield first, X[first:first + rows_per_chunk]


def _flat_chunk(block: np.ndarray, row_start: int, height: int) -> _InputChunk:
    """Flatten a block of frames ``(frames, height, bands)`` to rows ``(rows, bands)``."""
    if block.ndim == 3:
        n_frames, block_height, n_bands = block.shape
        return _InputChunk(block.reshape(n_frames * block_height, n_bands),
                           (n_frames, block_height), row_start)
    if block.ndim == 2:
        layout = (block.shape[0] // height, height) if block.shape[0] % height == 0 else None
        return _InputChunk(block, layout, row_start)
    log_and_raise("Prediction input must be 2-D or 3-D, got shape {}", block.shape, depth=2)


def _preprocess_chunk(signal: np.ndarray, preflight: PredictionPreflight) -> np.ndarray:
    """Apply the saved recipe; an in-memory predictor preprocesses by itself."""
    if preflight.predictor is not None:
        return signal
    return transform_preprocessing_steps(signal, preflight.preprocessing_steps)


def _predict_chunk(X: np.ndarray, preflight: PredictionPreflight) -> np.ndarray:
    """Predict one chunk; a single output column is returned as a vector."""
    if preflight.predictor is not None:
        y = np.asarray(preflight.predictor.predict(X), dtype=_WORKING_DTYPE)
    else:
        blob = preflight.model_blob or {}
        coef = np.asarray(blob["coef_"], dtype=_WORKING_DTYPE)
        y = X @ coef + np.asarray(blob["intercept_"], dtype=_WORKING_DTYPE)
    return y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y


# ---------------------------------------------------------------------------
# Outputs and progress
# ---------------------------------------------------------------------------


class _PredictionOutputs:
    """Collects chunk predictions in memory and/or in staged project artifacts.

    Used as a context manager: if the block raises (including cancellation),
    staged artifacts are discarded so that no half-written run remains.
    """

    def __init__(
        self,
        *,
        project_path: str | None,
        prediction_run_id: str,
        run_payload: dict[str, Any],
        preflight: PredictionPreflight,
        keep_arrays: bool,
        base_metadata: dict[str, Any],
    ) -> None:
        self._project_path = project_path
        self._run_id = prediction_run_id
        self._run_payload = run_payload
        self._preflight = preflight
        self._keep_arrays = keep_arrays
        self._metadata = base_metadata
        self._stack = ExitStack()
        self._writer: PredictionArtifactWriter | None = None
        self._staged = False
        self._rows_written = 0
        self._chunks = 0
        self._min = np.inf
        self._max = -np.inf
        self.y_pred: np.ndarray | None = None
        self.spatial_map: np.ndarray | None = None

    def __enter__(self) -> _PredictionOutputs:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stack.close()
        if exc_type is not None and self._staged and self._project_path is not None:
            discard_prediction_run_artifacts(self._project_path, self._run_id)

    def add(self, row_start: int, y_chunk: np.ndarray) -> None:
        if self._chunks == 0:
            self._start(tuple(int(dim) for dim in y_chunk.shape[1:]))
        if self.y_pred is not None:
            self.y_pred[row_start:row_start + y_chunk.shape[0]] = y_chunk
        if self._writer is not None:
            self._writer.write_chunk(row_start=row_start, y_pred_chunk=y_chunk)
        finite = y_chunk[np.isfinite(y_chunk)]
        if finite.size:
            self._min = min(self._min, float(finite.min()))
            self._max = max(self._max, float(finite.max()))
        self._rows_written += y_chunk.shape[0]
        self._chunks += 1

    def _start(self, output_shape: tuple[int, ...]) -> None:
        preflight = self._preflight
        y_shape = (preflight.n_rows, *output_shape)
        map_shape = (*preflight.spatial_shape, *output_shape)
        n_outputs = int(np.prod(output_shape)) if output_shape else 1
        self._metadata.update(
            prediction_shape=y_shape,
            n_outputs=n_outputs,
            spatial_shape=preflight.spatial_shape,
            spatial_map_shape=map_shape,
        )
        if n_outputs > 1:
            self._metadata["output_axis"] = output_shape
        else:
            self._metadata["output_index"] = 0
        if self._keep_arrays:
            self.y_pred = np.empty(y_shape, dtype=_WORKING_DTYPE)
            self.spatial_map = self.y_pred.reshape(map_shape)
        if self._project_path is not None:
            # y_pred is stored; spatial_map is its reshape by spatial_map_shape.
            initialize_prediction_run_artifacts(
                self._project_path,
                self._run_id,
                {**self._run_payload, "artifact_refs": dict(self._metadata)},
                y_pred_shape=y_shape,
                spatial_map_shape=None,
            )
            self._staged = True
            self._writer = self._stack.enter_context(
                open_prediction_run_artifact_writer(self._project_path, self._run_id)
            )

    def finish(self, stats: CalibrationStats) -> dict[str, Any]:
        """Check completeness, finalize staged artifacts and return the artifact metadata."""
        if self._rows_written != self._preflight.n_rows:
            log_and_raise("Prediction produced {} spectra, expected {}", self._rows_written,
                          self._preflight.n_rows)
        self._stack.close()
        self._writer = None
        metadata = dict(self._metadata)
        metadata.update(
            completed_chunks=self._chunks,
            clip_low_count=stats.clip_low_count,
            clip_high_count=stats.clip_high_count,
            near_zero_denominator_count=stats.near_zero_denominator_count,
        )
        if np.isfinite(self._min):
            metadata.update(prediction_min=self._min, prediction_max=self._max)
        if self._project_path is not None:
            finalize_prediction_run_artifacts(self._project_path, self._run_id, metadata)
        elif self.y_pred is not None:
            # spatial_map is y_pred reshaped by spatial_map_shape; it is not stored twice.
            metadata["y_pred"] = self.y_pred
        return metadata


def _artifact_metadata(
    project_path: str,
    data_path: str,
    preflight: PredictionPreflight,
    dataset: DatasetRef,
    model_run: ModelRun,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_path": str(data_path),
        "output_dtype": "float32",
        "chunk_axis": "frames" if dataset.cube_shape[1] > 1 else "rows",
        "chunk_size": preflight.chunk_size,
        "n_chunks": preflight.n_chunks,
    }
    # The target names travel with the prediction model: in the replayed artifact, in the
    # artifact_refs of a run fitted in this session, or else on disk (a hydrated model run
    # predicted with an in-memory predictor is metadata-only).
    model = preflight.model_blob or model_run.artifact_refs.get("prediction_model")
    if model is None:
        try:
            model = load_model_run_prediction_artifact(project_path, model_run.model_run_id)
        except KeyError:  # an in-memory model run that was never saved
            model = {}
    target_names = model.get("target_names")
    if target_names is not None and len(target_names):
        metadata["output_names"] = [str(name) for name in target_names]
    return metadata


class _Progress:
    """Reports progress events and checks for cancellation after each one."""

    def __init__(
        self,
        callback: PredictionProgressCallback | None,
        cancel_callback: PredictionCancelCallback | None,
    ) -> None:
        self._callback = callback
        self._cancel_callback = cancel_callback
        self.completed = 0
        self.total = 0

    def report(self, phase: str, message: str, *, check_cancel: bool = True) -> None:
        if self._callback is not None:
            self._callback(PredictionProgressEvent(
                phase=phase,
                message=message,
                completed=self.completed,
                total=self.total,
                unit="chunk" if self.total else "",
                indeterminate=self.total <= 0,
            ))
        if check_cancel and self._cancel_callback is not None and self._cancel_callback():
            logger.info("Prediction cancelled during {}", phase)
            raise PredictionCancelledError(f"Prediction cancelled during {phase}")


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "PredictionCancelCallback",
    "PredictionCancelledError",
    "PredictionPreflight",
    "PredictionProgressCallback",
    "PredictionProgressEvent",
    "PredictionResult",
    "build_dataset_ref_from_spectral_data",
    "predict_with_model_run",
    "preflight_prediction",
]
