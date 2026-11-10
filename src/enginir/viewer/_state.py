"""ViewerState — central data hub for the HSI viewer.

All panels read from and write to this object exclusively.  No maths lives
in the panels; all derived data is computed here.
"""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from ..io import file_format, load_spectra
from .._workflow import (
    DEFAULT_BLACK_LEVEL,
    DEFAULT_WHITE_LEVEL,
    Dataset,
    Project,
    data_file_labels,
    resolve_dataset,
    resolve_prediction_calibration,
)
from ..project import (
    CalibrationSpec,
    DatasetRef,
    ModelRun,
    SampleSet,
    SignalDomainContract,
    TargetAssignment,
    TargetDefinition,
    TrainingROI,
)
from ..preprocessing import PreprocessingRecipe
from ..preprocessing.recipe import SG_FILTERS
from ..prediction import PredictionResult, predict_with_model_run
from .._checks import log_and_raise
from .._logging import get_logger
from ..io.project_store import as_roi_storage_array
from ..project._ids import now_iso

logger = get_logger(__name__)


__all__ = [
    "DEFAULT_BLACK_LEVEL",
    "DEFAULT_SOURCE_MODALITY",
    "DEFAULT_WHITE_LEVEL",
    "PredictionJobContext",
    "ProjectState",
    "ViewerState",
]

# Default acquisition modality for viewer-imported datasets when the loader
# does not populate ``meta["source_modality"]``. The library is NIR-focused;
# DatasetRef requires a non-empty modality string.
DEFAULT_SOURCE_MODALITY: str = "NIR"
_EPS = 1e-10
_WORKING_DTYPE = np.float32
#: Predictions kept in memory in one viewer session (newest last); older ones are dropped.
SESSION_PREDICTIONS = 3


def _quoted_id_list(ids: list[str]) -> str:
    return ", ".join(f"'{item}'" for item in ids)


@dataclass(slots=True)
class PredictionJobContext:
    """Snapshot of the project state needed to compute one prediction run."""

    project_path: str
    model_run: ModelRun
    signal_domain: SignalDomainContract
    preprocessing_recipe_id: str
    recipe: PreprocessingRecipe
    datasets: dict[str, DatasetRef]
    calibrations: dict[str, CalibrationSpec]
    active_calibration_ids: dict[str, str]
    model_registry: dict[str, Any]


def _load_prediction_metadata(data_path: str, *, loader_kwargs: dict[str, Any] | None) -> Any:
    """Read the metadata of a prediction input; EIFF and HDF5 skip the pixel data."""
    kwargs = dict(loader_kwargs or {})
    if file_format(data_path) in ("eiff", "hdf5"):
        kwargs["header_only"] = True
    return load_spectra(data_path, **kwargs)


#: Every live project mirror of this process, to refresh windows on the same project.
_OPEN_STATES: weakref.WeakSet[ProjectState] = weakref.WeakSet()


class ProjectState(QObject):
    """Qt adapter of :class:`enginir.Project` for the viewer.

    Every write goes through the :class:`~enginir.Project` of the open file,
    which holds all workflow rules; this class mirrors its collections (keyed
    by id) for the widgets, keeps fitted predictors in memory and emits a
    signal for every change.
    """

    project_loaded = pyqtSignal()
    datasets_changed = pyqtSignal()
    calibrations_changed = pyqtSignal()
    rois_changed = pyqtSignal()
    targets_changed = pyqtSignal()
    target_assignments_changed = pyqtSignal()
    recipes_changed = pyqtSignal()
    signal_domains_changed = pyqtSignal()
    sample_sets_changed = pyqtSignal()
    model_runs_changed = pyqtSignal()
    predictions_changed = pyqtSignal()
    model_fitted = pyqtSignal(str)
    active_model_changed = pyqtSignal(str)
    prediction_completed = pyqtSignal(str)
    #: A short message after a user-visible write, e.g. "ROI 'A01' saved to project".
    saved = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.project: Project | None = None
        self.project_path: str | None = None
        self.project_meta: dict[str, Any] = {}
        self.datasets: dict[str, DatasetRef] = {}
        self.active_calibration_ids: dict[str, str] = {}
        self.calibrations: dict[str, CalibrationSpec] = {}
        self.rois: dict[str, TrainingROI] = {}
        self.targets: dict[str, TargetDefinition] = {}
        self.target_assignments: dict[tuple[str, str], TargetAssignment] = {}
        self.recipes: dict[str, dict[str, Any]] = {}
        self.signal_domains: dict[str, SignalDomainContract] = {}
        self.sample_sets: dict[str, SampleSet] = {}
        self.model_runs: dict[str, ModelRun] = {}
        #: Predictions of this session, newest last (never stored in the project).
        self.predictions: dict[str, PredictionResult] = {}
        #: Display label per dataset id: the file name, with its folder when names repeat.
        self.data_file_labels: dict[str, str] = {}
        self.model_registry: dict[str, Any] = {}
        self.active_model_name: str | None = None
        # Another window's write to the same project re-reads it once, on the next event loop pass.
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(0)
        self._reload_timer.timeout.connect(self.reload)
        _OPEN_STATES.add(self)

    # -- stale sample sets ---------------------------------------------------------

    def missing_roi_ids_for_sample_set(self, sample_set_id: str) -> list[str]:
        """Return deleted/missing ROI ids still referenced by a sample set."""
        return [] if self.project is None else self.project.missing_rois(sample_set_id)

    def sample_sets_referencing_roi(self, roi_id: str) -> list[str]:
        """Return sample-set ids that still reference the given ROI id."""
        return [sample_set_id for sample_set_id, sample_set in self.sample_sets.items()
                if roi_id in sample_set.roi_ids]

    def model_runs_for_sample_sets(self, sample_set_ids: list[str]) -> list[str]:
        """Return model-run ids bound to any of the given sample-set ids."""
        wanted = set(sample_set_ids)
        return [model_run_id for model_run_id, model_run in self.model_runs.items()
                if model_run.sample_set_id in wanted]

    def stale_sample_set_message(self, sample_set_id: str) -> str | None:
        """Return user-facing stale-reference details for a sample set if needed."""
        missing_roi_ids = self.missing_roi_ids_for_sample_set(sample_set_id)
        if not missing_roi_ids:
            return None
        return (
            f"SampleSet '{sample_set_id}' references deleted ROI(s) "
            f"{_quoted_id_list(missing_roi_ids)}. Existing saved model runs remain usable "
            "for prediction, but affected sample sets and model runs cannot be altered "
            "or refit until recreated."
        )

    # -- opening -------------------------------------------------------------------

    def new_project(self, path: str) -> None:
        """Create a new empty project file and set it as active."""
        self._set_project(Project.create(path))
        logger.info("New project: {}", path)

    def open_project(self, path: str, *, store: dict[str, Any] | None = None) -> None:
        """Open a project file; ``store`` is its already read contents, if any."""
        self._set_project(Project(path, store_contents=store))
        logger.info("Opened project: {} ({} ROIs)", path, len(self.rois))

    def _set_project(self, project: Project) -> None:
        if self.project_path is not None and self.project_path != project.path:
            self._reset_session_model_state()
        self.project = project
        self.project_path = project.path
        self._sync()
        self.project_loaded.emit()
        self._emit_workspace_signals()

    def _reset_session_model_state(self) -> None:
        """Drop fitted predictors and the active model when the project changes."""
        if self.model_registry or self.active_model_name is not None:
            logger.info(
                "Clearing {} in-memory predictor(s) and active model '{}' for project switch",
                len(self.model_registry),
                self.active_model_name,
            )
        self.model_registry.clear()
        self.active_model_name = None
        self.predictions.clear()

    # -- writing -------------------------------------------------------------------

    def register_dataset(self, dataset: DatasetRef) -> None:
        """Persist a dataset reference."""
        self._require_project("register_dataset").save_dataset(dataset)
        self._written()
        self.datasets_changed.emit()

    def add_data_file(self, path: str, **calibration: Any) -> str:
        """Register a data file with its calibration, or recalibrate a registered one.

        ``calibration`` are the reference keywords of
        :meth:`enginir.Project.add_dataset`. Returns the dataset id.
        """
        project = self._require_project("add_data_file")
        known = set(self.datasets)
        dataset_id = project.add_dataset(path, **calibration).ref.dataset_id
        self._written()
        self.datasets_changed.emit()
        self.calibrations_changed.emit()
        name = os.path.basename(str(path))
        self.saved.emit(f"Data file '{name}' added to project" if dataset_id not in known
                        else f"Calibration of '{name}' saved to project")
        return dataset_id

    def locate_data_file(self, dataset_id: str, path: str) -> None:
        """Record the new location of a moved data file after checking its pixels.

        See :meth:`enginir.Dataset.locate`; nothing is written if the file at
        ``path`` is not this data file.
        """
        project = self._require_project("locate_data_file")
        ref = self.datasets.get(dataset_id)
        if ref is None:
            log_and_raise("Unknown dataset '{}'", dataset_id, error=KeyError)
        Dataset(project, ref).locate(path)
        self._written()
        self.datasets_changed.emit()
        self.saved.emit(f"Data file '{os.path.basename(path)}' located")

    def data_file_label(self, dataset_id: str) -> str:
        """How a data file is shown: its file name, with the folder when file names repeat."""
        return self.data_file_labels.get(dataset_id, dataset_id)

    def active_calibration(self, dataset_id: str) -> CalibrationSpec | None:
        """The active calibration of a registered dataset, if any."""
        return self.calibrations.get(self.active_calibration_ids.get(dataset_id, ""))

    def set_active_calibration(self, calibration: CalibrationSpec) -> None:
        """Persist a calibration and make it its dataset's active one."""
        self._require_project("set_active_calibration").save_calibration(calibration)
        self._written()
        self.calibrations_changed.emit()

    def define_target(self, target: TargetDefinition) -> None:
        """Persist a target definition."""
        self._require_project("define_target").save_target(target)
        self._written()
        self.targets_changed.emit()
        self.saved.emit(f"Target '{target.name}' saved to project")

    def assign_target_to_roi(self, assignment: TargetAssignment) -> None:
        """Persist one ROI/target assignment."""
        self._require_project("assign_target_to_roi").save_assignment(assignment)
        self._written()
        self.target_assignments_changed.emit()
        target = self.targets.get(assignment.target_id)
        roi = self.rois.get(assignment.roi_id)
        unit = f" {target.unit}" if target is not None and target.unit else ""
        self.saved.emit(
            f"{target.name if target is not None else assignment.target_id} = "
            f"{assignment.value:g}{unit} saved for ROI "
            f"'{roi.name if roi is not None else assignment.roi_id}'")

    def set_roi_values(self, changes: list[tuple[str, str, float | None]]) -> None:
        """Set or remove target values: ``(roi_id, target_id, value)``, ``None`` removes.

        Every value is checked before any is written, so an invalid one leaves
        the project unchanged.
        """
        project = self._require_project("set_roi_values")
        now = now_iso()
        assignments = [TargetAssignment(roi_id=roi_id, target_id=target_id, value=value,
                                        assigned_at=now)
                       for roi_id, target_id, value in changes if value is not None]
        project.save_assignments(assignments)
        for roi_id, target_id, value in changes:
            if value is None:
                project.remove_assignment(roi_id, target_id)
        self._written()
        self.target_assignments_changed.emit()
        if len(changes) == 1:
            roi_id, target_id, value = changes[0]
            target, roi = self.targets.get(target_id), self.rois.get(roi_id)
            target_name = target.name if target is not None else target_id
            roi_name = roi.name if roi is not None else roi_id
            unit = f" {target.unit}" if target is not None and target.unit else ""
            self.saved.emit(f"{target_name} value removed for ROI '{roi_name}'" if value is None
                            else f"{target_name} = {value:g}{unit} saved for ROI '{roi_name}'")
        elif changes:
            self.saved.emit(f"{len(changes)} target values saved to project")

    def set_roi_samples(self, changes: list[tuple[str, str | None]]) -> None:
        """Set or remove sample labels: ``(roi_id, label)``; ``None`` or blank text removes.

        Labels only change which ROIs are one physical sample (fold groups);
        the ROIs themselves are not changed.
        """
        project = self._require_project("set_roi_samples")
        for roi_id, label in changes:
            project.save_roi_sample(roi_id, label)
        self._written()
        self.rois_changed.emit()
        if len(changes) == 1:
            roi_id, _label = changes[0]
            roi = self.rois.get(roi_id)
            name = roi.name if roi is not None else roi_id
            label = roi.sample if roi is not None else None
            self.saved.emit(f"Sample label removed for ROI '{name}'" if label is None
                            else f"Sample '{label}' saved for ROI '{name}'")
        elif changes:
            self.saved.emit(f"{len(changes)} sample labels saved to project")

    def set_roi_uses(self, changes: list[tuple[str, str]]) -> None:
        """Mark ROIs for ``"training"`` or ``"prediction"`` (``use``, roi-target contract)."""
        project = self._require_project("set_roi_uses")
        for roi_id, use in changes:
            project.save_roi_use(roi_id, use)
        self._written()
        self.rois_changed.emit()
        if len(changes) == 1:
            roi_id, use = changes[0]
            roi = self.rois.get(roi_id)
            name = roi.name if roi is not None else roi_id
            self.saved.emit(f"ROI '{name}' is now a {use} ROI")
        elif changes:
            self.saved.emit(f"{len(changes)} ROI uses saved to project")

    def add_roi(
        self,
        roi: TrainingROI,
        spectra: np.ndarray,
        *,
        dark_ref_spectra: np.ndarray | None = None,
        white_ref_spectra: np.ndarray | None = None,
    ) -> None:
        """Persist an ROI with its spectra."""
        self._require_project("add_roi").save_roi(roi, spectra, dark_ref_spectra=dark_ref_spectra,
                                                   white_ref_spectra=white_ref_spectra)
        self._written()
        logger.info("Added ROI '{}' ({} pixels)", roi.name, roi.n_pixels)
        self.rois_changed.emit()
        self.saved.emit(f"ROI '{roi.name}' saved to project ({roi.n_pixels} pixels)")

    def add_rois(self, rois: list[Any]) -> None:
        """Persist several ROIs (``(roi, spectra, refs)``) completely or not at all."""
        self._require_project("add_rois").save_rois(rois)
        self._written()
        logger.info("Added {} ROIs", len(rois))
        self.rois_changed.emit()
        self.saved.emit(f"{len(rois)} ROIs saved to project")

    def register_recipe(self, recipe_id: str, recipe_dict: dict[str, Any]) -> None:
        """Persist a preprocessing recipe."""
        recipe_id = str(recipe_id).strip()
        if not recipe_id:
            log_and_raise("recipe_id must be a non-empty string")
        self._require_project("register_recipe").save_recipe(recipe_id, recipe_dict)
        self._written()
        self.recipes_changed.emit()

    def register_signal_domain(self, signal_domain: SignalDomainContract) -> None:
        """Persist a signal-domain contract."""
        self._require_project("register_signal_domain").save_signal_domain(signal_domain)
        self._written()
        self.signal_domains_changed.emit()

    def create_sample_set(self, sample_set: SampleSet) -> None:
        """Persist a sample set."""
        self._require_project("create_sample_set").save_sample_set(sample_set)
        self._written()
        self.sample_sets_changed.emit()

    def build_sample_set(
        self,
        *,
        target_id: str,
        preprocessing_recipe_id: str,
        roi_ids: list[str] | None = None,
        sample_set_id: str | None = None,
        created_at: str | None = None,
    ) -> Any:
        """Build and persist one sample set from the current project workspace."""
        from ..modeling import build_sample_set as build_project_sample_set

        project_path = self._require_project_path("build_sample_set")
        sample_set_data = build_project_sample_set(
            project_path=project_path,
            rois=self.rois,
            target_assignments=self.target_assignments,
            target_id=target_id,
            preprocessing_recipe_id=preprocessing_recipe_id,
            roi_ids=roi_ids,
            sample_set_id=sample_set_id,
            created_at=created_at,
        )
        self.create_sample_set(sample_set_data.sample_set)
        return sample_set_data

    def register_model_run(self, model_run: ModelRun) -> None:
        """Persist a model run."""
        self._require_project("register_model_run").save_model_run(model_run)
        self._written()
        self.model_runs_changed.emit()

    def fit_model_run(
        self,
        *,
        sample_set_id: str,
        signal_domain_id: str,
        signal_domain: SignalDomainContract | None = None,
        model_type: str,
        hyperparameters: dict[str, Any] | None = None,
        cv: int | str = "auto",
        model_run_id: str | None = None,
        created_at: str | None = None,
    ) -> Any:
        """Fit, persist, and activate one grouped model run."""
        from ..modeling import fit_model_run as fit_project_model_run

        project = self._require_project("fit_model_run")
        if sample_set_id not in self.sample_sets:
            log_and_raise("Unknown sample_set_id '{}'", sample_set_id)
        project.require_current_sample_set(sample_set_id, action="fit model run")
        sample_set = self.sample_sets[sample_set_id]
        target = self.targets.get(sample_set.target_id)
        signal_domains = dict(self.signal_domains)
        if signal_domain is not None:
            signal_domains[signal_domain.signal_domain_id] = signal_domain
        if signal_domain_id not in signal_domains:
            log_and_raise("Unknown signal_domain_id '{}'", signal_domain_id)
        model_run_result = fit_project_model_run(
            project_path=project.path,
            sample_set=sample_set,
            datasets=self.datasets,
            calibrations=self.calibrations,
            rois=self.rois,
            target_assignments=self.target_assignments,
            recipes=self.recipes,
            signal_domains=signal_domains,
            signal_domain_id=signal_domain_id,
            model_type=model_type,
            hyperparameters=hyperparameters,
            cv=cv,
            model_run_id=model_run_id,
            created_at=created_at,
            target_name=None if target is None else target.name,
        )
        self.finalize_model_run_result(model_run_result=model_run_result,
                                       signal_domain=signal_domain)
        return model_run_result

    def finalize_model_run_result(
        self,
        *,
        model_run_result: Any,
        signal_domain: SignalDomainContract | None = None,
    ) -> None:
        """Persist one already-computed model run and make it the active model."""
        if signal_domain is not None and signal_domain.signal_domain_id not in self.signal_domains:
            self.register_signal_domain(signal_domain)
        model_run = model_run_result.model_run
        if model_run.name is None:
            model_run = replace(model_run, name=self._require_project(
                "finalize_model_run_result").default_model_name(model_run))
        self.register_model_run(model_run)
        self.model_registry[model_run.model_run_id] = model_run_result.predictor
        self.active_model_name = model_run.model_run_id
        self.model_fitted.emit(model_run.model_run_id)
        self.active_model_changed.emit(model_run.model_run_id)

    def build_prediction_job_context(self, *, model_run_id: str) -> PredictionJobContext:
        """Return a snapshot of everything a background prediction needs."""
        project_path = self._require_project_path("build_prediction_job_context")
        if model_run_id not in self.model_runs:
            log_and_raise("Unknown model_run_id '{}'", model_run_id)
        model_run = self.model_runs[model_run_id]
        if model_run.sample_set_id not in self.sample_sets:
            log_and_raise("ModelRun '{}' references missing sample_set_id '{}'",
                          model_run_id, model_run.sample_set_id)
        if model_run.signal_domain_id not in self.signal_domains:
            log_and_raise("ModelRun '{}' references missing signal_domain_id '{}'",
                          model_run_id, model_run.signal_domain_id)
        sample_set = self.sample_sets[model_run.sample_set_id]
        stale_message = self.stale_sample_set_message(sample_set.sample_set_id)
        if stale_message is not None:
            logger.warning("{} Prediction with saved model run '{}' is still allowed.",
                           stale_message, model_run_id)
        recipe_id = sample_set.preprocessing_recipe_id
        if recipe_id not in self.recipes:
            log_and_raise("ModelRun '{}' references missing preprocessing_recipe_id '{}'",
                          model_run_id, recipe_id)
        return PredictionJobContext(
            project_path=project_path,
            model_run=model_run,
            signal_domain=self.signal_domains[model_run.signal_domain_id],
            preprocessing_recipe_id=recipe_id,
            recipe=PreprocessingRecipe.from_dict(self.recipes[recipe_id]),
            datasets=dict(self.datasets),
            calibrations=dict(self.calibrations),
            active_calibration_ids=dict(self.active_calibration_ids),
            model_registry=dict(self.model_registry),
        )

    def add_prediction(self, result: PredictionResult) -> None:
        """Keep a computed prediction for this session and show it (nothing is written)."""
        run_id = result.prediction_run.prediction_run_id
        self.predictions[run_id] = result
        while len(self.predictions) > SESSION_PREDICTIONS:
            dropped = next(iter(self.predictions))
            del self.predictions[dropped]
            logger.info("Dropped session prediction '{}' (at most {} are kept)", dropped,
                        SESSION_PREDICTIONS)
        self.predictions_changed.emit()
        self.prediction_completed.emit(run_id)

    def predict_dataset(
        self,
        *,
        model_run_id: str,
        data_path: str,
        calibration: CalibrationSpec | None = None,
        dataset_id: str | None = None,
        prediction_run_id: str | None = None,
        created_at: str | None = None,
        loader_kwargs: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Resolve the dataset and calibration and predict in memory; nothing is written.

        The result (with the DModXPS and T²PS maps when the saved model has
        projection data) is kept for this session (:meth:`add_prediction`).
        """
        context = self.build_prediction_job_context(model_run_id=model_run_id)
        spectral_data = _load_prediction_metadata(data_path, loader_kwargs=loader_kwargs)
        dataset, _ = resolve_dataset(context.datasets, data_path, spectral_data,
                                     dataset_id=dataset_id, created_at=created_at)
        resolved_calibration, _ = resolve_prediction_calibration(
            dataset, context.calibrations, context.active_calibration_ids, calibration)
        result = predict_with_model_run(
            project_path=context.project_path,
            model_run=context.model_run,
            signal_domain=context.signal_domain,
            recipe=context.recipe,
            dataset=dataset,
            calibration=resolved_calibration,
            model_registry=context.model_registry,
            data_path=data_path,
            spectral_data=spectral_data,
            preprocessing_recipe_id=context.preprocessing_recipe_id,
            prediction_run_id=prediction_run_id,
            created_at=created_at,
            loader_kwargs=loader_kwargs,
            prefer_saved_model=True,
        )
        self.add_prediction(result)
        return result

    def remove_roi(self, roi_id: str) -> None:
        """Delete an ROI and its assignments."""
        removed = self.rois.get(roi_id)
        affected_sample_set_ids = self._require_project("remove_roi").remove_roi(roi_id)
        self._written()
        logger.info("Removed ROI '{}'", roi_id)
        self.rois_changed.emit()
        self.saved.emit(f"ROI '{removed.name if removed is not None else roi_id}' "
                        "removed from project")
        self.target_assignments_changed.emit()
        if affected_sample_set_ids:
            self.sample_sets_changed.emit()
            if self.model_runs_for_sample_sets(affected_sample_set_ids):
                self.model_runs_changed.emit()

    def get_target_assignment(self, roi_id: str, target_id: str) -> TargetAssignment | None:
        """Return one assignment by `(roi_id, target_id)` if present."""
        return self.target_assignments.get((roi_id, target_id))

    def _require_project(self, action: str) -> Project:
        if self.project is None:
            log_and_raise("{} called with no active project", action, error=RuntimeError, depth=2)
        return self.project

    def _require_project_path(self, action: str) -> str:
        return self._require_project(action).path

    # -- heavy artifacts -------------------------------------------------------------

    def ensure_model_run_artifact_refs(self, model_run_id: str) -> dict[str, Any]:
        """Load heavy model-run artifact refs on demand and merge them into memory."""
        from ..io.project_store import load_model_run_artifact_refs

        project_path = self._require_project_path("ensure_model_run_artifact_refs")
        model_run = self.model_runs.get(model_run_id)
        if model_run is None:
            log_and_raise("Unknown model_run_id '{}'", model_run_id, error=KeyError)
        loaded = load_model_run_artifact_refs(project_path, model_run_id)
        model_run.artifact_refs = _merge_artifact_refs(model_run.artifact_refs, loaded)
        return model_run.artifact_refs

    # -- mirroring -------------------------------------------------------------------

    def reload(self) -> None:
        """Read the project file again and refresh every view (several windows on one project)."""
        if self.project is None:
            return
        self.project.reload()
        self._sync()
        logger.info("Reloaded project {} after a write from another window", self.project_path)
        self._emit_workspace_signals()

    def _written(self) -> None:
        """Mirror this project after a write and have other windows on it re-read it."""
        self._sync()
        path = self.project_path
        if path is None:
            return
        for other in list(_OPEN_STATES):
            if other is not self and other.project_path is not None \
                    and os.path.normcase(os.path.abspath(other.project_path)) \
                    == os.path.normcase(os.path.abspath(path)):
                other._reload_timer.start()

    def _sync(self) -> None:
        """Mirror the collections of the open project."""
        if self.project is None:
            return
        ws = self.project.workspace
        self.project_meta = ws.project_meta
        self.datasets = dict(ws.datasets)
        self.active_calibration_ids = dict(ws.active_calibration_ids)
        self.calibrations = dict(ws.calibrations)
        self.rois = dict(ws.rois)
        self.targets = dict(ws.targets)
        self.target_assignments = dict(ws.target_assignments)
        self.recipes = dict(ws.recipes)
        self.signal_domains = dict(ws.signal_domains)
        self.sample_sets = dict(ws.sample_sets)
        self.model_runs = dict(ws.model_runs)
        self.data_file_labels = data_file_labels(self.datasets)

    def _emit_workspace_signals(self) -> None:
        self.datasets_changed.emit()
        self.calibrations_changed.emit()
        self.rois_changed.emit()
        self.targets_changed.emit()
        self.target_assignments_changed.emit()
        self.recipes_changed.emit()
        self.signal_domains_changed.emit()
        self.sample_sets_changed.emit()
        self.model_runs_changed.emit()
        self.predictions_changed.emit()


def _merge_artifact_refs(current: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in loaded.items():
        current_value = merged.get(key)
        merged[key] = (_merge_artifact_refs(current_value, value)
                       if isinstance(current_value, dict) and isinstance(value, dict) else value)
    return merged


class ViewerState(QObject):
    """Central state hub for the HSI viewer.

    Parameters
    ----------
    cube:
        Hyperspectral image cube, shape ``(n_frames, height, n_bands)``:
        integer raw counts (kept as uint16) or a floating precalibrated
        signal (kept as float32, see :attr:`calibrated`).
    wavelengths:
        Wavelength axis in nm, shape ``(n_bands,)``.
    meta:
        Loader metadata dictionary (optional).
    black_level:
        Default dark level used when no dark reference is loaded.
    white_level:
        Default white level used when no white reference is loaded.
    """

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    frame_changed = pyqtSignal(int)
    row_changed = pyqtSignal(int)
    band_changed = pyqtSignal(int)
    units_changed = pyqtSignal()

    def __init__(
        self,
        cube: np.ndarray,
        wavelengths: np.ndarray,
        meta: dict | None = None,
        *,
        black_level: float = DEFAULT_BLACK_LEVEL,
        white_level: float = DEFAULT_WHITE_LEVEL,
    ) -> None:
        super().__init__()
        self._reset_dataset_state(
            cube,
            wavelengths,
            meta,
            black_level=black_level,
            white_level=white_level,
        )

    def load_dataset(
        self,
        cube: np.ndarray,
        wavelengths: np.ndarray,
        meta: dict | None = None,
        *,
        black_level: float = DEFAULT_BLACK_LEVEL,
        white_level: float = DEFAULT_WHITE_LEVEL,
        dark_ref: np.ndarray | None = None,
        white_ref: np.ndarray | None = None,
    ) -> None:
        """Replace the current dataset payload in-place and notify all panels."""
        self._reset_dataset_state(
            cube,
            wavelengths,
            meta,
            black_level=black_level,
            white_level=white_level,
        )

        if dark_ref is not None:
            self.dark_ref = self._prepare_ref(dark_ref, "dark_ref")
        if white_ref is not None:
            self.white_ref = self._prepare_ref(white_ref, "white_ref")

        self.frame_changed.emit(self._frame)
        self.row_changed.emit(self._row)
        self.band_changed.emit(self._band)
        self.units_changed.emit()

    def _reset_dataset_state(
        self,
        cube: np.ndarray,
        wavelengths: np.ndarray,
        meta: dict | None,
        *,
        black_level: float,
        white_level: float,
    ) -> None:
        cube_arr = np.asarray(cube)
        if cube_arr.ndim != 3:
            log_and_raise("cube must be 3-D, got shape {}", cube_arr.shape)
        wavelengths_arr = np.asarray(wavelengths, dtype=_WORKING_DTYPE)
        if wavelengths_arr.ndim != 1:
            log_and_raise("wavelengths must be 1-D, got shape {}", wavelengths_arr.shape)
        if wavelengths_arr.shape[0] != cube_arr.shape[2]:
            log_and_raise(
                "wavelengths length {} does not match cube band count {}",
                wavelengths_arr.shape[0], cube_arr.shape[2],
            )
        if not np.all(np.isfinite(wavelengths_arr)):
            log_and_raise("wavelengths must contain only finite values")

        black_level = float(black_level)
        white_level = float(white_level)
        if not np.isfinite(black_level) or not np.isfinite(white_level):
            log_and_raise("black_level and white_level must be finite")

        # The dtype decides how the values are interpreted, as for stored ROI spectra:
        # floating cubes hold a precalibrated signal (float32), integer cubes raw counts
        # (uint16). Floats are never truncated to integers.
        self.cube = as_roi_storage_array(cube_arr)
        self.wavelengths = wavelengths_arr
        self.meta = meta or {}

        _, height, n_bands = self.cube.shape
        self._frame = 0
        self._row = height // 2
        self._band = n_bands // 2

        self.black_level = black_level
        self.white_level = white_level
        self.y_unit = "DN"
        self.x_spatial_unit = "pixel"
        self.pixel_size_mm = 1.0
        self.x_spectral_unit = "nm"
        self.dark_ref = None
        self.white_ref = None
        self.preprocess_filter = "none"
        self.sg_window = 11
        self.sg_polyorder = 2
        self.preprocess_snv = False
        self.preprocess_msc = False
        self.preprocess_emsc = False
        self.preprocess_detrend = False
        self.detrend_order = 1
        #: Band selection of the preview (the last fit dialog's ranges).
        self.bands_keep: tuple[tuple[float, float], ...] = ()
        self.bands_exclude: tuple[tuple[float, float], ...] = ()
        self._image_cache = None

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return int(self.cube.shape[0])

    @property
    def height(self) -> int:
        return int(self.cube.shape[1])

    @property
    def n_bands(self) -> int:
        return int(self.cube.shape[2])

    @property
    def has_dataset(self) -> bool:
        return self.n_frames > 0 and self.height > 0 and self.n_bands > 0

    # ------------------------------------------------------------------
    # Setters — validate, invalidate cache, emit
    # ------------------------------------------------------------------

    @property
    def frame(self) -> int:
        return self._frame

    @property
    def row(self) -> int:
        return self._row

    @property
    def band(self) -> int:
        return self._band

    def set_frame(self, f: int) -> None:
        if not self.has_dataset:
            return
        f = int(np.clip(f, 0, self.n_frames - 1))
        if f != self._frame:
            self._frame = f
            self._image_cache = None
            logger.debug("frame -> {} (cache invalidated)", f)
            self.frame_changed.emit(f)

    def set_row(self, r: int) -> None:
        if not self.has_dataset:
            return
        r = int(np.clip(r, 0, self.height - 1))
        if r != self._row:
            self._row = r
            self.row_changed.emit(r)

    def set_band(self, b: int) -> None:
        if not self.has_dataset:
            return
        b = int(np.clip(b, 0, self.n_bands - 1))
        if b != self._band:
            self._band = b
            self.band_changed.emit(b)

    def set_y_unit(self, unit: str) -> None:
        if unit != self.y_unit:
            self.y_unit = unit
            self._image_cache = None
            logger.debug("y_unit -> {} (cache invalidated)", unit)
            self.units_changed.emit()

    def set_x_spatial_unit(self, unit: str) -> None:
        if unit != self.x_spatial_unit:
            self.x_spatial_unit = unit
            self.units_changed.emit()

    def set_x_spectral_unit(self, unit: str) -> None:
        if unit != self.x_spectral_unit:
            self.x_spectral_unit = unit
            self.units_changed.emit()

    def set_pixel_size_mm(self, mm: float) -> None:
        self.pixel_size_mm = float(mm)
        if self.x_spatial_unit == "mm":
            self.units_changed.emit()

    def set_calibration(self, *, black_level: float, white_level: float,
                        dark_ref: np.ndarray | None = None,
                        white_ref: np.ndarray | None = None) -> None:
        """Convert with these detector levels and references (one units_changed signal)."""
        if not np.isfinite(float(black_level)) or not np.isfinite(float(white_level)):
            log_and_raise("black_level and white_level must be finite")
        self.black_level = float(black_level)
        self.white_level = float(white_level)
        self.dark_ref = None if dark_ref is None else self._prepare_ref(dark_ref, "dark_ref")
        self.white_ref = None if white_ref is None else self._prepare_ref(white_ref, "white_ref")
        self._image_cache = None
        self.units_changed.emit()

    def set_dark_ref(self, ref: np.ndarray | None) -> None:
        if ref is not None:
            ref = self._prepare_ref(ref, "dark_ref")
        self.dark_ref = ref
        self._image_cache = None
        self.units_changed.emit()

    def set_white_ref(self, ref: np.ndarray | None) -> None:
        if ref is not None:
            ref = self._prepare_ref(ref, "white_ref")
        self.white_ref = ref
        self._image_cache = None
        self.units_changed.emit()

    def _validate_ref_shape(self, ref: np.ndarray, name: str) -> None:
        """Check that a (height, n_bands) reference matches the cube."""
        if ref.ndim != 2:
            log_and_raise("{} must be 2-D (height, n_bands), got shape {}", name, ref.shape)
        if ref.shape[1] != self.n_bands:
            log_and_raise("{} has {} bands but cube has {}", name, ref.shape[1], self.n_bands)
        if ref.shape[0] != self.height:
            logger.warning(
                "{} has {} rows but cube has {}; resampling reference to dataset height",
                name,
                ref.shape[0],
                self.height,
            )

    def _prepare_ref(self, ref: np.ndarray, name: str) -> np.ndarray:
        ref = np.asarray(ref, dtype=_WORKING_DTYPE)
        self._validate_ref_shape(ref, name)
        if ref.shape[0] == self.height:
            return ref
        if ref.shape[0] == 1:
            return np.repeat(ref, self.height, axis=0)
        sample_rows = np.linspace(0, ref.shape[0] - 1, num=self.height)
        row_indices = np.rint(sample_rows).astype(int)
        return ref[row_indices]

    # Preprocessing setters -----------------------------------------------

    def set_preprocess_filter(self, mode: str) -> None:
        if mode != self.preprocess_filter:
            self.preprocess_filter = mode
            self._image_cache = None
            logger.debug("preprocess_filter -> {} (cache invalidated)", mode)
            self.units_changed.emit()

    def set_sg_window(self, w: int) -> None:
        w = max(3, int(w) | 1)  # enforce odd, minimum 3
        if w != self.sg_window:
            self.sg_window = w
            if self.preprocess_filter != "none":
                self._image_cache = None
                self.units_changed.emit()

    def set_sg_polyorder(self, p: int) -> None:
        p = max(1, int(p))
        if p != self.sg_polyorder:
            self.sg_polyorder = p
            if self.preprocess_filter != "none":
                self._image_cache = None
                self.units_changed.emit()

    def set_preprocess_snv(self, enabled: bool) -> None:
        if enabled != self.preprocess_snv:
            self.preprocess_snv = enabled
            self._image_cache = None
            self.units_changed.emit()

    def set_preprocess_msc(self, enabled: bool) -> None:
        if enabled != self.preprocess_msc:
            self.preprocess_msc = enabled
            self._image_cache = None
            self.units_changed.emit()

    def set_preprocess_emsc(self, enabled: bool) -> None:
        if enabled != self.preprocess_emsc:
            self.preprocess_emsc = enabled
            self._image_cache = None
            self.units_changed.emit()

    def set_preprocess_detrend(self, enabled: bool) -> None:
        if enabled != self.preprocess_detrend:
            self.preprocess_detrend = enabled
            self._image_cache = None
            self.units_changed.emit()

    def set_detrend_order(self, order: int) -> None:
        order = max(1, int(order))
        if order != self.detrend_order:
            self.detrend_order = order
            if self.preprocess_detrend:
                self._image_cache = None
                self.units_changed.emit()

    # ------------------------------------------------------------------
    # Recipe — read/write preprocessing configuration as a value object
    # ------------------------------------------------------------------

    @property
    def recipe(self) -> PreprocessingRecipe:
        """Current preprocessing configuration as a serializable recipe.

        The polynomial order is clamped into the range the selected filter
        accepts (derivative order up to ``sg_window - 1``).
        """
        deriv = SG_FILTERS.get(self.preprocess_filter, 0)
        return PreprocessingRecipe(
            filter=self.preprocess_filter,
            sg_window=self.sg_window,
            sg_polyorder=min(max(self.sg_polyorder, deriv), self.sg_window - 1),
            snv=self.preprocess_snv,
            msc=self.preprocess_msc,
            emsc=self.preprocess_emsc,
            detrend=self.preprocess_detrend,
            detrend_order=self.detrend_order,
            bands_keep=self.bands_keep,
            bands_exclude=self.bands_exclude,
        )

    def apply_recipe(self, recipe: PreprocessingRecipe) -> None:
        """Apply all settings from *recipe* atomically.

        Sets every preprocessing field from the recipe, then invalidates the
        image cache and emits ``units_changed`` once.  Prefer this over calling
        individual setters when loading a saved recipe or copying state.
        """
        self.preprocess_filter = recipe.filter
        self.sg_window = recipe.sg_window
        self.sg_polyorder = recipe.sg_polyorder
        self.preprocess_snv = recipe.snv
        self.preprocess_msc = recipe.msc
        self.preprocess_emsc = recipe.emsc
        self.preprocess_detrend = recipe.detrend
        self.detrend_order = recipe.detrend_order
        self.bands_keep = recipe.bands_keep
        self.bands_exclude = recipe.bands_exclude
        self._image_cache = None
        self.units_changed.emit()

    # ------------------------------------------------------------------
    # Computed — no maths in the panels
    # ------------------------------------------------------------------

    @property
    def calibrated(self) -> bool:
        """Whether the cube holds a precalibrated signal (a floating-point cube).

        Precalibrated cubes are persisted with ``precalibrated_import``
        calibrations; integer cubes are raw detector counts.
        """
        return self.cube.dtype.kind == "f"

    @property
    def y_unit_effective(self) -> str:
        """Actual unit in use.

        Always returns the requested unit — conversion is always possible
        because the default black/white levels (128 / 3500) are used when
        no explicit reference files are loaded.  DN is only returned when
        explicitly requested.
        """
        return self.y_unit

    def _to_reflectance(self, raw: np.ndarray) -> np.ndarray:
        """Convert one ``(height, n_bands)`` frame to reflectance like model training does.

        The conversion matches :func:`enginir.project.transform_spectra_to_signal_domain`
        for the calibration the viewer persists with an ROI:

        1. Precalibrated (floating) cubes are used as they are.
        2. Raw counts with dark/white reference spectra: ``(raw - d) / (w - d)``.
        3. Raw counts with the scalar black/white levels.

        Converted counts are clipped to ``[1e-6, 1]``.
        """
        if self.calibrated:
            return raw.astype(_WORKING_DTYPE, copy=False)

        if self.dark_ref is not None and self.white_ref is not None:
            logger.debug("Reflectance: explicit dark/white refs")
            denom = self.white_ref - self.dark_ref
            small = np.abs(denom) < _EPS
            n_small = int(np.sum(small))
            if n_small:
                logger.warning(
                    "Reflectance: {} explicit reference denominator element(s) are < {}; clamping",
                    n_small,
                    _EPS,
                )
                signs = np.where(denom >= 0.0, 1.0, -1.0)
                denom = np.where(small, signs * _EPS, denom)
            return np.clip((raw - self.dark_ref) / denom, 1e-6, 1.0).astype(
                _WORKING_DTYPE, copy=False
            )

        # Default: standard EVK black / white levels
        logger.debug(
            "Reflectance: default levels (black={}, white={})", self.black_level, self.white_level
        )
        denom = self._guard_scalar_denom(
            self.white_level - self.black_level,
            "Reflectance: white_level - black_level ({}) is near zero; clamping to {}",
        )
        return np.clip((raw - self.black_level) / denom, 1e-6, 1.0).astype(
            _WORKING_DTYPE, copy=False
        )

    def _guard_scalar_denom(self, denom: float, warning_template: str) -> float:
        if abs(denom) < _EPS:
            logger.warning(warning_template, denom, _EPS)
            return _EPS if denom >= 0.0 else -_EPS
        return denom

    def _apply_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Apply the current recipe to one frame, fitting MSC/EMSC on that frame.

        Bands a Savitzky-Golay filter drops (``sg_edges="drop"``) and bands a
        band selection leaves out are returned as NaN (missing), so band
        indices stay those of the data file.
        """
        n_bands = img.shape[1]
        for _, transformer in self.recipe.build_steps(self.wavelengths,
                                                      signal_unit=self.y_unit):
            img = np.asarray(transformer.fit_transform(img), dtype=_WORKING_DTYPE)
        if img.shape[1] != n_bands:
            padded = np.full((img.shape[0], n_bands), np.nan, dtype=_WORKING_DTYPE)
            padded[:, self.recipe.output_band_indices(self.wavelengths)] = img
            img = padded
        return img

    def current_frame_image(self) -> np.ndarray:
        """Return (height, n_bands) float32 for the current frame, fully processed."""
        if not self.has_dataset:
            return np.zeros((0, 0), dtype=_WORKING_DTYPE)
        if self._image_cache is not None:
            return self._image_cache

        raw = self.cube[self._frame].astype(_WORKING_DTYPE, copy=False)

        # 1. Signal unit
        if self.y_unit == "DN":
            img = raw
        else:
            R = self._to_reflectance(raw)
            if self.y_unit == "Reflectance":
                img = R
            else:
                img = np.asarray(np.clip(R, 1e-6, None), dtype=_WORKING_DTYPE, copy=False)
                np.log10(img, out=img)
                img *= -1.0

        # 2. Preprocessing recipe (SG filter, scatter correction, detrend)
        img = self._apply_preprocessing(img)

        self._image_cache = np.asarray(img, dtype=_WORKING_DTYPE, copy=False)
        return self._image_cache

    def raw_band_image(self, band: int) -> np.ndarray:
        """Return one raw band view as ``(height, n_frames)`` float32."""
        if not self.has_dataset:
            return np.zeros((0, 0), dtype=_WORKING_DTYPE)
        b = int(np.clip(band, 0, self.n_bands - 1))
        return self.cube[:, :, b].T.astype(_WORKING_DTYPE, copy=False)

    def raw_false_colour_rgb(
        self,
        *,
        red_nm: float = 1650.0,
        green_nm: float = 1300.0,
        blue_nm: float = 1050.0,
    ) -> np.ndarray:
        """Return a raw false-colour preview image as float RGB."""
        if not self.has_dataset:
            return np.zeros((0, 0, 3), dtype=np.float32)

        band_indices = [
            int(np.argmin(np.abs(self.wavelengths - red_nm))),
            int(np.argmin(np.abs(self.wavelengths - green_nm))),
            int(np.argmin(np.abs(self.wavelengths - blue_nm))),
        ]
        raw_rgb = (
            np.take(self.cube, band_indices, axis=2)
            .transpose(1, 0, 2)
            .astype(np.float32, copy=False)
        )
        percentiles = np.percentile(raw_rgb, [2, 98], axis=(0, 1))
        lo = percentiles[0]
        scale = np.maximum(percentiles[1] - percentiles[0], 1.0)
        return np.clip((raw_rgb - lo) / scale, 0.0, 1.0).astype(np.float32, copy=False)

    def raw_roi_spectra(self, f0: int, f1: int, r0: int, r1: int) -> np.ndarray:
        """Return raw ROI spectra as ``(n_pixels, n_bands)`` in the cube's dtype.

        That is uint16 for raw counts and float32 for precalibrated cubes, the
        storage dtypes of ROI spectra.
        """
        if not self.has_dataset:
            return np.zeros((0, 0), dtype=np.uint16)
        left, right, top, bottom = self._normalized_roi_bounds(f0, f1, r0, r1)
        return self.cube[left : right + 1, top : bottom + 1, :].reshape(-1, self.n_bands)

    def roi_reference_spectra(
        self,
        ref: np.ndarray,
        f0: int,
        f1: int,
        r0: int,
        r1: int,
    ) -> np.ndarray:
        """Return cross-track reference rows aligned to ``raw_roi_spectra`` order."""
        if not self.has_dataset:
            return np.zeros((0, 0), dtype=_WORKING_DTYPE)
        left, right, top, bottom = self._normalized_roi_bounds(f0, f1, r0, r1)
        n_frames = right - left + 1
        ref_arr = np.asarray(ref, dtype=_WORKING_DTYPE)
        if ref_arr.ndim != 2 or ref_arr.shape != (self.height, self.n_bands):
            log_and_raise(
                "ROI reference spectra require ref shape ({}, {}), got {}",
                self.height, self.n_bands, ref_arr.shape,
            )
        row_refs = ref_arr[top : bottom + 1]
        return np.tile(row_refs, (n_frames, 1)).astype(_WORKING_DTYPE, copy=False)

    def roi_dark_white_reference_spectra(
        self,
        f0: int,
        f1: int,
        r0: int,
        r1: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ROI-local dark/white refs aligned to stored raw ROI spectra."""
        if self.dark_ref is None or self.white_ref is None:
            return None
        return (
            self.roi_reference_spectra(self.dark_ref, f0, f1, r0, r1),
            self.roi_reference_spectra(self.white_ref, f0, f1, r0, r1),
        )

    def _normalized_roi_bounds(
        self,
        f0: int,
        f1: int,
        r0: int,
        r1: int,
    ) -> tuple[int, int, int, int]:
        f0 = int(np.clip(f0, 0, self.n_frames - 1))
        f1 = int(np.clip(f1, 0, self.n_frames - 1))
        r0 = int(np.clip(r0, 0, self.height - 1))
        r1 = int(np.clip(r1, 0, self.height - 1))
        left, right = sorted((f0, f1))
        top, bottom = sorted((r0, r1))
        return left, right, top, bottom

    def get_spectrum(self, row: int) -> np.ndarray:
        """Return (n_bands,) spectrum at *row* in the current frame."""
        if not self.has_dataset:
            return np.zeros((0,), dtype=_WORKING_DTYPE)
        r = int(np.clip(row, 0, self.height - 1))
        return self.current_frame_image()[r]

    def get_spatial_profile(self, band: int) -> np.ndarray:
        """Return (height,) spatial profile at *band* in the current frame."""
        if not self.has_dataset:
            return np.zeros((0,), dtype=_WORKING_DTYPE)
        b = int(np.clip(band, 0, self.n_bands - 1))
        return self.current_frame_image()[:, b]

    def spatial_x_axis(self) -> np.ndarray:
        if not self.has_dataset:
            return np.zeros((0,), dtype=_WORKING_DTYPE)
        rows = np.arange(self.height, dtype=_WORKING_DTYPE)
        if self.x_spatial_unit == "mm":
            return rows * self.pixel_size_mm
        return rows

    def spectral_x_axis(self) -> np.ndarray:
        if not self.has_dataset:
            return np.zeros((0,), dtype=_WORKING_DTYPE)
        if self.x_spectral_unit == "nm":
            return self.wavelengths.copy()
        return np.arange(self.n_bands, dtype=_WORKING_DTYPE)
