"""The project workflow in a few calls: datasets, ROIs, targets, models, predictions.

:class:`Project` creates the bookkeeping objects (calibrations, sample sets,
signal domains, stored recipes, provenance) itself and writes everything
through the validated project store. Its rules are the "Project Facade Rules"
of ``contracts/headless-workflow-contract_v3.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from ._checks import log_and_raise
from ._logging import get_logger
from .io import file_format, load_spectra
from .io import project_store as store
from .modeling import build_sample_set, fit_model_run
from .prediction import (
    PredictionResult,
    build_dataset_ref_from_spectral_data,
    predict_with_model_run,
    preflight_prediction,
)
from .preprocessing import PreprocessingRecipe
from .project import (
    CalibrationSpec,
    DatasetRef,
    ModelRun,
    PredictionRun,
    SampleSet,
    SignalDomainContract,
    TargetAssignment,
    TargetDefinition,
    TrainingROI,
    hydrate_project_store,
    validate_project_relationships,
)
from .project._ids import new_id, now_iso

logger = get_logger(__name__)

__all__ = ["DEFAULT_BLACK_LEVEL", "DEFAULT_WHITE_LEVEL", "Dataset", "Model", "Prediction",
           "Project", "make_calibration", "make_roi", "resolve_calibration", "resolve_dataset",
           "resolve_prediction_calibration"]

#: Dark and white detector levels of EVK cameras, used when a dataset is
#: calibrated to reflectance or absorbance without reference spectra.
DEFAULT_BLACK_LEVEL = 128
DEFAULT_WHITE_LEVEL = 3500

# Loaders that can read a frame range or only the header instead of the whole cube.
_FRAME_FORMATS = ("eiff", "hdf5")


class Project:
    """A project file and the workflow on it.

    Create one with :meth:`create` or :meth:`open`. ROIs, targets and models
    are addressed by name, datasets by file path; every object also keeps a
    stable id, which links the objects in the file.

    Examples
    --------
    >>> project = Project.create("tablets.h5")                       # doctest: +SKIP
    >>> scan = project.add_dataset("scan.eiff", dark="dark.eiff", white="white.eiff")
    >>> project.add_target("API content", unit="% w/w")
    >>> scan.add_roi("tablet 1", frames=(0, 4), rows=(0, 9), values={"API content": 1.0})
    >>> model = project.fit("API content", recipe=PreprocessingRecipe(snv=True),
    ...                     n_components=2)
    >>> model.metrics["rmsecv"]
    >>> model.predict("new_scan.eiff").map
    """

    def __init__(self, path: str | os.PathLike[str], *,
                 store_contents: dict[str, Any] | None = None) -> None:
        self.path = os.path.abspath(os.fspath(path))
        self._reload(store_contents)

    @classmethod
    def create(cls, path: str | os.PathLike[str]) -> Project:
        """Create an empty project file (an existing file is not overwritten)."""
        store.create_project(path)
        return cls(path)

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> Project:
        """Open an existing project file and validate it completely."""
        return cls(path)

    def __repr__(self) -> str:
        ws = self.workspace
        return (f"Project({self.path!r}: {len(ws.datasets)} datasets, {len(ws.rois)} ROIs, "
                f"{len(ws.targets)} targets, {len(ws.model_runs)} models, "
                f"{len(ws.predictions)} predictions)")

    # -- reading ---------------------------------------------------------------

    @property
    def datasets(self) -> dict[str, Dataset]:
        """Datasets by absolute file path."""
        return {ref.source_path: Dataset(self, ref) for ref in self.workspace.datasets.values()}

    @property
    def rois(self) -> dict[str, TrainingROI]:
        """ROIs by name."""
        return {roi.name: roi for roi in self.workspace.rois.values()}

    @property
    def targets(self) -> dict[str, TargetDefinition]:
        """Targets by name."""
        return {target.name: target for target in self.workspace.targets.values()}

    @property
    def models(self) -> dict[str, Model]:
        """Models by name."""
        return {self._model_name(run): Model(self, run) for run in self.workspace.model_runs.values()}

    @property
    def predictions(self) -> dict[str, Prediction]:
        """Predictions by id."""
        return {run_id: Prediction(self, run) for run_id, run in self.workspace.predictions.items()}

    def roi(self, name: str) -> TrainingROI:
        """Return the ROI called ``name``."""
        return _one_named(self.workspace.rois.values(), name, "ROI")

    def target(self, name: str) -> TargetDefinition:
        """Return the target called ``name``."""
        return _one_named(self.workspace.targets.values(), name, "target")

    def model(self, name: str) -> Model:
        """Return the model called ``name``."""
        runs = [run for run in self.workspace.model_runs.values()
                if self._model_name(run) == name]
        return Model(self, _only(runs, name, "model", id_field="model_run_id"))

    def dataset(self, path: str | os.PathLike[str]) -> Dataset:
        """Return the dataset registered for the file ``path``."""
        ref = self._dataset_ref(os.path.abspath(os.fspath(path)))
        if ref is None:
            log_and_raise("No dataset is registered for {}", path, error=KeyError)
        return Dataset(self, ref)

    # -- writing ---------------------------------------------------------------

    def add_dataset(
        self,
        path: str | os.PathLike[str],
        *,
        dark: str | os.PathLike[str] | np.ndarray | None = None,
        white: str | os.PathLike[str] | np.ndarray | None = None,
        unit: str = "Absorbance",
    ) -> Dataset:
        """Register a data file with its calibration.

        Parameters
        ----------
        path:
            Spectra or image file in any format :func:`enginir.io.load_spectra`
            reads. A file that is already registered is checked against its
            registration and reused.
        dark, white:
            Reference spectra: files or arrays, ``(height, n_bands)``; cubes
            are averaged over frames. Without them, raw counts are calibrated
            with the file's ``black_level`` and ``white_level`` metadata, or
            :data:`DEFAULT_BLACK_LEVEL` and :data:`DEFAULT_WHITE_LEVEL`.
        unit:
            Signal to work with: ``"Absorbance"``, ``"Reflectance"`` or
            ``"DN"`` (raw counts). Floating-point files are taken as already
            calibrated in this unit.

        Returns
        -------
        Dataset
        """
        dataset, calibration, new_dataset, new_calibration = self._resolve_dataset(
            path, dark=dark, white=white, unit=unit)
        self._write_dataset(dataset, calibration, new_dataset, new_calibration)
        return Dataset(self, self.workspace.datasets[dataset.dataset_id])

    def add_target(self, name: str, *, unit: str, value_type: str = "float",
                   missing_value_policy: str = "allowed") -> TargetDefinition:
        """Define a property to model, e.g. ``add_target("API content", unit="% w/w")``."""
        self._require_new_name(self.workspace.targets.values(), name, "target")
        target = TargetDefinition(target_id=new_id(), name=name, unit=unit,
                                  value_type=value_type,
                                  missing_value_policy=missing_value_policy,
                                  created_at=now_iso())
        self.save_target(target)
        return target

    def set_value(self, roi: str, target: str, value: float) -> None:
        """Set (or replace) the reference value of ``target`` for ``roi``."""
        roi_ref, target_ref = self.roi(roi), self.target(target)
        assignment = TargetAssignment(roi_id=roi_ref.roi_id, target_id=target_ref.target_id,
                                      value=value, assigned_at=now_iso())
        self.save_assignment(assignment)

    def fit(
        self,
        target: str,
        *,
        recipe: PreprocessingRecipe | Mapping[str, Any] | None = None,
        model: str = "PLS",
        name: str | None = None,
        cv: int | str = "auto",
        **hyperparameters: Any,
    ) -> Model:
        """Fit and cross-validate a model of ``target`` on every ROI with a value for it.

        Parameters
        ----------
        target:
            Target name.
        recipe:
            Preprocessing (default: none).
        model:
            ``"PLS"`` or ``"PCR"``.
        name:
            Model name (default: e.g. ``"PLS 2 LV, SNV"``).
        cv:
            Cross-validation over ROIs: ``"auto"``, a number of folds or
            ``"logo"`` (leave one ROI out).
        **hyperparameters:
            Estimator parameters, e.g. ``n_components=2``.

        Returns
        -------
        Model
        """
        ws = self.workspace
        target_ref = self.target(target)
        recipe_obj = _as_recipe(recipe)
        recipe_id, recipe_dict = _recipe_id(recipe_obj), recipe_obj.to_dict()
        if name is not None:
            self._require_new_name(ws.model_runs.values(), name, "model", key=self._model_name)
        recipes = {**ws.recipes, recipe_id: recipe_dict}

        built = build_sample_set(project_path=self.path, rois=ws.rois,
                                 target_assignments=ws.target_assignments,
                                 target_id=target_ref.target_id,
                                 preprocessing_recipe_id=recipe_id)
        sample_set = _existing(ws.sample_sets.values(), built.sample_set, _same_sample_set)
        domain = self._training_signal_domain(sample_set)
        domain = _existing(ws.signal_domains.values(), domain, _same_signal_domain)
        result = fit_model_run(
            project_path=self.path, sample_set=sample_set, datasets=ws.datasets,
            calibrations=ws.calibrations, rois=ws.rois,
            target_assignments=ws.target_assignments, recipes=recipes,
            signal_domains={**ws.signal_domains, domain.signal_domain_id: domain},
            signal_domain_id=domain.signal_domain_id, model_type=model,
            hyperparameters=hyperparameters, cv=cv, target_name=target_ref.name)
        run = replace(result.model_run,
                      name=name or self.default_model_name(result.model_run, recipe_obj))
        validate_project_relationships(
            datasets=ws.datasets, calibrations=ws.calibrations, rois=ws.rois,
            targets=ws.targets, target_assignments=ws.target_assignments.values(),
            recipes=recipes, signal_domains={domain.signal_domain_id: domain},
            sample_sets={sample_set.sample_set_id: sample_set},
            model_runs={run.model_run_id: run}, predictions={},
        )

        # Everything is computed and validated; write in dependency order.
        if recipe_id not in ws.recipes:
            self.save_recipe(recipe_id, recipe_obj)
        if sample_set.sample_set_id not in ws.sample_sets:
            self.save_sample_set(sample_set)
        if domain.signal_domain_id not in ws.signal_domains:
            self.save_signal_domain(domain)
        self.save_model_run(run)
        logger.info("Fitted model '{}': {}", run.name, _summary(run.metrics))
        return Model(self, self.workspace.model_runs[run.model_run_id])

    # -- typed objects (advanced) ----------------------------------------------
    #
    # Each method validates one complete object against the project, writes it
    # and rereads the project. The viewer stores everything through these.

    def save_dataset(self, dataset: DatasetRef) -> None:
        """Store a dataset reference."""
        store.save_dataset_ref(self.path, dataset.dataset_id, dataset.to_dict())
        self.workspace.datasets[dataset.dataset_id] = dataset

    def save_calibration(self, calibration: CalibrationSpec) -> None:
        """Store a calibration of a stored dataset and make it the dataset's active one."""
        store.save_calibration_spec(self.path, calibration.dataset_id,
                                    calibration.calibration_id, calibration.to_dict())
        self.workspace.calibrations[calibration.calibration_id] = calibration
        self.workspace.active_calibration_ids[calibration.dataset_id] = calibration.calibration_id

    def save_target(self, target: TargetDefinition) -> None:
        """Store a target definition."""
        store.save_target_definition(self.path, target.target_id, target.to_dict())
        self.workspace.targets[target.target_id] = target

    def save_assignment(self, assignment: TargetAssignment) -> None:
        """Store (or replace) the reference value of one target for one ROI."""
        self._validate(target_assignments=[assignment])
        store.save_target_assignment(self.path, assignment.target_id, assignment.roi_id,
                                     assignment.to_dict())
        self.workspace.target_assignments[(assignment.roi_id, assignment.target_id)] = assignment

    def save_roi(self, roi: TrainingROI, spectra: np.ndarray, *,
                 dark_ref_spectra: np.ndarray | None = None,
                 white_ref_spectra: np.ndarray | None = None) -> None:
        """Store an ROI of a stored dataset and calibration with its spectra."""
        self._validate(rois={roi.roi_id: roi}, target_assignments=[])
        store.save_training_roi(self.path, roi.roi_id, roi.to_dict(), spectra,
                                dark_ref_spectra=dark_ref_spectra,
                                white_ref_spectra=white_ref_spectra)
        self.workspace.rois[roi.roi_id] = roi

    def save_recipe(self, recipe_id: str, recipe: PreprocessingRecipe | Mapping[str, Any]) -> None:
        """Store a preprocessing recipe under ``recipe_id``."""
        recipe_dict = _as_recipe(recipe).to_dict()
        store.save_preprocessing_recipe(self.path, recipe_id, recipe_dict)
        self.workspace.recipes[recipe_id] = recipe_dict

    def save_signal_domain(self, signal_domain: SignalDomainContract) -> None:
        """Store a signal-domain contract."""
        store.save_signal_domain_contract(self.path, signal_domain.signal_domain_id,
                                          signal_domain.to_dict())
        self.workspace.signal_domains[signal_domain.signal_domain_id] = signal_domain

    def save_sample_set(self, sample_set: SampleSet) -> None:
        """Store a sample set; all its ROIs must exist."""
        ws = self.workspace
        # Strict for the new sample set, stale sample sets already stored are allowed.
        validate_project_relationships(
            datasets=ws.datasets, calibrations=ws.calibrations, rois=ws.rois,
            targets=ws.targets, target_assignments=ws.target_assignments.values(),
            recipes=ws.recipes, signal_domains=ws.signal_domains,
            sample_sets={sample_set.sample_set_id: sample_set}, model_runs={}, predictions={})
        self._validate(sample_sets={sample_set.sample_set_id: sample_set})
        store.save_sample_set(self.path, sample_set.sample_set_id, sample_set.to_dict())
        self.workspace.sample_sets[sample_set.sample_set_id] = sample_set

    def save_model_run(self, model_run: ModelRun) -> None:
        """Store a model run of a sample set whose ROIs all still exist."""
        self.require_current_sample_set(model_run.sample_set_id, action="store the model run")
        self._validate(model_runs={model_run.model_run_id: model_run})
        store.save_model_run(self.path, model_run.model_run_id, model_run.to_dict())
        # Artifact arrays stay on disk; they are read on demand.
        self.workspace.model_runs[model_run.model_run_id] = replace(model_run, artifact_refs={})

    def save_prediction_run(self, prediction: PredictionRun) -> None:
        """Store a complete prediction run."""
        self._validate(predictions={prediction.prediction_run_id: prediction})
        store.save_prediction_run(self.path, prediction.prediction_run_id, prediction.to_dict())
        self.workspace.predictions[prediction.prediction_run_id] = replace(
            prediction, artifact_refs={key: value for key, value in prediction.artifact_refs.items()
                                       if key not in ("y_pred", "spatial_map")})

    def remove_roi(self, roi_id: str) -> list[str]:
        """Delete an ROI and its reference values; return the sample sets still listing it.

        Model runs of those sample sets stay usable for prediction, but the
        sample sets can no longer be refitted.
        """
        affected = [sample_set_id for sample_set_id, sample_set in self.workspace.sample_sets.items()
                    if roi_id in sample_set.roi_ids]
        store.remove_roi_from_project(self.path, roi_id)
        ws = self.workspace
        ws.rois.pop(roi_id, None)
        for key in [key for key in ws.target_assignments if key[0] == roi_id]:
            del ws.target_assignments[key]
        if affected:
            logger.warning("Deleted ROI '{}' is still listed by sample set(s) {}; their model "
                           "runs remain usable for prediction, but they cannot be refitted",
                           roi_id, affected)
        return affected

    def missing_rois(self, sample_set_id: str) -> list[str]:
        """Ids of deleted ROIs a sample set still lists."""
        sample_set = self.workspace.sample_sets.get(sample_set_id)
        return [] if sample_set is None else [roi_id for roi_id in sample_set.roi_ids
                                              if roi_id not in self.workspace.rois]

    def require_current_sample_set(self, sample_set_id: str, *, action: str) -> None:
        """Refuse ``action`` on a sample set that lists deleted ROIs."""
        missing = self.missing_rois(sample_set_id)
        if missing:
            log_and_raise("Cannot {}: sample set '{}' lists deleted ROI(s) {}; its model runs "
                          "remain usable for prediction, but it cannot be altered or refit "
                          "until recreated", action, sample_set_id, missing, depth=2)

    def _validate(self, **objects: Any) -> None:
        """Check the references of ``objects`` together with everything stored."""
        ws = self.workspace
        graph: dict[str, Any] = dict(
            datasets=ws.datasets, calibrations=ws.calibrations, rois=ws.rois,
            targets=ws.targets, target_assignments=ws.target_assignments.values(),
            active_calibration_ids=ws.active_calibration_ids, recipes=ws.recipes,
            signal_domains=ws.signal_domains, sample_sets=ws.sample_sets,
            model_runs=ws.model_runs, predictions=ws.predictions,
            allow_stale_sample_set_rois=True)
        for key, value in objects.items():
            if key in ("target_assignments",):
                graph[key] = list(ws.target_assignments.values()) + list(value)
            elif isinstance(value, dict) and key != "active_calibration_ids":
                graph[key] = {**graph[key], **value}
            else:
                graph[key] = value
        validate_project_relationships(**graph)

    # -- internals -------------------------------------------------------------

    def _reload(self, store_contents: dict[str, Any] | None = None) -> None:
        if store_contents is None:
            store_contents = store.load_project_store(self.path)
        self.workspace = hydrate_project_store(store_contents, allow_stale_sample_set_rois=True)

    def _dataset_ref(self, abs_path: str) -> DatasetRef | None:
        return next((ref for ref in self.workspace.datasets.values()
                     if _same_path(ref.source_path, abs_path)), None)

    def _resolve_dataset(self, path, *, dark, white, unit):
        """Return ``(dataset, calibration, dataset_is_new, calibration_is_new)``, writing nothing."""
        ws = self.workspace
        data = _read_header(_native_path(path))
        dataset, new_dataset = resolve_dataset(ws.datasets, path, data)
        candidate = make_calibration(
            dataset, unit=unit, precalibrated=np.dtype(data.X.dtype).kind == "f",
            dark=None if dark is None else _reference(dark, dataset.cube_shape[1:], "dark"),
            white=None if white is None else _reference(white, dataset.cube_shape[1:], "white"),
            dark_source=_source(dark), white_source=_source(white),
            black_level=data.meta.get("black_level"), white_level=data.meta.get("white_level"))
        calibration, new_calibration = resolve_calibration(dataset, ws.calibrations, candidate)
        return dataset, calibration, new_dataset, new_calibration

    def _write_dataset(self, dataset, calibration, new_dataset, new_calibration) -> None:
        if new_dataset:
            self.save_dataset(dataset)
        if new_calibration or (self.workspace.active_calibration_ids.get(dataset.dataset_id)
                               != calibration.calibration_id):
            self.save_calibration(calibration)

    def _training_signal_domain(self, sample_set: SampleSet) -> SignalDomainContract:
        """The one signal domain of all training ROIs, refusing ROIs that need different ones."""
        ws = self.workspace
        domains: list[SignalDomainContract] = []
        for roi_id in sample_set.roi_ids:
            roi = ws.rois[roi_id]
            calibration = ws.calibrations[roi.calibration_id]
            domain = SignalDomainContract.for_training_data(
                dataset=ws.datasets[roi.dataset_id], calibration=calibration,
                required_signal_unit=calibration.output_signal_unit)
            if domains and not _same_signal_domain(domains[0], domain):
                log_and_raise("ROI '{}' was recorded with a different signal (format, unit, "
                              "calibration or wavelengths) than ROI '{}'; one model needs one "
                              "signal domain", roi.name, ws.rois[sample_set.roi_ids[0]].name)
            domains.append(domain)
        return domains[0]

    def _model_name(self, run: ModelRun) -> str:
        if run.name:
            return run.name
        n = run.hyperparameters.get("n_components")
        return f"{run.model_type} {n} LV" if n is not None else run.model_type

    def default_model_name(self, model_run: ModelRun,
                           recipe: PreprocessingRecipe | None = None) -> str:
        """A free name like ``"PLS 2 LV, SNV"`` for a model run of a stored sample set."""
        if recipe is None:
            ws = self.workspace
            recipe = PreprocessingRecipe.from_dict(
                ws.recipes[ws.sample_sets[model_run.sample_set_id].preprocessing_recipe_id])
        n = model_run.hyperparameters.get("n_components")
        base = f"{model_run.model_type}{f' {n} LV' if n is not None else ''}, {_recipe_label(recipe)}"
        taken = {self._model_name(run) for run in self.workspace.model_runs.values()}
        name, index = base, 2
        while name in taken:
            name, index = f"{base} ({index})", index + 1
        return name

    @staticmethod
    def _require_new_name(objects, name: str, kind: str, key=lambda obj: obj.name) -> None:
        if not isinstance(name, str) or not name.strip():
            log_and_raise("A {} name must be a non-empty string, got {!r}", kind, name, depth=2)
        if any(key(obj) == name for obj in objects):
            log_and_raise("A {} called '{}' already exists", kind, name, depth=2)


class Dataset:
    """A registered data file with its active calibration."""

    def __init__(self, project: Project, ref: DatasetRef) -> None:
        self.project = project
        self.ref = ref

    def __repr__(self) -> str:
        return f"Dataset({self.path!r}, shape={self.ref.cube_shape}, {self.calibration.mode})"

    @property
    def path(self) -> str:
        return self.ref.source_path

    @property
    def calibration(self) -> CalibrationSpec:
        """The calibration new ROIs of this dataset are recorded with."""
        ws = self.project.workspace
        return ws.calibrations[ws.active_calibration_ids[self.ref.dataset_id]]

    @property
    def rois(self) -> dict[str, TrainingROI]:
        """The ROIs of this dataset by name."""
        return {roi.name: roi for roi in self.project.workspace.rois.values()
                if roi.dataset_id == self.ref.dataset_id}

    def add_roi(self, name: str, *, frames: tuple[int, int], rows: tuple[int, int],
                values: Mapping[str, float] | None = None) -> TrainingROI:
        """Store the pixels ``frames`` x ``rows`` as an ROI with its reference values.

        Parameters
        ----------
        name:
            ROI name, unique in the project.
        frames, rows:
            Inclusive ``(first, last)`` index ranges.
        values:
            Reference values ``{target name: value}``.

        Returns
        -------
        TrainingROI
        """
        project, ws = self.project, self.project.workspace
        project._require_new_name(ws.rois.values(), name, "ROI")
        targets = {target_name: project.target(target_name) for target_name in values or {}}
        n_frames = self.ref.cube_shape[0]
        first, last = _index_range(frames, n_frames, "frames")
        roi, spectra, refs = make_roi(self.ref, self.calibration, name=name,
                                      frames=(first, last), rows=rows,
                                      pixels=_read_frames(self.path, first, last))
        created_at = roi.created_at
        assignments = [TargetAssignment(roi_id=roi.roi_id, target_id=target.target_id,
                                        value=float(values[target_name]), assigned_at=created_at)
                       for target_name, target in targets.items()] if values else []
        validate_project_relationships(
            datasets=ws.datasets, calibrations=ws.calibrations, rois={roi.roi_id: roi},
            targets=ws.targets, target_assignments=assignments, recipes={}, signal_domains={},
            sample_sets={}, model_runs={}, predictions={})

        project.save_roi(roi, spectra, dark_ref_spectra=refs[0] if refs else None,
                         white_ref_spectra=refs[1] if refs else None)
        for assignment in assignments:
            project.save_assignment(assignment)
        return roi


class Model:
    """A fitted, cross-validated model."""

    def __init__(self, project: Project, ref: ModelRun) -> None:
        self.project = project
        self.ref = ref

    def __repr__(self) -> str:
        return f"Model({self.name!r}, {_summary(self.metrics)})"

    @property
    def name(self) -> str:
        return self.project._model_name(self.ref)

    @property
    def metrics(self) -> dict[str, Any]:
        """Figures of merit: ROI-level ``rmsecv``, ``r2``, ``bias``, ``rpd``, ``rpiq``,
        and their pixel-level ``pixel_*`` variants."""
        return dict(self.ref.metrics)

    @property
    def target(self) -> TargetDefinition:
        ws = self.project.workspace
        return ws.targets[ws.sample_sets[self.ref.sample_set_id].target_id]

    @property
    def recipe(self) -> PreprocessingRecipe:
        ws = self.project.workspace
        return PreprocessingRecipe.from_dict(
            ws.recipes[ws.sample_sets[self.ref.sample_set_id].preprocessing_recipe_id])

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Cross-validation results and model diagnostics.

        ``roi_ids``, ``y_true_roi`` and ``y_pred_roi`` (measured and
        cross-validated values per ROI), and the stored model diagnostics
        such as coefficients, loadings and scores.
        """
        refs = store.load_model_run_artifact_refs(self.project.path, self.ref.model_run_id,
                                                  keys=("cv_outputs", "diagnostics"))
        return {**refs.get("diagnostics", {}), **refs.get("cv_outputs", {})}

    def predict(self, path: str | os.PathLike[str], *,
                dark: str | os.PathLike[str] | np.ndarray | None = None,
                white: str | os.PathLike[str] | np.ndarray | None = None) -> Prediction:
        """Predict the target for every spectrum of a data file and store the prediction.

        The file is registered as with :meth:`Project.add_dataset`, in the
        model's signal unit; compatibility with the model is checked before
        anything is read or written.
        """
        project, ws = self.project, self.project.workspace
        domain = ws.signal_domains[self.ref.signal_domain_id]
        recipe_id = ws.sample_sets[self.ref.sample_set_id].preprocessing_recipe_id
        recipe = PreprocessingRecipe.from_dict(ws.recipes[recipe_id])
        dataset, calibration, new_dataset, new_calibration = project._resolve_dataset(
            path, dark=dark, white=white, unit=domain.required_signal_unit)
        common: dict[str, Any] = dict(project_path=project.path, model_run=self.ref, signal_domain=domain,
                      recipe=recipe, dataset=dataset, calibration=calibration,
                      model_registry={})
        preflight_prediction(**common)

        project._write_dataset(dataset, calibration, new_dataset, new_calibration)
        result: PredictionResult = predict_with_model_run(
            **common, data_path=dataset.source_path, preprocessing_recipe_id=recipe_id,
            persist_artifacts=True, return_arrays=False)
        project._reload()
        return Prediction(project, project.workspace.predictions[
            result.prediction_run.prediction_run_id])


class Prediction:
    """Stored predictions of one model for one dataset."""

    def __init__(self, project: Project, ref: PredictionRun) -> None:
        self.project = project
        self.ref = ref

    def __repr__(self) -> str:
        return f"Prediction({self.ref.prediction_run_id!r}, dataset={self.dataset.path!r})"

    @property
    def model(self) -> Model:
        return Model(self.project, self.project.workspace.model_runs[self.ref.model_run_id])

    @property
    def dataset(self) -> Dataset:
        return Dataset(self.project, self.project.workspace.datasets[self.ref.dataset_id])

    @property
    def y_pred(self) -> np.ndarray:
        """Predictions, one per spectrum in file order."""
        return self._artifact("y_pred")

    @property
    def map(self) -> np.ndarray:
        """Predictions as an image ``(frames, height)`` (with an output axis for several targets)."""
        return self._artifact("spatial_map")

    def _artifact(self, key: str) -> np.ndarray:
        refs = store.load_prediction_run_artifact_refs(self.project.path,
                                                       self.ref.prediction_run_id, keys=(key,))
        return np.asarray(refs[key])


# -- helpers -------------------------------------------------------------------


def _read_header(path: str):
    """Load only the metadata of image files that allow it, otherwise the whole file."""
    if file_format(path) in _FRAME_FORMATS:
        return load_spectra(path, header_only=True)
    return load_spectra(path)


def _read_frames(path: str, first: int, last: int) -> np.ndarray:
    """Frames ``first..last`` as ``(frames, height, n_bands)`` in the file's dtype."""
    if file_format(path) in _FRAME_FORMATS:
        X = load_spectra(path, frames=slice(first, last + 1)).X
    else:
        X = load_spectra(path).X[first:last + 1]
    X = np.asarray(X)
    return X[:, np.newaxis, :] if X.ndim == 2 else X


def resolve_dataset(
    datasets: Mapping[str, DatasetRef],
    path: str | os.PathLike[str],
    data,
    *,
    dataset_id: str | None = None,
    created_at: str | None = None,
) -> tuple[DatasetRef, bool]:
    """Return the registered dataset of ``path``, or a new one; and whether it is new.

    ``data`` is the file's :class:`~enginir.io.SpectralData` (a header-only
    load is enough). A registered file must still have its registered format,
    format, shape and (if the file states one) modality; wavelength values
    are not compared. ``dataset_id`` selects or names the dataset explicitly.
    """
    abs_path = _native_path(path)
    registered = None
    if dataset_id is not None and dataset_id in datasets:
        registered = datasets[dataset_id]
        if not _same_path(registered.source_path, abs_path):
            log_and_raise("Dataset '{}' is registered for {}, not {}", dataset_id,
                          registered.source_path, abs_path)
    else:
        registered = next((ref for ref in datasets.values()
                           if _same_path(ref.source_path, abs_path)), None)
    if registered is None:
        return build_dataset_ref_from_spectral_data(
            data_path=abs_path, spectral_data=data, dataset_id=dataset_id,
            import_timestamp=created_at), True
    # Wavelength values are metadata, not a compatibility gate (see the
    # dataset-calibration contract); the modality is compared when the file states it.
    checks = [("format", file_format(abs_path), registered.source_format),
              ("shape", tuple(data.cube_shape), registered.cube_shape)]
    if data.meta.get("source_modality"):
        checks.append(("modality", data.meta["source_modality"], registered.source_modality))
    for label, now, then in checks:
        if now != then:
            log_and_raise("{} has {} {!r}, but it was registered with {!r}", abs_path, label,
                          now, then)
    return registered, False


def make_calibration(
    dataset: DatasetRef,
    *,
    unit: str,
    precalibrated: bool,
    dark: np.ndarray | None = None,
    white: np.ndarray | None = None,
    dark_source: str | None = None,
    white_source: str | None = None,
    black_level: float | None = None,
    white_level: float | None = None,
    created_at: str | None = None,
) -> CalibrationSpec:
    """Build the calibration of ``dataset`` by the contract rules.

    ``unit="DN"`` keeps raw counts; precalibrated (floating-point) data are
    imported as they are; otherwise counts are calibrated with the dark and
    white reference spectra ``(height, n_bands)`` or, without them, with
    ``black_level``/``white_level`` (default :data:`DEFAULT_BLACK_LEVEL` and
    :data:`DEFAULT_WHITE_LEVEL`).
    """
    if (dark is None) != (white is None):
        log_and_raise("Give both dark and white references, or neither")
    fields: dict[str, Any] = dict(dark_ref_source=None, white_ref_source=None,
                                  dark_ref_data=None, white_ref_data=None,
                                  black_level=None, white_level=None)
    if unit == "DN":
        mode = "raw_dn"
    elif precalibrated:
        mode = "precalibrated_import"
    else:
        mode = "dark_white_reference"
        if dark is not None and white is not None:
            fields.update(dark_ref_data=np.asarray(dark, dtype=np.float32),
                          white_ref_data=np.asarray(white, dtype=np.float32),
                          dark_ref_source=dark_source, white_ref_source=white_source)
        else:
            fields.update(
                black_level=float(DEFAULT_BLACK_LEVEL if black_level is None else black_level),
                white_level=float(DEFAULT_WHITE_LEVEL if white_level is None else white_level))
    return CalibrationSpec(calibration_id=new_id(), dataset_id=dataset.dataset_id, mode=mode,
                           output_signal_unit=unit, created_at=created_at or now_iso(), **fields)


def resolve_calibration(
    dataset: DatasetRef,
    calibrations: Mapping[str, CalibrationSpec],
    calibration: CalibrationSpec,
) -> tuple[CalibrationSpec, bool]:
    """Return the stored calibration with the content of ``calibration``, or it; and whether it is new.

    ``calibration`` is moved to ``dataset`` (and given an id if it has none).
    """
    candidate = replace(calibration, dataset_id=dataset.dataset_id,
                        calibration_id=calibration.calibration_id or new_id(),
                        created_at=calibration.created_at or now_iso())
    existing = next((cal for cal in calibrations.values() if cal.same_content_as(candidate)),
                    None)
    return (existing, False) if existing is not None else (candidate, True)


def resolve_prediction_calibration(
    dataset: DatasetRef,
    calibrations: Mapping[str, CalibrationSpec],
    active_calibration_ids: Mapping[str, str],
    calibration: CalibrationSpec | None,
) -> tuple[CalibrationSpec, bool]:
    """The calibration to predict ``dataset`` with, and whether it is new.

    An explicit ``calibration`` is resolved like :func:`resolve_calibration`;
    otherwise the dataset's active calibration is used.
    """
    if calibration is not None:
        return resolve_calibration(dataset, calibrations, calibration)
    active_id = active_calibration_ids.get(dataset.dataset_id)
    if active_id and active_id in calibrations:
        return calibrations[active_id], False
    log_and_raise("Prediction dataset '{}' has no calibration yet; explicit calibration required",
                  dataset.dataset_id)


def make_roi(
    dataset: DatasetRef,
    calibration: CalibrationSpec,
    *,
    name: str,
    frames: tuple[int, int],
    rows: tuple[int, int],
    pixels: np.ndarray,
    preview_recipe: PreprocessingRecipe | None = None,
    created_at: str | None = None,
) -> tuple[TrainingROI, np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    """Build an ROI of ``dataset`` from its frames ``pixels`` ``(frames, height, n_bands)``.

    ``frames`` and ``rows`` are inclusive ``(first, last)`` ranges; ``pixels``
    holds exactly the frames ``frames``. Returns the ROI, its spectra
    ``(n_pixels, n_bands)`` in the file's dtype and, for calibrations with
    reference spectra, the matching dark and white reference pixels.
    ``preview_recipe`` records the preprocessing shown while the ROI was
    drawn.
    """
    n_frames, height, n_bands = dataset.cube_shape
    f0, f1 = _index_range(frames, n_frames, "frames")
    r0, r1 = _index_range(rows, height, "rows")
    if np.shape(pixels)[0] != f1 - f0 + 1:
        log_and_raise("pixels hold {} frames, but frames {}..{} are {}", np.shape(pixels)[0],
                      f0, f1, f1 - f0 + 1)
    spectra = np.asarray(pixels)[:, r0:r1 + 1].reshape(-1, n_bands)
    refs: tuple[np.ndarray, np.ndarray] | None = None
    if calibration.dark_ref_data is not None and calibration.white_ref_data is not None:
        def pixels_of(ref: np.ndarray) -> np.ndarray:
            return np.tile(np.asarray(ref, dtype=np.float32)[r0:r1 + 1], (f1 - f0 + 1, 1))

        refs = (pixels_of(calibration.dark_ref_data), pixels_of(calibration.white_ref_data))
    created_at = created_at or now_iso()
    recipe = preview_recipe or PreprocessingRecipe()
    roi = TrainingROI(
        roi_id=new_id(), dataset_id=dataset.dataset_id,
        calibration_id=calibration.calibration_id, name=name,
        wavelengths=dataset.wavelengths, n_pixels=int(spectra.shape[0]),
        spatial_bounds=(f0, f1, r0, r1), source_file=dataset.source_path, created_at=created_at,
        provenance={"dataset_id": dataset.dataset_id,
                    "calibration_id": calibration.calibration_id,
                    "signal_unit": calibration.output_signal_unit,
                    "preview_recipe_summary": _recipe_label(recipe) if preview_recipe else "none",
                    "created_at": created_at, **(recipe.to_dict() if preview_recipe else {})},
    )
    return roi, spectra, refs


def _native_path(path: str | os.PathLike[str]) -> str:
    """Absolute path with this system's separators (projects move between systems)."""
    native = os.fspath(path).replace("\\", os.sep).replace("/", os.sep)
    return os.path.abspath(os.path.normpath(native))


def _same_path(left: str, right: str) -> bool:
    return os.path.normcase(_native_path(left)) == os.path.normcase(_native_path(right))


def _source(reference) -> str | None:
    return None if reference is None or isinstance(reference, np.ndarray) \
        else os.path.abspath(os.fspath(reference))


def _reference(source, sensor_shape: tuple[int, ...], label: str) -> np.ndarray:
    ref = np.asarray(source if isinstance(source, np.ndarray) else load_spectra(source).X)
    if ref.ndim == 3:
        ref = ref.mean(axis=0)
    if ref.shape != tuple(sensor_shape):
        log_and_raise("The {} reference has shape {}, the sensor {}", label, ref.shape,
                      tuple(sensor_shape), depth=3)
    return ref.astype(np.float32)


def _index_range(value: tuple[int, int], size: int, name: str) -> tuple[int, int]:
    first, last = (int(v) for v in value)
    if not 0 <= first <= last < size:
        log_and_raise("{} must be an inclusive range (first, last) within 0..{}, got {}", name,
                      size - 1, value, depth=2)
    return first, last


def _as_recipe(recipe: PreprocessingRecipe | Mapping[str, Any] | None) -> PreprocessingRecipe:
    if recipe is None:
        return PreprocessingRecipe()
    if isinstance(recipe, PreprocessingRecipe):
        return recipe
    return PreprocessingRecipe.from_dict(dict(recipe))


def _recipe_id(recipe: PreprocessingRecipe) -> str:
    """An id derived from the recipe's content, so identical recipes are stored once."""
    text = json.dumps(recipe.to_dict(), sort_keys=True)
    return "recipe_" + hashlib.sha256(text.encode()).hexdigest()[:12]


def _recipe_label(recipe: PreprocessingRecipe) -> str:
    steps = [{"sg_smooth": "SG", "sg_d1": "SG 1st der.", "sg_d2": "SG 2nd der."}[recipe.filter]
             ] if recipe.filter != "none" else []
    steps += [label for label, on in (("SNV", recipe.snv), ("MSC", recipe.msc),
                                      ("EMSC", recipe.emsc), ("detrend", recipe.detrend)) if on]
    return " + ".join(steps) or "raw"


def _same_sample_set(a: SampleSet, b: SampleSet) -> bool:
    return (set(a.roi_ids) == set(b.roi_ids) and a.target_id == b.target_id
            and a.preprocessing_recipe_id == b.preprocessing_recipe_id)


def _same_signal_domain(a: SignalDomainContract, b: SignalDomainContract) -> bool:
    from .io._hdf5_groups import values_equal

    def content(domain: SignalDomainContract) -> dict[str, Any]:
        return {k: v for k, v in domain.to_dict().items()
                if k not in ("signal_domain_id", "created_at")}

    return values_equal(content(a), content(b))


def _existing(objects, candidate, same):
    """Return the stored object with the content of ``candidate``, or ``candidate``."""
    return next((obj for obj in objects if same(obj, candidate)), candidate)


def _one_named(objects, name: str, kind: str):
    id_field = {"ROI": "roi_id", "target": "target_id"}[kind]
    return _only([obj for obj in objects if obj.name == name], name, kind, id_field=id_field)


def _only(matches: list, name: str, kind: str, *, id_field: str):
    if not matches:
        log_and_raise("No {} called '{}'", kind, name, error=KeyError, depth=3)
    if len(matches) > 1:
        log_and_raise("{} {}s share the name '{}' (ids {}); address them by id", len(matches),
                      kind, name, [getattr(obj, id_field) for obj in matches], error=KeyError,
                      depth=3)
    return matches[0]


def _summary(metrics: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={metrics[key]:.3g}" for key in ("rmsecv", "r2", "bias")
                     if isinstance(metrics.get(key), (int, float)))
