"""A Qt-free session for tests of the fit and prediction services on a stored project.

It opens a project with :class:`enginir.Project`, fits models and predicts files
through the library's services (``enginir.modeling``, ``enginir.prediction``),
with explicit ids, like the viewer's jobs do; it needs no ``QApplication``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from enginir import PreprocessingRecipe, Project
from enginir._checks import log_and_raise
from enginir._workflow import resolve_dataset, resolve_prediction_reference_correction
from enginir.io import file_format, load_spectra
from enginir.io.project_store import load_model_run_artifact_refs
from enginir.modeling import build_sample_set
from enginir.modeling import fit_model_run as fit_stored_model_run
from enginir.prediction import PredictionResult, predict_with_model_run
from enginir.project import (
    DatasetRecord,
    ModelRun,
    ReferenceCorrection,
    SignalDomainContract,
)



@dataclass(slots=True)
class PredictionContext:
    """What a prediction of one stored model needs."""

    project_path: str
    model_run: ModelRun
    signal_domain: SignalDomainContract
    preprocessing_recipe_id: str
    recipe: PreprocessingRecipe
    datasets: dict[str, DatasetRecord]
    reference_corrections: dict[str, ReferenceCorrection]
    active_reference_correction_ids: dict[str, str]
    model_registry: dict[str, Any]


class CoreSession:
    """An open project, its fitted predictors and the predictions of this session."""

    def __init__(self) -> None:
        self.project: Project | None = None
        self.model_registry: dict[str, Any] = {}
        self.predictions: dict[str, PredictionResult] = {}
        self.active_model_name: str | None = None

    # -- the stored objects, read live from the project -----------------------------

    def open_project(self, path: str) -> None:
        self.project = Project(path)
        self.model_registry.clear()
        self.predictions.clear()
        self.active_model_name = None

    @property
    def workspace(self):
        return self._project().workspace

    def _project(self) -> Project:
        if self.project is None:
            log_and_raise("No project is open", error=RuntimeError)
        return self.project

    datasets = property(lambda self: self.workspace.datasets)
    reference_corrections = property(lambda self: self.workspace.reference_corrections)
    active_reference_correction_ids = property(
        lambda self: self.workspace.active_reference_correction_ids)
    rois = property(lambda self: self.workspace.rois)
    targets = property(lambda self: self.workspace.targets)
    target_assignments = property(lambda self: self.workspace.target_assignments)
    recipes = property(lambda self: self.workspace.recipes)
    signal_domains = property(lambda self: self.workspace.signal_domains)
    sample_sets = property(lambda self: self.workspace.sample_sets)
    model_runs = property(lambda self: self.workspace.model_runs)

    def register_dataset(self, dataset: DatasetRecord) -> None:
        self._project().save_dataset(dataset)

    def set_active_reference_correction(self, reference_correction: ReferenceCorrection) -> None:
        self._project().save_reference_correction(reference_correction)

    def ensure_model_run_artifact_refs(self, model_run_id: str) -> dict[str, Any]:
        """The model's artifact references, read from the project file."""
        return load_model_run_artifact_refs(self._project().path, model_run_id)

    # -- fitting ------------------------------------------------------------------------

    def build_sample_set(self, *, target_id: str, preprocessing_recipe_id: str,
                         roi_ids: list[str] | None = None, sample_set_id: str | None = None,
                         created_at: str | None = None) -> Any:
        """Build one calibration set from the stored ROIs and store it."""
        project = self._project()
        data = build_sample_set(
            project_path=project.path, rois=self.rois, target_assignments=self.target_assignments,
            target_id=target_id, preprocessing_recipe_id=preprocessing_recipe_id, roi_ids=roi_ids,
            sample_set_id=sample_set_id, created_at=created_at)
        project.save_sample_set(data.sample_set)
        return data

    def fit_model_run(self, *, sample_set_id: str, signal_domain_id: str,
                      signal_domain: SignalDomainContract | None = None, model_type: str,
                      hyperparameters: dict[str, Any] | None = None, cv: int | str = "auto",
                      model_run_id: str | None = None, created_at: str | None = None) -> Any:
        """Fit a model on a stored calibration set, store it and keep its predictor."""
        project = self._project()
        if sample_set_id not in self.sample_sets:
            log_and_raise("Unknown sample_set_id '{}'", sample_set_id)
        project.require_current_sample_set(sample_set_id, action="fit model")
        sample_set = self.sample_sets[sample_set_id]
        target = self.targets.get(sample_set.target_id)
        signal_domains = dict(self.signal_domains)
        if signal_domain is not None:
            signal_domains[signal_domain.signal_domain_id] = signal_domain
        if signal_domain_id not in signal_domains:
            log_and_raise("Unknown signal_domain_id '{}'", signal_domain_id)
        result = fit_stored_model_run(
            project_path=project.path, sample_set=sample_set, datasets=self.datasets,
            reference_corrections=self.reference_corrections, rois=self.rois,
            target_assignments=self.target_assignments, recipes=self.recipes,
            signal_domains=signal_domains, signal_domain_id=signal_domain_id,
            model_type=model_type, hyperparameters=hyperparameters, cv=cv,
            model_run_id=model_run_id, created_at=created_at,
            target_name=None if target is None else target.name)
        if signal_domain is not None and signal_domain.signal_domain_id not in self.signal_domains:
            project.save_signal_domain(signal_domain)
        run = result.model_run
        if run.name is None:
            run = replace(run, name=project.default_model_name(run))
        project.save_model_run(run)
        self.model_registry[run.model_run_id] = result.predictor
        self.active_model_name = run.model_run_id
        return result

    # -- predicting ---------------------------------------------------------------------

    def build_prediction_job_context(self, *, model_run_id: str) -> PredictionContext:
        project = self._project()
        if model_run_id not in self.model_runs:
            log_and_raise("Unknown model_run_id '{}'", model_run_id)
        model_run = self.model_runs[model_run_id]
        sample_set = self.sample_sets[model_run.sample_set_id]
        recipe_id = sample_set.preprocessing_recipe_id
        return PredictionContext(
            project_path=project.path, model_run=model_run,
            signal_domain=self.signal_domains[model_run.signal_domain_id],
            preprocessing_recipe_id=recipe_id,
            recipe=PreprocessingRecipe.from_dict(self.recipes[recipe_id]),
            datasets=dict(self.datasets), reference_corrections=dict(self.reference_corrections),
            active_reference_correction_ids=dict(self.active_reference_correction_ids),
            model_registry=dict(self.model_registry))

    def predict_dataset(self, *, model_run_id: str, data_path: str,
                        reference_correction: ReferenceCorrection | None = None,
                        dataset_id: str | None = None, prediction_run_id: str | None = None,
                        created_at: str | None = None,
                        loader_kwargs: dict[str, Any] | None = None) -> PredictionResult:
        """Predict a file in memory (nothing is written) and keep the result."""
        context = self.build_prediction_job_context(model_run_id=model_run_id)
        kwargs = dict(loader_kwargs or {})
        if file_format(data_path) in ("eiff", "hdf5"):
            kwargs["header_only"] = True
        spectral_data = load_spectra(data_path, **kwargs)
        dataset, _ = resolve_dataset(context.datasets, data_path, spectral_data,
                                     dataset_id=dataset_id, created_at=created_at,
                                     project_path=context.project_path)
        resolved, _ = resolve_prediction_reference_correction(
            dataset, context.reference_corrections, context.active_reference_correction_ids,
            reference_correction)
        result = predict_with_model_run(
            project_path=context.project_path, model_run=context.model_run,
            signal_domain=context.signal_domain, recipe=context.recipe, dataset=dataset,
            reference_correction=resolved, model_registry=context.model_registry,
            data_path=data_path, spectral_data=spectral_data,
            preprocessing_recipe_id=context.preprocessing_recipe_id,
            prediction_run_id=prediction_run_id, created_at=created_at,
            loader_kwargs=loader_kwargs, prefer_saved_model=True)
        self.predictions[result.prediction_run.prediction_run_id] = result
        return result


def fit_stored_model(path: str, *, model_type: str = "PLS", n_components: int = 1,
                     model_run_id: str = "model_1") -> CoreSession:
    """Fit and store one model on the ROIs of ``write_model_project`` (``"logo"`` folds)."""
    session = CoreSession()
    session.open_project(path)
    session.build_sample_set(target_id="target_moisture", preprocessing_recipe_id="recipe_none",
                             sample_set_id="sample_set_1")
    session.fit_model_run(sample_set_id="sample_set_1", signal_domain_id="signal_domain_1",
                          model_type=model_type, hyperparameters={"n_components": n_components},
                          cv="logo", model_run_id=model_run_id)
    return session
