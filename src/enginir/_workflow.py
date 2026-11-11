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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ._checks import log_and_raise
from ._logging import get_logger
from .io import file_format, load_spectra
from .io import project_store as store
from .modeling import BandProposal, fit_model_run, plan_sample_set, propose_bands
from .modeling._service import fold_groups
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
    FitSpec,
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
from .project._models import check_roi_use, sample_label
from .prediction._rois import (
    RoiPrediction,
    RoiPredictionSummary,
    predict_rois,
    roi_prediction_statistics,
)

logger = get_logger(__name__)

__all__ = ["DEFAULT_BLACK_LEVEL", "DEFAULT_WHITE_LEVEL", "Dataset", "GridCell", "Model",
           "Prediction", "Project", "StoredPrediction", "check_data_file_location", "data_file_labels",
           "make_calibration", "make_roi", "read_frames", "read_header",
           "recipe_id_for", "resolve_calibration", "resolve_dataset",
           "resolve_prediction_calibration", "resolve_sample_set", "resolve_signal_domain",
           "roi_grid"]

#: Dark and white detector levels of EVK cameras, used when a dataset is
#: calibrated to reflectance or absorbance without reference spectra.
DEFAULT_BLACK_LEVEL = 128
DEFAULT_WHITE_LEVEL = 3500

# Loaders that can read a frame range or only the header instead of the whole cube.
_FRAME_FORMATS = ("eiff", "hdf5")


class Project:
    """A project file and the workflow on it.

    Create one with :meth:`create` or :meth:`open`. Every object has a
    stable id, its only identity and what links the objects in the file.
    Names are labels and may repeat. For convenience, ROIs, targets and
    models can be addressed by name and datasets by file path; reproducible
    scripts pass the objects themselves or look them up by id
    (``project.roi(roi_id=...)``), because a renamed or reused name refers
    to another object.

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
        """Create an empty project file (an existing file at ``path`` is replaced)."""
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
        """ROIs by name; an error if names repeat (``workspace.rois`` is by id)."""
        return _by_name(self.workspace.rois.values(), "ROI", "roi_id")

    @property
    def targets(self) -> dict[str, TargetDefinition]:
        """Targets by name; an error if names repeat (``workspace.targets`` is by id)."""
        return _by_name(self.workspace.targets.values(), "target", "target_id")

    @property
    def models(self) -> dict[str, Model]:
        """Models by name; an error if names repeat (``workspace.model_runs`` is by id)."""
        runs = _by_name(self.workspace.model_runs.values(), "model", "model_run_id",
                        key=self._model_name)
        return {name: Model(self, run) for name, run in runs.items()}

    @property
    def predictions(self) -> dict[str, StoredPrediction]:
        """Predictions by id."""
        return {run_id: StoredPrediction(self, run)
                for run_id, run in self.workspace.predictions.items()}

    def roi(self, name: str | None = None, *, roi_id: str | None = None) -> TrainingROI:
        """Return the ROI called ``name``, or the ROI with ``roi_id``.

        A name several ROIs share is an error listing their ids; the id is
        the reproducible reference.
        """
        if _by_id_or_name(name, roi_id, "ROI"):
            return _with_id(self.workspace.rois, roi_id, "ROI")
        return _one_named(self.workspace.rois.values(), str(name), "ROI")

    def target(self, name: str | None = None, *,
               target_id: str | None = None) -> TargetDefinition:
        """Return the target called ``name``, or the target with ``target_id``."""
        if _by_id_or_name(name, target_id, "target"):
            return _with_id(self.workspace.targets, target_id, "target")
        return _one_named(self.workspace.targets.values(), str(name), "target")

    def model(self, name: str | None = None, *, model_run_id: str | None = None) -> Model:
        """Return the model called ``name``, or the model with ``model_run_id``."""
        if _by_id_or_name(name, model_run_id, "model"):
            return Model(self, _with_id(self.workspace.model_runs, model_run_id, "model"))
        runs = [run for run in self.workspace.model_runs.values()
                if self._model_name(run) == name]
        return Model(self, _only(runs, str(name), "model", id_field="model_run_id"))

    def dataset(self, path: str | os.PathLike[str] | None = None, *,
                dataset_id: str | None = None) -> Dataset:
        """Return the dataset registered for the file ``path``, or the one with ``dataset_id``."""
        if (path is None) == (dataset_id is None):
            log_and_raise("Give the dataset's path or its dataset_id, not both or neither",
                          error=TypeError)
        if dataset_id is not None:
            return Dataset(self, _with_id(self.workspace.datasets, dataset_id, "dataset"))
        ref = self._dataset_ref(os.path.abspath(os.fspath(str(path))))
        if ref is None:
            log_and_raise("No dataset is registered for {}", path, error=KeyError)
        return Dataset(self, ref)

    def _roi_of(self, roi: str | TrainingROI) -> TrainingROI:
        """``roi`` itself when it is an ROI of this project, else the ROI with that name."""
        if isinstance(roi, TrainingROI):
            return _with_id(self.workspace.rois, roi.roi_id, "ROI")
        return self.roi(roi)

    def _target_of(self, target: str | TargetDefinition) -> TargetDefinition:
        if isinstance(target, TargetDefinition):
            return _with_id(self.workspace.targets, target.target_id, "target")
        return self.target(target)

    # -- writing ---------------------------------------------------------------

    def add_dataset(
        self,
        path: str | os.PathLike[str],
        *,
        dark: str | os.PathLike[str] | np.ndarray | None = None,
        white: str | os.PathLike[str] | np.ndarray | None = None,
        unit: str = "Absorbance",
        dark_frames: tuple[int, int] | None = None,
        white_frames: tuple[int, int] | None = None,
        references_from: str | os.PathLike[str] | None = None,
        black_level: float | None = None,
        white_level: float | None = None,
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
        dark_frames, white_frames:
            Inclusive ``(first, last)`` frames of this file to average into
            the dark or white reference instead of ``dark``/``white``, e.g. a
            white tile or shutter-closed frames scanned in the same image.
            Each can be combined with a reference file for the other one.
        references_from:
            A registered data file whose dark and white references this file
            uses too (one reference set for a session). Its arrays are copied
            into this file's own calibration. Not combined with the other
            reference arguments.
        black_level, white_level:
            Detector levels for raw counts calibrated without references,
            instead of the file's metadata or the defaults. Not combined with
            references; ``white_level`` must exceed ``black_level``.

        Returns
        -------
        Dataset

        Notes
        -----
        The dataset and its calibration are validated completely before
        either is written. Re-registering a file with another calibration
        makes that the active one; ROIs drawn before keep theirs.
        """
        dataset, calibration, new_dataset, new_calibration = self._resolve_dataset(
            path, dark=dark, white=white, unit=unit, dark_frames=dark_frames,
            white_frames=white_frames, references_from=references_from,
            black_level=black_level, white_level=white_level)
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

    def set_sample(self, roi: str | TrainingROI, label: str | None) -> None:
        """Set the sample label of ``roi`` (name or object); ``None`` or blank text removes it.

        ROIs with equal labels are one physical sample: with
        ``fit_spec={"folds": {"group_by": "sample"}}`` they are held out
        together. The label never changes the ROI itself.
        """
        self.save_roi_sample(self._roi_of(roi).roi_id, label)

    def set_use(self, roi: str | TrainingROI, use: str) -> None:
        """Mark ``roi`` (name or object) as ``"calibration"`` or ``"prediction"``.

        A prediction ROI is a sample to predict (``model.predict_rois()``); it
        never enters a calibration set, whatever target values it has.
        ``"training"`` (the name before 1.5) is refused.
        """
        self.save_roi_use(self._roi_of(roi).roi_id, use)

    def set_value(self, roi: str | TrainingROI, target: str | TargetDefinition,
                  value: float | None) -> None:
        """Set (or replace) the reference value of ``target`` for ``roi``; ``None`` removes it.

        ``roi`` and ``target`` are names or the objects themselves
        (reproducible: e.g. ``project.roi(roi_id=...)``).
        """
        roi_ref, target_ref = self._roi_of(roi), self._target_of(target)
        if value is None:
            self.remove_assignment(roi_ref.roi_id, target_ref.target_id)
            return
        assignment = TargetAssignment(roi_id=roi_ref.roi_id, target_id=target_ref.target_id,
                                      value=value, assigned_at=now_iso())
        self.save_assignment(assignment)

    def propose_bands(
        self,
        target: str | TargetDefinition,
        *,
        recipe: PreprocessingRecipe | Mapping[str, Any] | None = None,
        model: str = "PLS",
        n_intervals: int = 20,
        max_steps: int | None = None,
        fit_spec: FitSpec | Mapping[str, Any] | None = None,
        progress: Callable[[int, int], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        **hyperparameters: Any,
    ) -> BandProposal:
        """Propose band ranges for ``target`` by forward interval PLS, validated nested.

        The search runs on the recipe's axis after Savitzky-Golay (a band
        selection in ``recipe`` is ignored), split into ``n_intervals``
        intervals; each step adds the interval that lowers the RMSECV most
        and the search stops when the gain is within one standard error. The
        whole search is repeated inside every outer fold, so
        ``nested_rmsecv`` estimates the error of the procedure including the
        selection; compare it with ``all_bands_rmsecv``. Nothing is written
        to the project (``headless-workflow-contract_v3.md``, Band proposal).

        Parameters
        ----------
        target, recipe, model, fit_spec, **hyperparameters:
            As for :meth:`fit`; the folds, fold groups, pixel sample and
            component rule of ``fit_spec`` are used (no DModX exclusion).
            Without a number of components the rule chooses it.
        n_intervals:
            Number of contiguous intervals (2 to the band count).
        max_steps:
            Largest number of intervals to choose (default: all).
        progress, cancel:
            ``progress(done, total)`` after each candidate (``total`` is an
            upper bound); ``cancel()`` returning ``True`` stops with
            :class:`~enginir.modeling.BandProposalCancelledError`.

        Returns
        -------
        BandProposal
            ``ranges`` (for ``PreprocessingRecipe(bands_keep=...)``),
            ``steps``, ``intervals``, ``nested_rmsecv``, ``all_bands_rmsecv``
            and ``n_components``.

        Examples
        --------
        >>> proposal = project.propose_bands("API", recipe=PreprocessingRecipe(snv=True))
        ... # doctest: +SKIP
        >>> project.fit("API", recipe=PreprocessingRecipe(snv=True, bands_keep=proposal.ranges),
        ...             n_components=proposal.n_components)  # doctest: +SKIP
        """
        ws = self.workspace
        if fit_spec is not None and FitSpec.coerce(fit_spec).groups:
            log_and_raise("fit_spec folds/groups is written by the fit; leave it out and choose "
                          "folds/group_by instead")
        target_ref = self._target_of(target)
        recipe_obj = _as_recipe(recipe)
        recipe_id = _recipe_id(recipe_obj)
        recipes = {**ws.recipes, recipe_id: recipe_obj.to_dict()}
        sample_set, _new = resolve_sample_set(
            rois=ws.rois, target_assignments=ws.target_assignments,
            target_id=target_ref.target_id, recipe_id=recipe_id, sample_sets=ws.sample_sets)
        domain = resolve_signal_domain(self._training_signal_domain(sample_set),
                                       ws.signal_domains)
        return propose_bands(
            project_path=self.path, sample_set=sample_set, datasets=ws.datasets,
            calibrations=ws.calibrations, rois=ws.rois,
            target_assignments=ws.target_assignments, recipes=recipes,
            signal_domains={**ws.signal_domains, domain.signal_domain_id: domain},
            signal_domain_id=domain.signal_domain_id, model_type=model,
            hyperparameters=hyperparameters, n_intervals=n_intervals, max_steps=max_steps,
            fit_spec=fit_spec, progress=progress, cancel=cancel)

    def fit(
        self,
        target: str | TargetDefinition,
        *,
        recipe: PreprocessingRecipe | Mapping[str, Any] | None = None,
        model: str = "PLS",
        name: str | None = None,
        cv: int | str | None = None,
        fit_spec: FitSpec | Mapping[str, Any] | None = None,
        **hyperparameters: Any,
    ) -> Model:
        """Fit and cross-validate a model of ``target`` on every ROI with a value for it.

        Parameters
        ----------
        target:
            Target name, or the :class:`~enginir.project.TargetDefinition`.
        recipe:
            Preprocessing (default: none).
        model:
            ``"PLS"`` or ``"PCR"``.
        name:
            Model name (default: e.g. ``"PLS 2 LV, SNV"``).
        cv:
            Cross-validation over ROIs, the short form of the fold rule:
            ``"auto"`` (default: 7 round-robin folds), a number of
            round-robin folds, or ``"logo"`` (leave one ROI out).
        fit_spec:
            Full control of the fit (:class:`~enginir.project.FitSpec` or a
            mapping with the keys of ``sample-set-model-run-contract_v3.md``
            §7), e.g. ``{"pixels": {"per_roi": 50, "seed": 1}}`` or
            ``{"folds": {"group_by": "sample"}}`` (hold the ROIs of one sample
            out together, see :meth:`set_sample`); keys left out take their
            defaults. Not combined with ``cv`` when it sets the folds;
            ``folds/groups`` is written by the fit and refused here. The
            complete specification is stored with the model.
        **hyperparameters:
            Estimator parameters, e.g. ``n_components=2``.

        Returns
        -------
        Model
        """
        ws = self.workspace
        if fit_spec is not None and FitSpec.coerce(fit_spec).groups:
            log_and_raise("fit_spec folds/groups is written by the fit; leave it out and choose "
                          "folds/group_by instead")
        if fit_spec is not None and FitSpec.coerce(fit_spec).components_selected:
            log_and_raise("fit_spec components/selected is written by the fit; pass "
                          "n_components='auto' or a number instead")
        target_ref = self._target_of(target)
        recipe_obj = _as_recipe(recipe)
        recipe_id, recipe_dict = _recipe_id(recipe_obj), recipe_obj.to_dict()
        if name is not None:
            self._require_new_name(ws.model_runs.values(), name, "model", key=self._model_name)
        recipes = {**ws.recipes, recipe_id: recipe_dict}

        # The fit reads the spectra; the sample set itself needs none.
        sample_set, _new = resolve_sample_set(
            rois=ws.rois, target_assignments=ws.target_assignments,
            target_id=target_ref.target_id, recipe_id=recipe_id, sample_sets=ws.sample_sets)
        domain = resolve_signal_domain(self._training_signal_domain(sample_set),
                                       ws.signal_domains)
        result = fit_model_run(
            project_path=self.path, sample_set=sample_set, datasets=ws.datasets,
            calibrations=ws.calibrations, rois=ws.rois,
            target_assignments=ws.target_assignments, recipes=recipes,
            signal_domains={**ws.signal_domains, domain.signal_domain_id: domain},
            signal_domain_id=domain.signal_domain_id, model_type=model,
            hyperparameters=hyperparameters, cv=cv, fit_spec=fit_spec,
            target_name=target_ref.name)
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

    def save_assignments(self, assignments: Sequence[TargetAssignment]) -> None:
        """Store several reference values; all are validated before any is written."""
        self._validate(target_assignments=list(assignments))
        for assignment in assignments:
            self.save_assignment(assignment)

    def remove_assignment(self, roi_id: str, target_id: str) -> None:
        """Delete the reference value of ``target_id`` for ``roi_id`` (it becomes missing)."""
        store.remove_target_assignment(self.path, target_id, roi_id)
        self.workspace.target_assignments.pop((roi_id, target_id), None)

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

    def save_roi_use(self, roi_id: str, use: str) -> None:
        """Set a stored ROI's use: ``"calibration"`` or ``"prediction"`` (never in a calibration
        set)."""
        roi = _with_id(self.workspace.rois, roi_id, "ROI")
        use = check_roi_use(use)
        store.save_roi_use(self.path, roi_id, use)
        self.workspace.rois[roi_id] = replace(roi, use=use)

    def save_roi_sample(self, roi_id: str, sample: str | None) -> None:
        """Set (or, with ``None`` or blank text, remove) the sample label of a stored ROI."""
        roi = _with_id(self.workspace.rois, roi_id, "ROI")
        label = sample_label(sample)
        store.save_roi_sample(self.path, roi_id, label)
        self.workspace.rois[roi_id] = replace(roi, sample=label)

    def save_rois(self, rois: Sequence[tuple[TrainingROI, np.ndarray, tuple[np.ndarray,
                                                                        np.ndarray] | None]],
                  assignments: Sequence[TargetAssignment] = ()) -> None:
        """Store several ROIs (with spectra and reference snapshots) completely or not at all.

        Every ROI and assignment is validated before anything is written; ROI
        ids must be new and differ from each other. Names are labels and are
        not checked (headless-workflow contract, Names and ids). If a write
        fails, the ROIs already written are removed again (ROI/target
        contract, ROI grid).
        """
        ws = self.workspace
        ids = [roi.roi_id for roi, _spectra, _refs in rois]
        repeated = sorted({roi_id for roi_id in ids if ids.count(roi_id) > 1 or roi_id in ws.rois})
        if repeated:
            log_and_raise("ROI ids already exist or repeat within the batch: {}", repeated)
        self._validate(rois={roi.roi_id: roi for roi, _spectra, _refs in rois},
                       target_assignments=list(assignments))
        written: list[str] = []
        try:
            for roi, spectra, refs in rois:
                self.save_roi(roi, spectra, dark_ref_spectra=refs[0] if refs else None,
                              white_ref_spectra=refs[1] if refs else None)
                written.append(roi.roi_id)
            for assignment in assignments:
                self.save_assignment(assignment)
        except Exception as exc:
            logger.error("Writing {} ROIs failed after {}: {}; removing them again", len(rois),
                         len(written), exc)
            for roi_id in written:
                self.remove_roi(roi_id)
            raise

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
                # A new value of a (roi, target) pair replaces the stored one: updating
                # a value is the documented editing model (roi-target contract).
                merged = dict(ws.target_assignments)
                merged.update({(a.roi_id, a.target_id): a for a in value})
                graph[key] = list(merged.values())
            elif isinstance(value, dict) and key != "active_calibration_ids":
                graph[key] = {**graph[key], **value}
            else:
                graph[key] = value
        validate_project_relationships(**graph)

    # -- internals -------------------------------------------------------------

    def reload(self) -> None:
        """Read the project file again, e.g. after another window wrote to it."""
        self._reload()

    def _reload(self, store_contents: dict[str, Any] | None = None) -> None:
        if store_contents is None:
            store_contents = store.load_project_store(self.path)
        self.workspace = hydrate_project_store(store_contents, allow_stale_sample_set_rois=True)

    def _dataset_ref(self, abs_path: str) -> DatasetRef | None:
        return next((ref for ref in self.workspace.datasets.values()
                     if _same_path(ref.source_path, abs_path)), None)

    def _resolve_dataset(self, path, *, dark, white, unit, dark_frames=None, white_frames=None,
                         references_from=None, black_level=None, white_level=None):
        """Return ``(dataset, calibration, dataset_is_new, calibration_is_new)``, writing nothing."""
        ws = self.workspace
        levels_given = black_level is not None or white_level is not None
        if levels_given and any(value is not None for value in (
                dark, white, dark_frames, white_frames, references_from)):
            log_and_raise("Detector levels calibrate without references; do not combine "
                          "black_level/white_level with dark/white references")
        abs_path = _native_path(path)
        data = read_header(abs_path)
        dataset, new_dataset = resolve_dataset(ws.datasets, path, data)
        sensor = dataset.cube_shape[1:]
        if references_from is not None:
            if any(value is not None for value in (dark, white, dark_frames, white_frames)):
                log_and_raise("references_from takes both references from another data file; "
                              "do not combine it with dark, white or reference frames")
            references = self._references_of(references_from, sensor)
        else:
            references = tuple(
                self._one_reference(abs_path, file, frames, sensor, dataset.cube_shape[0], label)
                for file, frames, label in ((dark, dark_frames, "dark"),
                                            (white, white_frames, "white")))
        (dark_ref, dark_source), (white_ref, white_source) = references
        candidate = make_calibration(
            dataset, unit=unit, precalibrated=np.dtype(data.X.dtype).kind == "f",
            dark=dark_ref, white=white_ref, dark_source=dark_source, white_source=white_source,
            black_level=data.meta.get("black_level") if black_level is None else black_level,
            white_level=data.meta.get("white_level") if white_level is None else white_level)
        if candidate.black_level is not None and candidate.white_level is not None \
                and not candidate.white_level > candidate.black_level:
            log_and_raise("The white level ({:g}) must be above the black level ({:g})",
                          candidate.white_level, candidate.black_level)
        calibration, new_calibration = resolve_calibration(dataset, ws.calibrations, candidate)
        return dataset, calibration, new_dataset, new_calibration

    @staticmethod
    def _one_reference(abs_path, file, frames, sensor, n_frames, label):
        """``(reference, source)`` of a dark or white reference from a file or this file's frames."""
        if file is not None and frames is not None:
            log_and_raise("Give the {} reference as a file or as frames, not both", label, depth=2)
        if frames is not None:
            first, last = _frame_range(frames, n_frames, label)
            reference = read_frames(abs_path, first, last).mean(axis=0)
            return (_reference(reference, sensor, label),
                    f"{abs_path} frames {first}-{last}")
        if file is None:
            return None, None
        return _reference(file, sensor, label), _source(file)

    def _references_of(self, path, sensor):
        """The dark and white references of another registered data file, with their sources."""
        shared = self.dataset(path)
        source, calibration = shared.ref, shared.calibration
        if calibration.dark_ref_data is None or calibration.white_ref_data is None:
            log_and_raise("{} has no dark and white reference spectra to share",
                          source.source_path, depth=2)
        origin = f"references of {source.source_path}"
        return ((_reference(calibration.dark_ref_data, sensor, "dark"),
                 calibration.dark_ref_source or origin),
                (_reference(calibration.white_ref_data, sensor, "white"),
                 calibration.white_ref_source or origin))

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
        """The ROIs of this dataset by name; an error if names repeat."""
        return _by_name([roi for roi in self.project.workspace.rois.values()
                         if roi.dataset_id == self.ref.dataset_id], "ROI", "roi_id")

    def add_roi(self, name: str, *, frames: tuple[int, int], rows: tuple[int, int],
                values: Mapping[str, float] | None = None,
                use: str = "calibration") -> TrainingROI:
        """Store the pixels ``frames`` x ``rows`` as an ROI with its reference values.

        Parameters
        ----------
        name:
            ROI name. This function refuses a name the project already has,
            so a script can find the ROI by it again; ROIs stored otherwise
            (e.g. by the viewer) may repeat names.
        frames, rows:
            Inclusive ``(first, last)`` index ranges.
        values:
            Reference values ``{target name: value}``.
        use:
            ``"calibration"`` (default) or ``"prediction"``: a sample to
            predict (``model.predict_rois()``), never in a calibration set.

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
                                      pixels=read_frames(self.path, first, last), use=use)
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

    def locate(self, path: str | os.PathLike[str]) -> Dataset:
        """Record that this data file now lives at ``path`` (it was moved or renamed).

        The file must be this data file: same format, shape, modality and
        wavelengths, and for every ROI its pixels at the ROI's bounds equal
        the stored raw pixels (:func:`check_data_file_location`). Only
        ``source_path`` changes; nothing is written if the check fails.

        Returns
        -------
        Dataset
            This dataset at its new location.
        """
        project, ws = self.project, self.project.workspace
        abs_path = _native_path(path)
        other = project._dataset_ref(abs_path)
        if other is not None and other.dataset_id != self.ref.dataset_id:
            log_and_raise("{} is already registered as another data file of the project",
                          abs_path)
        rois = [roi for roi in ws.rois.values() if roi.dataset_id == self.ref.dataset_id]
        check_data_file_location(project.path, self.ref, rois, abs_path)
        store.relocate_dataset_ref(project.path, self.ref.dataset_id, abs_path)
        self.ref = replace(self.ref, source_path=abs_path)
        ws.datasets[self.ref.dataset_id] = self.ref
        return self

    def add_rois(self, cells: Sequence[GridCell]) -> list[TrainingROI]:
        """Store the cells of an ROI grid (see :func:`roi_grid`), all or none.

        Like :meth:`add_roi`, refuses names the project already has.

        Returns
        -------
        list of TrainingROI
            In the order of ``cells``.
        """
        if not cells:
            log_and_raise("An ROI grid needs at least one cell")
        names = [cell.name for cell in cells]
        for name in names:
            self.project._require_new_name(self.project.workspace.rois.values(), name, "ROI")
        if len(set(names)) != len(names):
            log_and_raise("ROI names repeat within the grid: {}",
                          sorted({name for name in names if names.count(name) > 1}))
        n_frames = self.ref.cube_shape[0]
        first = min(_index_range(cell.frames, n_frames, "frames")[0] for cell in cells)
        last = max(_index_range(cell.frames, n_frames, "frames")[1] for cell in cells)
        pixels = read_frames(self.path, first, last)
        built = [make_roi(self.ref, self.calibration, name=cell.name, frames=cell.frames,
                          rows=cell.rows, pixels=pixels[cell.frames[0] - first:
                                                        cell.frames[1] - first + 1],
                          grid_cell=cell)
                 for cell in cells]
        self.project.save_rois(built)
        return [roi for roi, _spectra, _refs in built]


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

    @property
    def component_curve(self) -> dict[str, Any] | None:
        """RMSECV, its standard error, Q², Q²(cum), R²X(cum) and R²Y(cum) per number of components.

        Arrays over ``n_components`` 1 … A_max and every rule's suggestion
        (``suggested_one_se``, ``suggested_min_rmsecv``, ``suggested_q2_limit``,
        ``suggested_f_test``, ``suggested_q2_lookahead``); from engiNIR 1.4 also
        ``r2y_cum_roi``, ``rmsee_roi``, ``noise``, ``dw_weights`` and
        ``structure_ceiling``. ``None`` for a model fitted before engiNIR 1.3
        (sample-set-model-run contract §7.4).
        """
        refs = store.load_model_run_artifact_refs(self.project.path, self.ref.model_run_id,
                                                  keys=("component_curve",))
        return refs.get("component_curve") or None

    @property
    def excluded_pixels(self) -> dict[str, np.ndarray]:
        """Pixels left out of the final model by DModX exclusion, per ROI id.

        Indices into each ROI's stored spectra; empty without exclusion
        (sample-set-model-run contract §7.2).
        """
        refs = store.load_model_run_artifact_refs(self.project.path, self.ref.model_run_id,
                                                  keys=("exclusion",))
        exclusion = refs.get("exclusion", {})
        if not exclusion:
            return {}
        roi_ids = [str(roi_id) for roi_id in np.atleast_1d(exclusion["roi_ids"])]
        counts = np.atleast_1d(exclusion["counts"]).astype(int)
        indices = np.atleast_1d(exclusion["pixel_indices"]).astype(np.int64)
        bounds = np.concatenate([[0], np.cumsum(counts)])
        return {roi_id: indices[bounds[i]:bounds[i + 1]] for i, roi_id in enumerate(roi_ids)}

    def refit(self, *, name: str | None = None) -> Model:
        """Fit this model again on exactly what it was fitted on, and store the new run.

        Uses the stored sample set (its ``roi_ids`` and preprocessing recipe),
        signal domain, model type, hyperparameters and fit specification
        (for a model stored without one, its legacy specification), not the
        ROIs that have values now. Folds depend only on the sample set and
        the pixel sample only on the seed and the ROI spectra, so unchanged
        data give the same metrics; reference values changed since the fit
        and a different pixel sample are reported. Refused when an ROI of the
        sample set was deleted.

        Parameters
        ----------
        name:
            Name of the new model (default: the usual name with ``(2)``, ...).

        Returns
        -------
        Model
            The new model; compare ``refitted.metrics`` with ``self.metrics``.
        """
        project, ws = self.project, self.project.workspace
        sample_set = ws.sample_sets[self.ref.sample_set_id]
        project.require_current_sample_set(sample_set.sample_set_id, action="refit the model")
        if name is not None:
            project._require_new_name(ws.model_runs.values(), name, "model",
                                      key=project._model_name)
        changed = self._changed_values(sample_set)
        if changed:
            logger.warning("Refitting model '{}': {} reference value(s) changed since it was "
                           "fitted (ROI ids {}); the metrics will differ", self.name,
                           len(changed), changed[:5])
        spec = self.ref.fit_specification
        self._warn_on_group_drift(spec, sample_set)
        hyperparameters = dict(self.ref.hyperparameters)
        if self.ref.fit_spec is None and not any(
                key in hyperparameters for key in ("scale", "x_scaling", "y_scaling")):
            hyperparameters["scale"] = True  # runs before 1.2: absent means True (§7.3)
        result = fit_model_run(
            project_path=project.path, sample_set=sample_set, datasets=ws.datasets,
            calibrations=ws.calibrations, rois=ws.rois,
            target_assignments=ws.target_assignments, recipes=ws.recipes,
            signal_domains=ws.signal_domains, signal_domain_id=self.ref.signal_domain_id,
            model_type=self.ref.model_type, hyperparameters=hyperparameters,
            fit_spec=spec, target_name=self.target.name)
        if spec.exclusion == "dmodx":
            self._warn_on_exclusion_drift(result.model_run)
        drawn = result.model_run.fit_specification.checksum
        if spec.checksum and drawn != spec.checksum:
            logger.warning("Refit of '{}' drew a different pixel sample than the model was "
                           "fitted on (checksum {} instead of {}): the stored spectra of an ROI "
                           "changed", self.name, drawn[:12], spec.checksum[:12])
        run = replace(result.model_run,
                      name=name or project.default_model_name(result.model_run, self.recipe))
        project.save_model_run(run)
        refitted = Model(project, project.workspace.model_runs[run.model_run_id])
        differences = _metric_differences(self.metrics, refitted.metrics)
        if differences:
            logger.warning("Refit of '{}' differs in {}", self.name, differences)
        else:
            logger.info("Refit of '{}' reproduced its metrics as '{}'", self.name, refitted.name)
        return refitted

    def _warn_on_exclusion_drift(self, refitted: ModelRun) -> None:
        """Warn about ROIs whose excluded pixels differ from the stored ones (contract §6)."""
        before = self.excluded_pixels
        exclusion = refitted.artifact_refs.get("exclusion", {})
        after: dict[str, np.ndarray] = {}
        start = 0
        for roi_id, count in zip(exclusion.get("roi_ids", []), exclusion.get("counts", [])):
            after[str(roi_id)] = np.asarray(exclusion["pixel_indices"][start:start + int(count)])
            start += int(count)
        differing = sorted(roi_id for roi_id in set(before) | set(after)
                           if not np.array_equal(before.get(roi_id, []), after.get(roi_id, [])))
        if differing:
            logger.warning("Refit of '{}' excluded other pixels than the model in {} ROI(s) "
                           "(ids {}); borderline pixels can change with floating-point "
                           "differences", self.name, len(differing), differing[:5])

    def _warn_on_group_drift(self, spec: FitSpec, sample_set: SampleSet) -> None:
        """Warn when current sample labels or values would group the ROIs differently.

        The refit keeps the stored groups (sample-set-model-run contract §6).
        """
        if not spec.groups:
            return
        ws = self.project.workspace
        try:
            now = fold_groups(spec.group_by, sample_set, ws.rois, ws.target_assignments)
        except (KeyError, ValueError) as exc:
            logger.warning("Refit of '{}' keeps its stored fold groups; the current ones cannot "
                           "be built: {}", self.name, exc)
            return
        if not np.array_equal(now, np.asarray(spec.groups)):
            logger.warning("Refit of '{}' keeps its stored fold groups (by {}); the current "
                           "{} would group the ROIs differently", self.name, spec.group_by,
                           "sample labels" if spec.group_by == "sample" else "target values")

    def _changed_values(self, sample_set: SampleSet) -> list[str]:
        """Ids of ROIs whose current reference value differs from the one the model used."""
        stored = self.diagnostics
        if "roi_ids" not in stored or "y_true_roi" not in stored:
            return []
        assignments = self.project.workspace.target_assignments
        changed = []
        for roi_id, used in zip(np.asarray(stored["roi_ids"]).astype(str),
                                np.asarray(stored["y_true_roi"], dtype=float).reshape(-1)):
            now = assignments.get((roi_id, sample_set.target_id))
            if now is None or not np.isclose(now.value, used, rtol=1e-9, atol=1e-12):
                changed.append(str(roi_id))
        return changed

    def predict_rois(self, rois: Sequence[str | TrainingROI] | None = None, *,
                     exclude_dmodx: float | bool | None = None) -> RoiPredictionSummary:
        """Predict the prediction ROIs from the project file alone, with per-ROI statistics.

        Each ROI's stored pixels are converted with its own calibration (as the
        fit does), preprocessed with the model's recipe and predicted; no data
        file is read and nothing is written (prediction-runtime contract,
        Prediction ROIs). The result is logged once.

        Parameters
        ----------
        rois:
            ROI names or objects; default every prediction ROI of the project.
        exclude_dmodx:
            Remove pixels above the DModX limit at this level from the
            statistics; ``False`` keeps every pixel; ``None`` (default) the
            model's own setting (its fit specification's DModX exclusion).

        Returns
        -------
        RoiPredictionSummary
            Per-ROI statistics, RMSEP and bias over ROIs with a value. Call
            :meth:`roi_prediction` and
            :func:`~enginir.prediction.roi_prediction_statistics` to change
            the exclusion without predicting again.
        """
        prediction = self.roi_prediction(rois)
        level = prediction.default_exclusion if exclude_dmodx is None else (
            None if exclude_dmodx is False else float(exclude_dmodx))
        summary = roi_prediction_statistics(
            prediction, target_assignments=self.project.workspace.target_assignments,
            exclude_dmodx=level)
        logger.info("Predicted {} ROI(s) of {} with model '{}' ({}); exclusion {}; {} refused; "
                    "RMSEP {} over {} ROI(s) with a value", len(prediction.pixels),
                    self.project.path, self.name, self.ref.model_run_id,
                    "none" if summary.exclusion is None else f"DModX {summary.exclusion}",
                    len(prediction.refused), summary.rmsep, summary.n_reference)
        return summary

    def roi_prediction(self, rois: Sequence[str | TrainingROI] | None = None) -> RoiPrediction:
        """The pixel predictions of the prediction ROIs (see :meth:`predict_rois`)."""
        ws = self.project.workspace
        chosen = ([roi for roi in ws.rois.values() if roi.use == "prediction"] if rois is None
                  else [self.project._roi_of(roi) for roi in rois])
        if not chosen:
            log_and_raise("No prediction ROIs to predict; mark ROIs with use='prediction'")
        return predict_rois(
            project_path=self.project.path, model_run=self.ref, model_name=self.name,
            rois=chosen, datasets=ws.datasets, calibrations=ws.calibrations,
            signal_domains=ws.signal_domains, recipes=ws.recipes, sample_sets=ws.sample_sets)

    def predict(self, path: str | os.PathLike[str], *,
                dark: str | os.PathLike[str] | np.ndarray | None = None,
                white: str | os.PathLike[str] | np.ndarray | None = None,
                checksum: bool = False) -> Prediction:
        """Predict the target for every spectrum of a data file, in memory.

        The file and its calibration are resolved as by
        :meth:`Project.add_dataset`, in the model's signal unit, but nothing
        is written to the project: a registered file reuses its stored
        calibration, a new file gets one in memory. Compatibility with the
        model is checked before the data are read. The prediction is logged
        once (model, data file with size and modification time, summary);
        ``checksum=True`` adds the SHA-256 of the data file.

        Returns
        -------
        Prediction
            The prediction map and vector, the DModXPS and T²PS maps with
            their 0.99 limits and summary, and the provenance.
        """
        project, ws = self.project, self.project.workspace
        domain = ws.signal_domains[self.ref.signal_domain_id]
        recipe_id = ws.sample_sets[self.ref.sample_set_id].preprocessing_recipe_id
        recipe = PreprocessingRecipe.from_dict(ws.recipes[recipe_id])
        registered = project._dataset_ref(_native_path(path))
        if registered is not None and dark is None and white is None:
            # A registered file is predicted with its own active calibration, the one its
            # ROIs were recorded with (not a new one from the detector levels).
            dataset = registered
            calibration = ws.calibrations[ws.active_calibration_ids[registered.dataset_id]]
        else:
            dataset, calibration, _new_dataset, _new_calibration = project._resolve_dataset(
                path, dark=dark, white=white, unit=domain.required_signal_unit)
        common: dict[str, Any] = dict(project_path=project.path, model_run=self.ref, signal_domain=domain,
                      recipe=recipe, dataset=dataset, calibration=calibration,
                      model_registry={})
        preflight_prediction(**common)
        result: PredictionResult = predict_with_model_run(
            **common, data_path=dataset.source_path, preprocessing_recipe_id=recipe_id,
            checksum=checksum)
        return Prediction(self, result)


class Prediction:
    """A prediction of one model for one data file, held in memory (never stored).

    ``map``/``y_pred`` are the predictions, ``dmodx_map``/``t2_map`` the
    applicability diagnostics (``None`` for a model fitted before engiNIR 1.3:
    refit it), :meth:`interval_map` the prediction intervals,
    ``diagnostics`` the 0.99 limits and the summary, and ``provenance`` the
    record also written to the log.
    """

    def __init__(self, model: Model, result: PredictionResult) -> None:
        self.model = model
        self.result = result

    def __repr__(self) -> str:
        return f"Prediction(model={self.model.name!r}, file={self.result.dataset.source_path!r})"

    @property
    def y_pred(self) -> np.ndarray:
        """Predictions, one per spectrum in file order."""
        return _required(self.result.y_pred)

    @property
    def map(self) -> np.ndarray:
        """Predictions as an image ``(frames, height)`` (with an output axis for several targets)."""
        return _required(self.result.spatial_map)

    @property
    def dmodx_map(self) -> np.ndarray | None:
        """DModXPS per pixel as an image, or ``None`` without projection data."""
        return self.result.diagnostic_map(self.result.dmodx)

    @property
    def t2_map(self) -> np.ndarray | None:
        """T²PS per pixel as an image, or ``None`` without projection data."""
        return self.result.diagnostic_map(self.result.t2)

    def interval_map(self, level: float = 0.95) -> np.ndarray | None:
        """Half-width of each pixel's prediction interval at ``level``, as an image.

        The interval is ŷ ± q · s · √(1 + h) with s the model's pixel RMSEE
        and h the leverage from T²PS (ASTM E1655); pixels outside the model
        (above DCrit or T²Crit at 0.99) are NaN. ``None`` when the model has
        no interval inputs (prediction-runtime contract, Prediction
        Intervals). Computed on demand, never stored.
        """
        return self.result.diagnostic_map(self.result.half_width(level))

    @property
    def diagnostics(self) -> dict[str, Any] | None:
        """Limits (``t2_crit_099``, ``dcrit_099``) and summary over the scan."""
        return self.result.diagnostics

    @property
    def provenance(self) -> dict[str, Any]:
        return self.result.provenance


def _required(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        log_and_raise("The prediction holds no arrays", depth=2)
    return values


class StoredPrediction:
    """A prediction persisted by the low-level writers (``enginir.io.project_store``)."""

    def __init__(self, project: Project, ref: PredictionRun) -> None:
        self.project = project
        self.ref = ref

    def __repr__(self) -> str:
        return f"StoredPrediction({self.ref.prediction_run_id!r}, dataset={self.dataset.path!r})"

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


def read_header(path: str):
    """Return a file's :class:`~enginir.io.SpectralData` without its pixels where possible.

    EIFF and HDF5 files are read header-only: ``cube_shape``, wavelengths,
    metadata and the pixel dtype are known, ``X`` holds no frames. Other
    formats are read completely.
    """
    if file_format(path) in _FRAME_FORMATS:
        return load_spectra(path, header_only=True)
    return load_spectra(path)


def read_frames(path: str, first: int, last: int) -> np.ndarray:
    """Return frames ``first..last`` (inclusive) as ``(frames, height, n_bands)``.

    EIFF and HDF5 files read only those frames; other formats are read
    completely and sliced. Values keep the file's dtype; point-spectrum files
    get a height of 1.
    """
    if file_format(path) in _FRAME_FORMATS:
        X = load_spectra(path, frames=slice(first, last + 1)).X
    else:
        X = load_spectra(path).X[first:last + 1]
    X = np.asarray(X)
    return X[:, np.newaxis, :] if X.ndim == 2 else X


#: Longest folder part of a data file label before it is shortened in the middle.
_LABEL_FOLDER_CHARS = 40


def data_file_labels(datasets: Mapping[str, DatasetRef], *,
                     shorten: bool = True) -> dict[str, str]:
    """A short, unique display label per dataset id (viewer identity rule).

    A data file is labelled by its file name. When several data files share
    a file name, each gets the folder that tells it apart, e.g.
    ``stream_hsi.h5 (API 10% - nozzle 0.6 mm)``: the parent folder, or more
    of the path where parent folders repeat too. Long folders are shortened
    in the middle unless that would make two labels equal (``shorten=False``
    keeps them whole, for lists with room). Labels are for display only;
    datasets are identified by id.
    """
    paths = {dataset_id: _native_path(ref.source_path).replace("\\", "/").split("/")
             if ref.source_path else [dataset_id] for dataset_id, ref in datasets.items()}
    by_name: dict[str, list[str]] = {}
    for dataset_id, parts in paths.items():
        by_name.setdefault(parts[-1].lower(), []).append(dataset_id)
    labels: dict[str, str] = {}
    for ids in by_name.values():
        if len(ids) == 1:
            labels[ids[0]] = paths[ids[0]][-1]
            continue
        depth = 1  # folders above the file needed to tell this group apart
        while depth < max(len(paths[i]) - 1 for i in ids) and len(
                {"/".join(paths[i][-1 - depth:-1]) for i in ids}) < len(ids):
            depth += 1
        folders = {i: "/".join(paths[i][-1 - depth:-1]) for i in ids}
        short = {i: _shorten_middle(folder, _LABEL_FOLDER_CHARS) if shorten else folder
                 for i, folder in folders.items()}
        if len(set(short.values())) < len(ids):
            short = folders
        for i in ids:
            labels[i] = f"{paths[i][-1]} ({short[i]})" if short[i] else paths[i][-1]
    return labels


def _shorten_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = limit - 1
    return f"{text[:keep - keep // 2]}…{text[-(keep // 2):]}"


def check_data_file_location(
    project_path: str | os.PathLike[str],
    dataset: DatasetRef,
    rois: Iterable[TrainingROI],
    path: str | os.PathLike[str],
) -> None:
    """Check that the file at ``path`` is the data file of ``dataset``; raise ``ValueError`` if not.

    The file must have the registered format, shape, modality (if the file
    states one) and wavelength axis, and for every ROI of ``rois`` its pixels
    at the ROI's ``spatial_bounds`` must equal the raw pixels stored in the
    project. Reads only the frames the ROIs cover; writes nothing.
    """
    abs_path = _native_path(path)
    if not os.path.isfile(abs_path):
        log_and_raise("{} does not exist", abs_path)
    header = read_header(abs_path)
    checks = [("format", file_format(abs_path), dataset.source_format),
              ("shape", tuple(int(n) for n in header.cube_shape), tuple(dataset.cube_shape))]
    if header.meta.get("source_modality"):
        checks.append(("modality", header.meta["source_modality"], dataset.source_modality))
    for label, now, then in checks:
        if now != then:
            log_and_raise("{} is not this data file: it has {} {!r}, the data file has {!r}",
                          abs_path, label, now, then)
    wavelengths = np.asarray(header.wavelengths, dtype=np.float32)
    if wavelengths.shape != np.shape(dataset.wavelengths) or not np.allclose(
            wavelengths, np.asarray(dataset.wavelengths, dtype=np.float32), atol=1e-6, rtol=0.0):
        log_and_raise("{} is not this data file: its wavelengths differ", abs_path)
    rois = list(rois)
    if not rois:
        logger.warning("Data file {} has no ROIs; only its format, shape and wavelengths "
                       "were compared", abs_path)
        return
    first = min(roi.spatial_bounds[0] for roi in rois)
    last = max(roi.spatial_bounds[1] for roi in rois)
    pixels = read_frames(abs_path, first, last)
    n_bands = dataset.cube_shape[2]
    for roi in rois:
        f0, f1, r0, r1 = roi.spatial_bounds
        found = store.as_roi_storage_array(
            pixels[f0 - first:f1 - first + 1, r0:r1 + 1].reshape(-1, n_bands))
        stored = store.load_roi_spectra(project_path, f"/rois/{roi.roi_id}/spectra")
        if found.shape != stored.shape or not np.array_equal(found, stored):
            log_and_raise("{} is not this data file: the pixels of ROI '{}' differ from the "
                          "stored ones", abs_path, roi.name)
    logger.info("{} matches data file '{}' ({} ROIs compared)", abs_path, dataset.dataset_id,
                len(rois))


@dataclass(frozen=True)
class GridCell:
    """One cell of an ROI grid: its grid position, pixel ranges and ROI name."""

    row: int
    column: int
    frames: tuple[int, int]
    rows: tuple[int, int]
    name: str
    #: How the grid was laid out, recorded in the ROI's provenance.
    grid: str = ""

    @property
    def position(self) -> str:
        """The cell's place in the tray, e.g. ``B4``."""
        return f"{_row_letters(self.row)}{self.column + 1}"


def roi_grid(
    *,
    n_rows: int,
    n_columns: int,
    size: tuple[int, int],
    origin: tuple[int, int] = (0, 0),
    spacing: tuple[int, int] = (0, 0),
    skip: Iterable[tuple[int, int]] = (),
    order: str = "rows",
    name_pattern: str = "{row}{column:02d}",
    file: str = "",
    image_size: tuple[int, int] | None = None,
) -> list[GridCell]:
    """Lay out a grid of equal ROIs (a tray of samples) in pixels.

    ``x`` runs along frames (the image's horizontal, scan axis) and ``y``
    along rows (across the sensor). Columns step along ``x``, rows along
    ``y``. Cell ``(r, c)`` covers frames ``origin_x + c * (width + spacing_x)``
    onwards for ``width`` frames, and rows ``origin_y + r * (height +
    spacing_y)`` onwards for ``height`` rows.

    Parameters
    ----------
    n_rows, n_columns:
        Grid size.
    size:
        ``(width, height)`` of every ROI, in frames and rows.
    origin:
        ``(x, y)`` of the first ROI's top-left pixel (frame, row).
    spacing:
        ``(x, y)`` gap between neighbouring ROIs; ``0`` makes them touch.
    skip:
        ``(row, column)`` positions (0-based) without a sample.
    order:
        ``"rows"`` (row by row) or ``"columns"``: the order of the cells and
        of ``{n}``.
    name_pattern:
        Format string with ``{row}`` (``A``, ``B``, ..., ``AA``),
        ``{column}`` (1-based), ``{n}`` (1-based running number in ``order``)
        and ``{file}``.
    file:
        Value of ``{file}``, e.g. the data file's name without extension.
    image_size:
        ``(n_frames, height)`` of the image; every created cell must lie
        inside it.

    Returns
    -------
    list of GridCell
    """
    n_rows, n_columns = int(n_rows), int(n_columns)
    width, height = (int(v) for v in size)
    x0, y0 = (int(v) for v in origin)
    gap_x, gap_y = (int(v) for v in spacing)
    if n_rows < 1 or n_columns < 1:
        log_and_raise("An ROI grid needs at least one row and column, got {} x {}", n_rows,
                      n_columns)
    if width < 1 or height < 1:
        log_and_raise("ROIs need a size of at least 1 x 1 pixel, got {} x {}", width, height)
    if x0 < 0 or y0 < 0:
        log_and_raise("The grid origin must not be negative, got ({}, {})", x0, y0)
    if gap_x < 0 or gap_y < 0:
        log_and_raise("The spacing must not be negative (ROIs would overlap), got ({}, {})",
                      gap_x, gap_y)
    if order not in ("rows", "columns"):
        log_and_raise("order must be 'rows' or 'columns', got {!r}", order)
    grid = (f"{n_rows} x {n_columns}, size {width} x {height}, origin ({x0}, {y0}), "
            f"spacing ({gap_x}, {gap_y})")
    skipped = {(int(r), int(c)) for r, c in skip}
    positions = [(r, c) for r in range(n_rows) for c in range(n_columns)]
    if order == "columns":
        positions.sort(key=lambda rc: (rc[1], rc[0]))
    cells: list[GridCell] = []
    for r, c in positions:
        if (r, c) in skipped:
            continue
        f0, r0 = x0 + c * (width + gap_x), y0 + r * (height + gap_y)
        try:
            name = name_pattern.format(row=_row_letters(r), column=c + 1, n=len(cells) + 1,
                                       file=file)
        except (KeyError, IndexError, ValueError) as exc:
            log_and_raise("Name pattern {!r} is not valid: {}", name_pattern, exc, cause=exc)
        cell = GridCell(row=r, column=c, frames=(f0, f0 + width - 1),
                        rows=(r0, r0 + height - 1), name=name, grid=grid)
        if image_size is not None and (cell.frames[1] >= image_size[0]
                                       or cell.rows[1] >= image_size[1]):
            log_and_raise("ROI {} ({}) lies outside the image of {} frames x {} rows",
                          cell.position, name, image_size[0], image_size[1])
        cells.append(cell)
    names = [cell.name for cell in cells]
    if len(set(names)) != len(names):
        log_and_raise("Name pattern {!r} gives repeated names; use {{row}} and {{column}} or "
                      "{{n}}", name_pattern)
    return cells


def _row_letters(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def recipe_id_for(recipe: PreprocessingRecipe | Mapping[str, Any] | None) -> str:
    """The content-derived id of a preprocessing recipe: the same steps give the same id."""
    return _recipe_id(_as_recipe(recipe))


def resolve_signal_domain(candidate: SignalDomainContract,
                          signal_domains: Mapping[str, SignalDomainContract]
                          ) -> SignalDomainContract:
    """Return the stored signal domain with the content of ``candidate``, or ``candidate``.

    Refitting with the same data and signal unit reuses the stored contract
    instead of adding an equal one with a new id.
    """
    return _existing(signal_domains.values(), candidate, _same_signal_domain)


def resolve_sample_set(
    *,
    rois: Mapping[str, TrainingROI],
    target_assignments: Mapping[tuple[str, str], TargetAssignment],
    target_id: str,
    recipe_id: str,
    sample_sets: Mapping[str, SampleSet],
) -> tuple[SampleSet, bool]:
    """The sample set of every ROI with a value for ``target_id``, and whether it is new.

    A stored sample set with the same ROIs, target and recipe is reused;
    otherwise a new one is returned (not yet stored). No spectra are read.

    Raises
    ------
    ValueError
        If no ROI has a value for the target, or the labelled ROIs have
        different wavelength axes (see :func:`enginir.modeling.plan_sample_set`).
    """
    planned = plan_sample_set(rois=dict(rois), target_assignments=dict(target_assignments),
                              target_id=target_id, preprocessing_recipe_id=recipe_id)
    stored = _existing(sample_sets.values(), planned, _same_sample_set)
    return stored, stored is planned


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
    grid_cell: GridCell | None = None,
    use: str = "calibration",
) -> tuple[TrainingROI, np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    """Build an ROI of ``dataset`` from its frames ``pixels`` ``(frames, height, n_bands)``.

    ``frames`` and ``rows`` are inclusive ``(first, last)`` ranges; ``pixels``
    holds exactly the frames ``frames``. Returns the ROI, its spectra
    ``(n_pixels, n_bands)`` in the file's dtype and, for calibrations with
    reference spectra, the matching dark and white reference pixels.
    ``preview_recipe`` records the preprocessing shown while the ROI was
    drawn; ``grid_cell`` records the grid position and layout of an ROI made
    by :func:`roi_grid`.
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
        roi_id=new_id(), use=check_roi_use(use), dataset_id=dataset.dataset_id,
        calibration_id=calibration.calibration_id, name=name,
        wavelengths=dataset.wavelengths, n_pixels=int(spectra.shape[0]),
        spatial_bounds=(f0, f1, r0, r1), source_file=dataset.source_path, created_at=created_at,
        provenance={"dataset_id": dataset.dataset_id,
                    "calibration_id": calibration.calibration_id,
                    "signal_unit": calibration.output_signal_unit,
                    "preview_recipe_summary": _recipe_label(recipe) if preview_recipe else "none",
                    "created_at": created_at, **(recipe.to_dict() if preview_recipe else {}),
                    **({"grid_position": grid_cell.position, "grid": grid_cell.grid}
                       if grid_cell is not None else {})},
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


def _frame_range(frames: tuple[int, int], n_frames: int, label: str) -> tuple[int, int]:
    first, last = (int(frames[0]), int(frames[1]))
    if not 0 <= first <= last < n_frames:
        log_and_raise("The {} reference frames {}..{} are outside the file's {} frames", label,
                      first, last, n_frames, depth=3)
    return first, last


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
    steps = ["AsLS"] if recipe.asls else []
    if recipe.filter != "none":
        sg = {"sg_smooth": "SG", "sg_d1": "SG 1st der.", "sg_d2": "SG 2nd der."}[recipe.filter]
        steps.append(sg if recipe.sg_edges == "drop" else f"{sg} (edges kept)")
    steps += [label for label, on in (("SNV", recipe.snv), ("MSC", recipe.msc),
                                      ("EMSC", recipe.emsc), ("detrend", recipe.detrend),
                                      ("bands", recipe.has_band_selection)) if on]
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


def _by_id_or_name(name: object, object_id: str | None, kind: str) -> bool:
    """Whether a lookup is by id; exactly one of name and id must be given."""
    if (name is None) == (object_id is None):
        log_and_raise("Give the {}'s name or its id, not both or neither", kind,
                      error=TypeError, depth=3)
    return object_id is not None


def _with_id(objects: Mapping[str, Any], object_id: str | None, kind: str):
    if object_id not in objects:
        log_and_raise("No {} with id '{}'", kind, object_id, error=KeyError, depth=3)
    return objects[str(object_id)]


def _by_name(objects, kind: str, id_field: str, key=lambda obj: obj.name) -> dict[str, Any]:
    """Objects by name; refuses repeated names instead of silently keeping one of them."""
    named: dict[str, Any] = {}
    repeated: dict[str, list[str]] = {}
    for obj in objects:
        name = key(obj)
        if name in named:
            repeated.setdefault(name, [getattr(named[name], id_field)]).append(
                getattr(obj, id_field))
        named[name] = obj
    if repeated:
        log_and_raise("{} names repeat ({}); use the objects by id instead (project.workspace)",
                      kind, "; ".join(f"'{name}': {ids}" for name, ids in repeated.items()),
                      error=KeyError, depth=3)
    return named


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


def _metric_differences(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Numeric metrics that differ between two model runs: ``{name: (before, after)}``."""
    differences: dict[str, Any] = {}
    for key, value in before.items():
        other = after.get(key)
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            if not isinstance(other, (int, float, np.number)) or not np.isclose(
                    float(value), float(other), rtol=1e-6, atol=1e-12, equal_nan=True):
                differences[key] = (value, other)
    return differences


def _summary(metrics: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={metrics[key]:.3g}" for key in ("rmsecv", "r2", "bias")
                     if isinstance(metrics.get(key), (int, float)))
