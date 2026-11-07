"""Building sample sets and fitting model runs from project ROIs.

Modeling follows the fixed v3 semantics of :class:`~enginir.project.SampleSet`:
every ROI pixel is one observation and carries the ROI's target value,
cross-validation keeps all pixels of an ROI in the same fold (ROI groups), and
figures of merit are reported per ROI from the mean prediction of its pixels.
Pixel-level metrics are kept as secondary diagnostics.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from sklearn.model_selection import GroupKFold

from enginir._checks import log_and_raise
from enginir._logging import get_logger

from ..io.project_store import (
    iter_roi_calibration_refs,
    load_roi_spectra_batch,
    load_roi_spectra_dtypes,
    load_roi_spectra_matrix,
)
from ..models import PCR, PLSRegression
from ..preprocessing import PreprocessingRecipe
from ..preprocessing._pipeline import (
    apply_preprocessing_recipe,
    fitted_preprocessing_state,
    make_model_pipeline,
    transform_preprocessing_steps,
)
from ..project import (
    CalibrationSpec,
    CalibrationStats,
    DatasetRef,
    ModelRun,
    SampleSet,
    SignalDomainContract,
    TargetAssignment,
    TrainingROI,
    check_signal_domain_compatibility,
    transform_spectra_to_signal_domain,
)
from ..project._fit_spec import FitSpec, fold_groups_problem
from ..project._ids import new_id, now_iso
from ..project._models import SAMPLE_SET_SEMANTICS
from ._pixels import draw_pixels, roi_digest, sample_checksum
from ..validation import (
    RoundRobinGroups,
    aggregate_regression_predictions,
    bias,
    cross_validate,
    r2_score,
    rmsecv,
    rpd,
    rpiq,
)

logger = get_logger(__name__)

_WORKING_DTYPE = np.float32
_ESTIMATORS = {"PLS": PLSRegression, "PCR": PCR}


@dataclass(slots=True)
class SampleSetData:
    """A sample set with its modeling arrays.

    Attributes
    ----------
    sample_set : SampleSet
        The sample-set description (to be stored in the project).
    X : numpy.ndarray
        Raw ROI pixel spectra as stored (uint16 counts or float32), one row
        per pixel.
    y : numpy.ndarray
        Target value of each pixel's ROI (float32 working dtype).
    groups : numpy.ndarray
        ROI id of each pixel (cross-validation groups).
    wavelengths : numpy.ndarray
        Common wavelength axis of the ROIs (float32).
    roi_slices : dict
        ``{roi_id: slice}`` with the rows of every ROI.
    pixel_indices : dict or None
        For a pixel sample, ``{roi_id: indices}`` of the drawn pixels in each
        ROI's stored spectra (and reference snapshot); ``None`` for all pixels.
    """

    sample_set: SampleSet
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    wavelengths: np.ndarray
    roi_slices: dict[str, slice]
    pixel_indices: dict[str, np.ndarray] | None = None


@dataclass(slots=True)
class ModelRunResult:
    """Outcome of :func:`fit_model_run`.

    Attributes
    ----------
    model_run : ModelRun
        Run description with ROI-level ``metrics`` (``rmsecv``, ``r2``,
        ``bias``, ``rpd``, ``rpiq``) and pixel-level ``pixel_*`` metrics, the
        ``cv_spec`` and the saved ``artifact_refs``.
    estimator : object
        The final PLS or PCR model fitted on all pixels.
    predictor : object
        ``estimator`` combined with the fitted preprocessing; predicts
        signal-domain spectra.
    sample_set_data : SampleSetData
        Training data.
    pixel_y_true, pixel_y_pred : numpy.ndarray
        Reference values and cross-validated predictions per pixel (float32).
    roi_ids : numpy.ndarray
        ROI ids in order of first appearance.
    roi_y_true, roi_y_pred : numpy.ndarray
        Reference value and mean cross-validated prediction per ROI (float32).
    """

    model_run: ModelRun
    estimator: Any
    predictor: Any
    sample_set_data: SampleSetData
    pixel_y_true: np.ndarray
    pixel_y_pred: np.ndarray
    roi_ids: np.ndarray
    roi_y_true: np.ndarray
    roi_y_pred: np.ndarray


@dataclass(slots=True)
class _StatelessRecipePredictor:
    """Estimator preceded by already-built stateless preprocessing steps (no refitting)."""

    recipe_aware = True

    steps: list[tuple[str, Any]]
    estimator: Any

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(transform_preprocessing_steps(X, self.steps))


def plan_sample_set(
    *,
    rois: Mapping[str, TrainingROI],
    target_assignments: Mapping[tuple[str, str], TargetAssignment],
    target_id: str,
    preprocessing_recipe_id: str,
    roi_ids: list[str] | None = None,
    sample_set_id: str | None = None,
    created_at: str | None = None,
) -> SampleSet:
    """The sample set of the labelled ROIs of a target, without reading any spectra.

    Applies the same selection rules as :func:`build_sample_set`: ROIs
    without a value for the target are left out, and all ROIs must share
    one wavelength axis.

    Raises
    ------
    ValueError
        If an ROI id is unknown or repeated, no selected ROI has a target
        value, or the ROIs have different wavelength axes.
    """
    selected = list(rois) if roi_ids is None else [str(roi_id) for roi_id in roi_ids]
    if not selected:
        log_and_raise("No ROIs selected for the sample set")
    repeated = sorted(roi_id for roi_id, count in Counter(selected).items() if count > 1)
    if repeated:
        log_and_raise("ROI(s) selected more than once for the sample set: {}", repeated)
    labelled: list[TrainingROI] = []
    unlabelled: list[str] = []
    for roi_id in selected:
        if roi_id not in rois:
            log_and_raise("Unknown ROI '{}' selected for the sample set", roi_id)
        if (roi_id, target_id) in target_assignments:
            labelled.append(rois[roi_id])
        else:
            unlabelled.append(roi_id)
    if not labelled:
        log_and_raise("No selected ROI has a value for target '{}'", target_id)
    if unlabelled:
        logger.warning("Skipping {} ROI(s) without a value for target '{}': {}",
                       len(unlabelled), target_id, unlabelled)
    wavelengths = labelled[0].wavelengths
    for roi in labelled[1:]:
        if roi.wavelengths.shape != wavelengths.shape or not np.allclose(
            roi.wavelengths, wavelengths, atol=1e-6, rtol=0.0
        ):
            log_and_raise("ROI '{}' has a different wavelength axis than ROI '{}'", roi.name,
                          labelled[0].name)
    return SampleSet(
        sample_set_id=sample_set_id or new_id(),
        roi_ids=[roi.roi_id for roi in labelled],
        target_id=target_id,
        preprocessing_recipe_id=preprocessing_recipe_id,
        dataset_ids=list(dict.fromkeys(roi.dataset_id for roi in labelled)),
        created_at=created_at or now_iso(),
        **SAMPLE_SET_SEMANTICS,
    )


def build_sample_set(
    *,
    project_path: str,
    rois: dict[str, TrainingROI],
    target_assignments: dict[tuple[str, str], TargetAssignment],
    target_id: str,
    preprocessing_recipe_id: str,
    roi_ids: list[str] | None = None,
    sample_set_id: str | None = None,
    created_at: str | None = None,
) -> SampleSetData:
    """Collect the labelled ROIs of a target into a sample set and read their spectra.

    Parameters
    ----------
    project_path:
        Project file with the ROI spectra.
    rois:
        ROIs by id.
    target_assignments:
        ``{(roi_id, target_id): TargetAssignment}``.
    target_id, preprocessing_recipe_id:
        Target to model and recipe to apply.
    roi_ids:
        ROIs to consider (default: all), each at most once; ROIs without a
        value for the target are skipped with a warning.
    sample_set_id, created_at:
        Id and timestamp of the new sample set (generated by default).

    Returns
    -------
    SampleSetData

    Raises
    ------
    ValueError
        If an ROI id is unknown or repeated, no selected ROI has a target
        value, or the ROIs have different wavelength axes.
    """
    sample_set = plan_sample_set(rois=rois, target_assignments=target_assignments,
                                 target_id=target_id,
                                 preprocessing_recipe_id=preprocessing_recipe_id,
                                 roi_ids=roi_ids, sample_set_id=sample_set_id,
                                 created_at=created_at)
    labelled = [(rois[roi_id], target_assignments[(roi_id, target_id)])
                for roi_id in sample_set.roi_ids]
    wavelengths = labelled[0][0].wavelengths.astype(np.float32)

    X, rows = load_roi_spectra_matrix(
        project_path, [roi.spectra_storage_key for roi, _ in labelled], n_bands=wavelengths.size
    )
    y = np.empty(X.shape[0], dtype=_WORKING_DTYPE)
    groups = np.empty(X.shape[0], dtype=object)
    roi_slices: dict[str, slice] = {}
    for roi, assignment in labelled:
        roi_rows = rows[roi.spectra_storage_key]
        y[roi_rows] = assignment.value
        groups[roi_rows] = roi.roi_id
        roi_slices[roi.roi_id] = roi_rows

    logger.info("Built sample set '{}' for target '{}': {} ROIs, {} pixels",
                sample_set.sample_set_id, target_id, len(labelled), X.shape[0])
    return SampleSetData(sample_set, X, y, groups, wavelengths, roi_slices)


def fit_model_run(
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
    model_type: str,
    hyperparameters: dict[str, Any] | None = None,
    cv: int | str | None = None,
    fit_spec: FitSpec | Mapping[str, Any] | None = None,
    model_run_id: str | None = None,
    created_at: str | None = None,
    target_name: str | None = None,
) -> ModelRunResult:
    """Cross-validate and fit a PLS or PCR model on a sample set.

    The ROI spectra (all pixels, or the pixels drawn by the fit
    specification) are converted to the signal domain (with each ROI's
    calibration), cross-validated with ROI groups, and a final model is
    fitted on all of them. A recipe without MSC/EMSC works spectrum by spectrum
    and is applied once beforehand. A recipe with MSC or EMSC runs completely
    inside every training fold, so that their references are fitted on
    training pixels only.

    Parameters
    ----------
    project_path:
        Project file with the ROI spectra and reference snapshots.
    sample_set:
        Sample set to fit.
    datasets, calibrations, rois, target_assignments, recipes, signal_domains:
        Project objects by id (as in :class:`~enginir.project.ProjectWorkspace`).
    signal_domain_id:
        Signal domain of the training spectra; its wavelengths must match
        the sample set.
    model_type:
        ``"PLS"`` or ``"PCR"``.
    hyperparameters:
        Estimator parameters, e.g. ``{"n_components": 5}``.
    cv:
        Short form of the fold rule: ``"auto"`` or ``None`` (round-robin over
        7 folds, :class:`~enginir.validation.RoundRobinGroups`), an integer
        number of round-robin folds, or ``"logo"`` (leave one ROI out).
        Refused together with a ``fit_spec`` that sets the folds.
    fit_spec:
        The fit specification (:class:`~enginir.project.FitSpec` or its
        mapping): fold rule, pixel sample and exclusion
        (``sample-set-model-run-contract_v3.md`` §7). The stored run gets the
        complete specification with the checksum of its pixel sample.
        Folds follow the order of the ROIs in the sample set and the pixel
        sample depends only on the seed and the ROI spectra, so repeating
        the analysis gives the same folds and metrics.
    model_run_id, created_at:
        Id and timestamp of the new model run (generated by default).
    target_name:
        Name stored with the model output (default: the target id).

    Returns
    -------
    ModelRunResult

    Raises
    ------
    ValueError
        If the inputs are inconsistent, ``cv`` is not ROI-grouped, or the
        model type is unknown.
    """
    spec, requested = resolve_fit_spec(cv, fit_spec)
    model_type = _model_type(model_type)
    if sample_set.preprocessing_recipe_id not in recipes:
        log_and_raise("Sample set '{}' references the unknown recipe '{}'",
                      sample_set.sample_set_id, sample_set.preprocessing_recipe_id)
    if signal_domain_id not in signal_domains:
        log_and_raise("Unknown signal domain '{}'", signal_domain_id)
    signal_domain = signal_domains[signal_domain_id]
    _check_fit_spec_feasible(spec, sample_set, rois)
    groups = _resolved_groups(spec, sample_set, rois, target_assignments)
    n_fold_groups = int(np.unique(groups).size)
    if n_fold_groups < 2:
        log_and_raise("Cross-validation needs at least 2 fold groups; grouping the {} ROIs of "
                      "sample set '{}' by {} gives {}", len(sample_set.roi_ids),
                      sample_set.sample_set_id, spec.group_by, n_fold_groups)
    spec = spec.with_groups(groups)

    data = build_sample_set(
        project_path=project_path,
        rois=rois,
        target_assignments=target_assignments,
        target_id=sample_set.target_id,
        preprocessing_recipe_id=sample_set.preprocessing_recipe_id,
        roi_ids=sample_set.roi_ids,
        sample_set_id=sample_set.sample_set_id,
        created_at=sample_set.created_at,
    )
    unlabelled = [roi_id for roi_id in sample_set.roi_ids if roi_id not in data.roi_slices]
    if unlabelled:
        log_and_raise("Sample set '{}' includes ROI(s) without a value for target '{}': {}; "
                      "build a new sample set", sample_set.sample_set_id, sample_set.target_id,
                      unlabelled)
    if signal_domain.wavelengths.shape != data.wavelengths.shape or not np.allclose(
        signal_domain.wavelengths, data.wavelengths, atol=1e-6, rtol=0.0
    ):
        log_and_raise("Signal domain '{}' has a different wavelength axis than sample set '{}'",
                      signal_domain_id, sample_set.sample_set_id)
    data, checksum = _pixel_sample(data, spec, project_path, rois)
    spec = spec.with_checksum(checksum)
    X_signal = _signal_domain_matrix(project_path, data, datasets, calibrations, rois,
                                     signal_domain)

    recipe = PreprocessingRecipe.from_dict(recipes[sample_set.preprocessing_recipe_id])
    hyperparameters = _with_default_scaling(model_type, dict(hyperparameters or {}))
    # Stateless recipes are applied once; MSC/EMSC must be fitted inside each fold.
    if recipe.requires_fitting:
        cv_estimator: Any = make_model_pipeline(recipe=recipe, wavelengths=data.wavelengths,
                                                estimator=_ESTIMATORS[model_type](**hyperparameters))
        X_cv = X_signal
    else:
        cv_estimator = _ESTIMATORS[model_type](**hyperparameters)
        X_cv = apply_preprocessing_recipe(X_signal, recipe=recipe, wavelengths=data.wavelengths)
        del X_signal  # only the preprocessed matrix is needed from here on
    # ROI ids are random, so the splitter gets each ROI's fold-group index
    # (its position in the sample set with group_by "roi") instead: folds
    # follow the recorded ROI order or target values and are the same
    # whenever the analysis is repeated (contract §4.1).
    roi_ids_present, roi_index = np.unique(data.groups, return_inverse=True)
    group_of = {roi_id: int(groups[i]) for i, roi_id in enumerate(sample_set.roi_ids)}
    cv_groups = np.array([group_of[roi_id] for roi_id in roi_ids_present])[roi_index]
    cv_result = cross_validate(cv_estimator, X_cv, data.y,
                               cv=_splitter(spec, n_fold_groups), groups=cv_groups)
    pixel_y_pred = np.asarray(cv_result["y_pred"], dtype=_WORKING_DTYPE)
    roi_ids, roi_y_true, roi_y_pred = aggregate_regression_predictions(
        data.y, pixel_y_pred, data.groups, aggregation=sample_set.roi_aggregation
    )
    roi_y_true = roi_y_true.astype(_WORKING_DTYPE)
    roi_y_pred = roi_y_pred.astype(_WORKING_DTYPE)

    estimator = _ESTIMATORS[model_type](**hyperparameters)
    if recipe.requires_fitting:
        predictor: Any = make_model_pipeline(recipe=recipe, wavelengths=data.wavelengths,
                                             estimator=estimator)
        predictor.fit(X_cv, data.y)  # X_cv is X_signal: MSC/EMSC are refitted here
    else:
        estimator.fit(X_cv, data.y)
        predictor = _StatelessRecipePredictor(recipe.build_steps(data.wavelengths), estimator)

    model_run = ModelRun(
        model_run_id=model_run_id or new_id(),
        sample_set_id=sample_set.sample_set_id,
        signal_domain_id=signal_domain_id,
        model_type=model_type,
        hyperparameters=hyperparameters,
        cv_spec={
            **cv_result["cv_scheme"],
            "requested": requested,
            "n_groups": n_fold_groups,
            **{key: SAMPLE_SET_SEMANTICS[key] for key in
               ("grouping_unit", "validation_unit", "reporting_level", "roi_aggregation")},
        },
        metrics={
            "rmsecv": rmsecv(roi_y_true, roi_y_pred),
            "r2": r2_score(roi_y_true, roi_y_pred),
            "bias": bias(roi_y_true, roi_y_pred),
            "rpd": rpd(roi_y_true, roi_y_pred),
            "rpiq": rpiq(roi_y_true, roi_y_pred),
            "primary_reporting_level": sample_set.reporting_level,
            "primary_roi_aggregation": sample_set.roi_aggregation,
            "n_rois": int(roi_ids.size),
            "n_pixels": int(data.X.shape[0]),
            "pixel_rmsecv": rmsecv(data.y, pixel_y_pred),
            "pixel_r2": r2_score(data.y, pixel_y_pred),
            "pixel_bias": bias(data.y, pixel_y_pred),
            "pixel_rpd": rpd(data.y, pixel_y_pred),
            "pixel_rpiq": rpiq(data.y, pixel_y_pred),
        },
        artifact_refs={
            "prediction_model": {
                "model_type": type(estimator).__name__,
                "wavelengths": data.wavelengths.astype(_WORKING_DTYPE),
                "target_names": [target_name or sample_set.target_id],
                "coef_": np.asarray(estimator.coef_, dtype=_WORKING_DTYPE),
                "intercept_": np.atleast_1d(np.asarray(estimator.intercept_, _WORKING_DTYPE)),
                **_preprocessing_state_entry(predictor, recipe),
            },
            "diagnostics": _diagnostics_artifact(estimator, data.wavelengths),
            "cv_outputs": {"roi_ids": roi_ids, "y_true_roi": roi_y_true,
                           "y_pred_roi": roi_y_pred},
        },
        created_at=created_at or now_iso(),
        fit_spec=spec.to_dict(),
    )
    logger.info("Fitted model run '{}' ({}) on sample set '{}': {} ROIs, {} pixels, RMSECV {:.4g}",
                model_run.model_run_id, model_type, sample_set.sample_set_id, roi_ids.size,
                data.X.shape[0], model_run.metrics["rmsecv"])
    return ModelRunResult(
        model_run=model_run,
        estimator=estimator,
        predictor=predictor,
        sample_set_data=data,
        pixel_y_true=data.y,
        pixel_y_pred=pixel_y_pred,
        roi_ids=roi_ids,
        roi_y_true=roi_y_true,
        roi_y_pred=roi_y_pred,
    )


def resolve_fit_spec(cv: int | str | None,
                     fit_spec: FitSpec | Mapping[str, Any] | None) -> tuple[FitSpec, Any]:
    """The fit specification of ``cv`` and ``fit_spec``, and the ``cv_spec/requested`` value.

    ``cv`` is the short form of the fold rule (contract §7); combined with a
    ``fit_spec`` it may only be given when that specification keeps the
    default folds, and then sets them.
    """
    if fit_spec is None:
        spec = FitSpec.from_cv(cv)
        return spec, "auto" if cv is None else cv
    spec = FitSpec.coerce(fit_spec)
    if cv is None:
        return spec, "fit_spec"
    if spec.sets_folds:
        log_and_raise("Give the folds either as cv={!r} or in fit_spec (rule {!r}, {} folds), "
                      "not both", cv, spec.fold_rule, spec.n_folds)
    folds = FitSpec.from_cv(cv)
    return replace(spec, fold_rule=folds.fold_rule, n_folds=folds.n_folds), cv


def _check_fit_spec_feasible(spec: FitSpec, sample_set: SampleSet,
                             rois: dict[str, TrainingROI]) -> None:
    """Block a fit whose specification cannot be met, before any spectra are read."""
    if len(sample_set.roi_ids) < 2:
        log_and_raise("A model needs at least 2 ROIs for cross-validation, sample set '{}' has {}",
                      sample_set.sample_set_id, len(sample_set.roi_ids))
    if spec.exclusion != "none":
        log_and_raise("Pixel exclusion '{}' is not available yet", spec.exclusion)
    if spec.pixels_per_roi is not None:
        known = [rois[roi_id] for roi_id in sample_set.roi_ids if roi_id in rois]
        smallest = min(known, key=lambda roi: roi.n_pixels, default=None)
        if smallest is not None and spec.pixels_per_roi > smallest.n_pixels:
            log_and_raise("Cannot draw {} pixels per ROI: ROI '{}' has only {}; choose at most {}",
                          spec.pixels_per_roi, smallest.name, smallest.n_pixels,
                          smallest.n_pixels)


def _with_default_scaling(model_type: str, hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Add the scaling defaults of new runs (contract §7) unless a scaling is given.

    Centred X, and for PLS unit-variance Y, the usual setting for spectra.
    Hyperparameters with ``scale`` (runs before engiNIR 1.2, refitted) are
    left as they are.
    """
    keys = ("scale", "x_scaling", "y_scaling") if model_type == "PLS" else ("scale", "x_scaling")
    if any(key in hyperparameters for key in keys):
        return hyperparameters
    defaults = {"x_scaling": "center", "y_scaling": "uv"} if model_type == "PLS" \
        else {"x_scaling": "center"}
    return {**hyperparameters, **defaults}


def fold_groups(group_by: str, sample_set: SampleSet, rois: dict[str, TrainingROI],
                target_assignments: dict[tuple[str, str], TargetAssignment]) -> np.ndarray:
    """Fold-group index of every ROI of ``sample_set``, in ``roi_ids`` order (contract §4.1).

    ``"roi"``: every ROI is its own group, in sample-set order. ``"sample"``:
    ROIs with the same sample label form one group (an ROI without a label is
    its own group), ordered by the position of the group's first ROI.
    ``"target"``: ROIs with the same stored value of the sample set's target
    form one group, ordered by value. Only equality of labels and the
    positions or values decide the groups, never the label text.
    """
    roi_ids = list(sample_set.roi_ids)
    if group_by == "roi":
        return np.arange(len(roi_ids), dtype=np.int64)
    if group_by == "sample":
        keys: list[Any] = []
        for roi_id in roi_ids:
            roi = rois.get(roi_id)
            label = roi.sample if roi is not None else None
            keys.append(("sample", label) if label is not None else ("roi", roi_id))
        first: dict[Any, int] = {}
        for key in keys:
            first.setdefault(key, len(first))
        return np.array([first[key] for key in keys], dtype=np.int64)
    if group_by == "target":
        values = []
        for roi_id in roi_ids:
            assignment = target_assignments.get((roi_id, sample_set.target_id))
            if assignment is None:
                log_and_raise("ROI '{}' has no value for target '{}'; it cannot be grouped by "
                              "target", roi_id, sample_set.target_id)
            values.append(float(assignment.value))
        levels = sorted(set(values))
        index = {value: j for j, value in enumerate(levels)}
        return np.array([index[value] for value in values], dtype=np.int64)
    log_and_raise("Unknown fold grouping {!r}", group_by)


def _resolved_groups(spec: FitSpec, sample_set: SampleSet, rois: dict[str, TrainingROI],
                     target_assignments: dict[tuple[str, str], TargetAssignment]) -> np.ndarray:
    """The groups the fit uses: those of ``spec`` (a refit), or computed from ``group_by``."""
    n_rois = len(sample_set.roi_ids)
    if spec.groups:
        problem = fold_groups_problem(spec.groups, spec.group_by, n_rois)
        if problem is not None:
            log_and_raise("folds/groups of the fit specification cannot be used for sample set "
                          "'{}': {}", sample_set.sample_set_id, problem)
        return np.asarray(spec.groups, dtype=np.int64)
    if spec.group_by == "sample":
        labels = [rois[roi_id].sample for roi_id in sample_set.roi_ids if roi_id in rois]
        present = sorted({label for label in labels if label is not None})
        if not present:
            logger.warning("Folds grouped by sample, but no ROI of sample set '{}' has a sample "
                           "label: every ROI is its own group", sample_set.sample_set_id)
        folded: dict[str, list[str]] = {}
        for label in present:
            folded.setdefault(label.casefold(), []).append(label)
        alike = [names for names in folded.values() if len(names) > 1]
        if alike:
            logger.warning("Sample labels that differ only in case form separate groups: {}; "
                           "if they are one sample, give them the same label", alike)
    return fold_groups(spec.group_by, sample_set, rois, target_assignments)


def _splitter(spec: FitSpec, n_groups: int) -> Any:
    """The cross-validation splitter of the fold rule over fold groups (contract §4, §7.3)."""
    if spec.fold_rule == "round_robin":
        return RoundRobinGroups(spec.n_folds)
    if spec.fold_rule == "leave_one_group_out" or n_groups < spec.n_folds:
        return "logo"
    return GroupKFold(n_splits=spec.n_folds)  # legacy runs (contract §7.3)


def _pixel_sample(data: SampleSetData, spec: FitSpec, project_path: str,
                  rois: dict[str, TrainingROI]) -> tuple[SampleSetData, str]:
    """Keep the pixels drawn by ``spec`` (contract §7.1); all pixels give an empty checksum."""
    if spec.pixels_per_roi is None:
        return data, ""
    roi_ids = list(data.sample_set.roi_ids)
    keys = {roi_id: rois[roi_id].spectra_storage_key for roi_id in roi_ids}
    stored_dtypes = load_roi_spectra_dtypes(project_path, keys.values())
    drawn: list[tuple[bytes, np.ndarray]] = []
    rows: list[np.ndarray] = []
    pixel_indices: dict[str, np.ndarray] = {}
    for roi_id in roi_ids:
        roi_rows = data.roi_slices[roi_id]
        stored = data.X[roi_rows].astype(stored_dtypes[keys[roi_id]], copy=False)
        digest = roi_digest(stored)
        indices = draw_pixels(stored, spec.pixels_per_roi, spec.seed, digest=digest)
        drawn.append((digest, indices))
        rows.append(roi_rows.start + indices)
        pixel_indices[roi_id] = indices
    take = np.concatenate(rows)
    n = spec.pixels_per_roi
    slices = {roi_id: slice(i * n, (i + 1) * n) for i, roi_id in enumerate(roi_ids)}
    sampled = SampleSetData(data.sample_set, data.X[take], data.y[take], data.groups[take],
                            data.wavelengths, slices, pixel_indices)
    logger.info("Drew {} pixels from each of {} ROIs (seed {})", n, len(roi_ids), spec.seed)
    return sampled, sample_checksum(drawn)


def aggregate_sample_set_predictions(
    *,
    sample_set_data: SampleSetData,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average pixel predictions per ROI.

    Returns
    -------
    roi_ids, roi_y_true, roi_y_pred : numpy.ndarray
        See :func:`enginir.validation.aggregate_regression_predictions`.
    """
    return aggregate_regression_predictions(
        sample_set_data.y, y_pred, sample_set_data.groups,
        aggregation=sample_set_data.sample_set.roi_aggregation,
    )


def _model_type(model_type: str) -> str:
    normalized = str(model_type).strip().upper()
    if normalized not in _ESTIMATORS:
        log_and_raise("Unsupported model type {!r}; use 'PLS' or 'PCR'", model_type, depth=2)
    return normalized


def _preprocessing_state_entry(predictor: Any, recipe: PreprocessingRecipe) -> dict[str, Any]:
    state = fitted_preprocessing_state(predictor, recipe=recipe)
    return {"preprocessing_state": state} if state else {}


def _diagnostics_artifact(model: Any, wavelengths: np.ndarray) -> dict[str, Any]:
    """Arrays for the diagnostics of a saved run (model-workspace diagnostics contract).

    Coefficients, loadings and training scores for every model; PLS adds the X
    weights and Y loadings, from which VIP can be computed without a refit.
    """
    names = (("coef_", "x_loadings_", "x_weights_", "y_loadings_", "x_scores_")
             if isinstance(model, PLSRegression) else ("coef_", "loadings_", "scores_"))
    artifact = {name: np.asarray(getattr(model, name), dtype=_WORKING_DTYPE) for name in names}
    artifact["wavelengths"] = np.asarray(wavelengths, dtype=_WORKING_DTYPE)
    return artifact


def roi_to_signal_domain(
    raw_spectra: np.ndarray,
    *,
    roi: TrainingROI,
    dataset: DatasetRef,
    calibration: CalibrationSpec,
    snapshot: tuple[np.ndarray, np.ndarray] | None,
    signal_domain: SignalDomainContract,
    stats: CalibrationStats | None = None,
) -> np.ndarray:
    """Convert one ROI's stored spectra to ``signal_domain``, exactly as model fitting does.

    Calibrations with per-pixel references use ``snapshot``, the ROI's stored
    ``(dark, white)`` reference spectra (one per ROI pixel). The fit and the
    viewer's ROI spectra both call this function.
    """
    check_signal_domain_compatibility(dataset=dataset, calibration=calibration,
                                      signal_domain=signal_domain)
    if calibration.dark_ref_data is not None:
        if snapshot is None:
            log_and_raise("ROI '{}' has no reference snapshot for calibration '{}'", roi.roi_id,
                          calibration.calibration_id)
        dark, white = snapshot
        if dark.shape != raw_spectra.shape or white.shape != raw_spectra.shape:
            log_and_raise("ROI '{}' reference snapshot shape {} does not match its spectra {}",
                          roi.roi_id, dark.shape, raw_spectra.shape)
        calibration = replace(calibration, dark_ref_data=dark, white_ref_data=white)
    return transform_spectra_to_signal_domain(
        raw_spectra, dataset=dataset, calibration=calibration, signal_domain=signal_domain,
        stats=stats, check_compatibility=False,
        per_spectrum_references=calibration.dark_ref_data is not None)


@dataclass(frozen=True)
class RoiSpectrumSummary:
    """Pixel spectra of one ROI in a signal unit and preprocessing, summarized per band."""

    roi_id: str
    mean: np.ndarray
    p05: np.ndarray
    median: np.ndarray
    p95: np.ndarray


def roi_spectrum_summaries(
    project_path: str,
    rois: Sequence[TrainingROI],
    *,
    datasets: Mapping[str, DatasetRef],
    calibrations: Mapping[str, CalibrationSpec],
    signal_unit: str,
    recipe: PreprocessingRecipe | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple[dict[str, RoiSpectrumSummary], dict[str, str]]:
    """Mean, median and 5–95 % spectra of each ROI as model fitting would see its pixels.

    Each ROI is converted with :func:`roi_to_signal_domain` (its own
    calibration and stored references) and preprocessed pixel by pixel with
    ``recipe``. ``signal_unit="DN"`` shows the stored raw counts of every ROI
    that has them. MSC and EMSC need a reference: here it is learned from the
    mean spectra of ``rois`` (model fitting learns it from the training
    pixels). ROIs are read one at a time, so memory stays bounded.

    Returns
    -------
    summaries : dict
        ``{roi_id: RoiSpectrumSummary}``.
    skipped : dict
        ``{roi_id: reason}`` for ROIs that cannot give ``signal_unit`` (for
        example raw-count ROIs asked for absorbance).
    """
    recipe = recipe or PreprocessingRecipe()
    signal: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    by_id = {roi.roi_id: roi for roi in rois}
    for roi_id, roi in by_id.items():
        if cancelled():
            break
        dataset, calibration = datasets[roi.dataset_id], calibrations[roi.calibration_id]
        key = f"/rois/{roi_id}/spectra"
        reason = _signal_unit_unavailable(calibration, signal_unit)
        if reason:
            skipped[roi_id] = reason
            continue
        if signal_unit == "DN":  # raw counts as stored; no conversion
            raw = load_roi_spectra_batch(project_path, [key])[key]
            signal[roi_id] = np.asarray(raw, dtype=_WORKING_DTYPE)
            continue
        try:
            domain = SignalDomainContract.for_training_data(
                dataset=dataset, calibration=calibration, required_signal_unit=signal_unit)
            raw = load_roi_spectra_batch(project_path, [key])[key]
            # One ROI's reference snapshot at a time: together they can be large.
            snapshot = next(iter(iter_roi_calibration_refs(project_path, [roi_id])))[1]
            signal[roi_id] = roi_to_signal_domain(
                raw, roi=roi, dataset=dataset, calibration=calibration, snapshot=snapshot,
                signal_domain=domain)
        except ValueError as exc:
            skipped[roi_id] = str(exc)
    summaries: dict[str, RoiSpectrumSummary] = {}
    if not signal:
        return summaries, skipped
    wavelengths = np.asarray(next(iter(by_id.values())).wavelengths)
    steps = recipe.build_steps(wavelengths)
    # Fit the steps in order on the ROI means: stateless steps ignore it, MSC/EMSC
    # learn their reference from it.
    fitted = np.stack([values.mean(axis=0) for values in signal.values()]).astype(_WORKING_DTYPE)
    for _name, step in steps:
        fitted = np.asarray(step.fit_transform(fitted), dtype=_WORKING_DTYPE)
    for roi_id, values in signal.items():
        pixels = transform_preprocessing_steps(values, steps)
        p05, median, p95 = np.percentile(pixels, [5, 50, 95], axis=0)
        summaries[roi_id] = RoiSpectrumSummary(
            roi_id, pixels.mean(axis=0), p05.astype(_WORKING_DTYPE),
            median.astype(_WORKING_DTYPE), p95.astype(_WORKING_DTYPE))
    return summaries, skipped


def _signal_unit_unavailable(calibration: CalibrationSpec, unit: str) -> str:
    """Why ROIs with ``calibration`` cannot be shown in ``unit`` ("" if they can)."""
    if calibration.mode == "precalibrated_import":
        return "" if unit != "DN" else "precalibrated data has no raw counts"
    if calibration.mode == "raw_dn" and unit != "DN":
        return f"raw counts without references cannot give {unit}"
    return ""


def _signal_domain_matrix(
    project_path: str,
    data: SampleSetData,
    datasets: dict[str, DatasetRef],
    calibrations: dict[str, CalibrationSpec],
    rois: dict[str, TrainingROI],
    signal_domain: SignalDomainContract,
) -> np.ndarray:
    """Convert the ROI spectra to the signal domain, each ROI with its own calibration.

    Calibrations with per-pixel references use the reference snapshot stored
    with each ROI (one reference spectrum per ROI pixel).
    """
    X_signal = np.empty(data.X.shape, dtype=_WORKING_DTYPE)
    stats = CalibrationStats()
    for roi_id, snapshot in iter_roi_calibration_refs(project_path, data.sample_set.roi_ids):
        roi = rois[roi_id]
        if roi.dataset_id not in datasets or roi.calibration_id not in calibrations:
            log_and_raise("ROI '{}' references an unknown dataset or calibration", roi_id)
        rows = data.roi_slices[roi_id]
        if snapshot is not None and data.pixel_indices is not None:
            picked = data.pixel_indices[roi_id]  # one reference spectrum per drawn pixel
            snapshot = (snapshot[0][picked], snapshot[1][picked])
        X_signal[rows] = roi_to_signal_domain(
            data.X[rows], roi=roi, dataset=datasets[roi.dataset_id],
            calibration=calibrations[roi.calibration_id], snapshot=snapshot,
            signal_domain=signal_domain, stats=stats)
    if stats.clip_low_count or stats.clip_high_count or stats.near_zero_denominator_count:
        logger.warning("Training signal-domain conversion of sample set '{}' clipped reflectance "
                       "values (clip_low={}, clip_high={}, near_zero_denominators={})",
                       data.sample_set.sample_set_id, stats.clip_low_count,
                       stats.clip_high_count, stats.near_zero_denominator_count)
    return X_signal


__all__ = [
    "ModelRunResult",
    "SampleSetData",
    "RoiSpectrumSummary",
    "aggregate_sample_set_predictions",
    "build_sample_set",
    "fit_model_run",
    "plan_sample_set",
    "roi_spectrum_summaries",
    "roi_to_signal_domain",
]
