"""Checks and verdicts: what a project needs next (headless-workflow contract, Guidance).

:func:`project_checks` lists what is done and what needs attention in each
step of the workflow (data, ROIs, values, model, predict); :func:`model_checks`
does so for one model, and :func:`model_verdict` condenses a model into one
level with its reasons. Everything is computed from the project file (and,
for the dataset files, from whether they are found), so scripts, the viewer
and reports show the same items. A check may name a :class:`Fix`: an action
with arguments that a caller (the viewer) can offer; nothing here changes a
project.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._logging import get_logger
from .validation.outliers import LIMIT_LEVEL

if TYPE_CHECKING:
    from ._workflow import Model, Project

logger = get_logger(__name__)

#: The workflow steps a check belongs to, in order.
STEPS = ("data", "rois", "values", "model", "predict")
#: ``ok``: done; ``info``: worth knowing; ``attention``: look at it;
#: ``blocking``: the next step cannot run.
SEVERITIES = ("ok", "info", "attention", "blocking")
#: The levels of a verdict, best first.
VERDICT_LEVELS = ("good", "usable_with_care", "not_usable")

#: Share of calibration pixels expected above the DModX limit at LIMIT_LEVEL (0.95: 5 %).
DMODX_SHARE_EXPECTED = 1.0 - LIMIT_LEVEL
#: Share of calibration pixels above the DModX limit that needs attention: three times
#: the expected share (15 % at 0.95; decided 2026-11-19).
DMODX_SHARE_ATTENTION = 3 * DMODX_SHARE_EXPECTED
#: Verdict thresholds on the cross-validated figures of merit per ROI
#: (model-workspace-diagnostics contract, Verdict). RPD = SD / RMSECV.
RPD_GOOD, RPD_MIN = 2.5, 1.5
Q2_GOOD, Q2_MIN = 0.8, 0.5


def figure_text(value: float) -> str:
    """A figure of merit as text: four significant digits (terminology contract, numbers).

    One rule for RMSECV, RMSEE, bias, R², Q² and RPD, wherever they are
    shown: the verdict, the Models table, the fit summary, the plots.
    """
    return f"{float(value):.4g}"


@dataclass(frozen=True)
class Fix:
    """An action that resolves a check: a name, a button label and its arguments."""

    action: str
    label: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Check:
    """One item of the workflow: what was checked, how it stands and what it refers to."""

    check_id: str
    step: str
    severity: str
    message: str
    roi_ids: tuple[str, ...] = ()
    dataset_ids: tuple[str, ...] = ()
    target_id: str | None = None
    model_run_id: str | None = None
    fix: Fix | None = None

    @property
    def is_open(self) -> bool:
        """Whether the check asks for an action (``attention`` or ``blocking``)."""
        return self.severity in ("attention", "blocking")


@dataclass(frozen=True)
class Verdict:
    """A model in one level (:data:`VERDICT_LEVELS`), one sentence and its evidence."""

    level: str
    summary: str
    reasons: tuple[str, ...]
    evidence: tuple[Check, ...]


def project_checks(project: Project, *, files: bool = True) -> list[Check]:
    """The checks of every step of ``project``, in workflow order.

    ``files=False`` skips looking for the dataset files (no file access
    beyond the project file).
    """
    checks = [*_data_checks(project, files=files), *_roi_checks(project),
              *_value_checks(project)]
    models = list(project.models.values())
    if not models:
        ready = _targets_ready_to_fit(project)
        if ready:
            checks.append(Check("model.none", "model", "info", "No model is fitted yet.",
                                target_id=ready[0],
                                fix=Fix("fit", "Fit Model", {"target_id": ready[0]})))
    for model in models:
        checks.extend(model_checks(model))
    checks.extend(_predict_checks(project))
    order = {step: index for index, step in enumerate(STEPS)}
    return sorted(checks, key=lambda check: order[check.step])  # stable within a step


def model_checks(model: Model) -> list[Check]:
    """The checks of one model: validation, calibration set, distance to the model, components."""
    run_id = model.ref.model_run_id
    checks = [*_validation_checks(model), *_calibration_set_checks(model)]
    distance = _distance_check(model)
    if distance is not None:
        checks.append(distance)
    components = _components_check(model)
    if components is not None:
        checks.append(components)
    metrics = model.metrics
    if "n_rois" in metrics and "n_pixels" in metrics:
        checks.append(Check(
            "model.calibration_set", "model", "ok",
            f"'{model.name}' is fitted on {int(metrics['n_rois'])} ROIs "
            f"({_count(int(metrics['n_pixels']))} pixels).", model_run_id=run_id))
    return checks


def model_verdict(model: Model) -> Verdict:
    """``model`` in one level with its reasons (model-workspace-diagnostics contract, Verdict).

    ``not_usable``: RPD below 1.5 or Q² below 0.5 (per ROI, cross-validated).
    ``good``: RPD at least 2.5, Q² at least 0.8 and no open check.
    ``usable_with_care``: everything between. A model without these figures
    of merit is judged by its checks alone.
    """
    checks = model_checks(model)
    open_checks = tuple(check for check in checks if check.is_open)
    metrics = model.metrics
    rpd, q2 = _finite(metrics.get("rpd")), _finite(metrics.get("r2"))
    reasons: list[str] = []
    level = "good"
    if rpd is not None and q2 is not None:
        if rpd < RPD_MIN or q2 < Q2_MIN:
            level = "not_usable"
            reasons.append(f"the cross-validated error is close to the spread of the values "
                           f"(RPD {figure_text(rpd)}, Q² {figure_text(q2)})")
        elif rpd < RPD_GOOD or q2 < Q2_GOOD:
            level = "usable_with_care"
            reasons.append(f"RPD {figure_text(rpd)} and Q² {figure_text(q2)} are below "
                           f"{RPD_GOOD} and {Q2_GOOD}")
    if open_checks and level == "good":
        level = "usable_with_care"
    reasons.extend(_reason(check) for check in open_checks)
    unit = "" if model.target.unit in ("", "1") else f" {model.target.unit}"
    figures = [] if "rmsecv" not in metrics else [f"RMSECV {figure_text(metrics['rmsecv'])}{unit}"]
    figures += [f"{name} {figure_text(value)}" for name, value in (("Q²", q2), ("RPD", rpd))
                if value is not None]
    label = {"good": "Good", "usable_with_care": "Usable with care",
             "not_usable": "Not usable"}[level]
    summary = f"{label}. {', '.join(figures)}." if figures else f"{label}."
    if open_checks and level != "not_usable":  # the first thing to act on
        summary += " " + _sentence(_reason(open_checks[0]))
    elif reasons:
        summary += " " + _sentence(reasons[0])
    logger.debug("Verdict of model '{}': {} ({} open check(s))", model.name, level,
                 len(open_checks))
    return Verdict(level, summary, tuple(reasons), tuple(checks))


# --- steps -----------------------------------------------------------------------------


def _data_checks(project: Project, *, files: bool) -> list[Check]:
    datasets = project.workspace.datasets
    if not datasets:
        return [Check("data.none", "data", "attention", "The project has no dataset.",
                      fix=Fix("add_dataset", "Add Dataset"))]
    checks: list[Check] = []
    if files:
        missing = tuple(dataset_id for dataset_id in datasets
                        if project.find_data_file(dataset_id) is None)
        if missing:
            checks.append(Check(
                "data.files_missing", "data", "attention",
                f"The files of {len(missing)} of {len(datasets)} datasets were not found. "
                "Stored ROIs and models still work; images and prediction maps need the "
                "files.", dataset_ids=missing,
                fix=Fix("locate_dataset", "Locate", {"dataset_id": missing[0]})))
        else:
            checks.append(Check("data.files_found", "data", "ok",
                                f"The files of all {len(datasets)} datasets were found.",
                                dataset_ids=tuple(datasets)))
    unnumbered = sorted({roi.dataset_id for roi in project.workspace.rois.values()
                         if _is_band_index(roi.wavelengths)})
    if unnumbered:
        checks.append(Check(
            "data.no_wavelengths", "data", "info",
            f"{len(unnumbered)} dataset(s) have no wavelengths; bands are numbered from 0.",
            dataset_ids=tuple(unnumbered)))
    return checks


def _roi_checks(project: Project) -> list[Check]:
    rois = project.workspace.rois
    if not rois:
        if not project.workspace.datasets:
            return []
        return [Check("rois.none", "rois", "attention", "The project has no ROI.",
                      fix=Fix("add_roi", "Add ROI"))]
    jobs = Counter(roi.job for roi in rois.values())
    n_datasets = len({roi.dataset_id for roi in rois.values()})
    return [Check("rois.present", "rois", "ok",
                  f"{jobs.get('calibration', 0)} calibration ROIs and "
                  f"{jobs.get('prediction', 0)} prediction ROIs in {n_datasets} dataset(s).")]


def _value_checks(project: Project) -> list[Check]:
    ws = project.workspace
    calibration_rois = [roi for roi in ws.rois.values() if roi.job == "calibration"]
    if not calibration_rois:
        return []
    if not ws.targets:
        return [Check("values.no_target", "values", "attention",
                      "The project has no target.", fix=Fix("add_target", "Add Target"))]
    checks: list[Check] = []
    for target_id, target in ws.targets.items():
        without = tuple(roi.roi_id for roi in calibration_rois
                        if (roi.roi_id, target_id) not in ws.target_assignments)
        n_with = len(calibration_rois) - len(without)
        if without:
            checks.append(Check(
                "values.missing", "values", "blocking" if n_with < 2 else "attention",
                f"{len(without)} of {len(calibration_rois)} calibration ROIs have no value for "
                f"'{target.name}'.", roi_ids=without, target_id=target_id,
                fix=Fix("edit_values", "Enter Values",
                        {"target_id": target_id, "roi_ids": without})))
        else:
            checks.append(Check(
                "values.complete", "values", "ok",
                f"All {len(calibration_rois)} calibration ROIs have a value for '{target.name}'.",
                target_id=target_id))
        checks.extend(_sample_value_checks(project, target_id))
    return checks


def _sample_value_checks(project: Project, target_id: str) -> list[Check]:
    """ROIs whose value differs from the other ROIs of their sample (a likely typing error)."""
    ws = project.workspace
    by_sample: dict[str, list[tuple[str, float]]] = {}
    for roi in ws.rois.values():
        assignment = ws.target_assignments.get((roi.roi_id, target_id))
        if roi.sample and assignment is not None:
            by_sample.setdefault(roi.sample, []).append((roi.roi_id, float(assignment.value)))
    checks = []
    for sample, members in by_sample.items():
        counts = Counter(value for _roi_id, value in members)
        if len(counts) < 2:
            continue
        usual, n_usual = counts.most_common(1)[0]
        # Without a majority every ROI of the sample is listed.
        odd = [roi_id for roi_id, value in members
               if value != usual or n_usual * 2 <= len(members)]
        names = ", ".join(ws.rois[roi_id].name for roi_id in odd[:3]) + (
            "" if len(odd) <= 3 else f" and {len(odd) - 3} more")
        message = (f"{names} of sample '{sample}': the value differs from the other ROIs of "
                   f"the sample ({_number(usual)})." if n_usual * 2 > len(members) else
                   f"The ROIs of sample '{sample}' have different values for "
                   f"'{ws.targets[target_id].name}'.")
        checks.append(Check("values.differ_within_sample", "values", "attention", message,
                            roi_ids=tuple(odd), target_id=target_id,
                            fix=Fix("select_roi", "Go to ROI", {"roi_id": odd[0]})))
    return checks


def _predict_checks(project: Project) -> list[Check]:
    models = list(project.models.values())
    if not models:
        return []
    prediction = tuple(roi.roi_id for roi in project.workspace.rois.values()
                       if roi.job == "prediction")
    if not prediction:
        return [Check("predict.no_rois", "predict", "info",
                      "No ROI is marked as a prediction ROI; a model can still predict a "
                      "whole dataset.")]
    newest = max(models, key=lambda model: model.ref.created_at)
    return [Check("predict.rois_ready", "predict", "info",
                  f"{len(prediction)} prediction ROIs can be predicted.", roi_ids=prediction,
                  model_run_id=newest.ref.model_run_id,
                  fix=Fix("predict_rois", "Predict ROIs",
                          {"model_run_id": newest.ref.model_run_id}))]


# --- one model -------------------------------------------------------------------------


def _validation_checks(model: Model) -> list[Check]:
    """The stored fold warnings (sample-set-model-run contract §7.4) as checks."""
    from .modeling._service import fold_warning_text

    run = model.ref
    codes = run.cv_spec.get("warnings") or ""
    if isinstance(codes, bytes):
        codes = codes.decode()
    codes = [code for code in (codes.split(",") if isinstance(codes, str) else codes) if code]
    ws = model.project.workspace
    sample_set = ws.sample_sets[run.sample_set_id]
    checks = []
    for code, text in zip(codes, fold_warning_text(codes)):
        fix = None
        if code == "one_level_per_file":
            members = [ws.rois[roi_id] for roi_id in sample_set.roi_ids if roi_id in ws.rois]
            labelled = bool(members) and all(roi.sample for roi in members)
            by_sample = run.fit_specification.group_by == "sample"
            if labelled and not by_sample:
                fix = Fix("fit_again", "Hold Out by Sample",
                          {"model_run_id": run.model_run_id, "group_by": "sample"})
            else:  # no labels, or held out by sample already: the labels need a look
                fix = Fix("edit_samples", "Set Sample Labels",
                          {"roi_ids": tuple(roi.roi_id for roi in members
                                            if by_sample or not roi.sample)})
        checks.append(Check(
            f"model.validation.{code}", "model", "attention",
            f"The cross-validation of '{model.name}' is likely optimistic: {text}.",
            model_run_id=run.model_run_id, fix=fix))
    return checks


def _calibration_set_checks(model: Model) -> list[Check]:
    """Whether the calibration set of ``model`` still equals the project's ROIs and values."""
    project, run = model.project, model.ref
    ws = project.workspace
    sample_set = ws.sample_sets[run.sample_set_id]
    refit = Fix("refit", "Fit Again", {"model_run_id": run.model_run_id})
    checks = []
    deleted = tuple(project.missing_rois(run.sample_set_id))
    if deleted:
        checks.append(Check(
            "model.rois_deleted", "model", "attention",
            f"'{model.name}' was fitted with {len(deleted)} ROI(s) that were deleted since; "
            "it can predict but cannot be fitted again.", roi_ids=deleted,
            model_run_id=run.model_run_id))
    changed = tuple(roi_id for roi_id in model._changed_values(sample_set)
                    if roi_id not in deleted)
    if changed:
        checks.append(Check(
            "model.values_changed", "model", "attention",
            f"{len(changed)} value(s) changed after '{model.name}' was fitted.",
            roi_ids=changed, model_run_id=run.model_run_id, fix=None if deleted else refit))
    added = tuple(roi.roi_id for roi in ws.rois.values()
                  if roi.job == "calibration" and roi.roi_id not in sample_set.roi_ids
                  and (roi.roi_id, sample_set.target_id) in ws.target_assignments)
    if added:
        checks.append(Check(
            "model.rois_added", "model", "info",
            f"{len(added)} calibration ROI(s) with a value are not in '{model.name}'.",
            roi_ids=added, model_run_id=run.model_run_id,
            fix=Fix("fit", "Fit Model", {"target_id": sample_set.target_id})))
    return checks


def _distance_check(model: Model) -> Check | None:
    """The share of calibration pixels above the DModX limit, with the datasets it comes from."""
    stored = _stored(model, "diagnostics", ("roi_share_above_dcrit", "roi_ids",
                                            "roi_score_pixels"))
    if "roi_share_above_dcrit" not in stored or "roi_ids" not in stored:
        return None
    shares = np.asarray(stored["roi_share_above_dcrit"], dtype=float).reshape(-1)
    roi_ids = [str(roi_id) for roi_id in np.asarray(stored["roi_ids"]).reshape(-1)]
    pixels = np.asarray(stored.get("roi_score_pixels", np.ones(len(shares))),
                        dtype=float).reshape(-1)
    if len(shares) != len(roi_ids) or len(pixels) != len(shares) or pixels.sum() <= 0:
        return None
    overall = float(np.sum(shares * pixels) / pixels.sum())
    run_id = model.ref.model_run_id
    if overall <= DMODX_SHARE_ATTENTION:
        return Check("model.distance", "model", "ok",
                     f"{overall:.1%} of the calibration pixels of '{model.name}' are above "
                     f"the DModX limit ({figure_text(100 * DMODX_SHARE_EXPECTED)} % expected).".replace("%", " %", 1),
                     model_run_id=run_id)
    ws = model.project.workspace
    by_dataset: dict[str, list[int]] = {}
    for index, roi_id in enumerate(roi_ids):
        if roi_id in ws.rois:
            by_dataset.setdefault(ws.rois[roi_id].dataset_id, []).append(index)
    share_of = {dataset_id: float(np.sum(shares[rows] * pixels[rows]) / pixels[rows].sum())
                for dataset_id, rows in by_dataset.items()}
    worst = tuple(sorted((d for d, share in share_of.items() if share > overall),
                         key=lambda d: -share_of[d]))
    high = tuple(roi_id for roi_id, share in zip(roi_ids, shares)
                 if share > DMODX_SHARE_ATTENTION and roi_id in ws.rois)
    where = "" if not worst or len(worst) == len(share_of) else (
        f", mostly in {len(worst)} of {len(share_of)} datasets "
        f"(up to {max(share_of.values()):.0%})".replace("%", " %"))
    return Check(
        "model.distance", "model", "attention",
        f"{overall:.1%}".replace("%", " %") + f" of the calibration pixels of '{model.name}' "
        f"are above the DModX limit ({figure_text(100 * DMODX_SHARE_EXPECTED)} % expected){where}; "
        "the model does not describe them "
        "well.", roi_ids=high, dataset_ids=worst, model_run_id=run_id,
        fix=Fix("show_page", "Open Distance Page", {"model_run_id": run_id,
                                                    "page": "Distance"}))


def _components_check(model: Model) -> Check | None:
    """The number of components against the selection rules and the structure ceiling."""
    curve = _stored(model, "component_curve", None) or None
    chosen = model.ref.hyperparameters.get("n_components")
    if not curve or chosen is None:
        return None
    chosen = int(chosen)
    suggestions = [int(value) for key, value in curve.items()
                   if key.startswith("suggested_") and np.ndim(value) == 0 and int(value) > 0]
    if not suggestions:
        return None
    agree = sum(1 for value in suggestions if value == chosen)
    ceiling = curve.get("structure_ceiling")
    run_id = model.ref.model_run_id
    if ceiling is not None and np.ndim(ceiling) == 0 and 0 < int(ceiling) < chosen:
        return Check(
            "model.components", "model", "attention",
            f"'{model.name}' uses {chosen} components; components beyond {int(ceiling)} "
            "describe noise.", model_run_id=run_id,
            fix=Fix("show_page", "Open Components Page", {"model_run_id": run_id,
                                                          "page": "Components"}))
    counts = Counter(suggestions)
    usual, n_usual = counts.most_common(1)[0]
    supported = agree == n_usual  # the chosen number is the most suggested one (ties count)
    return Check(
        "model.components", "model", "ok" if supported else "info",
        f"'{model.name}' uses {chosen} components; {agree} of {len(suggestions)} selection "
        f"rules suggest {chosen}" + ("." if supported else f", most suggest {usual}."),
        model_run_id=run_id)


# --- helpers ---------------------------------------------------------------------------

#: Stored model results read for checks, by (project file, model run, artifact).
#: A stored model never changes (headless-workflow contract, ids), so entries
#: stay valid; the viewer recomputes checks after every change.
_STORED: dict[tuple[str, str, str], dict[str, Any]] = {}


def _stored(model: Model, artifact: str, keys: tuple[str, ...] | None) -> dict[str, Any]:
    """The arrays ``keys`` (all with ``None``) of a stored model artifact, read once."""
    key = (str(model.project.path), model.ref.model_run_id, artifact)
    if key not in _STORED:
        if artifact == "component_curve":
            _STORED[key] = dict(model.component_curve or {})
        else:
            values = model.diagnostics if keys is None else _diagnostics(model, keys)
            _STORED[key] = dict(values)
    return _STORED[key]


def _diagnostics(model: Model, keys: tuple[str, ...]) -> dict[str, Any]:
    """Only ``keys`` of a model's diagnostics and cross-validation outputs.

    ``Model.diagnostics`` reads every array, including per-pixel scores;
    the checks need a few per-ROI ones.
    """
    from .io import project_store as store

    refs = store.load_model_run_artifact_refs(model.project.path, model.ref.model_run_id,
                                              keys=("cv_outputs",))
    found = {key: value for key, value in refs.get("cv_outputs", {}).items() if key in keys}
    import h5py

    from .io._hdf5_groups import read_mapping

    with h5py.File(model.project.path, "r") as handle:
        group = handle.get(f"model_runs/{model.ref.model_run_id}/artifact_refs/diagnostics")
        if isinstance(group, h5py.Group):
            # Short vectors (a model of 16 ROIs or fewer) are stored as attributes.
            found.update(read_mapping(group, keys=set(keys)))
    return found


def _targets_ready_to_fit(project: Project) -> list[str]:
    ws = project.workspace
    return [target_id for target_id in ws.targets
            if sum(1 for roi in ws.rois.values() if roi.job == "calibration"
                   and (roi.roi_id, target_id) in ws.target_assignments) >= 2]


def _is_band_index(wavelengths: Any) -> bool:
    axis = np.asarray(wavelengths, dtype=float).reshape(-1)
    return axis.size > 1 and bool(np.allclose(axis, np.arange(axis.size), atol=1e-6, rtol=0.0))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _count(value: int) -> str:
    """An integer with thin spaces between thousands (terminology contract, numbers)."""
    return f"{value:,}".replace(",", " ")


def _number(value: float) -> str:
    return f"{value:g}"


def _reason(check: Check) -> str:
    """The message of a check as a clause: no model name prefix, no final period."""
    text = check.message.rstrip(".")
    marker = "is likely optimistic: "
    return "the cross-validation " + marker + text.split(marker, 1)[1] if marker in text \
        else text[:1].lower() + text[1:]


def _sentence(clause: str) -> str:
    return clause[:1].upper() + clause[1:] + "."
