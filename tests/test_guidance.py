"""Checks and verdicts (headless-workflow contract, Guidance; plan gui-1.6, Phase 1).

``Project.checks()`` must find what a reviewer found by hand on a real
project: missing files, values that differ within a sample, an optimistic
cross-validation, a model that no longer matches its ROIs, pixels far from
the model. ``Model.verdict()`` condenses a model into one level.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace

import numpy as np
import pytest

from enginir import Project
from enginir import _guidance as guidance
from enginir._guidance import DMODX_SHARE_ATTENTION, Q2_GOOD, RPD_GOOD, RPD_MIN
from enginir._workflow import Model
from enginir.io import save_hdf5

N_BANDS, HEIGHT, FRAMES = 12, 4, 3


def _scan(path, contents, *, wavelengths=None, seed=1):
    """A scan with one block of FRAMES frames per value in ``contents``."""
    rng = np.random.default_rng(seed)
    peak = np.exp(-((np.arange(N_BANDS) - N_BANDS / 2) / 2) ** 2)
    content = np.repeat(np.asarray(contents, dtype=float), FRAMES)[:, None, None]
    cube = 2500 - 150 * content * peak + rng.normal(0, 3, (len(content), HEIGHT, N_BANDS))
    save_hdf5(str(path), cube.astype(np.uint16),
              np.linspace(1000, 1600, N_BANDS) if wavelengths is None else wavelengths)
    return str(path)


def _add_rois(dataset, prefix, contents, *, sample=None, target="API"):
    rois = []
    for i, value in enumerate(contents):
        roi = dataset.add_roi(f"{prefix}{i}", frames=(FRAMES * i, FRAMES * i + FRAMES - 1),
                              rows=(0, HEIGHT - 1), values={target: float(value)})
        if sample is not None:
            dataset.project.set_sample(roi, sample)
        rois.append(roi)
    return rois


def _project(tmp_path, contents=np.linspace(1, 6, 8)):
    """One scan, one ROI per value: nothing to complain about."""
    project = Project.create(tmp_path / "project.h5")
    project.add_target("API", unit="%")
    scan = project.add_dataset(_scan(tmp_path / "scan.h5", contents))
    _add_rois(scan, "s", contents)
    return project, scan


def _two_level_project(tmp_path, *, sample_labels):
    """Two scans, each with four ROIs of one value: the fit can learn the scan."""
    project = Project.create(tmp_path / "project.h5")
    project.add_target("API", unit="%")
    for name, value in (("low", 2.0), ("high", 5.0)):
        scan = project.add_dataset(_scan(tmp_path / f"{name}.h5", [value] * 4,
                                         seed=int(value)))
        _add_rois(scan, name, [value] * 4, sample=name if sample_labels else None)
    return project


def _by_id(checks):
    found = {}
    for check in checks:
        found.setdefault(check.check_id, []).append(check)
    return found


def _open_ids(checks):
    return sorted(check.check_id for check in checks if check.is_open)


def test_an_empty_project_asks_for_a_dataset_then_rois_then_a_target(tmp_path):
    project = Project.create(tmp_path / "project.h5")
    (only,) = project.checks()
    assert (only.check_id, only.step, only.severity) == ("data.none", "data", "attention")
    assert only.fix is not None and only.fix.action == "add_dataset"

    scan = project.add_dataset(_scan(tmp_path / "scan.h5", [1, 2, 3]))
    assert _open_ids(project.checks()) == ["rois.none"]

    scan.add_roi("a", frames=(0, 2), rows=(0, HEIGHT - 1))
    assert _open_ids(project.checks()) == ["values.no_target"]


def test_missing_values_block_a_fit_until_two_rois_have_one(tmp_path):
    project = Project.create(tmp_path / "project.h5")
    project.add_target("API", unit="%")
    scan = project.add_dataset(_scan(tmp_path / "scan.h5", [1, 2, 3]))
    rois = [scan.add_roi(name, frames=(FRAMES * i, FRAMES * i + 2), rows=(0, HEIGHT - 1))
            for i, name in enumerate("abc")]
    scan.add_roi("to predict", frames=(0, 1), rows=(0, 1), job="prediction")

    (missing,) = _by_id(project.checks())["values.missing"]
    assert missing.severity == "blocking"
    assert set(missing.roi_ids) == {roi.roi_id for roi in rois}, "prediction ROIs need no value"
    assert missing.fix is not None and missing.fix.arguments["roi_ids"] == missing.roi_ids

    project.set_value("a", "API", 1.0)
    project.set_value("b", "API", 2.0)
    (missing,) = _by_id(project.checks())["values.missing"]
    assert missing.severity == "attention" and missing.roi_ids == (rois[2].roi_id,)
    assert "model.none" in _by_id(project.checks()), "two values are enough to offer a fit"

    project.set_value("c", "API", 3.0)
    found = _by_id(project.checks())
    assert "values.missing" not in found and found["values.complete"][0].severity == "ok"


def test_a_value_that_differs_within_its_sample_is_named(tmp_path):
    project, scan = _project(tmp_path, contents=[4, 4, 4, 4, 7, 7])
    for name in ("s0", "s1", "s2", "s3"):
        project.set_sample(name, "L4")
    assert "values.differ_within_sample" not in _by_id(project.checks())

    project.set_value("s3", "API", 5.0)
    (odd,) = _by_id(project.checks())["values.differ_within_sample"]
    assert odd.severity == "attention"
    assert odd.roi_ids == (project.roi("s3").roi_id,)
    assert "s3" in odd.message and "'L4'" in odd.message and "(4)" in odd.message
    assert odd.fix is not None and odd.fix.arguments == {"roi_id": project.roi("s3").roi_id}

    # Without a majority no ROI is singled out.
    project.set_value("s2", "API", 5.0)
    (tie,) = _by_id(project.checks())["values.differ_within_sample"]
    assert len(tie.roi_ids) == 4


def test_files_that_are_gone_and_bands_without_wavelengths_are_reported(tmp_path):
    project = Project.create(tmp_path / "project.h5")
    project.add_target("API", unit="%")
    path = _scan(tmp_path / "scan.h5", [1, 2, 3], wavelengths=np.arange(N_BANDS, dtype=float))
    scan = project.add_dataset(path)
    _add_rois(scan, "s", [1, 2, 3])
    found = _by_id(project.checks())
    assert found["data.files_found"][0].severity == "ok"
    (axis,) = found["data.no_wavelengths"]
    assert axis.severity == "info" and axis.dataset_ids == (scan.record.dataset_id,)

    os.rename(path, tmp_path / "moved_away.bin")
    reopened = Project.open(project.path)
    (gone,) = _by_id(reopened.checks())["data.files_missing"]
    assert gone.severity == "attention" and gone.dataset_ids == (scan.record.dataset_id,)
    assert gone.fix is not None and gone.fix.action == "locate_dataset"
    assert not any(check.check_id.startswith("data.files")
                   for check in reopened.checks(files=False))


@pytest.mark.parametrize("sample_labels", [True, False])
def test_an_optimistic_cross_validation_is_a_check_with_its_fix(tmp_path, sample_labels):
    project = _two_level_project(tmp_path, sample_labels=sample_labels)
    model = project.fit("API", n_components=1, fit_spec={"folds": {"group_by": "roi"}})
    assert "one_level_per_file" in str(model.record.cv_spec["warnings"])

    (check,) = _by_id(model.checks())["model.validation.one_level_per_file"]
    assert check.severity == "attention" and check.model_run_id == model.record.model_run_id
    assert check.fix is not None
    if sample_labels:
        assert check.fix.action == "fit_again"
        assert check.fix.arguments == {"model_run_id": model.record.model_run_id,
                                       "group_by": "sample"}
    else:
        assert check.fix.action == "edit_samples" and len(check.fix.arguments["roi_ids"]) == 8
    assert check in project.checks(), "the project lists the checks of its models"

    if sample_labels:  # held out by sample already, the labels are what needs a look
        by_sample = project.fit("API", n_components=1, fit_spec={"folds": {"group_by": "sample"}})
        found = _by_id(by_sample.checks()).get("model.validation.one_level_per_file", [])
        assert all(c.fix is not None and c.fix.action == "edit_samples" for c in found)
    assert model.verdict().level != "good"


def test_a_model_reports_what_changed_in_its_calibration_set(tmp_path):
    project, scan = _project(tmp_path)
    model = project.fit("API", n_components=1)
    assert not [check for check in model.checks() if check.check_id in (
        "model.values_changed", "model.rois_deleted", "model.rois_added")]

    project.set_value("s2", "API", 9.0)
    (changed,) = _by_id(model.checks())["model.values_changed"]
    assert changed.roi_ids == (project.roi("s2").roi_id,)
    assert changed.fix is not None and changed.fix.action == "refit"

    scan.add_roi("new", frames=(0, 1), rows=(0, 1), values={"API": 1.0})
    (added,) = _by_id(model.checks())["model.rois_added"]
    assert added.severity == "info" and added.roi_ids == (project.roi("new").roi_id,)

    deleted_id = project.roi("s5").roi_id
    project.remove_roi(deleted_id)
    found = _by_id(project.model(model.name).checks())
    assert found["model.rois_deleted"][0].roi_ids == (deleted_id,)
    assert found["model.values_changed"][0].fix is None, "a refit is refused after a deletion"


def test_pixels_far_from_the_model_are_reported_with_their_dataset(tmp_path, monkeypatch):
    project = _two_level_project(tmp_path, sample_labels=True)
    model = project.fit("API", n_components=1)
    stored = model.diagnostics
    roi_ids = [str(roi_id) for roi_id in stored["roi_ids"]]
    low = [roi_id for roi_id in roi_ids if project.workspace.rois[roi_id].name.startswith("low")]
    pixels = np.asarray(stored["roi_score_pixels"], dtype=float)

    def with_shares(shares):
        guidance._STORED.clear()
        monkeypatch.setattr(guidance, "_diagnostics", lambda _model, keys: {
            **{key: stored[key] for key in keys if key in stored},
            "roi_share_above_dcrit": np.asarray(shares, dtype=np.float32)})
        return _by_id(model.checks())["model.distance"][0]

    assert with_shares([0.02] * 8).severity == "ok"
    assert with_shares([DMODX_SHARE_ATTENTION - 0.001] * 8).severity == "ok"
    assert with_shares([DMODX_SHARE_ATTENTION + 0.001] * 8).severity == "attention"
    # Decided 2026-11-19: three times the 5 % a 0.95 limit expects, so 15 %.
    assert with_shares([0.14] * 8).severity == "ok"
    assert "(5 % expected)" in with_shares([0.16] * 8).message
    assert with_shares([0.16] * 8).severity == "attention"

    shares = [0.6 if roi_id in low else 0.02 for roi_id in roi_ids]
    check = with_shares(shares)
    overall = float(np.sum(np.asarray(shares) * pixels) / pixels.sum())
    assert check.severity == "attention"
    assert f"{overall:.1%}".replace("%", " %") in check.message
    assert set(check.roi_ids) == set(low)
    assert check.dataset_ids == (project.workspace.rois[low[0]].dataset_id,)
    assert "1 of 2 datasets" in check.message
    assert check.fix is not None and check.fix.arguments["page"] == "Distance"


def _with_metrics(model, **metrics):
    return Model(model.project, replace(model.record, metrics={**model.record.metrics, **metrics}))


def test_the_verdict_follows_its_thresholds_and_the_open_checks(tmp_path):
    project, _scan_ = _project(tmp_path)
    model = project.fit("API", n_components=1)
    assert not [check for check in model.checks() if check.is_open], "a clean project"

    def level(**metrics):
        return _with_metrics(model, **metrics).verdict().level

    assert level(rpd=RPD_GOOD, r2=Q2_GOOD) == "good"
    assert level(rpd=RPD_GOOD - 0.01, r2=0.9) == "usable_with_care"
    assert level(rpd=3.0, r2=Q2_GOOD - 0.01) == "usable_with_care"
    assert level(rpd=RPD_MIN, r2=0.6) == "usable_with_care"
    assert level(rpd=RPD_MIN - 0.01, r2=0.6) == "not_usable"
    assert level(rpd=2.0, r2=0.49) == "not_usable"

    good = _with_metrics(model, rpd=4.0, r2=0.94, rmsecv=0.25).verdict()
    assert good.summary == "Good. RMSECV 0.25 %, Q² 0.94, RPD 4."
    assert good.reasons == () and good.evidence == tuple(model.checks())

    # An open check takes a model with good figures down one level, never to "not usable".
    project.set_value("s2", "API", 9.0)
    care = _with_metrics(model, rpd=4.0, r2=0.94).verdict()
    assert care.level == "usable_with_care"
    assert care.summary.endswith(f"1 value(s) changed after '{model.name}' was fitted.")
    assert len(care.reasons) == 1
    # A model that is not usable says why first, not what else is open.
    poor = _with_metrics(model, rpd=1.2, r2=0.3).verdict()
    assert poor.level == "not_usable" and len(poor.reasons) == 2
    assert poor.summary.endswith("(RPD 1.2, Q² 0.3).")


def test_checks_and_verdicts_change_nothing(tmp_path):
    project = _two_level_project(tmp_path, sample_labels=True)
    model = project.fit("API", n_components=1)

    def digest():
        with open(project.path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    before = digest()
    assert project.checks() and model.verdict()
    assert digest() == before


def test_a_small_model_gets_its_distance_check(tmp_path):
    # With 16 ROIs or fewer the per-ROI shares are stored as HDF5 attributes,
    # not datasets; the check must still read them (external review, 1.6.0).
    project, _scan = _project(tmp_path)
    model = project.fit("API", n_components=1)
    guidance._STORED.clear()
    assert np.asarray(model.diagnostics["roi_share_above_dcrit"]).size == 8
    assert "model.distance" in _by_id(model.checks())


# -- predict.rois_outside: prediction ROIs outside the newest model -------------------


def _count_predictions(monkeypatch) -> list[str]:
    calls: list[str] = []
    original = Model.roi_prediction

    def counted(self, rois=None):
        calls.append(self.record.model_run_id)
        return original(self, rois)

    monkeypatch.setattr(Model, "roi_prediction", counted)
    return calls


def test_a_prediction_roi_outside_the_newest_model_is_named_once_predicted(tmp_path, monkeypatch):
    from tests._scans import tray_project

    project, _dataset, _path = tray_project(tmp_path, odd="t09")
    project.set_job("t10", "calibration")  # prediction ROIs: t08 (normal) and t09 (odd)
    older = project.fit("API", n_components=1, cv=3)
    newest = project.fit("API", n_components=2, cv=3)
    calls = _count_predictions(monkeypatch)

    assert "predict.rois_outside" not in _by_id(project.checks(predict=False))
    assert calls == [], "predict=False predicts nothing"

    (check,) = _by_id(project.checks())["predict.rois_outside"]
    assert check.severity == "attention" and check.is_open
    assert check.roi_ids == (project.roi("t09").roi_id,)
    assert check.model_run_id == newest.record.model_run_id != older.record.model_run_id
    assert "1 prediction ROI(s) are outside" in check.message and "0.95" in check.message
    assert check.fix is not None and check.fix.arguments == {"roi_id": project.roi("t09").roi_id}
    assert calls == [newest.record.model_run_id]

    project.checks()
    assert calls == [newest.record.model_run_id], "a second call predicts nothing (cached)"


def test_prediction_rois_within_the_model_are_said_to_be(tmp_path):
    from tests._scans import tray_project

    project, _dataset, _path = tray_project(tmp_path)
    project.set_job("t10", "calibration")  # prediction ROIs: t08 and t09
    model = project.fit("API", n_components=2, cv=3)
    # t09 has some pixels above DCrit: more than the median calibration ROI, fewer than the
    # 90th percentile, so it is inside (a rule with the median would call it outside).
    high = np.percentile(np.asarray(model.diagnostics["roi_share_above_dcrit"]), 90)
    median = np.median(np.asarray(model.diagnostics["roi_share_above_dcrit"]))
    t09 = next(row for row in model.predict_rois(["t09"]).rois)
    assert median < t09.share_above_dcrit <= high
    checks = _by_id(project.checks())
    assert "predict.rois_outside" not in checks
    (ready,) = checks["predict.rois_ready"]
    assert ready.message.endswith(f"; all lie within '{model.name}'.")
    (fast,) = _by_id(project.checks(predict=False))["predict.rois_ready"]
    assert "all lie within" not in fast.message, "not claimed without predicting"
