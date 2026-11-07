"""Fold groups and ROI sample labels (sample-set-model-run contract §4.1, §6, §7; roi-target)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from enginir import Project
from enginir.io import save_hdf5
from enginir.modeling import _service
from enginir.project import FitSpec
from tests.test_fit_spec import _warnings_during

N_BANDS = 12
LEVELS = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}


def _blend_project(tmp_path, *, reverse=False):
    """5 concentration levels x 3 ROIs (A1 A2 A3 B1 ... E3, or E3 ... A1), 2 frames x 3 rows."""
    rng = np.random.default_rng(7)
    peak = np.exp(-((np.arange(N_BANDS) - 6) / 2.5) ** 2)
    names = [f"{level}{r}" for level in LEVELS for r in (1, 2, 3)]
    if reverse:
        names.reverse()
    cube = np.vstack([
        (2600 - 120 * LEVELS[name[0]] * peak + rng.normal(0, 25, (2, 3, N_BANDS)))
        for name in names]).astype(np.uint16)
    path = tmp_path / "scan.h5"
    save_hdf5(path, cube, np.linspace(1000, 1600, N_BANDS))
    project = Project.create(tmp_path / "blends.h5")
    scan = project.add_dataset(path)
    project.add_target("API", unit="%")
    for i, name in enumerate(names):
        scan.add_roi(name, frames=(2 * i, 2 * i + 1), rows=(0, 2),
                     values={"API": LEVELS[name[0]]})
    return project, names


def _label_by_level(project, names):
    for name in names:
        project.set_sample(name, f"blend {name[0]}")


def _held_out_folds(project, monkeypatch, **fit_args):
    """Fit, and return the ROI names held out in each cross-validation fold."""
    seen = {}
    real = _service.cross_validate

    def recording(estimator, X, y, *, cv, groups, **kwargs):
        seen["cv"], seen["groups"] = cv, groups
        return real(estimator, X, y, cv=cv, groups=groups, **kwargs)

    monkeypatch.setattr(_service, "cross_validate", recording)
    model = project.fit("API", n_components=1, **fit_args)
    run, ws = model.ref, project.workspace
    sample_set = ws.sample_sets[run.sample_set_id]
    data = _service.build_sample_set(
        project_path=project.path, rois=ws.rois, target_assignments=ws.target_assignments,
        target_id=sample_set.target_id, preprocessing_recipe_id=sample_set.preprocessing_recipe_id,
        roi_ids=sample_set.roi_ids)
    name_of = {roi_id: roi.name for roi_id, roi in ws.rois.items()}
    cv = seen["cv"]
    if cv == "logo":
        from sklearn.model_selection import LeaveOneGroupOut
        cv = LeaveOneGroupOut()
    folds = []
    for train, test in cv.split(np.zeros((len(seen["groups"]), 1)), groups=seen["groups"]):
        held = {name_of[roi_id] for roi_id in data.groups[test]}
        trained = {name_of[roi_id] for roi_id in data.groups[train]}
        assert not held & trained, "an ROI is on both sides of a fold"
        folds.append("".join(sorted(held)))
    return model, sorted(folds)


def _levels(*groups):
    return sorted("".join(f"{level}{r}" for level in group for r in (1, 2, 3))
                  for group in groups)


# -- folds ----------------------------------------------------------------------------


def test_grouping_by_sample_or_target_holds_out_whole_blends(tmp_path, monkeypatch):
    """Leave one group out by sample label or target value: each fold is one blend of 3 ROIs."""
    project, names = _blend_project(tmp_path)
    _label_by_level(project, names)
    for group_by in ("sample", "target"):
        _model, folds = _held_out_folds(
            project, monkeypatch,
            fit_spec={"folds": {"rule": "leave_one_group_out", "group_by": group_by}})
        assert folds == _levels("A", "B", "C", "D", "E"), group_by


def test_grouping_by_roi_holds_out_single_rois(tmp_path, monkeypatch):
    """Default grouping: leave one group out is leave one ROI out, labels are ignored."""
    project, names = _blend_project(tmp_path)
    _label_by_level(project, names)
    _model, folds = _held_out_folds(project, monkeypatch, cv="logo")
    assert folds == sorted(names)


def test_round_robin_by_target_deals_the_levels_in_value_order(tmp_path, monkeypatch):
    """Levels sorted by value, not by ROI order, are dealt like venetian blinds: with the ROIs
    stored E3 ... A1 and 3 folds the folds are A D | B E | C (by ROI order: E B | D A | C)."""
    project, _names = _blend_project(tmp_path, reverse=True)
    model, folds = _held_out_folds(project, monkeypatch,
                                   fit_spec={"folds": {"n_folds": 3, "group_by": "target"}})
    assert folds == _levels("AD", "BE", "C")
    assert model.ref.cv_spec["n_groups"] == 5
    assert model.ref.fit_specification.groups == tuple(j for j in (4, 3, 2, 1, 0) for _ in range(3))


def test_rois_without_a_label_are_their_own_group(tmp_path, monkeypatch):
    project, names = _blend_project(tmp_path)
    for name in names[:3]:
        project.set_sample(name, "blend A")
    _model, folds = _held_out_folds(
        project, monkeypatch, fit_spec={"folds": {"rule": "leave_one_group_out",
                                                 "group_by": "sample"}})
    assert folds == sorted(["A1A2A3", *names[3:]])


def test_fewer_than_two_groups_are_refused_before_reading(tmp_path, monkeypatch):
    project, names = _blend_project(tmp_path)
    for name in names:
        project.set_sample(name, "one tablet")
    monkeypatch.setattr(_service, "build_sample_set",
                        lambda **_: pytest.fail("spectra were read"))
    with pytest.raises(ValueError, match="at least 2 fold groups"):
        project.fit("API", n_components=1, fit_spec={"folds": {"group_by": "sample"}})


def test_labels_that_differ_only_in_case_warn(tmp_path):
    project, names = _blend_project(tmp_path)
    project.set_sample(names[0], "Blend A")
    project.set_sample(names[1], "blend a")
    _model, messages = _warnings_during(
        lambda: project.fit("API", n_components=1, fit_spec={"folds": {"group_by": "sample"}}))
    assert any("differ only in case" in message for message in messages)


# -- the stored groups ----------------------------------------------------------------


def test_a_refit_keeps_the_stored_groups_and_warns_when_labels_changed(tmp_path):
    project, names = _blend_project(tmp_path)
    _label_by_level(project, names)
    model = project.fit("API", n_components=1,
                        fit_spec={"folds": {"rule": "leave_one_group_out", "group_by": "sample"}})
    stored = model.ref.fit_specification.groups
    for name in names:  # every ROI its own sample now
        project.set_sample(name, name)
    refitted, messages = _warnings_during(model.refit)
    assert refitted.ref.fit_specification.groups == stored
    assert refitted.metrics["rmsecv"] == pytest.approx(model.metrics["rmsecv"], rel=1e-6)
    assert any("keeps its stored fold groups" in message for message in messages)


def test_groups_in_a_fit_request_are_refused(tmp_path):
    project, _names = _blend_project(tmp_path)
    with pytest.raises(ValueError, match="written by the fit"):
        project.fit("API", n_components=1, fit_spec={"folds": {"groups": list(range(15))}})


@pytest.mark.parametrize("groups", [[0, 1], [0, 0, 2] * 5, list(range(1, 16))])
def test_a_project_with_malformed_stored_groups_is_refused(tmp_path, groups):
    project, _names = _blend_project(tmp_path)
    model = project.fit("API", n_components=1, fit_spec={"folds": {"group_by": "target"}})
    with h5py.File(project.path, "r+") as f:
        f[f"model_runs/{model.ref.model_run_id}/fit_spec/folds"].attrs["groups"] = np.asarray(
            groups, dtype=np.int64)
    with pytest.raises(ValueError, match="folds/groups is not well-formed"):
        Project.open(project.path)


def test_group_kfold_needs_roi_groups():
    with pytest.raises(ValueError, match="group_kfold works only with folds/group_by roi"):
        FitSpec(fold_rule="group_kfold", group_by="sample")


# -- sample labels --------------------------------------------------------------------


def test_a_sample_label_is_set_changed_and_removed_without_touching_the_roi(tmp_path):
    project, names = _blend_project(tmp_path)
    before = project.roi(names[0]).to_dict()
    project.set_sample(names[0], "  tablet 7 ")
    reopened = Project.open(project.path)
    assert reopened.roi(names[0]).sample == "tablet 7"
    after = reopened.roi(names[0]).to_dict()
    assert before.pop("sample") is None and after.pop("sample") == "tablet 7"
    assert after.keys() == before.keys()
    assert all(np.array_equal(after[key], before[key]) if isinstance(before[key], np.ndarray)
               else after[key] == before[key] for key in before)
    reopened.set_sample(names[0], " ")
    assert Project.open(project.path).roi(names[0]).sample is None
