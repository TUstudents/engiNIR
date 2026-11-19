"""Fit specification and fold groups (sample-set-model-run contract §4, §4.1, §6, §7)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from enginir import Project
from enginir.io import save_hdf5
from enginir.modeling import _components, _gram_fit, _service
from enginir.modeling._pixels import draw_pixels, roi_digest, sample_checksum
from enginir.project import FitSpec
from enginir.validation import RoundRobinGroups
from tests import _simca_reference as ref
from tests._scans import uneven_project, warnings_during


# -- Fit specification (sample-set-model-run contract §4, §6, §7, §7.1, §7.3) ----------


N_BANDS = 12


def test_a_specification_round_trips_and_partial_mappings_take_the_defaults():
    spec = FitSpec(fold_rule="leave_one_group_out", pixels_per_roi=12, seed=5)
    assert FitSpec.from_dict(spec.to_dict()) == spec
    assert FitSpec.from_dict({"pixels": {"per_roi": 3}}) == FitSpec(pixels_per_roi=3)
    assert FitSpec().to_dict()["pixels"]["per_roi"] == "all"
    assert FitSpec.from_cv("logo").fold_rule == "leave_one_group_out"
    assert FitSpec.from_cv(4) == FitSpec(n_folds=4)


@pytest.mark.parametrize("payload, message", [
    ({"folds": {"rule": "shuffle"}}, "folds/rule"),
    ({"folds": {"n_folds": 1}}, "folds/n_folds"),
    ({"pixels": {"per_roi": 0}}, "pixels/per_roi"),
    ({"pixels": {"per_roi": "half"}}, "pixels/per_roi"),
    ({"pixels": {"seed": 2**63}}, "pixels/seed"),
    ({"exclusion": {"level": 0.4}}, "exclusion/level"),
    ({"exclusion": {"rounds": 3}}, "exclusion/rounds"),
    ({"exclusion": {"max_fraction": 1.0}}, "exclusion/max_fraction"),
    ({"pixels": {"count": 3}}, "unknown key"),
    ({"weights": {}}, "unknown part"),
])
def test_values_outside_the_contract_are_refused(payload, message):
    with pytest.raises(ValueError, match=message):
        FitSpec.from_dict(payload)


def test_every_key_is_stored_and_reopens_unchanged(tmp_path):
    project = uneven_project(tmp_path)
    model = project.fit("API", n_components=1, fit_spec={"pixels": {"per_roi": 4, "seed": 9}})
    with h5py.File(project.path, "r") as f:
        group = f[f"/model_runs/{model.record.model_run_id}/fit_spec"]
        stored = {part: dict(group[part].attrs) for part in group}
    assert {part: set(keys) for part, keys in stored.items()} == {
        part: set(keys) for part, keys in FitSpec().to_dict().items()}
    reopened = Project.open(project.path).model(model_run_id=model.record.model_run_id)
    assert reopened.record.fit_specification == model.record.fit_specification
    assert reopened.record.fit_specification.checksum == model.record.fit_specification.checksum != ""


def test_cv_and_a_fit_spec_with_folds_are_refused_together(tmp_path):
    project = uneven_project(tmp_path)
    with pytest.raises(ValueError, match="either as cv"):
        project.fit("API", n_components=1, cv=3, fit_spec={"folds": {"rule": "leave_one_group_out"}})
    model = project.fit("API", n_components=1, cv=3, fit_spec={"pixels": {"per_roi": 4}})
    spec = model.record.fit_specification
    assert (spec.fold_rule, spec.n_folds, spec.pixels_per_roi) == ("round_robin", 3, 4)
    assert model.record.cv_spec["requested"] == 3


def test_exclusion_without_residual_degrees_of_freedom_blocks_the_fit(tmp_path):
    """§7.2: K - A must be at least 1 (here A = K = 12 bands); nothing is stored."""
    project = uneven_project(tmp_path)
    with pytest.raises(ValueError, match="K - A >= 1"):
        project.fit("API", n_components=N_BANDS, fit_spec={"exclusion": {"method": "dmodx"}})
    assert project.models == {}


@pytest.mark.parametrize("requested", ["auto", 3])
def test_group_kfold_is_an_optional_rule_that_refits_identically(tmp_path, requested):
    """§4: GroupKFold is requested explicitly; it gives other folds than round-robin, and a
    refit keeps them."""
    project = uneven_project(tmp_path)
    n_folds = 7 if requested == "auto" else requested
    chosen = project.fit("API", n_components=2, name="group k-fold",
                         fit_spec={"folds": {"rule": "group_kfold", "n_folds": n_folds}})
    round_robin = project.fit("API", n_components=2, cv=n_folds)
    assert chosen.metrics["rmsecv"] != pytest.approx(round_robin.metrics["rmsecv"], rel=1e-6)
    assert chosen.record.cv_spec["type"] == "GroupKFold"

    refitted, warnings = warnings_during(chosen.refit)
    assert refitted.record.cv_spec["type"] == "GroupKFold"
    assert refitted.metrics["rmsecv"] == pytest.approx(chosen.metrics["rmsecv"], rel=1e-9)
    assert not warnings


def test_a_model_without_its_fit_specification_is_refused(tmp_path):
    """enginir-1.5: every model stores its fit specification; earlier runs are not read."""
    project = uneven_project(tmp_path)
    model = project.fit("API", n_components=2)
    with h5py.File(project.path, "a") as f:
        del f[f"/model_runs/{model.record.model_run_id}/fit_spec"]
    with pytest.raises(ValueError, match="fit_spec"):
        Project.open(project.path)


@pytest.mark.parametrize("n_groups, n_splits", [(3, 3), (7, 7), (20, 7)])
def test_round_robin_holds_out_the_group_at_position_i_in_fold_i_mod_k(n_groups, n_splits):
    groups = np.repeat(np.arange(n_groups), 3)[::-1]  # pixels in any order
    splitter = RoundRobinGroups(7)
    folds = list(splitter.split(np.zeros((groups.size, 1)), groups=groups))
    assert splitter.get_n_splits(groups=groups) == len(folds) == n_splits
    for f, (train, test) in enumerate(folds):
        held_out = set(groups[test])
        assert held_out == {g for g in range(n_groups) if g % n_splits == f}
        assert held_out.isdisjoint(groups[train])  # an ROI is never split


def test_more_folds_than_rois_hold_out_one_roi_each(tmp_path):
    project = uneven_project(tmp_path, n_rois=4)
    model = project.fit("API", n_components=1, cv=7)
    assert (model.record.cv_spec["type"], model.record.cv_spec["n_splits"]) == ("RoundRobinGroups", 4)


def test_the_fit_draws_the_pixels_of_the_simca_reference_table(tmp_path):
    """The library and the reference script implement §7.1 independently and agree."""
    from enginir.modeling import fit_model_run

    reference = ref.build(tmp_path)
    project = reference.project
    training = reference.names[:ref.N_TRAINING]
    model = project.fit("conc", n_components=1, cv="logo",
                        fit_spec={"pixels": {"per_roi": ref.PIXELS_PER_ROI, "seed": ref.SEED}})
    ws = project.workspace
    result = fit_model_run(
        project_path=project.path, sample_set=ws.sample_sets[model.record.sample_set_id],
        datasets=ws.datasets, reference_corrections=ws.reference_corrections, rois=ws.rois,
        target_assignments=ws.target_assignments, recipes=ws.recipes,
        signal_domains=ws.signal_domains, signal_domain_id=model.record.signal_domain_id,
        model_type="PLS", hyperparameters={"n_components": 1},
        fit_spec=model.record.fit_specification)
    ids, rois, _y, _X = ref.read_table(ref.DATA_DIR / "pixels.csv")
    data = result.sample_set_data  # the raw spectra stay in the project file (kernel-pls D7)
    assert data.pixel_indices is not None
    for name in reference.names:
        drawn = reference.raw[name][data.pixel_indices[ref.roi_id(name)]]
        if name in training:
            table = [int(i.split("_p")[1]) for i, roi in zip(ids, rois) if roi == name]
            np.testing.assert_array_equal(drawn, reference.raw[name][table])
    assert model.metrics["n_pixels"] == ref.PIXELS_PER_ROI * len(reference.names)
    assert result.model_run.fit_specification.checksum == model.record.fit_specification.checksum


def test_the_draw_depends_only_on_the_seed_and_the_roi_spectra():
    spectra = np.arange(40 * 5, dtype=np.uint16).reshape(40, 5)
    first = draw_pixels(spectra, 6, 1)
    assert np.array_equal(first, draw_pixels(spectra.copy(), 6, 1))
    assert np.array_equal(first, np.sort(first)) and len(set(first)) == 6
    assert not np.array_equal(first, draw_pixels(spectra, 6, 2))
    other = spectra.copy()
    other[3, 0] += 1
    assert not np.array_equal(first, draw_pixels(other, 6, 1))
    # The stored dtype is part of the digest: the same values as float32 are other spectra.
    assert roi_digest(spectra) != roi_digest(spectra.astype(np.float32))
    assert sample_checksum([(roi_digest(spectra), first)]) != sample_checksum(
        [(roi_digest(spectra), first[::-1])])


def test_adding_an_roi_keeps_the_other_rois_pixels_and_metrics_repeat(tmp_path):
    project = uneven_project(tmp_path)
    spec = {"pixels": {"per_roi": 5, "seed": 4}}
    first = project.fit("API", n_components=1, cv="logo", fit_spec=spec)
    again = project.fit("API", n_components=1, cv="logo", fit_spec=spec)
    assert again.metrics["rmsecv"] == pytest.approx(first.metrics["rmsecv"], rel=1e-12)
    assert again.record.fit_specification.checksum == first.record.fit_specification.checksum

    from enginir.io import project_store as store
    ws = project.workspace
    before = {roi_id: draw_pixels(store.load_roi_spectra(project.path, roi.spectra_storage_key),
                                  5, 4) for roi_id, roi in ws.rois.items()}
    project.datasets[next(iter(project.datasets))].add_roi(
        "extra", frames=(0, 1), rows=(0, 2), values={"API": 5.0})
    for roi_id, indices in before.items():
        stored = store.load_roi_spectra(project.path, ws.rois[roi_id].spectra_storage_key)
        assert np.array_equal(draw_pixels(stored, 5, 4), indices)


def test_more_pixels_than_the_smallest_roi_are_refused_before_reading(tmp_path, monkeypatch):
    project = uneven_project(tmp_path)  # the smallest ROIs have 2 x 3 = 6 pixels
    monkeypatch.setattr("enginir.modeling._service.load_roi_spectra_matrix",
                        lambda *a, **k: pytest.fail("spectra were read"))
    with pytest.raises(ValueError, match="Cannot draw 7 pixels per ROI.*choose at most 6"):
        project.fit("API", n_components=1, fit_spec={"pixels": {"per_roi": 7}})
    assert project.models == {}


def test_a_refit_reports_a_different_pixel_sample(tmp_path):
    project = uneven_project(tmp_path)
    model = project.fit("API", n_components=1, fit_spec={"pixels": {"per_roi": 5}})
    refitted, warnings = warnings_during(model.refit)
    assert not warnings
    assert refitted.record.fit_specification == model.record.fit_specification

    with h5py.File(project.path, "a") as f:  # as if an ROI's stored spectra had changed
        f[f"/model_runs/{model.record.model_run_id}/fit_spec/pixels"].attrs["checksum"] = "0" * 64
    changed = Project.open(project.path).model(model_run_id=model.record.model_run_id)
    _refitted, warnings = warnings_during(changed.refit)
    assert any("different pixel subsample" in w for w in warnings)


def test_runs_store_their_scaling_with_the_defaults_filled_in(tmp_path):
    """§7: defaults x_scaling center, y_scaling uv (PCR: x only), algorithm kernel (PCR: the
    one the guard chose, never "auto"); an explicit choice is kept."""
    project = uneven_project(tmp_path)
    assert project.fit("API", n_components=2).record.hyperparameters == {
        "n_components": 2, "x_scaling": "center", "y_scaling": "uv", "algorithm": "kernel"}
    assert project.fit("API", model="PCR", n_components=2).record.hyperparameters == {
        "n_components": 2, "x_scaling": "center", "algorithm": "gram"}
    assert project.fit("API", n_components=2, x_scaling="pareto").record.hyperparameters == {
        "n_components": 2, "x_scaling": "pareto", "y_scaling": "uv", "algorithm": "kernel"}
    assert project.fit("API", n_components=2, algorithm="nipals").record.hyperparameters[
        "algorithm"] == "nipals"
    with pytest.raises(ValueError, match="scale switch was removed"):
        project.fit("API", n_components=2, scale=True)


# -- Fold groups and ROI sample labels (sample-set-model-run contract §4.1, §6, §7; roi-target) ----


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

    real_folds = _gram_fit.folds

    def recording_folds(splitter, cv_groups, lay):
        seen["cv"], seen["groups"] = splitter, cv_groups
        return real_folds(splitter, cv_groups, lay)

    # the folds run in the component curve (and, with exclusion, again in the service);
    # a fit on statistics takes them from the same splitter and groups
    monkeypatch.setattr(_components, "cross_validate", recording)
    monkeypatch.setattr(_service, "cross_validate", recording)
    monkeypatch.setattr(_gram_fit, "folds", recording_folds)
    model = project.fit("API", n_components=1, **fit_args)
    run, ws = model.record, project.workspace
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
    assert model.record.cv_spec["n_groups"] == 5
    assert model.record.fit_specification.groups == tuple(j for j in (4, 3, 2, 1, 0) for _ in range(3))


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
    _model, messages = warnings_during(
        lambda: project.fit("API", n_components=1, fit_spec={"folds": {"group_by": "sample"}}))
    assert any("differ only in case" in message for message in messages)


def test_a_refit_keeps_the_stored_groups_and_warns_when_labels_changed(tmp_path):
    project, names = _blend_project(tmp_path)
    _label_by_level(project, names)
    model = project.fit("API", n_components=1,
                        fit_spec={"folds": {"rule": "leave_one_group_out", "group_by": "sample"}})
    stored = model.record.fit_specification.groups
    for name in names:  # every ROI its own sample now
        project.set_sample(name, name)
    refitted, messages = warnings_during(model.refit)
    assert refitted.record.fit_specification.groups == stored
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
        f[f"model_runs/{model.record.model_run_id}/fit_spec/folds"].attrs["groups"] = np.asarray(
            groups, dtype=np.int64)
    with pytest.raises(ValueError, match="folds/groups is not well-formed"):
        Project.open(project.path)


def test_group_kfold_needs_roi_groups():
    with pytest.raises(ValueError, match="group_kfold works only with folds/group_by roi"):
        FitSpec(fold_rule="group_kfold", group_by="sample")


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
