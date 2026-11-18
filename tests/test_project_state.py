"""ProjectState: the viewer's mirror of a project, written only through ``Project``.

Every change is written at once, signalled, and validated against the
project's relationships before anything is written; opening a project reads
no ROI spectra and no artifact arrays until they are needed.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from enginir.project import PredictionRun, ProjectRelationshipError
from enginir.viewer._state import ProjectState

from tests import _project_payloads as build

N_BANDS = 50


@pytest.fixture()
def wl():
    return np.linspace(900, 1700, N_BANDS, dtype=np.float32)


def _spectra(n_pixels: int = 4) -> np.ndarray:
    return np.arange(n_pixels * N_BANDS, dtype=np.uint16).reshape(n_pixels, N_BANDS)


@pytest.fixture()
def state(tmp_path, wl, qapp) -> ProjectState:
    """A project with two datasets, a target, a recipe, an input contract and one ROI."""
    ps = ProjectState()
    ps.new_project(str(tmp_path / "project.h5"))
    for name in ("a", "b"):
        ps.register_dataset(build.dataset(wl, f"dataset_{name}"))
        ps.set_active_calibration(build.calibration(f"dataset_{name}", f"cal_{name}"))
    ps.define_target(build.target())
    ps.register_recipe("recipe_1", build.recipe_payload())
    ps.register_signal_domain(build.signal_domain(wl))
    ps.add_roi(build.roi(wl, "roi_1", "dataset_a", "cal_a"), _spectra())
    return ps


@pytest.fixture()
def fitted(state) -> ProjectState:
    """``state`` with a value, a calibration set and a model with its artifacts."""
    wl = state.datasets["dataset_a"].wavelengths
    state.assign_target_to_roi(build.assignment("roi_1", value=12.3))
    state.create_sample_set(build.sample_set(["roi_1"], dataset_ids=["dataset_a"]))
    state.register_model_run(build.model_run(artifact_refs={
        "prediction_model": build.prediction_model_payload(wl),
        "cv_outputs": {"roi_ids": np.array(["roi_1"], dtype=object),
                       "y_true_roi": np.array([12.3], dtype=np.float32),
                       "y_pred_roi": np.array([12.0], dtype=np.float32)}}))
    return state


def test_a_new_project_is_written_empty_and_signalled(tmp_path, qapp):
    ps = ProjectState()
    loaded = []
    ps.project_loaded.connect(lambda: loaded.append(True))
    path = str(tmp_path / "new.h5")
    ps.new_project(path)
    assert loaded == [True] and ps.project_path == path
    assert ps.project_meta["FileSchemaVersion"] == "enginir-1.5"
    assert not any((ps.datasets, ps.calibrations, ps.rois, ps.targets, ps.target_assignments,
                    ps.recipes, ps.signal_domains, ps.sample_sets, ps.model_runs,
                    ps.predictions))


def test_rois_and_values_are_written_signalled_and_reopen_with_their_ids(state, wl):
    changed, targets, values = [], [], []
    state.rois_changed.connect(lambda: changed.append(True))
    state.targets_changed.connect(lambda: targets.append(True))
    state.target_assignments_changed.connect(lambda: values.append(True))

    spectra = _spectra(6)
    state.add_roi(build.roi(wl, "roi_2", "dataset_b", "cal_b", n_pixels=6,
                            spatial_bounds=(0, 1, 0, 2)), spectra)
    state.define_target(build.target("target_2", "protein"))
    state.assign_target_to_roi(build.assignment("roi_2", "target_2", 19.4))
    assert (changed, targets, values) == ([True], [True], [True])
    roi = state.rois["roi_2"]
    assert not hasattr(roi, "spectra"), "ROI spectra stay in the file"
    assert roi.spectra_storage_key == "/rois/roi_2/spectra"
    np.testing.assert_array_equal(roi.load_spectra(state.project_path), spectra)

    reopened = ProjectState()
    reopened.open_project(state.project_path)
    assert (reopened.rois["roi_2"].dataset_id, reopened.rois["roi_2"].calibration_id) == (
        "dataset_b", "cal_b")
    assert reopened.active_calibration_ids == {"dataset_a": "cal_a", "dataset_b": "cal_b"}
    assert reopened.get_target_assignment("roi_2", "target_2").value == pytest.approx(19.4)

    state.remove_roi("roi_2")
    assert "roi_2" not in state.rois and changed == [True, True]


def test_changes_are_written_without_rereading_the_project(state, wl):
    with patch.object(state.project, "_reload", side_effect=AssertionError("reload")):
        state.register_dataset(build.dataset(wl, "dataset_c"))
        state.set_active_calibration(build.calibration("dataset_c", "cal_c"))
        state.register_recipe("recipe_2", build.recipe_payload(snv=True))
        state.register_signal_domain(build.signal_domain(wl, "signal_2"))
        state.add_roi(build.roi(wl, "roi_2", "dataset_a", "cal_a"), _spectra())
        state.define_target(build.target("target_2", "protein"))
        state.assign_target_to_roi(build.assignment("roi_2", "target_2", 8.5))
        state.create_sample_set(build.sample_set(["roi_2"], target_id="target_2",
                                                 dataset_ids=["dataset_a"]))
        state.register_model_run(build.model_run())
        state.remove_roi("roi_2")
    assert state.model_runs["model_1"].sample_set_id == "sample_set_1"


def _refused(match: str, write) -> None:
    with pytest.raises(ProjectRelationshipError, match=match):
        write()


def test_broken_relationships_are_refused_before_anything_is_written(state, wl):
    _refused("missing calibration_id 'cal_missing'", lambda: state.add_roi(
        build.roi(wl, "roi_orphan", "dataset_a", "cal_missing"), _spectra()))
    _refused("missing target_id 'target_missing'", lambda: state.create_sample_set(
        build.sample_set(["roi_1"], target_id="target_missing", dataset_ids=["dataset_a"])))
    state.create_sample_set(build.sample_set(["roi_1"], dataset_ids=["dataset_a"]))
    _refused("missing signal_domain_id 'signal_missing'", lambda: state.register_model_run(
        build.model_run(signal_domain_id="signal_missing")))
    state.register_model_run(build.model_run(
        artifact_refs={"prediction_model": build.prediction_model_payload(wl)}))

    def prediction(**fields):
        values = dict(prediction_run_id="prediction_1", dataset_id="dataset_a",
                      calibration_id="cal_a", model_run_id="model_1",
                      signal_domain_id="signal_1", preprocessing_recipe_id="recipe_1",
                      artifact_refs={"y_pred": np.array([1.0], dtype=np.float32)},
                      created_at=build.T0)
        return PredictionRun(**{**values, **fields})

    _refused("missing model_run_id 'model_missing'", lambda: state.project.save_prediction_run(
        prediction(model_run_id="model_missing")))
    _refused("uses calibration 'cal_b' of dataset 'dataset_b'",
             lambda: state.project.save_prediction_run(prediction(calibration_id="cal_b")))

    reopened = ProjectState()
    reopened.open_project(state.project_path)
    assert set(reopened.rois) == {"roi_1"}
    assert set(reopened.sample_sets) == {"sample_set_1"}
    assert set(reopened.model_runs) == {"model_1"}
    assert reopened.project.workspace.predictions == {}


def test_opening_reads_objects_and_leaves_spectra_and_artifacts_on_disk(fitted, qapp):
    from enginir.io.project_store import load_project_store

    loaded = []
    ps = ProjectState()
    ps.project_loaded.connect(lambda: loaded.append(True))
    with patch("enginir.io.project_store.load_project_store",
               wraps=load_project_store) as reader:
        ps.open_project(fitted.project_path)
    assert reader.call_count == 1 and reader.call_args.kwargs == {}
    assert loaded == [True]
    assert set(ps.rois) == {"roi_1"} and not hasattr(ps.rois["roi_1"], "spectra")
    assert ps.get_target_assignment("roi_1", "target_1").value == pytest.approx(12.3)
    assert ps.model_runs["model_1"].artifact_refs["cv_outputs"] == {}
    outputs = ps.ensure_model_run_artifact_refs("model_1")["cv_outputs"]
    np.testing.assert_array_equal(outputs["roi_ids"], ["roi_1"])
    np.testing.assert_allclose(outputs["y_pred_roi"], [12.0])


def test_another_project_clears_the_in_memory_predictors(fitted, tmp_path):
    fitted.model_registry["model_1"] = object()
    fitted.active_model_name = "model_1"
    fitted.open_project(fitted.project_path)
    assert "model_1" in fitted.model_registry and fitted.active_model_name == "model_1"
    fitted.new_project(str(tmp_path / "other.h5"))
    assert fitted.model_registry == {} and fitted.active_model_name is None


def test_a_deleted_roi_leaves_its_calibration_set_stale_but_the_model_usable(fitted, wl):
    assert fitted.sample_sets_referencing_roi("roi_1") == ["sample_set_1"]
    assert fitted.model_runs_for_sample_sets(["sample_set_1"]) == ["model_1"]
    fitted.remove_roi("roi_1")

    reopened = ProjectState()
    reopened.open_project(fitted.project_path)
    assert "roi_1" not in reopened.rois
    assert reopened.sample_sets["sample_set_1"].roi_ids == ["roi_1"]
    assert reopened.missing_roi_ids_for_sample_set("sample_set_1") == ["roi_1"]
    assert reopened.build_prediction_job_context(model_run_id="model_1").model_run \
        .model_run_id == "model_1"
    with pytest.raises(ValueError, match="cannot be altered or refit"):
        reopened.fit_model_run(sample_set_id="sample_set_1", signal_domain_id="signal_1",
                               model_type="PLS", hyperparameters={"n_components": 1})

    reopened.add_roi(build.roi(wl, "roi_2", "dataset_a", "cal_a"), _spectra())
    reopened.create_sample_set(build.sample_set(["roi_2"], "sample_set_2",
                                                dataset_ids=["dataset_a"]))
    assert reopened.sample_sets["sample_set_2"].roi_ids == ["roi_2"]
