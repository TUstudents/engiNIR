"""Tests for TrainingROI and ProjectState."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from enginir.viewer._app import ensure_app

from ._project_payloads import recipe_payload, roi_provenance


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


@pytest.fixture()
def wl():
    return np.linspace(900, 1700, 50, dtype=np.float32)


@pytest.fixture()
def sample_provenance():
    return {
        "dataset_id": "dataset_1",
        "calibration_id": "cal_1",
        "signal_unit": "Reflectance",
        "preview_recipe_summary": "none",
        "created_at": "2026-04-02T10:00:00+00:00",
        "filter": "none",
        "sg_window": 11,
        "sg_polyorder": 2,
        "snv": False,
        "msc": False,
        "emsc": False,
        "detrend": False,
        "detrend_order": 1,
    }


def _register_dataset_context(
    ps, wl, *, dataset_id="dataset_1", calibration_id="cal_1", source_path="/tmp/cube.eiff"
):
    from enginir.project import CalibrationSpec, DatasetRef

    ps.register_dataset(
        DatasetRef(
            dataset_id=dataset_id,
            source_path=source_path,
            source_format="eiff",
            source_modality="nir_hsi",
            cube_shape=(20, 15, len(wl)),
            n_bands=len(wl),
            wavelengths=wl,
            loader_metadata={},
            import_timestamp="2026-04-02T10:00:00+00:00",
        )
    )
    ps.set_active_calibration(
        CalibrationSpec(
            calibration_id=calibration_id,
            dataset_id=dataset_id,
            mode="raw_dn",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=None,
            white_level=None,
            output_signal_unit="DN",
            created_at="2026-04-02T10:01:00+00:00",
        )
    )
    return dataset_id, calibration_id


@pytest.fixture()
def project_store_path(tmp_path, wl, sample_provenance):
    from enginir.io.project_store import (
        create_project,
        save_calibration_spec,
        save_dataset_ref,
        save_model_run,
        save_prediction_run,
        save_preprocessing_recipe,
        save_sample_set,
        save_signal_domain_contract,
        save_target_assignment,
        save_target_definition,
        save_training_roi,
    )

    path = str(tmp_path / "project_store.h5")
    create_project(path)
    save_dataset_ref(
        path,
        "dataset_1",
        {
            "source_path": "/data/train.eiff",
            "source_format": "eiff",
            "source_modality": "nir_hsi",
            "cube_shape": (20, 15, 50),
            "n_bands": 50,
            "wavelengths": wl,
            "loader_metadata": {"sensor": "AlphaVision"},
            "import_timestamp": "2026-04-02T10:00:00+00:00",
        },
    )
    save_calibration_spec(
        path,
        "dataset_1",
        "cal_1",
        {
            "mode": "dark_white_reference",
            "output_signal_unit": "Absorbance",
            "black_level": 128.0,
            "white_level": 3500.0,
            "created_at": "2026-04-02T10:01:00+00:00",
        },
    )
    save_target_definition(
        path,
        "target_moisture",
        {
            "name": "moisture",
            "unit": "% w/w",
            "value_type": "float",
            "missing_value_policy": "allowed",
            "created_at": "2026-04-02T10:02:00+00:00",
        },
    )
    save_preprocessing_recipe(
        path,
        "recipe_1",
        recipe_payload(filter="sg_d1", snv=True),
    )
    save_signal_domain_contract(
        path,
        "signal_domain_1",
        {
            "input_format": "eiff_raw",
            "source_modality": "nir_hsi",
            "required_signal_unit": "Absorbance",
            "required_calibration_mode": "dark_white_reference",
            "requires_dark_ref": True,
            "requires_white_ref": True,
            "calibrated_input_allowed": False,
            "n_bands": 50,
            "wavelengths": wl,
            "wavelength_tolerance_nm": 0.0,
            "band_mapping_policy": "exact",
            "output_dtype": "float32",
            "created_at": "2026-04-02T10:04:00+00:00",
        },
    )
    save_training_roi(
        path,
        "roi_1",
        {
            "dataset_id": "dataset_1",
            "calibration_id": "cal_1",
            "name": "ROI 1",
            "source_file": "/data/train.eiff",
            "n_pixels": 12,
            "created_at": "2026-04-02T10:05:00+00:00",
            "spatial_bounds": (0, 2, 0, 3),
            "wavelengths": wl,
            "provenance": {
                "dataset_id": "dataset_1",
                "calibration_id": "cal_1",
                "signal_unit": "Absorbance",
                "preview_recipe_summary": "SNV preview only",
                "created_at": "2026-04-02T10:05:00+00:00",
            },
        },
        np.zeros((12, 50), dtype=np.uint16),
    )
    save_target_assignment(
        path,
        "target_moisture",
        "roi_1",
        {
            "value": 12.3,
            "assigned_at": "2026-04-02T10:06:00+00:00",
        },
    )
    save_sample_set(
        path,
        "sample_set_1",
        {
            "roi_ids": ["roi_1"],
            "target_id": "target_moisture",
            "observation_unit": "pixel",
            "target_level": "roi",
            "grouping_unit": "roi",
            "validation_unit": "roi",
            "weighting_policy": "per_pixel_unweighted",
            "reporting_level": "roi",
            "roi_aggregation": "mean",
            "preprocessing_recipe_id": "recipe_1",
            "dataset_ids": ["dataset_1"],
            "created_at": "2026-04-02T10:07:00+00:00",
        },
    )
    save_model_run(
        path,
        "model_run_1",
        {
            "sample_set_id": "sample_set_1",
            "signal_domain_id": "signal_domain_1",
            "model_type": "PLS",
            "hyperparameters": {"n_components": 3},
            "cv_spec": {"type": "GroupKFold", "n_splits": 5},
            "metrics": {"rmsecv": 1.23},
            "artifact_refs": {
                "prediction_model": {
                    "model_type": "PLSRegression",
                    "coef_": np.ones(50, dtype=np.float32),
                    "intercept_": 0.0,
                    "x_mean_": np.zeros(50, dtype=np.float32),
                    "x_std_": np.ones(50, dtype=np.float32),
                    "y_mean_": np.zeros(1, dtype=np.float32),
                    "y_std_": np.ones(1, dtype=np.float32),
                    "wavelengths": wl,
                    "target_names": ["moisture"],
                },
                "cv_outputs": {
                    "roi_ids": np.array(["roi_1"], dtype=object),
                    "y_true_roi": np.array([12.3], dtype=np.float32),
                    "y_pred_roi": np.array([12.0], dtype=np.float32),
                },
            },
            "created_at": "2026-04-02T10:08:00+00:00",
        },
    )
    save_prediction_run(
        path,
        "prediction_run_1",
        {
            "dataset_id": "dataset_1",
            "calibration_id": "cal_1",
            "model_run_id": "model_run_1",
            "signal_domain_id": "signal_domain_1",
            "preprocessing_recipe_id": "recipe_1",
            "created_at": "2026-04-02T10:09:00+00:00",
            "artifact_refs": {
                "y_pred": np.array([11.9, 12.1], dtype=np.float32),
                "spatial_map": np.array([[11.9, 12.1]], dtype=np.float32),
            },
        },
    )
    return path


class TestTrainingROI:
    def test_fields(self, wl, sample_provenance):
        from enginir.viewer._state import TrainingROI

        roi = TrainingROI(
            roi_id="roi_grain_a",
            dataset_id="dataset_1",
            calibration_id="cal_1",
            name="grain_A",
            wavelengths=wl,
            n_pixels=12,
            spatial_bounds=(0, 2, 0, 3),
            source_file="/data/cube.eiff",
            created_at="2026-03-28T10:00:00+00:00",
            provenance=sample_provenance,
        )
        assert roi.roi_id == "roi_grain_a"
        assert roi.dataset_id == "dataset_1"
        assert roi.calibration_id == "cal_1"
        assert roi.name == "grain_A"
        assert roi.n_pixels == 12
        assert roi.spatial_bounds == (0, 2, 0, 3)
        assert roi.created_at == "2026-03-28T10:00:00+00:00"
        assert roi.spectra_storage_key == "/rois/roi_grain_a/spectra"

    def test_load_spectra_lazy(self, tmp_path, wl, sample_provenance):
        from enginir.viewer._state import TrainingROI
        from enginir.io.project_store import (
            create_project,
            save_calibration_spec,
            save_dataset_ref,
            save_training_roi,
        )

        rng = np.random.RandomState(0)
        spectra = rng.randint(0, 4095, (12, 50)).astype(np.uint16)
        path = str(tmp_path / "p.h5")
        create_project(path)
        save_dataset_ref(
            path,
            "dataset_1",
            {
                "source_path": "/x.eiff",
                "source_format": "eiff",
                "source_modality": "nir_hsi",
                "cube_shape": (3, 4, 50),
                "n_bands": 50,
                "wavelengths": wl,
                "loader_metadata": {},
                "import_timestamp": "2026-03-28T10:00:00",
            },
        )
        save_calibration_spec(
            path,
            "dataset_1",
            "cal_1",
            {
                "mode": "raw_dn",
                "dark_ref_source": None,
                "white_ref_source": None,
                "dark_ref_data": None,
                "white_ref_data": None,
                "black_level": None,
                "white_level": None,
                "output_signal_unit": "DN",
                "created_at": "2026-03-28T10:00:00",
            },
        )
        save_training_roi(
            path,
            "grain_A",
            {
                "dataset_id": "dataset_1",
                "calibration_id": "cal_1",
                "name": "grain_A",
                "wavelengths": wl,
                "source_file": "/x.eiff",
                "n_pixels": 12,
                "spatial_bounds": (0, 2, 0, 3),
                "created_at": "2026-03-28T10:00:00",
                "provenance": sample_provenance,
            },
            spectra,
        )
        roi = TrainingROI(
            roi_id="grain_A",
            dataset_id="dataset_1",
            calibration_id="cal_1",
            name="grain_A",
            wavelengths=wl,
            n_pixels=12,
            spatial_bounds=(0, 2, 0, 3),
            source_file="/x.eiff",
            created_at="2026-03-28T10:00:00",
            provenance=sample_provenance,
        )
        loaded = roi.load_spectra(path)
        assert loaded.dtype == np.uint16
        assert loaded.shape == (12, 50)
        np.testing.assert_array_equal(loaded, spectra)



def _add_fast_roi(ps, wl, dataset_id: str = "dataset_fast", calibration_id: str = "cal_fast") -> None:
    """Persist ROI ``roi_fast`` of one dataset, so that a sample set has a real member."""
    from enginir.viewer._state import TrainingROI

    roi = TrainingROI(
        roi_id="roi_fast",
        dataset_id=dataset_id,
        calibration_id=calibration_id,
        name="fast",
        wavelengths=wl,
        n_pixels=4,
        spatial_bounds=(0, 1, 0, 1),
        source_file="/tmp/fast.eiff",
        created_at="2026-04-02T10:02:30+00:00",
        provenance=roi_provenance(dataset_id=dataset_id, calibration_id=calibration_id),
    )
    ps.add_roi(roi, np.zeros((4, 50), dtype=np.uint16))

class TestProjectState:
    def test_new_project_creates_file_and_emits_project_loaded(self, tmp_path, qapp):
        from enginir.viewer._state import ProjectState

        ps = ProjectState()
        signals_received = []
        ps.project_loaded.connect(lambda: signals_received.append("loaded"))
        path = str(tmp_path / "proj.h5")
        ps.new_project(path)
        assert ps.project_path == path
        assert signals_received == ["loaded"]
        assert os.path.exists(path)
        assert ps.project_meta["schema_version"] == "viewer-v3"
        assert ps.datasets == {}
        assert ps.calibrations == {}
        assert ps.rois == {}
        assert ps.targets == {}
        assert ps.target_assignments == {}
        assert ps.recipes == {}
        assert ps.signal_domains == {}
        assert ps.sample_sets == {}
        assert ps.model_runs == {}
        assert ps.predictions == {}

    def test_add_roi_writes_to_hdf5_and_emits_rois_changed(
        self, tmp_path, wl, sample_provenance, qapp
    ):
        from enginir.viewer._state import ProjectState, TrainingROI

        ps = ProjectState()
        ps.new_project(str(tmp_path / "proj.h5"))
        dataset_id, calibration_id = _register_dataset_context(ps, wl)
        signals = []
        ps.rois_changed.connect(lambda: signals.append("changed"))
        spectra = np.zeros((6, 50), dtype=np.uint16)
        roi = TrainingROI(
            roi_id="roi_a",
            dataset_id=dataset_id,
            calibration_id=calibration_id,
            name="A",
            wavelengths=wl,
            n_pixels=6,
            spatial_bounds=(0, 1, 0, 2),
            source_file="/x.eiff",
            created_at="2026-03-28T10:00:00",
            provenance=sample_provenance,
        )
        ps.add_roi(roi, spectra)
        assert len(ps.rois) == 1
        assert "roi_a" in ps.rois
        assert ps.rois["roi_a"].roi_id == "roi_a"
        assert ps.rois["roi_a"].name == "A"
        assert ps.rois["roi_a"].dataset_id == dataset_id
        assert ps.rois["roi_a"].calibration_id == calibration_id
        assert signals == ["changed"]

    def test_add_roi_spectra_not_retained_in_memory(self, tmp_path, wl, sample_provenance, qapp):
        from enginir.viewer._state import ProjectState, TrainingROI

        ps = ProjectState()
        ps.new_project(str(tmp_path / "proj.h5"))
        dataset_id, calibration_id = _register_dataset_context(ps, wl)
        spectra = np.zeros((4, 50), dtype=np.uint16)
        roi = TrainingROI(
            roi_id="roi_b",
            dataset_id=dataset_id,
            calibration_id=calibration_id,
            name="B",
            wavelengths=wl,
            n_pixels=4,
            spatial_bounds=(0, 1, 0, 1),
            source_file="/tmp/cube_b.eiff",
            created_at="2026-03-28T10:00:00",
            provenance=sample_provenance,
        )
        ps.add_roi(roi, spectra)
        # TrainingROI has no spectra attribute
        assert not hasattr(ps.rois["roi_b"], "spectra")

    def test_remove_roi_emits_rois_changed(self, tmp_path, wl, sample_provenance, qapp):
        from enginir.viewer._state import ProjectState, TrainingROI

        ps = ProjectState()
        ps.new_project(str(tmp_path / "proj.h5"))
        dataset_id, calibration_id = _register_dataset_context(ps, wl)
        spectra = np.zeros((4, 50), dtype=np.uint16)
        roi = TrainingROI(
            roi_id="roi_c",
            dataset_id=dataset_id,
            calibration_id=calibration_id,
            name="C",
            wavelengths=wl,
            n_pixels=4,
            spatial_bounds=(0, 1, 0, 1),
            source_file="/tmp/cube_c.eiff",
            created_at="2026-03-28T10:00:00",
            provenance=sample_provenance,
        )
        ps.add_roi(roi, spectra)
        signals = []
        ps.rois_changed.connect(lambda: signals.append("changed"))
        ps.remove_roi("roi_c")
        assert ps.rois == {}
        assert signals == ["changed"]

    def test_deleted_roi_leaves_sample_set_stale_without_blocking_model_usage(
        self, project_store_path, wl, sample_provenance, qapp
    ):
        from enginir.project import SampleSet
        from enginir.viewer._state import ProjectState, TrainingROI

        ps = ProjectState()
        ps.open_project(project_store_path)
        assert ps.sample_sets_referencing_roi("roi_1") == ["sample_set_1"]
        assert ps.model_runs_for_sample_sets(["sample_set_1"]) == ["model_run_1"]

        ps.remove_roi("roi_1")

        reopened = ProjectState()
        reopened.open_project(project_store_path)
        assert "roi_1" not in reopened.rois
        assert reopened.sample_sets["sample_set_1"].roi_ids == ["roi_1"]
        assert reopened.model_runs["model_run_1"].sample_set_id == "sample_set_1"
        assert reopened.missing_roi_ids_for_sample_set("sample_set_1") == ["roi_1"]

        context = reopened.build_prediction_job_context(model_run_id="model_run_1")
        assert context.model_run.model_run_id == "model_run_1"

        with pytest.raises(ValueError, match="cannot be altered or refit"):
            reopened.fit_model_run(
                sample_set_id="sample_set_1",
                signal_domain_id="signal_domain_1",
                model_type="PLS",
                hyperparameters={"n_components": 1},
            )

        reopened.add_roi(
            TrainingROI(
                roi_id="roi_2",
                dataset_id="dataset_1",
                calibration_id="cal_1",
                name="ROI 2",
                wavelengths=wl,
                n_pixels=2,
                spatial_bounds=(0, 0, 0, 1),
                source_file="/data/train.eiff",
                created_at="2026-04-02T10:10:00+00:00",
                provenance=sample_provenance,
            ),
            np.zeros((2, 50), dtype=np.uint16),
        )
        reopened.create_sample_set(
            SampleSet(
                sample_set_id="sample_set_2",
                roi_ids=["roi_2"],
                target_id="target_moisture",
                observation_unit="pixel",
                target_level="roi",
                grouping_unit="roi",
                validation_unit="roi",
                weighting_policy="per_pixel_unweighted",
                reporting_level="roi",
                roi_aggregation="mean",
                preprocessing_recipe_id="recipe_1",
                dataset_ids=["dataset_1"],
                created_at="2026-04-02T10:11:00+00:00",
            )
        )
        assert reopened.sample_sets["sample_set_2"].roi_ids == ["roi_2"]

    def test_open_project_hydrates_first_class_workspace(self, project_store_path, qapp):
        from enginir.viewer._state import ProjectState

        ps = ProjectState()
        loaded_signals = []
        ps.project_loaded.connect(lambda: loaded_signals.append("loaded"))
        ps.open_project(project_store_path)
        assert ps.project_path == project_store_path
        assert ps.project_meta["schema_version"] == "viewer-v3"
        assert set(ps.datasets) == {"dataset_1"}
        assert ps.active_calibration_ids == {"dataset_1": "cal_1"}
        assert set(ps.calibrations) == {"cal_1"}
        assert set(ps.rois) == {"roi_1"}
        assert set(ps.targets) == {"target_moisture"}
        assert set(ps.recipes) == {"recipe_1"}
        assert set(ps.signal_domains) == {"signal_domain_1"}
        assert set(ps.sample_sets) == {"sample_set_1"}
        assert set(ps.model_runs) == {"model_run_1"}
        assert set(ps.predictions) == {"prediction_run_1"}
        assert len(ps.rois) == 1
        assert ps.rois["roi_1"].roi_id == "roi_1"
        assert ps.rois["roi_1"].dataset_id == "dataset_1"
        assert ps.rois["roi_1"].calibration_id == "cal_1"
        assert ps.rois["roi_1"].name == "ROI 1"
        assignment = ps.get_target_assignment("roi_1", "target_moisture")
        assert assignment is not None
        assert assignment.value == pytest.approx(12.3)
        assert ps.model_runs["model_run_1"].artifact_refs["cv_outputs"] == {}
        model_artifacts = ps.ensure_model_run_artifact_refs("model_run_1")
        np.testing.assert_array_equal(
            model_artifacts["cv_outputs"]["roi_ids"], np.array(["roi_1"], dtype=object)
        )
        np.testing.assert_allclose(
            model_artifacts["cv_outputs"]["y_true_roi"], np.array([12.3], dtype=np.float32)
        )
        np.testing.assert_allclose(
            model_artifacts["cv_outputs"]["y_pred_roi"], np.array([12.0], dtype=np.float32)
        )
        assert "y_pred" not in ps.predictions["prediction_run_1"].artifact_refs
        assert "spatial_map" not in ps.predictions["prediction_run_1"].artifact_refs
        prediction_artifacts = ps.ensure_prediction_artifact_refs("prediction_run_1")
        np.testing.assert_allclose(
            prediction_artifacts["y_pred"], np.array([11.9, 12.1], dtype=np.float32)
        )
        np.testing.assert_allclose(
            prediction_artifacts["spatial_map"],
            np.array([[11.9, 12.1]], dtype=np.float32),
        )

    def test_project_switch_clears_in_memory_predictors(self, project_store_path, tmp_path, qapp):
        from enginir.io.project_store import create_project
        from enginir.viewer._state import ProjectState

        ps = ProjectState()
        ps.open_project(project_store_path)
        ps.model_registry["model_run_1"] = object()
        ps.active_model_name = "model_run_1"

        same_project = str(project_store_path)
        ps.open_project(same_project)
        assert "model_run_1" in ps.model_registry
        assert ps.active_model_name == "model_run_1"

        other_path = str(tmp_path / "other_project.h5")
        create_project(other_path)
        ps.open_project(other_path)
        assert ps.model_registry == {}
        assert ps.active_model_name is None

    def test_open_project_skips_roi_spectra_materialization(self, project_store_path, qapp):
        from enginir.viewer._state import ProjectState
        from enginir.io.project_store import load_project_store

        with patch(
            "enginir.io.project_store.load_project_store",
            wraps=load_project_store,
        ) as mocked_load:
            ps = ProjectState()
            ps.open_project(project_store_path)

        assert mocked_load.call_count == 1
        assert mocked_load.call_args.kwargs == {}
        assert set(ps.rois) == {"roi_1"}
        assert not hasattr(ps.rois["roi_1"], "spectra")

    def test_define_and_assign_target_mutate_workspace_and_emit_signals(
        self, tmp_path, wl, sample_provenance, qapp
    ):
        from enginir.project import TargetAssignment, TargetDefinition
        from enginir.viewer._state import ProjectState, TrainingROI

        ps = ProjectState()
        ps.new_project(str(tmp_path / "proj.h5"))
        dataset_id, calibration_id = _register_dataset_context(ps, wl)
        roi = TrainingROI(
            roi_id="roi_d",
            dataset_id=dataset_id,
            calibration_id=calibration_id,
            name="D",
            wavelengths=wl,
            n_pixels=4,
            spatial_bounds=(0, 1, 0, 1),
            source_file="/tmp/cube.eiff",
            created_at="2026-03-28T10:00:00",
            provenance=sample_provenance,
        )
        ps.add_roi(roi, np.zeros((4, 50), dtype=np.uint16))

        target_signals = []
        assignment_signals = []
        ps.targets_changed.connect(lambda: target_signals.append("targets"))
        ps.target_assignments_changed.connect(lambda: assignment_signals.append("assignments"))

        ps.define_target(
            TargetDefinition(
                target_id="target_protein",
                name="protein",
                unit="% w/w",
                value_type="float",
                missing_value_policy="allowed",
                created_at="2026-04-02T10:10:00+00:00",
            )
        )
        ps.assign_target_to_roi(
            TargetAssignment(
                roi_id="roi_d",
                target_id="target_protein",
                value=19.4,
                assigned_at="2026-04-02T10:11:00+00:00",
            )
        )

        assert "target_protein" in ps.targets
        assignment = ps.get_target_assignment("roi_d", "target_protein")
        assert assignment is not None
        assert assignment.value == pytest.approx(19.4)
        assert target_signals == ["targets"]
        assert assignment_signals == ["assignments"]

    def test_hot_mutations_do_not_reload_full_workspace(
        self, tmp_path, wl, sample_provenance, qapp
    ):
        from enginir.project import TargetAssignment, TargetDefinition
        from enginir.viewer._state import ProjectState, TrainingROI

        ps = ProjectState()
        ps.new_project(str(tmp_path / "proj.h5"))
        dataset_id, calibration_id = _register_dataset_context(ps, wl)
        roi = TrainingROI(
            roi_id="roi_hot",
            dataset_id=dataset_id,
            calibration_id=calibration_id,
            name="Hot",
            wavelengths=wl,
            n_pixels=4,
            spatial_bounds=(0, 1, 0, 1),
            source_file="/tmp/cube.eiff",
            created_at="2026-03-28T10:00:00",
            provenance=sample_provenance,
        )

        with patch.object(ps.project, "_reload", side_effect=AssertionError("unexpected reload")):
            ps.add_roi(roi, np.zeros((4, 50), dtype=np.uint16))
            ps.define_target(
                TargetDefinition(
                    target_id="target_hot",
                    name="hot",
                    unit="% w/w",
                    value_type="float",
                    missing_value_policy="allowed",
                    created_at="2026-04-02T10:10:00+00:00",
                )
            )
            ps.assign_target_to_roi(
                TargetAssignment(
                    roi_id="roi_hot",
                    target_id="target_hot",
                    value=8.5,
                    assigned_at="2026-04-02T10:11:00+00:00",
                )
            )
            ps.remove_roi("roi_hot")

    def test_remaining_register_mutations_do_not_reload_full_workspace(self, tmp_path, wl, qapp):
        from enginir.project import (
            CalibrationSpec,
            DatasetRef,
            ModelRun,
            PredictionRun,
            SampleSet,
            SignalDomainContract,
            TargetDefinition,
        )
        from enginir.viewer._state import ProjectState

        ps = ProjectState()
        ps.new_project(str(tmp_path / "proj.h5"))
        dataset = DatasetRef(
            dataset_id="dataset_fast",
            source_path="/tmp/fast.eiff",
            source_format="eiff",
            source_modality="nir_hsi",
            cube_shape=(20, 15, 50),
            n_bands=50,
            wavelengths=wl,
            loader_metadata={},
            import_timestamp="2026-04-02T10:00:00+00:00",
        )
        calibration = CalibrationSpec(
            calibration_id="cal_fast",
            dataset_id="dataset_fast",
            mode="raw_dn",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=None,
            white_level=None,
            output_signal_unit="DN",
            created_at="2026-04-02T10:01:00+00:00",
        )
        signal_domain = SignalDomainContract(
            signal_domain_id="signal_fast",
            input_format="eiff",
            source_modality="nir_hsi",
            required_signal_unit="DN",
            required_calibration_mode="raw_dn",
            requires_dark_ref=False,
            requires_white_ref=False,
            calibrated_input_allowed=True,
            n_bands=50,
            wavelengths=wl,
            wavelength_tolerance_nm=0.0,
            band_mapping_policy="exact",
            output_dtype="float32",
            created_at="2026-04-02T10:02:00+00:00",
        )
        sample_set = SampleSet(
            sample_set_id="sample_fast",
            roi_ids=["roi_fast"],
            target_id="target_fast",
            observation_unit="pixel",
            target_level="roi",
            grouping_unit="roi",
            validation_unit="roi",
            weighting_policy="per_pixel_unweighted",
            reporting_level="roi",
            roi_aggregation="mean",
            preprocessing_recipe_id="recipe_fast",
            dataset_ids=["dataset_fast"],
            created_at="2026-04-02T10:03:00+00:00",
        )
        model_run = ModelRun(
            model_run_id="model_fast",
            sample_set_id="sample_fast",
            signal_domain_id="signal_fast",
            model_type="PLS",
            hyperparameters={"n_components": 2},
            cv_spec={"type": "GroupKFold", "n_splits": 2},
            metrics={"rmsecv": 1.0},
            artifact_refs={"coef": "artifact_refs/prediction_model/coef_"},
            created_at="2026-04-02T10:04:00+00:00",
        )
        prediction = PredictionRun(
            prediction_run_id="prediction_fast",
            dataset_id="dataset_fast",
            calibration_id="cal_fast",
            model_run_id="model_fast",
            signal_domain_id="signal_fast",
            preprocessing_recipe_id="recipe_fast",
            artifact_refs={"y_pred": np.array([1.0], dtype=np.float32)},
            created_at="2026-04-02T10:05:00+00:00",
        )

        with patch.object(ps.project, "_reload", side_effect=AssertionError("unexpected reload")):
            ps.register_dataset(dataset)
            ps.set_active_calibration(calibration)
            ps.define_target(
                TargetDefinition(
                    target_id="target_fast",
                    name="moisture",
                    unit="% w/w",
                    value_type="float",
                    missing_value_policy="allowed",
                    created_at="2026-04-02T10:01:30+00:00",
                )
            )
            ps.register_recipe("recipe_fast", recipe_payload())
            ps.register_signal_domain(signal_domain)
            _add_fast_roi(ps, wl)
            ps.create_sample_set(sample_set)
            ps.register_model_run(model_run)
            ps.register_prediction_run(prediction)

        assert ps.datasets["dataset_fast"].source_path == "/tmp/fast.eiff"
        assert ps.active_calibration_ids["dataset_fast"] == "cal_fast"
        assert ps.recipes["recipe_fast"]["filter"] == "none"
        assert ps.signal_domains["signal_fast"].required_calibration_mode == "raw_dn"
        assert ps.sample_sets["sample_fast"].dataset_ids == ["dataset_fast"]
        assert ps.model_runs["model_fast"].sample_set_id == "sample_fast"
        assert "y_pred" not in ps.predictions["prediction_fast"].artifact_refs
        np.testing.assert_allclose(
            ps.ensure_prediction_artifact_refs("prediction_fast")["y_pred"],
            np.array([1.0], dtype=np.float32),
        )

    def test_create_sample_set_rejects_invalid_relationships_before_persist(
        self, tmp_path, wl, qapp
    ):
        from enginir.project import CalibrationSpec, DatasetRef, ProjectRelationshipError, SampleSet
        from enginir.viewer._state import ProjectState

        ps = ProjectState()
        ps.new_project(str(tmp_path / "proj_invalid_sample_set.h5"))
        ps.register_dataset(
            DatasetRef(
                dataset_id="dataset_fast",
                source_path="/tmp/fast.eiff",
                source_format="eiff",
                source_modality="nir_hsi",
                cube_shape=(20, 15, 50),
                n_bands=50,
                wavelengths=wl,
                loader_metadata={},
                import_timestamp="2026-04-02T10:00:00+00:00",
            )
        )
        ps.set_active_calibration(
            CalibrationSpec(
                calibration_id="cal_fast",
                dataset_id="dataset_fast",
                mode="raw_dn",
                dark_ref_source=None,
                white_ref_source=None,
                dark_ref_data=None,
                white_ref_data=None,
                black_level=None,
                white_level=None,
                output_signal_unit="DN",
                created_at="2026-04-02T10:01:00+00:00",
            )
        )
        ps.register_recipe("recipe_fast", recipe_payload())

        sample_set = SampleSet(
            sample_set_id="sample_fast",
            roi_ids=["roi_fast"],
            target_id="target_missing",
            observation_unit="pixel",
            target_level="roi",
            grouping_unit="roi",
            validation_unit="roi",
            weighting_policy="per_pixel_unweighted",
            reporting_level="roi",
            roi_aggregation="mean",
            preprocessing_recipe_id="recipe_fast",
            dataset_ids=["dataset_fast"],
            created_at="2026-04-02T10:03:00+00:00",
        )

        _add_fast_roi(ps, wl)
        with patch("enginir.io.project_store.save_sample_set") as save_sample_set:
            with pytest.raises(
                ProjectRelationshipError, match="missing target_id 'target_missing'"
            ):
                ps.create_sample_set(sample_set)

        save_sample_set.assert_not_called()
        assert "sample_fast" not in ps.sample_sets

    def test_register_model_run_rejects_invalid_relationships_before_persist(
        self, tmp_path, wl, qapp
    ):
        from enginir.project import (
            CalibrationSpec,
            DatasetRef,
            ModelRun,
            ProjectRelationshipError,
            SampleSet,
            SignalDomainContract,
            TargetDefinition,
        )
        from enginir.viewer._state import ProjectState

        ps = ProjectState()
        ps.new_project(str(tmp_path / "proj_invalid_model_run.h5"))
        ps.register_dataset(
            DatasetRef(
                dataset_id="dataset_fast",
                source_path="/tmp/fast.eiff",
                source_format="eiff",
                source_modality="nir_hsi",
                cube_shape=(20, 15, 50),
                n_bands=50,
                wavelengths=wl,
                loader_metadata={},
                import_timestamp="2026-04-02T10:00:00+00:00",
            )
        )
        ps.set_active_calibration(
            CalibrationSpec(
                calibration_id="cal_fast",
                dataset_id="dataset_fast",
                mode="raw_dn",
                dark_ref_source=None,
                white_ref_source=None,
                dark_ref_data=None,
                white_ref_data=None,
                black_level=None,
                white_level=None,
                output_signal_unit="DN",
                created_at="2026-04-02T10:01:00+00:00",
            )
        )
        ps.define_target(
            TargetDefinition(
                target_id="target_fast",
                name="moisture",
                unit="% w/w",
                value_type="float",
                missing_value_policy="allowed",
                created_at="2026-04-02T10:02:00+00:00",
            )
        )
        ps.register_recipe("recipe_fast", recipe_payload())
        ps.register_signal_domain(
            SignalDomainContract(
                signal_domain_id="signal_fast",
                input_format="eiff",
                source_modality="nir_hsi",
                required_signal_unit="DN",
                required_calibration_mode="raw_dn",
                requires_dark_ref=False,
                requires_white_ref=False,
                calibrated_input_allowed=True,
                n_bands=50,
                wavelengths=wl,
                wavelength_tolerance_nm=0.0,
                band_mapping_policy="exact",
                output_dtype="float32",
                created_at="2026-04-02T10:02:00+00:00",
            )
        )
        _add_fast_roi(ps, wl)
        ps.create_sample_set(
            SampleSet(
                sample_set_id="sample_fast",
                roi_ids=["roi_fast"],
                target_id="target_fast",
                observation_unit="pixel",
                target_level="roi",
                grouping_unit="roi",
                validation_unit="roi",
                weighting_policy="per_pixel_unweighted",
                reporting_level="roi",
                roi_aggregation="mean",
                preprocessing_recipe_id="recipe_fast",
                dataset_ids=["dataset_fast"],
                created_at="2026-04-02T10:03:00+00:00",
            )
        )

        model_run = ModelRun(
            model_run_id="model_fast",
            sample_set_id="sample_fast",
            signal_domain_id="signal_missing",
            model_type="PLS",
            hyperparameters={"n_components": 2},
            cv_spec={"type": "GroupKFold", "n_splits": 2},
            metrics={"rmsecv": 1.0},
            artifact_refs={"coef": "artifact_refs/prediction_model/coef_"},
            created_at="2026-04-02T10:04:00+00:00",
        )

        with patch("enginir.io.project_store.save_model_run") as save_model_run:
            with pytest.raises(
                ProjectRelationshipError, match="missing signal_domain_id 'signal_missing'"
            ):
                ps.register_model_run(model_run)

        save_model_run.assert_not_called()
        assert "model_fast" not in ps.model_runs

    def test_register_prediction_run_rejects_invalid_relationships_before_persist(
        self, tmp_path, wl, qapp
    ):
        from enginir.project import (
            CalibrationSpec,
            DatasetRef,
            PredictionRun,
            ProjectRelationshipError,
            SignalDomainContract,
        )
        from enginir.viewer._state import ProjectState

        ps = ProjectState()
        ps.new_project(str(tmp_path / "proj_invalid_prediction_run.h5"))
        ps.register_dataset(
            DatasetRef(
                dataset_id="dataset_fast",
                source_path="/tmp/fast.eiff",
                source_format="eiff",
                source_modality="nir_hsi",
                cube_shape=(20, 15, 50),
                n_bands=50,
                wavelengths=wl,
                loader_metadata={},
                import_timestamp="2026-04-02T10:00:00+00:00",
            )
        )
        ps.set_active_calibration(
            CalibrationSpec(
                calibration_id="cal_fast",
                dataset_id="dataset_fast",
                mode="raw_dn",
                dark_ref_source=None,
                white_ref_source=None,
                dark_ref_data=None,
                white_ref_data=None,
                black_level=None,
                white_level=None,
                output_signal_unit="DN",
                created_at="2026-04-02T10:01:00+00:00",
            )
        )
        ps.register_recipe("recipe_fast", recipe_payload())
        ps.register_signal_domain(
            SignalDomainContract(
                signal_domain_id="signal_fast",
                input_format="eiff",
                source_modality="nir_hsi",
                required_signal_unit="DN",
                required_calibration_mode="raw_dn",
                requires_dark_ref=False,
                requires_white_ref=False,
                calibrated_input_allowed=True,
                n_bands=50,
                wavelengths=wl,
                wavelength_tolerance_nm=0.0,
                band_mapping_policy="exact",
                output_dtype="float32",
                created_at="2026-04-02T10:02:00+00:00",
            )
        )

        prediction = PredictionRun(
            prediction_run_id="prediction_fast",
            dataset_id="dataset_fast",
            calibration_id="cal_fast",
            model_run_id="model_missing",
            signal_domain_id="signal_fast",
            preprocessing_recipe_id="recipe_fast",
            artifact_refs={"y_pred": np.array([1.0], dtype=np.float32)},
            created_at="2026-04-02T10:05:00+00:00",
        )

        with patch("enginir.io.project_store.save_prediction_run") as save_prediction_run:
            with pytest.raises(
                ProjectRelationshipError, match="missing model_run_id 'model_missing'"
            ):
                ps.register_prediction_run(prediction)

        save_prediction_run.assert_not_called()
        assert "prediction_fast" not in ps.predictions

    def test_register_prediction_run_rejects_calibration_from_other_dataset(
        self, tmp_path, wl, qapp
    ):
        from enginir.project import (
            CalibrationSpec,
            DatasetRef,
            ModelRun,
            PredictionRun,
            ProjectRelationshipError,
            SampleSet,
            SignalDomainContract,
            TargetDefinition,
        )
        from enginir.viewer._state import ProjectState

        ps = ProjectState()
        ps.new_project(str(tmp_path / "proj_prediction_cross_dataset_calibration.h5"))
        ps.register_dataset(
            DatasetRef(
                dataset_id="dataset_a",
                source_path="/tmp/a.eiff",
                source_format="eiff",
                source_modality="nir_hsi",
                cube_shape=(20, 15, 50),
                n_bands=50,
                wavelengths=wl,
                loader_metadata={},
                import_timestamp="2026-04-02T10:00:00+00:00",
            )
        )
        ps.register_dataset(
            DatasetRef(
                dataset_id="dataset_b",
                source_path="/tmp/b.eiff",
                source_format="eiff",
                source_modality="nir_hsi",
                cube_shape=(20, 15, 50),
                n_bands=50,
                wavelengths=wl,
                loader_metadata={},
                import_timestamp="2026-04-02T10:00:30+00:00",
            )
        )
        ps.set_active_calibration(
            CalibrationSpec(
                calibration_id="cal_a",
                dataset_id="dataset_a",
                mode="raw_dn",
                dark_ref_source=None,
                white_ref_source=None,
                dark_ref_data=None,
                white_ref_data=None,
                black_level=None,
                white_level=None,
                output_signal_unit="DN",
                created_at="2026-04-02T10:01:00+00:00",
            )
        )
        ps.set_active_calibration(
            CalibrationSpec(
                calibration_id="cal_b",
                dataset_id="dataset_b",
                mode="raw_dn",
                dark_ref_source=None,
                white_ref_source=None,
                dark_ref_data=None,
                white_ref_data=None,
                black_level=None,
                white_level=None,
                output_signal_unit="DN",
                created_at="2026-04-02T10:01:30+00:00",
            )
        )
        ps.define_target(
            TargetDefinition(
                target_id="target_fast",
                name="moisture",
                unit="% w/w",
                value_type="float",
                missing_value_policy="allowed",
                created_at="2026-04-02T10:02:00+00:00",
            )
        )
        ps.register_recipe("recipe_fast", recipe_payload())
        ps.register_signal_domain(
            SignalDomainContract(
                signal_domain_id="signal_fast",
                input_format="eiff",
                source_modality="nir_hsi",
                required_signal_unit="DN",
                required_calibration_mode="raw_dn",
                requires_dark_ref=False,
                requires_white_ref=False,
                calibrated_input_allowed=True,
                n_bands=50,
                wavelengths=wl,
                wavelength_tolerance_nm=0.0,
                band_mapping_policy="exact",
                output_dtype="float32",
                created_at="2026-04-02T10:02:45+00:00",
            )
        )
        _add_fast_roi(ps, wl, "dataset_a", "cal_a")
        ps.create_sample_set(
            SampleSet(
                sample_set_id="sample_fast",
                roi_ids=["roi_fast"],
                target_id="target_fast",
                observation_unit="pixel",
                target_level="roi",
                grouping_unit="roi",
                validation_unit="roi",
                weighting_policy="per_pixel_unweighted",
                reporting_level="roi",
                roi_aggregation="mean",
                preprocessing_recipe_id="recipe_fast",
                dataset_ids=["dataset_a"],
                created_at="2026-04-02T10:03:00+00:00",
            )
        )
        ps.register_model_run(
            ModelRun(
                model_run_id="model_fast",
                sample_set_id="sample_fast",
                signal_domain_id="signal_fast",
                model_type="PLS",
                hyperparameters={"n_components": 2},
                cv_spec={"type": "GroupKFold", "n_splits": 2},
                metrics={"rmsecv": 1.0},
                artifact_refs={"coef": "artifact_refs/prediction_model/coef_"},
                created_at="2026-04-02T10:04:00+00:00",
            )
        )

        prediction = PredictionRun(
            prediction_run_id="prediction_cross_dataset",
            dataset_id="dataset_a",
            calibration_id="cal_b",
            model_run_id="model_fast",
            signal_domain_id="signal_fast",
            preprocessing_recipe_id="recipe_fast",
            artifact_refs={"y_pred": np.array([1.0], dtype=np.float32)},
            created_at="2026-04-02T10:05:00+00:00",
        )

        with patch("enginir.io.project_store.save_prediction_run") as save_prediction_run:
            with pytest.raises(
                ProjectRelationshipError,
                match="PredictionRun 'prediction_cross_dataset' uses calibration 'cal_b' of dataset 'dataset_b'",
            ):
                ps.register_prediction_run(prediction)

        save_prediction_run.assert_not_called()
        assert "prediction_cross_dataset" not in ps.predictions

    def test_save_and_reopen_preserves_ids_and_references(
        self, tmp_path, wl, sample_provenance, qapp
    ):
        from enginir.project import CalibrationSpec, DatasetRef, TargetAssignment, TargetDefinition
        from enginir.viewer._state import ProjectState, TrainingROI

        path = str(tmp_path / "roundtrip.h5")
        ps = ProjectState()
        ps.new_project(path)
        ps.register_dataset(
            DatasetRef(
                dataset_id="dataset_roundtrip",
                source_path="/tmp/sample.eiff",
                source_format="eiff",
                source_modality="nir_hsi",
                cube_shape=(20, 15, 50),
                n_bands=50,
                wavelengths=wl,
                loader_metadata={},
                import_timestamp="2026-04-02T10:00:00+00:00",
            )
        )
        ps.set_active_calibration(
            CalibrationSpec(
                calibration_id="cal_roundtrip",
                dataset_id="dataset_roundtrip",
                mode="dark_white_reference",
                dark_ref_source=None,
                white_ref_source=None,
                dark_ref_data=None,
                white_ref_data=None,
                black_level=128.0,
                white_level=3500.0,
                output_signal_unit="Reflectance",
                created_at="2026-04-02T10:01:00+00:00",
            )
        )
        ps.define_target(
            TargetDefinition(
                target_id="target_roundtrip",
                name="moisture",
                unit="% w/w",
                value_type="float",
                missing_value_policy="allowed",
                created_at="2026-04-02T10:02:00+00:00",
            )
        )
        ps.add_roi(
            TrainingROI(
                roi_id="roi_roundtrip",
                dataset_id="dataset_roundtrip",
                calibration_id="cal_roundtrip",
                name="ROI Roundtrip",
                wavelengths=wl,
                n_pixels=6,
                spatial_bounds=(0, 1, 0, 2),
                source_file="/tmp/sample.eiff",
                created_at="2026-04-02T10:03:00+00:00",
                provenance={
                    **sample_provenance,
                    "dataset_id": "dataset_roundtrip",
                    "calibration_id": "cal_roundtrip",
                },
            ),
            np.zeros((6, 50), dtype=np.uint16),
        )
        ps.assign_target_to_roi(
            TargetAssignment(
                roi_id="roi_roundtrip",
                target_id="target_roundtrip",
                value=41.2,
                assigned_at="2026-04-02T10:04:00+00:00",
            )
        )

        reopened = ProjectState()
        reopened.open_project(path)
        assert reopened.rois["roi_roundtrip"].dataset_id == "dataset_roundtrip"
        assert reopened.rois["roi_roundtrip"].calibration_id == "cal_roundtrip"
        assert reopened.active_calibration_ids["dataset_roundtrip"] == "cal_roundtrip"
        assignment = reopened.get_target_assignment("roi_roundtrip", "target_roundtrip")
        assert assignment is not None
        assert assignment.value == pytest.approx(41.2)


def test_add_roi_validates_relationships_before_writing(tmp_path, wl, sample_provenance, qapp):
    from enginir.project import ProjectRelationshipError
    from enginir.viewer._state import ProjectState, TrainingROI

    project_path = str(tmp_path / "proj.h5")
    ps = ProjectState()
    ps.new_project(project_path)
    dataset_id, _ = _register_dataset_context(ps, wl)
    provenance = {**sample_provenance, "dataset_id": dataset_id, "calibration_id": "cal_missing"}
    roi = TrainingROI(roi_id="roi_orphan", dataset_id=dataset_id, calibration_id="cal_missing",
                      name="orphan", wavelengths=wl, n_pixels=4, spatial_bounds=(0, 1, 0, 1),
                      source_file="/tmp/cube.eiff", created_at="2026-03-28T10:00:00",
                      provenance=provenance)

    with pytest.raises(ProjectRelationshipError, match="missing calibration_id 'cal_missing'"):
        ps.add_roi(roi, np.zeros((4, 50), dtype=np.uint16))

    assert "roi_orphan" not in ps.rois
    reopened = ProjectState()
    reopened.open_project(project_path)
    assert "roi_orphan" not in reopened.rois
