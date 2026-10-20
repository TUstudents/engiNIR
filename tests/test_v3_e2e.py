"""End-to-end v3 project integration test using the Keile EIFF dataset."""

from __future__ import annotations

import numpy as np
import pytest

from enginir.viewer._app import ensure_app
from enginir.viewer._state import ProjectState

from ._v3_fixture import (
    KEILE_EIFF_PATH,
    KEILE_CALIBRATION_ID,
    KEILE_DATASET_ID,
    KEILE_MODEL_RUN_ID,
    KEILE_PREDICTION_RUN_ID,
    KEILE_RECIPE_ID,
    KEILE_SAMPLE_SET_ID,
    KEILE_SIGNAL_DOMAIN_ID,
    KEILE_TARGET_ID,
    build_keile_v3_project,
)


def test_keile_v3_project_fixture_covers_full_project_graph(tmp_path):
    if not KEILE_EIFF_PATH.exists():
        pytest.skip(f"Keile EIFF fixture is not available at {KEILE_EIFF_PATH}")
    ensure_app()
    fixture = build_keile_v3_project(tmp_path / "keile_v3_fixture.h5")
    store = fixture["store"]
    dataset_store = store["datasets"][KEILE_DATASET_ID]

    assert len(store["datasets"]) == 1
    assert dataset_store["active_calibration_id"] == KEILE_CALIBRATION_ID
    assert len(dataset_store["calibrations"]) == 1
    assert len(store["rois"]) == 7
    assert len(store["targets"]) == 1
    assert len(store["target_assignments"]) == 1
    assert len(store["target_assignments"][KEILE_TARGET_ID]) == 7
    assert len(store["recipes"]) == 1
    assert len(store["signal_domains"]) == 1
    assert len(store["sample_sets"]) == 1
    assert len(store["model_runs"]) == 1
    assert len(store["predictions"]) == 1

    reopened = ProjectState()
    reopened.open_project(fixture["project_path"])

    assert KEILE_DATASET_ID in reopened.datasets
    assert KEILE_CALIBRATION_ID in reopened.calibrations
    assert KEILE_TARGET_ID in reopened.targets
    assert KEILE_SAMPLE_SET_ID in reopened.sample_sets
    assert KEILE_SIGNAL_DOMAIN_ID in reopened.signal_domains
    assert KEILE_MODEL_RUN_ID in reopened.model_runs
    assert KEILE_PREDICTION_RUN_ID in reopened.predictions

    dataset = reopened.datasets[KEILE_DATASET_ID]
    sample_set = reopened.sample_sets[KEILE_SAMPLE_SET_ID]
    model_run = reopened.model_runs[KEILE_MODEL_RUN_ID]
    prediction_run = reopened.predictions[KEILE_PREDICTION_RUN_ID]

    assert dataset.source_path == fixture["data_path"]
    assert dataset.cube_shape == fixture["cube_shape"]
    np.testing.assert_allclose(dataset.wavelengths, fixture["wavelengths"])

    assert sample_set.target_id == KEILE_TARGET_ID
    assert sample_set.preprocessing_recipe_id == KEILE_RECIPE_ID
    assert sample_set.roi_ids == fixture["roi_ids"]
    assert sample_set.dataset_ids == [KEILE_DATASET_ID]

    assert model_run.sample_set_id == KEILE_SAMPLE_SET_ID
    assert model_run.signal_domain_id == KEILE_SIGNAL_DOMAIN_ID
    assert model_run.metrics["primary_reporting_level"] == "roi"
    assert model_run.metrics["primary_roi_aggregation"] == "mean"
    assert int(model_run.metrics["n_rois"]) == 7

    assert prediction_run.dataset_id == KEILE_DATASET_ID
    assert prediction_run.calibration_id == KEILE_CALIBRATION_ID
    assert prediction_run.model_run_id == KEILE_MODEL_RUN_ID
    assert prediction_run.signal_domain_id == KEILE_SIGNAL_DOMAIN_ID
    assert prediction_run.preprocessing_recipe_id == KEILE_RECIPE_ID
    assert "spatial_map" not in prediction_run.artifact_refs
    prediction_artifacts = reopened.ensure_prediction_artifact_refs(KEILE_PREDICTION_RUN_ID)
    spatial_map = np.asarray(prediction_artifacts["spatial_map"], dtype=np.float32)
    assert spatial_map.shape == fixture["prediction_shape"]
    assert np.isfinite(spatial_map).all()

    for roi_id in fixture["roi_ids"]:
        roi = reopened.rois[roi_id]
        assignment = reopened.get_target_assignment(roi_id, KEILE_TARGET_ID)
        assert roi.dataset_id == KEILE_DATASET_ID
        assert roi.calibration_id == KEILE_CALIBRATION_ID
        assert assignment is not None
