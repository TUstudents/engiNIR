"""Tests for the v3 sample-set builder."""

from __future__ import annotations

import numpy as np
import pytest

from enginir.io.project_store import (
    create_project,
    save_calibration_spec,
    save_dataset_ref,
    save_target_assignment,
    save_target_definition,
    save_training_roi,

)
from enginir.modeling import build_sample_set
from enginir.viewer._state import ProjectState

from ._project_payloads import roi_provenance


def _create_phase5_project(path: str, *, mismatch_second_roi: bool = False) -> None:
    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    second_wavelengths = (
        np.array([1000.0, 1150.0, 1200.0], dtype=np.float32) if mismatch_second_roi else wavelengths
    )
    create_project(path)
    save_dataset_ref(
        path,
        "dataset_1",
        {
            "source_path": "/data/train.eiff",
            "source_format": "eiff",
            "source_modality": "nir_hsi",
            "cube_shape": (3, 3, 3),
            "n_bands": 3,
            "wavelengths": wavelengths,
            "loader_metadata": {"instrument": "alpha"},
            "import_timestamp": "2026-04-02T10:00:00+00:00",
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
    save_training_roi(
        path,
        "roi_1",
        {
            "dataset_id": "dataset_1",
            "calibration_id": "cal_1",
            "name": "ROI 1",
            "source_file": "/data/train.eiff",
            "n_pixels": 2,
            "created_at": "2026-04-02T10:03:00+00:00",
            "spatial_bounds": (0, 0, 0, 1),
            "wavelengths": wavelengths,
            "provenance": roi_provenance(created_at="2026-04-02T10:05:00+00:00"),
        },
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16),
    )
    save_training_roi(
        path,
        "roi_2",
        {
            "dataset_id": "dataset_1",
            "calibration_id": "cal_1",
            "name": "ROI 2",
            "source_file": "/data/train.eiff",
            "n_pixels": 1,
            "created_at": "2026-04-02T10:04:00+00:00",
            "spatial_bounds": (1, 1, 0, 0),
            "wavelengths": second_wavelengths,
            "provenance": roi_provenance(created_at="2026-04-02T10:06:00+00:00"),
        },
        np.array([[7, 8, 9]], dtype=np.uint16),
    )
    save_training_roi(
        path,
        "roi_3",
        {
            "dataset_id": "dataset_1",
            "calibration_id": "cal_1",
            "name": "ROI 3",
            "source_file": "/data/train.eiff",
            "n_pixels": 1,
            "created_at": "2026-04-02T10:05:00+00:00",
            "spatial_bounds": (2, 2, 0, 0),
            "wavelengths": wavelengths,
            "provenance": roi_provenance(created_at="2026-04-02T10:07:00+00:00"),
        },
        np.array([[10, 11, 12]], dtype=np.uint16),
    )
    save_target_assignment(
        path,
        "target_moisture",
        "roi_1",
        {"value": 10.0, "assigned_at": "2026-04-02T10:06:00+00:00"},
    )
    save_target_assignment(
        path,
        "target_moisture",
        "roi_2",
        {"value": 20.0, "assigned_at": "2026-04-02T10:07:00+00:00"},
    )


def test_build_sample_set_from_project_rois_and_assignments(tmp_path):
    project_path = str(tmp_path / "phase5_sample_set.h5")
    _create_phase5_project(project_path)

    state = ProjectState()
    state.open_project(project_path)

    result = build_sample_set(
        project_path=project_path,
        rois=state.rois,
        target_assignments=state.target_assignments,
        target_id="target_moisture",
        preprocessing_recipe_id="recipe_none",
        sample_set_id="sample_set_1",
    )

    np.testing.assert_array_equal(
        result.X,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64),
    )
    assert result.X.dtype == np.uint16
    np.testing.assert_array_equal(result.y, np.array([10.0, 10.0, 20.0], dtype=np.float64))
    np.testing.assert_array_equal(
        result.groups, np.array(["roi_1", "roi_1", "roi_2"], dtype=object)
    )
    np.testing.assert_array_equal(
        result.wavelengths, np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    )

    sample_set = result.sample_set
    assert sample_set.sample_set_id == "sample_set_1"
    assert sample_set.roi_ids == ["roi_1", "roi_2"]
    assert sample_set.target_id == "target_moisture"
    assert sample_set.observation_unit == "pixel"
    assert sample_set.target_level == "roi"
    assert sample_set.grouping_unit == "roi"
    assert sample_set.validation_unit == "roi"
    assert sample_set.weighting_policy == "per_pixel_unweighted"
    assert sample_set.reporting_level == "roi"
    assert sample_set.roi_aggregation == "mean"
    assert sample_set.preprocessing_recipe_id == "recipe_none"
    assert sample_set.dataset_ids == ["dataset_1"]


def test_sample_set_semantics_are_fixed():
    from enginir.project._models import SAMPLE_SET_SEMANTICS

    assert SAMPLE_SET_SEMANTICS == {
        "observation_unit": "pixel",
        "target_level": "roi",
        "grouping_unit": "roi",
        "validation_unit": "roi",
        "weighting_policy": "per_pixel_unweighted",
        "reporting_level": "roi",
        "roi_aggregation": "mean",
    }


def test_build_sample_set_excludes_unlabeled_rois(tmp_path):
    project_path = str(tmp_path / "phase5_sample_set_unlabeled.h5")
    _create_phase5_project(project_path)

    state = ProjectState()
    state.open_project(project_path)

    result = build_sample_set(
        project_path=project_path,
        rois=state.rois,
        target_assignments=state.target_assignments,
        target_id="target_moisture",
        preprocessing_recipe_id="recipe_none",
    )

    assert "roi_3" not in result.sample_set.roi_ids
    assert "roi_3" not in result.groups.tolist()


def test_build_sample_set_rejects_incompatible_wavelength_axes(tmp_path):
    project_path = str(tmp_path / "phase5_sample_set_mismatch.h5")
    _create_phase5_project(project_path, mismatch_second_roi=True)

    state = ProjectState()
    state.open_project(project_path)

    with pytest.raises(ValueError, match="different wavelength axis"):
        build_sample_set(
            project_path=project_path,
            rois=state.rois,
            target_assignments=state.target_assignments,
            target_id="target_moisture",
            preprocessing_recipe_id="recipe_none",
        )


def test_build_sample_set_rejects_repeated_rois(tmp_path):
    project_path = str(tmp_path / "phase5_sample_set_repeated.h5")
    _create_phase5_project(project_path)
    state = ProjectState()
    state.open_project(project_path)

    with pytest.raises(ValueError, match=r"more than once.*roi_1"):
        build_sample_set(
            project_path=project_path,
            rois=state.rois,
            target_assignments=state.target_assignments,
            target_id="target_moisture",
            preprocessing_recipe_id="recipe_none",
            roi_ids=["roi_1", "roi_2", "roi_1"],
        )


def test_build_sample_set_needs_at_least_one_labelled_roi(tmp_path):
    project_path = str(tmp_path / "phase5_sample_set_empty.h5")
    _create_phase5_project(project_path)
    state = ProjectState()
    state.open_project(project_path)
    common = {"project_path": project_path, "rois": state.rois,
              "target_assignments": state.target_assignments, "target_id": "target_moisture",
              "preprocessing_recipe_id": "recipe_none"}

    with pytest.raises(ValueError, match="No ROIs selected"):
        build_sample_set(**common, roi_ids=[])
    with pytest.raises(ValueError, match="No selected ROI has a value for target"):
        build_sample_set(**common, roi_ids=["roi_3"])  # roi_3 has no assignment
