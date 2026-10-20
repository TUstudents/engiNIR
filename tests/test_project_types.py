"""Contract tests for typed v3 project-domain models."""

from __future__ import annotations

import copy
from dataclasses import fields

import numpy as np
import pytest

from enginir.project import (
    CalibrationSpec,
    DatasetRef,
    ModelRun,
    PredictionRun,
    ProjectRelationshipError,
    SampleSet,
    SignalDomainContract,
    TargetAssignment,
    TargetDefinition,
    TrainingROI,
    validate_project_relationships,
)


def _assert_round_trip(instance):
    restored = type(instance)(**instance.to_dict())
    for field in fields(type(instance)):
        left = getattr(instance, field.name)
        right = getattr(restored, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert right == left


def _roi_provenance(
    *, dataset_id: str = "dataset_1", calibration_id: str = "cal_1"
) -> dict[str, str]:
    return {
        "dataset_id": dataset_id,
        "calibration_id": calibration_id,
        "signal_unit": "Absorbance",
        "preview_recipe_summary": "none",
        "created_at": "2026-04-02T10:02:00+00:00",
    }


@pytest.fixture()
def wavelengths():
    return np.linspace(900, 1700, 50, dtype=np.float32)


@pytest.fixture()
def valid_project_graph(wavelengths):
    return {
        "datasets": {
            "dataset_1": DatasetRef(
                dataset_id="dataset_1",
                source_path="/data/train.eiff",
                source_format="eiff",
                source_modality="nir_hsi",
                cube_shape=(20, 15, 50),
                n_bands=50,
                wavelengths=wavelengths,
                loader_metadata={"sensor": "AlphaVision"},
                import_timestamp="2026-04-02T10:00:00+00:00",
            )
        },
        "calibrations": {
            "cal_1": CalibrationSpec(
                calibration_id="cal_1",
                dataset_id="dataset_1",
                mode="dark_white_reference",
                dark_ref_source=None,
                white_ref_source=None,
                dark_ref_data=np.zeros((15, 50), dtype=np.float32),
                white_ref_data=np.ones((15, 50), dtype=np.float32),
                black_level=None,
                white_level=None,
                output_signal_unit="Absorbance",
                created_at="2026-04-02T10:01:00+00:00",
            )
        },
        "rois": {
            "roi_1": TrainingROI(
                roi_id="roi_1",
                dataset_id="dataset_1",
                calibration_id="cal_1",
                name="ROI 1",
                wavelengths=wavelengths,
                n_pixels=12,
                spatial_bounds=(0, 2, 0, 3),
                source_file="/data/train.eiff",
                created_at="2026-04-02T10:02:00+00:00",
                provenance=_roi_provenance(),
            )
        },
        "targets": {
            "target_1": TargetDefinition(
                target_id="target_1",
                name="moisture",
                unit="% w/w",
                value_type="float",
                missing_value_policy="allowed",
                created_at="2026-04-02T10:03:00+00:00",
            )
        },
        "target_assignments": [
            TargetAssignment(
                roi_id="roi_1",
                target_id="target_1",
                value=12.3,
                assigned_at="2026-04-02T10:04:00+00:00",
            )
        ],
        "recipes": {"recipe_1": {"filter": "sg_d1"}},
        "signal_domains": {
            "signal_domain_1": SignalDomainContract(
                signal_domain_id="signal_domain_1",
                input_format="eiff_raw",
                source_modality="nir_hsi",
                required_signal_unit="Absorbance",
                required_calibration_mode="dark_white_reference",
                requires_dark_ref=True,
                requires_white_ref=True,
                calibrated_input_allowed=False,
                n_bands=50,
                wavelengths=wavelengths,
                wavelength_tolerance_nm=0.0,
                band_mapping_policy="exact",
                output_dtype="float32",
                created_at="2026-04-02T10:04:00+00:00",
            )
        },
        "sample_sets": {
            "sample_set_1": SampleSet(
                sample_set_id="sample_set_1",
                roi_ids=["roi_1"],
                target_id="target_1",
                observation_unit="pixel",
                target_level="roi",
                grouping_unit="roi",
                validation_unit="roi",
                weighting_policy="per_pixel_unweighted",
                reporting_level="roi",
                roi_aggregation="mean",
                preprocessing_recipe_id="recipe_1",
                dataset_ids=["dataset_1"],
                created_at="2026-04-02T10:05:00+00:00",
            )
        },
        "model_runs": {
            "model_run_1": ModelRun(
                model_run_id="model_run_1",
                sample_set_id="sample_set_1",
                signal_domain_id="signal_domain_1",
                model_type="PLS",
                hyperparameters={"n_components": 3},
                cv_spec={"type": "GroupKFold", "n_splits": 5},
                metrics={"rmsecv": 1.23},
                artifact_refs={"weights": "artifact_refs/prediction_model/coef_"},
                created_at="2026-04-02T10:06:00+00:00",
            )
        },
        "predictions": {
            "prediction_run_1": PredictionRun(
                prediction_run_id="prediction_run_1",
                dataset_id="dataset_1",
                calibration_id="cal_1",
                model_run_id="model_run_1",
                signal_domain_id="signal_domain_1",
                preprocessing_recipe_id="recipe_1",
                artifact_refs={"y_pred": "/predictions/pred_1/y_pred"},
                created_at="2026-04-02T10:07:00+00:00",
            )
        },
    }


def test_project_models_round_trip_representative_instances(wavelengths):
    instances = [
        DatasetRef(
            dataset_id="dataset_1",
            source_path="/data/train.eiff",
            source_format="eiff",
            source_modality="nir_hsi",
            cube_shape=(20, 15, 50),
            n_bands=50,
            wavelengths=wavelengths,
            loader_metadata={"sensor": "AlphaVision"},
            import_timestamp="2026-04-02T10:00:00+00:00",
        ),
        CalibrationSpec(
            calibration_id="cal_1",
            dataset_id="dataset_1",
            mode="dark_white_reference",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=np.zeros((2, 3), dtype=np.float32),
            white_ref_data=np.ones((2, 3), dtype=np.float32),
            black_level=None,
            white_level=None,
            output_signal_unit="Absorbance",
            created_at="2026-04-02T10:01:00+00:00",
        ),
        TrainingROI(
            roi_id="roi_1",
            dataset_id="dataset_1",
            calibration_id="cal_1",
            name="ROI 1",
            wavelengths=wavelengths,
            n_pixels=12,
            spatial_bounds=(0, 2, 0, 3),
            source_file="/data/train.eiff",
            created_at="2026-04-02T10:02:00+00:00",
            provenance=_roi_provenance(),
        ),
        SignalDomainContract(
            signal_domain_id="signal_domain_1",
            input_format="eiff_raw",
            source_modality="nir_hsi",
            required_signal_unit="Absorbance",
            required_calibration_mode="dark_white_reference",
            requires_dark_ref=True,
            requires_white_ref=True,
            calibrated_input_allowed=False,
            n_bands=50,
            wavelengths=wavelengths,
            wavelength_tolerance_nm=0.0,
            band_mapping_policy="exact",
            output_dtype="float32",
            created_at="2026-04-02T10:04:00+00:00",
        ),
        PredictionRun(
            prediction_run_id="prediction_run_1",
            dataset_id="dataset_1",
            calibration_id="cal_1",
            model_run_id="model_run_1",
            signal_domain_id="signal_domain_1",
            preprocessing_recipe_id="recipe_1",
            artifact_refs={"y_pred": "/predictions/pred_1/y_pred"},
            created_at="2026-04-02T10:07:00+00:00",
        ),
    ]

    for instance in instances:
        _assert_round_trip(instance)


def test_validate_project_relationships_accepts_valid_graph(valid_project_graph):
    validate_project_relationships(**valid_project_graph)


def test_validate_project_relationships_allows_stale_sample_set_rois_when_requested(
    valid_project_graph,
):
    graph = copy.deepcopy(valid_project_graph)
    graph["rois"] = {}
    graph["target_assignments"] = []

    with pytest.raises(ProjectRelationshipError, match=r"missing roi_id\(s\) \['roi_1'\]"):
        validate_project_relationships(**graph)

    validate_project_relationships(**graph, allow_stale_sample_set_rois=True)


def test_validate_project_relationships_checks_active_calibration_ids(valid_project_graph):
    graph = copy.deepcopy(valid_project_graph)

    validate_project_relationships(**graph, active_calibration_ids={"dataset_1": "cal_1"})

    with pytest.raises(ProjectRelationshipError, match="missing calibration_id"):
        validate_project_relationships(
            **graph, active_calibration_ids={"dataset_1": "cal_missing"}
        )

    dataset_2 = copy.deepcopy(graph["datasets"]["dataset_1"])
    dataset_2.dataset_id = "dataset_2"
    graph["datasets"]["dataset_2"] = dataset_2
    with pytest.raises(ProjectRelationshipError, match="of dataset 'dataset_1'"):
        validate_project_relationships(**graph, active_calibration_ids={"dataset_2": "cal_1"})

    with pytest.raises(ProjectRelationshipError, match="missing dataset_id 'dataset_ghost'"):
        validate_project_relationships(
            **graph, active_calibration_ids={"dataset_ghost": "cal_1"}
        )


def test_validate_project_relationships_rejects_sample_set_omitting_roi_dataset(
    valid_project_graph,
):
    graph = copy.deepcopy(valid_project_graph)
    dataset_2 = copy.deepcopy(graph["datasets"]["dataset_1"])
    dataset_2.dataset_id = "dataset_2"
    graph["datasets"]["dataset_2"] = dataset_2
    graph["sample_sets"]["sample_set_1"].dataset_ids = ["dataset_2"]

    with pytest.raises(ProjectRelationshipError, match="does not list dataset_id 'dataset_1'"):
        validate_project_relationships(**graph)


def test_validate_project_relationships_rejects_invalid_edges(valid_project_graph, wavelengths):
    cases = [
        (
            "missing dataset",
            lambda graph: graph["rois"].update(
                {
                    "roi_1": TrainingROI(
                        roi_id="roi_1",
                        dataset_id="missing_dataset",
                        calibration_id="cal_1",
                        name="ROI 1",
                        wavelengths=wavelengths,
                        n_pixels=12,
                        spatial_bounds=(0, 2, 0, 3),
                        source_file="/data/train.eiff",
                        created_at="2026-04-02T10:02:00+00:00",
                        provenance=_roi_provenance(dataset_id="missing_dataset"),
                    )
                }
            ),
            "missing dataset_id",
        ),
        (
            "missing calibration",
            lambda graph: graph["rois"].update(
                {
                    "roi_1": TrainingROI(
                        roi_id="roi_1",
                        dataset_id="dataset_1",
                        calibration_id="missing_calibration",
                        name="ROI 1",
                        wavelengths=wavelengths,
                        n_pixels=12,
                        spatial_bounds=(0, 2, 0, 3),
                        source_file="/data/train.eiff",
                        created_at="2026-04-02T10:02:00+00:00",
                        provenance=_roi_provenance(calibration_id="missing_calibration"),
                    )
                }
            ),
            "missing calibration_id",
        ),
        (
            "missing target",
            lambda graph: graph.update(
                {
                    "target_assignments": [
                        TargetAssignment(
                            roi_id="roi_1",
                            target_id="missing_target",
                            value=12.3,
                            assigned_at="2026-04-02T10:04:00+00:00",
                        )
                    ]
                }
            ),
            "missing target_id",
        ),
        (
            "duplicate assignment",
            lambda graph: graph.update(
                {
                    "target_assignments": [
                        TargetAssignment(
                            roi_id="roi_1",
                            target_id="target_1",
                            value=12.3,
                            assigned_at="2026-04-02T10:04:00+00:00",
                        ),
                        TargetAssignment(
                            roi_id="roi_1",
                            target_id="target_1",
                            value=12.4,
                            assigned_at="2026-04-02T10:05:00+00:00",
                        ),
                    ]
                }
            ),
            "Duplicate target assignment",
        ),
    ]

    for _, mutate, message in cases:
        graph = copy.deepcopy(valid_project_graph)
        mutate(graph)
        with pytest.raises(ProjectRelationshipError, match=message):
            validate_project_relationships(**graph)


def test_calibration_spec_contract_validation():
    invalid_cases = [
        {
            "mode": "dark_white_reference",
            "dark_ref_data": np.zeros((2, 3), dtype=np.float32),
            "white_ref_data": np.ones((2, 3), dtype=np.float32),
            "black_level": 128.0,
            "white_level": 3500.0,
            "output_signal_unit": "Reflectance",
            "message": "exactly one of them",
        },
        {
            "mode": "dark_white_reference",
            "dark_ref_source": "/data/dark.eiff",
            "white_ref_source": "/data/white.eiff",
            "black_level": 128.0,
            "white_level": 3500.0,
            "output_signal_unit": "Reflectance",
            "message": "source paths are provenance only",
        },
        {
            "mode": "raw_dn",
            "output_signal_unit": "Reflectance",
            "message": "mode 'raw_dn' requires output_signal_unit",
        },
        {
            "mode": "dark_white_reference",
            "black_level": np.nan,
            "white_level": 3500.0,
            "output_signal_unit": "Reflectance",
            "message": "black_level must be a finite real number",
        },
    ]

    for case in invalid_cases:
        payload = {
            "calibration_id": "cal_1",
            "dataset_id": "dataset_1",
            "mode": case["mode"],
            "dark_ref_source": case.get("dark_ref_source"),
            "white_ref_source": case.get("white_ref_source"),
            "dark_ref_data": case.get("dark_ref_data"),
            "white_ref_data": case.get("white_ref_data"),
            "black_level": case.get("black_level"),
            "white_level": case.get("white_level"),
            "output_signal_unit": case["output_signal_unit"],
            "created_at": "2026-04-02T10:01:00+00:00",
        }
        with pytest.raises(ValueError, match=case["message"]):
            CalibrationSpec(**payload)

    calibration = CalibrationSpec(
        calibration_id="cal_scalar",
        dataset_id="dataset_1",
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
    assert calibration.black_level == pytest.approx(128.0)
    assert calibration.white_level == pytest.approx(3500.0)


def test_dataset_ref_rejects_inconsistent_band_metadata(wavelengths):
    with pytest.raises(ValueError, match="does not have n_bands=50"):
        DatasetRef(
            dataset_id="dataset_1",
            source_path="/data/train.eiff",
            source_format="eiff",
            source_modality="nir_hsi",
            cube_shape=(20, 15, 49),
            n_bands=50,
            wavelengths=wavelengths,
            loader_metadata={},
            import_timestamp="2026-04-02T10:00:00+00:00",
        )

    with pytest.raises(ValueError, match="wavelengths has 49 values, expected 50"):
        DatasetRef(
            dataset_id="dataset_1",
            source_path="/data/train.eiff",
            source_format="eiff",
            source_modality="nir_hsi",
            cube_shape=(20, 15, 50),
            n_bands=50,
            wavelengths=wavelengths[:-1],
            loader_metadata={},
            import_timestamp="2026-04-02T10:00:00+00:00",
        )


def test_training_roi_rejects_invalid_numeric_contracts(wavelengths):
    with pytest.raises(ValueError, match="n_pixels must be an integer >= 1"):
        TrainingROI(
            roi_id="roi_1",
            dataset_id="dataset_1",
            calibration_id="cal_1",
            name="ROI 1",
            wavelengths=wavelengths,
            n_pixels=0,
            spatial_bounds=(0, 2, 0, 3),
            source_file="/data/train.eiff",
            created_at="2026-04-02T10:02:00+00:00",
            provenance={},
        )

    with pytest.raises(ValueError, match="source_file must not be empty"):
        TrainingROI(
            roi_id="roi_1",
            dataset_id="dataset_1",
            calibration_id="cal_1",
            name="ROI 1",
            wavelengths=wavelengths,
            n_pixels=12,
            spatial_bounds=(0, 2, 0, 3),
            source_file="   ",
            created_at="2026-04-02T10:02:00+00:00",
            provenance={},
        )

    with pytest.raises(ValueError, match="spatial_bounds must be"):
        TrainingROI(
            roi_id="roi_1",
            dataset_id="dataset_1",
            calibration_id="cal_1",
            name="ROI 1",
            wavelengths=wavelengths,
            n_pixels=12,
            spatial_bounds=(3, 2, 0, 3),
            source_file="/data/train.eiff",
            created_at="2026-04-02T10:02:00+00:00",
            provenance={},
        )


def test_training_roi_requires_matching_complete_provenance(wavelengths):
    payload = {
        "roi_id": "roi_1",
        "dataset_id": "dataset_1",
        "calibration_id": "cal_1",
        "name": "ROI 1",
        "wavelengths": wavelengths,
        "n_pixels": 12,
        "spatial_bounds": (0, 2, 0, 3),
        "source_file": "/data/train.eiff",
        "created_at": "2026-04-02T10:02:00+00:00",
    }

    with pytest.raises(ValueError, match="provenance is missing"):
        TrainingROI(**payload, provenance={"signal_unit": "DN"})

    mismatched = _roi_provenance(dataset_id="other_dataset")
    with pytest.raises(ValueError, match="differs from the ROI's"):
        TrainingROI(**payload, provenance=mismatched)

    unsupported_signal = _roi_provenance()
    unsupported_signal["signal_unit"] = "radiance"
    with pytest.raises(ValueError, match="provenance signal_unit"):
        TrainingROI(**payload, provenance=unsupported_signal)

def test_sample_set_rejects_empty_or_repeated_membership():
    base_sample_set = {
        "sample_set_id": "sample_set_1",
        "roi_ids": ["roi_1"],
        "target_id": "target_1",
        "observation_unit": "pixel",
        "target_level": "roi",
        "grouping_unit": "roi",
        "validation_unit": "roi",
        "weighting_policy": "per_pixel_unweighted",
        "reporting_level": "roi",
        "roi_aggregation": "mean",
        "preprocessing_recipe_id": "recipe_1",
        "dataset_ids": ["dataset_1"],
        "created_at": "2026-04-02T10:05:00+00:00",
    }

    with pytest.raises(ValueError, match="has no dataset_ids"):
        SampleSet(**{**base_sample_set, "dataset_ids": []})
    with pytest.raises(ValueError, match="has no roi_ids"):
        SampleSet(**{**base_sample_set, "roi_ids": []})
    with pytest.raises(ValueError, match="lists an ROI more than once"):
        SampleSet(**{**base_sample_set, "roi_ids": ["roi_1", "roi_1"]})


def test_target_and_signal_domain_contract_validation(wavelengths):
    for field_name, field_value, message in [
        ("value_type", "int", "value_type must be one of"),
        ("missing_value_policy", "required", "missing_value_policy must be one of"),
    ]:
        payload = {
            "target_id": "target_1",
            "name": "moisture",
            "unit": "% w/w",
            "value_type": "float",
            "missing_value_policy": "allowed",
            "created_at": "2026-04-02T10:03:00+00:00",
        }
        payload[field_name] = field_value
        with pytest.raises(ValueError, match=message):
            TargetDefinition(**payload)

    base_signal = {
        "signal_domain_id": "signal_domain_1",
        "input_format": "eiff_raw",
        "source_modality": "nir_hsi",
        "required_signal_unit": "Absorbance",
        "required_calibration_mode": "dark_white_reference",
        "requires_dark_ref": True,
        "requires_white_ref": True,
        "calibrated_input_allowed": False,
        "n_bands": 50,
        "wavelengths": wavelengths,
        "wavelength_tolerance_nm": 0.0,
        "band_mapping_policy": "exact",
        "output_dtype": "float32",
        "created_at": "2026-04-02T10:04:00+00:00",
    }
    for field_name, field_value, message in [
        ("required_signal_unit", "Voltage", "required_signal_unit must be one of"),
        ("required_calibration_mode", "factory_default", "required_calibration_mode must be one of"),
        ("band_mapping_policy", "interpolate", "band_mapping_policy must be one of"),
    ]:
        payload = dict(base_signal)
        payload[field_name] = field_value
        with pytest.raises(ValueError, match=message):
            SignalDomainContract(**payload)

    with pytest.raises(ValueError, match="wavelengths has 49 values, expected 50"):
        SignalDomainContract(
            **{**base_signal, "wavelengths": np.linspace(900, 1700, 49, dtype=np.float32)}
        )
    with pytest.raises(ValueError, match="wavelength_tolerance_nm must be >= 0"):
        SignalDomainContract(**{**base_signal, "wavelength_tolerance_nm": -1.0})
    with pytest.raises(ValueError, match="target value must be a finite real number"):
        TargetAssignment(
            roi_id="roi_1",
            target_id="target_1",
            value=np.nan,
            assigned_at="2026-04-02T10:04:00+00:00",
        )


def test_sample_set_and_model_run_contract_validation():
    base_sample_set = {
        "sample_set_id": "sample_set_1",
        "roi_ids": ["roi_1"],
        "target_id": "target_1",
        "observation_unit": "pixel",
        "target_level": "roi",
        "grouping_unit": "roi",
        "validation_unit": "roi",
        "weighting_policy": "per_pixel_unweighted",
        "reporting_level": "roi",
        "roi_aggregation": "mean",
        "preprocessing_recipe_id": "recipe_1",
        "dataset_ids": ["dataset_1"],
        "created_at": "2026-04-02T10:05:00+00:00",
    }

    for field_name, field_value, message in [
        ("observation_unit", "roi", "observation_unit must be one of"),
        ("weighting_policy", "per_roi_equal", "weighting_policy must be one of"),
        ("roi_aggregation", "median", "roi_aggregation must be one of"),
    ]:
        payload = dict(base_sample_set)
        payload[field_name] = field_value
        with pytest.raises(ValueError, match=message):
            SampleSet(**payload)

    with pytest.raises(ValueError, match="model_type must be one of"):
        ModelRun(
            model_run_id="model_run_1",
            sample_set_id="sample_set_1",
            signal_domain_id="signal_domain_1",
            model_type="SIMCA",
            hyperparameters={},
            cv_spec={},
            metrics={},
            artifact_refs={},
            created_at="2026-04-02T10:06:00+00:00",
        )


# ---------------------------------------------------------------------------
# Strict hydration of persisted values
# ---------------------------------------------------------------------------


def _minimal_store(wavelengths):
    from enginir.io.project_store import PROJECT_SCHEMA_VERSION

    dataset = {
        "source_path": "/data/a.eiff",
        "source_format": "eiff",
        "source_modality": "nir_hsi",
        "cube_shape": np.array([20, 15, 50]),
        "n_bands": np.int64(50),
        "wavelengths": wavelengths,
        "loader_metadata": {},
        "import_timestamp": "2026-04-02T10:00:00+00:00",
        "active_calibration_id": "cal_1",
        "calibrations": {
            "cal_1": {
                "mode": "dark_white_reference",
                "black_level": 100.0,
                "white_level": 3000.0,
                "output_signal_unit": "Reflectance",
                "created_at": "2026-04-02T10:01:00+00:00",
            }
        },
    }
    return {
        "project_meta": {"project_id": "p", "schema_version": PROJECT_SCHEMA_VERSION,
                         "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"},
        "datasets": {"dataset_1": dataset},
        "targets": {}, "target_assignments": {}, "recipes": {}, "signal_domains": {},
        "rois": {}, "sample_sets": {}, "model_runs": {}, "predictions": {},
    }


def test_hydration_accepts_values_as_stored_in_hdf5(wavelengths):
    from enginir.project import hydrate_project_store

    workspace = hydrate_project_store(_minimal_store(wavelengths))

    assert workspace.datasets["dataset_1"].cube_shape == (20, 15, 50)
    assert workspace.active_calibration_ids == {"dataset_1": "cal_1"}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cube_shape", [20.7, 15, 50], "cube_shape"),
        ("n_bands", True, "n_bands must be an integer"),
        ("source_format", None, "source_format must be a string"),
        ("wavelengths", np.array(["a"] * 50), "wavelengths must be numeric"),
        ("loader_metadata", "not a mapping", "loader_metadata must be a mapping"),
    ],
)
def test_hydration_rejects_mistyped_values_and_names_the_location(wavelengths, field, value,
                                                                  message):
    from enginir.project import hydrate_project_store

    store = _minimal_store(wavelengths)
    store["datasets"]["dataset_1"][field] = value

    with pytest.raises(ValueError, match=f"/datasets/dataset_1: .*{message}"):
        hydrate_project_store(store)


def test_hydration_requires_an_active_calibration(wavelengths):
    from enginir.project import hydrate_project_store

    store = _minimal_store(wavelengths)
    del store["datasets"]["dataset_1"]["active_calibration_id"]

    with pytest.raises(ValueError, match="no active_calibration_id"):
        hydrate_project_store(store)


def test_signal_domain_rejects_contradictory_or_unordered_contracts(wavelengths):
    base = {
        "signal_domain_id": "sd", "input_format": "eiff", "source_modality": "nir_hsi",
        "required_signal_unit": "DN", "required_calibration_mode": "raw_dn",
        "requires_dark_ref": False, "requires_white_ref": False,
        "calibrated_input_allowed": False, "n_bands": 50, "wavelengths": wavelengths,
        "wavelength_tolerance_nm": 0.0, "band_mapping_policy": "exact",
        "output_dtype": "float32", "created_at": "2026-04-02T10:04:00+00:00",
    }
    SignalDomainContract(**base)
    with pytest.raises(ValueError, match="reference requirements need"):
        SignalDomainContract(**{**base, "requires_white_ref": True})
    with pytest.raises(ValueError, match="must be a bool"):
        SignalDomainContract(**{**base, "requires_dark_ref": "false"})
    shuffled = wavelengths.copy()
    shuffled[[0, 1]] = shuffled[[1, 0]]
    with pytest.raises(ValueError, match="strictly increasing or decreasing"):
        SignalDomainContract(**{**base, "wavelengths": shuffled})


def test_sample_set_rejects_a_bare_string_as_roi_list():
    with pytest.raises(ValueError, match="roi_ids must be a list of strings"):
        SampleSet(
            sample_set_id="s", roi_ids="roi_1", target_id="t", observation_unit="pixel",
            target_level="roi", grouping_unit="roi", validation_unit="roi",
            weighting_policy="per_pixel_unweighted", reporting_level="roi",
            roi_aggregation="mean", preprocessing_recipe_id="r", dataset_ids=["d"],
            created_at="2026-04-02T10:05:00+00:00",
        )


def test_hydration_rejects_an_active_calibration_without_calibrations(wavelengths):
    from enginir.project import hydrate_project_store

    store = _minimal_store(wavelengths)
    store["datasets"]["dataset_1"]["calibrations"] = {}
    store["datasets"]["dataset_1"]["active_calibration_id"] = "ghost"

    with pytest.raises(ValueError, match="ghost"):
        hydrate_project_store(store)


def test_hydration_accepts_matching_and_rejects_contradicting_stored_ids(wavelengths):
    from loguru import logger

    from enginir.project import hydrate_project_store

    store = _minimal_store(wavelengths)
    calibration = store["datasets"]["dataset_1"]["calibrations"]["cal_1"]
    calibration.update(calibration_id="cal_1", dataset_id="dataset_1")
    store["datasets"]["dataset_1"]["dataset_id"] = "dataset_1"
    warnings: list[str] = []
    sink_id = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        hydrate_project_store(store)
    finally:
        logger.remove(sink_id)
    assert not [text for text in warnings if "unknown field" in text]

    calibration["dataset_id"] = "dataset_2"
    with pytest.raises(ValueError, match="stores dataset_id='dataset_2'"):
        hydrate_project_store(store)


def test_calibration_content_identity_ignores_ids_and_timestamps():
    """Identical executable calibrations of one dataset are one calibration (contract)."""
    from dataclasses import replace

    from enginir.project import CalibrationSpec

    levels = CalibrationSpec(calibration_id="cal_1", dataset_id="dataset_1",
                             mode="dark_white_reference", dark_ref_source=None,
                             white_ref_source=None, dark_ref_data=None, white_ref_data=None,
                             black_level=100.0, white_level=3000.0,
                             output_signal_unit="Reflectance", created_at="2026-01-01T00:00:00")
    assert levels.same_content_as(replace(levels, calibration_id="cal_2", created_at="2026-01-02T00:00:00"))
    assert not levels.same_content_as(replace(levels, white_level=3001.0))
    assert not levels.same_content_as(replace(levels, dataset_id="dataset_2"))
    assert not levels.same_content_as(replace(levels, output_signal_unit="Absorbance"))

    dark, white = np.zeros((4, 3)), np.full((4, 3), 10.0)
    arrays = replace(levels, black_level=None, white_level=None, dark_ref_data=dark,
                     white_ref_data=white)
    assert arrays.same_content_as(replace(arrays, calibration_id="cal_3",
                                          dark_ref_data=dark.copy()))
    assert not arrays.same_content_as(replace(arrays, white_ref_data=white + 1.0))
    assert not arrays.same_content_as(levels)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("project_id", "", "project_id"),
        ("schema_version", "viewer-v2", "schema_version must be"),
        ("created_at", "yesterday", "ISO 8601"),
        ("updated_at", "2026-13-01", "ISO 8601"),
    ],
)
def test_hydration_rejects_malformed_project_meta(wavelengths, field, value, message):
    from enginir.project import hydrate_project_store

    store = _minimal_store(wavelengths)
    store["project_meta"][field] = value

    with pytest.raises(ValueError, match=message):
        hydrate_project_store(store)


@pytest.mark.parametrize("value", ["t", "2026-04-02 25:00", "", "02.04.2026"])
def test_object_timestamps_must_be_iso_8601(value):
    with pytest.raises(ValueError, match="created_at"):
        TargetDefinition(target_id="t1", name="m", unit="%", value_type="float",
                         missing_value_policy="allowed", created_at=value)


def test_raw_load_is_structural_and_hydration_validates(wavelengths, tmp_path):
    """G01: the raw reader accepts content that only hydration rejects."""
    import h5py

    from enginir.io.project_store import create_project, load_project_store
    from enginir.project import hydrate_project_store

    path = tmp_path / "p.h5"
    create_project(path)
    with h5py.File(path, "r+") as handle:
        handle["project_meta"].attrs["created_at"] = "not a timestamp"

    store = load_project_store(path)
    assert store["project_meta"]["created_at"] == "not a timestamp"
    with pytest.raises(ValueError, match="ISO 8601"):
        hydrate_project_store(store)
