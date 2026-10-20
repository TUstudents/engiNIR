"""Contract tests for the v3 HDF5 project API and model artifact serializers."""

from __future__ import annotations

import h5py
from loguru import logger
import numpy as np
import pytest


from enginir.io.project_store import (
    create_project,
    discard_prediction_run_artifacts,
    finalize_prediction_run_artifacts,
    initialize_prediction_run_artifacts,
    iter_roi_calibration_refs,
    open_prediction_run_artifact_writer,
    load_model_run_prediction_artifact,
    load_prediction_run_artifact_refs,
    load_prediction_spatial_map_preview,
    load_project_store,
    load_roi_spectra_batch,
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

from ._project_payloads import recipe_payload
from enginir.project import hydrate_project_store


@pytest.fixture()
def rng():
    return np.random.RandomState(42)


@pytest.fixture()
def project_path(tmp_path):
    return str(tmp_path / "project.h5")


@pytest.fixture()
def wavelengths():
    return np.linspace(900, 1700, 50, dtype=np.float32)


@pytest.fixture()
def regression_data(rng, wavelengths):
    X = rng.randn(40, wavelengths.shape[0])
    y = rng.randn(40)
    return X, wavelengths, y


@pytest.fixture()
def classification_data(rng, wavelengths):
    n_per_class = 20
    X = np.vstack([rng.randn(n_per_class, wavelengths.shape[0]) + 2 * idx for idx in range(3)])
    y = np.array(["A"] * n_per_class + ["B"] * n_per_class + ["C"] * n_per_class)
    return X, wavelengths, y


def _seed_dataset_context(
    path: str, wavelengths: np.ndarray, *, source_file: str = "/data/cube.eiff"
) -> None:
    save_dataset_ref(
        path,
        "dataset_1",
        {
            "source_path": source_file,
            "source_format": "eiff",
            "source_modality": "nir_hsi",
            "cube_shape": (20, 15, len(wavelengths)),
            "n_bands": len(wavelengths),
            "wavelengths": wavelengths,
            "loader_metadata": {},
            "import_timestamp": "2026-03-28T10:00:00+00:00",
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
            "created_at": "2026-03-28T10:00:00+00:00",
        },
    )


def _roi_payload(
    wavelengths: np.ndarray,
    *,
    name: str,
    source_file: str = "/data/cube.eiff",
    n_pixels: int,
) -> dict:
    return {
        "dataset_id": "dataset_1",
        "calibration_id": "cal_1",
        "name": name,
        "wavelengths": wavelengths,
        "source_file": source_file,
        "spatial_bounds": (0, 2, 0, 3),
        "n_pixels": n_pixels,
        "created_at": "2026-03-28T10:00:00+00:00",
        "provenance": {
            "dataset_id": "dataset_1",
            "calibration_id": "cal_1",
            "signal_unit": "DN",
            "preview_recipe_summary": "none",
            "created_at": "2026-03-28T10:00:00+00:00",
        },
    }


def test_v3_project_store_round_trip_and_skip_flags(project_path, rng, wavelengths):
    create_project(project_path)
    _seed_dataset_context(project_path, wavelengths, source_file="/data/train_a.eiff")

    save_target_definition(
        project_path,
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
        project_path,
        "recipe_1",
        recipe_payload(filter="sg_d1", snv=True),
    )
    save_signal_domain_contract(
        project_path,
        "signal_domain_1",
        {
            "input_format": "eiff_raw",
            "source_modality": "nir_hsi",
            "required_signal_unit": "Absorbance",
            "required_calibration_mode": "dark_white_reference",
            "requires_dark_ref": True,
            "requires_white_ref": True,
            "calibrated_input_allowed": False,
            "n_bands": len(wavelengths),
            "wavelengths": wavelengths,
            "wavelength_tolerance_nm": 0.0,
            "band_mapping_policy": "exact",
            "output_dtype": "float32",
            "created_at": "2026-04-02T10:04:00+00:00",
        },
    )

    spectra = rng.randint(0, 4095, (12, len(wavelengths))).astype(np.uint16)
    dark_refs = np.zeros_like(spectra, dtype=np.float32)
    white_refs = np.full_like(spectra, 4095.0, dtype=np.float32)
    save_training_roi(
        project_path,
        "roi_1",
        _roi_payload(wavelengths, name="ROI 1", n_pixels=12),
        spectra,
        dark_ref_spectra=dark_refs,
        white_ref_spectra=white_refs,
    )
    save_target_assignment(
        project_path,
        "target_moisture",
        "roi_1",
        {"value": 12.3, "assigned_at": "2026-04-02T10:06:00+00:00"},
    )
    save_sample_set(
        project_path,
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
        project_path,
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
                    "wavelengths": wavelengths,
                    "target_names": ["moisture"],
                },
                "cv_outputs": {
                    "roi_ids": np.array(["roi_1", "roi_2"], dtype=object),
                    "y_true_roi": np.array([1.0, 2.0], dtype=np.float32),
                    "y_pred_roi": np.array([1.1, 1.9], dtype=np.float32),
                },
            },
            "created_at": "2026-04-02T10:08:00+00:00",
        },
    )
    save_prediction_run(
        project_path,
        "prediction_run_1",
        {
            "dataset_id": "dataset_1",
            "calibration_id": "cal_1",
            "model_run_id": "model_run_1",
            "signal_domain_id": "signal_domain_1",
            "preprocessing_recipe_id": "recipe_1",
            "artifact_refs": {
                "source_path": "/data/predict.eiff",
                "y_pred": np.array([11.9, 12.1], dtype=np.float32),
                "spatial_map": np.array([[11.9, 12.1]], dtype=np.float32),
            },
            "created_at": "2026-04-02T10:09:00+00:00",
        },
    )

    full_store = load_project_store(
        project_path,
        include_roi_spectra=True,
        include_artifact_arrays=True,
    )
    trimmed_store = load_project_store(project_path)

    assert full_store["project_meta"]["schema_version"] == "viewer-v3"
    assert full_store["datasets"]["dataset_1"]["active_calibration_id"] == "cal_1"
    assert full_store["targets"]["target_moisture"]["name"] == "moisture"
    assert full_store["recipes"]["recipe_1"]["filter"] == "sg_d1"
    np.testing.assert_array_equal(full_store["rois"]["roi_1"]["spectra"], spectra)
    np.testing.assert_allclose(full_store["rois"]["roi_1"]["calibration_refs"]["dark"], dark_refs)
    np.testing.assert_allclose(full_store["rois"]["roi_1"]["calibration_refs"]["white"], white_refs)
    assert "spectra" not in trimmed_store["rois"]["roi_1"]
    assert trimmed_store["rois"]["roi_1"].get("calibration_refs") == {}
    loaded_refs = dict(iter_roi_calibration_refs(project_path, ["roi_1"]))
    assert loaded_refs["roi_1"] is not None
    loaded_dark, loaded_white = loaded_refs["roi_1"]
    np.testing.assert_allclose(loaded_dark, dark_refs)
    np.testing.assert_allclose(loaded_white, white_refs)
    assert full_store["sample_sets"]["sample_set_1"]["observation_unit"] == "pixel"
    assert full_store["model_runs"]["model_run_1"]["metrics"]["rmsecv"] == pytest.approx(1.23)
    np.testing.assert_allclose(
        full_store["predictions"]["prediction_run_1"]["artifact_refs"]["y_pred"],
        np.array([11.9, 12.1], dtype=np.float32),
    )

    assert "spectra" not in trimmed_store["rois"]["roi_1"]
    assert trimmed_store["model_runs"]["model_run_1"]["artifact_refs"]["cv_outputs"] == {}
    assert (
        trimmed_store["predictions"]["prediction_run_1"]["artifact_refs"]["source_path"]
        == "/data/predict.eiff"
    )
    assert "y_pred" not in trimmed_store["predictions"]["prediction_run_1"]["artifact_refs"]

    workspace = hydrate_project_store(trimmed_store)
    assert workspace.datasets["dataset_1"].n_bands == len(wavelengths)
    assert workspace.rois["roi_1"].spectra_storage_key == "/rois/roi_1/spectra"
    assert workspace.model_runs["model_run_1"].model_type == "PLS"
    assert workspace.predictions["prediction_run_1"].dataset_id == "dataset_1"

    trimmed_store["datasets"]["dataset_1"]["legacy_label"] = "ignored"
    trimmed_store["datasets"]["dataset_1"]["loader_metadata"]["sensor_extension"] = "kept"
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        workspace_with_extra = hydrate_project_store(trimmed_store)
    finally:
        logger.remove(sink_id)
    assert not hasattr(workspace_with_extra.datasets["dataset_1"], "legacy_label")
    assert (
        workspace_with_extra.datasets["dataset_1"].loader_metadata["sensor_extension"]
        == "kept"
    )
    assert any(
        "/datasets/dataset_1" in message and "legacy_label" in message
        for message in messages
    )

    invalid_dataset = dict(trimmed_store["datasets"]["dataset_1"])
    del invalid_dataset["source_modality"]
    invalid_store = dict(trimmed_store)
    invalid_store["datasets"] = {"dataset_1": invalid_dataset}
    with pytest.raises(
        ValueError,
        match=r"/datasets/dataset_1.*source_modality",
    ):
        hydrate_project_store(invalid_store)


def test_incremental_prediction_artifacts_are_disk_backed_and_key_loadable(project_path):
    create_project(project_path)
    payload = {
        "dataset_id": "dataset_1",
        "calibration_id": "cal_1",
        "model_run_id": "model_run_1",
        "signal_domain_id": "signal_domain_1",
        "preprocessing_recipe_id": "recipe_1",
        "artifact_refs": {
            "source_path": "/data/predict.eiff",
            "chunk_size": 1,
            "n_chunks": 2,
            "prediction_shape": (4,),
            "spatial_shape": (2, 2),
            "spatial_map_shape": (2, 2),
            "output_dtype": "float32",
        },
        "created_at": "2026-04-02T10:09:00+00:00",
    }

    initialize_prediction_run_artifacts(
        project_path,
        "prediction_stream",
        payload,
        y_pred_shape=(4,),
        spatial_map_shape=(2, 2),
    )
    with open_prediction_run_artifact_writer(project_path, "prediction_stream") as writer:
        writer.write_chunk(
            row_start=0,
            y_pred_chunk=np.array([1.0, 2.0], dtype=np.float32),
            frame_start=0,
            spatial_map_chunk=np.array([[1.0, 2.0]], dtype=np.float32),
        )

    with pytest.raises(ValueError, match="not written completely"):
        load_project_store(project_path)

    with open_prediction_run_artifact_writer(project_path, "prediction_stream") as writer:
        writer.write_chunk(
            row_start=2,
            y_pred_chunk=np.array([3.0, 4.0], dtype=np.float32),
            frame_start=1,
            spatial_map_chunk=np.array([[3.0, 4.0]], dtype=np.float32),
        )
    finalize_prediction_run_artifacts(
        project_path,
        "prediction_stream",
        {"completed_chunks": 2, "clip_low_count": 0, "clip_high_count": 1},
    )

    trimmed_store = load_project_store(project_path)
    artifact_refs = trimmed_store["predictions"]["prediction_stream"]["artifact_refs"]
    assert artifact_refs["artifact_write_status"] == "complete"
    assert artifact_refs["completed_chunks"] == 2
    assert "y_pred" not in artifact_refs
    assert "spatial_map" not in artifact_refs

    selected_refs = load_prediction_run_artifact_refs(
        project_path,
        "prediction_stream",
        keys=("spatial_map",),
    )
    assert set(selected_refs) == {"spatial_map"}
    np.testing.assert_allclose(
        selected_refs["spatial_map"],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        load_prediction_spatial_map_preview(project_path, "prediction_stream", max_pixels=1),
        np.array([[1.0]], dtype=np.float32),
    )

    all_refs = load_prediction_run_artifact_refs(project_path, "prediction_stream")
    np.testing.assert_allclose(all_refs["y_pred"], np.array([1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(ValueError, match="refusing to discard"):
        discard_prediction_run_artifacts(project_path, "prediction_stream")
    assert "prediction_stream" in load_project_store(project_path)["predictions"]
    with pytest.raises(ValueError, match="already exists"):
        initialize_prediction_run_artifacts(
            project_path, "prediction_stream", {}, y_pred_shape=(4,), spatial_map_shape=None
        )


def test_hydration_rejects_calibration_id_collision_across_datasets(project_path, wavelengths):
    create_project(project_path)
    _seed_dataset_context(project_path, wavelengths)
    save_dataset_ref(
        project_path,
        "dataset_2",
        {
            "source_path": "/data/other.eiff",
            "source_format": "eiff",
            "source_modality": "nir_hsi",
            "cube_shape": (10, 8, len(wavelengths)),
            "n_bands": len(wavelengths),
            "wavelengths": wavelengths,
            "loader_metadata": {},
            "import_timestamp": "2026-03-28T11:00:00+00:00",
        },
    )
    save_calibration_spec(
        project_path,
        "dataset_2",
        "cal_1",
        {
            "mode": "raw_dn",
            "output_signal_unit": "DN",
            "created_at": "2026-03-28T11:01:00+00:00",
        },
    )

    with pytest.raises(ValueError, match="is already used by dataset"):
        hydrate_project_store(load_project_store(project_path))


def test_hydration_rejects_dangling_or_foreign_active_calibration_id(project_path, wavelengths):
    create_project(project_path)
    _seed_dataset_context(project_path, wavelengths)

    with h5py.File(project_path, "r+") as handle:
        handle["datasets/dataset_1"].attrs["active_calibration_id"] = "cal_missing"
    with pytest.raises(ValueError, match="references missing calibration_id"):
        hydrate_project_store(load_project_store(project_path))

    save_dataset_ref(
        project_path,
        "dataset_2",
        {
            "source_path": "/data/other.eiff",
            "source_format": "eiff",
            "source_modality": "nir_hsi",
            "cube_shape": (10, 8, len(wavelengths)),
            "n_bands": len(wavelengths),
            "wavelengths": wavelengths,
            "loader_metadata": {},
            "import_timestamp": "2026-03-28T11:00:00+00:00",
        },
    )
    save_calibration_spec(
        project_path,
        "dataset_2",
        "cal_2",
        {
            "mode": "raw_dn",
            "output_signal_unit": "DN",
            "created_at": "2026-03-28T11:01:00+00:00",
        },
    )
    with h5py.File(project_path, "r+") as handle:
        handle["datasets/dataset_1"].attrs["active_calibration_id"] = "cal_2"
    with pytest.raises(ValueError, match="of dataset 'dataset_2'"):
        hydrate_project_store(load_project_store(project_path))


def test_prediction_run_artifact_write_status_lifecycle_enum(project_path):
    create_project(project_path)
    initialize_prediction_run_artifacts(
        project_path,
        "prediction_status",
        {
            "dataset_id": "dataset_1",
            "calibration_id": "cal_1",
            "model_run_id": "model_run_1",
            "signal_domain_id": "signal_domain_1",
            "preprocessing_recipe_id": "recipe_1",
            "artifact_refs": {"output_dtype": "float32", "prediction_shape": (2,)},
            "created_at": "2026-07-17T10:00:00+00:00",
        },
        y_pred_shape=(2,),
        spatial_map_shape=None,
    )
    finalize_prediction_run_artifacts(project_path, "prediction_status", {"completed_chunks": 1})

    with h5py.File(project_path, "r+") as handle:
        handle["predictions/prediction_status/artifact_refs"].attrs["artifact_write_status"] = (
            "finalizing"
        )
    with pytest.raises(ValueError, match="unknown artifact_write_status 'finalizing'"):
        load_project_store(project_path)

    with h5py.File(project_path, "r+") as handle:
        del handle["predictions/prediction_status/artifact_refs"].attrs["artifact_write_status"]
    store = load_project_store(project_path)
    assert "prediction_status" in store["predictions"]


def test_roi_storage_batch_loading(project_path, rng, wavelengths):
    create_project(project_path)
    _seed_dataset_context(project_path, wavelengths)
    spectra_a = rng.randint(0, 100, (3, len(wavelengths))).astype(np.uint16)
    spectra_b = rng.randint(0, 100, (2, len(wavelengths))).astype(np.uint16)
    save_training_roi(
        project_path,
        "A",
        _roi_payload(wavelengths, name="A", source_file="/a.eiff", n_pixels=3),
        spectra_a,
    )
    save_training_roi(
        project_path,
        "B",
        _roi_payload(wavelengths, name="B", source_file="/b.eiff", n_pixels=2),
        spectra_b,
    )

    loaded = load_roi_spectra_batch(project_path, ["/rois/A/spectra", "/rois/B/spectra"])

    np.testing.assert_array_equal(loaded["/rois/A/spectra"], spectra_a)
    np.testing.assert_array_equal(loaded["/rois/B/spectra"], spectra_b)
    with h5py.File(project_path, "r") as handle:
        assert handle["rois/A/spectra"].dtype == np.uint16


def test_load_project_store_rejects_wrong_schema_version(tmp_path):
    path = str(tmp_path / "bad_schema.h5")
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("project_meta")
        meta.attrs["schema_version"] = "viewer-v2"

    with pytest.raises(ValueError, match="schema_version"):
        load_project_store(path)


def test_save_recipe_enforces_current_schema_and_discards_extras(tmp_path):
    path = str(tmp_path / "recipe_schema.h5")
    create_project(path)
    payload = recipe_payload(snv=True, created_at="2026-01-01T00:00:00")
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        save_preprocessing_recipe(path, "recipe_1", payload)
    finally:
        logger.remove(sink_id)

    saved = load_project_store(path)["recipes"]["recipe_1"]
    assert saved == recipe_payload(snv=True)
    assert any("/recipes/recipe_1" in message and "created_at" in message for message in messages)

    incomplete = recipe_payload()
    del incomplete["msc"]
    with pytest.raises(ValueError, match=r"/recipes/incomplete.*msc"):
        save_preprocessing_recipe(path, "incomplete", incomplete)
    with h5py.File(path, "r") as handle:
        assert "incomplete" not in handle["recipes"]

    save_preprocessing_recipe(path, "recipe_1", recipe_payload(snv=True))
    with pytest.raises(ValueError, match="Refusing to overwrite PreprocessingRecipe"):
        save_preprocessing_recipe(path, "recipe_1", recipe_payload(msc=True))
    assert load_project_store(path)["recipes"]["recipe_1"] == recipe_payload(snv=True)


def test_failed_object_serialization_preserves_previous_group(tmp_path, monkeypatch):
    path = str(tmp_path / "atomic_object_write.h5")
    create_project(path)
    original = {
        "name": "moisture",
        "unit": "%",
        "value_type": "float",
        "missing_value_policy": "allowed",
        "created_at": "2026-04-02T10:00:00+00:00",
    }
    save_target_definition(path, "target_1", original)

    import enginir.io.project_store as store

    def write_then_fail(group, payload, compression):
        group.attrs["name"] = "partial"
        raise TypeError("cannot serialize")

    with monkeypatch.context() as patch:
        patch.setattr(store, "write_mapping", write_then_fail)
        with pytest.raises(TypeError):
            save_target_definition(path, "target_2", original)
    with h5py.File(path, "r") as handle:
        assert "target_2" not in handle["targets"]
        assert not any(name.startswith("__enginir_write_") for name in handle)

    with pytest.raises(ValueError, match="Refusing to overwrite TargetDefinition"):
        save_target_definition(path, "target_1", {**original, "unit": "g"})

    assert load_project_store(path)["targets"]["target_1"] == original
    with h5py.File(path, "r") as handle:
        assert not any(name.startswith("__enginir_write_") for name in handle)


def test_stable_object_ids_are_immutable_except_labels(project_path, wavelengths):
    create_project(project_path)
    _seed_dataset_context(project_path, wavelengths)

    target = {
        "name": "moisture",
        "unit": "%",
        "value_type": "float",
        "missing_value_policy": "allowed",
        "created_at": "2026-04-02T10:00:00+00:00",
    }
    save_target_definition(project_path, "target_imm", target)
    save_target_definition(project_path, "target_imm", dict(target))
    save_target_definition(project_path, "target_imm", {**target, "name": "renamed"})
    with pytest.raises(ValueError, match="Refusing to overwrite TargetDefinition"):
        save_target_definition(project_path, "target_imm", {**target, "unit": "mg"})

    spectra = np.arange(2 * len(wavelengths), dtype=np.uint16).reshape(2, len(wavelengths))
    save_training_roi(
        project_path,
        "roi_imm",
        _roi_payload(wavelengths, name="A", n_pixels=2),
        spectra,
    )
    save_training_roi(
        project_path,
        "roi_imm",
        _roi_payload(wavelengths, name="Renamed", n_pixels=2),
        spectra,
    )
    with pytest.raises(ValueError, match="Refusing to overwrite TrainingROI"):
        save_training_roi(
            project_path,
            "roi_imm",
            _roi_payload(wavelengths, name="Renamed", n_pixels=2),
            spectra + 1,
        )


def test_save_training_roi_rejects_metadata_spectra_mismatch(project_path, wavelengths):
    create_project(project_path)
    _seed_dataset_context(project_path, wavelengths)
    spectra = np.zeros((3, len(wavelengths)), dtype=np.uint16)

    with pytest.raises(ValueError, match="n_pixels=2 does not match 3 spectra rows"):
        save_training_roi(
            project_path,
            "roi_bad_pixels",
            _roi_payload(wavelengths, name="X", n_pixels=2),
            spectra,
        )
    with pytest.raises(ValueError, match="wavelengths shape"):
        save_training_roi(
            project_path,
            "roi_bad_bands",
            _roi_payload(wavelengths[:-1], name="X", n_pixels=3),
            spectra,
        )
    with h5py.File(project_path, "r") as handle:
        assert "roi_bad_pixels" not in handle["rois"]
        assert "roi_bad_bands" not in handle["rois"]


def test_roi_spectra_preserve_float32_for_precalibrated_sources(project_path, wavelengths):
    create_project(project_path)
    _seed_dataset_context(project_path, wavelengths)
    n_bands = len(wavelengths)
    reflectance = np.linspace(0.01, 0.99, 2 * n_bands, dtype=np.float32).reshape(2, n_bands)
    raw_dn = np.arange(2 * n_bands, dtype=np.uint16).reshape(2, n_bands) + 10
    save_training_roi(
        project_path,
        "roi_float",
        _roi_payload(wavelengths, name="Float", n_pixels=2),
        reflectance,
    )
    save_training_roi(
        project_path,
        "roi_raw",
        _roi_payload(wavelengths, name="Raw", n_pixels=2),
        raw_dn,
    )

    with h5py.File(project_path, "r") as handle:
        assert handle["rois/roi_float/spectra"].dtype == np.float32
        assert handle["rois/roi_raw/spectra"].dtype == np.uint16

    loaded = load_roi_spectra_batch(project_path, ["/rois/roi_float/spectra"])
    np.testing.assert_array_equal(loaded["/rois/roi_float/spectra"], reflectance)

    from enginir.io.project_store import load_roi_spectra_matrix

    matrix, slices = load_roi_spectra_matrix(
        project_path,
        ["/rois/roi_raw/spectra", "/rois/roi_float/spectra"],
        n_bands=len(wavelengths),
    )
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[slices["/rois/roi_float/spectra"]], reflectance)
    np.testing.assert_array_equal(
        matrix[slices["/rois/roi_raw/spectra"]],
        raw_dn.astype(np.float32),
    )

    raw_only, _ = load_roi_spectra_matrix(
        project_path,
        ["/rois/roi_raw/spectra"],
        n_bands=len(wavelengths),
    )
    assert raw_only.dtype == np.uint16


def test_load_project_store_rejects_corrupt_roi_arrays(project_path, wavelengths):
    create_project(project_path)
    _seed_dataset_context(project_path, wavelengths)
    save_training_roi(
        project_path,
        "roi_1",
        _roi_payload(wavelengths, name="A", n_pixels=2),
        np.zeros((2, len(wavelengths)), dtype=np.uint16),
    )

    with h5py.File(project_path, "r+") as handle:
        handle["rois/roi_1"].attrs["n_pixels"] = 5
    with pytest.raises(ValueError, match="n_pixels=5 does not match 2 stored rows"):
        load_project_store(project_path)

    with h5py.File(project_path, "r+") as handle:
        handle["rois/roi_1"].attrs["n_pixels"] = 2
        refs = handle["rois/roi_1"].create_group("calibration_refs")
        refs.create_dataset("dark", data=np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="calibration_refs/dark shape"):
        load_project_store(project_path)


def test_current_project_ignores_legacy_extras_at_root_and_roi(tmp_path, wavelengths):
    path = str(tmp_path / "legacy_extras.h5")
    create_project(path)
    _seed_dataset_context(path, wavelengths)
    save_training_roi(
        path,
        "roi_1",
        _roi_payload(wavelengths, name="ROI 1", n_pixels=2),
        np.zeros((2, len(wavelengths)), dtype=np.uint16),
    )
    with h5py.File(path, "a") as handle:
        handle.create_group("calibration")
        handle["rois/roi_1"].attrs["y_value"] = 1.0

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        workspace = hydrate_project_store(load_project_store(path))
    finally:
        logger.remove(sink_id)

    assert "roi_1" in workspace.rois
    assert not hasattr(workspace.rois["roi_1"], "y_value")
    assert any("/" in message and "calibration" in message for message in messages)
    assert any("/rois/roi_1" in message and "y_value" in message for message in messages)


def test_load_project_store_warns_once_and_ignores_extra_top_level_fields(tmp_path):
    path = str(tmp_path / "legacy_root_models.h5")
    create_project(path)
    with h5py.File(path, "a") as handle:
        legacy_model = handle.create_group("models").create_group("legacy_pls")
        legacy_model.attrs["model_type"] = "PLS"
        handle.create_group("future_extension")

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        store = load_project_store(path)
        load_project_store(path)
    finally:
        logger.remove(sink_id)

    assert "models" not in store
    assert store["model_runs"] == {}
    extra_field_messages = [
        message
        for message in messages
        if "unknown top-level group(s)" in message
    ]
    assert len(extra_field_messages) == 1
    assert "future_extension" in extra_field_messages[0]
    assert "models" in extra_field_messages[0]


def test_load_model_run_prediction_artifact_reads_only_lightweight_payload(project_path, wavelengths):
    create_project(project_path)
    _seed_dataset_context(project_path, wavelengths)
    save_model_run(
        project_path,
        "model_run_light",
        {
            "sample_set_id": "sample_set_1",
            "signal_domain_id": "signal_domain_1",
            "model_type": "PLS",
            "hyperparameters": {},
            "cv_spec": {},
            "metrics": {},
            "artifact_refs": {
                "prediction_model": {
                    "model_type": "PLSRegression",
                    "coef_": np.ones_like(wavelengths),
                    "intercept_": 0.0,
                },
                "diagnostics": {
                    "x_scores_": np.ones((100, 3), dtype=np.float32),
                },
            },
            "created_at": "2026-04-02T10:08:00+00:00",
        },
    )

    artifact = load_model_run_prediction_artifact(project_path, "model_run_light")

    assert artifact["model_type"] == "PLSRegression"
    assert "x_scores_" not in artifact
def test_resaving_unchanged_objects_is_a_no_op(tmp_path, wavelengths):
    from enginir.io.project_store import (
        save_calibration_spec,
        save_dataset_ref,
        save_model_run,
        save_target_definition,
    )

    project_path = str(tmp_path / "resave.h5")
    create_project(project_path)
    target = {"name": "T", "unit": "g", "value_type": "float",
              "missing_value_policy": "allowed", "created_at": "2026-01-01T00:00:00"}
    save_target_definition(project_path, "target_1", target)
    save_target_definition(project_path, "target_1", target)
    save_target_definition(project_path, "target_1", {**target, "name": "renamed"})

    dataset = {"source_path": "/data/a.eiff", "source_format": "eiff",
               "source_modality": "nir_hsi", "cube_shape": np.array([20, 15, 50]),
               "n_bands": 50, "wavelengths": wavelengths, "loader_metadata": {},
               "import_timestamp": "2026-01-01T00:00:00"}
    save_dataset_ref(project_path, "dataset_1", dataset)
    save_calibration_spec(project_path, "dataset_1", "cal_1", {
        "mode": "dark_white_reference", "black_level": 100.0, "white_level": 3000.0,
        "output_signal_unit": "Reflectance", "created_at": "2026-01-01T00:00:00"})
    save_dataset_ref(project_path, "dataset_1", dataset)

    # NaN is stored content like any other value: an unchanged re-save stays a no-op
    run = {"sample_set_id": "s", "signal_domain_id": "d", "model_type": "PLS",
           "hyperparameters": {"n_components": 1}, "cv_spec": {"type": "KFold"},
           "metrics": {"rmsecv": np.nan}, "artifact_refs": {},
           "created_at": "2026-01-01T00:00:00"}
    save_model_run(project_path, "run_1", run)
    save_model_run(project_path, "run_1", run)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        save_model_run(project_path, "run_1", {**run, "metrics": {"rmsecv": 0.0}})

    store = load_project_store(project_path)
    assert store["targets"]["target_1"]["name"] == "renamed"
    assert "cal_1" in store["datasets"]["dataset_1"]["calibrations"]
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        save_target_definition(project_path, "target_1", {**target, "unit": "%"})


@pytest.mark.parametrize("n_outputs", [None, 3])
def test_map_preview_subsamples_the_canonical_y_pred_layout(project_path, monkeypatch, n_outputs):
    """Without a stored spatial_map the preview is read from y_pred in frame blocks."""
    from enginir.io import project_store

    n_frames, height = 23, 7
    y_shape = (n_frames * height,) if n_outputs is None else (n_frames * height, n_outputs)
    map_shape = (n_frames, height) if n_outputs is None else (n_frames, height, n_outputs)
    y_pred = np.random.default_rng(0).random(y_shape).astype(np.float32)
    create_project(project_path)
    initialize_prediction_run_artifacts(project_path, "run", {"dataset_id": "d"},
                                        y_pred_shape=y_shape, spatial_map_shape=None)
    with open_prediction_run_artifact_writer(project_path, "run") as writer:
        writer.write_chunk(row_start=0, y_pred_chunk=y_pred)
    finalize_prediction_run_artifacts(project_path, "run", {"spatial_map_shape": map_shape})

    monkeypatch.setattr(project_store, "_PREVIEW_BLOCK_VALUES", 30)  # many small blocks
    preview = load_prediction_spatial_map_preview(project_path, "run", max_pixels=40)
    step = project_store._preview_step(map_shape, 40)
    assert step > 1
    np.testing.assert_array_equal(preview, y_pred.reshape(map_shape)[::step, ::step])


def test_updated_at_changes_only_after_a_successful_write(project_path, monkeypatch):
    import enginir.io.project_store as store

    monkeypatch.setattr(store, "_now_iso", lambda: "2026-01-01T00:00:00+00:00")
    create_project(project_path)
    target = {"name": "moisture", "unit": "%", "value_type": "float",
              "missing_value_policy": "allowed", "created_at": "2026-01-01T00:00:00+00:00"}

    monkeypatch.setattr(store, "_now_iso", lambda: "2026-02-02T00:00:00+00:00")
    with monkeypatch.context() as patch:
        patch.setattr(store, "write_mapping", lambda *args: (_ for _ in ()).throw(TypeError()))
        with pytest.raises(TypeError):
            save_target_definition(project_path, "target_1", target)
    meta = load_project_store(project_path)["project_meta"]
    assert meta["updated_at"] == "2026-01-01T00:00:00+00:00"

    save_target_definition(project_path, "target_1", target)
    meta = load_project_store(project_path)["project_meta"]
    assert meta["created_at"] == "2026-01-01T00:00:00+00:00"
    assert meta["updated_at"] == "2026-02-02T00:00:00+00:00"


@pytest.mark.parametrize(
    ("change", "message"),
    [({"created_at": "yesterday"}, "ISO 8601"), ({"colour": "red"}, "unknown field"),
     ({"unit": None}, "unit must be a string")],
)
def test_writers_refuse_payloads_that_hydration_would_reject(project_path, change, message):
    """G03: direct writes are validated by the typed models before anything is stored."""
    create_project(project_path)
    target = {"name": "moisture", "unit": "%", "value_type": "float",
              "missing_value_policy": "allowed", "created_at": "2026-01-01T00:00:00"}

    with pytest.raises(ValueError, match=message):
        save_target_definition(project_path, "target_1", {**target, **change})

    assert load_project_store(project_path)["targets"] == {}


def test_child_object_writers_validate_too(project_path, wavelengths):
    from enginir.io.project_store import save_calibration_spec, save_target_assignment

    create_project(project_path)
    _seed_dataset_context(project_path, wavelengths, source_file="/data/a.eiff")

    with pytest.raises(ValueError, match="mode"):
        save_calibration_spec(project_path, "dataset_1", "cal_x", {
            "mode": "guess", "output_signal_unit": "Reflectance",
            "created_at": "2026-01-01T00:00:00"})
    with pytest.raises(ValueError, match="assigned_at"):
        save_target_assignment(project_path, "target_1", "roi_1",
                               {"value": 1.0, "assigned_at": "now"})

    store = load_project_store(project_path)
    assert "cal_x" not in store["datasets"]["dataset_1"]["calibrations"]
    assert "roi_1" not in store["target_assignments"].get("target_1", {})
