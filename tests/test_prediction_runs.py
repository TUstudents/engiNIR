"""Tests for the v3 prediction workflow."""

from __future__ import annotations

import dataclasses

import h5py
import numpy as np
import pytest
from unittest.mock import patch

from enginir.io import SpectralData, load_hdf5, save_hdf5
from enginir.io.project_store import (
    create_project,
    load_model_run_prediction_artifact,
    save_calibration_spec,
    save_dataset_ref,
    save_preprocessing_recipe,
    save_signal_domain_contract,
    save_target_assignment,
    save_target_definition,
    save_training_roi,
)
from enginir.prediction import (
    PredictionCancelledError,
    PredictionProgressEvent,
    build_dataset_ref_from_spectral_data,
    predict_with_model_run,
    preflight_prediction,
)
from enginir.preprocessing import PreprocessingRecipe
from enginir.project import (
    CalibrationSpec,
    DatasetRef,
    ModelRun,
    PredictionRun,
    SignalDomainCompatibilityError,
    SignalDomainContract,
    transform_spectra_to_signal_domain,
)
from enginir.viewer._app import ensure_app
from enginir.viewer._state import ProjectState

from ._project_payloads import roi_provenance


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


def _create_phase6_model_project(path: str) -> None:
    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    create_project(path)
    save_dataset_ref(
        path,
        "dataset_train",
        {
            "source_path": "/data/train.h5",
            "source_format": "hdf5",
            "source_modality": "nir_hsi",
            "cube_shape": (4, 4, 3),
            "n_bands": 3,
            "wavelengths": wavelengths,
            "loader_metadata": {"instrument": "alpha"},
            "import_timestamp": "2026-04-02T10:00:00+00:00",
        },
    )
    save_calibration_spec(
        path,
        "dataset_train",
        "cal_train",
        {
            "mode": "dark_white_reference",
            "output_signal_unit": "Absorbance",
            "black_level": 0.0,
            "white_level": 100.0,
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
        "recipe_none",
        {
            "filter": "none",
            "sg_window": 11,
            "sg_polyorder": 2,
            "snv": False,
            "msc": False,
            "emsc": False,
            "detrend": False,
            "detrend_order": 1,
        },
    )
    save_preprocessing_recipe(
        path,
        "recipe_msc",
        {
            "filter": "none",
            "sg_window": 11,
            "sg_polyorder": 2,
            "snv": False,
            "msc": True,
            "emsc": False,
            "detrend": False,
            "detrend_order": 1,
        },
    )
    save_signal_domain_contract(
        path,
        "signal_domain_1",
        {
            "input_format": "hdf5",
            "source_modality": "nir_hsi",
            "required_signal_unit": "Absorbance",
            "required_calibration_mode": "dark_white_reference",
            "requires_dark_ref": True,
            "requires_white_ref": True,
            "calibrated_input_allowed": False,
            "n_bands": 3,
            "wavelengths": wavelengths,
            "wavelength_tolerance_nm": 0.0,
            "band_mapping_policy": "exact",
            "output_dtype": "float32",
            "created_at": "2026-04-02T10:04:00+00:00",
        },
    )

    roi_payloads = {
        "roi_1": (np.array([[10, 20, 30], [12, 22, 32]], dtype=np.uint16), 1.0),
        "roi_2": (np.array([[14, 24, 34], [16, 26, 36]], dtype=np.uint16), 2.0),
        "roi_3": (np.array([[18, 28, 38], [20, 30, 40]], dtype=np.uint16), 3.0),
        "roi_4": (np.array([[22, 32, 42], [24, 34, 44]], dtype=np.uint16), 4.0),
    }
    for index, (roi_id, (spectra, value)) in enumerate(roi_payloads.items(), start=1):
        save_training_roi(
            path,
            roi_id,
            {
                "dataset_id": "dataset_train",
                "calibration_id": "cal_train",
                "name": roi_id.upper(),
                "source_file": "/data/train.h5",
                "n_pixels": int(spectra.shape[0]),
                "created_at": f"2026-04-02T10:0{4 + index}:00+00:00",
                "spatial_bounds": (index, index, 0, 1),
                "wavelengths": wavelengths,
                "provenance": roi_provenance(
                    dataset_id="dataset_train",
                    calibration_id="cal_train",
                    created_at=f"2026-04-02T10:0{4 + index}:00+00:00"
                ),
            },
            spectra,
        )
        save_target_assignment(
            path,
            "target_moisture",
            roi_id,
            {"value": value, "assigned_at": f"2026-04-02T10:1{index}:00+00:00"},
        )


def _fit_phase6_model(project_path: str) -> ProjectState:
    state = ProjectState()
    state.open_project(project_path)
    sample_set = state.build_sample_set(
        target_id="target_moisture",
        preprocessing_recipe_id="recipe_none",
        sample_set_id="sample_set_1",
    )
    state.fit_model_run(
        sample_set_id=sample_set.sample_set.sample_set_id,
        signal_domain_id="signal_domain_1",
        model_type="PLS",
        hyperparameters={"n_components": 1},
        cv="logo",
        model_run_id="model_run_1",
    )
    return state


def test_preflight_prediction_uses_memory_budgeted_default_frame_chunks(tmp_path):
    class _DummyPredictor:
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=np.float32)

    def _preflight(cube_shape: tuple[int, int, int]):
        n_bands = cube_shape[2]
        wavelengths = np.arange(n_bands, dtype=np.float32)
        dataset = DatasetRef(
            dataset_id="dataset_large",
            source_path="/data/large.h5",
            source_format="hdf5",
            source_modality="nir_hsi",
            cube_shape=cube_shape,
            n_bands=n_bands,
            wavelengths=wavelengths,
            loader_metadata={},
            import_timestamp="2026-04-02T11:09:50+00:00",
        )
        calibration = CalibrationSpec(
            calibration_id="cal_raw",
            dataset_id=dataset.dataset_id,
            mode="raw_dn",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=None,
            white_level=None,
            output_signal_unit="DN",
            created_at="2026-04-02T11:09:50+00:00",
        )
        signal_domain = SignalDomainContract(
            signal_domain_id="signal_domain_raw",
            input_format="hdf5",
            source_modality="nir_hsi",
            required_signal_unit="DN",
            required_calibration_mode="raw_dn",
            requires_dark_ref=False,
            requires_white_ref=False,
            calibrated_input_allowed=False,
            n_bands=n_bands,
            wavelengths=wavelengths,
            wavelength_tolerance_nm=0.0,
            band_mapping_policy="exact",
            output_dtype="float32",
            created_at="2026-04-02T11:09:50+00:00",
        )
        model_run = ModelRun(
            model_run_id="model_large",
            sample_set_id="sample_large",
            signal_domain_id=signal_domain.signal_domain_id,
            model_type="PLS",
            hyperparameters={},
            cv_spec={},
            metrics={},
            artifact_refs={},
            created_at="2026-04-02T11:09:50+00:00",
        )
        return preflight_prediction(
            project_path=str(tmp_path / "unused.h5"),
            model_run=model_run,
            signal_domain=signal_domain,
            recipe=PreprocessingRecipe(),
            dataset=dataset,
            calibration=calibration,
            model_registry={model_run.model_run_id: _DummyPredictor()},
        )

    typical = _preflight((3490, 312, 220))
    oversized = _preflight((1000, 1000, 1000))

    assert typical.chunk_size == 512
    assert typical.n_chunks == (3490 + 511) // 512
    assert oversized.chunk_size == 512


def test_preflight_prediction_rejects_invalid_chunk_size(tmp_path):
    class _DummyPredictor:
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=np.float32)

    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    dataset = DatasetRef(
        dataset_id="dataset_chunk",
        source_path="/data/chunk.h5",
        source_format="hdf5",
        source_modality="nir_hsi",
        cube_shape=(2, 2, 3),
        n_bands=3,
        wavelengths=wavelengths,
        loader_metadata={},
        import_timestamp="2026-04-02T11:09:50+00:00",
    )
    calibration = CalibrationSpec(
        calibration_id="cal_chunk",
        dataset_id=dataset.dataset_id,
        mode="raw_dn",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=None,
        white_level=None,
        output_signal_unit="DN",
        created_at="2026-04-02T11:09:50+00:00",
    )
    signal_domain = SignalDomainContract(
        signal_domain_id="signal_domain_chunk",
        input_format="hdf5",
        source_modality="nir_hsi",
        required_signal_unit="DN",
        required_calibration_mode="raw_dn",
        requires_dark_ref=False,
        requires_white_ref=False,
        calibrated_input_allowed=False,
        n_bands=3,
        wavelengths=wavelengths,
        wavelength_tolerance_nm=0.0,
        band_mapping_policy="exact",
        output_dtype="float32",
        created_at="2026-04-02T11:09:50+00:00",
    )
    model_run = ModelRun(
        model_run_id="model_chunk",
        sample_set_id="sample_chunk",
        signal_domain_id=signal_domain.signal_domain_id,
        model_type="PLS",
        hyperparameters={},
        cv_spec={},
        metrics={},
        artifact_refs={},
        created_at="2026-04-02T11:09:50+00:00",
    )

    with pytest.raises(ValueError, match="chunk_size"):
        preflight_prediction(
            project_path=str(tmp_path / "unused.h5"),
            model_run=model_run,
            signal_domain=signal_domain,
            recipe=PreprocessingRecipe(),
            dataset=dataset,
            calibration=calibration,
            model_registry={model_run.model_run_id: _DummyPredictor()},
            chunk_size=0,
        )


def test_prediction_dataset_source_path_accepts_equivalent_slash_styles(tmp_path):
    source_path = tmp_path / "prediction_source.h5"
    spectral_data = SpectralData(
        X=np.empty((0, 2, 3), dtype=np.uint16),
        wavelengths=np.arange(3, dtype=np.float32),
        meta={"frames_total": 4, "source_modality": "nir_hsi"},
    )
    dataset = DatasetRef(
        dataset_id="dataset_slashes",
        source_path=str(source_path).replace("\\", "/"),
        source_format="hdf5",
        source_modality="nir_hsi",
        cube_shape=(4, 2, 3),
        n_bands=3,
        wavelengths=np.arange(3, dtype=np.float32),
        loader_metadata={},
        import_timestamp="2026-04-02T11:09:50+00:00",
    )

    from enginir._workflow import resolve_dataset

    resolved, is_new = resolve_dataset({dataset.dataset_id: dataset},
                                       str(source_path).replace("/", "\\"), spectral_data,
                                       dataset_id=dataset.dataset_id)

    assert resolved is dataset and not is_new


def _fit_phase6_model_with_recipe(
    project_path: str, *, recipe_id: str, model_run_id: str
) -> ProjectState:
    state = ProjectState()
    state.open_project(project_path)
    sample_set = state.build_sample_set(
        target_id="target_moisture",
        preprocessing_recipe_id=recipe_id,
        sample_set_id=f"sample_set_{recipe_id}",
    )
    state.fit_model_run(
        sample_set_id=sample_set.sample_set.sample_set_id,
        signal_domain_id="signal_domain_1",
        model_type="PLS",
        hyperparameters={"n_components": 1},
        cv="logo",
        model_run_id=model_run_id,
    )
    return state


def _write_prediction_cube(path: str, *, wavelengths: np.ndarray) -> None:
    cube = np.array(
        [
            [[11.0, 21.0, 31.0], [13.0, 23.0, 33.0]],
            [[15.0, 25.0, 35.0], [17.0, 27.0, 37.0]],
        ],
        dtype=np.float32,
    )
    save_hdf5(
        path,
        cube,
        wavelengths,
        meta={"source_modality": "nir_hsi", "instrument": "alpha"},
    )


def test_predict_dataset_reuses_dataset_local_calibration_defaults_after_reopen(tmp_path, qapp):
    project_path = str(tmp_path / "phase6_predict_defaults.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)

    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    pred_path = str(tmp_path / "prediction_defaults.h5")
    _write_prediction_cube(pred_path, wavelengths=wavelengths)
    state.register_dataset(
        DatasetRef(
            dataset_id="dataset_pred",
            source_path=pred_path,
            source_format="hdf5",
            source_modality="nir_hsi",
            cube_shape=(2, 2, 3),
            n_bands=3,
            wavelengths=wavelengths,
            loader_metadata={},
            import_timestamp="2026-04-02T11:00:00+00:00",
        )
    )
    state.set_active_calibration(
        CalibrationSpec(
            calibration_id="cal_pred",
            dataset_id="dataset_pred",
            mode="dark_white_reference",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=0.0,
            white_level=100.0,
            output_signal_unit="Absorbance",
            created_at="2026-04-02T11:01:00+00:00",
        )
    )

    in_memory_result = state.predict_dataset(
        model_run_id="model_run_1",
        data_path=pred_path,
        prediction_run_id="prediction_run_in_memory",
        return_arrays=True,
    )

    reopened = ProjectState()
    reopened.open_project(project_path)
    result = reopened.predict_dataset(
        model_run_id="model_run_1",
        data_path=pred_path,
        prediction_run_id="prediction_run_1",
    )

    assert result.dataset.dataset_id == "dataset_pred"
    assert result.calibration.calibration_id == "cal_pred"
    assert result.prediction_run.preprocessing_recipe_id == "recipe_none"
    assert result.y_pred is None
    assert result.spatial_map is None
    assert result.prediction_run.artifact_refs["prediction_shape"] == (4,)
    assert result.prediction_run.artifact_refs["spatial_map_shape"] == (2, 2)
    assert "prediction_run_1" in reopened.predictions
    assert "spatial_map" not in reopened.predictions["prediction_run_1"].artifact_refs
    artifact_refs = reopened.ensure_prediction_artifact_refs("prediction_run_1")
    np.testing.assert_allclose(artifact_refs["y_pred"], in_memory_result.y_pred, rtol=1e-5)
    np.testing.assert_allclose(artifact_refs["spatial_map"], in_memory_result.spatial_map, rtol=1e-5)


def test_reopened_prediction_run_lazily_loads_spatial_map_and_preserves_model_linkage(
    tmp_path, qapp
):
    project_path = str(tmp_path / "phase6_prediction_reopen_lazy.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)
    state.register_prediction_run(
        PredictionRun(
            prediction_run_id="prediction_run_1",
            dataset_id="dataset_train",
            calibration_id="cal_train",
            model_run_id="model_run_1",
            signal_domain_id="signal_domain_1",
            preprocessing_recipe_id="recipe_none",
            artifact_refs={
                "y_pred": np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
                "spatial_map": np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
            },
            created_at="2026-04-02T10:05:30+00:00",
        )
    )

    reopened = ProjectState()
    reopened.open_project(project_path)

    prediction = reopened.predictions["prediction_run_1"]
    assert prediction.model_run_id == "model_run_1"
    assert reopened.model_runs[prediction.model_run_id].model_type == "PLS"
    assert "y_pred" not in prediction.artifact_refs
    assert "spatial_map" not in prediction.artifact_refs

    artifact_refs = reopened.ensure_prediction_artifact_refs("prediction_run_1")

    np.testing.assert_allclose(
        artifact_refs["y_pred"], np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    )
    np.testing.assert_allclose(
        artifact_refs["spatial_map"],
        np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
    )
    assert artifact_refs["spatial_map"].shape == (2, 2)


def test_predict_dataset_requires_explicit_calibration_when_defaults_absent(tmp_path, qapp):
    project_path = str(tmp_path / "phase6_predict_no_defaults.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)

    pred_path = str(tmp_path / "prediction_no_defaults.h5")
    _write_prediction_cube(
        pred_path, wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    )

    with pytest.raises(ValueError, match="explicit calibration required"):
        state.predict_dataset(
            model_run_id="model_run_1",
            data_path=pred_path,
            prediction_run_id="prediction_run_missing_cal",
        )


def test_predict_dataset_persists_new_dataset_and_explicit_calibration(tmp_path, qapp):
    project_path = str(tmp_path / "phase6_predict_explicit.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)

    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    pred_path = str(tmp_path / "prediction_explicit.h5")
    _write_prediction_cube(pred_path, wavelengths=wavelengths)

    result = state.predict_dataset(
        model_run_id="model_run_1",
        data_path=pred_path,
        calibration=CalibrationSpec(
            calibration_id="cal_new",
            dataset_id="",
            mode="dark_white_reference",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=0.0,
            white_level=100.0,
            output_signal_unit="Absorbance",
            created_at="2026-04-02T11:02:00+00:00",
        ),
        prediction_run_id="prediction_run_2",
    )

    reopened = ProjectState()
    reopened.open_project(project_path)
    assert result.dataset.dataset_id in reopened.datasets
    assert reopened.active_calibration_ids[result.dataset.dataset_id] == "cal_new"
    assert "prediction_run_2" in reopened.predictions
    assert reopened.predictions["prediction_run_2"].dataset_id == result.dataset.dataset_id


def test_predict_dataset_allows_different_wavelength_metadata(tmp_path, qapp):
    project_path = str(tmp_path / "phase6_predict_different_wavelengths.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)

    pred_path = str(tmp_path / "prediction_different_wavelengths.h5")
    _write_prediction_cube(
        pred_path, wavelengths=np.array([1000.0, 1110.0, 1200.0], dtype=np.float32)
    )

    result = state.predict_dataset(
        model_run_id="model_run_1",
        data_path=pred_path,
        calibration=CalibrationSpec(
            calibration_id="cal_different_wl",
            dataset_id="",
            mode="dark_white_reference",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=0.0,
            white_level=100.0,
            output_signal_unit="Absorbance",
            created_at="2026-04-02T11:03:00+00:00",
        ),
        prediction_run_id="prediction_run_different_wl",
    )

    assert result.prediction_run.prediction_run_id == "prediction_run_different_wl"


def test_predict_dataset_rejects_incompatible_signal_unit(tmp_path, qapp):
    project_path = str(tmp_path / "phase6_predict_bad_signal.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)

    pred_path = str(tmp_path / "prediction_bad_signal.h5")
    _write_prediction_cube(
        pred_path, wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    )

    with pytest.raises(SignalDomainCompatibilityError, match="required_calibration_mode"):
        state.predict_dataset(
            model_run_id="model_run_1",
            data_path=pred_path,
            calibration=CalibrationSpec(
                calibration_id="cal_bad_signal",
                dataset_id="",
                mode="raw_dn",
                dark_ref_source=None,
                white_ref_source=None,
                dark_ref_data=None,
                white_ref_data=None,
                black_level=None,
                white_level=None,
                output_signal_unit="DN",
                created_at="2026-04-02T11:04:00+00:00",
            ),
            prediction_run_id="prediction_run_bad_signal",
        )


def _msc_prediction_calibration(calibration_id: str) -> CalibrationSpec:
    return CalibrationSpec(
        calibration_id=calibration_id,
        dataset_id="",
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=0.0,
        white_level=100.0,
        output_signal_unit="Absorbance",
        created_at="2026-04-02T11:05:00+00:00",
    )


def test_reopened_stateful_preprocessing_replays_persisted_training_state(tmp_path, qapp):
    project_path = str(tmp_path / "phase6_predict_reopened_msc.h5")
    _create_phase6_model_project(project_path)
    fitted = _fit_phase6_model_with_recipe(
        project_path, recipe_id="recipe_msc", model_run_id="model_run_msc"
    )

    in_memory_path = str(tmp_path / "prediction_in_memory_msc.h5")
    reopened_path = str(tmp_path / "prediction_reopened_msc.h5")
    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    _write_prediction_cube(in_memory_path, wavelengths=wavelengths)
    _write_prediction_cube(reopened_path, wavelengths=wavelengths)
    in_memory = fitted.predict_dataset(
        model_run_id="model_run_msc",
        data_path=in_memory_path,
        calibration=_msc_prediction_calibration("cal_msc_in_memory"),
        prediction_run_id="prediction_run_msc_in_memory",
        return_arrays=True,
    )

    reopened = ProjectState()
    reopened.open_project(project_path)
    replayed = reopened.predict_dataset(
        model_run_id="model_run_msc",
        data_path=reopened_path,
        calibration=_msc_prediction_calibration("cal_msc_reopened"),
        prediction_run_id="prediction_run_msc_reopened",
        return_arrays=True,
    )

    np.testing.assert_allclose(replayed.y_pred, in_memory.y_pred, rtol=1e-5, atol=1e-6)
    saved_model = load_model_run_prediction_artifact(project_path, "model_run_msc")
    np.testing.assert_array_equal(
        saved_model["preprocessing_state"]["msc"]["reference"],
        fitted.model_registry["model_run_msc"].named_steps["msc"].reference_,
    )


@pytest.mark.parametrize(
    ("damage", "match"),
    [("delete", "missing its fitted state"), ("wrong_length", "has 4 bands, the model needs 3")],
)
def test_reopened_stateful_preprocessing_rejects_missing_or_malformed_training_state(
    tmp_path, qapp, damage, match
):
    project_path = str(tmp_path / "phase6_predict_missing_msc_state.h5")
    _create_phase6_model_project(project_path)
    _fit_phase6_model_with_recipe(
        project_path, recipe_id="recipe_msc", model_run_id="model_run_msc"
    )
    state_path = "model_runs/model_run_msc/artifact_refs/prediction_model/preprocessing_state"
    with h5py.File(project_path, "a") as handle:
        if damage == "delete":
            del handle[state_path]
        else:  # a reference spectrum for a different band count
            msc_state = handle[f"{state_path}/msc"]
            del msc_state.attrs["reference"]
            msc_state.attrs["reference"] = np.array([1.0, 2.0, 3.0, 4.0])

    pred_path = str(tmp_path / "prediction_missing_msc_state.h5")
    _write_prediction_cube(
        pred_path, wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    )
    reopened = ProjectState()
    reopened.open_project(project_path)
    datasets_before = set(reopened.datasets)
    calibrations_before = set(reopened.calibrations)
    with pytest.raises(ValueError, match=match):
        reopened.predict_dataset(
            model_run_id="model_run_msc",
            data_path=pred_path,
            calibration=_msc_prediction_calibration("cal_msc_missing_state"),
            prediction_run_id="prediction_run_msc",
        )
    reopened.open_project(project_path)
    assert set(reopened.datasets) == datasets_before
    assert set(reopened.calibrations) == calibrations_before
    assert "prediction_run_msc" not in reopened.predictions


def test_predict_dataset_ignores_changed_wavelength_metadata_for_existing_dataset(tmp_path, qapp):
    project_path = str(tmp_path / "phase6_predict_changed_wavelength_metadata.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)

    pred_path = str(tmp_path / "prediction_stale.h5")
    _write_prediction_cube(
        pred_path, wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    )
    state.register_dataset(
        DatasetRef(
            dataset_id="dataset_stale",
            source_path=pred_path,
            source_format="hdf5",
            source_modality="nir_hsi",
            cube_shape=(2, 2, 3),
            n_bands=3,
            wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32),
            loader_metadata={},
            import_timestamp="2026-04-02T11:06:00+00:00",
        )
    )
    state.set_active_calibration(
        CalibrationSpec(
            calibration_id="cal_stale",
            dataset_id="dataset_stale",
            mode="dark_white_reference",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=0.0,
            white_level=100.0,
            output_signal_unit="Absorbance",
            created_at="2026-04-02T11:07:00+00:00",
        )
    )
    _write_prediction_cube(
        pred_path, wavelengths=np.array([1000.0, 1110.0, 1200.0], dtype=np.float32)
    )

    result = state.predict_dataset(
        model_run_id="model_run_1",
        data_path=pred_path,
        prediction_run_id="prediction_run_stale_dataset",
    )

    assert result.prediction_run.prediction_run_id == "prediction_run_stale_dataset"


def test_predict_with_model_run_accepts_uint16_cube_input_under_numpy2(tmp_path, qapp):
    project_path = str(tmp_path / "phase6_predict_numpy2_uint16.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)
    context = state.build_prediction_job_context(model_run_id="model_run_1")

    spectral_data = SpectralData(
        X=np.arange(12, dtype=np.uint16).reshape(2, 2, 3),
        wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float64),
        meta={"source_modality": "nir_hsi"},
    )
    dataset = build_dataset_ref_from_spectral_data(
        data_path=str(tmp_path / "prediction_uint16.h5"),
        spectral_data=spectral_data,
    )
    calibration = CalibrationSpec(
        calibration_id="cal_uint16",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=0.0,
        white_level=100.0,
        output_signal_unit="Absorbance",
        created_at="2026-04-02T11:08:00+00:00",
    )

    class _DummyPredictor:
        def predict(self, X):
            return np.sum(X, axis=1, dtype=np.float32)

    result = predict_with_model_run(
        project_path=context.project_path,
        model_run=context.model_run,
        signal_domain=context.signal_domain,
        recipe=context.recipe,
        dataset=dataset,
        calibration=calibration,
        model_registry={context.model_run.model_run_id: _DummyPredictor()},
        spectral_data=spectral_data,
        preprocessing_recipe_id=context.preprocessing_recipe_id,
        prediction_run_id="prediction_run_uint16",
        created_at="2026-04-02T11:09:00+00:00",
    )

    assert result.y_pred.shape == (4,)
    assert result.spatial_map is not None
    assert result.spatial_map.shape == (2, 2)
    assert np.shares_memory(result.y_pred, result.spatial_map)


def test_hdf5_prediction_reuses_source_and_artifact_file_handles(tmp_path, qapp):
    project_path = str(tmp_path / "prediction_handle_reuse_project.h5")
    source_path = str(tmp_path / "prediction_handle_reuse_source.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)
    context = state.build_prediction_job_context(model_run_id="model_run_1")
    save_hdf5(
        source_path,
        np.arange(18, dtype=np.uint16).reshape(3, 2, 3),
        np.array([1000.0, 1100.0, 1200.0], dtype=np.float32),
    )
    metadata = load_hdf5(source_path, header_only=True)
    dataset = build_dataset_ref_from_spectral_data(
        data_path=source_path,
        spectral_data=metadata,
        source_modality="nir_hsi",
    )
    calibration = CalibrationSpec(
        calibration_id="cal_handle_reuse",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=0.0,
        white_level=100.0,
        output_signal_unit="Reflectance",
        created_at="2026-04-02T11:08:30+00:00",
    )

    class _DummyPredictor:
        def predict(self, X):
            return np.sum(X, axis=1, dtype=np.float32)

    with patch("enginir.io.project_store.h5py.File", wraps=h5py.File) as mocked_file:
        predict_with_model_run(
            project_path=project_path,
            model_run=context.model_run,
            signal_domain=context.signal_domain,
            recipe=context.recipe,
            dataset=dataset,
            calibration=calibration,
            model_registry={context.model_run.model_run_id: _DummyPredictor()},
            data_path=source_path,
            spectral_data=metadata,
            preprocessing_recipe_id=context.preprocessing_recipe_id,
            prediction_run_id="prediction_handle_reuse",
            chunk_size=1,
            persist_artifacts=True,
            return_arrays=False,
        )

    opened_paths = [str(call.args[0]) for call in mocked_file.call_args_list]
    assert opened_paths.count(source_path) == 1
    # id check in preflight, initialize, chunk writer, finalize
    # A fixed number per job, independent of the chunk count: preflight, the target names of
    # the metadata-only model run, staging, the chunk writer and finalization.
    assert opened_paths.count(project_path) == 5


def test_predict_with_model_run_reports_chunk_progress_and_metadata(tmp_path, qapp):
    project_path = str(tmp_path / "phase6_predict_progress.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)
    context = state.build_prediction_job_context(model_run_id="model_run_1")
    spectral_data = SpectralData(
        X=np.array(
            [
                [[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]],
                [[20.0, 30.0, 40.0], [25.0, 35.0, 45.0]],
            ],
            dtype=np.float32,
        ),
        wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32),
        meta={"source_modality": "nir_hsi"},
    )
    dataset = build_dataset_ref_from_spectral_data(
        data_path=str(tmp_path / "prediction_progress.h5"),
        spectral_data=spectral_data,
    )
    calibration = CalibrationSpec(
        calibration_id="cal_progress",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=0.0,
        white_level=20.0,
        output_signal_unit="Reflectance",
        created_at="2026-04-02T11:09:30+00:00",
    )

    class _DummyPredictor:
        def predict(self, X):
            return np.sum(X, axis=1, dtype=np.float32)

    events: list[PredictionProgressEvent] = []
    result = predict_with_model_run(
        project_path=context.project_path,
        model_run=context.model_run,
        signal_domain=context.signal_domain,
        recipe=context.recipe,
        dataset=dataset,
        calibration=calibration,
        model_registry={context.model_run.model_run_id: _DummyPredictor()},
        spectral_data=spectral_data,
        preprocessing_recipe_id=context.preprocessing_recipe_id,
        prediction_run_id="prediction_run_progress",
        chunk_size=1,
        progress_callback=events.append,
    )

    assert result.prediction_run.artifact_refs["chunk_size"] == 1
    assert result.prediction_run.artifact_refs["n_chunks"] == 2
    assert result.prediction_run.artifact_refs["completed_chunks"] == 2
    assert result.prediction_run.artifact_refs["clip_high_count"] == 8
    assert "prediction_min" in result.prediction_run.artifact_refs
    assert "prediction_max" in result.prediction_run.artifact_refs
    assert {event.phase for event in events} >= {
        "preflight",
        "loading",
        "calibrating",
        "preprocessing",
        "predicting",
        "writing",
        "finalizing",
    }


def test_predict_with_model_run_cancels_between_chunks_and_discards_artifacts(tmp_path, qapp):
    project_path = str(tmp_path / "phase6_predict_cancel.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)
    context = state.build_prediction_job_context(model_run_id="model_run_1")
    spectral_data = SpectralData(
        X=np.ones((2, 2, 3), dtype=np.float32) * 10.0,
        wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32),
        meta={"source_modality": "nir_hsi"},
    )
    dataset = build_dataset_ref_from_spectral_data(
        data_path=str(tmp_path / "prediction_cancel.h5"),
        spectral_data=spectral_data,
    )
    calibration = CalibrationSpec(
        calibration_id="cal_cancel",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=0.0,
        white_level=20.0,
        output_signal_unit="Reflectance",
        created_at="2026-04-02T11:09:32+00:00",
    )

    class _DummyPredictor:
        def predict(self, X):
            return np.sum(X, axis=1, dtype=np.float32)

    events: list[PredictionProgressEvent] = []

    def cancel_after_first_chunk() -> bool:
        return any(event.phase == "predicting" and event.completed >= 1 for event in events)

    with pytest.raises(PredictionCancelledError, match="cancelled"):
        predict_with_model_run(
            project_path=context.project_path,
            model_run=context.model_run,
            signal_domain=context.signal_domain,
            recipe=context.recipe,
            dataset=dataset,
            calibration=calibration,
            model_registry={context.model_run.model_run_id: _DummyPredictor()},
            spectral_data=spectral_data,
            preprocessing_recipe_id=context.preprocessing_recipe_id,
            prediction_run_id="prediction_run_cancelled",
            chunk_size=1,
            progress_callback=events.append,
            cancel_callback=cancel_after_first_chunk,
            persist_artifacts=True,
            return_arrays=False,
        )

    reopened = ProjectState()
    reopened.open_project(project_path)
    assert "prediction_run_cancelled" not in reopened.predictions


def test_predict_with_model_run_can_persist_chunks_without_returning_arrays(tmp_path, qapp):
    from enginir.io.project_store import load_prediction_run_artifact_refs

    project_path = str(tmp_path / "phase6_predict_persist_chunks.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)
    context = state.build_prediction_job_context(model_run_id="model_run_1")
    spectral_data = SpectralData(
        X=np.ones((2, 2, 3), dtype=np.float32) * 50.0,
        wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32),
        meta={"source_modality": "nir_hsi"},
    )
    dataset = build_dataset_ref_from_spectral_data(
        data_path=str(tmp_path / "prediction_persist_chunks.h5"),
        spectral_data=spectral_data,
    )
    calibration = CalibrationSpec(
        calibration_id="cal_persist_chunks",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=0.0,
        white_level=100.0,
        output_signal_unit="Reflectance",
        created_at="2026-04-02T11:09:35+00:00",
    )

    class _DummyPredictor:
        def predict(self, X):
            return np.sum(X, axis=1, dtype=np.float32)

    result = predict_with_model_run(
        project_path=context.project_path,
        model_run=context.model_run,
        signal_domain=context.signal_domain,
        recipe=context.recipe,
        dataset=dataset,
        calibration=calibration,
        model_registry={context.model_run.model_run_id: _DummyPredictor()},
        spectral_data=spectral_data,
        preprocessing_recipe_id=context.preprocessing_recipe_id,
        prediction_run_id="prediction_run_persist_chunks",
        chunk_size=1,
        persist_artifacts=True,
        return_arrays=False,
    )

    assert result.artifacts_persisted is True
    assert result.y_pred is None
    assert result.spatial_map is None
    assert "y_pred" not in result.prediction_run.artifact_refs
    assert "spatial_map" not in result.prediction_run.artifact_refs

    artifact_refs = load_prediction_run_artifact_refs(
        project_path,
        "prediction_run_persist_chunks",
    )
    assert artifact_refs["artifact_write_status"] == "complete"
    assert artifact_refs["y_pred"].shape == (4,)
    assert artifact_refs["spatial_map"].shape == (2, 2)
    assert "prediction_min" in artifact_refs
    assert "prediction_max" in artifact_refs
    with h5py.File(project_path, "r") as handle:
        artifact_group = handle[
            "predictions/prediction_run_persist_chunks/artifact_refs"
        ]
        assert "y_pred" in artifact_group
        assert "spatial_map" not in artifact_group


def test_predict_with_model_run_rejects_bare_predictor_when_recipe_would_be_skipped(
    tmp_path, qapp
):
    project_path = str(tmp_path / "phase6_predict_bare_estimator.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)
    context = state.build_prediction_job_context(model_run_id="model_run_1")
    spectral_data = SpectralData(
        X=np.ones((2, 2, 3), dtype=np.float32),
        wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32),
        meta={"source_modality": "nir_hsi"},
    )
    dataset = build_dataset_ref_from_spectral_data(
        data_path=str(tmp_path / "prediction_bare.h5"),
        spectral_data=spectral_data,
    )
    calibration = CalibrationSpec(
        calibration_id="cal_bare",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=0.0,
        white_level=2.0,
        output_signal_unit="Reflectance",
        created_at="2026-04-02T11:09:40+00:00",
    )

    class _BareEstimator:
        def predict(self, X):
            return np.sum(X, axis=1, dtype=np.float32)

    with pytest.raises(ValueError, match="not recipe-aware"):
        predict_with_model_run(
            project_path=context.project_path,
            model_run=context.model_run,
            signal_domain=context.signal_domain,
            recipe=PreprocessingRecipe(snv=True),
            dataset=dataset,
            calibration=calibration,
            model_registry={context.model_run.model_run_id: _BareEstimator()},
            spectral_data=spectral_data,
            preprocessing_recipe_id="recipe_snv",
        )


def test_calibrations_normalize_the_case_of_enum_values():
    """Signal units and modes compare case-insensitively (dataset-calibration contract)."""
    calibration = CalibrationSpec(
        calibration_id="cal_signal_case",
        dataset_id="dataset_1",
        mode="Dark_White_Reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=0.0,
        white_level=100.0,
        output_signal_unit=" reflectance",
        created_at="2026-04-02T11:09:45+00:00",
    )
    assert calibration.mode == "dark_white_reference"
    assert calibration.output_signal_unit == "Reflectance"
    with pytest.raises(ValueError, match="output_signal_unit must be one of"):
        CalibrationSpec(
            calibration_id="cal_bad_unit",
            dataset_id="dataset_1",
            mode="dark_white_reference",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=0.0,
            white_level=100.0,
            output_signal_unit="transmittance",
            created_at="2026-04-02T11:09:45+00:00",
        )


def test_explicit_prediction_calibration_reuses_matching_dataset_calibration(qapp):
    dataset = DatasetRef(
        dataset_id="dataset_reuse",
        source_path="/tmp/predict.h5",
        source_format="hdf5",
        source_modality="nir_hsi",
        cube_shape=(1, 2, 3),
        n_bands=3,
        wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32),
        loader_metadata={},
        import_timestamp="2026-04-02T11:09:50+00:00",
    )
    existing = CalibrationSpec(
        calibration_id="cal_existing",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=0.0,
        white_level=100.0,
        output_signal_unit="Reflectance",
        created_at="2026-04-02T11:09:50+00:00",
    )
    explicit = CalibrationSpec(
        calibration_id="",
        dataset_id="",
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=0.0,
        white_level=100.0,
        output_signal_unit="Reflectance",
        created_at="2026-04-02T11:10:00+00:00",
    )

    from enginir._workflow import resolve_prediction_calibration

    resolved, is_new = resolve_prediction_calibration(
        dataset, {existing.calibration_id: existing}, {}, explicit)

    assert resolved is existing
    assert is_new is False


def test_predict_with_model_run_aligns_full_sensor_refs_before_flattening(tmp_path, qapp):
    project_path = str(tmp_path / "phase6_predict_embedded_refs.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)
    context = state.build_prediction_job_context(model_run_id="model_run_1")

    raw_cube = np.array(
        [
            [[20, 30, 40], [60, 70, 80]],
            [[30, 40, 50], [70, 80, 90]],
        ],
        dtype=np.uint16,
    )
    spectral_data = SpectralData(
        X=raw_cube,
        wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float64),
        meta={"source_modality": "nir_hsi"},
    )
    dataset = build_dataset_ref_from_spectral_data(
        data_path=str(tmp_path / "prediction_embedded_refs.h5"),
        spectral_data=spectral_data,
    )
    dark_ref = np.array([[0, 10, 20], [10, 20, 30]], dtype=np.float32)
    white_ref = dark_ref + 100.0
    calibration = CalibrationSpec(
        calibration_id="cal_embedded_refs",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=dark_ref,
        white_ref_data=white_ref,
        black_level=None,
        white_level=None,
        output_signal_unit="Absorbance",
        created_at="2026-04-02T11:10:00+00:00",
    )

    class _CapturingPredictor:
        def __init__(self):
            self.X: np.ndarray | None = None

        def predict(self, X):
            self.X = np.asarray(X, dtype=np.float32)
            return np.sum(self.X, axis=1, dtype=np.float32)

    predictor = _CapturingPredictor()
    result = predict_with_model_run(
        project_path=context.project_path,
        model_run=context.model_run,
        signal_domain=context.signal_domain,
        recipe=context.recipe,
        dataset=dataset,
        calibration=calibration,
        model_registry={context.model_run.model_run_id: predictor},
        spectral_data=spectral_data,
        preprocessing_recipe_id=context.preprocessing_recipe_id,
        prediction_run_id="prediction_run_embedded_refs",
        created_at="2026-04-02T11:11:00+00:00",
    )

    assert result.y_pred.shape == (4,)
    assert predictor.X is not None
    tiled_dark = np.tile(dark_ref, (2, 1))
    tiled_white = np.tile(white_ref, (2, 1))
    raw_flat = raw_cube.reshape(-1, raw_cube.shape[-1]).astype(np.float32)
    expected = -np.log10(np.clip((raw_flat - tiled_dark) / (tiled_white - tiled_dark), 1e-6, None))
    np.testing.assert_allclose(predictor.X, expected.astype(np.float32), rtol=1e-6)


def test_transform_spectra_infers_spatial_shape_for_full_sensor_refs():
    raw_cube = np.array(
        [
            [[20, 30, 40], [60, 70, 80]],
            [[30, 40, 50], [70, 80, 90]],
        ],
        dtype=np.uint16,
    )
    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    dataset = DatasetRef(
        dataset_id="dataset_flat",
        source_path="/data/prediction.h5",
        source_format="hdf5",
        source_modality="nir_hsi",
        cube_shape=(2, 2, 3),
        n_bands=3,
        wavelengths=wavelengths,
        loader_metadata={},
        import_timestamp="2026-04-02T11:12:00+00:00",
    )
    dark_ref = np.array([[0, 10, 20], [10, 20, 30]], dtype=np.float32)
    white_ref = dark_ref + 100.0
    calibration = CalibrationSpec(
        calibration_id="cal_flat_refs",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=dark_ref,
        white_ref_data=white_ref,
        black_level=None,
        white_level=None,
        output_signal_unit="Reflectance",
        created_at="2026-04-02T11:13:00+00:00",
    )
    signal_domain = SignalDomainContract(
        signal_domain_id="signal_domain_reflectance",
        input_format="hdf5",
        source_modality="nir_hsi",
        required_signal_unit="Reflectance",
        required_calibration_mode="dark_white_reference",
        requires_dark_ref=True,
        requires_white_ref=True,
        calibrated_input_allowed=False,
        n_bands=3,
        wavelengths=wavelengths,
        wavelength_tolerance_nm=0.0,
        band_mapping_policy="exact",
        output_dtype="float32",
        created_at="2026-04-02T11:00:00+00:00",
    )

    transformed = transform_spectra_to_signal_domain(
        raw_cube.reshape(-1, raw_cube.shape[-1]),
        dataset=dataset,
        calibration=calibration,
        signal_domain=signal_domain,
    )

    expected = (raw_cube.astype(np.float32) - dark_ref) / (white_ref - dark_ref)
    np.testing.assert_allclose(transformed, expected.reshape(-1, raw_cube.shape[-1]), rtol=1e-6)
    assert transformed.dtype == np.float32


@pytest.mark.parametrize(
    ("raw_spectra", "match"),
    [
        (np.ones((3,), dtype=np.float32), "2-D"),
        (np.empty((0, 3), dtype=np.float32), "at least one"),
        (np.array([[1.0, np.nan, 3.0]], dtype=np.float32), "finite"),
        (np.array([["bad", "data", "x"]], dtype=object), "numeric"),
    ],
)
def test_transform_spectra_to_signal_domain_rejects_invalid_raw_spectra(raw_spectra, match):
    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    dataset = DatasetRef(
        dataset_id="dataset_invalid_raw",
        source_path="/data/prediction.h5",
        source_format="hdf5",
        source_modality="nir_hsi",
        cube_shape=(1, 1, 3),
        n_bands=3,
        wavelengths=wavelengths,
        loader_metadata={},
        import_timestamp="2026-04-02T11:12:00+00:00",
    )
    calibration = CalibrationSpec(
        calibration_id="cal_invalid_raw",
        dataset_id=dataset.dataset_id,
        mode="raw_dn",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=None,
        white_level=None,
        output_signal_unit="DN",
        created_at="2026-04-02T11:13:00+00:00",
    )
    signal_domain = SignalDomainContract(
        signal_domain_id="signal_domain_invalid_raw",
        input_format="hdf5",
        source_modality="nir_hsi",
        required_signal_unit="DN",
        required_calibration_mode="raw_dn",
        requires_dark_ref=False,
        requires_white_ref=False,
        calibrated_input_allowed=False,
        n_bands=3,
        wavelengths=wavelengths,
        wavelength_tolerance_nm=0.0,
        band_mapping_policy="exact",
        output_dtype="float32",
        created_at="2026-04-02T11:00:00+00:00",
    )

    with pytest.raises(ValueError, match=match):
        transform_spectra_to_signal_domain(
            raw_spectra,
            dataset=dataset,
            calibration=calibration,
            signal_domain=signal_domain,
        )


def _scalar_reflectance_calibration(dataset_id: str) -> CalibrationSpec:
    return CalibrationSpec(
        calibration_id="cal_scalar",
        dataset_id=dataset_id,
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=0.0,
        white_level=200.0,
        output_signal_unit="Reflectance",
        created_at="2026-04-02T11:09:32+00:00",
    )


@pytest.mark.parametrize("cancel_at_poll", range(1, 12))
def test_cancelling_at_any_point_leaves_no_staged_run(tmp_path, qapp, cancel_at_poll):
    project_path = str(tmp_path / "cancel_any.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)
    context = state.build_prediction_job_context(model_run_id="model_run_1")
    spectral_data = SpectralData(
        X=np.full((2, 2, 3), 50.0, dtype=np.float32),
        wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32),
        meta={"source_modality": "nir_hsi"},
    )
    dataset = build_dataset_ref_from_spectral_data(
        data_path=str(tmp_path / "any.h5"), spectral_data=spectral_data
    )
    polls = 0

    def cancel() -> bool:
        nonlocal polls
        polls += 1
        return polls == cancel_at_poll

    try:
        predict_with_model_run(
            project_path=context.project_path,
            model_run=context.model_run,
            signal_domain=context.signal_domain,
            recipe=context.recipe,
            dataset=dataset,
            calibration=_scalar_reflectance_calibration(dataset.dataset_id),
            model_registry={},
            spectral_data=spectral_data,
            preprocessing_recipe_id=context.preprocessing_recipe_id,
            prediction_run_id="prediction_run_cancel_any",
            chunk_size=1,
            cancel_callback=cancel,
            persist_artifacts=True,
            return_arrays=False,
        )
    except PredictionCancelledError:
        cancelled = True
    else:
        cancelled = False

    from enginir.io.project_store import load_project_store

    store = load_project_store(project_path)  # raises on a half-written ("writing") run
    assert ("prediction_run_cancel_any" in store["predictions"]) is not cancelled


def test_chunked_saved_model_prediction_matches_one_shot_prediction(tmp_path, qapp):
    project_path = str(tmp_path / "chunked.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)
    context = state.build_prediction_job_context(model_run_id="model_run_1")
    rng = np.random.default_rng(5)
    spectral_data = SpectralData(
        X=rng.uniform(10.0, 190.0, size=(7, 3, 3)).astype(np.float32),
        wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32),
        meta={"source_modality": "nir_hsi"},
    )
    dataset = build_dataset_ref_from_spectral_data(
        data_path=str(tmp_path / "chunked_input.h5"), spectral_data=spectral_data
    )
    calibration = _scalar_reflectance_calibration(dataset.dataset_id)

    def predict(chunk_size: int, persist: bool, run_id: str):
        return predict_with_model_run(
            project_path=context.project_path,
            model_run=context.model_run,
            signal_domain=context.signal_domain,
            recipe=context.recipe,
            dataset=dataset,
            calibration=calibration,
            model_registry={},  # replay the saved artifact
            spectral_data=spectral_data,
            preprocessing_recipe_id=context.preprocessing_recipe_id,
            prediction_run_id=run_id,
            chunk_size=chunk_size,
            persist_artifacts=persist,
            return_arrays=True,
        )

    one_shot = predict(7, False, "one_shot")
    chunked = predict(3, True, "chunked")  # chunks of 3, 3 and 1 frames

    np.testing.assert_allclose(chunked.y_pred, one_shot.y_pred, rtol=1e-5, atol=1e-6)
    assert chunked.spatial_map.shape == (7, 3)
    from enginir.io.project_store import load_prediction_run_artifact_refs

    stored = load_prediction_run_artifact_refs(project_path, "chunked", keys=("y_pred",))
    np.testing.assert_allclose(stored["y_pred"], chunked.y_pred)
    assert chunked.prediction_run.artifact_refs["chunk_axis"] == "frames"
    assert chunked.prediction_run.artifact_refs["completed_chunks"] == 3


def test_training_and_prediction_use_identical_signal_conversion(tmp_path, qapp):
    project_path = str(tmp_path / "identity.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)
    context = state.build_prediction_job_context(model_run_id="model_run_1")
    roi = next(iter(state.rois.values()))
    raw = roi.load_spectra(project_path)
    calibration = state.calibrations[roi.calibration_id]
    dataset = state.datasets[roi.dataset_id]

    training = transform_spectra_to_signal_domain(
        raw, dataset=dataset, calibration=calibration, signal_domain=context.signal_domain
    )
    chunks = [
        transform_spectra_to_signal_domain(
            raw[i:i + 1], dataset=dataset, calibration=calibration,
            signal_domain=context.signal_domain, check_compatibility=False,
        )
        for i in range(raw.shape[0])
    ]

    np.testing.assert_array_equal(np.vstack(chunks), training)
    assert training.dtype == np.float32


def _second_review_setup(tmp_path, name: str, X: np.ndarray, **calibration_kwargs):
    project_path = str(tmp_path / f"{name}.h5")
    _create_phase6_model_project(project_path)
    state = _fit_phase6_model(project_path)
    context = state.build_prediction_job_context(model_run_id="model_run_1")
    spectral_data = SpectralData(X=X, wavelengths=np.array([1000.0, 1100.0, 1200.0]),
                                 meta={"source_modality": "nir_hsi"})
    dataset = build_dataset_ref_from_spectral_data(
        data_path=str(tmp_path / f"{name}_data.h5"), spectral_data=spectral_data,
    )
    levels = {"black_level": 0.0, "white_level": 20.0}
    if "dark_ref_data" in calibration_kwargs:
        levels = {"black_level": None, "white_level": None}
    fields = {"dark_ref_data": None, "white_ref_data": None, **levels, **calibration_kwargs}
    calibration = CalibrationSpec(
        calibration_id=f"cal_{name}", dataset_id=dataset.dataset_id,
        mode="dark_white_reference", dark_ref_source=None, white_ref_source=None,
        output_signal_unit="Reflectance", created_at="2026-04-02T11:09:30+00:00", **fields,
    )
    return context, spectral_data, dataset, calibration


class _SumPredictor:
    def predict(self, X):
        return np.sum(X, axis=1, dtype=np.float32)


def test_prediction_never_replaces_an_existing_run(tmp_path, qapp):
    X = np.ones((2, 2, 3), dtype=np.float32) * 10.0
    context, spectral_data, dataset, calibration = _second_review_setup(tmp_path, "reuse", X)
    kwargs = dict(
        project_path=context.project_path, model_run=context.model_run,
        signal_domain=context.signal_domain, recipe=context.recipe, dataset=dataset,
        calibration=calibration, model_registry={context.model_run.model_run_id: _SumPredictor()},
        spectral_data=spectral_data, preprocessing_recipe_id=context.preprocessing_recipe_id,
        prediction_run_id="prediction_once", persist_artifacts=True,
    )
    predict_with_model_run(**kwargs)

    with pytest.raises(ValueError, match="already exists"):
        predict_with_model_run(**kwargs)
    from enginir.io.project_store import load_prediction_run_artifact_refs

    refs = load_prediction_run_artifact_refs(context.project_path, "prediction_once",
                                             keys=("y_pred",))
    assert refs["y_pred"].shape == (4,)


def test_preflight_rejects_a_pipeline_without_the_recipe_steps(tmp_path, qapp):
    from types import SimpleNamespace

    X = np.ones((2, 2, 3), dtype=np.float32)
    context, spectral_data, dataset, calibration = _second_review_setup(tmp_path, "steps", X)
    pipeline = SimpleNamespace(named_steps={"model": _SumPredictor()},
                               predict=_SumPredictor().predict)

    with pytest.raises(ValueError, match=r"lacks the recipe step\(s\) \['snv'\]"):
        predict_with_model_run(
            project_path=context.project_path, model_run=context.model_run,
            signal_domain=context.signal_domain, recipe=PreprocessingRecipe(snv=True),
            dataset=dataset, calibration=calibration,
            model_registry={context.model_run.model_run_id: pipeline},
            spectral_data=spectral_data, preprocessing_recipe_id="recipe_snv",
        )


def test_every_chunk_reports_the_phases_in_order(tmp_path, qapp):
    X = np.ones((2, 2, 3), dtype=np.float32) * 10.0
    context, spectral_data, dataset, calibration = _second_review_setup(tmp_path, "phases", X)
    events: list[PredictionProgressEvent] = []
    predict_with_model_run(
        project_path=context.project_path, model_run=context.model_run,
        signal_domain=context.signal_domain, recipe=context.recipe, dataset=dataset,
        calibration=calibration, model_registry={context.model_run.model_run_id: _SumPredictor()},
        spectral_data=spectral_data, preprocessing_recipe_id=context.preprocessing_recipe_id,
        chunk_size=1, progress_callback=events.append,
    )

    per_chunk = ["loading", "calibrating", "preprocessing", "predicting", "writing"]
    assert [event.phase for event in events] == [
        "preflight", *per_chunk, *per_chunk, "finalizing", "finalizing"
    ]


def test_flat_spectra_with_per_sensor_references_chunk_like_one_shot(tmp_path, qapp):
    cube = np.array([[[20, 30, 40], [60, 70, 80]], [[30, 40, 50], [70, 80, 90]],
                     [[25, 35, 45], [65, 75, 85]]], dtype=np.uint16)
    dark = np.array([[0, 10, 20], [10, 20, 30]], dtype=np.float32)
    context, spectral_data, dataset, calibration = _second_review_setup(
        tmp_path, "flat_refs", cube, dark_ref_data=dark, white_ref_data=dark + 100.0
    )
    flat = SpectralData(X=cube.reshape(6, 3), wavelengths=spectral_data.wavelengths,
                        meta=spectral_data.meta)

    def predict(data, chunk_size):
        return predict_with_model_run(
            project_path=context.project_path, model_run=context.model_run,
            signal_domain=context.signal_domain, recipe=context.recipe, dataset=dataset,
            calibration=calibration,
            model_registry={context.model_run.model_run_id: _SumPredictor()},
            spectral_data=data, preprocessing_recipe_id=context.preprocessing_recipe_id,
            chunk_size=chunk_size,
        ).y_pred

    np.testing.assert_array_equal(predict(flat, 1), predict(spectral_data, 3))


def _predict_kwargs(context, spectral_data, dataset, calibration, predictor, **overrides):
    return {"project_path": context.project_path, "model_run": context.model_run,
            "signal_domain": context.signal_domain, "recipe": context.recipe, "dataset": dataset,
            "calibration": calibration,
            "model_registry": {context.model_run.model_run_id: predictor},
            "spectral_data": spectral_data,
            "preprocessing_recipe_id": context.preprocessing_recipe_id, **overrides}


def test_persisted_run_records_clip_counts_output_names_and_a_1d_single_output(tmp_path, qapp):
    """Runtime contract: clipping telemetry and output metadata are stored with the run."""
    from enginir.io.project_store import load_prediction_run_artifact_refs

    X = np.array([[[10.0, 15.0, 30.0], [12.5, 20.0, 25.0]],
                  [[5.0, 18.0, 40.0], [16.0, 11.0, 19.0]]], dtype=np.float32)
    black, white = 12.0, 20.0
    context, spectral_data, dataset, calibration = _second_review_setup(
        tmp_path, "telemetry", X, black_level=black, white_level=white)
    reflectance = (X - black) / (white - black)
    result = predict_with_model_run(**_predict_kwargs(
        context, spectral_data, dataset, calibration, _SumPredictor(),
        prediction_run_id="prediction_telemetry", persist_artifacts=True))

    stored = load_prediction_run_artifact_refs(context.project_path, "prediction_telemetry")
    assert stored["clip_low_count"] == int((reflectance < 1e-6).sum())
    assert stored["clip_high_count"] == int((reflectance > 1.0).sum())
    saved_model = load_model_run_prediction_artifact(context.project_path,
                                                     context.model_run.model_run_id)
    expected_names = list(saved_model["target_names"])
    assert list(stored["output_names"]) == expected_names
    assert result.y_pred.shape == (4,) and stored["y_pred"].shape == (4,)


def test_multi_output_predictions_store_their_output_axis(tmp_path, qapp):
    from enginir.io.project_store import load_prediction_run_artifact_refs

    class _TwoOutputs:
        def predict(self, X):
            return np.column_stack([X.sum(axis=1), X.mean(axis=1)]).astype(np.float32)

    X = np.full((3, 2, 3), 10.0, dtype=np.float32)
    context, spectral_data, dataset, calibration = _second_review_setup(tmp_path, "multi", X)
    result = predict_with_model_run(**_predict_kwargs(
        context, spectral_data, dataset, calibration, _TwoOutputs(),
        prediction_run_id="prediction_multi", persist_artifacts=True))

    assert result.spatial_map.shape == (3, 2, 2)
    stored = load_prediction_run_artifact_refs(context.project_path, "prediction_multi")
    assert int(stored["n_outputs"]) == 2
    assert list(np.ravel(stored["output_axis"])) == [2]
    assert "output_index" not in stored  # no output is chosen for the user


def test_preflight_rejects_a_signal_domain_of_another_model_run(tmp_path, qapp):
    from dataclasses import replace

    X = np.full((2, 2, 3), 10.0, dtype=np.float32)
    context, spectral_data, dataset, calibration = _second_review_setup(tmp_path, "foreign", X)
    foreign = replace(context.signal_domain, signal_domain_id="signal_domain_other")
    with pytest.raises(ValueError, match="is not the one of model run"):
        predict_with_model_run(**_predict_kwargs(
            context, spectral_data, dataset, calibration, _SumPredictor(),
            signal_domain=foreign))


def test_input_shorter_than_the_dataset_fails_and_leaves_no_run(tmp_path, qapp):
    X = np.full((3, 2, 3), 10.0, dtype=np.float32)
    context, spectral_data, dataset, calibration = _second_review_setup(tmp_path, "short", X)
    short = SpectralData(X[:2], spectral_data.wavelengths, spectral_data.meta)
    with pytest.raises(ValueError, match="produced 4 spectra, expected 6"):
        predict_with_model_run(**_predict_kwargs(
            context, short, dataset, calibration, _SumPredictor(),
            prediction_run_id="prediction_short", persist_artifacts=True))
    from enginir.io.project_store import has_prediction_run

    assert not has_prediction_run(context.project_path, "prediction_short")


@pytest.mark.parametrize(
    ("name", "frames"),
    [("cube.eiff", slice(None, None, -2)), ("cube.eiff", slice(1, None, 2)),
     ("cube.h5", slice(None, None, -3)), ("cube.hdr", None)],
)
def test_image_files_stream_into_the_same_prediction_as_in_memory_data(tmp_path, qapp,
                                                                       monkeypatch, name,
                                                                       frames):
    """G16: EIFF, HDF5 (any frame step) and ENVI are streamed, never read completely."""
    from enginir.io import load_spectra, save_envi, save_hdf5

    from .test_io import _build_eiff_file

    project_path = str(tmp_path / "stream.h5")
    _create_phase6_model_project(project_path)
    context = _fit_phase6_model(project_path).build_prediction_job_context(
        model_run_id="model_run_1")
    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    cube = np.random.default_rng(3).integers(10, 190, size=(7, 3, 3)).astype(np.uint16)
    path = tmp_path / name
    if name.endswith(".eiff"):
        path.write_bytes(_build_eiff_file(3, 3, list(cube), wavelengths))
    elif name.endswith(".h5"):
        save_hdf5(path, cube, wavelengths)
    else:
        save_envi(path, cube, wavelengths)
    loader_kwargs = {} if frames is None else {"frames": frames}
    selected = load_spectra(path, **loader_kwargs)
    dataset = build_dataset_ref_from_spectral_data(data_path=str(path), spectral_data=selected,
                                                   source_modality="nir_hsi")
    calibration = _scalar_reflectance_calibration(dataset.dataset_id)
    signal_domain = dataclasses.replace(context.signal_domain,
                                        input_format=dataset.source_format)

    def predict(spectral_data, run_id):
        return predict_with_model_run(
            project_path=context.project_path, model_run=context.model_run,
            signal_domain=signal_domain, recipe=context.recipe, dataset=dataset,
            calibration=calibration, model_registry={}, spectral_data=spectral_data,
            loader_kwargs=loader_kwargs,
            preprocessing_recipe_id=context.preprocessing_recipe_id,
            prediction_run_id=run_id, chunk_size=2, persist_artifacts=False,
            return_arrays=True,
        )

    in_memory = predict(selected, "in_memory")
    monkeypatch.setattr("enginir.prediction._service.load_spectra",
                        lambda *args, **kwargs: pytest.fail("the image was read completely"))
    streamed = predict(None, "streamed")

    np.testing.assert_allclose(streamed.y_pred, in_memory.y_pred, rtol=1e-6)
    assert streamed.spatial_map.shape == selected.X.shape[:2]
