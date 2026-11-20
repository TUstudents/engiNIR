"""Tests for the v3 prediction workflow."""

from __future__ import annotations


import dataclasses

import h5py
import numpy as np
import pytest

from tests._tables import check_all
from unittest.mock import patch

from enginir.io import SpectralData, load_hdf5, save_hdf5
from enginir.io.project_store import (
    load_model_run_prediction_artifact,
    load_project_store,
    save_prediction_run,
)
from enginir.prediction import (
    PredictionCancelledError,
    PredictionProgressEvent,
    build_dataset_ref_from_spectral_data,
    predict_with_model_run,
)
from enginir.preprocessing import PreprocessingRecipe
from enginir.project import (
    ReferenceCorrection,
    PredictionRun,
    transform_spectra_to_signal_domain,
)
from tests._sessions import CoreSession

from tests._eiff import eiff_bytes
from tests._project_payloads import write_model_project, write_prediction_cube
from tests import _project_payloads as build


def _fit_phase6_model(project_path: str) -> CoreSession:
    state = CoreSession()
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


def test_prediction_dataset_source_path_accepts_equivalent_slash_styles(tmp_path):
    source_path = tmp_path / "prediction_source.h5"
    spectral_data = SpectralData(
        X=np.empty((0, 2, 3), dtype=np.uint16),
        wavelengths=np.arange(3, dtype=np.float32),
        meta={"frames_total": 4, "source_modality": "nir_hsi"},
    )
    dataset = build.dataset(
        np.arange(3, dtype=np.float32),
        dataset_id="dataset_slashes",
        source_path=str(source_path).replace("\\", "/"),
        source_format="hdf5",
        cube_shape=(4, 2, 3),
        n_bands=3,
    )

    from enginir._workflow import resolve_dataset

    resolved, is_new = resolve_dataset({dataset.dataset_id: dataset},
                                       str(source_path).replace("/", "\\"), spectral_data,
                                       dataset_id=dataset.dataset_id)

    assert resolved is dataset and not is_new


def _fit_phase6_model_with_recipe(
    project_path: str, *, recipe_id: str, model_run_id: str
) -> CoreSession:
    state = CoreSession()
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


def test_predict_dataset_reuses_dataset_local_calibration_defaults_after_reopen(tmp_path):
    project_path = str(tmp_path / "phase6_predict_defaults.h5")
    write_model_project(project_path)
    state = _fit_phase6_model(project_path)

    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    pred_path = str(tmp_path / "prediction_defaults.h5")
    write_prediction_cube(pred_path, wavelengths=wavelengths)
    state.register_dataset(
        build.dataset(
            wavelengths,
            dataset_id="dataset_pred",
            source_path=pred_path,
            source_format="hdf5",
            cube_shape=(2, 2, 3),
            n_bands=3,
        )
    )
    state.set_active_reference_correction(
        build.reference_correction(
            reference_correction_id="cal_pred",
            dataset_id="dataset_pred",
            mode="dark_white_reference",
            black_level=0.0,
            white_level=100.0,
            output_signal_unit="Absorbance",
        )
    )

    in_memory_result = state.predict_dataset(
        model_run_id="model_run_1",
        data_path=pred_path,
        prediction_run_id="prediction_run_in_memory",
    )

    reopened = CoreSession()
    reopened.open_project(project_path)
    result = reopened.predict_dataset(
        model_run_id="model_run_1",
        data_path=pred_path,
        prediction_run_id="prediction_run_1",
    )

    assert result.dataset.dataset_id == "dataset_pred"
    assert result.reference_correction.reference_correction_id == "cal_pred"
    assert result.prediction_run.preprocessing_recipe_id == "recipe_none"
    assert result.prediction_run.artifact_refs["prediction_shape"] == (4,)
    assert result.prediction_run.artifact_refs["spatial_map_shape"] == (2, 2)
    assert reopened.predictions["prediction_run_1"] is result, "kept for the session"
    np.testing.assert_allclose(result.y_pred, in_memory_result.y_pred, rtol=1e-5)
    np.testing.assert_allclose(result.spatial_map, in_memory_result.spatial_map, rtol=1e-5)
    assert load_project_store(project_path)["predictions"] == {}, "nothing is stored"


def test_reopened_prediction_run_lazily_loads_spatial_map_and_preserves_model_linkage(tmp_path):
    project_path = str(tmp_path / "phase6_prediction_reopen_lazy.h5")
    write_model_project(project_path)
    state = _fit_phase6_model(project_path)
    del state
    # Low-level persistence (a script's explicit request); the viewer never stores predictions.
    run = PredictionRun(
        prediction_run_id="prediction_run_1",
        dataset_id="dataset_train",
        reference_correction_id="cal_train",
        model_run_id="model_run_1",
        signal_domain_id="signal_domain_1",
        preprocessing_recipe_id="recipe_none",
        artifact_refs={
            "y_pred": np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
            "spatial_map": np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
        },
        created_at="2026-04-02T10:05:30+00:00",
    )
    payload = run.to_dict()
    del payload["prediction_run_id"]
    save_prediction_run(project_path, "prediction_run_1", payload)

    from enginir import Project

    reopened = Project.open(project_path)
    prediction = reopened.predictions["prediction_run_1"]
    assert prediction.record.model_run_id == "model_run_1"
    assert reopened.workspace.model_runs[prediction.record.model_run_id].model_type == "PLS"
    assert "y_pred" not in prediction.record.artifact_refs
    assert "spatial_map" not in prediction.record.artifact_refs
    np.testing.assert_allclose(prediction.y_pred,
                               np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32))
    np.testing.assert_allclose(prediction.map,
                               np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32))


def test_predict_dataset_requires_explicit_calibration_when_defaults_absent(tmp_path):
    project_path = str(tmp_path / "phase6_predict_no_defaults.h5")
    write_model_project(project_path)
    state = _fit_phase6_model(project_path)

    pred_path = str(tmp_path / "prediction_no_defaults.h5")
    write_prediction_cube(
        pred_path, wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    )

    with pytest.raises(ValueError, match="explicit reference correction required"):
        state.predict_dataset(
            model_run_id="model_run_1",
            data_path=pred_path,
            prediction_run_id="prediction_run_missing_cal",
        )


def test_predict_dataset_allows_different_wavelength_metadata(tmp_path):
    project_path = str(tmp_path / "phase6_predict_different_wavelengths.h5")
    write_model_project(project_path)
    state = _fit_phase6_model(project_path)

    pred_path = str(tmp_path / "prediction_different_wavelengths.h5")
    write_prediction_cube(
        pred_path, wavelengths=np.array([1000.0, 1110.0, 1200.0], dtype=np.float32)
    )

    result = state.predict_dataset(
        model_run_id="model_run_1",
        data_path=pred_path,
        reference_correction=build.reference_correction(
            reference_correction_id="cal_different_wl",
            dataset_id="",
            mode="dark_white_reference",
            black_level=0.0,
            white_level=100.0,
            output_signal_unit="Absorbance",
        ),
        prediction_run_id="prediction_run_different_wl",
    )

    assert result.prediction_run.prediction_run_id == "prediction_run_different_wl"


def _msc_prediction_calibration(reference_correction_id: str) -> ReferenceCorrection:
    return build.reference_correction(
        reference_correction_id=reference_correction_id,
        dataset_id="",
        mode="dark_white_reference",
        black_level=0.0,
        white_level=100.0,
        output_signal_unit="Absorbance",
    )


def test_reopened_stateful_preprocessing_replays_persisted_training_state(tmp_path):
    project_path = str(tmp_path / "phase6_predict_reopened_msc.h5")
    write_model_project(project_path)
    fitted = _fit_phase6_model_with_recipe(
        project_path, recipe_id="recipe_msc", model_run_id="model_run_msc"
    )

    in_memory_path = str(tmp_path / "prediction_in_memory_msc.h5")
    reopened_path = str(tmp_path / "prediction_reopened_msc.h5")
    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    write_prediction_cube(in_memory_path, wavelengths=wavelengths)
    write_prediction_cube(reopened_path, wavelengths=wavelengths)
    # the fitted pipeline itself (an in-memory predictor), not the saved model
    context = fitted.build_prediction_job_context(model_run_id="model_run_msc")
    data = load_hdf5(in_memory_path)
    dataset = build_dataset_ref_from_spectral_data(data_path=in_memory_path, spectral_data=data,
                                                   dataset_id="dataset_msc_in_memory")
    reference_correction = dataclasses.replace(_msc_prediction_calibration("cal_msc_in_memory"),
                                      dataset_id=dataset.dataset_id)
    in_memory = predict_with_model_run(
        project_path=project_path, model_run=context.model_run,
        signal_domain=context.signal_domain, recipe=context.recipe, dataset=dataset,
        reference_correction=reference_correction, model_registry=context.model_registry, spectral_data=data,
        preprocessing_recipe_id=context.preprocessing_recipe_id)
    assert in_memory.dmodx is None, "an in-memory predictor alone gives no diagnostics"

    reopened = CoreSession()
    reopened.open_project(project_path)
    replayed = reopened.predict_dataset(
        model_run_id="model_run_msc",
        data_path=reopened_path,
        reference_correction=_msc_prediction_calibration("cal_msc_reopened"),
        prediction_run_id="prediction_run_msc_reopened",
    )

    np.testing.assert_allclose(replayed.y_pred, in_memory.y_pred, rtol=1e-5, atol=1e-6)
    saved_model = load_model_run_prediction_artifact(project_path, "model_run_msc")
    np.testing.assert_array_equal(
        saved_model["preprocessing_state"]["msc"]["reference"],
        fitted.model_registry["model_run_msc"].named_steps["msc"].reference_,
    )


def test_a_model_without_its_msc_state_is_refused_when_the_project_is_read(tmp_path):
    # The MSC/EMSC state is core (project-schema contract, Core and display
    # data): the reader refuses the project, before any prediction.
    project_path = str(tmp_path / "phase6_missing_msc_state.h5")
    write_model_project(project_path)
    _fit_phase6_model_with_recipe(
        project_path, recipe_id="recipe_msc", model_run_id="model_run_msc"
    )
    with h5py.File(project_path, "a") as handle:
        del handle["model_runs/model_run_msc/artifact_refs/prediction_model/preprocessing_state"]

    with pytest.raises(ValueError, match="no preprocessing_state"):
        CoreSession().open_project(project_path)


@pytest.mark.parametrize(
    ("damage", "match"),
    [("wrong_length", "has 4 bands, the recipe needs 3")],
)
def test_reopened_stateful_preprocessing_rejects_malformed_training_state(
    tmp_path, damage, match
):
    project_path = str(tmp_path / "phase6_predict_missing_msc_state.h5")
    write_model_project(project_path)
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
    write_prediction_cube(
        pred_path, wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    )
    reopened = CoreSession()
    reopened.open_project(project_path)
    datasets_before = set(reopened.datasets)
    calibrations_before = set(reopened.reference_corrections)
    with pytest.raises(ValueError, match=match):
        reopened.predict_dataset(
            model_run_id="model_run_msc",
            data_path=pred_path,
            reference_correction=_msc_prediction_calibration("cal_msc_missing_state"),
            prediction_run_id="prediction_run_msc",
        )
    reopened.open_project(project_path)
    assert set(reopened.datasets) == datasets_before
    assert set(reopened.reference_corrections) == calibrations_before
    assert "prediction_run_msc" not in reopened.predictions


def test_predict_dataset_ignores_changed_wavelength_metadata_for_existing_dataset(tmp_path):
    project_path = str(tmp_path / "phase6_predict_changed_wavelength_metadata.h5")
    write_model_project(project_path)
    state = _fit_phase6_model(project_path)

    pred_path = str(tmp_path / "prediction_stale.h5")
    write_prediction_cube(
        pred_path, wavelengths=np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    )
    state.register_dataset(
        build.dataset(
            np.array([1000.0, 1100.0, 1200.0], dtype=np.float32),
            dataset_id="dataset_stale",
            source_path=pred_path,
            source_format="hdf5",
            cube_shape=(2, 2, 3),
            n_bands=3,
        )
    )
    state.set_active_reference_correction(
        build.reference_correction(
            reference_correction_id="cal_stale",
            dataset_id="dataset_stale",
            mode="dark_white_reference",
            black_level=0.0,
            white_level=100.0,
            output_signal_unit="Absorbance",
        )
    )
    write_prediction_cube(
        pred_path, wavelengths=np.array([1000.0, 1110.0, 1200.0], dtype=np.float32)
    )

    result = state.predict_dataset(
        model_run_id="model_run_1",
        data_path=pred_path,
        prediction_run_id="prediction_run_stale_dataset",
    )

    assert result.prediction_run.prediction_run_id == "prediction_run_stale_dataset"


def test_hdf5_prediction_reuses_source_and_artifact_file_handles(tmp_path):
    project_path = str(tmp_path / "prediction_handle_reuse_project.h5")
    source_path = str(tmp_path / "prediction_handle_reuse_source.h5")
    write_model_project(project_path)
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
    reference_correction = build.reference_correction(
        reference_correction_id="cal_handle_reuse",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        black_level=0.0,
        white_level=100.0,
        output_signal_unit="Reflectance",
    )

    class _DummyPredictor:
        def predict(self, X):
            return np.sum(X, axis=1, dtype=np.float32)

    def opens(chunk_size: int) -> tuple[int, int]:
        """How often the project and the source file are opened by one persisted prediction."""
        with patch("enginir.io.project_store.h5py.File", wraps=h5py.File) as mocked_file:
            predict_with_model_run(
                project_path=project_path,
                model_run=context.model_run,
                signal_domain=context.signal_domain,
                recipe=context.recipe,
                dataset=dataset,
                reference_correction=reference_correction,
                model_registry={context.model_run.model_run_id: _DummyPredictor()},
                data_path=source_path,
                spectral_data=metadata,
                preprocessing_recipe_id=context.preprocessing_recipe_id,
                prediction_run_id=f"prediction_handle_reuse_{chunk_size}",
                chunk_size=chunk_size,
                persist_artifacts=True,
                return_arrays=False,
            )
        paths = [str(call.args[0]) for call in mocked_file.call_args_list]
        return paths.count(project_path), paths.count(source_path)

    # Three chunks of one frame against one chunk: the handles are reused, so the number of
    # opens does not grow with the chunks (the source is opened once either way).
    assert opens(chunk_size=1) == opens(chunk_size=3)
    assert opens(chunk_size=2)[1] == 1


def test_predict_with_model_run_cancels_between_chunks_and_discards_artifacts(tmp_path):
    project_path = str(tmp_path / "phase6_predict_cancel.h5")
    write_model_project(project_path)
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
    reference_correction = build.reference_correction(
        reference_correction_id="cal_cancel",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        black_level=0.0,
        white_level=20.0,
        output_signal_unit="Reflectance",
    )

    class _DummyPredictor:
        def predict(self, X):
            return np.sum(X, axis=1, dtype=np.float32)

    events: list[PredictionProgressEvent] = []

    def cancel_after_first_chunk() -> bool:
        return any(event.phase == "predicting" and event.completed >= 1 for event in events)

    with pytest.raises(PredictionCancelledError, match="canceled"):
        predict_with_model_run(
            project_path=context.project_path,
            model_run=context.model_run,
            signal_domain=context.signal_domain,
            recipe=context.recipe,
            dataset=dataset,
            reference_correction=reference_correction,
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

    reopened = CoreSession()
    reopened.open_project(project_path)
    assert "prediction_run_cancelled" not in reopened.predictions


def test_predict_with_model_run_rejects_bare_predictor_when_recipe_would_be_skipped(
    tmp_path
):
    project_path = str(tmp_path / "phase6_predict_bare_estimator.h5")
    write_model_project(project_path)
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
    reference_correction = build.reference_correction(
        reference_correction_id="cal_bare",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        black_level=0.0,
        white_level=2.0,
        output_signal_unit="Reflectance",
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
            reference_correction=reference_correction,
            model_registry={context.model_run.model_run_id: _BareEstimator()},
            spectral_data=spectral_data,
            preprocessing_recipe_id="recipe_snv",
        )


def test_explicit_prediction_calibration_reuses_matching_dataset_calibration():
    dataset = build.dataset(
        np.array([1000.0, 1100.0, 1200.0], dtype=np.float32),
        dataset_id="dataset_reuse",
        source_path="/tmp/predict.h5",
        source_format="hdf5",
        cube_shape=(1, 2, 3),
        n_bands=3,
    )
    existing = build.reference_correction(
        reference_correction_id="cal_existing",
        dataset_id=dataset.dataset_id,
        mode="dark_white_reference",
        black_level=0.0,
        white_level=100.0,
        output_signal_unit="Reflectance",
    )
    explicit = build.reference_correction(
        reference_correction_id="",
        dataset_id="",
        mode="dark_white_reference",
        black_level=0.0,
        white_level=100.0,
        output_signal_unit="Reflectance",
    )

    from enginir._workflow import resolve_prediction_reference_correction

    resolved, is_new = resolve_prediction_reference_correction(
        dataset, {existing.reference_correction_id: existing}, {}, explicit)

    assert resolved is existing
    assert is_new is False


def test_transform_spectra_to_signal_domain_rejects_invalid_raw_spectra():
    def check(case):
        raw_spectra, match = case
        wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
        dataset = build.dataset(
            wavelengths,
            dataset_id="dataset_invalid_raw",
            source_path="/data/prediction.h5",
            source_format="hdf5",
            cube_shape=(1, 1, 3),
            n_bands=3,
        )
        reference_correction = build.reference_correction(
            reference_correction_id="cal_invalid_raw",
            dataset_id=dataset.dataset_id,
        )
        signal_domain = build.signal_domain(
            wavelengths,
            signal_domain_id="signal_domain_invalid_raw",
            input_format="hdf5",
            corrected_input_allowed=False,
            n_bands=3,
        )

        with pytest.raises(ValueError, match=match):
            transform_spectra_to_signal_domain(
                raw_spectra,
                dataset=dataset,
                reference_correction=reference_correction,
                signal_domain=signal_domain,
            )

    check_all([
        (np.ones((3,), dtype=np.float32), "2-D"),
        (np.empty((0, 3), dtype=np.float32), "at least one"),
        (np.array([[1.0, np.nan, 3.0]], dtype=np.float32), "finite"),
        (np.array([["bad", "data", "x"]], dtype=object), "numeric"),
    ], check)


def _scalar_reflectance_calibration(dataset_id: str) -> ReferenceCorrection:
    return build.reference_correction(
        reference_correction_id="cal_scalar",
        dataset_id=dataset_id,
        mode="dark_white_reference",
        black_level=0.0,
        white_level=200.0,
        output_signal_unit="Reflectance",
    )


def _second_review_setup(tmp_path, name: str, X: np.ndarray, **calibration_kwargs):
    project_path = str(tmp_path / f"{name}.h5")
    write_model_project(project_path)
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
    reference_correction = ReferenceCorrection(
        reference_correction_id=f"cal_{name}", dataset_id=dataset.dataset_id,
        mode="dark_white_reference", dark_ref_source=None, white_ref_source=None,
        output_signal_unit="Reflectance", created_at="2026-04-02T11:09:30+00:00", **fields,
    )
    return context, spectral_data, dataset, reference_correction


class _SumPredictor:
    def predict(self, X):
        return np.sum(X, axis=1, dtype=np.float32)


def test_prediction_never_replaces_an_existing_run(tmp_path):
    X = np.ones((2, 2, 3), dtype=np.float32) * 10.0
    context, spectral_data, dataset, reference_correction = _second_review_setup(tmp_path, "reuse", X)
    kwargs = dict(
        project_path=context.project_path, model_run=context.model_run,
        signal_domain=context.signal_domain, recipe=context.recipe, dataset=dataset,
        reference_correction=reference_correction, model_registry={context.model_run.model_run_id: _SumPredictor()},
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


def test_preflight_rejects_a_pipeline_without_the_recipe_steps(tmp_path):
    from types import SimpleNamespace

    X = np.ones((2, 2, 3), dtype=np.float32)
    context, spectral_data, dataset, reference_correction = _second_review_setup(tmp_path, "steps", X)
    pipeline = SimpleNamespace(named_steps={"model": _SumPredictor()},
                               predict=_SumPredictor().predict)

    with pytest.raises(ValueError, match=r"lacks the recipe step\(s\) \['snv'\]"):
        predict_with_model_run(
            project_path=context.project_path, model_run=context.model_run,
            signal_domain=context.signal_domain, recipe=PreprocessingRecipe(snv=True),
            dataset=dataset, reference_correction=reference_correction,
            model_registry={context.model_run.model_run_id: pipeline},
            spectral_data=spectral_data, preprocessing_recipe_id="recipe_snv",
        )


def test_progress_text_calls_the_reference_correction_by_its_name(tmp_path):
    # Terminology contract: "calibration" means building a model, never the
    # reference correction; the phase id `correcting` is internal.
    from enginir.viewer._prediction import _phase_label

    X = np.ones((2, 2, 3), dtype=np.float32) * 10.0
    context, spectral_data, dataset, reference_correction = _second_review_setup(tmp_path, "words", X)
    events: list[PredictionProgressEvent] = []
    predict_with_model_run(
        project_path=context.project_path, model_run=context.model_run,
        signal_domain=context.signal_domain, recipe=context.recipe, dataset=dataset,
        reference_correction=reference_correction, model_registry={context.model_run.model_run_id: _SumPredictor()},
        spectral_data=spectral_data, preprocessing_recipe_id=context.preprocessing_recipe_id,
        chunk_size=1, progress_callback=events.append,
    )

    shown = [f"{_phase_label(event.phase)} {event.message}" for event in events]
    assert not [text for text in shown if "calibrat" in text.lower()]
    assert any(text.startswith("Reference correction") for text in shown)


def test_flat_spectra_with_per_sensor_references_chunk_like_one_shot(tmp_path):
    cube = np.array([[[20, 30, 40], [60, 70, 80]], [[30, 40, 50], [70, 80, 90]],
                     [[25, 35, 45], [65, 75, 85]]], dtype=np.uint16)
    dark = np.array([[0, 10, 20], [10, 20, 30]], dtype=np.float32)
    context, spectral_data, dataset, reference_correction = _second_review_setup(
        tmp_path, "flat_refs", cube, dark_ref_data=dark, white_ref_data=dark + 100.0
    )
    flat = SpectralData(X=cube.reshape(6, 3), wavelengths=spectral_data.wavelengths,
                        meta=spectral_data.meta)

    def predict(data, chunk_size):
        return predict_with_model_run(
            project_path=context.project_path, model_run=context.model_run,
            signal_domain=context.signal_domain, recipe=context.recipe, dataset=dataset,
            reference_correction=reference_correction,
            model_registry={context.model_run.model_run_id: _SumPredictor()},
            spectral_data=data, preprocessing_recipe_id=context.preprocessing_recipe_id,
            chunk_size=chunk_size,
        ).y_pred

    np.testing.assert_array_equal(predict(flat, 1), predict(spectral_data, 3))


def _predict_kwargs(context, spectral_data, dataset, reference_correction, predictor, **overrides):
    return {"project_path": context.project_path, "model_run": context.model_run,
            "signal_domain": context.signal_domain, "recipe": context.recipe, "dataset": dataset,
            "reference_correction": reference_correction,
            "model_registry": {context.model_run.model_run_id: predictor},
            "spectral_data": spectral_data,
            "preprocessing_recipe_id": context.preprocessing_recipe_id, **overrides}


def test_multi_output_predictions_store_their_output_axis(tmp_path):
    from enginir.io.project_store import load_prediction_run_artifact_refs

    class _TwoOutputs:
        def predict(self, X):
            return np.column_stack([X.sum(axis=1), X.mean(axis=1)]).astype(np.float32)

    X = np.full((3, 2, 3), 10.0, dtype=np.float32)
    context, spectral_data, dataset, reference_correction = _second_review_setup(tmp_path, "multi", X)
    result = predict_with_model_run(**_predict_kwargs(
        context, spectral_data, dataset, reference_correction, _TwoOutputs(),
        prediction_run_id="prediction_multi", persist_artifacts=True))

    assert result.spatial_map.shape == (3, 2, 2)
    stored = load_prediction_run_artifact_refs(context.project_path, "prediction_multi")
    assert int(stored["n_outputs"]) == 2
    assert list(np.ravel(stored["output_axis"])) == [2]
    assert "output_index" not in stored  # no output is chosen for the user


def test_preflight_rejects_a_signal_domain_of_another_model_run(tmp_path):
    from dataclasses import replace

    X = np.full((2, 2, 3), 10.0, dtype=np.float32)
    context, spectral_data, dataset, reference_correction = _second_review_setup(tmp_path, "foreign", X)
    foreign = replace(context.signal_domain, signal_domain_id="signal_domain_other")
    with pytest.raises(ValueError, match="is not the one of model"):
        predict_with_model_run(**_predict_kwargs(
            context, spectral_data, dataset, reference_correction, _SumPredictor(),
            signal_domain=foreign))


def test_input_shorter_than_the_dataset_fails_and_leaves_no_run(tmp_path):
    X = np.full((3, 2, 3), 10.0, dtype=np.float32)
    context, spectral_data, dataset, reference_correction = _second_review_setup(tmp_path, "short", X)
    short = SpectralData(X[:2], spectral_data.wavelengths, spectral_data.meta)
    with pytest.raises(ValueError, match="produced 4 spectra, expected 6"):
        predict_with_model_run(**_predict_kwargs(
            context, short, dataset, reference_correction, _SumPredictor(),
            prediction_run_id="prediction_short", persist_artifacts=True))
    from enginir.io.project_store import has_prediction_run

    assert not has_prediction_run(context.project_path, "prediction_short")


@pytest.mark.parametrize(
    ("name", "frames"),
    [("cube.eiff", slice(None, None, -2)), ("cube.eiff", slice(1, None, 2)),
     ("cube.h5", slice(None, None, -3)), ("cube.hdr", None)],
)
def test_image_files_stream_into_the_same_prediction_as_in_memory_data(tmp_path, monkeypatch, name,
                                                                       frames):
    """G16: EIFF, HDF5 (any frame step) and ENVI are streamed, never read completely."""
    from enginir.io import load_spectra, save_envi, save_hdf5

    project_path = str(tmp_path / "stream.h5")
    write_model_project(project_path)
    context = _fit_phase6_model(project_path).build_prediction_job_context(
        model_run_id="model_run_1")
    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    cube = np.random.default_rng(3).integers(10, 190, size=(7, 3, 3)).astype(np.uint16)
    path = tmp_path / name
    if name.endswith(".eiff"):
        path.write_bytes(eiff_bytes(3, 3, list(cube), wavelengths))
    elif name.endswith(".h5"):
        save_hdf5(path, cube, wavelengths)
    else:
        save_envi(path, cube, wavelengths)
    loader_kwargs = {} if frames is None else {"frames": frames}
    selected = load_spectra(path, **loader_kwargs)
    dataset = build_dataset_ref_from_spectral_data(data_path=str(path), spectral_data=selected,
                                                   source_modality="nir_hsi")
    reference_correction = _scalar_reflectance_calibration(dataset.dataset_id)
    signal_domain = dataclasses.replace(context.signal_domain,
                                        input_format=dataset.source_format)

    def predict(spectral_data, run_id):
        return predict_with_model_run(
            project_path=context.project_path, model_run=context.model_run,
            signal_domain=signal_domain, recipe=context.recipe, dataset=dataset,
            reference_correction=reference_correction, model_registry={}, spectral_data=spectral_data,
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
