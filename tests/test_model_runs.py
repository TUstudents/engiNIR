"""Fitting a model on a calibration set: the grouped pipeline, stored ROI spectra and ROI scores (sample-set-model-run contract)."""

from __future__ import annotations

from unittest.mock import patch
import h5py
from loguru import logger
import numpy as np
import pytest
from enginir import PreprocessingRecipe, Project
from enginir.io import save_hdf5
from enginir.io.project_store import (
    create_project,
    load_model_run_prediction_artifact,
    load_project_store,
    load_roi_band_images,
    save_reference_correction,
    save_dataset_ref,
    save_preprocessing_recipe,
    save_signal_domain_contract,
    save_target_assignment,
    save_target_definition,
    save_training_roi,
)
from enginir.modeling import build_sample_set, fit_model_run, roi_spectrum_summaries
from enginir.modeling._service import _StatelessRecipePredictor
from enginir.models import PLSRegression
from enginir.preprocessing._pipeline import (
    fitted_preprocessing_state,
    make_model_pipeline,
    restore_fitted_preprocessing_steps,
    transform_preprocessing_steps,
)
from enginir.validation import RoundRobinGroups
from enginir.viewer._modeling import _roi_score_payload
from enginir.viewer._state import ProjectState
from tests._project_payloads import recipe_payload, roi_provenance
from tests._scans import EXCLUSION, project_with_an_odd_pixel


# -- Tests for the v3 grouped model-run pipeline ---------------------------------------


def _create_model_project(path: str, *, seed_rois: bool = True) -> None:
    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    create_project(path)
    save_dataset_ref(
        path,
        "dataset_1",
        {
            "source_path": "/data/train.eiff",
            "source_format": "eiff",
            "source_modality": "nir_hsi",
            "cube_shape": (4, 4, 3),
            "n_bands": 3,
            "wavelengths": wavelengths,
            "loader_metadata": {"instrument": "alpha"},
            "import_timestamp": "2026-04-02T10:00:00+00:00",
        },
    )
    save_reference_correction(
        path,
        "dataset_1",
        "cal_1",
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
        recipe_payload(sg_edges="interp"),
    )
    save_signal_domain_contract(
        path,
        "signal_domain_1",
        {
            "input_format": "eiff",
            "source_modality": "nir_hsi",
            "required_signal_unit": "Absorbance",
            "required_correction_mode": "dark_white_reference",
            "requires_dark_ref": True,
            "requires_white_ref": True,
            "corrected_input_allowed": False,
            "n_bands": 3,
            "wavelengths": wavelengths,
            "wavelength_tolerance_nm": 0.0,
            "band_mapping_policy": "exact",
            "output_dtype": "float32",
            "created_at": "2026-04-02T10:04:00+00:00",
        },
    )

    if not seed_rois:
        return

    roi_payloads = {
        "roi_1": (np.array([[0, 0, 0], [0, 1, 0]], dtype=np.uint16), 1.0),
        "roi_2": (np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint16), 2.0),
        "roi_3": (np.array([[2, 0, 0], [2, 1, 0]], dtype=np.uint16), 3.0),
        "roi_4": (np.array([[3, 0, 0], [3, 1, 0]], dtype=np.uint16), 4.0),
    }
    for index, (roi_id, (spectra, value)) in enumerate(roi_payloads.items(), start=1):
        save_training_roi(
            path,
            roi_id,
            {
                "dataset_id": "dataset_1",
                "reference_correction_id": "cal_1",
                "name": roi_id.upper(),
                "source_file": "/data/train.eiff",
                "job": "calibration",
                "n_pixels": int(spectra.shape[0]),
                "created_at": f"2026-04-02T10:0{4 + index}:00+00:00",
                "spatial_bounds": (index, index, 0, 1),
                "wavelengths": wavelengths,
                "provenance": roi_provenance(
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


def test_project_state_fit_model_run_persists_grouped_model_run(tmp_path, qapp):
    project_path = str(tmp_path / "phase5_model_run.h5")
    _create_model_project(project_path)

    state = ProjectState()
    state.open_project(project_path)

    sample_set_data = state.build_sample_set(
        target_id="target_moisture",
        preprocessing_recipe_id="recipe_none",
        sample_set_id="sample_set_phase5",
    )
    result = state.fit_model_run(
        sample_set_id=sample_set_data.sample_set.sample_set_id,
        signal_domain_id="signal_domain_1",
        model_type="PLS",
        hyperparameters={"n_components": 1},
        cv="logo",
        model_run_id="model_run_phase5",
    )

    assert result.model_run.model_run_id == "model_run_phase5"
    assert result.model_run.model_type == "PLS"
    assert result.model_run.cv_spec["type"] == "LeaveOneGroupOut"
    assert result.model_run.metrics["primary_reporting_level"] == "roi"
    assert result.model_run.metrics["primary_roi_aggregation"] == "mean"
    assert result.roi_ids.shape == (4,)
    assert result.pixel_y_pred.shape == (8,)
    assert state.active_model_name == "model_run_phase5"
    assert "model_run_phase5" in state.model_registry
    assert "model_run_phase5" in state.model_runs

    reloaded = ProjectState()
    reloaded.open_project(project_path)
    assert "sample_set_phase5" in reloaded.sample_sets
    assert "model_run_phase5" in reloaded.model_runs
    assert reloaded.model_runs["model_run_phase5"].sample_set_id == "sample_set_phase5"
    assert reloaded.sample_sets["sample_set_phase5"].target_id == "target_moisture"
    assert reloaded.targets["target_moisture"].name == "moisture"
    assert reloaded.model_runs["model_run_phase5"].model_type == "PLS"
    assert reloaded.model_runs["model_run_phase5"].metrics["rmsecv"] == pytest.approx(
        result.model_run.metrics["rmsecv"]
    )
    assert reloaded.sample_sets["sample_set_phase5"].roi_ids == ["roi_1", "roi_2", "roi_3", "roi_4"]
    assert "fitted_model" not in reloaded.model_runs["model_run_phase5"].artifact_refs
    metadata = reloaded.model_runs["model_run_phase5"].artifact_refs.get("prediction_model")
    assert metadata["model_type"] == "PLSRegression"
    assert not {"coef_", "x_mean", "rotations", "loadings"} & set(metadata), "arrays not hydrated"
    assert "roi_ids" not in reloaded.model_runs["model_run_phase5"].artifact_refs.get(
        "cv_outputs", {}
    )
    np.testing.assert_array_equal(
        reloaded.ensure_model_run_artifact_refs("model_run_phase5")["cv_outputs"]["roi_ids"],
        ["roi_1", "roi_2", "roi_3", "roi_4"],
    )

    saved_model = load_model_run_prediction_artifact(project_path, "model_run_phase5")
    assert saved_model["model_type"] == "PLSRegression"

    store = load_project_store(project_path)
    assert store["model_runs"]["model_run_phase5"]["cv_spec"]["type"] == "LeaveOneGroupOut"


def test_project_state_fit_model_run_uses_roi_local_reference_snapshots(tmp_path, qapp):
    project_path = str(tmp_path / "phase5_model_run_roi_refs.h5")
    _create_model_project(project_path, seed_rois=False)
    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    full_dark = np.array(
        [[0, 10, 20], [10, 20, 30], [20, 30, 40], [30, 40, 50]],
        dtype=np.float32,
    )
    full_white = full_dark + 100.0
    save_reference_correction(
        project_path,
        "dataset_1",
        "cal_roi_refs",
        {
            "mode": "dark_white_reference",
            "output_signal_unit": "Absorbance",
            "dark_ref_data": full_dark,
            "white_ref_data": full_white,
            "created_at": "2026-04-02T10:01:00+00:00",
        },
    )
    roi_payloads = {
        "roi_1": (np.array([[30, 40, 50], [40, 50, 60]], dtype=np.uint16), 1.0),
        "roi_2": (np.array([[35, 45, 55], [45, 55, 65]], dtype=np.uint16), 2.0),
        "roi_3": (np.array([[40, 50, 60], [50, 60, 70]], dtype=np.uint16), 3.0),
        "roi_4": (np.array([[45, 55, 65], [55, 65, 75]], dtype=np.uint16), 4.0),
    }
    roi_dark = full_dark[0:2]
    roi_white = full_white[0:2]
    for index, (roi_id, (spectra, value)) in enumerate(roi_payloads.items(), start=1):
        save_training_roi(
            project_path,
            roi_id,
            {
                "dataset_id": "dataset_1",
                "reference_correction_id": "cal_roi_refs",
                "name": roi_id.upper(),
                "source_file": "/data/train.eiff",
                "job": "calibration",
                "n_pixels": int(spectra.shape[0]),
                "created_at": f"2026-04-02T10:0{4 + index}:00+00:00",
                "spatial_bounds": (index, index, 0, 1),
                "wavelengths": wavelengths,
                "provenance": roi_provenance(
                    reference_correction_id="cal_roi_refs",
                    created_at=f"2026-04-02T10:0{4 + index}:00+00:00"
                ),
            },
            spectra,
            dark_ref_spectra=roi_dark,
            white_ref_spectra=roi_white,
        )
        save_target_assignment(
            project_path,
            "target_moisture",
            roi_id,
            {"value": value, "assigned_at": f"2026-04-02T10:1{index}:00+00:00"},
        )

    state = ProjectState()
    state.open_project(project_path)
    sample_set_data = state.build_sample_set(
        target_id="target_moisture",
        preprocessing_recipe_id="recipe_none",
        sample_set_id="sample_set_roi_refs",
    )
    result = state.fit_model_run(
        sample_set_id=sample_set_data.sample_set.sample_set_id,
        signal_domain_id="signal_domain_1",
        model_type="PLS",
        hyperparameters={"n_components": 1},
        cv="logo",
        model_run_id="model_run_roi_refs",
    )

    assert result.model_run.model_run_id == "model_run_roi_refs"


def test_fit_model_run_logs_training_reflectance_clipping(tmp_path, qapp):
    project_path = str(tmp_path / "phase5_model_run_clipping.h5")
    _create_model_project(project_path)

    state = ProjectState()
    state.open_project(project_path)
    sample_set_data = state.build_sample_set(
        target_id="target_moisture",
        preprocessing_recipe_id="recipe_none",
        sample_set_id="sample_set_clipping",
    )

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        state.fit_model_run(
            sample_set_id=sample_set_data.sample_set.sample_set_id,
            signal_domain_id="signal_domain_1",
            model_type="PLS",
            hyperparameters={"n_components": 1},
            cv="logo",
            model_run_id="model_run_clipping",
        )
    finally:
        logger.remove(sink_id)

    clip_messages = [m for m in messages if "Conversion of calibration set" in m]
    assert clip_messages, "training-side reflectance clipping totals must be logged"
    assert "sample_set_clipping" in clip_messages[0]
    assert "clip_low=14" in clip_messages[0]
    assert "clip_high=0" in clip_messages[0]


def test_project_state_fit_model_run_uses_stateless_predictor_for_stateless_recipe(tmp_path, qapp):
    project_path = str(tmp_path / "phase5_model_run_stateless_predictor.h5")
    _create_model_project(project_path)

    state = ProjectState()
    state.open_project(project_path)

    sample_set_data = state.build_sample_set(
        target_id="target_moisture",
        preprocessing_recipe_id="recipe_none",
        sample_set_id="sample_set_stateless_predictor",
    )
    result = state.fit_model_run(
        sample_set_id=sample_set_data.sample_set.sample_set_id,
        signal_domain_id="signal_domain_1",
        model_type="PLS",
        hyperparameters={"n_components": 1},
        cv="logo",
        model_run_id="model_run_stateless_predictor",
    )

    predictor = state.model_registry["model_run_stateless_predictor"]
    assert isinstance(predictor, _StatelessRecipePredictor)
    assert result.predictor is predictor


def test_stateless_recipe_predictor_reuses_steps_without_refitting():
    class CountingTransformer:
        def __init__(self) -> None:
            self.fit_calls = 0
            self.transform_calls = 0

        def fit(self, X, y=None):
            del X, y
            self.fit_calls += 1
            raise AssertionError("stateless prediction must not refit transformers")

        def transform(self, X):
            self.transform_calls += 1
            return np.asarray(X, dtype=np.float32) + np.float32(1.0)

    class CaptureEstimator:
        def __init__(self) -> None:
            self.inputs: list[np.ndarray] = []

        def predict(self, X):
            self.inputs.append(np.asarray(X, dtype=np.float32).copy())
            return np.sum(X, axis=1, dtype=np.float32)

    transformer = CountingTransformer()
    estimator = CaptureEstimator()
    predictor = _StatelessRecipePredictor(
        steps=[("counting", transformer)],
        estimator=estimator,
    )
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    first = predictor.predict(X)
    second = predictor.predict(X)

    assert transformer.fit_calls == 0
    assert transformer.transform_calls == 2
    assert estimator.inputs[0].dtype == np.float32
    np.testing.assert_allclose(estimator.inputs[0], X + 1.0)
    np.testing.assert_allclose(first, second)


def test_msc_emsc_fitted_state_round_trips_without_refitting_prediction_data():
    wavelengths = np.linspace(1000.0, 1700.0, 9, dtype=np.float32)
    peak = np.exp(-0.5 * ((np.arange(9) - 4.0) / 1.5) ** 2)
    X_train = np.array(
        [np.linspace(10.0 + offset, 30.0 + 2 * offset, 9) + (1.0 + 0.3 * offset) * peak
         for offset in range(8)],
        dtype=np.float32,
    )
    y_train = np.linspace(1.0, 4.5, 8, dtype=np.float32)
    X_prediction = X_train[:3] + 0.25
    recipe = PreprocessingRecipe(msc=True, emsc=True)
    predictor = make_model_pipeline(
        recipe=recipe,
        wavelengths=wavelengths,
        estimator=PLSRegression(n_components=1),
    )
    predictor.fit(X_train, y_train)
    expected = predictor.predict(X_prediction)

    state = fitted_preprocessing_state(predictor, recipe=recipe)
    restored_steps = restore_fitted_preprocessing_steps(
        recipe.build_steps(wavelengths),
        recipe=recipe,
        state=state,
    )
    X_ready = transform_preprocessing_steps(X_prediction, restored_steps)
    actual = predictor.named_steps["model"].predict(X_ready)

    assert X_ready.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_fit_model_run_uses_roi_aggregated_primary_metrics(tmp_path, monkeypatch):
    project_path = str(tmp_path / "phase5_model_run_aggregated.h5")
    _create_model_project(project_path)

    state = ProjectState()
    state.open_project(project_path)
    sample_set_data = build_sample_set(
        project_path=project_path,
        rois=state.rois,
        target_assignments=state.target_assignments,
        target_id="target_moisture",
        preprocessing_recipe_id="recipe_none",
        roi_ids=["roi_1", "roi_2"],
        sample_set_id="sample_set_agg",
    )

    def _fake_cross_validate(estimator, X, y, cv, groups, scoring=None, return_train_score=False):
        del estimator, X, scoring, return_train_score
        assert isinstance(cv, RoundRobinGroups) and cv.n_splits == 7  # cv="auto"
        # One label per ROI: its position in the sample set's roi_ids.
        np.testing.assert_array_equal(groups, [0, 0, 1, 1])
        np.testing.assert_array_equal(y, np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
        return {
            "y_true": y,
            # the component curve's cross-validation: one column per number of components
            "y_pred": np.array([[0.0], [2.0], [1.0], [3.0]], dtype=np.float32),
            "cv_scheme": {"type": "LeaveOneGroupOut", "n_splits": 2},
        }

    monkeypatch.setattr("enginir.modeling._components.cross_validate", _fake_cross_validate)

    result = fit_model_run(
        project_path=project_path,
        sample_set=sample_set_data.sample_set,
        datasets=state.datasets,
        reference_corrections=state.reference_corrections,
        rois=state.rois,
        target_assignments=state.target_assignments,
        recipes=state.recipes,
        signal_domains=state.signal_domains,
        signal_domain_id="signal_domain_1",
        model_type="PLS",
        # the pipeline path (its cross_validate is replaced); kernel fits run on sums
        hyperparameters={"n_components": 1, "algorithm": "nipals"},
        model_run_id="model_run_agg",
    )

    assert result.model_run.metrics["rmsecv"] == pytest.approx(0.0)
    assert result.model_run.metrics["bias"] == pytest.approx(0.0)
    assert result.model_run.cv_spec["type"] == "LeaveOneGroupOut"
    assert result.model_run.cv_spec["requested"] == "auto"
    assert result.model_run.metrics["pixel_rmsecv"] == pytest.approx(1.0)
    assert result.model_run.metrics["pixel_bias"] == pytest.approx(0.0)
    np.testing.assert_array_equal(result.roi_ids, np.array(["roi_1", "roi_2"], dtype=object))
    np.testing.assert_allclose(result.roi_y_true, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(result.roi_y_pred, np.array([1.0, 2.0], dtype=np.float32))


def test_fit_model_run_preprocesses_stateless_recipe_once_before_cv(tmp_path, monkeypatch):
    project_path = str(tmp_path / "phase5_model_run_stateless_cv.h5")
    _create_model_project(project_path)
    save_preprocessing_recipe(
        project_path,
        "recipe_stateless",
        recipe_payload(filter="sg_smooth", sg_window=3, snv=True, detrend=True, sg_edges="interp"),
    )

    state = ProjectState()
    state.open_project(project_path)
    sample_set_data = build_sample_set(
        project_path=project_path,
        rois=state.rois,
        target_assignments=state.target_assignments,
        target_id="target_moisture",
        preprocessing_recipe_id="recipe_stateless",
        roi_ids=["roi_1", "roi_2"],
        sample_set_id="sample_set_stateless",
    )

    shift = np.linspace(0.0, 0.5, sample_set_data.X.shape[1], dtype=np.float32)
    preprocess_calls: list[np.ndarray] = []

    def _fake_apply_preprocessing_recipe(X, *, recipe, wavelengths, signal_unit=None):
        del recipe, wavelengths, signal_unit
        preprocess_calls.append(np.asarray(X))
        return (np.asarray(X) + shift).astype(np.float32)

    def _fake_cross_validate(estimator, X, y, cv, groups, scoring=None, return_train_score=False):
        del y, scoring, return_train_score
        assert isinstance(cv, RoundRobinGroups) and cv.n_splits == 7  # cv="auto"
        model = estimator.estimator  # the component curve's wrapper around the model
        assert isinstance(model, PLSRegression)
        assert model.n_components == 1  # A_max: 2 ROIs allow no more than the given 1
        np.testing.assert_array_equal(X, np.concatenate(preprocess_calls) + shift)
        # One label per ROI: its position in the sample set's roi_ids.
        np.testing.assert_array_equal(groups, [0, 0, 1, 1])
        return {
            "cv_scheme": {"type": "LeaveOneGroupOut", "n_splits": 2},
            "y_true": sample_set_data.y,
            "y_pred": np.array([[0.0], [2.0], [1.0], [3.0]], dtype=np.float32),
        }

    monkeypatch.setattr(
        "enginir.modeling._service.apply_preprocessing_recipe", _fake_apply_preprocessing_recipe
    )
    monkeypatch.setattr("enginir.modeling._components.cross_validate", _fake_cross_validate)

    fit_model_run(
        project_path=project_path,
        sample_set=sample_set_data.sample_set,
        datasets=state.datasets,
        reference_corrections=state.reference_corrections,
        rois=state.rois,
        target_assignments=state.target_assignments,
        recipes=state.recipes,
        signal_domains=state.signal_domains,
        signal_domain_id="signal_domain_1",
        model_type="PLS",
        hyperparameters={"n_components": 1, "algorithm": "nipals"},  # the pipeline path
        model_run_id="model_run_stateless_cv",
    )

    # Once per pixel, before the cross-validation: ROI by ROI, right after the conversion,
    # so the folds never preprocess (plans/kernel-pls.md, D7).
    assert [call.shape for call in preprocess_calls] == [(2, sample_set_data.X.shape[1])] * 2


def test_build_sample_set_loads_all_roi_spectra_with_one_project_open(tmp_path):
    project_path = str(tmp_path / "phase5_model_run_batch_read.h5")
    _create_model_project(project_path)

    state = ProjectState()
    state.open_project(project_path)

    with patch("enginir.io.project_store.h5py.File", wraps=h5py.File) as mocked_file:
        sample_set_data = build_sample_set(
            project_path=project_path,
            rois=state.rois,
            target_assignments=state.target_assignments,
            target_id="target_moisture",
            preprocessing_recipe_id="recipe_none",
            roi_ids=["roi_1", "roi_2", "roi_3", "roi_4"],
            sample_set_id="sample_set_batch",
        )

    assert mocked_file.call_count == 1
    assert sample_set_data.X.shape == (8, 3)
    assert sample_set_data.X.dtype == np.uint16
    assert sample_set_data.y.shape == (8,)


def test_fit_model_run_rejects_cross_validation_that_ignores_roi_groups(tmp_path):
    from enginir.modeling import fit_model_run

    with pytest.raises(ValueError, match="cv must be 'auto', 'logo' or a number of folds"):
        fit_model_run(
            project_path=str(tmp_path / "unused.h5"),
            sample_set=None,  # type: ignore[arg-type]
            datasets={}, reference_corrections={}, rois={}, target_assignments={}, recipes={},
            signal_domains={}, signal_domain_id="sd", model_type="PLS", cv="loo",
        )


def test_model_run_working_arrays_are_float32(tmp_path, qapp):
    """Targets, CV predictions and persisted ROI diagnostics use the float32 working dtype."""
    project_path = str(tmp_path / "dtype.h5")
    _create_model_project(project_path)
    state = ProjectState()
    state.open_project(project_path)
    sample_set_data = state.build_sample_set(
        target_id="target_moisture",
        preprocessing_recipe_id="recipe_none",
    )
    result = state.fit_model_run(
        sample_set_id=sample_set_data.sample_set.sample_set_id,
        signal_domain_id="signal_domain_1",
        model_type="PLS",
        hyperparameters={"n_components": 1},
        cv="logo",
    )

    assert sample_set_data.y.dtype == np.float32
    for array in (result.pixel_y_true, result.pixel_y_pred, result.roi_y_true, result.roi_y_pred):
        assert array.dtype == np.float32
    cv_outputs = result.model_run.artifact_refs["cv_outputs"]
    assert cv_outputs["y_true_roi"].dtype == cv_outputs["y_pred_roi"].dtype == np.float32


def _fit_kwargs(project_path, state, sample_set, **overrides):
    return {"project_path": project_path, "sample_set": sample_set, "datasets": state.datasets,
            "reference_corrections": state.reference_corrections, "rois": state.rois,
            "target_assignments": state.target_assignments, "recipes": state.recipes,
            "signal_domains": state.signal_domains, "signal_domain_id": "signal_domain_1",
            "model_type": "PLS", "hyperparameters": {"n_components": 1}, **overrides}


def test_fit_model_run_fits_msc_inside_every_cross_validation_fold(tmp_path, monkeypatch):
    """MSC learns a reference spectrum; fitting it on all pixels would leak held-out ROIs."""
    from sklearn.pipeline import Pipeline

    project_path = str(tmp_path / "model_run_msc_folds.h5")
    _create_model_project(project_path)
    save_preprocessing_recipe(project_path, "recipe_msc",
                              PreprocessingRecipe(msc=True).to_dict())
    state = ProjectState()
    state.open_project(project_path)
    data = build_sample_set(project_path=project_path, rois=state.rois,
                            target_assignments=state.target_assignments,
                            target_id="target_moisture", preprocessing_recipe_id="recipe_msc",
                            roi_ids=["roi_1", "roi_2", "roi_3"], sample_set_id="sample_set_msc")
    seen = {}

    def _no_preprocessing_outside_cv(X, *, recipe, wavelengths):
        raise AssertionError("a stateful recipe must not be applied before cross-validation")

    def _record_cross_validate(estimator, X, y, cv, groups, scoring=None,
                               return_train_score=False):
        seen["estimator"] = estimator.estimator  # inside the component curve's wrapper
        columns = estimator.estimator.get_params()["model__n_components"]
        return {"cv_scheme": {"type": "LeaveOneGroupOut", "n_splits": 3},
                "y_true": y,
                "y_pred": np.repeat(np.asarray(y, dtype=np.float32)[:, None], columns, axis=1)}

    monkeypatch.setattr("enginir.modeling._service.apply_preprocessing_recipe",
                        _no_preprocessing_outside_cv)
    monkeypatch.setattr("enginir.modeling._components.cross_validate", _record_cross_validate)
    fit_model_run(**_fit_kwargs(project_path, state, data.sample_set))

    estimator = seen["estimator"]
    assert isinstance(estimator, Pipeline)  # refitted, MSC included, on every training fold
    assert [name for name, _ in estimator.steps] == ["msc", "model"]


def test_fit_model_run_rejects_a_signal_domain_the_training_data_cannot_produce(tmp_path):
    from enginir.project import SignalDomainCompatibilityError

    project_path = str(tmp_path / "model_run_incompatible.h5")
    _create_model_project(project_path)
    domain = load_project_store(project_path)["signal_domains"]["signal_domain_1"]
    save_signal_domain_contract(project_path, "signal_domain_swir",
                                {**domain, "source_modality": "swir_line_scan"})
    state = ProjectState()
    state.open_project(project_path)
    data = build_sample_set(project_path=project_path, rois=state.rois,
                            target_assignments=state.target_assignments,
                            target_id="target_moisture", preprocessing_recipe_id="recipe_none",
                            roi_ids=["roi_1", "roi_2"], sample_set_id="sample_set_swir")

    with pytest.raises(SignalDomainCompatibilityError, match="source_modality"):
        fit_model_run(**_fit_kwargs(project_path, state, data.sample_set,
                                    signal_domain_id="signal_domain_swir"))


def test_fit_refuses_a_stored_sample_set_whose_roi_lost_its_target_value(tmp_path):
    """Refit-time blocker: unlabelled members are refused, not silently dropped."""
    project_path = str(tmp_path / "model_run_unlabelled.h5")
    _create_model_project(project_path)
    state = ProjectState()
    state.open_project(project_path)
    data = build_sample_set(project_path=project_path, rois=state.rois,
                            target_assignments=state.target_assignments,
                            target_id="target_moisture", preprocessing_recipe_id="recipe_none",
                            roi_ids=["roi_1", "roi_2"], sample_set_id="sample_set_unlabelled")
    without_roi_2 = {key: value for key, value in state.target_assignments.items()
                     if key[0] != "roi_2"}

    with pytest.raises(ValueError, match=r"without a value for target .*roi_2"):
        fit_model_run(**_fit_kwargs(project_path, state, data.sample_set,
                                    target_assignments=without_roi_2))


# -- Tests for the v3 sample-set builder -----------------------------------------------


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
    save_reference_correction(
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
            "reference_correction_id": "cal_1",
            "name": "ROI 1",
            "source_file": "/data/train.eiff",
            "job": "calibration",
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
            "reference_correction_id": "cal_1",
            "name": "ROI 2",
            "source_file": "/data/train.eiff",
            "job": "calibration",
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
            "reference_correction_id": "cal_1",
            "name": "ROI 3",
            "source_file": "/data/train.eiff",
            "job": "calibration",
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


# -- Stored ROIs read back without data files: patches and spectra as the fit sees them ----


@pytest.fixture()
def project(tmp_path):
    rng = np.random.default_rng(8)
    cube = rng.integers(1000, 3000, (24, 10, 7)).astype(np.uint16)
    cube[:2] = 100       # dark frames
    cube[-2:] = 3600     # white tile
    path = tmp_path / "tray.h5"
    save_hdf5(str(path), cube, np.linspace(1000.0, 1600.0, 7))
    raw_path = tmp_path / "raw.h5"
    save_hdf5(str(raw_path), cube, np.linspace(1000.0, 1600.0, 7))
    project = Project.create(tmp_path / "p.h5")
    tray = project.add_dataset(path, dark_frames=(0, 1), white_frames=(22, 23))
    tray.add_roi("T1", frames=(3, 6), rows=(1, 4))
    tray.add_roi("T2", frames=(10, 14), rows=(5, 8))
    project.add_dataset(raw_path, unit="DN").add_roi("R1", frames=(3, 5), rows=(0, 1))
    return project, cube


def test_band_images_rebuild_each_roi_patch_from_the_project(project):
    project_obj, cube = project
    rois = {roi.roi_id: roi for roi in project_obj.workspace.rois.values()}
    images = load_roi_band_images(project_obj.path, rois, [5, 1, 3])  # any band order
    t2 = project_obj.roi("T2")
    np.testing.assert_array_equal(images[t2.roi_id], cube[10:15, 5:9][:, :, [5, 1, 3]])


def _summaries(project_obj, unit, recipe=None):
    ws = project_obj.workspace
    return roi_spectrum_summaries(project_obj.path, list(ws.rois.values()), datasets=ws.datasets,
                                  reference_corrections=ws.reference_corrections, signal_unit=unit, recipe=recipe)


def test_means_follow_the_fit_conversion_and_skip_incompatible_rois(project):
    project_obj, cube = project
    summaries, skipped = _summaries(project_obj, "Absorbance")
    t1 = project_obj.roi("T1")
    dark, white = cube[:2].mean(axis=0), cube[-2:].mean(axis=0)
    pixels = cube[3:7, 1:5].astype(float)
    reflectance = np.clip((pixels - dark[1:5]) / (white[1:5] - dark[1:5]), 1e-6, 1.0)
    expected = -np.log10(reflectance).reshape(-1, cube.shape[2])
    np.testing.assert_allclose(summaries[t1.roi_id].mean, expected.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(summaries[t1.roi_id].median, np.median(expected, axis=0),
                               rtol=1e-5)
    r1 = project_obj.roi("R1").roi_id
    assert r1 in skipped and r1 not in summaries  # raw counts cannot give absorbance


def test_the_fit_and_the_summaries_convert_identically(project):
    """The same function converts the ROIs for the fit and for the Project ROIs tab."""
    from enginir.modeling import _service

    project_obj, _cube = project
    calls: list[str] = []
    original = _service.roi_to_signal_domain

    def spy(*args, **kwargs):
        calls.append(kwargs["roi"].name)
        return original(*args, **kwargs)

    _service.roi_to_signal_domain = spy
    try:
        project_obj.add_target("API", unit="%")
        project_obj.set_value("T1", "API", 1.0)
        project_obj.set_value("T2", "API", 2.0)
        project_obj.fit("API", n_components=1, cv="logo")
        _summaries(project_obj, "Absorbance")
    finally:
        _service.roi_to_signal_domain = original
    # Each training ROI went through it once for the fit and once for the summaries.
    assert calls.count("T1") == 2 and calls.count("T2") == 2


def test_preprocessing_is_applied_per_pixel_before_averaging(project):
    project_obj, _cube = project
    plain, _ = _summaries(project_obj, "Reflectance")
    snv, _ = _summaries(project_obj, "Reflectance", PreprocessingRecipe(snv=True))
    t1 = project_obj.roi("T1").roi_id
    # The mean of SNV-corrected pixels has mean ~0 but is not the SNV of the mean spectrum.
    assert abs(float(snv[t1].mean.mean())) < 1e-5
    mean = plain[t1].mean
    snv_of_mean = (mean - mean.mean()) / mean.std(ddof=1)
    assert not np.allclose(snv[t1].mean, snv_of_mean, atol=1e-4)
    assert np.all(snv[t1].p05 <= snv[t1].median) and np.all(snv[t1].median <= snv[t1].p95)


# -- ROI scores of the score plot, stored by the fit (model-workspace-diagnostics contract §2, ----


PIXELS = 12  # per ROI in the fixture


def _expected(model, drawn: int) -> tuple[np.ndarray, np.ndarray]:
    """ROI means of the final model's own training scores (rows: ROIs in order, drawn pixels,
    excluded rows left out), written out independently of the library."""
    diagnostics = model.diagnostics
    scores = np.asarray(diagnostics["x_scores_"], dtype=np.float64)
    excluded = model.excluded_pixels
    roi_ids = [str(r) for r in np.asarray(diagnostics["roi_ids"])]
    labels = []
    for index, roi_id in enumerate(roi_ids):
        kept = drawn - len(excluded.get(roi_id, ()))
        labels += [index] * kept
    labels = np.asarray(labels)
    assert labels.size == scores.shape[0]
    counts = np.bincount(labels, minlength=len(roi_ids))
    means = np.stack([np.bincount(labels, weights=scores[:, c], minlength=len(roi_ids))
                      for c in range(scores.shape[1])], axis=1) / counts[:, None]
    return means, counts


def test_roi_scores_with_exclusion_are_the_means_of_the_kept_pixels(tmp_path):
    project = project_with_an_odd_pixel(tmp_path)
    model = project.fit("API", n_components=2, fit_spec=EXCLUSION)
    assert model.metrics["n_excluded_pixels"] > 0
    means, counts = _expected(model, PIXELS)
    np.testing.assert_allclose(model.diagnostics["roi_scores"], means, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(model.diagnostics["roi_score_pixels"], counts)
    payload = _roi_score_payload(model.diagnostics)
    assert payload["scores_roi"].shape == (8, 2) and payload["scores_without_pixels"] == 0


def test_roi_scores_with_a_pixel_sample_and_exclusion(tmp_path):
    project = project_with_an_odd_pixel(tmp_path)
    for spec, drawn in (({"pixels": {"per_roi": 5, "seed": 1}}, 5),
                        ({"pixels": {"per_roi": 8, "seed": 2}, **EXCLUSION}, 8)):
        model = project.fit("API", n_components=2, fit_spec=spec)
        means, counts = _expected(model, drawn)
        np.testing.assert_allclose(model.diagnostics["roi_scores"], means, rtol=1e-4,
                                   atol=1e-5)
        np.testing.assert_array_equal(model.diagnostics["roi_score_pixels"], counts)


def test_the_viewer_shows_missing_roi_scores_as_not_available():
    payload = _roi_score_payload({"x_scores_": np.ones((10, 2))})
    assert "scores_roi" not in payload and "not available" in payload["scores_message"]
    nan_row = np.array([[1.0, 2.0], [np.nan, np.nan]], dtype=np.float32)
    assert _roi_score_payload({"roi_scores": nan_row})["scores_without_pixels"] == 1
