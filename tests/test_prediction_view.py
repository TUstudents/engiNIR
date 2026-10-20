"""Smoke tests for the split prediction view."""

from __future__ import annotations

import time
import types
from threading import Event

import numpy as np
import pytest
from PyQt6.QtWidgets import QMessageBox

from enginir.project import CalibrationSpec, ModelRun, PredictionRun, SignalDomainContract
from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


def _wait_until(qapp, predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    qapp.processEvents()
    assert predicate()


@pytest.fixture()
def project_state(tmp_path):
    from enginir.viewer._state import ProjectState

    state = ProjectState()
    state.new_project(str(tmp_path / "prediction.h5"))
    return state


def _seed_model_run(project_state):
    project_state.model_runs["model_1"] = ModelRun(
        model_run_id="model_1",
        sample_set_id="sample_set_1",
        signal_domain_id="signal_domain_1",
        model_type="PLS",
        hyperparameters={},
        cv_spec={},
        metrics={},
        artifact_refs={},
        created_at="2026-04-02T10:00:00+00:00",
    )
    project_state.active_model_name = "model_1"


def _allow_prediction_preflight(monkeypatch):
    monkeypatch.setattr("enginir.viewer._prediction.preflight_prediction", lambda **kwargs: None)


def _patch_prediction_flow(monkeypatch, project_state):
    """Patch the predict flow seams with contract-shaped domain objects."""
    from enginir.project import DatasetRef
    from enginir.viewer._state import PredictionJobContext

    _allow_prediction_preflight(monkeypatch)
    project_state.build_prediction_job_context = lambda *, model_run_id: PredictionJobContext(
        project_path=str(project_state.project_path),
        model_run=project_state.model_runs[model_run_id],
        signal_domain=types.SimpleNamespace(signal_domain_id="signal_domain_1"),
        preprocessing_recipe_id="recipe_1",
        recipe=types.SimpleNamespace(),
        datasets={},
        calibrations={},
        active_calibration_ids={},
        model_registry={},
    )
    monkeypatch.setattr(
        "enginir.viewer._prediction.load_spectra",
        lambda path, header_only=False: types.SimpleNamespace(
            X=[[1.0]], wavelengths=[1000.0], meta={}
        ),
    )
    monkeypatch.setattr(
        "enginir.viewer._prediction.resolve_dataset",
        lambda _datasets, data_path, _data, **kwargs: (DatasetRef(
            dataset_id="dataset_1",
            source_path=data_path,
            source_format="eiff",
            source_modality="nir_hsi",
            cube_shape=(1, 1, 1),
            n_bands=1,
            wavelengths=np.array([1000.0], dtype=np.float32),
            loader_metadata={},
            import_timestamp="2026-04-02T10:00:00+00:00",
        ), True),
    )
    monkeypatch.setattr(
        "enginir.viewer._prediction.resolve_prediction_calibration",
        lambda *args, **kwargs: (
            CalibrationSpec(
                calibration_id="cal_1",
                dataset_id="dataset_1",
                mode="raw_dn",
                dark_ref_source=None,
                white_ref_source=None,
                dark_ref_data=None,
                white_ref_data=None,
                black_level=None,
                white_level=None,
                output_signal_unit="DN",
                created_at="2026-04-02T10:00:00+00:00",
            ),
            False,
        ),
    )


def test_prediction_view_warns_when_no_models(qapp, project_state, monkeypatch):
    from enginir.viewer._prediction import PredictionView

    infos: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: infos.append("info"))

    view = PredictionView(project_state)
    view.predict_btn_clicked()

    assert infos == ["info"]
    assert "fit or load a model run" in view._readiness_status.text().lower()


def test_prediction_view_cancel_button_requests_cooperative_cancel(qapp, project_state):
    from enginir.viewer._prediction import PredictionView

    view = PredictionView(project_state)
    cancel_event = Event()
    view._job_cancel_event = cancel_event
    view._set_busy(True, "Running prediction...")

    assert view._btn_cancel.isEnabled()
    view._cancel_prediction_clicked()

    assert cancel_event.is_set()
    assert not view._btn_cancel.isEnabled()
    assert "cancelling" in view._service_status.label.text().lower()


def test_prediction_view_happy_path_calls_backend_and_updates_status(
    qapp, project_state, monkeypatch
):
    from enginir.viewer._prediction import PredictionRunDialog, PredictionView

    _seed_model_run(project_state)

    def fake_exec(self):
        self._path_edit.setText("/tmp/predict.eiff")
        return self.DialogCode.Accepted

    monkeypatch.setattr(PredictionRunDialog, "exec", fake_exec)
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: pytest.fail(f"unexpected warning: {args}")
    )
    _patch_prediction_flow(monkeypatch, project_state)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "enginir.viewer._prediction.predict_with_model_run",
        lambda **kwargs: (
            captured.update(kwargs)
            or types.SimpleNamespace(
                prediction_run=types.SimpleNamespace(prediction_run_id="prediction_1")
            )
        ),
    )
    finalized: list[str] = []
    monkeypatch.setattr(
        project_state,
        "finalize_prediction_result",
        lambda **kwargs: finalized.append(
            kwargs["prediction_result"].prediction_run.prediction_run_id
        ),
    )

    view = PredictionView(project_state)
    view.predict_btn_clicked()
    _wait_until(qapp, lambda: finalized == ["prediction_1"])

    assert captured["model_run"].model_run_id == "model_1"
    assert captured["data_path"] == "/tmp/predict.eiff"
    assert captured["persist_artifacts"] is True
    assert captured["return_arrays"] is False
    assert view._service_status.label.text() == "Service: prediction prediction_1"


def test_prediction_dialog_explicit_calibration_builds_and_validates(
    qapp, project_state, monkeypatch
):
    from enginir.viewer._prediction import PredictionRunDialog

    dialog = PredictionRunDialog(["model_1"], default_model_run_id="model_1")
    dialog._path_edit.setText("/tmp/predict.eiff")
    dialog._explicit_calibration.setChecked(True)
    dialog._mode_combo.setCurrentText("dark_white_reference")
    dialog._output_unit_combo.setCurrentText("Reflectance")
    dialog._black_level_spin.setValue(10.0)
    dialog._white_level_spin.setValue(100.0)

    calibration = dialog.calibration
    assert calibration is not None
    assert calibration.mode == "dark_white_reference"
    assert calibration.white_level == pytest.approx(100.0)

    dialog._mode_combo.setCurrentText("raw_dn")
    dialog._output_unit_combo.setCurrentText("Reflectance")
    warnings: list[str] = []
    accepted: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warnings.append(str(args[2]))
    )
    monkeypatch.setattr(dialog, "accept", lambda: accepted.append("accepted"))

    dialog._on_accept()

    assert warnings == ["Mode 'raw_dn' requires output unit 'DN'."]
    assert accepted == []


def test_prediction_dialog_exposes_default_scalar_calibration(qapp):
    from enginir.viewer._prediction import PredictionRunDialog

    dialog = PredictionRunDialog(["model_1"], default_model_run_id="model_1")

    assert dialog.calibration is None
    assert dialog._mode_combo.isEnabled()
    assert dialog._black_level_spin.isEnabled()

    calibration = dialog.default_calibration

    assert calibration.mode == "dark_white_reference"
    assert calibration.black_level == pytest.approx(128.0)
    assert calibration.white_level == pytest.approx(3500.0)


def test_prediction_view_uses_dialog_default_calibration_for_new_dataset(
    qapp, project_state, monkeypatch
):
    from enginir.viewer._prediction import PredictionRunDialog, PredictionView
    from enginir.viewer._state import PredictionJobContext

    _seed_model_run(project_state)

    def fake_exec(self):
        self._path_edit.setText("/tmp/predict_new.eiff")
        self._black_level_spin.setValue(10.0)
        self._white_level_spin.setValue(100.0)
        return self.DialogCode.Accepted

    from enginir.project import DatasetRef

    monkeypatch.setattr(PredictionRunDialog, "exec", fake_exec)
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: pytest.fail(f"unexpected warning: {args}")
    )
    _allow_prediction_preflight(monkeypatch)
    project_state.build_prediction_job_context = lambda *, model_run_id: PredictionJobContext(
        project_path=str(project_state.project_path),
        model_run=project_state.model_runs[model_run_id],
        signal_domain=types.SimpleNamespace(signal_domain_id="signal_domain_1"),
        preprocessing_recipe_id="recipe_1",
        recipe=types.SimpleNamespace(),
        datasets={},
        calibrations={},
        active_calibration_ids={},
        model_registry={},
    )
    monkeypatch.setattr(
        "enginir.viewer._prediction.load_spectra",
        lambda path, header_only=False: types.SimpleNamespace(
            X=[[1.0]], wavelengths=[1000.0], meta={}
        ),
    )
    monkeypatch.setattr(
        "enginir.viewer._prediction.resolve_dataset",
        lambda _datasets, data_path, _data, **kwargs: (DatasetRef(
            dataset_id="dataset_new",
            source_path=data_path,
            source_format="eiff",
            source_modality="nir_hsi",
            cube_shape=(1, 1, 1),
            n_bands=1,
            wavelengths=np.array([1000.0], dtype=np.float32),
            loader_metadata={},
            import_timestamp="2026-04-02T10:00:00+00:00",
        ), True),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "enginir.viewer._prediction.predict_with_model_run",
        lambda **kwargs: (
            captured.update(kwargs)
            or types.SimpleNamespace(
                prediction_run=types.SimpleNamespace(prediction_run_id="prediction_1")
            )
        ),
    )
    finalized: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        project_state,
        "finalize_prediction_result",
        lambda **kwargs: finalized.append(
            (
                kwargs["prediction_result"].prediction_run.prediction_run_id,
                kwargs["calibration_is_new"],
            )
        ),
    )

    view = PredictionView(project_state)
    view.predict_btn_clicked()
    _wait_until(qapp, lambda: finalized == [("prediction_1", True)])

    calibration = captured["calibration"]
    assert isinstance(calibration, CalibrationSpec)
    assert calibration.dataset_id == "dataset_new"
    assert calibration.mode == "dark_white_reference"
    assert calibration.black_level == pytest.approx(10.0)
    assert calibration.white_level == pytest.approx(100.0)


def test_prediction_view_reuses_existing_calibration_when_override_unchecked(
    qapp, project_state, monkeypatch
):
    from enginir.viewer._prediction import PredictionRunDialog, PredictionView
    from enginir.viewer._state import PredictionJobContext

    _seed_model_run(project_state)
    existing_dataset = types.SimpleNamespace(
        dataset_id="dataset_existing", source_path="/tmp/predict_existing.eiff"
    )
    existing_calibration = CalibrationSpec(
        calibration_id="cal_existing",
        dataset_id="dataset_existing",
        mode="dark_white_reference",
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=None,
        white_ref_data=None,
        black_level=1.0,
        white_level=200.0,
        output_signal_unit="Reflectance",
        created_at="2026-04-02T10:00:00+00:00",
    )

    def fake_exec(self):
        self._path_edit.setText("/tmp/predict_existing.eiff")
        self._black_level_spin.setValue(10.0)
        self._white_level_spin.setValue(100.0)
        return self.DialogCode.Accepted

    monkeypatch.setattr(PredictionRunDialog, "exec", fake_exec)
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: pytest.fail(f"unexpected warning: {args}")
    )
    _allow_prediction_preflight(monkeypatch)
    project_state.build_prediction_job_context = lambda *, model_run_id: PredictionJobContext(
        project_path=str(project_state.project_path),
        model_run=project_state.model_runs[model_run_id],
        signal_domain=types.SimpleNamespace(signal_domain_id="signal_domain_1"),
        preprocessing_recipe_id="recipe_1",
        recipe=types.SimpleNamespace(),
        datasets={"dataset_existing": existing_dataset},
        calibrations={"cal_existing": existing_calibration},
        active_calibration_ids={"dataset_existing": "cal_existing"},
        model_registry={},
    )
    monkeypatch.setattr(
        "enginir.viewer._prediction.load_spectra",
        lambda path, header_only=False: types.SimpleNamespace(
            X=[[1.0]], wavelengths=[1000.0], meta={}
        ),
    )
    monkeypatch.setattr(
        "enginir.viewer._prediction.resolve_dataset",
        lambda *args, **kwargs: (existing_dataset, False),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "enginir.viewer._prediction.predict_with_model_run",
        lambda **kwargs: (
            captured.update(kwargs)
            or types.SimpleNamespace(
                prediction_run=types.SimpleNamespace(prediction_run_id="prediction_existing")
            )
        ),
    )
    finalized: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        project_state,
        "finalize_prediction_result",
        lambda **kwargs: finalized.append(
            (
                kwargs["prediction_result"].prediction_run.prediction_run_id,
                kwargs["calibration_is_new"],
            )
        ),
    )

    view = PredictionView(project_state)
    view.predict_btn_clicked()
    _wait_until(qapp, lambda: finalized == [("prediction_existing", False)])

    assert captured["calibration"] is existing_calibration


def test_prediction_map_widget_orients_map_like_spatial_canvas_and_adds_colorbar(qapp):
    from enginir.viewer._prediction import PredictionMapWidget

    widget = PredictionMapWidget()
    spatial_map = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    widget.update_data({"spatial_map": spatial_map, "target_unit": "mg"})

    rendered = np.asarray(widget._ax.images[0].get_array(), dtype=np.float32)
    np.testing.assert_allclose(rendered, spatial_map.T)
    assert widget._ax.get_xlabel() == "Frame"
    assert widget._ax.get_ylabel() == "Row"
    assert widget._colorbar is not None
    assert widget._colorbar.ax.get_ylabel() == "Prediction (mg)"
    assert len(widget._fig.axes) == 2
    assert "shape=3 x 2" in widget._stats_label.text()

    widget.clear()

    assert widget._colorbar is None
    assert len(widget._fig.axes) == 1


def test_prediction_map_widget_exposes_multi_output_selector(qapp):
    from enginir.viewer._prediction import PredictionMapWidget

    widget = PredictionMapWidget()
    spatial_map = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=np.float32,
    )

    widget.update_data(
        {
            "spatial_map": spatial_map,
            "output_names": ["moisture", "density"],
        }
    )

    assert not widget._output_combo.isHidden()
    assert widget._output_combo.count() == 2
    np.testing.assert_allclose(widget._ax.images[0].get_array(), spatial_map[..., 0].T)

    widget._output_combo.setCurrentIndex(1)

    np.testing.assert_allclose(widget._ax.images[0].get_array(), spatial_map[..., 1].T)


def test_prediction_workspace_payload_cache_uses_artifact_version(qapp, project_state, monkeypatch):
    from enginir.viewer._prediction import PredictionView

    _seed_model_run(project_state)
    view = PredictionView(project_state)
    preview_payloads = {"prediction_1": np.array([[1.0, 2.0]], dtype=np.float32)}
    monkeypatch.setattr(
        project_state,
        "load_prediction_spatial_map_preview",
        lambda prediction_run_id: preview_payloads[prediction_run_id],
    )
    first = PredictionRun(
        prediction_run_id="prediction_1",
        dataset_id="dataset_1",
        calibration_id="calibration_1",
        model_run_id="model_1",
        signal_domain_id="signal_domain_1",
        preprocessing_recipe_id="recipe_1",
        artifact_refs={
            "spatial_map": np.array([[1.0, 2.0]], dtype=np.float32),
            "y_pred": np.array([1.0, 2.0], dtype=np.float32),
        },
        created_at="2026-04-02T10:01:00+00:00",
    )
    project_state._apply_prediction_run(first)

    first_payload = view.workspace_visual_payload()

    preview_payloads["prediction_1"] = np.array([[3.0, 4.0]], dtype=np.float32)
    second = PredictionRun(
        prediction_run_id="prediction_1",
        dataset_id="dataset_1",
        calibration_id="calibration_1",
        model_run_id="model_1",
        signal_domain_id="signal_domain_1",
        preprocessing_recipe_id="recipe_1",
        artifact_refs={
            "spatial_map": np.array([[3.0, 4.0]], dtype=np.float32),
            "y_pred": np.array([3.0, 4.0], dtype=np.float32),
        },
        created_at="2026-04-02T10:02:00+00:00",
    )
    project_state._apply_prediction_run(second)

    second_payload = view.workspace_visual_payload()

    assert first_payload is not second_payload
    assert second_payload is not None
    np.testing.assert_allclose(second_payload["spatial_map"], [[3.0, 4.0]])


def test_prediction_summary_uses_artifact_metadata_without_loading_map(
    qapp, project_state, monkeypatch
):
    from enginir.viewer._prediction import PredictionView

    _seed_model_run(project_state)
    view = PredictionView(project_state)
    prediction = PredictionRun(
        prediction_run_id="prediction_meta",
        dataset_id="dataset_1",
        calibration_id="calibration_1",
        model_run_id="model_1",
        signal_domain_id="signal_domain_1",
        preprocessing_recipe_id="recipe_1",
        artifact_refs={
            "spatial_shape": (2, 3),
            "prediction_min": 1.0,
            "prediction_max": 4.0,
            "clip_low_count": 2,
            "clip_high_count": 3,
            "near_zero_denominator_count": 1,
        },
        created_at="2026-04-02T10:01:00+00:00",
    )
    project_state._apply_prediction_run(prediction)
    monkeypatch.setattr(
        project_state,
        "load_prediction_artifact_refs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("loaded map")),
    )

    summary = view.workspace_summary()

    assert summary["map_shape"] == "3 x 2"
    assert summary["map_range"] == "1 .. 4"
    assert summary["clipping"] == "low=2 | high=3 | near-zero denom=1"


def test_prediction_view_reports_backend_failure(qapp, project_state, monkeypatch):
    from enginir.viewer._prediction import PredictionRunDialog, PredictionView

    _seed_model_run(project_state)

    def fake_exec(self):
        self._path_edit.setText("/tmp/predict.eiff")
        return self.DialogCode.Accepted

    warnings: list[str] = []
    monkeypatch.setattr(PredictionRunDialog, "exec", fake_exec)
    _patch_prediction_flow(monkeypatch, project_state)
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warnings.append(str(args[2]))
    )
    monkeypatch.setattr(
        "enginir.viewer._prediction.predict_with_model_run",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )

    view = PredictionView(project_state)
    view.predict_btn_clicked()
    _wait_until(qapp, lambda: warnings == ["boom"])

    assert view._service_status.label.text() == "Service: idle"


def test_prediction_view_rejects_incompatible_explicit_calibration(
    qapp, project_state, monkeypatch
):
    import enginir.viewer._prediction as prediction_mod
    from enginir.viewer._prediction import PredictionRunDialog, PredictionView
    from enginir.viewer._state import PredictionJobContext

    _seed_model_run(project_state)

    def fake_exec(self):
        self._path_edit.setText("/tmp/predict.eiff")
        self._explicit_calibration.setChecked(True)
        self._mode_combo.setCurrentText("raw_dn")
        self._output_unit_combo.setCurrentText("DN")
        return self.DialogCode.Accepted

    warnings: list[str] = []
    logs: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(PredictionRunDialog, "exec", fake_exec)
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warnings.append(str(args[2]))
    )
    monkeypatch.setattr(
        prediction_mod.logger, "error", lambda message, *args: logs.append((message, args))
    )
    project_state.build_prediction_job_context = lambda *, model_run_id: PredictionJobContext(
        project_path=str(project_state.project_path),
        model_run=project_state.model_runs[model_run_id],
        signal_domain=SignalDomainContract(
            signal_domain_id="signal_domain_1",
            input_format="eiff",
            source_modality="nir_hsi",
            required_signal_unit="Reflectance",
            required_calibration_mode="dark_white_reference",
            requires_dark_ref=True,
            requires_white_ref=True,
            calibrated_input_allowed=False,
            n_bands=1,
            wavelengths=[1000.0],
            wavelength_tolerance_nm=0.0,
            band_mapping_policy="exact",
            output_dtype="float32",
            created_at="2026-04-02T10:00:00+00:00",
        ),
        preprocessing_recipe_id="recipe_1",
        recipe=types.SimpleNamespace(),
        datasets={},
        calibrations={},
        active_calibration_ids={},
        model_registry={},
    )
    monkeypatch.setattr("enginir.viewer._prediction.load_spectra", lambda path: None)

    view = PredictionView(project_state)
    view.predict_btn_clicked()

    assert warnings == [
        "Calibration '' has mode='raw_dn', but the model requires required_calibration_mode='dark_white_reference'"
    ]
    assert len(logs) == 1
    assert logs[0][0] == "Prediction failed during preflight: {}"
