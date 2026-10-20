"""Smoke tests for the split modeling view."""

from __future__ import annotations

import time
import types

import numpy as np
import pytest
from PyQt6.QtWidgets import QDialog, QMessageBox

from enginir.project import (
    CalibrationSpec,
    DatasetRef,
    ModelRun,
    SampleSet,
    TargetAssignment,
    TargetDefinition,
    TrainingROI,
)
from enginir.viewer._app import ensure_app


def _roi_provenance() -> dict[str, str]:
    return {
        "dataset_id": "dataset_1",
        "calibration_id": "cal_1",
        "signal_unit": "DN",
        "preview_recipe_summary": "none",
        "created_at": "2026-04-02T10:01:00+00:00",
    }


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
def state_and_project(tmp_path):
    from enginir.viewer._state import ProjectState, ViewerState

    cube = np.random.randint(0, 1000, (6, 5, 8), dtype=np.uint16)
    wavelengths = np.linspace(1000.0, 1700.0, 8)
    state = ViewerState(cube, wavelengths, {"source_modality": "nir_hsi"})
    project = ProjectState()
    project.new_project(str(tmp_path / "modeling.h5"))
    project.register_dataset(
        DatasetRef(
            dataset_id="dataset_1",
            source_path=str(tmp_path / "cube.eiff"),
            source_format="eiff",
            source_modality="nir_hsi",
            cube_shape=cube.shape,
            n_bands=cube.shape[-1],
            wavelengths=wavelengths.astype("float32"),
            loader_metadata={},
            import_timestamp="2026-04-02T10:00:00+00:00",
        )
    )
    project.set_active_calibration(
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
            created_at="2026-04-02T10:00:30+00:00",
        )
    )
    return state, project


def test_modeling_view_can_define_target(qapp, state_and_project, monkeypatch):
    from enginir.viewer._modeling import ModelingView, TargetSelectionDialog

    state, project = state_and_project
    view = ModelingView(state, project)

    def fake_exec(self):
        self._target_combo.setCurrentText("protein")
        self._unit_edit.setText("% w/w")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(TargetSelectionDialog, "exec", fake_exec)
    view.select_target_btn_clicked()

    selected = view.current_target()
    assert selected is not None
    assert selected.name == "protein"
    assert "% w/w" in selected.unit


def test_new_roi_dialog_rejects_non_finite_target_value(qapp, state_and_project, monkeypatch):
    from enginir.viewer._modeling import NewROIDialog

    state, _project = state_and_project
    target = TargetDefinition(
        target_id="target_1",
        name="moisture",
        unit="% w/w",
        value_type="float",
        missing_value_policy="allowed",
        created_at="2026-04-02T10:00:00+00:00",
    )
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(str(args[2])))

    dialog = NewROIDialog(0, 1, 0, 1, state, target)
    dialog._name_edit.setText("roi")
    assert dialog._target_value_edit is not None
    dialog._target_value_edit.setText("nan")

    dialog._on_accept()

    assert any("finite numeric value" in message for message in warnings)
    assert dialog.result() == 0


def test_assign_target_dialog_rejects_non_finite_value(qapp, state_and_project, monkeypatch):
    from enginir.viewer._modeling import AssignTargetValueDialog

    _state, project = state_and_project
    target = TargetDefinition(
        target_id="target_1",
        name="moisture",
        unit="% w/w",
        value_type="float",
        missing_value_policy="allowed",
        created_at="2026-04-02T10:00:00+00:00",
    )
    roi = TrainingROI(
        roi_id="roi_1",
        dataset_id="dataset_1",
        calibration_id="cal_1",
        name="ROI 1",
        wavelengths=np.linspace(1000.0, 1700.0, 8),
        n_pixels=4,
        spatial_bounds=(0, 1, 0, 1),
        source_file=str(project.project_path or ""),
        created_at="2026-04-02T10:01:00+00:00",
        provenance=_roi_provenance(),
    )
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(str(args[2])))

    dialog = AssignTargetValueDialog(target, [roi], {})
    dialog._value_edit.setText("inf")

    dialog._on_accept()

    assert any("finite numeric value" in message for message in warnings)
    assert dialog.result() == 0


def test_build_sample_set_from_selected_target(qapp, state_and_project):
    from enginir.viewer._modeling import ModelingView

    state, project = state_and_project
    project.define_target(
        TargetDefinition(
            target_id="target_1",
            name="moisture",
            unit="% w/w",
            value_type="float",
            missing_value_policy="allowed",
            created_at="2026-04-02T10:00:00+00:00",
        )
    )
    for roi_id, value, start in [("roi_1", 3.2, 0), ("roi_2", 3.8, 2)]:
        project.add_roi(
            TrainingROI(
                roi_id=roi_id,
                dataset_id="dataset_1",
                calibration_id="cal_1",
                name=roi_id,
                wavelengths=state.wavelengths.astype("float32"),
                n_pixels=4,
                spatial_bounds=(start, start + 1, 0, 1),
                source_file="cube.eiff",
                created_at="2026-04-02T10:01:00+00:00",
                provenance=_roi_provenance(),
            ),
            np.arange(32, 64, dtype=np.uint16).reshape(4, 8)
            if roi_id == "roi_2"
            else np.arange(32, dtype=np.uint16).reshape(4, 8),
        )
        project.assign_target_to_roi(
            TargetAssignment(
                roi_id=roi_id,
                target_id="target_1",
                value=value,
                assigned_at="2026-04-02T10:02:00+00:00",
            )
        )

    view = ModelingView(state, project)
    view._selected_target_id = "target_1"
    view.build_sample_set_btn_clicked()
    _wait_until(qapp, lambda: len(project.sample_sets) == 1)

    sample_set = next(iter(project.sample_sets.values()))
    assert sample_set.target_id == "target_1"
    assert view._service_status.label.text().startswith("Service: sample set ")
    assert "fit a model" in view._readiness_status.text().lower()


def test_build_sample_set_failure_is_reported_without_persisting_recipe(
    qapp, state_and_project, monkeypatch
):
    from enginir.viewer._modeling import ModelingView

    state, project = state_and_project
    project.define_target(
        TargetDefinition(
            target_id="target_unlabeled",
            name="protein",
            unit="% w/w",
            value_type="float",
            missing_value_policy="allowed",
            created_at="2026-04-02T10:05:00+00:00",
        )
    )
    project.add_roi(
        TrainingROI(
            roi_id="roi_unlabeled",
            dataset_id="dataset_1",
            calibration_id="cal_1",
            name="grain_b",
            wavelengths=state.wavelengths.astype("float32"),
            n_pixels=4,
            spatial_bounds=(0, 1, 0, 1),
            source_file="cube.eiff",
            created_at="2026-04-02T10:06:00+00:00",
            provenance=_roi_provenance(),
        ),
        np.arange(32, dtype=np.uint16).reshape(4, 8),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warnings.append(str(args[2]))
    )

    view = ModelingView(state, project)
    view._selected_target_id = "target_unlabeled"
    recipes_before = set(project.recipes)
    sample_sets_before = set(project.sample_sets)
    view.build_sample_set_btn_clicked()
    _wait_until(qapp, lambda: bool(warnings))

    assert "has a value for target" in warnings[0]
    assert set(project.recipes) == recipes_before
    assert set(project.sample_sets) == sample_sets_before


def test_fit_model_builds_signal_domain_and_persists_run(qapp, state_and_project, monkeypatch):
    from enginir.viewer._modeling import FitModelRunDialog, ModelingView

    state, project = state_and_project
    project.set_active_calibration(
        CalibrationSpec(
            calibration_id="cal_1",
            dataset_id="dataset_1",
            mode="dark_white_reference",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=128.0,
            white_level=3500.0,
            output_signal_unit="Reflectance",
            created_at="2026-04-02T10:00:30+00:00",
        )
    )
    project.define_target(
        TargetDefinition(
            target_id="target_1",
            name="moisture",
            unit="% w/w",
            value_type="float",
            missing_value_policy="allowed",
            created_at="2026-04-02T10:01:00+00:00",
        )
    )
    for roi_id, value, start in [("roi_1", 3.2, 0), ("roi_2", 3.8, 2)]:
        project.add_roi(
            TrainingROI(
                roi_id=roi_id,
                dataset_id="dataset_1",
                calibration_id="cal_1",
                name=roi_id,
                wavelengths=state.wavelengths.astype("float32"),
                n_pixels=4,
                spatial_bounds=(start, start + 1, 0, 1),
                source_file="/tmp/cube.eiff",
                created_at="2026-04-02T10:01:30+00:00",
                provenance=_roi_provenance(),
            ),
            np.arange(32, 64, dtype=np.uint16).reshape(4, 8)
            if roi_id == "roi_2"
            else np.arange(32, dtype=np.uint16).reshape(4, 8),
        )
        project.assign_target_to_roi(
            TargetAssignment(
                roi_id=roi_id,
                target_id="target_1",
                value=value,
                assigned_at="2026-04-02T10:02:00+00:00",
            )
        )
    project.register_recipe("recipe_1", state.recipe.to_dict())
    project.create_sample_set(
        SampleSet(
            sample_set_id="sample_set_1",
            roi_ids=["roi_1", "roi_2"],
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
            created_at="2026-04-02T10:03:00+00:00",
        )
    )

    def fake_exec(self):
        self._signal_unit_combo.setCurrentIndex(self._signal_unit_combo.findText("Reflectance"))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(FitModelRunDialog, "exec", fake_exec)
    captured: dict[str, object] = {}

    def fake_fit_model_run(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            model_run=ModelRun(
                model_run_id="model_1",
                sample_set_id=kwargs["sample_set"].sample_set_id,
                signal_domain_id=kwargs["signal_domain_id"],
                model_type="PLS",
                hyperparameters={"n_components": 6},
                cv_spec={},
                metrics={"rmsecv": 1.23},
                artifact_refs={},
                created_at="2026-04-02T10:04:00+00:00",
            ),
            predictor=object(),
        )

    monkeypatch.setattr("enginir.modeling.fit_model_run", fake_fit_model_run)

    view = ModelingView(state, project)
    view._selected_target_id = "target_1"
    view.fit_model_btn_clicked()
    _wait_until(qapp, lambda: "model_1" in project.model_runs)

    assert captured["sample_set"].sample_set_id == "sample_set_1"
    assert "signal_domain_id" in captured
    assert captured["cv"] == "auto"
    assert view._service_status.label.text() == "Service: model model_1"


def test_fit_model_group_validation_guard(qapp, state_and_project, monkeypatch):
    from enginir.viewer._modeling import FitModelRunDialog, ModelingView

    state, project = state_and_project
    project.define_target(
        TargetDefinition(
            target_id="target_1",
            name="moisture",
            unit="% w/w",
            value_type="float",
            missing_value_policy="allowed",
            created_at="2026-04-02T10:01:00+00:00",
        )
    )
    for roi_id, value, start in [("roi_1", 3.2, 0), ("roi_2", 3.8, 2)]:
        project.add_roi(
            TrainingROI(
                roi_id=roi_id,
                dataset_id="dataset_1",
                calibration_id="cal_1",
                name=roi_id,
                wavelengths=state.wavelengths.astype("float32"),
                n_pixels=4,
                spatial_bounds=(start, start + 1, 0, 1),
                source_file="/tmp/cube.eiff",
                created_at="2026-04-02T10:01:30+00:00",
                provenance=_roi_provenance(),
            ),
            np.arange(32, 64, dtype=np.uint16).reshape(4, 8)
            if roi_id == "roi_2"
            else np.arange(32, dtype=np.uint16).reshape(4, 8),
        )
        project.assign_target_to_roi(
            TargetAssignment(
                roi_id=roi_id,
                target_id="target_1",
                value=value,
                assigned_at="2026-04-02T10:02:00+00:00",
            )
        )
    project.register_recipe("recipe_1", state.recipe.to_dict())
    project.create_sample_set(
        SampleSet(
            sample_set_id="sample_set_1",
            roi_ids=["roi_1", "roi_2"],
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
            created_at="2026-04-02T10:03:00+00:00",
        )
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warnings.append(str(args[2]))
    )

    def fake_exec(self):
        self._cv_combo.setCurrentIndex(self._cv_combo.findData(7))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(FitModelRunDialog, "exec", fake_exec)
    monkeypatch.setattr(
        "enginir.modeling.fit_model_run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not fit")),
    )

    view = ModelingView(state, project)
    view.fit_model_btn_clicked()

    assert warnings == [
        "7-fold grouped validation requires at least 7 ROI groups in the selected sample set."
    ]


@pytest.mark.parametrize("model_type", ["PLS", "PCR"])
def test_reopened_model_run_renders_every_diagnostic_and_selects_rois(qapp, tmp_path,
                                                                      model_type):
    """G15: a saved PLS/PCR run reopens with all plots drawn and clickable ROI points."""
    from enginir.viewer._model_diagnostics import ModelDiagnosticsWidget
    from enginir.viewer._modeling import ModelingView
    from enginir.viewer._state import ProjectState, ViewerState

    from .test_prediction_runs import _create_phase6_model_project

    path = str(tmp_path / "diagnostics.h5")
    _create_phase6_model_project(path)
    fitting = ProjectState()
    fitting.open_project(path)
    fitting.build_sample_set(target_id="target_moisture", preprocessing_recipe_id="recipe_none",
                             sample_set_id="sample_set_1")
    fitting.fit_model_run(sample_set_id="sample_set_1", signal_domain_id="signal_domain_1",
                          model_type=model_type, hyperparameters={"n_components": 2},
                          cv="logo", model_run_id="run")

    reopened = ProjectState()
    reopened.open_project(path)
    wavelengths = np.array([1000.0, 1100.0, 1200.0])
    view = ModelingView(ViewerState(np.zeros((4, 4, 3), np.uint16), wavelengths, {}), reopened)
    payload = view.workspace_diagnostics_payload()
    assert payload is not None
    n_rois = len(reopened.rois)

    widget = ModelDiagnosticsWidget()
    widget.update_data(payload)
    assert widget._pred_ax.collections[0].get_offsets().shape[0] == n_rois
    assert widget._resid_ax.collections[0].get_offsets().shape[0] == n_rois
    assert widget._target_dist_ax.patches
    assert widget._scores_ax.collections[0].get_offsets().shape[0] == n_rois
    for mode in ("coefficients", "loadings"):
        widget.set_spectral_mode(mode)
        line = widget._spectral_ax.lines[0]
        np.testing.assert_allclose(line.get_xdata(), wavelengths)
        np.testing.assert_allclose(line.get_ydata(), payload[mode])
    for ax in (widget._pred_ax, widget._resid_ax, widget._target_dist_ax, widget._scores_ax,
               widget._spectral_ax):
        assert not [t for t in ax.texts if "yet." in t.get_text()], "no empty state left"

    selected: list[str] = []
    widget.roi_selected.connect(selected.append)
    widget._canvas.draw()
    x, y = widget._scores_ax.collections[0].get_offsets()[1]
    px, py = widget._scores_ax.transData.transform((x, y))
    widget._on_canvas_click(types.SimpleNamespace(inaxes=widget._scores_ax, x=px, y=py))
    assert selected == [str(payload["roi_ids"][1])]
