"""Smoke tests for the split modeling view."""

from __future__ import annotations

import time
import types

import numpy as np
import pytest
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox

from enginir.project import (
    CalibrationSpec,
    DatasetRef,
    ModelRun,
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


def _labelled_rois(state, project, values=(3.2, 3.8), *, unlabelled=0):
    """A Reflectance calibration, target ``moisture`` and one ROI per value (plus unlabelled)."""
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
    for index in range(len(values) + unlabelled):
        roi_id = f"roi_{index + 1}"
        project.add_roi(
            TrainingROI(
                roi_id=roi_id,
                dataset_id="dataset_1",
                calibration_id="cal_1",
                name=roi_id,
                wavelengths=state.wavelengths.astype("float32"),
                n_pixels=4,
                spatial_bounds=(index, index, 0, 3),
                source_file="/tmp/cube.eiff",
                created_at="2026-04-02T10:01:30+00:00",
                provenance=_roi_provenance(),
            ),
            np.arange(32 * index, 32 * (index + 1), dtype=np.uint16).reshape(4, 8),
        )
        if index < len(values):
            project.assign_target_to_roi(
                TargetAssignment(
                    roi_id=roi_id,
                    target_id="target_1",
                    value=values[index],
                    assigned_at="2026-04-02T10:02:00+00:00",
                )
            )


def _fake_fit(captured: list[dict[str, object]], *, fail: bool = False):
    def fit(**kwargs):
        captured.append(kwargs)
        if fail:
            raise RuntimeError("fit exploded")
        return types.SimpleNamespace(
            model_run=ModelRun(
                model_run_id=f"model_{len(captured)}",
                sample_set_id=kwargs["sample_set"].sample_set_id,
                signal_domain_id=kwargs["signal_domain_id"],
                model_type="PLS",
                hyperparameters={"n_components": 1},
                cv_spec={},
                metrics={"rmsecv": 1.23},
                artifact_refs={},
                created_at="2026-04-02T10:04:00+00:00",
            ),
            predictor=object(),
        )

    return fit


def test_fit_dialog_starts_from_the_preview_and_explains_readiness(qapp, state_and_project):
    from enginir.viewer._modeling import FitModelRunDialog, ModelingView

    state, project = state_and_project
    _labelled_rois(state, project, unlabelled=1)
    project.define_target(TargetDefinition(
        target_id="target_2", name="fat", unit="%", value_type="float",
        missing_value_policy="allowed", created_at="2026-04-02T10:01:00+00:00"))
    state.set_preprocess_snv(True)
    view = ModelingView(state, project)
    dialog = FitModelRunDialog([("target_1", "moisture"), ("target_2", "fat")],
                               default_target_id="target_1", recipe=state.recipe,
                               default_signal_unit="Reflectance",
                               readiness=view.fit_readiness)
    assert dialog.target_id == "target_1" and dialog.recipe == state.recipe
    assert dialog.readiness_text.splitlines() == [
        "cube.eiff: 2 of 3 ROIs have a value",
        "The model uses 2 ROIs from 1 data file(s).",
    ]
    ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok.isEnabled()

    dialog._target_combo.setCurrentIndex(1)  # fat: no values yet
    assert dialog.readiness_text.startswith("cube.eiff: 0 of 3 ROIs have a value\nCannot fit:")
    assert not ok.isEnabled()


def test_a_fit_stores_its_sample_set_and_a_refit_reuses_it(qapp, state_and_project, monkeypatch):
    from enginir import PreprocessingRecipe
    from enginir._workflow import recipe_id_for
    from enginir.viewer._modeling import FitModelRunDialog, ModelingView

    state, project = state_and_project
    _labelled_rois(state, project)
    chosen = {"recipe": state.recipe}

    def fake_exec(self):
        self._signal_unit_combo.setCurrentIndex(self._signal_unit_combo.findText("Reflectance"))
        self.set_recipe(chosen["recipe"])
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(FitModelRunDialog, "exec", fake_exec)
    captured: list[dict[str, object]] = []
    monkeypatch.setattr("enginir.modeling.fit_model_run", _fake_fit(captured))

    view = ModelingView(state, project)
    view._selected_target_id = "target_1"
    view.fit_model_btn_clicked()
    _wait_until(qapp, lambda: "model_1" in project.model_runs)
    (sample_set,) = project.sample_sets.values()
    assert sample_set.roi_ids == ["roi_1", "roi_2"] and sample_set.target_id == "target_1"
    assert sample_set.preprocessing_recipe_id == recipe_id_for(state.recipe)
    assert captured[0]["cv"] == "auto"
    assert view._service_status.label.text() == "Service: model model_1"

    _wait_until(qapp, lambda: view._job_thread is None)
    view.fit_model_btn_clicked()  # the same choices: the same sample set
    _wait_until(qapp, lambda: "model_2" in project.model_runs)
    assert list(project.sample_sets) == [sample_set.sample_set_id]
    assert project.model_runs["model_2"].sample_set_id == sample_set.sample_set_id
    assert len(project.signal_domains) == 1
    assert project.model_runs["model_2"].signal_domain_id \
        == project.model_runs["model_1"].signal_domain_id

    _wait_until(qapp, lambda: view._job_thread is None)
    chosen["recipe"] = PreprocessingRecipe(snv=True)  # another preprocessing: a new sample set
    view.fit_model_btn_clicked()
    _wait_until(qapp, lambda: "model_3" in project.model_runs)
    snv_id = recipe_id_for(PreprocessingRecipe(snv=True))
    assert len(project.sample_sets) == 2 and snv_id in project.recipes
    assert project.sample_sets[project.model_runs["model_3"].sample_set_id] \
        .preprocessing_recipe_id == snv_id
    assert snv_id in captured[2]["recipes"]  # the fit was trained with the chosen recipe


def test_a_failed_fit_stores_nothing(qapp, state_and_project, monkeypatch):
    from enginir.viewer._modeling import FitModelRunDialog, ModelingView

    state, project = state_and_project
    _labelled_rois(state, project)
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warnings.append(str(args[2]))
    )
    state.set_y_unit("Reflectance")
    monkeypatch.setattr(FitModelRunDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    captured: list[dict[str, object]] = []
    monkeypatch.setattr("enginir.modeling.fit_model_run", _fake_fit(captured, fail=True))
    recipes_before = set(project.recipes)

    view = ModelingView(state, project)
    view.fit_model_btn_clicked()
    _wait_until(qapp, lambda: bool(warnings))

    assert captured and "fit exploded" in warnings[0]
    assert project.sample_sets == {} and set(project.recipes) == recipes_before
    assert project.model_runs == {}


def test_fit_model_group_validation_guard(qapp, state_and_project, monkeypatch):
    from enginir.viewer._modeling import FitModelRunDialog, ModelingView

    state, project = state_and_project
    _labelled_rois(state, project)
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
    assert project.sample_sets == {}, "a refused fit writes nothing"


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
    widget.resize(900, 600)
    widget.show()
    widget.update_data(payload)
    widget.grid.set_current_page("Spectral")
    qapp.processEvents()

    def axes(plot_id):
        panel = widget.grid.panel(plot_id)
        assert panel is not None
        return panel.figure.axes[0]

    assert axes("model.pred_vs_ref").collections[0].get_offsets().shape[0] == n_rois
    assert axes("model.residuals").collections[0].get_offsets().shape[0] == n_rois
    assert axes("model.target_distribution").patches
    assert axes("model.scores").collections[0].get_offsets().shape[0] == n_rois
    assert "training" in axes("model.scores").get_xlabel()
    for plot_id, key in (("model.coefficients", "coefficients"), ("model.loadings", "loadings")):
        line = axes(plot_id).lines[0]
        np.testing.assert_allclose(line.get_xdata(), wavelengths)
        np.testing.assert_allclose(line.get_ydata(), payload[key])
    for plot_id in ("model.pred_vs_ref", "model.residuals", "model.target_distribution",
                    "model.scores", "model.coefficients", "model.loadings"):
        assert not [t for t in axes(plot_id).texts if "yet." in t.get_text()], plot_id

    selected: list[str] = []
    widget.roi_selected.connect(selected.append)
    scores_panel = widget.grid.panel("model.scores")
    assert scores_panel is not None
    scores_ax = axes("model.scores")
    scores_panel.figure.canvas.draw()
    x, y = scores_ax.collections[0].get_offsets()[1]
    px, py = scores_ax.transData.transform((x, y))
    scores_panel._on_canvas_click(types.SimpleNamespace(inaxes=scores_ax, x=px, y=py))
    assert selected == [str(payload["roi_ids"][1])]
