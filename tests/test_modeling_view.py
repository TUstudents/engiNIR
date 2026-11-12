"""Smoke tests for the split modeling view."""

from __future__ import annotations

from enginir.project import FitSpec

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
            model_run=ModelRun(fit_spec=FitSpec(groups=tuple(range(len(kwargs["sample_set"].roi_ids)))).to_dict(), 
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
        "The model uses 2 ROIs from 1 dataset(s).",
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
    assert captured[0]["cv"] == "auto" and captured[0]["fit_spec"] is None
    assert captured[0]["hyperparameters"] == {"n_components": 6, "x_scaling": "center",
                                              "y_scaling": "uv"}
    named = project.model_runs["model_1"].name or "PLS model"
    assert view._service_status.label.text() == f"Service: model {named}", "a name, not an id"

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


def test_holding_out_samples_together_reaches_the_fit_as_its_fold_groups(
        qapp, state_and_project, monkeypatch):
    """"Hold out together: Sample label" with "Leave one out" fits with group_by sample."""
    from enginir.viewer._modeling import FitModelRunDialog, ModelingView

    state, project = state_and_project
    _labelled_rois(state, project)

    def fake_exec(self):
        self._signal_unit_combo.setCurrentIndex(self._signal_unit_combo.findText("Reflectance"))
        self._cv_combo.setCurrentIndex(self._cv_combo.findData("logo"))
        self._group_by_combo.setCurrentIndex(self._group_by_combo.findData("sample"))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(FitModelRunDialog, "exec", fake_exec)
    captured: list[dict[str, object]] = []
    monkeypatch.setattr("enginir.modeling.fit_model_run", _fake_fit(captured))
    view = ModelingView(state, project)
    view._selected_target_id = "target_1"
    view.fit_model_btn_clicked()
    _wait_until(qapp, lambda: "model_1" in project.model_runs)
    spec = captured[0]["fit_spec"]
    assert captured[0]["cv"] is None
    assert (spec.group_by, spec.fold_rule) == ("sample", "leave_one_group_out")


def test_the_fitting_section_becomes_the_fit_specification(qapp, state_and_project):
    """Pixels per ROI (at most the smallest ROI), scaling and DModX exclusion (contract §7)."""
    from enginir.viewer._modeling import FitModelRunDialog, ModelingView

    state, project = state_and_project
    _labelled_rois(state, project)
    view = ModelingView(state, project)
    dialog = FitModelRunDialog([("target_1", "moisture")], default_target_id="target_1",
                               recipe=state.recipe, default_signal_unit="Reflectance",
                               readiness=view.fit_readiness,
                               smallest_roi=view._smallest_labelled_roi)
    assert dialog.fit_spec is None, "the defaults need no fit specification"
    assert dialog.hyperparameters == {"n_components": 6, "x_scaling": "center",
                                      "y_scaling": "uv"}
    smallest = min(roi.n_pixels for roi in project.rois.values())
    assert dialog._pixels_spin.maximum() == smallest

    dialog._draw_check.setChecked(True)
    dialog._pixels_spin.setValue(smallest)
    dialog._seed_spin.setValue(3)
    dialog._exclude_check.setChecked(True)
    dialog._rounds_spin.setValue(2)
    dialog._share_spin.setValue(10)
    dialog._formula_combo.setCurrentIndex(dialog._formula_combo.findData("dmodx-2"))
    dialog._x_scaling_combo.setCurrentIndex(dialog._x_scaling_combo.findData("pareto"))
    dialog._cv_combo.setCurrentIndex(dialog._cv_combo.findData("logo"))
    spec = dialog.fit_spec
    assert (spec.pixels_per_roi, spec.seed, spec.fold_rule) == (smallest, 3,
                                                                 "leave_one_group_out")
    assert (spec.exclusion, spec.exclusion_rounds, spec.exclusion_max_fraction,
            spec.exclusion_formula, spec.exclusion_level) == ("dmodx", 2, 0.1, "dmodx-2", 0.99)
    assert dialog.hyperparameters["x_scaling"] == "pareto"

    dialog._model_type_combo.setCurrentText("PCR")
    assert "y_scaling" not in dialog.hyperparameters
    assert not dialog._y_scaling_combo.isEnabled()


def test_the_inspector_summarises_how_a_run_was_fitted(tmp_path):
    from enginir.io.project_store import load_model_run_artifact_refs
    from enginir.viewer._inspector import fit_summary
    from tests.test_exclusion import _project_with_an_odd_pixel

    project = _project_with_an_odd_pixel(tmp_path)
    model = project.fit("API", n_components=2, fit_spec={
        "pixels": {"per_roi": 10, "seed": 3},
        "exclusion": {"method": "dmodx", "formula": "dmodx-2", "level": 0.51,
                      "max_fraction": 0.1}})
    exclusion = load_model_run_artifact_refs(project.path, model.ref.model_run_id,
                                             keys=("exclusion",))["exclusion"]
    names = {roi_id: roi.name for roi_id, roi in project.workspace.rois.items()}
    lines = fit_summary(model.ref, names, exclusion)
    assert lines[0] == "Folds: round-robin, 7 folds; held out together: each ROI"
    assert lines[1] == "Pixels: 10 per ROI (seed 3)"
    assert lines[2] == "Scaling: X center, Y unit variance"
    assert lines[3].startswith("Software: enginir ") and "scikit-learn " in lines[3]
    assert lines[4].startswith("Pixel exclusion: DModX (weighted), level 0.51, 1 round(s), "
                               "at most 10% per ROI;")
    excluded = model.excluded_pixels
    assert lines[5] == "Excluded per ROI: " + ", ".join(
        f"{names[r]} {len(v)}" for r, v in excluded.items())
    assert any(line.startswith("Flagged ROIs") for line in lines)
    assert fit_summary(project.fit("API", n_components=2).ref, names)[4] \
        == "Pixel exclusion: none"


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
    """Cross-validation needs 2 ROIs; 7 folds over fewer ROIs hold out one ROI each."""
    from enginir.viewer._modeling import FitModelRunDialog, ModelingView

    state, project = state_and_project
    _labelled_rois(state, project, values=(3.2,))
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
        "Cross-validation requires at least 2 ROIs in the selected calibration set, it has 1."
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
    distribution = axes("model.target_distribution")
    assert distribution.patches or distribution.collections, "drawn as bins or as levels"
    assert axes("model.scores").collections[0].get_offsets().shape[0] == n_rois
    assert "calibration" in axes("model.scores").get_xlabel()
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


def test_auto_components_choose_by_a_rule(qapp, state_and_project):
    from enginir.viewer._modeling import FitModelRunDialog, ModelingView

    state, project = state_and_project
    _labelled_rois(state, project)
    view = ModelingView(state, project)
    dialog = FitModelRunDialog([("target_1", "moisture")], default_target_id="target_1",
                               recipe=state.recipe, default_signal_unit="Reflectance",
                               readiness=view.fit_readiness)
    assert not dialog._rule_combo.isEnabled()
    dialog._auto_check.setChecked(True)
    dialog._rule_combo.setCurrentIndex(dialog._rule_combo.findData("q2_limit"))
    dialog._max_spin.setValue(5)
    assert dialog.n_components == "auto" and not dialog._components_spin.isEnabled()
    assert dialog.hyperparameters["n_components"] == "auto"
    spec = dialog.fit_spec
    assert (spec.components_rule, spec.components_max) == ("q2_limit", 5)


def test_fit_again_presets_the_dialog_from_the_selected_model(tmp_path):
    """Every choice of a stored model comes back in the dialog, so OK refits it unchanged."""
    from enginir.viewer._app import ensure_app
    from enginir.viewer._modeling import FitModelRunDialog
    from tests.test_exclusion import _project_with_an_odd_pixel

    ensure_app()
    project = _project_with_an_odd_pixel(tmp_path)
    model = project.fit("API", n_components=3, x_scaling="pareto", cv="logo", fit_spec=None)
    dialog = FitModelRunDialog([("t", "API")], default_target_id="t",
                               recipe=__import__("enginir").PreprocessingRecipe(),
                               default_signal_unit="Absorbance",
                               readiness=lambda _t: ("", True))
    dialog.apply_model_run(model.ref)
    assert dialog.n_components == 3 and dialog.model_type == "PLS"
    assert dialog.hyperparameters == {"n_components": 3, "x_scaling": "pareto",
                                      "y_scaling": "uv"}
    assert dialog.cv == "logo"

    excluded = project.fit("API", n_components=2, fit_spec={
        "pixels": {"per_roi": 6, "seed": 4}, "exclusion": {"method": "dmodx", "rounds": 2}})
    dialog.apply_model_run(excluded.ref)
    spec = dialog.fit_spec
    assert (spec.pixels_per_roi, spec.seed, spec.exclusion, spec.exclusion_rounds) == (
        6, 4, "dmodx", 2)


def test_the_diagnostics_payload_carries_the_curve_and_the_training_distances(tmp_path):
    from enginir.viewer._modeling import _component_payload
    from tests.test_exclusion import _project_with_an_odd_pixel

    project = _project_with_an_odd_pixel(tmp_path)
    model = project.fit("API", n_components=2)
    from enginir.io.project_store import load_model_run_artifact_refs

    refs = load_model_run_artifact_refs(project.path, model.ref.model_run_id)
    artifact = {**refs["prediction_model"], **refs["diagnostics"],
                "component_curve": refs["component_curve"]}
    payload = _component_payload(model.ref, artifact)
    assert payload["n_components"] == 2 and "rmsecv" in payload["component_curve"]
    assert payload["loadings_matrix"].shape[1] == 2
    assert payload["dmodx_pixels"].shape == payload["t2_pixels"].shape == (96,)
    assert {"t2_crit_2_099", "dcrit_099"} <= set(payload)
    assert "rmsee" in payload["fit_statistics"]


def test_the_summary_names_the_model_and_its_validation_and_the_check_logs_no_error(
        qapp, state_and_project, monkeypatch):
    """Names, not ids; the stored folds as text; asking whether the Explore signal can be
    modelled answers in the status line without an ERROR in the log (a refused signal
    domain used to be logged by its constructor)."""
    from loguru import logger

    from enginir.viewer._modeling import FitModelRunDialog, ModelingView, describe_cv

    state, project = state_and_project
    _labelled_rois(state, project)

    def fake_exec(self):
        self._signal_unit_combo.setCurrentIndex(self._signal_unit_combo.findText("Reflectance"))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(FitModelRunDialog, "exec", fake_exec)
    captured: list[dict[str, object]] = []
    monkeypatch.setattr("enginir.modeling.fit_model_run", _fake_fit(captured))
    view = ModelingView(state, project)
    view._selected_target_id = "target_1"
    view.fit_model_btn_clicked()
    _wait_until(qapp, lambda: "model_1" in project.model_runs)
    _wait_until(qapp, lambda: view._job_thread is None)
    run = project.model_runs["model_1"]
    run.cv_spec = {"type": "RoundRobinGroups", "n_splits": 2, "n_groups": 2,
                   "grouping_unit": "roi"}
    view._invalidate_diagnostics_cache()
    summary = view.workspace_summary()
    assert summary["cv"] == "2 folds, round-robin over 2 samples"
    assert "model_1" not in summary["model"] + summary["readiness"]
    assert summary["sample_set"] == "2 ROIs"
    assert describe_cv({"cv": 5}) == "5" and describe_cv({}) == "-"

    errors: list[str] = []
    sink = logger.add(lambda m: errors.append(m), level="ERROR")
    try:
        # These ROIs have references: they cannot be modelled in raw counts.
        state.set_y_unit("DN")
        view._update_readiness_status()
    finally:
        logger.remove(sink)
    assert errors == []
    sample_set = next(iter(project.sample_sets.values()))
    assert "choose Reflectance or Absorbance" in (view._validate_signal_domain_preflight(
        sample_set=sample_set, required_signal_unit="DN") or "")


def test_few_reference_levels_get_one_bar_each_and_unit_1_is_not_printed():
    """Four levels (1, 4, 7, 10) in four equal bins over 1-10 made one flat block."""
    from matplotlib.figure import Figure

    from enginir.viewer._model_diagnostics import (
        _render_pred_vs_ref,
        _render_target_distribution,
        _table_target_distribution,
    )
    from enginir.viewer._theme import panel_colors

    y = np.repeat([1.0, 4.0, 7.0, 10.0], 5)
    payload = {"y_true_roi": y, "y_pred_roi": y + 0.3, "target_unit": "1"}
    fig = Figure()
    _render_target_distribution(fig, payload, panel_colors())
    (bars,) = fig.axes[0].collections
    segments = bars.get_segments()
    assert [float(seg[0][0]) for seg in segments] == [1.0, 4.0, 7.0, 10.0], "one bar per level"
    assert [float(seg[1][1]) for seg in segments] == [5, 5, 5, 5]
    assert not fig.axes[0].patches, "no binned block"
    table = _table_target_distribution(payload)
    assert [c.name for c in table.columns] == ["value", "roi_count"]

    continuous = {"y_true_roi": np.linspace(0, 10, 40), "y_pred_roi": np.linspace(0, 10, 40)}
    assert _table_target_distribution(continuous).columns[0].name == "bin_start"

    for unit, shown in (("1", False), ("%", True)):
        fig = Figure()
        _render_pred_vs_ref(fig, {**payload, "target_unit": unit}, panel_colors())
        stats = fig.axes[0].texts[0].get_text()
        assert (f" {unit}\n" in stats) is shown, (unit, stats)


def test_the_roi_ellipse_is_the_t2_limit_of_the_rois_and_both_are_in_the_legend():
    """ROI mean scores can be correlated: every ellipse point has the ROI-level T² limit
    under the ROIs' own covariance (an axis-aligned ellipse would not)."""
    from matplotlib.figure import Figure
    from scipy.stats import f

    from enginir.viewer._model_diagnostics import _render_scores, roi_t2_ellipse
    from enginir.viewer._theme import panel_colors

    rng = np.random.default_rng(4)
    lv1 = rng.normal(0, 1.0, 20)
    scores = np.column_stack([lv1, -0.4 * lv1 + rng.normal(0, 0.1, 20)])
    ellipse = roi_t2_ellipse(scores, 0.99)
    assert ellipse is not None
    centred = ellipse - scores.mean(axis=0)
    t2 = np.einsum("ij,jk,ik->i", centred, np.linalg.inv(np.cov(scores, rowvar=False)), centred)
    np.testing.assert_allclose(t2, 2 * 19 / 18 * f.ppf(0.99, 2, 18), rtol=1e-9)
    assert roi_t2_ellipse(scores[:3], 0.99) is None, "too few ROIs for a limit"

    y = np.linspace(1, 10, 20)
    payload = {"y_true_roi": y, "y_pred_roi": y, "scores_roi": scores,
               "score_variances": np.array([4.0, 1.0]), "t2_crit_2_095": 5.99,
               "t2_crit_2_099": 9.21}
    fig = Figure()
    _render_scores(fig, payload, panel_colors())
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert labels == ["T² 95 % pixels", "T² 99 % pixels", "T² 95 % ROIs", "T² 99 % ROIs"]


def test_band_importance_plots_keep_the_theme_and_name_band_axes():
    """The mean spectrum's twin axes lay a white background over VIP in the dark theme, and
    a file without wavelengths was labelled 'Wavelength' / '175 nm'."""
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure

    from enginir.viewer._model_diagnostics import _render_vip, _render_vip_sorted
    from enginir.viewer._theme import panel_colors

    colors = panel_colors()
    bands = np.arange(30.0)
    payload = {"vip": np.linspace(2.0, 0.5, 30), "spectral_axis": bands,
               "mean_spectrum": np.sin(bands / 5.0)}
    fig = Figure()
    _render_vip(fig, payload, colors)
    for ax in fig.axes:
        assert to_rgba(ax.get_facecolor()) == to_rgba(colors["ax_bg"]) \
            or not ax.patch.get_visible()
    assert fig.axes[0].get_xlabel() == "Band"
    fig = Figure()
    _render_vip_sorted(fig, payload, colors)
    listing = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "band 0" in listing and "nm" not in listing


def test_distance_histograms_show_the_tail_as_one_bar_and_rois_by_their_share():
    """One far pixel put every other pixel into the first of 40 bins; the per-ROI shares
    (stored by the fit) are sorted, clickable and exported."""
    from matplotlib.figure import Figure

    from enginir.viewer._model_diagnostics import (
        _distance_spec,
        _render_roi_distance,
        _table_roi_distance,
    )
    from enginir.viewer._theme import panel_colors

    rng = np.random.default_rng(1)
    pixels = np.concatenate([rng.gamma(4.0, 0.2, 5000), [35.0]])
    spec = _distance_spec("dmodx", "dcrit_099", "x", "x", "DModX")
    payload = {"dmodx_pixels": pixels, "dcrit_099": 1.5}
    fig = Figure()
    spec.render(fig, payload, panel_colors())
    main, overflow = fig.axes[0].patches
    edges = main.get_data().edges
    assert edges[-1] < 5.0, "the bins stop at the 99.5th percentile, not at the far pixel"
    assert overflow.get_data().values[0] == int(np.sum(pixels > edges[-1]))
    text = fig.axes[0].texts[0].get_text()
    assert "(1 % expected)" in text and "max 35" in text
    rows = spec.table(payload).columns[2].values
    assert int(np.sum(rows)) == pixels.size, "every pixel is in the CSV"

    ids = np.array(["a", "b", "c"], dtype=object)
    rois = {"y_true_roi": np.array([1.0, 4.0, 7.0]), "y_pred_roi": np.array([1.0, 4.0, 7.0]),
            "roi_ids": ids, "roi_names": ("API_L1_01", "API_L4_01", "API_L7_01"),
            "roi_share_above_dcrit": np.array([0.02, 0.9, 0.05]),
            "roi_share_above_t2crit": np.array([0.01, 0.3, 0.02])}
    fig = Figure()
    (points,) = _render_roi_distance(fig, rois, panel_colors())
    assert points.roi_ids == ("b", "c", "a") and list(points.y) == [90.0, 5.0, 2.0]
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == ["L4_01", "L7_01", "L1_01"]
    table = _table_roi_distance(rois)
    assert [c.name for c in table.columns][:3] == ["roi_id", "rank", "dmodx_share_above_percent"]


def test_the_components_page_explains_each_rule_and_the_guard():
    from matplotlib.figure import Figure

    from enginir.viewer._model_diagnostics import _render_components, _render_per_component
    from enginir.viewer._theme import panel_colors
    from tests.test_component_selection import EVIDENCE

    curve = {**EVIDENCE, "suggested_one_se": 7, "suggested_min_rmsecv": 15,
             "suggested_q2_limit": 1, "suggested_f_test": 6, "suggested_q2_lookahead": 6}
    payload = {"component_curve": curve, "n_components": 6, "target_unit": "1",
               "components_choice": {"rule": "one_se", "noise_guard": "on", "rule_choice": 7,
                                     "n_groups": 20}}
    fig = Figure()
    _render_components(fig, payload, panel_colors())
    text = fig.axes[0].texts[0].get_text()
    assert "noise guard: one SE 7 → 6" in text
    assert "Q² limit: 1 (Q²₂ = -0.0" in text and "≤ 0.05)" in text, "the failing component"
    assert "F-test: 6 (PRESS ratio < 1.36)" in text
    fig = Figure()
    _render_per_component(fig, payload, panel_colors())
    q2_bars = fig.axes[0].collections[0]
    grey = [tuple(c) for c in q2_bars.get_colors()]
    assert grey[6] == grey[14] != grey[5], "components 7-15 are noise and drawn grey"


def test_the_fit_dialog_offers_the_guard_and_warns_about_the_folds(qapp):
    from enginir.viewer._modeling import FitModelRunDialog

    seen: list[tuple] = []

    def warner(target_id, spec, cv):
        seen.append((target_id, spec, cv))
        return ["the ROIs of one dataset share one value"]

    dlg = FitModelRunDialog([("t1", "Layer")], default_target_id="t1",
                            recipe=__import__("enginir").PreprocessingRecipe(),
                            default_signal_unit="Absorbance",
                            readiness=lambda _t: ("Ready.", True), fold_warner=warner)
    try:
        assert "Validation likely optimistic: the ROIs of one dataset" in dlg.readiness_text
        dlg._auto_check.setChecked(True)
        spec = dlg.fit_spec
        assert spec is not None and spec.components_noise_guard == "on"
        dlg._guard_check.setChecked(False)
        assert dlg.fit_spec.components_noise_guard == "off"
        dlg._rule_combo.setCurrentIndex(dlg._rule_combo.findData("f_test"))
        assert dlg.fit_spec.components_rule == "f_test"
        calls = len(seen)
        dlg._group_by_combo.setCurrentIndex(dlg._group_by_combo.findData("sample"))
        assert len(seen) > calls and seen[-1][1].group_by == "sample", "follows the grouping"
    finally:
        dlg.close()


def test_every_model_plot_has_help_that_the_user_guide_repeats(qapp):
    """The ? text and the user guide are one text (model-workspace-diagnostics contract)."""
    import re
    from pathlib import Path

    from enginir.viewer._model_diagnostics import ModelDiagnosticsWidget
    from enginir.viewer._plot_registry import PLOT_SPECS, WORKSPACE_PAGES

    guide = re.sub(r"\s+", " ", (Path(__file__).parents[1] / "docs" / "user_guide"
                                 / "model_plots.rst").read_text(encoding="utf-8"))
    model_ids = [slot.plot_id for page in WORKSPACE_PAGES["Model"] for slot in page.slots]
    for plot_id in model_ids:
        text = PLOT_SPECS[plot_id].help
        assert text, f"{plot_id} has no help text"
        assert re.sub(r"\s+", " ", text) in guide, f"the user guide lacks the help of {plot_id}"
    widget = ModelDiagnosticsWidget()
    try:
        panel = widget.grid.panel(model_ids[0])
        assert panel is not None and panel._help_button is not None
        assert panel._help_button.toolTip() == PLOT_SPECS[model_ids[0]].help
        panel.show_help()  # shows a tooltip; must not raise
    finally:
        widget.close()


def test_fit_again_presets_the_models_signal_unit_and_its_full_seed(
        qapp, state_and_project, monkeypatch):
    """Fit Again repeats the model's signal unit, not Explore's (B1), and keeps a seed above
    2**31 - 1 (B2): the dialog must not change what the refit draws."""
    from enginir.viewer._modeling import FitModelRunDialog, ModelingView

    state, project = state_and_project
    _labelled_rois(state, project)
    seen: dict[str, object] = {}

    def fit_on_reflectance(self):
        self._signal_unit_combo.setCurrentIndex(self._signal_unit_combo.findText("Reflectance"))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(FitModelRunDialog, "exec", fit_on_reflectance)
    monkeypatch.setattr("enginir.modeling.fit_model_run", _fake_fit([]))
    view = ModelingView(state, project)
    view._selected_target_id = "target_1"
    view.fit_model_btn_clicked()
    _wait_until(qapp, lambda: "model_1" in project.model_runs)
    _wait_until(qapp, lambda: view._job_thread is None)
    assert state.y_unit != "Reflectance", "Explore shows another signal"

    def look(self):
        seen["signal_unit"] = self.signal_unit
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(FitModelRunDialog, "exec", look)
    monkeypatch.setattr(view, "_resolved_workspace_model_run",
                        lambda: project.model_runs["model_1"])
    view.fit_again_btn_clicked()
    assert seen["signal_unit"] == "Reflectance"

    dialog = FitModelRunDialog([("t", "API")], default_target_id="t",
                               recipe=__import__("enginir").PreprocessingRecipe(),
                               default_signal_unit="DN", readiness=lambda _t: ("", True))
    dialog._draw_check.setChecked(True)
    dialog._seed_spin.setValue(2**40)
    assert dialog.fit_spec.seed == 2**40


def test_the_fit_summary_shows_the_fold_warnings_and_wraps_long_values(qapp):
    """B6: a model's fold warnings are shown, also when its metric rows equal the previous
    model's; long summary values wrap instead of being cut off."""
    import numpy as np

    from enginir.viewer._viewer import HSIViewer

    viewer = HSIViewer(np.zeros((4, 4, 4), np.uint16), np.arange(4.0), {})
    viewer._workspace_tabs.setCurrentIndex(1)  # the Model page is built when first shown
    qapp.processEvents()
    rows = {"roi_metrics": [("Q² (CV R²)", "0.9")], "pixel_metrics": [],
            "r2": "0.9", "rmsecv": "1", "bias": "0"}
    viewer._set_metric_rows({**rows, "warnings": ""})
    assert not viewer._model_warnings.isVisibleTo(viewer)
    viewer._set_metric_rows({**rows, "warnings": "fewer than 7 fold groups"})
    assert viewer._model_warnings.isVisibleTo(viewer)
    assert "fewer than 7 fold groups" in viewer._model_warnings.text()
    assert viewer._model_card_cv_value.wordWrap()
    viewer.close()


@pytest.mark.parametrize("width", [2.2, 3.5, 6.0])
def test_the_distance_notes_stay_inside_their_plot(width):
    """W5: the notes of a distance histogram (limit, share above, overflow bar) are one block
    inside the axes, also in a narrow panel; an undefined (NaN) limit is not drawn."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from enginir.viewer._model_diagnostics import _distance_spec
    from enginir.viewer._theme import panel_colors

    values = np.concatenate([np.random.default_rng(0).gamma(4, 0.25, 5000), [36.3] * 20])
    spec = _distance_spec("dmodx", "dcrit_099", "model.dmodx", "DModX", "Normalized DModX")
    fig = Figure(figsize=(width, 2.4))
    canvas = FigureCanvasAgg(fig)
    spec.render(fig, {"dmodx_pixels": values, "dcrit_099": 1.17}, panel_colors())
    canvas.draw()
    renderer = canvas.get_renderer()
    (ax,) = fig.axes
    notes = [t for t in ax.texts if "limit 0.99" in t.get_text()]
    assert len(notes) == 1 and "last bar" in notes[0].get_text(), "one block with every note"
    box, axes_box = notes[0].get_window_extent(renderer), ax.get_window_extent(renderer)
    assert box.x0 >= axes_box.x0 - 1 and box.x1 <= axes_box.x1 + 1

    fig = Figure(figsize=(width, 2.4))
    FigureCanvasAgg(fig)
    spec.render(fig, {"dmodx_pixels": values, "dcrit_099": float("nan")}, panel_colors())
    assert not any("limit 0.99" in t.get_text() for t in fig.axes[0].texts)
