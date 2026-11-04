"""Export of all workspace plots and the plot layout in user settings (Plot Panel Policy)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


def _viewer(qapp):
    from enginir.viewer._viewer import HSIViewer

    cube = np.random.default_rng(2).integers(500, 3000, size=(20, 10, 6), dtype=np.uint16)
    viewer = HSIViewer(cube, np.linspace(1000.0, 1700.0, 6, dtype=np.float32))
    viewer.resize(1300, 850)
    viewer.show()
    qapp.processEvents()
    return viewer


@pytest.fixture()
def fitted_viewer(qapp, tmp_path):
    from .test_prediction_runs import _create_phase6_model_project

    path = str(tmp_path / "layout.h5")
    _create_phase6_model_project(path)
    viewer = _viewer(qapp)
    project = viewer.project_state
    project.open_project(path)
    project.build_sample_set(target_id="target_moisture", preprocessing_recipe_id="recipe_none",
                             sample_set_id="sample_set_1")
    project.fit_model_run(sample_set_id="sample_set_1", signal_domain_id="signal_domain_1",
                          model_type="PLS", hyperparameters={"n_components": 2}, cv="logo",
                          model_run_id="model_1")
    project.active_model_name = "model_1"
    viewer._workspace_tabs.setCurrentWidget(viewer._modeling_view)
    qapp.processEvents()
    yield viewer
    viewer.close()


def test_export_plots_writes_every_model_plot_including_unopened_pages(fitted_viewer, tmp_path):
    grid = fitted_viewer._model_diagnostics_widget.grid
    assert grid.current_page == "Fit" and grid.panel("model.loadings") is None  # not built

    out = tmp_path / "plots"
    out.mkdir()
    written, skipped = fitted_viewer.export_workspace_plots(out)
    plots = ("model.pred_vs_ref", "model.residuals", "model.target_distribution",
             "model.scores", "model.coefficients", "model.loadings")
    assert sorted(written) == sorted(f"{p}{s}" for p in plots for s in (".png", ".csv"))
    assert skipped == []
    assert grid.current_page == "Fit", "exporting does not switch the page"
    csv_text = (out / "model.loadings.csv").read_text(encoding="utf-8")
    assert "# model_run_id: model_1" in csv_text


def test_export_plots_skips_plots_without_data_and_needs_a_folder(qapp, tmp_path):
    viewer = _viewer(qapp)
    written, skipped = viewer.export_workspace_plots(tmp_path)
    # Histogram and RAW are hidden docks by default and have never drawn; there is
    # no project, so the Project ROIs plots have nothing to show.
    assert sorted(skipped) == ["explore.histogram", "explore.raw", "explore.roi_means",
                               "explore.roi_mosaic", "explore.roi_pixels"]
    assert {"explore.image.png", "explore.spectrum.csv", "explore.sensor.png",
            "explore.spatial.csv"} <= set(written)
    with pytest.raises(ValueError, match="does not exist"):
        viewer.export_workspace_plots(tmp_path / "missing")
    viewer.close()


def test_export_plots_asks_before_replacing_files(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    viewer = _viewer(qapp)
    (tmp_path / "explore.image.png").write_bytes(b"old")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    asked: list[str] = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda _p, _t, text, *a, **k: asked.append(text) or QMessageBox.StandardButton.No))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    viewer._action_export_plots()
    assert asked and "replaced" in asked[0]
    assert (tmp_path / "explore.image.png").read_bytes() == b"old"
    assert not (tmp_path / "explore.spectrum.csv").exists()
    viewer.close()


def test_maximized_plots_are_restored_in_a_new_window_until_reset(fitted_viewer, qapp):
    grid = fitted_viewer._model_diagnostics_widget.grid
    grid.maximize("model.residuals")
    fitted_viewer._workspace_tabs.setCurrentWidget(fitted_viewer._eda_view)
    qapp.processEvents()
    fitted_viewer._toggle_explore_maximized("spectrum")
    project_path = fitted_viewer.project_state.project_path

    second = _viewer(qapp)
    assert second._explore_maximized == "spectrum"
    assert not second._canvas_frame.isVisible()
    second.project_state.open_project(project_path)
    second._workspace_tabs.setCurrentWidget(second._modeling_view)
    qapp.processEvents()
    assert second._model_diagnostics_widget.grid.maximized_plot() == "model.residuals"

    from enginir.viewer import _layout_settings

    _layout_settings.save_maximized("Predict", "predict.histogram")  # Predict never opened here
    second.reset_layout()
    assert second._model_diagnostics_widget.grid.maximized_plot() is None
    second.close()
    third = _viewer(qapp)
    assert third._explore_maximized is None
    third._workspace_tabs.setCurrentWidget(third._modeling_view)
    qapp.processEvents()
    assert third._model_diagnostics_widget.grid.maximized_plot() is None
    third._workspace_tabs.setCurrentWidget(third._prediction_view)
    qapp.processEvents()
    assert third._predict_graph_panel.maximized is None
    third.close()


def test_layout_settings_tolerate_a_damaged_file():
    from enginir.viewer import _layout_settings
    from enginir.viewer._recent import config_dir

    config_dir().mkdir(parents=True, exist_ok=True)
    (config_dir() / "plot_layout.json").write_text("{not json", encoding="utf-8")
    assert _layout_settings.load_maximized() == {}
    _layout_settings.save_maximized("Model", "model.scores")
    assert _layout_settings.load_maximized() == {"Model": "model.scores"}
    assert os.path.exists(config_dir() / "plot_layout.json")
