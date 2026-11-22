"""Plot panels and plot grids: the Plot panel policy for every plot, export, maximize, lazy drawing and layout."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import types
from matplotlib.image import imread
import numpy as np
import pytest


# -- Plot panels and plot grids: export, maximize and lazy drawing (Plot panel policy) ----


def _scatter_spec(calls: list[str] | None = None, plot_id: str = "test.scatter"):
    """Reference vs prediction per ROI: the shape of the Model diagnostics."""
    from enginir.viewer._plot_panel import PlotColumn, PlotSpec, PlotTable, RoiPoints
    from enginir.viewer._ui_helpers import style_axes

    def render(fig, payload, colors):
        if calls is not None:
            calls.append(plot_id)
        ax = fig.add_subplot(111)
        style_axes(ax, colors)
        ax.scatter(payload["x"], payload["y"], color=colors["accent"])
        ax.set_xlabel("Reference (%)")
        return [RoiPoints(ax, np.asarray(payload["x"]), np.asarray(payload["y"]),
                          tuple(payload["roi_ids"]))]

    def table(payload):
        return PlotTable(columns=(
            PlotColumn("roi_name", payload["roi_names"]),
            PlotColumn("roi_id", payload["roi_ids"]),
            PlotColumn("reference", payload["x"], "%"),
            PlotColumn("predicted", payload["y"], "%"),
        ), metadata={"components": 3})

    return PlotSpec(plot_id, "Predicted vs Reference", render, table)


def _payload(n: int = 4, offset: float = 0.0) -> dict:
    x = np.linspace(1.0, 4.0, n, dtype=np.float32) + np.float32(1 / 3) + offset
    return {"x": x, "y": x + np.float32(0.25), "roi_ids": [f"id-{i}" for i in range(n)],
            "roi_names": [f"ROI, {i}" for i in range(n)]}


def _shown_panel(spec, payload, **kwargs):
    from enginir.viewer._plot_panel import PlotPanel

    panel = PlotPanel(spec)
    panel.resize(500, 350)
    panel.show()
    panel.update_data(payload, **kwargs)
    return panel


def test_plot_tables_reject_inconsistent_shapes():
    from enginir.viewer._plot_panel import PlotColumn, PlotTable

    with pytest.raises(ValueError, match="differ in length"):
        PlotTable(columns=(PlotColumn("a", [1, 2]), PlotColumn("b", [1, 2, 3])))
    with pytest.raises(ValueError, match="2-D"):
        PlotTable(matrix=np.zeros(3))


def test_images_are_saved_in_print_style_whatever_the_theme(qapp, tmp_path, monkeypatch):
    from enginir.viewer import _theme

    monkeypatch.setattr(_theme, "_current_dark", True)  # dark screen theme
    panel = _shown_panel(_scatter_spec(), _payload())
    png, svg = tmp_path / "plot.png", tmp_path / "plot.svg"
    panel.export_image(png)
    panel.export_image(svg)

    image = imread(png)
    np.testing.assert_allclose(image[0, 0, :3], 1.0)  # white print background
    # Same size as on screen, at 300 dpi.
    assert image.shape[1] == round(panel.figure.get_size_inches()[0] * 300)
    assert "<svg" in svg.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="png or .svg"):
        panel.export_image(tmp_path / "plot.jpg")

    from matplotlib.colors import to_hex
    from matplotlib.figure import Figure

    saved: list[Figure] = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **k: saved.append(fig))
    panel.export_image(png)
    ax = saved[0].axes[0]
    assert to_hex(ax.get_facecolor()) == "#ffffff"  # plot area, not only the margin
    assert to_hex(ax.xaxis.label.get_color()) == _theme.print_colors()["fg"]


def test_an_empty_panel_shows_why_and_cannot_be_exported(qapp, tmp_path):
    panel = _shown_panel(_scatter_spec(), None, empty_reason="Fit a model first.")
    texts = [text.get_text() for ax in panel.figure.axes for text in ax.texts]
    assert "Fit a model first." in texts
    actions = panel.findChildren(type(panel._csv_action))
    export_actions = [a for a in actions if a.text() != ""]
    assert export_actions and all(not a.isEnabled() for a in export_actions)
    assert all(a.toolTip() == "No data to export." for a in export_actions)
    with pytest.raises(ValueError, match="no data"):
        panel.export_csv(tmp_path / "empty.csv")
    assert not (tmp_path / "empty.csv").exists()

    panel.update_data(_payload())
    assert all(a.isEnabled() for a in export_actions)


def test_a_large_export_asks_before_writing(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    from enginir.viewer import _plot_panel

    monkeypatch.setattr(_plot_panel, "LARGE_EXPORT_VALUES", 10)
    path = tmp_path / "large.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    asked: list[str] = []

    def question(_parent, _title, text, *args, **kwargs):
        asked.append(text)
        return answers.pop(0)

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    panel = _shown_panel(_scatter_spec(), _payload(n=4))  # 16 values
    panel._csv_action.trigger()
    assert asked and not path.exists()  # declined
    panel._csv_action.trigger()
    assert path.exists()


def test_clicking_a_point_selects_its_roi(qapp):
    payload = _payload()
    panel = _shown_panel(_scatter_spec(), payload)
    qapp.processEvents()
    panel.figure.canvas.draw()
    ax = panel.figure.axes[0]
    px, py = ax.transData.transform((payload["x"][2], payload["y"][2]))
    selected: list[str] = []
    panel.roi_selected.connect(selected.append)
    panel._on_canvas_click(types.SimpleNamespace(inaxes=ax, x=px, y=py))
    panel._on_canvas_click(types.SimpleNamespace(inaxes=ax, x=px + 100, y=py + 100))
    assert selected == ["id-2"]  # the far click selects nothing


def _grid(calls: list[str]):
    from enginir.viewer._plot_grid import PlotGrid, PlotPage, PlotSlot

    specs = {plot_id: _scatter_spec(calls, plot_id) for plot_id in ("a", "b", "c")}
    pages = (PlotPage("Fit", (PlotSlot("a", 0, 0), PlotSlot("b", 0, 1))),
             PlotPage("Spectral", (PlotSlot("c", 0, 0),)))
    grid = PlotGrid(pages, specs)
    grid.resize(900, 600)
    grid.show()
    return grid


def test_double_clicking_the_header_toggles_maximize(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    from enginir.viewer._plot_panel import _PlotHeader

    grid = _grid([])
    panel = grid.panel("a")
    assert panel is not None
    header = panel.findChild(_PlotHeader)
    QTest.mouseDClick(header, Qt.MouseButton.LeftButton)
    assert grid.maximized_plot() == "a"
    QTest.mouseDClick(header, Qt.MouseButton.LeftButton)
    assert grid.maximized_plot() is None


def test_pages_and_registry_reject_bad_definitions():
    from enginir.viewer import _plot_registry
    from enginir.viewer._plot_grid import PlotGrid, PlotPage, PlotSlot

    with pytest.raises(ValueError, match="overlaps"):
        PlotPage("Fit", (PlotSlot("a", 0, 0, column_span=2), PlotSlot("b", 0, 1)))
    with pytest.raises(ValueError, match="twice"):
        PlotPage("Fit", (PlotSlot("a", 0, 0), PlotSlot("a", 0, 1)))
    with pytest.raises(ValueError, match="unknown plots"):
        PlotGrid((PlotPage("Fit", (PlotSlot("missing", 0, 0),)),), {})
    with pytest.raises(ValueError, match="unregistered"):
        _plot_registry.register_pages("test", (PlotPage("Fit", (PlotSlot("missing", 0, 0),)),))
    spec = _scatter_spec(plot_id="test.duplicate")
    _plot_registry.register_plot(spec)
    try:
        with pytest.raises(ValueError, match="already registered"):
            _plot_registry.register_plot(spec)
    finally:
        del _plot_registry.PLOT_SPECS["test.duplicate"]


# -- Every registered plot meets the Plot panel policy ---------------------------------


def _model_payload() -> dict:
    rng = np.random.default_rng(7)
    n_rois, n_bands = 12, 40
    y_true = rng.normal(10.0, 2.0, n_rois).astype(np.float32)
    return {
        "target_unit": "%",
        "y_true_roi": y_true,
        "y_pred_roi": (y_true + rng.normal(0.0, 0.4, n_rois)).astype(np.float32),
        "roi_ids": np.array([f"roi-{i}" for i in range(n_rois)], dtype=object),
        "roi_names": tuple(f"Tablet {i}" for i in range(n_rois)),
        "scores_roi": rng.normal(size=(n_rois, 3)),
        "spectral_axis": np.linspace(1000.0, 1700.0, n_bands, dtype=np.float32),
        "coefficients": rng.normal(size=n_bands).astype(np.float32),
        "loadings": rng.normal(size=n_bands).astype(np.float32),
        "loadings_matrix": rng.normal(size=(n_bands, 3)).astype(np.float32),
        "coef_cs": rng.normal(size=n_bands).astype(np.float32),
        "vip": np.abs(rng.normal(1.0, 0.4, size=n_bands)).astype(np.float32),
        "selectivity_ratio": np.abs(rng.normal(size=n_bands)).astype(np.float32),
        "mean_spectrum": rng.normal(1.0, 0.1, size=n_bands).astype(np.float32),
        "n_components": 3,
        "fit_statistics": {"rmsee": 0.3, "r2_adj": 0.9},
        "component_curve": {"rmsecv": np.array([1.0, 0.6, 0.45, 0.47]),
                            "rmsecv_se": np.array([0.1, 0.06, 0.05, 0.05]),
                            "q2_cum": np.array([0.6, 0.85, 0.9, 0.89]),
                            "r2y_cum": np.array([0.65, 0.9, 0.95, 0.96]),
                            "r2y_cum_roi": np.array([0.7, 0.92, 0.96, 0.97]),
                            "q2": np.array([0.6, 0.62, 0.33, -0.1]),
                            "rmsee_roi": np.array([0.9, 0.55, 0.4, 0.42]),
                            "dw_weights": np.array([0.01, 0.02, 0.2, 0.9]),
                            "noise": np.array([False, False, False, True]),
                            "r2x_cum": np.array([0.5, 0.8, 0.9, 0.95]),
                            "suggested_one_se": 3, "suggested_min_rmsecv": 3,
                            "suggested_q2_limit": 3},
        "score_variances": np.array([1.0, 0.5]),
        "t2_crit_2_095": 6.1, "t2_crit_2_099": 9.4, "t2_crit_095": 11.5, "dcrit_095": 1.3,
        "t2_pixels": rng.chisquare(3, 500).astype(np.float32),
        "dmodx_pixels": rng.gamma(4.0, 0.25, 500).astype(np.float32),
        "roi_share_above_dcrit": rng.uniform(0.0, 0.2, n_rois).astype(np.float32),
        "roi_share_above_t2crit": rng.uniform(0.0, 0.1, n_rois).astype(np.float32),
    }


def _predict_payload() -> dict:
    values = np.random.default_rng(3).normal(12.0, 1.5, (40, 25)).astype(np.float32)
    values[0, 0] = np.nan  # pixels without a prediction are left out
    return {"values": values, "target_unit": "%", "output": "moisture", "map_step": 2,
            "provenance": {"output": "moisture"}}


#: A payload with data for every plot, by the workspace prefix of its id.
def _explore_payload() -> dict:
    from enginir.modeling import RoiSpectrumSummary

    rng = np.random.default_rng(5)
    wavelengths = np.linspace(1000.0, 1700.0, 9)
    summaries = {}
    for i in range(3):
        base = rng.normal(0.3 + 0.1 * i, 0.01, 9).astype(np.float32)
        summaries[f"roi-{i}"] = RoiSpectrumSummary(f"roi-{i}", base, base - 0.02, base,
                                                   base + 0.02)
    return {"wavelengths": wavelengths, "signal_label": "Absorbance",
            "names": {k: f"T{i}" for i, k in enumerate(summaries)},
            "means": {k: s.mean for k, s in summaries.items()},
            "values": {"roi-0": 1.0, "roi-1": 3.0, "roi-2": None},
            "value_label": "API (%)", "selected": summaries["roi-1"], "selected_name": "T1"}


_PAYLOADS = {"model": _model_payload, "predict": _predict_payload, "explore": _explore_payload}


def _registry():
    import enginir.viewer._model_diagnostics  # noqa: F401 (registers the Model plots)
    import enginir.viewer._prediction  # noqa: F401 (registers the Predict plots)
    import enginir.viewer._project_rois  # noqa: F401 (registers the Project ROIs plots)
    from enginir.viewer._plot_registry import PLOT_SPECS, WORKSPACE_PAGES

    return PLOT_SPECS, WORKSPACE_PAGES


def _registered_plots() -> list[tuple[str, str]]:
    specs, _pages = _registry()
    return [(plot_id.split(".")[0], plot_id) for plot_id in sorted(specs)]


def _drawn_values(fig) -> np.ndarray:
    values: list[np.ndarray] = []
    for ax in fig.axes:
        for collection in ax.collections:
            values.append(np.ravel(collection.get_offsets()))
            values += [np.ravel(path.vertices) for path in collection.get_paths()]
            if collection.get_array() is not None:
                values.append(np.ravel(collection.get_array()))
        for line in ax.lines:
            values += [np.ravel(line.get_xdata()), np.ravel(line.get_ydata())]
        for patch in ax.patches:
            if hasattr(patch, "get_data"):  # StepPatch (histograms)
                data = patch.get_data()
                values += [np.ravel(data.values), np.ravel(data.edges)]
    return np.concatenate(values).astype(float) if values else np.array([])


def _csv_numbers(path: Path) -> np.ndarray:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if not line.startswith("# ")]
    header, *rows = list(csv.reader(lines))
    numbers = []
    for index, name in enumerate(header):
        if name in ("roi_name", "roi_id"):
            continue
        numbers += [float(row[index]) for row in rows]
    return np.array(numbers)


@pytest.mark.parametrize("workspace, plot_id", _registered_plots())
def test_registered_plot_meets_the_plot_panel_policy(qapp, tmp_path, workspace, plot_id):
    from enginir.viewer._plot_panel import PlotPanel

    specs, _pages = _registry()
    panel = PlotPanel(specs[plot_id])
    panel.resize(500, 350)
    panel.show()

    panel.update_data(None, empty_reason="Nothing fitted yet.")
    assert "Nothing fitted yet." in [t.get_text() for ax in panel.figure.axes for t in ax.texts]
    assert panel.export_reason()

    panel.update_data(_PAYLOADS[workspace](), provenance={"project_id": "p"})
    assert not panel.export_reason(), "a full payload is exportable"
    assert all(not ax.get_title() for ax in panel.figure.axes), "title only in the header"
    for name in ("plot.png", "plot.svg", "plot.csv"):
        (panel.export_csv if name.endswith(".csv") else panel.export_image)(tmp_path / name)
        assert (tmp_path / name).stat().st_size > 0

    drawn = _drawn_values(panel.figure)
    exported = _csv_numbers(tmp_path / "plot.csv")
    assert exported.size
    missing = [value for value in exported
               if not np.isclose(drawn, value, rtol=1e-6, atol=1e-9).any()]
    assert not missing, f"CSV values that the plot does not draw: {missing[:5]}"


# -- Export of all workspace plots and the plot layout in user settings (Plot panel policy) ----


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
    from tests._project_payloads import write_model_project
    from tests._sessions import fit_stored_model

    path = str(tmp_path / "layout.h5")
    write_model_project(path)
    fit_stored_model(path, model_type="PLS", n_components=2, model_run_id="model_1")
    viewer = _viewer(qapp)
    project = viewer.project_state
    project.open_project(path)
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
             "model.scores", "model.coefficients", "model.loadings", "model.coef_cs",
             "model.vip", "model.vip_sorted", "model.selectivity_ratio",
             "model.components", "model.r2_q2", "model.per_component", "model.r2_pixels",
             "model.dmodx", "model.t2",
             "model.roi_distance")
    assert sorted(written) == sorted(f"{p}{s}" for p in plots for s in (".png", ".csv"))
    assert skipped == []
    assert grid.current_page == "Fit", "exporting does not switch the page"
    csv_text = (out / "model.loadings.csv").read_text(encoding="utf-8")
    assert "# model_run_id: model_1" in csv_text


def test_export_plots_skips_plots_without_data_and_needs_a_folder(qapp, tmp_path):
    viewer = _viewer(qapp)
    written, skipped = viewer.export_workspace_plots(tmp_path)
    # The Histogram is a hidden dock by default and has never drawn; there is
    # no project, so the Project ROIs plots have nothing to show.
    assert sorted(skipped) == ["explore.histogram", "explore.roi_means",
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


# -- Empty plots and plot headers ------------------------------------------------------


# -- Long plot titles ------------------------------------------------------------------


