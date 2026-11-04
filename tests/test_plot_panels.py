"""Plot panels and plot grids: export, maximize and lazy drawing (Plot Panel Policy)."""

from __future__ import annotations

import csv
import types
from pathlib import Path

import numpy as np
import pytest
from matplotlib.image import imread

from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


def _read_csv(path: Path) -> tuple[dict[str, str], list[list[str]]]:
    provenance: dict[str, str] = {}
    rows: list[list[str]] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.startswith("# "):
                rows = list(csv.reader([line, *handle]))
                break
            key, _, value = line[2:].rstrip("\n").partition(": ")
            provenance[key] = value
    return provenance, rows


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


def test_csv_holds_the_plotted_values_and_the_provenance(qapp, tmp_path):
    payload = _payload()
    panel = _shown_panel(_scatter_spec(), payload, provenance={"project_id": "p-1"})
    path = tmp_path / "plot.csv"
    panel.export_csv(path)

    provenance, rows = _read_csv(path)
    assert provenance["plot"] == "test.scatter"
    assert provenance["project_id"] == "p-1"
    assert provenance["components"] == "3"
    assert provenance["enginir_version"] and provenance["exported_at"]
    header, *body = rows
    assert header == ["roi_name", "roi_id", "reference (%)", "predicted (%)"]
    assert [row[0] for row in body] == payload["roi_names"]  # commas survive quoting
    assert [row[1] for row in body] == payload["roi_ids"]
    # The CSV holds exactly the points drawn on screen.
    drawn = panel.figure.axes[0].collections[0].get_offsets()
    np.testing.assert_array_equal(np.array([[float(r[2]), float(r[3])] for r in body],
                                           dtype=np.float32), drawn)


def test_a_matrix_table_is_written_as_rows_by_frames(tmp_path):
    from enginir.viewer._plot_panel import PlotTable

    matrix = np.arange(6, dtype=np.float64).reshape(2, 3) / 7
    path = tmp_path / "map.csv"
    PlotTable(matrix=matrix).write_csv(path, {"plot": "map"})
    provenance, rows = _read_csv(path)
    assert provenance == {"plot": "map"}
    assert rows[0] == ["row\\frame", "0", "1", "2"]
    assert [row[0] for row in rows[1:]] == ["0", "1"]
    np.testing.assert_array_equal(np.array([r[1:] for r in rows[1:]], dtype=float), matrix)


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


def test_the_image_shows_the_current_axis_limits(qapp, tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    panel = _shown_panel(_scatter_spec(), _payload())
    panel.figure.axes[0].set_xlim(2.0, 3.0)  # the user zoomed in
    saved: list[Figure] = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **k: saved.append(fig))
    panel.export_image(tmp_path / "plot.png")
    assert saved[0].axes[0].get_xlim() == (2.0, 3.0)


def test_the_title_is_in_the_header_not_in_the_plot(qapp):
    from PyQt6.QtWidgets import QLabel

    panel = _shown_panel(_scatter_spec(), _payload())
    assert "Predicted vs Reference" in [label.text() for label in panel.findChildren(QLabel)]
    assert all(not ax.get_title() for ax in panel.figure.axes)


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


def test_a_hidden_panel_draws_when_shown_and_skips_unchanged_payloads(qapp):
    from enginir.viewer._plot_panel import PlotPanel

    calls: list[str] = []
    panel = PlotPanel(_scatter_spec(calls))
    payload = _payload()
    panel.update_data(payload)
    assert calls == []  # hidden: nothing drawn
    panel.show()
    qapp.processEvents()
    assert calls == ["test.scatter"]
    panel.update_data(payload)
    assert calls == ["test.scatter"]  # same payload object: no-op
    panel.update_data(_payload())
    assert calls == ["test.scatter"] * 2


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


def test_a_page_is_built_and_drawn_only_when_shown(qapp):
    calls: list[str] = []
    grid = _grid(calls)
    grid.update_data({"a": _payload(), "b": _payload(), "c": _payload()})
    qapp.processEvents()
    assert sorted(calls) == ["a", "b"]
    assert grid.panel("c") is None  # second page not built yet

    grid.set_current_page("Spectral")
    qapp.processEvents()
    assert grid.panel("c") is not None and calls[-1] == "c"


def test_maximize_fills_the_page_and_restore_redraws_nothing(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    calls: list[str] = []
    grid = _grid(calls)
    grid.update_data({"a": _payload(), "b": _payload()})
    qapp.processEvents()
    panel_a, panel_b = grid.panel("a"), grid.panel("b")
    assert panel_a is not None and panel_b is not None
    width_in_grid = panel_a.width()
    calls.clear()

    grid.maximize("a")
    qapp.processEvents()
    assert grid.maximized_plot() == "a" and panel_a.is_maximized
    assert panel_a.width() > 1.5 * width_in_grid
    assert not panel_b.isVisible()

    grid.activateWindow()
    assert QTest.qWaitForWindowActive(grid)
    panel_a.figure.canvas.setFocus()  # e.g. after clicking into the plot
    QTest.keyClick(panel_a.figure.canvas, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert grid.maximized_plot() is None and not panel_a.is_maximized
    assert panel_b.isVisible() and panel_a.width() == width_in_grid
    assert "b" not in calls  # the other panel kept its drawing


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
