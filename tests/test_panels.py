"""Focused smoke tests for viewer matplotlib panels."""

from __future__ import annotations

import numpy as np
import pytest

from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


@pytest.fixture()
def state():
    from enginir.viewer._state import ViewerState

    rng = np.random.default_rng(123)
    cube = rng.integers(128, 3500, size=(4, 8, 5), dtype=np.uint16)
    wavelengths = np.linspace(1000.0, 1700.0, 5, dtype=np.float32)
    return ViewerState(cube, wavelengths)


def test_hidden_histogram_defers_refresh_until_visible(qapp, state, monkeypatch):
    from enginir.viewer._panels import HistogramPanel

    panel = HistogramPanel(state)
    calls: list[None] = []
    original = state.current_frame_image

    def counting_current_frame_image():
        calls.append(None)
        return original()

    monkeypatch.setattr(state, "current_frame_image", counting_current_frame_image)

    state.set_frame(1)
    qapp.processEvents()
    assert calls == []  # hidden: nothing is computed
    assert panel._stairs is None

    panel.show()
    qapp.processEvents()
    assert calls
    stairs = panel._stairs
    expected, _ = np.histogram(original(), bins=256)
    np.testing.assert_array_equal(stairs.get_data().values, expected)

    state.set_frame(2)  # visible: the same artist is updated in place
    qapp.processEvents()
    assert panel._stairs is stairs
    expected, edges = np.histogram(original(), bins=256)
    np.testing.assert_array_equal(stairs.get_data().values, expected)
    x0, x1 = panel._ax.get_xlim()
    assert x0 < edges[0] and x1 > edges[-1], "every bin stays in view"
    assert panel._ax.get_ylim()[1] > expected.max()

    # A uniform frame puts every pixel into one bin, far above the earlier counts.
    monkeypatch.setattr(state, "current_frame_image", lambda: np.full((8, 5), 1000.0, np.float32))
    panel._on_update()
    assert panel._ax.get_ylim()[1] > 40


def test_sensor_panel_swaps_only_the_image_on_frame_change(qapp, state):
    from enginir.viewer._panels import SensorPanel

    panel = SensorPanel(state)
    panel.resize(400, 300)
    panel.show()
    qapp.processEvents()
    image_item = panel._image
    view_range = panel._view.viewRange()

    state.set_frame(2)
    qapp.processEvents()
    # The same pyqtgraph image item gets the new frame; the view is not reset.
    assert panel._image is image_item
    np.testing.assert_array_equal(panel._image.image, state.current_frame_image())
    assert panel._view.viewRange() == view_range
    data = state.current_frame_image()
    assert panel._levels == (float(data.min()), float(data.max()))


def test_image_panels_pick_the_clicked_pixel_and_follow_the_cursor(qapp, state):
    import types

    from PyQt6.QtCore import QPointF

    from enginir.viewer._panels import DC3Panel

    panel = DC3Panel(state)
    panel.resize(400, 300)
    panel.show()
    qapp.processEvents()
    picks: list[tuple[str, int]] = []
    panel.frame_pick.connect(lambda f: picks.append(("frame", f)))
    panel.row_pick.connect(lambda r: picks.append(("row", r)))
    # Click where pixel (row 6, frame 2) is drawn, a little off its centre.
    scene_pos = panel._image.mapToScene(QPointF(2.5 + 0.3, 6.5 - 0.3))
    panel._on_scene_click(types.SimpleNamespace(scenePos=lambda: scene_pos))
    assert picks == [("frame", 2), ("row", 6)]

    data = panel._data
    state.set_frame(3)
    state.set_row(1)
    assert panel._data is data, "a cursor move does not rebuild the band image"
    assert (panel._vline.value(), panel._hline.value()) == (3, 1)
    state.set_band(4)
    assert panel.plot_title.startswith("RAW  band 4 ")
    np.testing.assert_array_equal(panel._data, state.raw_band_image(4))


def test_model_diagnostics_show_empty_states_in_the_plots(qapp):
    """Blocking reasons and missing scores are visible, not just blank axes (contract)."""
    from enginir.viewer._model_diagnostics import ModelDiagnosticsWidget

    def texts(ax):
        return [text.get_text() for text in ax.texts]

    widget = ModelDiagnosticsWidget()
    widget.resize(900, 600)
    widget.show()
    widget.clear(empty_reason="Build a sample set before fitting.")
    for page in widget.grid.page_names:
        widget.grid.set_current_page(page)
        qapp.processEvents()
    panels = {plot_id: widget.grid.panel(plot_id) for plot_id in (
        "model.pred_vs_ref", "model.residuals", "model.target_distribution", "model.scores",
        "model.coefficients", "model.loadings")}
    for plot_id, panel in panels.items():
        assert panel is not None
        assert "Build a sample set before fitting." in texts(panel.figure.axes[0]), plot_id

    widget.grid.set_current_page("Fit")
    payload = {"y_true_roi": [1.0, 2.0, 3.0], "y_pred_roi": [1.1, 1.9, 3.2],
               "roi_ids": ["a", "b", "c"], "scores_roi": None}
    widget.update_data(payload)
    qapp.processEvents()
    pred = panels["model.pred_vs_ref"]
    scores = panels["model.scores"]
    assert pred is not None and scores is not None
    assert "Build a sample set before fitting." not in texts(pred.figure.axes[0])
    assert "No latent scores diagnostics yet." in texts(scores.figure.axes[0])
    assert scores.export_reason(), "a placeholder cannot be exported"
