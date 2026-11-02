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
    expected, _ = np.histogram(original(), bins=256)
    np.testing.assert_array_equal(stairs.get_data().values, expected)


def test_sensor_panel_updates_image_in_place_on_frame_change(qapp, state):
    from enginir.viewer._panels import SensorPanel

    panel = SensorPanel(state)
    panel.show()
    qapp.processEvents()

    original_handle = panel._img_handle
    assert original_handle is not None

    state.set_frame(2)
    qapp.processEvents()

    # A full rebuild would replace the AxesImage via a fresh imshow(); the
    # in-place refresh must reuse the same handle and only push new data.
    assert panel._img_handle is original_handle
    np.testing.assert_array_equal(
        np.asarray(panel._img_handle.get_array()),
        np.asarray(state.current_frame_image()),
    )


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
