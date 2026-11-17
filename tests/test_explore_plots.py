"""Explore plots as plot frames: exports, header titles and maximize (Plot panel policy)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from matplotlib.image import imread

from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


@pytest.fixture()
def viewer(qapp):
    from enginir.viewer._viewer import HSIViewer

    rng = np.random.default_rng(5)
    cube = rng.integers(500, 3000, size=(30, 12, 9), dtype=np.uint16)
    viewer = HSIViewer(cube, np.linspace(1000.0, 1700.0, 9, dtype=np.float32))
    viewer.resize(1400, 900)
    viewer.show()
    viewer._hist_dock.show()
    viewer._dc3_dock.show()
    viewer.state.set_frame(4)
    viewer.state.set_row(7)
    viewer.state.set_band(3)
    qapp.processEvents()
    yield viewer
    viewer.close()


def _read(path: Path) -> tuple[dict[str, str], list[str], np.ndarray]:
    provenance: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines[0].startswith("# "):
        key, _, value = lines.pop(0)[2:].partition(": ")
        provenance[key] = value
    header, *rows = list(csv.reader(lines))
    return provenance, header, np.array(rows, dtype=float)


def _frames(viewer) -> dict:
    return {
        "explore.image": viewer._canvas_frame,
        "explore.spectrum": viewer._spectrum_panel,
        "explore.sensor": viewer._sensor_frame,
        "explore.spatial": viewer._spatial_frame,
        **{frame.plot_id: frame for frame in viewer._dock_frames.values()},
    }


def test_every_explore_plot_exports_what_it_shows(viewer, tmp_path):
    state = viewer.state
    frames = _frames(viewer)
    assert sorted(frames) == ["explore.histogram", "explore.image", "explore.raw",
                              "explore.sensor", "explore.spatial", "explore.spectrum"]
    tables = {}
    for plot_id, frame in frames.items():
        assert not frame.export_reason(), plot_id
        frame.export_image(tmp_path / f"{plot_id}.png")
        frame.export_image(tmp_path / f"{plot_id}.svg")
        np.testing.assert_allclose(imread(tmp_path / f"{plot_id}.png")[0, 0, :3], 1.0)
        frame.export_csv(tmp_path / f"{plot_id}.csv")
        provenance, header, values = _read(tmp_path / f"{plot_id}.csv")
        assert provenance["plot"] == plot_id
        assert {"project_id", "data_file", "enginir_version"} <= set(provenance)
        tables[plot_id] = (provenance, header, values)

    _prov, header, values = tables["explore.spectrum"]
    assert header == ["wavelength (nm)", "value (DN)"]
    np.testing.assert_allclose(values[:, 0], state.wavelengths, rtol=1e-6)
    np.testing.assert_allclose(values[:, 1], state.get_spectrum(7), rtol=1e-6)
    assert _prov["row"] == "7" and _prov["frame"] == "4"

    _prov, header, values = tables["explore.spatial"]
    np.testing.assert_allclose(values[:, 1], state.get_spatial_profile(3), rtol=1e-6)
    assert _prov["band"] == "3"

    _prov, header, values = tables["explore.raw"]
    assert header[0] == "row\\frame"
    np.testing.assert_array_equal(values[:, 1:], state.raw_band_image(3))

    _prov, header, values = tables["explore.sensor"]
    assert header[0] == "row\\band"
    np.testing.assert_allclose(values[:, 1:], state.current_frame_image(), rtol=1e-6)

    _prov, header, values = tables["explore.histogram"]
    counts, edges = np.histogram(state.current_frame_image(), bins=256)
    np.testing.assert_array_equal(values[:, 2], counts)
    np.testing.assert_allclose(values[:, 0], edges[:-1], rtol=1e-6)

    _prov, header, values = tables["explore.image"]
    assert header == ["frame", "row", "red", "green", "blue"]
    rgb = state.raw_false_colour_rgb()
    frames_idx, rows_idx = values[:, 0].astype(int), values[:, 1].astype(int)
    np.testing.assert_allclose(values[:, 2:], rgb[rows_idx, frames_idx], rtol=1e-6)


def test_titles_are_in_the_headers_and_follow_the_band(viewer, qapp):
    frames = _frames(viewer)
    for plot_id, frame in frames.items():
        if plot_id == "explore.image":
            continue
        assert all(not ax.get_title() for ax in frame.print_figure().axes), plot_id
    raw = frames["explore.raw"]
    assert raw.title.startswith("RAW  band 3 ")
    viewer.state.set_band(5)
    qapp.processEvents()
    assert raw.title.startswith("RAW  band 5 ")


def test_a_panel_without_data_cannot_be_exported(qapp):
    from enginir.viewer._panels import SpectrumPanel
    from enginir.viewer._plot_panel import PlotFrame
    from enginir.viewer._state import ViewerState

    panel = SpectrumPanel(ViewerState(np.zeros((0, 0, 0), np.uint16), np.zeros(0, np.float32)))
    frame = PlotFrame("explore.spectrum", panel.plot_title, panel, panel)
    frame.show()
    assert frame.export_reason() == "No data to export."


def test_a_maximized_dock_covers_the_window_and_restores(viewer, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    dock, frame = viewer._hist_dock, viewer._dock_frames[viewer._hist_dock]
    dock.show()
    qapp.processEvents()
    docked_width = dock.width()
    frame.maximize_toggled.emit()
    qapp.processEvents()
    assert frame.is_maximized and dock.isFloating()
    assert dock.width() > 2 * docked_width

    dock.activateWindow()
    QTest.qWaitForWindowActive(dock)
    QTest.keyClick(dock.widget(), Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not frame.is_maximized and not dock.isFloating()

    frame.maximize_toggled.emit()
    dock.hide()  # the header's close button; reopening must show it docked
    qapp.processEvents()
    assert not frame.is_maximized and not dock.isFloating()

    dock.show()
    frame.maximize_toggled.emit()
    viewer._workspace_tabs.setCurrentWidget(viewer._modeling_view)  # leaving Explore restores
    qapp.processEvents()
    assert not frame.is_maximized and not dock.isFloating()


def test_the_spectrum_maximizes_within_the_explore_page(viewer, qapp):
    spectrum, image = viewer._spectrum_panel, viewer._canvas_frame
    height = spectrum.height()
    spectrum.maximize_toggled.emit()
    qapp.processEvents()
    assert spectrum.is_maximized and not image.isVisible()
    assert spectrum.height() > 2 * height

    spectrum.maximize_toggled.emit()  # the same control restores
    qapp.processEvents()
    assert not spectrum.is_maximized and image.isVisible() and spectrum.height() == height

    image.maximize_toggled.emit()
    qapp.processEvents()
    assert image.is_maximized and not spectrum.isVisible()
    viewer._on_explore_escape()
    qapp.processEvents()
    assert not image.is_maximized and spectrum.isVisible()


def test_sensor_frame_and_spatial_profile_share_the_image_rows(viewer, qapp):
    """They sit beside the Spatial image on the page, and one row lines up in all three."""
    image, sensor, spatial = viewer._canvas, viewer._sensor, viewer._spatial
    assert viewer._sensor_frame.parent() is viewer._rows_splitter
    assert viewer._spatial_frame.parent() is viewer._rows_splitter
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDockWidget

    right = [dock for dock in viewer.findChildren(QDockWidget) if dock.isVisible()
             and viewer.dockWidgetArea(dock) == Qt.DockWidgetArea.RightDockWidgetArea]
    assert right == [viewer._controls_dock], "the settings have the right side to themselves"
    image.view_box.setYRange(2.0, 6.0, padding=0)
    qapp.processEvents()
    for panel in (sensor, spatial):
        low, high = panel.view_box.viewRange()[1]
        assert (low, high) == pytest.approx((2.0, 6.0)), "zoom follows the image's rows"

    # Upright: the value along x, the row down; the red line is the row.
    x, y = spatial._curve.getData()
    np.testing.assert_allclose(x, viewer.state.get_spatial_profile(viewer.state.band), rtol=1e-6)
    np.testing.assert_array_equal(y, np.arange(viewer.state.height))
    viewer.state.set_row(5)
    assert spatial._marker.value() == 5
    spatial._marker.setValue(2.2)  # dragged
    assert viewer.state.row == 2


def test_the_sensor_frame_shows_wavelengths_like_the_spectral_profile(viewer, qapp):
    state, sensor = viewer.state, viewer._sensor
    state.set_x_spectral_unit("nm")
    qapp.processEvents()
    assert sensor._x_label == "Wavelength (nm)"
    left, width = sensor._column_span()
    assert left < state.wavelengths[0] < state.wavelengths[-1] < left + width
    assert sensor._vline.value() == pytest.approx(state.wavelengths[state.band])
    assert sensor._column_at(float(state.wavelengths[2]) + 0.1) == 2, "a click picks the band"
    state.set_x_spectral_unit("band")
    qapp.processEvents()
    assert sensor._x_label == "Band" and sensor._column_span() == (-0.5, state.n_bands)


def test_the_page_plots_close_maximize_and_come_back_from_the_view_menu(viewer, qapp):
    sensor, spatial = viewer._sensor_frame, viewer._spatial_frame
    action = viewer._page_plot_actions["sensor"]
    viewer._on_page_plot_visibility_action("sensor", False)  # the header's close button
    qapp.processEvents()
    assert not sensor.isVisible() and spatial.isVisible() and not action.isChecked()
    action.trigger()
    qapp.processEvents()
    assert sensor.isVisible() and action.isChecked()

    sensor.maximize_toggled.emit()
    qapp.processEvents()
    assert sensor.is_maximized and not viewer._canvas_frame.isVisible() \
        and not spatial.isVisible() and not viewer._spectrum_panel.isVisible()
    viewer._on_explore_escape()
    qapp.processEvents()
    assert not sensor.is_maximized and viewer._canvas_frame.isVisible() and spatial.isVisible()


def test_raw_count_axes_name_the_signal_in_full(viewer, qapp):
    """Terminology contract: the y axis of raw data says "Raw counts (DN)", never "DN" alone."""
    from enginir.viewer._panels import SpatialProfilePanel

    assert viewer.state.y_unit_effective == "DN"
    panels = [panel for panel in viewer.findChildren(SpatialProfilePanel)]
    assert panels
    qapp.processEvents()
    for panel in panels:
        assert panel.axis_labels()[0] == "Raw counts (DN)"
