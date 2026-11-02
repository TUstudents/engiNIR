"""Explore plots as plot frames: exports, header titles and maximize (Plot Panel Policy)."""

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

    dock, frame = viewer._sensor_dock, viewer._dock_frames[viewer._sensor_dock]
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
