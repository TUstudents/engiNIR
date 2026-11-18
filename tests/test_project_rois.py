"""The Project ROIs tab: stored ROIs shown from the project file alone (viewer UI contract)."""

from __future__ import annotations

import sys
import time

import h5py
import numpy as np
import pytest
from PyQt6.QtCore import Qt

from enginir import Project
from enginir.io import save_hdf5
from tests._qt import wait_until


@pytest.fixture()
def trays(tmp_path):
    """Two trays with references from frames; tray_a has A1 (API 2), A2 (API 5); tray_b has B1."""
    rng = np.random.default_rng(12)
    wavelengths = np.linspace(1000.0, 1700.0, 12)
    paths = {}
    for name in ("tray_a", "tray_b"):
        cube = rng.integers(1500, 2500, (24, 10, 12)).astype(np.uint16)
        cube[:2], cube[-2:] = 100, 3600
        paths[name] = tmp_path / f"{name}.h5"
        save_hdf5(str(paths[name]), cube, wavelengths)
    project = Project.create(tmp_path / "trays.h5")
    project.add_target("API", unit="%")
    tray_a = project.add_dataset(paths["tray_a"], dark_frames=(0, 1), white_frames=(22, 23))
    tray_a.add_roi("tray_a-A1", frames=(3, 6), rows=(1, 3), values={"API": 2.0})
    tray_a.add_roi("tray_a-A2", frames=(9, 12), rows=(5, 8), values={"API": 5.0})
    tray_b = project.add_dataset(paths["tray_b"], dark_frames=(0, 1), white_frames=(22, 23))
    tray_b.add_roi("tray_b-B1", frames=(4, 7), rows=(2, 4))
    return project, paths


def _settle(qapp, view, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if not view.busy() and not view.mosaic.missing_patches() \
                and not view._rebuild_timer.isActive():
            return
        time.sleep(0.01)
    raise AssertionError("Project ROIs did not finish loading")


def _open(qapp, project, target="API"):
    from enginir.viewer import open_project

    window = open_project(project.path)
    window.resize(1400, 900)
    window.show()
    modeling = window._modeling_view
    modeling._selected_target_id = next(t for t, ref in window.project_state.targets.items()
                                        if ref.name == target)
    modeling._update_target_status()
    qapp.processEvents()
    return window


def test_the_scan_layout_keeps_positions_and_stacks_data_files(trays):
    from enginir.project._data_paths import stored_path
    from enginir.viewer._project_rois import SCAN, mosaic_layout

    project, _paths = trays
    rois = list(project.workspace.rois.values())
    names = {d: stored_path(ref.source_path).name for d, ref in project.workspace.datasets.items()}
    layout = mosaic_layout(rois, names, SCAN)
    places = layout.places
    a1, a2, b1 = (project.roi(n).roi_id for n in ("tray_a-A1", "tray_a-A2", "tray_b-B1"))
    assert [name for _d, name, _x, _y, _w in layout.headers] == ["tray_a.h5", "tray_b.h5"]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = places[a1], places[a2]
    assert (x1, w1, h1) == (3.0, 4.0, 3.0) and (x2, w2, h2) == (9.0, 4.0, 4.0)
    assert y2 - y1 == 5 - 1, "rows keep their distance within a data file"
    header_b = layout.headers[1][3]
    assert header_b >= y2 + h2 + layout.unit, "the labels below keep their room"
    assert places[b1][1] >= header_b + 1.5 * layout.unit, "the header keeps its room"


def test_a_project_with_rois_opens_on_them_without_any_data_file(qapp, trays):
    project, paths = trays
    for path in paths.values():
        path.unlink()  # the tab needs only the project file
    window = _open(qapp, project)
    try:
        view = window._project_rois
        assert window._explore_tabs.currentWidget() is view
        assert window._explore_tabs.tabText(1) == "Project ROIs (3)"
        _settle(qapp, view)
        assert view.mosaic.tile_count() == 3
        labels = view.mosaic._labels
        assert labels[project.roi("tray_a-A2").roi_id] == "A2  5"
        assert labels[project.roi("tray_b-B1").roi_id] == "B1"
    finally:
        window.close()


def test_mean_spectra_match_the_fit_conversion_and_follow_the_signal_unit(qapp, trays):
    from enginir.modeling import roi_spectrum_summaries

    project, _paths = trays
    window = _open(qapp, project)
    try:
        view = window._project_rois
        window.state.set_y_unit("Absorbance")
        _settle(qapp, view)
        payload = view.means.payload
        ws = project.workspace
        expected, _ = roi_spectrum_summaries(project.path, list(ws.rois.values()),
                                             datasets=ws.datasets, calibrations=ws.calibrations,
                                             signal_unit="Absorbance")
        a1 = project.roi("tray_a-A1").roi_id
        np.testing.assert_allclose(payload["means"][a1], expected[a1].mean)
        assert payload["values"][a1] == 2.0 and payload["values"][project.roi(
            "tray_b-B1").roi_id] is None  # drawn grey
        assert payload["signal_label"] == "Absorbance"
        window.state.set_preprocess_snv(True)
        _settle(qapp, view)
        assert view.means.payload["signal_label"] == "Absorbance, SNV"
    finally:
        window.close()


def test_selecting_and_opening_a_tile(qapp, trays):
    project, paths = trays
    window = _open(qapp, project)
    try:
        view = window._project_rois
        _settle(qapp, view)
        a2 = project.roi("tray_a-A2").roi_id
        x, y, w, h = view.mosaic.places[a2]
        assert view.mosaic.roi_at(x + w / 2, y + h / 2) == a2

        view.mosaic.roi_selected.emit(a2)  # a click on the tile
        view.mosaic.select_roi(a2)
        view._on_tile_selected(a2)
        qapp.processEvents()
        assert view.pixels.payload["selected_name"] == "tray_a-A2"
        assert window._roi_table.view.currentIndex().isValid()

        view.mosaic.roi_opened.emit(a2)  # a double-click: show its data file
        window.wait_for_data_file()
        assert window._explore_tabs.currentWidget() is window._data_file_page
        assert window._cube_path == str(paths["tray_a"].resolve())
        assert window._canvas._selected_roi_id == a2
    finally:
        window.close()


def test_a_hidden_tab_reads_nothing(qapp, trays, monkeypatch):
    from enginir.viewer import _project_rois, open_viewer

    project, paths = trays
    calls: list[int] = []
    original = _project_rois.load_roi_band_images
    monkeypatch.setattr(_project_rois, "load_roi_band_images",
                        lambda *a, **k: calls.append(1) or original(*a, **k))
    window = open_viewer(paths["tray_a"], project=project.path)  # opens on the data file
    try:
        window.show()
        for _ in range(20):
            qapp.processEvents()
            time.sleep(0.01)
        assert window._explore_tabs.currentWidget() is window._data_file_page
        assert calls == [] and not window._project_rois.busy()
        window.show_project_rois()
        _settle(qapp, window._project_rois)
        assert calls
    finally:
        window.close()


def test_raw_counts_are_shown_in_dn_and_skipped_rois_are_explained(trays, tmp_path):
    from enginir.modeling import roi_spectrum_summaries

    project, paths = trays
    ws = project.workspace
    summaries, skipped = roi_spectrum_summaries(project.path, list(ws.rois.values()),
                                                datasets=ws.datasets,
                                                calibrations=ws.calibrations, signal_unit="DN")
    a1 = project.roi("tray_a-A1")
    from enginir.io.project_store import load_roi_spectra

    raw = load_roi_spectra(project.path, f"/rois/{a1.roi_id}/spectra")
    np.testing.assert_allclose(summaries[a1.roi_id].mean, raw.mean(axis=0), rtol=1e-6)
    assert skipped == {}

    calibrated = tmp_path / "calibrated.h5"
    save_hdf5(str(calibrated), np.full((4, 3, 12), 0.5, np.float32), np.linspace(1000, 1700, 12))
    project.add_dataset(calibrated, unit="Reflectance").add_roi("F1", frames=(0, 1), rows=(0, 1))
    ws = project.workspace
    _summaries, skipped = roi_spectrum_summaries(project.path, list(ws.rois.values()),
                                                 datasets=ws.datasets,
                                                 calibrations=ws.calibrations, signal_unit="DN")
    assert skipped == {project.roi("F1").roi_id: "precalibrated data has no raw counts"}


def test_a_new_value_relabels_the_tile_without_laying_out_again(qapp, trays, monkeypatch):
    project, _paths = trays
    window = _open(qapp, project)
    try:
        view = window._project_rois
        _settle(qapp, view)
        layouts: list[int] = []
        original = view.mosaic.set_rois
        monkeypatch.setattr(view.mosaic, "set_rois",
                            lambda *a, **k: layouts.append(1) or original(*a, **k))
        state = window.project_state
        b1 = project.roi("tray_b-B1").roi_id
        api = project.target("API").target_id
        state.set_roi_values([(b1, api, 7.0)])
        state.set_roi_values([(b1, api, 8.0)])  # a burst: one label update
        _settle(qapp, view)
        assert view.mosaic.label_text(b1) == "B1  8"
        assert view.mosaic.outline_color(b1) != view.mosaic.outline_color(
            project.roi("tray_a-A1").roi_id), "the outline takes the value's colour"
        assert view.mosaic.tile_count() == 3 and layouts == []
    finally:
        window.close()


def test_tiles_loaded_in_batches_share_one_stretch_at_the_end(qapp, trays, monkeypatch):
    from enginir.viewer import _project_rois

    monkeypatch.setattr(_project_rois, "_TILE_BATCH", 1)  # three batches, three stretches
    project, _paths = trays
    window = _open(qapp, project)
    try:
        view = window._project_rois
        _settle(qapp, view)
        mosaic = view.mosaic
        expected = _project_rois.stretch_to_rgb(mosaic._patches)
        assert mosaic.tile_count() == 3
        for roi_id, item in mosaic._images.items():
            np.testing.assert_allclose(item.image, np.transpose(expected[roi_id], (1, 0, 2)),
                                       atol=1e-6)
    finally:
        window.close()


def test_equal_values_get_the_colour_the_colour_bar_shows(qapp):
    """All ROIs at 10: the lines must take the colour of 10 on the bar, not its lowest colour.
    (A plain Normalize(10, 10) maps them to the bottom colour while the bar shows 9-11.)"""
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure

    from enginir.viewer._project_rois import _render_means
    from enginir.viewer._theme import panel_colors, value_colormap

    payload = {"means": {"a": np.ones(5), "b": np.full(5, 2.0)}, "values": {"a": 10.0, "b": 10.0},
               "wavelengths": np.arange(5.0), "axis_label": "Band", "value_label": "Layer"}
    figure = Figure()
    _render_means(figure, payload, panel_colors())
    ax, bar = figure.axes
    low, high = bar.get_ylim()
    expected = value_colormap(panel_colors())((10.0 - low) / (high - low))
    assert all(np.allclose(to_rgba(line.get_color()), expected) for line in ax.get_lines())
    assert ax.get_xlabel() == "Band"


def test_a_file_without_wavelengths_is_plotted_over_band_numbers(qapp, tmp_path):
    rng = np.random.default_rng(3)
    cube = rng.integers(1500, 2500, (12, 6, 10)).astype(np.uint16)
    path = tmp_path / "bands.h5"
    save_hdf5(str(path), cube, np.linspace(1000.0, 1700.0, 10))
    with h5py.File(path, "a") as f:
        del f["wavelengths"]  # a scan saved without its wavelengths
    project = Project.create(tmp_path / "bands_project.h5")
    scan = project.add_dataset(str(path), black_level=128.0, white_level=3500.0)
    scan.add_roi("r1", frames=(2, 5), rows=(1, 3))
    window = _open_without_target(qapp, project)
    try:
        view = window._project_rois
        _settle(qapp, view)
        assert view.means.payload["axis_label"] == "Band"
        assert view.means.table().columns[0].unit == "band"
    finally:
        window.close()


def _open_without_target(qapp, project):
    from enginir.viewer import open_project

    window = open_project(project.path)
    window.show()
    qapp.processEvents()
    return window


def test_data_file_docks_and_the_roi_table_follow_the_explore_tab(qapp, trays):
    project, _paths = trays
    window = _open(qapp, project)
    try:
        tabs, plots = window._explore_tabs, (window._sensor_frame, window._spatial_frame)
        table = window._roi_table_dock
        # Project ROIs: the table in the bottom panel; nothing at the cursor.
        assert tabs.currentWidget() is window._project_rois
        assert not table.isHidden() and not any(plot.isVisible() for plot in plots)
        assert window._roi_table._file_filter.currentIndex() == 0, "all data files"
        a_id = next(iter(project.workspace.datasets))
        window._project_rois._file_filter.setCurrentIndex(
            window._project_rois._file_filter.findData(a_id))
        assert window._roi_table._file_filter.currentData() == a_id, "follows the tab's filter"

        # Dataset: its plots; the table stays a tab of the bottom panel (plan gui-1.6).
        tabs.setCurrentWidget(window._data_file_page)
        qapp.processEvents()
        assert not table.isHidden() and all(plot.isVisible() for plot in plots)
        assert window.dockWidgetArea(table) == Qt.DockWidgetArea.BottomDockWidgetArea
        window._on_page_plot_visibility_action("spatial", False)  # the user closes one
        tabs.setCurrentWidget(window._project_rois)
        tabs.setCurrentWidget(window._data_file_page)
        qapp.processEvents()
        assert window._sensor_frame.isVisible() and not window._spatial_frame.isVisible()
        window._page_plot_actions["spatial"].trigger()

        # Asked for on the Data file tab, the table is the bottom panel's tab in front.
        window._eda_view.roi_table_requested.emit()
        qapp.processEvents()
        assert not table.isHidden() and table in window.tabifiedDockWidgets(window._checks_dock)
        assert table.visibleRegion().isEmpty() is False

        # Closed on Project ROIs, it stays closed there.
        tabs.setCurrentWidget(window._project_rois)
        table.close()
        tabs.setCurrentWidget(window._data_file_page)
        tabs.setCurrentWidget(window._project_rois)
        qapp.processEvents()
        assert table.isHidden()
    finally:
        window.close()


def _roi(roi_id, dataset_id, frames, rows, name=None, position=None):
    from types import SimpleNamespace

    return SimpleNamespace(roi_id=roi_id, dataset_id=dataset_id, name=name or roi_id,
                           spatial_bounds=(frames[0], frames[1], rows[0], rows[1]),
                           provenance={} if position is None else {"grid_position": position})


def test_the_grid_layout_uses_grid_positions_or_scan_order_with_room_for_text():
    from enginir.viewer._project_rois import GRID, grid_cells, mosaic_layout

    # A grid with an empty well (A2): B1 keeps its column, A3 its place.
    tray = [_roi("a1", "t", (0, 99), (0, 49), position="A1"),
            _roi("a3", "t", (400, 499), (0, 49), position="A3"),
            _roi("b1", "t", (0, 99), (200, 249), position="B1")]
    assert grid_cells(tray) == {"a1": (0, 0), "a3": (0, 2), "b1": (1, 0)}
    # Hand-drawn ROIs: rows by overlap, columns along the frames, gaps closed.
    drawn = [_roi("x2", "s", (900, 999), (12, 60)), _roi("x1", "s", (100, 199), (10, 58)),
             _roi("y1", "s", (500, 599), (300, 350))]
    assert grid_cells(drawn) == {"x1": (0, 0), "x2": (0, 1), "y1": (1, 0)}
    # Repeated positions (two grids on one file) fall back to scan order.
    twice = [_roi("p", "u", (0, 9), (0, 9), position="A1"),
             _roi("q", "u", (50, 59), (0, 9), position="A1")]
    assert grid_cells(twice) == {"p": (0, 0), "q": (0, 1)}

    layout = mosaic_layout(tray + drawn, {"t": "tray.h5", "s": "scan.h5"}, GRID)
    places, unit = layout.places, layout.unit
    assert places["x1"][1] == places["x2"][1] and places["x2"][0] > places["x1"][0]
    assert places["x2"][0] - (places["x1"][0] + 100) < 100, "the 700-frame gap is closed"
    for upper, lower in (("x1", "y1"), ("a1", "b1")):
        assert places[lower][1] >= places[upper][1] + places[upper][3] + 1.5 * unit
    scan_header, tray_header = (y for _d, _n, _x, y, _w in layout.headers)
    assert places["a1"][1] >= tray_header + 1.5 * unit
    assert tray_header >= places["y1"][1] + places["y1"][3] + 1.5 * unit


def test_short_names_drop_what_all_rois_share():
    from enginir.viewer._project_rois import _short_names

    names = {"1": "API_10p_L1_01", "2": "API_10p_L1_02", "3": "API_10p_L4_01"}
    files = dict.fromkeys(names, "stream_hsi.h5")
    assert _short_names(names, files) == {"1": "L1_01", "2": "L1_02", "3": "L4_01"}
    assert _short_names({"1": "tray03-B04", "2": "tray04-B04"},
                        {"1": "tray03.h5", "2": "tray04.h5"}) == {"1": "B04", "2": "B04"}
    assert _short_names({"1": "A1", "2": "A2"}, {}) == {"1": "A1", "2": "A2"}
    assert _short_names({"1": "API_10p_L1_01"}, {}) == {"1": "API_10p_L1_01"}


def test_labels_zoom_with_the_tiles_and_stay_in_their_room(qapp, trays):
    project, _paths = trays
    window = _open(qapp, project)
    try:
        view = window._project_rois
        _settle(qapp, view)
        mosaic = view.mosaic
        unit = mosaic._layout.unit
        for roi_id, (x, y, w, h) in mosaic.places.items():
            item = mosaic._label_items[roi_id]
            box = item.mapRectToParent(item.boundingRect())
            assert item.text() and box.height() >= 0.8 * unit, "readable, at the tiles' scale"
            assert box.top() >= y + h - 1e-6 and box.bottom() <= y + h + 1.5 * unit
            assert box.width() <= w + 1e-6, "elided to the tile"
    finally:
        window.close()


def test_the_view_button_switches_layout_and_stretch(qapp, trays):
    from enginir.viewer import _project_rois

    project, _paths = trays
    window = _open(qapp, project)
    try:
        view = window._project_rois
        _settle(qapp, view)
        mosaic = view.mosaic
        a2 = project.roi("tray_a-A2").roi_id
        grid_place = mosaic.places[a2]
        mosaic.set_layout_mode(_project_rois.SCAN)
        assert mosaic.places[a2][0] == 9.0 and mosaic.places[a2] != grid_place
        assert mosaic.tile_count() == 3, "tiles are redrawn, not reloaded"

        shared = {k: item.image.copy() for k, item in mosaic._images.items()}
        mosaic.set_stretch_mode(_project_rois.PER_FILE)
        groups = {roi_id: mosaic._rois[roi_id].dataset_id for roi_id in mosaic._patches}
        expected = _project_rois.stretch_to_rgb(mosaic._patches, groups)
        for roi_id, item in mosaic._images.items():
            np.testing.assert_allclose(item.image, np.transpose(expected[roi_id], (1, 0, 2)),
                                       atol=1e-6)
        assert any(not np.allclose(shared[k], mosaic._images[k].image) for k in shared)
        assert mosaic.plot_table().metadata == {"layout": "scan", "stretch": "per_file"}
    finally:
        window.close()


def test_the_window_never_demands_more_width_than_the_compositor_gives(qapp, trays):
    """WSLg ends the Wayland connection when a maximized window draws a buffer wider than
    the screen (seen at 1963 px on a 1920 px screen after a prediction). The window must not
    turn its docks' minimums into its own minimum size."""
    project, _paths = trays
    window = _open(qapp, project)
    try:
        for page in (window._eda_view, window._modeling_view, window._prediction_view):
            window._workspace_tabs.setCurrentWidget(page)
            qapp.processEvents()
            assert window.minimumSize().width() <= 1 and window.minimumSize().height() <= 1
        assert window._controls_dock.minimumSizeHint().width() < 400, "controls scroll"
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "win32", reason=(
    "offscreen Qt on Windows has no system fonts: its fallback font is about twice as "
    "wide as Segoe UI (a checkbox 336 px instead of 181), so widths are not real"))
def test_every_control_row_fits_in_the_controls_dock(qapp, trays):
    """The Settings column never runs out of the dock sideways: rows wrap and drop-downs
    shrink (plan gui-1.6); it scrolls up and down instead."""
    from PyQt6.QtWidgets import QScrollArea, QWidget

    project, _paths = trays
    window = _open(qapp, project)
    try:
        _settle(qapp, window._project_rois)
        scroll = window._controls_dock.findChild(QScrollArea)
        assert scroll is not None
        content, viewport = scroll.widget(), scroll.viewport()
        assert content is not None and viewport is not None
        for page in (window._eda_view, window._modeling_view, window._prediction_view):
            window._workspace_tabs.setCurrentWidget(page)
            window._prediction_view._readiness_status.setText("long text " * 60)
            qapp.processEvents()
            assert content.width() <= viewport.width(), type(page).__name__
            for child in page.findChildren(QWidget):
                if child.isVisible() and child.width() > 0:
                    right = child.mapTo(content, child.rect().topRight()).x()
                    assert right <= viewport.width() + 1, \
                        f"{type(page).__name__}: {type(child).__name__} {child.objectName()} " \
                        f"{getattr(child, 'text', lambda: '')()!r}"
    finally:
        _settle(qapp, window._project_rois)
        window.close()


# -- Mosaic layout and labels ----------------------------------------------------------


def test_the_mosaic_puts_blocks_side_by_side_in_a_wide_view_and_widens_label_columns():
    from enginir.viewer._project_rois import GRID, mosaic_layout

    rois = [_roi(f"{d}{i}", d, (10 * i, 10 * i + 5), (0, 19)) for d in "abcd" for i in range(5)]
    names = {d: f"{d}.h5" for d in "abcd"}
    stacked = mosaic_layout(rois, names, GRID)
    assert len({x for _d, _n, x, _y, _w in stacked.headers}) == 1, "one column by default"

    wide = mosaic_layout(rois, names, GRID, aspect=3.0)
    columns = sorted({x for _d, _n, x, _y, _w in wide.headers})
    assert len(columns) > 1, "blocks beside each other fill a wide view"
    assert [name for _d, name, _x, _y, _w in wide.headers] == ["a.h5", "b.h5", "c.h5", "d.h5"]
    for _d, _n, x, _y, width in wide.headers:  # no block reaches into its neighbour
        later = [c for c in columns if c > x]
        assert not later or x + width < later[0]
    tall = mosaic_layout(rois, names, GRID, aspect=0.3)
    assert len({x for _d, _n, x, _y, _w in tall.headers}) == 1

    def pitch(layout):
        return layout.places["a1"][0] - layout.places["a0"][0]

    labelled = mosaic_layout(rois, names, GRID, label_chars=10)
    assert pitch(labelled) > pitch(stacked), "room for the label under a narrow tile"
    assert pitch(mosaic_layout(rois, names, GRID, label_chars=200)) \
        == pitch(mosaic_layout(rois, names, GRID, label_chars=14)), "long labels are cut"


def _mosaic_numbers(mosaic) -> str:
    """Layout numbers for a failure message (fonts differ by platform)."""
    from PyQt6.QtGui import QFont, QFontMetricsF

    metrics = QFontMetricsF(QFont(mosaic.font()))
    longest = max(mosaic._labels.values(), key=len, default="")
    widths = sorted({round(place[2], 3) for place in mosaic._layout.places.values()})
    return (f"mode={mosaic.layout_mode} font={mosaic.font().family()!r} "
            f"{mosaic.font().pointSizeF()}pt height={metrics.height():.3f} "
            f"advance({longest!r})={metrics.horizontalAdvance(longest):.3f} "
            f"chars={mosaic._label_chars()} laid_out={mosaic._laid_out_chars} "
            f"unit={mosaic._layout.unit:.3f} label_width={mosaic._label_width:.3f} "
            f"tile_widths={widths[:4]} scale={mosaic._label_items and next(iter(mosaic._label_items.values())).scale():.4f}")


def _skip_without_real_fonts(mosaic) -> None:
    """Offscreen Qt on Windows has no fonts: every glyph is a box as wide as the line
    is high, so every label is over the limit and cut by design; nothing to test."""
    from enginir.viewer._project_rois import _MAX_LABEL_CHARS

    if mosaic._label_chars() > _MAX_LABEL_CHARS:
        pytest.skip(f"no real fonts on this platform: {_mosaic_numbers(mosaic)}")


def test_the_mosaic_shows_whole_tile_labels_for_narrow_tiles(qapp, window):
    window.show_step("rois")
    mosaic = window._project_rois.mosaic
    wait_until(qapp, lambda: mosaic._label_items)
    _skip_without_real_fonts(mosaic)
    shown = {roi_id: item.text() for roi_id, item in mosaic._label_items.items()}
    assert shown and all("…" not in text for text in shown.values()), \
        f"{_mosaic_numbers(mosaic)} {shown}"
    assert all(text == mosaic.label_text(roi_id) for roi_id, text in shown.items())


def test_mosaic_labels_are_measured_in_the_actual_font(qapp, window):
    """A wider font (Segoe UI on Windows) must not cut the labels."""
    from PyQt6.QtGui import QFont

    window.show_step("rois")
    mosaic = window._project_rois.mosaic
    wait_until(qapp, lambda: mosaic._label_items)
    font = QFont(mosaic.font())
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 140)
    mosaic.setFont(font)
    mosaic._lay_out()
    _skip_without_real_fonts(mosaic)
    shown = [item.text() for item in mosaic._label_items.values()]
    assert shown and all("\u2026" not in text for text in shown), \
        f"{_mosaic_numbers(mosaic)} {shown}"


def test_mosaic_labels_that_grow_later_are_not_cut(qapp, window):
    """Values arrive after the first layout (slow machines): the columns must widen."""
    window.show_step("rois")
    mosaic = window._project_rois.mosaic
    wait_until(qapp, lambda: mosaic._label_items)
    full = dict(mosaic._labels)
    mosaic.set_rois(list(mosaic._rois.values()), mosaic._file_names,
                    {roi_id: text[:2] for roi_id, text in full.items()}, mosaic._values)
    mosaic.set_labels(full)
    _skip_without_real_fonts(mosaic)
    shown = {roi_id: item.text() for roi_id, item in mosaic._label_items.items()}
    assert shown == {roi_id: full[roi_id] for roi_id in shown}, \
        f"{_mosaic_numbers(mosaic)} {shown}"
