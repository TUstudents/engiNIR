"""The Project ROIs tab: stored ROIs shown from the project file alone (viewer UI contract)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from enginir import Project
from enginir.io import save_hdf5
from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


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


def test_mosaic_layout_keeps_positions_and_stacks_data_files(trays):
    from enginir.viewer._project_rois import mosaic_layout

    project, _paths = trays
    rois = list(project.workspace.rois.values())
    names = {d: ref.source_path.split("/")[-1] for d, ref in project.workspace.datasets.items()}
    places, headers = mosaic_layout(rois, names)
    a1, a2, b1 = (project.roi(n).roi_id for n in ("tray_a-A1", "tray_a-A2", "tray_b-B1"))
    assert [name for name, _y in headers] == ["tray_a.h5", "tray_b.h5"]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = places[a1], places[a2]
    assert (x1, w1, h1) == (3.0, 4.0, 3.0) and (x2, w2, h2) == (9.0, 4.0, 4.0)
    assert y2 - y1 == 5 - 1, "rows keep their distance within a data file"
    assert places[b1][1] > y2 + h2, "the next data file starts below"


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
        assert labels[project.roi("tray_a-A2").roi_id] == "A2 5"
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
        x, y, w, h = view.mosaic._places[a2]
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
        assert view.mosaic._label_items[b1].toPlainText() == "B1 8"
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
