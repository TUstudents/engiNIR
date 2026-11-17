"""Fixes of the GUI review part A (plan gui-1.6, Phase 7; plans/gui-review-1.6.md)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from enginir import Project
from enginir._workflow import data_file_labels
from enginir.io import save_hdf5
from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


def _scan(path, *, wavelengths=True, white=3500, bright=0):
    rng = np.random.default_rng(3)
    cube = rng.integers(1500, 2500, (20, 6, 10)).astype(np.uint16)
    cube[:2], cube[-2:] = 100, white
    if bright:
        cube[5:7, :, :] = white + bright  # brighter than the white reference
    axis = np.linspace(1000, 1600, 10) if wavelengths else np.arange(10, dtype=float)
    save_hdf5(str(path), cube, axis)
    return path


def test_a_dataset_label_is_stored_shown_and_tolerated_when_missing(tmp_path):
    project = Project.create(tmp_path / "p.h5")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = project.add_dataset(_scan(tmp_path / "a" / "stream.h5"))
    b = project.add_dataset(_scan(tmp_path / "b" / "stream.h5"))
    labels = data_file_labels(project.workspace.datasets)
    assert labels[a.ref.dataset_id].startswith("stream.h5 (")
    a.set_label("Layer 7")
    reopened = Project.open(project.path)
    assert reopened.workspace.datasets[a.ref.dataset_id].label == "Layer 7"
    labels = data_file_labels(reopened.workspace.datasets)
    assert labels[a.ref.dataset_id] == "Layer 7"
    assert labels[b.ref.dataset_id] == "stream.h5 (b)", "the other keeps its file label"

    with h5py.File(project.path, "a") as handle:  # a 1.5.x project: no label attribute
        del handle["datasets"][a.ref.dataset_id].attrs["label"]
        handle["datasets"][b.ref.dataset_id].attrs["label"] = 7  # malformed display data
    reopened = Project.open(project.path)
    assert reopened.workspace.datasets[a.ref.dataset_id].label == ""
    assert reopened.workspace.datasets[b.ref.dataset_id].label == ""


def test_a_file_without_wavelengths_offers_band_numbers_only(qapp, tmp_path):
    from enginir.viewer import open_viewer

    window = open_viewer(str(_scan(tmp_path / "bands.h5", wavelengths=False)))
    try:
        qapp.processEvents()
        eda = window._eda_view
        assert not window.state.has_wavelengths
        assert not eda._rb_nm.isEnabled() and eda._rb_band.isChecked()
        assert window.state.spectral_axis_label() == "Band"
    finally:
        window.close()
    window = open_viewer(str(_scan(tmp_path / "nm.h5")))
    try:
        qapp.processEvents()
        assert window.state.has_wavelengths and window._eda_view._rb_nm.isEnabled()
    finally:
        window.close()


def test_values_above_white_are_counted_and_shown(qapp, tmp_path):
    from enginir.viewer import open_viewer

    window = open_viewer(str(_scan(tmp_path / "bright.h5", white=3500, bright=200)))
    try:
        state = window.state
        state.set_y_unit("Reflectance")
        qapp.processEvents()
        share = state.reflectance_clip_share()
        expected = np.count_nonzero(state.cube > state.white_level) / state.cube.size
        assert share == pytest.approx(expected) and share > 0.001
        assert "above white and clipped" in window._eda_view._signal_info.text()
        state.set_y_unit("DN")
        qapp.processEvents()
        assert "clipped" not in window._eda_view._signal_info.text()
    finally:
        window.close()


def test_a_long_plot_title_ends_in_an_ellipsis_with_the_whole_title_as_tooltip(qapp):
    from PyQt6.QtWidgets import QWidget

    from enginir.viewer._plot_panel import PlotFrame

    frame = PlotFrame("x.test", "Predicted vs target value, per ROI, cross-validated, with "
                      "all the detail one could want", QWidget(), None)
    frame.resize(220, 300)
    frame.show()
    qapp.processEvents()
    assert frame.title_elided
    assert frame.title.startswith("Predicted vs target value")
    assert frame._title.toolTip() == frame.title
    frame.resize(2000, 300)
    qapp.processEvents()
    assert not frame.title_elided
    frame.close()


def test_predict_names_the_waiting_rois_and_the_model(qapp, tmp_path):
    from enginir.viewer import open_project

    project = Project.create(tmp_path / "p.h5")
    project.add_target("API", unit="%")
    scan = project.add_dataset(_scan(tmp_path / "s.h5"), dark_frames=(0, 1),
                               white_frames=(18, 19))
    for i in range(4):
        scan.add_roi(f"c{i}", frames=(3 + 3 * i, 4 + 3 * i), rows=(0, 5),
                     values={"API": float(i)})
    scan.add_roi("p0", frames=(3, 4), rows=(0, 5), job="prediction")
    project.fit("API", n_components=1)
    window = open_project(project.path)
    try:
        summary = window._prediction_view.workspace_summary()
        assert summary["rois"] == "none predicted yet (1 prediction ROI)"
        assert summary["model"] != "none", "the model a prediction would use"
    finally:
        window.close()


def test_opening_a_dataset_from_the_project_tree_shows_its_image(qapp, tmp_path):
    from enginir.viewer import open_project

    project = Project.create(tmp_path / "p.h5")
    scan = project.add_dataset(_scan(tmp_path / "s.h5"))
    scan.add_roi("r", frames=(3, 5), rows=(0, 5))
    window = open_project(project.path)
    try:
        window.show_step("rois")
        window.show_dataset(scan.ref.dataset_id)
        window.wait_for_data_file()
        assert window._explore_tabs.currentWidget() is window._data_file_page
        assert window._eda_view.shown_dataset_id() == scan.ref.dataset_id
    finally:
        window.close()


def test_the_reference_frames_are_not_counted_above_white(qapp, tmp_path):
    """Half of a white reference's own values lie above its mean: those frames show
    the reference, not the sample (the example showed "1.2 % above white")."""
    from enginir._example import create_example_project
    from enginir._workflow import own_reference_frames
    from enginir.viewer import open_project

    project = Project.open(create_example_project(tmp_path))
    dataset = next(iter(project.workspace.datasets.values()))
    calibration = project.dataset(dataset_id=dataset.dataset_id).calibration
    n_frames = dataset.cube_shape[0]
    assert own_reference_frames(calibration, dataset.source_path) \
        == ((0, 1), (n_frames - 2, n_frames - 1))
    assert own_reference_frames(calibration, tmp_path / "other.h5") == ()

    window = open_project(project.path)
    try:
        window.show_dataset(dataset.dataset_id)
        window.wait_for_data_file()
        window.state.set_y_unit("Reflectance")
        qapp.processEvents()
        assert window.state.reference_frames == ((0, 1), (n_frames - 2, n_frames - 1))
        assert window.state.reflectance_clip_share() == 0.0
        assert "clipped" not in window._eda_view._signal_info.text()
    finally:
        window.close()
