"""The Explore step: the view settings, ROIs drawn on the image, exports and the Explore plots."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from matplotlib.image import imread
import numpy as np
import pytest
from enginir.project import FitSpec
from enginir.viewer._app import ensure_app
from enginir.viewer._ui_helpers import PROJECT_SAVE_FILTER, ask_save_path
from enginir import Project
from tests._scans import framed_scan
from tests import _project_payloads as build


# -- Smoke tests for the split EDA view ------------------------------------------------


@pytest.fixture()
def state_and_project(tmp_path):
    from enginir.viewer._state import ProjectState, ViewerState

    cube = np.random.randint(0, 1000, (6, 5, 8), dtype=np.uint16)
    wavelengths = np.linspace(1000.0, 1700.0, 8)
    state = ViewerState(cube, wavelengths, {"source_modality": "nir_hsi"})
    project = ProjectState()
    project.new_project(str(tmp_path / "eda.h5"))
    project.register_dataset(
        build.dataset(
            wavelengths.astype('float32'),
            dataset_id="dataset_1",
            source_path=str(tmp_path / 'cube.eiff'),
            cube_shape=cube.shape,
            n_bands=cube.shape[-1],
        )
    )
    project.set_active_reference_correction(
        build.reference_correction(reference_correction_id="cal_1", dataset_id="dataset_1")
    )
    return state, project


class _FakePanel:
    def __init__(self):
        self.values: list[bool] = []

    def toggle_roi_mode(self, enabled: bool) -> None:
        self.values.append(enabled)

    def toggle_training_roi_mode(self, enabled: bool) -> None:
        self.values.append(enabled)


class _FakeSignal:
    def __init__(self):
        self.values: list[tuple] = []
        self._callbacks: list[object] = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        self.values.append(args)


class _FakeCanvas:
    def __init__(self):
        self.rects: list[tuple[int, int, int, int, str, str]] = []
        self.entered = 0
        self.canceled = 0
        self.selected: list[str | None] = []
        self.roi_draw_finished = _FakeSignal()
        self.roi_draw_canceled = _FakeSignal()
        self.roi_selected = _FakeSignal()

    def clear_roi_rects(self) -> None:
        self.rects.clear()

    def add_roi_rect(self, f0: int, f1: int, r0: int, r1: int, name: str, roi_id: str) -> None:
        self.rects.append((f0, f1, r0, r1, name, roi_id))

    def enter_roi_draw_mode(self) -> None:
        self.entered += 1

    def cancel_roi_draw_mode(self) -> None:
        self.canceled += 1

    def select_roi(self, roi_id: str | None) -> None:
        self.selected.append(roi_id)


def _make_view(
    state,
    project,
    cube_path: str,
    *,
    dataset_attached: bool = True,
    selected_roi_id: list[str | None] | None = None,
):
    from enginir.viewer._eda import EDAView

    canvas = _FakeCanvas()
    selected = selected_roi_id if selected_roi_id is not None else [None]
    view = EDAView(
        state,
        project,
        dc3_panel=_FakePanel(),
        canvas=canvas,
        cube_path_provider=lambda: cube_path,
        dataset_attached_provider=lambda: dataset_attached,
        current_target_provider=lambda: None,
        selected_roi_id_provider=lambda: selected[0],
    )
    return view, canvas


def test_training_roi_selection_registers_dataset_calibration_and_overlay(
    qapp, state_and_project, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QDialog
    from enginir.viewer._viewer import NewROIDialog

    state, project = state_and_project
    view, canvas = _make_view(state, project, str(tmp_path / "cube.eiff"))
    monkeypatch.setattr(
        NewROIDialog,
        "exec",
        lambda self: self._name_edit.setText("roi_a") or QDialog.DialogCode.Accepted,
    )

    view.on_training_roi_selected(0, 1, 0, 1)

    roi = next(iter(project.rois.values()))
    assert roi.name == "roi_a"
    assert roi.dataset_id in project.datasets
    assert roi.reference_correction_id in project.reference_corrections
    assert canvas.rects[0][-1] == roi.roi_id
    assert canvas.selected[-1] == roi.roi_id


def test_training_roi_selection_persists_absorbance_calibration_context(
    qapp, state_and_project, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QDialog
    from enginir.viewer._viewer import NewROIDialog

    state, project = state_and_project
    state.set_y_unit("Absorbance")
    dark_ref = np.arange(state.height * state.n_bands, dtype=np.float64).reshape(
        state.height, state.n_bands
    )
    white_ref = dark_ref + 1000.0
    state.set_dark_ref(dark_ref)
    state.set_white_ref(white_ref)
    view, _ = _make_view(state, project, str(tmp_path / "cube.eiff"))
    monkeypatch.setattr(
        NewROIDialog,
        "exec",
        lambda self: self._name_edit.setText("roi_abs") or QDialog.DialogCode.Accepted,
    )

    view.on_training_roi_selected(0, 1, 0, 1)

    roi = next(iter(project.rois.values()))
    reference_correction = project.reference_corrections[roi.reference_correction_id]
    assert reference_correction.mode == "dark_white_reference"
    assert reference_correction.output_signal_unit == "Absorbance"
    assert reference_correction.dark_ref_data is not None
    assert reference_correction.white_ref_data is not None
    assert project.active_reference_correction_ids["dataset_1"] == roi.reference_correction_id
    from enginir.io.project_store import iter_roi_reference_spectra

    loaded_refs = dict(iter_roi_reference_spectra(project.project_path or "", [roi.roi_id]))
    assert loaded_refs[roi.roi_id] is not None
    loaded_dark, loaded_white = loaded_refs[roi.roi_id]
    expected_dark = np.tile(dark_ref[0:2], (2, 1)).astype(np.float32)
    expected_white = np.tile(white_ref[0:2], (2, 1)).astype(np.float32)
    np.testing.assert_allclose(loaded_dark, expected_dark)
    np.testing.assert_allclose(loaded_white, expected_white)


def test_training_roi_selection_requires_project_and_attached_dataset(qapp, tmp_path, monkeypatch):
    from enginir.viewer._eda import EDAView
    from enginir.viewer._state import ProjectState, ViewerState

    cube = np.random.randint(0, 1000, (4, 4, 4), dtype=np.uint16)
    state = ViewerState(cube, np.linspace(1000.0, 1300.0, 4))
    project = ProjectState()
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warnings.append(str(args[1]))
    )

    view = EDAView(
        state,
        project,
        dc3_panel=_FakePanel(),
        canvas=_FakeCanvas(),
        cube_path_provider=lambda: str(tmp_path / "cube.eiff"),
        dataset_attached_provider=lambda: False,
        current_target_provider=lambda: None,
        selected_roi_id_provider=lambda: None,
    )
    view.on_training_roi_selected(0, 1, 0, 1)

    assert warnings
    assert project.rois == {}


def test_roi_button_toggles_canvas_draw_mode(qapp, state_and_project, tmp_path):
    state, project = state_and_project
    view, canvas = _make_view(state, project, str(tmp_path / "cube.eiff"), dataset_attached=True)

    view._btn_roi_training.click()
    assert canvas.entered == 1

    view._btn_roi_training.setChecked(False)
    assert canvas.canceled == 1


def test_delete_roi_button_removes_selected_roi(qapp, state_and_project, tmp_path, monkeypatch):
    state, project = state_and_project
    selected_roi_id = ["roi_a"]
    view, _ = _make_view(
        state,
        project,
        str(tmp_path / "cube.eiff"),
        dataset_attached=True,
        selected_roi_id=selected_roi_id,
    )
    project.add_roi(
        build.roi(
            state.wavelengths.astype('float32'),
            roi_id="roi_a",
            dataset_id="dataset_1",
            reference_correction_id="cal_1",
            name="A",
            n_pixels=2,
            spatial_bounds=(0, 0, 0, 1),
            source_file=str(tmp_path / 'a.eiff'),
            provenance=build.roi_provenance(),
        ),
        np.zeros((2, state.wavelengths.shape[0]), dtype=np.uint16),
    )
    view.refresh_attachment_state()
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )

    view._delete_roi_btn_clicked()

    assert "roi_a" not in project.rois


def test_delete_roi_button_warns_when_roi_is_used_by_sample_set(
    qapp,
    state_and_project,
    tmp_path,
    monkeypatch,
):
    state, project = state_and_project
    selected_roi_id = ["roi_a"]
    view, _ = _make_view(
        state,
        project,
        str(tmp_path / "cube.eiff"),
        dataset_attached=True,
        selected_roi_id=selected_roi_id,
    )
    project.add_roi(
        build.roi(
            state.wavelengths.astype('float32'),
            roi_id="roi_a",
            dataset_id="dataset_1",
            reference_correction_id="cal_1",
            name="A",
            n_pixels=2,
            spatial_bounds=(0, 0, 0, 1),
            source_file=str(tmp_path / 'a.eiff'),
            provenance=build.roi_provenance(),
        ),
        np.zeros((2, state.wavelengths.shape[0]), dtype=np.uint16),
    )
    project.sample_sets["sample_set_1"] = build.sample_set(
        ['roi_a'],
        sample_set_id="sample_set_1",
        target_id="target_1",
        dataset_ids=['dataset_1'],
        recipe_id="recipe_1",
    )
    project.model_runs["model_run_1"] = build.model_run(
        fit_spec=FitSpec().to_dict(),
        model_run_id="model_run_1",
        sample_set_id="sample_set_1",
        signal_domain_id="signal_domain_1",
        hyperparameters={'n_components': 2},
        cv_spec={'type': 'GroupKFold', 'n_splits': 2},
        metrics={'rmsecv': 1.0},
        artifact_refs={'prediction_model': {'model_type': 'PLSRegression'}},
    )
    captured: dict[str, str] = {}

    def fake_question(*args, **kwargs):
        del kwargs
        captured["message"] = str(args[2])
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", fake_question)

    view._delete_roi_btn_clicked()

    assert "sample_set_1" in captured["message"]
    assert "model_run_1" in captured["message"]
    assert "remain usable for prediction" in captured["message"]
    assert "cannot be altered or refit" in captured["message"]
    assert "roi_a" not in project.rois


def test_export_csv_uses_batched_roi_load_and_preserves_order(
    qapp, state_and_project, tmp_path, monkeypatch
):
    state, project = state_and_project
    wl = state.wavelengths.astype("float32")
    spectra_a = np.arange(16, dtype=np.uint16).reshape(2, 8)
    spectra_b = np.arange(24, dtype=np.uint16).reshape(3, 8) + 100
    for roi_id, name, spectra in [("roi_a", "A", spectra_a), ("roi_b", "B", spectra_b)]:
        project.add_roi(
            build.roi(
                wl,
                roi_id=roi_id,
                dataset_id="dataset_1",
                reference_correction_id="cal_1",
                name=name,
                n_pixels=int(spectra.shape[0]),
                spatial_bounds=(0, 0, 0, spectra.shape[0] - 1),
                source_file=str(tmp_path / f'{name}.eiff'),
                provenance=build.roi_provenance(),
            ),
            spectra,
        )

    captured: dict[str, object] = {}
    batch_calls: list[list[str]] = []
    original_batch = __import__(
        "enginir.viewer._eda", fromlist=["load_roi_spectra_batch"]
    ).load_roi_spectra_batch

    monkeypatch.setattr(
        "enginir.viewer._eda.save_csv",
        lambda path, X, wavelengths, sample_names, columns: captured.update(
            {"path": path, "X": X, "wavelengths": wavelengths, "sample_names": sample_names,
             "columns": columns}
        ),
    )
    monkeypatch.setattr(
        "enginir.viewer._eda.load_roi_spectra_batch",
        lambda path, keys: batch_calls.append(list(keys)) or original_batch(path, keys),
    )

    view, _ = _make_view(state, project, str(tmp_path / "cube.eiff"))
    out_path = str(tmp_path / "out.csv")
    view.export_csv(out_path)

    assert batch_calls == [
        [project.rois["roi_a"].spectra_storage_key, project.rois["roi_b"].spectra_storage_key]
    ]
    np.testing.assert_allclose(captured["X"], np.vstack([spectra_a, spectra_b]).astype(np.float64))
    np.testing.assert_allclose(captured["wavelengths"], wl)
    # The id identifies each row's ROI; the name is a label next to it.
    assert captured["sample_names"] == ["roi_a", "roi_a", "roi_b", "roi_b", "roi_b"]
    assert captured["columns"]["roi_name"] == ["A", "A", "B", "B", "B"]
    assert captured["path"] == out_path


def test_export_csv_rejects_mixed_wavelength_axes(qapp, state_and_project, tmp_path):
    state, project = state_and_project
    wl_a = state.wavelengths.astype("float32")
    wl_b = (state.wavelengths + 1.0).astype("float32")
    project.register_dataset(
        build.dataset(
            wl_b,
            dataset_id="dataset_b",
            source_path=str(tmp_path / 'b.eiff'),
            cube_shape=(6, 5, 8),
            n_bands=8,
        )
    )
    project.set_active_reference_correction(
        build.reference_correction(reference_correction_id="cal_b", dataset_id="dataset_b")
    )
    project.add_roi(
        build.roi(
            wl_a,
            roi_id="roi_a",
            dataset_id="dataset_1",
            reference_correction_id="cal_1",
            name="A",
            n_pixels=2,
            spatial_bounds=(0, 0, 0, 1),
            source_file=str(tmp_path / 'a.eiff'),
            provenance=build.roi_provenance(),
        ),
        np.zeros((2, wl_a.shape[0]), dtype=np.uint16),
    )
    project.add_roi(
        build.roi(
            wl_b,
            roi_id="roi_b",
            dataset_id="dataset_b",
            reference_correction_id="cal_b",
            name="B",
            n_pixels=2,
            spatial_bounds=(1, 1, 0, 1),
            source_file=str(tmp_path / 'b.eiff'),
            provenance=build.roi_provenance(dataset_id='dataset_b', reference_correction_id='cal_b'),
        ),
        np.zeros((2, wl_b.shape[0]), dtype=np.uint16),
    )

    view, _ = _make_view(state, project, str(tmp_path / "cube.eiff"))
    with pytest.raises(ValueError, match="same wavelength axis"):
        view.export_csv(str(tmp_path / "mixed.csv"))


# -- Explore plots as plot frames: exports, header titles and maximize (Plot panel policy) ----


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
    assert raw.title.startswith("Band image, band 3 ")
    viewer.state.set_band(5)
    qapp.processEvents()
    assert raw.title.startswith("Band image, band 5 ")


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


# -- Focused smoke tests for viewer matplotlib panels ----------------------------------


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
    assert panel.plot_title.startswith("Band image, band 4 ")
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


# -- Save dialogs add the file extension the open dialogs filter on --------------------


@pytest.fixture()
def typed(monkeypatch):
    """Make the save dialog return the name in ``typed.name``."""
    class Typed:
        name = ""
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (Typed.name, PROJECT_SAVE_FILTER)))
    return Typed


@pytest.mark.parametrize(("name", "expected"), [
    ("project", "project.h5"),
    ("project.h5", "project.h5"),
    ("project.HDF5", "project.HDF5"),
    ("run.2026", "run.2026.h5"),          # a dot in the name is not an extension
])
def test_the_project_extension_is_added_when_missing(tmp_path, typed, name, expected):
    typed.name = str(tmp_path / name)
    assert ask_save_path(None, "New Project", PROJECT_SAVE_FILTER, ".h5", (".hdf5",)) \
        == str(tmp_path / expected)


def test_replacing_a_file_reached_by_the_added_extension_is_confirmed(tmp_path, typed,
                                                                      monkeypatch):
    (tmp_path / "project.h5").write_bytes(b"old")
    typed.name = str(tmp_path / "project")
    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    asked: list[str] = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda _p, _t, text, *a, **k: asked.append(text) or answers.pop(0)))
    assert ask_save_path(None, "New Project", PROJECT_SAVE_FILTER, ".h5") is None
    assert ask_save_path(None, "New Project", PROJECT_SAVE_FILTER, ".h5") \
        == str(tmp_path / "project.h5")
    assert len(asked) == 2 and "project.h5" in asked[0]


def test_cancelling_returns_nothing(typed):
    typed.name = ""
    assert ask_save_path(None, "New Project", PROJECT_SAVE_FILTER, ".h5") is None


def test_a_new_project_from_explore_gets_the_extension(tmp_path, typed):
    import numpy as np

    from enginir.viewer._state import ProjectState, ViewerState

    ensure_app()
    state = ViewerState(np.zeros((2, 2, 4), dtype=np.uint16), np.linspace(1000, 1300, 4))
    view, _canvas = _make_view(state, ProjectState(), str(tmp_path / "cube.eiff"),
                               dataset_attached=False)
    typed.name = str(tmp_path / "trial")
    view.new_project_btn_clicked()
    assert view.project_state.project_path == str(tmp_path / "trial.h5")
    assert (tmp_path / "trial.h5").exists() and not (tmp_path / "trial").exists()


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="needs a folder the user cannot write to")
def test_a_project_that_cannot_be_written_is_reported_and_nothing_changes(
        tmp_path, typed, monkeypatch):
    import numpy as np

    from enginir.viewer._state import ProjectState, ViewerState

    ensure_app()
    state = ViewerState(np.zeros((2, 2, 4), dtype=np.uint16), np.linspace(1000, 1300, 4))
    view, _canvas = _make_view(state, ProjectState(), str(tmp_path / "cube.eiff"),
                               dataset_attached=False)
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda _parent, title, text, *a: warnings.append(text)))
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        typed.name = str(locked / "trial.h5")
        view.new_project_btn_clicked()
    finally:
        locked.chmod(0o700)
    assert view.project_state.project_path is None
    assert len(warnings) == 1 and "Permission denied" in warnings[0]


# -- The working signal without a shown dataset ----------------------------------------


def test_a_project_without_a_shown_dataset_starts_in_the_working_signal(qapp, window):
    """Project ROIs show the ROIs in their reference correction's signal, not raw counts."""
    assert window._eda_view.shown_dataset_id() is None
    assert window.state.y_unit == "Absorbance"
    assert "Signal: Absorbance" in window._status_label.text()
    # No dataset levels are claimed: the ROIs are converted with their own references.
    assert window._eda_view._signal_info.text() == "each ROI with its own reference correction"


# -- Band numbers and counts above white -----------------------------------------------


def test_a_file_without_wavelengths_offers_band_numbers_only(qapp, tmp_path):
    from enginir.viewer import open_viewer

    window = open_viewer(str(framed_scan(tmp_path / "bands.h5", wavelengths=False)))
    try:
        qapp.processEvents()
        eda = window._eda_view
        assert not window.state.has_wavelengths
        assert not eda._rb_nm.isEnabled() and eda._rb_band.isChecked()
        assert window.state.spectral_axis_label() == "Band"
    finally:
        window.close()
    window = open_viewer(str(framed_scan(tmp_path / "nm.h5")))
    try:
        qapp.processEvents()
        assert window.state.has_wavelengths and window._eda_view._rb_nm.isEnabled()
    finally:
        window.close()


def test_values_above_white_are_counted_and_shown(qapp, tmp_path):
    from enginir.viewer import open_viewer

    window = open_viewer(str(framed_scan(tmp_path / "bright.h5", white=3500, bright=200)))
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


def test_the_reference_frames_are_not_counted_above_white(qapp, tmp_path):
    """Half of a white reference's own values lie above its mean: those frames show
    the reference, not the sample (the example showed "1.2 % above white")."""
    from enginir._example import create_example_project
    from enginir._workflow import own_reference_frames
    from enginir.viewer import open_project

    project = Project.open(create_example_project(tmp_path))
    dataset = next(iter(project.workspace.datasets.values()))
    reference_correction = project.dataset(dataset_id=dataset.dataset_id).reference_correction
    n_frames = dataset.cube_shape[0]
    assert own_reference_frames(reference_correction, dataset.source_path) \
        == ((0, 1), (n_frames - 2, n_frames - 1))
    assert own_reference_frames(reference_correction, tmp_path / "other.h5") == ()

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
