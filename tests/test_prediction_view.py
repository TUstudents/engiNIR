"""Smoke tests for the split prediction view."""

from __future__ import annotations

import time
from pathlib import Path
import types
from threading import Event

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from enginir import Project
from enginir.project import CalibrationSpec, ModelRun, PredictionRun
from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


def _wait_until(qapp, predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    qapp.processEvents()
    assert predicate()


@pytest.fixture()
def project_state(tmp_path):
    from enginir.viewer._state import ProjectState

    state = ProjectState()
    state.new_project(str(tmp_path / "prediction.h5"))
    return state


def _seed_model_run(project_state):
    project_state.model_runs["model_1"] = ModelRun(
        model_run_id="model_1",
        sample_set_id="sample_set_1",
        signal_domain_id="signal_domain_1",
        model_type="PLS",
        hyperparameters={},
        cv_spec={},
        metrics={},
        artifact_refs={},
        created_at="2026-04-02T10:00:00+00:00",
    )
    project_state.active_model_name = "model_1"


def test_prediction_view_warns_when_no_models(qapp, project_state, monkeypatch):
    from enginir.viewer._prediction import PredictionView

    infos: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: infos.append("info"))

    view = PredictionView(project_state)
    view.predict_btn_clicked()

    assert infos == ["info"]
    assert "fit or load a model" in view._readiness_status.text().lower()


def test_prediction_view_cancel_button_requests_cooperative_cancel(qapp, project_state):
    from enginir.viewer._prediction import PredictionView

    view = PredictionView(project_state)
    cancel_event = Event()
    view._job_cancel_event = cancel_event
    view._set_busy(True, "Running prediction...")

    assert view._btn_cancel.isEnabled()
    view._cancel_prediction_clicked()

    assert cancel_event.is_set()
    assert not view._btn_cancel.isEnabled()
    assert "canceling" in view._service_status.label.text().lower()


def test_prediction_map_widget_orients_map_like_spatial_canvas_and_adds_colorbar(qapp):
    from enginir.viewer._prediction import PredictionMapWidget

    widget = PredictionMapWidget()
    spatial_map = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    widget.update_data({"spatial_map": spatial_map, "target_unit": "mg"})

    rendered = np.asarray(widget._ax.images[0].get_array(), dtype=np.float32)
    np.testing.assert_allclose(rendered, spatial_map.T)
    assert widget._ax.get_xlabel() == "Frame"
    assert widget._ax.get_ylabel() == "Row"
    assert widget._colorbar is not None
    assert widget._colorbar.ax.get_ylabel() == "Prediction (mg)"
    assert len(widget._fig.axes) == 2
    assert "2 frames x 3 rows shown" in widget._stats_label.text()

    widget.clear()

    assert widget._colorbar is None
    assert len(widget._fig.axes) == 1


def test_prediction_map_widget_exposes_multi_output_selector(qapp):
    from enginir.viewer._prediction import PredictionMapWidget

    widget = PredictionMapWidget()
    spatial_map = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=np.float32,
    )

    widget.update_data(
        {
            "spatial_map": spatial_map,
            "output_names": ["moisture", "density"],
        }
    )

    assert not widget._output_combo.isHidden()
    assert widget._output_combo.count() == 2
    np.testing.assert_allclose(widget._ax.images[0].get_array(), spatial_map[..., 0].T)

    widget._output_combo.setCurrentIndex(1)

    np.testing.assert_allclose(widget._ax.images[0].get_array(), spatial_map[..., 1].T)


def _subsampled_map_payload() -> dict:
    frames, rows = np.meshgrid(np.arange(12), np.arange(5), indexing="ij")
    spatial_map = np.stack([frames + 0.1 * rows, -frames.astype(float)], axis=-1)
    return {"spatial_map": spatial_map.astype(np.float32), "target_unit": "%",
            "output_names": ["moisture", "fat"], "map_step": 3, "full_map_shape": (36, 15)}


def test_a_subsampled_map_shows_and_exports_true_frames_and_rows(qapp, tmp_path):
    """The preview keeps every 3rd frame and row; axes, hover and CSV use true indices."""
    import csv

    from enginir.viewer._plot_panel import PlotFrame
    from enginir.viewer._prediction import PredictionMapWidget

    widget = PredictionMapWidget()
    frame = PlotFrame("predict.map", widget.plot_title, widget, widget,
                      provenance=lambda: {"project_id": "p"})
    frame.show()
    payload = _subsampled_map_payload()
    widget.update_data(payload)
    assert frame.title == "Prediction map (%)"
    assert widget._ax.get_xlim()[1] == pytest.approx(12 * 3 - 0.5)

    widget._on_motion(types.SimpleNamespace(inaxes=widget._ax, xdata=7.2, ydata=4.0))
    assert widget.hover_text.startswith("Frame 6 | Row 3 |")  # preview pixel (2, 1)

    frame.export_csv(tmp_path / "map.csv")
    lines = (tmp_path / "map.csv").read_text(encoding="utf-8").splitlines()
    provenance = dict(line[2:].split(": ", 1) for line in lines if line.startswith("# "))
    assert provenance["preview_step"] == "3"
    assert provenance["full_map_shape"] == "36 frames x 15 rows"
    assert provenance["output"] == "moisture"
    header, *rows = list(csv.reader(line for line in lines if not line.startswith("# ")))
    assert header == ["row\\frame", *(str(3 * i) for i in range(12))]
    assert [row[0] for row in rows] == [str(3 * i) for i in range(5)]
    np.testing.assert_allclose(np.array([row[1:] for row in rows], dtype=float),
                               payload["spatial_map"][..., 0].T, rtol=1e-6)


def test_header_controls_select_the_output_and_pan_zoom_and_reset_the_view(qapp):
    from enginir.viewer._plot_panel import PlotPanel
    from enginir.viewer._prediction import PREDICTION_HISTOGRAM, PredictionMapWidget

    widget = PredictionMapWidget()
    histogram = PlotPanel(PREDICTION_HISTOGRAM)
    widget.displayed_map_changed.connect(histogram.update_data)
    widget.resize(600, 400)
    widget.show()
    histogram.show()
    payload = _subsampled_map_payload()
    widget.update_data(payload)
    _layer, combo, pan, zoom, reset = widget.header_widgets()

    def histogram_total():
        table = histogram.table()
        assert table is not None
        return int(np.sum(table.columns[2].values))

    assert histogram_total() == 12 * 5
    combo.setCurrentIndex(1)  # the histogram follows the selected output
    np.testing.assert_allclose(widget._ax.images[0].get_array(), payload["spatial_map"][..., 1].T)
    edges = histogram.table().columns[0].values  # type: ignore[union-attr]
    assert edges.min() == pytest.approx(-11.0)

    pan.click()
    assert pan.isChecked() and not zoom.isChecked()
    zoom.click()
    assert zoom.isChecked() and not pan.isChecked()
    zoom.click()
    assert not zoom.isChecked()

    full = widget._ax.get_xlim()
    widget._ax.set_xlim(3.0, 9.0)  # as after zooming in
    widget._toolbar.push_current()
    reset.click()
    assert widget._ax.get_xlim() == pytest.approx(full)


def _session_result(run_id, spatial_map, *, dmodx=None, t2=None, diagnostics=None,
                    artifact_refs=None, source="/data/scan.h5"):
    """An in-memory prediction as the service returns it."""
    from enginir.prediction import PredictionResult
    from enginir.project import DatasetRef

    frames, height = spatial_map.shape[:2]
    dataset = DatasetRef(dataset_id="dataset_1", source_path=source, source_format="hdf5",
                         source_modality="nir_hsi", cube_shape=(frames, height, 3), n_bands=3,
                         wavelengths=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                         loader_metadata={}, import_timestamp="2026-11-07T10:00:00+00:00")
    calibration = CalibrationSpec(
        calibration_id="cal_1", dataset_id="dataset_1", mode="raw_dn", dark_ref_source=None,
        white_ref_source=None, dark_ref_data=None, white_ref_data=None, black_level=None,
        white_level=None, output_signal_unit="DN", created_at="2026-11-07T10:00:00+00:00")
    run = PredictionRun(prediction_run_id=run_id, dataset_id="dataset_1",
                        calibration_id="cal_1", model_run_id="model_1",
                        signal_domain_id="signal_domain_1", preprocessing_recipe_id="recipe_1",
                        artifact_refs=dict(artifact_refs or {"spatial_map_shape": (frames, height)}),
                        created_at="2026-11-07T10:00:00+00:00")
    return PredictionResult(dataset=dataset, calibration=calibration, prediction_run=run,
                            y_pred=spatial_map.reshape(-1), spatial_map=spatial_map,
                            t2=None if t2 is None else t2.reshape(-1),
                            dmodx=None if dmodx is None else dmodx.reshape(-1),
                            diagnostics=diagnostics, provenance={"created_at": "now"})


def test_session_predictions_are_bounded_and_the_latest_is_shown(qapp, project_state):
    from enginir.viewer._prediction import PredictionView
    from enginir.viewer._state import SESSION_PREDICTIONS

    _seed_model_run(project_state)
    view = PredictionView(project_state)
    for index in range(SESSION_PREDICTIONS + 1):
        project_state.add_prediction(_session_result(
            f"p{index}", np.full((2, 3), float(index), dtype=np.float32)))
    assert list(project_state.predictions) == [f"p{i}" for i in range(1, SESSION_PREDICTIONS + 1)]
    payload = view.workspace_visual_payload()
    assert payload is not None
    np.testing.assert_allclose(payload["spatial_map"], SESSION_PREDICTIONS)
    assert payload["layers"] == {}, "no diagnostics without projection data"

    project_state.new_project(str(project_state.project_path) + ".other.h5")
    assert project_state.predictions == {}, "a project switch drops the session predictions"


def test_prediction_summary_reports_run_metadata_and_applicability(qapp, project_state):
    from enginir.viewer._prediction import PredictionView

    _seed_model_run(project_state)
    view = PredictionView(project_state)
    refs = {"spatial_shape": (2, 3), "prediction_min": 1.0, "prediction_max": 4.0,
            "clip_low_count": 2, "clip_high_count": 3, "near_zero_denominator_count": 1}
    summary = {"dcrit_099": 2.0, "t2_crit_099": 9.0, "n_pixels": 6, "n_above_dcrit": 3,
               "share_above_dcrit": 0.5, "n_above_t2crit": 0, "share_above_t2crit": 0.0,
               "mean_prediction_within_limits": 1.5, "mean_prediction_outside_limits": 3.5}
    grid = np.ones((2, 3), dtype=np.float32)
    project_state.add_prediction(_session_result("p", grid, dmodx=grid, t2=grid,
                                                 diagnostics=summary, artifact_refs=refs))
    text = view.workspace_summary()
    assert text["map_shape"] == "3 x 2" and text["map_range"] == "1 .. 4"
    assert text["clipping"] == ("reflectance clipped: 2 below 1e-6, 3 above 1, "
                                "1 with white ≈ dark")
    assert text["diagnostics"] == ("50 % above DCrit, 0 % above T²Crit (whole map, background "
                                   "included)\nMean prediction: 1.5 within the limits, "
                                   "3.5 outside")
    assert text["dataset"] == "scan.h5" and "not stored" in text["status"]

    project_state.add_prediction(_session_result("old", grid))
    assert "refit" in view.workspace_summary()["diagnostics"]


def test_the_layer_selector_shows_the_diagnostics_and_marks_pixels_above_the_limit(qapp):
    from enginir.viewer._prediction import PredictionMapWidget

    widget = PredictionMapWidget()
    prediction = np.arange(6, dtype=np.float32).reshape(3, 2)
    dmodx = np.array([[0.5, 0.7], [3.0, 0.2], [0.1, 0.9]], dtype=np.float32)
    payload = {"spatial_map": prediction, "layers": {"DModXPS": dmodx, "T²PS": dmodx * 2},
               "diagnostics": {"dcrit_099": 1.0, "n_above_dcrit": 1, "share_above_dcrit": 1 / 6,
                               "t2_crit_099": 4.0, "n_above_t2crit": 1,
                               "share_above_t2crit": 1 / 6}}
    shown: list = []
    widget.displayed_map_changed.connect(shown.append)
    widget.update_data(payload)
    assert len(widget._ax.images) == 1

    widget.set_layer("DModXPS")
    np.testing.assert_allclose(widget._ax.images[0].get_array(), dmodx.T)
    overlay = widget._ax.images[1].get_array()
    assert int((~np.ma.getmaskarray(overlay)).sum()) == 1, "one pixel above DCrit is marked"
    assert not np.ma.getmaskarray(overlay)[0, 1]  # frame 1, row 0 (shown transposed)
    assert "above DCrit 1: 1 pixels (17 %)" in widget._stats_label.text()
    assert widget.plot_title == "DModXPS map"
    assert shown[-1]["quantity"] == "DModXPS" and shown[-1]["limit"] == 1.0
    assert widget.plot_table_metadata()["layer"] == "DModXPS"

    widget.update_data({"spatial_map": prediction})  # a model without projection data
    model = widget._layer_combo.model()
    assert not any(model.item(i).isEnabled() for i in (1, 2, 3))
    assert widget._current_display_map is None
    texts = " ".join(text.get_text() for text in widget._ax.texts)
    assert "Refit" in texts


# -- Predict ROIs and Predict map on a real project (viewer contract §3) ---------------------


@pytest.fixture()
def fitted_tray(tmp_path):
    from enginir import PreprocessingRecipe
    from enginir.viewer._state import ProjectState
    from tests.test_prediction_rois import tray

    project, _scan, path = tray.__wrapped__(tmp_path)
    model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=2, cv=3)
    state = ProjectState()
    state.open_project(str(project.path))
    return project, model, state, path


def test_predict_map_names_the_chain_and_refuses_an_incompatible_file(
        qapp, fitted_tray, tmp_path):
    from enginir.io import save_hdf5
    from enginir.viewer._predict_dialogs import PredictMapDialog
    from enginir.viewer._prediction import PredictionView

    project, model, state, _path = fitted_tray
    raw = tmp_path / "raw.h5"
    save_hdf5(raw, np.full((4, 5, 24), 2000, dtype=np.uint16), np.linspace(1000, 1600, 24))
    raw_id = project.add_dataset(raw, unit="DN").ref.dataset_id
    state.open_project(str(project.path))
    view = PredictionView(state)
    choices = view._model_choices()
    assert [choice.name for choice in choices] == [model.name], "names, not ids"
    assert "Absorbance → SNV" in choices[0].chain and "dark/white references" in choices[0].chain
    tray_id = next(d for d in state.datasets if d != raw_id)
    assert view._map_check(model.ref.model_run_id, tray_id) is None
    reason = view._map_check(model.ref.model_run_id, raw_id)
    assert reason is not None and "raw_dn" in reason
    dialog = PredictMapDialog(choices, [(tray_id, "tray"), (raw_id, "raw")],
                              check=view._map_check, default_file=raw_id)
    ok = dialog._buttons.button(dialog._buttons.StandardButton.Ok)
    assert ok is not None and not ok.isEnabled() and "Cannot predict" in dialog._reason.text()
    dialog._file_combo.setCurrentIndex(dialog._file_combo.findData(tray_id))
    assert ok.isEnabled()


def test_predict_map_shows_the_map_and_the_rois_of_its_file(qapp, fitted_tray, monkeypatch):
    from enginir.viewer._predict_dialogs import PredictMapDialog
    from enginir.viewer._prediction import PredictionView

    project, model, state, _path = fitted_tray
    monkeypatch.setattr(PredictMapDialog, "exec", lambda self: self.DialogCode.Accepted)
    view = PredictionView(state)
    view.predict_btn_clicked()
    _wait_until(qapp, lambda: bool(state.predictions) and bool(view._roi_predictions),
                timeout=30)
    summary = view.workspace_summary()
    assert summary["model"] == model.name and "dark/white references" in summary["calibration"]
    payload = view.roi_results_payload()
    names = {row["roi"] for row in payload["roi_rows"]}
    assert names == {"t08", "t09", "t10"}
    visual = view.workspace_visual_payload()
    assert visual is not None and visual["colour_range"] == (1.0, 7.0), "training values"
    assert len(visual["roi_overlays"]) == 3
    # The map and the ROI path agree for an ROI (one chain).
    row = next(r for r in payload["roi_rows"] if r["roi"] == "t09")
    f0, f1, r0, r1 = project.roi("t09").spatial_bounds
    result = next(iter(state.predictions.values()))
    np.testing.assert_allclose(row["mean"], float(np.mean(
        result.spatial_map[f0:f1 + 1, r0:r1 + 1])), rtol=1e-5)


def test_the_prediction_interval_layer_shows_the_pixel_intervals_and_none_outside_the_model(
        qapp, fitted_tray, monkeypatch):
    """The half-width at 0.95 of the library, its own scale, pixels outside the model marked,
    the median and the share without an interval, and ± in the hover readout."""
    from enginir.viewer._predict_dialogs import PredictMapDialog
    from enginir.viewer._prediction import PredictionMapWidget, PredictionView

    _project, model, state, path = fitted_tray
    monkeypatch.setattr(PredictMapDialog, "exec", lambda self: self.DialogCode.Accepted)
    view = PredictionView(state)
    view.predict_btn_clicked()
    _wait_until(qapp, lambda: bool(state.predictions), timeout=30)
    expected = model.predict(path).interval_map(0.95)
    assert expected is not None
    outside = np.isnan(expected)
    assert outside.any() and not outside.all(), "the dark and white frames are outside"

    widget = PredictionMapWidget()
    payload = view.workspace_visual_payload()
    assert payload is not None
    widget.update_data(payload)
    widget.set_layer("Prediction interval")
    shown = widget._ax.images[0].get_array()
    np.testing.assert_allclose(np.ma.filled(shown, np.nan), expected.T, rtol=1e-6)
    assert widget._ax.images[0].get_clim() == pytest.approx(
        (np.nanmin(expected), np.nanmax(expected)), rel=1e-6)
    label = widget._stats_label.text()
    median = float(np.median(expected[~outside]))
    assert f"95 % interval: median ± {median:.3g}" in label
    assert f"no interval for {int(outside.sum())} pixels" in label
    assert widget.plot_title.startswith("± 95 % prediction interval")

    frame, row = np.argwhere(~outside)[0]
    assert f"± {expected[frame, row]:.3g} (95 %)" in widget.describe_pixel(int(frame), int(row))
    frame, row = np.argwhere(outside)[0]
    assert "no interval (outside the model)" in widget.describe_pixel(int(frame), int(row))
    assert "95 % interval: median ±" in view.workspace_summary()["diagnostics"]


def test_predict_rois_compares_models_and_the_exclusion_needs_no_new_prediction(
        qapp, fitted_tray, monkeypatch):
    from enginir.viewer import _prediction
    from enginir.viewer._predict_dialogs import PredictRoisDialog
    from enginir.viewer._prediction import PredictionView

    project, model, state, _path = fitted_tray
    other = project.fit("API", n_components=3, cv=3)
    state.open_project(str(project.path))

    def accept(self):
        for row in range(self._model_list.count()):
            item = self._model_list.item(row)
            item.setCheckState(item.checkState().Checked)
        return self.DialogCode.Accepted

    monkeypatch.setattr(PredictRoisDialog, "exec", accept)
    calls: list[str] = []
    real = _prediction.predict_rois
    monkeypatch.setattr(_prediction, "predict_rois",
                        lambda **kw: calls.append(kw["model_name"]) or real(**kw))
    view = PredictionView(state)
    view.predict_rois_clicked()
    _wait_until(qapp, lambda: len(view._roi_predictions) == 2, timeout=30)
    rows = view.roi_results_payload()["roi_rows"]
    assert {row["model"] for row in rows} == {model.name, other.name} and len(rows) == 6
    assert set(view.roi_results_payload()["rmsep"]) == {model.name, other.name}

    before = view.roi_results_payload()
    view._exclusion_check.setChecked(True)
    view._exclusion_level.setValue(0.5)  # a strict limit removes pixels
    after = view.roi_results_payload()
    assert after is not before and sum(row["removed"] for row in after["roi_rows"]) > 0
    assert len(calls) == 2, "recalculated from the predicted pixels"
    assert "excluded" in after["exclusion_text"]

    # The histogram shows the pixels behind the statistics: the kept ROI pixels per model.
    shown = {"values": np.full((4, 4), -60.0), "provenance": {}}  # a background-only map
    histogram = view.histogram_payload(shown)
    assert histogram is not None
    series = dict(histogram["series"])
    assert set(series) == {model.name, other.name}, "the ROI pixels, not the map"
    for name in series:
        kept = sum(row["pixels"] - row["removed"] for row in after["roi_rows"]
                   if row["model"] == name)
        assert series[name].size == kept
    view._exclusion_check.setChecked(False)
    every = view.histogram_payload(shown)["series"]
    assert sum(values.size for _name, values in every) > sum(v.size for v in series.values())


def test_a_map_of_a_file_without_prediction_rois_drops_the_old_roi_results(
        qapp, fitted_tray, monkeypatch, tmp_path):
    """Map and ROI results always describe the same data file."""
    import shutil

    from enginir.viewer._predict_dialogs import PredictMapDialog
    from enginir.viewer._prediction import PredictionView

    project, model, state, path = fitted_tray
    copy = tmp_path / "other" / "tray.h5"
    copy.parent.mkdir()
    shutil.copy(path, copy)
    n = 2 + 10 * 4 + 2
    other_id = project.add_dataset(copy, dark_frames=(0, 1),
                                   white_frames=(n - 2, n - 1)).ref.dataset_id
    state.open_project(str(project.path))
    view = PredictionView(state)
    view._set_roi_predictions([model.roi_prediction()], scope="tray")
    assert view.roi_results_payload()["roi_rows"]
    label = view._map_file_label(other_id)
    assert label.endswith("no prediction ROIs") and "other" in label
    monkeypatch.setattr(PredictMapDialog, "exec", lambda self: self.DialogCode.Accepted)
    monkeypatch.setattr(PredictMapDialog, "dataset_id", property(lambda self: other_id))
    view.predict_btn_clicked()
    _wait_until(qapp, lambda: bool(state.predictions), timeout=30)
    _wait_until(qapp, lambda: not view._roi_predictions, timeout=10)
    payload = view.roi_results_payload()
    assert not payload["roi_rows"] and "No prediction ROIs on" in payload["message"]
    assert view.workspace_summary()["rois"].startswith("No prediction ROIs on")


def test_hovering_a_map_pixel_describes_it_in_the_status_line(qapp):
    """Frame, row, prediction against the color range, DModX against DCrit and the
    exclusion limit, T² and the prediction ROI under the cursor; leaving clears it."""
    from enginir.viewer._prediction import PredictionMapWidget

    widget = PredictionMapWidget()
    shown: list[str] = []
    widget.hovered.connect(shown.append)
    prediction = np.array([[2.0, 12.0], [5.0, 0.5], [4.0, 6.0]], dtype=np.float32)
    dmodx = np.array([[0.5, 3.0], [1.2, 0.1], [0.2, 0.9]], dtype=np.float32)
    widget.update_data({"spatial_map": prediction, "layers": {"DModXPS": dmodx, "T²PS": dmodx * 2},
                        "diagnostics": {"dcrit_099": 1.0}, "colour_range": (1.0, 10.0),
                        "removed_limit": 2.0,
                        "roi_overlays": [((0, 0, 1, 1), "03  12", "03")]})
    # spatial_map is (frames, rows): the shown map is (rows, frames).
    text = widget.describe_pixel(1, 0)
    assert text == ("Frame 1 | Row 0 | Prediction 5 | DModXPS 1.2 (DCrit 0.99 1: above, "
                    "exclusion limit 2) | T²PS 2.4")
    text = widget.describe_pixel(0, 1)
    assert "Prediction 12 (above the color range)" in text
    assert "DModXPS 3 (DCrit 0.99 1: above, exclusion limit 2: excluded)" in text
    assert text.endswith("ROI 03")
    widget._on_motion(types.SimpleNamespace(inaxes=widget._ax, xdata=0.0, ydata=1.0))
    widget._on_motion(types.SimpleNamespace(inaxes=None, xdata=None, ydata=None))
    assert shown[-2].endswith("ROI 03") and shown[-1] == ""


def test_the_roi_table_shows_the_session_results_and_sorts_by_them(qapp, fitted_tray):
    """Per model: mean, SD, % excluded, % above DCrit, residual; read-only, following the
    pixel removal, sortable, and gone with another project (never stored)."""
    from enginir.viewer._prediction import PredictionView
    from enginir.viewer._roi_table import RoiTableModel

    project, model, state, _path = fitted_tray
    table = RoiTableModel(state)
    n_columns = table.columnCount()
    view = PredictionView(state)
    prediction = model.roi_prediction()
    glare = prediction.pixels[project.roi("t09").roi_id]
    dmodx = glare.dmodx.copy()
    dmodx[:4] = 1e3  # four pixels far outside the model
    prediction.pixels[project.roi("t09").roi_id] = type(glare)(y=glare.y, dmodx=dmodx,
                                                               t2=glare.t2)
    view._set_roi_predictions([prediction], scope="tray")
    assert table.columnCount() == n_columns + 5
    headers = [table.headerData(c, Qt.Orientation.Horizontal) for c in
               range(n_columns, n_columns + 5)]
    assert headers == [f"{model.name}: {h}" for h in
                       ("mean", "SD", "% excluded", "% above DCrit", "residual")]
    mean_column = n_columns
    row = table.row_of(project.roi("t09").roi_id)
    assert row is not None
    expected = next(r for r in view._roi_summaries[0].rois
                    if r.roi_id == project.roi("t09").roi_id)
    assert table.data(table.index(row, mean_column)) == f"{expected.mean:.3g}"
    assert not table.flags(table.index(row, mean_column)) & Qt.ItemFlag.ItemIsEditable
    training_row = table.row_of(project.roi("t01").roi_id)
    assert training_row is not None and table.data(table.index(training_row, mean_column)) == ""

    assert table.data(table.index(row, n_columns + 2)) == "0"
    view._exclusion_check.setChecked(True)
    removed = float(table.data(table.index(row, n_columns + 2)))
    assert removed >= 100 * 4 / glare.y.size - 0.5, "the table follows the pixel removal"

    table.sort(mean_column, Qt.SortOrder.DescendingOrder)
    means = [table.data(table.index(r, mean_column)) for r in range(table.rowCount())]
    filled = [float(m) for m in means if m]
    assert filled == sorted(filled, reverse=True) and means[len(filled):] == [""] * (
        len(means) - len(filled)), "by value, empty cells last"

    other = Project.create(Path(project.path).parent / "other.h5")
    state.open_project(str(other.path))
    assert state.roi_results == [] and table.columnCount() == len(
        ("Data file", "ROI", "Pixels", "Predict", "Sample"))


def test_the_per_roi_plot_rings_unusual_tablets_and_has_no_share_axis(qapp):
    from matplotlib.figure import Figure

    from enginir.viewer._predict_dialogs import _render_roi_results
    from enginir.viewer._theme import panel_colors

    def row(name, share):
        return {"roi": name, "roi_id": name, "model": "PLS", "mean": 5.0, "p05": 4.0,
                "p95": 6.0, "reference": None, "share_above_dcrit": share}

    def legend_texts(rows):
        fig = Figure()
        _render_roi_results(fig, {"roi_rows": rows, "training_share": (0.02, 0.1, 0.3)},
                            panel_colors())
        assert len(fig.axes) == 1, "no right axis"
        return [text.get_text() for text in fig.axes[0].get_legend().get_texts()]

    assert not any("unusual" in t for t in legend_texts([row("01", 0.05), row("02", 0.29)]))
    assert any("unusual" in t for t in legend_texts([row("01", 0.05), row("02", 0.6)]))


def test_the_summary_names_file_and_model_once_and_controls_change_the_map(
        qapp, fitted_tray, monkeypatch):
    from enginir.viewer._predict_dialogs import PredictMapDialog
    from enginir.viewer._prediction import PredictionView

    _project, model, state, _path = fitted_tray
    monkeypatch.setattr(PredictMapDialog, "exec", lambda self: self.DialogCode.Accepted)
    view = PredictionView(state)
    view.predict_btn_clicked()
    _wait_until(qapp, lambda: bool(state.predictions) and bool(view._roi_predictions),
                timeout=30)
    summary = view.workspace_summary()
    assert summary["headline"].startswith("Map of tray.h5 with " + model.name)
    shown = "\n".join(summary[key] for key in ("headline", "rois", "signal", "target",
                                                 "calibration", "map_shape", "clipping"))
    assert shown.count("tray.h5") == 1 and shown.count(model.name) == 1
    assert "Map: tray.h5" in view.status_text() and model.name in view.status_text()

    before = view.workspace_visual_payload()
    assert before is not None and before["show_roi_outlines"]
    view.set_colour_limits((2.0, 4.0))
    assert view.colour_limits == (2.0, 4.0) and view._colour_mode.currentData() == "own"
    view._show_outlines.setChecked(False)
    after = view.workspace_visual_payload()
    assert after is not before and not after["show_roi_outlines"]
    view._histogram_source.setCurrentIndex(view._histogram_source.findData("map"))
    map_payload = {"values": np.ones((2, 2)), "provenance": {}}
    assert view.histogram_payload(map_payload) is map_payload
    view.set_colour_limits(None)
    assert view.colour_limits is None
