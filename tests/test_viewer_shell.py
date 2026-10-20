"""Reduced shell smoke tests for the HSIViewer main window."""

from __future__ import annotations

import numpy as np
import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDockWidget, QMainWindow, QSplitter, QTreeWidget

from ._project_payloads import recipe_payload, roi_provenance


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture()
def cube_and_wavelengths():
    rng = np.random.default_rng(42)
    cube = rng.integers(128, 3500, size=(4, 8, 5), dtype=np.uint16)
    wavelengths = np.linspace(1000.0, 1700.0, 5)
    return cube, wavelengths


def _create_reopenable_model_project(path: str, wavelengths: np.ndarray) -> None:
    from enginir.io.project_store import (
        create_project,
        save_calibration_spec,
        save_dataset_ref,
        save_model_run,
        save_prediction_run,
        save_preprocessing_recipe,
        save_sample_set,
        save_signal_domain_contract,
        save_target_assignment,
        save_target_definition,
        save_training_roi,
    )
    from enginir.models.pls import PLSRegression

    create_project(path)
    save_dataset_ref(
        path,
        "dataset_1",
        {
            "source_path": "/data/train.eiff",
            "source_format": "eiff",
            "source_modality": "nir_hsi",
            "cube_shape": (4, 8, 4),
            "n_bands": 4,
            "wavelengths": wavelengths.astype(np.float32),
            "loader_metadata": {},
            "import_timestamp": "2026-04-13T10:00:00+00:00",
        },
    )
    save_calibration_spec(
        path,
        "dataset_1",
        "cal_1",
        {"mode": "raw_dn", "output_signal_unit": "DN", "created_at": "2026-04-13T10:00:01+00:00"},
    )
    save_target_definition(
        path,
        "target_1",
        {
            "name": "moisture",
            "unit": "% w/w",
            "value_type": "float",
            "missing_value_policy": "allowed",
            "created_at": "2026-04-13T10:00:02+00:00",
        },
    )
    save_preprocessing_recipe(
        path,
        "recipe_1",
        recipe_payload(),
    )
    save_signal_domain_contract(
        path,
        "signal_domain_1",
        {
            "input_format": "eiff",
            "source_modality": "nir_hsi",
            "required_signal_unit": "DN",
            "required_calibration_mode": "raw_dn",
            "requires_dark_ref": False,
            "requires_white_ref": False,
            "calibrated_input_allowed": False,
            "n_bands": int(len(wavelengths)),
            "wavelengths": wavelengths.astype(np.float32),
            "wavelength_tolerance_nm": 0.0,
            "band_mapping_policy": "exact",
            "output_dtype": "float32",
            "created_at": "2026-04-13T10:00:04+00:00",
        },
    )
    roi_specs = {
        "roi_1": (
            np.array([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1]], dtype=np.float64),
            1.0,
            (0, 1, 0, 1),
        ),
        "roi_2": (
            np.array([[2.0, 3.0, 4.0, 5.0], [2.1, 3.1, 4.1, 5.1]], dtype=np.float64),
            2.0,
            (2, 3, 0, 1),
        ),
    }
    for index, (roi_id, (spectra, value, bounds)) in enumerate(roi_specs.items(), start=1):
        save_training_roi(
            path,
            roi_id,
            {
                "dataset_id": "dataset_1",
                "calibration_id": "cal_1",
                "name": roi_id,
                "wavelengths": wavelengths.astype(np.float32),
                "source_file": "/data/train.eiff",
                "spatial_bounds": bounds,
                "n_pixels": int(spectra.shape[0]),
                "created_at": f"2026-04-13T10:00:1{index}+00:00",
                "provenance": roi_provenance(
                    created_at=f"2026-04-13T10:00:{10 + index:02d}+00:00"
                ),
            },
            spectra.astype(np.uint16),
        )
        save_target_assignment(
            path,
            "target_1",
            roi_id,
            {"value": value, "assigned_at": f"2026-04-13T10:00:2{index}+00:00"},
        )
    save_sample_set(
        path,
        "sample_set_1",
        {
            "roi_ids": ["roi_1", "roi_2"],
            "target_id": "target_1",
            "observation_unit": "pixel",
            "target_level": "roi",
            "grouping_unit": "roi",
            "validation_unit": "roi",
            "weighting_policy": "per_pixel_unweighted",
            "reporting_level": "roi",
            "roi_aggregation": "mean",
            "preprocessing_recipe_id": "recipe_1",
            "dataset_ids": ["dataset_1"],
            "created_at": "2026-04-13T10:00:30+00:00",
        },
    )
    X = np.vstack([roi_specs["roi_1"][0], roi_specs["roi_2"][0]])
    y = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    model = PLSRegression(n_components=2)
    model.fit(X, y)
    save_model_run(
        path,
        "model_1",
        {
            "sample_set_id": "sample_set_1",
            "signal_domain_id": "signal_domain_1",
            "model_type": "PLS",
            "hyperparameters": {"n_components": 2},
            "cv_spec": {"cv": "logo"},
            "metrics": {"rmsecv": 0.55, "r2": 0.9, "bias": 0.05},
            "artifact_refs": {
                "prediction_model": {
                    "model_type": "PLSRegression",
                    "wavelengths": wavelengths.astype(np.float32),
                    "target_names": ["moisture"],
                    "coef_": np.asarray(model.coef_, dtype=np.float32),
                    "intercept_": np.asarray(model.intercept_, dtype=np.float32),
                    "x_mean_": np.asarray(model.x_mean_, dtype=np.float32),
                    "x_std_": np.asarray(model.x_std_, dtype=np.float32),
                    "y_mean_": np.asarray(model.y_mean_, dtype=np.float32),
                    "y_std_": np.asarray(model.y_std_, dtype=np.float32),
                },
                "diagnostics": {
                    "wavelengths": wavelengths.astype(np.float32),
                    "coef_": np.asarray(model.coef_, dtype=np.float32),
                    "x_loadings_": np.asarray(model.x_loadings_, dtype=np.float32),
                    "x_weights_": np.asarray(model.x_weights_, dtype=np.float32),
                    "y_loadings_": np.asarray(model.y_loadings_, dtype=np.float32),
                    "x_scores_": np.asarray(model.x_scores_, dtype=np.float32),
                    "y_scores_": np.asarray(model.y_scores_, dtype=np.float32),
                },
            },
            "created_at": "2026-04-13T10:00:40+00:00",
        },
    )
    save_prediction_run(
        path,
        "prediction_1",
        {
            "dataset_id": "dataset_1",
            "calibration_id": "cal_1",
            "model_run_id": "model_1",
            "signal_domain_id": "signal_domain_1",
            "preprocessing_recipe_id": "recipe_1",
            "artifact_refs": {
                "source_path": "/data/predict.eiff",
                "y_pred": np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
                "spatial_map": np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
                "clip_low_count": 1,
                "clip_high_count": 2,
                "near_zero_denominator_count": 0,
            },
            "created_at": "2026-04-10T10:05:30+00:00",
        },
    )


def test_shell_initializes_main_window_canvas_and_tabs(qapp, cube_and_wavelengths):
    from enginir.viewer._viewer import HSIViewer

    cube, wavelengths = cube_and_wavelengths
    viewer = HSIViewer(cube, wavelengths)

    assert isinstance(viewer, QMainWindow)
    assert viewer._canvas is not None
    assert viewer._model_diagnostics_widget is None
    assert viewer._predict_map_widget is None

    viewer._workspace_tabs.setCurrentIndex(1)
    qapp.processEvents()
    assert isinstance(viewer._model_workspace_splitter, QSplitter)
    assert viewer._model_workspace_splitter.orientation() == Qt.Orientation.Horizontal
    assert viewer._model_workspace_splitter.widget(0) is viewer._model_graph_panel
    assert viewer._model_workspace_splitter.widget(1) is viewer._model_text_panel
    assert viewer._model_text_panel.maximumWidth() == 360

    viewer._workspace_tabs.setCurrentIndex(2)
    qapp.processEvents()
    assert isinstance(viewer._predict_workspace_splitter, QSplitter)
    assert viewer._predict_workspace_splitter.orientation() == Qt.Orientation.Horizontal
    assert viewer._predict_workspace_splitter.widget(0) is viewer._predict_graph_panel
    assert viewer._predict_workspace_splitter.widget(1) is viewer._predict_text_panel
    assert viewer._predict_text_panel.maximumWidth() == 360
    assert viewer._project_browser.minimumWidth() == 260
    assert viewer._project_browser.maximumWidth() == 260
    assert viewer._inspector.minimumWidth() == 260
    assert viewer._inspector.maximumWidth() == 260
    assert viewer._project_browser_dock.minimumWidth() == 260
    assert viewer._project_browser_dock.maximumWidth() == 260
    assert (
        viewer._project_browser_dock.features()
        == QDockWidget.DockWidgetFeature.DockWidgetClosable
    )
    assert viewer._inspector_dock.minimumWidth() == 260
    assert viewer._inspector_dock.maximumWidth() == 260
    assert viewer._inspector_dock.features() == QDockWidget.DockWidgetFeature.DockWidgetClosable
    assert viewer._controls_dock.minimumHeight() == viewer._controls_dock.maximumHeight()
    assert viewer._controls_dock.maximumHeight() == 230
    assert viewer._spectrum_panel.minimumHeight() == viewer._spectrum_panel.maximumHeight()
    assert viewer._spectrum_panel.maximumHeight() == 220
    assert viewer._spectrum_toggle_action.isCheckable()
    assert [viewer._workspace_tabs.tabText(i) for i in range(viewer._workspace_tabs.count())] == [
        "Explore",
        "Model",
        "Predict",
    ]


def test_status_strip_and_canvas_click_track_cursor_state(qapp, cube_and_wavelengths):
    from enginir.viewer._viewer import HSIViewer

    cube, wavelengths = cube_and_wavelengths
    viewer = HSIViewer(cube, wavelengths)
    viewer._on_canvas_click(2, 3)

    assert viewer.state.frame == 2
    assert viewer.state.row == 3
    status = viewer._status_label.text()
    assert "Frame: 2" in status
    assert "Row: 3" in status


def test_attach_dataset_path_propagates_unexpected_errors(qapp, cube_and_wavelengths, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from enginir.viewer._viewer import HSIViewer

    cube, wavelengths = cube_and_wavelengths
    viewer = HSIViewer(cube, wavelengths)
    criticals: list[str] = []
    monkeypatch.setattr(
        "enginir.viewer._viewer.load_spectra",
        lambda path: (_ for _ in ()).throw(AttributeError("unexpected bug")),
    )
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *args, **kwargs: criticals.append(str(args[2]))
    )

    with pytest.raises(AttributeError, match="unexpected bug"):
        viewer._attach_dataset_path("/tmp/broken.eiff")

    assert criticals == []


def test_reopened_project_renders_model_and_prediction_workspaces(
    qapp, cube_and_wavelengths, tmp_path
):
    from enginir.viewer._viewer import HSIViewer

    cube, wavelengths = cube_and_wavelengths
    project_path = tmp_path / "reopened.h5"
    _create_reopenable_model_project(str(project_path), wavelengths[:4])

    viewer = HSIViewer(cube, wavelengths)
    viewer.project_state.open_project(str(project_path))

    viewer._workspace_tabs.setCurrentWidget(viewer._modeling_view)
    qapp.processEvents()
    assert viewer._workspace_content_stack.currentWidget() is viewer._model_content_page
    assert viewer._model_card_model_value.text() != "none"

    viewer._workspace_tabs.setCurrentWidget(viewer._prediction_view)
    qapp.processEvents()
    assert viewer._workspace_content_stack.currentWidget() is viewer._predict_content_page
    assert viewer._predict_card_prediction_value.text() != "none"
    assert viewer._predict_clipping_value.text() == "low=1 | high=2 | near-zero denom=0"


def test_workspace_controls_height_is_stable_between_modes(qapp, cube_and_wavelengths):
    from enginir.viewer._viewer import HSIViewer

    cube, wavelengths = cube_and_wavelengths
    viewer = HSIViewer(cube, wavelengths)
    expected_height = viewer._controls_dock.maximumHeight()

    for widget, spectrum_visible in (
        (viewer._eda_view, True),
        (viewer._modeling_view, False),
        (viewer._prediction_view, False),
        (viewer._eda_view, True),
    ):
        viewer._workspace_tabs.setCurrentWidget(widget)
        qapp.processEvents()
        assert viewer._controls_dock.minimumHeight() == expected_height
        assert viewer._controls_dock.maximumHeight() == expected_height
        assert viewer._spectrum_panel.isHidden() is not spectrum_visible
        assert viewer._spectrum_toggle_action.isEnabled() is spectrum_visible


def test_project_browser_selection_syncs_inspector_and_canvas(qapp, cube_and_wavelengths, tmp_path):
    from enginir.viewer._viewer import HSIViewer

    cube, wavelengths = cube_and_wavelengths
    project_path = tmp_path / "browser_roi_selection.h5"
    _create_reopenable_model_project(str(project_path), wavelengths[:4])

    viewer = HSIViewer(cube, wavelengths)
    viewer.project_state.open_project(str(project_path))
    qapp.processEvents()

    assert isinstance(viewer._project_browser, QTreeWidget)
    sections = {
        viewer._project_browser.topLevelItem(i).text(0): viewer._project_browser.topLevelItem(i)
        for i in range(viewer._project_browser.topLevelItemCount())
    }
    roi_item = sections["ROIs"].child(1)
    viewer._project_browser.setCurrentItem(roi_item)
    qapp.processEvents()

    assert viewer._canvas._selected_roi_id == "roi_2"
    assert viewer._inspector.title_label.text() == "ROI"

    viewer._on_canvas_roi_selected("roi_1")
    qapp.processEvents()
    payload = viewer._project_browser.currentItem().data(0, Qt.ItemDataRole.UserRole)
    assert payload == {"kind": "roi", "id": "roi_1"}


def test_workspace_refresh_requests_are_coalesced(qapp, cube_and_wavelengths, monkeypatch):
    from enginir.viewer._viewer import HSIViewer

    cube, wavelengths = cube_and_wavelengths
    viewer = HSIViewer(cube, wavelengths)
    calls: list[bool] = []
    monkeypatch.setattr(
        viewer, "_refresh_workspace_content_pages", lambda *_args: calls.append(True)
    )

    viewer.schedule_workspace_content_refresh()
    viewer.schedule_workspace_content_refresh()
    viewer.schedule_workspace_content_refresh()
    qapp.processEvents()

    assert calls == [True]


@pytest.mark.parametrize("button", ["Copy Path", "Open Location"])
def test_help_log_file_dialog_copies_or_opens_the_log_location(
    qapp, cube_and_wavelengths, tmp_path, monkeypatch, button
):
    import enginir
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from enginir.viewer import _viewer
    from enginir.viewer._viewer import HSIViewer

    log_file = tmp_path / "viewer.log"
    shown: list[str] = []
    opened: list = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.append(box.text()))
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda box: next(b for b in box.buttons() if b.text() == button))
    monkeypatch.setattr(_viewer, "reveal_in_file_manager", lambda path: opened.append(path) or True)
    QApplication.clipboard().clear()
    enginir.enable_logging("CRITICAL", file=log_file)
    try:
        viewer = HSIViewer(*cube_and_wavelengths)
        help_entries = [a.text() for a in viewer._menu_help.actions() if a.text()]
        assert help_entries == ["View Log", "Log File...", "About engiNIR"]
        next(a for a in viewer._menu_help.actions() if a.text() == "Log File...").trigger()
    finally:
        enginir.disable_logging()

    assert str(log_file) in shown[0]
    if button == "Copy Path":
        assert QApplication.clipboard().text() == str(log_file) and opened == []
    else:
        assert opened == [log_file] and QApplication.clipboard().text() == ""


def test_about_box_has_no_log_path(qapp, cube_and_wavelengths, monkeypatch):
    from enginir.viewer._viewer import HSIViewer

    from PyQt6.QtWidgets import QMessageBox

    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.append(box.text()))
    viewer = HSIViewer(*cube_and_wavelengths)
    next(a for a in viewer._menu_help.actions() if a.text() == "About engiNIR").trigger()
    assert "engiNIR" in shown[0] and "Log" not in shown[0]


def test_workspace_navigation_preserves_context_and_redraws_only_the_visible_workspace(
    qapp, cube_and_wavelengths, tmp_path, monkeypatch
):
    """G17: tab switches keep the scientific context, create nothing, and skip hidden redraws."""
    from enginir.viewer._viewer import HSIViewer

    from .test_prediction_runs import _create_phase6_model_project

    cube, wavelengths = cube_and_wavelengths
    project_path = str(tmp_path / "navigation.h5")
    _create_phase6_model_project(project_path)
    viewer = HSIViewer(cube, wavelengths)
    project = viewer.project_state
    project.open_project(project_path)
    project.build_sample_set(target_id="target_moisture", preprocessing_recipe_id="recipe_none",
                             sample_set_id="sample_set_1")
    project.fit_model_run(sample_set_id="sample_set_1", signal_domain_id="signal_domain_1",
                          model_type="PLS", hyperparameters={"n_components": 1}, cv="logo",
                          model_run_id="model_1")
    viewer.state.set_frame(3)
    viewer.state.set_row(5)
    viewer.state.set_band(2)
    project.active_model_name = "model_1"

    def context():
        return (project.project_path, viewer.state.frame, viewer.state.row, viewer.state.band,
                viewer._modeling_view.current_target(), project.active_model_name,
                {name: len(getattr(project, name)) for name in (
                    "datasets", "calibrations", "rois", "targets", "sample_sets",
                    "model_runs", "predictions")})

    tabs = viewer._workspace_tabs
    tabs.setCurrentWidget(viewer._modeling_view)
    qapp.processEvents()
    before = context()
    diagnostics = viewer._model_diagnostics_widget
    renders: list[object] = []
    original = diagnostics._render_spectral
    monkeypatch.setattr(diagnostics, "_render_spectral", lambda: (renders.append(1), original()))

    for widget in (viewer._prediction_view, viewer._eda_view, viewer._modeling_view,
                   viewer._prediction_view):
        tabs.setCurrentWidget(widget)
        qapp.processEvents()
        assert context() == before
    renders_while_hidden = len(renders)
    viewer._modeling_view._invalidate_diagnostics_cache()  # a refresh would draw a new payload
    viewer.schedule_workspace_content_refresh()
    qapp.processEvents()
    assert len(renders) == renders_while_hidden, "hidden Model diagnostics must not redraw"

    tabs.setCurrentWidget(viewer._modeling_view)
    qapp.processEvents()
    assert viewer._modeling_view.workspace_diagnostics_payload() is not None
    shown = len(renders)
    viewer.schedule_workspace_content_refresh()
    qapp.processEvents()
    assert len(renders) == shown, "an unchanged diagnostics payload is a no-op redraw"
