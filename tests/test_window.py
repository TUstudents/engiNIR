"""The main window: menus, workflow strip, docks, project browser, Checks and Models panels."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDockWidget, QMainWindow, QSplitter, QTreeWidget
import numpy as np
import pytest
from enginir import Project
from enginir.io import save_hdf5
from enginir.project import FitSpec
from enginir.project._models import software_versions
from tests._project_payloads import projection_payload, recipe_payload, roi_provenance
from tests._qt import wait_until
from tests._scans import framed_scan


# -- The main window of engiNIR 1.6 (plan gui-1.6, Phases 3–5) -------------------------


N_BANDS, HEIGHT, FRAMES = 12, 4, 3


def _scan(path, contents, seed):
    rng = np.random.default_rng(seed)
    peak = np.exp(-((np.arange(N_BANDS) - N_BANDS / 2) / 2) ** 2)
    content = np.repeat(np.asarray(contents, dtype=float), FRAMES)[:, None, None]
    cube = 2500 - 150 * content * peak + rng.normal(0, 3, (len(content), HEIGHT, N_BANDS))
    save_hdf5(str(path), cube.astype(np.uint16), np.linspace(1000, 1600, N_BANDS))
    return str(path)


@pytest.fixture()
def layers(tmp_path):
    """Two scans, one sample each, four ROIs per scan; ROI high3 has a typing error."""
    project = Project.create(tmp_path / "layers.h5")
    project.add_target("Layers", unit="1")
    for name, value in (("low", 2.0), ("high", 5.0)):
        scan = project.add_dataset(_scan(tmp_path / f"{name}.h5", [value] * 4, int(value)))
        for i in range(4):
            roi = scan.add_roi(f"{name}{i}", frames=(FRAMES * i, FRAMES * i + 2),
                               rows=(0, HEIGHT - 1),
                               values={"Layers": value + (1.0 if name == "high" and i == 3
                                                          else 0.0)})
            project.set_sample(roi, name)
    return project


def _open(qapp, project):
    from enginir.viewer import open_project

    window = open_project(project.path)
    window.resize(1920, 1080)
    window.show()
    # The Project ROIs tiles load in the background and read the project file; a write
    # (the fit) waits until they are in, as a user would.
    view = window._project_rois
    wait_until(qapp, lambda: not view.busy() and not view.mosaic.missing_patches()
                and not view._rebuild_timer.isActive())
    window.refresh_checks()
    return window


def _fit(qapp, window, n_components=1, **form):
    window.show_step("model")  # the Fit settings are built when the step is first shown
    model = window._modeling_view
    fit_form = model.fit_form
    assert fit_form is not None
    fit_form._components_spin.setValue(n_components)
    fit_form._signal_unit_combo.setCurrentIndex(fit_form._signal_unit_combo.findData(
        "Absorbance"))
    if "group_by" in form:
        fit_form._group_by_combo.setCurrentIndex(fit_form._group_by_combo.findData(
            form["group_by"]))
    before = set(window.project_state.model_runs)
    window.show_step("model")
    model._btn_fit_model.click()
    wait_until(qapp, lambda: set(window.project_state.model_runs) - before
                and model._job_thread is None)
    (new,) = set(window.project_state.model_runs) - before
    window.refresh_checks()
    return window.project_state.model_runs[new]


def test_every_workspace_action_is_a_menu_item_and_follows_its_button(qapp, layers):
    window = _open(qapp, layers)
    try:
        titles = [action.text() for action in window.menuBar().actions()]
        assert titles == ["File", "Edit", "View", "Project", "ROI", "Model", "Predict",
                          "Tools", "Help"]
        assert window.findChildren(type(window._workflow_bar)) == [window._workflow_bar], \
            "the workflow strip is the only bar: no toolbar of actions"
        from enginir.viewer._viewer import _button_of

        # A page built on first use gives its buttons through a function (D8).
        pairs = [(action, _button_of(button)) for action, button in window._button_actions]
        buttons = {button for _action, button in pairs}
        assert window._modeling_view._btn_fit_model in buttons
        assert window._prediction_view._btn_predict_rois in buttons
        assert window._eda_view._btn_roi_grid in buttons
        for action, button in pairs:
            button.setEnabled(False)
            window._sync_button_actions()
            assert not action.isEnabled(), action.text()
            button.setEnabled(True)
            window._sync_button_actions()
            assert action.isEnabled(), action.text()
    finally:
        window.close()


def test_the_strip_opens_each_step_and_shows_the_open_checks(qapp, layers):
    window = _open(qapp, layers)
    try:
        strip = window._workflow
        for step, page in (("model", window._modeling_view),
                           ("predict", window._prediction_view),
                           ("data", window._eda_view), ("rois", window._eda_view)):
            strip.button(step).click()
            qapp.processEvents()
            assert window._workspace_tabs.currentWidget() is page, step
            assert strip.current() == step
        assert window._explore_tabs.currentWidget() is window._project_rois
        strip.button("data").click()
        assert window._explore_tabs.currentWidget() is window._data_file_page

        expected = {}
        for check in layers.checks():
            if check.is_open:
                expected[check.step] = expected.get(check.step, 0) + 1
        assert expected == {"values": 1}, "the typing error of ROI high3"
        assert strip.state("values") == ("open", 1)
        assert strip.state("data")[0] == "done" and strip.state("rois")[0] == "done"
        assert strip.button("values").text() == "Values", "no count in the name"
        assert strip.button("values").badge_text() == "1", "the open checks, in a badge"
        assert strip.button("data").badge_text() == ""
        assert window._checks_dock.windowTitle() == "Checks (1)"
    finally:
        window.close()


def test_the_checks_panel_shows_the_library_checks_and_runs_their_fixes(qapp, layers):
    window = _open(qapp, layers)
    try:
        panel = window._checks_panel
        assert [c.check_id for c in panel.checks()] == \
            [c.check_id for c in sorted(layers.checks(), key=lambda c: (
                {"blocking": 0, "attention": 1, "info": 2, "ok": 3}[c.severity]))]
        row = next(i for i, c in enumerate(panel.checks())
                   if c.check_id == "values.differ_within_sample")
        button = panel.fix_button(row)
        assert button is not None and button.text() == "Go to ROI"
        button.click()
        qapp.processEvents()
        assert window._workflow.current() == "values"
        assert window._roi_table_dock.isVisible()
        assert window._selected_roi_id() == layers.roi("high3").roi_id
    finally:
        window.close()


def test_the_fit_settings_dock_fits_without_a_dialog_and_marks_changes(qapp, layers,
                                                                       monkeypatch):
    from PyQt6.QtWidgets import QDialog

    from enginir.viewer._modeling import FitModelRunDialog

    monkeypatch.setattr(FitModelRunDialog, "exec", lambda self: pytest.fail("a dialog opened"))
    monkeypatch.setattr(QDialog, "exec", lambda self: pytest.fail("a dialog opened"))
    window = _open(qapp, layers)
    try:
        model_view = window._modeling_view
        assert model_view.fit_form is None, "built when the Model step is first shown (D8)"
        window.show_step("model")
        form = model_view.fit_form
        assert form is not None and form.embedded and form.isVisibleTo(window)
        assert form.parentWidget() is not None and not form.isWindow()

        run = _fit(qapp, window, n_components=2)
        assert run.hyperparameters["n_components"] == 2
        assert not model_view.fit_settings_changed()
        form._components_spin.setValue(1)
        assert model_view.fit_settings_changed()
        assert "Changed since the last fit" in model_view._fit_changed_label.text()

        model_view.select_model_run(run.model_run_id)
        model_view._btn_fit_again.click()  # "Load Model Settings"
        assert form._components_spin.value() == 2 and not model_view.fit_settings_changed()
        assert len(window.project_state.model_runs) == 1, "loading settings does not fit"
    finally:
        window.close()


def test_the_hold_out_fix_loads_the_model_with_samples_held_out(qapp, layers):
    window = _open(qapp, layers)
    try:
        run = _fit(qapp, window, group_by="roi")
        checks = window._checks_panel.checks()
        row = next(i for i, c in enumerate(checks)
                   if c.check_id == "model.validation.one_level_per_file")
        button = window._checks_panel.fix_button(row)
        assert button is not None and button.text() == "Hold Out by Sample"
        button.click()
        qapp.processEvents()
        form = window._modeling_view.fit_form
        assert form is not None and form._group_by_combo.currentData() == "sample"
        assert window._workflow.current() == "model"
        assert window._modeling_view._selected_model_run_id == run.model_run_id
        assert window._modeling_view.fit_settings_changed(), "a refit is one click away"
    finally:
        window.close()


def test_the_models_panel_shows_each_model_with_the_library_verdict(qapp, layers):
    from enginir._workflow import Model

    window = _open(qapp, layers)
    try:
        assert window._models_panel.model_ids() == []
        run = _fit(qapp, window)
        panel = window._models_panel
        assert panel.model_ids() == [run.model_run_id]
        verdict = Model(Project.open(layers.path), run).verdict()
        cell = panel.table.item(0, panel.COLUMNS.index("Verdict"))
        assert cell is not None and cell.toolTip() == verdict.summary
        panel.table.cellDoubleClicked.emit(0, 0)
        assert window._workspace_tabs.currentWidget() is window._modeling_view
    finally:
        window.close()


def test_the_window_is_menus_strip_three_columns_and_a_bottom_panel(qapp, layers):
    window = _open(qapp, layers)
    try:
        area = window.dockWidgetArea
        assert area(window._controls_dock) == Qt.DockWidgetArea.RightDockWidgetArea
        assert area(window._project_browser_dock) == Qt.DockWidgetArea.LeftDockWidgetArea
        bottom = [window._checks_dock, window._roi_table_dock, window._models_dock,
                  window._log_dock]
        assert all(area(dock) == Qt.DockWidgetArea.BottomDockWidgetArea for dock in bottom)
        assert set(window.tabifiedDockWidgets(window._checks_dock)) == set(bottom[1:])
        for dock in (*bottom, window._controls_dock, window._project_browser_dock):
            features = dock.features()
            assert not features & dock.DockWidgetFeature.DockWidgetFloatable, "fixed (Q6)"
            assert not features & dock.DockWidgetFeature.DockWidgetMovable
        assert not window._workspace_tabs.tabBar().isVisible(), "the strip navigates"
        assert window._project_browser_dock.windowTitle() == "Project"
        assert window._inspector_dock.windowTitle() == "Properties"
    finally:
        window.close()


def test_the_example_project_is_created_once_and_opens_with_a_model(qapp, tmp_path,
                                                                     monkeypatch):
    from enginir._example import create_example_project
    from enginir.viewer import _recent

    path = create_example_project(tmp_path)
    project = Project.open(path)
    assert len(project.workspace.datasets) == 5 and len(project.models) == 1
    jobs = [roi.job for roi in project.workspace.rois.values()]
    assert jobs.count("prediction") == 5 and jobs.count("calibration") == 20
    assert all(roi.sample for roi in project.workspace.rois.values())
    model = next(iter(project.models.values()))
    assert model.verdict().level in ("good", "usable_with_care")
    stamp = path.stat().st_mtime_ns
    assert create_example_project(tmp_path) == path and path.stat().st_mtime_ns == stamp, \
        "reused, not written again"
    (tmp_path / "tablets_to_predict.h5").rename(tmp_path / "tablets_05_layers.h5")
    assert create_example_project(tmp_path) == path and path.stat().st_mtime_ns != stamp, \
        "the example of 1.6.0rc3 (one layer count per scan) is created again"
    assert (tmp_path / "tablets_to_predict.h5").is_file()

    monkeypatch.setattr(_recent, "config_dir", lambda: tmp_path / "config")
    window = _open(qapp, project)
    try:
        window.open_example_project()
        wait_until(qapp, lambda: window.project_state.project_path is not None
                    and "example" in window.project_state.project_path)
        assert len(window.project_state.model_runs) == 1
    finally:
        window.close()


def test_no_bar_repeats_the_checks_and_the_fit_summary_gives_the_verdict(qapp, layers):
    """The Checks tab says what a step needs (no bar above the pages repeats its first
    line); the Fit summary has the library's verdict of the model shown."""
    from enginir._workflow import Model

    window = _open(qapp, layers)
    try:
        assert window.centralWidget() is window._workspace_content_stack, "no bar above"
        window.show_step("values")
        qapp.processEvents()
        checks = window._checks_panel.checks()
        row = next(i for i, c in enumerate(checks)
                   if c.check_id == "values.differ_within_sample")
        button = window._checks_panel.fix_button(row)
        assert button is not None and button.text() == "Go to ROI"
        button.click()
        assert window._selected_roi_id() == layers.roi("high3").roi_id

        window.show_step("predict")
        qapp.processEvents()
        assert window._predict_page_stack.currentWidget() is window._predict_empty, \
            "an invitation, not empty axes"

        run = _fit(qapp, window)
        window.show_step("model")
        window.select_model(run.model_run_id)
        qapp.processEvents()
        verdict = Model(Project.open(layers.path), run).verdict()
        words = {"good": "Good", "usable_with_care": "Usable with care",
                 "not_usable": "Not usable"}[verdict.level]
        assert window._model_verdict_value.text() == words
        assert window._model_verdict_value.toolTip() == verdict.summary
        assert window._model_verdict_value.isVisibleTo(window)
    finally:
        window.close()


def test_passed_checks_fold_into_one_row_that_unfolds(qapp, layers):
    window = _open(qapp, layers)
    try:
        panel = window._checks_panel
        checks = panel.checks()
        passed = [check for check in checks if check.severity == "ok"]
        others = [check for check in checks if check.severity != "ok"]
        assert len(passed) >= 2 and others, "the fixture has open and passed checks"
        assert panel.passed_folded()
        assert panel.row_checks() == [*others, None], "open checks first, then one row"
        item = panel.table.item(len(others), 1)
        assert item is not None and item.text() == f"{len(passed)} checks passed"
        panel.table.cellDoubleClicked.emit(len(others), 1)
        assert panel.row_checks() == [*others, None, *passed]
        window.refresh_checks()
        assert not panel.passed_folded(), "a refresh keeps them unfolded"
    finally:
        window.close()


def test_the_fold_warning_of_a_model_is_said_once_on_the_model_page(qapp, layers):
    """In the Checks tab; not again above the page, in the fit summary or in Properties."""
    from PyQt6.QtWidgets import QLabel

    window = _open(qapp, layers)
    try:
        _fit(qapp, window, group_by="roi")
        window.show_step("model")
        qapp.processEvents()
        window._refresh_workspace_content_pages()
        form = window._modeling_view.fit_form
        said = [label for label in window.findChildren(QLabel)
                if "likely optimistic" in label.text() and label.isVisibleTo(window)]
        assert said == [], "the Checks tab says it"
        form._components_spin.setValue(2)  # other settings: the form warns before the fit
        assert "likely optimistic" in form._fold_warning_label.text()
        assert form._fold_warning_label.isVisibleTo(window)
        assert any("likely optimistic" in check.message
                   for check in window._checks_panel.checks())
        assert "RMSECV" not in window._inspector.body_label.text(), \
            "the metrics are on the page, not again in Properties"
    finally:
        window.close()


def test_the_fit_form_shows_the_selected_model_until_it_is_edited(qapp, layers):
    window = _open(qapp, layers)
    try:
        view = window._modeling_view
        one = _fit(qapp, window, n_components=1)
        form = view.fit_form
        form._snv.setChecked(True)
        two = _fit(qapp, window, n_components=2)
        assert form._components_spin.value() == 2 and form._snv.isChecked()

        view.select_model_run(one.model_run_id)
        assert (form._components_spin.value(), form._snv.isChecked()) == (1, False)
        assert not view.fit_settings_changed()
        assert view._fit_changed_label.text() == "Settings of 'PLS 1 LV, raw'"

        form._components_spin.setValue(3)  # an edit stays when another model is selected
        view.select_model_run(two.model_run_id)
        assert form._components_spin.value() == 3 and view.fit_settings_changed()
        assert view._fit_changed_label.text().startswith("● Changed since the last fit")
        view._btn_fit_again.click()  # Load Model Settings discards the edit
        assert (form._components_spin.value(), form._snv.isChecked()) == (2, True)
    finally:
        window.close()


def test_a_fit_uses_the_forms_preprocessing_when_its_bands_differ_from_explore(qapp, layers):
    """Showing the fit's bands in Explore changes the preview; that must not put the
    preview's other preprocessing into the form or the fit."""
    window = _open(qapp, layers)
    try:
        window.show_step("model")
        form = window._modeling_view.fit_form
        form._snv.setChecked(True)  # the Explore preview has no SNV
        form._keep_edit.setText("1100-1500")
        run = _fit(qapp, window)
        state = window.project_state
        recipe = state.recipes[state.sample_sets[run.sample_set_id].preprocessing_recipe_id]
        assert recipe["snv"] is True and recipe["bands_keep"]
        assert form._snv.isChecked() and not window._modeling_view.fit_settings_changed()
        assert window.state.bands_keep, "Explore shows the bands of the fit"
    finally:
        window.close()
    reopened = _open(qapp, Project.open(layers.path))  # an opened project shows its model
    try:
        reopened.show_step("model")
        form = reopened._modeling_view.fit_form
        assert form._snv.isChecked() and form._keep_edit.text() == "1100-1500"
        assert not reopened._modeling_view.fit_settings_changed()
        assert not reopened.state.preprocess_snv, "the Explore preview stays as it was"
    finally:
        reopened.close()


def test_the_fit_form_folds_and_keeps_fit_model_in_view(qapp, layers):
    from PyQt6.QtWidgets import QScrollArea

    window = _open(qapp, layers)
    try:
        window.resize(1536, 864)
        view = window._modeling_view
        window.show_step("model")
        form = view.fit_form
        qapp.processEvents()
        assert [form.section(key).expanded for key in
                ("data", "preprocessing", "bands", "model", "validation", "fitting")] \
            == [True, True, False, True, True, False]
        assert form.section("bands").summary == "all bands"
        form._keep_edit.setText("1100-1500")
        form._exclude_check.setChecked(True)
        assert form.section("bands").summary == "keep 1100-1500"
        assert "DModX exclusion at 0.95" in form.section("fitting").summary
        assert not form._keep_edit.isVisibleTo(window)
        form.section("bands").header.click()
        assert form._keep_edit.isVisibleTo(window)

        scroll = window._controls_dock.findChild(QScrollArea)
        button = view._btn_fit_model
        assert button.isVisibleTo(window) and not scroll.isAncestorOf(button), \
            "under the scrolling settings, never scrolled away"
        for hidden in (view._btn_target, view.target_status_label, view._sample_set_status):
            assert not hidden.isVisibleTo(window), "the form's Target row says it"
        window.show_step("rois")
        qapp.processEvents()
        assert not button.isVisibleTo(window)
        window.show_step("model")  # Qt centres a shown dialog; the embedded form stays put
        qapp.processEvents()
        assert view.rect().contains(form.geometry()), form.geometry()
    finally:
        window.close()


def test_a_reopened_project_shows_its_latest_model_whatever_its_preprocessing(qapp, layers):
    """Two models of one target with different preprocessing have two calibration sets;
    the Model step shows the latest (no model is selected or active yet), not none.
    A model of another target is still not shown for the chosen target."""
    window = _open(qapp, layers)
    try:
        _fit(qapp, window, n_components=1)
        window._modeling_view.fit_form._snv.setChecked(True)
        latest = _fit(qapp, window, n_components=2)
    finally:
        window.close()
    project = Project.open(layers.path)
    other = project.add_target("Other", unit="%")
    reopened = _open(qapp, project)
    try:
        view = reopened._modeling_view
        assert reopened.project_state.active_model_name is None and \
            view._selected_model_run_id is None
        shown = view._resolved_workspace_model_run()
        assert shown is not None and shown.model_run_id == latest.model_run_id
        reopened.show_step("model")
        qapp.processEvents()
        assert reopened._model_verdict_level in ("good", "usable_with_care", "not_usable"), \
            "the verdict of the model shown"
        assert view.fit_form._snv.isChecked() and view.fit_form._components_spin.value() == 2

        view.selected_target_id = other.ref.target_id if hasattr(other, "ref") else other.target_id
        assert view._resolved_workspace_model_run() is None, "no model of this target"
    finally:
        reopened.close()


# -- Reduced shell smoke tests for the HSIViewer main window ---------------------------


@pytest.fixture()
def cube_and_wavelengths():
    rng = np.random.default_rng(42)
    cube = rng.integers(128, 3500, size=(4, 8, 5), dtype=np.uint16)
    wavelengths = np.linspace(1000.0, 1700.0, 5)
    return cube, wavelengths


def _create_reopenable_model_project(path: str, wavelengths: np.ndarray) -> None:
    from enginir.io.project_store import (
        create_project,
        save_reference_correction,
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
    save_reference_correction(
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
            "required_correction_mode": "raw_dn",
            "requires_dark_ref": False,
            "requires_white_ref": False,
            "corrected_input_allowed": False,
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
                "reference_correction_id": "cal_1",
                "name": roi_id,
                "wavelengths": wavelengths.astype(np.float32),
                "source_file": "/data/train.eiff",
                "job": "calibration",
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
            "fit_spec": FitSpec(groups=(0, 1)).to_dict(),
            "software": software_versions(),
            "hyperparameters": {"n_components": 2},
            "cv_spec": {"cv": "logo"},
            "metrics": {"rmsecv": 0.55, "r2": 0.9, "bias": 0.05},
            "artifact_refs": {
                "prediction_model": {
                    "n_components": 1,
                    "model_type": "PLSRegression",
                    "wavelengths": wavelengths.astype(np.float32),
                    "target_names": ["moisture"],
                    "coef_": np.asarray(model.coef_, dtype=np.float32),
                    "intercept_": np.asarray(model.intercept_, dtype=np.float32),
                    "x_mean_": np.asarray(model.x_mean_, dtype=np.float32),
                    "x_std_": np.asarray(model.x_std_, dtype=np.float32),
                    "y_mean_": np.asarray(model.y_mean_, dtype=np.float32),
                    "y_std_": np.asarray(model.y_std_, dtype=np.float32),
                    **projection_payload(wavelengths.size),
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
            "reference_correction_id": "cal_1",
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


def test_the_shell_builds_its_workspaces_and_docks_of_fixed_width(qapp, cube_and_wavelengths):
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
    assert viewer._model_workspace_splitter.widget(0) is viewer._model_diagnostics_widget
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
    # The Settings dock: the outer right column, a fixed width per screen-size class.
    assert viewer._controls_dock.minimumWidth() == viewer._controls_dock.maximumWidth()
    assert viewer._controls_dock.maximumWidth() in (340, 304)
    assert viewer.dockWidgetArea(viewer._controls_dock) == Qt.DockWidgetArea.RightDockWidgetArea
    assert viewer._spectrum_panel.minimumHeight() == viewer._spectrum_panel.maximumHeight()
    assert viewer._spectrum_panel.maximumHeight() == 220
    assert viewer._spectrum_toggle_action.isCheckable()
    assert [viewer._workspace_tabs.tabText(i) for i in range(viewer._workspace_tabs.count())] == [
        "Explore",
        "Model",
        "Predict",
    ]

    # The Settings dock keeps its width and takes the step's title.
    expected_width = viewer._controls_dock.maximumWidth()
    titles = {viewer._eda_view: "View settings", viewer._modeling_view: "Fit settings",
              viewer._prediction_view: "Prediction settings"}

    for widget, spectrum_visible in (
        (viewer._eda_view, True),
        (viewer._modeling_view, False),
        (viewer._prediction_view, False),
        (viewer._eda_view, True),
    ):
        viewer._workspace_tabs.setCurrentWidget(widget)
        qapp.processEvents()
        assert viewer._controls_dock.minimumWidth() == expected_width
        assert viewer._controls_dock.maximumWidth() == expected_width
        assert viewer._controls_dock.windowTitle() == titles[widget]
        assert viewer._spectrum_panel.isHidden() is not spectrum_visible
        assert viewer._spectrum_toggle_action.isEnabled() is spectrum_visible


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


def test_add_data_file_propagates_unexpected_errors(qapp, cube_and_wavelengths, monkeypatch,
                                                    tmp_path):
    """Expected file errors become a dialog; programming errors are not hidden in one."""
    from PyQt6.QtWidgets import QMessageBox

    from enginir import Project
    from enginir.viewer._viewer import HSIViewer

    cube, wavelengths = cube_and_wavelengths
    viewer = HSIViewer(cube, wavelengths)
    viewer.project_state.open_project(str(Project.create(tmp_path / "p.h5").path))
    criticals: list[str] = []
    monkeypatch.setattr(
        "enginir.viewer._viewer.read_header",
        lambda path: (_ for _ in ()).throw(AttributeError("unexpected bug")),
    )
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *args, **kwargs: criticals.append(str(args[2]))
    )

    with pytest.raises(AttributeError, match="unexpected bug"):
        viewer.add_data_file("/tmp/broken.eiff")

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
    # the stored prediction run of the file is not shown: predictions live in the session
    assert not viewer._predict_headline.text().startswith("Map of")


def test_project_browser_selection_syncs_inspector_and_canvas(qapp, cube_and_wavelengths, tmp_path):
    from enginir.viewer._viewer import HSIViewer

    cube, wavelengths = cube_and_wavelengths
    project_path = tmp_path / "browser_roi_selection.h5"
    _create_reopenable_model_project(str(project_path), wavelengths[:4])

    viewer = HSIViewer(cube, wavelengths)
    viewer._cube_path = "/data/train.eiff"  # the window shows the data file of the ROIs
    viewer.project_state.open_project(str(project_path))
    qapp.processEvents()

    assert isinstance(viewer._project_browser, QTreeWidget)
    sections = {
        viewer._project_browser.topLevelItem(i).text(0).split(" (")[0]:
            viewer._project_browser.topLevelItem(i)
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
    from PyQt6.QtWidgets import QMessageBox

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

    from tests._project_payloads import write_model_project

    cube, wavelengths = cube_and_wavelengths
    project_path = str(tmp_path / "navigation.h5")
    write_model_project(project_path)
    viewer = HSIViewer(cube, wavelengths)
    viewer.show()  # plots draw only while visible
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
                    "datasets", "reference_corrections", "rois", "targets", "sample_sets",
                    "model_runs", "predictions")})

    tabs = viewer._workspace_tabs
    tabs.setCurrentWidget(viewer._modeling_view)
    qapp.processEvents()
    before = context()
    from enginir.viewer._plot_panel import PlotPanel

    renders: list[object] = []
    original = PlotPanel._render

    def counting_render(panel):
        renders.append(panel.spec.plot_id)
        original(panel)

    monkeypatch.setattr(PlotPanel, "_render", counting_render)

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
    assert shown > renders_while_hidden, "the shown Model diagnostics draw the new payload"
    viewer.schedule_workspace_content_refresh()
    qapp.processEvents()
    assert len(renders) == shown, "an unchanged diagnostics payload is a no-op redraw"


# -- Project Browser: each model holds its calibration set and input contract (terminology contract) ----


def test_a_model_shows_its_calibration_set_and_input_contract_beneath_it(qapp, tray):
    from PyQt6.QtCore import Qt

    from enginir.viewer._browser import ProjectBrowser
    from enginir.viewer._state import ProjectState

    project, _scan, _path = tray
    model = project.fit("API", n_components=2, cv=3)
    state = ProjectState()
    state.open_project(str(project.path))
    browser = ProjectBrowser(state)
    browser.refresh()

    shown = [browser.topLevelItem(i).text(0) for i in range(browser.topLevelItemCount())]
    assert shown[0] == "Datasets (1)" and shown[-2:] == ["Models (1)", "Predictions"], \
        "a section counts its objects; an empty one has no count"
    sections = [text.split(" (")[0] for text in shown]
    assert sections == ["Datasets", "Reference corrections", "ROIs", "Targets", "Target values",
                        "Models", "Predictions"]
    assert not browser.topLevelItem(0).icon(0).isNull()
    models = browser.topLevelItem(sections.index("Models"))
    assert models.childCount() == 1, "nothing left over: the set and the contract are used"
    item = models.child(0)
    assert item.text(0).startswith(model.name)
    kinds = {item.child(i).data(0, Qt.ItemDataRole.UserRole)["kind"]: item.child(i).text(0)
             for i in range(item.childCount())}
    assert set(kinds) == {"sample_set", "signal_domain"}
    assert kinds["sample_set"].startswith("Calibration set: 7 ROIs, API")
    assert kinds["signal_domain"].startswith("Input contract: ")

    browser.select_payload({"kind": "signal_domain", "id": model.ref.signal_domain_id})
    assert browser.currentItem() is not None
    assert browser.currentItem().parent() is item, "selection finds the nested item"


def test_leaves_show_names_in_natural_order_and_keep_the_id_in_the_tooltip(qapp, tmp_path):
    from enginir import Project
    from enginir.viewer._browser import ProjectBrowser
    from enginir.viewer._state import ProjectState

    project = Project.create(tmp_path / "p.h5")
    project.add_target("API", unit="%")
    from tests._scans import framed_scan

    scan = project.add_dataset(framed_scan(tmp_path / "s.h5"), dark_frames=(0, 1),
                               white_frames=(18, 19))
    for name in ("r10", "r2", "r1"):
        scan.add_roi(name, frames=(3, 5), rows=(0, 5), values={"API": 1.5})
    state = ProjectState()
    state.open_project(str(project.path))
    browser = ProjectBrowser(state)
    browser.refresh()

    def leaves(section: str) -> list[str]:
        top = next(browser.topLevelItem(i) for i in range(browser.topLevelItemCount())
                   if browser.topLevelItem(i).text(0).startswith(section))
        return [top.child(i).text(0) for i in range(top.childCount())]

    assert leaves("ROIs") == ["r1", "r2", "r10"], "names in natural order, no ids"
    assert leaves("Targets") == ["API (%)"]
    assert leaves("Target values") == ["r1: API = 1.5", "r2: API = 1.5", "r10: API = 1.5"]
    assert leaves("Reference corrections") == ["s.h5: dark/white references"]
    rois = next(browser.topLevelItem(i) for i in range(browser.topLevelItemCount())
                if browser.topLevelItem(i).text(0).startswith("ROIs"))
    roi_ids = {roi.name: roi_id for roi_id, roi in state.rois.items()}
    assert rois.child(0).toolTip(0).endswith(f"id {roi_ids['r1']}"), "the id stays at hand"
    every = [text for section in ("Datasets", "ROIs", "Targets", "Models") for text in leaves(section)]
    assert not any(object_id in text for text in every for object_id in state.rois), every


# -- Strip, status line, ROI table and project tree on the example project -------------


def _add_roi_without_value(project) -> None:
    dataset = next(iter(project.workspace.datasets))
    project.dataset(dataset_id=dataset).add_roi("new", frames=(8, 10), rows=(8, 20))


def test_the_strip_shows_order_badges_and_names_project_and_model(qapp, edited_window):
    window = edited_window(_add_roi_without_value)
    from PyQt6.QtWidgets import QLabel

    from enginir.viewer._workflow_strip import STEPS

    strip = window._workflow
    assert [step.key for step in STEPS] == ["data", "rois", "values", "model", "predict"], \
        "no Report step before there is a report"
    arrows = [label for label in strip.findChildren(QLabel)
              if label.objectName() == "WorkflowArrow"]
    assert len(arrows) == len(STEPS) - 1, "an arrow between the steps: their order"
    assert [strip.button(step.key).text() for step in STEPS] \
        == ["Data", "ROIs", "Values", "Model", "Predict"], "no numbers, no counts"
    assert all(strip.state(step.key)[0] != "open" for step in STEPS
               if step.key not in ("values", "predict")), \
        "the example's only open check is in Predict (a tablet outside the model)"
    state, count = strip.state("values")
    button = strip.button("values")
    assert state == "open" and count and button.badge_text() == str(count)
    assert "open check" in button.toolTip(), "what the number is"
    badge = button.findChild(QLabel)
    assert badge.isVisibleTo(window) and button.rect().contains(badge.geometry())
    assert strip.context() == "engiNIR example.h5   ·   Model: PLS 2 LV"


def test_the_roi_table_gets_more_height_while_it_is_in_front(qapp, window):
    window.show_step("rois")
    wait_until(qapp, lambda: not window._roi_table_in_front()
                and window._checks_dock.height() < 0.3 * window.height())
    low = window._checks_dock.height()
    window.show_step("values")  # the Values step brings the ROI table to the front
    wait_until(qapp, lambda: window._roi_table_in_front()
                and window._roi_table_dock.height() >= 0.38 * window.height())
    view = window._roi_table.view
    rows = view.viewport().height() // view.rowHeight(0)
    assert rows >= 10, f"only {rows} ROIs in view"
    # The next step, by its workspace tab as by the strip, gets its page's height back:
    # Model with the Models in front, Predict with the Checks.
    window._workspace_tabs.setCurrentWidget(window._modeling_view)
    wait_until(qapp, lambda: not window._roi_table_in_front()
                and not window._models_panel.visibleRegion().isEmpty()
                and window._models_dock.height() <= low + 2)
    for leave in (lambda: window.show_step("model"),
                  lambda: window._workspace_tabs.setCurrentWidget(window._prediction_view)):
        window.show_step("values")
        wait_until(qapp, window._roi_table_in_front)
        leave()
        wait_until(qapp, lambda: not window._roi_table_in_front()
                    and window._models_dock.height() <= low + 2)
    assert not window._checks_panel.visibleRegion().isEmpty(), "Predict: the Checks"


def test_the_status_strip_follows_the_step(qapp, window):
    window.show_step("data")
    window.show_dataset(sorted(window.project_state.datasets)[0])
    window.wait_for_data_file()
    qapp.processEvents()
    assert "Frame: " in window._status_label.text() and "Band: " in window._status_label.text()
    window.show_step("rois")
    qapp.processEvents()
    text = window._status_label.text()
    assert "Dataset: " in text and "Frame" not in text, "no cursor on the Project ROIs page"
    window.show_step("model")
    qapp.processEvents()
    assert window._status_label.text() == ("Project: engiNIR example.h5 | Model: PLS 2 LV"
                                           " | Target: Layers | Calibration set: 20 ROIs")


def test_the_project_tree_counts_and_properties_are_a_table_with_plain_text(qapp, window):
    browser = window._project_browser
    texts = [browser.topLevelItem(i).text(0) for i in range(browser.topLevelItemCount())]
    assert texts[:3] == ["Datasets (5)", "Reference corrections (5)", "ROIs (25)"]
    assert texts[-1] == "Predictions", "nothing to count"
    body = window._inspector.body_label
    body.setText("Target: Layers\nA sentence that runs on for a while before: its colon.\nNote")
    assert body.text().startswith("Target: Layers\n"), "tests and copies read the lines"
    from PyQt6.QtWidgets import QLabel

    rendered = QLabel.text(body)  # what the label displays
    assert "<table" in rendered and rendered.count("<tr>") == 3
    assert rendered.count('colspan="2"') == 2, "only 'Target' is a key"


def test_the_example_has_one_open_check_a_tablet_outside_the_model_and_a_good_model(example):
    """The example shows good practice: values vary within each scan, one sample per tablet.

    Its one open check is in Predict: the prediction tablet P_T4 lies outside the model.
    """
    from enginir import Project

    project = Project.open(example)
    open_checks = [check for check in project.checks() if check.is_open]
    assert [check.check_id for check in open_checks] == ["predict.rois_outside"]
    assert open_checks[0].roi_ids == (project.roi("P_T4").roi_id,)
    assert not [check for check in project.checks(predict=False) if check.is_open]
    assert project.model("PLS 2 LV").verdict().level == "good"


# -- Opening a dataset from the project tree -------------------------------------------


def test_opening_a_dataset_from_the_project_tree_shows_its_image(qapp, tmp_path):
    from enginir.viewer import open_project

    project = Project.create(tmp_path / "p.h5")
    scan = project.add_dataset(framed_scan(tmp_path / "s.h5"))
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


def test_the_prediction_check_arrives_after_the_other_checks(qapp, window):
    """The panel shows the checks at once; the prediction-ROI check comes from a job."""
    def ids():
        return {check.check_id for check in window._checks}

    window.refresh_checks()
    assert "predict.rois_ready" in ids() and "predict.rois_outside" not in ids(), \
        "the fast checks first, without predicting"
    wait_until(qapp, lambda: "predict.rois_outside" in ids(), timeout=30)
    (outside,) = [check for check in window._checks if check.check_id == "predict.rois_outside"]
    assert window._workflow.state("predict")[0] == "open"
    assert window._checks_panel.open_count() >= 1

    # A result of an earlier refresh is dropped.
    from enginir._guidance import Check

    fast = list(window._checks)
    window.refresh_checks()
    stale = window._checks_generation - 1
    window._apply_prediction_checks(stale, window.project_state.project, fast,
                                    [Check("predict.stale", "predict", "attention", "stale")])
    assert "predict.stale" not in ids()
    wait_until(qapp, lambda: "predict.rois_outside" in ids(), timeout=30)
    assert outside.roi_ids == next(c for c in window._checks
                                   if c.check_id == "predict.rois_outside").roi_ids


def test_a_refresh_does_not_bring_a_known_check_to_the_front_again(qapp, window):
    """The prediction check arrives after each refresh; only a new open check raises the tab."""
    from enginir.viewer._jobs import _RUNNING_JOBS

    assert "predict.rois_outside" in {check.check_id for check in window._checks}
    window.show_step("values")  # the ROI table in front of the Checks
    wait_until(qapp, lambda: window._roi_table_in_front())
    window.refresh_checks()
    wait_until(qapp, lambda: not _RUNNING_JOBS)
    qapp.processEvents()
    assert "predict.rois_outside" in {check.check_id for check in window._checks}
    assert window._roi_table_in_front(), "the known prediction check raised the Checks tab"


# -- The Model and Predict pages on first use (plan startup-speed-1.7.1, D8) -----------


def _example_window(qapp, tmp_path):
    from enginir._example import create_example_project
    from enginir.viewer import open_project

    window = open_project(str(create_example_project(tmp_path)))
    window.resize(1600, 900)
    window.show()
    qapp.processEvents()
    return window


def test_the_fit_settings_and_the_predict_page_are_built_when_first_shown(qapp, tmp_path):
    window = _example_window(qapp, tmp_path)
    try:
        view = window._modeling_view
        assert view.fit_form is None and window._prediction_page is None, \
            "a window that stays in Explore builds neither"
        assert window._modeling_view.current_target() is not None, \
            "the current target is there without the Model page (Explore and the ROI table)"
        window.show_step("model")
        form = view.fit_form
        assert form is not None and form.isVisibleTo(window)
        assert window._fit_footer is not None and window._fit_footer.isVisibleTo(window)
        window.show_step("predict")
        page = window._prediction_page
        assert page is not None and window._workspace_tabs.currentWidget() is page
        assert window._controls_dock.windowTitle() == "Prediction settings"
        window.show_step("data")
        window.show_step("model")
        window.show_step("predict")
        assert view.fit_form is form and window._prediction_view is page, "built once"
        assert not window._fit_footer.isVisibleTo(window), "Fit Model shows with Model only"
    finally:
        window.close()


def test_menus_and_fixes_reach_pages_that_were_never_shown(qapp, tmp_path, monkeypatch):
    """Each way into the Model or Predict page builds it first: no fit dialog opens in
    place of the Fit settings, and Predict ROIs opens its own dialog."""
    from PyQt6.QtWidgets import QDialog

    from enginir._guidance import Fix

    opened: list[str] = []
    monkeypatch.setattr(QDialog, "exec", lambda self: opened.append(type(self).__name__) or 0)
    window = _example_window(qapp, tmp_path)
    try:
        load = next(action for action in window._menu_model.actions()
                    if action.text() == "Load Model Settings")
        window._menu_model.aboutToShow.emit()
        load.trigger()
        form = window._modeling_view.fit_form
        assert form is not None and opened == [], opened
        assert not window._modeling_view.fit_settings_changed(), "the model's settings"
        predict = next(action for action in window._menu_predict.actions()
                       if action.text() == "Predict ROIs...")
        window._menu_predict.aboutToShow.emit()
        assert window._prediction_page is not None, "the Predict menu builds its page"
        predict.trigger()
        assert opened == ["PredictRoisDialog"], opened
    finally:
        window.close()
    window = _example_window(qapp, tmp_path)
    try:
        target_id = next(iter(window.project_state.targets))
        window.run_fix(Fix("fit", "Fit", {"target_id": target_id}))
        form = window._modeling_view.fit_form
        assert form is not None and form._target_combo.currentData() == target_id
        window.run_fix(Fix("predict_rois", "Predict"))
        assert window._prediction_page is not None
    finally:
        window.close()
