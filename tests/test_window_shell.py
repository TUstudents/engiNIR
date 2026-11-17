"""The main window of engiNIR 1.6 (plan gui-1.6, Phases 3–5).

Menus hold every action, the workflow strip navigates and shows the state of
each step, the bottom panel lists the library's checks with their fixes, the
Settings dock holds the fit settings (no fit dialog), and the Models panel
shows each model's verdict. The viewer judges nothing itself: everything
shown comes from ``Project.checks()`` and ``Model.verdict()``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from enginir import Project
from enginir.io import save_hdf5
from enginir.viewer._app import ensure_app

N_BANDS, HEIGHT, FRAMES = 12, 4, 3


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


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


def _wait_until(qapp, predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def _open(qapp, project):
    from enginir.viewer import open_project

    window = open_project(project.path)
    window.resize(1920, 1080)
    window.show()
    # The Project ROIs tiles load in the background and read the project file; a write
    # (the fit) waits until they are in, as a user would.
    view = window._project_rois
    _wait_until(qapp, lambda: not view.busy() and not view.mosaic.missing_patches()
                and not view._rebuild_timer.isActive())
    window.refresh_checks()
    return window


def _fit(qapp, window, n_components=1, **form):
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
    _wait_until(qapp, lambda: set(window.project_state.model_runs) - before
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
        buttons = {button for _action, button in window._button_actions}
        assert window._modeling_view._btn_fit_model in buttons
        assert window._prediction_view._btn_predict_rois in buttons
        assert window._eda_view._btn_roi_grid in buttons
        for action, button in window._button_actions:
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
        form = model_view.fit_form
        window.show_step("model")
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
        _wait_until(qapp, lambda: window.project_state.project_path is not None
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


# -- polish after the design review (repetition, Fit settings form) -------------------------


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
        form = view.fit_form
        one = _fit(qapp, window, n_components=1)
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
        form = view.fit_form
        window.show_step("model")
        qapp.processEvents()
        assert [form.section(key).expanded for key in
                ("data", "preprocessing", "bands", "model", "validation", "fitting")] \
            == [True, True, False, True, True, False]
        assert form.section("bands").summary == "all bands"
        form._keep_edit.setText("1100-1500")
        form._exclude_check.setChecked(True)
        assert form.section("bands").summary == "keep 1100-1500"
        assert "DModX exclusion at 0.99" in form.section("fitting").summary
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
