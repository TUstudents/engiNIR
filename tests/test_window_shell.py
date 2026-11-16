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
        assert "1" in strip.button("values").text()
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

    monkeypatch.setattr(_recent, "config_dir", lambda: tmp_path / "config")
    window = _open(qapp, project)
    try:
        window.open_example_project()
        _wait_until(qapp, lambda: window.project_state.project_path is not None
                    and "example" in window.project_state.project_path)
        assert len(window.project_state.model_runs) == 1
    finally:
        window.close()


def test_the_guide_bar_shows_the_verdict_or_the_steps_next_check(qapp, layers):
    from enginir._workflow import Model

    window = _open(qapp, layers)
    try:
        bar = window._guide_bar
        window.show_step("values")
        qapp.processEvents()
        (check,) = [c for c in layers.checks() if c.check_id == "values.differ_within_sample"]
        assert bar.isVisible() and bar.text.text() == check.message and bar.level == "open"
        assert bar.fix.isVisible() and bar.fix.text() == "Go to ROI"
        bar.fix.click()
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
        assert bar.text.text() == verdict.summary and bar.level == verdict.level
    finally:
        window.close()
