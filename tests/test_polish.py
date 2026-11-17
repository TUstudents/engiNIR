"""Polish of the 1.6 window after the design review: plots and details (groups 3 and 4)."""

from __future__ import annotations

import numpy as np
import pytest

from enginir._example import create_example_project
from enginir._guidance import figure_text
from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    return create_example_project(tmp_path_factory.mktemp("example"))


@pytest.fixture()
def window(qapp, example):
    from enginir.viewer import open_project

    window = open_project(example)
    window.resize(1536, 864)
    window.show()
    qapp.processEvents()
    window.refresh_checks()
    yield window
    window.close()


@pytest.fixture()
def edited_window(qapp, tmp_path):
    """A window on a copy of the example after ``edit(project)``; the example has no open check."""
    from enginir import Project
    from enginir.viewer import open_project

    windows = []

    def open_edited(edit):
        project = Project.open(create_example_project(tmp_path))
        edit(project)
        window = open_project(project.path)
        window.resize(1536, 864)
        window.show()
        qapp.processEvents()
        window.refresh_checks()
        windows.append(window)
        return window

    yield open_edited
    for window in windows:
        window.close()


def _add_roi_without_value(project) -> None:
    dataset = next(iter(project.workspace.datasets))
    project.dataset(dataset_id=dataset).add_roi("new", frames=(8, 10), rows=(8, 20))


def _wait_until(qapp, predicate, timeout=20.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def _roi(roi_id, dataset_id, frames, rows):
    from tests.test_project_rois import _roi as make

    return make(roi_id, dataset_id, frames, rows)


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
    _wait_until(qapp, lambda: mosaic._label_items)
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
    _wait_until(qapp, lambda: mosaic._label_items)
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
    _wait_until(qapp, lambda: mosaic._label_items)
    full = dict(mosaic._labels)
    mosaic.set_rois(list(mosaic._rois.values()), mosaic._file_names,
                    {roi_id: text[:2] for roi_id, text in full.items()}, mosaic._values)
    mosaic.set_labels(full)
    _skip_without_real_fonts(mosaic)
    shown = {roi_id: item.text() for roi_id, item in mosaic._label_items.items()}
    assert shown == {roi_id: full[roi_id] for roi_id in shown}, \
        f"{_mosaic_numbers(mosaic)} {shown}"


def test_a_dimensionless_unit_is_not_printed(qapp, window):
    from enginir.viewer._ui_helpers import with_unit

    assert (with_unit("API", "%"), with_unit("Layers", "1"), with_unit("Layers", "")) \
        == ("API (%)", "Layers", "Layers")
    form = window._modeling_view.fit_form
    assert form._target_combo.currentText() == "Layers"  # the example's unit is "1"
    assert window._modeling_view.workspace_summary()["target"] == "Layers"


def test_an_empty_plot_shows_its_message_without_axes():
    from matplotlib.figure import Figure

    from enginir.viewer._plot_panel import draw_placeholder
    from enginir.viewer._theme import panel_colors

    ax = Figure().add_subplot(111)
    draw_placeholder(ax, "Select an ROI to see the spread of its pixels.", panel_colors())
    assert not ax.axison and [text.get_text() for text in ax.texts] \
        == ["Select an ROI to see the spread of its pixels."]


def test_roi_names_on_the_image_have_a_plate_and_plot_headers_have_icons(qapp, window):
    from PyQt6.QtWidgets import QToolButton

    window.show_step("data")
    dataset_id = sorted(window.project_state.datasets)[0]
    window.show_dataset(dataset_id)
    window.wait_for_data_file()
    qapp.processEvents()
    entries = list(window._canvas._roi_item_map.values())
    assert entries and all(entry.text.fill.color().alpha() > 150 for entry in entries), \
        "a name is readable on a bright image"
    buttons = [button for button in window._canvas_frame._header.findChildren(QToolButton)
               if button.property("icon_name")]  # not the frame's own "Reset View"
    assert len(buttons) >= 2, "maximize, export"
    for button in buttons:
        assert not button.icon().isNull() and button.text() == "", button.property("icon_name")
    frame = window._canvas_frame
    frame.set_maximized(True)
    assert frame._max_button.property("icon_name") == "arrows-minimize"
    frame.set_maximized(False)


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
    assert all(strip.state(step.key)[0] != "open" for step in STEPS if step.key != "values"), \
        "the example itself has no open check"
    state, count = strip.state("values")
    button = strip.button("values")
    assert state == "open" and count and button.badge_text() == str(count)
    assert "open check" in button.toolTip(), "what the number is"
    badge = button.findChild(QLabel)
    assert badge.isVisibleTo(window) and button.rect().contains(badge.geometry())
    assert strip.context() == "engiNIR example.h5   ·   Model: PLS 2 LV"


def test_the_roi_table_gets_more_height_while_it_is_in_front(qapp, window):
    window.show_step("rois")
    _wait_until(qapp, lambda: not window._roi_table_in_front()
                and window._checks_dock.height() < 0.3 * window.height())
    low = window._checks_dock.height()
    window.show_step("values")  # the Values step brings the ROI table to the front
    _wait_until(qapp, lambda: window._roi_table_in_front()
                and window._roi_table_dock.height() >= 0.38 * window.height())
    view = window._roi_table.view
    rows = view.viewport().height() // view.rowHeight(0)
    assert rows >= 10, f"only {rows} ROIs in view"
    # The next step, by its workspace tab as by the strip, gets its page's height back:
    # Model with the Models in front, Predict with the Checks.
    window._workspace_tabs.setCurrentWidget(window._modeling_view)
    _wait_until(qapp, lambda: not window._roi_table_in_front()
                and not window._models_panel.visibleRegion().isEmpty()
                and window._models_dock.height() <= low + 2)
    for leave in (lambda: window.show_step("model"),
                  lambda: window._workspace_tabs.setCurrentWidget(window._prediction_view)):
        window.show_step("values")
        _wait_until(qapp, window._roi_table_in_front)
        leave()
        _wait_until(qapp, lambda: not window._roi_table_in_front()
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


def test_one_number_format_for_the_figures_of_merit(qapp, window):
    from enginir import Project
    from enginir._workflow import Model

    assert (figure_text(1.10724), figure_text(0.89104), figure_text(12345.6)) \
        == ("1.107", "0.891", "1.235e+04")
    project = Project.open(window.project_state.project_path)
    run = next(iter(window.project_state.model_runs.values()))
    verdict = Model(project, run).verdict()
    metrics = run.metrics
    for key in ("rmsecv", "r2", "rpd"):
        assert figure_text(metrics[key]) in verdict.summary, key
    summary = window._modeling_view.workspace_summary()
    assert summary["rmsecv"] == figure_text(metrics["rmsecv"])
    table = window._models_panel.table
    shown = [table.item(0, column).data(0) for column in (3, 4, 5)]
    assert shown == [float(figure_text(metrics[key])) for key in ("rmsecv", "r2", "rpd")]
    assert np.isfinite(shown).all()


def test_icons_and_properties_take_the_colours_of_a_new_theme(qapp, window):
    from PyQt6.QtWidgets import QLabel

    from enginir.viewer._theme import TOKENS, is_dark

    was_dark = is_dark()
    body = window._inspector.body_label
    body.setText("Target: Layers")
    section = window._project_browser.topLevelItem(0)

    def tree_icon():
        return section.icon(0).pixmap(16, 16).toImage()

    try:
        window._set_theme_mode(True)
        qapp.processEvents()
        dark_icon = tree_icon()
        assert TOKENS[True]["text_secondary"] in QLabel.text(body)
        window._set_theme_mode(False)
        qapp.processEvents()
        assert TOKENS[False]["text_secondary"] in QLabel.text(body), "the keys' colour"
        assert tree_icon() != dark_icon, "the icon is drawn again in the light theme's colour"
    finally:
        window._set_theme_mode(was_dark)


def test_the_fit_form_takes_only_what_changed_in_the_explore_preview(qapp, edited_window):
    """Opening a dataset switches the preview's signal unit; that must not replace the
    shown model's preprocessing (SNV) by the preview's (none)."""
    from enginir.preprocessing import PreprocessingRecipe

    window = edited_window(lambda project: project.fit(
        "Layers", recipe=PreprocessingRecipe(snv=True), n_components=2, name="SNV model",
        fit_spec={"folds": {"group_by": "sample"}}))
    window.show_step("model")
    qapp.processEvents()
    view = window._modeling_view
    form = view.fit_form
    assert form._snv.isChecked() and not view.fit_settings_changed(), "the model's settings"
    assert not window.state.preprocess_snv

    window.show_step("data")
    window.show_dataset(sorted(window.project_state.datasets)[0])
    window.wait_for_data_file()
    qapp.processEvents()
    assert form._snv.isChecked() and not view.fit_settings_changed()

    window.state.set_preprocess_filter("sg_d1")  # an edit of the preview
    qapp.processEvents()
    assert form._filter_combo.currentData() == "sg_d1", "the changed setting follows"
    assert form._snv.isChecked(), "the others stay"
    assert view.fit_settings_changed()
    window.state.set_preprocess_filter("none")


def test_running_jobs_finish_before_the_interpreter_exits(qapp):
    """A job thread still reading a file at exit crashed or hung the process."""
    import time

    from PyQt6.QtCore import QObject

    from enginir.viewer._jobs import _RUNNING_JOBS, start_background_job, wait_for_running_jobs

    owner = QObject()
    done = []
    start_background_job(owner=owner, fn=lambda: time.sleep(0.3), on_success=done.append,
                         on_error=done.append, on_finished=lambda: None)
    assert _RUNNING_JOBS
    assert wait_for_running_jobs(timeout=10.0), "no event loop runs at exit"
    _wait_until(qapp, lambda: not _RUNNING_JOBS)
    assert done == [None], "the result is still delivered when the loop runs again"


@pytest.mark.parametrize("size", [(3.2, 2.4), (5.0, 3.5)])
def test_the_statistics_of_predicted_vs_target_cover_no_point(size):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from enginir.viewer._model_diagnostics import _covered, _render_pred_vs_ref
    from enginir.viewer._theme import panel_colors

    rng = np.random.default_rng(1)
    y_true = np.repeat([1.0, 4.0, 7.0, 10.0], 5)
    y_pred = y_true + rng.normal(0, 1.2, y_true.size)
    y_pred[:3] = 9.5  # wrong predictions where the block sits by default (top left)
    fig = Figure(figsize=size, dpi=100)
    FigureCanvasAgg(fig)
    _render_pred_vs_ref(fig, {"y_true_roi": y_true, "y_pred_roi": y_pred}, panel_colors())
    fig.canvas.draw()
    ax = fig.axes[0]
    block = next(text for text in ax.texts if "RMSECV" in text.get_text())
    assert _covered(ax, block, fig.canvas.get_renderer(), y_true, y_pred) == 0


def test_the_example_opens_without_open_checks_and_a_good_model(example):
    """The example shows good practice: values vary within each scan, one sample per tablet."""
    from enginir import Project

    project = Project.open(example)
    open_checks = [check.message for check in project.checks()
                   if check.severity in ("attention", "blocked")]
    assert open_checks == []
    assert project.model("PLS 2 LV").verdict().level == "good"
