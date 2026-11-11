"""The ROI table: target values of all ROIs, typed, removed and pasted (ROI/target contract)."""

from __future__ import annotations

import numpy as np
import pytest

from enginir import Project
from enginir.io import save_hdf5
from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


@pytest.fixture()
def project(tmp_path):
    """Two trays: tray_a with A1, A2, A10 (API of A1 = 2), tray_b with B1; targets API and fat."""
    rng = np.random.default_rng(6)
    wavelengths = np.linspace(1000.0, 1700.0, 6)
    paths = {}
    for name in ("tray_a", "tray_b"):
        paths[name] = tmp_path / f"{name}.h5"
        save_hdf5(str(paths[name]), rng.integers(500, 3000, (20, 6, 6)).astype(np.uint16),
                  wavelengths)
    project = Project.create(tmp_path / "trays.h5")
    project.add_target("API", unit="%")
    project.add_target("fat", unit="%")
    tray_a = project.add_dataset(paths["tray_a"])
    tray_a.add_roi("A10", frames=(12, 13), rows=(0, 1))
    tray_a.add_roi("A2", frames=(4, 5), rows=(0, 1))
    tray_a.add_roi("A1", frames=(0, 1), rows=(0, 1), values={"API": 2.0})
    project.add_dataset(paths["tray_b"]).add_roi("B1", frames=(0, 1), rows=(0, 1))
    return project, paths


@pytest.fixture()
def table(qapp, project):
    from enginir.viewer._roi_table import RoiTableWidget
    from enginir.viewer._state import ProjectState

    state = ProjectState()
    state.open_project(project[0].path)
    widget = RoiTableWidget(state)
    widget.resize(600, 300)
    widget.show()
    return widget


def _cells(widget) -> list[list[str]]:
    model = widget.model
    return [[str(model.data(model.index(r, c)) or "") for c in range(model.columnCount())]
            for r in range(model.rowCount())]


def _stored(project_path, roi_name, target):
    reopened = Project.open(project_path)
    key = (reopened.roi(roi_name).roi_id, reopened.target(target).target_id)
    assignment = reopened.workspace.target_assignments.get(key)
    return None if assignment is None else assignment.value


def test_rows_columns_and_natural_order(table):
    model = table.model
    headers = [model.headerData(c, __import__("PyQt6.QtCore").QtCore.Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    assert headers == ["Dataset", "ROI", "Pixels", "Set", "Sample", "API (%)", "fat (%)"]
    assert _cells(table) == [["tray_a.h5", "A1", "4", "Calibration", "", "2", ""],
                             ["tray_a.h5", "A2", "4", "Calibration", "", "", ""],
                             ["tray_a.h5", "A10", "4", "Calibration", "", "", ""],
                             ["tray_b.h5", "B1", "4", "Calibration", "", "", ""]]


def test_typing_stores_values_and_clearing_removes_them(table, project):
    model, path = table.model, project[0].path
    assert model.setData(model.index(1, 5), "3,5")  # decimal comma
    assert _stored(path, "A2", "API") == 3.5
    assert model.setData(model.index(0, 5), "")  # clear A1
    assert _stored(path, "A1", "API") is None, "a cleared cell is a missing value, not zero"
    assert table.project_state.target_assignments.get(
        (Project.open(path).roi("A1").roi_id, Project.open(path).target("API").target_id)) is None
    assert _cells(table)[0][5] == "" and _cells(table)[1][5] == "3.5"


def test_invalid_text_is_refused_and_nothing_changes(table, project, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    errors: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda _p, _t, text, *a, **k: errors.append(text)))
    model = table.model
    assert not model.setData(model.index(0, 5), "two")
    assert not model.setData(model.index(0, 5), "inf")
    assert len(errors) == 2 and "not a number" in errors[0]
    assert _stored(project[0].path, "A1", "API") == 2.0


def test_enter_stores_and_moves_to_the_next_row(qapp, table, project):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QAbstractItemView

    view = table.view
    view.setCurrentIndex(table.model.index(1, 6))
    view.edit(view.currentIndex())
    QTest.keyClicks(view.focusWidget() or view, "7.25")
    QTest.keyClick(view.focusWidget() or view, Qt.Key.Key_Return)
    qapp.processEvents()
    assert _stored(project[0].path, "A2", "fat") == 7.25
    assert view.currentIndex().row() == 2 and view.currentIndex().column() == 6
    assert view.state() == QAbstractItemView.State.EditingState


def test_pasting_a_column_fills_rows_downwards(table, project):
    view, model, path = table.view, table.model, project[0].path
    view.setCurrentIndex(model.index(0, 5))
    assert table.paste_clipboard("1.5\n\n2.5\tignored\n")  # A1, A2 (cleared), A10
    assert (_stored(path, "A1", "API"), _stored(path, "A2", "API"),
            _stored(path, "A10", "API")) == (1.5, None, 2.5)


@pytest.mark.parametrize(("text", "message"), [
    ("1\nx\n3", "Line 2: 'x' is not a number"),
    ("1\n2\n3\n4\n5", "5 values do not fit below this row"),
])
def test_an_invalid_paste_writes_nothing(table, project, monkeypatch, text, message):
    from PyQt6.QtWidgets import QMessageBox

    errors: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda _p, _t, text, *a, **k: errors.append(text)))
    table.view.setCurrentIndex(table.model.index(0, 5))
    assert not table.paste_clipboard(text)
    assert errors and errors[0].startswith(message)
    assert _stored(project[0].path, "A1", "API") == 2.0
    assert _stored(project[0].path, "A2", "API") is None


def test_filters_by_data_file_and_missing_values(table):
    state = table.project_state
    tray_b = next(d for d, ref in state.datasets.items() if ref.source_path.endswith("tray_b.h5"))
    table.show_data_file(tray_b)
    assert [row[1] for row in _cells(table)] == ["B1"]
    table.show_data_file(None)
    table._missing.setChecked(True)  # without a value for API (the first target)
    assert [row[1] for row in _cells(table)] == ["A2", "A10", "B1"]
    assert table._count.text().startswith("3 ROIs | API: 0 of 3")


def test_the_table_works_without_any_data_file(qapp, project):
    from enginir.viewer import open_project

    project_obj, paths = project
    for path in paths.values():
        path.unlink()
    window = open_project(project_obj.path)
    try:
        window.show_roi_table()
        table = window._roi_table
        assert window._roi_table_dock.isVisible() and table.model.rowCount() == 4
        assert table.model.setData(table.model.index(3, 6), "12")
        assert _stored(project_obj.path, "B1", "fat") == 12.0
    finally:
        window.close()


def test_target_values_button_opens_the_table_on_the_shown_file_and_target(qapp, project):
    from enginir.viewer import open_viewer

    project_obj, paths = project
    window = open_viewer(paths["tray_b"], project=project_obj.path)
    qapp.processEvents()  # the browser fills its tree on the next event-loop pass
    try:
        modeling = window._modeling_view
        fat = next(t for t, ref in window.project_state.targets.items() if ref.name == "fat")
        modeling._selected_target_id = fat
        modeling._btn_assign_target.click()
        table = window._roi_table
        assert window._roi_table_dock.isVisible()
        assert [row[1] for row in _cells(table)] == ["B1"]
        assert table.model.target_id(table.view.currentIndex().column()) == fat

        # Selecting a row selects the ROI in the browser and on the image.
        table.show_data_file(None)
        selected: list[str] = []
        window._project_browser.itemSelectionChanged.connect(
            lambda: selected.append(str(window._selected_project_browser_payload())))
        table.view.setCurrentIndex(table.model.index(3, 1))  # B1
        qapp.processEvents()
        b1 = next(r for r, ref in window.project_state.rois.items() if ref.name == "B1")
        assert window._canvas._selected_roi_id == b1
    finally:
        window.close()


def test_the_library_removes_values_and_checks_batches_first(project):
    from enginir.project import TargetAssignment

    project_obj, _paths = project
    project_obj.set_value("A1", "API", None)
    assert _stored(project_obj.path, "A1", "API") is None
    a2, api = project_obj.roi("A2").roi_id, project_obj.target("API").target_id
    with pytest.raises(ValueError, match="finite"):
        project_obj.save_assignments([
            TargetAssignment(roi_id=a2, target_id=api, value=1.0, assigned_at="2026-11-03T00:00:00"),
            TargetAssignment(roi_id=a2, target_id=api, value=float("nan"),
                             assigned_at="2026-11-03T00:00:00")])
    assert _stored(project_obj.path, "A2", "API") is None


def test_a_stored_value_can_be_replaced(project):
    """Updating a value is the documented editing model (it used to fail as a duplicate)."""
    project_obj, _paths = project
    project_obj.set_value("A1", "API", 4.5)
    assert _stored(project_obj.path, "A1", "API") == 4.5


def test_sample_labels_are_typed_pasted_and_removed(table, project):
    """The Sample column: text is stored without surrounding spaces; blank removes it."""
    model, view, path = table.model, table.view, project[0].path
    assert model.setData(model.index(0, 4), "  blend 1 ")
    assert Project.open(path).roi("A1").sample == "blend 1"
    assert _cells(table)[0][4] == "blend 1"
    view.setCurrentIndex(model.index(1, 4))
    assert table.paste_clipboard("blend 1\tignored\n\nblend 2\n")  # A2, A10 (blank), B1
    reopened = Project.open(path)
    assert [reopened.roi(name).sample for name in ("A1", "A2", "A10", "B1")] == [
        "blend 1", "blend 1", None, "blend 2"]
    assert model.setData(model.index(0, 4), " ")
    assert Project.open(path).roi("A1").sample is None
    assert _stored(path, "A1", "API") == 2.0, "labels never touch target values"


def test_the_set_column_switches_an_roi_between_calibration_and_prediction(qapp, project):
    from PyQt6.QtCore import Qt

    from enginir.viewer._roi_table import RoiTableModel, _SetDelegate
    from enginir.viewer._state import ProjectState

    project, _paths = project
    state = ProjectState()
    state.open_project(str(project.path))
    model = RoiTableModel(state)
    assert model.headerData(3, Qt.Orientation.Horizontal) == "Set"
    index = model.index(0, 3)
    assert model.flags(index) & Qt.ItemFlag.ItemIsEditable
    assert model.data(index) == "Calibration"
    # The drop-down editor of the Set column (terminology contract, Calibration and Prediction).
    delegate = _SetDelegate()
    editor = delegate.createEditor(None, None, index)
    assert [editor.itemText(i) for i in range(editor.count())] == ["Calibration", "Prediction"]
    delegate.setEditorData(editor, index)
    assert editor.currentText() == "Calibration"
    editor.setCurrentIndex(1)
    delegate.setModelData(editor, model, index)
    roi_id = model.roi_id(0)
    assert state.rois[roi_id].use == "prediction"
    assert Project.open(project.path).workspace.rois[roi_id].use == "prediction", "stored"
    assert model.data(model.index(0, 3)) == "Prediction"
    failed: list[str] = []
    model.edit_failed.connect(failed.append)
    assert not model.setData(model.index(0, 3), "Test", Qt.ItemDataRole.EditRole)
    assert failed and state.rois[roi_id].use == "prediction", "an unknown set is refused"
