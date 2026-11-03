"""Projects with ROIs from several data files (multi-file plan, attachment contract)."""

from __future__ import annotations

import numpy as np
import pytest

from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


@pytest.fixture()
def two_file_project(tmp_path):
    """A project with two ROIs drawn on tray_a.h5 and one on tray_b.h5."""
    from enginir import Project
    from enginir.io import save_hdf5

    rng = np.random.default_rng(1)
    wavelengths = np.linspace(1000.0, 1700.0, 12)
    paths = {}
    for name in ("tray_a", "tray_b"):
        paths[name] = tmp_path / f"{name}.h5"
        save_hdf5(str(paths[name]), rng.integers(500, 3000, (16, 8, 12)).astype(np.uint16),
                  wavelengths)
    project_path = tmp_path / "trays.h5"
    project = Project.create(project_path)
    project.add_target("API", unit="%")
    tray_a = project.add_dataset(paths["tray_a"])
    tray_a.add_roi("A1", frames=(0, 3), rows=(0, 3), values={"API": 2.0})
    tray_a.add_roi("A2", frames=(5, 8), rows=(0, 3), values={"API": 4.0})
    project.add_dataset(paths["tray_b"]).add_roi("B1", frames=(2, 6), rows=(2, 5),
                                                 values={"API": 6.0})
    return project_path, paths


def _drawn_roi_names(viewer) -> list[str]:
    rois = viewer.project_state.rois
    return sorted(rois[roi_id].name for roi_id in viewer._canvas._roi_item_map)


def test_a_window_draws_only_the_rois_of_its_data_file(qapp, two_file_project):
    from enginir.viewer import open_viewer

    project_path, paths = two_file_project
    window_a = open_viewer(paths["tray_a"], project=project_path)
    window_b = open_viewer(paths["tray_b"], project=project_path)
    try:
        assert len(window_a.project_state.rois) == 3  # the project holds all of them
        assert _drawn_roi_names(window_a) == ["A1", "A2"]
        assert _drawn_roi_names(window_b) == ["B1"]
    finally:
        window_a.close()
        window_b.close()


def test_a_project_only_window_draws_the_rois_of_the_file_it_attaches(qapp, two_file_project):
    from enginir.viewer import open_project

    project_path, paths = two_file_project
    window = open_project(project_path)
    try:
        assert len(window.project_state.rois) == 3
        assert _drawn_roi_names(window) == []
        window._open_data_path(str(paths["tray_b"]))  # File > Open Data
        window.wait_for_data_file()
        assert _drawn_roi_names(window) == ["B1"]
    finally:
        window.close()


def test_project_writes_are_confirmed_in_the_status_bar(qapp, two_file_project, monkeypatch):
    from PyQt6.QtWidgets import QDialog

    import enginir.viewer._eda as eda
    from enginir.project import TargetAssignment
    from enginir.project._ids import now_iso
    from enginir.viewer import open_viewer

    project_path, paths = two_file_project
    monkeypatch.setattr(eda.NewROIDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(eda.NewROIDialog, "roi_name", property(lambda self: "A3"))
    monkeypatch.setattr(eda.NewROIDialog, "target_value", property(lambda self: None))
    window = open_viewer(paths["tray_a"], project=project_path)
    try:
        window._eda_view.on_training_roi_selected(10, 13, 4, 7)
        assert window._saved_label.text() == "✓ ROI 'A3' saved to project (16 pixels)"
        assert _drawn_roi_names(window) == ["A1", "A2", "A3"]

        state = window.project_state
        roi_id = next(r.roi_id for r in state.rois.values() if r.name == "A3")
        target_id = next(iter(state.targets))
        state.assign_target_to_roi(TargetAssignment(roi_id=roi_id, target_id=target_id,
                                                    value=5.5, assigned_at=now_iso()))
        assert window._saved_label.text() == "✓ API = 5.5 % saved for ROI 'A3'"

        state.remove_roi(roi_id)
        assert window._saved_label.text() == "✓ ROI 'A3' removed from project"
        assert _drawn_roi_names(window) == ["A1", "A2"]
    finally:
        window.close()


@pytest.fixture()
def tray_c(tmp_path):
    """A third data file, not in the project: dark frames 0-1, white tile in frames 14-15."""
    from enginir.io import save_hdf5

    rng = np.random.default_rng(9)
    cube = rng.integers(1000, 2000, (16, 8, 12)).astype(np.uint16)
    cube[:2] = 100
    cube[14:] = 3500
    path = tmp_path / "tray_c.h5"
    save_hdf5(str(path), cube, np.linspace(1000.0, 1700.0, 12))
    return path, cube


def _dialog_answers(monkeypatch, configure):
    """Make AddDataFileDialog.exec apply ``configure`` and press OK."""
    from PyQt6.QtWidgets import QDialog

    from enginir.viewer._data_files import AddDataFileDialog

    seen: list[AddDataFileDialog] = []

    def exec_(self):
        seen.append(self)
        configure(self)
        self._accept()
        return (QDialog.DialogCode.Accepted if self.request is not None
                else QDialog.DialogCode.Rejected)

    monkeypatch.setattr(AddDataFileDialog, "exec", exec_)
    return seen


def test_open_data_adds_the_file_with_frame_references_in_the_same_window(
        qapp, two_file_project, tray_c, monkeypatch):
    from enginir.viewer import open_viewer

    project_path, paths = two_file_project
    path_c, cube_c = tray_c
    dialogs = _dialog_answers(monkeypatch, lambda d: (
        d.set_choice("references"), d.dark_source().use_frames(0, 1),
        d.white_source().use_frames(14, 15), d.set_unit("Reflectance")))
    window = open_viewer(paths["tray_a"], project=project_path)
    try:
        window._open_data_path(str(path_c))  # File > Open Data in a window showing tray_a
        window.wait_for_data_file()
        assert len(dialogs) == 1 and window._child_viewers == []
        state = window.project_state
        dataset_id = window._eda_view.shown_dataset_id()
        assert state.datasets[dataset_id].source_path == str(path_c.resolve())
        calibration = state.active_calibration(dataset_id)
        assert calibration is not None
        assert calibration.white_ref_source == f"{path_c.resolve()} frames 14-15"
        np.testing.assert_allclose(calibration.white_ref_data, cube_c[14:].mean(axis=0))
        np.testing.assert_array_equal(window.state.white_ref, calibration.white_ref_data)
        assert window.state.y_unit == "Reflectance"
        assert _drawn_roi_names(window) == []  # tray_c has no ROIs yet
        assert "White: mean 3500 counts, 0.00 % saturated" in dialogs[0].check_text()
        combo = window._data_file_combo
        assert [combo.itemText(i) for i in range(combo.count()) if combo.itemData(i)] == [
            "tray_a.h5", "tray_b.h5", "tray_c.h5", "Add data file..."]
        assert window._saved_label.text() == "✓ Data file 'tray_c.h5' added to project"
    finally:
        window.close()


def test_rois_on_an_added_file_keep_its_calibration(qapp, two_file_project, tray_c,
                                                    monkeypatch):
    from PyQt6.QtWidgets import QDialog

    import enginir.viewer._eda as eda
    from enginir.viewer import open_project

    project_path, _paths = two_file_project
    path_c, _cube = tray_c
    _dialog_answers(monkeypatch, lambda d: (
        d.set_choice("references"), d.dark_source().use_frames(0, 1),
        d.white_source().use_frames(14, 15)))
    monkeypatch.setattr(eda.NewROIDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(eda.NewROIDialog, "roi_name", property(lambda self: "C1"))
    monkeypatch.setattr(eda.NewROIDialog, "target_value", property(lambda self: None))
    window = open_project(project_path)
    try:
        dataset_id = window.add_data_file(str(path_c))
        window.wait_for_data_file()
        calibration = window.project_state.active_calibration(str(dataset_id))
        n_calibrations = len(window.project_state.calibrations)
        window.state.set_y_unit("DN")  # a preview only
        window._eda_view.on_training_roi_selected(3, 5, 1, 3)
        roi = window.project_state.rois[next(r for r, v in window.project_state.rois.items()
                                             if v.name == "C1")]
        assert calibration is not None and roi.calibration_id == calibration.calibration_id
        assert len(window.project_state.calibrations) == n_calibrations
    finally:
        window.close()


def test_new_references_on_a_file_with_rois_ask_first(qapp, two_file_project, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QMessageBox

    import enginir.viewer._eda as eda
    from enginir.viewer import open_viewer

    project_path, paths = two_file_project
    asked: list[str] = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda _p, _t, text, *a, **k: asked.append(text) or QMessageBox.StandardButton.No))
    monkeypatch.setattr(eda.NewROIDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(eda.NewROIDialog, "roi_name", property(lambda self: "A3"))
    monkeypatch.setattr(eda.NewROIDialog, "target_value", property(lambda self: None))
    window = open_viewer(paths["tray_a"], project=project_path)
    try:
        window.state.set_dark_ref(np.full((8, 12), 90.0))
        window.state.set_white_ref(np.full((8, 12), 3600.0))
        window._eda_view.on_training_roi_selected(10, 13, 4, 7)
        assert asked and "2 ROI(s) drawn before keep their own calibration" in asked[0]
        assert "A3" not in {roi.name for roi in window.project_state.rois.values()}
    finally:
        window.close()


def test_switching_data_files_and_sharing_references(qapp, two_file_project, tray_c,
                                                     monkeypatch):
    from enginir.viewer import open_project

    project_path, paths = two_file_project
    path_c, _cube = tray_c
    window = open_project(project_path)
    try:
        dialogs = _dialog_answers(monkeypatch, lambda d: (
            d.set_choice("references"), d.dark_source().use_frames(0, 1),
            d.white_source().use_frames(14, 15)))
        window.add_data_file(str(path_c))
        window.wait_for_data_file()
        combo = window._data_file_combo
        combo.setCurrentIndex(combo.findText("tray_a.h5"))
        window._on_data_file_chosen(combo.currentIndex())
        window.wait_for_data_file()
        assert _drawn_roi_names(window) == ["A1", "A2"]
        assert window._cube_path == str(paths["tray_a"].resolve())

        # tray_b is registered already, so it is shown without the dialog.
        window._open_data_path(str(paths["tray_b"]))
        window.wait_for_data_file()
        assert _drawn_roi_names(window) == ["B1"]
        assert len(dialogs) == 1, "only the new file asked for its calibration"
    finally:
        window.close()


def test_a_missing_data_file_is_reported_and_nothing_changes(qapp, two_file_project,
                                                             monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from enginir.viewer import open_project

    project_path, paths = two_file_project
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda _p, title, text, *a, **k: warnings.append(f"{title}: {text}")))
    paths["tray_b"].unlink()
    window = open_project(project_path)
    try:
        combo = window._data_file_combo
        assert "tray_b.h5 (not found)" in [combo.itemText(i) for i in range(combo.count())]
        dataset_id = window._registered_dataset_id(str(paths["tray_b"]))
        window.show_data_file(str(dataset_id))
        assert warnings and warnings[0].startswith("Data file not found")
        assert window._cube_path is None
    finally:
        window.close()


def test_the_dialog_blocks_ok_until_the_references_are_valid(qapp, tray_c):
    from PyQt6.QtWidgets import QDialogButtonBox

    from enginir._workflow import read_header
    from enginir.viewer._data_files import AddDataFileDialog, reference_summary

    path_c, cube = tray_c
    dialog = AddDataFileDialog(str(path_c), read_header(str(path_c)), {}, {})
    dialog.set_choice("references")
    dialog.white_source().use_file("")  # no white file chosen
    assert not dialog.check_references()
    ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok is not None and not ok.isEnabled()
    dialog.white_source().use_frames(14, 15)
    assert dialog.check_references() and ok.isEnabled()
    request = dialog.build_request()
    assert (request.dark_frames, request.white_frames) == ((0, 2), (14, 15))

    saturated = cube[14:].copy()
    saturated[0, 0, :6] = 65535  # 6 of 192 values at the uint16 maximum
    _mean, share = reference_summary(saturated, 65535.0)
    assert share == pytest.approx(6 / saturated.size * 100)


def test_a_registered_file_opens_with_its_stored_calibration(qapp, tmp_path, tray_c):
    from enginir import Project
    from enginir.viewer import open_viewer

    path_c, cube_c = tray_c
    project = Project.create(tmp_path / "c.h5")
    project.add_dataset(path_c, dark_frames=(0, 1), white_frames=(14, 15), unit="Reflectance")
    window = open_viewer(path_c, project=project.path)
    try:
        np.testing.assert_allclose(window.state.white_ref, cube_c[14:].mean(axis=0))
        np.testing.assert_allclose(window.state.dark_ref, cube_c[:2].mean(axis=0))
        assert window.state.y_unit == "Reflectance"
    finally:
        window.close()
