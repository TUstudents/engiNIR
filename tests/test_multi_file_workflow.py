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
        d.white_source().use_frames(14, 15)))
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
        assert window.state.y_unit == "Absorbance"  # the working signal, no choice made yet
        assert calibration.output_signal_unit == "Absorbance"
        assert _drawn_roi_names(window) == []  # tray_c has no ROIs yet
        assert "White: mean 3500 counts, 0.00 % saturated" in dialogs[0].check_text()
        combo = window._data_file_combo
        assert [combo.itemText(i) for i in range(combo.count()) if combo.itemData(i)] == [
            "tray_a.h5", "tray_b.h5", "tray_c.h5", "Add data file..."]
        assert window._saved_label.text() == "✓ Data file 'tray_c.h5' added to project"
    finally:
        window.close()


def test_explore_signals_follow_the_shown_file_and_keep_the_choice(qapp, tmp_path, tray_c):
    """Raw counts are always stored; the shown file decides which signals it can give."""
    from enginir import Project
    from enginir.io import save_hdf5
    from enginir.viewer import open_viewer

    path_c, cube = tray_c
    raw = tmp_path / "raw.h5"
    save_hdf5(str(raw), cube, np.linspace(1000.0, 1700.0, 12))
    project = Project.create(tmp_path / "s.h5")
    calibrated = project.add_dataset(path_c, dark_frames=(0, 1), white_frames=(14, 15))
    project.add_dataset(raw, unit="DN")
    window = open_viewer(path_c, project=project.path)
    try:
        eda = window._eda_view
        buttons = eda._signal_buttons()
        assert window.state.y_unit == "Absorbance" and buttons["Absorbance"].isChecked()
        assert all(button.isEnabled() for button in buttons.values())

        buttons["Reflectance"].click()  # the user's choice
        assert window.state.y_unit == "Reflectance"
        window.show_data_file(window._registered_dataset_id(str(raw)))
        window.wait_for_data_file()
        assert window.state.y_unit == "DN" and buttons["DN"].isChecked()
        assert not buttons["Reflectance"].isEnabled() and "without references" in \
            buttons["Reflectance"].toolTip()

        window.show_data_file(calibrated.ref.dataset_id)  # back: the choice returns
        window.wait_for_data_file()
        assert window.state.y_unit == "Reflectance" and buttons["Reflectance"].isChecked()
        assert buttons["Reflectance"].isEnabled()

        floats = tmp_path / "calibrated.h5"  # already absorbance: no raw counts to show
        save_hdf5(str(floats), np.full((4, 8, 12), 0.3), np.linspace(1000.0, 1700.0, 12))
        window.project_state.add_data_file(str(floats), unit="Absorbance")
        window.show_data_file(window._registered_dataset_id(str(floats)))
        window.wait_for_data_file()
        assert not buttons["DN"].isEnabled() and window.state.y_unit == "Reflectance"
    finally:
        window.close()


def test_the_dialog_asks_the_unit_only_for_calibrated_files(qapp, tmp_path, tray_c):
    from enginir._workflow import read_header
    from enginir.io import save_hdf5
    from enginir.viewer._data_files import AddDataFileDialog

    path_c, _cube = tray_c
    counts = AddDataFileDialog(str(path_c), read_header(str(path_c)), {}, {})
    assert not counts._unit.isVisibleTo(counts)
    for choice, unit in (("references", "Absorbance"), ("levels", "Absorbance"),
                         ("raw", "DN")):
        counts.set_choice(choice)
        assert counts.build_request().unit == unit
    floats = tmp_path / "f.h5"
    save_hdf5(str(floats), np.full((4, 8, 12), 0.5), np.linspace(1000.0, 1700.0, 12))
    dialog = AddDataFileDialog(str(floats), read_header(str(floats)), {}, {})
    assert dialog._unit.isVisibleTo(dialog)
    dialog.set_unit("Reflectance")
    assert dialog.build_request().unit == "Reflectance"


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
    monkeypatch.setattr(QMessageBox, "exec", lambda self: warnings.append(
        f"{self.windowTitle()}: {self.text()}") or 0)  # then Cancel: nothing is located
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


def test_the_dialog_plot_shows_the_data_and_what_each_choice_uses(qapp, tmp_path, tray_c):
    """Every choice plots the file's counts; levels and shared references are drawn too."""
    from enginir import Project
    from enginir._workflow import read_header
    from enginir.viewer._data_files import AddDataFileDialog

    path_c, cube = tray_c
    project = Project.create(tmp_path / "c.h5")
    shared = project.add_dataset(path_c, dark_frames=(0, 1), white_frames=(14, 15))
    ws = project.workspace
    other = tmp_path / "tray_d.h5"
    from enginir.io import save_hdf5

    save_hdf5(str(other), cube, np.linspace(1000.0, 1700.0, 12))
    dialog = AddDataFileDialog(str(other), read_header(str(other)), ws.datasets,
                               {d: ws.calibrations[c] for d, c in ws.active_calibration_ids.items()})

    def drawn() -> dict[str, np.ndarray]:
        dialog.check_references()
        (ax,) = dialog._figure.axes
        assert ax.get_ylabel() == "Counts"
        return {line.get_label(): np.asarray(line.get_ydata()) for line in ax.get_lines()}

    middle = cube[6:11].reshape(-1, 12).mean(axis=0)  # frames 6..10 of 16
    for choice in ("references", "shared", "levels", "raw"):
        dialog.set_choice(choice)
        lines = drawn()
        np.testing.assert_allclose(lines["Data (mean)"], middle)
    dialog.set_choice("shared")
    np.testing.assert_allclose(drawn()["White"],
                               shared.calibration.white_ref_data.mean(axis=0), rtol=1e-6)
    dialog.set_choice("levels")
    levels = drawn()
    assert levels["Black level 128"][0] == 128 and levels["White level 3500"][0] == 3500
    dialog.set_choice("raw")
    assert set(drawn()) == {"Data (mean)"}


def test_the_dialog_levels_are_editable_and_checked(qapp, tray_c):
    from PyQt6.QtWidgets import QDialogButtonBox

    from enginir._workflow import read_header
    from enginir.viewer._data_files import AddDataFileDialog

    path_c, _cube = tray_c
    dialog = AddDataFileDialog(str(path_c), read_header(str(path_c)), {}, {})
    dialog.set_choice("levels")
    assert dialog.build_request().add_dataset_kwargs()["black_level"] == 128.0  # the default
    dialog.set_levels(90.0, 3600.0)
    assert dialog.check_references()
    (ax,) = dialog._figure.axes
    assert {line.get_label() for line in ax.get_lines()} >= {"Black level 90",
                                                            "White level 3600"}
    request = dialog.build_request()
    assert (request.black_level, request.white_level) == (90.0, 3600.0)
    dialog.set_levels(3600.0, 3600.0)
    assert not dialog.check_references()
    ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok is not None and not ok.isEnabled()
    assert "must be above" in dialog.check_text()
    dialog.set_choice("references")  # references take no levels
    assert dialog.build_request().black_level is None


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


def _grid_answers(monkeypatch, configure, seen_preview=None):
    """Make RoiGridDialog.exec apply ``configure`` and press OK (or Cancel if no cells)."""
    from PyQt6.QtWidgets import QDialog

    from enginir.viewer._roi_grid import RoiGridDialog

    def exec_(self):
        configure(self)
        if seen_preview is not None:
            seen_preview.append(self.parent()._canvas.grid_preview_count)
        return QDialog.DialogCode.Accepted if self.cells else QDialog.DialogCode.Rejected

    monkeypatch.setattr(RoiGridDialog, "exec", exec_)


def test_a_grid_starts_from_the_options_defaults_every_time(qapp, two_file_project,
                                                             monkeypatch):
    from enginir.viewer import open_viewer
    from enginir.viewer._grid_settings import GridDefaults, save_grid_defaults

    project_path, paths = two_file_project
    defaults = GridDefaults(n_rows=2, n_columns=2, width=4, height=4, origin_x=8,
                            origin_y=0, spacing_x=0, spacing_y=0)
    save_grid_defaults(defaults)
    opened: list[GridDefaults] = []
    previews: list[int] = []

    def first(dialog):
        opened.append(dialog.form.values())
        # The grid is on the image as soon as the dialog opens: 4 cells with labels.
        assert dialog.parent()._canvas.grid_preview_count == 8
        dialog.toggle_cell(0, 1)  # an empty position

    _grid_answers(monkeypatch, first, previews)
    window = open_viewer(paths["tray_b"], project=project_path)
    try:
        eda = window._eda_view
        eda._btn_roi_grid.click()  # opens the dialog at once; no dragging
        state = window.project_state
        names = sorted(r.name for r in state.rois.values() if r.name.startswith("tray_b-"))
        assert names == ["tray_b-A01", "tray_b-B01", "tray_b-B02"]
        roi = next(r for r in state.rois.values() if r.name == "tray_b-B02")
        assert roi.spatial_bounds == (12, 15, 4, 7)
        assert roi.provenance["grid_position"] == "B2"
        from enginir.io.project_store import load_roi_spectra

        stored = load_roi_spectra(str(project_path), f"/rois/{roi.roi_id}/spectra")
        np.testing.assert_array_equal(
            stored, window.state.cube[12:16, 4:8].reshape(-1, window.state.n_bands))
        assert window._saved_label.text() == "✓ 3 ROIs saved to project"
        # 3 cells with labels and 1 empty cell were previewed, then removed.
        assert previews == [7] and window._canvas.grid_preview_count == 0

        def second(dialog):  # starts from the defaults, not from the first grid's changes
            opened.append(dialog.form.values())
            dialog.set_grid(width=2, name_pattern="{file}-X{n}")

        _grid_answers(monkeypatch, second)
        eda._btn_roi_grid.click()
        _grid_answers(monkeypatch, lambda d: opened.append(d.form.values()) or
                      d.set_grid(n_rows=1, n_columns=1, name_pattern="{file}-Y"))
        eda._btn_roi_grid.click()
        assert opened == [defaults, defaults, defaults]
    finally:
        window.close()


def test_a_grid_with_a_taken_name_is_refused_in_the_dialog(qapp, tmp_path, monkeypatch):
    """Scans with the same file name in different folders give the same {file}; the dialog
    refuses the clash before OK, so no error box follows."""
    from PyQt6.QtWidgets import QDialogButtonBox

    from enginir import Project
    from enginir.io import save_hdf5
    from enginir.viewer import open_viewer
    from enginir.viewer._grid_settings import GridDefaults, save_grid_defaults

    paths = []
    for folder in ("run1", "run2"):
        (tmp_path / folder).mkdir()
        paths.append(tmp_path / folder / "stream_hsi.h5")
        save_hdf5(str(paths[-1]), np.full((16, 8, 6), 1000, np.uint16),
                  np.linspace(1000.0, 1500.0, 6))
    project = Project.create(tmp_path / "p.h5")
    first = project.add_dataset(paths[0])
    first.add_roi("stream_hsi-A01", frames=(0, 3), rows=(0, 3))
    project.add_dataset(paths[1])
    save_grid_defaults(GridDefaults(n_rows=1, n_columns=2, width=4, height=4, origin_x=0,
                                    origin_y=0, spacing_x=0))
    seen: list[tuple[str, bool]] = []

    def answer(dialog):
        ok = dialog._buttons_box.button(QDialogButtonBox.StandardButton.Ok)
        seen.append((dialog.status_text, ok.isEnabled()))
        dialog.set_grid(name_pattern="run2-{row}{column:02d}")  # a pattern that is free
        seen.append((dialog.status_text, ok.isEnabled()))

    _grid_answers(monkeypatch, answer)
    window = open_viewer(paths[1], project=project.path)
    try:
        before = set(window.project_state.rois)
        window._eda_view._btn_roi_grid.click()
        assert seen[0] == ("1 name(s) are already used in the project, e.g. 'stream_hsi-A01'. "
                           "Choose another name pattern (several data files may have the "
                           "same file name).", False)
        assert seen[1] == ("2 ROIs will be created", True)
        names = {r.name for r in window.project_state.rois.values()} - {
            window.project_state.rois[r].name for r in before}
        assert names == {"run2-A01", "run2-A02"}
    finally:
        window.close()


def test_the_grid_dialog_reports_counts_outside_cells_and_invalid_patterns(qapp):
    from PyQt6.QtWidgets import QDialogButtonBox

    from enginir.viewer._grid_settings import GridDefaults
    from enginir.viewer._roi_grid import RoiGridDialog

    dialog = RoiGridDialog((40, 20), file="tray03", defaults=GridDefaults(
        n_rows=2, n_columns=4, width=10, height=10, origin_x=0, origin_y=0, spacing_x=0,
        spacing_y=0))
    previews: list[tuple[int, int, int]] = []
    dialog.preview_changed.connect(lambda cells, skipped, outside: previews.append(
        (len(cells), len(skipped), len(outside))))
    assert dialog.status_text == "8 ROIs will be created"
    dialog.toggle_cell(1, 3)
    assert dialog.status_text == "7 ROIs will be created, 1 position(s) empty"
    assert previews[-1] == (7, 1, 0)
    assert [cell.name for cell in dialog.cells][:2] == ["tray03-A01", "tray03-A02"]
    ok = dialog._buttons_box.button(QDialogButtonBox.StandardButton.Ok)
    assert ok is not None and ok.isEnabled()

    dialog.set_grid(origin_x=5)  # the last column now ends at frame 44 of 40
    assert not ok.isEnabled() and dialog.cells == []
    assert "1 ROI(s) lie outside the image" in dialog.status_text
    assert previews[-1] == (6, 1, 1)
    dialog.toggle_cell(0, 3)  # marking it empty brings the grid back inside
    assert ok.isEnabled() and len(dialog.cells) == 6

    dialog._select_all(False)
    assert not ok.isEnabled() and dialog.cells == []
    dialog._select_all(True)  # A4 and B4 are outside again
    assert not ok.isEnabled() and previews[-1] == (6, 0, 2)
    dialog.set_grid(origin_x=0, name_pattern="{file}")
    assert not ok.isEnabled() and "repeated names" in dialog.status_text
    dialog.set_grid(name_pattern="T{n:02d}", order="columns")
    assert ok.isEnabled() and [c.name for c in dialog.cells][:3] == ["T01", "T02", "T03"]
    assert [(c.row, c.column) for c in dialog.cells][:2] == [(0, 0), (1, 0)]


def test_options_grid_defaults_are_stored_in_the_user_settings(qapp, monkeypatch):
    from enginir.viewer._grid_settings import GridDefaults, load_grid_defaults
    from enginir.viewer._roi_grid import GridDefaultsDialog
    from enginir.viewer._viewer import HSIViewer

    assert load_grid_defaults() == GridDefaults()  # built-in values without settings

    def exec_(self):
        self.form.set_values(GridDefaults(n_rows=8, n_columns=12, width=30, height=30,
                                          origin_x=12, origin_y=5, spacing_x=6, spacing_y=6,
                                          name_pattern="{n:03d}", order="columns"))
        self._accept()
        return self.result()

    monkeypatch.setattr(GridDefaultsDialog, "exec", exec_)
    window = HSIViewer(np.zeros((4, 4, 4), np.uint16), np.arange(4.0), {})
    try:
        assert window.edit_grid_defaults()
    finally:
        window.close()
    stored = load_grid_defaults()
    assert (stored.n_rows, stored.n_columns, stored.origin_x, stored.name_pattern,
            stored.order) == (8, 12, 12, "{n:03d}", "columns")


def test_profile_axes_fit_their_labels_after_a_file_is_shown(qapp, tmp_path):
    """A project window starts with empty plots; a file shown later must not clip tick labels."""
    from enginir import Project
    from enginir.io import save_hdf5
    from enginir.viewer import open_project

    rng = np.random.default_rng(4)
    cube = rng.integers(300, 900, (12, 40, 30)).astype(np.uint16)  # absorbance about 0.6-1.1
    path = tmp_path / "scan.h5"
    save_hdf5(str(path), cube, np.linspace(1000.0, 1700.0, 30))
    project = Project.create(tmp_path / "p.h5")
    project.add_dataset(path, black_level=0.0, white_level=10000.0)
    window = open_project(project.path)
    try:
        window.resize(1400, 900)
        window.show()
        qapp.processEvents()  # the plots draw their empty state first
        window._explore_tabs.setCurrentWidget(window._data_file_page)
        window.show_data_file(next(iter(window.project_state.datasets)))
        window.wait_for_data_file()
        assert window.state.y_unit == "Absorbance"
        for _ in range(5):
            qapp.processEvents()
        for panel in (window._spectrum, window._spatial):
            figure = panel.figure
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            (ax,) = figure.axes
            labels = [label for label in ax.get_yticklabels() if label.get_text()]
            assert labels
            assert min(label.get_window_extent(renderer).x0 for label in labels) >= 0, \
                type(panel).__name__
    finally:
        window.close()


def test_make_defaults_in_the_grid_dialog_sets_the_next_grid(qapp):
    from enginir.viewer._grid_settings import GridDefaults, load_grid_defaults
    from enginir.viewer._roi_grid import RoiGridDialog

    built_in = GridDefaults()
    assert (built_in.n_columns, built_in.n_rows, built_in.width, built_in.height,
            built_in.origin_x, built_in.origin_y, built_in.spacing_x, built_in.spacing_y) \
        == (5, 1, 180, 110, 150, 64, 214, 10)
    dialog = RoiGridDialog((2000, 300), file="tray", defaults=load_grid_defaults())
    dialog.set_grid(n_rows=2, width=170, spacing_y=20)
    assert load_grid_defaults() == built_in  # a change in the dialog is for this grid only
    dialog._make_defaults.click()
    assert load_grid_defaults() == dialog.form.values()
    assert load_grid_defaults().width == 170
    following = RoiGridDialog((2000, 300), file="tray", defaults=load_grid_defaults())
    assert following.form.values() == dialog.form.values()
