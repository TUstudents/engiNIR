"""Reopening a project, moved data files and several windows (project attachment contract)."""

from __future__ import annotations

import shutil
import time

import numpy as np
import pytest

from enginir import Project
from enginir.io import save_hdf5
from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


@pytest.fixture()
def trays(tmp_path):
    """tray_a.h5 with A1 (API 2), tray_b.h5 with B1; target API."""
    rng = np.random.default_rng(31)
    wavelengths = np.linspace(1000.0, 1700.0, 6)
    paths = {}
    for name in ("tray_a", "tray_b"):
        paths[name] = tmp_path / f"{name}.h5"
        save_hdf5(str(paths[name]), rng.integers(500, 3000, (12, 6, 6)).astype(np.uint16),
                  wavelengths)
    project = Project.create(tmp_path / "trays.h5")
    project.add_target("API", unit="%")
    project.add_dataset(paths["tray_a"]).add_roi("A1", frames=(1, 3), rows=(0, 2),
                                                 values={"API": 2.0})
    project.add_dataset(paths["tray_b"]).add_roi("B1", frames=(4, 6), rows=(1, 3))
    return project, paths


def _dataset_id(window, name):
    return next(d for d, ref in window.project_state.datasets.items()
                if ref.source_path.endswith(name))


def _process(qapp, n=10):
    for _ in range(n):
        qapp.processEvents()
        time.sleep(0.005)


def test_the_data_file_tab_loads_the_last_shown_file(qapp, trays):
    from enginir.viewer import open_project

    project, paths = trays
    first = open_project(project.path)
    try:
        first.show_data_file(_dataset_id(first, "tray_b.h5"))
        first.wait_for_data_file()
    finally:
        first.close()

    window = open_project(project.path)  # a project with ROIs opens on them, nothing loaded
    try:
        assert window._explore_tabs.currentWidget() is window._project_rois
        assert not window.has_attached_dataset()
        window._explore_tabs.setCurrentWidget(window._data_file_page)
        window.wait_for_data_file()
        assert window._cube_path == str(paths["tray_b"].resolve())
    finally:
        window.close()


def test_a_missing_last_file_is_not_loaded(qapp, trays):
    from enginir.viewer import open_project

    project, paths = trays
    first = open_project(project.path)
    try:
        first.show_data_file(_dataset_id(first, "tray_a.h5"))
        first.wait_for_data_file()
    finally:
        first.close()
    paths["tray_a"].unlink()
    window = open_project(project.path)
    try:
        window._explore_tabs.setCurrentWidget(window._data_file_page)
        window.wait_for_data_file()
        assert not window.has_attached_dataset()
        assert "tray_a.h5 (not found)" in [window._data_file_combo.itemText(i)
                                          for i in range(window._data_file_combo.count())]
    finally:
        window.close()


def test_locate_file_from_the_not_found_question(qapp, trays, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    from enginir.viewer import open_project

    project, paths = trays
    moved = paths["tray_a"].parent / "archive" / "tray_a_renamed.h5"
    moved.parent.mkdir()
    shutil.move(paths["tray_a"], moved)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: next(
        b for b in self.buttons() if b.text() == "Locate file..."))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(moved), "")))
    window = open_project(project.path)
    try:
        window.show_data_file(_dataset_id(window, "tray_a.h5"))
        window.wait_for_data_file()
        assert window._cube_path == str(moved.resolve())
        (ref,) = [r for r in Project.open(project.path).workspace.datasets.values()
                  if r.source_path.endswith("renamed.h5")]
        assert ref.source_path == str(moved.resolve())
    finally:
        window.close()


def test_a_different_file_is_refused_and_nothing_changes(qapp, trays, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from enginir.viewer import open_project

    project, paths = trays
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda _p, _t, text, *a, **k: warnings.append(text)))
    tray_a = str(paths["tray_a"].resolve())
    window = open_project(project.path)
    try:
        assert not window.locate_data_file(_dataset_id(window, "tray_a.h5"),
                                           str(paths["tray_b"]))  # same shape, other pixels
        assert warnings and "already registered" in warnings[0]
        impostor = paths["tray_a"].parent / "impostor.h5"
        save_hdf5(str(impostor), np.full((12, 6, 6), 700, np.uint16),
                  np.linspace(1000.0, 1700.0, 6))
        assert not window.locate_data_file(_dataset_id(window, "tray_a.h5"), str(impostor))
        assert "the pixels of ROI 'A1' differ" in warnings[1]
        assert window.project_state.datasets[_dataset_id(window, "tray_a.h5")].source_path \
            == tray_a
        assert not window.has_attached_dataset()
    finally:
        window.close()
    assert any(r.source_path == tray_a for r in Project.open(project.path).workspace
               .datasets.values())


def test_a_second_window_on_the_project_refreshes(qapp, trays):
    from enginir.viewer import open_project

    project, _paths = trays
    first, second = open_project(project.path), open_project(project.path)
    try:
        api = next(iter(first.project_state.targets))
        b1 = next(r for r, roi in first.project_state.rois.items() if roi.name == "B1")
        first.project_state.set_roi_values([(b1, api, 7.5)])
        _process(qapp)
        assert second.project_state.target_assignments[(b1, api)].value == 7.5
        second.show_roi_table()
        model = second._roi_table.model
        cells = [str(model.data(model.index(r, 5)) or "") for r in range(model.rowCount())]
        assert cells == ["2", "7.5"]
        # The refreshed window writes against the current project, not a stale copy.
        second.project_state.set_roi_values([(b1, api, None)])
        _process(qapp)
        assert (b1, api) not in first.project_state.target_assignments
    finally:
        first.close()
        second.close()
