"""Smoke tests for WelcomeWindow launcher behavior."""

from __future__ import annotations

import os
import sys

import pytest

from tests._qt import wait_until

@pytest.fixture()
def welcome(qapp):
    from enginir.viewer._welcome import WelcomeWindow

    window = WelcomeWindow()
    yield window
    window.close()


def test_reference_checkboxes_toggle_browse_buttons(welcome):
    from PyQt6.QtWidgets import QCheckBox, QPushButton

    checkboxes = welcome.findChildren(QCheckBox)
    browse_buttons = [
        button for button in welcome.findChildren(QPushButton) if button.text() == "Browse..."
    ]
    assert all(not button.isEnabled() for button in browse_buttons)

    checkboxes[0].setChecked(True)
    assert sum(1 for button in browse_buttons if button.isEnabled()) == 1
    checkboxes[0].setChecked(False)
    checkboxes[1].setChecked(True)
    assert sum(1 for button in browse_buttons if button.isEnabled()) == 1


def test_welcome_loading_context_disables_controls_and_shows_progress(welcome):
    from enginir.viewer._loading import LoadingProgressEvent, ViewerBusyContext

    assert welcome.loading_snapshot() == ("Ready", False)
    context = ViewerBusyContext(
        "Opening dataset...",
        status_widget=welcome.loading_status_widget(),
        disabled_controls=welcome.loading_controls(),
    )

    context.begin()
    try:
        assert welcome.loading_snapshot() == ("Opening dataset...", True)
        assert welcome.loading_status_widget().progress.maximum() == 0  # native busy mode
        context.set_progress(
            LoadingProgressEvent(
                message="Loading cube: 2/4 frames",
                completed=2,
                total=4,
                indeterminate=False,
            )
        )
        assert welcome.loading_snapshot() == ("Loading cube: 2/4 frames", True)
        assert welcome.loading_status_widget().progress.maximum() == 4
        assert welcome.loading_status_widget().progress.value() == 2
        context.set_progress(LoadingProgressEvent(message="Opening viewer..."))
        assert welcome.loading_snapshot() == ("Opening viewer...", True)
        assert welcome.loading_status_widget().progress.maximum() == 0  # busy again
        assert all(not control.isEnabled() for control in welcome.loading_controls())
    finally:
        context.end()

    assert welcome.loading_snapshot() == ("Ready", False)
    assert welcome._btn_open.isEnabled()


def test_open_dataset_button_emits_selected_paths(welcome, monkeypatch, tmp_path):
    dataset, dark, white = (str(tmp_path / f"{name}.eiff") for name in ("sample", "dark", "white"))
    picked = {"Dark reference": dark, "White reference": white}
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getOpenFileName",
        lambda _parent, title, *args, **kwargs: (picked.get(title, dataset), ""))
    received: list[tuple[str, str | None, str | None]] = []
    welcome.file_selected.connect(lambda path, d, w: received.append((path, d, w)))

    welcome._cb_dark.setChecked(True)
    welcome._cb_white.setChecked(True)
    welcome._btn_dark.click()   # the Browse... buttons of the two references
    welcome._btn_white.click()
    welcome._btn_open.click()
    assert received == [(dataset, dark, white)]

    welcome._cb_dark.setChecked(False)  # a reference that is not ticked is not used
    welcome._btn_open.click()
    assert received[-1] == (dataset, None, white)


def test_project_buttons_emit_open_and_create_signals(welcome, monkeypatch, tmp_path):
    project_path = str(tmp_path / "project.h5")
    created_path = str(tmp_path / "created.h5")
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getOpenFileName", lambda *args, **kwargs: (project_path, "")
    )
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName", lambda *args, **kwargs: (created_path, "")
    )
    opened: list[str] = []
    created: list[str] = []
    welcome.project_selected.connect(opened.append)
    welcome.project_created.connect(created.append)

    welcome._btn_open_project.click()
    welcome._btn_new_project.click()

    assert opened == [project_path]
    assert created == [created_path]


def test_browse_reference_updates_label_and_path(welcome, monkeypatch, tmp_path):
    dark_path = str(tmp_path / "dark.eiff")
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getOpenFileName", lambda *args, **kwargs: (dark_path, "")
    )
    welcome._cb_dark.setChecked(True)

    welcome._btn_dark.click()

    assert welcome._dark_path == dark_path
    assert welcome._lbl_dark.text() == os.path.basename(dark_path)


def test_recent_list_filters_missing_and_emits_correct_signal(qapp, tmp_path, monkeypatch):
    import enginir.viewer._recent as recent
    from importlib import reload
    from enginir.io.project_store import create_project
    from enginir.viewer._welcome import WelcomeWindow
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QListWidget

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    reload(recent)

    dataset = str(tmp_path / "sample.eiff")
    open(dataset, "w").close()
    project_path = str(tmp_path / "sample_project.h5")
    create_project(project_path)
    recent.add(dataset)
    recent.add("/nonexistent/ghost.eiff")
    recent.add(project_path)

    window = WelcomeWindow()
    recent_list = window.findChildren(QListWidget)[0]
    wait_until(qapp, lambda: window._recent_checked)
    enabled = [recent_list.item(i).text() for i in range(recent_list.count())
               if recent_list.item(i).flags() & Qt.ItemFlag.ItemIsEnabled]
    assert any("sample.eiff" in item for item in enabled)
    assert any("sample_project.h5" in item for item in enabled)
    assert not any("ghost.eiff" in item for item in enabled), "a missing file is grayed out"

    dataset_received: list[tuple[str, str | None, str | None]] = []
    project_received: list[str] = []
    window.file_selected.connect(lambda path, d, w: dataset_received.append((path, d, w)))
    window.project_selected.connect(project_received.append)

    recent_list.itemActivated.emit(recent_list.item(0))
    recent_list.itemActivated.emit(recent_list.item(1))

    assert dataset_received or project_received


def _record_file_access(monkeypatch, release):
    """Patch the recent-file checks to wait for ``release`` and note their threads."""
    import threading

    import enginir.viewer._files as files

    threads: list[str] = []
    real_exists, real_is_project = os.path.exists, files._is_project_file

    def exists(path):
        if "recent_" in str(path):
            threads.append(threading.current_thread().name)
            release.wait(10)  # an unreachable network drive
        return real_exists(path)

    def is_project(path):
        threads.append(threading.current_thread().name)
        return real_is_project(path)

    monkeypatch.setattr(os.path, "exists", exists)
    monkeypatch.setattr(files, "_is_project_file", is_project)
    return threads


def _recent_files(tmp_path, monkeypatch, *, kinds):
    """A config with a project, a dataset and a missing file (kinds stored or not)."""
    import json

    import enginir.viewer._recent as recent
    from enginir.io.project_store import create_project

    monkeypatch.setattr(recent, "config_dir", lambda: tmp_path / "config")
    project = tmp_path / "recent_project.h5"
    create_project(project)
    dataset = tmp_path / "recent_scan.eiff"
    dataset.write_bytes(b"")
    ghost = tmp_path / "gone" / "recent_ghost.eiff"
    paths = [str(project), str(dataset), str(ghost)]
    entries = ([{"path": str(project), "kind": "project"},
                {"path": str(dataset), "kind": "dataset"},
                {"path": str(ghost), "kind": "dataset"}] if kinds else paths)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "recent.json").write_text(json.dumps(entries), encoding="utf-8")
    return paths


def test_open_recent_never_touches_a_recent_file_on_the_ui_thread(window, qapp, tmp_path,
                                                                  monkeypatch):
    import threading

    project, dataset, ghost = _recent_files(tmp_path, monkeypatch, kinds=True)
    release = threading.Event()
    threads = _record_file_access(monkeypatch, release)
    menu = window._menu_open_recent
    try:
        menu.aboutToShow.emit()
        entries = [act for act in menu.actions() if not act.isSeparator()]
        assert [act.text() for act in entries] == [project, dataset, ghost], \
            "the project first, then the data files"
        assert all(act.isEnabled() for act in entries), "not checked yet"
        release.set()
        wait_until(qapp, lambda: not entries[2].isEnabled())
        assert entries[0].isEnabled() and entries[1].isEnabled()
        assert threads and threading.main_thread().name not in threads, threads
    finally:
        release.set()


def test_the_project_check_of_a_recent_file_waits_for_the_warmups_imports(tmp_path,
                                                                           monkeypatch):
    """A recent entry of unknown kind (written before 1.7.1) is opened with h5py to tell a
    project from a dataset. While the warm-up imports h5py in the background, that check
    waits: two threads importing one package at once can deadlock."""
    import threading

    from enginir.io.project_store import create_project
    from enginir.viewer import _recent, _warmup

    path = tmp_path / "old.h5"
    create_project(path)
    imports = threading.Event()  # the warm-up's background imports run
    monkeypatch.setattr(_warmup, "_LIBRARY_IMPORTED", imports)
    result: dict = {}
    entry = _recent.RecentEntry(str(path), None)
    checker = threading.Thread(target=lambda: result.update(_recent.check([entry])),
                               daemon=True)
    checker.start()
    checker.join(0.3)
    assert result == {}, "checked while the warm-up imports"
    imports.set()
    checker.join(5)
    assert result == {str(path): (True, "project")}
