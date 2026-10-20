"""Smoke tests for WelcomeWindow launcher behavior."""

from __future__ import annotations

import os
import sys

import pytest

from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


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
    dataset = str(tmp_path / "sample.eiff")
    dark = str(tmp_path / "dark.eiff")
    white = str(tmp_path / "white.eiff")
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getOpenFileName", lambda *args, **kwargs: (dataset, "")
    )
    welcome._dark_path = dark
    welcome._white_path = white
    welcome._cb_dark.setChecked(True)
    welcome._cb_white.setChecked(True)
    received: list[tuple[str, str | None, str | None]] = []
    welcome.file_selected.connect(lambda path, d, w: received.append((path, d, w)))

    welcome._btn_open.click()

    assert received == [(dataset, dark, white)]


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
    items = [recent_list.item(i).text() for i in range(recent_list.count())]
    assert any("sample.eiff" in item for item in items)
    assert any("sample_project.h5" in item for item in items)
    assert not any("ghost.eiff" in item for item in items)

    dataset_received: list[tuple[str, str | None, str | None]] = []
    project_received: list[str] = []
    window.file_selected.connect(lambda path, d, w: dataset_received.append((path, d, w)))
    window.project_selected.connect(project_received.append)

    recent_list.itemActivated.emit(recent_list.item(0))
    recent_list.itemActivated.emit(recent_list.item(1))

    assert dataset_received or project_received
