"""Save dialogs add the file extension the open dialogs filter on."""

from __future__ import annotations

import os

import pytest
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from enginir.viewer._app import ensure_app
from enginir.viewer._ui_helpers import PROJECT_SAVE_FILTER, ask_save_path


@pytest.fixture()
def typed(monkeypatch):
    """Make the save dialog return the name in ``typed.name``."""
    class Typed:
        name = ""
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (Typed.name, PROJECT_SAVE_FILTER)))
    return Typed


@pytest.mark.parametrize(("name", "expected"), [
    ("project", "project.h5"),
    ("project.h5", "project.h5"),
    ("project.HDF5", "project.HDF5"),
    ("run.2026", "run.2026.h5"),          # a dot in the name is not an extension
])
def test_the_project_extension_is_added_when_missing(tmp_path, typed, name, expected):
    typed.name = str(tmp_path / name)
    assert ask_save_path(None, "New Project", PROJECT_SAVE_FILTER, ".h5", (".hdf5",)) \
        == str(tmp_path / expected)


def test_replacing_a_file_reached_by_the_added_extension_is_confirmed(tmp_path, typed,
                                                                      monkeypatch):
    (tmp_path / "project.h5").write_bytes(b"old")
    typed.name = str(tmp_path / "project")
    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    asked: list[str] = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda _p, _t, text, *a, **k: asked.append(text) or answers.pop(0)))
    assert ask_save_path(None, "New Project", PROJECT_SAVE_FILTER, ".h5") is None
    assert ask_save_path(None, "New Project", PROJECT_SAVE_FILTER, ".h5") \
        == str(tmp_path / "project.h5")
    assert len(asked) == 2 and "project.h5" in asked[0]


def test_cancelling_returns_nothing(typed):
    typed.name = ""
    assert ask_save_path(None, "New Project", PROJECT_SAVE_FILTER, ".h5") is None


def test_a_new_project_from_explore_gets_the_extension(tmp_path, typed):
    import numpy as np

    from enginir.viewer._state import ProjectState, ViewerState
    from tests.test_eda_view import _make_view

    ensure_app()
    state = ViewerState(np.zeros((2, 2, 4), dtype=np.uint16), np.linspace(1000, 1300, 4))
    view, _canvas = _make_view(state, ProjectState(), str(tmp_path / "cube.eiff"),
                               dataset_attached=False)
    typed.name = str(tmp_path / "trial")
    view.new_project_btn_clicked()
    assert view.project_state.project_path == str(tmp_path / "trial.h5")
    assert (tmp_path / "trial.h5").exists() and not (tmp_path / "trial").exists()



@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="needs a folder the user cannot write to")
def test_a_project_that_cannot_be_written_is_reported_and_nothing_changes(
        tmp_path, typed, monkeypatch):
    import numpy as np

    from enginir.viewer._state import ProjectState, ViewerState
    from tests.test_eda_view import _make_view

    ensure_app()
    state = ViewerState(np.zeros((2, 2, 4), dtype=np.uint16), np.linspace(1000, 1300, 4))
    view, _canvas = _make_view(state, ProjectState(), str(tmp_path / "cube.eiff"),
                               dataset_attached=False)
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda _parent, title, text, *a: warnings.append(text)))
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        typed.name = str(locked / "trial.h5")
        view.new_project_btn_clicked()
    finally:
        locked.chmod(0o700)
    assert view.project_state.project_path is None
    assert len(warnings) == 1 and "Permission denied" in warnings[0]
