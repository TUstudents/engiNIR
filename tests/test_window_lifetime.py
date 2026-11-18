"""Viewer windows are deleted when they close (viewer-workspace-ui contract, Window lifetime).

A closed window is deleted by Qt at a safe point of the event loop, keeps no
data, receives no project refresh, and nothing calls it afterwards
(plan ``plans/window-lifetime.md``).
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from enginir import Project
from enginir.io import save_hdf5


def _deleted_later(qapp) -> None:
    """Run Qt's pending deletions and Python's collector, as after an event-loop turn."""
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QApplication

    qapp.processEvents()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    gc.collect()


def _scan(path, value=1500) -> str:
    rng = np.random.default_rng(value)
    save_hdf5(str(path), rng.integers(value, value + 500, (8, 4, 6)).astype(np.uint16),
              np.linspace(1000.0, 1600.0, 6))
    return str(path)


@pytest.fixture()
def project_path(tmp_path):
    """A project with one dataset and one ROI with a value."""
    project = Project.create(tmp_path / "project.h5")
    project.add_target("API", unit="%")
    project.add_dataset(_scan(tmp_path / "scan.h5")).add_roi(
        "A1", frames=(0, 2), rows=(0, 1), values={"API": 2.0})
    return project.path


def test_a_closed_window_is_deleted(qapp, project_path):
    from PyQt6 import sip

    from enginir.viewer import open_project

    window = open_project(project_path)
    window.close()
    _deleted_later(qapp)
    assert sip.isdeleted(window)


def test_a_closed_window_releases_its_data_while_its_opener_stays_open(qapp, tmp_path):
    """Open Data... in a window without a project opens a second window; closing it frees the cube."""
    from PyQt6 import sip
    from PyQt6.QtWidgets import QApplication

    from enginir.viewer import open_viewer
    from enginir.viewer._viewer import HSIViewer

    opener = open_viewer(_scan(tmp_path / "first.h5"))
    before = [w for w in QApplication.topLevelWidgets() if isinstance(w, HSIViewer)]
    opener._open_data_path(_scan(tmp_path / "second.h5", 2500))
    (opened,) = [w for w in QApplication.topLevelWidgets()
                 if isinstance(w, HSIViewer) and not any(w is b for b in before)]
    cube = weakref.ref(opened.state.cube)
    opened.close()
    _deleted_later(qapp)
    assert cube() is None, "the closed window still holds its cube"
    assert not sip.isdeleted(opener) and opener.state.cube.shape == (8, 4, 6)
    opener.close()


def test_a_closed_window_is_not_refreshed_by_writes_in_another(qapp, project_path, monkeypatch):
    from enginir.viewer import _state, open_project

    reloads: list[object] = []
    original = _state.ProjectState.reload

    def counted(self, *args, **kwargs):
        reloads.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_state.ProjectState, "reload", counted)
    writer, closed = open_project(project_path), open_project(project_path)
    closed_state = closed.project_state
    closed.close()
    _deleted_later(qapp)
    assert closed_state not in _state._OPEN_STATES
    api = next(iter(writer.project_state.targets))
    roi = next(iter(writer.project_state.rois))
    writer.project_state.set_roi_values([(roi, api, 3.0)])
    _deleted_later(qapp)
    assert closed_state not in reloads
    writer.close()


def test_the_welcome_window_is_deleted_when_it_closes(qapp):
    from PyQt6 import sip

    from enginir.viewer._welcome import WelcomeWindow

    welcome = WelcomeWindow()
    welcome.show()
    welcome.close()
    _deleted_later(qapp)
    assert sip.isdeleted(welcome)


def test_a_job_that_ends_after_its_window_closed_calls_nothing(qapp, project_path):
    """The job's thread finishes; none of its callbacks reaches the deleted window."""
    import threading

    from PyQt6 import sip

    from enginir.viewer import open_project
    from enginir.viewer._jobs import start_background_job, wait_for_running_jobs

    window = open_project(project_path)
    release = threading.Event()
    called: list[str] = []
    start_background_job(
        owner=window, fn=lambda: release.wait(5) and "done",
        on_success=lambda result: called.append(f"success {window.windowTitle()}"),
        on_error=lambda message: called.append(f"error {window.windowTitle()}"),
        on_finished=lambda: called.append(f"finished {window.windowTitle()}"))
    window.close()
    _deleted_later(qapp)
    assert sip.isdeleted(window)
    release.set()
    assert wait_for_running_jobs(timeout=5.0)
    _deleted_later(qapp)
    assert called == []
