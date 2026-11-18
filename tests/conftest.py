"""Shared pytest configuration and fixtures."""

import os
import sys
import tempfile

import pytest
from loguru import logger

# Use offscreen Qt platform so tests pass in headless CI environments.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Viewer tests enable file logging; keep it out of the user's home directory.
os.environ.setdefault(
    "ENGINIR_LOG_FILE", os.path.join(tempfile.mkdtemp(prefix="enginir-tests-"), "enginir.log")
)
# The viewer stores recent files and the theme in the user's config directory;
# point it at a temporary one for the whole session (module-scoped Qt fixtures
# start before function fixtures), and per test in _isolated_user_config.
_SESSION_CONFIG = tempfile.mkdtemp(prefix="enginir-tests-config-")
os.environ["XDG_CONFIG_HOME"] = _SESSION_CONFIG
os.environ["APPDATA"] = _SESSION_CONFIG


@pytest.fixture(autouse=True)
def _isolated_user_config(tmp_path, monkeypatch):
    """Keep the viewer's settings out of the user's real config.

    HOME is redirected on every platform, so a test that unsets
    XDG_CONFIG_HOME still writes into ``tmp_path``.
    """
    config = tmp_path / "user_config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))  # Linux
    monkeypatch.setenv("APPDATA", str(config))          # Windows
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # macOS, and Linux without XDG


@pytest.fixture(autouse=True)
def _enginir_records_enabled():
    """Let tests observe enginir log records (the library is muted by default)."""
    logger.enable("enginir")
    yield


@pytest.fixture(autouse=True)
def _unexpected_message_boxes_fail(monkeypatch):
    """An unexpected modal dialog or message box fails the test instead of blocking it.

    Under the offscreen Qt platform a modal dialog waits forever, and the
    per-test timeout does not interrupt Qt's event loop. Tests that expect a
    dialog replace these functions themselves (their monkeypatch wins).
    """
    try:
        from PyQt6.QtWidgets import QDialog, QFileDialog, QMessageBox
    except ImportError:  # Qt-free environments
        return

    def unexpected(kind):
        def show(*args, **_kwargs):
            title = args[1] if len(args) > 1 else ""
            text = args[2] if len(args) > 2 else ""
            raise AssertionError(f"unexpected {kind}: {title!r}: {text!r}")
        return staticmethod(show)

    for kind in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, kind, unexpected(f"QMessageBox.{kind}"))
    for kind in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, kind, unexpected(f"QFileDialog.{kind}"))

    def unexpected_exec(dialog, *_args):
        raise AssertionError(f"unexpected modal dialog {type(dialog).__name__}: "
                             f"{dialog.windowTitle()!r}")

    # Dialog classes that a test drives replace ``exec`` on the subclass, which wins.
    monkeypatch.setattr(QDialog, "exec", unexpected_exec)


@pytest.fixture(autouse=True)
def _exceptions_in_qt_slots_fail():
    """An exception raised in a Qt slot fails the test (PyQt only prints it).

    PyQt passes such exceptions to ``sys.excepthook`` and carries on, so a
    broken signal handler would otherwise go unnoticed.
    """

    import traceback

    raised: list[str] = []
    original = sys.excepthook
    sys.excepthook = lambda kind, value, tb: raised.append(
        "".join(traceback.format_exception(kind, value, tb)))
    yield
    sys.excepthook = original
    if raised:
        pytest.fail(f"exception in a Qt slot:\n{raised[0]}")


@pytest.fixture()
def tray(tmp_path):
    """The prediction-ROI tray (``tests._scans.tray_project``): ``(project, dataset, path)``."""
    from tests._scans import tray_project

    return tray_project(tmp_path)


@pytest.fixture(scope="session")
def qapp():
    """The one QApplication of the run, with the engiNIR theme (``ensure_app``)."""
    from enginir.viewer._app import ensure_app

    return ensure_app()


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    """The example project (synthetic tablets with a fitted model), once per module."""
    from enginir._example import create_example_project

    return create_example_project(tmp_path_factory.mktemp("example"))


@pytest.fixture()
def window(qapp, example):
    """A window on :func:`example` at 1536 x 864, with its checks."""
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
    from enginir._example import create_example_project
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


@pytest.fixture(autouse=True)
def _windows_are_destroyed_after_the_test():
    """Destroy the windows a test opened, at a quiet point right after it.

    A closed window otherwise lives on until Python's garbage collector frees
    it at a random moment, possibly while Qt paints another window; CI runs
    crashed now and then that way (segmentation fault in ``QPainter::restore``).
    Only widgets that Python owns are destroyed: parentless widgets that Qt
    owns (pyqtgraph's menu widgets belong to a ``QWidgetAction``) are left to
    their owner, which deletes them itself.
    """
    import sys

    if "PyQt6.QtWidgets" not in sys.modules:
        yield
        return
    from PyQt6 import sip
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QApplication, QMenu

    app = QApplication.instance()
    before = {id(widget) for widget in app.topLevelWidgets()} if app is not None else set()
    yield
    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        if (id(widget) not in before and widget.parent() is None
                and not isinstance(widget, QMenu) and sip.ispyowned(widget)):
            widget.close()
            widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
