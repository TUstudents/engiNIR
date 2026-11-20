"""The viewer application: Qt platform and theme start-up, background jobs and a clean exit."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication
import pytest


# -- Tests for enginir.viewer._app -----------------------------------------------------


def test_ensure_app_does_not_restyle_an_app_that_has_the_theme(qapp, monkeypatch):
    """Opening another window must not restyle every live widget (it crashed Qt in tests)."""
    import enginir.viewer._app as appmod

    appmod.ensure_app()
    called: list[bool] = []
    monkeypatch.setattr(appmod, "apply_qt_theme", lambda app, *, dark: called.append(dark) or app)
    monkeypatch.setattr(appmod, "load_preference", lambda: bool(qapp.property(
        "_enginir_theme_dark")))
    assert appmod.ensure_app() is qapp
    assert called == []

    # An application created by someone else gets the engiNIR icon.
    from PyQt6.QtGui import QIcon

    qapp.setWindowIcon(QIcon())
    appmod.ensure_app()
    assert not qapp.windowIcon().isNull()


@pytest.mark.parametrize(
    ("explicit_platform", "wayland", "display", "wsl", "expected"),
    [
        (None, "wayland-0", ":0", None, "xcb"),
        ("offscreen", "wayland-0", ":0", None, None),
        (None, "wayland-0", None, None, None),
        (None, "wayland-0", ":0", "Ubuntu", "wayland"),     # WSLg: menus stay in front
        ("xcb", "wayland-0", ":0", "Ubuntu", None),         # an explicit choice wins
        (None, None, ":0", "Ubuntu", None),                 # WSL without WSLg
    ],
)
def test_recommended_qt_platform_linux_wayland_cases(
    monkeypatch, explicit_platform, wayland, display, wsl, expected
):
    import enginir.viewer._app as appmod

    if wsl is None:
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    else:
        monkeypatch.setenv("WSL_DISTRO_NAME", wsl)

    if explicit_platform is None:
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    else:
        monkeypatch.setenv("QT_QPA_PLATFORM", explicit_platform)
    if wayland is None:
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    else:
        monkeypatch.setenv("WAYLAND_DISPLAY", wayland)
    if display is None:
        monkeypatch.delenv("DISPLAY", raising=False)
    else:
        monkeypatch.setenv("DISPLAY", display)
    monkeypatch.setattr(appmod.sys, "platform", "linux")

    assert appmod._recommended_qt_platform() == expected


def test_wslg_uses_wayland_with_qts_title_bar(monkeypatch):
    import enginir.viewer._app as appmod

    for name in ("QT_QPA_PLATFORM", "QT_WAYLAND_DISABLE_WINDOWDECORATION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(appmod.sys, "platform", "linux")
    appmod._configure_qt_platform()
    assert os.environ["QT_QPA_PLATFORM"] == "wayland"
    # WSLg's own frame has no icon; Qt's title bar shows it.
    assert "QT_WAYLAND_DISABLE_WINDOWDECORATION" not in os.environ


def test_run_is_noop_in_jupyter(qapp, monkeypatch):
    import builtins
    from enginir.viewer import _app

    class FakeIP:
        def enable_gui(self, backend):
            pass

    started: list[str] = []
    monkeypatch.setattr(builtins, "get_ipython", lambda: FakeIP(), raising=False)
    monkeypatch.setattr(_app, "ensure_app", lambda: started.append("app"))
    _app.run()
    assert started == [], "the notebook's event loop runs Qt: no application, no exec()"


def test_the_launcher_reports_a_target_it_cannot_open_and_opens_a_project(
        qapp, monkeypatch, tmp_path, boxes):
    """Through the real opening code: a missing file is an error box and exit code 1; a
    project opens in a window and the launcher goes on to the event loop (exit code 0)."""

    from enginir import Project
    import enginir.viewer.__main__ as launcher

    shown = boxes.critical
    monkeypatch.setattr("enginir.viewer._app.run", lambda: None)
    missing = str(tmp_path / "missing.eiff")
    assert launcher._main([missing]) == 1
    assert shown and "missing.eiff" in shown[0]

    project = Project.create(tmp_path / "empty.h5")
    before = set(QApplication.topLevelWidgets())
    try:
        assert launcher._main(["--project", str(project.path)]) == 0
        opened = set(QApplication.topLevelWidgets()) - before
        assert any(widget.isVisible() for widget in opened), "a window shows the project"
    finally:
        for widget in set(QApplication.topLevelWidgets()) - before:
            widget.close()
    assert len(shown) == 1, "opening a project is not an error"


def test_combo_popups_release_their_surface_on_wslg_wayland(qapp, monkeypatch):
    """WSLg keeps a popup the app hid on screen (microsoft/wslg#1522); destroying the hidden
    popup's native window removes it. The combo must still open again afterwards."""
    import time

    from PyQt6.QtWidgets import QComboBox

    from enginir.viewer import _app

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert not _app._releases_combo_popups(qapp), "only with Qt's Wayland platform"

    released = []

    def spy(popup):
        released.append(popup)
        _app.release_popup_surface.__wrapped__(popup)

    spy.__wrapped__ = _app.release_popup_surface
    monkeypatch.setattr(_app, "release_popup_surface", spy)
    fix = _app._DialogCentering(release_combo_popups=True)
    qapp.installEventFilter(fix)
    combo = QComboBox()
    combo.addItems(["Reflectance", "Absorbance", "DN"])
    combo.show()
    try:
        for index in (1, 2):
            combo.showPopup()
            popup = combo.view().window()
            assert popup.isVisible(), "opens again after its surface was released"
            combo.setCurrentIndex(index)
            combo.hidePopup()
            deadline = time.monotonic() + 2
            while len(released) < index and time.monotonic() < deadline:
                qapp.processEvents()
            assert released[-1] is popup and not popup.isVisible()
        assert combo.currentText() == "DN"
        QComboBox().hide()  # an ordinary widget hiding is not a combo popup
        qapp.processEvents()
        assert len(released) == 2
    finally:
        qapp.removeEventFilter(fix)
        combo.close()


@pytest.mark.skipif(os.name != "nt", reason="the taskbar app id is a Windows feature")
def test_the_viewer_has_its_own_taskbar_identity_on_windows():
    """Without its own app id, Windows groups the viewer under python.exe and shows no
    engiNIR icon in the taskbar. Checked in a fresh process: the id must be set before
    the first window."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import ctypes
        from ctypes import wintypes
        from enginir.viewer._app import WINDOWS_APP_ID, ensure_app
        ensure_app()
        value = wintypes.LPWSTR()
        ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(value))
        print(value.value == WINDOWS_APP_ID, value.value)
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, timeout=60)
    assert result.stdout.startswith("True"), result.stdout + result.stderr


# -- Lifecycle of viewer background jobs (G17) -----------------------------------------


def _run(qapp, fn):
    from enginir.viewer import _jobs

    events: list[tuple[str, object, bool]] = []
    main = threading.get_ident()

    def record(kind):
        return lambda *args: events.append((kind, args[0] if args else None,
                                            threading.get_ident() == main))

    owner = QObject()
    job = _jobs.start_background_job(owner=owner, fn=fn, on_success=record("success"),
                                     on_error=record("error"), on_finished=record("finished"))
    del job  # the helper keeps the job alive by itself
    deadline = time.monotonic() + 5
    while (not events or events[-1][0] != "finished") and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    return events, _jobs._RUNNING_JOBS


# -- The viewer process exits cleanly after a project window was open ------------------


_SESSION = textwrap.dedent("""
    import sys
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication
    import enginir
    from enginir.viewer import open_project
    from enginir.viewer._app import ensure_app, run

    app = ensure_app()
    enginir.Project.create(sys.argv[1])
    open_project(sys.argv[1])
    QTimer.singleShot(300, lambda: [w.close() for w in QApplication.topLevelWidgets()])
    run()
""")


#: Each viewer process gets this long to start, close its window and exit.
_SESSION_TIMEOUT = 50


# Four processes in one test: on macOS runners each start takes 10-15 s, so the
# suite's 60 s per-test limit cut the test off in the middle of a healthy run. The
# limit covers the four timeouts; a process that hangs fails with its own
# TimeoutExpired (which run, and its output) instead of the suite's generic timeout.
@pytest.mark.timeout(4 * _SESSION_TIMEOUT + 30)
def test_closing_a_project_viewer_exits_without_a_crash(tmp_path):
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen",
           "ENGINIR_LOG_FILE": str(tmp_path / "viewer.log")}
    for run in range(4):
        result = subprocess.run(
            [sys.executable, "-c", _SESSION, str(tmp_path / f"p{run}.h5")],
            env=env, capture_output=True, text=True, timeout=_SESSION_TIMEOUT)
        assert result.returncode == 0, (
            f"run {run} ended with {result.returncode}:\n{result.stderr[-2000:]}")


# -- Jobs at exit ----------------------------------------------------------------------


