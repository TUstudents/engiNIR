"""A fast start: what the launcher loads before the welcome window (plan startup-speed-1.7.1)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

#: Not loaded before the welcome window is shown (D2): the scientific stack and the parts of
#: engiNIR that need it. Each would cost seconds on a cold Windows start.
HEAVY = ("sklearn", "scipy", "pandas", "h5py", "matplotlib", "pyqtgraph",
         "enginir._workflow", "enginir.io", "enginir.project", "enginir.models",
         "enginir.modeling", "enginir.prediction", "enginir.validation", "enginir.viewer._viewer")

_LAUNCHER = textwrap.dedent("""
    import json, sys
    import enginir.viewer.__main__
    from enginir.viewer._app import enable_viewer_logging, ensure_app
    enable_viewer_logging()
    ensure_app()
    from enginir.viewer._welcome import WelcomeWindow
    from PyQt6.QtWidgets import QApplication
    window = WelcomeWindow()
    window.show()
    QApplication.processEvents()
    print(json.dumps(sorted(sys.modules)))
""")


def _modules_at_welcome(tmp_path) -> set[str]:
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "HOME": str(home),
           "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
           "APPDATA": str(home / "AppData"), "ENGINIR_LOG_FILE": str(tmp_path / "viewer.log")}
    result = subprocess.run([sys.executable, "-c", _LAUNCHER], env=env, capture_output=True,
                            text=True, timeout=60, check=True)
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_the_welcome_window_shows_without_the_scientific_stack(tmp_path):
    modules = _modules_at_welcome(tmp_path)
    assert "PyQt6.QtWidgets" in modules, "the launcher did reach the welcome window"
    loaded = [name for name in HEAVY if name in modules]
    assert loaded == [], f"loaded before the welcome window: {loaded}"


@pytest.mark.parametrize("package", ["enginir", "enginir.viewer"])
def test_every_public_name_still_resolves(package):
    """The names load on first use; each one in __all__ still exists."""
    import importlib

    module = importlib.import_module(package)
    for name in module.__all__:
        assert getattr(module, name) is not None, name
    with pytest.raises(AttributeError):
        module.__getattr__("no_such_name")


_WARM_THEN_OPEN = textwrap.dedent("""
    import json, sys, time
    import enginir.viewer.__main__
    from enginir.viewer._app import enable_viewer_logging, ensure_app
    enable_viewer_logging()
    ensure_app()
    from enginir.viewer._welcome import WelcomeWindow
    from enginir.viewer._warmup import start_warmup
    from PyQt6.QtWidgets import QApplication
    window = WelcomeWindow()
    window.show()
    QApplication.processEvents()
    done = []
    start_warmup(window, on_done=lambda: done.append(True))
    wait = sys.argv[2] == "wait"
    while wait and not done:
        QApplication.processEvents()
        time.sleep(0.005)
    before = set(sys.modules)
    from enginir.viewer import _open
    viewer = _open._open_loaded_project_viewer(_open._load_project_startup(sys.argv[1]))
    QApplication.processEvents()
    print(json.dumps({"new": sorted(set(sys.modules) - before),
                      "opened": type(viewer).__name__, "done": bool(done)}))
""")


def _open_after_warmup(tmp_path, mode):
    from enginir.io.project_store import create_project

    path = tmp_path / "empty.h5"
    create_project(path)
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "HOME": str(home),
           "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
           "APPDATA": str(home / "AppData"), "ENGINIR_LOG_FILE": str(tmp_path / "viewer.log")}
    result = subprocess.run([sys.executable, "-c", _WARM_THEN_OPEN, str(path), mode], env=env,
                            capture_output=True, text=True, timeout=90, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_after_the_warmup_opening_a_project_imports_nothing_heavy(tmp_path):
    """D3: the warm-up has loaded the library and the main window before the first pick."""
    outcome = _open_after_warmup(tmp_path, "wait")
    assert outcome["done"] and outcome["opened"] == "HSIViewer"
    heavy = [name for name in outcome["new"] if name.split(".")[0] in
             ("sklearn", "scipy", "pandas", "h5py", "matplotlib", "pyqtgraph")
             or name in HEAVY]
    assert heavy == [], f"imported only when the project opened: {heavy}"


def test_a_pick_before_the_warmup_ends_opens_the_project(tmp_path):
    outcome = _open_after_warmup(tmp_path, "now")
    assert outcome["opened"] == "HSIViewer"


_QUIT_DURING_WARMUP = textwrap.dedent("""
    import enginir.viewer.__main__
    from enginir.viewer._app import enable_viewer_logging, ensure_app, run
    enable_viewer_logging()
    ensure_app()
    from enginir.viewer._welcome import WelcomeWindow
    from enginir.viewer._warmup import start_warmup
    from PyQt6.QtCore import QTimer
    window = WelcomeWindow()
    window.show()
    start_warmup(window)
    QTimer.singleShot(50, window.close)  # the user closes the launcher at once
    run()
""")


def test_closing_during_the_warmup_exits_cleanly(tmp_path):
    """The warm-up's import job ends before the interpreter does (no crash, no hang)."""
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "HOME": str(home),
           "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
           "APPDATA": str(home / "AppData"), "ENGINIR_LOG_FILE": str(tmp_path / "viewer.log")}
    result = subprocess.run([sys.executable, "-c", _QUIT_DURING_WARMUP], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Traceback" not in result.stderr, result.stderr[-3000:]


def test_the_process_age_counts_from_the_process_start(tmp_path):
    """D7: the age comes from the operating system, so it includes the time before Python
    reached engiNIR (here a sleep before the import)."""
    script = ("import time; time.sleep(2.2)\n"
              "import enginir; enginir.enable_logging('WARNING')\n"
              "from enginir.viewer._startup_clock import process_age; print(process_age())")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=60, check=True)
    said = result.stdout.strip().splitlines()[-1]
    assert said != "None", result.stderr[-2000:]  # the reason is logged as a warning
    age = float(said)
    assert 2.0 <= age < 30.0, age  # macOS reads it from ps, to the second


_LAUNCH_UNTIL_WARM = textwrap.dedent("""
    import os, sys
    from enginir.viewer._app import ensure_app
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication
    app = ensure_app()
    log = os.environ["ENGINIR_LOG_FILE"]

    def close_when_warm():
        text = open(log, encoding="utf-8").read() if os.path.exists(log) else ""
        if "warm-up done" in text or "warm-up stopped" in text:
            QApplication.closeAllWindows()

    timer = QTimer()
    timer.timeout.connect(close_when_warm)
    timer.start(100)
    QTimer.singleShot(80_000, QApplication.closeAllWindows)
    import enginir.viewer.__main__ as launcher
    sys.exit(launcher._main([]))
""")


@pytest.mark.timeout(120)
def test_the_log_shows_the_start_from_the_process_start(tmp_path):
    """D7: the log states when the launcher ran, the welcome window was shown and the
    warm-up ended, each counted from the process start."""
    home = tmp_path / "home"
    home.mkdir()
    log = tmp_path / "viewer.log"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "HOME": str(home),
           "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
           "APPDATA": str(home / "AppData"), "ENGINIR_LOG_FILE": str(log)}
    result = subprocess.run([sys.executable, "-c", _LAUNCH_UNTIL_WARM], env=env,
                            capture_output=True, text=True, timeout=110)
    assert result.returncode == 0, result.stderr[-2000:]
    text = log.read_text(encoding="utf-8")
    for line in ("the launcher ran", "welcome window shown", "warm-up done"):
        found = [entry for entry in text.splitlines() if line in entry]
        assert found and "after the process started" in found[0], (line, text[-3000:])
