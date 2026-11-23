"""A fast start: what the launcher loads before the welcome window."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap

import pytest

#: Not loaded before the welcome window is shown: the scientific stack and the parts of
#: engiNIR that need it, Loguru (with asyncio and multiprocessing) and importlib.metadata
#: (plans/fast-start.md, phase 1). Each would cost time on a cold Windows start.
HEAVY = ("numpy", "sklearn", "scipy", "pandas", "h5py", "matplotlib", "pyqtgraph",
         "enginir._workflow", "enginir.io", "enginir.project", "enginir.models",
         "enginir.modeling", "enginir.prediction", "enginir.validation", "enginir.viewer._viewer",
         "loguru", "asyncio", "multiprocessing", "importlib.metadata")
#: Not loaded for the main window: SciPy and scikit-learn load after it is shown.
MODELING = ("scipy", "sklearn")

_LAUNCHER = textwrap.dedent("""
    import json, sys
    import enginir.viewer.__main__
    from enginir._logging import defer_logging
    from enginir.viewer._app import ensure_app
    defer_logging()  # as the launcher: the log sink is added after the welcome window
    ensure_app()
    from enginir.viewer._welcome import WelcomeWindow
    from PyQt6.QtWidgets import QApplication
    window = WelcomeWindow()
    window.show()
    QApplication.processEvents()
    print(json.dumps(sorted(sys.modules)))
""")


def _run(args: list[str], *, timeout: float, env: dict[str, str] | None = None):
    """Run a Python subprocess; on failure, the assertion shows its stderr (CI logs)."""
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, f"exit {result.returncode}:\n{result.stderr[-4000:]}"
    return result


def _modules_at_welcome(tmp_path) -> set[str]:
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "HOME": str(home),
           "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
           "APPDATA": str(home / "AppData"), "ENGINIR_LOG_FILE": str(tmp_path / "viewer.log")}
    result = _run([sys.executable, "-c", _LAUNCHER], env=env, timeout=60)
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


_WITHOUT_PANDAS = textwrap.dedent("""
    import sys
    class NoPandas:  # as if pandas were not installed
        def find_spec(self, name, path=None, target=None):
            if name == "pandas" or name.startswith("pandas."):
                raise ModuleNotFoundError(f"No module named {name!r}")
    sys.meta_path.insert(0, NoPandas())
    import numpy as np
    from enginir.viewer._warmup import BACKGROUND_STEPS, _import_all
    _import_all(BACKGROUND_STEPS)
    from enginir.io import load_csv, save_csv
    from enginir.validation import metrics_report
    save_csv(sys.argv[1], np.array([[1.5, 2.0]]), np.array([1000.0, 1001.0]), ["a"])
    assert load_csv(sys.argv[1]).X.tolist() == [[1.5, 2.0]]
    try:
        metrics_report(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.0, 2.9]), as_dataframe=True)
    except ImportError as exc:
        print(exc)
""")


def test_enginir_works_without_pandas(tmp_path):
    """pandas is optional (plan cold-start, phase 1): scikit-learn imports it whenever it is
    installed, so a start without it opens fewer files. The warm-up, CSV files and metrics
    work without it; only a metrics DataFrame needs it, and says how to get it."""
    result = _run([sys.executable, "-c", _WITHOUT_PANDAS, str(tmp_path / "s.csv")], timeout=60)
    assert "enginir[pandas]" in result.stdout, result.stderr[-3000:]


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
    from enginir.viewer._warmup import modeling_ready
    while wait and not (done and modeling_ready()):
        QApplication.processEvents()
        time.sleep(0.005)
    before = set(sys.modules)
    launcher = enginir.viewer.__main__  # the launcher's own entry points
    viewer = launcher._open_loaded_project_viewer(launcher._load_project_startup(sys.argv[1]))
    QApplication.processEvents()
    print(json.dumps({"new": sorted(set(sys.modules) - before),
                      "opened": type(viewer).__name__, "done": bool(done)}))
    # Exit as the launcher does (run()): a warm-up import still running when Python
    # shuts down fails ("can't register atexit after shutdown").
    from enginir.viewer._jobs import wait_for_running_jobs
    wait_for_running_jobs(60)
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
    result = _run([sys.executable, "-c", _WARM_THEN_OPEN, str(path), mode], env=env, timeout=90)
    assert "Traceback" not in result.stderr, result.stderr[-4000:]  # e.g. an import at exit
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_after_the_warmup_opening_a_project_imports_nothing_heavy(tmp_path):
    """The warm-up (both stages) has loaded the library and the main window before the
    first pick."""
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
    """The age comes from the operating system, so it includes the time before Python
    reached engiNIR (here a sleep before the import)."""
    script = ("import time; time.sleep(2.2)\n"
              "import enginir; enginir.enable_logging('WARNING')\n"
              "from enginir.viewer._startup_clock import process_age; print(process_age())")
    result = _run([sys.executable, "-c", script], timeout=60)
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
        if "modeling libraries ready" in text or "stopped" in text:
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
    """The log states when the launcher ran, the welcome window was shown and the
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
    for line in ("the launcher ran", "welcome window shown", "warm-up done",
                 "modeling libraries ready"):
        found = [entry for entry in text.splitlines() if line in entry]
        assert found and "after the process started" in found[0], (line, text[-3000:])
    # Where a cold start's time goes: one line per warm-up step, with what it loaded.
    from enginir.viewer._warmup import BACKGROUND_STEPS, MODELING_STEPS, UI_MODULES

    imported = re.findall(r"warm-up imported (.+?) in [\d.]+s \((\d+) modules, (\d+) native",
                          text)
    assert [step for step, _, _ in imported] == [step for step, _ in BACKGROUND_STEPS] + list(
        UI_MODULES) + [step for step, _ in MODELING_STEPS], text[-3000:]
    # The main window's stage ends before SciPy loads (plans/fast-start.md).
    assert text.index("warm-up done") < text.index("warm-up imported scipy"), text[-3000:]
    assert dict((step, int(n)) for step, n, _ in imported)["scipy"] > 100



def _record_job_starts(monkeypatch):
    """Record each job start in the viewer: (owner, a window was being shown, its window
    was visible). The job starter is wrapped in every viewer module, ``show`` in the
    window classes."""
    import enginir.viewer._jobs as jobs
    from enginir.viewer._viewer import HSIViewer
    from enginir.viewer._welcome import WelcomeWindow

    started: list[tuple[str, bool, bool]] = []
    showing: list[bool] = []
    real = jobs.start_background_job

    def recording(**kwargs):
        owner = kwargs["owner"]
        started.append((type(owner).__name__, bool(showing), owner.window().isVisible()))
        return real(**kwargs)

    for name, module in list(sys.modules.items()):
        if (name == "enginir.viewer" or name.startswith("enginir.viewer.")) \
                and getattr(module, "start_background_job", None) is real:
            monkeypatch.setattr(module, "start_background_job", recording)
    for cls in (HSIViewer, WelcomeWindow):
        real_show = cls.show

        def show(self, _real=real_show):
            showing.append(True)
            try:
                _real(self)
            finally:
                showing.pop()

        monkeypatch.setattr(cls, "show", show)
    return started


def test_no_job_thread_starts_while_a_window_is_built_or_shown(qapp, tmp_path, monkeypatch):
    """A job thread that started during show() hung a project window on macOS
    (2026-11-19): the jobs of opening a project, of the welcome window and of the
    Open Recent menu start from the event loop, once their window is shown."""
    from enginir._example import create_example_project
    from enginir.viewer import _recent, open_project
    from enginir.viewer._welcome import WelcomeWindow
    from tests._qt import wait_until

    path = create_example_project(tmp_path)
    monkeypatch.setattr(_recent, "config_dir", lambda: tmp_path / "config")
    _recent.add(str(path), "project")
    started = _record_job_starts(monkeypatch)

    def early() -> list[tuple[str, bool, bool]]:
        return [job for job in started if job[1] or not job[2]]

    window = open_project(str(path))
    try:
        wait_until(qapp, lambda: not window.project_state.data_files_unknown())
        assert any(owner == "HSIViewer" for owner, _, _ in started), "the lookup ran"
        assert early() == [], f"started before or while the window was shown: {early()}"
        started.clear()
        window._menu_open_recent.aboutToShow.emit()
        assert started == [], "the Open Recent check started while the menu popped up"
        wait_until(qapp, lambda: started)
    finally:
        window.close()

    started.clear()
    welcome = WelcomeWindow()
    assert started == [], f"started while the welcome window was built: {started}"
    welcome.show()
    try:
        wait_until(qapp, lambda: welcome._recent_checked)
        assert started and early() == [], started
    finally:
        welcome.close()


@pytest.mark.parametrize("loader", ["_load_dataset_startup", "_load_project_startup",
                                    "create_project", "_load_example_startup"])
def test_a_welcome_pick_waits_for_the_warmup_before_loading(loader, monkeypatch, tmp_path):
    """Two threads importing the library at once can deadlock (2026-11-19): a pick on the
    welcome window loads nothing until the warm-up's imports have ended. The example
    project did not wait (found on Windows, 2026-11-22)."""
    import threading

    import enginir._example
    import enginir.io.project_store
    import enginir.viewer._open
    from enginir.viewer import __main__ as launcher
    from enginir.viewer import _recent, _warmup

    called: list[str] = []
    for module, name in ((enginir.viewer._open, "_load_dataset_startup"),
                         (enginir.viewer._open, "_load_project_startup"),
                         (enginir.io.project_store, "create_project"),
                         (enginir._example, "create_example_project")):
        monkeypatch.setattr(module, name, lambda *args, _name=name, **kwargs: called.append(_name))
    monkeypatch.setattr(_recent, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(_warmup, "_LIBRARY_IMPORTED", threading.Event())  # a warm-up runs
    args = () if loader == "_load_example_startup" else (str(tmp_path / "x.h5"),)
    thread = threading.Thread(target=getattr(launcher, loader), args=args, daemon=True)
    thread.start()
    thread.join(0.3)
    assert called == [], f"{loader} loaded during the warm-up: {called}"
    _warmup._LIBRARY_IMPORTED.set()
    thread.join(5)
    assert called, f"{loader} did not load after the warm-up"


_LIGHT = textwrap.dedent("""
    import json, sys
    for name in sys.argv[1:]:
        __import__(name)
    print(json.dumps(sorted(sys.modules)))
""")


def test_the_main_window_and_the_workflow_load_no_scipy_or_scikit_learn(tmp_path):
    """SciPy and scikit-learn load after the main window is shown (plans/fast-start.md):
    nothing the window or the project workflow imports pulls them in."""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    light = ["enginir.project", "enginir._workflow", "enginir._guidance", "enginir.io",
             "enginir.prediction", "enginir.preprocessing.recipe", "enginir.modeling._stored",
             "enginir.viewer._open", "enginir.viewer._viewer"]
    result = _run([sys.executable, "-c", _LIGHT, *light], env=env, timeout=60)
    modules = set(json.loads(result.stdout.strip().splitlines()[-1]))
    assert "enginir.viewer._viewer" in modules
    loaded = sorted(name for name in modules if name.split(".")[0] in MODELING)
    assert loaded == [], f"loaded with the main window: {loaded[:10]}"


_WINDOW_WHILE_LOADING = textwrap.dedent("""
    import json, sys, threading, time
    main = threading.main_thread()
    attempts = []

    class Watch:  # records where SciPy or scikit-learn would be imported, and refuses
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("scipy", "sklearn"):
                attempts.append([name, threading.current_thread() is main])
                raise ModuleNotFoundError(f"not yet: {name}")

    sys.meta_path.insert(0, Watch())
    from enginir.viewer import _warmup
    _warmup._MODELING_IMPORTED.clear()  # the modeling stage runs
    from enginir.viewer._app import ensure_app
    from enginir.viewer import open_project
    from PyQt6.QtWidgets import QApplication
    ensure_app()
    window = open_project(sys.argv[1])
    for step in ("data", "rois", "values", "model", "predict"):
        window.show_step(step)
        for _ in range(30):
            QApplication.processEvents()
            time.sleep(0.01)
    window.select_model(next(iter(window.project_state.model_runs)))
    from enginir.viewer._model_diagnostics import ModelDiagnosticsWidget
    for plots in window.findChildren(ModelDiagnosticsWidget):
        for page in plots.grid.page_names:  # every page of the model's plots is drawn
            plots.grid.set_current_page(page)
            for _ in range(10):
                QApplication.processEvents()
                time.sleep(0.01)
    print(json.dumps({"shown": window.isVisible(), "attempts": attempts}))
    from enginir.viewer._jobs import wait_for_running_jobs
    wait_for_running_jobs(30)
""")


def test_the_main_window_works_while_the_modeling_libraries_load(tmp_path):
    """A project with a model, a calibration set and a stored prediction opens, every
    workflow step and every page of the model's plots show, and the UI thread never imports SciPy or
    scikit-learn meanwhile (two threads importing them at once can deadlock).
    Worker threads may: the import gate makes them wait."""
    from tests._scans import every_stored_object_project

    path = every_stored_object_project(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "HOME": str(home),
           "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
           "APPDATA": str(home / "AppData"), "ENGINIR_LOG_FILE": str(tmp_path / "viewer.log")}
    result = _run([sys.executable, "-c", _WINDOW_WHILE_LOADING, str(path)], env=env, timeout=90)
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["shown"]
    on_ui = [name for name, ui in outcome["attempts"] if ui]
    assert on_ui == [], f"the UI thread imported while the libraries load: {on_ui[:5]}"


def test_while_the_modeling_libraries_load_another_thread_waits_to_import_them(
        tmp_path, monkeypatch):
    """The gate: an import of a modeling package from a thread other than the loader's
    waits until the stage has ended."""
    import builtins
    import threading

    from enginir.viewer import _warmup

    (tmp_path / "fake_modeling_package.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(_warmup, "_MODELING_PACKAGES",
                        (*_warmup._MODELING_PACKAGES, "fake_modeling_package"))
    event = threading.Event()
    monkeypatch.setattr(_warmup, "_MODELING_IMPORTED", event)
    gate = _warmup._ModelingGate()
    monkeypatch.setattr(builtins, "__import__", gate)
    imported = threading.Event()

    def import_it() -> None:
        import fake_modeling_package  # noqa: F401

        imported.set()

    thread = threading.Thread(target=import_it, daemon=True)
    thread.start()
    assert not imported.wait(0.3), "imported while the modeling libraries load"
    event.set()
    assert imported.wait(5), "not imported once they are loaded"
    sys.modules.pop("fake_modeling_package", None)


_DEFERRED_LOG = textwrap.dedent("""
    import sys, time
    from enginir._logging import defer_logging, enable_logging
    defer_logging()
    from enginir.viewer import _startup_clock  # an engiNIR module that logs
    _startup_clock.logger.info("kept until the sink exists")
    assert "loguru" not in sys.modules, "Loguru imported before the welcome window"
    time.sleep(1.2)
    enable_logging("ERROR", file=sys.argv[1])
    print("ok")
""")


def test_messages_before_the_log_sink_keep_their_time_and_place(tmp_path):
    """D1: the launcher's first messages are kept without Loguru and written with the
    sink, with their own time, module, function and line."""
    log = tmp_path / "viewer.log"
    result = _run([sys.executable, "-c", _DEFERRED_LOG, str(log)], timeout=60)
    assert result.stdout.strip().endswith("ok"), result.stderr[-2000:]
    (line,) = [entry for entry in log.read_text(encoding="utf-8").splitlines()
               if "kept until the sink exists" in entry]
    assert "enginir.viewer._startup_clock:<module>:6" in line, line
    import datetime

    stamp = datetime.datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f")
    written = datetime.datetime.fromtimestamp(log.stat().st_mtime)
    assert (written - stamp).total_seconds() > 1.0, "the time it was logged, not written"


def test_the_import_gate_knows_every_module_that_loads_scipy_or_scikit_learn():
    """Every engiNIR module that imports SciPy or scikit-learn when it is imported (itself
    or through another engiNIR module) waits at the gate while they load; one missing
    would hold its own import lock while waiting, and the loader could wait for it."""
    import ast
    import importlib.util
    from pathlib import Path

    import enginir
    from enginir.viewer._warmup import _MODELING_MODULES

    root = Path(enginir.__file__).parent
    imports: dict[str, set[str]] = {}
    for path in sorted(root.rglob("*.py")):
        parts = path.with_suffix("").relative_to(root.parent).parts
        module = ".".join(parts).removesuffix(".__init__")
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        targets: set[str] = set()
        for node in ast.parse(path.read_text(encoding="utf-8")).body:  # import time only
            if isinstance(node, ast.Import):
                targets |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                base = (importlib.util.resolve_name("." * node.level + (node.module or ""),
                                                    package) if node.level else node.module)
                targets |= {base or "", *(f"{base}.{alias.name}" for alias in node.names)}
        imports[module] = targets
    heavy = {module for module, targets in imports.items()
             if any(target.split(".")[0] in MODELING for target in targets)}
    while extra := {module for module, targets in imports.items()
                    if module not in heavy and targets & heavy}:
        heavy |= extra
    assert heavy <= _MODELING_MODULES, sorted(heavy - _MODELING_MODULES)


def test_a_stalled_modeling_stage_logs_where_every_thread_is(monkeypatch):
    """If loading the modeling libraries hangs (seen once on a cold Windows start), the
    log shows where, so the next report can tell."""
    import threading

    from enginir.viewer import _warmup

    stuck = threading.Event()
    worker = threading.Thread(target=stuck.wait, name="stuck-loader", daemon=True)
    worker.start()
    warnings: list[str] = []
    monkeypatch.setattr(_warmup.logger, "warning",
                        lambda message, *args: warnings.append(message.format(*args)))
    _warmup._report_stall(0.0)
    stuck.set()
    (text,) = warnings
    block = text[text.index("Thread stuck-loader:"):]
    assert "in wait" in block.split("Thread ", 2)[1], block[:800]
