"""Load the library and the viewer in the background, in two stages.

The launcher shows the welcome window without the scientific stack. Then:

1. **window stage**: a background job imports what the main window shows with (NumPy,
   h5py, matplotlib's core, engiNIR's light modules); then the Qt modules (pyqtgraph,
   matplotlib's Qt backend, the main window) are imported on the UI thread, one per
   idle-timer step, because Qt modules are only safe to import on the main thread. A
   pick on the welcome window waits for this stage only (:func:`wait_for_library`).
2. **modeling stage**: a background job imports SciPy, scikit-learn and engiNIR's
   fitting, prediction and preprocessing code, while the main window is already shown
   (``plans/fast-start.md``). What needs them runs through
   :func:`when_modeling_ready` (UI thread) or waits for them
   (:func:`wait_for_modeling`, other threads).

Two threads importing the same packages at once can deadlock Python's import locks (a
module's lock taken before its package's in one thread, the other way round in the
other; seen on Python 3.13.18 and 3.14, 2026-11-19). So from the start of the warm-up
until the modeling stage has ended, an import of SciPy, scikit-learn or engiNIR's
modeling code from any other thread waits for the stage (:class:`_ModelingGate`).
From the start, not from the stage's: a worker that began such an import before the
stage would hold its locks while the stage loads, and the first import it waits for
would never come (Windows log of 2026-11-23). The UI thread must not wait (the window
freezes): every such path goes through :func:`when_modeling_ready`; one that does not
is logged as a warning.

Each step is logged with its time and what it loaded (modules, how many of them
native, their size), so a cold start's log shows where the time goes
(``plans/cold-start.md``, phase 2).
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, QTimer

from .._logging import get_logger

logger = get_logger(__name__)

#: Window stage, imported by a background job (no Qt objects at import time), as named
#: steps in dependency order, so that each step's time is its own.
BACKGROUND_STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("numpy", ("numpy",)),
    ("h5py", ("h5py",)),
    ("matplotlib", ("matplotlib.figure", "matplotlib.backends.backend_agg")),
    ("engiNIR", ("enginir._workflow", "enginir.io", "enginir.prediction._service",
                 "enginir.prediction._rois", "enginir._guidance",
                 "enginir.modeling._roi_spectra")),
)
#: Window stage, imported on the UI thread, one per step, after the background job.
UI_MODULES = (
    "pyqtgraph",
    "matplotlib.backends.backend_qtagg",
    "enginir.viewer._open",
    "enginir.viewer._viewer",
)
#: Modeling stage, imported by a background job while the main window is shown.
#: ``scipy.stats`` loads most of SciPy; ``sklearn.base`` loads the rest of what
#: scikit-learn needs from it.
MODELING_STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scipy", ("scipy.stats", "scipy.signal", "scipy.spatial.distance")),
    ("scikit-learn", ("sklearn.base", "sklearn.model_selection", "sklearn.pipeline")),
    ("engiNIR modeling", (
        "enginir.preprocessing.scatter", "enginir.preprocessing.baseline",
        "enginir.preprocessing.derivatives", "enginir.preprocessing.selection",
        "enginir.models.pls", "enginir.models.pcr", "enginir.validation.cross_val",
        "enginir.validation.components", "enginir.validation.outliers",
        "enginir.modeling._service", "enginir.modeling._bands")),
)

#: Imports that wait while the modeling stage runs: the top-level packages, and every
#: engiNIR module that imports them when it is imported. All of them, not only the ones
#: the stage names: a thread that imported one of the others would hold that module's
#: import lock while it waits, and the stage would wait for that lock (a deadlock until
#: the timeout). ``tests/test_startup.py`` recomputes the list from the source.
_MODELING_PACKAGES = ("scipy", "sklearn", "joblib", "threadpoolctl")
_MODELING_MODULES = frozenset({
    "enginir._estimator",
    "enginir.modeling._bands", "enginir.modeling._components", "enginir.modeling._exclusion",
    "enginir.modeling._gram_fit", "enginir.modeling._service",
    "enginir.models.classification", "enginir.models.ml", "enginir.models.pcr",
    "enginir.models.pls",
    "enginir.preprocessing.baseline", "enginir.preprocessing.derivatives",
    "enginir.preprocessing.scatter", "enginir.preprocessing.selection",
    "enginir.validation.cross_val",
})


def _timed_import(step: str, names: tuple[str, ...]) -> None:
    """Import ``names`` and log the time and what they loaded."""
    before = set(sys.modules.copy())
    started = time.perf_counter()
    for name in names:
        importlib.import_module(name)
    elapsed = time.perf_counter() - started
    added = [sys.modules.get(name) for name in set(sys.modules.copy()) - before]
    native = [path for path in (getattr(module, "__file__", None) for module in added)
              if path and path.endswith((".pyd", ".so"))]
    size = sum(os.path.getsize(path) for path in native if os.path.exists(path))
    logger.info("Startup timing: warm-up imported {} in {:.2f}s ({} modules, {} native, "
                "{:.0f} MB native)", step, elapsed, len(added), len(native), size / 1e6)


#: Set when the window stage's background imports have ended (or failed).
_LIBRARY_IMPORTED = threading.Event()
_LIBRARY_IMPORTED.set()
#: Set when the modeling stage has ended (or failed); unset while it runs.
_MODELING_IMPORTED = threading.Event()
_MODELING_IMPORTED.set()


def _import_all(steps: tuple[tuple[str, tuple[str, ...]], ...],
                done: threading.Event | None = None) -> None:
    try:
        for step, names in steps:
            _timed_import(step, names)
    finally:
        (_LIBRARY_IMPORTED if done is None else done).set()


def wait_for_library(timeout: float = 120.0) -> None:
    """Wait until the window stage's background imports have ended (at once if none runs).

    Call it from any thread before code that imports the library (``_open``,
    the project store), so no two threads import the same packages at once.
    """
    if not _LIBRARY_IMPORTED.is_set():
        started = time.perf_counter()
        if not _LIBRARY_IMPORTED.wait(timeout):
            logger.warning("The warm-up's imports did not end within {:.0f}s", timeout)
        logger.debug("Waited {:.3f}s for the warm-up's imports", time.perf_counter() - started)


def modeling_ready() -> bool:
    """``True`` unless the modeling stage is still loading SciPy and scikit-learn."""
    return _MODELING_IMPORTED.is_set()


def wait_for_modeling(timeout: float = 300.0) -> None:
    """Wait (on a worker thread) until the modeling stage has ended; at once if none runs."""
    if not _MODELING_IMPORTED.is_set():
        started = time.perf_counter()
        if not _MODELING_IMPORTED.wait(timeout):
            logger.warning("The modeling libraries did not load within {:.0f}s", timeout)
        logger.debug("Waited {:.3f}s for the modeling libraries", time.perf_counter() - started)


#: Shown in a page's busy line while an action waits for the modeling libraries.
LOADING_TEXT = "Loading the modeling libraries..."

#: Callbacks waiting for the modeling stage, run on the UI thread when it ends, with
#: the object whose deletion cancels them.
_WHEN_READY: list[tuple[Callable[[], None], QObject | None]] = []


def when_modeling_ready(callback: Callable[[], None], owner: QObject | None = None) -> bool:
    """Run ``callback`` on the UI thread now, or when the modeling libraries are loaded.

    Fit, Predict and Propose bands go through it (decision D2 of ``plans/fast-start.md``):
    the click is accepted and the action starts by itself when loading is done; the page
    shows :data:`LOADING_TEXT` meanwhile. Nothing runs if ``owner`` is deleted before.
    Returns ``True`` if ``callback`` ran now.
    """
    if modeling_ready():
        callback()
        return True
    if (callback, owner) not in _WHEN_READY:  # a repeated request runs once
        logger.info("Waiting for the modeling libraries before: {}",
                    getattr(callback, "__qualname__", "an action"))
        _WHEN_READY.append((callback, owner))
    return False


def _run_waiting() -> None:
    from PyQt6 import sip

    waiting = list(_WHEN_READY)
    _WHEN_READY.clear()
    for callback, owner in waiting:
        if owner is not None and sip.isdeleted(owner):
            continue
        try:
            callback()
        except Exception as exc:  # one failing action must not drop the others
            logger.exception("An action waiting for the modeling libraries failed: {}", exc)


def _loaded(name: str) -> bool:
    """``True`` if ``name`` is imported completely (not missing, not being initialized)."""
    module = sys.modules.get(name)
    spec = getattr(module, "__spec__", None)
    return module is not None and not getattr(spec, "_initializing", False)


class _ModelingGate:
    """From the start of the warm-up until the modeling stage has ended, an import of its
    packages from another thread waits for it, so no two threads import them at once (see
    the module docstring).

    Installed as ``builtins.__import__`` by :func:`_hold_modeling_imports` and removed
    when the stage ends. Before the stage has started, the UI thread does not wait (it
    starts the stage, so the wait would not end); nothing else imports then.
    """

    def __init__(self) -> None:
        self.loader_ident: int | None = None
        self.started = False  # set on the UI thread when the stage's job is started
        self.original = builtins.__import__

    def _is_modeling(self, name: str) -> bool:
        if name.split(".", 1)[0] in _MODELING_PACKAGES:
            return True
        return name in _MODELING_MODULES

    def __call__(self, name: str, globals: dict[str, Any] | None = None,
                 locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if (not _MODELING_IMPORTED.is_set()
                and threading.get_ident() != self.loader_ident):
            absolute = name
            if level:
                package = (globals or {}).get("__package__") or ""
                absolute = importlib.util.resolve_name("." * level + name, package)
            wanted = [absolute] + [f"{absolute}.{item}" for item in (fromlist or ())
                                   if isinstance(item, str) and item != "*"]
            if any(self._is_modeling(item) and not _loaded(item) for item in wanted):
                started = time.perf_counter()
                on_ui = threading.current_thread() is threading.main_thread()
                if on_ui and not self.started:
                    logger.warning("The UI thread imports {} before the modeling libraries "
                                   "load", absolute)
                    return self.original(name, globals, locals, fromlist, level)
                if on_ui:
                    logger.warning("The UI thread imports {} while the modeling libraries "
                                   "load; it waits for them", absolute)
                _MODELING_IMPORTED.wait(300.0)
                logger.debug("Import of {} waited {:.3f}s for the modeling libraries",
                             absolute, time.perf_counter() - started)
        return self.original(name, globals, locals, fromlist, level)


_GATE: _ModelingGate | None = None


def _hold_modeling_imports() -> _ModelingGate:
    """Install the gate (if it is not installed): modeling imports wait for the stage."""
    global _GATE
    if _GATE is None:
        _GATE = _ModelingGate()
        _MODELING_IMPORTED.clear()
        builtins.__import__ = _GATE
    return _GATE


def _release_modeling_imports(gate: _ModelingGate) -> None:
    global _GATE
    if builtins.__import__ is gate:
        builtins.__import__ = gate.original
    if _GATE is gate:
        _GATE = None


#: Seconds after which a modeling stage that has not ended logs where every thread is.
STALL_REPORT_AFTER = 90.0


def _report_stall(started: float) -> None:
    """Log the stack of every thread: the modeling stage has not ended in time."""
    import traceback

    names = {thread.ident: thread.name for thread in threading.enumerate()}
    stacks = "\n".join(f"Thread {names.get(ident, ident)}:\n" + "".join(
        traceback.format_stack(frame)) for ident, frame in sys._current_frames().items())
    logger.warning("The modeling libraries have not loaded after {:.0f}s; the threads are "
                   "at:\n{}", time.perf_counter() - started, stacks)


def start_modeling_warmup(owner: QObject | None = None) -> None:
    """Load the modeling libraries in the background (the modeling stage).

    ``owner`` keeps the job's callbacks; by default the application, so the stage
    survives the welcome window. Callbacks waiting in :func:`when_modeling_ready` run
    on the UI thread when it ends.
    """
    if _GATE is not None and _GATE.started:
        return  # already running
    from PyQt6.QtWidgets import QApplication

    from ._jobs import start_background_job

    owner = owner if owner is not None else QApplication.instance()
    if owner is None:
        logger.warning("No application to own the modeling warm-up; loading on use")
        if _GATE is not None:
            _release_modeling_imports(_GATE)
        _MODELING_IMPORTED.set()
        return
    started = time.perf_counter()
    watchdog = threading.Timer(STALL_REPORT_AFTER, _report_stall, args=(started,))
    watchdog.daemon = True
    gate = _hold_modeling_imports()
    gate.started = True

    def load() -> None:
        gate.loader_ident = threading.get_ident()
        try:
            _import_all(MODELING_STEPS, _MODELING_IMPORTED)
        finally:
            _release_modeling_imports(gate)

    def ended(reason: str) -> None:
        watchdog.cancel()
        from ._startup_clock import since_start

        logger.info("Startup timing: modeling libraries {} {:.2f}s after the process "
                    "started ({:.3f}s)", reason, since_start(), time.perf_counter() - started)
        _run_waiting()

    def failed(message: str) -> None:
        logger.warning("Loading the modeling libraries failed: {}", message)
        ended("stopped")

    # The gate is installed before the thread starts, the loader's ident set first thing
    # in it; until then the gate lets the loader's own imports through only by ident,
    # so the first statement of ``load`` imports nothing.
    start_background_job(owner=owner, fn=load, on_success=lambda _result: ended("ready"),
                         on_error=failed, on_finished=lambda: None)
    watchdog.start()


def start_warmup(owner: QObject, *, on_done: Callable[[], None] | None = None) -> None:
    """Warm up the imports. The modeling stage follows the window stage by itself; from
    now until it has ended, modeling imports from other threads wait for it.

    The window stage's job and timer belong to the application (``owner``, the welcome
    window, if there is none), so that the stage ends, and the modeling stage starts, even
    when the welcome window is gone. ``on_done`` is called on the UI thread when the
    window stage is imported (or at its first failure, which is logged; opening a file
    then imports what is left).
    """
    from PyQt6.QtWidgets import QApplication

    from ._jobs import start_background_job

    app = QApplication.instance()
    owner = app if app is not None else owner
    _hold_modeling_imports()
    started = time.perf_counter()
    pending = list(UI_MODULES)
    timer = QTimer(owner)
    timer.setInterval(0)

    def finish(reason: str) -> None:
        timer.stop()
        from ._startup_clock import since_start

        logger.info("Startup timing: warm-up {} {:.2f}s after the process started ({:.3f}s)",
                    reason, since_start(), time.perf_counter() - started)
        start_modeling_warmup()
        if on_done is not None:
            on_done()

    def step() -> None:
        if not pending:
            finish("done")
            return
        name = pending.pop(0)
        try:
            _timed_import(name, (name,))
        except Exception as exc:  # a warm-up must never take the launcher down
            logger.warning("Warm-up import of {} failed: {}", name, exc)
            finish("stopped")

    def background_done(_result: object) -> None:
        logger.debug("Warm-up: library imported in {:.3f}s", time.perf_counter() - started)
        timer.start()

    def background_failed(message: str) -> None:
        logger.warning("Warm-up of the library failed: {}", message)
        finish("stopped")

    timer.timeout.connect(step)
    _LIBRARY_IMPORTED.clear()
    start_background_job(owner=owner, fn=lambda: _import_all(BACKGROUND_STEPS),
                         on_success=background_done, on_error=background_failed,
                         on_finished=lambda: None)


__all__ = ["BACKGROUND_STEPS", "LOADING_TEXT", "MODELING_STEPS", "UI_MODULES",
           "modeling_ready", "start_modeling_warmup", "start_warmup", "wait_for_library",
           "wait_for_modeling", "when_modeling_ready"]
