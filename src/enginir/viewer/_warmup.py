"""Load the library and the viewer while the welcome screen waits for the user.

The launcher shows the welcome window without the scientific stack (plan
startup-speed-1.7.1, D2). While the user reads it, the warm-up imports what
opening a file needs (D3), so the first project or data file opens without
that wait:

1. a background job imports the non-Qt stack: the workflow (scikit-learn,
   SciPy, pandas), the readers (h5py) and matplotlib's core;
2. when it has ended, the Qt modules (pyqtgraph, matplotlib's Qt backend,
   the main window) are imported on the UI thread, one per idle-timer step,
   because Qt modules are only safe to import on the main thread.

A pick before the warm-up ends waits for its background imports
(:func:`wait_for_library`) before anything else imports the library: two
threads importing the same packages at once can deadlock Python's import locks
(a module's lock taken before its package's in one thread, the other way round
in the other; seen on Python 3.13.18 and 3.14, 2026-11-19). The wait costs
nothing, since those imports would have blocked the pick anyway.
"""

from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Callable

from PyQt6.QtCore import QObject, QTimer

from .._logging import get_logger

logger = get_logger(__name__)

#: Imported by the background job (no Qt objects at import time).
BACKGROUND_MODULES = (
    "enginir._workflow",
    "enginir.io",
    "enginir.modeling",
    "enginir.prediction",
    "matplotlib.figure",
    "matplotlib.backends.backend_agg",
)
#: Imported on the UI thread, one per step, after the background job.
UI_MODULES = (
    "pyqtgraph",
    "matplotlib.backends.backend_qtagg",
    "enginir.viewer._open",
    "enginir.viewer._viewer",
)


#: Set when the background imports have ended (or failed); unset while they run.
_LIBRARY_IMPORTED = threading.Event()
_LIBRARY_IMPORTED.set()


def _import_all(names: tuple[str, ...]) -> None:
    try:
        for name in names:
            importlib.import_module(name)
    finally:
        _LIBRARY_IMPORTED.set()


def wait_for_library(timeout: float = 120.0) -> None:
    """Wait until the warm-up's background imports have ended (at once if none runs).

    Call it from any thread before code that imports the library (``_open``,
    the project store), so no two threads import the same packages at once.
    """
    if not _LIBRARY_IMPORTED.is_set():
        started = time.perf_counter()
        if not _LIBRARY_IMPORTED.wait(timeout):
            logger.warning("The warm-up's imports did not end within {:.0f}s", timeout)
        logger.debug("Waited {:.3f}s for the warm-up's imports", time.perf_counter() - started)


def start_warmup(owner: QObject, *, on_done: Callable[[], None] | None = None) -> None:
    """Warm up the imports; ``owner`` (the welcome window) owns the job and the timer.

    ``on_done`` is called on the UI thread when everything is imported (or at the
    first failure, which is logged; opening a file then imports what is left).
    """
    from ._jobs import start_background_job

    started = time.perf_counter()
    pending = list(UI_MODULES)
    timer = QTimer(owner)
    timer.setInterval(0)

    def finish(reason: str) -> None:
        timer.stop()
        from ._startup_clock import since_start

        logger.info("Startup timing: warm-up {} {:.2f}s after the process started ({:.3f}s)",
                    reason, since_start(), time.perf_counter() - started)
        if on_done is not None:
            on_done()

    def step() -> None:
        if not pending:
            finish("done")
            return
        name = pending.pop(0)
        try:
            importlib.import_module(name)
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
    start_background_job(owner=owner, fn=lambda: _import_all(BACKGROUND_MODULES),
                         on_success=background_done, on_error=background_failed,
                         on_finished=lambda: None)


__all__ = ["BACKGROUND_MODULES", "UI_MODULES", "start_warmup", "wait_for_library"]
