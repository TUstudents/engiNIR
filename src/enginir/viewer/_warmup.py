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

A pick before the warm-up ends is not delayed by it: an import the warm-up has
started makes the other thread wait for that module only (Python's import
lock), never for the rest.
"""

from __future__ import annotations

import importlib
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


def _import_all(names: tuple[str, ...]) -> None:
    for name in names:
        importlib.import_module(name)


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
        logger.info("Startup timing: warm-up {} in {:.3f}s", reason, time.perf_counter() - started)
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
    start_background_job(owner=owner, fn=lambda: _import_all(BACKGROUND_MODULES),
                         on_success=background_done, on_error=background_failed,
                         on_finished=lambda: None)


__all__ = ["BACKGROUND_MODULES", "UI_MODULES", "start_warmup"]
