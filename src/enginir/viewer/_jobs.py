"""Reusable background-job helpers for responsive viewer service actions."""

from __future__ import annotations

import atexit
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QCoreApplication, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from .._logging import get_logger
from ..modeling import BandProposalCancelledError
from ..prediction import PredictionCancelledError
from ._theme import secondary_text_style

if TYPE_CHECKING:
    from ._loading import LoadingProgressEvent

logger = get_logger(__name__)

#: Raised when the user cancels a job: reported to the caller, logged as info, not as a failure.
_CANCELLED = (BandProposalCancelledError, PredictionCancelledError)


class BackgroundWorker(QObject):
    """Execute one callable in a worker thread and surface success/error."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._fn = fn

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self._fn()
        except _CANCELLED as exc:
            logger.info("Background viewer job canceled: {}", exc)
            self.failed.emit(str(exc))
        except Exception as exc:
            logger.exception("Background viewer job failed: {}", exc)
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()


class BusyStatusWidget(QWidget):
    """Compact status label plus indeterminate progress bar.

    The widget also emits ``status_changed(text, busy)`` so shell-level summary
    pages can mirror the same busy state without polling the owning workflow
    view.
    """

    #: Emitted on every status change. Args: (text, busy)
    status_changed = pyqtSignal(str, bool)

    def __init__(self, *, idle_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._idle_text = idle_text

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.label = QLabel(idle_text)
        self.label.setStyleSheet(secondary_text_style())
        layout.addWidget(self.label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(6)
        self.progress.hide()
        layout.addWidget(self.progress)

    def set_idle(self, text: str | None = None) -> None:
        """Show one idle/status message and hide the progress bar."""
        resolved_text = self._idle_text if text is None else text
        self.label.setText(resolved_text)
        self.progress.hide()
        self.status_changed.emit(resolved_text, False)

    def set_busy(self, text: str) -> None:
        """Show one busy message and an indeterminate progress bar."""
        self.label.setText(text)
        self.progress.setRange(0, 0)  # Qt's own busy animation
        self.progress.show()
        self.status_changed.emit(text, True)

    def set_progress(self, event: LoadingProgressEvent) -> None:
        """Show a progress message; the bar is determinate when the event has a total."""
        self.label.setText(event.message)
        if event.total > 0 and not event.indeterminate:
            self.progress.setRange(0, event.total)
            self.progress.setValue(max(0, min(event.completed, event.total)))
        else:
            self.progress.setRange(0, 0)
        self.progress.show()
        self.status_changed.emit(event.message, True)

    def snapshot(self) -> tuple[str, bool]:
        """Return the shown message and whether the busy indicator is visible."""
        return self.label.text(), not self.progress.isHidden()


class _CallbackRelay(QObject):
    """Ensure background job callbacks run on the owner's thread."""

    def __init__(
        self,
        *,
        on_success: Callable[[Any], None],
        on_error: Callable[[str], None],
        on_finished: Callable[[], None],
        parent: QObject,
    ) -> None:
        super().__init__(parent)
        self._on_success = on_success
        self._on_error = on_error
        self._on_finished = on_finished

    @pyqtSlot(object)
    def handle_success(self, result: Any) -> None:
        self._on_success(result)

    @pyqtSlot(str)
    def handle_error(self, message: str) -> None:
        self._on_error(message)

    @pyqtSlot()
    def handle_finished(self) -> None:
        self._on_finished()


# Jobs whose thread is still running. Holding them here guarantees that a
# QThread is never garbage-collected while it runs, which would make Qt abort
# the process (for example when the view that started the job is closed).
_RUNNING_JOBS: set[tuple[QThread, BackgroundWorker, QObject]] = set()


def start_background_job(
    *,
    owner: QObject,
    fn: Callable[[], Any],
    on_success: Callable[[Any], None],
    on_error: Callable[[str], None],
    on_finished: Callable[[], None],
) -> tuple[QThread, BackgroundWorker, QObject]:
    """Run ``fn`` in a new QThread and call the callbacks on ``owner``'s thread.

    ``on_success(result)`` or ``on_error(message)`` is called, then
    ``on_finished()``. The job is kept alive until its thread has finished,
    even if the caller drops the returned references.

    Returns
    -------
    tuple
        ``(thread, worker, relay)``.
    """
    relay = _CallbackRelay(
        on_success=on_success,
        on_error=on_error,
        on_finished=on_finished,
        parent=owner,
    )
    thread = QThread()
    worker = BackgroundWorker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.succeeded.connect(relay.handle_success)
    worker.failed.connect(relay.handle_error)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    job = (thread, worker, relay)
    _RUNNING_JOBS.add(job)
    thread.finished.connect(relay.handle_finished)
    thread.finished.connect(lambda: _RUNNING_JOBS.discard(job))
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return job


def wait_for_running_jobs(timeout: float = 10.0) -> bool:
    """Let running jobs finish (up to ``timeout`` seconds); ``True`` if none is left.

    Called before the interpreter exits: a job thread that still reads a project
    file while Python shuts down crashes or hangs the process (h5py, Qt). Events
    are processed while waiting, since a job's thread is told to quit through
    the event loop.
    """
    deadline = time.monotonic() + timeout
    while _RUNNING_JOBS and time.monotonic() < deadline:
        app = QCoreApplication.instance()
        if app is not None:
            app.processEvents()
        for thread, _worker, _relay in list(_RUNNING_JOBS):
            try:
                if thread.isFinished():
                    _RUNNING_JOBS.discard((thread, _worker, _relay))
            except RuntimeError:  # the C++ thread object is gone already
                _RUNNING_JOBS.discard((thread, _worker, _relay))
        time.sleep(0.01)
    if _RUNNING_JOBS:
        logger.warning("{} background job(s) still running at exit", len(_RUNNING_JOBS))
    return not _RUNNING_JOBS


atexit.register(wait_for_running_jobs)


__all__ = ["BackgroundWorker", "BusyStatusWidget", "start_background_job",
           "wait_for_running_jobs"]
