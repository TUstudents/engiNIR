"""Lifecycle of viewer background jobs (G17)."""

from __future__ import annotations

import threading
import time

import pytest
from PyQt6.QtCore import QObject

from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


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


def test_job_result_arrives_on_the_ui_thread_then_finishes(qapp):
    worker_thread: list[int] = []

    def work():
        worker_thread.append(threading.get_ident())
        return 42

    events, running = _run(qapp, work)

    assert worker_thread and worker_thread[0] != threading.get_ident()
    assert events == [("success", 42, True), ("finished", None, True)]
    assert not running


def test_job_failure_is_reported_once_with_its_message_and_finishes(qapp):
    def work():
        raise ValueError("bad sample set")

    events, running = _run(qapp, work)

    assert [kind for kind, _, _ in events] == ["error", "finished"]
    assert "bad sample set" in str(events[0][1])
    assert all(on_ui for _, _, on_ui in events)
    assert not running
