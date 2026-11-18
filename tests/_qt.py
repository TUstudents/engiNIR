"""Qt helpers shared by the viewer tests."""

from __future__ import annotations

import time
from collections.abc import Callable


def wait_until(qapp, predicate: Callable[[], object], timeout: float = 20.0) -> None:
    """Process events until ``predicate()`` is true; fail after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")
