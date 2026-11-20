"""Run one check over a table of cases and report every failing case at once.

A table of inputs that all prove one rule is one test, not one test per row.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def check_all(cases: Iterable[Any], check: Callable[[Any], None]) -> None:
    """Call ``check(case)`` for each case; fail once, naming every case that failed."""
    failures: list[str] = []
    for case in cases:
        try:
            check(case)
        except BaseException as exc:  # noqa: BLE001 - the case's own failure, reported below
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failures.append(f"{case!r}: {type(exc).__name__}: {exc}"[:300])
    assert not failures, f"{len(failures)} case(s) failed:\n" + "\n".join(failures)
