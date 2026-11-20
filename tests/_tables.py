"""Run one check over a table of cases and report every failing case at once.

A table of inputs that all prove one rule is one test, not one test per row.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def check_all(cases: Iterable[Any], check: Callable[[Any], None]) -> None:
    """Call ``check(case)`` for each case; fail once, naming every case that failed."""
    failures: list[str] = []
    checked = 0
    for case in cases:
        checked += 1
        try:
            check(case)
        except Exception as exc:  # noqa: BLE001 - the case's own failure, reported below
            if type(exc).__module__.startswith("_pytest.outcomes"):
                raise  # a skip or an exit inside a case is the test's, not a failed case
            failures.append(f"{case!r}: {type(exc).__name__}: {exc}"[:1000])
    assert checked, "no cases: the table is empty, so nothing was checked"
    assert not failures, f"{len(failures)} case(s) failed:\n" + "\n".join(failures)
