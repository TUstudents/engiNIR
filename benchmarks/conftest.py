"""The timing benchmarks run with the isolation of the test suite.

Run them on demand, with output shown::

    ENGINIR_PERF=1 QT_QPA_PLATFORM=offscreen uv run pytest benchmarks -s
"""

from tests.conftest import *  # noqa: F403  (offscreen Qt, temporary config, autouse fixtures)
