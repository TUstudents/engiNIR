"""Timing benchmark of a fit on statistics (plans/kernel-pls.md).

Skipped unless ``ENGINIR_PERF=1``; run it with output shown::

    ENGINIR_PERF=1 QT_QPA_PLATFORM=offscreen uv run pytest benchmarks/test_kernel_pls_performance.py -s

24 ROIs of up to 4800 pixels (40 bands, 7 folds): the fit takes seconds, not minutes.
"""

from __future__ import annotations

import os
import time

import pytest

from enginir import PreprocessingRecipe
from tests._scans import kernel_project

pytestmark = pytest.mark.skipif(os.environ.get("ENGINIR_PERF") != "1",
                                reason="timing benchmark; set ENGINIR_PERF=1")


def test_a_fit_on_statistics_is_fast(tmp_path):
    project = kernel_project(tmp_path, n_rois=12, frames=(780, 820), seed=4)
    start = time.perf_counter()
    model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components="auto")
    seconds = time.perf_counter() - start
    print(f"fit on statistics: {seconds:.1f} s")
    assert model.metrics["n_pixels"] > 100_000
    assert seconds < 30, f"{seconds:.1f} s"
