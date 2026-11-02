"""Timing benchmark for the Model diagnostics plots (plot-panel plan, phases 0 and 7).

Skipped unless ``ENGINIR_PERF=1``; run it with output shown::

    ENGINIR_PERF=1 QT_QPA_PLATFORM=offscreen uv run pytest tests/test_plot_performance.py -s

It measures real Qt widgets at 1400x900: building the diagnostics, redrawing
the visible plots for a new payload, and changing one plot. Each step
includes the event-loop pass that paints the canvas.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable

import numpy as np
import pytest

from enginir.viewer._app import ensure_app

pytestmark = pytest.mark.skipif(
    os.environ.get("ENGINIR_PERF") != "1", reason="timing benchmark; set ENGINIR_PERF=1"
)

_SIZE = (1400, 900)
_REPEATS = 15
# Phase 0 baseline (single-figure diagnostics, 2026-11-02, WSL2, 10 cores). The
# limits hold on that machine; on others, compare against a run of the baseline.
_BASELINE_FULL_MS = 187.0
_BASELINE_ONE_MS = 121.0
# Explore frame step with all plots in matplotlib (before RAW and Sensor Frame
# moved to pyqtgraph), same machine.
_BASELINE_SCRUB_MS = 73.0


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


def _payload(rng: np.random.Generator, *, n_rois: int = 60, n_bands: int = 224) -> dict:
    y_true = rng.normal(10.0, 2.0, n_rois)
    return {
        "y_true_roi": y_true,
        "y_pred_roi": y_true + rng.normal(0.0, 0.4, n_rois),
        "roi_ids": [f"roi-{i}" for i in range(n_rois)],
        "scores_roi": rng.normal(size=(n_rois, 6)),
        "target_unit": "%",
        "spectral_axis": np.linspace(900.0, 1700.0, n_bands),
        "coefficients": rng.normal(size=n_bands),
        "loadings": rng.normal(size=n_bands),
    }


def _median_ms(step: Callable[[], None], qapp) -> float:
    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        step()
        qapp.processEvents()  # draw_idle paints in the next event-loop pass
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def test_model_diagnostics_timings(qapp):
    from enginir.viewer._model_diagnostics import ModelDiagnosticsWidget

    rng = np.random.default_rng(0)
    widgets = []

    def build() -> None:
        widget = ModelDiagnosticsWidget()
        widget.resize(*_SIZE)
        widget.show()
        widget.update_data(_payload(rng))
        widgets.append(widget)

    build_ms = _median_ms(build, qapp)
    for widget in widgets:
        widget.close()

    widget = ModelDiagnosticsWidget()
    widget.resize(*_SIZE)
    widget.show()
    widget.update_data(_payload(rng))
    qapp.processEvents()

    # A new payload object; the plots skip redraws for the same object.
    full_ms = _median_ms(lambda: widget.update_data(_payload(rng)), qapp)

    # One plot gets new data, the others keep theirs.
    panel = widget.grid.panel("model.pred_vs_ref")
    assert panel is not None
    one_ms = _median_ms(lambda: panel.update_data(_payload(rng)), qapp)
    widget.close()

    print(f"\nModel diagnostics at {_SIZE[0]}x{_SIZE[1]} (median of {_REPEATS}):"
          f"\n  build + first draw   {build_ms:7.1f} ms"
          f"\n  redraw all plots     {full_ms:7.1f} ms"
          f"\n  change one plot      {one_ms:7.1f} ms")
    assert one_ms < _BASELINE_ONE_MS, "changing one plot must not redraw the others"
    assert full_ms <= 1.1 * _BASELINE_FULL_MS


def test_explore_frame_scrubbing_timings(qapp):
    """One frame step with every Explore plot shown (the slider/click path)."""
    from enginir.viewer._viewer import HSIViewer

    cube = np.random.default_rng(0).integers(500, 3000, (400, 320, 224), dtype=np.uint16)
    viewer = HSIViewer(cube, np.linspace(900.0, 1700.0, 224, dtype=np.float32))
    viewer.resize(1500, 950)
    viewer.show()
    viewer._hist_dock.show()
    viewer._dc3_dock.show()
    for _ in range(5):
        qapp.processEvents()
    frames = iter(range(10, 10 + 3 * _REPEATS))
    step_ms = _median_ms(lambda: viewer.state.set_frame(next(frames)), qapp)
    viewer.close()
    print(f"\nExplore frame step, 400x320x224 cube, all plots shown: {step_ms:7.1f} ms")
    assert step_ms < _BASELINE_SCRUB_MS
