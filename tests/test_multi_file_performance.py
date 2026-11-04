"""Timing benchmark for projects with ROIs from many data files (multi-file plan).

Skipped unless ``ENGINIR_PERF=1``; run it with output shown::

    ENGINIR_PERF=1 QT_QPA_PLATFORM=offscreen uv run pytest tests/test_multi_file_performance.py -s

The project holds 10 data files with a 5 x 8 tray each: 400 ROIs of 2,000
pixels (40 frames x 50 rows) and 224 bands, about 360 MB of raw ROI spectra.
It measures what the plan's targets refer to: opening the project, reading
every ROI's pixels, the fit, the Project ROIs tab (first tile, all tiles,
all mean spectra, event-loop gaps meanwhile) and the Locate file check.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.skipif(os.environ.get("ENGINIR_PERF") != "1",
                       reason="timing benchmark; set ENGINIR_PERF=1"),
    pytest.mark.timeout(900),
]

N_FILES, TRAY_ROWS, TRAY_COLS = 10, 5, 8
ROI_FRAMES, ROI_ROWS, N_BANDS = 40, 50, 224


@pytest.fixture(scope="module")
def project_path(tmp_path_factory) -> Path:
    from enginir import Project
    from enginir.io import save_hdf5

    folder = tmp_path_factory.mktemp("multi_file")
    rng = np.random.default_rng(0)
    wavelengths = np.linspace(900.0, 1700.0, N_BANDS)
    peak = np.exp(-((np.arange(N_BANDS) - 120) / 15.0) ** 2)
    path = folder / "trays.h5"
    started = time.perf_counter()
    project = Project.create(path)
    project.add_target("API", unit="% w/w")
    for file_index in range(N_FILES):
        content = rng.uniform(1.0, 9.0, (TRAY_ROWS, TRAY_COLS))
        # One sample per cell; its absorption at the peak grows with its content.
        cell_content = np.repeat(np.repeat(content, ROI_ROWS, axis=0), ROI_FRAMES, axis=1).T
        cube = (2500.0 - 100.0 * cell_content[:, :, None] * peak
                + rng.normal(0.0, 5.0, (TRAY_COLS * ROI_FRAMES, TRAY_ROWS * ROI_ROWS, N_BANDS)))
        data_file = folder / f"tray{file_index:02d}.h5"
        save_hdf5(str(data_file), cube.astype(np.uint16), wavelengths)
        dataset = project.add_dataset(data_file)
        for row in range(TRAY_ROWS):
            for col in range(TRAY_COLS):
                dataset.add_roi(
                    f"tray{file_index:02d}-{'ABCDE'[row]}{col + 1:02d}",
                    frames=(col * ROI_FRAMES, (col + 1) * ROI_FRAMES - 1),
                    rows=(row * ROI_ROWS, (row + 1) * ROI_ROWS - 1),
                    values={"API": float(content[row, col])})
    print(f"\nbuilt {N_FILES * TRAY_ROWS * TRAY_COLS} ROIs in {time.perf_counter() - started:.1f} s"
          f", project {path.stat().st_size / 1e6:.0f} MB")
    return path


def _timed(step) -> tuple[float, object]:
    started = time.perf_counter()
    result = step()
    return (time.perf_counter() - started) * 1000.0, result


def test_multi_file_project_timings(project_path):
    from enginir import Project
    from enginir.io.project_store import load_roi_spectra_batch
    from enginir.viewer._app import ensure_app
    from enginir.viewer._state import ProjectState

    app = ensure_app()
    state = ProjectState()
    open_ms, _ = _timed(lambda: state.open_project(str(project_path)))
    assert len(state.rois) == N_FILES * TRAY_ROWS * TRAY_COLS

    from enginir.viewer import open_project

    window_ms, window = _timed(lambda: open_project(project_path))
    app.processEvents()
    window.close()

    keys = [f"/rois/{roi_id}/spectra" for roi_id in state.rois]
    first_ms, _ = _timed(lambda: load_roi_spectra_batch(str(project_path), keys[:1]))
    all_ms, spectra = _timed(lambda: load_roi_spectra_batch(str(project_path), keys))

    def mean_spectra() -> dict:
        return {key: values.mean(axis=0) for key, values in spectra.items()}

    means_ms, _ = _timed(mean_spectra)
    spectra.clear()

    project = Project.open(project_path)
    fit_ms, model = _timed(lambda: project.fit("API", n_components=5))

    print(f"\nMulti-file project ({N_FILES} files, {len(keys)} ROIs, "
          f"{ROI_FRAMES * ROI_ROWS} px, {N_BANDS} bands):"
          f"\n  open project (ProjectState)        {open_ms:9.1f} ms"
          f"\n  open project window (project-only) {window_ms:9.1f} ms"
          f"\n  read one ROI's pixels              {first_ms:9.1f} ms"
          f"\n  read all ROI pixels                {all_ms:9.1f} ms"
          f"\n  mean spectra (raw, in memory)      {means_ms:9.1f} ms"
          f"\n  fit PLS 5 LV, auto CV              {fit_ms:9.1f} ms  ({model})")
    assert np.isfinite([open_ms, window_ms, first_ms, all_ms, means_ms, fit_ms]).all()


def test_project_rois_tab_and_locate_timings(project_path, tmp_path):
    """Phase 9 targets: first tile under 1 s, UI responsive while the tab loads."""
    import shutil

    from PyQt6.QtCore import QElapsedTimer, QTimer

    from enginir import Project
    from enginir.viewer import open_project
    from enginir.viewer._app import ensure_app

    app = ensure_app()
    started = time.perf_counter()
    window = open_project(project_path)  # a project with ROIs opens on Project ROIs
    window.resize(1400, 900)
    window.show()
    view = window._project_rois
    gaps: list[float] = []
    clock = QElapsedTimer()
    clock.start()

    def tick() -> None:
        gaps.append(clock.restart())

    ticker = QTimer()
    ticker.setInterval(10)
    ticker.timeout.connect(tick)
    ticker.start()
    first_tile = all_tiles = means = None
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        app.processEvents()
        now = time.perf_counter() - started
        if first_tile is None and view.mosaic.tile_count() > 0:
            first_tile = now
        if all_tiles is None and not view.mosaic.missing_patches() and view.mosaic.tile_count():
            all_tiles = now
        if all_tiles is not None and not view.busy() and view.means.payload:
            means = now
            break
        time.sleep(0.002)
    ticker.stop()
    window.close()
    assert first_tile is not None and all_tiles is not None and means is not None

    # Locate a moved data file: compares the stored pixels of its 40 ROIs.
    project = Project.open(project_path)
    dataset = next(iter(project.datasets.values()))
    moved = tmp_path / "moved.h5"
    shutil.copy(dataset.path, moved)
    from enginir._workflow import check_data_file_location

    rois = list(dataset.rois.values())
    locate_ms, _ = _timed(lambda: check_data_file_location(project.path, dataset.ref, rois,
                                                           moved))
    gaps_sorted = sorted(gaps)
    print(f"\nProject ROIs tab ({len(project.workspace.rois)} ROIs):"
          f"\n  window open to first tile          {first_tile * 1000:9.1f} ms"
          f"\n  window open to all tiles           {all_tiles * 1000:9.1f} ms"
          f"\n  window open to all mean spectra    {means * 1000:9.1f} ms"
          f"\n  event loop gap: median {gaps_sorted[len(gaps) // 2]:.0f} ms, "
          f"95% {gaps_sorted[int(len(gaps) * 0.95)]:.0f} ms, max {gaps_sorted[-1]:.0f} ms"
          f"\n  Locate check, {len(rois)} ROIs            {locate_ms:9.1f} ms")
