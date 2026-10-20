"""engiNIR viewer public API.

Public API
----------
``ensure_app()`` -> QApplication singleton helper
``run()`` -> Qt event-loop helper
``open_viewer(path=None, *, dark=None, white=None, project=None)`` -> :class:`HSIViewer`
``open_project(path)`` -> :class:`HSIViewer`
``ViewerState`` / ``ProjectState`` -> viewer and project workspace state objects

CLI
---
``uv run python -m enginir.viewer file.eiff [--dark d.eiff] [--white w.eiff]``
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..io import file_format, load_spectra
from ..io.project_store import is_project_file, load_project_store
from .._checks import log_and_raise
from .._logging import get_logger
from ._attachment import confirm_project_dataset_binding
from ._app import ensure_app, run
from ._state import ProjectState, ViewerState

if TYPE_CHECKING:
    from ._viewer import HSIViewer

logger = get_logger(__name__)


@dataclass(slots=True)
class _DatasetStartupPayload:
    """Pure data loaded before constructing one dataset-backed viewer."""

    path: str
    X: np.ndarray
    wavelengths: np.ndarray
    meta: dict[str, object]
    dark_cube: np.ndarray | None = None
    white_cube: np.ndarray | None = None


@dataclass(slots=True)
class _ProjectStartupPayload:
    """Pure project metadata loaded before constructing one project-only viewer."""

    path: str
    store: dict[str, Any]


def open_viewer(
    path: str | Path | None = None,
    *,
    dark: str | Path | None = None,
    white: str | Path | None = None,
    project: str | Path | None = None,
) -> "HSIViewer":
    """Open the interactive HSI viewer for a hyperspectral data file.

    Creates a QApplication if one does not exist.  Inside a Jupyter kernel
    the Qt event loop is integrated automatically via ``enable_gui("qt")``.
    Outside Jupyter, call :func:`enginir.viewer.run` after this
    function to start the blocking event loop.

    Parameters
    ----------
    path:
        Optional path to the data file. Any format accepted by
        :func:`enginir.io.load_spectra` is supported. If *path* points to a
        v3 project file, project-only startup is used automatically.
    dark:
        Optional dark-reference file (same format).
    white:
        Optional white-reference file (same format).
    project:
        Optional project file to load after the dataset is opened. When
        *path* is omitted, this enables project-only startup.

    Returns
    -------
    HSIViewer
        Viewer instance.  Use ``.state`` for programmatic control.
    """
    if path is None and project is None:
        log_and_raise("open_viewer requires a dataset path, a project path, or both")
    if path is not None and _is_project_file(path):
        if dark is not None or white is not None:
            log_and_raise("dark/white reference paths are only valid for dataset startup")
        if project is not None:
            log_and_raise("Pass either a project path as 'path' or via 'project', not both")
        return open_project(path)
    if path is None:
        if project is None:
            log_and_raise("open_viewer requires a project path when dataset path is omitted")
        return open_project(project)

    ensure_qt_ready()
    viewer = _open_dataset_viewer(path, dark=dark, white=white)
    if project is not None:
        try:
            _load_project_into_viewer(viewer, project, dataset_path=viewer._cube_path)
        except Exception as exc:
            logger.error("Failed to attach project {} to dataset viewer: {}", project, exc)
            viewer.close()
            raise
    return viewer


def open_project(path: str | Path) -> "HSIViewer":
    """Open a project in project-only mode without synthesizing dataset context."""
    ensure_qt_ready()
    payload = _load_project_startup(path)
    return _open_loaded_project_viewer(payload)


def _load_project_startup(path: str | Path) -> _ProjectStartupPayload:
    """Load project metadata for later UI-thread viewer construction."""
    project_path = os.path.abspath(str(path))
    logger.info("Loading project metadata: {}", project_path)
    store = load_project_store(project_path)
    logger.info(
        "Loaded project metadata: {} ({} datasets, {} ROIs, {} model runs, {} predictions)",
        project_path,
        len(store.get("datasets", {})),
        len(store.get("rois", {})),
        len(store.get("model_runs", {})),
        len(store.get("predictions", {})),
    )
    return _ProjectStartupPayload(path=project_path, store=store)


def _open_loaded_project_viewer(payload: _ProjectStartupPayload) -> "HSIViewer":
    """Construct and show a project-only viewer from preloaded project metadata."""
    ensure_qt_ready()

    from ._viewer import HSIViewer

    logger.info("Constructing project-only viewer for {}", payload.path)
    t0 = time.perf_counter()
    viewer = HSIViewer(
        np.zeros((0, 0, 0), dtype=np.uint16),
        np.zeros((0,), dtype=np.float64),
        {},
    )
    logger.info(
        "Project-only viewer constructed in {:.3f}s for {}",
        time.perf_counter() - t0,
        payload.path,
    )
    logger.info("Hydrating project-only viewer for {}", payload.path)
    t1 = time.perf_counter()
    viewer._cube_path = None
    try:
        _load_project_store_into_viewer(
            viewer,
            payload.path,
            store=payload.store,
            dataset_path=None,
        )
    except Exception as exc:
        logger.error("Failed to open project-only viewer for {}: {}", payload.path, exc)
        viewer.close()
        raise
    logger.info(
        "Project-only viewer hydrated in {:.3f}s for {}",
        time.perf_counter() - t1,
        payload.path,
    )
    t2 = time.perf_counter()
    viewer.show()
    viewer.raise_()
    viewer.activateWindow()
    logger.info(
        "Project viewer shown without attached dataset in {:.3f}s for {}",
        time.perf_counter() - t2,
        payload.path,
    )
    from PyQt6.QtWidgets import QApplication

    t3 = time.perf_counter()
    QApplication.processEvents()
    logger.debug(
        "Project viewer first event processing took {:.3f}s for {}",
        time.perf_counter() - t3,
        payload.path,
    )
    return viewer


def ensure_qt_ready() -> None:
    """Create QApplication if needed and integrate with Jupyter."""
    from ._app import ensure_app

    ensure_app()

    # Integrate Qt event loop when inside a Jupyter kernel
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        if ip is not None:
            ip.enable_gui("qt")
    except NameError:
        pass  # not in Jupyter


def _open_dataset_viewer(
    path: str | Path,
    *,
    dark: str | Path | None = None,
    white: str | Path | None = None,
) -> "HSIViewer":
    """Open one dataset-backed viewer window."""
    payload = _load_dataset_startup(path, dark=dark, white=white)
    return _open_loaded_dataset_viewer(payload)


def _load_dataset_startup(
    path: str | Path,
    *,
    dark: str | Path | None = None,
    white: str | Path | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> _DatasetStartupPayload:
    """Load dataset and optional references without constructing Qt widgets."""

    startup_started = time.perf_counter()
    logger.info("Loading cube: {}", path)
    t0 = time.perf_counter()
    X, wavelengths, meta = _load_spectra_startup_with_progress(
        path,
        label="cube",
        progress_callback=progress_callback,
    )
    elapsed = time.perf_counter() - t0
    if X.ndim == 2:
        X = X[:, np.newaxis, :]  # (n_samples, 1, n_bands)
    logger.info("Cube loaded in {:.1f}s - shape {}", elapsed, X.shape)

    dark_cube: np.ndarray | None = None
    if dark is not None:
        ref_started = time.perf_counter()
        dark_X, _, _ = _load_spectra_startup_with_progress(
            dark,
            label="dark reference",
            progress_callback=progress_callback,
        )
        dark_cube = dark_X.astype(np.float32, copy=False)
        logger.info(
            "Dark reference loaded in {:.3f}s - shape {}",
            time.perf_counter() - ref_started,
            dark_X.shape,
        )

    white_cube: np.ndarray | None = None
    if white is not None:
        ref_started = time.perf_counter()
        white_X, _, _ = _load_spectra_startup_with_progress(
            white,
            label="white reference",
            progress_callback=progress_callback,
        )
        white_cube = white_X.astype(np.float32, copy=False)
        logger.info(
            "White reference loaded in {:.3f}s - shape {}",
            time.perf_counter() - ref_started,
            white_X.shape,
        )

    payload = _DatasetStartupPayload(
        path=os.path.abspath(str(path)),
        X=X,
        wavelengths=np.asarray(wavelengths),
        meta=dict(meta),
        dark_cube=dark_cube,
        white_cube=white_cube,
    )
    logger.info(
        "Dataset startup payload prepared in {:.3f}s for {}",
        time.perf_counter() - startup_started,
        payload.path,
    )
    return payload


def _load_spectra_startup_with_progress(
    path: str | Path,
    *,
    label: str,
    progress_callback: Callable[[str, int, int], None] | None,
) -> Any:
    if progress_callback is not None:
        progress_callback(f"Loading {label}...", 0, 0)
    return load_spectra(str(path))


def _open_loaded_dataset_viewer(payload: _DatasetStartupPayload) -> "HSIViewer":
    """Construct and show a dataset-backed viewer from preloaded spectral data."""
    ensure_qt_ready()
    from ._viewer import HSIViewer

    t0 = time.perf_counter()
    logger.info("Constructing dataset viewer for {}", payload.path)
    viewer = HSIViewer(
        payload.X,
        payload.wavelengths,
        payload.meta,
        dark=payload.dark_cube,
        white=payload.white_cube,
    )
    logger.info(
        "Dataset viewer constructed in {:.3f}s for {}",
        time.perf_counter() - t0,
        payload.path,
    )
    t1 = time.perf_counter()
    viewer._cube_path = payload.path
    viewer._update_status_strip()
    from ._recent import add as _recent_add

    _recent_add(payload.path)
    logger.info(
        "Dataset viewer metadata/status prepared in {:.3f}s for {}",
        time.perf_counter() - t1,
        payload.path,
    )
    t2 = time.perf_counter()
    viewer.show()
    viewer.raise_()
    viewer.activateWindow()
    logger.info("Dataset viewer shown in {:.3f}s for {}", time.perf_counter() - t2, payload.path)

    from PyQt6.QtWidgets import QApplication

    t3 = time.perf_counter()
    QApplication.processEvents()
    logger.debug(
        "Dataset viewer first event processing took {:.3f}s for {}",
        time.perf_counter() - t3,
        payload.path,
    )
    return viewer


def _load_project_into_viewer(
    viewer: "HSIViewer",
    path: str | Path,
    *,
    dataset_path: str | None,
) -> None:
    """Load a project into an existing viewer.

    A project whose ROI datasets do not match the viewer's dataset asks for
    confirmation through the same dialog as attaching in the window, and
    raises :class:`ValueError` when the user cancels.
    """
    project_path = os.path.abspath(str(path))
    store = load_project_store(project_path)
    _load_project_store_into_viewer(
        viewer,
        project_path,
        store=store,
        dataset_path=dataset_path,
    )


def _load_project_store_into_viewer(
    viewer: "HSIViewer",
    project_path: str,
    *,
    store: dict[str, Any],
    dataset_path: str | None,
) -> None:
    """Attach preloaded project metadata to an existing viewer."""
    if not confirm_project_dataset_binding(
        viewer,
        project_path=project_path,
        store=store,
        dataset_path=dataset_path,
        source_format=None if not dataset_path else file_format(dataset_path),
        source_modality=str(viewer.state.meta.get("source_modality", "")),
        cube_shape=(
            int(viewer.state.cube.shape[0]),
            int(viewer.state.cube.shape[1]),
            int(viewer.state.cube.shape[2]),
        ),
        wavelengths=viewer.state.wavelengths,
    ):
        log_and_raise("Project attachment cancelled due to source mismatch")
    viewer.project_state.open_project(project_path, store=store)
    viewer._update_status_strip()
    from ._recent import add as _recent_add

    _recent_add(project_path)


def _is_project_file(path: str | Path | None) -> bool:
    """Return True when *path* is a v3 project file."""
    if path is None:
        return False
    candidate = Path(path)
    if candidate.suffix.lower() not in {".h5", ".hdf5"} or not candidate.exists():
        return False
    try:
        import h5py

        with h5py.File(candidate, "r") as handle:
            return is_project_file(handle)
    except OSError:
        return False


__all__ = [
    "ensure_app",
    "run",
    "open_project",
    "open_viewer",
    "ProjectState",
    "ViewerState",
]
