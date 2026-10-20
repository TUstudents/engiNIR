"""CLI entry point for the engiNIR HSI viewer.

Usage
-----
    enginir-viewer                           # show welcome screen
    enginir-viewer path/to/file.eiff         # open file directly
    enginir-viewer path/to/project.h5        # open project directly
    enginir-viewer --project project.h5      # project-only startup
    enginir-viewer data.eiff --project p.h5  # dataset-first + project
    enginir-viewer file.eiff --dark d.eiff --white w.eiff
    uv run python -m enginir.viewer file.eiff
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from typing import Any

from .._logging import get_logger, log_file_path

logger = get_logger(__name__)


def _main(argv: list[str] | None = None) -> int:
    from ._app import enable_viewer_logging

    app_started = time.perf_counter()
    enable_viewer_logging()
    logger.info("enginir-viewer started")
    logger.info("Log file: {}", log_file_path())

    parser = argparse.ArgumentParser(
        prog="enginir-viewer",
        description="Open the engiNIR interactive HSI viewer.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to a data file or viewer project file. Omit to show the welcome screen.",
    )
    parser.add_argument(
        "--project",
        default=None,
        metavar="FILE",
        help="Project file to open. Can be used alone for project-only startup or together with a dataset path.",
    )
    parser.add_argument("--dark", default=None, metavar="FILE", help="Dark-reference file")
    parser.add_argument("--white", default=None, metavar="FILE", help="White-reference file")
    args = parser.parse_args(argv)

    from ._app import ensure_app, run
    from . import open_project, open_viewer

    ensure_app()
    logger.debug(
        "Startup timing: QApplication ready in {:.3f}s",
        time.perf_counter() - app_started,
    )

    _viewer_ref = None  # hold a reference so Qt doesn't GC the viewer

    if args.path is not None or args.project is not None:
        startup_label = args.project if args.path is None else args.path
        logger.info("Opening startup target: {}", startup_label)
        target_started = time.perf_counter()
        try:
            if args.path is None:
                _viewer_ref = open_project(args.project)
            else:
                _viewer_ref = open_viewer(
                    args.path,
                    dark=args.dark,
                    white=args.white,
                    project=args.project,
                )
            logger.info(
                "Startup timing: direct startup target opened in {:.3f}s: {}",
                time.perf_counter() - target_started,
                startup_label,
            )
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox

            logger.error("Failed to open startup target {}: {}", startup_label, exc)
            QMessageBox.critical(None, "Cannot open startup target", str(exc))
            return 1
    else:
        logger.info("Showing welcome screen")
        from PyQt6.QtWidgets import QApplication, QMessageBox

        from ..io.project_store import create_project
        from . import (
            _load_dataset_startup,
            _load_project_startup,
            _open_loaded_dataset_viewer,
            _open_loaded_project_viewer,
        )
        from ._jobs import start_background_job
        from ._loading import LoadingProgressEvent, LoadingProgressRelay, ViewerBusyContext
        from ._welcome import WelcomeWindow

        welcome = WelcomeWindow()
        welcome_created_at = time.perf_counter()
        selected_at: float | None = None
        _startup_job: object | None = None
        _startup_busy: ViewerBusyContext | None = None
        _startup_progress_relay: LoadingProgressRelay | None = None

        def _clear_startup_busy(text: str | None = None) -> None:
            nonlocal _startup_busy
            if _startup_busy is None:
                return
            _startup_busy.end(text)
            _startup_busy = None

        def _start_welcome_job(
            *,
            text: str,
            fn: Callable[[Callable[[str, int, int], None] | None], Any],
            on_success: Callable[[Any], None],
            error_title: str,
            log_context: str,
        ) -> None:
            nonlocal _startup_job, _startup_busy, _startup_progress_relay
            if _startup_job is not None:
                return
            job_started = time.perf_counter()
            logger.debug("Starting welcome loading job: {}", text)
            logger.debug(
                "Startup timing: welcome job queued {:.3f}s after selection ({})",
                0.0 if selected_at is None else job_started - selected_at,
                text,
            )
            _startup_busy = ViewerBusyContext(
                text,
                status_widget=welcome.loading_status_widget(),
                disabled_controls=welcome.loading_controls(),
            )
            _startup_busy.begin()
            _startup_progress_relay = LoadingProgressRelay(welcome)

            def _handle_progress_event(event: LoadingProgressEvent) -> None:
                if _startup_busy is not None:
                    _startup_busy.set_progress(event)

            _startup_progress_relay.progressed.connect(_handle_progress_event)

            def _progress(message: str, completed: int = 0, total: int = 0) -> None:
                if _startup_progress_relay is not None:
                    _startup_progress_relay.report(
                        message,
                        completed=completed,
                        total=total,
                        indeterminate=total <= 0,
                    )

            def _run() -> Any:
                worker_started = time.perf_counter()
                try:
                    return fn(_progress)
                finally:
                    logger.debug(
                        "Startup timing: welcome worker finished in {:.3f}s ({})",
                        time.perf_counter() - worker_started,
                        text,
                    )

            def _handle_success(payload: Any) -> None:
                logger.info("Welcome loading job completed: {}", text)
                handoff_started = time.perf_counter()
                if _startup_busy is not None:
                    _startup_busy.set_progress(
                        LoadingProgressEvent(
                            message="Opening viewer...",
                            completed=0,
                            total=0,
                            indeterminate=True,
                        )
                    )
                    QApplication.processEvents()
                try:
                    on_success(payload)
                except Exception as exc:
                    _clear_startup_busy()
                    logger.error("{}: {}", log_context, exc)
                    QMessageBox.critical(welcome, error_title, str(exc))
                    return
                logger.debug(
                    "Startup timing: viewer handoff completed in {:.3f}s ({})",
                    time.perf_counter() - handoff_started,
                    text,
                )
                welcome.close()
                _clear_startup_busy()

            def _handle_error(message: str) -> None:
                logger.info(
                    "Startup timing: welcome job failed after {:.3f}s ({})",
                    time.perf_counter() - job_started,
                    text,
                )
                _clear_startup_busy()
                logger.error("{}: {}", log_context, message)
                QMessageBox.critical(welcome, error_title, message)

            def _handle_finished() -> None:
                nonlocal _startup_job, _startup_progress_relay
                _startup_job = None
                _startup_progress_relay = None
                logger.debug("Welcome loading job finished: {}", text)
                logger.debug(
                    "Startup timing: welcome job lifecycle {:.3f}s ({})",
                    time.perf_counter() - job_started,
                    text,
                )
                _clear_startup_busy()

            _startup_job = start_background_job(
                owner=welcome,
                fn=_run,
                on_success=_handle_success,
                on_error=_handle_error,
                on_finished=_handle_finished,
            )

        def _show_loaded_dataset(payload: Any) -> None:
            nonlocal _viewer_ref
            _viewer_ref = _open_loaded_dataset_viewer(payload)

        def _show_loaded_project(payload: Any) -> None:
            nonlocal _viewer_ref
            _viewer_ref = _open_loaded_project_viewer(payload)

        def _on_dataset_selected(path: str, dark: str | None, white: str | None) -> None:
            nonlocal selected_at
            selected_at = time.perf_counter()
            logger.debug("Dataset selected from welcome screen: {}", path)
            logger.debug(
                "Startup timing: dataset selected {:.3f}s after welcome creation: {}",
                selected_at - welcome_created_at,
                path,
            )
            _start_welcome_job(
                text=f"Opening dataset: {path}",
                fn=lambda progress: _load_dataset_startup(
                    path,
                    dark=dark,
                    white=white,
                    progress_callback=progress,
                ),
                on_success=_show_loaded_dataset,
                error_title="Cannot open file",
                log_context=f"Failed to open {path}",
            )

        def _on_project_selected(path: str) -> None:
            nonlocal selected_at
            selected_at = time.perf_counter()
            logger.debug("Project selected from welcome screen: {}", path)
            logger.debug(
                "Startup timing: project selected {:.3f}s after welcome creation: {}",
                selected_at - welcome_created_at,
                path,
            )
            _start_welcome_job(
                text=f"Opening project: {path}",
                fn=lambda _progress: _load_project_startup(path),
                on_success=_show_loaded_project,
                error_title="Cannot open project",
                log_context=f"Failed to open project {path}",
            )

        def _on_project_created(path: str) -> None:
            nonlocal selected_at
            selected_at = time.perf_counter()
            logger.debug("Creating project from welcome screen: {}", path)
            logger.debug(
                "Startup timing: create project selected {:.3f}s after welcome creation: {}",
                selected_at - welcome_created_at,
                path,
            )
            _start_welcome_job(
                text=f"Creating project: {path}",
                fn=lambda _progress: (
                    create_project(path),
                    _load_project_startup(path),
                )[1],
                on_success=_show_loaded_project,
                error_title="Cannot create project",
                log_context=f"Failed to create project {path}",
            )

        welcome.file_selected.connect(_on_dataset_selected)
        welcome.project_selected.connect(_on_project_selected)
        welcome.project_created.connect(_on_project_created)
        welcome.show()
        logger.info(
            "Startup timing: welcome window shown in {:.3f}s",
            time.perf_counter() - app_started,
        )

    run()
    logger.debug("Shutdown")
    logger.debug("Startup timing: app session duration {:.3f}s", time.perf_counter() - app_started)
    return 0


def main(argv: list[str] | None = None) -> None:
    try:
        sys.exit(_main(argv))
    except Exception:
        logger.exception("Launcher crashed")
        sys.exit(1)


if __name__ == "__main__":
    main()
