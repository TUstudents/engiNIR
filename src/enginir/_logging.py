"""Logging configuration for engiNIR.

engiNIR logs through `Loguru <https://github.com/Delgan/loguru>`_. Following
Loguru's guidance for libraries, logging is **disabled when the package is
imported**: importing :mod:`enginir` adds no sinks, creates no files and
installs no exception hooks, so it never changes the logging of the program
that uses it.

Applications opt in with :func:`enable_logging`. The HSI viewer does so at
startup (warnings to stderr, INFO and above to a rotating log file, and logging
of uncaught exceptions). Scripts and notebooks can do the same::

    import enginir
    enginir.enable_logging("INFO", file="analysis.log")

Modules obtain their logger with ``logger = get_logger(__name__)`` and use
Loguru's ``{}`` placeholders.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

__all__ = [
    "add_log_sink",
    "default_log_files",
    "disable_logging",
    "enable_logging",
    "get_logger",
    "log_file_path",
    "logging_enabled",
    "remove_log_sink",
]

_PACKAGE = "enginir"

_STDERR_FORMAT = (
    "<level>{level:<7}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "{message}"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} | {message}"

# Library code must never log at import time without opt-in.
logger.disable(_PACKAGE)


@dataclass
class _LoggingState:
    """Sinks and hooks installed by :func:`enable_logging`, removed by :func:`disable_logging`."""

    sink_ids: list[int] = field(default_factory=list)
    log_file: Path | None = None
    previous_hooks: tuple[Any, Any] | None = None
    own_hooks: tuple[Any, Any] | None = None
    default_handler_rerouted: bool = False


_state = _LoggingState()


def _is_enginir_record(record: Any) -> bool:
    name = record["name"] or ""
    return name == _PACKAGE or name.startswith(_PACKAGE + ".")


def _is_enginir_console_record(record: Any) -> bool:
    # Uncaught exceptions are printed by the chained interpreter hook already.
    return _is_enginir_record(record) and not record["extra"].get("uncaught", False)


def get_logger(name: str) -> Any:
    """Return the Loguru logger with records attributed to module ``name``.

    Call once per module as ``logger = get_logger(__name__)``.
    """
    return logger.patch(lambda record: record.update(name=name))


def default_log_files() -> list[Path]:
    """Return the default log file locations in order of preference.

    ``$ENGINIR_LOG_FILE`` (if set), ``~/.enginir/enginir.log`` and
    ``/tmp/enginir/enginir.log``; the first writable one is used.
    """
    candidates = [Path.home() / ".enginir" / "enginir.log", Path("/tmp/enginir/enginir.log")]
    configured = os.environ.get("ENGINIR_LOG_FILE")
    return [Path(configured), *candidates] if configured else candidates


def enable_logging(
    level: str = "WARNING",
    *,
    file: bool | str | os.PathLike[str] = False,
    file_level: str = "INFO",
    catch_uncaught: bool = False,
) -> Path | None:
    """Turn on engiNIR log output.

    Calling the function again replaces the previous configuration.

    Parameters
    ----------
    level:
        Minimum level of engiNIR messages printed to stderr, e.g. ``"DEBUG"``,
        ``"INFO"`` or ``"WARNING"`` (default).
    file:
        ``False`` (default) for no log file, ``True`` for the first writable
        location of :func:`default_log_files`, or a path. The file rotates at 10 MB and
        keeps three old files.
    file_level:
        Minimum level written to the log file. Default ``"INFO"``.
    catch_uncaught:
        If ``True``, uncaught exceptions of the whole process (main thread and
        other threads) are also logged at CRITICAL level before the previous
        exception hooks run. Intended for applications such as the viewer.

    Returns
    -------
    pathlib.Path or None
        The log file in use, or ``None``.

    Notes
    -----
    If Loguru's default stderr handler (id 0) is still installed, it is
    replaced, once per process, by an identical handler that skips engiNIR
    records, so that engiNIR messages are not printed twice and other Loguru
    users keep their output. :func:`disable_logging` does not bring back
    handler id 0.

    If the configuration fails (e.g. an unknown level), nothing stays
    installed and engiNIR remains muted.
    """
    disable_logging()
    try:
        logger.enable(_PACKAGE)
        _reroute_default_handler()
        _state.sink_ids.append(
            logger.add(
                sys.stderr,
                level=level,
                format=_STDERR_FORMAT,
                filter=_is_enginir_console_record,
                diagnose=False,
            )
        )
        if file is True:
            for candidate in default_log_files():
                if _add_file_sink(candidate, file_level):
                    break
        elif file:
            _add_file_sink(Path(file), file_level)
        if catch_uncaught:
            _install_exception_hooks()
    except BaseException:
        disable_logging()
        raise
    return _state.log_file


def disable_logging() -> None:
    """Undo :func:`enable_logging`: remove its sinks and hooks and mute engiNIR again."""
    for sink_id in _state.sink_ids:
        logger.remove(sink_id)
    _state.sink_ids.clear()
    _state.log_file = None
    if _state.previous_hooks is not None:
        # Restore a hook only if it is still ours; a hook installed after ours stays.
        previous_hook, previous_thread_hook = _state.previous_hooks
        own_hook, own_thread_hook = _state.own_hooks or (None, None)
        if sys.excepthook is own_hook:
            sys.excepthook = previous_hook
        if threading.excepthook is own_thread_hook:
            threading.excepthook = previous_thread_hook
        _state.previous_hooks = None
    logger.disable(_PACKAGE)


def logging_enabled() -> bool:
    """Return ``True`` while a configuration from :func:`enable_logging` is active."""
    return bool(_state.sink_ids)


def log_file_path() -> Path | None:
    """Return the active log file, or ``None`` if no file is being written."""
    return _state.log_file


def add_log_sink(*args: Any, **kwargs: Any) -> int:
    """Add a sink to the Loguru logger (passed to :meth:`loguru.logger.add`) and return its id."""
    return logger.add(*args, **kwargs)


def remove_log_sink(sink_id: int) -> None:
    """Remove a sink added with :func:`add_log_sink`."""
    logger.remove(sink_id)


def _reroute_default_handler() -> None:
    if _state.default_handler_rerouted:
        return
    _state.default_handler_rerouted = True
    try:
        logger.remove(0)
    except ValueError:
        return  # The application already configured Loguru itself.
    logger.add(sys.stderr, filter=lambda record: not _is_enginir_record(record))


def _add_file_sink(path: Path, level: str) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sink_id = logger.add(
            str(path),
            level=level,
            format=_FILE_FORMAT,
            filter=_is_enginir_record,
            rotation="10 MB",
            retention=3,
            encoding="utf-8",
            diagnose=False,
        )
    except OSError as exc:
        logger.warning("Cannot write the log file {}: {}", path, exc)
        return False
    _state.sink_ids.append(sink_id)
    _state.log_file = path
    return True


def _install_exception_hooks() -> None:
    previous_hook, previous_thread_hook = sys.excepthook, threading.excepthook
    _state.previous_hooks = (previous_hook, previous_thread_hook)
    uncaught_logger = logger.bind(uncaught=True)

    def log_uncaught(exc_type: Any, exc: Any, traceback: Any) -> None:
        if not issubclass(exc_type, KeyboardInterrupt):
            uncaught_logger.opt(exception=(exc_type, exc, traceback)).critical(
                "Uncaught exception"
            )
        previous_hook(exc_type, exc, traceback)

    def log_uncaught_in_thread(args: Any) -> None:
        # Like Python's default thread hook, a thread ending with SystemExit is not an error.
        if not issubclass(args.exc_type, SystemExit):
            uncaught_logger.opt(
                exception=(args.exc_type, args.exc_value, args.exc_traceback)
            ).critical("Uncaught exception in thread {}", getattr(args.thread, "name", "?"))
        previous_thread_hook(args)

    sys.excepthook = log_uncaught
    threading.excepthook = log_uncaught_in_thread
    _state.own_hooks = (log_uncaught, log_uncaught_in_thread)
