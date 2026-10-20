"""QApplication singleton and event-loop helpers."""

from __future__ import annotations

import os
import sys

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QDialog

from .._logging import enable_logging, get_logger, logging_enabled
from ._theme import apply_qt_theme, load_preference

__all__ = ["enable_viewer_logging", "ensure_app", "run"]

# Module-level reference prevents CPython from garbage-collecting the
# QApplication when callers discard ensure_app()'s return value.
_app: QApplication | None = None
logger = get_logger(__name__)


def _recommended_qt_platform() -> str | None:
    """Return a safer Qt platform plugin for the current session, if needed."""
    if os.environ.get("QT_QPA_PLATFORM"):
        return None
    if not sys.platform.startswith("linux"):
        return None
    if not os.environ.get("WAYLAND_DISPLAY"):
        return None
    if not os.environ.get("DISPLAY"):
        return None
    return "xcb"


def _configure_qt_platform() -> None:
    """Apply conservative platform-plugin overrides before QApplication starts."""
    platform = _recommended_qt_platform()
    if platform is None:
        return
    os.environ["QT_QPA_PLATFORM"] = platform
    logger.warning(
        "Wayland session detected with DISPLAY={}; forcing QT_QPA_PLATFORM={} to avoid viewer startup crashes",
        os.environ.get("DISPLAY"),
        platform,
    )


class _DialogCentering(QObject):
    """Centre every dialog over its parent window (or the screen) when it is shown.

    Window managers without a position hint (e.g. WSLg/xcb) otherwise place
    dialogs, including ``QMessageBox`` convenience calls and file dialogs, at
    the top-left corner of the screen.
    """

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is not None and a1.type() == QEvent.Type.Show and isinstance(a0, QDialog):
            centre_dialog(a0)
        return False


def centre_dialog(dialog: QDialog) -> None:
    """Move ``dialog`` so that it is centred over its parent window or the screen."""
    parent = dialog.parentWidget()
    window = parent.window() if parent is not None else None
    if window is not None and window.isVisible():
        area = window.frameGeometry()
    else:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
    dialog.move(area.center() - dialog.rect().center())


_dialog_centering: _DialogCentering | None = None


def _install_dialog_centering(app: QApplication) -> None:
    global _dialog_centering
    if _dialog_centering is None:
        _dialog_centering = _DialogCentering()
        app.installEventFilter(_dialog_centering)


def enable_viewer_logging() -> None:
    """Enable engiNIR logging for a viewer session unless it is already configured.

    Warnings go to stderr, INFO and above to the default log file (see :func:`enginir._logging.default_log_files`),
    and uncaught exceptions are logged. Installing an exception hook also keeps
    PyQt from aborting the process when an exception escapes a Qt slot.
    """
    if not logging_enabled():
        enable_logging("WARNING", file=True, catch_uncaught=True)


def ensure_app() -> QApplication:
    """Return the existing QApplication or create one.

    Also enables viewer logging (see :func:`enable_viewer_logging`). Safe to
    call multiple times; returns the singleton.
    """
    global _app
    enable_viewer_logging()
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        _install_dialog_centering(existing)
        return apply_qt_theme(existing, dark=load_preference())
    _configure_qt_platform()
    _app = QApplication([])
    _install_dialog_centering(_app)
    return apply_qt_theme(_app, dark=load_preference())


def run() -> None:
    """Start the Qt event loop (blocking).

    No-op when running inside a Jupyter kernel — the IPython event loop
    handles Qt integration via ``enable_gui("qt")``.
    """
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        if ip is not None:
            return
    except NameError:
        pass
    app = ensure_app()
    app.exec()
