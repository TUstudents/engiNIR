"""QApplication singleton and event-loop helpers."""

from __future__ import annotations

import atexit
import importlib.resources
import os
import sys

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtGui import QGuiApplication, QIcon
from PyQt6.QtWidgets import QApplication, QComboBox, QDialog, QWidget

from .._logging import enable_logging, get_logger, logging_enabled
from ._theme import apply_qt_theme, load_preference, theme_applied
from ._windows_shell import WINDOWS_APP_ID

__all__ = ["enable_viewer_logging", "ensure_app", "run"]

# Module-level reference prevents CPython from garbage-collecting the
# QApplication when callers discard ensure_app()'s return value.
_app: QApplication | None = None
logger = get_logger(__name__)


def _is_wslg() -> bool:
    """Whether the session is WSLg (Windows Subsystem for Linux with GUI apps)."""
    return bool(os.environ.get("WSL_DISTRO_NAME")) and bool(os.environ.get("WAYLAND_DISPLAY"))


def _recommended_qt_platform() -> str | None:
    """Return the Qt platform plugin for the current session, if it needs a choice.

    On WSLg the viewer uses native Wayland: under X11 WSLg shows every popup
    as its own Windows window, and menus fall behind the main window. On other
    Linux sessions with both Wayland and X11, X11 is kept (Qt draws its own
    window decorations on Wayland, and Wayland start-up crashed there). An
    explicit ``QT_QPA_PLATFORM`` always wins.
    """
    if os.environ.get("QT_QPA_PLATFORM"):
        return None
    if not sys.platform.startswith("linux"):
        return None
    if not os.environ.get("WAYLAND_DISPLAY"):
        return None
    if _is_wslg():
        return "wayland"
    if not os.environ.get("DISPLAY"):
        return None
    return "xcb"


def _configure_qt_platform() -> None:
    """Choose the platform plugin before QApplication starts."""
    platform = _recommended_qt_platform()
    if platform is None:
        return
    os.environ["QT_QPA_PLATFORM"] = platform
    if platform == "wayland":
        # Qt draws the title bar (with the window icon); WSLg's own frame has none.
        logger.info("WSLg session: using QT_QPA_PLATFORM=wayland (set QT_QPA_PLATFORM=xcb "
                    "to use X11 instead)")
        return
    logger.warning(
        "Wayland session detected with DISPLAY={}; forcing QT_QPA_PLATFORM={} to avoid viewer startup crashes",
        os.environ.get("DISPLAY"),
        platform,
    )


def _set_windows_app_id() -> None:
    """Give the process its own taskbar identity on Windows (no-op elsewhere)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        result = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except (AttributeError, OSError) as exc:
        logger.debug("Windows app id not set: {}", exc)
        return
    if result != 0:
        logger.debug("Windows app id not set (HRESULT {})", result)


def app_icon() -> QIcon:
    """The engiNIR window icon (empty when the bundled file is unavailable)."""
    try:
        resource = importlib.resources.files("enginir._resources").joinpath("enginir.ico")
        with importlib.resources.as_file(resource) as path:
            return QIcon(str(path))
    except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
        logger.debug("Window icon unavailable: {}", exc)
        return QIcon()


class _DialogCentering(QObject):
    """Application-wide window fixes: centre dialogs; on WSLg Wayland, close combo popups.

    Window managers without a position hint (e.g. WSLg/xcb) otherwise place
    dialogs, including ``QMessageBox`` convenience calls and file dialogs, at
    the top-left corner of the screen.

    With ``release_combo_popups``, the native window of a combo box's list is
    destroyed after it hides (see :func:`release_popup_surface`).
    """

    def __init__(self, release_combo_popups: bool = False) -> None:
        super().__init__()
        self.release_combo_popups = release_combo_popups

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is None:
            return False
        kind = a1.type()
        if kind == QEvent.Type.Show and isinstance(a0, QDialog):
            centre_dialog(a0)
        elif kind == QEvent.Type.Hide and self.release_combo_popups \
                and isinstance(a0, QWidget) and _is_combo_popup(a0):
            QTimer.singleShot(0, lambda popup=a0: release_popup_surface(popup))
        return False


def _is_combo_popup(obj: QWidget) -> bool:
    """The popup window holding a combo box's list (Qt's private container)."""
    return (obj.isWindow()
            and obj.windowType() == Qt.WindowType.Popup
            and isinstance(obj.parentWidget(), QComboBox))


def release_popup_surface(popup: QWidget) -> None:
    """Destroy the native window of a hidden popup; Qt creates a new one when it opens again.

    WSLg's compositor keeps a popup on screen when the application unmaps it
    (choosing an item), and removes it only when the compositor dismisses it
    (a click outside) or the surface is destroyed (microsoft/wslg#1522).
    """
    if sip.isdeleted(popup) or popup.isVisible():
        return
    handle = popup.windowHandle()
    if handle is not None:
        handle.destroy()


def _releases_combo_popups(app: QApplication) -> bool:
    """Whether combo popups need :func:`release_popup_surface`: WSLg with native Wayland."""
    return _is_wslg() and app.platformName() == "wayland"


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
        release = _releases_combo_popups(app)
        if release:
            logger.info("WSLg Wayland: combo box popups release their surface when closed")
        _dialog_centering = _DialogCentering(release_combo_popups=release)
        app.installEventFilter(_dialog_centering)
        # atexit runs hooks last-registered-first, so this runs before PyQt's
        # exit cleanup (registered when PyQt6.QtCore was imported).
        atexit.register(_remove_dialog_centering)


def _remove_dialog_centering() -> None:
    """Remove the application-wide event filter before PyQt's exit cleanup.

    At interpreter exit PyQt deletes the remaining top-level Qt objects while
    walking sip's table of wrappers. A Python event filter on the application
    runs for the events of every object being deleted and creates wrappers,
    which can reallocate the table mid-walk; the walk then reads freed memory
    and the process ends with a segmentation fault (sip issue, unfixed in
    PyQt6-sip 13.13). Without Python code in that path the walk is safe.
    """
    global _dialog_centering
    centering, _dialog_centering = _dialog_centering, None
    if centering is None or sip.isdeleted(centering):
        return
    app = QApplication.instance()
    if app is not None and not sip.isdeleted(app):
        app.removeEventFilter(centering)


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
    call multiple times; returns the singleton. The theme is applied only when
    the application does not have it yet in the saved mode, because applying
    it restyles every open widget.
    """
    global _app
    enable_viewer_logging()
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        if existing.windowIcon().isNull():
            existing.setWindowIcon(app_icon())
        _install_dialog_centering(existing)
        dark = load_preference()
        if theme_applied(existing, dark=dark):
            # Restyling the application restyles every live widget; the theme
            # is already in place, so a second window does not repeat it.
            return existing
        return apply_qt_theme(existing, dark=dark)
    _configure_qt_platform()
    _set_windows_app_id()  # before the first window, so the taskbar uses our icon
    _app = QApplication([])
    _app.setWindowIcon(app_icon())
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
