"""Progress events and busy state for viewer loading flows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication, QWidget

from ._jobs import BusyStatusWidget

__all__ = [
    "LoadingProgressEvent",
    "LoadingProgressRelay",
    "ViewerBusyContext",
]


@dataclass(frozen=True, slots=True)
class LoadingProgressEvent:
    """Structured progress event for startup loading jobs."""

    message: str
    completed: int = 0
    total: int = 0
    indeterminate: bool = True


class LoadingProgressRelay(QObject):
    """Thread-safe signal relay for startup progress callbacks."""

    progressed = pyqtSignal(object)

    def report(
        self,
        message: str,
        *,
        completed: int = 0,
        total: int = 0,
        indeterminate: bool | None = None,
    ) -> None:
        self.progressed.emit(
            LoadingProgressEvent(
                message=message,
                completed=int(completed),
                total=int(total),
                indeterminate=(total <= 0 if indeterminate is None else bool(indeterminate)),
            )
        )


class ViewerBusyContext:
    """Manage wait cursor, disabled controls, and optional status widgets."""

    def __init__(
        self,
        text: str,
        *,
        status_widget: BusyStatusWidget | None = None,
        disabled_controls: Iterable[Any] = (),
        status_setter: Callable[[str], None] | None = None,
        status_clearer: Callable[[], None] | None = None,
    ) -> None:
        self._text = text
        self._status_widget = status_widget
        self._controls = list(disabled_controls)
        self._status_setter = status_setter
        self._status_clearer = status_clearer
        self._enabled_states: list[tuple[Any, bool]] = []
        self._cursor_states: list[tuple[QWidget, QCursor]] = []
        self._active = False

    def begin(self) -> None:
        if self._active:
            return
        self._active = True
        self._enabled_states = []
        for control in self._controls:
            is_enabled = getattr(control, "isEnabled", None)
            set_enabled = getattr(control, "setEnabled", None)
            if callable(is_enabled) and callable(set_enabled):
                self._enabled_states.append((control, bool(is_enabled())))
                set_enabled(False)
        if self._status_widget is not None:
            self._status_widget.set_busy(self._text)
        if self._status_setter is not None:
            self._status_setter(self._text)
        wait_cursor = QCursor(Qt.CursorShape.WaitCursor)
        for control in self._controls:
            self._set_wait_cursor(control, wait_cursor)
        self._set_wait_cursor(self._status_widget, wait_cursor)
        QApplication.setOverrideCursor(wait_cursor)
        QApplication.processEvents()

    def end(self, text: str | None = None) -> None:
        if not self._active:
            return
        self._active = False
        for control, was_enabled in self._enabled_states:
            set_enabled = getattr(control, "setEnabled", None)
            if callable(set_enabled):
                set_enabled(was_enabled)
        self._enabled_states = []
        if self._status_widget is not None:
            self._status_widget.set_idle(text)
        if self._status_clearer is not None:
            self._status_clearer()
        for widget, cursor in reversed(self._cursor_states):
            widget.setCursor(cursor)
        self._cursor_states = []
        QApplication.restoreOverrideCursor()
        QApplication.processEvents()

    def set_progress(self, event: LoadingProgressEvent) -> None:
        if self._status_widget is not None:
            self._status_widget.set_progress(event)
        if self._status_setter is not None:
            self._status_setter(event.message)

    def _set_wait_cursor(self, widget: Any, cursor: QCursor) -> None:
        if not isinstance(widget, QWidget):
            return
        self._remember_cursor(widget, cursor)
        window = widget.window()
        if isinstance(window, QWidget):
            self._remember_cursor(window, cursor)

    def _remember_cursor(self, widget: QWidget, cursor: QCursor) -> None:
        if any(existing is widget for existing, _ in self._cursor_states):
            return
        self._cursor_states.append((widget, widget.cursor()))
        widget.setCursor(cursor)
