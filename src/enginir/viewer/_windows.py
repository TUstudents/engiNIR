"""The open top-level windows of the viewer (viewer-workspace-ui contract, Window lifetime).

A window is kept here from its creation until Qt deletes it after it closes
(``WA_DeleteOnClose``). The registry is what keeps a window's Python object
alive; nothing else needs to (no "parent" window owns another). When Qt
deletes the window, the registry forgets it and runs its ``on_deleted``
callback, which releases what the Python side still holds (the cube).
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QWidget

from .._logging import get_logger

logger = get_logger(__name__)

_OPEN_WINDOWS: set[QWidget] = set()


def keep_open(window: QWidget, *, on_deleted: Callable[[], None] | None = None) -> None:
    """Keep ``window`` alive until Qt deletes it; then forget it and call ``on_deleted``."""
    _OPEN_WINDOWS.add(window)
    name = type(window).__name__

    def forget(*_args: object) -> None:
        _OPEN_WINDOWS.discard(window)
        if on_deleted is not None:
            on_deleted()
        logger.debug("{} deleted; {} window(s) open", name, len(_OPEN_WINDOWS))

    window.destroyed.connect(forget)


def open_windows() -> list[QWidget]:
    """The windows that are open (created and not yet deleted)."""
    return list(_OPEN_WINDOWS)
