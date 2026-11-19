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

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the names below load on first use (PEP 562), see __getattr__
    from ._app import ensure_app, run
    from ._files import _is_project_file as _is_project_file
    from ._open import open_project, open_viewer
    from ._state import ProjectState, ViewerState

#: Name -> the module it lives in. Importing ``enginir.viewer`` (or a light module of it,
#: such as the launcher's) loads none of them (plan startup-speed-1.7.1, D2).
_LAZY = {
    "ensure_app": "._app", "run": "._app",
    "ProjectState": "._state", "ViewerState": "._state",
    "_is_project_file": "._files",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name, "._open")
    try:
        value = getattr(import_module(module, __name__), name)
    except AttributeError:
        raise AttributeError(f"module 'enginir.viewer' has no attribute {name!r}") from None
    globals()[name] = value  # resolved once
    return value


__all__ = [
    "ensure_app",
    "run",
    "open_project",
    "open_viewer",
    "ProjectState",
    "ViewerState",
]
