"""engiNIR viewer public API.

Public API
----------
``ensure_app()`` -> QApplication singleton helper
``run()`` -> Qt event-loop helper
``open_viewer(path=None, *, dark=None, white=None, project=None)`` -> :class:`HSIViewer`
``open_project(path)`` -> :class:`HSIViewer`
``ViewerState`` / ``ProjectState`` -> viewer and project workspace state objects

Widgets for other applications (e.g. a production-line status window)
----------------------------------------------------------------------
``apply_theme(app, *, dark=None)``, ``theme_tokens()``, ``value_colormap()``,
``heat_colormap()`` -> the viewer's look
``PredictionMapWidget`` -> prediction, DModXPS and T²PS maps (``show_maps``)
``FalseColorCanvas`` -> a scan's false-color image with layout cells (``show_cells``)

CLI
---
``uv run python -m enginir.viewer file.eiff [--dark d.eiff] [--white w.eiff]``
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the names below load on first use (PEP 562), see __getattr__
    from ._app import ensure_app, run
    from ._canvas import FalseColorCanvas
    from ._prediction import PredictionMapWidget
    from ._theme import apply_theme, heat_colormap, theme_tokens, value_colormap
    from ._files import _is_project_file as _is_project_file
    from ._open import open_project, open_viewer
    from ._state import ProjectState, ViewerState

#: Name -> the module it lives in. Importing ``enginir.viewer`` (or a light module of it,
#: such as the launcher's) loads none of them.
_LAZY = {
    "ensure_app": "._app", "run": "._app",
    "ProjectState": "._state", "ViewerState": "._state",
    "_is_project_file": "._files",
    "FalseColorCanvas": "._canvas", "PredictionMapWidget": "._prediction",
    **{name: "._theme" for name in ("apply_theme", "heat_colormap", "theme_tokens",
                                     "value_colormap")},
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
    "apply_theme",
    "theme_tokens",
    "value_colormap",
    "heat_colormap",
    "PredictionMapWidget",
    "FalseColorCanvas",
]
