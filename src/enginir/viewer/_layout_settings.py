"""Plot layout state kept in the user settings, not in the project file.

Stores the maximized plot of each workspace (Plot Panel Policy,
``viewer-workspace-ui-contract_v3.md``) in ``plot_layout.json`` next to the
recent-files list. Floating docks and the active page are not stored: a dock
maximize is transient, and a new model run always opens the `Fit` page.
"""

from __future__ import annotations

import json

from .._logging import get_logger
from ._recent import config_dir

logger = get_logger(__name__)

_FILE = "plot_layout.json"


def load_maximized() -> dict[str, str]:
    """The maximized plot id of each workspace that has one."""
    path = config_dir() / _FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable plot layout settings {}: {}", path, exc)
        return {}
    maximized = data.get("maximized") if isinstance(data, dict) else None
    if not isinstance(maximized, dict):
        return {}
    return {str(k): str(v) for k, v in maximized.items() if isinstance(v, str)}


def save_maximized(workspace: str, plot_id: str | None) -> None:
    """Record the maximized plot of ``workspace`` (``None``: nothing maximized)."""
    maximized = load_maximized()
    if plot_id is None:
        if maximized.pop(workspace, None) is None:
            return
    elif maximized.get(workspace) == plot_id:
        return
    else:
        maximized[workspace] = plot_id
    _write({"maximized": maximized})


def clear() -> None:
    """Forget all stored plot layout state (``Reset Layout``)."""
    path = config_dir() / _FILE
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove plot layout settings {}: {}", path, exc)


def _write(data: dict[str, object]) -> None:
    path = config_dir() / _FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not save plot layout settings {}: {}", path, exc)


__all__ = ["clear", "load_maximized", "save_maximized"]
