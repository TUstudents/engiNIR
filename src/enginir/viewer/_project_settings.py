"""Per-project UI state kept in the user settings, never in the project file.

Stores the last shown data file of each project by ``project_id``
(``project-dataset-attachment-contract_v3.md``, Reopening a project) in
``project_ui.json`` next to the recent-files list.
"""

from __future__ import annotations

import json
from typing import Any

from .._logging import get_logger
from ._recent import config_dir

logger = get_logger(__name__)

_FILE = "project_ui.json"


def load_last_data_file(project_id: str) -> str | None:
    """The dataset id last shown for the project ``project_id``, if any."""
    value = _read().get("last_data_file", {}).get(project_id)
    return value if isinstance(value, str) else None


def save_last_data_file(project_id: str, dataset_id: str) -> None:
    """Remember ``dataset_id`` as the last shown data file of ``project_id``."""
    data = _read()
    last = data.setdefault("last_data_file", {})
    if last.get(project_id) == dataset_id:
        return
    last[project_id] = dataset_id
    path = config_dir() / _FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not save project settings {}: {}", path, exc)


def _read() -> dict[str, Any]:
    path = config_dir() / _FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable project settings {}: {}", path, exc)
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("last_data_file", {}), dict):
        return {}
    return data


__all__ = ["load_last_data_file", "save_last_data_file"]
