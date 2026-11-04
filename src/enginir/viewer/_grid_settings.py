"""Default ROI grid geometry, kept in the user settings (Options > ROI Grid Defaults).

Every new ROI grid starts from these values; changes made in the grid dialog
apply to that grid only. Stored in ``roi_grid.json`` next to the
recent-files list, never in a project.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

from .._logging import get_logger
from ._recent import config_dir

logger = get_logger(__name__)

_FILE = "roi_grid.json"

#: Name patterns offered in the grid dialogs (any other format string can be typed).
NAME_PATTERNS = (
    "{file}-{row}{column:02d}",
    "{row}{column:02d}",
    "{file}-{n:03d}",
    "{n:03d}",
)


@dataclass(frozen=True)
class GridDefaults:
    """The geometry and naming a new ROI grid starts with (pixels; x = frames, y = rows)."""

    n_rows: int = 1
    n_columns: int = 5
    width: int = 180
    height: int = 110
    origin_x: int = 150
    origin_y: int = 64
    spacing_x: int = 214
    spacing_y: int = 10
    name_pattern: str = NAME_PATTERNS[0]
    order: str = "rows"


def load_grid_defaults() -> GridDefaults:
    """The stored defaults; built-in values for anything missing or unreadable."""
    path = config_dir() / _FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return GridDefaults()
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable ROI grid defaults {}: {}", path, exc)
        return GridDefaults()
    if not isinstance(data, dict):
        return GridDefaults()
    built_in = GridDefaults()
    values = {}
    for field in fields(GridDefaults):
        value = data.get(field.name, getattr(built_in, field.name))
        expected = type(getattr(built_in, field.name))
        values[field.name] = value if isinstance(value, expected) else getattr(built_in,
                                                                                field.name)
    return GridDefaults(**values)


def save_grid_defaults(defaults: GridDefaults) -> None:
    """Store the defaults new ROI grids start with."""
    path = config_dir() / _FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(defaults)), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not save ROI grid defaults {}: {}", path, exc)


__all__ = ["NAME_PATTERNS", "GridDefaults", "load_grid_defaults", "save_grid_defaults"]
