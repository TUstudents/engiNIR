"""Recent files list - pure I/O, no Qt dependency.

Config file location (per platform):
  Linux:   ~/.config/enginir/recent.json
  Windows: %APPDATA%/enginir/recent.json
  macOS:   ~/Library/Application Support/enginir/recent.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from enginir._logging import get_logger

logger = get_logger(__name__)

_MAX = 10

__all__ = ["config_dir", "load", "add", "remove"]


def config_dir() -> Path:
    """Return the platform-appropriate config directory for enginir."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", Path.home())
        return Path(base) / "enginir"
    if sys.platform == "darwin":
        base = os.environ.get("HOME", Path.home())
        return Path(base) / "Library" / "Application Support" / "enginir"
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    home = os.environ.get("HOME", Path.home())
    base = Path(xdg) if xdg else Path(home) / ".config"
    return base / "enginir"


def _path() -> Path:
    return config_dir() / "recent.json"


def load() -> list[str]:
    """Return list of recent file paths. Returns [] if missing or malformed."""
    path = _path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(p, str) for p in data):
            return data
        logger.warning("Recent files payload at {} is not a list of paths", path)
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load recent files from {}: {}", path, exc)
    return []


def add(path: str) -> None:
    """Prepend abspath(path) to the list, deduplicate, cap at 10, write back."""
    recent_path = _path()
    try:
        abspath = _canonical_recent_path(path)
        current = [p for p in load() if p != abspath]
        current.insert(0, abspath)
        current = current[:_MAX]
        cfg = config_dir()
        cfg.mkdir(parents=True, exist_ok=True)
        recent_path.write_text(json.dumps(current), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write recent files to {}: {}", recent_path, exc)


def remove(path: str) -> None:
    """Remove path from the list. No-op if not present."""
    recent_path = _path()
    try:
        abspath = _canonical_recent_path(path)
        current = [p for p in load() if p != abspath]
        config_dir().mkdir(parents=True, exist_ok=True)
        recent_path.write_text(json.dumps(current), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not update recent files at {}: {}", recent_path, exc)


def _canonical_recent_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if sys.platform != "win32" and expanded.startswith("/"):
        return os.path.normpath(expanded).replace("\\", "/")
    return os.path.abspath(expanded)
