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
from dataclasses import dataclass
from pathlib import Path

from enginir._logging import get_logger

logger = get_logger(__name__)

_MAX = 10

__all__ = ["RecentEntry", "config_dir", "load", "entries", "add", "remove", "set_kind",
           "check", "last_check"]

#: What a recent path is: "project" or "dataset" (None: not known yet, e.g. an entry
#: written before 1.7.1). Stored with the entry, so the list is drawn without opening the
#: file (plan startup-speed-1.7.1, D4).
KINDS = ("project", "dataset")


@dataclass(frozen=True)
class RecentEntry:
    """One recent file: its path and, when known, whether it is a project or a dataset."""

    path: str
    kind: str | None = None


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


def entries() -> list[RecentEntry]:
    """The recent files, newest first; [] if the list is missing or malformed.

    Reads only the settings file: no recent path is touched (it may be on a slow or
    unreachable network drive). Entries are a path (written before 1.7.1) or
    ``{"path": ..., "kind": "project" | "dataset"}``.
    """
    path = _path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load recent files from {}: {}", path, exc)
        return []
    if not isinstance(data, list):
        logger.warning("Recent files payload at {} is not a list", path)
        return []
    found: list[RecentEntry] = []
    for item in data:
        if isinstance(item, str):
            found.append(RecentEntry(item))
        elif isinstance(item, dict) and isinstance(item.get("path"), str):
            kind = item.get("kind")
            found.append(RecentEntry(item["path"], kind if kind in KINDS else None))
        else:
            logger.warning("Recent files payload at {} has an entry that is not a path", path)
            return []
    return found


def load() -> list[str]:
    """Return list of recent file paths. Returns [] if missing or malformed."""
    return [entry.path for entry in entries()]


def _write(current: list[RecentEntry]) -> None:
    recent_path = _path()
    try:
        config_dir().mkdir(parents=True, exist_ok=True)
        recent_path.write_text(json.dumps(
            [{"path": entry.path, "kind": entry.kind} if entry.kind else entry.path
             for entry in current[:_MAX]]), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write recent files to {}: {}", recent_path, exc)


def add(path: str, kind: str | None = None) -> None:
    """Put ``path`` first (once, up to 10 entries); ``kind`` is "project" or "dataset"."""
    abspath = _canonical_recent_path(path)
    current = entries()
    known = next((entry.kind for entry in current if entry.path == abspath), None)
    _write([RecentEntry(abspath, kind if kind in KINDS else known),
            *(entry for entry in current if entry.path != abspath)])


def set_kind(path: str, kind: str) -> None:
    """Store what ``path`` is, learned by :func:`check`, without moving it in the list."""
    if kind not in KINDS:
        return
    current = entries()
    if any(entry.path == path and entry.kind != kind for entry in current):
        _write([RecentEntry(entry.path, kind) if entry.path == path else entry
                for entry in current])


def remove(path: str) -> None:
    """Remove path from the list. No-op if not present."""
    abspath = _canonical_recent_path(path)
    _write([entry for entry in entries() if entry.path != abspath])


#: The last result of :func:`check` in this process: path -> (exists, kind).
_LAST_CHECK: dict[str, tuple[bool, str | None]] = {}


def check(recent: list[RecentEntry]) -> dict[str, tuple[bool, str | None]]:
    """Whether each recent file exists, and its kind when it was not known.

    File access, possibly over a slow network: run it in a background job, never on
    the UI thread. The result is also kept for :func:`last_check`.
    """
    from ._files import _is_project_file

    result: dict[str, tuple[bool, str | None]] = {}
    for entry in recent:
        try:
            exists = os.path.exists(entry.path)
        except OSError:
            exists = False
        kind = entry.kind
        if exists and kind is None:
            kind = "project" if _is_project_file(entry.path) else "dataset"
        result[entry.path] = (exists, kind)
    _LAST_CHECK.update(result)
    return result


def last_check() -> dict[str, tuple[bool, str | None]]:
    """The last :func:`check` result in this process (empty before the first one)."""
    return dict(_LAST_CHECK)


def _canonical_recent_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if sys.platform != "win32" and expanded.startswith("/"):
        return os.path.normpath(expanded).replace("\\", "/")
    return os.path.abspath(expanded)
