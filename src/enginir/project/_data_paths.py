"""Where a dataset's data file may be: stored paths read on any platform.

A project moves between computers and platforms; the same share is
``/mnt/z/…`` in WSL and ``Z:\\…`` on Windows. These functions only compute
paths (no file is opened); :meth:`enginir.Project.find_data_file` tries them
in order (``project-dataset-attachment-contract_v3.md``, Missing and moved
files).
"""

from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import PurePath, PurePosixPath, PureWindowsPath

from enginir._logging import get_logger

logger = get_logger(__name__)

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:([\\/]|$)")
_DEFAULT_WSL_ROOT = "/mnt/"
_WSL_CONF = "/etc/wsl.conf"

#: How a candidate was derived, for the log ("found next to the project").
RELATIVE = "next to the project"
STORED = "at its stored path"
#: Followed by the drive letter: "via drive Z:".
DRIVE = "via drive"


@dataclass(frozen=True, slots=True)
class DataFileCandidate:
    """A path where a dataset's data file may be, and how it was derived."""

    path: str
    how: str


def stored_path(text: str) -> PurePath:
    """A stored path read by its own syntax, whatever the platform reading it.

    A drive letter or a ``\\\\`` prefix makes it a Windows path; anything else
    is a POSIX path when it starts with ``/``. A relative path follows its
    separators (backslashes: Windows).
    """
    if _WINDOWS_DRIVE.match(text) or text.startswith("\\\\"):
        return PureWindowsPath(text)
    if text.startswith("/"):
        return PurePosixPath(text)
    return PureWindowsPath(text) if "\\" in text else PurePosixPath(text)


def native_path(path: PurePath, *, windows: bool | None = None) -> str | None:
    """``path`` as this platform names it, or ``None`` if it is the other platform's path."""
    windows = os.name == "nt" if windows is None else windows
    if isinstance(path, PureWindowsPath) == windows:
        return str(path)
    return None


@cache
def wsl_mount_root(conf: str = _WSL_CONF) -> str:
    """The WSL automount root (``[automount] root`` of ``/etc/wsl.conf``), default ``/mnt/``."""
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(conf, encoding="utf-8")
        root = parser.get("automount", "root", fallback=_DEFAULT_WSL_ROOT)
    except (configparser.Error, OSError, UnicodeDecodeError) as exc:
        logger.warning("Cannot read {} ({}); the WSL mount root is {}", conf, exc,
                       _DEFAULT_WSL_ROOT)
        return _DEFAULT_WSL_ROOT
    root = root.strip().strip('"') or _DEFAULT_WSL_ROOT
    return root if root.endswith("/") else root + "/"


def drive_mapped(path: PurePath, *, windows: bool | None = None,
                 wsl_root: str | None = None) -> str | None:
    """``path`` moved between a Windows drive and its WSL mount, or ``None``.

    ``Z:\\data\\x.h5`` read on Linux is ``<root>z/data/x.h5``; ``/mnt/z/data/x.h5``
    read on Windows is ``Z:\\data\\x.h5``. On Windows the WSL root of the
    writer is unknown, so both ``/mnt/z/…`` and ``/z/…`` are mapped.
    """
    windows = os.name == "nt" if windows is None else windows
    if windows:
        if not isinstance(path, PurePosixPath):
            return None
        parts = path.parts  # ("/", "mnt", "z", ...) or ("/", "z", ...)
        offset = 2 if len(parts) > 2 and parts[1] == "mnt" else 1
        if len(parts) <= offset or not re.fullmatch(r"[A-Za-z]", parts[offset]):
            return None
        return str(PureWindowsPath(f"{parts[offset].upper()}:\\", *parts[offset + 1:]))
    if not isinstance(path, PureWindowsPath) or not re.fullmatch(r"[A-Za-z]:", path.drive):
        return None
    root = wsl_mount_root() if wsl_root is None else wsl_root
    return str(PurePosixPath(root, path.drive[0].lower(), *path.parts[1:]))


def relative_data_path(project_path: str | os.PathLike[str],
                       data_path: str | os.PathLike[str]) -> str:
    """The data file's path relative to the project file's folder, with ``/``; ``""`` if none.

    On Windows there is no relative path between two drives.
    """
    folder = os.path.dirname(os.path.abspath(os.fspath(project_path)))
    try:
        relative = os.path.relpath(os.path.abspath(os.fspath(data_path)), folder)
    except ValueError:
        return ""
    return PurePath(relative).as_posix()


def data_file_candidates(project_path: str | os.PathLike[str], source_path: str,
                         relative_path: str = "", *, windows: bool | None = None,
                         wsl_root: str | None = None) -> list[DataFileCandidate]:
    """Where to look for a data file, in order: relative, stored, drive-mapped.

    Paths that this platform cannot name are left out, and a path is listed
    once (the first way it was derived).
    """
    windows = os.name == "nt" if windows is None else windows
    found: list[DataFileCandidate] = []
    if relative_path:
        folder = os.path.dirname(os.path.abspath(os.fspath(project_path)))
        found.append(DataFileCandidate(
            os.path.normpath(os.path.join(folder, *relative_path.split("/"))), RELATIVE))
    if source_path:
        stored = stored_path(source_path)
        native = native_path(stored, windows=windows)
        if native is not None:
            found.append(DataFileCandidate(native, STORED))
        mapped = drive_mapped(stored, windows=windows, wsl_root=wsl_root)
        if mapped is not None:
            letter = (mapped if windows else stored.drive)[0].upper()
            found.append(DataFileCandidate(mapped, f"{DRIVE} {letter}:"))
    unique: list[DataFileCandidate] = []
    for candidate in found:
        if not any(same_path(candidate.path, other.path) for other in unique):
            unique.append(candidate)
    return unique


def same_path(left: str, right: str) -> bool:
    """Whether two paths name the same file (case-insensitive on Windows)."""
    return os.path.normcase(os.path.normpath(left)) == os.path.normcase(os.path.normpath(right))


__all__ = [
    "DRIVE",
    "RELATIVE",
    "STORED",
    "DataFileCandidate",
    "data_file_candidates",
    "drive_mapped",
    "moved_path",
    "native_path",
    "relative_data_path",
    "same_path",
    "stored_path",
    "wsl_mount_root",
]


def moved_path(old_file: str, new_file: str, other: str) -> str | None:
    """``other`` moved the way ``old_file`` moved to ``new_file``, or ``None``.

    The folder change is what remains after the common end of both paths
    (compared case-insensitively): ``/mnt/z/data/run1/x.h5`` →
    ``E:\\copy\\data\\run1\\x.h5`` moves ``/mnt/z`` to ``E:\\copy``. ``other`` must be
    under the old folder. Stored paths are read by their own syntax, and the
    result is a path of this platform.
    """
    old, new, moved = (stored_path(old_file).parts, PurePath(new_file).parts,
                       stored_path(other).parts)
    common = 0
    while (common < min(len(old), len(new)) - 1
           and old[-1 - common].lower() == new[-1 - common].lower()):
        common += 1
    if common == 0:
        return None
    old_folder, new_folder = old[:len(old) - common], new[:len(new) - common]
    if len(moved) <= len(old_folder) or [p.lower() for p in moved[:len(old_folder)]] != [
            p.lower() for p in old_folder]:
        return None
    return str(PurePath(*new_folder, *moved[len(old_folder):]))
