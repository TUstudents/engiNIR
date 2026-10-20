"""Helpers shared by the file-format readers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, BinaryIO

from enginir._checks import log_and_raise


class EnginirIOError(ValueError):
    """Base class of all errors raised for invalid or unsupported data files."""


def require_file(path: str | os.PathLike[str]) -> Path:
    """Return ``path`` as a :class:`~pathlib.Path`, raising if it is not an existing file."""
    resolved = Path(path)
    if not resolved.is_file():
        log_and_raise("No such file: {}", resolved, error=FileNotFoundError, depth=2)
    return resolved


def read_exact(
    f: BinaryIO,
    n_bytes: int,
    what: str,
    error: type[Exception] = EnginirIOError,
) -> bytes:
    """Read exactly ``n_bytes`` from ``f`` or raise ``error`` (the file is truncated)."""
    offset = f.tell()
    data = f.read(n_bytes)
    if len(data) < n_bytes:
        log_and_raise(
            "Expected {} bytes of {} at offset {}, got {} (file truncated?)",
            n_bytes,
            what,
            offset,
            len(data),
            error=error,
            depth=2,
        )
    return data


def base_meta(path: Path, file_format: str, x_units: str) -> dict[str, Any]:
    """Return the metadata keys every loader provides.

    ``source_path`` (absolute path), ``format`` (loader name, e.g. ``"spc"``)
    and ``x_units`` (unit of the returned spectral axis, e.g. ``"nm"`` or
    ``"cm-1"``; ``"unknown"`` if the file does not say).
    """
    return {"source_path": str(path.resolve()), "format": file_format, "x_units": x_units}


__all__ = ["EnginirIOError", "base_meta", "read_exact", "require_file"]
