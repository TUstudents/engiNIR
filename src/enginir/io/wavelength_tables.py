"""Wavelength tables: the wavelengths of a camera setting, for files that store none.

Some files hold only band numbers, e.g. the ``stream_hsi.h5`` scans of the
Kestreleye software. A wavelength table gives their bands wavelengths in nm.
Its values are copied into the project when a dataset is added with it
(:meth:`enginir.Project.add_dataset`) or given it later
(:meth:`enginir.Dataset.set_wavelengths`), so a project never depends on a
table file on one computer.

A table is a CSV file: comment lines ``# key: value`` (``name`` is required,
``camera`` is shown), then the header ``band,wavelength_nm`` and one row per
band, numbered from 0, with strictly increasing wavelengths. The tables that
ship with engiNIR are listed by :func:`wavelength_tables`, together with the
user's own in ``~/.enginir/wavelength_tables/`` (``%USERPROFILE%\\.enginir\\
wavelength_tables\\`` on Windows).
"""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .._checks import log_and_raise
from .._logging import get_logger

logger = get_logger(__name__)

_BUILT_IN = Path(__file__).resolve().parent.parent / "_resources" / "wavelength_tables"


@dataclass(frozen=True)
class WavelengthTable:
    """A named wavelength axis for files whose bands have no wavelengths.

    Attributes
    ----------
    name:
        The table's name, e.g. ``"Kestreleye 176 bands"``; shown and stored.
    wavelengths:
        Wavelength of each band in nm, float32 as a project stores it,
        strictly increasing.
    camera:
        The camera and setting it belongs to (shown; may be ``""``).
    path:
        The CSV file it was read from.
    """

    name: str
    wavelengths: np.ndarray = field(repr=False)
    camera: str = ""
    path: str = ""

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        return int(self.wavelengths.size)

    @property
    def table_id(self) -> str:
        """Content id: the same values give the same id, changed values a new one."""
        digest = hashlib.sha256(np.ascontiguousarray(self.wavelengths, dtype="<f4").tobytes())
        return f"wlt-{digest.hexdigest()[:16]}"

    def describe(self) -> str:
        """``"Kestreleye 176 bands (1041.75-1589.5 nm, EVK-Helios-EC-32 ...)"``."""
        span = f"{self.wavelengths[0]:g}-{self.wavelengths[-1]:g} nm"
        return f"{self.name} ({span}{', ' + self.camera if self.camera else ''})"


def load_wavelength_table(path: str | os.PathLike[str]) -> WavelengthTable:
    """Read a wavelength table from its CSV file (format in the module description).

    Raises
    ------
    ValueError
        If the file has no name, no ``band,wavelength_nm`` header, bands not
        numbered 0, 1, 2, ..., or wavelengths that are not finite, positive and
        strictly increasing.
    """
    path = Path(path)
    keys: dict[str, str] = {}
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            key, colon, value = text[1:].partition(":")
            if colon and key.strip() and " " not in key.strip():
                keys.setdefault(key.strip().lower(), value.strip())
            continue
        rows.append(next(csv.reader([text])))
    name = keys.get("name", "")
    if not name:
        log_and_raise("Wavelength table {} has no '# name: ...' line", path)
    if not rows or [cell.strip().lower() for cell in rows[0]] != ["band", "wavelength_nm"]:
        log_and_raise("Wavelength table {} needs the header 'band,wavelength_nm'", path)
    try:
        bands = [int(row[0]) for row in rows[1:]]
        values = np.array([float(row[1]) for row in rows[1:]], dtype=np.float64)
    except (IndexError, ValueError) as exc:
        log_and_raise("Wavelength table {} has a row that is not 'band,wavelength_nm': {}",
                      path, exc)
    if len(bands) < 2 or bands != list(range(len(bands))):
        log_and_raise("Wavelength table {} must number at least two bands 0, 1, 2, ... in "
                      "order", path)
    if not np.all(np.isfinite(values)) or values.min() <= 0 or np.any(np.diff(values) <= 0):
        log_and_raise("Wavelength table {}: the wavelengths must be positive and strictly "
                      "increasing", path)
    axis = values.astype(np.float32)
    if np.any(np.diff(axis) <= 0):
        log_and_raise("Wavelength table {}: two wavelengths are equal in float32", path)
    axis.setflags(write=False)
    return WavelengthTable(name=name, wavelengths=axis, camera=keys.get("camera", ""),
                           path=str(path))


def is_band_numbers(wavelengths: object) -> bool:
    """Whether an axis is the band numbers 0, 1, ..., n - 1 of a file without wavelengths."""
    axis = np.asarray(wavelengths, dtype=np.float64).reshape(-1)
    return axis.size > 1 and bool(np.allclose(axis, np.arange(axis.size), atol=1e-6, rtol=0.0))


def user_table_folder() -> Path:
    """The folder of the user's own tables: ``~/.enginir/wavelength_tables/``."""
    return Path.home() / ".enginir" / "wavelength_tables"


def wavelength_tables(n_bands: int | None = None) -> list[WavelengthTable]:
    """The tables that ship with engiNIR, then the user's; only those of ``n_bands`` if given.

    A user table that cannot be read, or has the name of an earlier table, is
    skipped with a warning.
    """
    tables: dict[str, WavelengthTable] = {}
    for folder, built_in in ((_BUILT_IN, True), (user_table_folder(), False)):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.csv")):
            try:
                table = load_wavelength_table(path)
            except (OSError, ValueError) as exc:
                if built_in:
                    raise
                logger.warning("Wavelength table {} is skipped: {}", path, exc)
                continue
            if table.name in tables:
                logger.warning("Wavelength table {} is skipped: '{}' is already the name of {}",
                               path, table.name, tables[table.name].path)
                continue
            tables[table.name] = table
    return [table for table in tables.values() if n_bands is None or table.n_bands == n_bands]


def wavelength_table(table: str | os.PathLike[str] | WavelengthTable) -> WavelengthTable:
    """A table by name (:func:`wavelength_tables`) or CSV path; a table is returned as is."""
    if isinstance(table, WavelengthTable):
        return table
    text = os.fspath(table)
    for known in wavelength_tables():
        if known.name == text:
            return known
    if text.lower().endswith(".csv") and Path(text).is_file():
        return load_wavelength_table(text)
    log_and_raise("Unknown wavelength table '{}'; known: {}", text,
                  ", ".join(repr(t.name) for t in wavelength_tables()) or "none", error=KeyError)


__all__ = ["WavelengthTable", "is_band_numbers", "load_wavelength_table", "user_table_folder",
           "wavelength_table", "wavelength_tables"]
