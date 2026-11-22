"""CSV and Excel (``.xlsx``) reader and CSV writer for spectra tables."""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from enginir._checks import log_and_raise
from enginir._logging import get_logger

from ._common import base_meta, require_file
from ._types import SpectralData

logger = get_logger(__name__)


def load_csv(
    path: str | os.PathLike[str],
    wavelength_row: int = 0,
    sample_col: str | int | None = None,
    sheet_name: str | int = 0,
    x_units: str = "nm",
) -> SpectralData:
    """Load a table of spectra with one spectrum per row from CSV or Excel.

    Expected layout (the header row holds the spectral axis)::

        sample_id, 1000.0, 1002.0, ..., 2500.0
        sample_1,    0.42,   0.41, ...,   0.38
        sample_2,    0.51,   0.50, ...,   0.47

    Further columns between the sample column and the spectra whose header is
    not a number (e.g. ``roi_name``, ``data_file`` written by
    :func:`save_csv`) are kept as text in ``meta["sample_columns"]``.

    Parameters
    ----------
    path:
        ``.csv`` or ``.xlsx`` file.
    wavelength_row:
        0-based row index of the header row with the spectral axis values.
    sample_col:
        Name or 0-based index of the column with sample identifiers. If
        ``None`` (default), the first column is used when its header is not a
        number; otherwise samples are numbered from 0.
    sheet_name:
        Sheet name or index for Excel files; ignored for CSV.
    x_units:
        Unit of the header values, stored in ``meta["x_units"]``.
        Default ``"nm"``.

    Returns
    -------
    SpectralData
        ``X`` of shape ``(n_samples, n_wavelengths)`` (float64), the header
        values as ``wavelengths``, and ``meta`` with the common keys (see
        :class:`~enginir.io.SpectralData`) plus ``"sample_names"`` and
        ``"sample_columns"`` (``{header: values}``, possibly empty).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the header or a cell cannot be read as a number.
    """
    path = require_file(path)
    if path.suffix.lower() == ".xlsx":
        rows = _read_xlsx(path, sheet_name)
    else:
        rows = _read_csv(path)
    if len(rows) <= wavelength_row:
        log_and_raise("{} has no header row {} (it has {} rows)", path, wavelength_row, len(rows))
    headers: list[Any] = ["" if cell is None else cell for cell in rows[wavelength_row]]
    body = rows[wavelength_row + 1:]
    if any(len(row) > len(headers) for row in body):
        log_and_raise("{} has rows with more cells than its header", path)
    columns = [[row[index] if index < len(row) else None for row in body]
               for index in range(len(headers))]

    if sample_col is None and headers and not _is_number(headers[0]):
        sample_col = 0
    if sample_col is None:
        sample_names = [str(index) for index in range(len(body))]
    else:
        if isinstance(sample_col, int):
            if not 0 <= sample_col < len(headers):
                log_and_raise(
                    "sample_col index {} is outside the {} columns of {}",
                    sample_col,
                    len(headers),
                    path,
                )
            position = sample_col
        elif sample_col in headers:
            position = headers.index(sample_col)
        else:
            log_and_raise("sample_col {!r} is not a column of {}", sample_col, path)
        sample_names = _text_column(columns.pop(position))
        del headers[position]

    sample_columns: dict[str, list[str]] = {}
    while headers and not _is_number(headers[0]):
        sample_columns[str(headers.pop(0))] = _text_column(columns.pop(0))
    if not headers:
        log_and_raise("{} contains no spectral columns", path)
    try:
        wavelengths = np.array([float(header) for header in headers])
    except (TypeError, ValueError) as exc:
        log_and_raise("Header of {} contains non-numeric wavelengths", path, cause=exc)
    try:
        X = np.array([[_number(cell) for cell in column] for column in columns],
                     dtype=np.float64).T.reshape(len(body), len(headers))
    except (TypeError, ValueError) as exc:
        log_and_raise("{} contains non-numeric spectral values", path, cause=exc)

    meta = base_meta(path, "csv", x_units)
    meta["sample_names"] = sample_names
    meta["sample_columns"] = sample_columns
    logger.info("Loaded CSV: shape {} from {}", X.shape, path)
    return SpectralData(X, wavelengths, meta)


def save_csv(
    path: str | os.PathLike[str],
    X: np.ndarray,
    wavelengths: np.ndarray,
    sample_names: list[str] | None = None,
    columns: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Write spectra to a CSV file readable by :func:`load_csv`.

    Parameters
    ----------
    path:
        Output file.
    X:
        Spectra, shape ``(n_samples, n_wavelengths)``.
    wavelengths:
        Spectral axis, shape ``(n_wavelengths,)``; written as the header row.
    sample_names:
        Optional sample identifiers, written as a first ``sample_id`` column.
    columns:
        Optional text columns ``{header: values}`` written after it, e.g. a
        label next to an id; :func:`load_csv` returns them in
        ``meta["sample_columns"]``. Headers must not be numbers.
    """
    X = np.asarray(X)
    wavelengths = np.asarray(wavelengths)
    if X.ndim != 2:
        log_and_raise("X must be 2-D, got shape {}", X.shape)
    if wavelengths.shape != (X.shape[1],):
        log_and_raise(
            "wavelengths must have shape ({},) to match X, got {}", X.shape[1], wavelengths.shape
        )
    text_columns: dict[str, Sequence[str]] = {}
    if sample_names is not None:
        if len(sample_names) != X.shape[0]:
            log_and_raise(
                "sample_names has {} entries but X has {} rows", len(sample_names), X.shape[0]
            )
        text_columns["sample_id"] = sample_names
    for header, values in (columns or {}).items():
        if _is_number(header) or len(values) != X.shape[0]:
            log_and_raise("Column {!r} needs a non-numeric header and {} values, got {}",
                          header, X.shape[0], len(values))
        text_columns[str(header)] = values
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator=os.linesep)
        writer.writerow([*text_columns, *(str(w) for w in wavelengths)])
        for index, row in enumerate(X):
            writer.writerow([*(values[index] for values in text_columns.values()),
                             *(_cell_text(value) for value in row)])
    logger.info("Saved CSV: shape {} to {}", X.shape, path)


#: Cells read as missing (NaN), as pandas reads them.
_MISSING = frozenset({"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
                      "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a",
                      "nan", "null"})


def _read_csv(path: Path) -> list[list[Any]]:
    """The rows of a CSV file as text cells; blank lines are skipped, a BOM is dropped."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return [row for row in csv.reader(file) if row]


def _read_xlsx(path: Path, sheet_name: str | int) -> list[list[Any]]:
    """The rows of one worksheet as cell values (``None`` for empty cells)."""
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if isinstance(sheet_name, int):
            if not 0 <= sheet_name < len(workbook.worksheets):
                log_and_raise("Worksheet {} not found: {} has {} sheets", sheet_name, path,
                              len(workbook.worksheets))
            sheet = workbook.worksheets[sheet_name]
        elif sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
        else:
            log_and_raise("Worksheet named {!r} not found", sheet_name)
        rows = [list(row) for row in sheet.iter_rows(values_only=True)]
    finally:
        workbook.close()
    return rows


def _is_missing(cell: Any) -> bool:
    return cell is None or (isinstance(cell, str) and cell.strip() in _MISSING)


def _number(cell: Any) -> float:
    return np.nan if _is_missing(cell) else float(cell)


def _text_column(cells: list[Any]) -> list[str]:
    """Cells of a text column as pandas gave them: a column of numbers is read
    as numbers (``"001"`` gives ``"1"``; ``"1.0"`` once a cell is missing or
    not whole), a missing cell is ``"nan"``."""
    present = [cell for cell in cells if not _is_missing(cell)]
    if present and all(_is_int(cell) for cell in present) and len(present) == len(cells):
        return [str(int(cell)) for cell in cells]
    if present and all(_is_number(cell) for cell in present):
        return [str(_number(cell)) for cell in cells]
    return ["nan" if _is_missing(cell) else str(cell) for cell in cells]


def _is_int(cell: Any) -> bool:
    if isinstance(cell, bool):
        return False
    if isinstance(cell, int):
        return True
    if isinstance(cell, str):
        try:
            int(cell)
        except ValueError:
            return False
        return True
    return False


def _cell_text(value: Any) -> str:
    """A number as pandas writes it: shortest repr of its own dtype, NaN empty."""
    if isinstance(value, np.floating) and np.isnan(value):
        return ""
    return str(value)


def _is_number(value: object) -> bool:
    try:
        float(str(value))
    except ValueError:
        return False
    return True


__all__ = ["load_csv", "save_csv"]
