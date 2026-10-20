"""CSV and Excel (``.xlsx``) reader and CSV writer for spectra tables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
        :class:`~enginir.io.SpectralData`) plus ``"sample_names"``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the header or a cell cannot be read as a number.
    """
    path = require_file(path)
    if path.suffix.lower() == ".xlsx":
        table = pd.read_excel(path, sheet_name=sheet_name, header=wavelength_row)
    else:
        table = pd.read_csv(path, header=wavelength_row)

    if sample_col is None and not _is_number(table.columns[0]):
        sample_col = 0
    if sample_col is None:
        sample_names = [str(index) for index in table.index]
    else:
        column: Any = sample_col
        if isinstance(sample_col, int):
            if not 0 <= sample_col < len(table.columns):
                log_and_raise(
                    "sample_col index {} is outside the {} columns of {}",
                    sample_col,
                    len(table.columns),
                    path,
                )
            column = table.columns[sample_col]
        if column not in table.columns:
            log_and_raise("sample_col {!r} is not a column of {}", column, path)
        sample_names = [str(name) for name in table[column]]
        table = table.drop(columns=[column])

    if table.shape[1] == 0:
        log_and_raise("{} contains no spectral columns", path)
    try:
        wavelengths = np.array([float(column) for column in table.columns])
    except (TypeError, ValueError) as exc:
        log_and_raise("Header of {} contains non-numeric wavelengths", path, cause=exc)
    try:
        X = table.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        log_and_raise("{} contains non-numeric spectral values", path, cause=exc)

    meta = base_meta(path, "csv", x_units)
    meta["sample_names"] = sample_names
    logger.info("Loaded CSV: shape {} from {}", X.shape, path)
    return SpectralData(X, wavelengths, meta)


def save_csv(
    path: str | os.PathLike[str],
    X: np.ndarray,
    wavelengths: np.ndarray,
    sample_names: list[str] | None = None,
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
    """
    X = np.asarray(X)
    wavelengths = np.asarray(wavelengths)
    if X.ndim != 2:
        log_and_raise("X must be 2-D, got shape {}", X.shape)
    if wavelengths.shape != (X.shape[1],):
        log_and_raise(
            "wavelengths must have shape ({},) to match X, got {}", X.shape[1], wavelengths.shape
        )
    table = pd.DataFrame(X, columns=[str(w) for w in wavelengths])
    if sample_names is not None:
        if len(sample_names) != X.shape[0]:
            log_and_raise(
                "sample_names has {} entries but X has {} rows", len(sample_names), X.shape[0]
            )
        table.insert(0, "sample_id", sample_names)
    table.to_csv(Path(path), index=False)
    logger.info("Saved CSV: shape {} to {}", X.shape, path)


def _is_number(value: object) -> bool:
    try:
        float(str(value))
    except ValueError:
        return False
    return True


__all__ = ["load_csv", "save_csv"]
