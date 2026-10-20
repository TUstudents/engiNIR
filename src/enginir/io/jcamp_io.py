"""Reader for JCAMP-DX spectra (``.dx`` / ``.jdx``).

Supported: JCAMP-DX 4.24 and 5.x files with uncompressed (AFFN) tables in the
``(X++(Y..Y))`` or ``(XY..XY)`` form, single blocks and compound files whose
data blocks share one X axis. Not supported: compressed ASDF tables
(SQZ/DIF/DUP characters) and peak tables; such files raise
:class:`JCAMPError`.

References
----------
R. S. McDonald, P. A. Wilks, "JCAMP-DX: a standard form for exchange of
infrared spectra in computer readable form", *Applied Spectroscopy* 42 (1988)
151-162.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from enginir._checks import log_and_raise
from enginir._logging import get_logger

from ._common import EnginirIOError, base_meta, require_file
from ._types import SpectralData

logger = get_logger(__name__)

_X_UNITS = {"1/CM": "cm-1", "NANOMETERS": "nm", "MICROMETERS": "um", "SECONDS": "s"}


class JCAMPError(EnginirIOError):
    """Raised for invalid or unsupported JCAMP-DX files."""


@dataclass
class _Block:
    """Labelled data records (LDRs) and table lines of one JCAMP-DX block."""

    labels: dict[str, str] = field(default_factory=dict)
    table_format: str | None = None
    table_lines: list[str] = field(default_factory=list)

    def number(self, label: str) -> float | None:
        value = self.labels.get(label)
        if value is None:
            return None
        try:
            return float(value.split()[0])
        except (ValueError, IndexError):
            log_and_raise("JCAMP-DX label ##{} has non-numeric value {!r}", label, value,
                          error=JCAMPError)


def load_jcamp(path: str | os.PathLike[str]) -> SpectralData:
    """Load the spectra of a JCAMP-DX file.

    Parameters
    ----------
    path:
        ``.dx`` or ``.jdx`` file.

    Returns
    -------
    SpectralData
        ``X`` of shape ``(n_blocks, n_points)``, the X axis as
        ``wavelengths`` in ``meta["x_units"]`` (``"cm-1"``, ``"nm"``,
        ``"um"``, ...), and ``meta`` with the common keys plus the labelled
        data records of the first block (upper-case keys such as ``"TITLE"``,
        ``"XUNITS"``, ``"YUNITS"``).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    JCAMPError
        If the file has no readable data block, uses compressed tables, or
        its blocks have different X axes.
    """
    path = require_file(path)
    text = path.read_text(encoding="latin-1")
    blocks = [_parse_block(block_text) for block_text in _split_blocks(text)]
    if not blocks:
        log_and_raise("No JCAMP-DX data block found in {}", path, error=JCAMPError)

    spectra: list[np.ndarray] = []
    axis: np.ndarray | None = None
    first_block: _Block | None = None
    for number, block in enumerate(blocks, start=1):
        parsed = _read_table(block)
        if parsed is None:
            logger.warning(
                "Skipping JCAMP-DX block {} in {} with unsupported table {!r}",
                number,
                path,
                block.table_format,
            )
            continue
        x, y = parsed
        if axis is None:
            axis, first_block = x, block
        elif x.shape != axis.shape or not np.allclose(x, axis, atol=1e-6, rtol=0.0):
            log_and_raise(
                "JCAMP-DX block {} in {} has a different X axis than the first block",
                number,
                path,
                error=JCAMPError,
            )
        spectra.append(y)
    if axis is None or first_block is None:
        log_and_raise("No supported JCAMP-DX data table found in {}", path, error=JCAMPError)

    x_units = _X_UNITS.get(first_block.labels.get("XUNITS", "").upper(), "unknown")
    meta = base_meta(path, "jcamp", x_units)
    meta.update(first_block.labels)
    X = np.vstack(spectra)
    logger.info("Loaded JCAMP-DX: {} spectra x {} points from {}", X.shape[0], X.shape[1], path)
    return SpectralData(X, axis, meta)


def _split_blocks(text: str) -> list[str]:
    """Split a (compound) file at ``##END=`` and keep the blocks that contain a data table."""
    parts = re.split(r"##END\s*=", text, flags=re.IGNORECASE)
    return [part for part in parts if re.search(r"##(XYDATA|XYPOINTS)\s*=", part, re.IGNORECASE)]


def _parse_block(text: str) -> _Block:
    """Collect the labels and table lines of one block, dropping ``$$`` comments."""
    block = _Block()
    in_table = False
    for raw_line in text.splitlines():
        line = raw_line.split("$$", 1)[0].strip()
        if not line:
            continue
        if line.startswith("##"):
            label, _, value = line[2:].partition("=")
            label = label.strip().upper()
            in_table = label in ("XYDATA", "XYPOINTS")
            if in_table:
                block.table_format = value.strip().upper().replace(" ", "")
            else:
                block.labels[label] = value.strip()
        elif in_table:
            block.table_lines.append(line)
    return block


def _read_table(block: _Block) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the ``(x, y)`` arrays of a block, or ``None`` for unsupported tables."""
    x_factor = block.number("XFACTOR") or 1.0
    y_factor = block.number("YFACTOR") or 1.0
    if block.table_format == "(X++(Y..Y))":
        return _read_x_plus_y(block, x_factor, y_factor)
    if block.table_format == "(XY..XY)":
        values = np.array([v for line in block.table_lines for v in _numbers(line)])
        if values.size % 2:
            log_and_raise("(XY..XY) table has an odd number of values", error=JCAMPError)
        return values[0::2] * x_factor, values[1::2] * y_factor
    return None


def _read_x_plus_y(
    block: _Block, x_factor: float, y_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Read an ``(X++(Y..Y))`` table; every line is ``X Y1 Y2 ...`` with X the abscissa of Y1."""
    line_checks: list[tuple[int, float]] = []  # (index of first Y on the line, X check value)
    y_values: list[float] = []
    for line in block.table_lines:
        numbers = _numbers(line)
        if len(numbers) < 2:
            continue
        line_checks.append((len(y_values), numbers[0] * x_factor))
        y_values.extend(numbers[1:])
    n_points = len(y_values)
    declared = block.number("NPOINTS")
    if declared is not None and int(declared) != n_points:
        log_and_raise(
            "##NPOINTS={} but the table contains {} Y values", int(declared), n_points,
            error=JCAMPError,
        )
    first_x, last_x = block.number("FIRSTX"), block.number("LASTX")
    if first_x is None or last_x is None:
        log_and_raise("(X++(Y..Y)) table requires ##FIRSTX and ##LASTX", error=JCAMPError)
    # FIRSTX/LASTX are in real units; XFACTOR only scales the tabulated X values.
    x = np.linspace(first_x, last_x, n_points)
    step = abs(last_x - first_x) / max(n_points - 1, 1)
    deviations = [abs(x[index] - check) for index, check in line_checks]
    if deviations and max(deviations) > 0.5 * step:
        logger.warning(
            "JCAMP-DX X check values deviate from the FIRSTX/LASTX axis by up to {:.3g} "
            "(point spacing {:.3g})",
            max(deviations),
            step,
        )
    return x, np.asarray(y_values) * y_factor


def _numbers(line: str) -> list[float]:
    """Parse the space- or comma-separated numbers of an uncompressed (AFFN) table line."""
    values: list[float] = []
    for token in re.split(r"[\s,]+", line):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            log_and_raise(
                "Unsupported JCAMP-DX token {!r}: compressed (SQZ/DIF/DUP) tables are not "
                "supported",
                token[:40],
                error=JCAMPError,
            )
    return values


__all__ = ["JCAMPError", "load_jcamp"]
