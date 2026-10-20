"""Reader and writer for ENVI image cubes (``.hdr`` header plus binary data file).

Supported: data types 1, 2, 3, 4, 5, 12 and 13, BSQ/BIL/BIP interleave,
either byte order and a header offset. The header format is described in the
ENVI documentation ("ENVI Header Files", NV5 Geospatial / L3Harris).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import numpy as np

from enginir._checks import check_integer, log_and_raise
from enginir._logging import get_logger

from ._common import EnginirIOError, base_meta
from ._types import SpectralData

logger = get_logger(__name__)

#: ENVI ``data type`` code -> numpy dtype
_ENVI_DTYPES = {
    1: np.dtype("uint8"),
    2: np.dtype("int16"),
    3: np.dtype("int32"),
    4: np.dtype("float32"),
    5: np.dtype("float64"),
    12: np.dtype("uint16"),
    13: np.dtype("uint32"),
}
_DTYPE_CODES = {dtype: code for code, dtype in _ENVI_DTYPES.items()}
_DATA_SUFFIXES = (".img", ".dat", ".raw", ".bil", ".bsq", ".bip", "")
#: ``wavelength units`` header value -> factor converting to nm
_TO_NANOMETERS = {"nanometers": 1.0, "nm": 1.0, "micrometers": 1000.0, "um": 1000.0,
                  "microns": 1000.0}
_COMPUTED_KEYS = {"samples", "lines", "bands", "data type", "interleave", "byte order",
                  "header offset", "file type", "wavelength", "wavelength units",
                  # engiNIR metadata of the loaded file, not ENVI header fields
                  "source_path", "format", "x_units"}
#: Header fields whose values ENVI writes in braces; load_envi returns them without.
_BRACE_KEYS = {"band names", "bbl", "coordinate system string", "data gain values",
               "data offset values", "default bands", "description", "fwhm", "map info",
               "spectra names"}


class ENVIError(EnginirIOError):
    """Raised for invalid or unsupported ENVI files."""


def load_envi(path: str | os.PathLike[str], flatten: bool = False) -> SpectralData:
    """Load an ENVI image cube.

    Parameters
    ----------
    path:
        The ``.hdr`` header, or the binary data file (``.img``, ``.dat``,
        ``.raw``, ``.bil``, ``.bsq``, ``.bip`` or no extension) next to it.
        Headers may be named ``cube.hdr`` or ``cube.img.hdr``.
    flatten:
        If ``True``, return the pixels as rows, shape ``(lines * samples,
        bands)``.

    Returns
    -------
    SpectralData
        ``X`` of shape ``(lines, samples, bands)`` (rows, columns, bands) in
        the file's data type, ``wavelengths`` in nm (converted from
        micrometers if necessary; unconverted with ``meta["x_units"] ==
        "unknown"`` if the header does not state the units; empty if it has
        none), and ``meta`` with all header fields as strings (lower-case
        keys, braces removed) plus the common keys.

    Raises
    ------
    FileNotFoundError
        If the header or the data file is missing.
    ENVIError
        If the header is invalid or inconsistent with the data file.

    Notes
    -----
    The viewer expects push-broom cubes ``(n_frames, height, n_bands)``; ENVI
    cubes may need transposing first.
    """
    header_path, header, view = _map_envi(path)
    # The one copy that reorders the file to (lines, samples, bands) and
    # converts it to native byte order; the file is never read twice.
    cube = np.array(view, dtype=view.dtype.newbyteorder("="), order="C")
    del view
    bands = cube.shape[2]
    if flatten:
        cube = cube.reshape(-1, bands)

    wavelengths, x_units = _wavelengths_in_nm(header, bands, header_path)
    meta: dict = dict(header)
    meta.update(base_meta(header_path, "envi", x_units))
    logger.info("Loaded ENVI: shape {} from {}", cube.shape, header_path)
    return SpectralData(cube, wavelengths, meta)


def iter_envi_frame_chunks(
    path: str | os.PathLike[str], *, chunk_size: int
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield blocks of lines of an ENVI cube without reading the whole file.

    The data file is memory-mapped; each block is copied once to native byte
    order, so only one block is held in memory.

    Parameters
    ----------
    path:
        Header or data file, as in :func:`load_envi`.
    chunk_size:
        Maximum number of lines per block.

    Yields
    ------
    offset : int
        Index of the block's first line.
    block : numpy.ndarray, shape (n, samples, bands)
        Up to ``chunk_size`` lines in the file's data type.
    """
    chunk_size = check_integer(chunk_size, "chunk_size", minimum=1)
    _, _, view = _map_envi(path)
    native = view.dtype.newbyteorder("=")
    for offset in range(0, view.shape[0], chunk_size):
        yield offset, np.array(view[offset:offset + chunk_size], dtype=native, order="C")


def _map_envi(path: str | os.PathLike[str]) -> tuple[Path, dict[str, str], np.ndarray]:
    """Validate the header and return it with a ``(lines, samples, bands)`` memmap view."""
    header_path, data_path = _resolve_paths(Path(path))
    header = _parse_header(header_path)
    lines, samples, bands = (_header_int(header, key) for key in ("lines", "samples", "bands"))
    if min(lines, samples, bands) <= 0:
        log_and_raise("Invalid ENVI dimensions lines={}, samples={}, bands={}", lines, samples,
                      bands, error=ENVIError, depth=2)
    dtype = _ENVI_DTYPES.get(_header_int(header, "data type"))
    if dtype is None:
        log_and_raise("Unsupported ENVI data type {}; supported: {}", header["data type"],
                      sorted(_ENVI_DTYPES), error=ENVIError, depth=2)
    byte_order = _header_int(header, "byte order", default=0)
    if byte_order not in (0, 1):
        log_and_raise("ENVI byte order must be 0 or 1, got {}", byte_order, error=ENVIError,
                      depth=2)
    dtype = dtype.newbyteorder(">" if byte_order else "<")
    offset = _header_int(header, "header offset", default=0)

    expected = lines * samples * bands
    available = (data_path.stat().st_size - offset) // dtype.itemsize
    if available < expected:
        log_and_raise("{} holds {} values but the header declares {}", data_path, available,
                      expected, error=ENVIError, depth=2)
    if available > expected:
        logger.warning("{} holds {} values; ignoring {} values after the declared cube",
                       data_path, available, available - expected)
    mapped = np.memmap(data_path, dtype=dtype, mode="r", offset=offset, shape=(expected,))
    view = _to_rows_cols_bands(mapped, header.get("interleave", "bsq"), lines, samples, bands)
    return header_path, header, view


def save_envi(
    path: str | os.PathLike[str],
    X: np.ndarray,
    wavelengths: np.ndarray,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write an image cube or a matrix of spectra as an ENVI BIP file pair.

    Parameters
    ----------
    path:
        Output base name; ``<base>.hdr`` and ``<base>.img`` are written. A
        trailing ``.hdr``, ``.img`` or ``.dat`` is removed first.
    X:
        Cube ``(lines, samples, bands)``, or spectra ``(n, bands)`` written
        as ``n`` lines of one sample (they load back as ``(n, 1, bands)``).
    wavelengths:
        Wavelengths in nm, shape ``(bands,)``, or empty.
    meta:
        Further header fields, e.g. the ``meta`` of :func:`load_envi`.
        Sequences, and the values of ENVI list fields such as ``band names``
        or ``map info``, are written in braces. Fields computed from ``X``
        (dimensions, data type, interleave, ...) and engiNIR's
        ``source_path``, ``format`` and ``x_units`` are ignored.
    """
    X = np.asarray(X)
    wavelengths = np.asarray(wavelengths)
    if X.ndim not in (2, 3):
        log_and_raise("X must be 2-D or 3-D, got shape {}", X.shape, error=ENVIError)
    cube = X[:, np.newaxis, :] if X.ndim == 2 else X
    lines, samples, bands = cube.shape
    if wavelengths.ndim != 1 or wavelengths.size not in (0, bands):
        log_and_raise("wavelengths must be empty or have shape ({},), got {}", bands,
                      wavelengths.shape, error=ENVIError)
    code = _DTYPE_CODES.get(X.dtype.newbyteorder("="))
    if code is None:
        log_and_raise("Unsupported dtype {} for ENVI; supported: {}", X.dtype,
                      [str(dtype) for dtype in _ENVI_DTYPES.values()], error=ENVIError)

    base = Path(path)
    if base.suffix.lower() in (".hdr", ".img", ".dat"):
        base = base.with_suffix("")
    # Band-interleaved-by-pixel is the in-memory layout, so a C-contiguous
    # little-endian cube is written without a copy.
    np.ascontiguousarray(cube, dtype=X.dtype.newbyteorder("<")).tofile(f"{base}.img")

    header = [
        "ENVI",
        f"samples = {samples}",
        f"lines = {lines}",
        f"bands = {bands}",
        "header offset = 0",
        "file type = ENVI Standard",
        f"data type = {code}",
        "interleave = bip",
        "byte order = 0",
    ]
    if wavelengths.size:
        header += ["wavelength units = Nanometers",
                   "wavelength = {" + ", ".join(f"{w:.6f}" for w in wavelengths) + "}"]
    for key, value in (meta or {}).items():
        if key.lower() in _COMPUTED_KEYS:
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            value = "{" + ", ".join(str(item) for item in value) + "}"
        elif key.lower() in _BRACE_KEYS and not str(value).startswith("{"):
            value = "{" + str(value) + "}"
        header.append(f"{key} = {value}")
    Path(f"{base}.hdr").write_text("\n".join(header) + "\n", encoding="utf-8")
    logger.info("Saved ENVI: shape {} to {}.hdr/.img", cube.shape, base)


def _resolve_paths(path: Path) -> tuple[Path, Path]:
    """Return ``(header_path, data_path)`` for a header or data file path."""
    if path.suffix.lower() == ".hdr":
        header_path = path
        candidates = [path.with_suffix(suffix) for suffix in _DATA_SUFFIXES]
    elif path.suffix.lower() in _DATA_SUFFIXES:
        # Both "cube.hdr" and "cube.img.hdr" are common header names.
        headers = [path.with_suffix(".hdr"), path.with_name(path.name + ".hdr")]
        header_path = next((header for header in headers if header.is_file()), headers[0])
        candidates = [path]
    else:
        log_and_raise("Expected an ENVI header (.hdr) or data file ({}), got {}",
                      ", ".join(suffix for suffix in _DATA_SUFFIXES if suffix), path,
                      error=ENVIError)
    if not header_path.is_file():
        log_and_raise("ENVI header not found: {}", header_path, error=FileNotFoundError)
    data_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if data_path is None:
        log_and_raise("No ENVI data file found for {}", header_path, error=FileNotFoundError)
    return header_path, data_path


def _parse_header(path: Path) -> dict[str, str]:
    """Return the header fields with lower-case keys; ``{...}`` values are joined without braces."""
    fields: dict[str, str] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in re.finditer(r"^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", text, re.MULTILINE):
        key, value = match.group(1).strip().lower(), match.group(2).strip()
        if value.startswith("{"):
            value = " ".join(value[1:-1].split())
        fields[key] = value
    return fields


def _header_int(header: dict[str, str], key: str, default: int | None = None) -> int:
    value = header.get(key)
    if value is None:
        if default is None:
            log_and_raise("ENVI header has no '{}' field", key, error=ENVIError, depth=2)
        return default
    try:
        return int(value)
    except ValueError as exc:
        log_and_raise("ENVI header field '{}' is not an integer: {!r}", key, value,
                      error=ENVIError, cause=exc, depth=2)


def _wavelengths_in_nm(header: dict[str, str], bands: int,
                       path: Path) -> tuple[np.ndarray, str]:
    """Return the wavelengths in nm and ``"nm"``, or unconverted and ``"unknown"``.

    Values are kept unconverted (with a warning) when the header gives no
    ``wavelength units`` or ``Unknown``; other unknown units raise.
    """
    text = header.get("wavelength", "")
    try:
        values = np.array([float(token) for token in re.split(r"[,\s]+", text) if token])
    except ValueError as exc:
        log_and_raise("ENVI wavelength list in {} is not numeric", path, error=ENVIError,
                      cause=exc, depth=2)
    if values.size == 0:
        return values, "unknown"
    if values.size != bands:
        log_and_raise("{} lists {} wavelengths for {} bands", path, values.size, bands,
                      error=ENVIError, depth=2)
    units = header.get("wavelength units", "unknown").strip().lower()
    if units == "unknown":
        logger.warning("{} does not state its wavelength units; the values are kept as they "
                       "are (x_units='unknown')", path)
        return values, "unknown"
    factor = _TO_NANOMETERS.get(units)
    if factor is None:
        log_and_raise("Unsupported ENVI wavelength units {!r} in {}", units, path,
                      error=ENVIError, depth=2)
    return values * factor, "nm"


def _to_rows_cols_bands(
    flat: np.ndarray, interleave: str, lines: int, samples: int, bands: int
) -> np.ndarray:
    """Return a ``(lines, samples, bands)`` view of the data file contents."""
    interleave = interleave.strip().lower()
    if interleave == "bsq":
        return flat.reshape(bands, lines, samples).transpose(1, 2, 0)
    if interleave == "bil":
        return flat.reshape(lines, bands, samples).transpose(0, 2, 1)
    if interleave == "bip":
        return flat.reshape(lines, samples, bands)
    log_and_raise("Unknown ENVI interleave {!r}", interleave, error=ENVIError, depth=2)


__all__ = ["ENVIError", "iter_envi_frame_chunks", "load_envi", "save_envi"]
