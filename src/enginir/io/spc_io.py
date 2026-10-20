"""Reader for Galactic / Thermo Scientific SPC files (new format, version ``0x4B``).

Supported: single- and multi-spectrum files sharing one X axis (``TMULTI``),
evenly spaced or explicit X values (``TXVALS``), IEEE float32 and scaled
32-bit or 16-bit (``TSPREC``) integer Y values, and the log-block text. Not
supported: the old ``0x4D`` format and files with one X array per spectrum
(``TXYXYS``). The binary layout is described in ``docs/spc_format.md``.
"""

from __future__ import annotations

import datetime as dt
import os
import struct
from typing import Any, BinaryIO

import numpy as np

from enginir._checks import log_and_raise
from enginir._logging import get_logger

from ._common import EnginirIOError, base_meta, read_exact, require_file
from ._types import SpectralData

logger = get_logger(__name__)

_MAIN_HEADER_SIZE = 512
_SUB_HEADER_SIZE = 32
_VERSION_NEW = 0x4B

# ftflgs bit masks
_TSPREC = 0x01  # 16-bit integer Y values
_TMULTI = 0x04  # several spectra
_TXYXYS = 0x40  # every spectrum has its own X array
_TXVALS = 0x80  # explicit X array after the main header

_FEXP_FLOAT32 = -128  # fexp byte 0x80: Y values are IEEE float32

#: fxtype code -> (description, unit of the X axis)
_X_TYPES: dict[int, tuple[str, str]] = {
    0: ("arbitrary", "unknown"),
    1: ("wavenumber", "cm-1"),
    2: ("wavelength", "um"),
    3: ("wavelength", "nm"),
    4: ("time", "s"),
    5: ("time", "min"),
    6: ("frequency", "Hz"),
    7: ("frequency", "kHz"),
    8: ("frequency", "MHz"),
    9: ("mass", "m/z"),
    10: ("chemical shift", "ppm"),
    11: ("time", "d"),
    12: ("time", "a"),
    13: ("raman shift", "cm-1"),
    14: ("energy", "eV"),
}

#: fytype code -> description of the Y values
_Y_TYPES: dict[int, str] = {
    0: "arbitrary",
    1: "interferogram",
    2: "absorbance",
    3: "kubelka_munk",
    4: "counts",
    5: "volts",
    6: "degrees",
    7: "milliamps",
    8: "millimeters",
    9: "millivolts",
    10: "log_1_r",
    11: "percent",
    12: "intensity",
    13: "relative_intensity",
    14: "energy",
    16: "decibel",
    19: "temperature_f",
    20: "temperature_c",
    21: "temperature_k",
    22: "index_of_refraction",
    23: "extinction_coeff",
    24: "real",
    25: "imaginary",
    26: "complex",
    128: "transmission",
    129: "reflectance",
    130: "valley_absorbance",
    131: "emission",
}


class SPCError(EnginirIOError):
    """Raised for invalid or unsupported SPC files."""


def load_spc(path: str | os.PathLike[str]) -> SpectralData:
    """Load all spectra of an SPC file.

    Parameters
    ----------
    path:
        ``.spc`` file.

    Returns
    -------
    SpectralData
        ``X`` of shape ``(n_spectra, n_points)`` (float64; single-spectrum
        files give one row), the X axis as ``wavelengths`` in the unit
        ``meta["x_units"]``, and ``meta`` with the common keys plus
        ``"x_label"``, ``"y_label"``, ``"comment"``, ``"timestamp"``
        (ISO 8601, instrument local time, or ``None``), ``"n_spectra"``,
        ``"n_points"`` and, if present, ``"log"``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    SPCError
        If the file is truncated, malformed or uses an unsupported variant.
    """
    path = require_file(path)
    with open(path, "rb") as f:
        header = _parse_main_header(read_exact(f, _MAIN_HEADER_SIZE, "main header", SPCError))
        n_points, n_spectra, flags = header["fnpts"], header["fnsub"], header["ftflgs"]
        if flags & _TXYXYS:
            log_and_raise(
                "{}: SPC files with one X array per spectrum (TXYXYS) are not supported",
                path,
                error=SPCError,
            )
        if n_points == 0 or n_spectra == 0:
            log_and_raise(
                "{} declares {} points and {} spectra", path, n_points, n_spectra, error=SPCError
            )

        if flags & _TXVALS:
            x_data = read_exact(f, 4 * n_points, "X values", SPCError)
            wavelengths = np.frombuffer(x_data, dtype="<f4").astype(np.float64)
        else:
            wavelengths = np.linspace(header["ffirst"], header["flast"], n_points)

        bytes_per_value = 2 if flags & _TSPREC else 4
        spectra = np.empty((n_spectra, n_points), dtype=np.float64)
        for index in range(n_spectra):
            sub_header = read_exact(f, _SUB_HEADER_SIZE, f"sub-header {index}", SPCError)
            sub_exponent = struct.unpack_from("b", sub_header, 1)[0]
            sub_points = struct.unpack_from("<I", sub_header, 16)[0]
            if flags & _TMULTI and sub_points not in (0, n_points):
                log_and_raise(
                    "Spectrum {} has {} points but the file declares {}; "
                    "spectra of different lengths are not supported",
                    index,
                    sub_points,
                    n_points,
                    error=SPCError,
                )
            y_data = read_exact(f, bytes_per_value * n_points, f"Y values of spectrum {index}",
                                SPCError)
            # The sub-header exponent applies when the main header leaves it at 0.
            exponent = sub_exponent if header["fexp"] == 0 else header["fexp"]
            spectra[index] = _decode_y_values(y_data, exponent, sixteen_bit=bool(flags & _TSPREC))

        log_text = _read_log_text(f, header["flogoff"])

    description, x_units = _X_TYPES.get(header["fxtype"], (f"unknown({header['fxtype']})",
                                                           "unknown"))
    meta = base_meta(path, "spc", x_units)
    meta.update(
        x_label=description,
        y_label=_Y_TYPES.get(header["fytype"], f"unknown({header['fytype']})"),
        comment=header["comment"],
        timestamp=header["timestamp"],
        n_spectra=n_spectra,
        n_points=n_points,
    )
    if log_text:
        meta["log"] = log_text
    logger.info("Loaded SPC: {} spectra x {} points from {}", n_spectra, n_points, path)
    return SpectralData(spectra, wavelengths, meta)


def _parse_main_header(data: bytes) -> dict[str, Any]:
    """Decode the fields of the 512-byte main header that the reader uses."""
    version = data[1]
    if version != _VERSION_NEW:
        log_and_raise(
            "Unsupported SPC version 0x{:02X}; only the new format 0x4B is supported",
            version,
            error=SPCError,
            depth=2,
        )
    return {
        "ftflgs": data[0],
        "fexp": struct.unpack_from("b", data, 3)[0],
        "fnpts": struct.unpack_from("<I", data, 4)[0],
        "ffirst": struct.unpack_from("<d", data, 8)[0],
        "flast": struct.unpack_from("<d", data, 16)[0],
        "fnsub": struct.unpack_from("<I", data, 24)[0],
        "fxtype": data[28],
        "fytype": data[29],
        "timestamp": _decode_date(struct.unpack_from("<I", data, 32)[0]),
        "comment": data[88:218].split(b"\x00", 1)[0].decode("latin-1", errors="replace"),
        "flogoff": struct.unpack_from("<I", data, 248)[0],
    }


def _decode_date(packed: int) -> str | None:
    """Decode the packed ``fdate`` field (year 12 bits, month 4, day 5, hour 5, minute 6)."""
    if packed == 0:
        return None
    try:
        timestamp = dt.datetime(
            (packed >> 20) & 0xFFF,
            (packed >> 16) & 0x0F,
            (packed >> 11) & 0x1F,
            (packed >> 6) & 0x1F,
            packed & 0x3F,
        )
    except ValueError:
        logger.warning("SPC date field 0x{:08X} is not a valid date; ignoring it", packed)
        return None
    return timestamp.isoformat()


def _decode_y_values(data: bytes, exponent: int, *, sixteen_bit: bool) -> np.ndarray:
    """Decode float32 values, or integers scaled by ``2**(exponent - bits)``."""
    if exponent == _FEXP_FLOAT32:
        return np.frombuffer(data, dtype="<f4").astype(np.float64)
    if sixteen_bit:
        return np.frombuffer(data, dtype="<i2").astype(np.float64) * 2.0 ** (exponent - 16)
    return np.frombuffer(data, dtype="<i4").astype(np.float64) * 2.0 ** (exponent - 32)


def _read_log_text(f: BinaryIO, offset: int) -> str:
    """Return the ASCII text of the log block at ``offset`` (empty if absent or unreadable)."""
    if offset == 0:
        return ""
    f.seek(offset)
    log_header = f.read(64)
    if len(log_header) < 64:
        logger.warning("SPC log block at offset {} is truncated; ignoring it", offset)
        return ""
    # LOGSTC: logsizd (disk size of the block) at 0, logtxto (text offset) at 8.
    block_size = struct.unpack_from("<I", log_header, 0)[0]
    text_offset = struct.unpack_from("<I", log_header, 8)[0]
    if text_offset == 0 or block_size <= text_offset:
        return ""
    f.seek(offset + text_offset)
    text = f.read(min(block_size - text_offset, 65536))
    return text.split(b"\x00", 1)[0].decode("latin-1", errors="replace")


__all__ = ["SPCError", "load_spc"]
