"""Reader for Thermo Scientific OMNIC ``.spa`` single-spectrum files.

The byte offsets used here have not yet been validated against files written
by OMNIC; the reader is tested only with synthetic files.
"""

from __future__ import annotations

import os
import struct

import numpy as np

from enginir._checks import log_and_raise
from enginir._logging import get_logger

from ._common import EnginirIOError, base_meta, require_file
from ._types import SpectralData

logger = get_logger(__name__)

# Byte offsets used by this reader:
#   0 - 29   title (null-padded ASCII)
#  32 - 59   date string (ASCII)
#  60 - 61   data type code (uint16 LE): 2/3 single beam, 11 transmittance,
#            12 reflectance, 13 absorbance, 17 emission
# 564 - 567  first wavenumber (float32 LE)
# 568 - 571  last wavenumber (float32 LE)
# 572 - 575  number of points (uint32 LE)
# 576 - 579  offset of the float32 data array (uint32 LE)
_MIN_HEADER_SIZE = 580
_DATA_TYPE_MAP = {
    2: "single_beam",
    3: "single_beam",
    11: "transmittance",
    12: "reflectance",
    13: "absorbance",
    17: "emission",
}


class SPAError(EnginirIOError):
    """Raised for invalid or unreadable SPA files."""


def load_spa(path: str | os.PathLike[str]) -> SpectralData:
    """Load the spectrum of a Thermo OMNIC ``.spa`` file.

    Parameters
    ----------
    path:
        ``.spa`` file.

    Returns
    -------
    SpectralData
        ``X`` of shape ``(1, n_points)``, the wavenumber axis in cm-1 as
        ``wavelengths``, and ``meta`` with the common keys plus ``"title"``,
        ``"date"`` and ``"data_type"``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    SPAError
        If the file is too small or truncated.
    """
    path = require_file(path)
    with open(path, "rb") as fh:
        header = fh.read(_MIN_HEADER_SIZE)
        if len(header) < _MIN_HEADER_SIZE:
            log_and_raise("{} has {} bytes; an SPA file has at least {}", path, len(header),
                          _MIN_HEADER_SIZE, error=SPAError)
        title = header[:30].decode("latin-1").rstrip("\x00").strip()
        date = header[32:60].decode("latin-1").rstrip("\x00").strip()
        data_type = _DATA_TYPE_MAP.get(struct.unpack_from("<H", header, 60)[0], "unknown")
        x_first, x_last = struct.unpack_from("<ff", header, 564)
        n_points, data_offset = struct.unpack_from("<II", header, 572)
        if n_points == 0:
            log_and_raise("{} declares zero data points", path, error=SPAError)
        fh.seek(data_offset)
        raw = fh.read(4 * n_points)
    if len(raw) < 4 * n_points:
        log_and_raise("{} is truncated: expected {} bytes at offset {}, got {}", path,
                      4 * n_points, data_offset, len(raw), error=SPAError)

    meta = base_meta(path, "spa", "cm-1")
    meta.update(title=title, date=date, data_type=data_type)
    X = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(1, -1)
    wavenumbers = np.linspace(float(x_first), float(x_last), n_points)
    logger.info("Loaded SPA: shape {} from {}", X.shape, path)
    return SpectralData(X, wavenumbers, meta)


__all__ = ["SPAError", "load_spa"]
