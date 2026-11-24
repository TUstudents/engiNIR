"""The false-color image of a scan: three raw bands, stretched (the viewer's dataset view)."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from ._checks import log_and_raise
from ._logging import get_logger

logger = get_logger(__name__)

#: Wavelengths (nm) of the red, green and blue channels.
FALSE_COLOR_NM = (1650.0, 1300.0, 1050.0)
#: Frames read at a time, so a scan is never in memory with all its bands.
_BLOCK_FRAMES = 256


def false_color_bands(wavelengths: np.ndarray) -> list[int]:
    """The band indices nearest :data:`FALSE_COLOR_NM` on ``wavelengths``."""
    axis = np.asarray(wavelengths, dtype=np.float64)
    return [int(np.argmin(np.abs(axis - nm))) for nm in FALSE_COLOR_NM]


def stretch_rgb(raw: np.ndarray) -> np.ndarray:
    """``raw`` (..., 3) stretched per channel from its 2nd to 98th percentile to [0, 1], float32."""
    raw = np.asarray(raw, dtype=np.float32)
    lo, hi = np.percentile(raw.reshape(-1, 3), [2, 98], axis=0)
    scale = np.maximum(hi - lo, 1.0)
    return np.clip((raw - lo) / scale, 0.0, 1.0).astype(np.float32, copy=False)


def false_color_image(path: str | os.PathLike[str], *, bands: Sequence[int] | None = None,
                       binning: int = 1) -> np.ndarray:
    """The false-color image of a scan, as the viewer shows a dataset.

    Red, green and blue are the raw bands nearest 1650, 1300 and 1050 nm, each
    stretched from its 2nd to its 98th percentile. Only these three bands are
    kept: the file is read in blocks of frames, read-only. The image keeps the
    scan's pixels, so cells of a layout (:func:`roi_grid`) lie on it as on the
    scan; show it with :meth:`enginir.viewer.FalseColorCanvas.set_image`.

    Parameters
    ----------
    path:
        An image file (EIFF, HDF5, ENVI).
    bands:
        Three band indices (red, green, blue). Required for a file without
        wavelengths (band numbers, e.g. a Kestreleye ``stream_hsi.h5``), whose
        bands cannot be chosen by wavelength.
    binning:
        As for :func:`read_frames`: frames and rows are binned pixels.

    Returns
    -------
    numpy.ndarray
        ``(rows, frames, 3)`` float32 in [0, 1].
    """
    from ._workflow import _FRAME_FORMATS, read_frames, read_header
    from .io import file_format, is_band_numbers

    header = read_header(path, binning=binning)
    n_frames, height, n_bands = header.cube_shape
    if bands is None:
        if header.meta.get("x_units") == "band_index" or is_band_numbers(header.wavelengths):
            log_and_raise("{} has no wavelengths: give the three bands of the image "
                          "(bands=(red, green, blue))", os.fspath(path))
        chosen = false_color_bands(header.wavelengths)
    else:
        chosen = [int(b) for b in bands]
        if len(chosen) != 3 or not all(0 <= b < n_bands for b in chosen):
            log_and_raise("bands must be three band indices in 0..{}, got {}", n_bands - 1,
                          list(bands))
    raw = np.empty((n_frames, height, 3), dtype=np.float32)
    # Other formats are read completely by every read_frames call: read them once.
    block = _BLOCK_FRAMES if file_format(os.fspath(path)) in _FRAME_FORMATS else max(n_frames, 1)
    for first in range(0, n_frames, block):
        last = min(first + block, n_frames) - 1
        raw[first:last + 1] = read_frames(path, first, last, binning)[..., chosen]
    logger.info("False-color image of {}: {} frames x {} rows, bands {}", os.fspath(path),
                n_frames, height, chosen)
    return stretch_rgb(raw.transpose(1, 0, 2))
