"""Spatial pixel binning of image cubes (plan ``plans/pixel-binning.md``).

``k × k`` blocks of pixels, ``k`` frames by ``k`` rows, become one pixel: the
mean of the block, taken on the values as stored. Integer cubes (raw counts)
keep their dtype: the mean is rounded half up, so a binned cube is still raw
counts for the rest of engiNIR (which tells raw counts from corrected data by
the dtype). Floating cubes (already corrected) keep float32 means. Frames and
rows that do not fill a last block are dropped. Bands are never binned.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from .._checks import check_integer, log_and_raise
from .._levels import BINNING_FACTORS
from .._logging import get_logger
from ._types import SpectralData

logger = get_logger(__name__)


def check_binning(binning: Any) -> int:
    """The binning factor as an int (1 = none); refuses anything else."""
    k = check_integer(binning, "binning", minimum=1)
    if k not in BINNING_FACTORS:
        log_and_raise("binning must be one of {}, got {}", BINNING_FACTORS, k, depth=2)
    return k


def binned_shape(shape: tuple[int, int, int], binning: int) -> tuple[int, int, int]:
    """``(frames, rows, bands)`` after ``binning``: incomplete blocks are dropped."""
    k = check_binning(binning)
    frames, rows, bands = (int(n) for n in shape)
    return frames // k, rows // k, bands


def bin_blocks(cube: np.ndarray, binning: int) -> np.ndarray:
    """Average ``k × k`` blocks of frames and rows of a ``(frames, rows, bands)`` cube.

    Integer cubes keep their dtype (the mean rounded half up, computed exactly
    in 64-bit integers); floating cubes give float32 means. ``binning=1``
    returns the cube unchanged.
    """
    k = check_binning(binning)
    cube = np.asarray(cube)
    if cube.ndim != 3:
        log_and_raise("Binning needs an image cube (frames, rows, bands), got shape {}",
                      cube.shape)
    if k == 1:
        return cube
    frames, rows, bands = binned_shape(cube.shape, k)
    if frames == 0 or rows == 0:
        log_and_raise("A cube of {} frames and {} rows has no {} x {} block", cube.shape[0],
                      cube.shape[1], k, k)
    blocks = cube[:frames * k, :rows * k].reshape(frames, k, rows, k, bands)
    if cube.dtype.kind in "iu":
        total = blocks.sum(axis=(1, 3), dtype=np.int64)
        n = k * k
        return np.floor_divide(total + n // 2, n).astype(cube.dtype, copy=False)
    if cube.dtype.kind == "f":
        return blocks.mean(axis=(1, 3), dtype=np.float64).astype(np.float32, copy=False)
    log_and_raise("Binning needs a numeric cube, got dtype {}", cube.dtype)


def bin_rows(reference: np.ndarray, binning: int) -> np.ndarray:
    """Average blocks of ``k`` rows of a ``(rows, bands)`` reference (float32).

    A dark or white reference holds one spectrum per detector row; a binned
    dataset needs one per binned row, made like its pixels (the incomplete last
    block dropped).
    """
    k = check_binning(binning)
    reference = np.asarray(reference)
    if reference.ndim != 2:
        log_and_raise("A reference to bin has shape (rows, bands), got {}", reference.shape)
    rows = reference.shape[0] // k
    if rows == 0:
        log_and_raise("A reference of {} rows has no block of {}", reference.shape[0], k)
    blocks = reference[:rows * k].reshape(rows, k, reference.shape[1])
    return blocks.mean(axis=1, dtype=np.float64).astype(np.float32, copy=False)


def binned_frames(frames: slice | None, binning: int) -> slice | None:
    """The file's frames behind a slice of binned frames (step 1 only)."""
    k = check_binning(binning)
    if frames is None or k == 1:
        return frames
    if frames.step not in (None, 1):
        log_and_raise("A binned cube is read in order: frames must have step 1, got {}",
                      frames.step, depth=2)
    start = None if frames.start is None else int(frames.start) * k
    stop = None if frames.stop is None else int(frames.stop) * k
    if (start is not None and start < 0) or (stop is not None and stop < 0):
        log_and_raise("Binned frames are counted from the start, got {}", frames, depth=2)
    return slice(start, stop)


def bin_spectral_data(data: SpectralData, binning: int) -> SpectralData:
    """``data`` with its cube binned; a header-only load gets the binned shape."""
    k = check_binning(binning)
    if k == 1:
        return data
    meta = dict(data.meta)
    shape = binned_shape(data.cube_shape, k)
    if shape[1] == 0 or (np.size(data.X) and np.ndim(data.X) != 3):
        log_and_raise("Only image cubes can be binned; {} holds spectra of shape {}",
                      meta.get("source_path", "the data"), data.cube_shape)
    X = bin_blocks(data.X, k) if np.size(data.X) else data.X
    meta["source_shape"] = shape
    if "frames_total" in meta:
        meta["frames_total"] = shape[0]
    meta["binning"] = k
    return SpectralData(X=X, wavelengths=data.wavelengths, meta=meta)


def bin_frame_chunks(chunks: Iterable[tuple[int, np.ndarray]],
                     binning: int) -> Iterator[tuple[int, np.ndarray]]:
    """Bin consecutive ``(offset, frames)`` blocks of a cube, in file order.

    Blocks of any size are joined so that no ``k``-frame block is split; the
    offsets are those of the binned frames. Frames left over at the end are
    dropped, as by :func:`bin_blocks`.
    """
    k = check_binning(binning)
    if k == 1:
        yield from chunks
        return
    pending: np.ndarray | None = None
    expected = 0
    binned = 0
    for offset, block in chunks:
        if int(offset) != expected:
            log_and_raise("Binned frames are read in order: expected the block at frame {}, "
                          "got {}", expected, offset)
        block = np.asarray(block)
        expected += block.shape[0]
        pending = block if pending is None else np.concatenate([pending, block])
        usable = (pending.shape[0] // k) * k
        if usable:
            out = bin_blocks(pending[:usable], k)
            yield binned, out
            binned += out.shape[0]
            pending = pending[usable:]
    if pending is not None and pending.shape[0]:
        logger.debug("Binning dropped the last {} frame(s) (fewer than {})",
                     pending.shape[0], k)


__all__ = ["BINNING_FACTORS", "bin_blocks", "bin_frame_chunks", "bin_rows", "bin_spectral_data",
           "binned_frames", "binned_shape", "check_binning"]
