"""Reader for the EIFF files of EVK HELIOS push-broom NIR cameras.

An EIFF file is a chunked container: a root ``LIST`` chunk holds ``PROP``
chunks (image header, wavelength table, device information) followed by one
``FORM``/``BODY`` chunk per camera frame. Frames are zlib-compressed and are
streamed and decompressed directly into a pre-allocated cube, optionally in
several threads, so the compressed file is never held in memory. The format is
documented in ``docs/eiff_format.md``.

Each frame is one line of the scene: ``height`` spatial pixels times ``width``
spectral bands, so cubes have shape ``(n_frames, height, n_bands)``.
"""

from __future__ import annotations

import datetime as dt
import itertools
import os
import struct
import zlib
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import BinaryIO

import numpy as np

from enginir._checks import check_integer, log_and_raise
from enginir._logging import get_logger

from ._common import EnginirIOError, base_meta, read_exact, require_file
from ._types import SpectralData

logger = get_logger(__name__)

# -- Chunk layout constants ---------------------------------------------------
_NAME_LEN = 10  # chunk name field: 10 bytes, space-padded
_SIZE_LEN = 8  # chunk size field: big-endian uint64
_HEADER_LEN = _NAME_LEN + _SIZE_LEN  # 18 bytes per chunk header
_CONTENT_TYPE_LEN = 10  # group chunks (FORM/LIST/PROP) have 10-byte content-type

# Compression envelope: [u32 LE comp_type][u32 LE uncomp_size][u8 0x01][zlib]
_COMP_HEADER_LEN = 9  # 4 + 4 + 1

# EVKSTNRD struct: 54 bytes, big-endian u16 fields with date as individual bytes
_STNRD_SIZE = 54

# Safety limits
_MAX_CHUNK_SIZE = 4 * 1024**3  # 4 GiB; reject obviously corrupt size fields
_MAX_FRAME_BYTES = 512 * 1024**2  # 512 MiB per frame; sanity cap
_DEFAULT_MAX_BYTES = 16 * 1024**3  # 16 GiB; default memory cap for output cube
_DEFAULT_FRAME_WORKERS = max(1, min(4, os.cpu_count() or 1))
_PARALLEL_FRAME_THRESHOLD = 64
_MAX_PENDING_FACTOR = 2
_PROGRESS_UPDATE_STEPS = 20
EIFFProgressCallback = Callable[[int, int], None]

# Minimum file: LIST header (18) + content_type (10) + PROP header (18) +
#               content_type (10) + EVKSTNRD header (18) + EVKSTNRD data (54)
_MIN_FILE_SIZE = 128


class EIFFError(EnginirIOError):
    """Raised for invalid, corrupted, or unreadable EIFF files."""


class EIFFTruncatedError(EIFFError):
    """Raised when the file is truncated or a read returns fewer bytes than expected."""


class EIFFMemoryError(EIFFError):
    """Raised when the decoded cube would exceed the memory limit."""


# -- Low-level helpers --------------------------------------------------------


def _read_exact(f: BinaryIO, n: int, what: str) -> bytes:
    return read_exact(f, n, what, EIFFTruncatedError)


def _read_chunk_header(f: BinaryIO, file_size: int = 0, *,
                       limit: bool = True) -> tuple[bytes, int]:
    """Read a chunk header and return ``(name, payload size)``.

    Parameters
    ----------
    f:
        File positioned at a chunk header.
    file_size:
        Total file size; if non-zero, a payload extending past it raises
        :class:`EIFFTruncatedError`.
    limit:
        Reject sizes above 4 GiB as corrupt. Disabled for the root chunk,
        which spans the whole file.
    """
    raw = _read_exact(f, _HEADER_LEN, "chunk header")
    name = raw[:_NAME_LEN].strip(b"\x00 ")
    size = struct.unpack(">Q", raw[_NAME_LEN:])[0]
    position = f.tell() - _HEADER_LEN
    if limit and size > _MAX_CHUNK_SIZE:
        log_and_raise("Chunk {!r} at offset {} declares {} bytes ({:.1f} GiB), more than the "
                      "4 GiB safety limit; the file is likely corrupt", name, position, size,
                      size / 1024**3, error=EIFFError, depth=2)
    if file_size and f.tell() + size > file_size:
        log_and_raise("Chunk {!r} at offset {} declares {} bytes of payload, but only {} bytes "
                      "remain in the file", name, position, size, file_size - f.tell(),
                      error=EIFFTruncatedError, depth=2)
    return name, size


def _decompress_payload(data: bytes, context: str = "") -> bytes:
    """Decompress a payload with the EIFF compression envelope.

    Envelope format: [4B comp_type LE][4B uncomp_size LE][1B flag][zlib data]
    comp_type=2 means zlib compression. comp_type=0 means uncompressed.
    """
    if len(data) < _COMP_HEADER_LEN:
        return data
    comp_type = struct.unpack("<I", data[:4])[0]
    uncomp_size = struct.unpack("<I", data[4:8])[0]

    if comp_type == 0:
        return data[_COMP_HEADER_LEN:]

    if comp_type == 2:
        if uncomp_size > _MAX_FRAME_BYTES:
            log_and_raise("Declared uncompressed size {} exceeds the safety limit ({} bytes){}",
                          uncomp_size, _MAX_FRAME_BYTES, context, error=EIFFError)
        try:
            decompressed = zlib.decompress(data[_COMP_HEADER_LEN:])
        except zlib.error as exc:
            log_and_raise("zlib decompression failed{}: {}", context, exc, error=EIFFError,
                          cause=exc)
        if len(decompressed) != uncomp_size:
            log_and_raise("Decompressed size mismatch: header says {}, got {}{}", uncomp_size,
                          len(decompressed), context, error=EIFFError)
        return decompressed

    log_and_raise("Unknown EIFF compression type: {}{}", comp_type, context, error=EIFFError)


# -- Metadata parsers ---------------------------------------------------------


def _parse_stnrd(data: bytes) -> dict:
    """Parse EVKSTNRD standard header (54 bytes).

    Layout (all multi-byte integers are big-endian u16):
        offset  0: version (u16)
        offset  2: reserved (4 bytes)
        offset  6: padding (2 bytes)
        offset  8: width (u16)
        offset 10: padding (2 bytes)
        offset 12: height (u16)
        offset 14: bytes_per_value (u16)
        offset 16: compression (u16)
        offset 18: bits_per_value (u16)
        offset 20: unknown1 (u16)
        offset 22: unknown2 (u16)
        offset 24: reserved (4 bytes)
        offset 28: count (u16) - number of frames
        offset 30: year (u16)
        offset 32: month (u8)
        offset 33: day (u8)
        offset 34: hour (u8)
        offset 35: minute (u8)
        offset 36: second (u8)
        offset 37: centisecond (u8)
        offset 38: reserved (16 bytes)
    """
    if len(data) < _STNRD_SIZE:
        log_and_raise("EVKSTNRD payload is {} bytes, expected {}", len(data), _STNRD_SIZE,
                      error=EIFFError)

    def u16(off: int) -> int:
        return struct.unpack(">H", data[off : off + 2])[0]

    meta: dict[str, int | str | None] = {
        "version": u16(0),
        "width": u16(8),
        "height": u16(12),
        "bytes_per_value": u16(14),
        "compression": u16(16),
        "bits_per_value": u16(18),
        # Offsets 20-23: two reserved u16 fields whose meaning is not yet
        # documented.  Surfaced as raw values so callers can inspect them;
        # one or both may encode a calibration mode or black/white level.
        "stnrd_u16_20": u16(20),
        "stnrd_u16_22": u16(22),
        "count": u16(28),
    }

    # Validate critical fields
    if meta["width"] == 0 or meta["height"] == 0:
        log_and_raise("Invalid image dimensions: {}x{}", meta["width"], meta["height"],
                      error=EIFFError)
    if meta["bytes_per_value"] == 0:
        log_and_raise("EVKSTNRD bytes_per_value is 0", error=EIFFError)

    # Date/time fields
    year = u16(30)
    month, day = data[32], data[33]
    hour, minute = data[34], data[35]
    second, cs = data[36], data[37]
    try:
        # Camera clock, local time; no time zone is stored in the file.
        meta["timestamp"] = dt.datetime(
            year, month, day, hour, minute, second, cs * 10_000
        ).isoformat()
    except (ValueError, OverflowError):
        logger.warning("EIFF acquisition date fields are invalid; timestamp set to None")
        meta["timestamp"] = None
    return meta


def _parse_wavelengths(data: bytes) -> np.ndarray:
    """Decode EVKWAVTB wavelength table.

    Layout after decompression:
        [2B version LE][2B type LE][2B reserved LE][2B count LE]
        followed by *count* uint32 LE values in units of 0.025 nm.
    """
    if len(data) < 8:
        logger.warning("EIFF wavelength table is too short ({} bytes); ignoring it", len(data))
        return np.array([], dtype=np.float64)
    count = struct.unpack("<H", data[6:8])[0]
    n_available = (len(data) - 8) // 4
    if count > n_available:
        logger.warning(
            "EIFF wavelength table declares {} entries but holds {}", count, n_available
        )
        count = n_available
    return np.frombuffer(data, dtype="<u4", count=count, offset=8).astype(np.float64) * 0.025


# EVKDVINF: fixed-offset UTF-16LE string fields after 54-byte binary header.
_DVINF_FIELDS = (
    (54, "serial"),
    (120, "camera_model"),
    (296, "firmware"),
    (362, "sdk_version"),
    (460, "manufacturer"),
    (526, "interface"),
)


def _read_utf16le_string(data: bytes, offset: int) -> str:
    """Read a null-terminated UTF-16LE string from ``data`` at ``offset``."""
    payload = data[offset:]
    payload = payload[: len(payload) - len(payload) % 2]  # whole code units only
    return payload.decode("utf-16le", errors="ignore").split("\x00", 1)[0]


def _parse_device_info(data: bytes) -> dict:
    """Extract the device strings present in decompressed EVKDVINF data."""
    fields = {name: _read_utf16le_string(data, offset) for offset, name in _DVINF_FIELDS}
    return {name: value for name, value in fields.items() if value}


# -- Chunk scanning -----------------------------------------------------------


def _scan_props(
    f: BinaryIO,
    list_end: int,
    file_size: int,
) -> tuple[dict, np.ndarray]:
    """Scan PROP chunks from current position up to list_end.

    Returns (meta, wavelengths). Stops at the first non-PROP chunk
    (i.e. when FORM frame data begins).
    """
    meta: dict = {}
    wavelengths = np.array([], dtype=np.float64)

    while f.tell() < list_end:
        pos = f.tell()
        name, size = _read_chunk_header(f, file_size)
        chunk_end = f.tell() + size

        if chunk_end > list_end:
            log_and_raise("PROP chunk at offset {} extends past the LIST boundary "
                          "(chunk_end={}, list_end={})", pos, chunk_end, list_end,
                          error=EIFFError)

        if name != b"PROP":
            f.seek(pos)
            break

        content_type = _read_exact(f, _CONTENT_TYPE_LEN, "PROP content type").strip(b"\x00 ")

        if content_type == b"EVKSTNRD":
            _, sub_size = _read_chunk_header(f, file_size)
            meta.update(_parse_stnrd(_read_exact(f, sub_size, "EVKSTNRD")))
        else:
            # Other PROPs: PROPINFO (12B) + data sub-chunk (compressed)
            _, pi_size = _read_chunk_header(f, file_size)
            _read_exact(f, pi_size, "PROPINFO")

            sub_name, sub_size = _read_chunk_header(f, file_size)
            sub_data = _read_exact(f, sub_size, sub_name.decode("ascii", errors="replace"))
            try:
                payload = _decompress_payload(
                    sub_data,
                    context=f" in {content_type.decode('ascii', errors='replace')} chunk",
                )
            except EIFFError as exc:
                # Non-essential metadata: skip corrupt optional chunks.
                logger.warning("Skipping corrupt EIFF PROP chunk: {}", exc)
                f.seek(chunk_end)
                continue

            if content_type == b"EVKWAVTB":
                wavelengths = _parse_wavelengths(payload)
            elif content_type == b"EVKDVINF":
                meta["device"] = _parse_device_info(payload)

        f.seek(chunk_end)

    return meta, wavelengths


def _decode_frame_into(
    body: bytes,
    target: np.ndarray,
    frame_index: int,
) -> None:
    """Decompress one BODY payload into ``target`` (a ``(height, width)`` uint16 view)."""
    raw = _decompress_payload(body, context=f" in frame {frame_index}")
    if len(raw) != target.nbytes:
        log_and_raise(
            "Frame {} decompressed to {} bytes, expected {}",
            frame_index,
            len(raw),
            target.nbytes,
            error=EIFFError,
        )
    np.copyto(target, np.frombuffer(raw, dtype="<u2").reshape(target.shape))


def _iter_frame_bodies(
    f: BinaryIO,
    end: int,
    wanted: Sequence[int],
) -> Iterator[tuple[int, bytes]]:
    """Yield ``(frame_index, BODY payload)`` for the ``wanted`` frames, in file order.

    Frames are numbered by counting ``FORM`` chunks from the current position.
    Iteration stops after the last wanted frame or at the first chunk that
    does not fit before ``end`` (truncated file). Payloads of unwanted frames
    are skipped without reading them. A complete ``FORM`` without a valid
    ``BODY`` raises :class:`EIFFError`: skipping it would shift every later
    frame.
    """
    wanted_set = set(wanted)
    last_wanted = max(wanted, default=-1)
    index = 0
    while index <= last_wanted and f.tell() + _HEADER_LEN <= end:
        name, size = _read_chunk_header(f)
        chunk_end = f.tell() + size
        if chunk_end > end:
            return
        if name == b"FORM":
            if index in wanted_set:
                f.seek(_CONTENT_TYPE_LEN, os.SEEK_CUR)
                body_name, body_size = _read_chunk_header(f)
                if body_name != b"BODY" or f.tell() + body_size > chunk_end:
                    log_and_raise("EIFF frame {} has no valid BODY chunk", index,
                                  error=EIFFError)
                yield index, f.read(body_size)
            index += 1
        f.seek(chunk_end)


def _decode_frames(
    bodies: Iterator[tuple[int, bytes]],
    cube: np.ndarray,
    position_of_frame: dict[int, int],
    workers: int,
    progress_callback: EIFFProgressCallback | None,
) -> np.ndarray:
    """Decode frame payloads into ``cube``; return a mask of the filled cube rows."""
    total = cube.shape[0]
    filled = np.zeros(total, dtype=bool)
    interval = max(1, total // _PROGRESS_UPDATE_STEPS)
    done = 0

    def report_one() -> None:
        nonlocal done
        done += 1
        if progress_callback is not None and (done == total or done % interval == 0):
            progress_callback(done, total)

    if progress_callback is not None:
        progress_callback(0, total)
    if workers <= 1 or total < _PARALLEL_FRAME_THRESHOLD:
        for index, body in bodies:
            position = position_of_frame[index]
            _decode_frame_into(body, cube[position], index)
            filled[position] = True
            report_one()
        return filled

    pending: set[Future[None]] = set()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eiff-frame") as executor:
        for index, body in bodies:
            position = position_of_frame[index]
            pending.add(executor.submit(_decode_frame_into, body, cube[position], index))
            filled[position] = True
            if len(pending) >= workers * _MAX_PENDING_FACTOR:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    future.result()
                    report_one()
        for future in wait(pending).done:
            future.result()
            report_one()
    return filled


def _read_header(f: BinaryIO, path: Path) -> tuple[int, dict, np.ndarray]:
    """Read the root chunk and the PROP chunks; leave ``f`` at the first frame.

    Returns the end offset of the frame data, the image header fields and the
    wavelength table.
    """
    file_size = path.stat().st_size
    if file_size < _MIN_FILE_SIZE:
        log_and_raise("{} has {} bytes; an EIFF file has at least {}", path, file_size,
                      _MIN_FILE_SIZE, error=EIFFError, depth=2)
    # The root LIST spans the whole file, so it is not subject to the chunk size limit;
    # it may also declare more bytes than a truncated file holds.
    root_name, root_size = _read_chunk_header(f, limit=False)
    end = min(f.tell() + root_size, file_size)
    root_type = _read_exact(f, _CONTENT_TYPE_LEN, "root content type").strip(b"\x00 ")
    if root_name != b"LIST" or root_type != b"SPEHLIOS":
        log_and_raise("{} is not an EVK HELIOS EIFF file (root {!r}/{!r})", path,
                      root_name, root_type, error=EIFFError, depth=2)
    header, wavelengths = _scan_props(f, end, file_size)
    if "width" not in header:
        log_and_raise("{} has no EVKSTNRD image header", path, error=EIFFError, depth=2)
    if header["bytes_per_value"] != 2:
        log_and_raise("Unsupported bytes_per_value={}; only 2 (uint16) is supported",
                      header["bytes_per_value"], error=EIFFError, depth=2)
    if wavelengths.size not in (0, header["width"]):
        logger.warning("{}: wavelength table has {} entries for {} bands", path,
                       wavelengths.size, header["width"])
    return end, header, wavelengths


def iter_eiff_frame_chunks(
    path: str | os.PathLike[str],
    *,
    chunk_size: int,
    frames: slice | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield blocks of frames, reading the file once from front to back.

    The file stays open and is scanned forward a single time, so streaming a
    long scan costs no more I/O than :func:`load_eiff` while only one block is
    held in memory.

    Parameters
    ----------
    path:
        ``.eiff`` file.
    chunk_size:
        Maximum number of frames per block.
    frames:
        Optional slice of frames to iterate over, as in :func:`load_eiff`
        (any step). For a negative step the blocks are yielded last first, in
        file order; place them by their ``offset``.

    Yields
    ------
    offset : int
        Index of the block's first frame relative to the start of ``frames``.
    block : numpy.ndarray of uint16, shape (n, height, n_bands)
        Up to ``chunk_size`` frames.

    Raises
    ------
    EIFFTruncatedError
        If the file ends before the requested frames.
    """
    chunk_size = check_integer(chunk_size, "chunk_size", minimum=1)
    path = require_file(path)
    with open(path, "rb") as f:
        end, header, _ = _read_header(f, path)
        frame_indices = range(header["count"])[frames or slice(None)]
        bodies = _iter_frame_bodies(f, end, frame_indices)
        offsets = range(0, len(frame_indices), chunk_size)
        reverse = frame_indices.step < 0
        # The file is scanned forward once, so a reversed selection is read
        # from its last block to its first and each block is filled backwards.
        for offset in reversed(offsets) if reverse else offsets:
            wanted = frame_indices[offset:offset + chunk_size]
            block = np.empty((len(wanted), header["height"], header["width"]), dtype=np.uint16)
            n_read = 0
            for position, (index, body) in enumerate(itertools.islice(bodies, len(wanted))):
                _decode_frame_into(body, block[len(wanted) - 1 - position if reverse
                                               else position], index)
                n_read = position + 1
            if n_read < len(wanted):
                missing = wanted[len(wanted) - 1 - n_read] if reverse else wanted[n_read]
                log_and_raise("{} is truncated: frame {} is missing", path, missing,
                              error=EIFFTruncatedError)
            yield offset, block
            del block  # do not hold the previous block while decoding the next


def load_eiff(
    path: str | os.PathLike[str],
    frames: slice | None = None,
    header_only: bool = False,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    frame_workers: int | None = None,
    progress_callback: EIFFProgressCallback | None = None,
) -> SpectralData:
    """Load a hyperspectral cube from an EVK EIFF file.

    Parameters
    ----------
    path:
        ``.eiff`` file.
    frames:
        Optional slice of frames to load, e.g. ``slice(0, 100)`` or
        ``slice(None, None, 2)``; negative steps are supported. Default: all.
    header_only:
        If ``True``, read only the metadata and wavelengths and return an empty
        cube of shape ``(0, height, n_bands)``.
    max_bytes:
        Refuse (:class:`EIFFMemoryError`) to allocate a cube larger than this
        many bytes; ``0`` disables the check. Default 16 GiB.
    frame_workers:
        Number of threads decompressing frames. Default: up to 4. ``1`` reads
        sequentially.
    progress_callback:
        Called as ``progress_callback(frames_done, frames_requested)`` while
        frames are decoded.

    Returns
    -------
    SpectralData
        ``X`` of shape ``(n_frames, height, n_bands)`` (raw uint16 counts),
        ``wavelengths`` in nm from the file's wavelength table, and ``meta``
        with the common keys plus the image header fields (``"width"``,
        ``"height"``, ``"count"``, ``"bits_per_value"``, ``"timestamp"`` in
        camera local time, ...), ``"device"`` information, and:

        ``"frames_total"``
            Number of frames in the file.
        ``"frames_loaded"``
            Number of frames in ``X``.
        ``"source_shape"``
            Shape of the requested cube, also when ``header_only=True``.
        ``"truncated"``
            ``True`` if the file ended before all requested frames; ``X`` then
            holds only the frames that were read.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    EIFFError
        If the file structure is invalid (see also its subclasses).
    EIFFMemoryError
        If the cube would exceed ``max_bytes``.
    """
    path = require_file(path)
    with open(path, "rb") as f:
        end, header, wavelengths = _read_header(f, path)
        width, height, n_frames_total = header["width"], header["height"], header["count"]
        frame_indices = list(range(n_frames_total)[frames or slice(None)])
        meta = base_meta(path, "eiff", "nm" if wavelengths.size else "unknown")
        meta.update(header)
        meta.update(
            frames_total=n_frames_total,
            source_shape=(len(frame_indices), height, width),
            truncated=False,
        )
        cube_bytes = len(frame_indices) * height * width * 2
        if header_only or not frame_indices:
            meta["frames_loaded"] = 0
            return SpectralData(np.empty((0, height, width), np.uint16), wavelengths, meta)
        if max_bytes and cube_bytes > max_bytes:
            log_and_raise(
                "Loading {} frames of {} x {} needs {:.1f} GiB, more than max_bytes={:.1f} GiB; "
                "load a subset with frames= or raise max_bytes",
                len(frame_indices), height, width, cube_bytes / 1024**3, max_bytes / 1024**3,
                error=EIFFMemoryError,
            )

        cube = np.empty((len(frame_indices), height, width), dtype=np.uint16)
        workers = _DEFAULT_FRAME_WORKERS if frame_workers is None else max(1, int(frame_workers))
        filled = _decode_frames(
            _iter_frame_bodies(f, end, frame_indices),
            cube,
            {frame: position for position, frame in enumerate(frame_indices)},
            workers,
            progress_callback,
        )

    n_loaded = int(filled.sum())
    if n_loaded < len(frame_indices):
        logger.warning("{} is truncated: read {} of {} requested frames", path, n_loaded,
                       len(frame_indices))
        meta["truncated"] = True
        cube = cube[:n_loaded] if filled[:n_loaded].all() else cube[filled]
    meta["frames_loaded"] = n_loaded
    logger.info("Loaded EIFF: {} frames x {} x {} from {}", n_loaded, height, width, path)
    return SpectralData(cube, wavelengths, meta)


__all__ = [
    "EIFFError",
    "EIFFMemoryError",
    "EIFFProgressCallback",
    "EIFFTruncatedError",
    "iter_eiff_frame_chunks",
    "load_eiff",
]
