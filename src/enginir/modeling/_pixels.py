"""The pixel subsample of a fit (``sample-set-model-run-contract_v3.md`` §7.1).

The pixels of an ROI are chosen by SHA-256 keys of the seed, the ROI's stored
spectra and the pixel index; the ``n`` smallest keys win. Only SHA-256 is
involved, so the choice is the same on every platform and in every version,
and it depends on the ROI's spectra, never on its random id.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

from .._checks import log_and_raise
from .._logging import get_logger

logger = get_logger(__name__)


def roi_digest(spectra: np.ndarray) -> bytes:
    """SHA-256 of an ROI's stored spectra (contract §7.1, step 1).

    ``spectra`` must have the ROI's stored dtype (e.g. uint16 counts). The
    bytes are little-endian and row-major, preceded by
    ``"{dtype}:{n_pixels}x{n_bands}:"``.
    """
    stored = np.asarray(spectra)
    if stored.ndim != 2:
        log_and_raise("ROI spectra must be 2-D (n_pixels, n_bands), got shape {}", stored.shape)
    little = np.ascontiguousarray(stored, dtype=stored.dtype.newbyteorder("<"))
    header = f"{stored.dtype.name}:{stored.shape[0]}x{stored.shape[1]}:".encode("ascii")
    return hashlib.sha256(header + little.tobytes(order="C")).digest()


def draw_pixels(spectra: np.ndarray, n: int, seed: int, *,
                digest: bytes | None = None) -> np.ndarray:
    """Indices (ascending) of the ``n`` pixels drawn from an ROI (contract §7.1, steps 2-3).

    Parameters
    ----------
    spectra:
        The ROI's stored spectra, shape ``(n_pixels, n_bands)``.
    n:
        Number of pixels to draw, ``1 <= n <= n_pixels``.
    seed:
        Seed of the pixel subsample, ``0 <= seed < 2**63``.
    digest:
        :func:`roi_digest` of ``spectra``, if already computed.
    """
    n_pixels = int(np.shape(spectra)[0])
    if not 1 <= n <= n_pixels:
        log_and_raise("Cannot draw {} of {} pixels", n, n_pixels)
    digest = roi_digest(spectra) if digest is None else digest
    prefix = int(seed).to_bytes(8, "little") + digest
    keys = [hashlib.sha256(prefix + j.to_bytes(4, "little")).digest() for j in range(n_pixels)]
    chosen = sorted(range(n_pixels), key=keys.__getitem__)[:n]
    return np.array(sorted(chosen), dtype=np.int64)


def sample_checksum(drawn: Iterable[tuple[bytes, np.ndarray]]) -> str:
    """SHA-256 (hex) of the pixel subsample: per ROI in sample-set order, its digest and indices."""
    hasher = hashlib.sha256()
    for digest, indices in drawn:
        hasher.update(digest)
        hasher.update(np.asarray(indices, dtype="<u4").tobytes())
    return hasher.hexdigest()


__all__ = ["draw_pixels", "roi_digest", "sample_checksum"]
