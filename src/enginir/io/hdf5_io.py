"""Reader and writer for spectra and image cubes stored in plain HDF5 files.

The default layout, written by :func:`save_hdf5`, is a root dataset
``spectra`` (2-D spectra or a 3-D cube with the spectral axis last), a root
dataset ``wavelengths`` and an optional group ``meta`` whose attributes and
datasets become :class:`~enginir.io.SpectralData` metadata. Raw cubes of the
KestrelEye camera software (a root dataset ``data`` without ``spectra``) are
recognised as well. Blosc-compressed datasets are readable when the
``hdf5plugin`` package is installed.

engiNIR project files are HDF5 files too, but are read with
:mod:`enginir.io.project_store`.
"""
# pyright: reportIndexIssue=false, reportAttributeAccessIssue=false
# h5py's stubs do not describe Group/Dataset item access.

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from enginir._checks import check_integer, log_and_raise
from enginir._logging import get_logger

from ._common import base_meta, require_file
from ._hdf5_groups import read_mapping, write_mapping
from ._types import SpectralData
from .project_store import is_project_file

logger = get_logger(__name__)

_SCHEMA_SIMPLE = "enginir_simple_spectral_hdf5"
_SCHEMA_KESTRELEYE = "kestreleye_raw_hdf5"
_SCHEMA_CUSTOM = "custom_hdf5"


def load_hdf5(
    path: str | os.PathLike[str],
    spectra_key: str = "spectra",
    wavelengths_key: str = "wavelengths",
    meta_group: str = "meta",
    frames: slice | None = None,
    header_only: bool = False,
) -> SpectralData:
    """Load spectra or an image cube from an HDF5 file.

    Parameters
    ----------
    path:
        ``.h5`` / ``.hdf5`` file.
    spectra_key:
        Dataset with the spectra or cube. With the default ``"spectra"``, a
        KestrelEye raw file's ``data`` dataset is used if ``spectra`` is absent.
    wavelengths_key:
        Dataset with the wavelength axis. If it is missing, band indices
        ``0, 1, ...`` are used and ``meta["x_units"]`` is ``"band_index"``.
    meta_group:
        Group whose attributes and datasets are returned as metadata.
    frames:
        Optional slice of the first axis (frames of a cube, rows of a matrix).
    header_only:
        If ``True``, return an empty array with the trailing dimensions of the
        data instead of reading it.

    Returns
    -------
    SpectralData
        ``X`` in the stored dtype, ``wavelengths``, and ``meta`` with the
        common keys, the entries of ``meta_group``, ``"hdf5_schema"``,
        ``"hdf5_spectra_key"``, ``"source_dtype"``, ``"frames_total"`` and
        ``"source_shape"`` (shape of the selected data).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` is an engiNIR project file.
    KeyError
        If the spectra dataset is missing.
    """
    path = require_file(path)
    _register_compression_plugins()
    with h5py.File(path, "r") as f:
        dataset, schema = _spectral_dataset(f, path, spectra_key)
        selected_shape = _selected_shape(dataset.shape, frames)
        if header_only:
            X = np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
        else:
            X = _read_frames(dataset, frames)
        n_bands = dataset.shape[-1]
        if wavelengths_key in f:
            wavelengths, x_units = np.asarray(f[wavelengths_key][()]), "nm"
        else:
            _warn_missing_wavelengths(str(path), wavelengths_key, n_bands)
            wavelengths, x_units = np.arange(n_bands, dtype=np.float64), "band_index"

        group = f.get(meta_group)
        meta = read_mapping(group) if isinstance(group, h5py.Group) else {}
        # The file's identity always describes this file, even if the stored
        # metadata came from another source (e.g. re-saved EIFF metadata).
        meta.update(base_meta(path, "hdf5", x_units))
        meta.update(
            hdf5_schema=schema,
            hdf5_spectra_key=(dataset.name or "").lstrip("/"),
            source_dtype=str(dataset.dtype),
            frames_total=int(dataset.shape[0]),
            frames_loaded=int(X.shape[0]),
            source_shape=selected_shape,
        )
    logger.info("Loaded HDF5: shape {} from {}", X.shape, path)
    return SpectralData(X, wavelengths, meta)


def save_hdf5(
    path: str | os.PathLike[str],
    X: np.ndarray,
    wavelengths: np.ndarray,
    meta: dict[str, Any] | None = None,
    spectra_key: str = "spectra",
    wavelengths_key: str = "wavelengths",
    meta_group: str = "meta",
    compression: str = "gzip",
) -> None:
    """Write spectra or an image cube to a new HDF5 file readable by :func:`load_hdf5`.

    Parameters
    ----------
    path:
        Output file. It is written to a temporary file next to it first and
        only replaces an existing file once complete.
    X:
        Spectra ``(n_samples, n_wavelengths)`` or cube with the spectral axis
        last.
    wavelengths:
        Wavelengths in nm, shape ``(n_wavelengths,)``.
    meta:
        Metadata written to ``meta_group`` like project objects: scalars and
        short numeric vectors as attributes, arrays and string lists as
        datasets, nested mappings as subgroups; ``None`` values are omitted.
        On loading, ``source_path``, ``format`` and ``x_units`` describe the
        loaded file and take precedence over stored keys of the same name.
    spectra_key, wavelengths_key, meta_group:
        Names of the datasets and the metadata group.
    compression:
        HDF5 compression filter (``"gzip"`` or ``"lzf"``).
    """
    X = np.asarray(X)
    wavelengths = np.asarray(wavelengths)
    if wavelengths.shape != (X.shape[-1],):
        log_and_raise("wavelengths must have shape ({},) to match X of shape {}, got {}",
                      X.shape[-1], X.shape, wavelengths.shape)
    if not Path(path).expanduser().resolve().parent.is_dir():
        log_and_raise("Output directory of {} does not exist", path, error=FileNotFoundError)
    target = Path(path)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with h5py.File(temporary, "w") as f:
            f.create_dataset(spectra_key, data=X, compression=compression)
            f.create_dataset(wavelengths_key, data=wavelengths, compression=compression)
            if meta:
                write_mapping(f.create_group(meta_group), dict(meta), compression)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    logger.info("Saved HDF5: shape {} to {}", X.shape, path)


def iter_hdf5_frame_chunks(
    path: str | os.PathLike[str],
    *,
    chunk_size: int,
    frames: slice | None = None,
    spectra_key: str = "spectra",
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield consecutive blocks of frames without reading the whole cube.

    Parameters
    ----------
    path:
        HDF5 file as accepted by :func:`load_hdf5`.
    chunk_size:
        Maximum number of frames per block.
    frames:
        Optional slice of frames to iterate over, as in :func:`load_hdf5`
        (any step, including negative).
    spectra_key:
        Dataset with the cube, as in :func:`load_hdf5`.

    Yields
    ------
    offset : int
        Index of the block's first frame relative to the start of ``frames``.
    block : numpy.ndarray
        Up to ``chunk_size`` frames in the stored dtype.
    """
    chunk_size = check_integer(chunk_size, "chunk_size", minimum=1)
    path = require_file(path)
    _register_compression_plugins()
    with h5py.File(path, "r") as f:
        dataset, _ = _spectral_dataset(f, path, spectra_key)
        indices = range(dataset.shape[0])[frames or slice(None)]
        for offset in range(0, len(indices), chunk_size):
            wanted = indices[offset:offset + chunk_size]
            stop = wanted.stop if wanted.stop >= 0 else None
            yield offset, _read_frames(dataset, slice(wanted.start, stop, wanted.step))


def _spectral_dataset(f: h5py.File, path: Path, spectra_key: str) -> tuple[h5py.Dataset, str]:
    """Return the spectra dataset and the name of the recognised file layout."""
    if is_project_file(f):
        log_and_raise("{} is an engiNIR project file; open it with the project API", path,
                      depth=2)
    if spectra_key in f:
        key, schema = spectra_key, _SCHEMA_SIMPLE if spectra_key == "spectra" else _SCHEMA_CUSTOM
    elif spectra_key == "spectra" and isinstance(f.get("data"), h5py.Dataset):
        key, schema = "data", _SCHEMA_KESTRELEYE
    else:
        log_and_raise("{} has no dataset '{}'", path, spectra_key, error=KeyError, depth=2)
    dataset = f[key]
    if not isinstance(dataset, h5py.Dataset) or dataset.ndim < 2:
        log_and_raise("'{}' in {} is not a 2-D or 3-D dataset", key, path, depth=2)
    return dataset, schema


def _read_frames(dataset: h5py.Dataset, frames: slice | None) -> np.ndarray:
    """Read ``dataset[frames]``; h5py only slices with positive steps, so reverse after."""
    if frames is None:
        return np.asarray(dataset[()])
    indices = range(dataset.shape[0])[frames]
    if not indices:
        return np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
    if indices.step > 0:
        return np.asarray(dataset[indices.start:indices.stop:indices.step])
    return np.asarray(dataset[indices[-1]:indices[0] + 1:-indices.step])[::-1]


def _selected_shape(shape: tuple[int, ...], frames: slice | None) -> tuple[int, ...]:
    if frames is None:
        return tuple(int(dim) for dim in shape)
    return (len(range(*frames.indices(shape[0]))), *(int(dim) for dim in shape[1:]))


@cache
def _register_compression_plugins() -> None:
    """Import ``hdf5plugin`` once so that Blosc and other filters can be decoded."""
    import hdf5plugin  # noqa: F401  (a dependency; importing registers the filters)


@cache
def _warn_missing_wavelengths(path: str, wavelengths_key: str, n_bands: int) -> None:
    logger.warning("{} has no '{}' dataset; using band indices 0..{} as wavelengths", path,
                   wavelengths_key, n_bands - 1)


__all__ = ["iter_hdf5_frame_chunks", "load_hdf5", "save_hdf5"]
