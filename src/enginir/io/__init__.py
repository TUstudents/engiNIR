"""Readers and writers for spectral data files and engiNIR project files.

Spectral data formats (all loaders return :class:`SpectralData`)

==============  =========================================  ======================
Format          Functions                                  Extensions
==============  =========================================  ======================
CSV / Excel     :func:`load_csv`, :func:`save_csv`         ``.csv``, ``.xlsx``
Galactic SPC    :func:`load_spc`                           ``.spc``
Thermo SPA      :func:`load_spa`                           ``.spa``
JCAMP-DX        :func:`load_jcamp`                         ``.dx``, ``.jdx``
ENVI            :func:`load_envi`, :func:`save_envi`       ``.hdr``, ``.img``, ``.dat``, ``.raw``
HDF5 spectra    :func:`load_hdf5`, :func:`save_hdf5`       ``.h5``, ``.hdf5``
EVK EIFF        :func:`load_eiff`                          ``.eiff``
==============  =========================================  ======================

:func:`load_spectra` picks the reader from the file extension. Invalid or
unsupported files of the text and binary formats raise a subclass of
:class:`EnginirIOError` (itself a :class:`ValueError`); the HDF5 readers raise
:class:`ValueError` or :class:`KeyError`, or h5py's own errors for damaged
files. Missing files raise :class:`FileNotFoundError`.

The Thermo SPA reader follows the published reverse-engineered layout and has
only been tested with synthetic files; check its output against OMNIC before
relying on it.

Project files (the v3 scientific workspace) are read and written by the
functions of :mod:`enginir.io.project_store` re-exported here.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._common import EnginirIOError
from ._types import SpectralData
from .csv_io import load_csv, save_csv
from .spa_io import load_spa, SPAError
from .spc_io import load_spc, SPCError
from .jcamp_io import load_jcamp, JCAMPError
from .envi_io import iter_envi_frame_chunks, load_envi, save_envi, ENVIError
from .hdf5_io import iter_hdf5_frame_chunks, load_hdf5, save_hdf5
from .project_store import (
    PROJECT_SCHEMA_VERSION,
    PredictionArtifactWriter,
    create_project,
    discard_prediction_run_artifacts,
    has_prediction_run,
    finalize_prediction_run_artifacts,
    initialize_prediction_run_artifacts,
    iter_roi_calibration_refs,
    load_model_run_artifact_refs,
    load_model_run_prediction_artifact,
    load_prediction_run_artifact_refs,
    load_prediction_spatial_map_preview,
    load_project_store,
    load_roi_spectra,
    load_roi_spectra_batch,
    load_roi_spectra_matrix,
    open_prediction_run_artifact_writer,
    remove_roi_from_project,
    save_calibration_spec,
    save_dataset_ref,
    save_model_run,
    save_prediction_run,
    save_preprocessing_recipe,
    save_sample_set,
    save_signal_domain_contract,
    save_target_assignment,
    save_target_definition,
    save_training_roi,
)
from .eiff_io import (
    EIFFError,
    EIFFMemoryError,
    EIFFTruncatedError,
    iter_eiff_frame_chunks,
    load_eiff,
)

from enginir._checks import log_and_raise


#: File extension -> (format name reported in ``meta["format"]``, reader).
_READERS: dict[str, tuple[str, Callable[..., SpectralData]]] = {
    ".csv": ("csv", load_csv),
    ".xlsx": ("csv", load_csv),
    ".spc": ("spc", load_spc),
    ".spa": ("spa", load_spa),
    ".dx": ("jcamp", load_jcamp),
    ".jdx": ("jcamp", load_jcamp),
    ".hdr": ("envi", load_envi),
    ".img": ("envi", load_envi),
    ".dat": ("envi", load_envi),
    ".raw": ("envi", load_envi),
    ".h5": ("hdf5", load_hdf5),
    ".hdf5": ("hdf5", load_hdf5),
    ".eiff": ("eiff", load_eiff),
}


def file_format(path: str | Path) -> str:
    """Return the format name of a data file from its extension (e.g. ``"hdf5"``).

    Unknown extensions are returned without the dot, in lower case.
    """
    suffix = Path(path).suffix.lower()
    return _READERS[suffix][0] if suffix in _READERS else suffix.lstrip(".")


def load_spectra(path: str | Path, **kwargs: Any) -> SpectralData:
    """Load a spectral data file, choosing the reader from the file extension.

    Parameters
    ----------
    path:
        Data file with one of the extensions listed in :mod:`enginir.io`.
    **kwargs:
        Passed to the format reader (e.g. ``frames=`` for EIFF and HDF5).

    Returns
    -------
    SpectralData
        See the format reader.

    Raises
    ------
    ValueError
        If the extension is not supported.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in _READERS:
        log_and_raise("Unsupported file extension {!r} of {}; supported: {}", suffix, path,
                      sorted(_READERS))
    return _READERS[suffix][1](path, **kwargs)


__all__ = [
    "EnginirIOError",
    "file_format",
    "SpectralData",
    "load_spectra",
    # spectral data formats
    "ENVIError",
    "EIFFError",
    "EIFFMemoryError",
    "EIFFTruncatedError",
    "JCAMPError",
    "SPAError",
    "SPCError",
    "iter_eiff_frame_chunks",
    "iter_envi_frame_chunks",
    "iter_hdf5_frame_chunks",
    "load_csv",
    "load_eiff",
    "load_envi",
    "load_hdf5",
    "load_jcamp",
    "load_spa",
    "load_spc",
    "save_csv",
    "save_envi",
    "save_hdf5",
    # project files
    "PROJECT_SCHEMA_VERSION",
    "PredictionArtifactWriter",
    "create_project",
    "discard_prediction_run_artifacts",
    "has_prediction_run",
    "finalize_prediction_run_artifacts",
    "initialize_prediction_run_artifacts",
    "iter_roi_calibration_refs",
    "load_model_run_artifact_refs",
    "load_model_run_prediction_artifact",
    "load_prediction_run_artifact_refs",
    "load_prediction_spatial_map_preview",
    "load_project_store",
    "load_roi_spectra",
    "load_roi_spectra_batch",
    "load_roi_spectra_matrix",
    "open_prediction_run_artifact_writer",
    "remove_roi_from_project",
    "save_calibration_spec",
    "save_dataset_ref",
    "save_model_run",
    "save_prediction_run",
    "save_preprocessing_recipe",
    "save_sample_set",
    "save_signal_domain_contract",
    "save_target_assignment",
    "save_target_definition",
    "save_training_roi",
]
