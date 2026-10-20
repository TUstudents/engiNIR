"""Return type shared by all spectral data loaders."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from enginir._checks import log_and_raise


class SpectralData(NamedTuple):
    """Spectra or an image cube with its spectral axis and metadata.

    A named tuple, so it can be unpacked or used by attribute::

        X, wavelengths, meta = load_spectra("spectra.csv")
        data = load_spectra("cube.eiff")
        data.X.shape

    Attributes
    ----------
    X : numpy.ndarray
        Spectra of shape ``(n_samples, n_wavelengths)``, or an image cube with
        the spectral axis last: ``(n_frames, height, n_wavelengths)`` for
        push-broom formats (EIFF, HDF5) and ``(lines, samples, n_wavelengths)``
        for ENVI.
    wavelengths : numpy.ndarray, shape (n_wavelengths,)
        Spectral axis in the unit ``meta["x_units"]`` (nm for the imaging
        formats; cm-1 for most FT-IR files).
    meta : dict
        Metadata. Every loader sets ``"source_path"`` (absolute path),
        ``"format"`` (e.g. ``"eiff"``) and ``"x_units"`` (``"nm"``,
        ``"cm-1"``, ``"um"``, ... or ``"unknown"``); the remaining keys are
        format specific and documented by each loader.
    """

    X: np.ndarray
    wavelengths: np.ndarray
    meta: dict[str, Any]

    @property
    def cube_shape(self) -> tuple[int, int, int]:
        """Shape of the data as ``(n_frames, height, n_bands)``.

        Uses ``meta["source_shape"]`` (or ``meta["frames_total"]``) when
        present, so that header-only loads (empty ``X``) report the shape of
        the data they describe. Spectra matrices ``(n, n_bands)`` count as
        ``(n, 1, n_bands)``.
        """
        shape = tuple(int(dim) for dim in self.meta.get("source_shape", np.shape(self.X)))
        if shape and shape[0] == 0 and "frames_total" in self.meta:
            shape = (int(self.meta["frames_total"]), *shape[1:])
        if len(shape) == 2:
            return shape[0], 1, shape[1]
        if len(shape) != 3:
            log_and_raise("Spectral data must be 2-D or 3-D, got shape {}", shape)
        return shape[0], shape[1], shape[2]


__all__ = ["SpectralData"]
