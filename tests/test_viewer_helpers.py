"""Focused tests for small viewer helper modules."""

from __future__ import annotations


import numpy as np


def test_wavelength_axes_match_rejects_malformed_axes():
    from enginir.viewer._data_helpers import wavelength_axes_match

    assert wavelength_axes_match(np.array([900.0, 901.0]), np.array([900.0, 901.0]))
    assert not wavelength_axes_match(
        np.array([[900.0, 901.0]]),
        np.array([900.0, 901.0]),
    )
    assert not wavelength_axes_match(
        np.array([900.0, np.nan]),
        np.array([900.0, 901.0]),
    )


def test_wavelength_axes_match_a_file_axis_with_its_stored_float32_copy():
    """A project stores wavelengths as float32; re-reading the file must still match."""
    from enginir.viewer._data_helpers import wavelength_axes_match

    file_axis = np.linspace(1000.0, 1700.0, 12)
    stored = file_axis.astype(np.float32)
    assert np.abs(stored.astype(np.float64) - file_axis).max() > 1e-6  # plain allclose fails
    assert wavelength_axes_match(stored, file_axis)
    assert not wavelength_axes_match(stored, file_axis + 0.01)  # a real shift still differs
