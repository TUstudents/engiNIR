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
