"""Matplotlib figures for spectra, calibrations and prediction maps.

Spectra and model vectors
    :func:`plot_spectra`, :func:`plot_loadings`, :func:`plot_vip`
Samples
    :func:`plot_prediction` (reference vs predicted), :func:`plot_scores`,
    :func:`plot_leverage` (influence plot)
Images
    :func:`plot_image_map` (prediction map)

All functions draw into an optional ``ax``, return the Axes and pass extra
keyword arguments to the underlying matplotlib call.
"""

from .plots import (
    plot_image_map,
    plot_leverage,
    plot_loadings,
    plot_prediction,
    plot_scores,
    plot_spectra,
    plot_vip,
)

__all__ = [
    "plot_image_map",
    "plot_leverage",
    "plot_loadings",
    "plot_prediction",
    "plot_scores",
    "plot_spectra",
    "plot_vip",
]
