enginir.utils
=============

General-purpose helpers for reshaping hyperspectral cubes between image and
spectral-matrix forms.

``flatten_cube`` expects a non-empty cube with shape
``(rows, cols, n_wavelengths)`` and returns both the 2-D matrix and the spatial
shape needed for round-trip restoration. ``unflatten_cube`` validates that the
matrix row count matches ``rows * cols`` before reshaping.

.. autofunction:: enginir.utils.flatten_cube

.. autofunction:: enginir.utils.unflatten_cube
