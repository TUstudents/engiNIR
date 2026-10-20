"""Reshaping between image cubes and per-pixel matrices."""

from __future__ import annotations

import numpy as np
import pytest

from enginir.utils import flatten_cube, unflatten_cube


def test_flatten_keeps_pixel_order_and_round_trips_as_views():
    cube = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    X_2d, spatial_shape = flatten_cube(cube)
    assert spatial_shape == (5, 4)
    np.testing.assert_array_equal(X_2d[1 * 4 + 2], cube[1, 2])  # row-major pixel order
    assert np.shares_memory(X_2d, cube)  # documented: a view, no copy
    restored = unflatten_cube(X_2d, spatial_shape)
    np.testing.assert_array_equal(restored, cube)
    assert np.shares_memory(restored, cube)


def test_one_value_per_pixel_becomes_a_2d_map():
    predictions = np.arange(20.0)
    assert unflatten_cube(predictions, (4, 5)).shape == (4, 5)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: flatten_cube(np.ones((4, 5))), "X_3d must have shape"),
        (lambda: unflatten_cube(np.ones((20, 3)), (4, 0)), "must be an integer >= 1"),
        (lambda: unflatten_cube(np.ones((20, 3)), (4.0, 5)), "must be an integer"),
        (lambda: unflatten_cube(np.ones((50, 3)), (4, 5)), "X_2d has 50 rows"),
        (lambda: unflatten_cube(np.ones((20, 3)), 20), "spatial_shape must be a"),
    ],
)
def test_reshapes_reject_inconsistent_shapes(call, match):
    with pytest.raises(ValueError, match=match):
        call()
