"""Shared validators (enginir._checks) and reshaping between cubes and pixel matrices."""

from __future__ import annotations

import numpy as np
import pytest
from enginir._checks import (
    as_float_array,
    as_float_matrix,
    as_float_vector,
    check_choice,
    check_integer,
    check_real,
)
from enginir.utils import flatten_cube, unflatten_cube


# -- Tests for the shared validators in enginir._checks --------------------------------


def test_numbers_must_be_real_numbers_not_booleans():
    assert check_integer(np.int16(3), "n", minimum=1) == 3
    assert check_real(np.float32(0.5), "x", positive=True) == 0.5
    for value in (True, np.bool_(True), 2.5, "3"):
        with pytest.raises(ValueError):
            check_integer(value, "n")
    for value in (False, float("nan"), float("inf"), "1.0"):
        with pytest.raises(ValueError):
            check_real(value, "x")


def test_choice_rejects_unhashable_values_with_a_value_error():
    assert check_choice("a", "mode", {"a", "b"}) == "a"
    with pytest.raises(ValueError, match="mode must be one of"):
        check_choice(["a"], "mode", {"a", "b"})


def test_float_arrays_keep_float32_and_float64_and_promote_the_rest():
    assert as_float_array(np.ones(3, np.float32), "X").dtype == np.float32
    assert as_float_array(np.ones(3, np.float64), "X").dtype == np.float64
    for dtype in (np.uint16, np.int64, np.bool_, np.float16):
        assert as_float_array(np.ones(3, dtype), "X").dtype == np.float32


@pytest.mark.parametrize(
    "value",
    [np.ones(3, np.complex64), np.array(["1.5", "2"]), [1.0, None],
     np.array(["2026-01-01"], dtype="datetime64[D]")],
)
def test_float_arrays_reject_non_real_input(value):
    with pytest.raises(ValueError, match="real numeric array"):
        as_float_array(value, "X")


def test_float_matrices_must_be_2d_non_empty_and_finite():
    with pytest.raises(ValueError, match="2-D"):
        as_float_matrix(np.ones(3), "X")
    with pytest.raises(ValueError, match="at least one"):
        as_float_matrix(np.empty((0, 3)), "X")
    with pytest.raises(ValueError, match="finite"):
        as_float_matrix(np.array([[1.0, np.inf]]), "X")
    with pytest.raises(ValueError, match="finite"):
        as_float_vector(np.array([1.0, np.nan]), "w")
    with pytest.raises(ValueError, match="has 2 values but X has 3"):
        as_float_vector(np.ones(2), "w", size=3)


# -- Reshaping between image cubes and per-pixel matrices ------------------------------


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
