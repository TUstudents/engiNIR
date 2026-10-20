"""Tests for the shared validators in enginir._checks."""

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
