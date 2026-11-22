"""Contract tests for ViewerState."""

from __future__ import annotations

import numpy as np
import pytest

from enginir.viewer._state import DEFAULT_BLACK_LEVEL, DEFAULT_WHITE_LEVEL, _EPS, ViewerState


@pytest.fixture()
def cube_and_wavelengths():
    rng = np.random.default_rng(42)
    cube = rng.integers(128, 3500, size=(3, 10, 5), dtype=np.uint16)
    wavelengths = np.linspace(1000.0, 1600.0, 5, dtype=np.float32)
    return cube, wavelengths


@pytest.fixture()
def state(qapp, cube_and_wavelengths):
    cube, wavelengths = cube_and_wavelengths
    return ViewerState(cube, wavelengths)


@pytest.mark.parametrize("shape", [(10, 5), (2, 3, 4, 5)])
def test_constructor_rejects_non_3d_input(qapp, shape):
    with pytest.raises(ValueError, match="3-D"):
        ViewerState(np.zeros(shape), np.arange(shape[-1]))


def test_constructor_rejects_invalid_wavelength_axis(qapp):
    cube = np.zeros((2, 3, 4), dtype=np.uint16)

    with pytest.raises(ValueError, match="wavelengths length 3 does not match cube band count 4"):
        ViewerState(cube, np.arange(3))

    with pytest.raises(ValueError, match="wavelengths must be 1-D"):
        ViewerState(cube, np.zeros((1, 4)))

    with pytest.raises(ValueError, match="wavelengths must contain only finite values"):
        ViewerState(cube, np.array([1.0, 2.0, np.nan, 4.0]))


def test_constructor_rejects_non_finite_default_levels(qapp, cube_and_wavelengths):
    cube, wavelengths = cube_and_wavelengths

    with pytest.raises(ValueError, match="black_level and white_level must be finite"):
        ViewerState(cube, wavelengths, black_level=np.nan)


def test_current_frame_image_default_levels_and_absorbance(qapp):
    """Raw counts at the black level, halfway and at the white level (known values)."""
    black, white = DEFAULT_BLACK_LEVEL, DEFAULT_WHITE_LEVEL
    halfway = (black + white) // 2  # 1614 for 128 and 3100: reflectance 0.5 up to one count
    cube = np.array([[[black, halfway, white]]], dtype=np.uint16)
    state = ViewerState(cube, np.array([1000.0, 1100.0, 1200.0], dtype=np.float32))

    state.set_y_unit("Reflectance")
    reflectance = state.current_frame_image()[0]
    assert reflectance[0] == pytest.approx(1e-6)  # clamped: a log of zero is never taken
    assert reflectance[1] == pytest.approx(0.5, abs=1e-3)
    assert reflectance[2] == pytest.approx(1.0)

    state.set_y_unit("Absorbance")
    absorbance = state.current_frame_image()[0]
    assert absorbance[1] == pytest.approx(0.30103, abs=1e-3)  # -log10(0.5)
    assert absorbance[2] == pytest.approx(0.0, abs=1e-6)


def test_current_frame_image_uses_explicit_refs_and_clamps_small_denominators(qapp):
    cube = np.full((1, 3, 2), 200, dtype=np.uint16)
    wavelengths = np.linspace(1000.0, 1100.0, 2)
    dark = np.full((3, 2), 100.0, dtype=np.float32)
    white = dark.copy()
    viewer_state = ViewerState(cube, wavelengths)
    viewer_state.set_dark_ref(dark)
    viewer_state.set_white_ref(white)
    viewer_state.set_y_unit("Reflectance")

    image = viewer_state.current_frame_image()
    expected = np.clip((cube[0].astype(np.float32) - dark) / _EPS, 1e-6, 1.0)
    np.testing.assert_allclose(image, expected)


def test_float_cubes_are_precalibrated_and_never_truncated(qapp):
    """A float cube is a precalibrated signal; casting it to uint16 zeroed reflectance."""
    cube = np.array([[[0.2, 0.4, 0.8], [0.1, 0.5, 1.2]]], dtype=np.float64)
    wavelengths = np.linspace(1000.0, 1200.0, 3)
    viewer_state = ViewerState(cube, wavelengths)
    assert viewer_state.calibrated and viewer_state.cube.dtype == np.float32

    viewer_state.set_y_unit("Reflectance")  # used as is, as in training (no clipping at 1)
    np.testing.assert_allclose(viewer_state.current_frame_image(), cube[0], rtol=1e-6)
    viewer_state.set_y_unit("Absorbance")
    np.testing.assert_allclose(viewer_state.current_frame_image(), -np.log10(cube[0]), rtol=1e-6)

    raw = ViewerState(np.full((1, 2, 3), 1000, dtype=np.int32), wavelengths)
    assert not raw.calibrated and raw.cube.dtype == np.uint16
    with pytest.raises(ValueError, match="fit into uint16"):
        ViewerState(np.full((1, 2, 3), 70000, dtype=np.int32), wavelengths)


def test_profile_and_axis_helpers_follow_current_units(state):
    image = state.current_frame_image()

    np.testing.assert_array_equal(state.get_spectrum(999), image[-1])
    np.testing.assert_array_equal(state.get_spatial_profile(-1), image[:, 0])

    state.set_x_spatial_unit("mm")
    state.set_pixel_size_mm(0.25)
    np.testing.assert_allclose(
        state.spatial_x_axis(),
        np.arange(state.height, dtype=np.float32) * 0.25,
    )

    state.set_x_spectral_unit("band")
    np.testing.assert_array_equal(
        state.spectral_x_axis(),
        np.arange(state.n_bands, dtype=np.float32),
    )


def test_reference_shapes_validate_resample_and_broadcast(qapp):
    cube = np.zeros((2, 4, 3), dtype=np.uint16)
    wavelengths = np.linspace(1000.0, 1200.0, 3)
    viewer_state = ViewerState(cube, wavelengths)

    with pytest.raises(ValueError, match="bands"):
        viewer_state.set_dark_ref(np.zeros((4, 2), dtype=np.float32))

    dark = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    white = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
    viewer_state.set_dark_ref(dark)
    viewer_state.set_white_ref(white)

    assert viewer_state.dark_ref is not None
    assert viewer_state.white_ref is not None
    assert viewer_state.dark_ref.shape == (4, 3)
    assert viewer_state.white_ref.shape == (4, 3)
    np.testing.assert_allclose(viewer_state.white_ref, np.repeat(white, 4, axis=0))


def test_raw_helpers_return_expected_shapes_and_ranges(state):
    band_image = state.raw_band_image(2)
    false_colour = state.raw_false_colour_rgb()

    assert band_image.shape == (state.height, state.n_frames)
    assert false_colour.shape == (state.height, state.n_frames, 3)
    assert false_colour.dtype == np.float32
    assert float(false_colour.min()) >= 0.0
    assert float(false_colour.max()) <= 1.0


