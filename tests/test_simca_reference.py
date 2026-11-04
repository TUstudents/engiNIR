"""The SIMCA reference tables are what engiNIR fits on (plan fit-specification, Phase R)."""

from __future__ import annotations

import numpy as np
import pytest

from tests import _simca_reference as ref


@pytest.fixture(scope="module")
def computed(tmp_path_factory):
    return ref.tables(ref.build(tmp_path_factory.mktemp("simca_reference")))


@pytest.mark.parametrize("file_name", ["pixels.csv", "roi_means.csv", "pixels_test.csv"])
def test_the_committed_simca_tables_equal_the_spectra_engiNIR_fits_on(computed, file_name):
    """SIMCA and engiNIR start from the same numbers: same pixels, same absorbance, exactly."""
    ids, rois, y, X = ref.read_table(ref.DATA_DIR / file_name)
    want_ids, want_rois, want_y, want_X = computed[file_name]
    assert ids == want_ids
    assert rois == want_rois
    np.testing.assert_allclose(y, want_y, rtol=1e-9, atol=0)
    assert X.dtype == want_X.dtype == np.float32
    np.testing.assert_array_equal(X, want_X)  # 9 significant digits are exact for float32


def test_the_pixel_draw_depends_only_on_the_seed_and_the_roi_spectra():
    """Contract §7.1: same spectra and seed, same pixels (new ids or ROI order do not matter)."""
    cube, *_ = ref.make_cube()
    a = cube[0:6].reshape(-1, cube.shape[2])
    b = cube[6:12].reshape(-1, cube.shape[2])
    first = ref.draw(a, 10, ref.SEED)
    assert np.array_equal(first, ref.draw(a.copy(), 10, ref.SEED))
    assert np.all(np.diff(first) > 0) and first.size == 10 and first.max() < a.shape[0]
    assert not np.array_equal(first, ref.draw(b, 10, ref.SEED))
    assert not np.array_equal(first, ref.draw(a, 10, ref.SEED + 1))
    changed = a.copy()
    changed[0, 0] += 1
    assert not np.array_equal(first, ref.draw(changed, 10, ref.SEED))
