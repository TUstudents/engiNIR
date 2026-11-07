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
    # 9 significant digits are exact for float32, but the conversion to
    # absorbance (vectorised log10) may round the last bit differently on
    # another CPU or NumPy build: allow a few float32 rounding steps, far below
    # what any SIMCA comparison resolves (~1e-5).
    np.testing.assert_allclose(X, want_X, rtol=1e-6, atol=0)


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


def _simca_export_module():
    """The SIMCA export script, loaded with a stand-in for SIMCA's ``umetrics``."""
    import importlib.util
    import sys
    import types

    spec = importlib.util.spec_from_file_location(
        "simca_export", ref.DATA_DIR / "simca_export.py")
    module = importlib.util.module_from_spec(spec)
    saved = sys.modules.get("umetrics")
    sys.modules["umetrics"] = types.ModuleType("umetrics")
    try:
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            del sys.modules["umetrics"]
        else:
            sys.modules["umetrics"] = saved
    return module


def test_simca_default_cv_groups_on_the_imported_order_are_the_round_robin_roi_folds():
    """SIMCA's every-7th groups on the interleaved pixels equal engiNIR's round-robin folds."""
    from enginir.validation import RoundRobinGroups

    _, body = _simca_export_module().read_table("pixels")
    ordered = _simca_export_module().interleave_by_fold(body)
    assert sorted(ordered) == sorted(body)
    simca_groups = {}
    for k, row in enumerate(ordered):  # SIMCA default: every 7th observation in one group
        simca_groups.setdefault(k % 7, set()).add(row[1])
    rois = [row[1] for row in body]
    roi_order = list(dict.fromkeys(rois))
    engi_groups = [{rois[i] for i in test} for _, test in
                   RoundRobinGroups(7).split(np.zeros((len(rois), 1)), groups=rois)]
    assert sorted(map(sorted, simca_groups.values())) == sorted(map(sorted, engi_groups))
    assert all(len(group) == 2 for group in simca_groups.values())
    assert roi_order[0] in simca_groups[0] and roi_order[7] in simca_groups[0]
