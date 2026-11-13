"""The kernel engine (plan ``plans/kernel-pls.md``, Phases 1 and 3b).

Kernel PLS and the Gram PCA must give the models of NIPALS and the full SVD
to floating-point precision (sample-set-model-run contract §7, `algorithm`),
from one chunked pass over the rows and without a full copy of X.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from enginir.models import PCR, PLSRegression
from enginir.models._gram import GramStats, blas_thread_count, gram_condition

_ATTRIBUTES = ("coef_", "intercept_", "x_mean_", "x_std_", "y_mean_", "y_std_", "x_weights_",
               "x_loadings_", "y_loadings_", "x_rotations_", "x_scores_", "y_scores_",
               "x_score_ss_")


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _spectra(n, k, *, targets=1, offset=0.0, dtype=np.float64, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, 8))
    X = latent @ rng.standard_normal((8, k)) + 0.05 * rng.standard_normal((n, k)) + offset
    Y = latent[:, :targets] * 2 + 0.1 * rng.standard_normal((n, targets))
    return X.astype(dtype), (Y[:, 0] if targets == 1 else Y)


@pytest.mark.parametrize("x_scaling", ["center", "uv", "pareto"])
@pytest.mark.parametrize(("n", "k", "offset", "dtype"), [
    (2000, 60, 0.0, np.float64),
    (2000, 60, 3000.0, np.float32),    # raw counts: float32 rows, a large offset
    (40, 120, 0.0, np.float64),        # more bands than samples
])
def test_kernel_pls_is_nipals(n, k, offset, dtype, x_scaling):
    X, y = _spectra(n, k, offset=offset, dtype=dtype)
    kernel = PLSRegression(n_components=6, x_scaling=x_scaling).fit(X, y)
    nipals = PLSRegression(n_components=6, x_scaling=x_scaling, algorithm="nipals").fit(X, y)
    for name in _ATTRIBUTES:
        assert _rel(getattr(kernel, name), getattr(nipals, name)) < 1e-8, name
    assert _rel(kernel.vip_scores(), nipals.vip_scores()) < 1e-8
    assert _rel(kernel.predict(X), nipals.predict(X)) < 1e-6  # in the dtype of X
    # Signs as scikit-learn's: the largest absolute element of every weight vector is positive.
    weights = kernel.x_weights_
    assert np.all(weights[np.argmax(np.abs(weights), axis=0), np.arange(6)] > 0)


@pytest.mark.parametrize("x_scaling", ["center", "uv"])
def test_kernel_pls2_is_the_ikpls_oracle(x_scaling):
    """PLS2: scikit-learn's NIPALS iterates for each weight vector (about 1e-6); ikpls is exact."""
    from ikpls.numpy import PLS

    X, Y = _spectra(1500, 40, targets=3, seed=4)
    ours = PLSRegression(n_components=5, x_scaling=x_scaling, y_scaling="center").fit(X, Y)
    oracle = PLS(algorithm=2, scale_X=x_scaling == "uv", scale_Y=False, ddof=1).fit(X, Y, A=5)
    assert _rel(ours.predict(X), oracle.predict(X, n_components=5)) < 1e-9
    nipals = PLSRegression(n_components=5, x_scaling=x_scaling, y_scaling="center",
                           algorithm="nipals", tol=1e-12, max_iter=5000).fit(X, Y)
    assert _rel(ours.coef_, nipals.coef_) < 1e-5


def test_kernel_pls1_is_the_ikpls_oracle():
    from ikpls.numpy import PLS

    X, y = _spectra(3000, 80, seed=7)
    ours = PLSRegression(n_components=10).fit(X, y)
    oracle = PLS(algorithm=2, scale_X=False, scale_Y=False).fit(X, y.reshape(-1, 1), A=10)
    for a in (1, 4, 10):
        assert _rel(ours.predict_components(X)[:, a - 1],
                    oracle.predict(X, n_components=a)[:, 0]) < 1e-9


def test_statistics_summed_from_blocks_equal_the_statistics_of_the_joined_rows():
    X, y = _spectra(5000, 30, offset=500.0, dtype=np.float32, seed=3)
    ranges = [(0, 1200, 0, 0), (1200, 2600, 1, 1), (2600, 4000, 0, 2), (4000, 5000, 2, 3)]
    blocks = GramStats.from_ranges(X, y, ranges, n_blocks=3, n_rois=4)
    joined = GramStats.from_rows(X, y)
    a, b = blocks.moments(), joined.moments()
    for name in ("x_mean", "y_mean", "cxx", "cxy", "cyy"):
        assert _rel(getattr(a, name), getattr(b, name)) < 1e-10, name
    part = blocks.moments([0, 2])
    rows = np.r_[0:1200, 2600:5000]
    direct = GramStats.from_rows(X[rows], y[rows]).moments()
    assert _rel(part.cxx, direct.cxx) < 1e-10 and _rel(part.x_mean, direct.x_mean) < 1e-10
    means, values = blocks.roi_means()
    np.testing.assert_allclose(means[1], X[1200:2600].astype(np.float64).mean(axis=0),
                               rtol=1e-10)
    np.testing.assert_allclose(values[2, 0], y[2600:4000].mean(), rtol=1e-12)
    # A downdate (the DModX exclusion) equals summing without those rows.
    keep = np.ones(5000, dtype=bool)
    keep[[5, 70, 1300, 4999]] = False
    removed = blocks.copy()
    block_of_row = np.repeat([0, 1, 0, 2], [1200, 1400, 1400, 1000])
    roi_of_row = np.repeat([0, 1, 2, 3], [1200, 1400, 1400, 1000])
    removed.remove_rows(X, y, np.flatnonzero(~keep), block_of_row, roi_of_row)
    kept = GramStats.from_ranges(X, y, ranges, n_blocks=3, n_rois=4, keep=keep)
    assert _rel(removed.moments().cxx, kept.moments().cxx) < 1e-10
    assert np.array_equal(removed.roi_n, kept.roi_n)


def test_raw_counts_with_an_offset_keep_their_precision():
    """The shift before summing keeps raw counts from cancelling when centering.

    Counts around 30 000 with a signal of a few counts, 200 000 rows: without
    the shift, Σx² − n·x̄² loses about eight digits (fault-injected).
    """
    X, y = _spectra(200_000, 20, offset=30_000.0, dtype=np.float32, seed=11)
    reference = PLSRegression(n_components=5, algorithm="nipals").fit(X.astype(np.float64), y)
    kernel = PLSRegression(n_components=5).fit(X, y)
    assert _rel(kernel.coef_, reference.coef_) < 1e-8


def test_a_kernel_fit_makes_no_full_copy_of_x():
    """Peak memory of a fit of 400 000 × 176 float32 rows: below a quarter of a float64 copy."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400_000, 176), dtype=np.float32)
    y = X[:, :3].sum(axis=1)
    tracemalloc.start()
    PLSRegression(n_components=6).fit(X, y)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < X.size * 8 / 4, f"peak {peak / 1e6:.0f} MB"


@pytest.mark.parametrize("x_scaling", ["center", "uv", "pareto"])
@pytest.mark.parametrize(("n", "k", "offset", "dtype"), [
    (3000, 80, 0.0, np.float64), (3000, 80, 3000.0, np.float32), (50, 120, 0.0, np.float64)])
def test_gram_pcr_is_the_full_svd(n, k, offset, dtype, x_scaling):
    """Against the SVD of the same rows in float64 (scikit-learn's PCA keeps float32 input)."""
    X, y = _spectra(n, k, offset=offset, dtype=dtype, seed=2)
    gram = PCR(n_components=8, x_scaling=x_scaling, algorithm="gram").fit(X, y)
    svd = PCR(n_components=8, x_scaling=x_scaling, algorithm="svd").fit(X.astype(np.float64), y)
    assert gram.algorithm_ == "gram" and svd.algorithm_ == "svd"
    for name in ("coef_", "intercept_", "scores_", "explained_variance_ratio_"):
        assert _rel(getattr(gram, name), getattr(svd, name)) < 1e-8, name
    cosines = np.sum(gram.pca_.components_ * svd.pca_.components_, axis=1)
    assert np.all(cosines > 0), "signs as scikit-learn's PCA"
    assert np.degrees(np.arccos(np.clip(cosines, -1, 1))).max() < 1e-4
    X64 = X.astype(np.float64)  # predictions work in the dtype of their input
    assert _rel(gram.predict_components(X64), svd.predict_components(X64)) < 1e-8


def test_pcr_auto_uses_the_svd_beyond_the_conditioning_limit():
    """Near the rank limit (e.g. after SNV) the Gram eigenvalues lose precision: the guard."""
    rng = np.random.default_rng(9)
    X = rng.standard_normal((500, 3)) @ rng.standard_normal((3, 40))   # rank 3
    X += 1e-9 * rng.standard_normal(X.shape)                          # tiny noise beyond it
    y = X[:, 0]
    assert gram_condition(GramStats.from_rows(X, y).moments(), 6, "center") < 1e-6
    assert PCR(n_components=6).fit(X, y).algorithm_ == "svd"
    assert PCR(n_components=3).fit(X, y).algorithm_ == "gram"


def test_the_engine_caps_its_blas_threads(monkeypatch):
    monkeypatch.setenv("ENGINIR_BLAS_THREADS", "3")
    assert blas_thread_count() == 3
    monkeypatch.setenv("ENGINIR_BLAS_THREADS", "many")
    assert blas_thread_count() >= 1
    monkeypatch.delenv("ENGINIR_BLAS_THREADS")
    assert 1 <= blas_thread_count()


def test_an_unknown_algorithm_is_refused():
    X, y = _spectra(100, 10)
    with pytest.raises(ValueError, match="algorithm"):
        PLSRegression(algorithm="svd").fit(X, y)
    with pytest.raises(ValueError, match="algorithm"):
        PCR(algorithm="kernel").fit(X, y)


def test_exhausted_components_have_zero_y_scores_as_with_nipals():
    """SNV-like rank loss: more components than the data support (1.5.1 gave NaN)."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((50, 3)) @ rng.standard_normal((3, 20))
    y = X[:, 0] + 0.1 * rng.standard_normal(50)
    with np.errstate(all="raise"):
        kernel = PLSRegression(n_components=6, algorithm="kernel").fit(X, y)
    nipals = PLSRegression(n_components=6, algorithm="nipals").fit(X, y)
    assert np.isfinite(kernel.y_scores_).all()
    assert np.allclose(kernel.y_scores_[:, 3:], 0.0)
    assert np.allclose(kernel.y_scores_, nipals.y_scores_, atol=1e-8)
