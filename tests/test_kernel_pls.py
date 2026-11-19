"""The kernel engine: kernel PLS and Gram PCA, and fits on statistics equal to the pipeline path."""

from __future__ import annotations

import time
import tracemalloc
import numpy as np
import pytest
from enginir import PreprocessingRecipe, Project
from enginir.io import save_hdf5
from enginir.modeling import _gram_fit, fit_model_run
from enginir.models import PCR, PLSRegression, _gram
from enginir.models._gram import GramStats, blas_thread_count, gram_condition


# -- Fits on statistics equal the pipeline path (plan ``plans/kernel-pls.md``, Phase 2) ----


N_BANDS = 40


_CV_KEYS = {"rmsecv", "r2", "rpd", "rpiq", "pixel_rmsecv", "pixel_r2", "pixel_rpd",
            "pixel_rpiq", "interval_coverage_pixel", "interval_coverage_roi"}


#: Curve values from pixel predictions: the pipeline computes those in float32.
_CV_CURVE = {"rmsecv", "rmsecv_se", "q2", "q2_cum", "r2y_cum", "r2y_cum_roi", "rmsee_roi"}


def _project(tmp_path, *, n_datasets=2, n_rois=6, frames=(3, 8), seed=0):
    """Two scans of 6 ROIs (3 … 7 frames × 6 rows), API from 2 to 9 %, an odd pixel."""
    rng = np.random.default_rng(seed)
    axis = np.arange(N_BANDS)
    peaks = [np.exp(-((axis - c) / w) ** 2) for c, w in ((8, 3), (20, 5), (31, 2.5))]
    project = Project.create(tmp_path / "p.h5")
    project.add_target("API", unit="%")
    for d in range(n_datasets):
        blocks, rois, start = [], [], 0
        for i in range(n_rois):
            size = frames[0] + (i * 7 + d) % (frames[1] - frames[0])
            api = 2 + 1.3 * i + 0.4 * d
            spectra = (2600 - 140 * api * peaks[0] - 90 * rng.uniform(0, 3) * peaks[1]
                       - 60 * rng.uniform(0, 1) * peaks[2])
            block = (spectra + rng.normal(0, 18, (size, 6, N_BANDS))
                     + rng.normal(0, 30, (size, 6, 1)))
            if i == 2 and d == 0:
                block[1, 2] += 400 * peaks[2]  # a pixel the DModX exclusion should find
            blocks.append(block)
            rois.append((f"d{d}r{i}", start, start + size - 1, api))
            start += size
        path = tmp_path / f"scan{d}.h5"
        save_hdf5(str(path), np.vstack(blocks).astype(np.uint16),
                  np.linspace(1000, 1600, N_BANDS))
        dataset = project.add_dataset(path)
        for name, a, b, api in rois:
            dataset.add_roi(name, frames=(a, b), rows=(0, 5), values={"API": float(api)})
    return project


def _both(project, *, model="PLS", recipe=None, n_components=3, fit_spec=None, reference=None):
    """The run on statistics and the pipeline run (``reference``) of the same request."""
    first = project.fit("API", model=model, recipe=recipe or PreprocessingRecipe(snv=True),
                        n_components=n_components, fit_spec=fit_spec)
    ws = project.workspace
    results = []
    for algorithm in (None, reference or ("nipals" if model == "PLS" else "svd")):
        hyperparameters: dict = {"n_components": n_components}
        if algorithm is not None:
            hyperparameters["algorithm"] = algorithm
        results.append(fit_model_run(
            project_path=project.path, sample_set=ws.sample_sets[first.record.sample_set_id],
            datasets=ws.datasets, reference_corrections=ws.reference_corrections, rois=ws.rois,
            target_assignments=ws.target_assignments, recipes=ws.recipes,
            signal_domains=ws.signal_domains, signal_domain_id=first.record.signal_domain_id,
            model_type=model, hyperparameters=hyperparameters,
            fit_spec=first.record.fit_specification.to_dict()))
    return results


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    if a.size == 0:
        return 0.0
    assert np.array_equal(np.isnan(a), np.isnan(b)), "undefined in one path only"
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _assert_equal_mapping(ours, reference, where, tolerance) -> None:
    assert set(ours) == set(reference), where
    for key, value in reference.items():
        if isinstance(value, dict):
            _assert_equal_mapping(ours[key], value, f"{where}/{key}", tolerance)
            continue
        array = np.asarray(value)
        if array.dtype.kind in "fiub":
            assert _rel(ours[key], value) < tolerance(key), f"{where}/{key}"
        else:
            assert list(np.atleast_1d(ours[key])) == list(np.atleast_1d(value)), key


def _assert_equal_runs(ours, reference, *, model_tol, cv_tol):
    a, b = ours.model_run, reference.model_run
    assert a.hyperparameters["n_components"] == b.hyperparameters["n_components"]
    assert a.cv_spec["type"] == b.cv_spec["type"] and a.cv_spec["n_splits"] == b.cv_spec["n_splits"]
    assert a.cv_spec["warnings"] == b.cv_spec["warnings"]
    spread = float(np.std(reference.pixel_y_true))
    for key, value in b.metrics.items():
        if isinstance(value, str) or isinstance(value, bool):
            assert a.metrics[key] == value, key
        elif key in ("bias", "pixel_bias"):
            # Near zero, so no relative tolerance: absolute, at the precision of the other
            # cross-validation metrics (1e-6 alone failed by 1 % for PCR's float32 paths).
            assert abs(a.metrics[key] - value) < cv_tol * spread, key
        elif key.endswith("_n") or key.startswith("n_"):
            assert a.metrics[key] == value, key
        else:
            assert _rel(a.metrics[key], value) < (cv_tol if key in _CV_KEYS else model_tol), key
    for group in ("component_curve", "diagnostics", "prediction_model", "exclusion"):
        _assert_equal_mapping(a.artifact_refs[group], b.artifact_refs[group], group,
                              lambda key, g=group: (cv_tol if g == "component_curve"
                                                    and key in _CV_CURVE else model_tol))
    assert _rel(a.artifact_refs["cv_outputs"]["y_pred_roi"],
                b.artifact_refs["cv_outputs"]["y_pred_roi"]) < cv_tol
    assert _rel(ours.pixel_y_pred, reference.pixel_y_pred) < cv_tol


@pytest.mark.parametrize(("fit_spec", "recipe", "n_components"), [
    (None, PreprocessingRecipe(snv=True), 4),
    (None, PreprocessingRecipe(), 3),
    ({"folds": {"rule": "leave_one_group_out"}, "pixels": {"per_roi": 10, "seed": 3}},
     PreprocessingRecipe(snv=True), "auto"),
    ({"folds": {"rule": "group_kfold", "n_folds": 4}}, PreprocessingRecipe(), 3),
    ({"folds": {"group_by": "target"}}, PreprocessingRecipe(snv=True), 3),
    ({"exclusion": {"method": "dmodx", "level": 0.95}}, PreprocessingRecipe(snv=True), 3),
    ({"exclusion": {"method": "dmodx", "formula": "dmodx-2", "rounds": 2, "level": 0.95}},
     PreprocessingRecipe(), 3),
], ids=["snv", "raw", "logo-subsample-auto", "group-kfold", "group-by-target",
        "exclusion", "exclusion-dmodx2-2-rounds"])
def test_pls_on_statistics_equals_the_pipeline(tmp_path, fit_spec, recipe, n_components):
    ours, reference = _both(_project(tmp_path), recipe=recipe, n_components=n_components,
                            fit_spec=fit_spec)
    assert ours.model_run.hyperparameters["algorithm"] == "kernel"
    _assert_equal_runs(ours, reference, model_tol=1e-8, cv_tol=1e-5)
    if fit_spec and "exclusion" in fit_spec:
        assert ours.model_run.metrics["n_excluded_pixels"] >= 1


@pytest.mark.parametrize("fit_spec", [None, {"exclusion": {"method": "dmodx"}}])
def test_pcr_on_statistics_equals_the_svd_pipeline(tmp_path, fit_spec):
    ours, reference = _both(_project(tmp_path), model="PCR", n_components=4, fit_spec=fit_spec)
    assert ours.model_run.hyperparameters["algorithm"] == "gram"
    assert reference.model_run.hyperparameters["algorithm"] == "svd"
    _assert_equal_runs(ours, reference, model_tol=1e-4, cv_tol=1e-4)


def test_msc_recipes_take_the_pipeline_path_with_the_kernel_engine(tmp_path, monkeypatch):
    """MSC/EMSC are fitted inside every fold: no fit on statistics, still kernel PLS."""
    calls: list[int] = []
    real = _gram_fit.component_curve
    monkeypatch.setattr(_gram_fit, "component_curve",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    ours, reference = _both(_project(tmp_path), recipe=PreprocessingRecipe(msc=True))
    assert not calls and ours.model_run.hyperparameters["algorithm"] == "kernel"
    _assert_equal_runs(ours, reference, model_tol=1e-6, cv_tol=1e-5)


@pytest.mark.parametrize("fault", ["leakage", "no-centering", "no-downdate", "wrong-fold"])
def test_the_equivalence_test_catches_the_faults_of_a_fit_on_statistics(tmp_path, monkeypatch,
                                                                        fault):
    """Each fault of the statistics path makes the comparison with the pipeline fail."""
    project = _project(tmp_path)
    spec = {"exclusion": {"method": "dmodx", "level": 0.95}} if fault == "no-downdate" else None
    if fault == "leakage":   # the held-out blocks stay in their fold's sums
        real_folds = _gram_fit.folds

        def leaky(splitter, cv_groups, lay):
            folds, scheme = real_folds(splitter, cv_groups, lay)
            return [_gram_fit.Fold(np.arange(lay.n_blocks), f.test_rois) for f in folds], scheme
        monkeypatch.setattr(_gram_fit, "folds", leaky)
    elif fault == "no-centering":
        real_moments = _gram.GramStats.moments

        def uncentred(self, blocks=None, columns=None):
            m = real_moments(self, blocks, columns)
            m.cxx = m.cxx + m.n * np.outer(m.x_mean - self.shift_x, m.x_mean - self.shift_x)
            return m
        monkeypatch.setattr(_gram.GramStats, "moments", uncentred)
    elif fault == "no-downdate":
        monkeypatch.setattr(_gram.GramStats, "remove_rows", lambda self, *a, **k: None)
    else:                    # every pixel predicted by the first fold's model
        real_held_out = _gram_fit.held_out_predictions
        monkeypatch.setattr(_gram_fit, "held_out_predictions",
                            lambda X, lay, folds, coefs: real_held_out(X, lay, folds,
                                                                      [coefs[0]] * len(coefs)))
    ours, reference = _both(project, fit_spec=spec)
    with pytest.raises(AssertionError):
        _assert_equal_runs(ours, reference, model_tol=1e-8, cv_tol=1e-5)


def test_a_model_without_algorithm_refits_with_nipals_and_a_kernel_model_with_kernel(tmp_path):
    """Contract §7: a 1.5 model (no `algorithm`) reads nipals; a refit reproduces exactly."""
    import h5py

    project = _project(tmp_path)
    model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=3)
    assert model.record.hyperparameters["algorithm"] == "kernel"
    again = model.refit()
    assert again.record.hyperparameters["algorithm"] == "kernel"
    assert again.metrics["rmsecv"] == model.metrics["rmsecv"]
    with h5py.File(project.path, "r+") as f:  # as a 1.5 project stored it
        del f[f"model_runs/{model.record.model_run_id}/hyperparameters"].attrs["algorithm"]
    old = Project.open(project.path).model(model.name)
    assert "algorithm" not in old.record.hyperparameters
    nipals = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=3,
                         algorithm="nipals")
    refitted = old.refit()
    assert refitted.record.hyperparameters["algorithm"] == "nipals"
    assert refitted.metrics["rmsecv"] == nipals.metrics["rmsecv"]


def test_a_fit_on_statistics_is_fast_and_reads_the_pixels_a_few_times(tmp_path):
    """A benchmark with a generous bound: 24 ROIs × 4800 pixels × 40 bands, 7 folds."""
    project = _project(tmp_path, n_rois=12, frames=(780, 820), seed=4)
    start = time.perf_counter()
    model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components="auto")
    seconds = time.perf_counter() - start
    assert model.metrics["n_pixels"] > 100_000
    assert seconds < 30, f"{seconds:.1f} s"


@pytest.mark.parametrize(("limit", "stored"), [(0.0, "gram"), (2.0, "svd")])
def test_a_pcr_run_stores_the_algorithm_its_guard_chose(tmp_path, monkeypatch, limit, stored):
    """PCR's "auto" is decided once for the run on the whole set; never stored as "auto"."""
    from enginir.modeling import _service

    monkeypatch.setattr(_service, "GRAM_CONDITION_LIMIT", limit)
    model = _project(tmp_path).fit("API", model="PCR", recipe=PreprocessingRecipe(snv=True),
                                   n_components=3)
    assert model.record.hyperparameters["algorithm"] == stored
    assert model.refit().record.hyperparameters["algorithm"] == stored


def test_a_stored_model_without_algorithm_reads_nipals_or_svd():
    from enginir.modeling import stored_hyperparameters

    assert stored_hyperparameters("PLS", {"n_components": 2})["algorithm"] == "nipals"
    assert stored_hyperparameters("PCR", {"n_components": 2})["algorithm"] == "svd"
    assert stored_hyperparameters("PLS", {"algorithm": "kernel"})["algorithm"] == "kernel"


# -- The kernel engine (plan ``plans/kernel-pls.md``, Phases 1 and 3b) -----------------


_ATTRIBUTES = ("coef_", "intercept_", "x_mean_", "x_std_", "y_mean_", "y_std_", "x_weights_",
               "x_loadings_", "y_loadings_", "x_rotations_", "x_scores_", "y_scores_",
               "x_score_ss_")


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
