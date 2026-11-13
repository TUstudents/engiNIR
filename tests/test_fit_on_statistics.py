"""Fits on statistics equal the pipeline path (plan ``plans/kernel-pls.md``, Phase 2).

``fit_model_run`` with a recipe without MSC/EMSC and a ``kernel`` (PLS) or
``gram`` (PCR) model works on sums per fold group; with ``nipals``/``svd``
it runs the pipeline (cross-validation with cloned estimators on the rows).
Every stored result must agree:

- the model, its diagnostics, projection data and exclusion to 1e-8;
- cross-validated values to 1e-5: the pipeline predicts held-out pixels in
  float32 (the working dtype), the statistics path predicts ROI means in
  float64; bias, near zero, to 1e-6 of the spread of y;
- PCR to 1e-4: scikit-learn's PCA computes float32 input in float32
  (``test_kernel_pls.py`` shows the Gram path equals a float64 SVD to 1e-8).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from enginir import PreprocessingRecipe, Project
from enginir.io import save_hdf5
from enginir.modeling import _gram_fit, fit_model_run
from enginir.models import _gram

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
            project_path=project.path, sample_set=ws.sample_sets[first.ref.sample_set_id],
            datasets=ws.datasets, calibrations=ws.calibrations, rois=ws.rois,
            target_assignments=ws.target_assignments, recipes=ws.recipes,
            signal_domains=ws.signal_domains, signal_domain_id=first.ref.signal_domain_id,
            model_type=model, hyperparameters=hyperparameters,
            fit_spec=first.ref.fit_specification.to_dict()))
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
            assert abs(a.metrics[key] - value) < 1e-6 * spread, key
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
    assert model.ref.hyperparameters["algorithm"] == "kernel"
    again = model.refit()
    assert again.ref.hyperparameters["algorithm"] == "kernel"
    assert again.metrics["rmsecv"] == model.metrics["rmsecv"]
    with h5py.File(project.path, "r+") as f:  # as a 1.5 project stored it
        del f[f"model_runs/{model.ref.model_run_id}/hyperparameters"].attrs["algorithm"]
    old = Project.open(project.path).model(model.name)
    assert "algorithm" not in old.ref.hyperparameters
    nipals = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=3,
                         algorithm="nipals")
    refitted = old.refit()
    assert refitted.ref.hyperparameters["algorithm"] == "nipals"
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
    assert model.ref.hyperparameters["algorithm"] == stored
    assert model.refit().ref.hyperparameters["algorithm"] == stored


def test_a_stored_model_without_algorithm_reads_nipals_or_svd():
    from enginir.modeling import stored_hyperparameters

    assert stored_hyperparameters("PLS", {"n_components": 2})["algorithm"] == "nipals"
    assert stored_hyperparameters("PCR", {"n_components": 2})["algorithm"] == "svd"
    assert stored_hyperparameters("PLS", {"algorithm": "kernel"})["algorithm"] == "kernel"
