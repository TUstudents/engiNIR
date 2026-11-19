"""DModX and pixel exclusion (sample-set-model-run contract §6, §7.2)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from enginir import Project
from enginir.modeling import _exclusion, _gram_fit, _service
from enginir.validation import dmodx, dmodx_limit, modelling_power
from tests._scans import EXCLUSION, project_with_an_odd_pixel, warnings_during

N_BANDS = 30


# -- the formulas -----------------------------------------------------------------------


@pytest.mark.parametrize(("n", "k", "a", "level", "simca"), [
    (140, 100, 4, 0.95, 1.1757993698120117),
    (140, 100, 4, 0.99, 1.2577115297317505),
    (14, 100, 2, 0.95, 1.3428428173065186),
    (14, 100, 2, 0.99, 1.5213791131973267),
])
def test_dcrit_equals_simca_18(n, k, a, level, simca):
    """DCrit of the SIMCA reference models M1 and M4 (exported from SIMCA 18.1)."""
    assert dmodx_limit(n, k, a, level) == pytest.approx(simca, rel=1e-5)


def test_dmodx_is_normalised_so_that_its_mean_square_is_one():
    """With v = sqrt(N / (N - A - A0)): sum of normalised DModX^2 = N; s0 is the pooled SD."""
    rng = np.random.default_rng(1)
    residuals = rng.normal(size=(50, 20)) * rng.uniform(0.5, 2, (50, 1))
    distance, s0 = dmodx(residuals, 3)
    assert np.mean(distance ** 2) == pytest.approx(1.0, rel=1e-12)
    assert s0 == pytest.approx(np.sqrt((residuals ** 2).sum() / ((50 - 3 - 1) * (20 - 3))))
    assert np.argmax(distance) == np.argmax((residuals ** 2).sum(axis=1))


def test_modelling_power_is_the_explained_sd_clipped_at_zero():
    """MPow_k = max(0, 1 - SV_k / SV0_k), SV with N - A - A0 and SV0 with N - 1 (dmodx-2)."""
    rng = np.random.default_rng(2)
    centred = rng.normal(size=(12, 4))
    centred -= centred.mean(axis=0)
    residuals = centred * np.array([0.1, 0.5, 1.0, 0.0])
    mpow = modelling_power(residuals, centred, 2)
    ratio = np.sqrt((residuals ** 2).sum(0) / (12 - 2 - 1)) / np.sqrt((centred ** 2).sum(0) / 11)
    np.testing.assert_allclose(mpow, np.clip(1 - ratio, 0, None))
    assert mpow[2] == 0.0, "a residual SD above the initial SD (few observations) is clipped"
    assert mpow[3] == 1.0 and mpow[0] > mpow[1] > 0


def test_weighted_dmodx_scales_each_band_by_its_weight():
    rng = np.random.default_rng(3)
    residuals = rng.normal(size=(40, 10))
    plain, s0 = dmodx(residuals, 2)
    ones, s0_ones = dmodx(residuals, 2, weights=np.ones(10))
    np.testing.assert_allclose(ones, plain)
    assert s0_ones == pytest.approx(s0)
    odd = residuals.copy()
    odd[7, 9] += 25.0  # a large residual in band 9 only
    weights = np.ones(10)
    weights[9] = 0.0
    assert np.argmax(dmodx(odd, 2)[0]) == 7
    np.testing.assert_allclose(dmodx(odd, 2, weights=weights)[0],
                               dmodx(residuals, 2, weights=weights)[0])


def test_dmodx_needs_degrees_of_freedom():
    with pytest.raises(ValueError, match="N - A - A0 >= 1"):
        dmodx(np.ones((5, 10)), 4)
    with pytest.raises(ValueError, match="K - A >= 1"):
        dmodx_limit(50, 4, 4, 0.99)


# -- exclusion in a fit -----------------------------------------------------------------


def test_an_odd_pixel_is_excluded_and_stored(tmp_path):
    project = project_with_an_odd_pixel(tmp_path)
    model = project.fit("API", n_components=2, fit_spec=EXCLUSION)
    s3 = project.roi("s3").roi_id
    excluded = model.excluded_pixels
    assert 5 in excluded[s3]
    assert model.metrics["n_excluded_pixels"] == sum(len(v) for v in excluded.values())
    assert model.metrics["n_pixels"] == 8 * 12, "n_pixels counts the drawn pixels"
    budget = int(np.floor(0.2 * 12))
    assert all(len(indices) <= budget for indices in excluded.values())
    plain = project.fit("API", n_components=2)
    assert plain.excluded_pixels == {} and plain.metrics["n_excluded_pixels"] == 0


def test_an_exclusion_without_a_level_uses_simcas_095(tmp_path):
    """The default level is 0.95: DCrit of the fit before the exclusion, at 0.95, is stored."""
    project = project_with_an_odd_pixel(tmp_path)
    model = project.fit("API", n_components=2, cv=4, fit_spec={"exclusion": {"method": "dmodx"}})
    assert model.record.fit_specification.exclusion_level == 0.95
    assert model.metrics["n_excluded_pixels"] > 0
    n_drawn = int(model.metrics["n_pixels"])  # every drawn pixel, the excluded ones included
    k = int(np.asarray(model.diagnostics["mean_spectrum"]).size)
    with h5py.File(project.path, "r") as f:
        dcrit = float(f[f"model_runs/{model.record.model_run_id}/artifact_refs/exclusion"]
                      .attrs["dcrit"])
    assert dcrit == pytest.approx(dmodx_limit(n_drawn, k, 2, 0.95), rel=1e-6)


def test_no_roi_loses_more_than_its_share(tmp_path):
    """A low level puts many pixels above DCrit; every ROI keeps at least 1 - max_fraction."""
    project = project_with_an_odd_pixel(tmp_path)
    model = project.fit("API", n_components=2, fit_spec={"exclusion": {
        "method": "dmodx", "level": 0.51, "rounds": 2, "max_fraction": 0.1}})
    assert all(len(indices) <= 1 for indices in model.excluded_pixels.values())
    refs = project.workspace.model_runs[model.record.model_run_id]
    assert refs.fit_specification.exclusion_rounds == 2


def test_the_exclusion_sees_only_the_training_pixels_of_each_fold(tmp_path, monkeypatch):
    """Leakage rule: in every fold, the exclusion is fitted on exactly that fold's training rows."""
    project = project_with_an_odd_pixel(tmp_path)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    seen: list[np.ndarray] = []
    real_cv, real_fit = _service.cross_validate, _exclusion.DModXExclusion.fit

    def recording_cv(estimator, X, y, *, cv, groups, **kwargs):
        for train, test in cv.split(X, y, groups):
            folds.append((train, test))
        return real_cv(estimator, X, y, cv=cv, groups=groups, **kwargs)

    def recording_fit(self, X, y, sample_index=None):
        seen.append(np.arange(X.shape[0]) if sample_index is None else np.asarray(sample_index))
        return real_fit(self, X, y, sample_index=sample_index)

    monkeypatch.setattr(_service, "cross_validate", recording_cv)
    monkeypatch.setattr(_exclusion.DModXExclusion, "fit", recording_fit)
    project.fit("API", n_components=2, fit_spec=EXCLUSION, algorithm="nipals")  # pipeline path
    assert len(seen) == len(folds) + 1  # every fold, then the final model on all pixels
    for (train, test), rows in zip(folds, seen):
        assert np.array_equal(np.sort(rows), np.sort(train))
        assert not np.intersect1d(rows, test).size


def test_on_statistics_the_exclusion_sees_only_the_training_blocks_of_each_fold(tmp_path,
                                                                                monkeypatch):
    """The same leakage rule for fits on statistics: each fold excludes within its own blocks."""
    project = project_with_an_odd_pixel(tmp_path)
    folds: list[_gram_fit.Fold] = []
    seen: list[np.ndarray | None] = []
    real_folds, real_exclude = _gram_fit.folds, _gram_fit.exclude

    def recording_folds(*args, **kwargs):
        result = real_folds(*args, **kwargs)
        folds.extend(result[0])
        return result

    def recording_exclude(make, n, stats, X, y, lay, blocks, **kwargs):
        seen.append(None if blocks is None else np.asarray(blocks))
        result = real_exclude(make, n, stats, X, y, lay, blocks, **kwargs)
        if blocks is not None:
            excluded_rows = np.flatnonzero(result.excluded_)
            assert np.all(np.isin(lay.block_of_row[excluded_rows], blocks)), "a held-out pixel"
        return result

    monkeypatch.setattr(_gram_fit, "folds", recording_folds)
    monkeypatch.setattr(_gram_fit, "exclude", recording_exclude)
    project.fit("API", n_components=2, fit_spec=EXCLUSION)
    assert len(seen) == len(folds) + 1 and seen[-1] is None  # every fold, then all pixels
    for fold, blocks in zip(folds, seen):
        assert blocks is not None and np.array_equal(blocks, fold.train_blocks)


def test_a_refit_repeats_the_exclusion_and_reports_a_different_one(tmp_path):
    project = project_with_an_odd_pixel(tmp_path)
    model = project.fit("API", n_components=2, fit_spec=EXCLUSION)
    refitted, messages = warnings_during(model.refit)
    assert refitted.metrics["rmsecv"] == pytest.approx(model.metrics["rmsecv"], rel=1e-9)
    assert not any("excluded other pixels" in message for message in messages)
    with h5py.File(project.path, "r+") as f:  # pretend the stored model excluded another pixel
        group = f[f"model_runs/{model.record.model_run_id}/artifact_refs/exclusion"]
        indices = np.array(group["pixel_indices"]) if "pixel_indices" in group \
            else np.array(group.attrs["pixel_indices"])
        indices[0] = (indices[0] + 1) % 12
        if "pixel_indices" in group:
            del group["pixel_indices"]
        group.attrs["pixel_indices"] = indices.astype(np.uint32)
    reopened = Project.open(project.path).model(model_run_id=model.record.model_run_id)
    _refitted, messages = warnings_during(reopened.refit)
    assert any("excluded other pixels" in message for message in messages)


@pytest.mark.parametrize("formula", ["dmodx-1", "dmodx-2"])
def test_the_formula_of_the_specification_is_used_and_stored(tmp_path, monkeypatch, formula):
    """dmodx-2 weights by the modelling power (computed in every round), dmodx-1 does not."""
    project = project_with_an_odd_pixel(tmp_path)
    calls: list[int] = []
    real, real_gram = _exclusion.modelling_power, _gram_fit._modelling_power
    monkeypatch.setattr(_exclusion, "modelling_power",
                        lambda *args, **kwargs: calls.append(1) or real(*args, **kwargs))
    monkeypatch.setattr(_gram_fit, "_modelling_power",
                        lambda *args, **kwargs: calls.append(1) or real_gram(*args, **kwargs))
    model = project.fit("API", n_components=2,
                        fit_spec={"exclusion": {"method": "dmodx", "formula": formula}})
    assert model.record.fit_specification.exclusion_formula == formula
    assert bool(calls) == (formula == "dmodx-2")
    assert 5 in model.excluded_pixels[project.roi("s3").roi_id]
