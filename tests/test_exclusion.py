"""DModX and pixel exclusion (sample-set-model-run contract §6, §7.2)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from enginir import Project
from enginir.io import save_hdf5
from enginir.modeling import _exclusion, _service
from enginir.validation import dmodx, dmodx_limit
from tests.test_fit_spec import _warnings_during

N_BANDS = 30
EXCLUSION = {"exclusion": {"method": "dmodx", "level": 0.99}}


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


def test_dmodx_needs_degrees_of_freedom():
    with pytest.raises(ValueError, match="N - A - A0 >= 1"):
        dmodx(np.ones((5, 10)), 4)
    with pytest.raises(ValueError, match="K - A >= 1"):
        dmodx_limit(50, 4, 4, 0.99)


# -- exclusion in a fit -----------------------------------------------------------------


def _project_with_an_odd_pixel(tmp_path):
    """8 ROIs of 4 frames x 3 rows; pixel 5 of ROI s3 carries a band the others lack."""
    rng = np.random.default_rng(4)
    content = np.linspace(1.0, 8.0, 8)
    bands = np.arange(N_BANDS)
    peak = np.exp(-((bands - 12) / 3.0) ** 2)
    cube = np.vstack([2600 - 150 * c * peak + rng.normal(0, 8, (4, 3, N_BANDS))
                      for c in content])
    odd = np.exp(-((bands - 24) / 1.5) ** 2)
    cube[3 * 4 + 1, 2] += 400 * odd  # ROI s3: frame 1, row 2 -> pixel 1 * 3 + 2 = 5
    path = tmp_path / "scan.h5"
    save_hdf5(path, cube.astype(np.uint16), np.linspace(1000, 1600, N_BANDS))
    project = Project.create(tmp_path / "odd.h5")
    scan = project.add_dataset(path)
    project.add_target("API", unit="%")
    for i, value in enumerate(content):
        scan.add_roi(f"s{i}", frames=(4 * i, 4 * i + 3), rows=(0, 2), values={"API": float(value)})
    return project


def test_an_odd_pixel_is_excluded_and_stored(tmp_path):
    project = _project_with_an_odd_pixel(tmp_path)
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


def test_no_roi_loses_more_than_its_share(tmp_path):
    """A low level puts many pixels above DCrit; every ROI keeps at least 1 - max_fraction."""
    project = _project_with_an_odd_pixel(tmp_path)
    model = project.fit("API", n_components=2, fit_spec={"exclusion": {
        "method": "dmodx", "level": 0.51, "rounds": 2, "max_fraction": 0.1}})
    assert all(len(indices) <= 1 for indices in model.excluded_pixels.values())
    refs = project.workspace.model_runs[model.ref.model_run_id]
    assert refs.fit_specification.exclusion_rounds == 2


def test_the_exclusion_sees_only_the_training_pixels_of_each_fold(tmp_path, monkeypatch):
    """Leakage rule: in every fold, the exclusion is fitted on exactly that fold's training rows."""
    project = _project_with_an_odd_pixel(tmp_path)
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
    project.fit("API", n_components=2, fit_spec=EXCLUSION)
    assert len(seen) == len(folds) + 1  # every fold, then the final model on all pixels
    for (train, test), rows in zip(folds, seen):
        assert np.array_equal(np.sort(rows), np.sort(train))
        assert not np.intersect1d(rows, test).size


def test_a_refit_repeats_the_exclusion_and_reports_a_different_one(tmp_path):
    project = _project_with_an_odd_pixel(tmp_path)
    model = project.fit("API", n_components=2, fit_spec=EXCLUSION)
    refitted, messages = _warnings_during(model.refit)
    assert refitted.metrics["rmsecv"] == pytest.approx(model.metrics["rmsecv"], rel=1e-9)
    assert not any("excluded other pixels" in message for message in messages)
    with h5py.File(project.path, "r+") as f:  # pretend the stored model excluded another pixel
        group = f[f"model_runs/{model.ref.model_run_id}/artifact_refs/exclusion"]
        indices = np.array(group["pixel_indices"]) if "pixel_indices" in group \
            else np.array(group.attrs["pixel_indices"])
        indices[0] = (indices[0] + 1) % 12
        if "pixel_indices" in group:
            del group["pixel_indices"]
        group.attrs["pixel_indices"] = indices.astype(np.uint32)
    reopened = Project.open(project.path).model(model_run_id=model.ref.model_run_id)
    _refitted, messages = _warnings_during(reopened.refit)
    assert any("excluded other pixels" in message for message in messages)
