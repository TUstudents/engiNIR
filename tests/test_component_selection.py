"""Choosing the number of components (sample-set-model-run contract §5, §6, §7, §7.4, §7.5)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from scipy.stats import f

from enginir import Project
from enginir.modeling import _components
from enginir.models import PCR, PLSRegression
from enginir.validation import RoundRobinGroups, choose_components, dmodx
from tests.test_fit_spec import _uneven_project, _warnings_during


# -- the rules ---------------------------------------------------------------------------


@pytest.mark.parametrize(("curve", "rule", "expected"), [
    ({"rmsecv": [3.0, 2.0, 2.0, 2.5], "rmsecv_se": [0, 0, 0, 0], "q2": [.5, .5, .1, -1]},
     "min_rmsecv", 2),                                    # ties: the smaller number
    ({"rmsecv": [3.0, 2.1, 2.0, 2.05], "rmsecv_se": [0, 0, 0.2, 0], "q2": [.5, .5, .1, 0]},
     "one_se", 2),                                        # within one SE of the minimum
    ({"rmsecv": [1.0, 1.0, 1.0], "rmsecv_se": [0, 0, 0], "q2": [0.2, 0.0, 0.0]},
     "one_se", 1),                                        # a flat curve: one component
    ({"rmsecv": [5, 4, 3, 2, 1.0], "rmsecv_se": [0] * 5, "q2": [.9, .5, .3, .2, .1]},
     "q2_limit", 5),                                      # every component passes
    ({"rmsecv": [5, 4, 3.0], "rmsecv_se": [0] * 3, "q2": [.01, .5, .5]},
     "q2_limit", 1),                                      # the first fails: 1
    ({"rmsecv": [5, 4, 3.0], "rmsecv_se": [0] * 3, "q2": [.5, float("nan"), .5]},
     "q2_limit", 1),                                      # a NaN Q² fails
])
def test_the_rules_choose_as_the_contract_says(curve, rule, expected):
    assert choose_components(curve, rule, n_samples=20) == expected


def test_the_q2_limit_is_lower_for_more_than_100_samples():
    curve = {"rmsecv": [3, 2, 1.0], "rmsecv_se": [0] * 3, "q2": [0.5, 0.03, 0.01]}
    assert choose_components(curve, "q2_limit", n_samples=100) == 1
    assert choose_components(curve, "q2_limit", n_samples=101) == 3


# -- the curve ---------------------------------------------------------------------------


def _three_sources(seed=5, n_rois=30, pixels=6, bands=40):
    """Spectra built from exactly three independent sources; y follows the first."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, bands)
    sources = np.stack([np.exp(-((grid - c) / 0.08) ** 2) for c in (0.25, 0.5, 0.75)])
    amounts = rng.uniform(0.2, 1.0, (n_rois, 3))
    X = np.repeat(amounts, pixels, axis=0) @ sources + rng.normal(0, 0.002, (n_rois * pixels,
                                                                              bands))
    y = np.repeat(amounts[:, 0], pixels)
    roi_index = np.repeat(np.arange(n_rois), pixels)
    return X, y, roi_index


def test_the_one_se_rule_finds_the_number_of_independent_sources():
    X, y, roi_index = _three_sources()
    curve, oof, _scheme = _components.component_curve(
        PLSRegression(n_components=8, x_scaling="center"), X, y,
        splitter=RoundRobinGroups(7), cv_groups=roi_index, roi_index=roi_index, n_samples=30)
    assert curve["suggested_one_se"] == 3
    assert oof.shape == (X.shape[0], 8)
    assert np.all(np.diff(curve["r2y_cum"]) >= -1e-12), "pixel R²Y never decreases"
    assert curve["r2x_cum"][2] > 0.999 and curve["q2_cum"][2] > 0.99


def test_r2x_of_pcr_is_its_explained_variance():
    X, y, roi_index = _three_sources()
    curve, _oof, _scheme = _components.component_curve(
        PCR(n_components=5, x_scaling="center"), X, y, splitter=RoundRobinGroups(7),
        cv_groups=roi_index, roi_index=roi_index, n_samples=30)
    expected = np.cumsum(PCR(n_components=5, x_scaling="center").fit(X, y)
                         .explained_variance_ratio_)
    np.testing.assert_allclose(curve["r2x_cum"], expected, rtol=1e-8)


def test_the_curve_at_the_runs_components_is_the_runs_cross_validation(tmp_path):
    """Without exclusion one cross-validation serves both; the curve uses the run's folds."""
    project = _uneven_project(tmp_path)
    model = project.fit("API", n_components=2)
    curve = model.component_curve
    assert curve["rmsecv"][1] == pytest.approx(model.metrics["rmsecv"], rel=1e-6)
    spec = model.ref.fit_specification
    assert (spec.components_rule, spec.components_selected) == ("fixed", 2)
    assert spec.components_max == len(curve["rmsecv"]) >= 2


def test_auto_chooses_by_the_rule_and_a_refit_keeps_the_number(tmp_path):
    project = _uneven_project(tmp_path)
    model = project.fit("API", n_components="auto")
    curve = model.component_curve
    chosen = model.ref.hyperparameters["n_components"]
    assert chosen == curve["suggested_one_se"]
    spec = model.ref.fit_specification
    assert (spec.components_rule, spec.components_selected) == ("one_se", chosen)
    by_minimum = project.fit("API", n_components="auto",
                             fit_spec={"components": {"rule": "min_rmsecv"}})
    assert by_minimum.ref.hyperparameters["n_components"] == curve["suggested_min_rmsecv"]

    other = 1 if chosen != 1 else 2  # pretend the stored model chose another number
    with h5py.File(project.path, "r+") as f:
        group = f[f"model_runs/{model.ref.model_run_id}"]
        group["hyperparameters"].attrs["n_components"] = other
        group["fit_spec/components"].attrs["selected"] = other
    stored = Project.open(project.path).model(model_run_id=model.ref.model_run_id)
    refitted, messages = _warnings_during(stored.refit)
    assert refitted.ref.hyperparameters["n_components"] == other
    assert any("would now choose" in message for message in messages)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"n_components": 2, "fit_spec": {"components": {"rule": "one_se"}}}, "given number"),
    ({"n_components": 2, "fit_spec": {"components": {"selected": 2}}}, "written by the fit"),
    ({"n_components": "auto", "fit_spec": {"components": {"max": 500}}}, "at most"),
    ({"n_components": "many"}, "number or 'auto'"),
])
def test_requests_that_cannot_be_met_are_refused(tmp_path, kwargs, message):
    project = _uneven_project(tmp_path)
    with pytest.raises(ValueError, match=message):
        project.fit("API", **kwargs)
    assert project.models == {}


# -- training diagnostics ------------------------------------------------------------------


def test_training_statistics_follow_their_formulas(tmp_path):
    """RMSEE (N - 1 - A), T² with its F-form limit, DModX (dmodx-1), CoeffCS (§5, §7.5)."""
    project = _uneven_project(tmp_path)
    model = project.fit("API", n_components=2)
    diagnostics, metrics = model.diagnostics, model.metrics
    scores = np.asarray(diagnostics["x_scores_"], dtype=np.float64)
    n, a = scores.shape
    t2 = (scores ** 2 / scores.var(axis=0, ddof=1)).sum(axis=1)
    np.testing.assert_allclose(diagnostics["t2"], t2, rtol=1e-4)
    assert diagnostics["t2_crit_099"] == pytest.approx(
        a * (n - 1) / (n - a) * f.ppf(0.99, a, n - a), rel=1e-9)
    assert diagnostics["dmodx"].shape == (n,) and diagnostics["dcrit_099"] > 1
    assert metrics["pixel_rmsee"] > 0 and metrics["pixel_r2_adj"] < 1
    assert metrics["rmsee"] > 0  # 9 ROIs, 2 components: R - 1 - A = 6
    coef = np.asarray(diagnostics["coef_"], dtype=np.float64)
    assert np.isfinite(diagnostics["coef_cs"]).all() and diagnostics["coef_cs"].shape == coef.shape


def test_training_dmodx_equals_the_exclusion_formula():
    """§7.5 DModX of the fitted pixels is dmodx-1 of the model's residuals."""
    X, y, roi_index = _three_sources()
    model = PLSRegression(n_components=2, x_scaling="center").fit(X, y)
    _metrics, diagnostics, _projection = _components.training_diagnostics(
        model, X, y, np.ones(X.shape[0], bool), roi_index, y_divisor=float(model.y_std_[0]))
    from enginir.modeling._exclusion import x_residuals

    residuals, a, _centred = x_residuals(model, X)
    expected, _s0 = dmodx(residuals, a)
    np.testing.assert_allclose(diagnostics["dmodx"], expected, rtol=1e-5)
    assert diagnostics["roi_share_above_dcrit"].shape == (30,)


def test_a_selected_number_that_differs_from_the_model_is_refused_on_reading(tmp_path):
    project = _uneven_project(tmp_path)
    model = project.fit("API", n_components=2)
    with h5py.File(project.path, "r+") as f:
        f[f"model_runs/{model.ref.model_run_id}/fit_spec/components"].attrs["selected"] = 3
    with pytest.raises(ValueError, match="differs from hyperparameters/n_components"):
        Project.open(project.path)


def test_the_standard_error_is_the_delta_method_of_the_roi_errors():
    """SE(a) = s(e_r²) / (2 RMSECV(a) √R), e_r the ROI-mean cross-validated errors (§7.4)."""
    X, y, roi_index = _three_sources(seed=8)
    curve, oof, _scheme = _components.component_curve(
        PLSRegression(n_components=4, x_scaling="center"), X, y, splitter=RoundRobinGroups(7),
        cv_groups=roi_index, roi_index=roi_index, n_samples=30)
    errors = np.stack([np.bincount(roi_index, weights=oof[:, a]) / np.bincount(roi_index)
                       for a in range(4)], axis=1) - (np.bincount(roi_index, weights=y)
                                                      / np.bincount(roi_index))[:, None]
    rmsecv = np.sqrt((errors ** 2).mean(axis=0))
    np.testing.assert_allclose(curve["rmsecv"], rmsecv, rtol=1e-10)
    np.testing.assert_allclose(
        curve["rmsecv_se"], (errors ** 2).std(axis=0, ddof=1) / (2 * rmsecv * np.sqrt(30)),
        rtol=1e-10)
