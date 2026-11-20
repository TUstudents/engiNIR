"""Choosing the number of components (sample-set-model-run contract §5, §6, §7, §7.4, §7.5)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from tests._tables import check_all
from scipy.stats import f

from enginir import Project
from enginir.modeling import _components
from enginir.models import PLSRegression
from enginir.validation import (
    RoundRobinGroups,
    choose_components,
    dmodx,
    dmodx_limit,
    guarded_choice,
    noise_components,
    structure_ceiling,
)
from tests._scans import uneven_project, warnings_during


# -- the rules ---------------------------------------------------------------------------


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


def test_auto_chooses_by_the_rule_and_a_refit_keeps_the_number(tmp_path):
    project = uneven_project(tmp_path)
    model = project.fit("API", n_components="auto")
    curve = model.component_curve
    chosen = model.record.hyperparameters["n_components"]
    spec = model.record.fit_specification
    assert spec.components_rule_choice == curve["suggested_one_se"]
    assert chosen == guarded_choice(curve, curve["suggested_one_se"], n_samples=9)
    assert (spec.components_rule, spec.components_selected, spec.components_noise_guard) \
        == ("one_se", chosen, "on")
    by_minimum = project.fit("API", n_components="auto",
                             fit_spec={"components": {"rule": "min_rmsecv"}})
    # "auto" turns the noise guard on unless the request says otherwise (contract §7.4).
    assert by_minimum.record.hyperparameters["n_components"] == guarded_choice(
        curve, curve["suggested_min_rmsecv"], n_samples=9)
    unguarded = project.fit("API", n_components="auto", fit_spec={"components": {
        "rule": "min_rmsecv", "noise_guard": "off"}})
    assert unguarded.record.hyperparameters["n_components"] == curve["suggested_min_rmsecv"]

    other = 1 if chosen != 1 else 2  # pretend the stored model chose another number
    with h5py.File(project.path, "r+") as f:
        group = f[f"model_runs/{model.record.model_run_id}"]
        group["hyperparameters"].attrs["n_components"] = other
        group["fit_spec/components"].attrs["selected"] = other
    stored = Project.open(project.path).model(model_run_id=model.record.model_run_id)
    refitted, messages = warnings_during(stored.refit)
    assert refitted.record.hyperparameters["n_components"] == other
    assert any("would now choose" in message for message in messages)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"n_components": 2, "fit_spec": {"components": {"rule": "one_se"}}}, "given number"),
    ({"n_components": 2, "fit_spec": {"components": {"selected": 2}}}, "written by the fit"),
    ({"n_components": "auto", "fit_spec": {"components": {"max": 500}}}, "at most"),
    ({"n_components": "many"}, "number or 'auto'"),
])
def test_requests_that_cannot_be_met_are_refused(tmp_path, kwargs, message):
    project = uneven_project(tmp_path)
    with pytest.raises(ValueError, match=message):
        project.fit("API", **kwargs)
    assert project.models == {}


# -- training diagnostics ------------------------------------------------------------------


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


def test_the_limits_in_use_and_the_roi_shares_are_at_simcas_level_095():
    """DCrit and T²Crit at 0.95 (and 0.99 for display); per ROI the share above the 0.95 pair."""
    X, y, roi_index = _three_sources()
    model = PLSRegression(n_components=2, x_scaling="center").fit(X, y)
    _metrics, diagnostics, projection = _components.training_diagnostics(
        model, X, y, np.ones(X.shape[0], bool), roi_index, y_divisor=float(model.y_std_[0]))
    n, k, a = X.shape[0], X.shape[1], 2
    for level, suffix in ((0.95, "095"), (0.99, "099")):
        t2_crit = a * (n - 1) / (n - a) * f.ppf(level, a, n - a)
        assert diagnostics[f"t2_crit_{suffix}"] == pytest.approx(t2_crit, rel=1e-12)
        assert diagnostics[f"dcrit_{suffix}"] == pytest.approx(dmodx_limit(n, k, a, level))
        assert projection[f"dcrit_{suffix}"] == diagnostics[f"dcrit_{suffix}"]
        assert projection[f"t2_crit_{suffix}"] == diagnostics[f"t2_crit_{suffix}"]
    for key, values, limit in (("roi_share_above_dcrit", diagnostics["dmodx"], "dcrit_095"),
                               ("roi_share_above_t2crit", diagnostics["t2"], "t2_crit_095")):
        expected = [np.mean(np.asarray(values)[roi_index == i] > diagnostics[limit])
                    for i in range(roi_index.max() + 1)]
        np.testing.assert_allclose(diagnostics[key], expected, rtol=1e-6)
    assert np.mean(diagnostics["roi_share_above_dcrit"]) > np.mean(
        [np.mean(np.asarray(diagnostics["dmodx"])[roi_index == i] > diagnostics["dcrit_099"])
         for i in range(roi_index.max() + 1)]), "0.95 flags more pixels than 0.99"


def test_a_selected_number_that_differs_from_the_model_is_refused_on_reading(tmp_path):
    project = uneven_project(tmp_path)
    model = project.fit("API", n_components=2)
    with h5py.File(project.path, "r+") as f:
        f[f"model_runs/{model.record.model_run_id}/fit_spec/components"].attrs["selected"] = 3
    with pytest.raises(ValueError, match="differs from hyperparameters/n_components"):
        Project.open(project.path)


# -- autofit beyond the 1.3 rules (contract §7.4, plan component-diagnostics-and-autofit) ----


def test_the_lookahead_keeps_a_failing_component_before_a_passing_one():
    def check(case):
        q2, expected = case
        curve = {"rmsecv": [1.0] * len(q2), "rmsecv_se": [0.0] * len(q2), "q2": q2}
        assert choose_components(curve, "q2_lookahead", n_samples=20) == expected

    check_all([
    ([0.5, 0.02, 0.3, 0.01, 0.0], 3),     # 2 fails but 3 passes: both taken; 4 and 5 fail
    ([0.01, 0.4, 0.3], 3),                 # the first fails, the second passes
    ([0.01, 0.02, 0.4], 1),                # two in a row fail: 1
    ([0.4, 0.3, 0.2], 3),                  # all pass
    ([0.4, 0.3, 0.01], 2),                 # the last fails with no successor
    ([0.4, float("nan"), 0.3], 3),         # NaN fails, the next passes
], check)


def test_the_ceiling_comes_before_the_noise_drop():
    """Guard first, ceiling second would end on the noise component 6 (7 predicts)."""
    curve = {"r2x_cum": np.cumsum([0.6, 0.2, 0.1, 0.05, 0.02, 0.003, 0.002, 0.001]),
             "q2": [0.5, 0.3, 0.2, 0.1, 0.1, -0.1, 0.2, -0.1]}
    assert structure_ceiling(curve) == 6
    assert list(noise_components(curve, n_samples=20)) == [False] * 5 + [True, False, True]
    assert guarded_choice(curve, 8, n_samples=20) == 5


def test_a_stored_specification_must_have_every_key():
    from enginir.project import FitSpec

    request = FitSpec.from_dict({"components": {"rule": "one_se", "max": 5}})
    assert request.components_noise_guard == "default", "a request leaves the default"
    stored = FitSpec(components_noise_guard="on", components_rule_choice=4)
    assert FitSpec.from_dict(stored.to_dict(), stored=True) == stored
    incomplete = stored.to_dict()
    del incomplete["components"]["noise_guard"]
    with pytest.raises(ValueError, match="noise_guard"):
        FitSpec.from_dict(incomplete, stored=True)


def test_training_statistics_follow_their_formulas(tmp_path):
    """RMSEE (N - 1 - A), T² with its F-form limit, DModX (dmodx-1), CoeffCS (§5, §7.5)."""
    project = uneven_project(tmp_path)
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


def test_the_guard_can_be_switched_off_and_the_curve_stores_the_new_fields(tmp_path):
    project = uneven_project(tmp_path)
    model = project.fit("API", n_components="auto",
                        fit_spec={"components": {"rule": "min_rmsecv", "noise_guard": "off"}})
    curve = model.component_curve
    spec = model.record.fit_specification
    assert spec.components_noise_guard == "off"
    assert model.record.hyperparameters["n_components"] == curve["suggested_min_rmsecv"] \
        == spec.components_rule_choice
    a = model.record.hyperparameters["n_components"]
    assert curve["rmsee_roi"][a - 1] == pytest.approx(model.metrics["rmsee"], rel=1e-6)
    assert np.array_equal(curve["noise"], noise_components(curve, n_samples=9))
    assert int(curve["structure_ceiling"]) == structure_ceiling(curve)
    roughness = np.asarray(curve["dw_weights"])
    assert roughness.shape == (len(curve["rmsecv"]),) and np.all(roughness >= 0)
    fixed = project.fit("API", n_components=2)
    assert fixed.record.fit_specification.components_noise_guard == "off"
    assert fixed.record.fit_specification.components_rule_choice == 2
    assert "warnings" in fixed.record.cv_spec
