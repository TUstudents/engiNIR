"""Figures of merit, cross-validation, sample selection and outlier limits."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from enginir.models import PCR, SIMCA, PLSRegression
from enginir.validation import (
    aggregate_regression_predictions,
    bias,
    cross_validate,
    kennard_stone,
    metrics_report,
)
from enginir.validation import cross_val
from enginir.validation.metrics import r2_score, rmsep, rpd, rpiq, sep
from enginir.validation.outliers import hotelling_t2, leverage_outliers, spectral_outliers

RNG = np.random.default_rng(7)
N, P = 40, 80
X_DUMMY = RNG.random((N, P))
Y_DUMMY = RNG.random(N)


# --- figures of merit ---------------------------------------------------------------------------

def test_metrics_match_hand_computed_values_and_their_identity():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.0, 3.5, 4.0, 6.0])  # errors 0.5, 0, 0.5, 0, 1; mean 0.4
    assert bias(y_true, y_pred) == pytest.approx(0.4)
    assert sep(y_true, y_pred) == pytest.approx(np.sqrt(np.sum((y_pred - y_true - 0.4) ** 2) / 4))
    assert rpd(y_true, y_pred) == pytest.approx(np.std(y_true, ddof=1) / np.sqrt(0.3))
    assert rpiq(y_true, y_pred) == pytest.approx(2.0 / np.sqrt(0.3))
    assert r2_score(y_true, y_pred) == pytest.approx(1 - 1.5 / 10.0)
    n = y_true.size
    assert rmsep(y_true, y_pred) ** 2 == pytest.approx(0.4**2 + sep(y_true, y_pred) ** 2 * (n - 1) / n)


def test_metrics_handle_degenerate_cases_and_reject_misuse():
    assert np.isnan(r2_score(np.ones(5), np.ones(5)))
    assert rpd(np.arange(4.0), np.arange(4.0)) == float("inf")
    with pytest.raises(ValueError, match="1-D"):  # multi-target metrics are per column
        rmsep(np.ones((3, 2)), np.ones((3, 2)))
    with pytest.raises(ValueError, match="has 2 values but y_pred has 1"):
        rmsep(np.array([1.0, 2.0]), np.array([1.0]))


def test_metrics_report_keys_and_paired_test_arrays():
    y = np.linspace(0, 10, 20)
    y_pred = y + RNG.random(20) * 0.1
    assert set(metrics_report(y, y_pred)) == {"RMSEC", "R2_cal"}
    full = metrics_report(y, y_pred, y_pred_cv=y_pred, y_true_test=y[:10], y_pred_test=y_pred[:10])
    assert set(full) == {"RMSEC", "R2_cal", "RMSECV", "R2_cv", "RMSEP", "SEP", "bias", "R2",
                         "RPD", "RPIQ"}
    with pytest.raises(ValueError, match="must be given together"):
        metrics_report(y, y, y_true_test=y)


# --- cross-validation ---------------------------------------------------------------------------

def test_cross_validate_matches_sklearn_and_leaves_the_estimator_unfitted():
    from sklearn.model_selection import KFold, cross_val_predict

    model = PLSRegression(n_components=3)
    result = cross_validate(model, X_DUMMY, Y_DUMMY, cv=5, random_state=0)
    expected = cross_val_predict(model, X_DUMMY, Y_DUMMY, cv=KFold(5, shuffle=True, random_state=0))
    np.testing.assert_allclose(result["y_pred"], np.ravel(expected))
    assert result["cv_scheme"] == {"type": "KFold", "n_splits": 5}
    assert result["rmsecv"] == pytest.approx(rmsep(Y_DUMMY, result["y_pred"]))
    assert not hasattr(model, "x_scores_")


@pytest.mark.parametrize(
    ("n_groups", "cv", "scheme"),
    [
        (8, "auto", {"type": "GroupKFold", "n_splits": 7}),
        (5, "auto", {"type": "LeaveOneGroupOut", "n_splits": 5}),  # fewer groups than folds
        (5, "logo", {"type": "LeaveOneGroupOut", "n_splits": 5}),
        (5, 3, {"type": "GroupKFold", "n_splits": 3}),
        (3, "loo", {"type": "LeaveOneOut", "n_splits": 12}),  # explicit: ignores groups
    ],
)
def test_cross_validate_chooses_the_splitter(n_groups, cv, scheme):
    groups = np.repeat(np.arange(n_groups), 4)
    X = RNG.random((groups.size, 10))
    result = cross_validate(PLSRegression(n_components=1), X, groups.astype(float), cv=cv,
                            groups=groups)
    assert result["cv_scheme"] == scheme


def test_cross_validate_rejects_impossible_or_ambiguous_requests():
    groups = np.repeat(np.arange(5), 4)
    X, y = RNG.random((20, 10)), RNG.random(20)
    with pytest.raises(ValueError, match="7-fold grouped validation requires at least 7 groups"):
        cross_validate(PLSRegression(n_components=1), X, y, cv=7, groups=groups)
    with pytest.raises(ValueError, match="cv='logo' requires groups"):
        cross_validate(PLSRegression(n_components=1), X, y, cv="logo")
    with pytest.raises(ValueError, match="groups must have shape"):
        cross_validate(PLSRegression(n_components=1), X, y, groups=groups[:, None])
    with pytest.raises(ValueError, match="scoring must be a list"):
        cross_validate(PLSRegression(n_components=1), X, y, scoring="r2")


def test_grouped_cross_validation_never_splits_a_group():
    class RecordingPCR(PCR):
        seen: list = []

        def fit(self, X, y):
            RecordingPCR.seen.append(set(np.round(X[:, 0], 12)))
            return super().fit(X, y)

    groups = np.repeat(np.arange(10), 4)
    X = RNG.random((40, 5))
    cross_validate(RecordingPCR(n_components=2), X, RNG.random(40), groups=groups)
    assert RecordingPCR.seen
    for training_rows in RecordingPCR.seen:
        for group in range(10):
            members = set(np.round(X[groups == group, 0], 12))
            assert members <= training_rows or not members & training_rows


def test_cross_validate_reports_per_target_and_per_fold_errors():
    y = np.column_stack([Y_DUMMY, 100 * Y_DUMMY + RNG.random(N)])
    result = cross_validate(PLSRegression(n_components=2), X_DUMMY, y, cv=4,
                            return_train_score=True)
    assert result["rmsecv"].shape == (2,)
    assert result["rmsecv"][1] == pytest.approx(rmsep(y[:, 1], result["y_pred"][:, 1]))
    assert len(result["fold_results"]) == 4


def test_cross_validate_stratifies_classifiers_and_keeps_their_labels():
    y = np.repeat([0, 1], 20)
    result = cross_validate(SIMCA(n_components=2), RNG.random((40, 10)) + y[:, None], y, cv=4,
                            return_train_score=True)
    assert result["cv_scheme"] == {"type": "StratifiedKFold", "n_splits": 4}
    assert result["y_pred"].dtype.kind == "i"
    assert "rmsecv" not in result and "fold_results" not in result
    with pytest.raises(ValueError, match="needs a class with at least 7 samples"):
        cross_validate(SIMCA(n_components=1), X_DUMMY[:12], np.array(["a", "b"] * 6))


def test_aggregation_averages_per_group_in_order_of_first_appearance():
    groups = np.array(["roi_2", "roi_2", "roi_1", "roi_1"], dtype=object)
    ids, y_true, y_pred = aggregate_regression_predictions(
        np.array([1.0, 1.0, 3.0, 3.0]), np.array([0.0, 2.0, 2.0, 4.0]), groups
    )
    np.testing.assert_array_equal(ids, ["roi_2", "roi_1"])
    np.testing.assert_allclose(y_true, [1.0, 3.0])
    np.testing.assert_allclose(y_pred, [1.0, 3.0])
    with pytest.raises(ValueError, match="aggregation must be one of"):
        aggregate_regression_predictions(y_true, y_pred, ids, aggregation="median")
    with pytest.raises(ValueError, match="same length"):
        aggregate_regression_predictions(np.ones(2), np.ones(3), np.arange(2))


# --- Kennard-Stone ------------------------------------------------------------------------------

def test_kennard_stone_selects_extremes_first_and_partitions_the_samples():
    X = np.array([[0.0], [10.0], [5.0], [1.0], [9.0], [6.0]])
    calibration, test = kennard_stone(X, n_select=3)
    np.testing.assert_array_equal(calibration, [0, 1, 2])
    np.testing.assert_array_equal(test, [3, 4, 5])


def test_kennard_stone_blockwise_search_matches_the_full_distance_matrix(monkeypatch):
    X = RNG.integers(0, 4, (300, 3)).astype(float)  # many tied distances
    distances = cdist(X, X)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    selected = [int(first), int(second)]
    nearest = np.minimum(distances[first], distances[second])
    nearest[selected] = -np.inf
    while len(selected) < 30:
        best = int(np.argmax(nearest))
        selected.append(best)
        nearest = np.minimum(nearest, distances[best])
        nearest[best] = -np.inf

    monkeypatch.setattr(cross_val, "_KS_BLOCK_ELEMENTS", 1000)  # forces ~100 row blocks
    np.testing.assert_array_equal(kennard_stone(X, n_select=30)[0], selected)


def test_kennard_stone_mahalanobis_matches_scipy_and_rejects_degenerate_input():
    X = RNG.random((30, 4))
    VI = np.linalg.inv(np.cov(X, rowvar=False))
    whitened = cross_val._whitened_scores(X)
    np.testing.assert_allclose(cdist(whitened, whitened), cdist(X, X, "mahalanobis", VI=VI),
                               atol=1e-10)
    with pytest.raises(ValueError, match="degenerate"):
        kennard_stone(RNG.random((6, 50)), n_select=3, metric="mahalanobis")


# --- outlier limits -----------------------------------------------------------------------------

def test_leverage_equals_hat_matrix_diagonal_for_rank_deficient_x():
    scores = RNG.standard_normal((30, 3))
    X = scores @ RNG.standard_normal((3, 10))  # rank 3, 10 columns
    design = np.column_stack([np.ones(30), scores])
    leverage, mask = leverage_outliers(X)
    np.testing.assert_allclose(leverage, np.diag(design @ np.linalg.pinv(design)), atol=1e-10)
    assert leverage.sum() == pytest.approx(4.0)
    np.testing.assert_array_equal(mask, leverage > 3 * 4 / 30)


def test_hotelling_limits_have_nominal_false_alarm_rates():
    rng = np.random.default_rng(0)
    n, n_components, repeats = 20, 3, 2000
    calibration_alarms = new_alarms = 0
    for _ in range(repeats):
        calibration = rng.standard_normal((n, n_components))
        calibration_alarms += int(hotelling_t2(calibration)[2].sum())
        new = rng.standard_normal((1, n_components))
        new_alarms += int(hotelling_t2(new, calibration_scores=calibration)[2].sum())
    assert calibration_alarms / (n * repeats) == pytest.approx(0.05, abs=0.006)
    assert new_alarms / repeats == pytest.approx(0.05, abs=0.015)
    with pytest.raises(ValueError, match="calibration samples"):
        hotelling_t2(np.ones((3, 5)))


def test_q_residuals_flag_bad_reconstructions_with_a_scale_invariant_limit():
    X = RNG.standard_normal((50, 20))
    X_reconstructed = X + 0.01 * RNG.standard_normal((50, 20))
    X_reconstructed[0] += 1.0
    q, q_crit, mask = spectral_outliers(X, X_reconstructed)
    assert mask[0]
    np.testing.assert_allclose(q, ((X - X_reconstructed) ** 2).sum(axis=1))
    q_small, q_crit_small, mask_small = spectral_outliers(1e-3 * X, 1e-3 * X_reconstructed)
    assert q_crit_small == pytest.approx(1e-6 * q_crit)
    np.testing.assert_array_equal(mask, mask_small)
    assert not spectral_outliers(X, X)[2].any()  # no spread: limit is max(Q)


def test_q_limit_recovers_the_chi_square_quantile_for_gaussian_residuals():
    from scipy.stats import chi2

    X = np.random.default_rng(5).standard_normal((20000, 20))  # Q is exactly chi2(20)
    q_crit = spectral_outliers(X, np.zeros_like(X), alpha=0.05)[1]
    assert q_crit == pytest.approx(chi2.ppf(0.95, 20), rel=0.02)
