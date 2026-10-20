"""Chemometric and machine-learning estimators: numerics and sklearn contracts."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor

from enginir.models import (
    PCR,
    SIMCA,
    LDAClassifier,
    MLPModel,
    PLSRegression,
    RandomForestModel,
    SVRModel,
)
from enginir.models.classification import jackson_mudholkar_limit

RNG = np.random.default_rng(0)
N, P = 50, 100
X_DUMMY = RNG.random((N, P))
COEF_TRUE = np.r_[RNG.random(10), np.zeros(P - 10)]
Y_REG = X_DUMMY @ COEF_TRUE + 0.05 * RNG.standard_normal(N)

_rng = np.random.default_rng(7)
X_SEP = np.vstack([0.5 * _rng.standard_normal((20, P)) + centre for centre in (0.0, 10.0, 20.0)])
Y_SEP = np.repeat([0, 1, 2], 20)


# --- linear regression --------------------------------------------------------------------------

@pytest.mark.parametrize("model", [PCR(n_components=10, scale=True), PCR(n_components=10, scale=False),
                                   PLSRegression(n_components=5, scale=True),
                                   PLSRegression(n_components=5, scale=False)])
def test_linear_models_fold_scaling_and_centring_into_raw_space_coefficients(model):
    """``X @ coef_ + intercept_`` must equal the model evaluated step by step in score space."""

    def stepwise(model, X):
        if isinstance(model, PCR):
            scaled = X if model.scaler_ is None else model.scaler_.transform(X)
            return model.reg_.predict(model.pca_.transform(scaled))
        y_scaled = model.transform(X) @ model.y_loadings_.T  # PLS: T Q^T in scaled Y units
        return np.squeeze(y_scaled * model.y_std_ + model.y_mean_)

    for y in (Y_REG, np.column_stack([Y_REG, RNG.random(N)])):
        model.fit(X_DUMMY, y)
        np.testing.assert_allclose(X_DUMMY @ model.coef_ + model.intercept_,
                                   stepwise(model, X_DUMMY), rtol=1e-8, atol=1e-10)
    assert model.coef_.shape == (P, 2)


def test_pls_matches_scikit_learn_and_keeps_2d_targets_2d():
    from sklearn.cross_decomposition import PLSRegression as SklearnPLS

    model = PLSRegression(n_components=4).fit(X_DUMMY, Y_REG)
    reference = SklearnPLS(n_components=4).fit(X_DUMMY, Y_REG)
    np.testing.assert_allclose(model.predict(X_DUMMY), reference.predict(X_DUMMY), rtol=1e-10)
    np.testing.assert_allclose(model.transform(X_DUMMY), reference.transform(X_DUMMY), rtol=1e-10)
    column_model = PLSRegression(n_components=4).fit(X_DUMMY, Y_REG[:, np.newaxis])
    assert column_model.coef_.shape == (P, 1)
    assert column_model.predict(X_DUMMY).shape == (N, 1)


def test_vip_matches_textbook_definition():
    """VIP weights each component by the Y variance it explains (Chong & Jun 2005)."""
    model = PLSRegression(n_components=4).fit(X_DUMMY, Y_REG)
    X = (X_DUMMY - X_DUMMY.mean(0)) / X_DUMMY.std(0, ddof=1)
    f = (Y_REG - Y_REG.mean()) / Y_REG.std(ddof=1)
    weights, explained = [], []
    for _ in range(4):  # NIPALS PLS1
        w = X.T @ f
        w /= np.linalg.norm(w)
        t = X @ w
        q = f @ t / (t @ t)
        X = X - np.outer(t, X.T @ t / (t @ t))
        f = f - q * t
        weights.append(w)
        explained.append(q**2 * (t @ t))
    weights, explained = np.array(weights).T, np.array(explained)
    expected = np.sqrt(P * (weights**2 @ explained) / explained.sum())
    np.testing.assert_allclose(model.vip_scores(), expected, rtol=1e-8)
    assert np.mean(model.vip_scores() ** 2) == pytest.approx(1.0)


def test_component_count_is_limited_by_the_rank_of_centred_x():
    X, y = RNG.random((5, 20)), RNG.random(5)
    with pytest.raises(ValueError, match="at most 4"):
        PLSRegression(n_components=5).fit(X, y)
    with pytest.raises(ValueError, match="at most 4"):
        PCR(n_components=5).fit(X, y)
    assert np.all(np.isfinite(PLSRegression(n_components=4).fit(X, y).vip_scores()))
    assert PCR(n_components=None).fit(X, y).pca_.n_components_ == 4
    with pytest.raises(ValueError, match="at least 2 samples"):
        PLSRegression(n_components=1).fit(X[:1], y[:1])


@pytest.mark.parametrize("factory", [PLSRegression, PCR])
def test_linear_models_predict_in_float32_and_reject_a_different_band_count(factory):
    X, y = X_DUMMY.astype(np.float32), Y_REG.astype(np.float32)
    model = factory(n_components=5).fit(X, y)
    assert model.predict(X).dtype == np.float32
    assert model.transform(X).dtype == np.float32
    with pytest.raises(ValueError, match=f"{factory.__name__} was fitted with {P} features"):
        model.predict(X[:, :-2])


# --- SIMCA --------------------------------------------------------------------------------------

def test_simca_limit_uses_exact_discarded_eigenvalues():
    X = RNG.standard_normal((20, P))
    model = SIMCA(n_components=3, scale=False).fit(X, np.zeros(20, dtype=int))
    singular_values = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    eigenvalues = singular_values[:19] ** 2 / 19  # rank of 20 centred samples is 19
    assert model.boundaries_[0] == pytest.approx(jackson_mudholkar_limit(eigenvalues[3:], 0.05))


def test_simca_accepts_about_95_percent_of_new_class_members():
    rng = np.random.default_rng(3)
    loadings = rng.standard_normal((3, P))

    def draw(m):
        return 3 * rng.standard_normal((m, 3)) @ loadings + 0.3 * rng.standard_normal((m, P))

    model = SIMCA(n_components=3, scale=False).fit(draw(200), np.zeros(200, dtype=int))
    assert 0.92 <= np.mean(model.predict(draw(2000)) == 0) <= 0.98


@pytest.mark.parametrize("scale", [True, False])
def test_simca_q_residuals_are_the_squared_distance_to_each_class_subspace(scale):
    model = SIMCA(n_components=3, scale=scale).fit(X_SEP, Y_SEP)
    q = model.q_residuals(X_SEP[:5])
    for column, label in enumerate(model.classes_):
        scaled = model.scalers_[label].transform(X_SEP[:5]) if scale else X_SEP[:5]
        pca = model.pca_models_[label]
        expected = ((scaled - pca.inverse_transform(pca.transform(scaled))) ** 2).sum(axis=1)
        np.testing.assert_allclose(q[:, column], expected, rtol=1e-10)
    np.testing.assert_array_equal(model.predict(X_SEP), Y_SEP)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_simca_reduces_components_to_the_class_rank_and_accepts_its_training_data(dtype):
    X = np.vstack([RNG.random((30, 200)), RNG.random((30, 200)) + 2.0]).astype(dtype)
    y = np.repeat([0, 1], 30)
    model = SIMCA(n_components=40).fit(X, y)
    assert model.n_components_ == {0: 28, 1: 28}  # 30 centred samples: rank 29, one left over
    assert np.mean(model.predict(X) == y) > 0.9


@pytest.mark.parametrize(
    ("labels", "rejected", "dtype_kind"),
    [
        (np.array([1, 2]), -1, "i"),
        (np.array([1, 2], dtype=np.uint64), -1, "i"),  # -1 needs a signed dtype
        (np.array(["a", "b"]), "unknown", "U"),
        (np.array(["a", "b"], dtype=object), "unknown", "U"),  # e.g. a pandas Series
        (np.array([0.5, 1.5]), np.nan, "f"),
        (np.array([True, False]), None, "O"),
    ],
)
def test_simca_rejection_label_matches_the_label_dtype(labels, rejected, dtype_kind):
    X = np.vstack([RNG.standard_normal((15, 10)), RNG.standard_normal((15, 10)) + 5])
    prediction = SIMCA(n_components=2).fit(X, np.repeat(labels, 15)).predict(np.full((1, 10), 1e3))
    assert prediction.dtype.kind == dtype_kind
    if rejected is None:
        assert prediction[0] is None
    elif isinstance(rejected, float):
        assert np.isnan(prediction[0])
    else:
        assert prediction[0] == rejected


def test_simca_rejects_unusable_class_definitions():
    with pytest.raises(ValueError, match="at least 3"):
        SIMCA(n_components=2).fit(RNG.random((4, P)), np.array([0, 0, 0, 1]))
    with pytest.raises(ValueError, match="no entry for class"):
        SIMCA(n_components={0: 2}).fit(X_SEP, Y_SEP)


# --- LDA and the machine-learning wrappers ------------------------------------------------------

def test_lda_classifier_separates_classes_and_transform_needs_a_projecting_solver():
    lda = LDAClassifier(n_pca_components=5).fit(X_SEP, Y_SEP)
    assert lda.transform(X_SEP).shape == (X_SEP.shape[0], 2)  # n_classes - 1 axes
    assert np.mean(lda.predict(X_SEP) == Y_SEP) >= 0.95
    with pytest.raises(NotImplementedError, match="lsqr"):
        LDAClassifier(solver="lsqr").fit(X_SEP, Y_SEP).transform(X_SEP)


@pytest.mark.parametrize(
    "model",
    [SVRModel(), RandomForestModel(n_estimators=10), MLPModel(hidden_layer_sizes=(16,), max_iter=50)],
    ids=["SVR", "RandomForest", "MLP"],
)
def test_regression_wrappers_are_sklearn_regressors(model):
    assert is_regressor(model) and not hasattr(model, "predict_proba")
    y_pred = model.fit(X_DUMMY, Y_REG).predict(X_DUMMY)
    assert y_pred.shape == (N,) and np.isfinite(y_pred).all()


@pytest.mark.parametrize(
    "model",
    [RandomForestModel(n_estimators=20, task="classification"),
     MLPModel(hidden_layer_sizes=(16,), max_iter=300, task="classification")],
    ids=["RandomForest", "MLP"],
)
def test_classification_wrappers_follow_the_classifier_contract(model):
    model.fit(X_SEP, Y_SEP)
    assert is_classifier(model) and not is_regressor(model)
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    assert model.predict_proba(X_SEP).shape == (X_SEP.shape[0], 3)
    assert model.score(X_SEP, Y_SEP) == pytest.approx(np.mean(model.predict(X_SEP) == Y_SEP))
    assert model.score(X_SEP, Y_SEP) >= 0.9
    assert clone(model).get_params() == model.get_params()


def test_random_forest_importances_cover_every_wavelength():
    model = RandomForestModel(n_estimators=10).fit(X_DUMMY, Y_REG)
    assert model.feature_importances_.shape == (P,)
    assert model.feature_importances_.sum() == pytest.approx(1.0)


def test_pickles_record_the_version_and_refuse_pre_1_0_state():
    """A pre-1.0 PLS pickle predicted wrong values silently; it is refused instead."""
    import pickle

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(30, 8)), rng.normal(size=30)
    model = PLSRegression(n_components=2).fit(X, y)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))

    old_state = model.__getstate__()
    del old_state["_enginir_version"]
    with pytest.raises(ValueError, match="saved by engiNIR before 1.0"):
        PLSRegression.__new__(PLSRegression).__setstate__(old_state)
