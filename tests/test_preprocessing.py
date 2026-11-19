"""Preprocessing: the transformers, the persisted recipe, AsLS baselines and Savitzky-Golay edges."""

from __future__ import annotations

import tracemalloc
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from enginir import PreprocessingRecipe
from enginir._workflow import recipe_id_for
from enginir.preprocessing import (
    AsLSBaseline,
    Detrend,
    EMSC,
    MSC,
    SNV,
    SavitzkyGolay,
    Smoother,
)
from tests._scans import N_BANDS, labelled_project, predicts_its_training_scan


# -- Behaviour of the spectral preprocessing transformers ------------------------------


RNG = np.random.default_rng(42)


N, P = 20, 100


X_DUMMY = RNG.random((N, P))


def _reference_emsc(X, *, order, reference=None, interferences=None, weights=None):
    """Textbook EMSC by ``lstsq``: corrected spectra and coefficients."""
    reference = X.mean(axis=0) if reference is None else reference
    A = np.column_stack([np.vander(np.linspace(-1.0, 1.0, X.shape[1]), order + 1, increasing=True),
                         reference])
    if interferences is not None:
        A = np.column_stack([A, interferences.T])
    w = np.ones(X.shape[1]) if weights is None else weights
    coeffs = np.linalg.lstsq(A * w[:, None], (X * w).T, rcond=None)[0]
    additive = coeffs.copy()
    additive[order + 1] = 0.0
    return (X - (A @ additive).T) / coeffs[order + 1][:, None], coeffs.T


def test_snv_uses_the_sample_std_and_zeroes_constant_spectra():
    np.testing.assert_allclose(SNV().transform(np.array([[1.0, 2.0, 3.0]])), [[-1.0, 0.0, 1.0]])

    X = np.full((2, 220), 0.123, dtype=np.float32)  # float32 constant: std is rounding noise
    X[1] = np.linspace(0.0, 1.0, 220)
    out = SNV().transform(X)
    np.testing.assert_array_equal(out[0], 0.0)
    assert out[1].mean() == pytest.approx(0.0, abs=1e-6)
    assert out[1].std(ddof=1) == pytest.approx(1.0, rel=1e-5)


def test_msc_maps_its_reference_onto_itself_and_rejects_a_constant_one():
    msc = MSC().fit(X_DUMMY)
    np.testing.assert_allclose(msc.reference_, X_DUMMY.mean(axis=0))
    # a spectrum a + b * r is mapped back onto r: both scatter terms are removed
    shifted = 0.4 + 2.5 * msc.reference_
    np.testing.assert_allclose(msc.transform(shifted[None]), msc.reference_[None], rtol=1e-10)
    with pytest.raises(ValueError, match="constant"):
        MSC(reference=np.ones(P)).fit(X_DUMMY)


@pytest.mark.parametrize("transformer", [MSC(), EMSC()], ids=["MSC", "EMSC"])
def test_fitted_scatter_correction_rejects_a_different_band_count(transformer):
    transformer.fit(X_DUMMY)
    with pytest.raises(ValueError, match="features"):
        transformer.transform(RNG.random((5, P - 2)))


def test_emsc_matches_the_reference_algorithm_with_interferences_and_weights():
    interferences = RNG.random((2, P))
    weights = np.linspace(0.25, 1.0, P)
    emsc = EMSC(order=1, interferences=interferences, weights=weights).fit(X_DUMMY)
    expected, expected_coeffs = _reference_emsc(
        X_DUMMY, order=1, interferences=interferences, weights=weights
    )
    np.testing.assert_allclose(emsc.transform(X_DUMMY), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(emsc.compute_coefficients(X_DUMMY), expected_coeffs,
                               rtol=1e-12, atol=1e-12)


def test_emsc_reference_can_be_the_median_or_supplied():
    np.testing.assert_allclose(EMSC(method="median", order=0).fit(X_DUMMY).reference_,
                               np.median(X_DUMMY, axis=0))
    reference = X_DUMMY[0].copy()
    expected, _ = _reference_emsc(X_DUMMY, order=2, reference=reference)
    np.testing.assert_allclose(EMSC(reference=reference).fit(X_DUMMY).transform(X_DUMMY),
                               expected, rtol=1e-12, atol=1e-12)


def test_emsc_rejects_invalid_weights_and_rank_deficient_designs():
    with pytest.raises(ValueError, match="non-negative"):
        EMSC(weights=np.r_[-1.0, np.ones(P - 1)]).fit(X_DUMMY)
    with pytest.raises(ValueError, match="positive"):
        EMSC(weights=np.zeros(P)).fit(X_DUMMY)
    with pytest.raises(ValueError, match="interferences have"):
        EMSC(interferences=RNG.random((2, P - 1))).fit(X_DUMMY)
    with pytest.raises(ValueError, match="rank"):  # reference is a degree-1 polynomial
        EMSC(reference=np.linspace(0.0, 1.0, P)).fit(X_DUMMY)
    with pytest.raises(ValueError, match="rank"):  # interference collinear with the offset
        EMSC(interferences=np.ones((1, P))).fit(X_DUMMY)


def test_emsc_keeps_output_finite_for_a_zero_multiplicative_term():
    assert np.isfinite(EMSC(order=0).fit(X_DUMMY).transform(np.zeros((3, P)))).all()


def test_emsc_transform_replays_the_fitted_order_not_the_current_parameter():
    emsc = EMSC(order=2).fit(X_DUMMY)
    expected = emsc.transform(X_DUMMY)
    emsc.set_params(order=1)
    np.testing.assert_array_equal(emsc.transform(X_DUMMY), expected)


def test_savitzky_golay_differentiates_a_polynomial_exactly():
    x = np.arange(30.0)
    X = np.tile(0.5 * x**3, (2, 1))
    np.testing.assert_allclose(SavitzkyGolay(7, 3, deriv=1).transform(X), np.tile(1.5 * x**2, (2, 1)),
                               atol=1e-8)
    np.testing.assert_allclose(SavitzkyGolay(7, 3, deriv=2).transform(X), np.tile(3.0 * x, (2, 1)),
                               atol=1e-8)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"window_length": 4}, "odd"),
        ({"window_length": 5, "polyorder": 5}, "polyorder"),
        ({"window_length": 11, "polyorder": 2, "deriv": 3}, "deriv"),
        ({"window_length": 101}, "exceeds"),
    ],
)
def test_savitzky_golay_rejects_inconsistent_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SavitzkyGolay(**kwargs).transform(X_DUMMY)


def test_detrend_removes_polynomial_baselines_on_a_raw_wavelength_axis():
    wavelengths = np.linspace(991.0, 1677.0, 220)
    baseline = 0.3 + 2e-4 * wavelengths - 1e-7 * wavelengths**2 + 1e-11 * wavelengths**3
    np.testing.assert_allclose(Detrend(3, wavelengths).transform(np.tile(baseline, (4, 1))), 0.0,
                               atol=1e-9)
    # float32 stays accurate because the abscissa is rescaled to [-1, 1]
    X = (baseline + 0.01 * np.sin(wavelengths / 20.0))[None]
    single = Detrend(3, wavelengths).transform(X.astype(np.float32))
    np.testing.assert_allclose(single, Detrend(3, wavelengths).transform(X), atol=1e-7)
    with pytest.raises(ValueError, match="order=30"):
        Detrend(order=30).transform(RNG.random((3, 30)))


def test_moving_average_repeats_edge_values():
    out = Smoother(window_length=5).transform(np.array([[0.0, 5.0, 10.0, 15.0, 20.0]]))
    np.testing.assert_allclose(out, [[3.0, 6.0, 10.0, 14.0, 17.0]])  # edges padded 0,0 / 20,20
    with pytest.raises(ValueError, match="odd"):
        Smoother(window_length=4).transform(X_DUMMY)


def test_whittaker_matches_the_sparse_solution_and_stays_accurate_in_float32():
    X = RNG.standard_normal((4, 32))
    lam, order = 1e3, 2
    D = sp.eye(32, format="csr")
    for _ in range(order):
        D = D[1:] - D[:-1]  # type: ignore[reportIndexIssue]
    A = (sp.eye(32) + lam * D.T @ D).tocsc()
    expected = np.vstack([spla.spsolve(A, row) for row in X])
    np.testing.assert_allclose(Smoother(method="whittaker", lam=lam).transform(X), expected)

    X = np.random.default_rng(1).random((20, 200)) + np.linspace(0.0, 1.0, 200)
    for lam in (1e4, 1e8):  # condition number ~16 lam: solved in float64 blocks
        smoother = Smoother(method="whittaker", lam=lam)
        np.testing.assert_allclose(smoother.transform(X.astype(np.float32)), smoother.transform(X),
                                   atol=1e-5)


@pytest.mark.parametrize(
    "transformer",
    [SNV(), MSC(), EMSC(), Detrend(order=2, wavelengths=np.linspace(1000, 1700, 220)),
     Smoother(method="whittaker"), Smoother(), SavitzkyGolay()],
    ids=lambda t: f"{type(t).__name__}-{getattr(t, 'method', '')}",
)
def test_transformers_work_in_float32_without_full_size_upcasts(transformer):
    """Working arrays stay float32 (prediction-runtime contract, memory rules)."""
    X = np.random.default_rng(0).random((5000, 220)).astype(np.float32)
    transformer.fit(X)
    tracemalloc.start()
    out = transformer.transform(X)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert out.dtype == np.float32
    assert peak < 2.0 * X.nbytes  # a float64 copy of X alone would be 2x


# -- PreprocessingRecipe: persisted schema, validation and step construction -----------


CONTRACT_FIELDS = {"filter", "sg_window", "sg_polyorder", "snv", "msc", "emsc", "detrend",
                   "detrend_order", "sg_edges", "asls", "asls_lambda", "asls_p",
                   "asls_iterations", "bands_keep", "bands_exclude"}


def test_persisted_form_has_every_contract_field_and_round_trips():
    # enginir-1.5: every field is written, also when it has no effect.
    for recipe in (PreprocessingRecipe(filter="sg_smooth", sg_window=9, sg_polyorder=3,
                                       msc=True, sg_edges="interp"),
                   PreprocessingRecipe(),
                   PreprocessingRecipe(asls=True, bands_keep=[(1100, 1300)])):
        assert set(recipe.to_dict()) == CONTRACT_FIELDS
        assert PreprocessingRecipe.from_dict(recipe.to_dict()) == recipe


def test_from_dict_requires_every_field_and_discards_extras():
    payload = PreprocessingRecipe(snv=True).to_dict()
    payload["created_at"] = "2026-04-02T10:03:00+00:00"
    assert PreprocessingRecipe.from_dict(payload) == PreprocessingRecipe(snv=True)

    del payload["msc"]
    with pytest.raises(ValueError, match=r"/recipes/recipe_1.*msc"):
        PreprocessingRecipe.from_dict(payload, location="/recipes/recipe_1")


def test_from_dict_normalises_hdf5_scalars_but_rejects_wrong_types():
    payload = PreprocessingRecipe().to_dict()
    payload.update(sg_window=np.int64(15), snv=np.bool_(True))
    recipe = PreprocessingRecipe.from_dict(payload)
    assert type(recipe.sg_window) is int and recipe.snv is True

    for field_name, value in (("snv", "false"), ("sg_window", 11.0)):
        bad = PreprocessingRecipe().to_dict() | {field_name: value}
        with pytest.raises(ValueError, match=field_name):
            PreprocessingRecipe.from_dict(bad)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"filter": "median"}, "filter"),
        ({"sg_window": 4}, "sg_window must be odd"),
        ({"sg_window": 5, "sg_polyorder": 5}, "smaller than sg_window"),
        ({"filter": "sg_d2", "sg_polyorder": 1}, "sg_polyorder >= 2"),
    ],
)
def test_rejects_inconsistent_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PreprocessingRecipe(**kwargs)


def test_steps_follow_the_contract_order():
    recipe = PreprocessingRecipe(filter="sg_d1", snv=True, msc=True, emsc=True, detrend=True,
                                 detrend_order=2)
    steps = recipe.build_steps(np.linspace(1000.0, 1700.0, 50))
    assert [name for name, _ in steps] == ["sg", "snv", "msc", "emsc", "detrend"]
    assert steps[0][1].deriv == 1 and steps[-1][1].order == 2
    assert recipe.requires_fitting
    assert recipe.describe() == "SG d1 (w=11, p=2) > SNV > MSC > EMSC > Detrend (order=2)"


def test_window_longer_than_the_spectrum_is_rejected_not_shrunk():
    with pytest.raises(ValueError, match="exceeds the 8 available wavelengths"):
        PreprocessingRecipe(filter="sg_smooth", sg_window=11).build_steps(np.arange(8.0))


# -- SIMCA preprocessing: AsLS baseline correction and Savitzky-Golay edge dropping ----


#: A recipe as stored before enginir-1.5: only the first eight fields.
LEGACY = {"filter": "sg_d1", "sg_window": 11, "sg_polyorder": 2, "snv": True, "msc": False,
          "emsc": False, "detrend": False, "detrend_order": 1}


#: The same recipe in the enginir-1.5 form: every field.
FULL = PreprocessingRecipe(filter="sg_d1", snv=True, sg_edges="interp").to_dict()


def _dense_asls(z, lam, p, n_iter, direction="up"):
    """Eilers & Boelens AsLS with a dense solver (the reference)."""
    n = z.size
    D = np.diff(np.eye(n), n=2, axis=0)
    w = np.ones(n)
    sign = 1.0 if direction == "up" else -1.0
    for _ in range(n_iter):
        b = np.linalg.solve(np.diag(w) + lam * D.T @ D, w * z)
        w = np.where(sign * (z - b) > 0, p, 1 - p)
    return z - b


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("n_bands", [3, 5, 9, 60])
def test_asls_equals_a_dense_solver(direction, n_bands):
    X = np.random.default_rng(n_bands).normal(size=(4, n_bands)).cumsum(axis=1)
    out = AsLSBaseline(lam=50.0, p=0.02, n_iter=6, direction=direction).fit_transform(X)
    expected = np.stack([_dense_asls(x, 50.0, 0.02, 6, direction) for x in X])
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_down_is_asls_of_the_negated_spectrum():
    X = np.random.default_rng(1).normal(size=(3, 40)).cumsum(axis=1)
    down = AsLSBaseline(lam=1e3, direction="down").fit_transform(X)
    np.testing.assert_allclose(down, -AsLSBaseline(lam=1e3).fit_transform(-X), atol=1e-10)


def _curved_spectra(sign):
    x = np.linspace(0.0, 1.0, 150)
    baseline = 0.8 + 0.6 * x - 0.5 * x ** 2
    peaks = np.exp(-((x - 0.3) / 0.02) ** 2) + 0.5 * np.exp(-((x - 0.7) / 0.03) ** 2)
    return (baseline + sign * peaks)[None, :], sign * peaks


@pytest.mark.parametrize(("sign", "direction"), [(1.0, "up"), (-1.0, "down")])
def test_asls_removes_a_curved_baseline_and_keeps_the_peaks(sign, direction):
    spectra, peaks = _curved_spectra(sign)
    corrected = AsLSBaseline(lam=1e3, p=0.001, direction=direction).fit_transform(spectra)[0]
    assert np.abs(corrected - peaks).max() < 0.05  # peaks of height 0.5 to 1
    wrong = AsLSBaseline(lam=1e3, p=0.001,
                         direction="down" if direction == "up" else "up").fit_transform(spectra)
    assert np.abs(wrong[0] - peaks).max() > 0.3, "the wrong direction fits the peaks instead"


def test_asls_removes_constant_and_linear_baselines_and_keeps_float32():
    x = np.linspace(0, 1, 80)
    X = np.stack([np.full(80, 3.0), 2.0 - 5.0 * x]).astype(np.float32)
    out = AsLSBaseline().fit_transform(X)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 0.0, atol=1e-4)


def test_a_recipe_without_the_newer_fields_is_refused():
    # No backward compatibility (enginir-1.5): a recipe missing a field is refused.
    with pytest.raises(ValueError, match="missing required field"):
        PreprocessingRecipe.from_dict(LEGACY)
    recipe = PreprocessingRecipe.from_dict(FULL)
    assert recipe.sg_edges == "interp" and not recipe.asls
    assert recipe_id_for(FULL) == recipe_id_for(recipe)
    none = {**FULL, "filter": "none"}
    assert recipe_id_for(none) == recipe_id_for(PreprocessingRecipe(snv=True))
    assert recipe_id_for(PreprocessingRecipe(filter="sg_d1", snv=True)) != recipe_id_for(FULL)


def test_new_fields_are_normalized_and_validated():
    asls = PreprocessingRecipe(asls=True, asls_lambda=10000, asls_iterations=np.int64(10))
    assert recipe_id_for(asls) == recipe_id_for(PreprocessingRecipe(asls=True,
                                                                     asls_lambda=1e4))
    assert PreprocessingRecipe.from_dict(asls.to_dict()) == asls
    assert PreprocessingRecipe(asls_lambda=5.0) == PreprocessingRecipe()  # no effect: normal form
    payload = asls.to_dict()
    del payload["asls_p"]
    with pytest.raises(ValueError, match="asls_p"):
        PreprocessingRecipe.from_dict(payload)
    with pytest.raises(ValueError, match="sg_edges"):
        PreprocessingRecipe.from_dict(FULL | {"sg_edges": "mirror"})
    with pytest.raises(ValueError, match="asls_p"):
        PreprocessingRecipe(asls=True, asls_p=0.7)
    with pytest.raises(ValueError, match="signal unit"):
        asls.build_steps(np.arange(20.0))


def test_the_order_and_the_direction_follow_the_contract():
    recipe = PreprocessingRecipe(asls=True, filter="sg_smooth", snv=True, msc=True, emsc=True,
                                 detrend=True)
    steps = recipe.build_steps(np.arange(30.0), signal_unit="Absorbance")
    assert [name for name, _ in steps] == ["asls", "sg", "snv", "msc", "emsc", "detrend"]
    assert dict(steps)["asls"].direction == "up"
    for unit in ("Reflectance", "DN", "reflectance"):
        assert dict(recipe.build_steps(np.arange(30.0), signal_unit=unit))["asls"].direction \
            == "down"
    assert dict(steps)["detrend"].wavelengths.size == 30 - 2 * 5  # the output axis


def test_resaving_a_stored_recipe_is_a_no_op_and_a_change_is_refused(tmp_path):
    from enginir.io.project_store import create_project, save_preprocessing_recipe

    path = str(tmp_path / "p.h5")
    create_project(path)
    save_preprocessing_recipe(path, "r", FULL)
    save_preprocessing_recipe(path, "r", FULL)  # same content
    with pytest.raises(ValueError, match="different content"):
        save_preprocessing_recipe(path, "r", FULL | {"sg_edges": "drop"})


def test_dropping_the_edges_keeps_the_interior():
    X = np.random.default_rng(3).normal(size=(4, 40)).cumsum(axis=1).astype(np.float32)
    kept = SavitzkyGolay(window_length=15, polyorder=2, deriv=1).fit_transform(X)
    dropped = SavitzkyGolay(window_length=15, polyorder=2, deriv=1,
                            edges="drop").fit_transform(X)
    assert dropped.shape == (4, 40 - 14) and dropped.dtype == np.float32
    np.testing.assert_array_equal(dropped, kept[:, 7:33])
    recipe = PreprocessingRecipe(filter="sg_d1", sg_window=15)
    assert recipe.output_wavelengths(np.arange(40.0)).tolist() == list(range(7, 33))


@pytest.mark.parametrize("unit", ["Absorbance", "Reflectance"])
def test_a_model_on_the_trimmed_axis_predicts_like_it_was_trained(tmp_path, unit):
    project, scan = labelled_project(tmp_path, unit=unit)
    recipe = PreprocessingRecipe(asls=True, asls_lambda=100.0, filter="sg_smooth",
                                 sg_window=5, detrend=True)
    model = project.fit("API", recipe=recipe, n_components=2)
    assert model.name.startswith("PLS 2 LV, AsLS + SG")
    blob_bands = N_BANDS - 4
    from enginir.io import load_model_run_prediction_artifact

    blob = load_model_run_prediction_artifact(project.path, model.record.model_run_id)
    assert np.shape(blob["coef_"])[0] == np.size(blob["wavelengths"]) == blob_bands
    assert np.size(blob["x_mean"]) == blob_bands
    assert np.size(model.diagnostics["wavelengths"]) == blob_bands
    prediction = predicts_its_training_scan(model, scan)
    assert np.isfinite(prediction.map).all()


def test_a_recipe_mapping_must_be_complete_and_keeps_its_edges(tmp_path):
    project, scan = labelled_project(tmp_path)
    with pytest.raises(ValueError, match="missing required field"):
        project.fit("API", recipe={**LEGACY, "sg_window": 5}, n_components=2)
    model = project.fit("API", recipe={**FULL, "sg_window": 5}, n_components=2)
    assert np.size(model.diagnostics["wavelengths"]) == N_BANDS
    assert "(edges kept)" in model.name
    predicts_its_training_scan(model, scan)


def test_the_fit_dialog_keeps_asls_and_the_edge_handling_of_a_preset_recipe():
    from enginir.viewer._app import ensure_app
    from enginir.viewer._modeling import FitModelRunDialog

    ensure_app()
    preset = PreprocessingRecipe(asls=True, asls_lambda=500.0, filter="sg_d1",
                                 sg_edges="interp", snv=True)
    dialog = FitModelRunDialog([("t", "target")], default_target_id="t", recipe=preset,
                               default_signal_unit="Absorbance",
                               readiness=lambda _t: ("ready", True))
    assert dialog.recipe == preset
    assert "AsLS" in dialog._asls_label.text()
    dialog._snv.setChecked(False)
    assert dialog.recipe == PreprocessingRecipe(asls=True, asls_lambda=500.0, filter="sg_d1",
                                                sg_edges="interp")
    dialog.set_recipe(PreprocessingRecipe(filter="sg_d1"))
    assert dialog.recipe.sg_edges == "drop" and not dialog.recipe.asls
    dialog.close()


def test_the_explore_preview_shows_dropped_bands_as_missing():
    from enginir.viewer._app import ensure_app
    from enginir.viewer._state import ViewerState

    ensure_app()
    cube = np.random.default_rng(0).integers(500, 3000, size=(4, 3, 20), dtype=np.uint16)
    state = ViewerState(cube, np.linspace(1000, 1700, 20))
    state.preprocess_filter, state.sg_window = "sg_smooth", 7
    state._image_cache = None
    assert state.recipe.sg_trim == 3
    image = state.current_frame_image()
    assert image.shape == (3, 20)
    assert np.isnan(image[:, :3]).all() and np.isnan(image[:, -3:]).all()
    assert np.isfinite(image[:, 3:-3]).all()
