"""SIMCA preprocessing: AsLS baseline correction and Savitzky-Golay edge dropping
(project-schema contract §4; plan simca-preprocessing)."""

from __future__ import annotations


import numpy as np
import pytest

from enginir import PreprocessingRecipe
from enginir._workflow import recipe_id_for
from enginir.preprocessing import AsLSBaseline, SavitzkyGolay
from tests.test_workflow import N_BANDS, _labelled_project

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


# -- AsLS ----------------------------------------------------------------------------------


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


# -- the recipe ----------------------------------------------------------------------------


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


# -- Savitzky-Golay edges --------------------------------------------------------------------


def test_dropping_the_edges_keeps_the_interior():
    X = np.random.default_rng(3).normal(size=(4, 40)).cumsum(axis=1).astype(np.float32)
    kept = SavitzkyGolay(window_length=15, polyorder=2, deriv=1).fit_transform(X)
    dropped = SavitzkyGolay(window_length=15, polyorder=2, deriv=1,
                            edges="drop").fit_transform(X)
    assert dropped.shape == (4, 40 - 14) and dropped.dtype == np.float32
    np.testing.assert_array_equal(dropped, kept[:, 7:33])
    recipe = PreprocessingRecipe(filter="sg_d1", sg_window=15)
    assert recipe.output_wavelengths(np.arange(40.0)).tolist() == list(range(7, 33))


def _training_scan_identity(model, scan):
    """Prediction of the training scan replays the training preprocessing exactly."""
    prediction = model.predict(scan.path)
    np.testing.assert_allclose(prediction.t2_map.reshape(-1), model.diagnostics["t2"],
                               rtol=1e-3)
    return prediction


@pytest.mark.parametrize("unit", ["Absorbance", "Reflectance"])
def test_a_model_on_the_trimmed_axis_predicts_like_it_was_trained(tmp_path, unit):
    project, scan = _labelled_project(tmp_path, unit=unit)
    recipe = PreprocessingRecipe(asls=True, asls_lambda=100.0, filter="sg_smooth",
                                 sg_window=5, detrend=True)
    model = project.fit("API", recipe=recipe, n_components=2)
    assert model.name.startswith("PLS 2 LV, AsLS + SG")
    blob_bands = N_BANDS - 4
    from enginir.io import load_model_run_prediction_artifact

    blob = load_model_run_prediction_artifact(project.path, model.ref.model_run_id)
    assert np.shape(blob["coef_"])[0] == np.size(blob["wavelengths"]) == blob_bands
    assert np.size(blob["x_mean"]) == blob_bands
    assert np.size(model.diagnostics["wavelengths"]) == blob_bands
    prediction = _training_scan_identity(model, scan)
    assert np.isfinite(prediction.map).all()


def test_a_recipe_mapping_must_be_complete_and_keeps_its_edges(tmp_path):
    project, scan = _labelled_project(tmp_path)
    with pytest.raises(ValueError, match="missing required field"):
        project.fit("API", recipe={**LEGACY, "sg_window": 5}, n_components=2)
    model = project.fit("API", recipe={**FULL, "sg_window": 5}, n_components=2)
    assert np.size(model.diagnostics["wavelengths"]) == N_BANDS
    assert "(edges kept)" in model.name
    _training_scan_identity(model, scan)


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
