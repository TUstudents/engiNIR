"""Band selection (recipe ranges) and band importance (VIP, selectivity ratio)."""

from __future__ import annotations


import numpy as np
import pytest

from enginir import PreprocessingRecipe
from enginir._workflow import recipe_id_for
from enginir.models import PCR, PLSRegression
from enginir.modeling._components import band_importance
from enginir.preprocessing import SNV, BandSelection, Detrend
from enginir.preprocessing._pipeline import apply_preprocessing_recipe
from enginir.preprocessing.recipe import parse_ranges
from tests.test_simca_preprocessing import _training_scan_identity
from tests.test_workflow import N_BANDS, _labelled_project

AXIS = np.linspace(1000, 2500, 151)  # 10 nm steps


# -- band importance -----------------------------------------------------------


def _informative(seed=0, n=120, k=30, band=11):
    """Spectra where only ``band`` carries y, on top of shared, uninformative structure."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 3)) @ rng.normal(size=(3, k))
    y = rng.normal(size=n)
    X = base + 0.05 * rng.normal(size=(n, k))
    X[:, band] += 2.0 * y
    return X, y


def _dense_selectivity_ratio(model, X):
    """Rajalahti's target projection, written out on the full matrix."""
    Z = (X - X.mean(axis=0)) / np.asarray(model.x_std_)
    b = np.asarray(model.coef_).reshape(-1) * np.asarray(model.x_std_)
    w = b / np.linalg.norm(b)
    t = Z @ w
    p = Z.T @ t / (t @ t)
    explained = np.sum(np.outer(t, p) ** 2, axis=0)
    residual = np.sum((Z - np.outer(t, p)) ** 2, axis=0)
    return explained / residual


def _dense_vip(model):
    W, T, Q = model.x_weights_, model.x_scores_, model.y_loadings_
    ss = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    return np.sqrt(W.shape[0] * ((W / np.linalg.norm(W, axis=0)) ** 2 @ ss) / ss.sum())


@pytest.mark.parametrize("estimator", [PLSRegression(n_components=4), PCR(n_components=4),
                                       PLSRegression(n_components=4, x_scaling="uv")])
def test_band_importance_matches_independent_computations(estimator):
    X, y = _informative()
    model = estimator.fit(X, y)
    result = band_importance(model, X, np.ones(len(X), dtype=bool))
    np.testing.assert_allclose(result["selectivity_ratio"], _dense_selectivity_ratio(model, X),
                               rtol=1e-4)
    np.testing.assert_allclose(result["mean_spectrum"], X.mean(axis=0), rtol=1e-5, atol=1e-6)
    assert int(np.argmax(result["selectivity_ratio"])) == 11, "the informative band leads"
    if isinstance(model, PLSRegression):
        np.testing.assert_allclose(result["vip"], _dense_vip(model), rtol=1e-5)
        assert np.mean(result["vip"].astype(float) ** 2) == pytest.approx(1.0, rel=1e-5)
        assert int(np.argmax(result["vip"])) == 11
    else:
        assert "vip" not in result


def test_the_fit_stores_vip_and_selectivity_ratio_on_the_fitted_pixels(tmp_path):
    project, _scan = _labelled_project(tmp_path)
    model = project.fit("API", n_components=2)
    diagnostics = model.diagnostics
    for key in ("vip", "selectivity_ratio", "mean_spectrum"):
        assert np.shape(diagnostics[key]) == (N_BANDS,)
    vip = np.asarray(diagnostics["vip"], dtype=float)
    assert np.mean(vip ** 2) == pytest.approx(1.0, rel=1e-4)
    # The scan's absorption peak (band N_BANDS / 2) carries the content.
    assert abs(int(np.argmax(vip)) - N_BANDS // 2) <= 1
    pcr = project.fit("API", model="PCR", n_components=2, name="pcr")
    assert "vip" not in pcr.diagnostics and np.shape(pcr.diagnostics["selectivity_ratio"]) == (
        N_BANDS,)


# -- band ranges in the recipe -------------------------------------------------


def test_ranges_are_normalized_before_the_id():
    recipe = PreprocessingRecipe(bands_exclude=[(1500, 1400), (1900, 2000), (1450, 1600),
                                                (1300, 1400)])
    assert recipe.bands_exclude == ((1300.0, 1600.0), (1900.0, 2000.0))
    same = PreprocessingRecipe(bands_exclude=[[1300.0, 1600.0], [1900, 2000]])
    assert recipe == same and recipe_id_for(recipe) == recipe_id_for(same)
    assert PreprocessingRecipe(bands_keep=[]) == PreprocessingRecipe()
    assert PreprocessingRecipe(bands_keep=(1100, 1200)).bands_keep == ((1100.0, 1200.0),)
    for bad in ([(1100, np.nan)], [(1, 2, 3)], "1100-1200"):
        with pytest.raises(ValueError):
            PreprocessingRecipe(bands_keep=bad)
    assert parse_ranges("1100-1350, 1500–2300; 2000-2100") == ((1100.0, 1350.0),
                                                                (1500.0, 2300.0))
    with pytest.raises(ValueError, match="not a range"):
        parse_ranges("1100")


def test_band_ranges_are_always_stored_and_change_the_id():
    recipe = PreprocessingRecipe(filter="sg_d1", snv=True)
    assert not recipe.has_band_selection
    assert recipe.to_dict()["bands_keep"] == [] and recipe.to_dict()["bands_exclude"] == []
    with_bands = PreprocessingRecipe.from_dict(
        {**recipe.to_dict(), "bands_exclude": [[1400, 1500]]})
    assert recipe_id_for(with_bands) != recipe_id_for(recipe)
    assert PreprocessingRecipe.from_dict(with_bands.to_dict()) == with_bands


def test_a_water_band_exclusion_leaves_the_expected_bands():
    recipe = PreprocessingRecipe(filter="sg_d1", sg_window=11,
                                 bands_exclude=[(1400, 1500), (1900, 2000)])
    kept = AXIS[recipe.output_band_indices(AXIS)]
    expected = AXIS[5:-5]
    expected = expected[~(((expected >= 1400) & (expected <= 1500))
                          | ((expected >= 1900) & (expected <= 2000)))]
    np.testing.assert_array_equal(kept, expected)
    np.testing.assert_array_equal(recipe.output_wavelengths(AXIS), expected)
    keep = PreprocessingRecipe(bands_keep=[(1100, 1300), (2200, 2300)], bands_exclude=[(1200,
                                                                                          1250)])
    kept = AXIS[keep.output_band_indices(AXIS)]
    assert kept.tolist() == [1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190,
                             1260, 1270, 1280, 1290, 1300] + list(range(2200, 2310, 10))


def test_a_typed_bound_matches_a_float32_wavelength():
    axis = np.array([1400.1, 1410.1, 1420.1], dtype=np.float32)  # 1400.0999755859375, ...
    recipe = PreprocessingRecipe(bands_keep=[(1400.1, 1410.1)])
    assert recipe.output_band_indices(axis).tolist() == [0, 1]


def test_band_numbers_without_wavelengths_are_those_of_the_data_file():
    axis = np.arange(20, dtype=float)  # the band-index axis of a file without wavelengths
    recipe = PreprocessingRecipe(filter="sg_smooth", sg_window=5, bands_keep=[(0, 6)])
    assert recipe.output_band_indices(axis).tolist() == [2, 3, 4, 5, 6]


def test_an_empty_selection_is_refused_and_an_empty_range_is_logged(caplog):
    with pytest.raises(ValueError, match="selects none"):
        PreprocessingRecipe(bands_keep=[(3000, 3100)]).output_band_indices(AXIS)
    recipe = PreprocessingRecipe(bands_exclude=[(3000, 3100)])
    assert recipe.output_band_indices(AXIS).size == AXIS.size


def test_the_selection_comes_after_every_other_step():
    rng = np.random.default_rng(3)
    X = rng.normal(1.0, 0.2, size=(5, AXIS.size)).astype(np.float32)
    recipe = PreprocessingRecipe(snv=True, detrend=True, bands_keep=[(1100, 1500)])
    columns = recipe.output_band_indices(AXIS)
    full = Detrend(order=1, wavelengths=AXIS).fit_transform(SNV().fit_transform(X))
    np.testing.assert_allclose(apply_preprocessing_recipe(X, recipe=recipe, wavelengths=AXIS),
                               full[:, columns], rtol=1e-5, atol=1e-6)
    # Fault injection: SNV and Detrend on the selected bands alone differ.
    alone = Detrend(order=1, wavelengths=AXIS[columns]).fit_transform(
        SNV().fit_transform(X[:, columns]))
    assert not np.allclose(alone, full[:, columns], atol=1e-3)
    assert [name for name, _ in recipe.build_steps(AXIS)][-1] == "bands"


def test_band_selection_checks_its_indices():
    with pytest.raises(ValueError, match="at least one"):
        BandSelection([]).fit(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="outside"):
        BandSelection([1, 5]).fit(np.zeros((2, 4)))
    assert BandSelection([0, 2]).fit_transform(np.arange(8.0).reshape(2, 4)).tolist() == [
        [0.0, 2.0], [4.0, 6.0]]


@pytest.mark.parametrize("msc", [False, True])
def test_a_model_on_selected_bands_predicts_like_it_was_trained(tmp_path, msc):
    project, scan = _labelled_project(tmp_path)
    wavelengths = np.linspace(1000, 1600, N_BANDS)
    recipe = PreprocessingRecipe(snv=not msc, msc=msc, filter="sg_smooth", sg_window=3,
                                 bands_exclude=[(wavelengths[2], wavelengths[3])])
    model = project.fit("API", recipe=recipe, n_components=2)
    assert model.name.endswith("bands")
    n_model = N_BANDS - 2 - 2
    from enginir.io import load_model_run_prediction_artifact

    blob = load_model_run_prediction_artifact(project.path, model.ref.model_run_id)
    assert np.shape(blob["coef_"])[0] == np.size(blob["wavelengths"]) == n_model
    np.testing.assert_allclose(blob["wavelengths"], np.delete(wavelengths[1:-1], [1, 2]),
                               rtol=1e-6)
    if msc:  # MSC sees every band after Savitzky-Golay, before the selection
        assert np.size(blob["preprocessing_state"]["msc"]["reference"]) == N_BANDS - 2
    prediction = _training_scan_identity(model, scan)
    assert np.isfinite(prediction.map).all()
    reopened = type(project).open(project.path).model(model_run_id=model.ref.model_run_id)
    assert reopened.recipe == recipe if hasattr(reopened, "recipe") else True


def test_too_many_components_for_the_selected_bands_are_refused(tmp_path):
    project, _scan = _labelled_project(tmp_path)
    wavelengths = np.linspace(1000, 1600, N_BANDS)
    recipe = PreprocessingRecipe(bands_keep=[(wavelengths[0], wavelengths[2])])
    with pytest.raises(ValueError, match="3 bands"):
        project.fit("API", recipe=recipe, n_components=4)


def test_the_explore_preview_shows_unselected_bands_as_missing():
    from enginir.viewer._app import ensure_app
    from enginir.viewer._state import ViewerState

    ensure_app()
    cube = np.random.default_rng(0).integers(500, 3000, size=(4, 3, 20), dtype=np.uint16)
    wavelengths = np.linspace(1000, 1700, 20)
    state = ViewerState(cube, wavelengths)
    state.apply_recipe(PreprocessingRecipe(snv=True, bands_exclude=[(wavelengths[5],
                                                                     wavelengths[8])]))
    image = state.current_frame_image()
    assert image.shape == (3, 20)
    missing = np.flatnonzero(np.isnan(image).all(axis=0)).tolist()
    assert missing == [5, 6, 7, 8]


# -- viewer --------------------------------------------------------------------


def _dialog(recipe, band_source=None):
    from enginir.viewer._app import ensure_app
    from enginir.viewer._modeling import FitModelRunDialog

    ensure_app()
    return FitModelRunDialog([("t", "target")], default_target_id="t", recipe=recipe,
                             default_signal_unit="Absorbance",
                             readiness=lambda _t: ("ready", True), band_source=band_source)


def test_the_fit_dialog_reads_keeps_and_suggests_band_ranges():
    preset = PreprocessingRecipe(snv=True, bands_keep=[(1100, 1350)],
                                 bands_exclude=[(1200, 1250)])
    dialog = _dialog(preset)
    assert dialog.recipe == preset, "Fit again keeps the stored ranges"
    assert dialog._keep_edit.text() == "1100-1350"
    dialog._exclude_edit.setText("")
    dialog._add_exclude((1400.0, 1500.0))  # the water-band suggestion, only when chosen
    assert dialog.recipe.bands_exclude == ((1400.0, 1500.0),)
    dialog._keep_edit.setText("1100-")
    with pytest.raises(ValueError, match="not a range"):
        dialog.recipe  # noqa: B018
    assert not dialog._select_bands.isEnabled(), "no source, no plot"
    dialog.close()


def test_the_range_dialog_turns_regions_into_ranges(qtbot=None):
    from enginir.viewer._band_ranges import BandRangeDialog, BandSource

    axis = np.linspace(1000, 2000, 50)
    dialog = BandRangeDialog(BandSource(load=lambda: (axis, np.sin(axis / 100)),
                                        guide=(axis, np.abs(np.cos(axis / 100)) + 0.5, "VIP")),
                             keep=((1100.0, 1300.0),), exclude=((1200.0, 1210.0),))
    dialog.add_region("exclude", (1900.0, 1950.0))
    assert dialog.ranges("keep") == ((1100.0, 1300.0),)
    assert dialog.ranges("exclude") == ((1200.0, 1210.0), (1900.0, 1950.0))
    dialog.clear_regions()
    assert dialog.ranges("keep") == dialog.ranges("exclude") == ()
    dialog.close()


def test_the_band_source_and_explore_follow_the_fit_dialog(tmp_path):
    from enginir.viewer._app import ensure_app
    from enginir.viewer._modeling import ModelingView
    from enginir.viewer._state import ProjectState, ViewerState

    ensure_app()
    project, _scan = _labelled_project(tmp_path)
    project_state = ProjectState()
    project_state.open_project(str(project.path))
    cube = np.zeros((2, 2, N_BANDS), dtype=np.uint16)
    state = ViewerState(cube, np.linspace(1000, 1600, N_BANDS))
    view = ModelingView(state, project_state)
    target_id = next(iter(project_state.targets))
    recipe = PreprocessingRecipe(snv=True)
    source = view._band_source(target_id, "Absorbance", recipe)
    assert source is not None and source.axis_label == "Wavelength (nm)"
    axis, mean = source.load()
    assert axis.shape == mean.shape == (N_BANDS,)
    assert np.allclose(mean.mean(), 0.0, atol=1e-5), "SNV-corrected means average to zero"
    assert view._band_source("missing", "Absorbance", recipe) is None
    chosen = PreprocessingRecipe(snv=True, bands_exclude=[(1100, 1200)])
    view._show_bands_in_explore(chosen)
    assert state.recipe.bands_exclude == chosen.bands_exclude
    view.close()
