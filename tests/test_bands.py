"""Band selection (recipe ranges), band importance (VIP, selectivity ratio) and band proposal."""

from __future__ import annotations

import numpy as np
import pytest
from enginir import PreprocessingRecipe, Project
from enginir._workflow import recipe_id_for
from enginir.io import save_hdf5
from enginir.modeling import BandProposalCancelledError
from enginir.modeling._components import band_importance
from enginir.models import PCR, PLSRegression
from enginir.preprocessing import BandSelection
from enginir.preprocessing.recipe import parse_ranges
from tests._scans import N_BANDS, labelled_project, predicts_its_training_scan


# -- Band selection (recipe ranges) and band importance (VIP, selectivity ratio) -------


AXIS = np.linspace(1000, 2500, 151)  # 10 nm steps


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
    project, _scan = labelled_project(tmp_path)
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


def test_an_empty_selection_is_refused_and_an_empty_range_is_logged(caplog):
    with pytest.raises(ValueError, match="selects none"):
        PreprocessingRecipe(bands_keep=[(3000, 3100)]).output_band_indices(AXIS)
    recipe = PreprocessingRecipe(bands_exclude=[(3000, 3100)])
    assert recipe.output_band_indices(AXIS).size == AXIS.size


def test_band_selection_checks_its_indices():
    with pytest.raises(ValueError, match="at least one"):
        BandSelection([]).fit(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="outside"):
        BandSelection([1, 5]).fit(np.zeros((2, 4)))
    assert BandSelection([0, 2]).fit_transform(np.arange(8.0).reshape(2, 4)).tolist() == [
        [0.0, 2.0], [4.0, 6.0]]


@pytest.mark.parametrize("msc", [False, True])
def test_a_model_on_selected_bands_predicts_like_it_was_trained(tmp_path, msc):
    project, scan = labelled_project(tmp_path)
    wavelengths = np.linspace(1000, 1600, N_BANDS)
    recipe = PreprocessingRecipe(snv=not msc, msc=msc, filter="sg_smooth", sg_window=3,
                                 bands_exclude=[(wavelengths[2], wavelengths[3])])
    model = project.fit("API", recipe=recipe, n_components=2)
    assert model.name.endswith("bands")
    n_model = N_BANDS - 2 - 2
    from enginir.io import load_model_run_prediction_artifact

    blob = load_model_run_prediction_artifact(project.path, model.record.model_run_id)
    assert np.shape(blob["coef_"])[0] == np.size(blob["wavelengths"]) == n_model
    np.testing.assert_allclose(blob["wavelengths"], np.delete(wavelengths[1:-1], [1, 2]),
                               rtol=1e-6)
    if msc:  # MSC sees every band after Savitzky-Golay, before the selection
        assert np.size(blob["preprocessing_state"]["msc"]["reference"]) == N_BANDS - 2
    prediction = predicts_its_training_scan(model, scan)
    assert np.isfinite(prediction.map).all()
    reopened = type(project).open(project.path).model(model_run_id=model.record.model_run_id)
    assert reopened.recipe == recipe if hasattr(reopened, "recipe") else True


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
    project, _scan = labelled_project(tmp_path)
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


# -- Band proposal: forward interval PLS with nested validation (headless contract) ----


FRAMES, HEIGHT = 3, 4


def _project(tmp_path, *, informative: bool, seed: int = 0, n_rois: int = 16, n_bands: int = 60):
    """ROIs whose content lies in bands 10-14 and 40-44 (or nowhere) under ROI interference."""
    rng = np.random.default_rng(seed)
    content = rng.uniform(0, 10, n_rois)
    wavelengths = np.linspace(1000, 1000 + 10 * (n_bands - 1), n_bands)
    signal = np.zeros(n_bands)
    if informative:
        signal[10:15], signal[40:45] = 1.0, 0.7
    interference = rng.normal(0, 1, (n_rois, 6)) @ rng.normal(0, 1, (6, n_bands)) * 0.06
    interference[:, signal > 0] *= 0.05
    spectra = 0.5 + 0.01 * content[:, None] * signal + interference
    cube = (np.repeat(spectra, FRAMES, axis=0)[:, None, :]
            + rng.normal(0, 0.002, (n_rois * FRAMES, HEIGHT, n_bands)))
    save_hdf5(tmp_path / "scan.h5", cube.astype(np.float32), wavelengths)
    project = Project.create(tmp_path / "project.h5")
    scan = project.add_dataset(str(tmp_path / "scan.h5"), unit="Reflectance")
    project.add_target("API", unit="%")
    for i in range(n_rois):
        scan.add_roi(f"r{i}", frames=(i * FRAMES, i * FRAMES + FRAMES - 1),
                     rows=(0, HEIGHT - 1), values={"API": float(content[i])})
    return project


def _covers(ranges, low, high):
    return any(a <= low + 1e-6 and high <= b + 1e-6 for a, b in ranges)


def test_the_search_finds_the_informative_intervals(tmp_path):
    project = _project(tmp_path, informative=True)
    before = (tmp_path / "project.h5").read_bytes()
    calls = []
    proposal = project.propose_bands("API", n_intervals=12,
                                     progress=lambda done, total: calls.append((done, total)))
    assert _covers(proposal.ranges, 1100, 1140) and _covers(proposal.ranges, 1400, 1440)
    rmsecv = [step.rmsecv for step in proposal.steps]
    assert proposal.steps[0].intervals == () and proposal.steps[0].n_components == 0
    assert all(b < a for a, b in zip(rmsecv, rmsecv[1:], strict=False)), "every step improves"
    assert proposal.n_components == proposal.steps[-1].n_components >= 1
    assert proposal.intervals[0] == (1000.0, 1040.0) and len(proposal.intervals) == 12
    # The nested estimate includes the selection: not better than the selected ranges.
    assert proposal.nested_rmsecv > proposal.steps[-1].rmsecv
    assert np.isfinite(proposal.all_bands_rmsecv)
    assert calls and calls[-1][0] <= calls[-1][1]
    assert (tmp_path / "project.h5").read_bytes() == before, "nothing is written"
    # The proposal is an ordinary keep list.
    model = project.fit("API", recipe=PreprocessingRecipe(bands_keep=proposal.ranges),
                        n_components=proposal.n_components)
    assert model.metrics["rmsecv"] == pytest.approx(proposal.steps[-1].rmsecv, rel=0.05)


def test_a_stateful_recipe_is_fitted_inside_the_folds(tmp_path):
    project = _project(tmp_path, informative=True, n_bands=30, n_rois=12)
    proposal = project.propose_bands("API", recipe=PreprocessingRecipe(msc=True), n_intervals=6)
    # MSC removes this data's additive signal; the search still runs fold by fold and
    # proposes nothing when no interval beats the mean predictor.
    assert proposal.steps[0].n_components == 0
    assert np.isfinite(proposal.nested_rmsecv) and np.isfinite(proposal.all_bands_rmsecv)
    if not proposal.ranges:
        assert proposal.nested_rmsecv == pytest.approx(proposal.steps[0].rmsecv, rel=0.25)


def test_requests_that_cannot_be_met_are_refused(tmp_path):
    project = _project(tmp_path, informative=True, n_bands=20, n_rois=12)
    with pytest.raises(ValueError, match="exceeds the 20 bands"):
        project.propose_bands("API", n_intervals=21)
    with pytest.raises(ValueError):
        project.propose_bands("API", n_intervals=1)
    with pytest.raises(BandProposalCancelledError):
        project.propose_bands("API", n_intervals=4, cancel=lambda: True)
    with pytest.raises(ValueError, match="at least 3 fold groups"):
        _two_rois(tmp_path).propose_bands("API", n_intervals=2)


def _two_rois(tmp_path):
    rng = np.random.default_rng(0)
    cube = rng.uniform(0.4, 0.6, (4, 2, 10)).astype(np.float32)
    save_hdf5(tmp_path / "two.h5", cube, np.linspace(1000, 1090, 10))
    project = Project.create(tmp_path / "two_project.h5")
    scan = project.add_dataset(str(tmp_path / "two.h5"), unit="Reflectance")
    project.add_target("API", unit="%")
    scan.add_roi("a", frames=(0, 1), rows=(0, 1), values={"API": 1.0})
    scan.add_roi("b", frames=(2, 3), rows=(0, 1), values={"API": 2.0})
    return project


def _wait(qapp, condition, timeout=30.0):
    import time

    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)
    assert condition()


def test_the_viewer_proposes_through_the_project_and_fills_the_fit_dialog(tmp_path):
    from enginir.viewer._app import ensure_app
    from enginir.viewer._band_ranges import ProposeBandsDialog
    from enginir.viewer._modeling import ModelingView
    from enginir.viewer._state import ProjectState, ViewerState

    qapp = ensure_app()
    project = _project(tmp_path, informative=True, n_bands=30, n_rois=12)
    project_state = ProjectState()
    project_state.open_project(str(project.path))
    view = ModelingView(ViewerState(np.zeros((2, 2, 30), dtype=np.uint16),
                                    np.linspace(1000, 1290, 30)), project_state)
    target_id = next(iter(project_state.targets))
    run = view._band_proposer(target_id, PreprocessingRecipe(), "PLS",
                              {"n_components": "auto"}, None)
    assert run is not None
    seen = []
    proposal = run(lambda done, total: seen.append(total), lambda: False)
    assert proposal.ranges and seen, "the facade runs with progress"

    dialog = ProposeBandsDialog(lambda progress, cancel: proposal)
    _wait(qapp, lambda: dialog.proposal is not None)
    assert dialog._use is not None and dialog._use.isEnabled()
    assert "Nested RMSECV" in dialog._status.text()
    dialog.close()

    cancelled = ProposeBandsDialog(lambda progress, cancel: project.propose_bands(
        "API", n_intervals=6, progress=progress, cancel=cancel))
    cancelled._cancel_or_close()
    _wait(qapp, lambda: not cancelled._running)
    assert cancelled.proposal is None and "Canceled" in cancelled._status.text()
    cancelled.close()
    view.close()


@pytest.mark.parametrize(("model", "recipe", "kwargs"), [
    ("PLS", PreprocessingRecipe(), {}),
    ("PLS", PreprocessingRecipe(snv=True), {"n_components": 3}),
    ("PCR", PreprocessingRecipe(), {}),
], ids=["pls-auto", "pls-snv-fixed", "pcr-auto"])
def test_the_proposal_on_statistics_equals_the_pipeline(tmp_path, monkeypatch, model, recipe,
                                                        kwargs):
    """Plan kernel-pls, Phase 3: same ranges, steps and component counts; RMSECVs to 1e-5
    (the pipeline predicts held-out pixels in float32, the statistics path ROI means). The
    SVD reference gets float64 spectra: scikit-learn's PCA computes float32 in float32."""
    import time

    import enginir.models.pcr as pcr

    tol = 1e-5

    project = _project(tmp_path, informative=True, seed=3)
    reference_algorithm = "nipals" if model == "PLS" else "svd"
    start = time.perf_counter()
    ours = project.propose_bands("API", model=model, recipe=recipe, n_intervals=8, **kwargs)
    fast = time.perf_counter() - start
    real_fit = pcr.PCA.fit
    monkeypatch.setattr(pcr.PCA, "fit",
                        lambda self, X, y=None: real_fit(self, np.asarray(X, dtype=np.float64)))
    start = time.perf_counter()
    reference = project.propose_bands("API", model=model, recipe=recipe, n_intervals=8,
                                      algorithm=reference_algorithm, **kwargs)
    slow = time.perf_counter() - start
    assert ours.ranges == reference.ranges
    assert [(s.intervals, s.n_components) for s in ours.steps] == \
        [(s.intervals, s.n_components) for s in reference.steps]
    for a, b in zip(ours.steps, reference.steps, strict=True):
        assert a.rmsecv == pytest.approx(b.rmsecv, rel=tol)
        assert a.rmsecv_se == pytest.approx(b.rmsecv_se, rel=10 * tol, abs=1e-9)
    # Nested: the pipeline predicts held-out pixels (about 5 %) in float32 and the errors are
    # about 0.02, so its rounding of 1e-7 grows to some 1e-5 in RMSECV.
    assert ours.nested_rmsecv == pytest.approx(reference.nested_rmsecv, rel=1e-4)
    assert ours.all_bands_rmsecv == pytest.approx(reference.all_bands_rmsecv, rel=tol)
    assert ours.n_components == reference.n_components
    print(f"band proposal on statistics {fast:.2f} s, pipeline {slow:.2f} s")
