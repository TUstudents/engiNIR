"""Band proposal: forward interval PLS with nested validation (headless contract)."""

from __future__ import annotations

import numpy as np
import pytest

from enginir import PreprocessingRecipe, Project
from enginir.io import save_hdf5
from enginir.modeling import BandProposalCancelledError

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


def test_the_nested_estimate_is_not_optimistic_on_noise(tmp_path):
    """With a target unrelated to the spectra, the honest nested RMSECV is no better than
    the mean predictor on average. Fault injection (the inner search choosing on every
    ROI, not only the outer training part) gives about 0.89 here and fails."""
    ratios = []
    for seed in range(6):
        folder = tmp_path / f"s{seed}"
        folder.mkdir()
        project = _project(folder, informative=False, seed=seed, n_rois=14, n_bands=40)
        proposal = project.propose_bands("API", n_intervals=8)
        ratios.append(proposal.nested_rmsecv / proposal.steps[0].rmsecv)
    assert np.mean(ratios) >= 0.97


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
    assert cancelled.proposal is None and "Cancelled" in cancelled._status.text()
    cancelled.close()
    view.close()
