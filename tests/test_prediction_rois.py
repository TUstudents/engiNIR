"""Prediction ROIs (roi-target contract ``use``; prediction-runtime contract, Prediction ROIs)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from enginir import Project
from enginir.io import save_hdf5
from tests._scans import TRAY_BANDS as N_BANDS, TRAY_HEIGHT as HEIGHT, warnings_during


def test_the_api_accepts_only_calibration_and_prediction(tray):
    project, scan, _path = tray
    with pytest.raises(ValueError, match="job must be one of"):
        project.set_job("t01", "training")
    with pytest.raises(ValueError, match="job must be one of"):
        scan.add_roi("t11", frames=(0, 1), rows=(0, 1), job="training")
    assert project.roi("t01").job == "calibration"


def test_prediction_rois_never_enter_a_fit(tray):
    project, _scan, _path = tray
    model = project.fit("API", n_components=2, cv=3)
    sample_set = project.workspace.sample_sets[model.record.sample_set_id]
    names = {project.workspace.rois[roi_id].name for roi_id in sample_set.roi_ids}
    assert names == {f"t{i:02d}" for i in range(1, 8)}, "t08-t10 are prediction ROIs"

    # A stored sample set keeps its ROIs: switching one later only warns on refit.
    project.set_job("t03", "prediction")
    refitted, messages = warnings_during(model.refit)
    assert any("now prediction ROIs" in message for message in messages)
    assert refitted.record.sample_set_id == model.record.sample_set_id

    for name in [f"t{i:02d}" for i in range(1, 8)]:
        project.set_job(name, "prediction")
    with pytest.raises(ValueError, match=r"No calibration ROI has a value .*10 prediction ROI"):
        project.fit("API", n_components=2, cv=3)


def test_an_roi_that_cannot_give_the_models_signal_is_refused_and_the_others_predicted(
        tray, tmp_path):
    project, _scan, _path = tray
    model = project.fit("API", n_components=2, cv=3)  # absorbance
    raw = tmp_path / "raw.h5"
    save_hdf5(raw, np.full((4, HEIGHT, N_BANDS), 2000, dtype=np.uint16),
              np.linspace(1000, 1600, N_BANDS))
    counts = project.add_dataset(raw, unit="DN")
    counts.add_roi("raw", frames=(0, 3), rows=(0, HEIGHT - 1), job="prediction")
    prediction = model.roi_prediction()
    assert set(prediction.refused) == {project.roi("raw").roi_id}
    assert len(prediction.pixels) == 3


def test_the_default_exclusion_is_the_models_own(tray):
    project, _scan, _path = tray
    plain = project.fit("API", n_components=2, cv=3)
    excluding = project.fit("API", n_components=2, cv=3,
                            fit_spec={"exclusion": {"method": "dmodx", "level": 0.95}})
    assert plain.roi_prediction().default_exclusion is None
    assert excluding.roi_prediction().default_exclusion == 0.95
    assert excluding.predict_rois().exclusion == 0.95
    assert excluding.predict_rois(exclude_dmodx=False).exclusion is None


# -- Prediction ROIs outside the model (one rule: the check and the Predict ring) ------


def test_a_model_without_calibration_shares_has_no_rule(tmp_path):
    """A converted model (shares renamed to *_099) is never judged with the 0.99 shares."""
    from enginir.prediction import unusual_rois
    from tests._scans import tray_project

    project, _dataset, _path = tray_project(tmp_path, odd="t09")
    model = project.fit("API", n_components=2, cv=3)
    with h5py.File(project.path, "r+") as f:
        diagnostics = f[f"model_runs/{model.record.model_run_id}/artifact_refs/diagnostics"]
        diagnostics.attrs["roi_share_above_dcrit_099"] = diagnostics.attrs["roi_share_above_dcrit"]
        del diagnostics.attrs["roi_share_above_dcrit"]
    model = Project.open(project.path).model(model_run_id=model.record.model_run_id)
    assert unusual_rois(model.roi_prediction([project.roi("t09")])) == ()


# -- A layout on a scan (Prediction.roi_statistics): the statistics of stored ROIs -------


def test_a_layout_on_a_scan_gives_the_statistics_of_the_same_stored_rois(tmp_path):
    """Per-tablet values on a production line are those of the same pixels stored as
    prediction ROIs: one function, so a tablet has one value everywhere."""
    from enginir import GridCell
    from enginir.prediction import unusual_rois
    from tests._scans import tray_project

    project, _dataset, path = tray_project(tmp_path, odd="t10")
    model = project.fit("API", n_components=2, cv=3,
                        fit_spec={"exclusion": {"method": "dmodx", "level": 0.95}})
    rois = [project.roi(f"t{i:02d}") for i in (8, 9, 10)]
    layout = [GridCell(row=0, column=i, frames=roi.spatial_bounds[:2],
                       rows=roi.spatial_bounds[2:], name=roi.name) for i, roi in enumerate(rois)]
    stored = {project.workspace.rois[row.roi_id].name: row for row in model.predict_rois(rois).rois}
    scan = model.predict(path)
    on_scan = {row.roi_id: row for row in scan.roi_statistics(layout).rois}

    assert set(on_scan) == set(stored) == {"t08", "t09", "t10"}
    for name, row in on_scan.items():
        expected = stored[name]
        assert (row.n_pixels, row.n_removed) == (expected.n_pixels, expected.n_removed), name
        for field in ("mean", "sd", "median", "share_above_dcrit", "share_above_t2crit"):
            assert getattr(row, field) == pytest.approx(getattr(expected, field), rel=1e-4, nan_ok=True,
                                                        abs=1e-6), (name, field)
    assert unusual_rois(scan.roi_prediction(layout)) == ("t10",)
    assert scan.roi_statistics(layout, exclude_dmodx=False).exclusion is None


def test_a_layout_cell_outside_the_scan_or_a_repeated_name_is_refused(tray):
    from enginir import GridCell

    project, _scan, path = tray
    prediction = project.fit("API", n_components=2, cv=3).predict(path)
    n_frames, height = prediction.map.shape
    inside = GridCell(row=0, column=0, frames=(2, 5), rows=(0, height - 1), name="A1")
    with pytest.raises(ValueError, match="Cell 'B1' .* is outside the scan"):
        prediction.roi_statistics([inside, GridCell(row=1, column=0, frames=(2, 5),
                                                    rows=(0, height), name="B1")])
    with pytest.raises(ValueError, match="Cell names must be unique"):
        prediction.roi_statistics([inside, inside])


# -- The reference correction of a prediction (Model.predict, as Project.add_dataset) -----


def test_predict_takes_the_reference_choices_of_add_dataset(tray, tmp_path):
    """A line chooses how each new scan is corrected: frames of the scan or detector levels
    give the same correction as the registered scan's own (its frames are dark 120 and white
    3400 counts); nothing given uses the defaults; levels exclude references."""
    import shutil

    project, _scan, path = tray
    model = project.fit("API", n_components=2, cv=3)
    registered = model.predict(path).map
    new = shutil.copy(path, tmp_path / "new_scan.h5")
    n = registered.shape[0]  # every frame of the scan, its 2 dark and 2 white frames too
    by_frames = model.predict(new, dark_frames=(0, 1), white_frames=(n - 2, n - 1)).map
    by_levels = model.predict(new, black_level=120, white_level=3400).map
    np.testing.assert_allclose(by_frames, registered, rtol=1e-5)
    np.testing.assert_allclose(by_levels, registered, rtol=1e-5)
    assert not np.allclose(model.predict(new).map, registered, rtol=1e-3)
    assert not np.allclose(model.predict(path, black_level=0, white_level=4000).map, registered,
                           rtol=1e-3), "a registered scan with levels given is corrected anew"
    with pytest.raises(ValueError, match="Detector levels correct without references"):
        model.predict(new, black_level=120, dark_frames=(0, 1))
