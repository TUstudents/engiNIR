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
