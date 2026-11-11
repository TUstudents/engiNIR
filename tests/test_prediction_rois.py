"""Prediction ROIs (roi-target contract ``use``; prediction-runtime contract, Prediction ROIs)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from enginir import PreprocessingRecipe, Project
from enginir.io import save_hdf5
from enginir.prediction import roi_prediction_statistics
from tests.test_fit_spec import _warnings_during

FRAMES_PER_ROI, HEIGHT, N_BANDS = 4, 5, 24


@pytest.fixture()
def tray(tmp_path):
    """One tray of 10 ROIs (levels 1..10, references from frames); ROIs 8-10 for prediction."""
    rng = np.random.default_rng(11)
    peak = np.exp(-((np.arange(N_BANDS) - 9) / 3.0) ** 2)
    blocks = [np.full((2, HEIGHT, N_BANDS), 120.0)]  # dark frames
    values = np.arange(1.0, 11.0)
    for value in values:
        blocks.append(2600 - 110 * value * peak + rng.normal(0, 20, (FRAMES_PER_ROI, HEIGHT,
                                                                     N_BANDS)))
    blocks.append(np.full((2, HEIGHT, N_BANDS), 3400.0))  # white frames
    cube = np.vstack(blocks).astype(np.uint16)
    path = tmp_path / "tray.h5"
    save_hdf5(path, cube, np.linspace(1000, 1600, N_BANDS))
    project = Project.create(tmp_path / "project.h5")
    n = cube.shape[0]
    scan = project.add_dataset(path, dark_frames=(0, 1), white_frames=(n - 2, n - 1))
    project.add_target("API", unit="%")
    for i, value in enumerate(values):
        first = 2 + i * FRAMES_PER_ROI
        scan.add_roi(f"t{i + 1:02d}", frames=(first, first + FRAMES_PER_ROI - 1),
                     rows=(0, HEIGHT - 1), values={"API": float(value)},
                     use="prediction" if i >= 7 else "calibration")
    return project, scan, path


def test_the_use_is_stored_read_back_and_switched_in_place(tray):
    """Calibration and prediction are stored explicitly and switched in place; an unknown
    stored value makes the project fail to open."""
    project, _scan, _path = tray
    reopened = Project.open(project.path)
    assert reopened.roi("t09").use == "prediction" and reopened.roi("t01").use == "calibration"
    with h5py.File(project.path, "r") as f:
        assert f[f"rois/{project.roi('t01').roi_id}"].attrs["use"] == "calibration"
    reopened.set_use("t09", "calibration")
    assert Project.open(project.path).roi("t09").use == "calibration"
    with h5py.File(project.path, "r+") as f:
        f[f"rois/{project.roi('t02').roi_id}"].attrs["use"] = "validation"
    with pytest.raises(ValueError, match="use must be one of"):
        Project.open(project.path)


@pytest.mark.parametrize("stored", [None, "training"])
def test_older_stored_forms_read_as_calibration_and_resave_unchanged(tray, stored):
    """A missing use (before 1.4) and 'training' (1.4) read as calibration, and re-saving such
    an ROI is not refused as a change of content (terminology contract)."""
    from enginir.io import project_store as store

    project, _scan, _path = tray
    roi = project.roi("t01")
    with h5py.File(project.path, "r+") as f:
        attrs = f[f"rois/{roi.roi_id}"].attrs
        del attrs["use"]
        if stored is not None:
            attrs["use"] = stored
    reopened = Project.open(project.path)
    assert reopened.roi("t01").use == "calibration"
    key = f"/rois/{roi.roi_id}/spectra"
    spectra = store.load_roi_spectra_batch(project.path, [key])[key]
    refs = next(iter(store.iter_roi_calibration_refs(project.path, [roi.roi_id])))[1]
    dark, white = refs if refs is not None else (None, None)
    store.save_training_roi(project.path, roi.roi_id, reopened.roi("t01").to_dict(), spectra,
                            dark_ref_spectra=dark, white_ref_spectra=white)
    with h5py.File(project.path, "r") as f:
        assert f[f"rois/{roi.roi_id}"].attrs.get("use") == stored, "re-saving changes nothing"
    assert reopened.fit("API", n_components=2, cv=3).ref is not None, "still a calibration ROI"


def test_the_api_refuses_the_old_name_training(tray):
    project, scan, _path = tray
    with pytest.raises(ValueError, match="renamed to 'calibration'"):
        project.set_use("t01", "training")
    with pytest.raises(ValueError, match="renamed to 'calibration'"):
        scan.add_roi("t11", frames=(0, 1), rows=(0, 1), use="training")
    assert project.roi("t01").use == "calibration"


def test_prediction_rois_never_enter_a_fit(tray):
    project, _scan, _path = tray
    model = project.fit("API", n_components=2, cv=3)
    sample_set = project.workspace.sample_sets[model.ref.sample_set_id]
    names = {project.workspace.rois[roi_id].name for roi_id in sample_set.roi_ids}
    assert names == {f"t{i:02d}" for i in range(1, 8)}, "t08-t10 are prediction ROIs"

    # A stored sample set keeps its ROIs: switching one later only warns on refit.
    project.set_use("t03", "prediction")
    refitted, messages = _warnings_during(model.refit)
    assert any("now prediction ROIs" in message for message in messages)
    assert refitted.ref.sample_set_id == model.ref.sample_set_id

    for name in [f"t{i:02d}" for i in range(1, 8)]:
        project.set_use(name, "prediction")
    with pytest.raises(ValueError, match=r"No training ROI has a value .*10 prediction ROI"):
        project.fit("API", n_components=2, cv=3)


def test_predicting_rois_equals_predicting_the_scan_at_their_pixels(tray):
    """One chain: the ROI path (stored pixels, the ROI's calibration) gives the scan path's
    pixel predictions and diagnostics."""
    project, _scan, path = tray
    model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=2, cv=3)
    roi_prediction = model.roi_prediction()
    scan = model.predict(path)
    assert set(roi_prediction.pixels) == {project.roi(f"t{i:02d}").roi_id for i in (8, 9, 10)}
    for roi_id, pixels in roi_prediction.pixels.items():
        f0, f1, r0, r1 = project.workspace.rois[roi_id].spatial_bounds
        np.testing.assert_allclose(pixels.y, scan.map[f0:f1 + 1, r0:r1 + 1].reshape(-1),
                                   rtol=1e-5, atol=1e-5)
        dmodx_map = scan.dmodx_map
        assert dmodx_map is not None
        np.testing.assert_allclose(pixels.dmodx, dmodx_map[f0:f1 + 1, r0:r1 + 1].reshape(-1),
                                   rtol=1e-4, atol=1e-5)


def test_statistics_follow_the_contract_and_the_exclusion_removes_glare(tray):
    project, _scan, _path = tray
    model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=2, cv=3)
    prediction = model.roi_prediction()
    roi_id = project.roi("t09").roi_id
    pixels = prediction.pixels[roi_id]
    # Glare: three pixels far outside the model, with absurd predictions.
    y, dmodx = pixels.y.copy(), pixels.dmodx.copy()
    y[:3], dmodx[:3] = 80.0, 50.0
    prediction.pixels[roi_id] = type(pixels)(y=y, dmodx=dmodx, t2=pixels.t2)
    assignments = project.workspace.target_assignments

    plain = roi_prediction_statistics(prediction, target_assignments=assignments)
    row = next(r for r in plain.rois if r.roi_id == roi_id)
    np.testing.assert_allclose(row.mean, y.astype(np.float64).mean())
    np.testing.assert_allclose(row.sd, y.astype(np.float64).std(ddof=1))
    np.testing.assert_allclose((row.p05, row.p95), np.percentile(y.astype(np.float64), [5, 95]))
    assert row.n_removed == 0 and row.reference == 9.0
    assert row.share_above_dcrit >= 3 / y.size

    cleaned = roi_prediction_statistics(prediction, target_assignments=assignments,
                                        exclude_dmodx=0.99)
    row = next(r for r in cleaned.rois if r.roi_id == roi_id)
    limit = prediction.dmodx_limit_at(0.99)
    assert limit is not None
    kept = dmodx <= limit
    assert row.n_removed == int((~kept).sum()) >= 3
    np.testing.assert_allclose(row.mean, y[kept].astype(np.float64).mean())
    assert row.mean < 15, "the glare no longer shifts the ROI"
    assert row.share_above_dcrit == next(r for r in plain.rois
                                         if r.roi_id == roi_id).share_above_dcrit

    residuals = np.array([r.residual for r in cleaned.rois])
    assert cleaned.n_reference == 3
    np.testing.assert_allclose(cleaned.rmsep, np.sqrt(np.mean(residuals ** 2)))
    np.testing.assert_allclose(cleaned.bias, residuals.mean())
    assert prediction.dmodx_limit_at(0.95) < limit


def test_an_roi_that_cannot_give_the_models_signal_is_refused_and_the_others_predicted(
        tray, tmp_path):
    project, _scan, _path = tray
    model = project.fit("API", n_components=2, cv=3)  # absorbance
    raw = tmp_path / "raw.h5"
    save_hdf5(raw, np.full((4, HEIGHT, N_BANDS), 2000, dtype=np.uint16),
              np.linspace(1000, 1600, N_BANDS))
    counts = project.add_dataset(raw, unit="DN")
    counts.add_roi("raw", frames=(0, 3), rows=(0, HEIGHT - 1), use="prediction")
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
