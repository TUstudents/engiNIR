"""Prediction intervals by the leverage formula (prediction-runtime contract, Prediction Intervals)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from scipy import stats

from enginir.io import load_model_run_prediction_artifact, load_spectra
from enginir.prediction import roi_prediction_statistics
from enginir.project import transform_spectra_to_signal_domain
from tests.test_prediction_rois import FRAMES_PER_ROI, N_BANDS
from tests.test_prediction_rois import tray as _tray

N_TRAINING_ROIS = 7


@pytest.fixture()
def tray(tmp_path):
    """The prediction-ROI tray: ROIs t01-t07 for training, t08-t10 to predict."""
    return _tray.__wrapped__(tmp_path)


def _scores(project, model, path) -> np.ndarray:
    """Scores of every scan pixel, from the stored projection data (independent of prediction)."""
    blob = load_model_run_prediction_artifact(project.path, model.ref.model_run_id)
    dataset = project._dataset_ref(str(path))
    calibration = project.workspace.calibrations[
        project.workspace.active_calibration_ids[dataset.dataset_id]]
    domain = project.workspace.signal_domains[model.ref.signal_domain_id]
    raw = load_spectra(path).X
    X = transform_spectra_to_signal_domain(
        raw.reshape(-1, N_BANDS), dataset=dataset, calibration=calibration,
        signal_domain=domain, spatial_shape=raw.shape[:2]).astype(np.float64)
    return ((X - blob["x_mean"]) / blob["x_std"]) @ np.asarray(blob["rotations"], np.float64)


def _q(level: float, n_rois: int, a: int) -> float:
    return float(stats.t.ppf(1 - (1 - level) / 2, n_rois - a - 1))


def test_pixel_intervals_follow_the_leverage_formula_of_the_hat_matrix(tray):
    """T²PS/(N − 1) is the hat-matrix term of the training scores; with the ROIs as the samples
    h = 1/R + T²PS/(R − 1) and the half-width is q · pixel RMSEE · √(1 + h), q at R − A − 1."""
    project, _scan, path = tray
    model = project.fit("API", n_components=2, cv=3)
    prediction = model.predict(path)
    scores = _scores(project, model, path)
    height = prediction.map.shape[1]
    first, last = 2 * height, (2 + N_TRAINING_ROIS * FRAMES_PER_ROI) * height
    training = scores[first:last]  # the training ROIs tile these frames
    n, a = training.shape
    centred = training - training.mean(axis=0)
    hat = np.einsum("ij,jk,ik->i", scores, np.linalg.inv(centred.T @ centred), scores)
    r = N_TRAINING_ROIS
    leverage = 1 / r + hat * (n - 1) / (r - 1)
    expected = (_q(0.95, N_TRAINING_ROIS, a) * model.metrics["pixel_rmsee"]
                * np.sqrt(1 + leverage))

    half = prediction.uncertainty_map(0.95)
    assert half is not None and half.shape == prediction.map.shape
    half = half.reshape(-1)
    inside = np.isfinite(half)
    assert inside[first:last].mean() > 0.9, "almost every training pixel is inside its model"
    np.testing.assert_allclose(half[inside], expected[inside], rtol=1e-5)
    wider = prediction.uncertainty_map(0.99)
    assert wider is not None and np.all(wider.reshape(-1)[inside] > half[inside])
    with pytest.raises(ValueError, match="between 0 and 1"):
        prediction.uncertainty_map(1.0)


def test_pixels_outside_the_model_have_no_interval(tray, tmp_path):
    project, _scan, path = tray
    model = project.fit("API", n_components=2, cv=3)
    odd = tmp_path / "odd.h5"
    odd.write_bytes(path.read_bytes())
    with h5py.File(odd, "r+") as f:
        cube = f["spectra"][...]
        cube[10, 2, : N_BANDS // 2] += 600  # a step the model never saw
        f["spectra"][...] = cube
    prediction = model.predict(odd)
    half, dmodx, t2 = prediction.uncertainty_map(), prediction.dmodx_map, prediction.t2_map
    assert half is not None and dmodx is not None and t2 is not None
    limits = prediction.diagnostics or {}
    assert dmodx[10, 2] > limits["dcrit_099"]
    assert np.isnan(half[10, 2])
    outside = (dmodx > limits["dcrit_099"]) | (t2 > limits["t2_crit_099"])
    assert np.array_equal(np.isnan(half), outside)


def test_the_mean_interval_of_a_prediction_roi_uses_the_training_roi_scores(tray):
    """h̄ = 1/R′ + d²/(R′ − 1) against the stored ROI scores, s = RMSEE; it follows the pixel
    exclusion, and a ROI beyond the ROI-level T² limit has none."""
    project, _scan, _path = tray
    model = project.fit("API", n_components=2, cv=3)
    prediction = model.roi_prediction()
    roi_scores = np.asarray(model.diagnostics["roi_scores"], dtype=np.float64)
    r, a = roi_scores.shape
    centre, cov_inv = roi_scores.mean(axis=0), np.linalg.inv(np.cov(roi_scores, rowvar=False))
    q = _q(0.95, N_TRAINING_ROIS, a)

    def expected(mean_scores: np.ndarray) -> float:
        d2 = (mean_scores - centre) @ cov_inv @ (mean_scores - centre)
        return q * model.metrics["rmsee"] * np.sqrt(1 + 1 / r + d2 / (r - 1))

    assignments = project.workspace.target_assignments
    summary = roi_prediction_statistics(prediction, target_assignments=assignments)
    for row in summary.rois:
        pixels = prediction.pixels[row.roi_id]
        assert pixels.scores is not None
        np.testing.assert_allclose(row.interval, expected(pixels.scores.mean(axis=0)), rtol=1e-5)

    # Glare in t09: three pixels far outside the model; removing them changes the interval.
    roi_id = project.roi("t09").roi_id
    pixels = prediction.pixels[roi_id]
    assert pixels.scores is not None
    scores, dmodx = pixels.scores.copy(), pixels.dmodx.copy()
    scores[:3] += 50 * np.sqrt(np.diag(np.cov(roi_scores, rowvar=False)))
    dmodx[:3] = 50.0
    prediction.pixels[roi_id] = type(pixels)(y=pixels.y, dmodx=dmodx, t2=pixels.t2,
                                             scores=scores)
    glare = next(row for row in roi_prediction_statistics(
        prediction, target_assignments=assignments).rois if row.roi_id == roi_id)
    assert np.isnan(glare.interval), "the glare moves the mean beyond the ROI-level limit"
    cleaned = next(row for row in roi_prediction_statistics(
        prediction, target_assignments=assignments, exclude_dmodx=0.99).rois
        if row.roi_id == roi_id)
    np.testing.assert_allclose(cleaned.interval, expected(scores[3:].mean(axis=0)), rtol=1e-5)


def test_the_fit_reports_how_often_held_out_predictions_fall_inside_their_intervals(tray):
    project, _scan, _path = tray
    model = project.fit("API", n_components=2, cv=3)
    metrics, diagnostics = model.metrics, model.diagnostics
    roi_scores = np.asarray(diagnostics["roi_scores"], dtype=np.float64)
    r, a = roi_scores.shape
    centred = roi_scores - roi_scores.mean(axis=0)
    d2 = np.einsum("ij,jk,ik->i", centred, np.linalg.inv(np.cov(roi_scores, rowvar=False)),
                   centred)
    half = (_q(0.95, N_TRAINING_ROIS, a) * metrics["rmsee"]
            * np.sqrt(1 + 1 / r + d2 / (r - 1)))
    crit = a * (r - 1) / (r - a) * stats.f.ppf(0.99, a, r - a)
    error = np.abs(np.asarray(diagnostics["y_pred_roi"]) - np.asarray(diagnostics["y_true_roi"]))
    counted = d2 <= crit
    assert metrics["interval_coverage_roi_n"] == int(counted.sum())
    assert metrics["interval_coverage_roi"] == pytest.approx(
        np.mean(error[counted] <= half[counted]))

    inside = ((np.asarray(diagnostics["t2"]) <= diagnostics["t2_crit_099"])
              & (np.asarray(diagnostics["dmodx"]) <= diagnostics["dcrit_099"]))
    assert metrics["interval_coverage_pixel_n"] == int(inside.sum())
    # Pixel noise is normal here, so the held-out pixels fall inside about as often as stated.
    assert 0.85 <= metrics["interval_coverage_pixel"] <= 1.0
