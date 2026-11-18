"""Applicability of predictions: DModXPS and T²PS, prediction intervals and the wavelength warning (prediction-runtime contract)."""

from __future__ import annotations

import hashlib
import h5py
import numpy as np
import pytest
from scipy import stats
from enginir import PreprocessingRecipe, Project
from enginir.io import load_model_run_prediction_artifact, load_spectra
from enginir.prediction import (
    _diagnostics,
    _service,
    roi_prediction_statistics,
    wavelength_mismatch,
)
from enginir.project import transform_spectra_to_signal_domain
from tests._scans import (
    N_BANDS,
    TRAY_BANDS,
    TRAY_FRAMES,
    labelled_project,
    scan_file,
    stored_counts,
    warnings_during,
)


# -- Applicability diagnostics of predictions and in-memory results (prediction-runtime contract) ----


def _file_digest(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _independent(prediction_x: np.ndarray, blob: dict) -> tuple[np.ndarray, np.ndarray]:
    """T²PS and DModXPS from the stored projection data, written out plainly in float64."""
    z = (prediction_x - blob["x_mean"]) / blob["x_std"]
    t = z @ blob["rotations"]
    e = z - t @ blob["loadings"].T
    k, a = z.shape[1], t.shape[1]
    return (t ** 2 / blob["score_variances"]).sum(axis=1), \
        np.sqrt((e ** 2).sum(axis=1) / (k - a)) / float(blob["s0"])


@pytest.mark.parametrize("model_type", ["PLS", "PCR"])
def test_on_the_training_scan_the_statistics_are_the_training_ones(tmp_path, model_type):
    """T²PS equals the training T²; DModXPS is the training DModX without its factor v."""
    project, scan = labelled_project(tmp_path)
    model = project.fit("API", model=model_type, recipe=PreprocessingRecipe(snv=True),
                        n_components=2)
    prediction = model.predict(scan.path)
    diagnostics = model.diagnostics
    n, a = len(diagnostics["t2"]), model.ref.hyperparameters["n_components"]
    v = np.sqrt(n / (n - a - 1))
    # the ROIs tile the scan frame by frame, in the order of the map
    np.testing.assert_allclose(prediction.t2_map.reshape(-1), diagnostics["t2"], rtol=1e-4)
    np.testing.assert_allclose(prediction.dmodx_map.reshape(-1), diagnostics["dmodx"] / v,
                               rtol=1e-4)
    limits = prediction.diagnostics
    assert limits["t2_crit_099"] == pytest.approx(diagnostics["t2_crit_099"])
    assert limits["dcrit_099"] == pytest.approx(diagnostics["dcrit_099"])
    assert limits["n_pixels"] == n
    assert prediction.map.shape == prediction.dmodx_map.shape == prediction.t2_map.shape


def test_the_statistics_equal_their_formulas_and_an_odd_pixel_exceeds_dcrit(tmp_path):
    project, _scan_ref = labelled_project(tmp_path)
    model = project.fit("API", n_components=2)
    path = scan_file(tmp_path / "new.h5", offset=10.0)
    with h5py.File(path, "r+") as f:
        cube = f["spectra"][...]
        cube[3, 1, : N_BANDS // 2] += 400  # an odd pixel: a step the model never saw
        f["spectra"][...] = cube
    prediction = model.predict(path)

    from enginir.io import load_model_run_prediction_artifact, load_spectra
    from enginir.project import transform_spectra_to_signal_domain

    blob = load_model_run_prediction_artifact(project.path, model.ref.model_run_id)
    domain = project.workspace.signal_domains[model.ref.signal_domain_id]
    dataset, calibration, _, _ = project._resolve_dataset(
        path, dark=None, white=None, unit=domain.required_signal_unit)
    raw = load_spectra(path).X
    X = transform_spectra_to_signal_domain(
        raw.reshape(-1, N_BANDS), dataset=dataset, calibration=calibration,
        signal_domain=domain, spatial_shape=raw.shape[:2]).astype(np.float64)
    t2, distance = _independent(X, blob)
    np.testing.assert_allclose(prediction.t2_map.reshape(-1), t2, rtol=1e-5)
    np.testing.assert_allclose(prediction.dmodx_map.reshape(-1), distance, rtol=1e-5)

    dcrit = prediction.diagnostics["dcrit_099"]
    odd = prediction.dmodx_map[3, 1]
    assert odd > dcrit and odd == prediction.dmodx_map.max()
    summary = prediction.diagnostics
    assert summary["n_above_dcrit"] >= 1
    assert summary["share_above_dcrit"] == pytest.approx(
        summary["n_above_dcrit"] / summary["n_pixels"])


def test_chunked_diagnostics_equal_one_chunk(tmp_path, monkeypatch):
    project, scan = labelled_project(tmp_path)
    model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=2)
    whole = model.predict(scan.path)
    monkeypatch.setattr(_service, "DEFAULT_CHUNK_SIZE", 3)
    monkeypatch.setattr(_diagnostics, "_BLOCK_ROWS", 5)
    chunked = model.predict(scan.path)
    np.testing.assert_allclose(chunked.dmodx_map, whole.dmodx_map, rtol=1e-6)
    np.testing.assert_allclose(chunked.t2_map, whole.t2_map, rtol=1e-6)
    np.testing.assert_allclose(chunked.map, whole.map, rtol=1e-6)
    assert chunked.diagnostics == pytest.approx(whole.diagnostics, nan_ok=True)


def test_a_prediction_writes_nothing_to_the_project_and_logs_one_entry(tmp_path):
    project, scan = labelled_project(tmp_path)
    model = project.fit("API", n_components=2)
    new = scan_file(tmp_path / "new.h5", offset=5.0)
    before, digest = stored_counts(project), _file_digest(project.path)

    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        prediction = model.predict(new, checksum=True)
        model.predict(scan.path)
    finally:
        logger.remove(sink)
    assert stored_counts(project) == before
    assert _file_digest(project.path) == digest, "the project file is unchanged"
    assert new not in Project.open(project.path).datasets
    entries = [m for m in messages if "Prediction with model" in m]
    assert len(entries) == 2 and model.ref.model_run_id in entries[0]
    record = prediction.provenance["files"][0]
    assert record["sha256"] == _file_digest(new) and record["size"] > 0
    assert "sha256" not in model.predict(new).provenance["files"][0]


def test_a_stored_model_without_a_core_member_is_refused_at_open(tmp_path):
    # The executable model is core (project-schema contract): a project whose
    # model lacks coefficients or projection data does not open, whether the
    # member is a dataset or a short vector stored as an attribute.
    project, _scan = labelled_project(tmp_path)
    model = project.fit("API", n_components=2)
    location = f"model_runs/{model.ref.model_run_id}/artifact_refs/prediction_model"
    Project.open(project.path)
    for key in ("coef_", "score_variances"):
        copy = tmp_path / f"without_{key}.h5"
        copy.write_bytes(open(project.path, "rb").read())
        with h5py.File(copy, "r+") as f:
            blob = f[location]
            if key in blob:
                del blob[key]
            else:
                del blob.attrs[key]
        with pytest.raises(ValueError, match=f"missing required field.*{key}"):
            Project.open(copy)


def test_diagnostics_without_a_usable_s0_are_nan_and_not_counted():
    rng = np.random.default_rng(0)
    projection = _diagnostics.Projection(
        x_mean=np.zeros(5), x_std=np.ones(5), rotations=np.eye(5)[:, :2],
        loadings=np.eye(5)[:, :2], score_variances=np.ones(2), s0=0.0, t2_crit=1.0,
        dcrit=None)
    t2, distance = _diagnostics.chunk_diagnostics(rng.normal(size=(7, 5)).astype(np.float32),
                                                  projection)
    assert np.isnan(distance).all() and np.isfinite(t2).all()
    summary = _diagnostics.DiagnosticsSummary(t2_crit=1.0, dcrit=None)
    summary.add(np.zeros(7), t2, distance)
    assert summary.as_dict()["n_pixels"] == 0


def test_a_model_fitted_in_the_viewer_session_gets_the_maps_and_nothing_is_written(tmp_path):
    from enginir.io import load_project_store
    from enginir.viewer._app import ensure_app
    from enginir.viewer._state import ProjectState
    from tests._project_payloads import write_model_project, write_prediction_cube

    ensure_app()
    path = str(tmp_path / "session.h5")
    write_model_project(path)
    state = ProjectState()
    state.open_project(path)
    state.build_sample_set(target_id="target_moisture", preprocessing_recipe_id="recipe_none",
                           sample_set_id="sample_set_1")
    state.fit_model_run(sample_set_id="sample_set_1", signal_domain_id="signal_domain_1",
                        model_type="PLS", hyperparameters={"n_components": 1}, cv="logo",
                        model_run_id="model_1")
    assert "model_1" in state.model_registry  # the in-memory predictor of the session
    scan = str(tmp_path / "scan.h5")
    write_prediction_cube(scan, wavelengths=np.array([1000.0, 1100.0, 1200.0], np.float32))
    before = load_project_store(path)
    calibration = state.calibrations["cal_train"]
    from dataclasses import replace

    result = state.predict_dataset(model_run_id="model_1", data_path=scan,
                                   calibration=replace(calibration, calibration_id="cal_scan",
                                                       dataset_id=""))
    assert result.dmodx is not None and result.t2 is not None
    assert result.diagnostics is not None and result.diagnostics["n_pixels"] == result.dmodx.size
    after = load_project_store(path)
    assert {key: set(value) for key, value in after.items() if isinstance(value, dict)} == \
        {key: set(value) for key, value in before.items() if isinstance(value, dict)}


def test_envi_provenance_records_the_files_the_reader_uses(tmp_path):
    # With .img and .raw next to one header the reader takes .img; the record
    # (and its checksum) must name that file, also for a "cube.img.hdr" header.
    from enginir.io import load_spectra, save_envi

    cube = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    save_envi(str(tmp_path / "sample"), cube, np.linspace(1000, 1300, 4))
    (tmp_path / "sample.raw").write_bytes(b"not the data")
    assert np.array_equal(load_spectra(str(tmp_path / "sample.hdr")).X, cube)
    records = _diagnostics.file_records(str(tmp_path / "sample.hdr"), checksum=True)
    assert [r["path"] for r in records] == [str(tmp_path / "sample.hdr"),
                                            str(tmp_path / "sample.img")]
    assert records[1]["sha256"] == _file_digest(tmp_path / "sample.img")

    (tmp_path / "sample.hdr").rename(tmp_path / "sample.img.hdr")
    records = _diagnostics.file_records(str(tmp_path / "sample.img"))
    assert [r["path"] for r in records] == [str(tmp_path / "sample.img"),
                                            str(tmp_path / "sample.img.hdr")]


def test_a_replaced_registered_file_is_not_predicted(tmp_path, monkeypatch):
    # The file at a registered path must still be that dataset (attachment
    # contract, Mismatch validation); the stored shape and axis are not
    # trusted blindly (external review, 1.6.0).
    from enginir import _workflow
    from enginir.io import load_spectra, save_hdf5

    project, scan = labelled_project(tmp_path)
    model = project.fit("API", n_components=2)
    data = load_spectra(scan.path)
    cube, wavelengths = np.asarray(data.X), np.asarray(data.wavelengths)

    save_hdf5(scan.path, cube, wavelengths + 100.0)
    with pytest.raises(ValueError, match="Not predicted: .*wavelengths differ"):
        model.predict(scan.path)

    frames, height, bands = cube.shape
    swapped = cube.reshape(height, frames, bands)
    save_hdf5(scan.path, swapped, wavelengths)
    with pytest.raises(ValueError, match="Not predicted: .*shape"):
        model.predict(scan.path)
    # The runtime refuses it too, should a caller skip the check.
    monkeypatch.setattr(_workflow, "_header_mismatch", lambda *_args: None)
    with pytest.raises(ValueError, match="not predicted"):
        model.predict(scan.path)

    save_hdf5(scan.path, cube, wavelengths)
    assert model.predict(scan.path).map.shape == (frames, height)


# -- Prediction intervals by the leverage formula (prediction-runtime contract, Prediction intervals) ----


N_TRAINING_ROIS = 7


def _scores(project, model, path) -> np.ndarray:
    """Scores of every scan pixel, from the stored projection data (independent of prediction)."""
    blob = load_model_run_prediction_artifact(project.path, model.ref.model_run_id)
    dataset = project._dataset_ref(str(path))
    calibration = project.workspace.calibrations[
        project.workspace.active_calibration_ids[dataset.dataset_id]]
    domain = project.workspace.signal_domains[model.ref.signal_domain_id]
    raw = load_spectra(path).X
    X = transform_spectra_to_signal_domain(
        raw.reshape(-1, TRAY_BANDS), dataset=dataset, calibration=calibration,
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
    first, last = 2 * height, (2 + N_TRAINING_ROIS * TRAY_FRAMES) * height
    training = scores[first:last]  # the training ROIs tile these frames
    n, a = training.shape
    centred = training - training.mean(axis=0)
    hat = np.einsum("ij,jk,ik->i", scores, np.linalg.inv(centred.T @ centred), scores)
    r = N_TRAINING_ROIS
    leverage = 1 / r + hat * (n - 1) / (r - 1)
    expected = (_q(0.95, N_TRAINING_ROIS, a) * model.metrics["pixel_rmsee"]
                * np.sqrt(1 + leverage))

    half = prediction.interval_map(0.95)
    assert not hasattr(prediction, "uncertainty_map"), "renamed in 1.5 (terminology contract)"
    assert half is not None and half.shape == prediction.map.shape
    half = half.reshape(-1)
    inside = np.isfinite(half)
    assert inside[first:last].mean() > 0.9, "almost every training pixel is inside its model"
    np.testing.assert_allclose(half[inside], expected[inside], rtol=1e-5)
    wider = prediction.interval_map(0.99)
    assert wider is not None and np.all(wider.reshape(-1)[inside] > half[inside])
    with pytest.raises(ValueError, match="between 0 and 1"):
        prediction.interval_map(1.0)


def test_pixels_outside_the_model_have_no_interval(tray, tmp_path):
    project, _scan, path = tray
    model = project.fit("API", n_components=2, cv=3)
    odd = tmp_path / "odd.h5"
    odd.write_bytes(path.read_bytes())
    with h5py.File(odd, "r+") as f:
        cube = f["spectra"][...]
        cube[10, 2, : TRAY_BANDS // 2] += 600  # a step the model never saw
        f["spectra"][...] = cube
    prediction = model.predict(odd)
    half, dmodx, t2 = prediction.interval_map(), prediction.dmodx_map, prediction.t2_map
    assert half is not None and dmodx is not None and t2 is not None
    limits = prediction.diagnostics or {}
    assert dmodx[10, 2] > limits["dcrit_099"]
    assert np.isnan(half[10, 2])
    outside = (dmodx > limits["dcrit_099"]) | (t2 > limits["t2_crit_099"])
    assert np.array_equal(np.isnan(half), outside)


def test_the_mean_interval_of_a_prediction_roi_uses_the_calibration_roi_scores(tray):
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


# -- Prediction-time wavelength warning (G18: warn only, never a gate) -----------------


AXIS = np.linspace(1000, 1600, N_BANDS)


@pytest.mark.parametrize(("scan", "model", "tolerance", "warns"), [
    (AXIS + 2.0, AXIS, 0.0, True),
    (AXIS + 2.0, AXIS, 2.5, False),             # within the signal domain's tolerance
    (AXIS + 1e-7, AXIS, 0.0, False),            # float noise
    (AXIS, AXIS, 0.0, False),
    (np.arange(N_BANDS, dtype=float), AXIS, 0.0, False),   # band-index fallback: no axis
    (AXIS, np.arange(N_BANDS, dtype=float), 0.0, False),
    (np.array([]), AXIS, 0.0, False),
    (AXIS[:-1], AXIS, 0.0, False),              # band counts are checked elsewhere
])
def test_axes_are_compared_only_when_both_are_real(scan, model, tolerance, warns):
    assert (wavelength_mismatch(scan, model, tolerance) is not None) is warns


def test_a_shifted_scan_warns_and_predicts_exactly_as_before(tmp_path):
    project, scan = labelled_project(tmp_path)
    model = project.fit("API", n_components=2)
    same = model.predict(scan.path)
    shifted_path = scan_file(tmp_path / "shifted.h5", wavelengths=AXIS + 5.0)
    shifted, messages = warnings_during(lambda: model.predict(shifted_path))
    assert same.provenance["wavelength_warning"] is None
    warning = shifted.provenance["wavelength_warning"]
    assert warning is not None and "up to 5 nm" in warning
    assert sum("predicted anyway" in m for m in messages) == 1, "logged once"
    np.testing.assert_array_equal(shifted.map, same.map)
