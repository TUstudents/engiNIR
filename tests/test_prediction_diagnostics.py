"""Applicability diagnostics of predictions and in-memory results (prediction-runtime contract)."""

from __future__ import annotations

import hashlib

import h5py
import numpy as np
import pytest

from enginir import PreprocessingRecipe, Project
from enginir.prediction import _diagnostics, _service
from tests.test_workflow import N_BANDS, _counts, _labelled_project, _scan


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
    project, scan = _labelled_project(tmp_path)
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
    project, _scan_ref = _labelled_project(tmp_path)
    model = project.fit("API", n_components=2)
    path = _scan(tmp_path / "new.h5", offset=10.0)
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
    project, scan = _labelled_project(tmp_path)
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
    project, scan = _labelled_project(tmp_path)
    model = project.fit("API", n_components=2)
    new = _scan(tmp_path / "new.h5", offset=5.0)
    before, digest = _counts(project), _file_digest(project.path)

    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        prediction = model.predict(new, checksum=True)
        model.predict(scan.path)
    finally:
        logger.remove(sink)
    assert _counts(project) == before
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
    project, _scan = _labelled_project(tmp_path)
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
    from tests.test_prediction_runs import _create_phase6_model_project, _write_prediction_cube

    ensure_app()
    path = str(tmp_path / "session.h5")
    _create_phase6_model_project(path)
    state = ProjectState()
    state.open_project(path)
    state.build_sample_set(target_id="target_moisture", preprocessing_recipe_id="recipe_none",
                           sample_set_id="sample_set_1")
    state.fit_model_run(sample_set_id="sample_set_1", signal_domain_id="signal_domain_1",
                        model_type="PLS", hyperparameters={"n_components": 1}, cv="logo",
                        model_run_id="model_1")
    assert "model_1" in state.model_registry  # the in-memory predictor of the session
    scan = str(tmp_path / "scan.h5")
    _write_prediction_cube(scan, wavelengths=np.array([1000.0, 1100.0, 1200.0], np.float32))
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

    project, scan = _labelled_project(tmp_path)
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
