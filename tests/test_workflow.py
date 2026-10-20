"""The Project facade rules of the headless workflow contract."""

from __future__ import annotations

import numpy as np
import pytest

from enginir import PreprocessingRecipe, Project
from enginir.io import load_project_store, save_hdf5

N_SAMPLES, FRAMES_PER_SAMPLE, HEIGHT, N_BANDS = 6, 3, 4, 12
CONTENT = np.linspace(1, 6, N_SAMPLES)


def _scan(path, *, offset=0.0, wavelengths=None, dtype=np.uint16):
    """A scan of N_SAMPLES samples whose absorption grows with CONTENT."""
    rng = np.random.default_rng(1)
    peak = np.exp(-((np.arange(N_BANDS) - N_BANDS / 2) / 2) ** 2)
    content = np.repeat(CONTENT, FRAMES_PER_SAMPLE)[:, None, None]
    cube = 2500 - 150 * content * peak + offset + rng.normal(0, 3, (len(content), HEIGHT, N_BANDS))
    if np.dtype(dtype).kind == "f":
        cube = cube / 3500
    if wavelengths is None:
        wavelengths = np.linspace(1000, 1600, N_BANDS)
    save_hdf5(path, cube.astype(dtype), wavelengths)
    return str(path)


def _labelled_project(tmp_path, **dataset_kwargs):
    project = Project.create(tmp_path / "project.h5")
    scan = project.add_dataset(_scan(tmp_path / "scan.h5"), **dataset_kwargs)
    project.add_target("API", unit="%")
    for i, value in enumerate(CONTENT):
        first = FRAMES_PER_SAMPLE * i
        scan.add_roi(f"s{i}", frames=(first, first + FRAMES_PER_SAMPLE - 1),
                     rows=(0, HEIGHT - 1), values={"API": value})
    return project, scan


def _counts(project):
    stored = load_project_store(project.path)
    return {key: len(value) for key, value in stored.items() if key != "project_meta"}


def test_workflow_fits_predicts_and_reopens_identically(tmp_path):
    project, scan = _labelled_project(tmp_path)
    model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=2)
    prediction = model.predict(scan.path)

    assert model.name == "PLS 2 LV, SNV"
    assert model.metrics["r2"] > 0.9
    per_sample = prediction.map.reshape(N_SAMPLES, -1).mean(axis=1)
    np.testing.assert_allclose(per_sample, CONTENT, atol=0.5)
    assert set(model.diagnostics) >= {"roi_ids", "y_true_roi", "y_pred_roi"}

    reopened = Project.open(project.path)
    assert list(reopened.models) == [model.name]
    assert reopened.model(model.name).metrics == model.metrics
    assert set(reopened.rois) == set(project.rois)
    np.testing.assert_array_equal(next(iter(reopened.predictions.values())).map,
                                  prediction.map)


def test_names_are_unique_and_ambiguous_lookups_fail(tmp_path):
    from enginir.io.project_store import save_target_definition

    project, scan = _labelled_project(tmp_path)
    with pytest.raises(ValueError, match="ROI called 's0' already exists"):
        scan.add_roi("s0", frames=(0, 0), rows=(0, 0))
    with pytest.raises(ValueError, match="target called 'API' already exists"):
        project.add_target("API", unit="%")

    first = project.fit("API", n_components=1)
    second = project.fit("API", n_components=1)
    assert (first.name, second.name) == ("PLS 1 LV, raw", "PLS 1 LV, raw (2)")
    with pytest.raises(ValueError, match="model called 'PLS 1 LV, raw' already exists"):
        project.fit("API", n_components=1, name="PLS 1 LV, raw")

    # A file written by another tool may repeat a name: opening works, the lookup refuses.
    save_target_definition(project.path, "other_id", {
        "name": "API", "unit": "%", "value_type": "float", "missing_value_policy": "allowed",
        "created_at": "2026-01-01T00:00:00"})
    with pytest.raises(KeyError, match="2 targets share the name 'API'"):
        Project.open(project.path).target("API")


def test_a_registered_file_is_reused_and_a_changed_one_refused(tmp_path):
    project = Project.create(tmp_path / "project.h5")
    path = _scan(tmp_path / "scan.h5")
    project.add_dataset(path)
    project.add_dataset(path)
    assert _counts(project)["datasets"] == 1
    assert len(project.workspace.calibrations) == 1  # same content, not duplicated

    save_hdf5(path, np.zeros((2, HEIGHT, N_BANDS), np.uint16), np.linspace(1000, 1600, N_BANDS))
    with pytest.raises(ValueError, match="has shape"):
        project.add_dataset(path)


@pytest.mark.parametrize(
    ("dtype", "kwargs", "mode", "levels"),
    [(np.uint16, {}, "dark_white_reference", (128.0, 3500.0)),
     (np.uint16, {"unit": "DN"}, "raw_dn", (None, None)),
     (np.float32, {}, "precalibrated_import", (None, None))],
)
def test_calibration_follows_the_data_and_unit(tmp_path, dtype, kwargs, mode, levels):
    project = Project.create(tmp_path / "project.h5")
    calibration = project.add_dataset(_scan(tmp_path / "scan.h5", dtype=dtype),
                                      **kwargs).calibration
    assert calibration.mode == mode
    assert (calibration.black_level, calibration.white_level) == levels


def test_reference_spectra_calibrate_and_travel_with_each_roi(tmp_path):
    from enginir.io import load_roi_spectra_batch, iter_roi_calibration_refs

    dark = np.full((HEIGHT, N_BANDS), 100.0)
    white = np.full((2, HEIGHT, N_BANDS), 3000.0)  # a cube: averaged over frames
    project, _ = _labelled_project(tmp_path, dark=dark, white=white)

    calibration = next(iter(project.datasets.values())).calibration
    np.testing.assert_array_equal(calibration.white_ref_data, white.mean(axis=0))
    roi = project.roi("s1")
    dark_pixels, _ = dict(iter_roi_calibration_refs(project.path, [roi.roi_id]))[roi.roi_id]
    assert dark_pixels.shape == load_roi_spectra_batch(
        project.path, [roi.spectra_storage_key])[roi.spectra_storage_key].shape
    with pytest.raises(ValueError, match="both dark and white"):
        project.add_dataset(_scan(tmp_path / "other.h5"), dark=dark)


def test_refitting_reuses_sample_set_recipe_and_signal_domain(tmp_path):
    project, _ = _labelled_project(tmp_path)
    project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=1)
    project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=2)
    counts = _counts(project)
    assert (counts["recipes"], counts["sample_sets"], counts["signal_domains"],
            counts["model_runs"]) == (1, 1, 1, 2)


def test_one_model_refuses_rois_recorded_in_different_signals(tmp_path):
    project, _ = _labelled_project(tmp_path)
    other = project.add_dataset(_scan(tmp_path / "dn.h5"), unit="DN")
    other.add_roi("dn sample", frames=(0, 2), rows=(0, 3), values={"API": 1.0})
    before = _counts(project)

    with pytest.raises(ValueError, match="different signal"):
        project.fit("API", n_components=1)
    assert _counts(project) == before


def test_failed_calls_write_nothing(tmp_path):
    project, scan = _labelled_project(tmp_path)
    before = _counts(project)

    with pytest.raises(KeyError, match="No target called 'water'"):
        scan.add_roi("new", frames=(0, 0), rows=(0, 0), values={"water": 1.0})
    with pytest.raises(ValueError, match="frames must be an inclusive range"):
        scan.add_roi("new", frames=(0, 99), rows=(0, 0))
    with pytest.raises(ValueError):
        project.fit("API", n_components=99)
    assert _counts(project) == before


def test_prediction_is_refused_before_an_incompatible_file_is_registered(tmp_path):
    project, _ = _labelled_project(tmp_path)
    model = project.fit("API", n_components=1)
    before = _counts(project)
    path = tmp_path / "fewer_bands.h5"
    save_hdf5(path, np.full((3, HEIGHT, N_BANDS - 1), 2000, np.uint16),
              np.linspace(1000, 1600, N_BANDS - 1))

    with pytest.raises(ValueError, match="n_bands"):
        model.predict(path)
    assert _counts(project) == before
