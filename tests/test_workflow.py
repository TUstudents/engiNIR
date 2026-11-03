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


def test_the_facade_refuses_names_it_creates_itself(tmp_path):
    from enginir._workflow import roi_grid

    project, scan = _labelled_project(tmp_path)
    with pytest.raises(ValueError, match="ROI called 's0' already exists"):
        scan.add_roi("s0", frames=(0, 0), rows=(0, 0))
    with pytest.raises(ValueError, match="ROI called 's1' already exists"):
        scan.add_rois(roi_grid(n_rows=1, n_columns=2, size=(1, 1), name_pattern="s{n}"))
    with pytest.raises(ValueError, match="target called 'API' already exists"):
        project.add_target("API", unit="%")

    first = project.fit("API", n_components=1)
    second = project.fit("API", n_components=1)
    assert (first.name, second.name) == ("PLS 1 LV, raw", "PLS 1 LV, raw (2)")
    with pytest.raises(ValueError, match="model called 'PLS 1 LV, raw' already exists"):
        project.fit("API", n_components=1, name="PLS 1 LV, raw")


def _with_repeated_names(tmp_path):
    """The labelled project plus a second scan whose ROIs repeat the names s0..s5,
    stored through the typed, id-based layer as the viewer does."""
    from enginir._workflow import make_roi, read_frames

    project, scan = _labelled_project(tmp_path)
    other = project.add_dataset(_scan(tmp_path / "other.h5", offset=40.0))
    pixels = read_frames(other.path, 0, N_SAMPLES * FRAMES_PER_SAMPLE - 1)
    built = [make_roi(other.ref, other.calibration, name=f"s{i}",
                      frames=(FRAMES_PER_SAMPLE * i, FRAMES_PER_SAMPLE * (i + 1) - 1),
                      rows=(0, HEIGHT - 1),
                      pixels=pixels[FRAMES_PER_SAMPLE * i:FRAMES_PER_SAMPLE * (i + 1)])
             for i in range(N_SAMPLES)]
    project.save_rois(built)
    return project, scan, other, [roi for roi, _spectra, _refs in built]


def test_names_are_labels_and_may_repeat(tmp_path):
    project, scan, other, repeated = _with_repeated_names(tmp_path)
    reopened = Project.open(project.path)
    assert len(reopened.workspace.rois) == 2 * N_SAMPLES  # both s0..s5 sets are stored

    # A shared name is never resolved to one of them, and name-keyed views refuse.
    with pytest.raises(KeyError, match="2 ROIs share the name 's0'"):
        reopened.roi("s0")
    with pytest.raises(KeyError, match="ROI names repeat"):
        _ = reopened.rois
    assert set(reopened.dataset(other.path).rois) == {f"s{i}" for i in range(N_SAMPLES)}

    # Ids and objects are the reproducible references.
    s0_other = reopened.roi(roi_id=repeated[0].roi_id)
    assert s0_other.dataset_id == other.ref.dataset_id
    api = reopened.target(target_id=project.target("API").target_id)
    for i, roi in enumerate(repeated):
        reopened.set_value(reopened.roi(roi_id=roi.roi_id), api, float(CONTENT[i]))
    model = reopened.fit(api, n_components=1)
    sample_set = reopened.workspace.sample_sets[model.ref.sample_set_id]
    assert len(sample_set.roi_ids) == 2 * N_SAMPLES
    assert reopened.model(model_run_id=model.ref.model_run_id).name == model.name
    assert reopened.dataset(dataset_id=other.ref.dataset_id).path == other.path


def test_id_lookups_are_strict(tmp_path):
    project, scan = _labelled_project(tmp_path)
    with pytest.raises(TypeError, match="not both or neither"):
        project.roi()
    with pytest.raises(TypeError, match="not both or neither"):
        project.roi("s0", roi_id=project.roi("s0").roi_id)
    with pytest.raises(KeyError, match="No ROI with id 's0'"):
        project.roi(roi_id="s0")  # a name is never taken for an id
    (tmp_path / "b").mkdir()
    elsewhere, _scan_b = _labelled_project(tmp_path / "b")
    with pytest.raises(KeyError, match="No ROI with id"):
        project.set_value(elsewhere.roi("s0"), "API", 1.0)  # an ROI of another project


def test_typed_writes_check_ids_not_names(tmp_path):
    from enginir._workflow import make_roi, read_frames

    project, scan = _labelled_project(tmp_path)
    pixels = read_frames(scan.path, 0, 0)
    roi, spectra, refs = make_roi(scan.ref, scan.calibration, name="s0", frames=(0, 0),
                                  rows=(0, 0), pixels=pixels)
    with pytest.raises(ValueError, match="ROI ids already exist or repeat"):
        project.save_rois([(roi, spectra, refs), (roi, spectra, refs)])
    project.save_rois([(roi, spectra, refs)])  # "s0" exists already: a label, accepted
    assert sum(r.name == "s0" for r in project.workspace.rois.values()) == 2


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


def test_cross_validation_does_not_depend_on_the_random_roi_ids(tmp_path, monkeypatch):
    """Repeating an analysis in a new project (new ids) reproduces its CV metrics.

    With 8 ROIs, ``cv="auto"`` is a 7-fold GroupKFold, so two ROIs share a fold;
    which two must follow the recorded ROI order, not the sort order of the ids.
    """
    content = np.linspace(1, 9, 8)
    rng = np.random.default_rng(0)
    peak = np.exp(-((np.arange(N_BANDS) - N_BANDS / 2) / 2) ** 2)
    cube = (2500 - 100 * np.repeat(content, 2)[:, None, None] * peak
            + rng.normal(0, 5, (16, HEIGHT, N_BANDS))).astype(np.uint16)
    scan_path = str(tmp_path / "scan.h5")
    save_hdf5(scan_path, cube, np.linspace(1000, 1600, N_BANDS))

    def fit(folder, roi_rank):
        project = Project.create(tmp_path / f"{folder}.h5")
        scan = project.add_dataset(scan_path)
        project.add_target("API", unit="%")
        ranks = iter(roi_rank)
        with monkeypatch.context() as patch:  # the id of each ROI, in creation order
            patch.setattr("enginir._workflow.new_id", lambda: f"{next(ranks):02d}-{folder}")
            for i, value in enumerate(content):
                scan.add_roi(f"s{i}", frames=(2 * i, 2 * i + 1), rows=(0, HEIGHT - 1),
                             values={"API": value})
        model = project.fit("API", n_components=2)
        names = {roi.roi_id: name for name, roi in project.rois.items()}
        cv = dict(zip((names[i] for i in model.diagnostics["roi_ids"]),
                      model.diagnostics["y_pred_roi"], strict=True))
        return model, cv

    # The ROI ids sort in creation order in the first project and rotated by one in
    # the second, so sorting by id would put different ROIs into the shared fold.
    first, first_cv = fit("a", range(8))
    second, second_cv = fit("b", [1, 2, 3, 4, 5, 6, 7, 0])

    assert first.ref.cv_spec["type"] == "GroupKFold"
    assert second_cv == pytest.approx(first_cv)
    assert second.metrics["rmsecv"] == pytest.approx(first.metrics["rmsecv"])


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


def _warnings_during(action):
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        result = action()
    finally:
        logger.remove(sink_id)
    return result, messages


def test_refit_reproduces_a_model_from_its_stored_sample_set(tmp_path):
    project, scan = _labelled_project(tmp_path)
    model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=2, cv=3)
    # A value added later changes what fit() would use, not what refit() uses.
    scan.add_roi("late", frames=(0, 1), rows=(0, 1), values={"API": 9.0})

    refitted, warnings = _warnings_during(model.refit)
    assert warnings == []
    assert refitted.ref.model_run_id != model.ref.model_run_id
    assert refitted.name == "PLS 2 LV, SNV (2)"
    for key in ("sample_set_id", "signal_domain_id", "model_type", "hyperparameters"):
        assert getattr(refitted.ref, key) == getattr(model.ref, key)
    assert refitted.ref.cv_spec["requested"] == 3
    for key in ("rmsecv", "r2", "bias", "pixel_rmsecv", "n_rois"):
        assert refitted.metrics[key] == pytest.approx(model.metrics[key], rel=1e-9)
    assert project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=2,
                       cv=3).metrics["n_rois"] == N_SAMPLES + 1

    reopened = Project.open(project.path)
    again = reopened.model(model_run_id=model.ref.model_run_id).refit(name="check")
    assert again.metrics["rmsecv"] == pytest.approx(model.metrics["rmsecv"], rel=1e-9)


def test_refit_reports_changed_values_and_refuses_deleted_rois(tmp_path):
    project, scan = _labelled_project(tmp_path)
    model = project.fit("API", n_components=1, cv="logo")
    project.set_value("s2", "API", 7.5)
    refitted, warnings = _warnings_during(model.refit)
    assert any("1 reference value(s) changed" in w for w in warnings)
    assert any("differs in" in w and "rmsecv" in w for w in warnings)
    assert refitted.metrics["rmsecv"] != pytest.approx(model.metrics["rmsecv"])
    with pytest.raises(ValueError, match="model called 'PLS 1 LV, raw' already exists"):
        model.refit(name="PLS 1 LV, raw")

    project.remove_roi(project.roi("s0").roi_id)
    with pytest.raises(ValueError, match="Cannot refit the model: sample set .* lists deleted"):
        model.refit()
