"""The project file format ``enginir-1.5``: a strict core and tolerant display data.

``project-schema-contract_v3.md`` (Object identity rules §1, Core and display
data, Compatibility policy): the reader refuses a file whose core is
incomplete, before anything is shown; display data that is missing never
stops a project from opening. Every core field below is removed from a copy
of a complete project, one at a time, and the copy must be refused.
"""

from __future__ import annotations

import shutil

import h5py
import pytest

from enginir import PreprocessingRecipe, Project
from enginir.io import FILE_SCHEMA_VERSION, load_project_store
from tests.test_workflow import _labelled_project

#: Core fields of the format, as HDF5 paths: ``group@attribute`` or a
#: dataset/group path. Placeholders are filled with the ids of the project.
#: Phases 3c–3d of plans/release-1.5x.md add ``fit_spec`` and ``software``.
CORE_FIELDS = (
    "project_meta@project_id",
    "project_meta@FileSchemaVersion",
    "project_meta@FileCreator",
    "project_meta@FileWriter",
    "project_meta@created_at",
    "project_meta@updated_at",
    *(f"datasets/{{dataset}}@{name}" for name in (
        "source_path", "source_format", "source_modality", "cube_shape", "n_bands",
        "import_timestamp", "active_calibration_id")),
    "datasets/{dataset}/wavelengths",
    "datasets/{dataset}/loader_metadata",
    *(f"datasets/{{dataset}}/calibrations/{{calibration}}@{name}" for name in (
        "mode", "output_signal_unit", "created_at")),
    *(f"rois/{{roi}}@{name}" for name in (
        "dataset_id", "calibration_id", "name", "n_pixels", "created_at", "spatial_bounds",
        "source_file", "job")),
    "rois/{roi}/wavelengths",
    "rois/{roi}/spectra",
    "rois/{roi}/provenance",
    *(f"targets/{{target}}@{name}" for name in (
        "name", "unit", "value_type", "missing_value_policy", "created_at")),
    "target_assignments/{target}/{roi}@value",
    "target_assignments/{target}/{roi}@assigned_at",
    *(f"recipes/{{recipe}}@{name}" for name in (
        "filter", "sg_window", "sg_polyorder", "snv", "msc", "emsc", "detrend",
        "detrend_order", "sg_edges", "asls", "asls_lambda", "asls_p", "asls_iterations")),
    "recipes/{recipe}/bands_keep",
    "recipes/{recipe}/bands_exclude",
    *(f"signal_domains/{{signal_domain}}@{name}" for name in (
        "input_format", "source_modality", "required_signal_unit", "required_calibration_mode",
        "requires_dark_ref", "requires_white_ref", "calibrated_input_allowed", "n_bands",
        "wavelength_tolerance_nm", "band_mapping_policy", "output_dtype", "created_at")),
    "signal_domains/{signal_domain}/wavelengths",
    *(f"sample_sets/{{sample_set}}@{name}" for name in (
        "target_id", "observation_unit", "target_level", "grouping_unit", "validation_unit",
        "weighting_policy", "reporting_level", "roi_aggregation", "preprocessing_recipe_id",
        "created_at")),
    "sample_sets/{sample_set}/roi_ids",
    "sample_sets/{sample_set}/dataset_ids",
    *(f"model_runs/{{model}}@{name}" for name in (
        "sample_set_id", "signal_domain_id", "model_type", "created_at")),
    "model_runs/{model}/hyperparameters",
    "model_runs/{model}/artifact_refs",
    "model_runs/{model}/artifact_refs/prediction_model",
    "model_runs/{model}/artifact_refs/prediction_model@model_type",
    "model_runs/{model}/artifact_refs/prediction_model@n_components",
)

#: Display data: removing it never stops the project from opening.
DISPLAY_FIELDS = (
    "model_runs/{model}/metrics",
    "model_runs/{model}/cv_spec",
)


@pytest.fixture(scope="module")
def qapp():
    from enginir.viewer._app import ensure_app

    return ensure_app()


@pytest.fixture(scope="module")
def complete_project(tmp_path_factory):
    """One complete project (dataset, ROIs, target values, a fitted model) and its ids."""
    tmp_path = tmp_path_factory.mktemp("format")
    project, _scan = _labelled_project(tmp_path)
    model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=2)
    ws = project.workspace
    roi = next(iter(ws.rois))
    ids = {
        "dataset": next(iter(ws.datasets)),
        "calibration": next(iter(ws.calibrations)),
        "roi": roi,
        "target": next(iter(ws.targets)),
        "recipe": next(iter(ws.recipes)),
        "signal_domain": next(iter(ws.signal_domains)),
        "sample_set": next(iter(ws.sample_sets)),
        "model": model.ref.model_run_id,
    }
    return project.path, ids


def _without(source, tmp_path, field, ids):
    """A copy of the project with ``field`` removed."""
    copy = tmp_path / "copy.h5"
    shutil.copy(source, copy)
    path = field.format(**ids)
    with h5py.File(copy, "r+") as f:
        if "@" in path:
            group, attribute = path.split("@")
            assert attribute in f[group].attrs, f"{path} is not in the complete project"
            del f[group].attrs[attribute]
        elif path in f:
            del f[path]
        else:  # a small array is stored as an attribute of its group
            group, name = path.rsplit("/", 1)
            assert name in f[group].attrs, f"{path} is not in the complete project"
            del f[group].attrs[name]
    return copy


def test_the_complete_project_opens(complete_project):
    path, _ids = complete_project
    meta = load_project_store(path)["project_meta"]
    assert meta["FileSchemaVersion"] == FILE_SCHEMA_VERSION
    Project.open(path)


@pytest.mark.parametrize("field", CORE_FIELDS)
def test_a_project_without_a_core_field_is_refused(complete_project, tmp_path, field):
    path, ids = complete_project
    copy = _without(path, tmp_path, field, ids)

    with pytest.raises(Exception):  # noqa: B017 - ValueError or a relationship error
        Project.open(copy)


@pytest.mark.parametrize("field", DISPLAY_FIELDS)
def test_a_project_without_display_data_opens(complete_project, tmp_path, field):
    path, ids = complete_project
    copy = _without(path, tmp_path, field, ids)

    model = Project.open(copy).models
    name = next(iter(model))
    assert isinstance(model[name].metrics, dict)


def test_every_write_records_the_writer_and_the_time(complete_project, tmp_path):
    path, _ids = complete_project
    copy = tmp_path / "copy.h5"
    shutil.copy(path, copy)
    with h5py.File(copy, "r+") as f:
        f["project_meta"].attrs.update(FileWriter="engiNIR 0.0.1",
                                       updated_at="2000-01-01T00:00:00.000000+00:00")

    Project.open(copy).add_target("moisture", unit="%")

    meta = load_project_store(copy)["project_meta"]
    assert meta["FileWriter"].startswith("engiNIR ") and meta["FileWriter"] != "engiNIR 0.0.1"
    assert meta["updated_at"] > "2000-01-01T00:00:00.000000+00:00"
    assert meta["FileCreator"] == load_project_store(path)["project_meta"]["FileCreator"]


def test_reading_a_project_changes_nothing(complete_project):
    path, _ids = complete_project
    before = dict(load_project_store(path)["project_meta"])
    Project.open(path)
    assert load_project_store(path)["project_meta"] == before


def test_the_inspector_shows_the_file_format_and_its_writers(complete_project, qapp):
    from enginir.viewer._inspector import ProjectInspector
    from enginir.viewer._state import ProjectState

    path, _ids = complete_project
    state = ProjectState()
    state.open_project(str(path))
    inspector = ProjectInspector(state)
    inspector.show_default()

    text = inspector.body_label.text()
    assert "File format: enginir-1.5" in text
    assert "Created by: engiNIR " in text and "Last written by: engiNIR " in text
