"""Renaming and retiring models, removing datasets, compacting a project (menu-polish plan,
phases 4 and 5; project-schema contract §5)."""

from __future__ import annotations

import hashlib

import h5py
import numpy as np
import pytest

from enginir import Project
from enginir.io import load_hdf5
from tests._scans import every_stored_object_project, scan_file


def _contents(path) -> dict:
    """Every attribute and array of an HDF5 file, by path, as text (NaN equals NaN)."""
    found: dict = {}
    with h5py.File(path, "r") as f:
        found["/"] = repr({key: np.asarray(value).tolist() for key, value in f.attrs.items()})

        def read(name, obj):
            found[f"{name}@attrs"] = repr({key: np.asarray(value).tolist()
                                           for key, value in obj.attrs.items()})
            if isinstance(obj, h5py.Dataset):
                found[name] = repr((obj.dtype.str, obj.compression,
                                    np.asarray(obj[()]).tolist()))

        f.visititems(read)
    return found


def _digest(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# -- Rename Model ----------------------------------------------------------------------


def test_a_model_is_renamed_and_an_empty_or_taken_name_is_refused(tray):
    project, _scan, _path = tray
    first = project.fit("API", n_components=2, cv=3)
    second = project.fit("API", n_components=3, cv=3)
    first.rename("  Tablets v2  ")
    assert Project.open(project.path).model(model_run_id=first.record.model_run_id).name \
        == "Tablets v2"
    before = _digest(project.path)
    for name in ("", "   ", "Tablets v2"):
        with pytest.raises(ValueError, match="non-empty|already exists"):
            second.rename(name)
    assert _digest(project.path) == before, "a refused rename writes nothing"
    second.rename(second.name)  # its own name is not taken by another model


# -- Remove Dataset --------------------------------------------------------------------


def test_a_dataset_without_rois_is_removed_and_its_file_can_be_added_again(tray, tmp_path):
    project, _scan, _path = tray
    extra = scan_file(tmp_path / "extra.h5")
    file_before = _digest(extra)
    dataset = project.add_dataset(extra)
    corrections = {cid for cid, c in project.workspace.reference_corrections.items()
                   if c.dataset_id == dataset.record.dataset_id}
    assert corrections
    project.remove_dataset(dataset.record.dataset_id)

    reopened = Project.open(project.path)
    assert dataset.record.dataset_id not in reopened.workspace.datasets
    assert not corrections & set(reopened.workspace.reference_corrections)
    assert _digest(extra) == file_before, "the data file is never touched"
    again = reopened.add_dataset(extra, binning=2)  # another binning: a new dataset
    assert again.record.binning == 2


def test_a_dataset_something_refers_to_is_not_removed_and_nothing_is_written(tmp_path):
    project = Project.open(every_stored_object_project(tmp_path))
    scan = project.dataset(tmp_path / "scan.h5")
    before = _digest(project.path)
    with pytest.raises(ValueError, match="its ROIs s0, s1"):
        project.remove_dataset(scan.record.dataset_id)
    for name in [f"s{i}" for i in range(6)]:
        project.remove_roi(project.roi(name).roi_id)
    stored = _digest(project.path)
    with pytest.raises(ValueError, match=r"calibration set\(s\) .*stored prediction\(s\) "
                                         r"stored-prediction"):
        project.remove_dataset(scan.record.dataset_id)
    assert before != stored and _digest(project.path) == stored, "a refusal writes nothing"


# -- Retire Model ----------------------------------------------------------------------


def test_retiring_removes_only_the_per_pixel_diagnostics_and_stops_predicting(tray):
    project, _scan, path = tray
    retired = project.fit("API", n_components=2, cv=3)
    active = project.fit("API", n_components=3, cv=3)
    runs = {model.record.model_run_id: model for model in (retired, active)}
    before = _contents(project.path)
    retired.retire("superseded")

    after = _contents(project.path)
    rid, aid = retired.record.model_run_id, active.record.model_run_id
    per_pixel = {f"model_runs/{rid}/artifact_refs/diagnostics/{name}{suffix}"
                 for name in ("x_scores_", "t2", "dmodx") for suffix in ("", "@attrs")}
    assert per_pixel <= set(before) and not per_pixel & set(after)
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    assert changed == per_pixel | {f"model_runs/{rid}@attrs", "project_meta@attrs"}, changed
    assert not any(key.startswith(f"model_runs/{aid}") and before[key] != after.get(key)
                   for key in before), "the other model is unchanged"

    reopened = Project.open(project.path)
    assert list(reopened.models) == [active.name]
    assert list(reopened.retired_models) == [retired.name]
    model = reopened.retired_models[retired.name]
    assert model.record.retired_reason == "superseded" and model.record.retired_at
    assert model.record.retired_enginir
    for action in (lambda: model.predict(path), lambda: model.predict_rois()):
        with pytest.raises(ValueError, match="is retired"):
            action()
    with pytest.raises(ValueError, match="already retired"):
        model.retire()
    refit = model.refit()
    assert not refit.retired and "t2" in refit.diagnostics
    assert refit.metrics["rmsecv"] == pytest.approx(runs[rid].metrics["rmsecv"], rel=1e-9)


# -- Compact Project -------------------------------------------------------------------


def test_compacting_shrinks_the_file_and_keeps_every_stored_value(tmp_path):
    path = every_stored_object_project(tmp_path)
    project = Project.open(path)
    next(iter(project.models.values())).retire()
    kept = _contents(path)
    before, after = project.compact()
    assert after < before
    assert _contents(path) == kept
    reopened = Project.open(path)
    assert set(reopened.workspace.predictions) == {"stored-prediction"}
    assert load_hdf5(tmp_path / "scan.h5").X.shape[0] > 0


def test_a_failed_compaction_leaves_the_project_as_it_was(tmp_path, monkeypatch):
    path = every_stored_object_project(tmp_path)
    before = _digest(path)
    monkeypatch.setattr(h5py.Group, "copy", lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        Project.open(path).compact()
    assert _digest(path) == before
    assert not list(tmp_path.glob("*.compacting"))
