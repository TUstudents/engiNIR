"""Moved data files: `Dataset.locate` accepts only the same pixels (project attachment contract)."""

from __future__ import annotations

import shutil
from dataclasses import replace

import numpy as np
import pytest

from enginir import Project
from enginir.io import save_hdf5
from enginir.io._hdf5_groups import values_equal

WAVELENGTHS = np.linspace(1000.0, 1700.0, 6)


@pytest.fixture()
def moved(tmp_path):
    """tray.h5 with two ROIs, then moved to moved/tray.h5; the old path no longer exists."""
    rng = np.random.default_rng(21)
    cube = rng.integers(500, 3000, (20, 8, 6)).astype(np.uint16)
    path = tmp_path / "tray.h5"
    save_hdf5(str(path), cube, WAVELENGTHS)
    project = Project.create(tmp_path / "p.h5")
    project.add_target("API", unit="%")
    tray = project.add_dataset(path)
    tray.add_roi("A1", frames=(2, 4), rows=(1, 3), values={"API": 1.0})
    tray.add_roi("A2", frames=(12, 15), rows=(4, 6), values={"API": 2.0})
    new_path = tmp_path / "moved" / "tray.h5"
    new_path.parent.mkdir()
    shutil.move(path, new_path)
    return project, path, new_path, cube


def _source_path(project):
    (ref,) = Project.open(project.path).workspace.datasets.values()
    return ref.source_path


def test_a_moved_file_is_accepted_and_only_its_path_changes(moved):
    project, old_path, new_path, _cube = moved
    before = Project.open(project.path).workspace
    project.dataset(old_path).locate(new_path)

    after = Project.open(project.path).workspace
    (ref,) = after.datasets.values()
    assert ref.source_path == str(new_path.resolve())
    (old_ref,) = before.datasets.values()
    assert ref.relative_path == "moved/tray.h5", "the relative path follows the new location"
    assert values_equal(replace(old_ref, source_path=ref.source_path,
                                relative_path=ref.relative_path).to_dict(), ref.to_dict())
    assert after.rois.keys() == before.rois.keys()
    # New ROIs read the file at its new location.
    Project.open(project.path).dataset(new_path).add_roi("A3", frames=(6, 7), rows=(0, 1))


def _write(path, cube, wavelengths=WAVELENGTHS):
    path.unlink(missing_ok=True)
    save_hdf5(str(path), cube, wavelengths)


@pytest.mark.parametrize(("change", "message"), [
    ("other_pixels", "the pixels of ROI 'A1' differ"),
    ("one_pixel", "the pixels of ROI 'A2' differ"),
    ("shape", "it has shape"),
    ("wavelengths", "its wavelengths differ"),
])
def test_another_file_is_refused_and_nothing_changes(moved, change, message):
    project, old_path, new_path, cube = moved
    candidate = new_path.parent / "candidate.h5"
    if change == "other_pixels":
        _write(candidate, np.random.default_rng(5).integers(500, 3000, cube.shape)
               .astype(np.uint16))
    elif change == "one_pixel":
        changed = cube.copy()
        changed[15, 6, 5] += 1  # the last pixel of A2
        _write(candidate, changed)
    elif change == "shape":
        _write(candidate, cube[:-1])
    else:
        _write(candidate, cube, WAVELENGTHS + 0.5)
    with pytest.raises(ValueError, match=message):
        project.dataset(old_path).locate(candidate)
    assert _source_path(project) == str(old_path.resolve())


def test_a_file_of_another_data_file_is_refused(moved, tmp_path):
    project, old_path, new_path, cube = moved
    other = tmp_path / "other.h5"
    save_hdf5(str(other), cube, WAVELENGTHS)
    project.add_dataset(other)
    with pytest.raises(ValueError, match="already registered"):
        project.dataset(old_path).locate(other)


def test_precalibrated_pixels_are_compared_as_stored(tmp_path):
    cube = np.random.default_rng(3).random((10, 4, 6))  # float64 reflectance
    path = tmp_path / "calibrated.h5"
    save_hdf5(str(path), cube, WAVELENGTHS)
    project = Project.create(tmp_path / "p.h5")
    project.add_dataset(path, unit="Reflectance").add_roi("F1", frames=(1, 3), rows=(0, 2))
    new_path = tmp_path / "renamed.h5"
    path.rename(new_path)
    project.dataset(path).locate(new_path)
    assert _source_path(project) == str(new_path.resolve())
