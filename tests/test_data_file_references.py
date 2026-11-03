"""Data files and their references in a project (dataset-calibration contract, headless)."""

from __future__ import annotations

import numpy as np
import pytest

from enginir import Project
from enginir.io import save_hdf5

N_FRAMES, HEIGHT, N_BANDS = 20, 6, 8


@pytest.fixture()
def files(tmp_path):
    """A tray with dark frames 0-2 and a white tile in frames 17-19, and separate references."""
    rng = np.random.default_rng(3)
    wavelengths = np.linspace(1000.0, 1700.0, N_BANDS)
    tray = rng.integers(1000, 2000, (N_FRAMES, HEIGHT, N_BANDS)).astype(np.uint16)
    tray[:3] = rng.integers(90, 110, (3, HEIGHT, N_BANDS))
    tray[17:] = rng.integers(3400, 3600, (3, HEIGHT, N_BANDS))
    paths = {name: tmp_path / f"{name}.h5" for name in ("tray", "tray2", "dark", "white")}
    save_hdf5(str(paths["tray"]), tray, wavelengths)
    save_hdf5(str(paths["tray2"]), rng.integers(1000, 2000, tray.shape).astype(np.uint16),
              wavelengths)
    save_hdf5(str(paths["dark"]), rng.integers(90, 110, (4, HEIGHT, N_BANDS)).astype(np.uint16),
              wavelengths)
    save_hdf5(str(paths["white"]),
              rng.integers(3400, 3600, (4, HEIGHT, N_BANDS)).astype(np.uint16), wavelengths)
    return paths, tray


def test_references_from_frames_of_the_same_file(tmp_path, files):
    paths, tray = files
    project = Project.create(tmp_path / "p.h5")
    calibration = project.add_dataset(paths["tray"], dark_frames=(0, 2),
                                      white_frames=(17, 19)).calibration
    assert calibration.mode == "dark_white_reference"
    np.testing.assert_allclose(calibration.dark_ref_data, tray[0:3].mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(calibration.white_ref_data, tray[17:20].mean(axis=0), rtol=1e-6)
    tray_path = str(paths["tray"].resolve())
    assert calibration.dark_ref_source == f"{tray_path} frames 0-2"
    assert calibration.white_ref_source == f"{tray_path} frames 17-19"


def test_a_reference_file_and_frames_combine(tmp_path, files):
    paths, tray = files
    project = Project.create(tmp_path / "p.h5")
    calibration = project.add_dataset(paths["tray"], dark=paths["dark"],
                                      white_frames=(17, 19)).calibration
    assert calibration.dark_ref_source == str(paths["dark"].resolve())
    np.testing.assert_allclose(calibration.white_ref_data, tray[17:20].mean(axis=0), rtol=1e-6)


def test_a_session_shares_one_reference_set(tmp_path, files):
    paths, _tray = files
    project = Project.create(tmp_path / "p.h5")
    first = project.add_dataset(paths["tray"], dark=paths["dark"], white=paths["white"])
    second = project.add_dataset(paths["tray2"], references_from=paths["tray"])
    shared, own = first.calibration, second.calibration
    assert own.dataset_id == second.ref.dataset_id != shared.dataset_id  # dataset-local copy
    assert own.calibration_id != shared.calibration_id
    np.testing.assert_array_equal(own.dark_ref_data, shared.dark_ref_data)
    np.testing.assert_array_equal(own.white_ref_data, shared.white_ref_data)
    assert (own.dark_ref_source, own.white_ref_source) == (shared.dark_ref_source,
                                                           shared.white_ref_source)


def test_invalid_reference_requests_write_nothing(tmp_path, files):
    paths, _tray = files
    project = Project.create(tmp_path / "p.h5")
    project.add_dataset(paths["tray"])  # default levels: no reference spectra
    with pytest.raises(ValueError, match="no dark and white reference spectra"):
        project.add_dataset(paths["tray2"], references_from=paths["tray"])
    with pytest.raises(ValueError, match="do not combine"):
        project.add_dataset(paths["tray2"], references_from=paths["tray"], dark=paths["dark"])
    with pytest.raises(ValueError, match="outside the file's 20 frames"):
        project.add_dataset(paths["tray2"], dark_frames=(0, 2), white_frames=(18, 25))
    with pytest.raises(ValueError, match="as a file or as frames"):
        project.add_dataset(paths["tray2"], dark=paths["dark"], dark_frames=(0, 2),
                            white=paths["white"])
    reopened = Project.open(project.path)
    assert list(reopened.datasets) == [str(paths["tray"].resolve())]


def test_a_new_calibration_applies_to_later_rois_only(tmp_path, files):
    paths, _tray = files
    project = Project.create(tmp_path / "p.h5")
    tray = project.add_dataset(paths["tray"])
    tray.add_roi("early", frames=(4, 6), rows=(0, 2))
    tray = project.add_dataset(paths["tray"], dark_frames=(0, 2), white_frames=(17, 19))
    tray.add_roi("late", frames=(8, 10), rows=(0, 2))
    early, late = project.roi("early"), project.roi("late")
    assert late.calibration_id == tray.calibration.calibration_id
    assert early.calibration_id != late.calibration_id
    assert project.workspace.calibrations[early.calibration_id].black_level is not None
