"""Stored ROIs read back without data files: patches and spectra as the fit sees them."""

from __future__ import annotations

import numpy as np
import pytest

from enginir import PreprocessingRecipe, Project
from enginir.io import save_hdf5
from enginir.io.project_store import load_roi_band_images
from enginir.modeling import roi_spectrum_summaries


@pytest.fixture()
def project(tmp_path):
    rng = np.random.default_rng(8)
    cube = rng.integers(1000, 3000, (24, 10, 7)).astype(np.uint16)
    cube[:2] = 100       # dark frames
    cube[-2:] = 3600     # white tile
    path = tmp_path / "tray.h5"
    save_hdf5(str(path), cube, np.linspace(1000.0, 1600.0, 7))
    raw_path = tmp_path / "raw.h5"
    save_hdf5(str(raw_path), cube, np.linspace(1000.0, 1600.0, 7))
    project = Project.create(tmp_path / "p.h5")
    tray = project.add_dataset(path, dark_frames=(0, 1), white_frames=(22, 23))
    tray.add_roi("T1", frames=(3, 6), rows=(1, 4))
    tray.add_roi("T2", frames=(10, 14), rows=(5, 8))
    project.add_dataset(raw_path, unit="DN").add_roi("R1", frames=(3, 5), rows=(0, 1))
    return project, cube


def test_band_images_rebuild_each_roi_patch_from_the_project(project):
    project_obj, cube = project
    rois = {roi.roi_id: roi for roi in project_obj.workspace.rois.values()}
    images = load_roi_band_images(project_obj.path, rois, [5, 1, 3])  # any band order
    t2 = project_obj.roi("T2")
    np.testing.assert_array_equal(images[t2.roi_id], cube[10:15, 5:9][:, :, [5, 1, 3]])


def _summaries(project_obj, unit, recipe=None):
    ws = project_obj.workspace
    return roi_spectrum_summaries(project_obj.path, list(ws.rois.values()), datasets=ws.datasets,
                                  calibrations=ws.calibrations, signal_unit=unit, recipe=recipe)


def test_means_follow_the_fit_conversion_and_skip_incompatible_rois(project):
    project_obj, cube = project
    summaries, skipped = _summaries(project_obj, "Absorbance")
    t1 = project_obj.roi("T1")
    dark, white = cube[:2].mean(axis=0), cube[-2:].mean(axis=0)
    pixels = cube[3:7, 1:5].astype(float)
    reflectance = np.clip((pixels - dark[1:5]) / (white[1:5] - dark[1:5]), 1e-6, 1.0)
    expected = -np.log10(reflectance).reshape(-1, cube.shape[2])
    np.testing.assert_allclose(summaries[t1.roi_id].mean, expected.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(summaries[t1.roi_id].median, np.median(expected, axis=0),
                               rtol=1e-5)
    r1 = project_obj.roi("R1").roi_id
    assert r1 in skipped and r1 not in summaries  # raw counts cannot give absorbance


def test_the_fit_and_the_summaries_convert_identically(project):
    """The same function converts the ROIs for the fit and for the Project ROIs tab."""
    from enginir.modeling import _service

    project_obj, _cube = project
    calls: list[str] = []
    original = _service.roi_to_signal_domain

    def spy(*args, **kwargs):
        calls.append(kwargs["roi"].name)
        return original(*args, **kwargs)

    _service.roi_to_signal_domain = spy
    try:
        project_obj.add_target("API", unit="%")
        project_obj.set_value("T1", "API", 1.0)
        project_obj.set_value("T2", "API", 2.0)
        project_obj.fit("API", n_components=1, cv="logo")
        _summaries(project_obj, "Absorbance")
    finally:
        _service.roi_to_signal_domain = original
    # Each training ROI went through it once for the fit and once for the summaries.
    assert calls.count("T1") == 2 and calls.count("T2") == 2


def test_preprocessing_is_applied_per_pixel_before_averaging(project):
    project_obj, _cube = project
    plain, _ = _summaries(project_obj, "Reflectance")
    snv, _ = _summaries(project_obj, "Reflectance", PreprocessingRecipe(snv=True))
    t1 = project_obj.roi("T1").roi_id
    # The mean of SNV-corrected pixels has mean ~0 but is not the SNV of the mean spectrum.
    assert abs(float(snv[t1].mean.mean())) < 1e-5
    mean = plain[t1].mean
    snv_of_mean = (mean - mean.mean()) / mean.std(ddof=1)
    assert not np.allclose(snv[t1].mean, snv_of_mean, atol=1e-4)
    assert np.all(snv[t1].p05 <= snv[t1].median) and np.all(snv[t1].median <= snv[t1].p95)
