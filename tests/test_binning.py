"""Spatial pixel binning of image cubes (plan ``plans/pixel-binning.md``, phase 1)."""

from __future__ import annotations

import numpy as np
import pytest

from enginir.io import (
    bin_blocks,
    bin_frame_chunks,
    iter_eiff_frame_chunks,
    iter_envi_frame_chunks,
    iter_hdf5_frame_chunks,
    load_spectra,
    save_envi,
    save_hdf5,
)
from tests._eiff import eiff_bytes

N_FRAMES, HEIGHT, N_BANDS = 14, 11, 5
WAVELENGTHS = np.linspace(1000.0, 1600.0, N_BANDS)


def _oracle(cube: np.ndarray, k: int) -> np.ndarray:
    """Mean of each k x k block, by hand: floats as float32, counts rounded half up."""
    frames, rows = cube.shape[0] // k, cube.shape[1] // k
    out = np.empty((frames, rows, cube.shape[2]), dtype=np.float64)
    for f in range(frames):
        for r in range(rows):
            out[f, r] = cube[f * k:(f + 1) * k, r * k:(r + 1) * k].astype(np.float64).mean(
                axis=(0, 1))
    if cube.dtype.kind == "f":
        return out.astype(np.float32)
    return np.floor(out + 0.5).astype(cube.dtype)


@pytest.fixture()
def counts():
    return np.random.default_rng(3).integers(0, 65536, (N_FRAMES, HEIGHT, N_BANDS),
                                             dtype=np.uint16)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_a_block_is_the_mean_of_its_counts_and_stays_raw_counts(counts, k):
    """uint16 stays uint16 (no overflow at 65535, no float that would read as corrected
    data), the mean is rounded half up, and frames and rows short of a block are dropped."""
    binned = bin_blocks(counts, k)
    assert binned.dtype == np.uint16
    assert binned.shape == (N_FRAMES // k, HEIGHT // k, N_BANDS)
    np.testing.assert_array_equal(binned, _oracle(counts, k))
    assert bin_blocks(np.full((2, 2, 1), 65535, np.uint16), 2)[0, 0, 0] == 65535
    assert bin_blocks(np.array([[[1], [2]], [[1], [2]]], np.uint16), 2)[0, 0, 0] == 2  # 1.5 up


def test_corrected_data_are_averaged_as_float32_and_one_is_no_binning(counts):
    reflectance = counts.astype(np.float32) / 65535
    binned = bin_blocks(reflectance, 2)
    assert binned.dtype == np.float32
    np.testing.assert_allclose(binned, _oracle(reflectance, 2), rtol=1e-6)
    assert bin_blocks(counts, 1) is counts


def test_binning_is_refused_where_it_has_no_meaning():
    with pytest.raises(ValueError, match="binning must be one of"):
        bin_blocks(np.zeros((4, 4, 2)), 5)
    with pytest.raises(ValueError, match="image cube"):
        bin_blocks(np.zeros((4, 2)), 2)
    with pytest.raises(ValueError, match="no 4 x 4 block"):
        bin_blocks(np.zeros((8, 3, 2)), 4)


def _files(tmp_path, counts):
    hdf5 = tmp_path / "cube.h5"
    save_hdf5(str(hdf5), counts, WAVELENGTHS)
    eiff = tmp_path / "cube.eiff"
    eiff.write_bytes(eiff_bytes(N_BANDS, HEIGHT, list(counts), WAVELENGTHS))
    envi = tmp_path / "cube.hdr"
    save_envi(envi, counts, WAVELENGTHS)
    return {"hdf5": hdf5, "eiff": eiff, "envi": envi}


def test_every_cube_format_reads_binned_like_the_binned_cube(tmp_path, counts):
    """Whole cube, a slice of binned frames, and the header alone."""
    expected = bin_blocks(counts, 2)
    for name, path in _files(tmp_path, counts).items():
        data = load_spectra(path, binning=2)
        np.testing.assert_array_equal(data.X, expected, err_msg=name)
        assert data.cube_shape == expected.shape and data.meta["binning"] == 2, name
        np.testing.assert_allclose(data.wavelengths, WAVELENGTHS, rtol=1e-6)
        if name == "envi":
            continue  # ENVI reads whole files only
        part = load_spectra(path, binning=2, frames=slice(2, 5)).X
        np.testing.assert_array_equal(part, expected[2:5], err_msg=name)
        header = load_spectra(path, binning=2, header_only=True)
        assert header.cube_shape == expected.shape, name


def test_binned_frames_are_counted_from_the_start_in_order(tmp_path, counts):
    path = _files(tmp_path, counts)["hdf5"]
    with pytest.raises(ValueError, match="step 1"):
        load_spectra(path, binning=2, frames=slice(0, 4, 2))
    with pytest.raises(ValueError, match="Only image cubes"):
        csv = tmp_path / "spectra.csv"
        csv.write_text("sample,1000,1001\na,1,2\nb,3,4\n", encoding="utf-8")
        load_spectra(csv, binning=2)


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 14])
def test_chunked_reading_bins_like_one_shot(tmp_path, counts, chunk_size):
    """Blocks of any size are joined so no k-frame block is split across chunks."""
    files = _files(tmp_path, counts)
    expected = bin_blocks(counts, 3)
    for name, chunks in (
            ("hdf5", iter_hdf5_frame_chunks(files["hdf5"], chunk_size=chunk_size)),
            ("eiff", iter_eiff_frame_chunks(files["eiff"], chunk_size=chunk_size)),
            ("envi", iter_envi_frame_chunks(files["envi"], chunk_size=chunk_size))):
        got = np.zeros_like(expected)
        offsets = []
        for offset, block in bin_frame_chunks(chunks, 3):
            got[offset:offset + block.shape[0]] = block
            offsets.append(offset)
        np.testing.assert_array_equal(got, expected, err_msg=name)
        assert offsets == sorted(offsets) and offsets[0] == 0, name
    with pytest.raises(ValueError, match="in order"):
        list(bin_frame_chunks([(0, counts[:2]), (5, counts[5:7])], 2))


# -- A binned dataset (phase 2) --------------------------------------------------------


def _tray(tmp_path):
    """24 frames x 8 rows x 6 bands: dark frames 0-1, white frames 22-23, counts between;
    the same scan binned 2 x 2 with NumPy, as a file of its own."""
    rng = np.random.default_rng(11)
    cube = rng.integers(1500, 2500, (24, 8, 6)).astype(np.uint16)
    cube[:2] = rng.integers(90, 110, (2, 8, 6))
    cube[-2:] = rng.integers(3000, 3200, (2, 8, 6))
    raw, prebinned = tmp_path / "raw.h5", tmp_path / "prebinned.h5"
    save_hdf5(str(raw), cube, WAVELENGTHS[:1].repeat(6) + np.arange(6.0))
    save_hdf5(str(prebinned), bin_blocks(cube, 2), WAVELENGTHS[:1].repeat(6) + np.arange(6.0))
    return raw, prebinned


def _pixels(project, roi):
    from enginir.io.project_store import load_roi_spectra_batch

    key = roi.spectra_storage_key
    return load_roi_spectra_batch(project.path, [key])[key]


def test_a_binned_dataset_is_the_file_binned_with_numpy(tmp_path):
    """The acceptance test: the dataset of a file added with binning=2 has the pixels of
    the same scan pre-binned with NumPy and added as it is, and its references are made
    from the stored frames (dark 0-1, white 22-23), binned over the rows only."""
    from enginir import Project

    raw, prebinned = _tray(tmp_path)
    project = Project.create(tmp_path / "p.h5")
    binned = project.add_dataset(raw, binning=2, dark_frames=(0, 1), white_frames=(22, 23))
    plain = project.add_dataset(prebinned, dark_frames=(0, 0), white_frames=(11, 11))
    assert binned.record.binning == 2 and binned.record.cube_shape == (12, 4, 6)
    assert plain.record.binning == 1

    a = binned.add_roi("a", frames=(2, 5), rows=(0, 3))
    b = plain.add_roi("b", frames=(2, 5), rows=(0, 3))
    np.testing.assert_array_equal(_pixels(project, a), _pixels(project, b))
    assert _pixels(project, a).dtype == np.uint16, "still raw counts"
    # The binned dataset's references are means of the stored counts; the pre-binned
    # file's are means of counts already rounded: they differ by rounding only.
    for name in ("dark_ref_data", "white_ref_data"):
        np.testing.assert_allclose(getattr(binned.reference_correction, name),
                                   getattr(plain.reference_correction, name), atol=0.5)

    reopened = Project.open(project.path)
    assert reopened.dataset(str(raw)).record.binning == 2, "stored with the dataset"


def test_a_file_has_one_binning_in_a_project(tmp_path):
    from enginir import Project

    raw, prebinned = _tray(tmp_path)
    project = Project.create(tmp_path / "p.h5")
    project.add_dataset(raw, binning=2, dark_frames=(0, 1), white_frames=(22, 23))
    with pytest.raises(ValueError, match="registered with binning 2"):
        project.add_dataset(raw, binning=3)
    again = project.add_dataset(raw, dark_frames=(0, 1), white_frames=(22, 23))
    assert again.record.binning == 2, "no binning given: the registered one"
    with pytest.raises(ValueError, match="binned 2, this file 1"):
        project.add_dataset(prebinned, references_from=raw)


def test_an_unbinned_dataset_keeps_the_content_it_had(tmp_path):
    """Binning is stored only when there is one: a dataset of 1.7 re-saves as a no-op."""
    import h5py

    from enginir import Project

    raw, _ = _tray(tmp_path)
    project = Project.create(tmp_path / "p.h5")
    dataset = project.add_dataset(raw)
    with h5py.File(project.path, "r") as f:
        assert "binning" not in f[f"datasets/{dataset.record.dataset_id}"].attrs
    project.save_dataset(dataset.record)  # identical: allowed


def test_a_moved_binned_file_is_found_and_another_scan_is_not(tmp_path):
    """Locate compares the binned shape and pixels with the registration."""
    import shutil

    from enginir import Project

    raw, prebinned = _tray(tmp_path)
    project = Project.create(tmp_path / "p.h5")
    dataset = project.add_dataset(raw, binning=2, dark_frames=(0, 1), white_frames=(22, 23))
    dataset.add_roi("a", frames=(2, 5), rows=(0, 3))
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(raw, moved / "raw.h5")
    with pytest.raises(ValueError):
        Project.open(project.path).dataset(str(raw)).locate(prebinned)  # another shape
    located = Project.open(project.path).dataset(str(raw)).locate(moved / "raw.h5")
    assert located.record.binning == 2


# -- Models and prediction (phase 3) ---------------------------------------------------


def _fitted(tmp_path, name, path, binning=None):
    """A model on eight ROIs of a tray (detector levels, no references)."""
    from enginir import Project

    project = Project.create(tmp_path / f"{name}.h5")
    project.add_target("API", unit="%")
    dataset = project.add_dataset(path, binning=binning, unit="Absorbance")
    for i in range(8):
        dataset.add_roi(f"r{i}", frames=(1 + i, 1 + i), rows=(0, 3), values={"API": 2.0 + i})
    return project, project.fit("API", n_components=1, cv=4)


def test_a_model_of_a_binned_dataset_fits_and_predicts_like_the_numpy_binned_file(tmp_path):
    """Phase 3 of the acceptance test: same metrics, and a new scan, not registered, is
    binned like the model before it is predicted (it equals the pre-binned scan's map)."""
    raw, prebinned = _tray(tmp_path)
    _, binned_model = _fitted(tmp_path, "binned", raw, binning=2)
    _, plain_model = _fitted(tmp_path, "plain", prebinned)
    assert binned_model.metrics["rmsecv"] == pytest.approx(plain_model.metrics["rmsecv"],
                                                            rel=1e-6)

    scan = np.random.default_rng(5).integers(1500, 2500, (10, 8, 6)).astype(np.uint16)
    new_raw, new_prebinned = tmp_path / "new_raw.h5", tmp_path / "new_prebinned.h5"
    axis = WAVELENGTHS[:1].repeat(6) + np.arange(6.0)
    save_hdf5(str(new_raw), scan, axis)
    save_hdf5(str(new_prebinned), bin_blocks(scan, 2), axis)
    binned_map = binned_model.predict(new_raw).map
    assert binned_map.shape == (5, 4), "10 frames x 8 rows binned 2 x 2"
    np.testing.assert_allclose(binned_map, plain_model.predict(new_prebinned).map,
                               rtol=1e-5, atol=1e-6)


def test_a_registered_file_of_another_binning_is_not_predicted(tmp_path):
    """D1: its stored references have the other row count."""
    raw, prebinned = _tray(tmp_path)
    project, model = _fitted(tmp_path, "p", raw, binning=2)
    project.add_dataset(prebinned)  # registered without binning
    with pytest.raises(ValueError, match="binned 1 x 1, but the model was fitted on 2 x 2"):
        model.predict(prebinned)


def test_a_calibration_set_of_two_binnings_is_refused(tmp_path):
    from enginir import Project

    raw, prebinned = _tray(tmp_path)
    project = Project.create(tmp_path / "p.h5")
    project.add_target("API", unit="%")
    for path, binning, prefix in ((raw, 2, "b"), (prebinned, None, "p")):
        dataset = project.add_dataset(path, binning=binning, unit="Absorbance")
        for i in range(4):
            dataset.add_roi(f"{prefix}{i}", frames=(1 + i, 1 + i), rows=(0, 3),
                            values={"API": 2.0 + i})
    with pytest.raises(ValueError):
        project.fit("API", n_components=1, cv=4)


# -- The viewer (phase 4) --------------------------------------------------------------


def test_add_dataset_bins_the_file_and_the_viewer_shows_the_binned_image(qapp, tmp_path):
    from enginir import Project
    from enginir.viewer import open_project
    from enginir.viewer._data_files import AddDataFileDialog
    from enginir._workflow import read_header

    raw, _ = _tray(tmp_path)
    project = Project.create(tmp_path / "p.h5")
    dialog = AddDataFileDialog(str(raw), read_header(str(raw)), {}, {}, binning=2)
    assert dialog._binning is not None and dialog._binning.currentData() == 2, \
        "opened from Predict: the model's binning"
    request = dialog.build_request()
    dataset = project.add_dataset(request.path, **request.add_dataset_kwargs())
    assert dataset.record.binning == 2

    window = open_project(project.path)
    try:
        window.show_dataset(dataset.record.dataset_id)
        window.wait_for_data_file()
        assert window.state.cube.shape == (12, 4, 6)
        assert "Binned 2 × 2" in window._status_label.text()
    finally:
        window.close()
