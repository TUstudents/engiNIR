"""Data files of a project: labels, references, moved files and paths across platforms (attachment and dataset-calibration contracts)."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import PurePath, PureWindowsPath
import shutil
import time
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
from enginir import Project
from enginir._workflow import data_file_labels
from enginir.io import save_hdf5
from enginir.io._hdf5_groups import values_equal
from enginir.project import _data_paths
from enginir.project._data_paths import (
    RELATIVE,
    STORED,
    data_file_candidates,
    drive_mapped,
    moved_path,
    native_path,
    relative_data_path,
    stored_path,
    wsl_mount_root,
)
from tests._scans import framed_scan


# -- Data files and their references in a project (dataset-calibration contract, headless) ----


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


def test_detector_levels_can_be_given_and_are_checked(tmp_path, files):
    paths, tray = files
    project = Project.create(tmp_path / "p.h5")
    calibration = project.add_dataset(paths["tray"], black_level=100.0,
                                      white_level=3000.0).calibration
    assert (calibration.mode, calibration.black_level, calibration.white_level) \
        == ("dark_white_reference", 100.0, 3000.0)
    roi = project.dataset(paths["tray"]).add_roi("R", frames=(5, 6), rows=(0, 1))
    from enginir.modeling import roi_spectrum_summaries  # the fit's conversion

    ws = project.workspace
    summaries, _ = roi_spectrum_summaries(project.path, [roi], datasets=ws.datasets,
                                          calibrations=ws.calibrations, signal_unit="Reflectance")
    expected = (tray[5:7, 0:2].reshape(-1, N_BANDS) - 100.0) / (3000.0 - 100.0)
    np.testing.assert_allclose(summaries[roi.roi_id].mean,
                               np.clip(expected, 1e-6, 1.0).mean(axis=0), rtol=1e-5)

    before = Project.open(project.path).workspace
    with pytest.raises(ValueError, match="must be above the black level"):
        project.add_dataset(paths["tray2"], black_level=3000.0, white_level=3000.0)
    with pytest.raises(ValueError, match="do not combine"):
        project.add_dataset(paths["tray2"], dark_frames=(0, 2), white_frames=(17, 19),
                            black_level=100.0)
    after = Project.open(project.path).workspace
    assert after.datasets.keys() == before.datasets.keys()


# -- Data file labels: file names, with the folder that tells repeated names apart (viewer ids) ----


def _refs(*paths: str) -> dict[str, SimpleNamespace]:
    return {f"d{i}": SimpleNamespace(source_path=os.path.join(os.sep, *p.split("/")))
            for i, p in enumerate(paths)}


def test_unique_file_names_are_kept_and_repeated_ones_get_their_folder():
    labels = data_file_labels(_refs("z/run1/stream_hsi.h5", "z/run2/stream_hsi.h5",
                                    "z/run2/tray03.h5"))
    assert labels == {"d0": "stream_hsi.h5 (run1)", "d1": "stream_hsi.h5 (run2)",
                      "d2": "tray03.h5"}


def test_more_folders_are_used_when_parent_folders_repeat_too():
    labels = data_file_labels(_refs("z/day1/raw/scan.h5", "z/day2/raw/scan.h5"))
    assert labels == {"d0": "scan.h5 (day1/raw)", "d1": "scan.h5 (day2/raw)"}


def test_long_folders_are_shortened_unless_that_makes_labels_equal():
    a = "API 10% - nozzle 0.6 mm__20261102_160818-black-alu-foil"
    b = "API 10% - nozzle 0.6 mm__20261102_170818-white-alu-foil"
    labels = data_file_labels(_refs(f"z/{a}/stream_hsi.h5", f"z/{b}/stream_hsi.h5"))
    folder = labels["d0"].removeprefix("stream_hsi.h5 (").removesuffix(")")
    assert len(folder) == 40 and "…" in folder  # shortened in the middle, ends kept
    assert folder.startswith("API 10% - nozzle") and folder.endswith("black-alu-foil")
    assert len(set(labels.values())) == 2
    same_ends = ["x" * 25 + mid + "y" * 25 for mid in ("1", "2")]  # differ only in the middle
    labels = data_file_labels(_refs(f"z/{same_ends[0]}/s.h5", f"z/{same_ends[1]}/s.h5"))
    assert labels == {"d0": f"s.h5 ({same_ends[0]})", "d1": f"s.h5 ({same_ends[1]})"}


def test_the_viewer_tells_repeated_file_names_apart(qapp, tmp_path):
    from enginir import Project
    from enginir.io import save_hdf5
    from enginir.viewer import open_viewer

    paths = []
    for folder in ("run1", "run2"):
        (tmp_path / folder).mkdir()
        paths.append(tmp_path / folder / "stream_hsi.h5")
        save_hdf5(str(paths[-1]), np.full((8, 4, 6), 1000, np.uint16),
                  np.linspace(1000.0, 1500.0, 6))
    project = Project.create(tmp_path / "p.h5")
    project.add_target("API", unit="%")
    for path, name in zip(paths, ("A1", "B1")):
        project.add_dataset(path).add_roi(name, frames=(0, 1), rows=(0, 1), values={"API": 1.0})
    window = open_viewer(paths[1], project=project.path)
    try:
        combo = window._data_file_combo
        items = {combo.itemText(i): combo.itemData(i, 3) for i in range(combo.count())
                 if isinstance(combo.itemData(i), str) and combo.itemData(i) != "__add_data_file__"}
        assert items == {"stream_hsi.h5 (run1)": str(paths[0].resolve()),
                         "stream_hsi.h5 (run2)": str(paths[1].resolve())}  # with tooltips
        window.show_roi_table()
        model = window._roi_table.model
        window._roi_table.show_data_file(None)
        # The table cell is the file name only; its tooltip tells the files apart.
        files = sorted(str(model.data(model.index(r, 0))) for r in range(model.rowCount()))
        assert files == ["stream_hsi.h5", "stream_hsi.h5"]
        tips = sorted(str(model.data(model.index(r, 0), 3)) for r in range(model.rowCount()))
        assert tips == ["stream_hsi.h5 (run1)", "stream_hsi.h5 (run2)"]
        readiness, _ok = window._modeling_view.fit_readiness(project.target("API").target_id)
        assert "stream_hsi.h5 (run1): 1 of 1 ROIs have a value" in readiness
        assert "stream_hsi.h5 (run2): 1 of 1 ROIs have a value" in readiness
        window._project_rois.rebuild()  # the tab lays out its blocks (hidden tabs wait)
        headers = window._project_rois.mosaic.header_names()
        assert sorted(headers) == ["stream_hsi.h5 (run1)", "stream_hsi.h5 (run2)"]
    finally:
        window.close()


# -- Moved data files: `Dataset.locate` accepts only the same pixels (project attachment contract) ----


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


# -- Data files across platforms (attachment contract, Missing and moved files) --------


def test_a_stored_path_is_read_by_its_own_syntax_on_any_platform():
    windows = stored_path(r"Z:\NonKF\API 10%\stream_hsi.h5")
    assert isinstance(windows, PureWindowsPath)
    assert windows.parts == ("Z:\\", "NonKF", "API 10%", "stream_hsi.h5")
    assert isinstance(stored_path(r"\\172.16.0.61\z\x.h5"), PureWindowsPath)
    assert native_path(windows, windows=False) is None, "not a Linux path, never cwd/Z:/…"
    assert native_path(stored_path("/mnt/z/x.h5"), windows=False) == "/mnt/z/x.h5"
    assert native_path(stored_path("/mnt/z/x.h5"), windows=True) is None


@pytest.mark.parametrize(("stored", "windows", "root", "expected"), [
    (r"Z:\NonKF\a b\x.h5", False, "/mnt/", "/mnt/z/NonKF/a b/x.h5"),
    (r"Z:\NonKF\x.h5", False, "/", "/z/NonKF/x.h5"),          # wsl.conf root = /
    ("/mnt/z/NonKF/a b/x.h5", True, None, r"Z:\NonKF\a b\x.h5"),
    ("/z/NonKF/x.h5", True, None, r"Z:\NonKF\x.h5"),
    ("/mnt/data/x.h5", True, None, None),                     # not a drive letter
    ("/home/user/x.h5", True, None, None),
    (r"\\172.16.0.61\z\x.h5", False, "/mnt/", None),          # UNC: a configured mapping
    ("/mnt/z/x.h5", False, "/mnt/", None),                    # already this platform's
])
def test_the_drive_mapping_both_ways(stored, windows, root, expected):
    assert drive_mapped(stored_path(stored), windows=windows, wsl_root=root) == expected


def test_the_wsl_mount_root_is_read_from_wsl_conf(tmp_path):
    conf = tmp_path / "wsl.conf"
    conf.write_text("[automount]\nroot = /\noptions = metadata\n")
    assert wsl_mount_root(str(conf)) == "/"
    conf.write_text("[network]\nhostname = pc\n")
    wsl_mount_root.cache_clear()
    assert wsl_mount_root(str(conf)) == "/mnt/"
    assert wsl_mount_root(str(tmp_path / "missing.conf")) == "/mnt/"


def test_the_relative_path_and_the_order_of_the_candidates():
    assert relative_data_path("/mnt/z/p/demo.h5", "/mnt/z/p/lay 1/x.h5") == "lay 1/x.h5"
    assert relative_data_path("/mnt/z/p/demo.h5", "/mnt/z/data/x.h5") == "../data/x.h5"
    candidates = data_file_candidates("/home/u/copy/demo.h5", r"Z:\p\lay1\x.h5", "lay1/x.h5",
                                      windows=False, wsl_root="/mnt/")
    next_to = os.path.join(os.path.abspath("/home/u/copy"), "lay1", "x.h5")  # C:\home\… on Windows
    assert [(c.path, c.how) for c in candidates] == [
        (next_to, RELATIVE), ("/mnt/z/p/lay1/x.h5", "via drive Z:")]
    project = os.path.join(os.path.abspath("/share/p"), "demo.h5")  # this platform's paths
    data = os.path.join(os.path.abspath("/share/p"), "lay1", "x.h5")
    same = data_file_candidates(project, data, "lay1/x.h5")
    assert [c.how for c in same] == [RELATIVE], "one path is listed once"


def test_a_folder_change_moves_the_other_files_the_same_way():
    # The result is a path of this platform (the located file's own syntax).
    assert moved_path("/mnt/z/data/run1/x.h5", "/media/copy/data/run1/x.h5",
                      "/mnt/z/data/run2/x.h5") == str(PurePath("/media/copy/data/run2/x.h5"))
    assert moved_path(r"Z:\data\run1\x.h5", "/mnt/q/data/run1/x.h5",
                      r"Z:\data\run2\x.h5") == str(PurePath("/mnt/q/data/run2/x.h5"))
    if os.name == "nt":  # located on Windows: the other files move to the same drive
        assert moved_path("/mnt/z/data/run1/x.h5", r"E:\copy\data\run1\x.h5",
                          "/mnt/z/data/run2/x.h5") == r"E:\copy\data\run2\x.h5"
    assert moved_path("/a/x.h5", "/b/y.h5", "/a/z.h5") is None, "nothing in common"
    assert moved_path("/mnt/z/d/r1/x.h5", "/e/d/r1/x.h5", "/other/r2/x.h5") is None


def _layers(folder, n=2):
    """A project in ``folder`` with one data file per layer folder and one ROI each."""
    rng = np.random.default_rng(3)
    project = Project.create(folder / "demo.h5")
    for layer in range(1, n + 1):
        (folder / f"lay-{layer}").mkdir()
        path = folder / f"lay-{layer}" / "stream_hsi.h5"
        save_hdf5(str(path), rng.integers(500, 3000, (8, 4, 6)).astype(np.uint16), WAVELENGTHS)
        project.add_dataset(path).add_roi(f"lay-{layer}_A1", frames=(1, 2), rows=(0, 1))
    return project


def _meta(path):
    with h5py.File(path, "r") as f:
        return dict(f["project_meta"].attrs)


def test_a_copied_project_folder_finds_its_data_next_to_it_and_writes_nothing(tmp_path):
    (tmp_path / "share").mkdir()
    original = _layers(tmp_path / "share")
    (ref,) = [r for r in original.workspace.datasets.values() if "lay-1" in r.source_path]
    assert ref.relative_path == "lay-1/stream_hsi.h5"
    copy = tmp_path / "laptop"
    shutil.copytree(tmp_path / "share", copy)
    before = _meta(copy / "demo.h5")

    project = Project.open(copy / "demo.h5")
    found = project.find_data_file(ref.dataset_id)
    assert found is not None and found.how == RELATIVE, "relative first: the copy, not the share"
    assert found.path == str(copy / "lay-1" / "stream_hsi.h5")
    assert project.dataset(found.path).ref.dataset_id == ref.dataset_id, "never registered twice"
    assert project.workspace.datasets[ref.dataset_id].source_path == ref.source_path
    time.sleep(0.01)
    assert _meta(copy / "demo.h5") == before, "finding a file writes nothing"


def test_a_moved_project_folder_finds_its_data_when_the_stored_path_is_gone(tmp_path):
    (tmp_path / "share").mkdir()
    original = _layers(tmp_path / "share")
    ids = list(original.workspace.datasets)
    shutil.move(tmp_path / "share", tmp_path / "moved")
    project = Project.open(tmp_path / "moved" / "demo.h5")
    assert all(project.find_data_file(i).how == RELATIVE for i in ids)
    # The workflow reads the file where it was found.
    dataset = project.dataset(project.data_file(ids[0]))
    dataset.add_roi("lay-1_A2", frames=(4, 5), rows=(0, 1))


@pytest.mark.skipif(os.name == "nt", reason="a WSL mount exists only on Linux")
def test_a_windows_path_is_found_through_its_wsl_mount(tmp_path, monkeypatch):
    (tmp_path / "share").mkdir()
    project = _layers(tmp_path / "share", n=1)
    (dataset_id,) = project.workspace.datasets
    mount = tmp_path / "mnt"
    shutil.copytree(tmp_path / "share", mount / "z" / "data")
    with h5py.File(project.path, "r+") as f:  # as written on Windows, across drives
        f[f"datasets/{dataset_id}"].attrs["source_path"] = r"Z:\data\lay-1\stream_hsi.h5"
        f[f"datasets/{dataset_id}"].attrs["relative_path"] = ""
    monkeypatch.setattr(_data_paths, "wsl_mount_root", lambda: f"{mount}/")

    reopened = Project.open(project.path)
    found = reopened.find_data_file(dataset_id)
    assert found is not None and found.how == "via drive Z:"
    assert found.path == str(mount / "z" / "data" / "lay-1" / "stream_hsi.h5")
    assert reopened.dataset(found.path).ref.dataset_id == dataset_id


def test_a_look_alike_file_at_a_candidate_path_is_refused(tmp_path):
    (tmp_path / "share").mkdir()
    project = _layers(tmp_path / "share", n=1)
    (dataset_id,) = project.workspace.datasets
    copy = tmp_path / "copy"
    copy.mkdir()
    shutil.copy(project.path, copy / "demo.h5")
    (copy / "lay-1").mkdir()
    # Same name, format, shape and wavelengths; other pixels.
    save_hdf5(str(copy / "lay-1" / "stream_hsi.h5"), np.full((8, 4, 6), 700, np.uint16),
              WAVELENGTHS)
    found = Project.open(copy / "demo.h5").find_data_file(dataset_id)
    assert found is not None and found.how == STORED, "the look-alike is skipped"
    (tmp_path / "share" / "lay-1" / "stream_hsi.h5").unlink()
    assert Project.open(copy / "demo.h5").find_data_file(dataset_id) is None


def test_a_project_without_relative_paths_still_opens(tmp_path):
    (tmp_path / "share").mkdir()
    project = _layers(tmp_path / "share", n=1)
    (dataset_id,) = project.workspace.datasets
    with h5py.File(project.path, "r+") as f:
        del f[f"datasets/{dataset_id}"].attrs["relative_path"]
    reopened = Project.open(project.path)
    assert reopened.workspace.datasets[dataset_id].relative_path == ""
    assert reopened.find_data_file(dataset_id).how == STORED


def test_one_locate_offers_the_same_folder_change_for_the_others(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from enginir.viewer import open_project
    from enginir.viewer._app import ensure_app

    ensure_app()
    (tmp_path / "share").mkdir()
    _layers(tmp_path / "share", n=3)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.move(tmp_path / "share" / "demo.h5", elsewhere / "demo.h5")  # relative paths break
    shutil.move(tmp_path / "share", tmp_path / "archive")                # stored paths break
    questions: list[str] = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda _p, _t, text, *a, **k: questions.append(text) or QMessageBox.StandardButton.Yes))
    window = open_project(str(elsewhere / "demo.h5"))
    try:
        state = window.project_state
        assert all(state.find_data_file(i) is None for i in state.datasets)
        first = next(i for i, r in state.datasets.items() if "lay-1" in r.source_path)
        assert window.locate_data_file(first, str(tmp_path / "archive" / "lay-1"
                                                  / "stream_hsi.h5"))
        window.wait_for_data_file()
        assert len(questions) == 1 and questions[0].startswith("2 more dataset file(s)")
        stored = Project.open(elsewhere / "demo.h5").workspace.datasets
        assert all(str(tmp_path / "archive") in r.source_path for r in stored.values())
        assert all(r.relative_path.startswith("../archive/") for r in stored.values())
    finally:
        window.close()


def test_the_dataset_list_shows_only_the_path_a_file_is_read_from(tmp_path):
    """How the file was found and the stored path are logged, not shown (user, 2026-11-13)."""
    from PyQt6.QtCore import Qt

    from enginir.viewer import open_project
    from enginir.viewer._app import ensure_app

    ensure_app()
    (tmp_path / "share").mkdir()
    _layers(tmp_path / "share", n=1)
    shutil.move(tmp_path / "share", tmp_path / "moved")  # found next to the project
    window = open_project(str(tmp_path / "moved" / "demo.h5"))
    try:
        combo = window._data_file_combo
        tips = [combo.itemData(i, Qt.ItemDataRole.ToolTipRole) for i in range(combo.count())
                if isinstance(combo.itemData(i), str)]
        assert str(tmp_path / "moved" / "lay-1" / "stream_hsi.h5") in tips
        assert not any("found" in str(tip) or "share" in str(tip) for tip in tips)
    finally:
        window.close()


# -- Dataset labels --------------------------------------------------------------------


def test_a_dataset_label_is_stored_shown_and_tolerated_when_missing(tmp_path):
    project = Project.create(tmp_path / "p.h5")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = project.add_dataset(framed_scan(tmp_path / "a" / "stream.h5"))
    b = project.add_dataset(framed_scan(tmp_path / "b" / "stream.h5"))
    labels = data_file_labels(project.workspace.datasets)
    assert labels[a.ref.dataset_id].startswith("stream.h5 (")
    a.set_label("Layer 7")
    reopened = Project.open(project.path)
    assert reopened.workspace.datasets[a.ref.dataset_id].label == "Layer 7"
    labels = data_file_labels(reopened.workspace.datasets)
    assert labels[a.ref.dataset_id] == "Layer 7"
    assert labels[b.ref.dataset_id] == "stream.h5 (b)", "the other keeps its file label"

    with h5py.File(project.path, "a") as handle:  # a 1.5.x project: no label attribute
        del handle["datasets"][a.ref.dataset_id].attrs["label"]
        handle["datasets"][b.ref.dataset_id].attrs["label"] = 7  # malformed display data
    reopened = Project.open(project.path)
    assert reopened.workspace.datasets[a.ref.dataset_id].label == ""
    assert reopened.workspace.datasets[b.ref.dataset_id].label == ""
