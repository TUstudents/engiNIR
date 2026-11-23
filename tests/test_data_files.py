"""Data files of a project: labels, references, moved files and paths across platforms (attachment and dataset-reference-correction contracts)."""

from __future__ import annotations

import os
from pathlib import PurePath
import shutil
import time
from types import SimpleNamespace
import h5py
import numpy as np
import pytest

from tests._tables import check_all
from enginir import Project
from enginir._workflow import data_file_labels
from enginir.io import save_hdf5
from enginir.project import _data_paths
from enginir.project._data_paths import (
    STORED,
    drive_mapped,
    moved_path,
    stored_path,
)
from tests._scans import framed_scan


# -- Data files and their references in a project (dataset-reference-correction contract, headless) ----


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


def test_a_reference_file_and_frames_combine(tmp_path, files):
    paths, tray = files
    project = Project.create(tmp_path / "p.h5")
    reference_correction = project.add_dataset(paths["tray"], dark=paths["dark"],
                                      white_frames=(17, 19)).reference_correction
    assert reference_correction.dark_ref_source == str(paths["dark"].resolve())
    np.testing.assert_allclose(reference_correction.white_ref_data, tray[17:20].mean(axis=0), rtol=1e-6)


def test_a_session_shares_one_reference_set(tmp_path, files):
    paths, _tray = files
    project = Project.create(tmp_path / "p.h5")
    first = project.add_dataset(paths["tray"], dark=paths["dark"], white=paths["white"])
    second = project.add_dataset(paths["tray2"], references_from=paths["tray"])
    shared, own = first.reference_correction, second.reference_correction
    assert own.dataset_id == second.record.dataset_id != shared.dataset_id  # dataset-local copy
    assert own.reference_correction_id != shared.reference_correction_id
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


def test_detector_levels_can_be_given_and_are_checked(tmp_path, files):
    paths, tray = files
    project = Project.create(tmp_path / "p.h5")
    reference_correction = project.add_dataset(paths["tray"], black_level=100.0,
                                      white_level=3000.0).reference_correction
    assert (reference_correction.mode, reference_correction.black_level, reference_correction.white_level) \
        == ("dark_white_reference", 100.0, 3000.0)
    roi = project.dataset(paths["tray"]).add_roi("R", frames=(5, 6), rows=(0, 1))
    from enginir.modeling import roi_spectrum_summaries  # the fit's conversion

    ws = project.workspace
    summaries, _ = roi_spectrum_summaries(project.path, [roi], datasets=ws.datasets,
                                          reference_corrections=ws.reference_corrections, signal_unit="Reflectance")
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
    from tests._qt import wait_until

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
        # The files are looked up in the background; the list says so until then.
        wait_until(qapp, lambda: not window.project_state.data_files_unknown())
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


# -- Data files across platforms (attachment contract, Missing and moved files) --------


def test_the_drive_mapping_both_ways():
    def check(case):
        stored, windows, root, expected = case
        assert drive_mapped(stored_path(stored), windows=windows, wsl_root=root) == expected

    check_all([
    (r"Z:\NonKF\a b\x.h5", False, "/mnt/", "/mnt/z/NonKF/a b/x.h5"),
    (r"Z:\NonKF\x.h5", False, "/", "/z/NonKF/x.h5"),          # wsl.conf root = /
    ("/mnt/z/NonKF/a b/x.h5", True, None, r"Z:\NonKF\a b\x.h5"),
    ("/z/NonKF/x.h5", True, None, r"Z:\NonKF\x.h5"),
    ("/mnt/data/x.h5", True, None, None),                     # not a drive letter
    ("/home/user/x.h5", True, None, None),
    (r"\\172.16.0.61\z\x.h5", False, "/mnt/", None),          # UNC: a configured mapping
    ("/mnt/z/x.h5", False, "/mnt/", None),                    # already this platform's
], check)


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
    assert reopened.dataset(found.path).record.dataset_id == dataset_id


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


# -- Dataset labels --------------------------------------------------------------------


def test_a_dataset_label_is_stored_shown_and_tolerated_when_missing(tmp_path):
    project = Project.create(tmp_path / "p.h5")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = project.add_dataset(framed_scan(tmp_path / "a" / "stream.h5"))
    b = project.add_dataset(framed_scan(tmp_path / "b" / "stream.h5"))
    labels = data_file_labels(project.workspace.datasets)
    assert labels[a.record.dataset_id].startswith("stream.h5 (")
    a.set_label("Layer 7")
    reopened = Project.open(project.path)
    assert reopened.workspace.datasets[a.record.dataset_id].label == "Layer 7"
    labels = data_file_labels(reopened.workspace.datasets)
    assert labels[a.record.dataset_id] == "Layer 7"
    assert labels[b.record.dataset_id] == "stream.h5 (b)", "the other keeps its file label"

    with h5py.File(project.path, "a") as handle:  # a 1.5.x project: no label attribute
        del handle["datasets"][a.record.dataset_id].attrs["label"]
        handle["datasets"][b.record.dataset_id].attrs["label"] = 7  # malformed display data
    reopened = Project.open(project.path)
    assert reopened.workspace.datasets[a.record.dataset_id].label == ""
    assert reopened.workspace.datasets[b.record.dataset_id].label == ""


def test_opening_a_project_never_looks_for_data_files_on_the_ui_thread(qapp, tmp_path,
                                                                       monkeypatch):
    """A slow drive delays only the background lookup.

    The window opens with the files "checking…"; the selector, the checks and the last
    shown file follow when the lookup has answered, and nothing looks again.
    """
    import threading

    from enginir.viewer import open_project
    from enginir.viewer._project_settings import save_last_data_file
    from tests._qt import wait_until

    drive = tmp_path / "drive"
    drive.mkdir()
    paths = [drive / "present.h5", drive / "gone.h5"]
    project = Project.create(tmp_path / "p.h5")
    project.add_target("API", unit="%")
    ids = []
    for path, name in zip(paths, ("A1", "B1")):
        save_hdf5(str(path), np.full((8, 4, 6), 1000, np.uint16), np.linspace(1000.0, 1500.0, 6))
        dataset = project.add_dataset(path)
        dataset.add_roi(name, frames=(0, 1), rows=(0, 1), values={"API": 1.0})
        ids.append(dataset.record.dataset_id)
    paths[1].unlink()
    present_id = ids[0]
    save_last_data_file(str(project.workspace.project_meta["project_id"]), present_id)

    release = threading.Event()
    lookups: list[tuple[str, str]] = []
    real_isfile = os.path.isfile

    def isfile(path):
        if str(drive) in os.fspath(path):
            lookups.append((threading.current_thread().name, os.fspath(path)))
            release.wait(10)  # a network drive that answers slowly
        return real_isfile(path)

    monkeypatch.setattr(os.path, "isfile", isfile)
    window = open_project(str(project.path))
    try:
        window._explore_tabs.setCurrentWidget(window._data_file_page)  # shows the last file
        settle = time.monotonic() + 0.5  # the checks run from a 150 ms timer
        while time.monotonic() < settle:
            qapp.processEvents()
            time.sleep(0.01)
        main = threading.main_thread().name
        assert [path for thread, path in lookups if thread == main] == [], "looked up on the UI"
        combo = window._data_file_combo
        assert any("checking" in combo.itemText(i) for i in range(combo.count()))
        assert not any(c.check_id.startswith("data.files") for c in window._checks_panel.checks())
        release.set()
        wait_until(qapp, lambda: not window.project_state.data_files_unknown())
        wait_until(qapp, lambda: any(c.check_id == "data.files_missing"
                                     for c in window._checks_panel.checks()))
        texts = [combo.itemText(i) for i in range(combo.count())]
        assert any(text.endswith("(not found)") for text in texts), texts
        assert not any("checking" in text for text in texts), texts
        wait_until(qapp, lambda: window._eda_view.shown_dataset_id() == present_id)
        assert [path for thread, path in lookups if thread == main] == [], "looked up on the UI"
        count = len(lookups)
        window.refresh_checks()
        window._refresh_data_file_choices()
        assert len(lookups) == count, "each file is looked up once per session"
    finally:
        release.set()
        window.close()


def test_a_dataset_is_named_when_it_is_added(tmp_path):
    """add_dataset(name=...): stored with the new dataset; for a registered one it renames,
    None keeps the name (the Add Dataset dialog's Name field)."""
    path = tmp_path / "stream_hsi.h5"
    save_hdf5(str(path), np.full((4, 3, 6), 2000, np.uint16), np.linspace(1000, 1500, 6))
    project = Project.create(tmp_path / "p.h5")
    dataset = project.add_dataset(path, name=" Layer 7 ")
    assert dataset.record.label == "Layer 7"
    assert Project.open(project.path).workspace.datasets[dataset.record.dataset_id].label \
        == "Layer 7", "stored"
    project.add_dataset(path)  # None: the name stays
    assert project.workspace.datasets[dataset.record.dataset_id].label == "Layer 7"
    project.add_dataset(path, name="Layer 8")
    reopened = Project.open(project.path).workspace.datasets[dataset.record.dataset_id]
    assert reopened.label == "Layer 8", "a registered dataset is renamed"


