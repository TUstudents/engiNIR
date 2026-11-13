"""Data files across platforms (attachment contract, Missing and moved files).

The same share is ``/mnt/z/…`` in WSL and ``Z:\\…`` on Windows, and a project
folder may be copied with its data. The data file is looked for next to the
project, at its stored path and through the drive mapping; finding it never
writes the project.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import PurePath, PureWindowsPath

import h5py
import numpy as np
import pytest

from enginir import Project
from enginir.io import save_hdf5
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

WAVELENGTHS = np.linspace(1000.0, 1700.0, 6)


# -- paths only ------------------------------------------------------------------


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


# -- a project ---------------------------------------------------------------------


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


# -- the viewer ----------------------------------------------------------------------


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
