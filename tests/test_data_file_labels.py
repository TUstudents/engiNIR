"""Data file labels: file names, with the folder that tells repeated names apart (viewer ids)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from enginir._workflow import data_file_labels
from enginir.viewer._app import ensure_app


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


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


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
        files = sorted(str(model.data(model.index(r, 0))) for r in range(model.rowCount()))
        assert files == ["stream_hsi.h5 (run1)", "stream_hsi.h5 (run2)"]
        readiness, _ok = window._modeling_view.fit_readiness(project.target("API").target_id)
        assert "stream_hsi.h5 (run1): 1 of 1 ROIs have a value" in readiness
        assert "stream_hsi.h5 (run2): 1 of 1 ROIs have a value" in readiness
        window._project_rois.rebuild()  # the tab lays out its blocks (hidden tabs wait)
        headers = [name for name, _y in window._project_rois.mosaic._headers]
        assert sorted(headers) == ["stream_hsi.h5 (run1)", "stream_hsi.h5 (run2)"]
    finally:
        window.close()
