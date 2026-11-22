"""ROI grids for trays (ROI/target contract): cell layout, names, all-or-nothing writes."""

from __future__ import annotations

import numpy as np
import pytest

from tests._tables import check_all

from enginir import Project
from enginir._workflow import roi_grid
from enginir.io import save_hdf5


def test_invalid_grids_are_refused():
    def check(case):
        kwargs, message = case
        grid = {"n_rows": 2, "n_columns": 2, "size": (10, 10), **kwargs}
        with pytest.raises(ValueError, match=message):
            roi_grid(**grid)

    check_all([
    ({"n_rows": 0}, "at least one row"),
    ({"size": (0, 5)}, "at least 1 x 1"),
    ({"origin": (-1, 0)}, "must not be negative"),
    ({"spacing": (0, -2)}, "would overlap"),
    ({"order": "diagonal"}, "order must be"),
    ({"name_pattern": "{file}"}, "repeated names"),
    ({"name_pattern": "{well}"}, "is not valid"),
    ({"image_size": (19, 40)}, "A2 .* lies outside the image of 19 frames"),
], check)


@pytest.fixture()
def tray(tmp_path):
    cube = np.random.default_rng(4).integers(500, 3000, (40, 12, 6)).astype(np.uint16)
    path = tmp_path / "tray03.h5"
    save_hdf5(str(path), cube, np.linspace(1000.0, 1500.0, 6))
    project = Project.create(tmp_path / "p.h5")
    return project, project.add_dataset(path), cube


def test_a_grid_stores_one_roi_per_cell_with_its_pixels(tray):
    project, dataset, cube = tray
    cells = roi_grid(n_rows=2, n_columns=4, size=(8, 5), origin=(1, 1), spacing=(2, 1),
                     name_pattern="{file}-{row}{column:02d}", file="tray03")
    rois = dataset.add_rois(cells)
    assert [roi.name for roi in rois] == [cell.name for cell in cells]
    stored = Project.open(project.path)
    assert len(stored.rois) == 8
    roi = stored.roi("tray03-B03")
    cell = next(c for c in cells if c.name == "tray03-B03")
    assert roi.spatial_bounds == (*cell.frames, *cell.rows)
    from enginir.io.project_store import load_roi_spectra

    spectra = load_roi_spectra(project.path, f"/rois/{roi.roi_id}/spectra")
    expected = cube[cell.frames[0]:cell.frames[1] + 1, cell.rows[0]:cell.rows[1] + 1]
    np.testing.assert_array_equal(spectra, expected.reshape(-1, cube.shape[2]))
    assert roi.provenance["grid_position"] == "B3"
    assert roi.provenance["grid"] == "2 x 4, size 8 x 5, origin (1, 1), spacing (2, 1)"


def test_a_failed_write_removes_the_rois_already_written(tray, monkeypatch):
    from enginir.io import project_store

    project, dataset, _cube = tray
    original = project_store.save_training_roi
    calls: list[str] = []

    def fail_on_third(*args, **kwargs):
        calls.append(args[1])
        if len(calls) == 3:
            raise OSError("disk full")
        return original(*args, **kwargs)

    monkeypatch.setattr(project_store, "save_training_roi", fail_on_third)
    with pytest.raises(OSError, match="disk full"):
        dataset.add_rois(roi_grid(n_rows=1, n_columns=4, size=(10, 10)))
    assert len(calls) == 3
    assert Project.open(project.path).rois == {}
    assert project.workspace.rois == {}


def test_a_grid_whose_cells_repeat_a_name_writes_nothing(tray):
    from enginir._workflow import GridCell

    project, dataset, _cube = tray
    cells = [GridCell(0, 0, (0, 1), (0, 1), "T1"), GridCell(0, 1, (2, 3), (0, 1), "T1")]
    with pytest.raises(ValueError, match="ROI names repeat within the grid"):
        dataset.add_rois(cells)
    assert Project.open(project.path).workspace.rois == {}


