"""ROI grids for trays (ROI/target contract): cell layout, names, all-or-nothing writes."""

from __future__ import annotations

import numpy as np
import pytest

from enginir import Project
from enginir._workflow import roi_grid
from enginir.io import save_hdf5


def test_cells_split_the_tray_evenly_and_leave_out_the_inset():
    cells = roi_grid((0, 59), (0, 19), n_rows=2, n_columns=3, inset=0.1)
    assert [(c.row, c.column) for c in cells] == [(0, 0), (0, 1), (0, 2),
                                                  (1, 0), (1, 1), (1, 2)]
    # Each cell is 20 frames x 10 rows; 10 % inset leaves out 2 frames and 1 row per side.
    assert cells[0].frames == (2, 17) and cells[0].rows == (1, 8)
    assert cells[2].frames == (42, 57) and cells[5].rows == (11, 18)
    assert [c.name for c in cells] == ["A01", "A02", "A03", "B01", "B02", "B03"]


def test_uneven_sizes_spread_the_remainder_and_cells_never_overlap():
    cells = roi_grid((5, 26), (0, 6), n_rows=1, n_columns=4)  # 22 frames into 4 columns
    spans = [c.frames for c in cells]
    assert spans == [(5, 9), (10, 15), (16, 20), (21, 26)]
    assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))


def test_skipped_cells_order_and_name_patterns():
    cells = roi_grid((0, 39), (0, 39), n_rows=2, n_columns=2, skip=[(0, 1)], order="columns",
                     name_pattern="{file}-{n:03d}-{row}{column}", file="tray03")
    assert [(c.row, c.column) for c in cells] == [(0, 0), (1, 0), (1, 1)]
    assert [c.name for c in cells] == ["tray03-001-A1", "tray03-002-B1", "tray03-003-B2"]
    many = roi_grid((0, 0), (0, 27), n_rows=28, n_columns=1)
    assert [c.name[:2] for c in many[25:]] == ["Z0", "AA", "AB"]


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"n_rows": 0, "n_columns": 2}, "at least one row"),
    ({"n_rows": 2, "n_columns": 50}, "cannot be split into 50 cells"),
    ({"n_rows": 2, "n_columns": 2, "inset": 0.5}, r"inset must be in \[0, 0.5\)"),
    ({"n_rows": 2, "n_columns": 2, "order": "diagonal"}, "order must be"),
    ({"n_rows": 2, "n_columns": 2, "name_pattern": "{file}"}, "repeated names"),
    ({"n_rows": 2, "n_columns": 2, "name_pattern": "{well}"}, "is not valid"),
])
def test_invalid_grids_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        roi_grid((0, 19), (0, 19), **kwargs)


@pytest.fixture()
def tray(tmp_path):
    cube = np.random.default_rng(4).integers(500, 3000, (40, 12, 6)).astype(np.uint16)
    path = tmp_path / "tray03.h5"
    save_hdf5(str(path), cube, np.linspace(1000.0, 1500.0, 6))
    project = Project.create(tmp_path / "p.h5")
    return project, project.add_dataset(path), cube


def test_a_grid_stores_one_roi_per_cell_with_its_pixels(tray):
    project, dataset, cube = tray
    cells = roi_grid((0, 39), (0, 11), n_rows=2, n_columns=4, inset=0.1,
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


def test_a_grid_with_a_taken_name_writes_nothing(tray):
    project, dataset, _cube = tray
    dataset.add_roi("A02", frames=(0, 1), rows=(0, 1))
    with pytest.raises(ValueError, match="'A02' already exists"):
        dataset.add_rois(roi_grid((0, 39), (0, 11), n_rows=1, n_columns=3))
    assert list(Project.open(project.path).rois) == ["A02"]


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
        dataset.add_rois(roi_grid((0, 39), (0, 11), n_rows=1, n_columns=4))
    assert len(calls) == 3
    assert Project.open(project.path).rois == {}
    assert project.workspace.rois == {}


def test_a_batch_with_repeated_names_writes_nothing(tray):
    from enginir._workflow import make_roi, read_frames

    project, dataset, _cube = tray
    pixels = read_frames(dataset.path, 0, 1)
    built = [make_roi(dataset.ref, dataset.calibration, name="T1", frames=(0, 1), rows=rows,
                      pixels=pixels) for rows in ((0, 1), (2, 3))]
    with pytest.raises(ValueError, match="repeat within the batch"):
        project.save_rois(built)
    assert Project.open(project.path).rois == {}
