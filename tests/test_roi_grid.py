"""ROI grids for trays (ROI/target contract): cell layout, names, all-or-nothing writes."""

from __future__ import annotations

import numpy as np
import pytest

from enginir import Project
from enginir._workflow import roi_grid
from enginir.io import save_hdf5


def test_cells_follow_size_origin_and_spacing():
    cells = roi_grid(n_rows=2, n_columns=3, size=(10, 6), origin=(4, 2), spacing=(5, 3))
    assert [(c.row, c.column) for c in cells] == [(0, 0), (0, 1), (0, 2),
                                                  (1, 0), (1, 1), (1, 2)]
    # x steps by width + spacing = 15 frames, y by height + spacing = 9 rows.
    assert [c.frames for c in cells[:3]] == [(4, 13), (19, 28), (34, 43)]
    assert cells[0].rows == (2, 7) and cells[3].rows == (11, 16)
    assert [c.name for c in cells] == ["A01", "A02", "A03", "B01", "B02", "B03"]
    assert cells[4].position == "B2"
    assert cells[0].grid == "2 x 3, size 10 x 6, origin (4, 2), spacing (5, 3)"


def test_defaults_start_at_zero_and_touching_cells_never_overlap():
    cells = roi_grid(n_rows=1, n_columns=3, size=(4, 2))
    spans = [c.frames for c in cells]
    assert spans == [(0, 3), (4, 7), (8, 11)]
    assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))


def test_skipped_cells_order_and_name_patterns():
    cells = roi_grid(n_rows=2, n_columns=2, size=(5, 5), skip=[(0, 1)], order="columns",
                     name_pattern="{file}-{n:03d}-{row}{column}", file="tray03")
    assert [(c.row, c.column) for c in cells] == [(0, 0), (1, 0), (1, 1)]
    assert [c.name for c in cells] == ["tray03-001-A1", "tray03-002-B1", "tray03-003-B2"]
    many = roi_grid(n_rows=28, n_columns=1, size=(1, 1))
    assert [c.name[:2] for c in many[25:]] == ["Z0", "AA", "AB"]


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"n_rows": 0}, "at least one row"),
    ({"size": (0, 5)}, "at least 1 x 1"),
    ({"origin": (-1, 0)}, "must not be negative"),
    ({"spacing": (0, -2)}, "would overlap"),
    ({"order": "diagonal"}, "order must be"),
    ({"name_pattern": "{file}"}, "repeated names"),
    ({"name_pattern": "{well}"}, "is not valid"),
    ({"image_size": (19, 40)}, "A2 .* lies outside the image of 19 frames"),
])
def test_invalid_grids_are_refused(kwargs, message):
    grid = {"n_rows": 2, "n_columns": 2, "size": (10, 10), **kwargs}
    with pytest.raises(ValueError, match=message):
        roi_grid(**grid)


def test_skipped_cells_outside_the_image_are_allowed():
    cells = roi_grid(n_rows=1, n_columns=3, size=(10, 10), skip=[(0, 2)], image_size=(25, 10))
    assert [c.position for c in cells] == ["A1", "A2"]


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


def test_a_grid_with_a_taken_name_writes_nothing(tray):
    project, dataset, _cube = tray
    dataset.add_roi("A02", frames=(0, 1), rows=(0, 1))
    with pytest.raises(ValueError, match="'A02' already exists"):
        dataset.add_rois(roi_grid(n_rows=1, n_columns=3, size=(10, 10)))
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


def test_name_and_folder_tell_scans_with_one_file_name_apart():
    """Every Kestreleye scan is stream_hsi.h5: {name} is the dataset's name (else the file
    name), {folder} the file's folder (plan roi-workflow, R1, R2)."""
    def names(**kwargs):
        return [cell.name for cell in roi_grid(n_rows=1, n_columns=2, size=(2, 2),
                                               file="stream_hsi", **kwargs)]

    assert names(name_pattern="{name}-{row}{column}", name="Layer 7") == ["Layer 7-A1",
                                                                            "Layer 7-A2"]
    assert names(name_pattern="{name}-{row}{column}") == ["stream_hsi-A1", "stream_hsi-A2"]
    assert names(name_pattern="{folder}_{n}", folder="lay-10_API-5") == ["lay-10_API-5_1",
                                                                         "lay-10_API-5_2"]
    assert names(name_pattern="{file}-{n}", name="Layer 7") == ["stream_hsi-1", "stream_hsi-2"]


def test_a_name_pattern_being_typed_logs_no_error(qapp):
    """Each keystroke of '{folder}' made the dialog log an ERROR ('{fol}' is not valid);
    the grid that is made is still checked, with a log line."""
    from enginir.viewer._grid_settings import GridDefaults
    from enginir.viewer._roi_grid import RoiGridDialog
    from tests._scans import warnings_during

    dialog = RoiGridDialog((100, 50), file="stream_hsi", folder="lay-10",
                           defaults=GridDefaults(n_rows=1, n_columns=2, width=5, height=5,
                                                 origin_x=0, origin_y=0, spacing_x=0,
                                                 spacing_y=0))
    _, said = warnings_during(lambda: [dialog.set_grid(name_pattern=typed) for typed in
                                       ("{fo}-{n}", "{fol}-{n}", "{folder}", "{folder}-{n}")])
    assert said == [], said
    assert [cell.name for cell in dialog.cells] == ["lay-10-1", "lay-10-2"]
    _, said = warnings_during(lambda: pytest.raises(ValueError, roi_grid, n_rows=1, n_columns=1,
                                                    size=(2, 2), name_pattern="{fol}"))
    assert any("is not valid" in line for line in said), "roi_grid itself still logs"
    dialog.close()
