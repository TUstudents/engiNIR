"""Wavelength tables for files without wavelengths (plan wavelength-tables, D1-D3)."""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pytest

from enginir import Project
from enginir.io import load_wavelength_table, save_hdf5, wavelength_table, wavelength_tables
from enginir.io.wavelength_tables import user_table_folder

KESTRELEYE = "Kestreleye 176 bands"
N_BANDS = 176
CONTENT = np.linspace(1.0, 6.0, 6)


def stream_file(path, *, n_bands=N_BANDS) -> str:
    """A Kestreleye-like ``stream_hsi.h5``: only ``data``, no wavelengths; six samples of
    three frames whose absorption around band 90 grows with CONTENT."""
    rng = np.random.default_rng(3)
    peak = np.exp(-((np.arange(n_bands) - 90) / 12) ** 2)
    content = np.repeat(CONTENT, 3)[:, None, None]
    cube = 2500 - 150 * content * peak + rng.normal(0, 3, (len(content), 4, n_bands))
    with h5py.File(path, "w") as f:
        f["data"] = cube.astype(np.uint16)
    return str(path)


def labelled(project, path, **kwargs):
    if not project.workspace.targets:
        project.add_target("API", unit="%")
    dataset = project.add_dataset(path, **kwargs)
    for i, value in enumerate(CONTENT):
        dataset.add_roi(f"{dataset.record.dataset_id[:4]}{i}", frames=(3 * i, 3 * i + 2),
                        rows=(0, 3), values={"API": value})
    return dataset


def _table(tmp_path, name="My table", rows=None, *, header="band,wavelength_nm") -> str:
    rows = rows if rows is not None else [(i, 1000.0 + 5 * i) for i in range(4)]
    path = tmp_path / f"{name.replace(' ', '_')}.csv"
    lines = ([f"# name: {name}"] if name else []) + ["# camera: Test camera", header]
    path.write_text("\n".join(lines + [f"{b},{w}" for b, w in rows]) + "\n", encoding="utf-8")
    return str(path)


# -- the tables -----------------------------------------------------------------------


@pytest.mark.parametrize("problem", ["no name", "header", "gap", "not increasing", "text"])
def test_a_malformed_table_is_refused(tmp_path, problem):
    rows = [(i, 1000.0 + 5 * i) for i in range(4)]
    kwargs = {}
    if problem == "no name":
        kwargs["name"] = ""
    elif problem == "header":
        kwargs["header"] = "index,nm"
    elif problem == "gap":
        rows[2] = (3, rows[2][1])
    elif problem == "not increasing":
        rows[2] = (2, rows[1][1])
    else:
        rows[1] = (1, "one thousand")
    with pytest.raises(ValueError):
        load_wavelength_table(_table(tmp_path, rows=rows, **kwargs))


def test_the_users_tables_are_found_and_a_bad_one_is_skipped(tmp_path):
    folder = user_table_folder()  # in the test's HOME (conftest)
    folder.mkdir(parents=True)
    shutil.copy(_table(tmp_path, "Mine"), folder / "mine.csv")
    shutil.copy(_table(tmp_path, KESTRELEYE), folder / "clash.csv")  # a name taken
    (folder / "broken.csv").write_text("# name: Broken\nband,wavelength_nm\n0,x\n",
                                       encoding="utf-8")
    assert [t.name for t in wavelength_tables()] == [KESTRELEYE, "Mine"]
    assert wavelength_table("Mine").n_bands == 4
    with pytest.raises(KeyError):
        wavelength_table("Broken")


# -- adding a dataset with a table ----------------------------------------------------


def test_a_table_is_refused_for_a_file_with_wavelengths_or_other_bands(tmp_path):
    project = Project.create(tmp_path / "p.h5")
    own = tmp_path / "own.h5"
    save_hdf5(str(own), np.full((3, 4, N_BANDS), 2000, np.uint16), np.linspace(1000, 1600, 176))
    with pytest.raises(ValueError, match="its own wavelengths"):
        project.add_dataset(own, wavelength_table=KESTRELEYE)
    with pytest.raises(ValueError, match="bands"):
        project.add_dataset(stream_file(tmp_path / "s.h5", n_bands=100),
                            wavelength_table=KESTRELEYE)
    assert project.workspace.datasets == {}, "nothing was written"


def test_a_registered_band_number_dataset_gets_a_table_only_through_set_wavelengths(tmp_path):
    path = stream_file(tmp_path / "stream_hsi.h5")
    project = Project.create(tmp_path / "p.h5")
    project.add_dataset(path)
    with pytest.raises(ValueError, match="set_wavelengths"):
        project.add_dataset(path, wavelength_table=KESTRELEYE)


# -- Set Wavelengths (D1) -------------------------------------------------------------


def test_set_wavelengths_changes_the_dataset_and_its_rois_and_keeps_the_pixels(tmp_path):
    path = stream_file(tmp_path / "stream_hsi.h5")
    project = Project.create(tmp_path / "p.h5")
    dataset = labelled(project, path)
    from enginir.io import load_roi_spectra

    before = {roi_id: load_roi_spectra(project.path, f"/rois/{roi_id}/spectra")
              for roi_id in project.workspace.rois}
    dataset.set_wavelengths(KESTRELEYE)
    table = wavelength_table(KESTRELEYE)
    reopened = Project.open(project.path)
    ref = reopened.workspace.datasets[dataset.record.dataset_id]
    assert ref.wavelength_table == KESTRELEYE
    np.testing.assert_array_equal(ref.wavelengths, table.wavelengths)
    for roi_id, roi in reopened.workspace.rois.items():
        np.testing.assert_array_equal(roi.wavelengths, table.wavelengths)
        np.testing.assert_array_equal(load_roi_spectra(reopened.path, f"/rois/{roi_id}/spectra"),
                                      before[roi_id])
    stamp = (tmp_path / "p.h5").stat().st_mtime_ns
    dataset.set_wavelengths(KESTRELEYE)  # the same table again: nothing changes
    assert (tmp_path / "p.h5").stat().st_mtime_ns == stamp


def test_set_wavelengths_is_refused_for_wavelengths_and_other_band_counts(tmp_path):
    project = Project.create(tmp_path / "p.h5")
    own = tmp_path / "own.h5"
    save_hdf5(str(own), np.full((3, 4, N_BANDS), 2000, np.uint16), np.linspace(1000, 1600, 176))
    with pytest.raises(ValueError, match="already has wavelengths"):
        project.add_dataset(own).set_wavelengths(KESTRELEYE)
    other = project.add_dataset(stream_file(tmp_path / "s.h5", n_bands=100))
    with pytest.raises(ValueError, match="100"):
        other.set_wavelengths(KESTRELEYE)


def test_a_model_fitted_on_band_numbers_predicts_the_same_after_set_wavelengths(tmp_path):
    """Prediction builds its steps from the model's own axis (D1)."""
    path = stream_file(tmp_path / "stream_hsi.h5")
    project = Project.create(tmp_path / "p.h5")
    dataset = labelled(project, path)
    model = project.fit("API", n_components=2, cv=3)
    before = model.predict(path).map
    dataset.set_wavelengths(KESTRELEYE)
    reopened = Project.open(project.path)
    after = next(iter(reopened.models.values())).predict(path).map
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-6)
    refit = reopened.fit("API", n_components=2, cv=3)  # a new model: in nm
    contract = reopened.workspace.signal_domains[refit.record.signal_domain_id]
    assert contract.wavelengths[0] == pytest.approx(1041.75)


# -- the check (D2) -------------------------------------------------------------------


# -- the viewer -----------------------------------------------------------------------


def test_the_add_dataset_dialog_preselects_the_one_table(qapp, tmp_path):
    from enginir._workflow import read_header
    from enginir.viewer._data_files import AddDataFileDialog

    path = stream_file(tmp_path / "stream_hsi.h5")
    dialog = AddDataFileDialog(path, read_header(path), {}, {})
    combo = dialog._wavelengths
    assert combo is not None and combo.currentData() == KESTRELEYE, "preselected (D2)"
    assert dialog.build_request().add_dataset_kwargs()["wavelength_table"] == KESTRELEYE
    combo.setCurrentIndex(0)
    assert dialog.build_request().wavelength_table is None, "band numbers stay possible"
    own = tmp_path / "own.h5"
    save_hdf5(str(own), np.full((3, 4, N_BANDS), 2000, np.uint16), np.linspace(1000, 1600, 176))
    assert AddDataFileDialog(str(own), read_header(str(own)), {}, {})._wavelengths is None


def test_a_table_dataset_is_shown_in_nm(qapp, tmp_path):
    from enginir.viewer import open_project, open_viewer

    path = stream_file(tmp_path / "stream_hsi.h5")
    project = Project.create(tmp_path / "p.h5")
    dataset = labelled(project, path, wavelength_table=KESTRELEYE)
    window = open_project(str(project.path))
    try:
        window.show_data_file(dataset.record.dataset_id)
        window.wait_for_data_file()
        assert window.state.wavelengths[0] == pytest.approx(1041.75)
    finally:
        window.close()
    window = open_viewer(path, project=str(project.path))  # the file first, then its project
    try:
        assert window.state.wavelengths[0] == pytest.approx(1041.75)
    finally:
        window.close()


def test_the_checks_fix_sets_the_wavelengths_of_the_shown_dataset(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QDialog

    from enginir.viewer import open_project
    from enginir.viewer._data_files import SetWavelengthsDialog

    path = stream_file(tmp_path / "stream_hsi.h5")
    project = Project.create(tmp_path / "p.h5")
    dataset = labelled(project, path)
    seen = []

    def accept(dialog):
        seen.append([dialog._list.item(i).checkState() for i in range(dialog._list.count())])
        dialog._accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SetWavelengthsDialog, "exec", accept)
    window = open_project(str(project.path))
    try:
        window.show_data_file(dataset.record.dataset_id)
        window.wait_for_data_file()
        assert window.state.wavelengths[0] == 0.0, "band numbers"
        window.refresh_checks()
        check = next(c for c in window._checks_panel.checks()
                     if c.check_id == "data.no_wavelengths")
        assert check.fix is not None
        window.run_fix(check.fix)
        assert seen, "the fix asks first: it changes stored ROIs"
        ref = window.project_state.datasets[dataset.record.dataset_id]
        assert ref.wavelength_table == KESTRELEYE
        assert all(roi.wavelengths[0] == pytest.approx(1041.75)
                   for roi in window.project_state.rois.values())
        assert window.state.wavelengths[0] == pytest.approx(1041.75), "the shown file too"
    finally:
        window.close()


def test_the_kestreleye_table_holds_the_helios_wavelengths():
    """Extracted from an .eiff of the Kestreleye setting (EVK-Helios-EC-32, 176 bands from
    wavelength start 36); a changed value changes the id."""
    table = wavelength_table(KESTRELEYE)
    assert table.n_bands == N_BANDS and "Helios" in table.camera
    assert table.wavelengths[[0, 1, 90, 175]].tolist() == pytest.approx(
        [1041.75, 1044.9, 1323.45, 1589.5], abs=1e-4)
    assert np.all(np.diff(table.wavelengths) > 3.1) and np.all(np.diff(table.wavelengths) < 3.2)
    assert table.table_id == "wlt-4927c6969fd7d903"
    assert [t.name for t in wavelength_tables(N_BANDS)] == [KESTRELEYE]
    assert wavelength_tables(220) == []
