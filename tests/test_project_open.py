"""Opening a project and attaching data files: mismatch checks, reopening, relinking and several windows."""

from __future__ import annotations

import os
import shutil
import time
import h5py
import numpy as np
import pytest
from enginir import Project
from enginir.io import save_hdf5
from tests._project_payloads import roi_provenance


# -- Smoke tests for project-open and project-attach workflows -------------------------


@pytest.fixture()
def small_cube():
    return np.random.randint(0, 1000, (20, 15, 50), dtype=np.uint16)


@pytest.fixture()
def wavelengths():
    return np.linspace(900, 1700, 50)


def _save_project_roi(
    path: str, *, roi_id: str, wavelengths: np.ndarray, source_file: str, spectra: np.ndarray
) -> None:
    from enginir.io.project_store import save_reference_correction, save_dataset_ref, save_training_roi

    save_dataset_ref(
        path,
        "dataset_1",
        {
            "source_path": source_file,
            "source_format": "eiff",
            "source_modality": "nir_hsi",
            "cube_shape": (20, 15, len(wavelengths)),
            "n_bands": len(wavelengths),
            "wavelengths": wavelengths.astype(np.float32),
            "loader_metadata": {},
            "import_timestamp": "2026-03-28T10:00:00+00:00",
        },
    )
    save_reference_correction(
        path,
        "dataset_1",
        "cal_1",
        {
            "mode": "raw_dn",
            "dark_ref_source": None,
            "white_ref_source": None,
            "dark_ref_data": None,
            "white_ref_data": None,
            "black_level": None,
            "white_level": None,
            "output_signal_unit": "DN",
            "created_at": "2026-03-28T10:00:00+00:00",
        },
    )
    save_training_roi(
        path,
        roi_id,
        {
            "dataset_id": "dataset_1",
            "reference_correction_id": "cal_1",
            "name": roi_id,
            "wavelengths": wavelengths.astype(np.float32),
            "source_file": source_file,
            "job": "calibration",
            "spatial_bounds": (0, 1, 0, 2),
            "n_pixels": int(spectra.shape[0]),
            "created_at": "2026-03-28T10:00:00+00:00",
            "provenance": roi_provenance(created_at="2026-04-12T10:00:02+00:00"),
        },
        spectra,
    )


@pytest.fixture()
def project_with_roi(tmp_path, wavelengths):
    from enginir.io.project_store import create_project

    path = str(tmp_path / "proj.h5")
    create_project(path)
    _save_project_roi(
        path,
        roi_id="roi_x",
        wavelengths=wavelengths,
        source_file=os.path.abspath("/data/cube.eiff"),
        spectra=np.zeros((6, 50), dtype=np.uint16),
    )
    return path


@pytest.mark.parametrize(
    ("reply", "should_load"),
    [
        ("Cancel", False),
        ("Yes", True),
    ],
)
def test_a_registered_file_that_changed_uses_the_warning_gate(
    project_with_roi,
    small_cube,
    wavelengths,
    qapp,
    monkeypatch,
    reply,
    should_load,
):
    """The file at a registered path no longer matches its DatasetRecord (here: its shape)."""
    from PyQt6.QtWidgets import QMessageBox

    from enginir.viewer._viewer import HSIViewer

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: (
            warnings.append(str(args[2])) or getattr(QMessageBox.StandardButton, reply)
        ),
    )

    viewer = HSIViewer(small_cube[:, :10], wavelengths)  # registered with 15 rows
    viewer._cube_path = os.path.abspath("/data/cube.eiff")
    viewer._open_project_btn_clicked_with_path(project_with_roi)

    assert len(warnings) == 1
    assert "no longer matches" in warnings[0]
    assert "shape (frames, rows, bands): registered (20, 15, 50), file (20, 10, 50)" in warnings[0]
    assert "'roi_x'" in warnings[0], "the ROIs drawn from it are named"
    assert (viewer.project_state.project_path == project_with_roi) is should_load


def test_open_project_rejects_invalid_non_v3_file(
    tmp_path, small_cube, wavelengths, qapp, monkeypatch
):
    from PyQt6.QtWidgets import QMessageBox

    from enginir.viewer._viewer import HSIViewer

    invalid_path = str(tmp_path / "not_a_project.h5")
    with h5py.File(invalid_path, "w") as handle:
        handle.create_group("spectra")

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(str(args[2])) or QMessageBox.StandardButton.Ok,
    )

    viewer = HSIViewer(small_cube, wavelengths)
    viewer._open_project_btn_clicked_with_path(invalid_path)

    assert len(warnings) == 1
    assert "is not an engiNIR project file" in warnings[0]
    assert viewer.project_state.project_path is None


def test_open_viewer_sets_cube_path_and_can_attach_project(tmp_path, qapp, monkeypatch):
    import enginir.viewer as viewer_mod
    from enginir.io._types import SpectralData
    from enginir.io.project_store import create_project

    cube = np.random.randint(0, 1000, (10, 8, 20), dtype=np.uint16)
    wavelengths = np.linspace(900, 1700, 20)
    fake_path = str(tmp_path / "fake.eiff")
    open(fake_path, "w").close()
    project_path = str(tmp_path / "attached.h5")
    create_project(project_path)

    monkeypatch.setattr(
        "enginir.viewer._open.load_spectra",
        lambda path, **kwargs: SpectralData(cube, wavelengths, {}),
    )

    viewer = viewer_mod.open_viewer(fake_path, project=project_path)

    assert viewer._cube_path == os.path.abspath(fake_path)
    assert viewer.project_state.project_path == os.path.abspath(project_path)


def test_open_viewer_logs_project_attach_failure_before_reraise(tmp_path, qapp, monkeypatch):
    import enginir.viewer._open as viewer_mod

    class DummyViewer:
        def __init__(self) -> None:
            self._cube_path = str(tmp_path / "cube.eiff")
            self.closed = False

        def close(self) -> None:
            self.closed = True

    viewer = DummyViewer()
    logs: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(viewer_mod, "_open_dataset_viewer", lambda *args, **kwargs: viewer)
    monkeypatch.setattr(
        viewer_mod,
        "_load_project_into_viewer",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("project load boom")),
    )
    monkeypatch.setattr(
        viewer_mod.logger, "error", lambda message, *args: logs.append((message, args))
    )

    with pytest.raises(RuntimeError, match="project load boom"):
        viewer_mod.open_viewer(str(tmp_path / "cube.eiff"), project=str(tmp_path / "project.h5"))

    assert viewer.closed is True
    assert len(logs) == 1
    assert logs[0][0] == "Failed to attach project {} to dataset viewer: {}"


def test_open_project_logs_failure_before_reraise(tmp_path, qapp, monkeypatch):
    import enginir.viewer._open as viewer_mod
    import enginir.viewer._viewer as viewer_impl

    class DummyViewer:
        def __init__(self, *args, **kwargs) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def show(self) -> None:
            pass

        def raise_(self) -> None:
            pass

        def activateWindow(self) -> None:
            pass

    logs: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(viewer_mod, "ensure_qt_ready", lambda: None)
    monkeypatch.setattr(viewer_impl, "HSIViewer", DummyViewer)
    monkeypatch.setattr(
        viewer_mod,
        "_load_project_startup",
        lambda path: viewer_mod._ProjectStartupPayload(path=str(path), store={}),
    )
    monkeypatch.setattr(
        viewer_mod,
        "_load_project_store_into_viewer",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("project-only boom")),
    )
    monkeypatch.setattr(
        viewer_mod.logger, "error", lambda message, *args: logs.append((message, args))
    )

    with pytest.raises(RuntimeError, match="project-only boom"):
        viewer_mod.open_project(str(tmp_path / "project_only.h5"))

    assert len(logs) == 1
    assert logs[0][0] == "Failed to open project-only viewer for {}: {}"


# -- Reopening a project, moved data files and several windows (project attachment contract) ----


@pytest.fixture()
def trays(tmp_path):
    """tray_a.h5 with A1 (API 2), tray_b.h5 with B1; target API."""
    rng = np.random.default_rng(31)
    wavelengths = np.linspace(1000.0, 1700.0, 6)
    paths = {}
    for name in ("tray_a", "tray_b"):
        paths[name] = tmp_path / f"{name}.h5"
        save_hdf5(str(paths[name]), rng.integers(500, 3000, (12, 6, 6)).astype(np.uint16),
                  wavelengths)
    project = Project.create(tmp_path / "trays.h5")
    project.add_target("API", unit="%")
    project.add_dataset(paths["tray_a"]).add_roi("A1", frames=(1, 3), rows=(0, 2),
                                                 values={"API": 2.0})
    project.add_dataset(paths["tray_b"]).add_roi("B1", frames=(4, 6), rows=(1, 3))
    return project, paths


def _dataset_id(window, name):
    return next(d for d, ref in window.project_state.datasets.items()
                if ref.source_path.endswith(name))


def _process(qapp, n=10):
    for _ in range(n):
        qapp.processEvents()
        time.sleep(0.005)


def test_locate_file_from_the_not_found_question(qapp, trays, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    from enginir.viewer import open_project

    project, paths = trays
    moved = paths["tray_a"].parent / "archive" / "tray_a_renamed.h5"
    moved.parent.mkdir()
    shutil.move(paths["tray_a"], moved)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: next(
        b for b in self.buttons() if b.text() == "Locate File..."))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(moved), "")))
    window = open_project(project.path)
    try:
        window.show_data_file(_dataset_id(window, "tray_a.h5"))
        window.wait_for_data_file()
        assert window._cube_path == str(moved.resolve())
        (ref,) = [r for r in Project.open(project.path).workspace.datasets.values()
                  if r.source_path.endswith("renamed.h5")]
        assert ref.source_path == str(moved.resolve())
    finally:
        window.close()


def test_a_different_file_is_refused_and_nothing_changes(qapp, trays, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from enginir.viewer import open_project

    project, paths = trays
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda _p, _t, text, *a, **k: warnings.append(text)))
    tray_a = str(paths["tray_a"].resolve())
    window = open_project(project.path)
    try:
        assert not window.locate_data_file(_dataset_id(window, "tray_a.h5"),
                                           str(paths["tray_b"]))  # same shape, other pixels
        assert warnings and "already registered" in warnings[0]
        impostor = paths["tray_a"].parent / "impostor.h5"
        save_hdf5(str(impostor), np.full((12, 6, 6), 700, np.uint16),
                  np.linspace(1000.0, 1700.0, 6))
        assert not window.locate_data_file(_dataset_id(window, "tray_a.h5"), str(impostor))
        assert "the pixels of ROI 'A1' differ" in warnings[1]
        assert window.project_state.datasets[_dataset_id(window, "tray_a.h5")].source_path \
            == tray_a
        assert not window.has_attached_dataset()
    finally:
        window.close()
    assert any(r.source_path == tray_a for r in Project.open(project.path).workspace
               .datasets.values())


def test_a_second_window_on_the_project_refreshes(qapp, trays):
    from enginir.viewer import open_project

    project, _paths = trays
    first, second = open_project(project.path), open_project(project.path)
    try:
        api = next(iter(first.project_state.targets))
        b1 = next(r for r, roi in first.project_state.rois.items() if roi.name == "B1")
        first.project_state.set_roi_values([(b1, api, 7.5)])
        _process(qapp)
        assert second.project_state.target_assignments[(b1, api)].value == 7.5
        second.show_roi_table()
        model = second._roi_table.model
        cells = [str(model.data(model.index(r, 5)) or "") for r in range(model.rowCount())]
        assert cells == ["2", "7.5"]
        # The refreshed window writes against the current project, not a stale copy.
        second.project_state.set_roi_values([(b1, api, None)])
        _process(qapp)
        assert (b1, api) not in first.project_state.target_assignments
    finally:
        first.close()
        second.close()


# -- Focused tests for small viewer helper modules -------------------------------------


def test_wavelength_axes_match_rejects_malformed_axes():
    from enginir.viewer._data_helpers import wavelength_axes_match

    assert wavelength_axes_match(np.array([900.0, 901.0]), np.array([900.0, 901.0]))
    assert not wavelength_axes_match(
        np.array([[900.0, 901.0]]),
        np.array([900.0, 901.0]),
    )
    assert not wavelength_axes_match(
        np.array([900.0, np.nan]),
        np.array([900.0, 901.0]),
    )


