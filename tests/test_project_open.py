"""Smoke tests for project-open and project-attach workflows."""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from enginir.viewer._app import ensure_app

from ._project_payloads import roi_provenance


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


@pytest.fixture()
def small_cube():
    return np.random.randint(0, 1000, (20, 15, 50), dtype=np.uint16)


@pytest.fixture()
def wavelengths():
    return np.linspace(900, 1700, 50)


def _save_project_roi(
    path: str, *, roi_id: str, wavelengths: np.ndarray, source_file: str, spectra: np.ndarray
) -> None:
    from enginir.io.project_store import save_calibration_spec, save_dataset_ref, save_training_roi

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
    save_calibration_spec(
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
            "calibration_id": "cal_1",
            "name": roi_id,
            "wavelengths": wavelengths.astype(np.float32),
            "source_file": source_file,
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


def test_open_project_starts_in_project_only_mode(tmp_path, qapp):
    import enginir.viewer as viewer_mod
    from enginir.io.project_store import create_project

    project_path = str(tmp_path / "project_only.h5")
    create_project(project_path)

    viewer = viewer_mod.open_project(project_path)

    assert viewer.project_state.project_path == os.path.abspath(project_path)
    assert viewer._cube_path is None
    assert viewer.state.has_dataset is False


def test_open_project_from_project_only_window_skips_mismatch_warning(
    project_with_roi, small_cube, wavelengths, qapp, monkeypatch
):
    from PyQt6.QtWidgets import QMessageBox

    from enginir.viewer._viewer import HSIViewer

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(str(args[2])) or QMessageBox.StandardButton.Cancel,
    )

    viewer = HSIViewer(small_cube, wavelengths)
    viewer._open_project_btn_clicked_with_path(project_with_roi)

    assert warnings == []
    assert viewer.project_state.project_path == project_with_roi


@pytest.mark.parametrize(
    ("reply", "should_load"),
    [
        ("Cancel", False),
        ("Yes", True),
    ],
)
def test_attached_dataset_project_mismatch_uses_warning_gate(
    project_with_roi,
    small_cube,
    wavelengths,
    qapp,
    monkeypatch,
    reply,
    should_load,
):
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

    viewer = HSIViewer(small_cube, wavelengths)
    viewer._cube_path = os.path.abspath("/other/cube.eiff")
    viewer._open_project_btn_clicked_with_path(project_with_roi)

    assert len(warnings) == 1
    assert ("source file" in warnings[0]) or ("does not match" in warnings[0])
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
    assert "is not an engiNIR viewer-v3 project file" in warnings[0]
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
        "enginir.viewer.load_spectra",
        lambda path, **kwargs: SpectralData(cube, wavelengths, {}),
    )

    viewer = viewer_mod.open_viewer(fake_path, project=project_path)

    assert viewer._cube_path == os.path.abspath(fake_path)
    assert viewer.project_state.project_path == os.path.abspath(project_path)


def test_open_viewer_logs_project_attach_failure_before_reraise(tmp_path, qapp, monkeypatch):
    import enginir.viewer as viewer_mod

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
    import enginir.viewer as viewer_mod
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
