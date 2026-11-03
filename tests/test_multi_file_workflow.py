"""Projects with ROIs from several data files (multi-file plan, attachment contract)."""

from __future__ import annotations

import numpy as np
import pytest

from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


@pytest.fixture()
def two_file_project(tmp_path):
    """A project with two ROIs drawn on tray_a.h5 and one on tray_b.h5."""
    from enginir import Project
    from enginir.io import save_hdf5

    rng = np.random.default_rng(1)
    wavelengths = np.linspace(1000.0, 1700.0, 12)
    paths = {}
    for name in ("tray_a", "tray_b"):
        paths[name] = tmp_path / f"{name}.h5"
        save_hdf5(str(paths[name]), rng.integers(500, 3000, (16, 8, 12)).astype(np.uint16),
                  wavelengths)
    project_path = tmp_path / "trays.h5"
    project = Project.create(project_path)
    project.add_target("API", unit="%")
    tray_a = project.add_dataset(paths["tray_a"])
    tray_a.add_roi("A1", frames=(0, 3), rows=(0, 3), values={"API": 2.0})
    tray_a.add_roi("A2", frames=(5, 8), rows=(0, 3), values={"API": 4.0})
    project.add_dataset(paths["tray_b"]).add_roi("B1", frames=(2, 6), rows=(2, 5),
                                                 values={"API": 6.0})
    return project_path, paths


def _drawn_roi_names(viewer) -> list[str]:
    rois = viewer.project_state.rois
    return sorted(rois[roi_id].name for roi_id in viewer._canvas._roi_item_map)


def test_a_window_draws_only_the_rois_of_its_data_file(qapp, two_file_project):
    from enginir.viewer import open_viewer

    project_path, paths = two_file_project
    window_a = open_viewer(paths["tray_a"], project=project_path)
    window_b = open_viewer(paths["tray_b"], project=project_path)
    try:
        assert len(window_a.project_state.rois) == 3  # the project holds all of them
        assert _drawn_roi_names(window_a) == ["A1", "A2"]
        assert _drawn_roi_names(window_b) == ["B1"]
    finally:
        window_a.close()
        window_b.close()


def test_a_project_only_window_draws_the_rois_of_the_file_it_attaches(qapp, two_file_project):
    from enginir.viewer import open_project

    project_path, paths = two_file_project
    window = open_project(project_path)
    try:
        assert len(window.project_state.rois) == 3
        assert _drawn_roi_names(window) == []
        window._attach_dataset_path(str(paths["tray_b"]))  # File > Open Data
        assert _drawn_roi_names(window) == ["B1"]
    finally:
        window.close()


def test_project_writes_are_confirmed_in_the_status_bar(qapp, two_file_project, monkeypatch):
    from PyQt6.QtWidgets import QDialog

    import enginir.viewer._eda as eda
    from enginir.project import TargetAssignment
    from enginir.project._ids import now_iso
    from enginir.viewer import open_viewer

    project_path, paths = two_file_project
    monkeypatch.setattr(eda.NewROIDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(eda.NewROIDialog, "roi_name", property(lambda self: "A3"))
    monkeypatch.setattr(eda.NewROIDialog, "target_value", property(lambda self: None))
    window = open_viewer(paths["tray_a"], project=project_path)
    try:
        window._eda_view.on_training_roi_selected(10, 13, 4, 7)
        assert window._saved_label.text() == "✓ ROI 'A3' saved to project (16 pixels)"
        assert _drawn_roi_names(window) == ["A1", "A2", "A3"]

        state = window.project_state
        roi_id = next(r.roi_id for r in state.rois.values() if r.name == "A3")
        target_id = next(iter(state.targets))
        state.assign_target_to_roi(TargetAssignment(roi_id=roi_id, target_id=target_id,
                                                    value=5.5, assigned_at=now_iso()))
        assert window._saved_label.text() == "✓ API = 5.5 % saved for ROI 'A3'"

        state.remove_roi(roi_id)
        assert window._saved_label.text() == "✓ ROI 'A3' removed from project"
        assert _drawn_roi_names(window) == ["A1", "A2"]
    finally:
        window.close()
