"""Smoke tests for the split EDA view."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QMessageBox

from enginir.project import CalibrationSpec, DatasetRef, ModelRun, SampleSet, TrainingROI
from enginir.viewer._app import ensure_app


def _roi_provenance(
    *, dataset_id: str = "dataset_1", calibration_id: str = "cal_1"
) -> dict[str, str]:
    return {
        "dataset_id": dataset_id,
        "calibration_id": calibration_id,
        "signal_unit": "DN",
        "preview_recipe_summary": "none",
        "created_at": "2026-04-02T10:00:00+00:00",
    }


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


@pytest.fixture()
def state_and_project(tmp_path):
    from enginir.viewer._state import ProjectState, ViewerState

    cube = np.random.randint(0, 1000, (6, 5, 8), dtype=np.uint16)
    wavelengths = np.linspace(1000.0, 1700.0, 8)
    state = ViewerState(cube, wavelengths, {"source_modality": "nir_hsi"})
    project = ProjectState()
    project.new_project(str(tmp_path / "eda.h5"))
    project.register_dataset(
        DatasetRef(
            dataset_id="dataset_1",
            source_path=str(tmp_path / "cube.eiff"),
            source_format="eiff",
            source_modality="nir_hsi",
            cube_shape=cube.shape,
            n_bands=cube.shape[-1],
            wavelengths=wavelengths.astype("float32"),
            loader_metadata={},
            import_timestamp="2026-04-02T10:00:00+00:00",
        )
    )
    project.set_active_calibration(
        CalibrationSpec(
            calibration_id="cal_1",
            dataset_id="dataset_1",
            mode="raw_dn",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=None,
            white_level=None,
            output_signal_unit="DN",
            created_at="2026-04-02T10:01:00+00:00",
        )
    )
    return state, project


class _FakePanel:
    def __init__(self):
        self.values: list[bool] = []

    def toggle_roi_mode(self, enabled: bool) -> None:
        self.values.append(enabled)

    def toggle_training_roi_mode(self, enabled: bool) -> None:
        self.values.append(enabled)


class _FakeSignal:
    def __init__(self):
        self.values: list[tuple] = []
        self._callbacks: list[object] = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        self.values.append(args)


class _FakeCanvas:
    def __init__(self):
        self.rects: list[tuple[int, int, int, int, str, str]] = []
        self.entered = 0
        self.canceled = 0
        self.selected: list[str | None] = []
        self.roi_draw_finished = _FakeSignal()
        self.roi_draw_canceled = _FakeSignal()
        self.roi_selected = _FakeSignal()

    def clear_roi_rects(self) -> None:
        self.rects.clear()

    def add_roi_rect(self, f0: int, f1: int, r0: int, r1: int, name: str, roi_id: str) -> None:
        self.rects.append((f0, f1, r0, r1, name, roi_id))

    def enter_roi_draw_mode(self) -> None:
        self.entered += 1

    def cancel_roi_draw_mode(self) -> None:
        self.canceled += 1

    def select_roi(self, roi_id: str | None) -> None:
        self.selected.append(roi_id)


def _make_view(
    state,
    project,
    cube_path: str,
    *,
    dataset_attached: bool = True,
    selected_roi_id: list[str | None] | None = None,
):
    from enginir.viewer._eda import EDAView

    canvas = _FakeCanvas()
    selected = selected_roi_id if selected_roi_id is not None else [None]
    view = EDAView(
        state,
        project,
        dc3_panel=_FakePanel(),
        canvas=canvas,
        cube_path_provider=lambda: cube_path,
        dataset_attached_provider=lambda: dataset_attached,
        current_target_provider=lambda: None,
        selected_roi_id_provider=lambda: selected[0],
    )
    return view, canvas


def test_training_roi_selection_registers_dataset_calibration_and_overlay(
    qapp, state_and_project, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QDialog
    from enginir.viewer._viewer import NewROIDialog

    state, project = state_and_project
    view, canvas = _make_view(state, project, str(tmp_path / "cube.eiff"))
    monkeypatch.setattr(
        NewROIDialog,
        "exec",
        lambda self: self._name_edit.setText("roi_a") or QDialog.DialogCode.Accepted,
    )

    view.on_training_roi_selected(0, 1, 0, 1)

    roi = next(iter(project.rois.values()))
    assert roi.name == "roi_a"
    assert roi.dataset_id in project.datasets
    assert roi.calibration_id in project.calibrations
    assert canvas.rects[0][-1] == roi.roi_id
    assert canvas.selected[-1] == roi.roi_id


def test_training_roi_selection_persists_absorbance_calibration_context(
    qapp, state_and_project, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QDialog
    from enginir.viewer._viewer import NewROIDialog

    state, project = state_and_project
    state.set_y_unit("Absorbance")
    dark_ref = np.arange(state.height * state.n_bands, dtype=np.float64).reshape(
        state.height, state.n_bands
    )
    white_ref = dark_ref + 1000.0
    state.set_dark_ref(dark_ref)
    state.set_white_ref(white_ref)
    view, _ = _make_view(state, project, str(tmp_path / "cube.eiff"))
    monkeypatch.setattr(
        NewROIDialog,
        "exec",
        lambda self: self._name_edit.setText("roi_abs") or QDialog.DialogCode.Accepted,
    )

    view.on_training_roi_selected(0, 1, 0, 1)

    roi = next(iter(project.rois.values()))
    calibration = project.calibrations[roi.calibration_id]
    assert calibration.mode == "dark_white_reference"
    assert calibration.output_signal_unit == "Absorbance"
    assert calibration.dark_ref_data is not None
    assert calibration.white_ref_data is not None
    assert project.active_calibration_ids["dataset_1"] == roi.calibration_id
    from enginir.io.project_store import iter_roi_calibration_refs

    loaded_refs = dict(iter_roi_calibration_refs(project.project_path or "", [roi.roi_id]))
    assert loaded_refs[roi.roi_id] is not None
    loaded_dark, loaded_white = loaded_refs[roi.roi_id]
    expected_dark = np.tile(dark_ref[0:2], (2, 1)).astype(np.float32)
    expected_white = np.tile(white_ref[0:2], (2, 1)).astype(np.float32)
    np.testing.assert_allclose(loaded_dark, expected_dark)
    np.testing.assert_allclose(loaded_white, expected_white)


def test_training_roi_selection_requires_project_and_attached_dataset(qapp, tmp_path, monkeypatch):
    from enginir.viewer._eda import EDAView
    from enginir.viewer._state import ProjectState, ViewerState

    cube = np.random.randint(0, 1000, (4, 4, 4), dtype=np.uint16)
    state = ViewerState(cube, np.linspace(1000.0, 1300.0, 4))
    project = ProjectState()
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warnings.append(str(args[1]))
    )

    view = EDAView(
        state,
        project,
        dc3_panel=_FakePanel(),
        canvas=_FakeCanvas(),
        cube_path_provider=lambda: str(tmp_path / "cube.eiff"),
        dataset_attached_provider=lambda: False,
        current_target_provider=lambda: None,
        selected_roi_id_provider=lambda: None,
    )
    view.on_training_roi_selected(0, 1, 0, 1)

    assert warnings
    assert project.rois == {}


def test_roi_button_toggles_canvas_draw_mode(qapp, state_and_project, tmp_path):
    state, project = state_and_project
    view, canvas = _make_view(state, project, str(tmp_path / "cube.eiff"), dataset_attached=True)

    view._btn_roi_training.click()
    assert canvas.entered == 1

    view._btn_roi_training.setChecked(False)
    assert canvas.canceled == 1


def test_delete_roi_button_removes_selected_roi(qapp, state_and_project, tmp_path, monkeypatch):
    state, project = state_and_project
    selected_roi_id = ["roi_a"]
    view, _ = _make_view(
        state,
        project,
        str(tmp_path / "cube.eiff"),
        dataset_attached=True,
        selected_roi_id=selected_roi_id,
    )
    project.add_roi(
        TrainingROI(
            roi_id="roi_a",
            dataset_id="dataset_1",
            calibration_id="cal_1",
            name="A",
            wavelengths=state.wavelengths.astype("float32"),
            n_pixels=2,
            spatial_bounds=(0, 0, 0, 1),
            source_file=str(tmp_path / "a.eiff"),
            created_at="2026-04-02T10:00:00+00:00",
            provenance=_roi_provenance(),
        ),
        np.zeros((2, state.wavelengths.shape[0]), dtype=np.uint16),
    )
    view.refresh_attachment_state()
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )

    view._delete_roi_btn_clicked()

    assert "roi_a" not in project.rois


def test_delete_roi_button_warns_when_roi_is_used_by_sample_set(
    qapp,
    state_and_project,
    tmp_path,
    monkeypatch,
):
    state, project = state_and_project
    selected_roi_id = ["roi_a"]
    view, _ = _make_view(
        state,
        project,
        str(tmp_path / "cube.eiff"),
        dataset_attached=True,
        selected_roi_id=selected_roi_id,
    )
    project.add_roi(
        TrainingROI(
            roi_id="roi_a",
            dataset_id="dataset_1",
            calibration_id="cal_1",
            name="A",
            wavelengths=state.wavelengths.astype("float32"),
            n_pixels=2,
            spatial_bounds=(0, 0, 0, 1),
            source_file=str(tmp_path / "a.eiff"),
            created_at="2026-04-02T10:00:00+00:00",
            provenance=_roi_provenance(),
        ),
        np.zeros((2, state.wavelengths.shape[0]), dtype=np.uint16),
    )
    project.sample_sets["sample_set_1"] = SampleSet(
        sample_set_id="sample_set_1",
        roi_ids=["roi_a"],
        target_id="target_1",
        observation_unit="pixel",
        target_level="roi",
        grouping_unit="roi",
        validation_unit="roi",
        weighting_policy="per_pixel_unweighted",
        reporting_level="roi",
        roi_aggregation="mean",
        preprocessing_recipe_id="recipe_1",
        dataset_ids=["dataset_1"],
        created_at="2026-04-02T10:01:00+00:00",
    )
    project.model_runs["model_run_1"] = ModelRun(
        model_run_id="model_run_1",
        sample_set_id="sample_set_1",
        signal_domain_id="signal_domain_1",
        model_type="PLS",
        hyperparameters={"n_components": 2},
        cv_spec={"type": "GroupKFold", "n_splits": 2},
        metrics={"rmsecv": 1.0},
        artifact_refs={"prediction_model": {"model_type": "PLSRegression"}},
        created_at="2026-04-02T10:02:00+00:00",
    )
    captured: dict[str, str] = {}

    def fake_question(*args, **kwargs):
        del kwargs
        captured["message"] = str(args[2])
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", fake_question)

    view._delete_roi_btn_clicked()

    assert "sample_set_1" in captured["message"]
    assert "model_run_1" in captured["message"]
    assert "remain usable for prediction" in captured["message"]
    assert "cannot be altered or refit" in captured["message"]
    assert "roi_a" not in project.rois


def test_export_csv_uses_batched_roi_load_and_preserves_order(
    qapp, state_and_project, tmp_path, monkeypatch
):
    state, project = state_and_project
    wl = state.wavelengths.astype("float32")
    spectra_a = np.arange(16, dtype=np.uint16).reshape(2, 8)
    spectra_b = np.arange(24, dtype=np.uint16).reshape(3, 8) + 100
    for roi_id, name, spectra in [("roi_a", "A", spectra_a), ("roi_b", "B", spectra_b)]:
        project.add_roi(
            TrainingROI(
                roi_id=roi_id,
                dataset_id="dataset_1",
                calibration_id="cal_1",
                name=name,
                wavelengths=wl,
                n_pixels=int(spectra.shape[0]),
                spatial_bounds=(0, 0, 0, spectra.shape[0] - 1),
                source_file=str(tmp_path / f"{name}.eiff"),
                created_at="2026-04-02T10:00:00+00:00",
                provenance=_roi_provenance(),
            ),
            spectra,
        )

    captured: dict[str, object] = {}
    batch_calls: list[list[str]] = []
    original_batch = __import__(
        "enginir.viewer._eda", fromlist=["load_roi_spectra_batch"]
    ).load_roi_spectra_batch

    monkeypatch.setattr(
        "enginir.viewer._eda.save_csv",
        lambda path, X, wavelengths, sample_names: captured.update(
            {"path": path, "X": X, "wavelengths": wavelengths, "sample_names": sample_names}
        ),
    )
    monkeypatch.setattr(
        "enginir.viewer._eda.load_roi_spectra_batch",
        lambda path, keys: batch_calls.append(list(keys)) or original_batch(path, keys),
    )

    view, _ = _make_view(state, project, str(tmp_path / "cube.eiff"))
    out_path = str(tmp_path / "out.csv")
    view.export_csv(out_path)

    assert batch_calls == [
        [project.rois["roi_a"].spectra_storage_key, project.rois["roi_b"].spectra_storage_key]
    ]
    np.testing.assert_allclose(captured["X"], np.vstack([spectra_a, spectra_b]).astype(np.float64))
    np.testing.assert_allclose(captured["wavelengths"], wl)
    assert captured["sample_names"] == ["A", "A", "B", "B", "B"]
    assert captured["path"] == out_path


def test_export_csv_rejects_mixed_wavelength_axes(qapp, state_and_project, tmp_path):
    state, project = state_and_project
    wl_a = state.wavelengths.astype("float32")
    wl_b = (state.wavelengths + 1.0).astype("float32")
    project.register_dataset(
        DatasetRef(
            dataset_id="dataset_b",
            source_path=str(tmp_path / "b.eiff"),
            source_format="eiff",
            source_modality="nir_hsi",
            cube_shape=(6, 5, 8),
            n_bands=8,
            wavelengths=wl_b,
            loader_metadata={},
            import_timestamp="2026-04-02T10:02:00+00:00",
        )
    )
    project.set_active_calibration(
        CalibrationSpec(
            calibration_id="cal_b",
            dataset_id="dataset_b",
            mode="raw_dn",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=None,
            white_level=None,
            output_signal_unit="DN",
            created_at="2026-04-02T10:03:00+00:00",
        )
    )
    project.add_roi(
        TrainingROI(
            roi_id="roi_a",
            dataset_id="dataset_1",
            calibration_id="cal_1",
            name="A",
            wavelengths=wl_a,
            n_pixels=2,
            spatial_bounds=(0, 0, 0, 1),
            source_file=str(tmp_path / "a.eiff"),
            created_at="2026-04-02T10:00:00+00:00",
            provenance=_roi_provenance(),
        ),
        np.zeros((2, wl_a.shape[0]), dtype=np.uint16),
    )
    project.add_roi(
        TrainingROI(
            roi_id="roi_b",
            dataset_id="dataset_b",
            calibration_id="cal_b",
            name="B",
            wavelengths=wl_b,
            n_pixels=2,
            spatial_bounds=(1, 1, 0, 1),
            source_file=str(tmp_path / "b.eiff"),
            created_at="2026-04-02T10:01:00+00:00",
            provenance=_roi_provenance(dataset_id="dataset_b", calibration_id="cal_b"),
        ),
        np.zeros((2, wl_b.shape[0]), dtype=np.uint16),
    )

    view, _ = _make_view(state, project, str(tmp_path / "cube.eiff"))
    with pytest.raises(ValueError, match="same wavelength axis"):
        view.export_csv(str(tmp_path / "mixed.csv"))
