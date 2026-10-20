"""EDA view widgets for dataset controls, calibration, and ROI persistence."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from PyQt6.QtWidgets import (
    QFileDialog,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QRadioButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .._checks import log_and_raise
from .._logging import get_logger
from .._workflow import make_calibration, make_roi, resolve_calibration, resolve_dataset
from ..io import SpectralData, save_csv
from ..io.project_store import load_project_store, load_roi_spectra_batch
from ..project import CalibrationSpec, DatasetRef, TargetDefinition
from ._attachment import confirm_project_dataset_binding
from ._data_helpers import wavelength_axes_match
from ..io import file_format
from ._loading import ViewerBusyContext
from ._modeling import NewROIDialog
from ._state import DEFAULT_SOURCE_MODALITY, ProjectState, ViewerState
from ._theme import secondary_text_style
from ..project._ids import now_iso

logger = get_logger(__name__)


class _CanvasLike(Protocol):
    roi_draw_finished: Any
    roi_draw_canceled: Any
    roi_selected: Any

    def clear_roi_rects(self) -> None: ...
    def add_roi_rect(self, f0: int, f1: int, r0: int, r1: int, label: str, roi_id: str) -> None: ...
    def enter_roi_draw_mode(self) -> None: ...
    def cancel_roi_draw_mode(self) -> None: ...
    def select_roi(self, roi_id: str | None) -> None: ...


class EDAView(QWidget):
    """EDA controls for signal configuration, project files, and ROI drawing."""

    def __init__(
        self,
        viewer_state: ViewerState,
        project_state: ProjectState,
        *,
        dc3_panel: QWidget,
        canvas: _CanvasLike,
        cube_path_provider: Callable[[], str | None],
        dataset_attached_provider: Callable[[], bool],
        current_target_provider: Callable[[], TargetDefinition | None],
        selected_roi_id_provider: Callable[[], str | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = viewer_state
        self.project_state = project_state
        self._dc3 = dc3_panel
        self._canvas = canvas
        self._cube_path_provider = cube_path_provider
        self._dataset_attached_provider = dataset_attached_provider
        self._current_target_provider = current_target_provider
        self._selected_roi_id_provider = selected_roi_id_provider

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._project_group())
        layout.addWidget(self._signal_group())
        layout.addWidget(self._filtering_group())
        layout.addWidget(self._normalisation_group())
        layout.addWidget(self._axis_group())
        layout.addWidget(self._roi_group())

        self.project_state.project_loaded.connect(self._update_project_status)
        self.project_state.project_loaded.connect(self._on_rois_changed)
        self.project_state.project_loaded.connect(self.refresh_attachment_state)
        self.project_state.rois_changed.connect(self._on_rois_changed)
        self.project_state.rois_changed.connect(self.refresh_attachment_state)
        self._canvas.roi_draw_finished.connect(self._on_canvas_roi_draw_finished)
        self._canvas.roi_draw_canceled.connect(self._on_canvas_roi_draw_canceled)
        self.refresh_attachment_state()

    def _project_group(self) -> QGroupBox:
        box = QGroupBox("Project")
        layout = QVBoxLayout(box)
        layout.setSpacing(2)

        btn_row = QHBoxLayout()
        btn_new = QToolButton()
        btn_new.setText("New")
        btn_new.clicked.connect(self.new_project_btn_clicked)

        btn_open = QToolButton()
        btn_open.setText("Open")
        btn_open.clicked.connect(self.open_project_btn_clicked)

        btn_row.addWidget(btn_new)
        btn_row.addWidget(btn_open)
        layout.addLayout(btn_row)

        self.project_status_label = QLabel("No project")
        self.project_status_label.setStyleSheet(secondary_text_style())
        layout.addWidget(self.project_status_label)
        self.project_autosave_label = QLabel("Changes are saved automatically.")
        self.project_autosave_label.setStyleSheet(secondary_text_style())
        self.project_autosave_label.setWordWrap(True)
        layout.addWidget(self.project_autosave_label)
        return box

    def _signal_group(self) -> QGroupBox:
        box = QGroupBox("Signal")
        layout = QVBoxLayout(box)
        layout.setSpacing(2)

        self._rb_dn = QRadioButton("Raw DN")
        self._rb_ref = QRadioButton("Reflectance")
        self._rb_abs = QRadioButton("Absorbance")
        self._rb_dn.setChecked(True)

        self._signal_bg = QButtonGroup()
        for rb in (self._rb_dn, self._rb_ref, self._rb_abs):
            self._signal_bg.addButton(rb)
            layout.addWidget(rb)

        self._signal_info = QLabel()
        self._signal_info.setStyleSheet(secondary_text_style())
        layout.addWidget(self._signal_info)

        self._signal_bg.buttonClicked.connect(self._on_signal_changed)
        return box

    def _filtering_group(self) -> QGroupBox:
        box = QGroupBox("Filtering")
        layout = QGridLayout(box)
        layout.setSpacing(2)

        self._rb_filt_none = QRadioButton("None")
        self._rb_filt_sg0 = QRadioButton("SG Smooth")
        self._rb_filt_sg1 = QRadioButton("SG 1st deriv")
        self._rb_filt_sg2 = QRadioButton("SG 2nd deriv")
        self._rb_filt_none.setChecked(True)

        self._filt_bg = QButtonGroup()
        for col, rb in enumerate(
            (
                self._rb_filt_none,
                self._rb_filt_sg0,
                self._rb_filt_sg1,
                self._rb_filt_sg2,
            )
        ):
            self._filt_bg.addButton(rb)
            layout.addWidget(rb, 0, col)

        layout.addWidget(QLabel("Window:"), 1, 0)
        self._spin_win = QSpinBox()
        self._spin_win.setRange(3, 101)
        self._spin_win.setSingleStep(2)
        self._spin_win.setValue(11)
        self._spin_win.setEnabled(False)
        layout.addWidget(self._spin_win, 1, 1)

        layout.addWidget(QLabel("Poly:"), 1, 2)
        self._spin_pol = QSpinBox()
        self._spin_pol.setRange(1, 9)
        self._spin_pol.setValue(2)
        self._spin_pol.setEnabled(False)
        layout.addWidget(self._spin_pol, 1, 3)

        self._filt_bg.buttonClicked.connect(self._on_filter_changed)
        self._spin_win.valueChanged.connect(self.state.set_sg_window)
        self._spin_pol.valueChanged.connect(self.state.set_sg_polyorder)
        return box

    def _normalisation_group(self) -> QGroupBox:
        box = QGroupBox("Normalisation")
        layout = QGridLayout(box)
        layout.setSpacing(2)

        self._cb_snv = QCheckBox("SNV")
        self._cb_msc = QCheckBox("MSC")
        self._cb_emsc = QCheckBox("EMSC")
        self._cb_det = QCheckBox("Detrend")

        layout.addWidget(self._cb_snv, 0, 0)
        layout.addWidget(self._cb_msc, 0, 1)
        layout.addWidget(self._cb_emsc, 1, 0)
        layout.addWidget(self._cb_det, 1, 1)

        layout.addWidget(QLabel("order:"), 2, 0)
        self._spin_det = QSpinBox()
        self._spin_det.setRange(1, 5)
        self._spin_det.setValue(1)
        self._spin_det.setEnabled(False)
        layout.addWidget(self._spin_det, 2, 1)

        self._cb_snv.toggled.connect(self.state.set_preprocess_snv)
        self._cb_msc.toggled.connect(self.state.set_preprocess_msc)
        self._cb_emsc.toggled.connect(self.state.set_preprocess_emsc)
        self._cb_det.toggled.connect(self._on_detrend_toggled)
        self._spin_det.valueChanged.connect(self.state.set_detrend_order)
        return box

    def _axis_group(self) -> QGroupBox:
        box = QGroupBox("Axis")
        layout = QGridLayout(box)
        layout.setSpacing(2)

        layout.addWidget(QLabel("Spectral:"), 0, 0)
        self._rb_nm = QRadioButton("nm")
        self._rb_band = QRadioButton("band")
        self._rb_nm.setChecked(True)
        self._spectral_bg = QButtonGroup()
        self._spectral_bg.addButton(self._rb_nm)
        self._spectral_bg.addButton(self._rb_band)
        layout.addWidget(self._rb_nm, 0, 1)
        layout.addWidget(self._rb_band, 0, 2)

        layout.addWidget(QLabel("Spatial:"), 1, 0)
        self._rb_px = QRadioButton("pixel")
        self._rb_mm = QRadioButton("mm")
        self._rb_px.setChecked(True)
        self._spatial_bg = QButtonGroup()
        self._spatial_bg.addButton(self._rb_px)
        self._spatial_bg.addButton(self._rb_mm)
        layout.addWidget(self._rb_px, 1, 1)
        layout.addWidget(self._rb_mm, 1, 2)

        layout.addWidget(QLabel("px (mm):"), 2, 0)
        self._spin_px = QDoubleSpinBox()
        self._spin_px.setRange(0.001, 100.0)
        self._spin_px.setSingleStep(0.01)
        self._spin_px.setDecimals(3)
        self._spin_px.setValue(1.0)
        self._spin_px.setEnabled(False)
        layout.addWidget(self._spin_px, 2, 1, 1, 2)

        self._spectral_bg.buttonClicked.connect(self._on_spectral_axis_changed)
        self._spatial_bg.buttonClicked.connect(self._on_spatial_axis_changed)
        self._spin_px.valueChanged.connect(self.state.set_pixel_size_mm)
        return box

    def _roi_group(self) -> QGroupBox:
        box = QGroupBox("ROI")
        layout = QVBoxLayout(box)
        layout.setSpacing(2)

        self._btn_roi_training = QToolButton()
        self._btn_roi_training.setText("Add ROI")
        self._btn_roi_training.setCheckable(True)
        self._btn_roi_training.toggled.connect(self._on_add_roi_toggled)
        layout.addWidget(self._btn_roi_training)

        self._btn_roi_delete = QToolButton()
        self._btn_roi_delete.setText("Delete ROI")
        self._btn_roi_delete.clicked.connect(self._delete_roi_btn_clicked)
        layout.addWidget(self._btn_roi_delete)

        btn_export = QToolButton()
        btn_export.setText("Export csv")
        btn_export.clicked.connect(self.export_csv_btn_clicked)
        layout.addWidget(btn_export)

        self._roi_hint_label = QLabel()
        self._roi_hint_label.setStyleSheet(secondary_text_style())
        self._roi_hint_label.setWordWrap(True)
        layout.addWidget(self._roi_hint_label)
        return box

    def _on_signal_changed(self, button) -> None:
        unit_map = {
            self._rb_dn: "DN",
            self._rb_ref: "Reflectance",
            self._rb_abs: "Absorbance",
        }
        unit = unit_map.get(button, "DN")

        if unit in {"Reflectance", "Absorbance"}:
            if self.state.calibrated:
                msg = ""
            elif self.state.dark_ref is not None and self.state.white_ref is not None:
                msg = ""
            else:
                bl, wl = int(self.state.black_level), int(self.state.white_level)
                msg = f"default: black={bl}  white={wl}"
        else:
            msg = ""
        self._signal_info.setText(msg)
        self.state.set_y_unit(unit)

    def _on_filter_changed(self, button) -> None:
        mode_map = {
            self._rb_filt_none: "none",
            self._rb_filt_sg0: "sg_smooth",
            self._rb_filt_sg1: "sg_d1",
            self._rb_filt_sg2: "sg_d2",
        }
        mode = mode_map.get(button, "none")
        sg_active = mode != "none"
        self._spin_win.setEnabled(sg_active)
        self._spin_pol.setEnabled(sg_active)
        self.state.set_preprocess_filter(mode)

    def _on_detrend_toggled(self, checked: bool) -> None:
        self._spin_det.setEnabled(checked)
        self.state.set_preprocess_detrend(checked)

    def _on_spectral_axis_changed(self, button) -> None:
        unit = "nm" if button is self._rb_nm else "band"
        self.state.set_x_spectral_unit(unit)

    def _on_spatial_axis_changed(self, button) -> None:
        mm = button is self._rb_mm
        self._spin_px.setEnabled(mm)
        self.state.set_x_spatial_unit("mm" if mm else "pixel")

    def export_csv_btn_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            self.export_csv(path)
        except ValueError as exc:
            message = str(exc)
            if message == "No training ROIs defined yet.":
                QMessageBox.information(self, "Export csv", message)
                return
            QMessageBox.warning(self, "Export csv failed", message)

    def export_csv(self, path: str) -> None:
        if self.project_state.project_path is None:
            log_and_raise("Open a project before exporting CSV.")
        rois = list(self.project_state.rois.values())
        if not rois:
            log_and_raise("No training ROIs defined yet.")
        wavelengths = np.asarray(rois[0].wavelengths, dtype=np.float32)
        for roi in rois:
            roi_wavelengths = np.asarray(roi.wavelengths, dtype=np.float32)
            if not wavelength_axes_match(roi_wavelengths, wavelengths):
                log_and_raise(
                    "Export csv requires all persisted ROIs to share the same wavelength axis.",
                )

        spectra_by_key = load_roi_spectra_batch(
            self.project_state.project_path,
            [roi.spectra_storage_key for roi in rois],
        )
        total_rows = sum(
            int(np.asarray(spectra_by_key[roi.spectra_storage_key]).shape[0]) for roi in rois
        )
        X = np.empty((total_rows, int(wavelengths.shape[0])), dtype=np.float64)
        labels: list[str] = []
        start = 0
        for roi in rois:
            spectra = np.asarray(spectra_by_key[roi.spectra_storage_key], dtype=np.float64)
            stop = start + spectra.shape[0]
            X[start:stop] = spectra
            label = roi.name or roi.roi_id
            labels.extend([label] * spectra.shape[0])
            start = stop
        save_csv(path, X, wavelengths, sample_names=labels)

    def new_project_btn_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "New Project", "", "HDF5 (*.h5)")
        if not path:
            return
        if pathlib.Path(path).exists():
            result = QMessageBox.warning(
                self,
                "File exists",
                f"'{pathlib.Path(path).name}' already exists.\n"
                "Creating a new project will permanently overwrite it.\nContinue?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            )
            if result != QMessageBox.StandardButton.Ok:
                return
        self.project_state.new_project(path)
        self._set_project_active()

    def open_project_btn_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "HDF5 (*.h5)")
        if not path:
            return
        self.open_project_btn_clicked_with_path(path)

    def open_project_btn_clicked_with_path(self, path: str) -> None:
        cube_path = self._cube_path_provider() if self._dataset_attached_provider() else None
        busy = ViewerBusyContext(f"Opening project: {path}")
        busy.begin()
        try:
            store = load_project_store(path)
        except ValueError as exc:
            busy.end()
            logger.error("Invalid project file '{}': {}", path, exc)
            QMessageBox.warning(self, "Invalid project file", str(exc))
            return
        finally:
            busy.end()
        try:
            if not confirm_project_dataset_binding(
                self,
                project_path=path,
                store=store,
                dataset_path=cube_path,
                source_format=None if not cube_path else file_format(cube_path),
                source_modality=str(self.state.meta.get("source_modality") or DEFAULT_SOURCE_MODALITY),
                cube_shape=(
                    int(self.state.cube.shape[0]),
                    int(self.state.cube.shape[1]),
                    int(self.state.cube.shape[2]),
                ),
                wavelengths=self.state.wavelengths,
            ):
                return
            self.project_state.open_project(path, store=store)
        except ValueError as exc:
            logger.error("Invalid project file '{}': {}", path, exc)
            QMessageBox.warning(self, "Invalid project file", str(exc))
            return
        self._set_project_active()

    def _set_project_active(self) -> None:
        self._update_project_status()
        self.refresh_attachment_state()

    def refresh_attachment_state(self) -> None:
        project_open = self.project_state.project_path is not None
        dataset_attached = self._dataset_attached_provider()
        selected_roi_id = self._selected_roi_id_provider()
        self._btn_roi_training.setEnabled(project_open and dataset_attached)
        self._btn_roi_delete.setEnabled(project_open and selected_roi_id is not None)
        tooltip = ""
        hint = "Open or create a project to draw ROIs."
        if project_open and not dataset_attached:
            tooltip = "Attach a dataset to this project before drawing ROIs."
            hint = "Attach a dataset to draw ROIs or run dataset-backed actions."
        elif project_open and dataset_attached:
            if selected_roi_id is not None:
                roi = self.project_state.rois.get(selected_roi_id)
                roi_name = roi.name if roi is not None and roi.name else selected_roi_id
                hint = f"Selected ROI: {roi_name}. Delete ROI removes it from the project."
            else:
                hint = "Click an ROI to inspect it, or Add ROI to draw one on Spatial Image."
        self._btn_roi_training.setToolTip(tooltip)
        self._roi_hint_label.setText(hint)

    def _update_project_status(self) -> None:
        ps = self.project_state
        if ps.project_path is None:
            self.project_status_label.setText("No project")
            return
        fname = os.path.basename(ps.project_path)
        n_rois = len(ps.rois)
        dataset_status = "dataset attached" if self._dataset_attached_provider() else "no dataset"
        self.project_status_label.setText(f"{fname} | ROIs: {n_rois} | {dataset_status}")

    def update_project_status(self) -> None:
        self._update_project_status()

    def _on_rois_changed(self) -> None:
        self._canvas.clear_roi_rects()
        for roi in self.project_state.rois.values():
            f0, f1, r0, r1 = roi.spatial_bounds
            self._canvas.add_roi_rect(f0, f1, r0, r1, roi.name, roi.roi_id)
        self._update_project_status()

    def _on_add_roi_toggled(self, enabled: bool) -> None:
        if enabled:
            self._canvas.enter_roi_draw_mode()
            self._roi_hint_label.setText("Drag on Spatial Image to define an ROI. Esc cancels.")
            return
        self._canvas.cancel_roi_draw_mode()
        self.refresh_attachment_state()

    def _set_add_roi_checked(self, checked: bool) -> None:
        self._btn_roi_training.blockSignals(True)
        self._btn_roi_training.setChecked(checked)
        self._btn_roi_training.blockSignals(False)

    def _on_canvas_roi_draw_finished(self, f0: int, f1: int, r0: int, r1: int) -> None:
        self._set_add_roi_checked(False)
        self.refresh_attachment_state()
        self.on_training_roi_selected(f0, f1, r0, r1)

    def _on_canvas_roi_draw_canceled(self) -> None:
        self._set_add_roi_checked(False)
        self.refresh_attachment_state()

    def _delete_roi_btn_clicked(self) -> None:
        roi_id = self._selected_roi_id_provider()
        if self.project_state.project_path is None or roi_id is None:
            return
        name = roi_id
        roi = self.project_state.rois.get(roi_id)
        if roi is not None and roi.name:
            name = roi.name
        message = f"Delete ROI '{name}'?"
        affected_sample_set_ids = self.project_state.sample_sets_referencing_roi(roi_id)
        if affected_sample_set_ids:
            affected_model_run_ids = self.project_state.model_runs_for_sample_sets(
                affected_sample_set_ids
            )
            model_text = ", ".join(affected_model_run_ids) if affected_model_run_ids else "none"
            message += (
                "\n\nThis ROI is used by sample set(s) "
                f"{', '.join(affected_sample_set_ids)}. Existing saved model run(s) "
                f"{model_text} remain usable for prediction, but affected sample sets "
                "and model runs cannot be altered or refit until recreated."
            )
        result = QMessageBox.question(
            self,
            "Delete ROI",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if result != QMessageBox.StandardButton.Yes:
            return
        self.project_state.remove_roi(roi_id)

    def on_training_roi_selected(self, f0: int, f1: int, r0: int, r1: int) -> None:
        if self.project_state.project_path is None:
            QMessageBox.warning(
                self,
                "No project open",
                "No project open. Use Project > New or Project > Open first.",
            )
            return
        if not self._dataset_attached_provider():
            QMessageBox.warning(
                self,
                "No dataset attached",
                "Attach a real dataset to this project before drawing ROIs.",
            )
            return
        try:
            dataset, calibration = self._ensure_current_dataset_context()
        except Exception as exc:
            logger.error("Dataset context error while creating ROI: {}", exc)
            QMessageBox.warning(self, "Dataset context error", str(exc))
            return

        dlg = NewROIDialog(
            f0,
            f1,
            r0,
            r1,
            self.state,
            self._current_target_provider(),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        left, right, top, bottom = self.state._normalized_roi_bounds(f0, f1, r0, r1)
        roi, spectra, refs = make_roi(
            dataset, calibration, name=dlg.roi_name, frames=(left, right), rows=(top, bottom),
            pixels=self.state.cube[left:right + 1], preview_recipe=self.state.recipe)
        self.project_state.add_roi(roi, spectra,
                                   dark_ref_spectra=None if refs is None else refs[0],
                                   white_ref_spectra=None if refs is None else refs[1])
        self._canvas.select_roi(roi.roi_id)
        self._canvas.roi_selected.emit(roi.roi_id)

        current_target = self._current_target_provider()
        if current_target is not None and dlg.target_value is not None:
            from ..project import TargetAssignment

            self.project_state.assign_target_to_roi(
                TargetAssignment(
                    roi_id=roi.roi_id,
                    target_id=current_target.target_id,
                    value=dlg.target_value,
                    assigned_at=now_iso(),
                )
            )

    def _ensure_current_dataset_context(self) -> tuple[DatasetRef, CalibrationSpec]:
        """Register the attached dataset and its current calibration (or reuse them)."""
        cube_path = self._cube_path_provider()
        if not self._dataset_attached_provider() or not cube_path:
            log_and_raise("A real dataset must be attached before creating project-backed ROIs.")
        state, project = self.state, self.project_state
        dataset, is_new = resolve_dataset(
            project.datasets, cube_path, SpectralData(state.cube, state.wavelengths, state.meta))
        if is_new:
            project.register_dataset(dataset)
        has_refs = state.dark_ref is not None and state.white_ref is not None
        candidate = make_calibration(
            dataset, unit=str(state.y_unit_effective), precalibrated=state.calibrated,
            dark=state.dark_ref if has_refs else None, white=state.white_ref if has_refs else None,
            black_level=state.black_level, white_level=state.white_level)
        calibration, is_new = resolve_calibration(dataset, project.calibrations, candidate)
        if is_new or project.active_calibration_ids.get(dataset.dataset_id) != \
                calibration.calibration_id:
            project.set_active_calibration(calibration)
        return project.datasets[dataset.dataset_id], calibration


__all__ = ["EDAView"]
