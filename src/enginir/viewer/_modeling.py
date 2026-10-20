"""Modeling view widgets for target editing and model workflows."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .._checks import log_and_raise
from .._logging import get_logger
from ..io.project_store import load_model_run_artifact_refs
from ..modeling import build_sample_set as build_project_sample_set
from ..project import check_signal_domain_compatibility
from ..project import (
    ModelRun,
    SampleSet,
    SignalDomainContract,
    TargetAssignment,
    TargetDefinition,
    TrainingROI,
)
from ._jobs import BusyStatusWidget, BackgroundWorker, start_background_job
from ._state import ProjectState, ViewerState
from ._theme import secondary_text_style
from ..project._ids import now_iso

logger = get_logger(__name__)


def _finite_float_from_text(text: str) -> float | None:
    """Return one finite float from dialog text, or ``None`` when invalid."""
    try:
        value = float(text)
    except ValueError:
        return None
    if not np.isfinite(value):
        return None
    return value


class NewROIDialog(QDialog):
    """Dialog for naming a new training ROI and optionally assigning one target value."""

    def __init__(
        self,
        f0: int,
        f1: int,
        r0: int,
        r1: int,
        viewer_state: ViewerState,
        selected_target: TargetDefinition | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Training ROI")
        self._selected_target = selected_target

        n_pixels = (f1 - f0 + 1) * (r1 - r0 + 1)
        layout = QFormLayout(self)

        self._name_edit = QLineEdit()
        layout.addRow("Name:", self._name_edit)

        self._target_value_edit: QLineEdit | None = None
        if selected_target is not None:
            self._target_value_edit = QLineEdit()
            self._target_value_edit.setPlaceholderText("leave blank = unlabelled")
            layout.addRow(f"{selected_target.name}:", self._target_value_edit)
        else:
            layout.addRow("Target:", QLabel("none selected"))

        layout.addRow("Pixels:", QLabel(str(n_pixels)))

        s = viewer_state
        layout.addRow("Signal:", QLabel(s.y_unit))

        filt = s.preprocess_filter
        if filt != "none":
            filt = f"{filt}  w={s.sg_window}  p={s.sg_polyorder}"
        layout.addRow("Filter:", QLabel(filt))

        norm_parts = []
        if s.preprocess_snv:
            norm_parts.append("SNV")
        if s.preprocess_msc:
            norm_parts.append("MSC")
        if s.preprocess_emsc:
            norm_parts.append("EMSC")
        if s.preprocess_detrend:
            norm_parts.append(f"Detrend(order={s.detrend_order})")
        layout.addRow("Norm:", QLabel(", ".join(norm_parts) or "none"))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Name cannot be empty.")
            return
        target_text = ""
        if self._target_value_edit is not None:
            target_text = self._target_value_edit.text().strip()
        if target_text:
            if _finite_float_from_text(target_text) is None:
                label = self._selected_target.name if self._selected_target is not None else "Value"
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"'{target_text}' is not a finite numeric value for {label}.",
                )
                return
        self.accept()

    @property
    def roi_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def target_value(self) -> float | None:
        """Return the optional finite target value entered for the new ROI."""
        if self._target_value_edit is None:
            return None
        text = self._target_value_edit.text().strip()
        if not text:
            return None
        value = _finite_float_from_text(text)
        if value is None:
            log_and_raise("NewROIDialog target_value requires finite numeric text")
        return value


class TargetSelectionDialog(QDialog):
    """Dialog for selecting an active target or defining a new target."""

    def __init__(
        self,
        targets: list[TargetDefinition],
        current_target_id: str | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Target")
        self._targets_by_id = {target.target_id: target for target in targets}
        name_counts: dict[str, int] = {}
        for target in targets:
            name_counts[target.name] = name_counts.get(target.name, 0) + 1

        layout = QFormLayout(self)
        self._target_combo = QComboBox()
        self._target_combo.setEditable(True)
        self._target_combo.addItem("", None)
        for target in targets:
            label = target.name
            if name_counts[target.name] > 1:
                label = f"{target.name} [{target.target_id}]"
            self._target_combo.addItem(label, target.target_id)
        layout.addRow("Target:", self._target_combo)

        self._unit_edit = QLineEdit()
        self._unit_edit.setPlaceholderText("required for a new target")
        layout.addRow("Unit:", self._unit_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._target_combo.currentTextChanged.connect(self._sync_target_fields)
        current_target = (
            None if current_target_id is None else self._targets_by_id.get(current_target_id)
        )
        if current_target is not None:
            current_index = self._target_combo.findData(current_target.target_id)
            if current_index >= 0:
                self._target_combo.setCurrentIndex(current_index)
        self._sync_target_fields(self._target_combo.currentText())

    def _sync_target_fields(self, _text: str) -> None:
        target = self._selected_target()
        if target is None:
            if not self._unit_edit.isEnabled():
                self._unit_edit.clear()
            self._unit_edit.setEnabled(bool(self._target_combo.currentText().strip()))
            return
        self._unit_edit.setText(target.unit)
        self._unit_edit.setEnabled(False)

    def _on_accept(self) -> None:
        target_name = self.target_name
        if not target_name:
            self.accept()
            return
        if self.selected_target_id is not None:
            self.accept()
            return
        if not self.target_unit:
            QMessageBox.warning(self, "Validation", "Unit is required for a new target.")
            return
        self.accept()

    @property
    def target_name(self) -> str:
        """Return the selected or newly entered target name."""
        target = self._selected_target()
        if target is not None:
            return target.name
        return self._target_combo.currentText().strip()

    @property
    def target_unit(self) -> str:
        """Return the unit entered for a new target, or the selected target unit."""
        return self._unit_edit.text().strip()

    @property
    def selected_target_id(self) -> str | None:
        """Return the selected existing target id, or ``None`` for a new target."""
        current_index = self._target_combo.currentIndex()
        if current_index <= 0:
            return None
        current_text = self._target_combo.currentText().strip()
        if current_text != self._target_combo.itemText(current_index).strip():
            return None
        target_id = self._target_combo.currentData()
        return None if target_id is None else str(target_id)

    def _selected_target(self) -> TargetDefinition | None:
        selected_target_id = self.selected_target_id
        if selected_target_id is None:
            return None
        return self._targets_by_id.get(selected_target_id)


class AssignTargetValueDialog(QDialog):
    """Dialog for assigning one selected target to one ROI."""

    def __init__(
        self,
        target: TargetDefinition,
        rois: list[TrainingROI],
        current_assignments: dict[tuple[str, str], TargetAssignment],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Assign Target Value")
        self._target = target
        self._current_assignments = current_assignments

        layout = QFormLayout(self)
        layout.addRow(
            "Target:", QLabel(f"{target.name} ({target.unit})" if target.unit else target.name)
        )

        self._roi_combo = QComboBox()
        for roi in rois:
            label = f"{roi.name} [{roi.roi_id}]" if roi.name else roi.roi_id
            self._roi_combo.addItem(label, roi.roi_id)
        layout.addRow("ROI:", self._roi_combo)

        self._value_edit = QLineEdit()
        self._value_edit.setPlaceholderText("required")
        layout.addRow("Value:", self._value_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._roi_combo.currentIndexChanged.connect(self._prefill_assignment)
        self._prefill_assignment()

    def _prefill_assignment(self, _index: int = 0) -> None:
        assignment = self._current_assignments.get((self.roi_id, self._target.target_id))
        self._value_edit.setText("" if assignment is None else str(assignment.value))

    def _on_accept(self) -> None:
        text = self._value_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "Validation", "Value is required.")
            return
        if _finite_float_from_text(text) is None:
            QMessageBox.warning(self, "Validation", f"'{text}' is not a finite numeric value.")
            return
        self.accept()

    @property
    def roi_id(self) -> str:
        """Return the selected ROI id for the target assignment."""
        return str(self._roi_combo.currentData())

    @property
    def value(self) -> float:
        """Return the finite target value entered by the user."""
        value = _finite_float_from_text(self._value_edit.text().strip())
        if value is None:
            log_and_raise("AssignTargetValueDialog value requires finite numeric text")
        return value


class FitModelRunDialog(QDialog):
    """Dialog for selecting one sample set and model-fit configuration."""

    def __init__(
        self,
        sample_sets: list[tuple[str, str]],
        *,
        default_sample_set_id: str | None,
        default_signal_unit: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fit Model Run")

        layout = QFormLayout(self)

        self._sample_set_combo = QComboBox()
        for sample_set_id, label in sample_sets:
            self._sample_set_combo.addItem(label, sample_set_id)
        layout.addRow("Sample set:", self._sample_set_combo)

        self._model_type_combo = QComboBox()
        self._model_type_combo.addItems(["PLS", "PCR"])
        layout.addRow("Model:", self._model_type_combo)

        self._components_spin = QSpinBox()
        self._components_spin.setRange(1, 128)
        self._components_spin.setValue(6)
        layout.addRow("Components:", self._components_spin)

        self._signal_unit_combo = QComboBox()
        self._signal_unit_combo.addItems(["DN", "Reflectance", "Absorbance"])
        current_signal_index = self._signal_unit_combo.findText(default_signal_unit)
        if current_signal_index >= 0:
            self._signal_unit_combo.setCurrentIndex(current_signal_index)
        layout.addRow("Signal unit:", self._signal_unit_combo)

        self._cv_combo = QComboBox()
        self._cv_combo.addItem("Auto grouped (7 folds or leave-one-ROI-out)", "auto")
        self._cv_combo.addItem("Leave-one-ROI-out", "logo")
        self._cv_combo.addItem("7 folds", 7)
        layout.addRow("Validation:", self._cv_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        if default_sample_set_id is not None:
            current_index = self._sample_set_combo.findData(default_sample_set_id)
            if current_index >= 0:
                self._sample_set_combo.setCurrentIndex(current_index)

    def _on_accept(self) -> None:
        if self._sample_set_combo.currentIndex() < 0:
            QMessageBox.warning(self, "Validation", "Select a sample set first.")
            return
        self.accept()

    @property
    def sample_set_id(self) -> str:
        """Return the selected sample-set id."""
        return str(self._sample_set_combo.currentData())

    @property
    def model_type(self) -> str:
        """Return the selected model family."""
        return self._model_type_combo.currentText()

    @property
    def n_components(self) -> int:
        """Return the requested latent component count."""
        return int(self._components_spin.value())

    @property
    def signal_unit(self) -> str:
        """Return the signal unit requested for fitting."""
        return self._signal_unit_combo.currentText()

    @property
    def cv(self) -> int | str:
        """Return the selected grouped-validation policy."""
        return self._cv_combo.currentData()


class ModelingView(QWidget):
    """Modeling controls for target editing, sample-set builds, and model fitting."""

    view_state_changed = pyqtSignal()

    def __init__(
        self,
        viewer_state: ViewerState,
        project_state: ProjectState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = viewer_state
        self.project_state = project_state
        self._selected_target_id: str | None = None
        self._selected_model_run_id: str | None = None
        self._last_sample_set_id: str | None = None
        self._job_thread: QThread | None = None
        self._job_worker: BackgroundWorker | None = None
        # Hold the relay strongly so Qt callbacks remain connected for the full job lifetime.
        self._job_relay: object | None = None
        self._cached_diagnostics_model_key: tuple[str, str] | None = None
        self._cached_diagnostics_model_artifact: dict[str, Any] | None = None
        # Model runs are immutable, so their CV outputs are read once per run.
        self._cached_cv_outputs_key: tuple[str, str] | None = None
        self._cached_cv_outputs: dict[str, Any] | None = None
        self._cached_workspace_diagnostics_token: tuple[Any, ...] | None = None
        self._cached_workspace_diagnostics_payload: dict[str, Any] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_group())

        self.project_state.project_loaded.connect(self._invalidate_diagnostics_cache)
        self.project_state.project_loaded.connect(self._sync_selected_target)
        self.project_state.datasets_changed.connect(self._update_readiness_status)
        self.project_state.calibrations_changed.connect(self._update_readiness_status)
        self.project_state.targets_changed.connect(self._sync_selected_target)
        self.project_state.rois_changed.connect(self._invalidate_diagnostics_cache)
        self.project_state.rois_changed.connect(self._update_readiness_status)
        self.project_state.target_assignments_changed.connect(self._update_readiness_status)
        self.project_state.sample_sets_changed.connect(self._invalidate_diagnostics_cache)
        self.project_state.sample_sets_changed.connect(self._update_model_status)
        self.project_state.model_runs_changed.connect(self._invalidate_diagnostics_cache)
        self.project_state.model_runs_changed.connect(self._update_model_status)
        self.project_state.active_model_changed.connect(self._on_active_model_changed)
        self.state.units_changed.connect(self._update_readiness_status)
        self._sync_selected_target()

    def _build_group(self) -> QGroupBox:
        box = QGroupBox("Modeling")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        action_panel = QWidget(box)
        action_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        action_grid = QGridLayout(action_panel)
        action_grid.setContentsMargins(0, 0, 0, 0)
        action_grid.setHorizontalSpacing(8)
        action_grid.setVerticalSpacing(4)

        def configure_status(label: QLabel) -> None:
            label.setStyleSheet(secondary_text_style())
            label.setWordWrap(True)
            label.setMaximumWidth(132)

        def configure_action(button: QToolButton) -> None:
            button.setMinimumWidth(132)
            button.setMaximumWidth(132)

        self.target_status_label = QLabel("Target: none")
        configure_status(self.target_status_label)
        action_grid.addWidget(self.target_status_label, 0, 0)

        self._sample_set_status = QLabel("Sample set: none")
        configure_status(self._sample_set_status)
        action_grid.addWidget(self._sample_set_status, 0, 1)

        self._readiness_status = QLabel("Ready: define or select a target.")
        configure_status(self._readiness_status)
        action_grid.addWidget(self._readiness_status, 2, 0)

        self._model_status = QLabel("Model: none")
        configure_status(self._model_status)
        action_grid.addWidget(self._model_status, 2, 1)

        self._btn_target = QToolButton()
        self._btn_target.setText("Target...")
        configure_action(self._btn_target)
        self._btn_target.clicked.connect(self.select_target_btn_clicked)
        action_grid.addWidget(self._btn_target, 1, 0)

        self._btn_build_sample_set = QToolButton()
        self._btn_build_sample_set.setText("Build sample set")
        configure_action(self._btn_build_sample_set)
        self._btn_build_sample_set.clicked.connect(self.build_sample_set_btn_clicked)
        action_grid.addWidget(self._btn_build_sample_set, 1, 1)

        self._btn_assign_target = QToolButton()
        self._btn_assign_target.setText("Assign target")
        configure_action(self._btn_assign_target)
        self._btn_assign_target.clicked.connect(self.assign_target_btn_clicked)
        action_grid.addWidget(self._btn_assign_target, 3, 0)

        self._btn_fit_model = QToolButton()
        self._btn_fit_model.setText("Fit model")
        configure_action(self._btn_fit_model)
        self._btn_fit_model.clicked.connect(self.fit_model_btn_clicked)
        action_grid.addWidget(self._btn_fit_model, 3, 1)

        self._service_status = BusyStatusWidget(idle_text="Service: idle", parent=self)
        self._service_status.setMaximumWidth(272)
        self._service_status.label.setWordWrap(True)
        action_grid.addWidget(self._service_status, 4, 0, 1, 2)

        layout.addWidget(action_panel)
        layout.addStretch(1)

        return box

    def visible_targets(self) -> list[TargetDefinition]:
        return sorted(
            self.project_state.targets.values(),
            key=lambda target: target.name,
        )

    def current_target(self) -> TargetDefinition | None:
        if self._selected_target_id is None:
            return None
        target = self.project_state.targets.get(self._selected_target_id)
        if target is None:
            return None
        return target

    @property
    def selected_target_id(self) -> str | None:
        return self._selected_target_id

    @selected_target_id.setter
    def selected_target_id(self, value: str | None) -> None:
        self._selected_target_id = None if value is None else str(value)

    def select_target_btn_clicked(self) -> None:
        if self.project_state.project_path is None:
            QMessageBox.warning(
                self,
                "No project open",
                "No project open. Use Project > New or Project > Open first.",
            )
            return
        dlg = TargetSelectionDialog(self.visible_targets(), self._selected_target_id, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if not dlg.target_name:
            self._selected_target_id = None
            self._update_target_status()
            self._update_model_status()
            return
        if dlg.selected_target_id is not None:
            self._selected_target_id = dlg.selected_target_id
            self._update_target_status()
            self._update_model_status()
            return

        target = TargetDefinition(
            target_id=str(uuid.uuid4()),
            name=dlg.target_name,
            unit=dlg.target_unit,
            value_type="float",
            missing_value_policy="allowed",
            created_at=now_iso(),
        )
        self.project_state.define_target(target)
        self._selected_target_id = target.target_id
        self._update_target_status()
        self._update_model_status()

    def assign_target_btn_clicked(self) -> None:
        target = self.current_target()
        if self.project_state.project_path is None:
            QMessageBox.warning(
                self,
                "No project open",
                "No project open. Use Project > New or Project > Open first.",
            )
            return
        if target is None:
            QMessageBox.warning(self, "No target selected", "Select or create a target first.")
            return
        rois = list(self.project_state.rois.values())
        if not rois:
            QMessageBox.information(
                self, "No ROIs", "Draw a training ROI before assigning target values."
            )
            return
        dlg = AssignTargetValueDialog(
            target, rois, self.project_state.target_assignments, parent=self
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.project_state.assign_target_to_roi(
            TargetAssignment(
                roi_id=dlg.roi_id,
                target_id=target.target_id,
                value=dlg.value,
                assigned_at=now_iso(),
            )
        )

    def build_sample_set_btn_clicked(self) -> None:
        if self._job_thread is not None:
            return
        if self.project_state.project_path is None:
            QMessageBox.warning(
                self, "No project open", "Open a project before building a sample set."
            )
            return
        target = self.current_target()
        if target is None:
            QMessageBox.warning(
                self, "No target selected", "Select a target before building a sample set."
            )
            return
        if not self.project_state.rois:
            QMessageBox.information(
                self, "No ROIs", "Draw at least one ROI before building a sample set."
            )
            return

        recipe_id = str(uuid.uuid4())
        recipe_dict = self.state.recipe.to_dict()
        project_path = self._require_project_path()
        rois = dict(self.project_state.rois)
        target_assignments = dict(self.project_state.target_assignments)

        def job():
            return build_project_sample_set(
                project_path=project_path,
                rois=rois,
                target_assignments=target_assignments,
                target_id=target.target_id,
                preprocessing_recipe_id=recipe_id,
            )

        self._set_busy(True, "Building sample set...")
        self._job_thread, self._job_worker, self._job_relay = start_background_job(
            owner=self,
            fn=job,
            on_success=lambda sample_set_data: self._on_build_sample_set_success(
                sample_set_data=sample_set_data,
                recipe_id=recipe_id,
                recipe_dict=recipe_dict,
                project_path=project_path,
            ),
            on_error=lambda msg: self._on_job_error("Build sample set failed", msg),
            on_finished=self._clear_job,
        )

    def fit_model_btn_clicked(self) -> None:
        if self._job_thread is not None:
            return
        if self.project_state.project_path is None:
            QMessageBox.warning(self, "No project open", "Open a project before fitting a model.")
            return
        if not self.project_state.sample_sets:
            QMessageBox.information(
                self,
                "No sample sets",
                "Build a sample set before fitting a model.",
            )
            return

        dlg = FitModelRunDialog(
            self._sample_set_choices(),
            default_sample_set_id=self._default_sample_set_id(),
            default_signal_unit=self.state.y_unit,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        sample_set = self.project_state.sample_sets[dlg.sample_set_id]
        preflight_error = self._validate_fit_preflight(
            sample_set_id=dlg.sample_set_id,
            cv=dlg.cv,
            required_signal_unit=dlg.signal_unit,
        )
        if preflight_error is not None:
            QMessageBox.warning(self, "Fit model failed", preflight_error)
            return

        signal_domain = self._build_signal_domain_contract(
            sample_set_id=dlg.sample_set_id,
            required_signal_unit=dlg.signal_unit,
        )
        project_path = self._require_project_path()
        target = self.project_state.targets.get(sample_set.target_id)
        target_name = None if target is None else target.name
        resolved_signal_domains = dict(self.project_state.signal_domains)
        resolved_signal_domains[signal_domain.signal_domain_id] = signal_domain
        datasets = dict(self.project_state.datasets)
        calibrations = dict(self.project_state.calibrations)
        rois = dict(self.project_state.rois)
        target_assignments = dict(self.project_state.target_assignments)
        recipes = dict(self.project_state.recipes)

        from ..modeling import fit_model_run as fit_project_model_run

        def job():
            return fit_project_model_run(
                project_path=project_path,
                sample_set=sample_set,
                datasets=datasets,
                calibrations=calibrations,
                rois=rois,
                target_assignments=target_assignments,
                recipes=recipes,
                signal_domains=resolved_signal_domains,
                signal_domain_id=signal_domain.signal_domain_id,
                model_type=dlg.model_type,
                hyperparameters={"n_components": dlg.n_components},
                cv=dlg.cv,
                target_name=target_name,
            )

        self._set_busy(True, "Fitting model...")
        self._job_thread, self._job_worker, self._job_relay = start_background_job(
            owner=self,
            fn=job,
            on_success=lambda model_run_result: self._on_fit_model_success(
                model_run_result=model_run_result,
                signal_domain=signal_domain,
                project_path=project_path,
            ),
            on_error=lambda msg: self._on_job_error("Fit model failed", msg),
            on_finished=self._clear_job,
        )

    def _validate_fit_preflight(
        self,
        *,
        sample_set_id: str,
        cv: int | str,
        required_signal_unit: str,
    ) -> str | None:
        sample_set = self.project_state.sample_sets[sample_set_id]
        stale_message = self.project_state.stale_sample_set_message(sample_set_id)
        if stale_message is not None:
            return stale_message
        n_roi_groups = self._sample_set_group_count(sample_set)
        cv_error = self._validate_cv_preflight(n_roi_groups=n_roi_groups, cv=cv)
        if cv_error is not None:
            return cv_error
        signal_domain_error = self._validate_signal_domain_preflight(
            sample_set_id=sample_set_id,
            required_signal_unit=required_signal_unit,
        )
        if signal_domain_error is not None:
            return signal_domain_error
        return None

    def _sample_set_choices(self) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = []
        for sample_set_id, sample_set in self.project_state.sample_sets.items():
            target = self.project_state.targets.get(sample_set.target_id)
            target_label = target.name if target is not None else sample_set.target_id
            label = f"{target_label} [{sample_set_id}]"
            if self.project_state.missing_roi_ids_for_sample_set(sample_set_id):
                label += " (stale: deleted ROI)"
            choices.append((sample_set_id, label))
        return choices

    def _default_sample_set_id(self) -> str | None:
        current_target = self.current_target()
        if self._last_sample_set_id in self.project_state.sample_sets and (
            current_target is None
            or self.project_state.sample_sets[self._last_sample_set_id].target_id
            == current_target.target_id
        ):
            return self._last_sample_set_id
        if current_target is not None:
            for sample_set in self.project_state.sample_sets.values():
                if sample_set.target_id == current_target.target_id:
                    return sample_set.sample_set_id
            return None
        return next(iter(self.project_state.sample_sets), None)

    def _build_signal_domain_contract(
        self,
        *,
        sample_set_id: str,
        required_signal_unit: str,
    ) -> SignalDomainContract:
        sample_set = self.project_state.sample_sets[sample_set_id]
        stale_message = self.project_state.stale_sample_set_message(sample_set_id)
        if stale_message is not None:
            log_and_raise("{}", stale_message)
        # Opening the project validated that every ROI's dataset and calibration exist.
        roi = self.project_state.rois[sample_set.roi_ids[0]]
        return SignalDomainContract.for_training_data(
            dataset=self.project_state.datasets[roi.dataset_id],
            calibration=self.project_state.calibrations[roi.calibration_id],
            required_signal_unit=required_signal_unit,
        )

    def _sync_selected_target(self) -> None:
        visible_targets = self.visible_targets()
        visible_target_ids = {target.target_id for target in visible_targets}
        if self._selected_target_id not in visible_target_ids:
            self._selected_target_id = None if not visible_targets else visible_targets[0].target_id
        self._update_target_status()
        self._update_model_status()
        self._update_readiness_status()

    def select_model_run(self, model_run_id: str | None) -> None:
        selected_model_run_id = None if model_run_id is None else str(model_run_id)
        if selected_model_run_id not in self.project_state.model_runs:
            selected_model_run_id = None
        if selected_model_run_id == self._selected_model_run_id:
            return
        self._selected_model_run_id = selected_model_run_id
        self._invalidate_diagnostics_cache()
        self._update_model_status()

    def workspace_summary(self) -> dict[str, str]:
        model_run = self._resolved_workspace_model_run()
        sample_set = self._resolved_workspace_sample_set(model_run=model_run)
        target = self._resolved_workspace_target(model_run=model_run, sample_set=sample_set)
        target_text = "none"
        if target is not None:
            target_text = target.name if not target.unit else f"{target.name} ({target.unit})"

        roi_count = 0
        pixel_count = 0
        if sample_set is not None:
            roi_count = len(sample_set.roi_ids)
            pixel_count = sum(
                int(self.project_state.rois[roi_id].n_pixels)
                for roi_id in sample_set.roi_ids
                if roi_id in self.project_state.rois
            )

        model_text = "none"
        components_text = "-"
        cv_text = "-"
        r2_text = "-"
        rmsecv_text = "-"
        bias_text = "-"
        if model_run is not None:
            model_text = f"{model_run.model_type} [{model_run.model_run_id}]"
            n_components = model_run.hyperparameters.get("n_components")
            if n_components is not None:
                components_text = str(n_components)
            cv_value = model_run.cv_spec.get("cv")
            if cv_value is None and len(model_run.cv_spec) == 1:
                cv_value = next(iter(model_run.cv_spec.values()))
            if cv_value is not None:
                cv_text = str(cv_value)
            if "r2" in model_run.metrics:
                r2_text = f"{float(model_run.metrics['r2']):.4g}"
            if "rmsecv" in model_run.metrics:
                rmsecv_text = f"{float(model_run.metrics['rmsecv']):.4g}"
            if "bias" in model_run.metrics:
                bias_text = f"{float(model_run.metrics['bias']):.4g}"

        return {
            "title": "Model Diagnostics",
            "target": target_text,
            "sample_set": "none" if sample_set is None else sample_set.sample_set_id,
            "model": model_text,
            "components": components_text,
            "cv": cv_text,
            "roi_count": str(roi_count),
            "pixel_count": str(pixel_count),
            "r2": r2_text,
            "rmsecv": rmsecv_text,
            "bias": bias_text,
            "readiness": self._readiness_status.text(),
        }

    def workspace_diagnostics_payload(self) -> dict[str, Any] | None:
        """Return one stable diagnostics payload for the selected Model workspace run.

        The payload object is cached by the selected run and loaded artifact
        identities so repeated shell refreshes can treat unchanged diagnostics as
        a no-op redraw.
        """
        model_run = self._resolved_workspace_model_run()
        if model_run is None:
            self._cached_workspace_diagnostics_token = None
            self._cached_workspace_diagnostics_payload = None
            return None

        sample_set = self._resolved_workspace_sample_set(model_run=model_run)
        target = self._resolved_workspace_target(model_run=model_run, sample_set=sample_set)
        target_unit = "" if target is None else target.unit

        cv_outputs = model_run.artifact_refs.get("cv_outputs")
        if (
            not isinstance(cv_outputs, dict)
            or cv_outputs.get("y_true_roi") is None
            or cv_outputs.get("y_pred_roi") is None
        ):
            cv_outputs = self._load_model_cv_outputs(model_run)
        if not isinstance(cv_outputs, dict):
            self._cached_workspace_diagnostics_token = None
            self._cached_workspace_diagnostics_payload = None
            return None
        y_true_roi = cv_outputs.get("y_true_roi")
        y_pred_roi = cv_outputs.get("y_pred_roi")
        if y_true_roi is None or y_pred_roi is None:
            self._cached_workspace_diagnostics_token = None
            self._cached_workspace_diagnostics_payload = None
            return None

        artifact = self._load_model_artifact(model_run)
        token = (
            model_run.model_run_id,
            target_unit,
            id(cv_outputs),
            id(y_true_roi),
            id(y_pred_roi),
            id(cv_outputs.get("roi_ids")),
            id(artifact),
        )
        if self._cached_workspace_diagnostics_token == token:
            return self._cached_workspace_diagnostics_payload

        payload: dict[str, Any] = {
            "target_unit": target_unit,
            "y_true_roi": np.asarray(y_true_roi, dtype=np.float32),
            "y_pred_roi": np.asarray(y_pred_roi, dtype=np.float32),
        }
        roi_ids = cv_outputs.get("roi_ids")
        if roi_ids is not None:
            payload["roi_ids"] = np.asarray(roi_ids, dtype=object).reshape(-1)
        scores_roi = self._extract_roi_scores(model_run, artifact)
        if scores_roi is not None:
            payload["scores_roi"] = scores_roi
        if artifact is not None:
            coefficients = artifact.get("coef_")
            if coefficients is not None:
                coef_arr = np.asarray(coefficients, dtype=np.float32).squeeze()
                if coef_arr.ndim == 2:
                    coef_arr = coef_arr[:, 0]
                if coef_arr.ndim == 1:
                    payload["coefficients"] = coef_arr
            loadings = artifact.get("x_loadings_")
            if loadings is None:
                loadings = artifact.get("loadings_")
            if loadings is not None:
                loadings_arr = np.asarray(loadings, dtype=np.float32).squeeze()
                if loadings_arr.ndim == 2:
                    loadings_arr = loadings_arr[:, 0]
                if loadings_arr.ndim == 1:
                    payload["loadings"] = loadings_arr
            wavelengths = artifact.get("wavelengths")
            if wavelengths is not None:
                spectral_axis = np.asarray(wavelengths, dtype=np.float32).squeeze()
                if spectral_axis.ndim == 1:
                    payload["spectral_axis"] = spectral_axis
        self._cached_workspace_diagnostics_token = token
        self._cached_workspace_diagnostics_payload = payload
        return payload

    def _load_model_cv_outputs(self, model_run: ModelRun) -> dict[str, Any] | None:
        project_path = self.project_state.project_path
        if not project_path:
            return None
        cache_key = (project_path, model_run.model_run_id)
        if self._cached_cv_outputs_key == cache_key:
            return self._cached_cv_outputs
        try:
            artifact_refs = load_model_run_artifact_refs(
                project_path,
                model_run.model_run_id,
                keys=("cv_outputs",),
            )
        except Exception as exc:
            logger.warning(
                "Could not load model CV outputs '{}': {}",
                model_run.model_run_id,
                exc,
            )
            return None
        cv_outputs = artifact_refs.get("cv_outputs")
        self._cached_cv_outputs_key = cache_key
        self._cached_cv_outputs = cv_outputs if isinstance(cv_outputs, dict) else None
        return self._cached_cv_outputs

    def _load_model_artifact(self, model_run) -> dict[str, Any] | None:
        project_path = self.project_state.project_path
        if not project_path:
            return None
        cache_key = (project_path, model_run.model_run_id)
        if self._cached_diagnostics_model_key == cache_key:
            return self._cached_diagnostics_model_artifact
        try:
            artifact_refs = load_model_run_artifact_refs(
                project_path,
                model_run.model_run_id,
                keys=("prediction_model", "diagnostics"),
            )
        except Exception as exc:
            logger.warning(
                "Could not load model diagnostics artifact '{}': {}",
                model_run.model_run_id,
                exc,
            )
            return None
        artifact = {}
        prediction_model = artifact_refs.get("prediction_model")
        diagnostics = artifact_refs.get("diagnostics")
        if isinstance(prediction_model, dict):
            artifact.update(prediction_model)
        if isinstance(diagnostics, dict):
            artifact.update(diagnostics)
        if not artifact:
            return None
        self._cached_diagnostics_model_key = cache_key
        self._cached_diagnostics_model_artifact = artifact
        return artifact

    def _extract_roi_scores(self, model_run, artifact: dict[str, Any] | None) -> np.ndarray | None:
        if artifact is None:
            return None
        sample_set = self.project_state.sample_sets.get(model_run.sample_set_id)
        if sample_set is None:
            return None
        scores = None
        if "x_scores_" in artifact:
            scores = artifact["x_scores_"]
        elif "scores_" in artifact:
            scores = artifact["scores_"]
        if scores is None:
            return None

        scores_arr = np.asarray(scores, dtype=np.float32)
        if scores_arr.ndim != 2 or scores_arr.shape[0] == 0:
            return None

        counts: list[int] = []
        for roi_id in sample_set.roi_ids:
            roi = self.project_state.rois.get(roi_id)
            if roi is None:
                return None
            n_pixels = int(getattr(roi, "n_pixels", 0))
            if n_pixels <= 0:
                return None
            counts.append(n_pixels)

        counts_arr = np.asarray(counts, dtype=np.int64)
        total_pixels = int(counts_arr.sum())
        if total_pixels > scores_arr.shape[0]:
            logger.warning(
                "Model diagnostics scores for '{}' are shorter than sample-set ROI pixel counts",
                model_run.model_run_id,
            )
            return None
        if total_pixels != scores_arr.shape[0]:
            logger.warning(
                "Model diagnostics scores for '{}' have {} rows but sample-set ROI counts consume {}",
                model_run.model_run_id,
                scores_arr.shape[0],
                total_pixels,
            )
        starts = np.concatenate(([0], np.cumsum(counts_arr[:-1], dtype=np.int64)))
        sums = np.add.reduceat(scores_arr[:total_pixels], starts, axis=0)
        return (sums / counts_arr[:, None]).astype(np.float32, copy=False)

    def _update_target_status(self) -> None:
        target = self.current_target()
        if target is None:
            self.target_status_label.setText("Target: none")
            self.view_state_changed.emit()
            return
        label = target.name if not target.unit else f"{target.name} ({target.unit})"
        self.target_status_label.setText(f"Target: {label}")
        self.view_state_changed.emit()

    def _update_model_status(self) -> None:
        model_run = self._resolved_workspace_model_run()
        sample_set = self._resolved_workspace_sample_set(model_run=model_run)
        sample_set_id = None if sample_set is None else sample_set.sample_set_id
        if sample_set_id is None:
            self._sample_set_status.setText("Sample set: none")
        else:
            suffix = (
                " (stale: deleted ROI)"
                if self.project_state.missing_roi_ids_for_sample_set(sample_set_id)
                else ""
            )
            self._sample_set_status.setText(f"Sample set: {sample_set_id}{suffix}")

        if model_run is None:
            self._model_status.setText("Model: none")
        else:
            metric = model_run.metrics.get("rmsecv")
            suffix = "" if metric is None else f" | RMSECV={float(metric):.4g}"
            self._model_status.setText(f"Model: {model_run.model_run_id}{suffix}")
        self._update_readiness_status()
        self.view_state_changed.emit()

    def _on_active_model_changed(self, model_run_id: str) -> None:
        logger.info("Active model -> {}", model_run_id)
        self._invalidate_diagnostics_cache()
        self._update_model_status()

    def _invalidate_diagnostics_cache(self) -> None:
        self._cached_diagnostics_model_key = None
        self._cached_diagnostics_model_artifact = None
        self._cached_cv_outputs_key = None
        self._cached_cv_outputs = None
        self._cached_workspace_diagnostics_token = None
        self._cached_workspace_diagnostics_payload = None

    def _require_project_path(self) -> str:
        project_path = self.project_state.project_path
        if project_path is not None:
            return project_path
        log_and_raise(
            "ModelingView requires an open project before starting a background job.", error=RuntimeError,
        )

    def _update_readiness_status(self) -> None:
        model_run = self._resolved_workspace_model_run()
        if model_run is not None:
            message = f"Ready: model {model_run.model_run_id} is available for prediction."
            stale_message = self.project_state.stale_sample_set_message(model_run.sample_set_id)
            if stale_message is not None:
                message += f" {stale_message}"
            self._readiness_status.setText(message)
            return

        target = self.current_target()
        if target is None:
            self._readiness_status.setText("Ready: define or select a target.")
            return

        labeled_roi_ids = {
            roi_id
            for roi_id in self.project_state.rois
            if (roi_id, target.target_id) in self.project_state.target_assignments
        }
        if not self.project_state.rois:
            self._readiness_status.setText("Ready: draw ROIs in Explore.")
            return
        if not labeled_roi_ids:
            self._readiness_status.setText("Ready: assign target values to ROIs.")
            return
        if self._default_sample_set_id() is None:
            labeled_count = len(labeled_roi_ids)
            label = "ROI" if labeled_count == 1 else "ROIs"
            self._readiness_status.setText(
                f"Ready: build a sample set from {labeled_count} labeled {label}."
            )
            return

        sample_set_id = self._default_sample_set_id()
        sample_set = (
            None if sample_set_id is None else self.project_state.sample_sets.get(sample_set_id)
        )
        if model_run is None:
            details: list[str] = []
            if sample_set is not None:
                roi_count = len(sample_set.roi_ids)
                roi_label = "ROI" if roi_count == 1 else "ROIs"
                n_roi_groups = self._sample_set_group_count(sample_set)
                group_label = "group" if n_roi_groups == 1 else "groups"
                details.append(
                    f"{roi_count} labeled {roi_label} across {n_roi_groups} ROI {group_label}"
                )
                stale_message = self.project_state.stale_sample_set_message(
                    sample_set.sample_set_id
                )
                if stale_message is not None:
                    details.append(stale_message)
                else:
                    if n_roi_groups < 2:
                        details.append("Grouped validation unavailable")
                    elif n_roi_groups < 7:
                        details.append("Auto validation will use leave-one-ROI-out")
                    signal_domain_error = self._validate_signal_domain_preflight(
                        sample_set_id=sample_set.sample_set_id,
                        required_signal_unit=str(self.state.y_unit_effective),
                    )
                    if signal_domain_error is not None:
                        details.append(
                            f"Current signal unit {self.state.y_unit_effective} is incompatible: {signal_domain_error}"
                        )
            message = "Ready: fit a model from the selected sample set."
            if details:
                message += " " + " ".join(
                    detail if detail.endswith(".") else f"{detail}." for detail in details
                )
            self._readiness_status.setText(message)
            return

    def _job_result_is_stale(self, project_path: str, *, what: str) -> bool:
        """Return True (and inform the user) when the project changed mid-job."""
        if self.project_state.project_path == project_path:
            return False
        msg = "Discarding {} computed for project '{}' because project '{}' is now open"
        logger.warning(msg, what, project_path, self.project_state.project_path)
        self._service_status.set_idle("Service: idle")
        QMessageBox.information(
            self,
            "Result discarded",
            msg.format(what, project_path, self.project_state.project_path) + ".",
        )
        return True

    def _on_build_sample_set_success(
        self,
        *,
        sample_set_data: Any,
        recipe_id: str,
        recipe_dict: dict[str, object],
        project_path: str,
    ) -> None:
        if self._job_result_is_stale(project_path, what="sample set"):
            return
        try:
            self.project_state.register_recipe(recipe_id, recipe_dict)
            self.project_state.create_sample_set(sample_set_data.sample_set)
        except Exception as exc:
            logger.exception("Persisting the built sample set failed")
            self._on_job_error("Build sample set failed", str(exc))
            return
        self._last_sample_set_id = sample_set_data.sample_set.sample_set_id
        self._service_status.set_idle(
            f"Service: sample set {sample_set_data.sample_set.sample_set_id}"
        )
        self._update_model_status()

    def _on_fit_model_success(
        self,
        *,
        model_run_result: Any,
        signal_domain: SignalDomainContract,
        project_path: str,
    ) -> None:
        if self._job_result_is_stale(project_path, what="model run"):
            return
        try:
            self.project_state.finalize_model_run_result(
                model_run_result=model_run_result,
                signal_domain=signal_domain,
            )
        except Exception as exc:
            logger.exception("Persisting the fitted model run failed")
            self._on_job_error("Fit model failed", str(exc))
            return
        self._service_status.set_idle(f"Service: model {model_run_result.model_run.model_run_id}")
        self._update_model_status()

    def _on_job_error(self, title: str, message: str) -> None:
        self._service_status.set_idle("Service: idle")
        logger.error("{}: {}", title, message)
        QMessageBox.warning(self, title, message)

    def service_status_snapshot(self) -> tuple[str, bool]:
        """Return the current service label text and busy/progress visibility."""
        return self._service_status.label.text(), self._service_status.progress.isVisible()

    def _set_busy(self, busy: bool, text: str | None = None) -> None:
        self._btn_target.setEnabled(not busy)
        self._btn_assign_target.setEnabled(not busy)
        self._btn_build_sample_set.setEnabled(not busy)
        self._btn_fit_model.setEnabled(not busy)
        if busy:
            self._service_status.set_busy(text or "Working...")
        elif text is not None:
            self._service_status.set_idle(text)
        else:
            self._service_status.set_idle()

    def _clear_job(self) -> None:
        self._job_thread = None
        self._job_worker = None
        self._job_relay = None
        self._btn_target.setEnabled(True)
        self._btn_assign_target.setEnabled(True)
        self._btn_build_sample_set.setEnabled(True)
        self._btn_fit_model.setEnabled(True)

    def _selected_workspace_model_run(self) -> ModelRun | None:
        if self._selected_model_run_id is None:
            return None
        model_run = self.project_state.model_runs.get(self._selected_model_run_id)
        if model_run is None:
            self._selected_model_run_id = None
            return None
        return model_run

    def _resolved_workspace_model_run(self) -> ModelRun | None:
        selected_model_run = self._selected_workspace_model_run()
        if selected_model_run is not None:
            return selected_model_run

        model_run_id = self.project_state.active_model_name
        if model_run_id is None and self.project_state.model_runs:
            model_run_id = next(reversed(self.project_state.model_runs))
        if model_run_id is None:
            return None
        model_run = self.project_state.model_runs.get(model_run_id)
        if model_run is None:
            return None

        sample_set_id = self._default_sample_set_id()
        current_target = self.current_target()
        if (
            sample_set_id is not None
            and current_target is not None
            and model_run.sample_set_id != sample_set_id
        ):
            return None
        return model_run

    def _resolved_workspace_sample_set(self, *, model_run: ModelRun | None) -> SampleSet | None:
        if model_run is not None:
            return self.project_state.sample_sets.get(model_run.sample_set_id)
        sample_set_id = self._default_sample_set_id()
        if sample_set_id is None:
            return None
        return self.project_state.sample_sets.get(sample_set_id)

    def _resolved_workspace_target(
        self,
        *,
        model_run: ModelRun | None,
        sample_set: SampleSet | None,
    ) -> TargetDefinition | None:
        del model_run
        if sample_set is not None:
            target = self.project_state.targets.get(sample_set.target_id)
            if target is not None:
                return target
        return self.current_target()

    @staticmethod
    def _sample_set_group_count(sample_set: Any) -> int:
        return len({str(roi_id) for roi_id in sample_set.roi_ids})

    @staticmethod
    def _validate_cv_preflight(*, n_roi_groups: int, cv: int | str) -> str | None:
        if cv == "auto" and n_roi_groups < 2:
            return "Automatic grouped validation requires at least 2 ROI groups in the selected sample set."
        if cv == "logo" and n_roi_groups < 2:
            return "Leave-one-ROI-out validation requires at least 2 ROI groups in the selected sample set."
        if isinstance(cv, int) and n_roi_groups < cv:
            return f"{cv}-fold grouped validation requires at least {cv} ROI groups in the selected sample set."
        return None

    def _validate_signal_domain_preflight(
        self,
        *,
        sample_set_id: str,
        required_signal_unit: str,
    ) -> str | None:
        stale_message = self.project_state.stale_sample_set_message(sample_set_id)
        if stale_message is not None:
            return stale_message
        try:
            signal_domain = self._build_signal_domain_contract(
                sample_set_id=sample_set_id,
                required_signal_unit=required_signal_unit,
            )
        except ValueError as exc:
            return str(exc)

        sample_set = self.project_state.sample_sets[sample_set_id]
        checked_contexts: set[tuple[str, str]] = set()
        for roi_id in sample_set.roi_ids:
            if roi_id not in self.project_state.rois:
                return f"SampleSet '{sample_set_id}' references missing roi_id '{roi_id}'"
            roi = self.project_state.rois[roi_id]
            context_key = (roi.dataset_id, roi.calibration_id)
            if context_key in checked_contexts:
                continue
            checked_contexts.add(context_key)
            if roi.dataset_id not in self.project_state.datasets:
                return f"ROI '{roi.roi_id}' references unknown dataset_id '{roi.dataset_id}'"
            if roi.calibration_id not in self.project_state.calibrations:
                return (
                    f"ROI '{roi.roi_id}' references unknown calibration_id '{roi.calibration_id}'"
                )
            try:
                check_signal_domain_compatibility(
                    dataset=self.project_state.datasets[roi.dataset_id],
                    calibration=self.project_state.calibrations[roi.calibration_id],
                    signal_domain=signal_domain,
                )
            except ValueError as exc:
                return str(exc)
        return None


__all__ = [
    "AssignTargetValueDialog",
    "FitModelRunDialog",
    "ModelingView",
    "NewROIDialog",
    "TargetSelectionDialog",
]
