"""Read-only project browser for the workspace shell."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFrame, QSizePolicy, QTreeWidget, QTreeWidgetItem, QWidget

from ._state import ProjectState

__all__ = ["ProjectBrowser"]


class ProjectBrowser(QTreeWidget):
    """Persistent read-only browser for the active project workspace."""

    _SECTION_ORDER = [
        ("datasets", "Datasets"),
        ("calibrations", "Calibrations"),
        ("rois", "ROIs"),
        ("targets", "Targets"),
        ("target_assignments", "Target Assignments"),
        ("sample_sets", "Sample Sets"),
        ("signal_domains", "Signal Domains"),
        ("model_runs", "Model Runs"),
        ("predictions", "Predictions"),
    ]
    _KIND_TO_SECTION = {
        "dataset": "datasets",
        "calibration": "calibrations",
        "roi": "rois",
        "target": "targets",
        "target_assignment": "target_assignments",
        "sample_set": "sample_sets",
        "signal_domain": "signal_domains",
        "model_run": "model_runs",
        "prediction": "predictions",
    }

    def __init__(self, project_state: ProjectState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_state = project_state
        self.setObjectName("ProjectBrowser")
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setIndentation(10)
        self.setMinimumWidth(260)
        self.setMaximumWidth(260)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._refresh_pending = False
        self._sections: dict[str, QTreeWidgetItem] = {}
        for key, label in self._SECTION_ORDER:
            item = QTreeWidgetItem([label])
            item.setData(
                0, Qt.ItemDataRole.UserRole, {"kind": "section", "id": key, "label": label}
            )
            item.setExpanded(True)
            self.addTopLevelItem(item)
            self._sections[key] = item

        self.project_state.project_loaded.connect(self.schedule_refresh)
        self.project_state.datasets_changed.connect(self.schedule_refresh)
        self.project_state.calibrations_changed.connect(self.schedule_refresh)
        self.project_state.rois_changed.connect(self.schedule_refresh)
        self.project_state.targets_changed.connect(self.schedule_refresh)
        self.project_state.target_assignments_changed.connect(self.schedule_refresh)
        self.project_state.sample_sets_changed.connect(self.schedule_refresh)
        self.project_state.signal_domains_changed.connect(self.schedule_refresh)
        self.project_state.model_runs_changed.connect(self.schedule_refresh)
        self.project_state.predictions_changed.connect(self.schedule_refresh)
        self.refresh()

    def schedule_refresh(self) -> None:
        """Coalesce repeated browser rebuilds into one event-loop refresh."""
        if self._refresh_pending:
            return
        self._refresh_pending = True
        QTimer.singleShot(0, self._flush_refresh)

    def _flush_refresh(self) -> None:
        self._refresh_pending = False
        self.refresh()

    def refresh(self) -> None:
        """Rebuild browser sections from the current project workspace."""
        selected_payload = self._current_payload()
        for section in self._sections.values():
            section.takeChildren()

        self._populate_datasets()
        self._populate_calibrations()
        self._populate_rois()
        self._populate_targets()
        self._populate_target_assignments()
        self._populate_sample_sets()
        self._populate_signal_domains()
        self._populate_model_runs()
        self._populate_predictions()

        item = self._item_for_payload(selected_payload)
        if item is not None:
            self.setCurrentItem(item)
            return
        self.setCurrentItem(None)

    def select_payload(self, payload: dict[str, str] | None) -> None:
        """Select one section or leaf item by payload."""
        item = self._item_for_payload(payload)
        if item is None:
            self.setCurrentItem(None)
            return
        self.setCurrentItem(item)

    def _current_payload(self) -> dict[str, str] | None:
        item = self.currentItem()
        if item is None:
            return None
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(payload, dict):
            return payload
        return None

    def _item_for_payload(self, payload: dict[str, str] | None) -> QTreeWidgetItem | None:
        if not payload:
            return None
        kind = payload.get("kind")
        object_id = payload.get("id")
        if kind == "section":
            if object_id is None:
                return None
            if object_id in self._sections:
                return self._sections[object_id]
            return None

        if kind is None:
            return None
        section_key = self._KIND_TO_SECTION.get(kind)
        if section_key is None:
            return None
        section = self._sections[section_key]
        for index in range(section.childCount()):
            child = section.child(index)
            if child is None:
                continue
            child_payload = child.data(0, Qt.ItemDataRole.UserRole)
            if self._payload_matches(payload, child_payload):
                return child
        return None

    @staticmethod
    def _payload_matches(
        requested_payload: dict[str, str],
        candidate_payload: object,
    ) -> bool:
        if not isinstance(candidate_payload, dict):
            return False
        if requested_payload.get("kind") != candidate_payload.get("kind"):
            return False
        if requested_payload.get("kind") == "target_assignment":
            return requested_payload.get("target_id") == candidate_payload.get(
                "target_id"
            ) and requested_payload.get("roi_id") == candidate_payload.get("roi_id")
        requested_id = requested_payload.get("id")
        if requested_id is None:
            return False
        return candidate_payload.get("id") == requested_id

    def _populate_datasets(self) -> None:
        section = self._sections["datasets"]
        for dataset_id, dataset in self.project_state.datasets.items():
            label = dataset_id
            if dataset.source_path:
                label = f"{dataset_id} | {Path(dataset.source_path).name}"
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "dataset", "id": dataset_id})
            section.addChild(item)

    def _populate_calibrations(self) -> None:
        section = self._sections["calibrations"]
        for calibration_id, calibration in self.project_state.calibrations.items():
            label = f"{calibration_id} | {calibration.mode}"
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "calibration", "id": calibration_id})
            section.addChild(item)

    def _populate_rois(self) -> None:
        section = self._sections["rois"]
        for roi_id, roi in self.project_state.rois.items():
            label = f"{roi.name} [{roi_id}]" if roi.name else roi_id
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "roi", "id": roi_id})
            section.addChild(item)

    def _populate_targets(self) -> None:
        section = self._sections["targets"]
        for target_id, target in self.project_state.targets.items():
            label = f"{target.name} [{target_id}]"
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "target", "id": target_id})
            section.addChild(item)

    def _populate_target_assignments(self) -> None:
        section = self._sections["target_assignments"]
        assignments = sorted(
            self.project_state.target_assignments.values(),
            key=lambda assignment: (assignment.target_id, assignment.roi_id),
        )
        for assignment in assignments:
            target = self.project_state.targets.get(assignment.target_id)
            roi = self.project_state.rois.get(assignment.roi_id)
            target_label = target.name if target is not None else assignment.target_id
            roi_label = roi.name if roi is not None and roi.name else assignment.roi_id
            label = f"{target_label} -> {roi_label} | {assignment.value:.4g}"
            item = QTreeWidgetItem([label])
            item.setData(
                0,
                Qt.ItemDataRole.UserRole,
                {
                    "kind": "target_assignment",
                    "target_id": assignment.target_id,
                    "roi_id": assignment.roi_id,
                },
            )
            section.addChild(item)

    def _populate_sample_sets(self) -> None:
        section = self._sections["sample_sets"]
        for sample_set_id, sample_set in self.project_state.sample_sets.items():
            label = f"{sample_set_id} | target={sample_set.target_id}"
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "sample_set", "id": sample_set_id})
            section.addChild(item)

    def _populate_signal_domains(self) -> None:
        section = self._sections["signal_domains"]
        for signal_domain_id, signal_domain in self.project_state.signal_domains.items():
            label = (
                f"{signal_domain_id} | "
                f"{signal_domain.required_signal_unit}/{signal_domain.required_calibration_mode}"
            )
            item = QTreeWidgetItem([label])
            item.setData(
                0, Qt.ItemDataRole.UserRole, {"kind": "signal_domain", "id": signal_domain_id}
            )
            section.addChild(item)

    def _populate_model_runs(self) -> None:
        section = self._sections["model_runs"]
        for model_run_id, model_run in self.project_state.model_runs.items():
            label = f"{model_run_id} | {model_run.model_type}"
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "model_run", "id": model_run_id})
            section.addChild(item)

    def _populate_predictions(self) -> None:
        section = self._sections["predictions"]
        for prediction_id, prediction in self.project_state.predictions.items():
            label = f"{prediction_id} | model={prediction.model_run_id}"
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "prediction", "id": prediction_id})
            section.addChild(item)
