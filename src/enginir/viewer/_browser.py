"""Read-only project browser for the workspace shell."""

from __future__ import annotations

from pathlib import Path


from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFrame, QSizePolicy, QTreeWidget, QTreeWidgetItem, QWidget

from ._state import ProjectState, signal_label

__all__ = ["ProjectBrowser"]


class ProjectBrowser(QTreeWidget):
    """Persistent read-only browser for the active project workspace.

    Each model shows its calibration set and its input contract beneath it
    (terminology contract); calibration sets and input contracts that no
    model uses follow the models, marked "(no model)".
    """

    _SECTION_ORDER = [
        ("datasets", "Datasets"),
        ("calibrations", "Reference corrections"),
        ("rois", "ROIs"),
        ("targets", "Targets"),
        ("target_assignments", "Target values"),
        ("model_runs", "Models"),
        ("predictions", "Predictions"),
    ]
    _KIND_TO_SECTION = {
        "dataset": "datasets",
        "calibration": "calibrations",
        "roi": "rois",
        "target": "targets",
        "target_assignment": "target_assignments",
        "sample_set": "model_runs",
        "signal_domain": "model_runs",
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
        pending = [self._sections[section_key]]
        while pending:  # depth first: a model's calibration set and input contract are nested
            parent = pending.pop(0)
            for index in range(parent.childCount()):
                child = parent.child(index)
                if child is None:
                    continue
                if self._payload_matches(payload, child.data(0, Qt.ItemDataRole.UserRole)):
                    return child
                pending.append(child)
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
            label = f"{self.project_state.data_file_label(dataset_id)} [{dataset_id}]"
            item = QTreeWidgetItem([label])
            item.setToolTip(0, dataset.source_path or dataset_id)
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
            item.setToolTip(0, self.project_state.data_file_label(roi.dataset_id))
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

    def _sample_set_item(self, sample_set_id: str, suffix: str = "") -> QTreeWidgetItem:
        sample_set = self.project_state.sample_sets[sample_set_id]
        target = self.project_state.targets.get(sample_set.target_id)
        target_label = target.name if target is not None else sample_set.target_id
        item = QTreeWidgetItem([f"Calibration set: {len(sample_set.roi_ids)} ROIs, "
                                f"{target_label}{suffix}"])
        item.setToolTip(0, sample_set_id)
        item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "sample_set", "id": sample_set_id})
        return item

    def _signal_domain_item(self, signal_domain_id: str, suffix: str = "") -> QTreeWidgetItem:
        domain = self.project_state.signal_domains[signal_domain_id]
        item = QTreeWidgetItem([f"Input contract: {domain.input_format}, "
                                f"{signal_label(domain.required_signal_unit)}{suffix}"])
        item.setToolTip(0, signal_domain_id)
        item.setData(0, Qt.ItemDataRole.UserRole,
                     {"kind": "signal_domain", "id": signal_domain_id})
        return item

    def _populate_model_runs(self) -> None:
        section = self._sections["model_runs"]
        state = self.project_state
        used_sets: set[str] = set()
        used_domains: set[str] = set()
        for model_run_id, model_run in state.model_runs.items():
            name = model_run.name or model_run.model_type
            item = QTreeWidgetItem([f"{name} [{model_run_id}]"])
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "model_run", "id": model_run_id})
            if model_run.sample_set_id in state.sample_sets:
                item.addChild(self._sample_set_item(model_run.sample_set_id))
                used_sets.add(model_run.sample_set_id)
            if model_run.signal_domain_id in state.signal_domains:
                item.addChild(self._signal_domain_item(model_run.signal_domain_id))
                used_domains.add(model_run.signal_domain_id)
            section.addChild(item)
        for sample_set_id in state.sample_sets:
            if sample_set_id not in used_sets:
                section.addChild(self._sample_set_item(sample_set_id, " (no model)"))
        for signal_domain_id in state.signal_domains:
            if signal_domain_id not in used_domains:
                section.addChild(self._signal_domain_item(signal_domain_id, " (no model)"))

    def _populate_predictions(self) -> None:
        section = self._sections["predictions"]
        for prediction_id, prediction in self.project_state.predictions.items():
            model_run = self.project_state.model_runs.get(prediction.prediction_run.model_run_id)
            model = model_run.name if model_run is not None and model_run.name else "model"
            label = f"{Path(prediction.dataset.source_path).name} | {model} (this session)"
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "prediction", "id": prediction_id})
            section.addChild(item)
