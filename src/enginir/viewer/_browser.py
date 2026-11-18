"""Read-only project browser for the workspace shell."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import QFrame, QSizePolicy, QTreeWidget, QTreeWidgetItem, QWidget

from .._guidance import figure_text
from ..project import ReferenceCorrection
from ._icons import icon
from ._state import ProjectState, signal_label
from ._ui_helpers import with_unit

#: The icon of each section of the project tree.
_SECTION_ICONS = {"datasets": "database", "reference_corrections": "adjustments-horizontal",
                  "rois": "grid-dots", "targets": "target",
                  "target_assignments": "list-details", "model_runs": "chart-dots",
                  "predictions": "map"}

__all__ = ["ProjectBrowser"]


class ProjectBrowser(QTreeWidget):
    """Persistent read-only browser for the active project workspace.

    Each model shows its calibration set and its input contract beneath it
    (terminology contract); calibration sets and input contracts that no
    model uses follow the models, marked "(no model)".
    """

    _SECTION_ORDER = [
        ("datasets", "Datasets"),
        ("reference_corrections", "Reference corrections"),
        ("rois", "ROIs"),
        ("targets", "Targets"),
        ("target_assignments", "Target values"),
        ("model_runs", "Models"),
        ("predictions", "Predictions"),
    ]
    _KIND_TO_SECTION = {
        "dataset": "datasets",
        "reference_correction": "reference_corrections",
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
        self.setIndentation(14)
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
        self.project_state.reference_corrections_changed.connect(self.schedule_refresh)
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
        self._populate_reference_corrections()
        self._populate_rois()
        self._populate_targets()
        self._populate_target_assignments()
        self._populate_model_runs()
        self._populate_predictions()
        self._label_sections()

        item = self._item_for_payload(selected_payload)
        if item is not None:
            self.setCurrentItem(item)
            return
        self.setCurrentItem(None)

    def _label_sections(self) -> None:
        """Each section with the number of its objects and its icon (in the theme's colour)."""
        state = self.project_state
        counts = {"datasets": len(state.datasets), "reference_corrections": len(state.reference_corrections),
                  "rois": len(state.rois), "targets": len(state.targets),
                  "target_assignments": len(state.target_assignments),
                  "model_runs": len(state.model_runs), "predictions": len(state.predictions)}
        for key, label in self._SECTION_ORDER:
            section = self._sections[key]
            section.setText(0, f"{label} ({counts[key]})" if counts[key] else label)
            section.setIcon(0, icon(_SECTION_ICONS[key]))

    def changeEvent(self, event: QEvent | None) -> None:  # noqa: N802 (Qt API)
        """The section icons follow the theme."""
        super().changeEvent(event)
        sections = getattr(self, "_sections", None)  # not yet during construction
        if event is not None and event.type() == QEvent.Type.PaletteChange and sections:
            for key, _label in self._SECTION_ORDER:
                sections[key].setIcon(0, icon(_SECTION_ICONS[key]))

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

    # Leaves show names; ids are in the tooltip and in Properties (an id is no name).

    def _populate_datasets(self) -> None:
        section = self._sections["datasets"]
        state = self.project_state
        for dataset_id in sorted(state.datasets,
                                 key=lambda d: state.data_file_label(d).casefold()):
            dataset = state.datasets[dataset_id]
            item = QTreeWidgetItem([state.data_file_label(dataset_id)])
            item.setToolTip(0, _with_id(dataset.source_path or "", dataset_id))
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "dataset", "id": dataset_id})
            section.addChild(item)

    def _populate_reference_corrections(self) -> None:
        section = self._sections["reference_corrections"]
        state = self.project_state

        def file_of(reference_correction_id: str) -> str:
            return state.data_file_label(state.reference_corrections[reference_correction_id].dataset_id)

        for reference_correction_id in sorted(state.reference_corrections, key=lambda c: file_of(c).casefold()):
            reference_correction = state.reference_corrections[reference_correction_id]
            item = QTreeWidgetItem([f"{file_of(reference_correction_id)}: {correction_text(reference_correction)}"])
            item.setToolTip(0, _with_id("Reference correction", reference_correction_id))
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "reference_correction", "id": reference_correction_id})
            section.addChild(item)

    def _populate_rois(self) -> None:
        section = self._sections["rois"]
        state = self.project_state
        rois = sorted(state.rois.items(), key=lambda pair: (
            state.data_file_label(pair[1].dataset_id).casefold(),
            _natural(pair[1].name or pair[0])))
        for roi_id, roi in rois:
            item = QTreeWidgetItem([roi.name or roi_id])
            item.setToolTip(0, _with_id(state.data_file_label(roi.dataset_id), roi_id))
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "roi", "id": roi_id})
            section.addChild(item)

    def _populate_targets(self) -> None:
        section = self._sections["targets"]
        targets = self.project_state.targets
        for target_id in sorted(targets, key=lambda t: targets[t].name.casefold()):
            target = targets[target_id]
            item = QTreeWidgetItem([with_unit(target.name, target.unit)])
            item.setToolTip(0, _with_id("Target", target_id))
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "target", "id": target_id})
            section.addChild(item)

    def _populate_target_assignments(self) -> None:
        section = self._sections["target_assignments"]
        state = self.project_state

        def names(assignment: Any) -> tuple[str, str]:
            target = state.targets.get(assignment.target_id)
            roi = state.rois.get(assignment.roi_id)
            return (target.name if target is not None else assignment.target_id,
                    roi.name if roi is not None and roi.name else assignment.roi_id)

        assignments = sorted(state.target_assignments.values(), key=lambda a: (
            names(a)[0].casefold(), _natural(names(a)[1])))
        for assignment in assignments:
            target_label, roi_label = names(assignment)
            item = QTreeWidgetItem([f"{roi_label}: {target_label} = "
                                    f"{figure_text(assignment.value)}"])
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
        for model_run_id, model_run in sorted(state.model_runs.items(),
                                              key=lambda pair: str(pair[1].created_at)):
            name = model_run.name or model_run.model_type
            item = QTreeWidgetItem([name])
            item.setToolTip(0, _with_id("Model", model_run_id))
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


def correction_text(reference_correction: ReferenceCorrection) -> str:
    """A reference correction as people say it (terminology contract, Reference correction)."""
    if reference_correction.mode == "dark_white_reference":
        if reference_correction.black_level is not None and reference_correction.white_level is not None:
            return (f"detector levels {reference_correction.black_level:g} / "
                    f"{reference_correction.white_level:g}")
        return "dark/white references"
    if reference_correction.mode == "already_corrected":
        return "already corrected"
    return "none (raw counts)"


def _with_id(text: str, object_id: str) -> str:
    """A tooltip: ``text`` and the stable id (for error reports)."""
    return f"{text}\nid {object_id}" if text else f"id {object_id}"


def _natural(text: str) -> tuple[Any, ...]:
    """Sort key that orders "L2" before "L10"."""
    return tuple(int(part) if part.isdigit() else part.casefold()
                 for part in re.split(r"(\d+)", text))
