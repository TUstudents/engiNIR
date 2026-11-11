"""Project object inspector for the workspace shell."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QFrame, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from typing import Any

import numpy as np

from .._logging import get_logger
from ..io.project_store import load_model_run_artifact_refs
from ..project import ModelRun
from ._state import ProjectState

__all__ = ["ProjectInspector", "fit_summary"]

logger = get_logger(__name__)

_FOLD_RULES = {"round_robin": "round-robin, {n} folds", "leave_one_group_out": "leave one out",
               "group_kfold": "GroupKFold, {n} folds (before 1.2)"}
_GROUPS = {"roi": "each ROI", "sample": "ROIs with the same Sample label",
           "target": "ROIs with the same target value"}
_SCALINGS = {"center": "center", "uv": "unit variance", "pareto": "Pareto"}
#: What a calibration set was fitted on (terminology contract, Pixel: "Fitted on: ...").
_FITTED_ON = {"pixel": "pixels", "roi_mean": "ROI means"}
#: Reference correction modes as people say them (terminology contract, Reference correction).
_CORRECTION_MODES = {"raw_dn": "none (raw counts)",
                     "dark_white_reference": "dark/white references or detector levels",
                     "precalibrated_import": "already corrected"}


def fit_summary(model_run: ModelRun, roi_names: dict[str, str],
                exclusion: dict[str, Any] | None = None) -> list[str]:
    """How a model run was fitted, as lines for people (sample-set-model-run contract §7).

    ``exclusion`` is the run's stored ``artifact_refs/exclusion`` (with its
    arrays); ROI ids are shown by name.
    """
    spec = model_run.fit_specification
    folds = _FOLD_RULES.get(spec.fold_rule, spec.fold_rule).format(n=spec.n_folds)
    lines = [f"Folds: {folds}; held out together: {_GROUPS.get(spec.group_by, spec.group_by)}"]
    lines.append("Pixels: all" if spec.pixels_per_roi is None
                 else f"Pixels: {spec.pixels_per_roi} per ROI (seed {spec.seed})")
    hyper = model_run.hyperparameters
    if "x_scaling" in hyper:
        scaling = f"X {_SCALINGS.get(str(hyper['x_scaling']), hyper['x_scaling'])}"
        if "y_scaling" in hyper:
            scaling += f", Y {_SCALINGS.get(str(hyper['y_scaling']), hyper['y_scaling'])}"
    else:
        scaling = "unit variance (before 1.2)" if hyper.get("scale", True) else "center"
    lines.append(f"Scaling: {scaling}")
    if spec.exclusion == "none":
        lines.append("Pixel exclusion: none")
        return lines
    weighting = "weighted" if spec.exclusion_formula == "dmodx-2" else "unweighted"
    lines.append(f"Pixel exclusion: DModX ({weighting}), level {spec.exclusion_level:g}, "
                 f"{spec.exclusion_rounds} round(s), at most "
                 f"{spec.exclusion_max_fraction:.0%} per ROI; "
                 f"{model_run.metrics.get('n_excluded_pixels', 0)} pixels excluded")
    if exclusion:
        roi_ids = [str(r) for r in np.atleast_1d(exclusion.get("roi_ids", []))]
        counts = np.atleast_1d(exclusion.get("counts", [])).astype(int)
        if roi_ids:
            lines.append("Excluded per ROI: " + ", ".join(
                f"{roi_names.get(r, r)} {n}" for r, n in zip(roi_ids, counts)))
        flagged = [str(r) for r in np.atleast_1d(exclusion.get("flagged_roi_ids", []))]
        if flagged:
            lines.append("Flagged ROIs (more outlying pixels than allowed): "
                         + ", ".join(roi_names.get(r, r) for r in flagged))
        if "dcrit" in exclusion:
            lines.append(f"DCrit: {float(exclusion['dcrit']):.4g}")
    return lines


class ProjectInspector(QWidget):
    """Simple selection-driven inspector for project browser objects."""

    _SECTION_ENTRIES = {
        "datasets": lambda state: sorted(state.datasets),
        "calibrations": lambda state: sorted(state.calibrations),
        "rois": lambda state: sorted(state.rois),
        "targets": lambda state: sorted(state.targets),
        "target_assignments": lambda state: sorted(
            f"{target_id}->{roi_id}" for roi_id, target_id in state.target_assignments
        ),
        "sample_sets": lambda state: sorted(state.sample_sets),
        "signal_domains": lambda state: sorted(state.signal_domains),
        "model_runs": lambda state: sorted(state.model_runs),
        "predictions": lambda state: sorted(state.predictions),
    }

    def __init__(self, project_state: ProjectState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_state = project_state
        self.setMinimumWidth(260)
        self.setMaximumWidth(260)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 6, 4, 6)
        content_layout.setSpacing(6)

        self.title_label = QLabel("Inspector")
        self.body_label = QLabel("Select a project object to inspect details.")
        self.body_label.setWordWrap(True)
        self.body_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)

        content_layout.addWidget(self.title_label)
        content_layout.addWidget(self.body_label)
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def show_default(self) -> None:
        self.title_label.setText("Inspector")
        self.body_label.setText("Select a project object to inspect details.")

    def show_workspace_summary(self, title: str, lines: list[str]) -> None:
        self.title_label.setText(title)
        self.body_label.setText("\n".join(lines))

    def show_item(self, payload: dict[str, str] | None) -> None:
        if not payload:
            self.show_default()
            return
        kind = payload.get("kind", "")
        object_id = payload.get("id", "")
        if kind == "section":
            self._show_section(object_id, payload.get("label", object_id))
            return
        handler = self._item_handlers().get(kind)
        if handler is None:
            self.show_default()
            return
        if kind == "target_assignment":
            self._show_target_assignment(payload)
            return
        handler(object_id)

    def _item_handlers(self) -> dict[str, Callable[[str], None]]:
        return {
            "dataset": self._show_dataset,
            "calibration": self._show_calibration,
            "roi": self._show_roi,
            "target": self._show_target,
            "sample_set": self._show_sample_set,
            "signal_domain": self._show_signal_domain,
            "model_run": self._show_model_run,
            "prediction": self._show_prediction,
        }

    def _show_section(self, section_id: str, label: str) -> None:
        entry_loader = self._SECTION_ENTRIES.get(section_id)
        if entry_loader is None:
            self.show_default()
            return
        entries = entry_loader(self.project_state)

        preview = ", ".join(entries[:3]) if entries else "none"
        self.title_label.setText(label)
        self.body_label.setText(
            "\n".join(
                [
                    f"Count: {len(entries)}",
                    f"Preview: {preview}",
                ]
            )
        )

    def _show_dataset(self, dataset_id: str) -> None:
        dataset = self.project_state.datasets.get(dataset_id)
        if dataset is None:
            self.show_default()
            return
        self.title_label.setText("Dataset")
        self.body_label.setText(
            "\n".join(
                [
                    f"ID: {dataset.dataset_id}",
                    f"Path: {Path(dataset.source_path).name if dataset.source_path else 'none'}",
                    f"Format: {dataset.source_format}",
                    f"Modality: {dataset.source_modality}",
                    f"Cube shape: {dataset.cube_shape}",
                    f"Bands: {dataset.n_bands}",
                ]
            )
        )

    def _show_calibration(self, calibration_id: str) -> None:
        calibration = self.project_state.calibrations.get(calibration_id)
        if calibration is None:
            self.show_default()
            return
        self.title_label.setText("Reference correction")
        self.body_label.setText(
            "\n".join(
                [
                    f"ID: {calibration.calibration_id}",
                    f"Dataset: {calibration.dataset_id}",
                    f"Mode: {_CORRECTION_MODES.get(calibration.mode, calibration.mode)}",
                    f"Output unit: {calibration.output_signal_unit}",
                    f"Has dark context: {calibration.dark_ref_source is not None or calibration.dark_ref_data is not None or calibration.black_level is not None}",
                    f"Has white context: {calibration.white_ref_source is not None or calibration.white_ref_data is not None or calibration.white_level is not None}",
                ]
            )
        )

    def _show_roi(self, roi_id: str) -> None:
        roi = self.project_state.rois.get(roi_id)
        if roi is None:
            self.show_default()
            return
        self.title_label.setText("ROI")
        assignment_lines: list[str] = []
        for (assigned_roi_id, target_id), assignment in sorted(
            self.project_state.target_assignments.items()
        ):
            if assigned_roi_id != roi_id:
                continue
            target = self.project_state.targets.get(target_id)
            target_label = target.name if target is not None else target_id
            assignment_lines.append(f"{target_label}: {assignment.value:.4g}")
        if not assignment_lines:
            assignment_text = "none"
        else:
            assignment_text = "; ".join(assignment_lines[:3])
        self.body_label.setText(
            "\n".join(
                [
                    f"ID: {roi.roi_id}",
                    f"Name: {roi.name or 'unnamed'}",
                    f"Dataset: {self.project_state.data_file_label(roi.dataset_id)}",
                    f"Dataset ID: {roi.dataset_id}",
                    f"Reference correction ID: {roi.calibration_id}",
                    f"Pixels: {roi.n_pixels}",
                    f"Bounds: {roi.spatial_bounds}",
                    f"Target assignments: {assignment_text}",
                ]
            )
        )

    def _show_target(self, target_id: str) -> None:
        target = self.project_state.targets.get(target_id)
        if target is None:
            self.show_default()
            return
        assignments = [
            assignment
            for assignment in self.project_state.target_assignments.values()
            if assignment.target_id == target_id
        ]
        self.title_label.setText("Target")
        self.body_label.setText(
            "\n".join(
                [
                    f"ID: {target.target_id}",
                    f"Name: {target.name}",
                    f"Unit: {target.unit}",
                    f"Value type: {target.value_type}",
                    f"Missing policy: {target.missing_value_policy}",
                    f"Assignments: {len(assignments)}",
                ]
            )
        )

    def _show_sample_set(self, sample_set_id: str) -> None:
        sample_set = self.project_state.sample_sets.get(sample_set_id)
        if sample_set is None:
            self.show_default()
            return
        self.title_label.setText("Calibration set")
        self.body_label.setText(
            "\n".join(
                [
                    f"ID: {sample_set.sample_set_id}",
                    f"Target: {sample_set.target_id}",
                    f"Preprocessing recipe: {sample_set.preprocessing_recipe_id}",
                    f"Datasets: {', '.join(sample_set.dataset_ids)}",
                    f"ROIs: {len(sample_set.roi_ids)}",
                    f"Fitted on: {_FITTED_ON.get(sample_set.observation_unit, sample_set.observation_unit)}",
                    f"Validation unit: {sample_set.validation_unit}",
                ]
            )
        )

    def _show_signal_domain(self, signal_domain_id: str) -> None:
        signal_domain = self.project_state.signal_domains.get(signal_domain_id)
        if signal_domain is None:
            self.show_default()
            return
        self.title_label.setText("Input contract")
        self.body_label.setText(
            "\n".join(
                [
                    f"ID: {signal_domain.signal_domain_id}",
                    f"Input: {signal_domain.input_format} | {signal_domain.source_modality}",
                    f"Signal unit: {signal_domain.required_signal_unit}",
                    f"Reference correction: {_CORRECTION_MODES.get(signal_domain.required_calibration_mode, signal_domain.required_calibration_mode)}",
                    f"Requires refs: dark={signal_domain.requires_dark_ref}, white={signal_domain.requires_white_ref}",
                    f"Corrected input allowed: {signal_domain.calibrated_input_allowed}",
                    f"Bands: {signal_domain.n_bands}",
                    f"Band mapping: {signal_domain.band_mapping_policy}",
                ]
            )
        )

    def _show_model_run(self, model_run_id: str) -> None:
        model_run = self.project_state.model_runs.get(model_run_id)
        if model_run is None:
            self.show_default()
            return
        self.title_label.setText("Model")
        self.body_label.setText(
            "\n".join(
                [
                    f"ID: {model_run.model_run_id}",
                    f"Type: {model_run.model_type}",
                    f"Calibration set: {model_run.sample_set_id}",
                    f"Input contract: {model_run.signal_domain_id}",
                    f"CV: {model_run.cv_spec}",
                    *fit_summary(model_run, {r: roi.name for r, roi in
                                             self.project_state.rois.items()},
                                 self._exclusion_of(model_run)),
                    f"Metrics: {model_run.metrics or 'none'}",
                ]
            )
        )

    def _exclusion_of(self, model_run: ModelRun) -> dict[str, Any] | None:
        """The run's stored exclusion arrays, read on demand (``None`` without exclusion)."""
        path = self.project_state.project_path
        if model_run.fit_specification.exclusion == "none" or path is None:
            return None
        try:
            refs = load_model_run_artifact_refs(path, model_run.model_run_id,
                                                keys=("exclusion",))
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("Excluded pixels of model '{}' not read: {}",
                           model_run.model_run_id, exc)
            return None
        return refs.get("exclusion")

    def _show_prediction(self, prediction_id: str) -> None:
        result = self.project_state.predictions.get(prediction_id)
        if result is None:
            self.show_default()
            return
        run = result.prediction_run
        shape = run.artifact_refs.get("spatial_shape", run.artifact_refs.get("spatial_map_shape"))
        map_shape = ("none" if shape is None
                     else " x ".join(str(int(v)) for v in np.asarray(shape).tolist()))
        model_run = self.project_state.model_runs.get(run.model_run_id)
        summary = result.diagnostics or {}
        lines = [
            "Not stored: predictions live in this session only.",
            f"Dataset file: {result.dataset.source_path}",
            f"Reference correction: {_CORRECTION_MODES.get(result.calibration.mode, result.calibration.mode)}"
            f" -> {result.calibration.output_signal_unit}",
            f"Model: {model_run.name if model_run is not None and model_run.name else run.model_run_id}",
            f"Spatial map: {map_shape}",
        ]
        if summary:
            lines.append(f"Above DCrit: {100 * float(summary.get('share_above_dcrit', 0)):.2g} % | "
                         f"above T²Crit: {100 * float(summary.get('share_above_t2crit', 0)):.2g} %")
        else:
            lines.append("No DModX/T² maps (refit the model to get them)")
        self.title_label.setText("Prediction")
        self.body_label.setText("\n".join(lines))

    def _show_target_assignment(self, payload: dict[str, str]) -> None:
        roi_id = str(payload.get("roi_id", ""))
        target_id = str(payload.get("target_id", ""))
        assignment = self.project_state.target_assignments.get((roi_id, target_id))
        if assignment is None:
            self.show_default()
            return
        roi = self.project_state.rois.get(roi_id)
        target = self.project_state.targets.get(target_id)
        self.title_label.setText("Target value")
        self.body_label.setText(
            "\n".join(
                [
                    f"Target: {target.name if target is not None else target_id}",
                    f"ROI: {roi.name if roi is not None and roi.name else roi_id} [{roi_id}]",
                    f"Value: {assignment.value:.4g}",
                    f"Assigned at: {assignment.assigned_at}",
                ]
            )
        )
