"""Shared project/dataset attachment validation helpers."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from PyQt6.QtWidgets import QMessageBox, QWidget

from .._logging import get_logger
from ..io.project_store import load_project_store
from ..project import hydrate_project_store
from ._data_helpers import wavelength_axes_match
from ._state import ProjectState

logger = get_logger(__name__)


def confirm_project_dataset_binding(
    parent: QWidget | None,
    *,
    project_path: str,
    dataset_path: str | None,
    project_state: ProjectState | None = None,
    store: dict[str, Any] | None = None,
    source_format: str | None = None,
    source_modality: str | None = None,
    cube_shape: tuple[int, int, int] | None = None,
    wavelengths: np.ndarray | None = None,
) -> bool:
    """Return ``True`` when a project may be attached to the given dataset context.

    Validation is per data file (attachment contract): the attached file is
    compared with its own registration in the project, if it has one. A file
    that is not registered yet is a new data file of the project, and ROIs of
    other data files are never a mismatch. A missing dataset path means
    project-only mode and is always allowed.
    """
    if not dataset_path:
        return True
    mismatches = _collect_binding_mismatches(
        project_path=project_path,
        dataset_path=dataset_path,
        project_state=project_state,
        store=store,
        source_format=source_format,
        source_modality=source_modality,
        cube_shape=cube_shape,
        wavelengths=wavelengths,
    )
    if not mismatches:
        logger.info("Project dataset binding ok for {}", dataset_path)
        return True
    logger.warning(
        "Project dataset binding found {} mismatch(es) for {}: {}",
        len(mismatches),
        dataset_path,
        "; ".join(mismatches),
    )
    result = QMessageBox.warning(
        parent,
        "Data file changed",
        f"'{os.path.basename(dataset_path)}' no longer matches the data file registered in "
        "this project, so its ROIs may not line up with the image.\n"
        "Open anyway?\n\n" + "\n".join(f"- {detail}" for detail in mismatches),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
    )
    logger.info(
        "Project dataset binding {} for {}",
        "overridden by user" if result == QMessageBox.StandardButton.Yes else "cancelled by user",
        dataset_path,
    )
    return result == QMessageBox.StandardButton.Yes


def _collect_binding_mismatches(
    *,
    project_path: str,
    dataset_path: str,
    project_state: ProjectState | None,
    store: dict[str, Any] | None,
    source_format: str | None,
    source_modality: str | None,
    cube_shape: tuple[int, int, int] | None,
    wavelengths: np.ndarray | None,
) -> list[str]:
    """Compare the attached file with the project dataset registered at its path."""
    if project_state is not None:
        rois, datasets = project_state.rois, project_state.datasets
    else:
        workspace = hydrate_project_store(
            store if store is not None else load_project_store(project_path)
        )
        rois, datasets = workspace.rois, workspace.datasets

    attached_path = os.path.abspath(dataset_path)
    dataset = next((dataset for dataset in datasets.values()
                    if dataset.source_path
                    and os.path.abspath(dataset.source_path) == attached_path), None)
    if dataset is None:
        return []  # a new data file of the project
    mismatches: list[str] = []
    if source_format and dataset.source_format != source_format:
        mismatches.append(f"format: registered {dataset.source_format!r}, "
                          f"file {source_format!r}")
    if source_modality and dataset.source_modality != source_modality:
        mismatches.append(f"modality: registered {dataset.source_modality!r}, "
                          f"file {source_modality!r}")
    if cube_shape is not None and dataset.cube_shape != tuple(int(n) for n in cube_shape):
        mismatches.append(f"shape (frames, rows, bands): registered {dataset.cube_shape}, "
                          f"file {tuple(int(n) for n in cube_shape)}")
    if wavelengths is not None and not wavelength_axes_match(dataset.wavelengths, wavelengths):
        mismatches.append("wavelength axis differs")
    if mismatches:
        names = sorted(roi.name or roi.roi_id for roi in rois.values()
                       if roi.dataset_id == dataset.dataset_id)
        if names:
            shown = ", ".join(f"'{name}'" for name in names[:5])
            more = f" and {len(names) - 5} more" if len(names) > 5 else ""
            mismatches.append(f"ROIs drawn from it: {shown}{more}")
    return mismatches


__all__ = ["confirm_project_dataset_binding"]
