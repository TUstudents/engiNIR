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

    A missing dataset path means project-only mode and is always allowed.
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
        "Dataset mismatch",
        "The attached dataset does not match one or more ROI-backed project datasets.\n"
        "Overlays may not align with the current image.\n"
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
    """Compare every ROI-backed dataset of the project with the dataset being attached."""
    if project_state is not None:
        rois, datasets = project_state.rois, project_state.datasets
    else:
        workspace = hydrate_project_store(
            store if store is not None else load_project_store(project_path)
        )
        rois, datasets = workspace.rois, workspace.datasets

    attached_path = os.path.abspath(dataset_path)
    mismatches: list[str] = []
    compared: set[str] = set()
    for roi in rois.values():
        if os.path.abspath(roi.source_file) != attached_path:
            mismatches.append(f"ROI '{roi.roi_id}' was drawn from '{roi.source_file}', "
                              f"not '{attached_path}'.")
            continue
        dataset = datasets.get(roi.dataset_id)
        if dataset is None or roi.dataset_id in compared:
            continue
        compared.add(roi.dataset_id)
        label = f"ROI-backed dataset '{dataset.dataset_id}'"
        if os.path.abspath(dataset.source_path) != attached_path:
            mismatches.append(f"{label} points to '{os.path.abspath(dataset.source_path)}', "
                              f"not '{attached_path}'.")
            continue
        if source_format and dataset.source_format != source_format:
            mismatches.append(f"{label} expects source_format={dataset.source_format!r}, "
                              f"got {source_format!r}.")
        if source_modality and dataset.source_modality != source_modality:
            mismatches.append(f"{label} expects source_modality={dataset.source_modality!r}, "
                              f"got {source_modality!r}.")
        if cube_shape is not None and dataset.cube_shape != tuple(int(n) for n in cube_shape):
            mismatches.append(f"{label} expects cube_shape={dataset.cube_shape!r}, "
                              f"got {tuple(int(n) for n in cube_shape)!r}.")
        if wavelengths is not None and not wavelength_axes_match(dataset.wavelengths,
                                                                 wavelengths):
            mismatches.append(f"{label} wavelength axis does not match the attached dataset.")
    return mismatches


__all__ = ["confirm_project_dataset_binding"]
