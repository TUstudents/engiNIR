"""Plan a calibration set without loading its pixels (a light module: the viewer's Model
page calls it before the modelling libraries are loaded). :mod:`enginir.modeling`
re-exports it."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import numpy as np

from .._checks import log_and_raise
from .._logging import get_logger
from ..project import SampleSet, TargetAssignment, TrainingROI
from ..project._ids import new_id, now_iso
from ..project._models import SAMPLE_SET_SEMANTICS

logger = get_logger(__name__)


def plan_sample_set(
    *,
    rois: Mapping[str, TrainingROI],
    target_assignments: Mapping[tuple[str, str], TargetAssignment],
    target_id: str,
    preprocessing_recipe_id: str,
    roi_ids: list[str] | None = None,
    sample_set_id: str | None = None,
    created_at: str | None = None,
    target_name: str | None = None,
) -> SampleSet:
    """The sample set of the labelled ROIs of a target, without reading any spectra.

    Applies the same selection rules as :func:`build_sample_set`: ROIs
    without a value for the target are left out, and all ROIs must share
    one wavelength axis.

    Raises
    ------
    ValueError
        If an ROI id is unknown or repeated, no selected ROI has a target
        value, or the ROIs have different wavelength axes. Messages name the
        target by ``target_name`` when it is given, else by its id.
    """
    target_label = target_name or target_id
    explicit = roi_ids is not None
    selected = ([roi_id for roi_id, roi in rois.items() if roi.job != "prediction"]
                if roi_ids is None else [str(roi_id) for roi_id in roi_ids])
    left_out = 0 if explicit else sum(
        1 for roi_id, roi in rois.items()
        if roi.job == "prediction" and (roi_id, target_id) in target_assignments)
    if not selected:
        if left_out:
            log_and_raise("No calibration ROI has a value for target '{}' ({} prediction ROI(s) "
                          "with a value left out)", target_label, left_out)
        if not explicit and rois:
            log_and_raise("Every ROI is a prediction ROI; a model needs calibration ROIs")
        log_and_raise("No ROIs selected for the calibration set")
    repeated = sorted(roi_id for roi_id, count in Counter(selected).items() if count > 1)
    if repeated:
        log_and_raise("ROI(s) selected more than once for the calibration set: {}", repeated)
    labelled: list[TrainingROI] = []
    unlabelled: list[str] = []
    for roi_id in selected:
        if roi_id not in rois:
            log_and_raise("Unknown ROI '{}' selected for the calibration set", roi_id)
        if (roi_id, target_id) in target_assignments:
            labelled.append(rois[roi_id])
        else:
            unlabelled.append(roi_id)
    if not labelled:
        if left_out:
            log_and_raise("No calibration ROI has a value for target '{}' ({} prediction ROI(s) "
                          "with a value left out)", target_label, left_out)
        log_and_raise("No selected ROI has a value for target '{}'", target_label)
    switched = [roi.name for roi in labelled if roi.job == "prediction"]
    if explicit and switched:  # a stored sample set keeps its ROIs (sample-set contract §1)
        logger.warning("ROI(s) of the stored selection are now prediction ROIs and are used "
                       "as stored: {}", switched)
    if unlabelled:
        logger.warning("Skipping {} ROI(s) without a value for target '{}': {}",
                       len(unlabelled), target_id, unlabelled)
    wavelengths = labelled[0].wavelengths
    for roi in labelled[1:]:
        if roi.wavelengths.shape != wavelengths.shape or not np.allclose(
            roi.wavelengths, wavelengths, atol=1e-6, rtol=0.0
        ):
            log_and_raise("ROI '{}' has a different wavelength axis than ROI '{}'", roi.name,
                          labelled[0].name)
    return SampleSet(
        sample_set_id=sample_set_id or new_id(),
        roi_ids=[roi.roi_id for roi in labelled],
        target_id=target_id,
        preprocessing_recipe_id=preprocessing_recipe_id,
        dataset_ids=list(dict.fromkeys(roi.dataset_id for roi in labelled)),
        created_at=created_at or now_iso(),
        **SAMPLE_SET_SEMANTICS,
    )
