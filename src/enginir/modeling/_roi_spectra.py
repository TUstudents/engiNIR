"""ROI spectra in a signal unit, as model fitting converts them (light: no SciPy or
scikit-learn at import, so the viewer's ROI spectra show while those load; the
preprocessing steps import them when a recipe needs them).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from enginir._checks import log_and_raise
from enginir._logging import get_logger

from ..io.project_store import iter_roi_reference_spectra, load_roi_spectra_batch
from ..preprocessing.recipe import PreprocessingRecipe
from ..preprocessing._pipeline import transform_preprocessing_steps
from ..project import (
    DatasetRecord,
    ReferenceCorrection,
    ReferenceCorrectionStats,
    SignalDomainContract,
    TrainingROI,
    check_signal_domain_compatibility,
    transform_spectra_to_signal_domain,
)

logger = get_logger(__name__)

_WORKING_DTYPE = np.float32


def roi_to_signal_domain(
    raw_spectra: np.ndarray,
    *,
    roi: TrainingROI,
    dataset: DatasetRecord,
    reference_correction: ReferenceCorrection,
    snapshot: tuple[np.ndarray, np.ndarray] | None,
    signal_domain: SignalDomainContract,
    stats: ReferenceCorrectionStats | None = None,
) -> np.ndarray:
    """Convert one ROI's stored spectra to ``signal_domain``, exactly as model fitting does.

    Reference corrections with per-pixel references use ``snapshot``, the ROI's stored
    ``(dark, white)`` reference spectra (one per ROI pixel). The fit and the
    viewer's ROI spectra both call this function.
    """
    check_signal_domain_compatibility(dataset=dataset, reference_correction=reference_correction,
                                      signal_domain=signal_domain)
    if reference_correction.dark_ref_data is not None:
        if snapshot is None:
            log_and_raise("ROI '{}' has no reference snapshot for reference correction '{}'", roi.roi_id,
                          reference_correction.reference_correction_id)
        dark, white = snapshot
        if dark.shape != raw_spectra.shape or white.shape != raw_spectra.shape:
            log_and_raise("ROI '{}' reference snapshot shape {} does not match its spectra {}",
                          roi.roi_id, dark.shape, raw_spectra.shape)
        reference_correction = replace(reference_correction, dark_ref_data=dark, white_ref_data=white)
    return transform_spectra_to_signal_domain(
        raw_spectra, dataset=dataset, reference_correction=reference_correction, signal_domain=signal_domain,
        stats=stats, check_compatibility=False,
        per_spectrum_references=reference_correction.dark_ref_data is not None)


@dataclass(frozen=True)
class RoiSpectrumSummary:
    """Pixel spectra of one ROI in a signal unit and preprocessing, summarized per band."""

    roi_id: str
    mean: np.ndarray
    p05: np.ndarray
    median: np.ndarray
    p95: np.ndarray


def roi_spectrum_summaries(
    project_path: str,
    rois: Sequence[TrainingROI],
    *,
    datasets: Mapping[str, DatasetRecord],
    reference_corrections: Mapping[str, ReferenceCorrection],
    signal_unit: str,
    recipe: PreprocessingRecipe | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple[dict[str, RoiSpectrumSummary], dict[str, str]]:
    """Mean, median and 5–95 % spectra of each ROI as model fitting would see its pixels.

    Each ROI is converted with :func:`roi_to_signal_domain` (its own
    reference correction and stored references) and preprocessed pixel by pixel with
    ``recipe``. ``signal_unit="DN"`` shows the stored raw counts of every ROI
    that has them. MSC and EMSC need a reference: here it is learned from the
    mean spectra of ``rois`` (model fitting learns it from the training
    pixels). ROIs are read one at a time, so memory stays bounded.

    Returns
    -------
    summaries : dict
        ``{roi_id: RoiSpectrumSummary}``.
    skipped : dict
        ``{roi_id: reason}`` for ROIs that cannot give ``signal_unit`` (for
        example raw-count ROIs asked for absorbance).
    """
    recipe = recipe or PreprocessingRecipe()
    signal: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    by_id = {roi.roi_id: roi for roi in rois}
    for roi_id, roi in by_id.items():
        if cancelled():
            break
        dataset, reference_correction = datasets[roi.dataset_id], reference_corrections[roi.reference_correction_id]
        key = f"/rois/{roi_id}/spectra"
        reason = _signal_unit_unavailable(reference_correction, signal_unit)
        if reason:
            skipped[roi_id] = reason
            continue
        if signal_unit == "DN":  # raw counts as stored; no conversion
            raw = load_roi_spectra_batch(project_path, [key])[key]
            signal[roi_id] = np.asarray(raw, dtype=_WORKING_DTYPE)
            continue
        try:
            domain = SignalDomainContract.for_training_data(
                dataset=dataset, reference_correction=reference_correction, required_signal_unit=signal_unit)
            raw = load_roi_spectra_batch(project_path, [key])[key]
            # One ROI's reference snapshot at a time: together they can be large.
            snapshot = next(iter(iter_roi_reference_spectra(project_path, [roi_id])))[1]
            signal[roi_id] = roi_to_signal_domain(
                raw, roi=roi, dataset=dataset, reference_correction=reference_correction, snapshot=snapshot,
                signal_domain=domain)
        except ValueError as exc:
            skipped[roi_id] = str(exc)
    summaries: dict[str, RoiSpectrumSummary] = {}
    if not signal:
        return summaries, skipped
    wavelengths = np.asarray(next(iter(by_id.values())).wavelengths)
    steps = recipe.build_steps(wavelengths, signal_unit=signal_unit)
    # Fit the steps in order on the ROI means: stateless steps ignore it, MSC/EMSC
    # learn their reference from it.
    fitted = np.stack([values.mean(axis=0) for values in signal.values()]).astype(_WORKING_DTYPE)
    for _name, step in steps:
        fitted = np.asarray(step.fit_transform(fitted), dtype=_WORKING_DTYPE)
    for roi_id, values in signal.items():
        pixels = transform_preprocessing_steps(values, steps)
        p05, median, p95 = np.percentile(pixels, [5, 50, 95], axis=0)
        summaries[roi_id] = RoiSpectrumSummary(
            roi_id, pixels.mean(axis=0), p05.astype(_WORKING_DTYPE),
            median.astype(_WORKING_DTYPE), p95.astype(_WORKING_DTYPE))
    return summaries, skipped


def _signal_unit_unavailable(reference_correction: ReferenceCorrection, unit: str) -> str:
    """Why ROIs with ``reference_correction`` cannot be shown in ``unit`` ("" if they can)."""
    if reference_correction.mode == "already_corrected":
        return "" if unit != "DN" else "already corrected data has no raw counts"
    if reference_correction.mode == "raw_dn" and unit != "DN":
        return f"raw counts without references cannot give {unit}"
    return ""


__all__ = ["RoiSpectrumSummary", "roi_spectrum_summaries", "roi_to_signal_domain"]
