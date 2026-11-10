"""Predicting prediction ROIs from the project file (prediction-runtime contract, Prediction ROIs).

:func:`predict_rois` predicts the stored pixels of ROIs with one model run:
each ROI is converted with its own calibration and stored references as model
fitting does (:func:`enginir.modeling.roi_to_signal_domain`), checked and
preprocessed by the same preflight and recipe steps as a scan
(:func:`~enginir.prediction.preflight_prediction`), and predicted with
DModXPS and T²PS per pixel. :func:`roi_prediction_statistics` summarises the
pixels per ROI, with an optional DModX pixel exclusion, and gives RMSEP and
bias over the ROIs that have a value. No data file is read and nothing is
written to the project.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._checks import log_and_raise
from .._logging import get_logger
from ..io.project_store import (
    iter_roi_calibration_refs,
    load_model_run_artifact_refs,
    load_roi_spectra_batch,
)
from ..preprocessing import PreprocessingRecipe
from ..project import (
    CalibrationSpec,
    DatasetRef,
    ModelRun,
    SampleSet,
    SignalDomainContract,
    TargetAssignment,
    TrainingROI,
)
from ..validation.outliers import dmodx_limit
from ._diagnostics import chunk_diagnostics, chunk_scores
from ._service import _predict_chunk, _preprocess_chunk, preflight_prediction
from ._uncertainty import DEFAULT_LEVEL, IntervalInputs, interval_inputs, mean_half_width

logger = get_logger(__name__)


@dataclass(frozen=True)
class RoiPixels:
    """The pixel predictions of one ROI by one model, with DModXPS and T²PS (NaN without).

    ``scores`` are the pixels' scores ``(n, A)`` for the mean interval (``None``
    without projection data).
    """

    y: np.ndarray
    dmodx: np.ndarray
    t2: np.ndarray
    scores: np.ndarray | None = None


@dataclass
class RoiPrediction:
    """One model's prediction of a set of ROIs, in memory (not stored).

    ``pixels`` holds the predicted ROIs; ``refused`` the ROIs that could not
    be predicted, with the reason. ``limit_args`` is ``(n_train, K, A)`` for
    :func:`~enginir.validation.dmodx_limit` (``None`` without projection
    data); ``default_exclusion`` the DModX level of the run's own fit
    specification (``None``: no exclusion); ``training_share_range`` the
    10th, 50th and 90th percentile of the training ROIs' shares of pixels
    above DCrit (0.99): their typical range and median (the extremes are
    single unusual ROIs); ``intervals`` the run's prediction-interval inputs
    (``None`` without them).
    """

    model_run_id: str
    model_name: str
    target_id: str
    pixels: dict[str, RoiPixels] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)
    limit_args: tuple[int, int, int] | None = None
    dcrit_099: float | None = None
    t2_crit_099: float | None = None
    default_exclusion: float | None = None
    training_share_range: tuple[float, float, float] | None = None
    intervals: IntervalInputs | None = None
    wavelength_warnings: dict[str, str] = field(default_factory=dict)

    def dmodx_limit_at(self, level: float) -> float | None:
        """DCrit at ``level`` for this model (the stored one at 0.99), or ``None``."""
        if self.limit_args is None:
            return None
        if abs(level - 0.99) < 1e-12 and self.dcrit_099 is not None:
            return self.dcrit_099
        n_train, n_bands, n_components = self.limit_args
        return dmodx_limit(n_train, n_bands, n_components, level, centred=True)


@dataclass(frozen=True)
class RoiStatistics:
    """Statistics of one ROI's pixels predicted by one model (contract, Prediction ROIs)."""

    roi_id: str
    model_run_id: str
    n_pixels: int
    n_removed: int
    mean: float
    median: float
    sd: float
    p05: float
    p95: float
    mean_dmodx: float
    mean_t2: float
    share_above_dcrit: float
    share_above_t2crit: float
    reference: float | None
    residual: float | None
    #: Half-width of the mean's 0.95 prediction interval; NaN without one (outside the
    #: model at the ROI level, no kept pixel, or no interval inputs).
    interval: float = float("nan")


@dataclass(frozen=True)
class RoiPredictionSummary:
    """Per-ROI statistics of one model, and RMSEP and bias over the ROIs with a value."""

    model_run_id: str
    model_name: str
    exclusion: float | None
    rois: tuple[RoiStatistics, ...]
    rmsep: float | None
    bias: float | None
    n_reference: int


def predict_rois(
    *,
    project_path: str,
    model_run: ModelRun,
    model_name: str,
    rois: Sequence[TrainingROI],
    datasets: Mapping[str, DatasetRef],
    calibrations: Mapping[str, CalibrationSpec],
    signal_domains: Mapping[str, SignalDomainContract],
    recipes: Mapping[str, dict[str, Any]],
    sample_sets: Mapping[str, SampleSet],
    cancelled: Callable[[], bool] = lambda: False,
) -> RoiPrediction:
    """Predict the stored pixels of ``rois`` with ``model_run`` (contract, Prediction ROIs).

    Each ROI uses its own calibration and stored references; an ROI that does
    not fit the model is refused with the reason, the others are predicted.
    The saved model is used (with its projection data, so the pixels get
    DModXPS and T²PS). Nothing is written to the project.
    """
    from ..modeling import roi_to_signal_domain  # modeling imports prediction

    if model_run.signal_domain_id not in signal_domains:
        log_and_raise("Model run '{}' has an unknown signal domain '{}'",
                      model_run.model_run_id, model_run.signal_domain_id)
    domain = signal_domains[model_run.signal_domain_id]
    sample_set = sample_sets[model_run.sample_set_id]
    recipe = PreprocessingRecipe.from_dict(recipes[sample_set.preprocessing_recipe_id])
    spec = model_run.fit_specification
    training = (load_model_run_artifact_refs(project_path, model_run.model_run_id,
                                             keys=("diagnostics",)).get("diagnostics") or {})
    result = RoiPrediction(
        model_run_id=model_run.model_run_id, model_name=model_name,
        target_id=sample_set.target_id,
        default_exclusion=(float(spec.exclusion_level) if spec.exclusion == "dmodx" else None),
        training_share_range=_share_percentiles(training.get("roi_share_above_dcrit")))
    preflights: dict[tuple[str, str], Any] = {}
    for roi in rois:
        if cancelled():
            break
        key = (roi.dataset_id, roi.calibration_id)
        try:
            if key not in preflights:
                preflights[key] = preflight_prediction(
                    project_path=project_path, model_run=model_run, signal_domain=domain,
                    recipe=recipe, dataset=datasets[roi.dataset_id],
                    calibration=calibrations[roi.calibration_id], model_registry={})
                warning = preflights[key].wavelength_warning
                if warning and roi.dataset_id not in result.wavelength_warnings:
                    result.wavelength_warnings[roi.dataset_id] = str(warning)
            preflight = preflights[key]
            spectra_key = f"/rois/{roi.roi_id}/spectra"
            raw = load_roi_spectra_batch(project_path, [spectra_key])[spectra_key]
            snapshot = next(iter(iter_roi_calibration_refs(project_path, [roi.roi_id])))[1]
            signal = roi_to_signal_domain(
                raw, roi=roi, dataset=datasets[roi.dataset_id],
                calibration=calibrations[roi.calibration_id], snapshot=snapshot,
                signal_domain=domain)
        except (ValueError, KeyError) as exc:
            result.refused[roi.roi_id] = str(exc)
            logger.warning("ROI '{}' not predicted with model '{}': {}", roi.name, model_name,
                           exc)
            continue
        X = _preprocess_chunk(signal, preflight)
        y = np.asarray(_predict_chunk(X, preflight), dtype=np.float32).reshape(-1)
        if preflight.projection is not None:
            t2, dmodx = chunk_diagnostics(X, preflight.projection)
            scores = chunk_scores(X, preflight.projection)
            projection = preflight.projection
            result.limit_args = (int((preflight.model_blob or {}).get("n_train", 0)) or 1,
                                 int(projection.x_mean.size), projection.n_components)
            result.dcrit_099, result.t2_crit_099 = projection.dcrit, projection.t2_crit
        else:
            t2 = dmodx = np.full(y.shape, np.nan, dtype=np.float32)
            scores = None
        if preflight.projection is not None and result.intervals is None:
            result.intervals = interval_inputs(model_run.metrics, preflight.model_blob,
                                               training.get("roi_scores"))
        result.pixels[roi.roi_id] = RoiPixels(y=y, dmodx=np.asarray(dmodx, np.float32),
                                              t2=np.asarray(t2, np.float32), scores=scores)
    for dataset_id, warning in result.wavelength_warnings.items():
        logger.warning("Data file '{}': {}", dataset_id, warning)
    return result


def roi_prediction_statistics(
    prediction: RoiPrediction,
    *,
    target_assignments: Mapping[tuple[str, str], TargetAssignment],
    exclude_dmodx: float | None = None,
) -> RoiPredictionSummary:
    """Per-ROI statistics of ``prediction`` (contract, Prediction ROIs); no prediction is redone.

    ``exclude_dmodx``: a level in (0, 1); pixels whose DModXPS is strictly
    above the DCrit at that level are removed from every statistic except the
    shares above the limits. ``None``: keep every pixel.
    """
    limit = None
    if exclude_dmodx is not None:
        if not 0.0 < float(exclude_dmodx) < 1.0:
            log_and_raise("exclude_dmodx must be a level between 0 and 1, got {}",
                          exclude_dmodx)
        limit = prediction.dmodx_limit_at(float(exclude_dmodx))
        if limit is None:
            logger.warning("Model '{}' has no projection data: pixels cannot be removed by "
                           "DModX", prediction.model_name)
    rows: list[RoiStatistics] = []
    for roi_id, pixels in prediction.pixels.items():
        valid = np.isfinite(pixels.y)
        removed = valid & (np.nan_to_num(pixels.dmodx, nan=-np.inf) > limit) if limit is not None \
            else np.zeros_like(valid)
        kept = valid & ~removed
        values = pixels.y[kept].astype(np.float64)
        n_valid = int(valid.sum())

        def share(values_: np.ndarray, crit: float | None) -> float:
            if crit is None or n_valid == 0 or not np.isfinite(values_[valid]).any():
                return float("nan")
            return float(np.mean(values_[valid] > crit))

        assignment = target_assignments.get((roi_id, prediction.target_id))
        reference = None if assignment is None else float(assignment.value)
        mean = float(values.mean()) if values.size else float("nan")
        rows.append(RoiStatistics(
            roi_id=roi_id, model_run_id=prediction.model_run_id, n_pixels=n_valid,
            n_removed=int(removed.sum()), mean=mean,
            median=float(np.median(values)) if values.size else float("nan"),
            sd=float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            p05=float(np.percentile(values, 5)) if values.size else float("nan"),
            p95=float(np.percentile(values, 95)) if values.size else float("nan"),
            mean_dmodx=_nanmean(pixels.dmodx[kept]), mean_t2=_nanmean(pixels.t2[kept]),
            share_above_dcrit=share(pixels.dmodx, prediction.dcrit_099),
            share_above_t2crit=share(pixels.t2, prediction.t2_crit_099),
            reference=reference,
            residual=None if reference is None or not values.size else mean - reference,
            interval=mean_half_width(
                None if pixels.scores is None or not kept.any()
                else pixels.scores[kept].astype(np.float64).mean(axis=0),
                prediction.intervals, level=DEFAULT_LEVEL)))
    residuals = np.array([row.residual for row in rows if row.residual is not None])
    return RoiPredictionSummary(
        model_run_id=prediction.model_run_id, model_name=prediction.model_name,
        exclusion=exclude_dmodx if limit is not None else None, rois=tuple(rows),
        rmsep=float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else None,
        bias=float(np.mean(residuals)) if residuals.size else None,
        n_reference=int(residuals.size))


def _nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _share_percentiles(shares: Any) -> tuple[float, float, float] | None:
    """10th, 50th and 90th percentile of the training ROIs' share above DCrit (0.99)."""
    if shares is None or not np.size(shares):
        return None
    values = np.asarray(shares, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    low, median, high = np.percentile(values, [10, 50, 90])
    return float(low), float(median), float(high)


__all__ = ["RoiPixels", "RoiPrediction", "RoiPredictionSummary", "RoiStatistics",
           "predict_rois", "roi_prediction_statistics"]
