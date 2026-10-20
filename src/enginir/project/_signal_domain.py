"""Conversion of raw spectra into the signal domain of a model.

Training (:func:`enginir.modeling.fit_model_run`) and prediction
(:func:`enginir.prediction.predict_with_model_run`) both use
:func:`transform_spectra_to_signal_domain`, so a model always sees spectra
computed in exactly the same way.

For a ``"dark_white_reference"`` calibration, raw detector counts ``x`` are
converted to reflectance

.. math:: R = \\operatorname{clip}\\left(\\frac{x - d}{w - d},\\ 10^{-6},\\ 1\\right)

with the dark and white references ``d`` and ``w`` (per band, per sensor
pixel, or scalar levels), and optionally to absorbance :math:`A = -\\log_{10} R`.
Denominators ``|w - d| < 1e-10`` are replaced by ``+/-1e-10``. Clipping and
replaced denominators are counted in :class:`CalibrationStats` and logged.
All results are ``float32``. Raw spectra stay in their source dtype until
this conversion, which allocates at most one float32 working array; float32
input that needs no conversion is returned without a copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NoReturn

import numpy as np

from enginir._checks import log_and_raise
from enginir._logging import get_logger

from ._models import CalibrationSpec, DatasetRef, SignalDomainContract

logger = get_logger(__name__)

REFLECTANCE_MIN = np.float32(1e-6)
REFLECTANCE_MAX = np.float32(1.0)
_EPS = np.float32(1e-10)
_WORKING_DTYPE = np.float32
_FORMAT_ALIASES = {"eiff_raw": "eiff"}


class SignalDomainCompatibilityError(ValueError):
    """A dataset or calibration cannot produce the spectra a model was trained on."""


@dataclass(slots=True)
class CalibrationStats:
    """Counters of a signal-domain conversion, accumulated over calls.

    Attributes
    ----------
    clip_low_count, clip_high_count : int
        Reflectance values raised to ``1e-6`` or lowered to ``1``.
    near_zero_denominator_count : int
        Reference denominators ``|w - d| < 1e-10`` that were replaced.
    """

    clip_low_count: int = 0
    clip_high_count: int = 0
    near_zero_denominator_count: int = 0
    # Denominator w - d of the reference arrays it was computed from, reused
    # across chunks of the same calibration.
    _denominator: np.ndarray | None = field(default=None, repr=False)
    _denominator_source: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)


def _fail(message: str, *args: Any) -> NoReturn:
    log_and_raise(message, *args, error=SignalDomainCompatibilityError, depth=3)


def check_signal_domain_compatibility(
    *,
    dataset: DatasetRef,
    calibration: CalibrationSpec,
    signal_domain: SignalDomainContract,
) -> None:
    """Check that ``dataset`` with ``calibration`` can produce ``signal_domain`` spectra.

    The source format, modality and band count of the dataset must match the
    contract, embedded dark/white references must be one spectrum
    ``(n_bands,)`` or one per sensor pixel ``(height, n_bands)`` of the
    dataset, and the calibration must pass
    :func:`check_calibration_signal_domain_compatibility`.

    Raises
    ------
    SignalDomainCompatibilityError
    """
    dataset_format = _FORMAT_ALIASES.get(dataset.source_format.lower(), dataset.source_format.lower())
    domain_format = _FORMAT_ALIASES.get(signal_domain.input_format.lower(),
                                        signal_domain.input_format.lower())
    if dataset_format != domain_format:
        _fail("Dataset '{}' has source_format {!r}, but signal domain '{}' requires {!r}",
              dataset.dataset_id, dataset.source_format, signal_domain.signal_domain_id,
              signal_domain.input_format)
    if dataset.source_modality != signal_domain.source_modality:
        _fail("Dataset '{}' has source_modality {!r}, but the model requires {!r}",
              dataset.dataset_id, dataset.source_modality, signal_domain.source_modality)
    if dataset.n_bands != signal_domain.n_bands:
        _fail("Dataset '{}' has n_bands={}, but the model requires n_bands={}",
              dataset.dataset_id, dataset.n_bands, signal_domain.n_bands)
    if calibration.dark_ref_data is not None:
        allowed = ((dataset.n_bands,), (dataset.cube_shape[1], dataset.n_bands))
        if calibration.dark_ref_data.shape not in allowed:
            _fail("Calibration '{}' references have shape {}, but dataset '{}' needs {} or {}",
                  calibration.calibration_id, calibration.dark_ref_data.shape,
                  dataset.dataset_id, *allowed)
    check_calibration_signal_domain_compatibility(calibration=calibration,
                                                  signal_domain=signal_domain)


def check_calibration_signal_domain_compatibility(
    *,
    calibration: CalibrationSpec,
    signal_domain: SignalDomainContract,
) -> None:
    """Check that ``calibration`` can produce the signal unit ``signal_domain`` requires.

    The calibration mode must equal the required mode (or be
    ``"precalibrated_import"`` where the contract allows calibrated input),
    required dark/white references must be present, and the calibration's
    output unit must equal the required unit or convert to it (reflectance and
    absorbance convert into each other).

    Raises
    ------
    SignalDomainCompatibilityError
    """
    mode = calibration.mode
    mode_ok = mode == signal_domain.required_calibration_mode or (
        mode == "precalibrated_import" and signal_domain.calibrated_input_allowed
    )
    if not mode_ok:
        _fail("Calibration '{}' has mode={!r}, but the model requires "
              "required_calibration_mode={!r}", calibration.calibration_id, mode,
              signal_domain.required_calibration_mode)
    if mode == "dark_white_reference":
        has_references = calibration.dark_ref_data is not None or calibration.black_level is not None
        if (signal_domain.requires_dark_ref or signal_domain.requires_white_ref) and not has_references:
            _fail("Calibration '{}' has no dark/white reference", calibration.calibration_id)
    if not _can_produce(calibration.output_signal_unit, signal_domain.required_signal_unit):
        _fail("Calibration '{}' produces {!r} and cannot provide required_signal_unit={!r}",
              calibration.calibration_id, calibration.output_signal_unit,
              signal_domain.required_signal_unit)


def transform_spectra_to_signal_domain(
    raw_spectra: np.ndarray,
    *,
    dataset: DatasetRef,
    calibration: CalibrationSpec,
    signal_domain: SignalDomainContract,
    spatial_shape: tuple[int, int] | None = None,
    stats: CalibrationStats | None = None,
    check_compatibility: bool = True,
    per_spectrum_references: bool = False,
) -> np.ndarray:
    """Calibrate raw spectra and convert them to the model's signal unit.

    Parameters
    ----------
    raw_spectra:
        Spectra as stored (e.g. uint16 counts), shape ``(n_spectra, n_bands)``.
    dataset, calibration, signal_domain:
        Source dataset, its calibration and the model's signal-domain contract.
    spatial_shape:
        ``(n_frames, height)`` of the rows of ``raw_spectra``. Needed when the
        calibration stores one reference spectrum per sensor pixel,
        ``(height, n_bands)``. If omitted, the rows are taken to be whole
        frames of the dataset (``height = dataset.cube_shape[1]``), starting
        at a frame boundary.
    stats:
        Optional counters to accumulate clipping and flooring statistics into.
    check_compatibility:
        Run :func:`check_signal_domain_compatibility` first. Callers that
        convert many chunks of one dataset check once and pass ``False``.
    per_spectrum_references:
        If ``True``, the calibration's reference arrays hold one spectrum per
        row of ``raw_spectra`` (e.g. the reference snapshot stored with an
        ROI) and must have its shape. Combine with
        ``check_compatibility=False`` after checking the dataset calibration
        itself.

    Returns
    -------
    numpy.ndarray of float32, shape (n_spectra, n_bands)

    Raises
    ------
    SignalDomainCompatibilityError
        If the inputs are incompatible.
    ValueError
        If ``raw_spectra`` is not a finite real 2-D array.
    """
    if check_compatibility:
        check_signal_domain_compatibility(dataset=dataset, calibration=calibration,
                                          signal_domain=signal_domain)
    raw = _raw_matrix(raw_spectra)
    if spatial_shape is None:
        spatial_shape = _whole_frames_spatial_shape(raw, dataset)
    stats = CalibrationStats() if stats is None else stats

    if calibration.mode == "dark_white_reference":
        signal = _reflectance(raw, calibration, spatial_shape, stats, per_spectrum_references)
        unit = "Reflectance"
    else:
        # Raw counts are converted to float32 here, once; float32 input is not copied.
        signal = np.asarray(raw, dtype=_WORKING_DTYPE)
        unit = calibration.output_signal_unit
    target = signal_domain.required_signal_unit
    if unit != target and np.shares_memory(signal, raw):
        signal = signal.copy()  # the unit conversion below works in place
    if unit == "Reflectance" and target == "Absorbance":
        _reflectance_to_absorbance(signal, calibration, stats)
    elif unit == "Absorbance" and target == "Reflectance":
        np.negative(signal, out=signal)
        np.power(_WORKING_DTYPE(10.0), signal, out=signal)
    elif unit != target:
        _fail("Cannot convert {!r} spectra to {!r}", unit, target)
    return signal


def _can_produce(unit: str, required: str) -> bool:
    return unit == required or {unit, required} == {"Reflectance", "Absorbance"}


def _raw_matrix(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        log_and_raise("raw_spectra must be a real numeric array, got dtype {}", raw.dtype,
                      depth=3)
    if raw.ndim != 2 or 0 in raw.shape:
        log_and_raise("raw_spectra must be a 2-D matrix with at least one spectrum and band, "
                      "got shape {}", raw.shape, depth=3)
    if raw.dtype.kind == "f" and not np.all(np.isfinite(raw)):
        log_and_raise("raw_spectra must contain only finite values", depth=3)
    return raw


def _whole_frames_spatial_shape(raw: np.ndarray, dataset: DatasetRef) -> tuple[int, int] | None:
    height = dataset.cube_shape[1]
    if height > 0 and raw.shape[0] % height == 0:
        return raw.shape[0] // height, height
    return None


def _reflectance(
    raw: np.ndarray,
    calibration: CalibrationSpec,
    spatial_shape: tuple[int, int] | None,
    stats: CalibrationStats,
    per_spectrum_references: bool,
) -> np.ndarray:
    """Return clipped reflectance ``(raw - d) / (w - d)`` as float32."""
    out = np.empty(raw.shape, dtype=_WORKING_DTYPE)
    if calibration.dark_ref_data is not None and calibration.white_ref_data is not None:
        dark = calibration.dark_ref_data
        white = calibration.white_ref_data
        cached = stats._denominator_source
        if stats._denominator is None or cached is None or cached[0] is not dark \
                or cached[1] is not white:
            stats._denominator = _safe_denominator(white - dark, calibration, stats)
            stats._denominator_source = (dark, white)
        denominator = stats._denominator
        rows_shape = _reference_layout(raw, dark, spatial_shape, calibration,
                                       per_spectrum_references)
        np.subtract(raw.reshape(rows_shape), dark, out=out.reshape(rows_shape))
        np.divide(out.reshape(rows_shape), denominator, out=out.reshape(rows_shape))
    elif calibration.black_level is not None and calibration.white_level is not None:
        denominator = _safe_denominator(
            np.asarray(calibration.white_level - calibration.black_level, dtype=_WORKING_DTYPE),
            calibration, stats,
        )
        np.subtract(raw, _WORKING_DTYPE(calibration.black_level), out=out)
        np.divide(out, denominator, out=out)
    else:
        _fail("Calibration '{}' has no dark/white reference", calibration.calibration_id)
    _clip_reflectance(out, calibration, stats)
    return out


def _safe_denominator(
    denominator: np.ndarray, calibration: CalibrationSpec, stats: CalibrationStats
) -> np.ndarray:
    denominator = np.asarray(denominator, dtype=_WORKING_DTYPE)
    small = np.abs(denominator) < _EPS
    n_small = int(np.count_nonzero(small))
    if n_small:
        stats.near_zero_denominator_count += n_small
        logger.warning("Calibration '{}': {} white - dark value(s) are near zero; using +/-{}",
                       calibration.calibration_id, n_small, float(_EPS))
        denominator = np.where(small, np.where(denominator >= 0, _EPS, -_EPS), denominator)
    return denominator.astype(_WORKING_DTYPE, copy=False)


def _reference_layout(
    raw: np.ndarray,
    reference: np.ndarray,
    spatial_shape: tuple[int, int] | None,
    calibration: CalibrationSpec,
    per_spectrum_references: bool,
) -> tuple[int, ...]:
    """Return the shape to view ``raw`` in so that ``reference`` broadcasts against it.

    References are one spectrum ``(n_bands,)``, one per sensor pixel
    ``(height, n_bands)``, which needs the ``(n_frames, height)`` layout, or,
    with ``per_spectrum_references``, one per row of ``raw`` (ROI snapshots).
    """
    n_bands = raw.shape[1]
    if per_spectrum_references:
        if reference.shape != raw.shape:
            _fail("Calibration '{}' per-spectrum references of shape {} do not match spectra "
                  "of shape {}", calibration.calibration_id, reference.shape, raw.shape)
        return raw.shape
    if reference.shape == (n_bands,):
        return raw.shape
    if spatial_shape is None:
        _fail("Calibration '{}' has references of shape {} for spectra of shape {}; the "
              "spectra's spatial layout is needed", calibration.calibration_id,
              reference.shape, raw.shape)
    n_frames, height = spatial_shape
    if n_frames * height != raw.shape[0] or reference.shape != (height, n_bands):
        _fail("Calibration '{}' references of shape {} do not fit spectra of shape {} with "
              "spatial layout {}", calibration.calibration_id, reference.shape, raw.shape,
              spatial_shape)
    return (n_frames, height, n_bands)


def _clip_reflectance(out: np.ndarray, calibration: CalibrationSpec,
                      stats: CalibrationStats) -> None:
    n_low = int(np.count_nonzero(out < REFLECTANCE_MIN))
    n_high = int(np.count_nonzero(out > REFLECTANCE_MAX))
    stats.clip_low_count += n_low
    stats.clip_high_count += n_high
    if n_low or n_high:
        logger.warning("Calibration '{}': clipped {} reflectance value(s) below {} and {} "
                       "above {}", calibration.calibration_id, n_low, float(REFLECTANCE_MIN),
                       n_high, float(REFLECTANCE_MAX))
    np.clip(out, REFLECTANCE_MIN, REFLECTANCE_MAX, out=out)


def _reflectance_to_absorbance(reflectance: np.ndarray, calibration: CalibrationSpec,
                               stats: CalibrationStats) -> None:
    # Clipped dark/white reflectance is already >= REFLECTANCE_MIN; precalibrated
    # reflectance may not be, and log10 needs a positive floor.
    n_low = int(np.count_nonzero(reflectance < REFLECTANCE_MIN))
    if n_low:
        stats.clip_low_count += n_low
        logger.warning("Calibration '{}': raised {} reflectance value(s) below {} to {} before "
                       "converting to absorbance", calibration.calibration_id, n_low,
                       float(REFLECTANCE_MIN), float(REFLECTANCE_MIN))
    np.maximum(reflectance, REFLECTANCE_MIN, out=reflectance)
    np.log10(reflectance, out=reflectance)
    np.negative(reflectance, out=reflectance)


__all__ = [
    "REFLECTANCE_MAX",
    "REFLECTANCE_MIN",
    "CalibrationStats",
    "SignalDomainCompatibilityError",
    "check_calibration_signal_domain_compatibility",
    "check_signal_domain_compatibility",
    "transform_spectra_to_signal_domain",
]
