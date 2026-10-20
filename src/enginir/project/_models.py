"""Typed objects of the engiNIR project workspace.

Every class validates its fields on construction and raises :class:`ValueError`
(after logging it) for missing, mistyped or contradictory values. Values are
checked, not converted: a float where an integer is required, or ``None``
where text is required, is an error. Relationships *between* objects (ids
that must exist) are checked by
:func:`enginir.project.validate_project_relationships`.

The objects are mutable dataclasses, but validation runs only on
construction; create a new object (e.g. with :func:`dataclasses.replace`)
instead of assigning fields. Classes holding NumPy arrays compare by identity.
The field semantics are specified in the ``contracts/*_v3.md`` documents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

import numpy as np

from enginir._checks import check_choice, check_integer, check_real, log_and_raise

SIGNAL_UNITS = ("DN", "Reflectance", "Absorbance")
CALIBRATION_MODES = ("raw_dn", "dark_white_reference", "precalibrated_import")
MODEL_TYPES = ("PLS", "PCR")
#: The only allowed value of each modeling-semantics field of a :class:`SampleSet`.
SAMPLE_SET_SEMANTICS = {
    "observation_unit": "pixel",
    "target_level": "roi",
    "grouping_unit": "roi",
    "validation_unit": "roi",
    "weighting_policy": "per_pixel_unweighted",
    "reporting_level": "roi",
    "roi_aggregation": "mean",
}


# ---------------------------------------------------------------------------
# Field checks
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True) -> str:
    """Return ``value`` stripped; it must be a ``str`` (non-empty if ``required``)."""
    if not isinstance(value, str):
        log_and_raise("{} must be a string, got {!r}", name, value, depth=2)
    text = value.strip()
    if required and not text:
        log_and_raise("{} must not be empty", name, depth=2)
    return text


def _timestamp(value: Any, name: str) -> str:
    """Return an ISO 8601 timestamp as given; anything ``datetime`` cannot parse is an error."""
    text = _text(value, name)
    try:
        datetime.fromisoformat(text)
    except ValueError:
        log_and_raise("{} must be an ISO 8601 timestamp, got {!r}", name, text, depth=2)
    return text


def validate_project_meta(meta: Any, *, schema_version: str) -> dict[str, str]:
    """Return validated project metadata: id, the current schema version and timestamps."""
    meta = _mapping(meta, "project_meta")
    if meta.get("schema_version") != schema_version:
        log_and_raise("project_meta schema_version must be {!r}, got {!r}", schema_version,
                      meta.get("schema_version"))
    return {
        "project_id": _text(meta.get("project_id"), "project_id"),
        "schema_version": schema_version,
        "created_at": _timestamp(meta.get("created_at"), "project created_at"),
        "updated_at": _timestamp(meta.get("updated_at"), "project updated_at"),
    }


def _optional_text(value: Any, name: str) -> str | None:
    """Return ``None`` for ``None`` or blank text, else the stripped text."""
    return None if value is None else (_text(value, name, required=False) or None)


def _text_list(value: Any, name: str) -> list[str]:
    if isinstance(value, str) or not isinstance(value, (list, tuple, np.ndarray)):
        log_and_raise("{} must be a list of strings, got {!r}", name, value, depth=2)
    return [_text(item, f"{name} entry") for item in value]


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        log_and_raise("{} must be a bool, got {!r}", name, value, depth=2)
    return bool(value)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        log_and_raise("{} must be a mapping, got {!r}", name, type(value).__name__, depth=2)
    return dict(value)


def _float_vector(value: Any, name: str, *, size: int | None = None) -> np.ndarray:
    """Return a finite, non-empty, 1-D float32 array (optionally of length ``size``)."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        log_and_raise("{} must be numeric, got dtype {}", name, array.dtype, depth=2)
    if array.ndim != 1 or array.size == 0:
        log_and_raise("{} must be a non-empty 1-D array, got shape {}", name, array.shape,
                      depth=2)
    if size is not None and array.size != size:
        log_and_raise("{} has {} values, expected {}", name, array.size, size, depth=2)
    if not np.all(np.isfinite(array)):
        log_and_raise("{} must contain only finite values", name, depth=2)
    return array.astype(np.float32)


def _wavelengths(value: Any, name: str, *, size: int | None = None) -> np.ndarray:
    """Return a wavelength axis: a strictly monotonic float32 vector of ``size`` values."""
    axis = _float_vector(value, name, size=size)
    steps = np.diff(axis)
    if axis.size > 1 and not (np.all(steps > 0) or np.all(steps < 0)):
        log_and_raise("{} must be strictly increasing or decreasing", name, depth=2)
    return axis


def _float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.size == 0 or not np.all(np.isfinite(array)):
        log_and_raise("{} must be a non-empty array of finite numbers", name, depth=2)
    return array.astype(np.float32)


def _int_tuple(value: Any, name: str, length: int, *, minimum: int) -> tuple[int, ...]:
    items = list(value) if isinstance(value, (list, tuple, np.ndarray)) else None
    if items is None or len(items) != length:
        log_and_raise("{} must hold {} integers, got {!r}", name, length, value, depth=2)
    return tuple(check_integer(item, name, minimum=minimum) for item in items)


def _choice(value: Any, name: str, choices: tuple[str, ...]) -> str:
    """Return the canonical spelling of an enum value, ignoring case and outer whitespace."""
    text = _text(value, name)
    canonical = {choice.lower(): choice for choice in choices}.get(text.lower(), text)
    return check_choice(canonical, name, choices)


def _signal_unit(value: Any, name: str) -> str:
    return _choice(value, name, SIGNAL_UNITS)


@dataclass(slots=True, eq=False)
class SerializableModel:
    """Base class providing :meth:`to_dict`."""

    def to_dict(self) -> dict[str, Any]:
        """Return the fields as a dict; top-level arrays, dicts and lists are shallow copies."""
        payload: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (np.ndarray, dict, list)):
                value = value.copy()
            payload[field.name] = value
        return payload


# ---------------------------------------------------------------------------
# Datasets and calibrations
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class DatasetRef(SerializableModel):
    """Reference to a hyperspectral data file used in the project.

    The data stay in their source file; the project stores where to find them
    and what they look like.

    Attributes
    ----------
    dataset_id : str
        Stable id.
    source_path : str
        Path of the data file.
    source_format : str
        Loader name, e.g. ``"eiff"`` or ``"hdf5"``.
    source_modality : str
        Measurement modality, e.g. ``"NIR"``.
    cube_shape : tuple of int
        ``(n_frames, height, n_bands)`` of the full cube.
    n_bands : int
        Number of spectral bands, equal to ``cube_shape[2]``.
    wavelengths : numpy.ndarray of float32, shape (n_bands,)
        Strictly monotonic wavelength axis in nm.
    loader_metadata : dict
        Metadata reported by the loader.
    import_timestamp : str
        ISO 8601 time of import.
    """

    dataset_id: str
    source_path: str
    source_format: str
    source_modality: str
    cube_shape: tuple[int, int, int]
    n_bands: int
    wavelengths: np.ndarray
    loader_metadata: dict[str, Any]
    import_timestamp: str

    def __post_init__(self) -> None:
        self.dataset_id = _text(self.dataset_id, "dataset_id")
        self.source_path = _text(self.source_path, "source_path")
        self.source_format = _text(self.source_format, "source_format")
        self.source_modality = _text(self.source_modality, "source_modality")
        self.n_bands = check_integer(self.n_bands, "n_bands", minimum=1)
        shape = _int_tuple(self.cube_shape, "cube_shape", 3, minimum=1)
        self.cube_shape = (shape[0], shape[1], shape[2])
        if self.cube_shape[2] != self.n_bands:
            log_and_raise("cube_shape {} does not have n_bands={} bands", self.cube_shape,
                          self.n_bands)
        self.wavelengths = _wavelengths(self.wavelengths, "wavelengths", size=self.n_bands)
        self.loader_metadata = _mapping(self.loader_metadata, "loader_metadata")
        self.import_timestamp = _timestamp(self.import_timestamp, "import_timestamp")


@dataclass(slots=True, eq=False)
class CalibrationSpec(SerializableModel):
    """How the raw spectra of one dataset are converted to physical units.

    ``mode`` selects the conversion:

    ``"raw_dn"``
        No conversion; spectra stay detector counts (``output_signal_unit``
        ``"DN"``). No reference context may be stored.
    ``"dark_white_reference"``
        Reflectance ``(raw - dark) / (white - dark)`` from either embedded
        dark/white reference arrays or scalar ``black_level``/``white_level``
        (not both), optionally converted to absorbance ``-log10(R)``.
        ``output_signal_unit`` is ``"Reflectance"`` or ``"Absorbance"``.
    ``"precalibrated_import"``
        Spectra are already reflectance or absorbance. No reference context
        may be stored.

    Attributes
    ----------
    calibration_id : str
        Stable id, unique within the project. Empty only for an unbound
        template that receives its ids before it is stored.
    dataset_id : str
        Dataset the calibration belongs to (empty for an unbound template).
    mode : str
        One of the modes above.
    dark_ref_source, white_ref_source : str or None
        Provenance: files the reference arrays were read from (both or
        neither; only together with the arrays).
    dark_ref_data, white_ref_data : numpy.ndarray of float32 or None
        Dark and white reference spectra, ``(n_bands,)`` or one per sensor
        pixel ``(height, n_bands)``; same shape, both or neither.
    black_level, white_level : float or None
        Scalar dark and white levels (both or neither).
    output_signal_unit : str
        ``"DN"``, ``"Reflectance"`` or ``"Absorbance"``.
    created_at : str
        ISO 8601 creation time.
    """

    calibration_id: str
    dataset_id: str
    mode: str
    dark_ref_source: str | None
    white_ref_source: str | None
    dark_ref_data: np.ndarray | None
    white_ref_data: np.ndarray | None
    black_level: float | None
    white_level: float | None
    output_signal_unit: str
    created_at: str

    def __post_init__(self) -> None:
        # Empty ids mark an unbound template (e.g. from the prediction dialog) that
        # receives its ids before it is stored; stored calibrations always have ids.
        self.calibration_id = _text(self.calibration_id, "calibration_id", required=False)
        self.dataset_id = _text(self.dataset_id, "dataset_id", required=False)
        self.mode = _choice(self.mode, "mode", CALIBRATION_MODES)
        self.output_signal_unit = _signal_unit(self.output_signal_unit, "output_signal_unit")
        self.created_at = _timestamp(self.created_at, "created_at")
        self.dark_ref_source = _optional_text(self.dark_ref_source, "dark_ref_source")
        self.white_ref_source = _optional_text(self.white_ref_source, "white_ref_source")
        for name in ("dark_ref_data", "white_ref_data"):
            value = getattr(self, name)
            setattr(self, name, None if value is None else _float_array(value, name))
        for name in ("black_level", "white_level"):
            value = getattr(self, name)
            setattr(self, name, None if value is None else check_real(value, name))
        self._check_mode()

    def same_content_as(self, other: CalibrationSpec) -> bool:
        """Whether ``other`` is the same executable calibration of the same dataset.

        Compares the dataset, mode, output unit, reference provenance, scalar
        levels and reference arrays; ids and creation times are ignored. An
        explicit calibration that matches an existing one must reuse it
        instead of creating a duplicate (dataset-calibration contract).
        """
        scalars = ("dataset_id", "mode", "output_signal_unit", "dark_ref_source",
                   "white_ref_source", "black_level", "white_level")
        if any(getattr(self, name) != getattr(other, name) for name in scalars):
            return False
        return all(
            (mine is None and theirs is None)
            or (mine is not None and theirs is not None and np.array_equal(mine, theirs))
            for mine, theirs in ((self.dark_ref_data, other.dark_ref_data),
                                 (self.white_ref_data, other.white_ref_data))
        )

    def _check_mode(self) -> None:
        has_paths = self.dark_ref_source is not None
        has_arrays = self.dark_ref_data is not None
        has_levels = self.black_level is not None
        for first, second in (("dark_ref_source", "white_ref_source"),
                              ("dark_ref_data", "white_ref_data"),
                              ("black_level", "white_level")):
            if (getattr(self, first) is None) != (getattr(self, second) is None):
                log_and_raise("Calibration '{}': {} and {} must be set together",
                              self.calibration_id, first, second)
        if self.mode != "dark_white_reference":
            expected_units = ("DN",) if self.mode == "raw_dn" else ("Reflectance", "Absorbance")
            if self.output_signal_unit not in expected_units:
                log_and_raise("Calibration '{}': mode {!r} requires output_signal_unit in {}",
                              self.calibration_id, self.mode, list(expected_units))
            if has_paths or has_arrays or has_levels:
                log_and_raise("Calibration '{}': mode {!r} must not store reference context",
                              self.calibration_id, self.mode)
            return
        if self.output_signal_unit == "DN":
            log_and_raise("Calibration '{}': mode 'dark_white_reference' produces Reflectance "
                          "or Absorbance, not DN", self.calibration_id)
        if has_arrays == has_levels:
            log_and_raise("Calibration '{}' needs either reference arrays or scalar levels "
                          "(exactly one of them)", self.calibration_id)
        if has_paths and not has_arrays:
            log_and_raise("Calibration '{}': reference source paths are provenance only and "
                          "need the reference arrays", self.calibration_id)
        if (self.dark_ref_data is not None and self.white_ref_data is not None
                and self.dark_ref_data.shape != self.white_ref_data.shape):
            log_and_raise("Calibration '{}': dark and white reference arrays differ in shape",
                          self.calibration_id)


# ---------------------------------------------------------------------------
# ROIs and targets
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class TrainingROI(SerializableModel):
    """Region of interest whose pixel spectra are stored in the project file.

    Attributes
    ----------
    roi_id : str
        Stable id.
    dataset_id, calibration_id : str
        Dataset the pixels were taken from and the calibration active then.
    name : str
        Display name (may be changed).
    wavelengths : numpy.ndarray of float32
        Wavelength axis of the stored spectra.
    n_pixels : int
        Number of stored pixel spectra.
    spatial_bounds : tuple of int
        ``(x0, x1, y0, y1)`` with ``x0 <= x1`` and ``y0 <= y1``.
    source_file : str
        Absolute path of the data file.
    created_at : str
        ISO 8601 creation time.
    provenance : dict
        Capture context with at least ``dataset_id`` and ``calibration_id``
        (matching the ROI), ``signal_unit``, ``preview_recipe_summary`` and
        ``created_at``.
    """

    roi_id: str
    dataset_id: str
    calibration_id: str
    name: str
    wavelengths: np.ndarray
    n_pixels: int
    spatial_bounds: tuple[int, int, int, int]
    source_file: str
    created_at: str
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        self.roi_id = _text(self.roi_id, "roi_id")
        self.dataset_id = _text(self.dataset_id, "dataset_id")
        self.calibration_id = _text(self.calibration_id, "calibration_id")
        self.name = _text(self.name, "name")
        self.n_pixels = check_integer(self.n_pixels, "n_pixels", minimum=1)
        self.wavelengths = _wavelengths(self.wavelengths, "wavelengths")
        x0, x1, y0, y1 = _int_tuple(self.spatial_bounds, "spatial_bounds", 4, minimum=0)
        if x0 > x1 or y0 > y1:
            log_and_raise("ROI '{}': spatial_bounds must be (x0, x1, y0, y1) with x0 <= x1 "
                          "and y0 <= y1, got {}", self.roi_id, (x0, x1, y0, y1))
        self.spatial_bounds = (x0, x1, y0, y1)
        self.source_file = os.path.abspath(_text(self.source_file, "source_file"))
        self.created_at = _timestamp(self.created_at, "created_at")
        self.provenance = self._checked_provenance(_mapping(self.provenance, "provenance"))

    def _checked_provenance(self, provenance: dict[str, Any]) -> dict[str, Any]:
        required = ("dataset_id", "calibration_id", "signal_unit", "preview_recipe_summary",
                    "created_at")
        missing = [key for key in required if key not in provenance]
        if missing:
            log_and_raise("ROI '{}': provenance is missing {}", self.roi_id, missing)
        for key in required:
            provenance[key] = _text(provenance[key], f"provenance {key}")
        for key in ("dataset_id", "calibration_id"):
            if provenance[key] != getattr(self, key):
                log_and_raise("ROI '{}': provenance {} {!r} differs from the ROI's {!r}",
                              self.roi_id, key, provenance[key], getattr(self, key))
        provenance["signal_unit"] = _signal_unit(provenance["signal_unit"],
                                                 "provenance signal_unit")
        return provenance

    @property
    def spectra_storage_key(self) -> str:
        """HDF5 path of the ROI spectra in the project file."""
        return f"/rois/{self.roi_id}/spectra"

    def load_spectra(self, project_path: str) -> np.ndarray:
        """Read the ROI spectra ``(n_pixels, n_bands)`` from the project file."""
        from ..io.project_store import load_roi_spectra

        return load_roi_spectra(project_path, self.spectra_storage_key)


@dataclass(slots=True)
class TargetDefinition(SerializableModel):
    """A measured reference property (e.g. moisture content) assigned to ROIs.

    Attributes
    ----------
    target_id : str
        Stable id.
    name : str
        Display name (may be changed).
    unit : str
        Unit of the values, may be empty.
    value_type : str
        ``"float"``.
    missing_value_policy : str
        ``"allowed"`` or ``"forbidden_for_modeling"``.
    created_at : str
        ISO 8601 creation time.
    """

    target_id: str
    name: str
    unit: str
    value_type: str
    missing_value_policy: str
    created_at: str

    def __post_init__(self) -> None:
        self.target_id = _text(self.target_id, "target_id")
        self.name = _text(self.name, "target name")
        self.unit = _text(self.unit, "unit", required=False)
        self.value_type = _choice(self.value_type, "value_type", ("float",))
        self.missing_value_policy = _choice(self.missing_value_policy, "missing_value_policy",
                                            ("allowed", "forbidden_for_modeling"))
        self.created_at = _timestamp(self.created_at, "created_at")


@dataclass(slots=True)
class TargetAssignment(SerializableModel):
    """The reference value of one target for one ROI.

    Attributes
    ----------
    roi_id, target_id : str
        The ROI and the target.
    value : float
        Finite reference value.
    assigned_at : str
        ISO 8601 time of assignment.
    """

    roi_id: str
    target_id: str
    value: float
    assigned_at: str

    def __post_init__(self) -> None:
        self.roi_id = _text(self.roi_id, "roi_id")
        self.target_id = _text(self.target_id, "target_id")
        self.value = check_real(self.value, "target value")
        self.assigned_at = _timestamp(self.assigned_at, "assigned_at")


# ---------------------------------------------------------------------------
# Modeling objects
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class SignalDomainContract(SerializableModel):
    """The spectral input a model was trained on, and that prediction must reproduce.

    Attributes
    ----------
    signal_domain_id : str
        Stable id.
    input_format : str
        Source format of the training data, e.g. ``"eiff"``.
    source_modality : str
        Measurement modality, e.g. ``"NIR"``.
    required_signal_unit : str
        ``"DN"``, ``"Reflectance"`` or ``"Absorbance"``.
    required_calibration_mode : str
        Calibration mode of the training data.
    requires_dark_ref, requires_white_ref : bool
        Whether prediction data need dark/white reference context; only
        allowed for ``"dark_white_reference"``.
    calibrated_input_allowed : bool
        Whether precalibrated input may be used.
    n_bands : int
        Number of bands.
    wavelengths : numpy.ndarray of float32, shape (n_bands,)
        Wavelength axis in nm.
    wavelength_tolerance_nm : float
        Allowed wavelength difference, ``>= 0``.
    band_mapping_policy : str
        ``"exact"`` or ``"reject"``.
    output_dtype : str
        ``"float32"``.
    created_at : str
        ISO 8601 creation time.
    """

    signal_domain_id: str
    input_format: str
    source_modality: str
    required_signal_unit: str
    required_calibration_mode: str
    requires_dark_ref: bool
    requires_white_ref: bool
    calibrated_input_allowed: bool
    n_bands: int
    wavelengths: np.ndarray
    wavelength_tolerance_nm: float
    band_mapping_policy: str
    output_dtype: str
    created_at: str

    def __post_init__(self) -> None:
        self.signal_domain_id = _text(self.signal_domain_id, "signal_domain_id")
        self.input_format = _text(self.input_format, "input_format")
        self.source_modality = _text(self.source_modality, "source_modality")
        self.required_signal_unit = _signal_unit(self.required_signal_unit,
                                                 "required_signal_unit")
        self.required_calibration_mode = _choice(
            self.required_calibration_mode, "required_calibration_mode", CALIBRATION_MODES
        )
        self.requires_dark_ref = _flag(self.requires_dark_ref, "requires_dark_ref")
        self.requires_white_ref = _flag(self.requires_white_ref, "requires_white_ref")
        self.calibrated_input_allowed = _flag(self.calibrated_input_allowed,
                                              "calibrated_input_allowed")
        produces_dn = self.required_calibration_mode == "raw_dn"
        if produces_dn != (self.required_signal_unit == "DN"):
            log_and_raise("Signal domain '{}': calibration mode {!r} cannot produce {!r} spectra",
                          self.signal_domain_id, self.required_calibration_mode,
                          self.required_signal_unit)
        if ((self.requires_dark_ref or self.requires_white_ref)
                and self.required_calibration_mode != "dark_white_reference"):
            log_and_raise("Signal domain '{}': reference requirements need calibration mode "
                          "'dark_white_reference', got {!r}", self.signal_domain_id,
                          self.required_calibration_mode)
        self.n_bands = check_integer(self.n_bands, "n_bands", minimum=1)
        self.wavelengths = _wavelengths(self.wavelengths, "wavelengths", size=self.n_bands)
        self.wavelength_tolerance_nm = check_real(self.wavelength_tolerance_nm,
                                                  "wavelength_tolerance_nm")
        if self.wavelength_tolerance_nm < 0:
            log_and_raise("wavelength_tolerance_nm must be >= 0, got {}",
                          self.wavelength_tolerance_nm)
        self.band_mapping_policy = _choice(self.band_mapping_policy, "band_mapping_policy",
                                           ("exact", "reject"))
        self.output_dtype = _choice(self.output_dtype, "output_dtype", ("float32",))
        self.created_at = _timestamp(self.created_at, "created_at")

    @classmethod
    def for_training_data(
        cls,
        *,
        dataset: DatasetRef,
        calibration: CalibrationSpec,
        required_signal_unit: str,
        signal_domain_id: str | None = None,
        created_at: str | None = None,
    ) -> SignalDomainContract:
        """Return the contract of a model trained on ``dataset`` with ``calibration``.

        Follows the writer rules of the dataset-calibration contract: the
        dataset's format, modality, bands and wavelengths; the calibration's
        mode; dark/white references required when the calibration has any
        reference context; precalibrated input allowed exactly when the
        training data were precalibrated; an ``"exact"`` band mapping, a
        wavelength tolerance of 0 and ``float32`` output.
        """
        from ._ids import new_id, now_iso

        if calibration.dataset_id != dataset.dataset_id:
            log_and_raise("Calibration '{}' belongs to dataset '{}', not '{}'",
                          calibration.calibration_id, calibration.dataset_id, dataset.dataset_id)
        has_dark = any(value is not None for value in (
            calibration.dark_ref_source, calibration.dark_ref_data, calibration.black_level))
        has_white = any(value is not None for value in (
            calibration.white_ref_source, calibration.white_ref_data, calibration.white_level))
        return cls(
            signal_domain_id=signal_domain_id or new_id(),
            input_format=dataset.source_format,
            source_modality=dataset.source_modality,
            required_signal_unit=required_signal_unit,
            required_calibration_mode=calibration.mode,
            requires_dark_ref=has_dark,
            requires_white_ref=has_white,
            calibrated_input_allowed=calibration.mode == "precalibrated_import",
            n_bands=dataset.n_bands,
            wavelengths=dataset.wavelengths,
            wavelength_tolerance_nm=0.0,
            band_mapping_policy="exact",
            output_dtype="float32",
            created_at=created_at or now_iso(),
        )


@dataclass(slots=True)
class SampleSet(SerializableModel):
    """The ROIs and target used to fit a model, with the fixed v3 modeling semantics.

    Pixels are the observations, each ROI's target value applies to all of its
    pixels, cross-validation keeps the pixels of an ROI together, and results
    are reported per ROI as the mean over its pixels. The fields
    ``observation_unit`` to ``roi_aggregation`` therefore have exactly one
    allowed value each.

    Attributes
    ----------
    sample_set_id : str
        Stable id.
    roi_ids : list of str
        ROIs in the sample set (non-empty, no duplicates).
    target_id : str
        Modeled target.
    observation_unit : str
        ``"pixel"``.
    target_level, grouping_unit, validation_unit, reporting_level : str
        ``"roi"``.
    weighting_policy : str
        ``"per_pixel_unweighted"``.
    roi_aggregation : str
        ``"mean"``.
    preprocessing_recipe_id : str
        Recipe applied before modeling.
    dataset_ids : list of str
        Datasets the ROIs come from (non-empty).
    created_at : str
        ISO 8601 creation time.
    """

    sample_set_id: str
    roi_ids: list[str]
    target_id: str
    observation_unit: str
    target_level: str
    grouping_unit: str
    validation_unit: str
    weighting_policy: str
    reporting_level: str
    roi_aggregation: str
    preprocessing_recipe_id: str
    dataset_ids: list[str]
    created_at: str

    def __post_init__(self) -> None:
        self.sample_set_id = _text(self.sample_set_id, "sample_set_id")
        self.target_id = _text(self.target_id, "target_id")
        self.preprocessing_recipe_id = _text(self.preprocessing_recipe_id,
                                             "preprocessing_recipe_id")
        self.roi_ids = _text_list(self.roi_ids, "roi_ids")
        if not self.roi_ids:
            log_and_raise("Sample set '{}' has no roi_ids", self.sample_set_id)
        if len(set(self.roi_ids)) != len(self.roi_ids):
            log_and_raise("Sample set '{}' lists an ROI more than once: {}", self.sample_set_id,
                          self.roi_ids)
        self.dataset_ids = _text_list(self.dataset_ids, "dataset_ids")
        if not self.dataset_ids:
            log_and_raise("Sample set '{}' has no dataset_ids", self.sample_set_id)
        for name, allowed in SAMPLE_SET_SEMANTICS.items():
            setattr(self, name, _choice(getattr(self, name), name, (allowed,)))
        self.created_at = _timestamp(self.created_at, "created_at")


@dataclass(slots=True, eq=False)
class ModelRun(SerializableModel):
    """One fitted model with its cross-validation results and saved artifacts.

    Attributes
    ----------
    model_run_id : str
        Stable id.
    sample_set_id, signal_domain_id : str
        Training data and their spectral domain.
    model_type : str
        ``"PLS"`` or ``"PCR"``.
    hyperparameters : dict
        Estimator parameters, e.g. ``{"n_components": 5}``.
    cv_spec : dict
        Cross-validation scheme.
    metrics : dict
        Figures of merit (ROI-level, with pixel-level variants).
    artifact_refs : dict
        Saved artifacts, e.g. ``prediction_model`` and ``diagnostics``.
    created_at : str
        ISO 8601 creation time.
    name : str or None
        Display name (a label that may change); ``None`` for runs saved
        without one.
    """

    model_run_id: str
    sample_set_id: str
    signal_domain_id: str
    model_type: str
    hyperparameters: dict[str, Any]
    cv_spec: dict[str, Any]
    metrics: dict[str, Any]
    artifact_refs: dict[str, Any]
    created_at: str
    name: str | None = None

    def __post_init__(self) -> None:
        self.model_run_id = _text(self.model_run_id, "model_run_id")
        if self.name is not None:
            self.name = _text(self.name, "name")
        self.sample_set_id = _text(self.sample_set_id, "sample_set_id")
        self.signal_domain_id = _text(self.signal_domain_id, "signal_domain_id")
        self.model_type = _choice(self.model_type, "model_type", MODEL_TYPES)
        for name in ("hyperparameters", "cv_spec", "metrics", "artifact_refs"):
            setattr(self, name, _mapping(getattr(self, name), name))
        self.created_at = _timestamp(self.created_at, "created_at")


@dataclass(slots=True, eq=False)
class PredictionRun(SerializableModel):
    """Predictions of one model run for one dataset.

    Attributes
    ----------
    prediction_run_id : str
        Stable id.
    dataset_id, calibration_id : str
        Predicted dataset and the calibration used.
    model_run_id, signal_domain_id, preprocessing_recipe_id : str
        Model and the spectral domain and recipe it was applied with.
    artifact_refs : dict
        Prediction outputs (``y_pred``, ``spatial_map``) and their metadata.
    created_at : str
        ISO 8601 creation time.
    """

    prediction_run_id: str
    dataset_id: str
    calibration_id: str
    model_run_id: str
    signal_domain_id: str
    preprocessing_recipe_id: str
    artifact_refs: dict[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        for name in ("prediction_run_id", "dataset_id", "calibration_id", "model_run_id",
                     "signal_domain_id", "preprocessing_recipe_id", "created_at"):
            setattr(self, name, _text(getattr(self, name), name))
        self.artifact_refs = _mapping(self.artifact_refs, "artifact_refs")


__all__ = [
    "validate_project_meta",
    "CALIBRATION_MODES",
    "MODEL_TYPES",
    "SAMPLE_SET_SEMANTICS",
    "SIGNAL_UNITS",
    "CalibrationSpec",
    "DatasetRef",
    "ModelRun",
    "PredictionRun",
    "SampleSet",
    "SerializableModel",
    "SignalDomainContract",
    "TargetAssignment",
    "TargetDefinition",
    "TrainingROI",
]
