"""Tests for v3 signal-domain compatibility rules."""

from __future__ import annotations

import numpy as np
import pytest

from enginir.project import (
    CalibrationStats,
    SignalDomainCompatibilityError,
    check_signal_domain_compatibility,
    transform_spectra_to_signal_domain,
)
from enginir.project import CalibrationSpec, DatasetRef, SignalDomainContract


def _dataset(
    *,
    wavelengths: np.ndarray,
    source_modality: str = "nir_hsi",
    source_format: str = "hdf5",
) -> DatasetRef:
    return DatasetRef(
        dataset_id="dataset_1",
        source_path="/data/sample.h5",
        source_format=source_format,
        source_modality=source_modality,
        cube_shape=(2, 2, int(wavelengths.shape[0])),
        n_bands=int(wavelengths.shape[0]),
        wavelengths=wavelengths,
        loader_metadata={},
        import_timestamp="2026-04-02T10:00:00+00:00",
    )


def _signal_domain(
    *,
    wavelengths: np.ndarray,
    input_format: str = "hdf5",
    calibrated_input_allowed: bool = False,
    wavelength_tolerance_nm: float = 0.0,
    band_mapping_policy: str = "exact",
    requires_dark_ref: bool = True,
    requires_white_ref: bool = True,
    output_dtype: str = "float32",
) -> SignalDomainContract:
    return SignalDomainContract(
        signal_domain_id="signal_domain_1",
        input_format=input_format,
        source_modality="nir_hsi",
        required_signal_unit="Absorbance",
        required_calibration_mode="dark_white_reference",
        requires_dark_ref=requires_dark_ref,
        requires_white_ref=requires_white_ref,
        calibrated_input_allowed=calibrated_input_allowed,
        n_bands=int(wavelengths.shape[0]),
        wavelengths=wavelengths,
        wavelength_tolerance_nm=wavelength_tolerance_nm,
        band_mapping_policy=band_mapping_policy,
        output_dtype=output_dtype,
        created_at="2026-04-02T10:01:00+00:00",
    )


def _calibration(
    *,
    mode: str = "dark_white_reference",
    output_signal_unit: str = "Absorbance",
    dark_ref_data: np.ndarray | None = None,
    white_ref_data: np.ndarray | None = None,
) -> CalibrationSpec:
    black_level = None
    white_level = None
    if mode == "dark_white_reference" and dark_ref_data is None and white_ref_data is None:
        black_level = 0.0
        white_level = 100.0
    return CalibrationSpec(
        calibration_id="cal_1",
        dataset_id="dataset_1",
        mode=mode,
        dark_ref_source=None,
        white_ref_source=None,
        dark_ref_data=dark_ref_data,
        white_ref_data=white_ref_data,
        black_level=black_level,
        white_level=white_level,
        output_signal_unit=output_signal_unit,
        created_at="2026-04-02T10:02:00+00:00",
    )


WL = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
REFS = {"dark_ref_data": np.zeros(3, np.float32), "white_ref_data": np.ones(3, np.float32)}
PRECALIBRATED = {"mode": "precalibrated_import", "output_signal_unit": "Absorbance"}
NO_REFS = {"requires_dark_ref": False, "requires_white_ref": False}


@pytest.mark.parametrize(
    ("dataset", "calibration", "domain", "error"),
    [
        ({}, REFS, {}, None),
        # wavelength values are metadata, not a gate, whatever the declared tolerance
        ({"wavelengths": np.array([900.0, 950.0, 1000.0], np.float32)}, REFS,
         {"wavelength_tolerance_nm": 0.001}, None),
        # formats compare case-insensitively, with legacy "eiff_raw" meaning "eiff"
        ({"source_format": "EIFF"}, REFS, {"input_format": "eiff_raw"}, None),
        ({"source_format": "eiff"}, REFS, {"input_format": "hdf5"}, "source_format"),
        ({"source_modality": "swir_line_scan"}, REFS, {}, "source_modality"),
        ({"wavelengths": WL[:2]}, {}, {}, "n_bands"),
        ({}, PRECALIBRATED, {"calibrated_input_allowed": True, **NO_REFS}, None),
        ({}, PRECALIBRATED, {"calibrated_input_allowed": False}, "required_calibration_mode"),
    ],
    ids=["match", "wavelength-values", "format-alias-and-case", "format", "modality", "band-count",
         "precalibrated-allowed", "precalibrated-refused"],
)
def test_signal_domain_compatibility(dataset, calibration, domain, error):
    """Executable contract checks: format, modality, band count and calibration mode."""
    kwargs = {
        "dataset": _dataset(**{"wavelengths": WL, **dataset}),
        "calibration": _calibration(**calibration),
        "signal_domain": _signal_domain(**{"wavelengths": WL, **domain}),
    }
    if error is None:
        check_signal_domain_compatibility(**kwargs)
    else:
        with pytest.raises(SignalDomainCompatibilityError, match=error):
            check_signal_domain_compatibility(**kwargs)


def test_signal_domain_contract_rejects_non_float32_output_dtype():
    with pytest.raises(ValueError, match="output_dtype must be one of"):
        _signal_domain(wavelengths=WL, output_dtype="float64")


def _references(height: int, n_bands: int) -> tuple[np.ndarray, np.ndarray]:
    dark = np.zeros((height, n_bands), dtype=np.float32)
    white = np.repeat(10.0 ** np.arange(1, height + 1, dtype=np.float32)[:, None], n_bands, axis=1)
    return dark, white


def test_per_sensor_references_follow_the_frame_layout():
    wavelengths = np.linspace(1000.0, 1100.0, 4)
    dataset = _dataset(wavelengths=wavelengths)  # cube_shape (2, 2, 4): height 2
    domain = _signal_domain(wavelengths=wavelengths)
    domain.required_signal_unit = "Reflectance"
    dark, white = _references(2, 4)
    calibration = _calibration(output_signal_unit="Reflectance", dark_ref_data=dark,
                               white_ref_data=white)
    raw = np.full((4, 4), 5, dtype=np.uint16)  # two frames of two sensor rows

    out = transform_spectra_to_signal_domain(raw, dataset=dataset, calibration=calibration,
                                             signal_domain=domain)
    np.testing.assert_allclose(out[:, 0], [0.5, 0.05, 0.5, 0.05])

    with pytest.raises(SignalDomainCompatibilityError, match="spatial layout"):
        transform_spectra_to_signal_domain(raw[:3], dataset=dataset, calibration=calibration,
                                           signal_domain=domain)
    with pytest.raises(SignalDomainCompatibilityError, match="per-spectrum"):
        transform_spectra_to_signal_domain(raw[:3], dataset=dataset, calibration=calibration,
                                           signal_domain=domain, check_compatibility=False,
                                           per_spectrum_references=True)


def test_compatibility_rejects_references_that_do_not_fit_the_dataset():
    wavelengths = np.linspace(1000.0, 1100.0, 4)
    dark, white = _references(3, 4)  # the dataset has height 2
    calibration = _calibration(dark_ref_data=dark, white_ref_data=white)

    with pytest.raises(SignalDomainCompatibilityError, match="references have shape"):
        check_signal_domain_compatibility(dataset=_dataset(wavelengths=wavelengths),
                                          calibration=calibration,
                                          signal_domain=_signal_domain(wavelengths=wavelengths))


def test_precalibrated_reflectance_floor_is_counted():
    wavelengths = np.linspace(1000.0, 1100.0, 4)
    calibration = _calibration(mode="precalibrated_import", output_signal_unit="Reflectance")
    stats = CalibrationStats()
    raw = np.array([[0.0, -0.1, 0.5, 1.0]], dtype=np.float32)

    out = transform_spectra_to_signal_domain(
        raw, dataset=_dataset(wavelengths=wavelengths), calibration=calibration,
        signal_domain=_signal_domain(wavelengths=wavelengths, calibrated_input_allowed=True,
                                     requires_dark_ref=False, requires_white_ref=False),
        stats=stats,
    )
    np.testing.assert_allclose(out[0], [6.0, 6.0, np.log10(2.0), 0.0], atol=1e-6)
    assert stats.clip_low_count == 2


def test_absorbance_converts_back_to_reflectance():
    wavelengths = np.linspace(1000.0, 1100.0, 4)
    calibration = _calibration(mode="precalibrated_import", output_signal_unit="Absorbance")
    domain = _signal_domain(wavelengths=wavelengths, calibrated_input_allowed=True,
                            requires_dark_ref=False, requires_white_ref=False)
    domain.required_signal_unit = "Reflectance"
    raw = np.array([[0.0, 1.0, 2.0, 0.5]], dtype=np.float32)

    out = transform_spectra_to_signal_domain(raw, dataset=_dataset(wavelengths=wavelengths),
                                             calibration=calibration, signal_domain=domain)
    np.testing.assert_allclose(out[0], 10.0 ** -raw[0], rtol=1e-6)


def test_signal_domain_rejects_a_unit_its_calibration_mode_cannot_produce():
    wavelengths = np.linspace(1000.0, 1100.0, 4)
    with pytest.raises(ValueError, match="cannot produce 'DN'"):
        SignalDomainContract(
            signal_domain_id="sd", input_format="hdf5", source_modality="nir_hsi",
            required_signal_unit="DN", required_calibration_mode="dark_white_reference",
            requires_dark_ref=True, requires_white_ref=True, calibrated_input_allowed=False,
            n_bands=4, wavelengths=wavelengths, wavelength_tolerance_nm=0.0,
            band_mapping_policy="exact", output_dtype="float32", created_at="2026-01-01T00:00:00",
        )


def test_signal_domain_for_training_data_follows_the_contract_writer_rules():
    from dataclasses import replace

    dataset = _dataset(wavelengths=WL, source_format="eiff")
    levels = _calibration(output_signal_unit="Reflectance")  # scalar black/white levels
    domain = SignalDomainContract.for_training_data(dataset=dataset, calibration=levels,
                                                    required_signal_unit="Absorbance")
    assert (domain.input_format, domain.source_modality, domain.n_bands) == ("eiff", "nir_hsi", 3)
    assert domain.required_calibration_mode == "dark_white_reference"
    assert domain.requires_dark_ref and domain.requires_white_ref
    assert not domain.calibrated_input_allowed
    assert (domain.band_mapping_policy, domain.wavelength_tolerance_nm) == ("exact", 0.0)
    np.testing.assert_array_equal(domain.wavelengths, WL)

    precalibrated = _calibration(mode="precalibrated_import", output_signal_unit="Reflectance")
    domain = SignalDomainContract.for_training_data(dataset=dataset, calibration=precalibrated,
                                                    required_signal_unit="Reflectance")
    assert domain.calibrated_input_allowed
    assert not (domain.requires_dark_ref or domain.requires_white_ref)

    with pytest.raises(ValueError, match="belongs to dataset"):
        SignalDomainContract.for_training_data(dataset=dataset,
                                               calibration=replace(levels, dataset_id="other"),
                                               required_signal_unit="Absorbance")
