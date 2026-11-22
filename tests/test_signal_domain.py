"""Tests for v3 signal-domain compatibility rules."""

from __future__ import annotations

import numpy as np
import pytest

from tests._tables import check_all

from enginir.project import (
    ReferenceCorrectionStats,
    SignalDomainCompatibilityError,
    check_signal_domain_compatibility,
    transform_spectra_to_signal_domain,
)
from enginir.project import ReferenceCorrection, DatasetRecord, SignalDomainContract
from tests import _project_payloads as build


def _dataset(
    *,
    wavelengths: np.ndarray,
    source_modality: str = "nir_hsi",
    source_format: str = "hdf5",
) -> DatasetRecord:
    return build.dataset(
        wavelengths,
        dataset_id="dataset_1",
        source_path="/data/sample.h5",
        source_format=source_format,
        source_modality=source_modality,
        cube_shape=(2, 2, int(wavelengths.shape[0])),
        n_bands=int(wavelengths.shape[0]),
    )


def _signal_domain(
    *,
    wavelengths: np.ndarray,
    input_format: str = "hdf5",
    corrected_input_allowed: bool = False,
    wavelength_tolerance_nm: float = 0.0,
    band_mapping_policy: str = "exact",
    requires_dark_ref: bool = True,
    requires_white_ref: bool = True,
    output_dtype: str = "float32",
) -> SignalDomainContract:
    return build.signal_domain(
        wavelengths,
        signal_domain_id="signal_domain_1",
        input_format=input_format,
        required_signal_unit="Absorbance",
        required_correction_mode="dark_white_reference",
        requires_dark_ref=requires_dark_ref,
        requires_white_ref=requires_white_ref,
        corrected_input_allowed=corrected_input_allowed,
        n_bands=int(wavelengths.shape[0]),
        wavelength_tolerance_nm=wavelength_tolerance_nm,
        band_mapping_policy=band_mapping_policy,
        output_dtype=output_dtype,
    )


def _calibration(
    *,
    mode: str = "dark_white_reference",
    output_signal_unit: str = "Absorbance",
    dark_ref_data: np.ndarray | None = None,
    white_ref_data: np.ndarray | None = None,
) -> ReferenceCorrection:
    black_level = None
    white_level = None
    if mode == "dark_white_reference" and dark_ref_data is None and white_ref_data is None:
        black_level = 0.0
        white_level = 100.0
    return build.reference_correction(
        reference_correction_id="cal_1",
        dataset_id="dataset_1",
        mode=mode,
        dark_ref_data=dark_ref_data,
        white_ref_data=white_ref_data,
        black_level=black_level,
        white_level=white_level,
        output_signal_unit=output_signal_unit,
    )


WL = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
REFS = {"dark_ref_data": np.zeros(3, np.float32), "white_ref_data": np.ones(3, np.float32)}
PRECALIBRATED = {"mode": "already_corrected", "output_signal_unit": "Absorbance"}
NO_REFS = {"requires_dark_ref": False, "requires_white_ref": False}


def test_signal_domain_compatibility():
    def check(case):
        dataset, reference_correction, domain, error = case
        """Executable contract checks: format, modality, band count and calibration mode."""
        kwargs = {
            "dataset": _dataset(**{"wavelengths": WL, **dataset}),
            "reference_correction": _calibration(**reference_correction),
            "signal_domain": _signal_domain(**{"wavelengths": WL, **domain}),
        }
        if error is None:
            check_signal_domain_compatibility(**kwargs)
        else:
            with pytest.raises(SignalDomainCompatibilityError, match=error):
                check_signal_domain_compatibility(**kwargs)

    check_all([
        ({}, REFS, {}, None),
        # wavelength values are metadata, not a gate, whatever the declared tolerance
        ({"wavelengths": np.array([900.0, 950.0, 1000.0], np.float32)}, REFS,
         {"wavelength_tolerance_nm": 0.001}, None),
        # formats compare case-insensitively; the earlier "eiff_raw" is no alias any more
        ({"source_format": "EIFF"}, REFS, {"input_format": "eiff"}, None),
        ({"source_format": "eiff"}, REFS, {"input_format": "eiff_raw"}, "source_format"),
        ({"source_format": "eiff"}, REFS, {"input_format": "hdf5"}, "source_format"),
        ({"source_modality": "swir_line_scan"}, REFS, {}, "source_modality"),
        ({"wavelengths": WL[:2]}, {}, {}, "n_bands"),
        ({}, PRECALIBRATED, {"corrected_input_allowed": True, **NO_REFS}, None),
        ({}, PRECALIBRATED, {"corrected_input_allowed": False}, "required_correction_mode"),
    ], check)


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
    reference_correction = _calibration(output_signal_unit="Reflectance", dark_ref_data=dark,
                               white_ref_data=white)
    raw = np.full((4, 4), 5, dtype=np.uint16)  # two frames of two sensor rows

    out = transform_spectra_to_signal_domain(raw, dataset=dataset, reference_correction=reference_correction,
                                             signal_domain=domain)
    np.testing.assert_allclose(out[:, 0], [0.5, 0.05, 0.5, 0.05])

    with pytest.raises(SignalDomainCompatibilityError, match="spatial layout"):
        transform_spectra_to_signal_domain(raw[:3], dataset=dataset, reference_correction=reference_correction,
                                           signal_domain=domain)
    with pytest.raises(SignalDomainCompatibilityError, match="per-spectrum"):
        transform_spectra_to_signal_domain(raw[:3], dataset=dataset, reference_correction=reference_correction,
                                           signal_domain=domain, check_compatibility=False,
                                           per_spectrum_references=True)


def test_compatibility_rejects_references_that_do_not_fit_the_dataset():
    wavelengths = np.linspace(1000.0, 1100.0, 4)
    dark, white = _references(3, 4)  # the dataset has height 2
    reference_correction = _calibration(dark_ref_data=dark, white_ref_data=white)

    with pytest.raises(SignalDomainCompatibilityError, match="references have shape"):
        check_signal_domain_compatibility(dataset=_dataset(wavelengths=wavelengths),
                                          reference_correction=reference_correction,
                                          signal_domain=_signal_domain(wavelengths=wavelengths))


def test_precalibrated_reflectance_floor_is_counted():
    wavelengths = np.linspace(1000.0, 1100.0, 4)
    reference_correction = _calibration(mode="already_corrected", output_signal_unit="Reflectance")
    stats = ReferenceCorrectionStats()
    raw = np.array([[0.0, -0.1, 0.5, 1.0]], dtype=np.float32)

    out = transform_spectra_to_signal_domain(
        raw, dataset=_dataset(wavelengths=wavelengths), reference_correction=reference_correction,
        signal_domain=_signal_domain(wavelengths=wavelengths, corrected_input_allowed=True,
                                     requires_dark_ref=False, requires_white_ref=False),
        stats=stats,
    )
    np.testing.assert_allclose(out[0], [6.0, 6.0, np.log10(2.0), 0.0], atol=1e-6)
    assert stats.clip_low_count == 2


def test_absorbance_converts_back_to_reflectance():
    wavelengths = np.linspace(1000.0, 1100.0, 4)
    reference_correction = _calibration(mode="already_corrected", output_signal_unit="Absorbance")
    domain = _signal_domain(wavelengths=wavelengths, corrected_input_allowed=True,
                            requires_dark_ref=False, requires_white_ref=False)
    domain.required_signal_unit = "Reflectance"
    raw = np.array([[0.0, 1.0, 2.0, 0.5]], dtype=np.float32)

    out = transform_spectra_to_signal_domain(raw, dataset=_dataset(wavelengths=wavelengths),
                                             reference_correction=reference_correction, signal_domain=domain)
    np.testing.assert_allclose(out[0], 10.0 ** -raw[0], rtol=1e-6)


def test_signal_domain_rejects_a_unit_its_calibration_mode_cannot_produce():
    wavelengths = np.linspace(1000.0, 1100.0, 4)
    with pytest.raises(ValueError, match="cannot produce 'DN'"):
        build.signal_domain(
            wavelengths,
            signal_domain_id="sd",
            input_format="hdf5",
            required_correction_mode="dark_white_reference",
            requires_dark_ref=True,
            requires_white_ref=True,
            corrected_input_allowed=False,
            n_bands=4,
        )


def test_signal_domain_for_training_data_follows_the_contract_writer_rules():
    from dataclasses import replace

    dataset = _dataset(wavelengths=WL, source_format="eiff")
    levels = _calibration(output_signal_unit="Reflectance")  # scalar black/white levels
    domain = SignalDomainContract.for_training_data(dataset=dataset, reference_correction=levels,
                                                    required_signal_unit="Absorbance")
    assert (domain.input_format, domain.source_modality, domain.n_bands) == ("eiff", "nir_hsi", 3)
    assert domain.required_correction_mode == "dark_white_reference"
    assert domain.requires_dark_ref and domain.requires_white_ref
    assert not domain.corrected_input_allowed
    assert (domain.band_mapping_policy, domain.wavelength_tolerance_nm) == ("exact", 0.0)
    np.testing.assert_array_equal(domain.wavelengths, WL)

    already_corrected = _calibration(mode="already_corrected", output_signal_unit="Reflectance")
    domain = SignalDomainContract.for_training_data(dataset=dataset, reference_correction=already_corrected,
                                                    required_signal_unit="Reflectance")
    assert domain.corrected_input_allowed
    assert not (domain.requires_dark_ref or domain.requires_white_ref)

    with pytest.raises(ValueError, match="belongs to dataset"):
        SignalDomainContract.for_training_data(dataset=dataset,
                                               reference_correction=replace(levels, dataset_id="other"),
                                               required_signal_unit="Absorbance")


