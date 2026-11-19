"""Builders for complete current-schema project objects and payloads used by the tests.

The typed builders return valid objects with test defaults; keyword arguments
replace fields. Payload builders return the dicts the low-level writers take.
"""

from __future__ import annotations

from typing import Any

import numpy as np

T0 = "2026-04-02T10:00:00+00:00"


def roi_provenance(
    *,
    dataset_id: str = "dataset_1",
    reference_correction_id: str = "cal_1",
    signal_unit: str = "DN",
    created_at: str = "2026-04-02T10:05:00+00:00",
) -> dict[str, str]:
    """Return the required ROI provenance fields."""
    return {
        "dataset_id": dataset_id,
        "reference_correction_id": reference_correction_id,
        "signal_unit": signal_unit,
        "preview_recipe_summary": "none",
        "created_at": created_at,
    }


def recipe_payload(**overrides: object) -> dict[str, object]:
    """Return all required fields for one persisted preprocessing recipe."""
    payload: dict[str, object] = {
        "filter": "none",
        "sg_window": 11,
        "sg_polyorder": 2,
        "snv": False,
        "msc": False,
        "emsc": False,
        "detrend": False,
        "detrend_order": 1,
        "sg_edges": "drop",
        "asls": False,
        "asls_lambda": 10000.0,
        "asls_p": 0.001,
        "asls_iterations": 10,
        "bands_keep": [],
        "bands_exclude": [],
    }
    payload.update(overrides)
    return payload


def projection_payload(n_bands: int, n_components: int = 1) -> dict[str, Any]:
    """The projection data every stored PLS/PCR model has (core of the format)."""
    components = np.zeros((n_bands, n_components))
    components[0] = 1.0
    return {"x_mean": np.zeros(n_bands), "x_std": np.ones(n_bands),
            "rotations": components, "loadings": components,
            "score_variances": np.ones(n_components), "s0": 1.0, "n_train": 10,
            "t2_crit_095": 10.0, "dcrit_095": 2.0}


def prediction_model_payload(wavelengths: Any, target: str = "moisture") -> dict[str, Any]:
    """A complete executable ``prediction_model`` artifact of one-component PLS."""
    wavelengths = np.asarray(wavelengths, dtype=np.float32)
    return {"model_type": "PLSRegression", "n_components": 1, "wavelengths": wavelengths,
            "target_names": [target], "coef_": np.ones(wavelengths.size, dtype=np.float32),
            "intercept_": np.zeros(1, dtype=np.float32),
            **projection_payload(wavelengths.size)}


# -- typed objects ---------------------------------------------------------------


def dataset(wavelengths: Any, dataset_id: str = "dataset_1", **fields: Any):
    from enginir.project import DatasetRef

    wavelengths = np.asarray(wavelengths, dtype=np.float32)
    values: dict[str, Any] = dict(
        dataset_id=dataset_id, source_path=f"/tmp/{dataset_id}.eiff", source_format="eiff",
        source_modality="nir_hsi", cube_shape=(20, 15, wavelengths.size),
        n_bands=wavelengths.size, wavelengths=wavelengths, loader_metadata={},
        import_timestamp=T0)
    values.update(fields)
    return DatasetRef(**values)


def reference_correction(dataset_id: str = "dataset_1", reference_correction_id: str = "cal_1", **fields: Any):
    """A reference correction; raw counts unless ``fields`` say otherwise."""
    from enginir.project import ReferenceCorrection

    values: dict[str, Any] = dict(
        reference_correction_id=reference_correction_id, dataset_id=dataset_id, mode="raw_dn",
        dark_ref_source=None, white_ref_source=None, dark_ref_data=None, white_ref_data=None,
        black_level=None, white_level=None, output_signal_unit="DN", created_at=T0)
    values.update(fields)
    return ReferenceCorrection(**values)


def target(target_id: str = "target_1", name: str = "moisture", **fields: Any):
    from enginir.project import TargetDefinition

    values: dict[str, Any] = dict(target_id=target_id, name=name, unit="% w/w",
                                  value_type="float", missing_value_policy="allowed",
                                  created_at=T0)
    values.update(fields)
    return TargetDefinition(**values)


def assignment(roi_id: str, target_id: str = "target_1", value: float = 1.0):
    from enginir.project import TargetAssignment

    return TargetAssignment(roi_id=roi_id, target_id=target_id, value=value, assigned_at=T0)


def signal_domain(wavelengths: Any, signal_domain_id: str = "signal_1", **fields: Any):
    """The input contract of raw counts (DN) of an EIFF file."""
    from enginir.project import SignalDomainContract

    wavelengths = np.asarray(wavelengths, dtype=np.float32)
    values: dict[str, Any] = dict(
        signal_domain_id=signal_domain_id, input_format="eiff", source_modality="nir_hsi",
        required_signal_unit="DN", required_correction_mode="raw_dn",
        requires_dark_ref=False, requires_white_ref=False, corrected_input_allowed=True,
        n_bands=wavelengths.size, wavelengths=wavelengths, wavelength_tolerance_nm=0.0,
        band_mapping_policy="exact", output_dtype="float32", created_at=T0)
    values.update(fields)
    return SignalDomainContract(**values)


def roi(wavelengths: Any, roi_id: str = "roi_1", dataset_id: str = "dataset_1",
        reference_correction_id: str = "cal_1", *, n_pixels: int = 4,
        spatial_bounds: tuple[int, int, int, int] = (0, 1, 0, 1), **fields: Any):
    from enginir.viewer._state import TrainingROI

    values: dict[str, Any] = dict(
        roi_id=roi_id, dataset_id=dataset_id, reference_correction_id=reference_correction_id, name=roi_id,
        wavelengths=np.asarray(wavelengths, dtype=np.float32), n_pixels=n_pixels,
        spatial_bounds=spatial_bounds, source_file=f"/tmp/{dataset_id}.eiff", created_at=T0,
        provenance=roi_provenance(dataset_id=dataset_id, reference_correction_id=reference_correction_id))
    values.update(fields)
    return TrainingROI(**values)


def sample_set(roi_ids: list[str], sample_set_id: str = "sample_set_1", *,
               target_id: str = "target_1", recipe_id: str = "recipe_1",
               dataset_ids: list[str] | None = None):
    from enginir.project import SampleSet

    return SampleSet(
        sample_set_id=sample_set_id, roi_ids=list(roi_ids), target_id=target_id,
        observation_unit="pixel", target_level="roi", grouping_unit="roi",
        validation_unit="roi", weighting_policy="per_pixel_unweighted", reporting_level="roi",
        roi_aggregation="mean", preprocessing_recipe_id=recipe_id,
        dataset_ids=list(dataset_ids or ["dataset_1"]), created_at=T0)


def model_run(model_run_id: str = "model_1", sample_set_id: str = "sample_set_1",
              signal_domain_id: str = "signal_1", **fields: Any):
    from enginir.project import FitSpec, ModelRun
    from enginir.project._models import software_versions

    values: dict[str, Any] = dict(software=software_versions(), 
        model_run_id=model_run_id, sample_set_id=sample_set_id,
        signal_domain_id=signal_domain_id, model_type="PLS", hyperparameters={"n_components": 1},
        cv_spec={"type": "GroupKFold", "n_splits": 2}, metrics={"rmsecv": 1.0},
        artifact_refs={}, created_at=T0, fit_spec=FitSpec(groups=(0,)).to_dict())
    values.update(fields)
    return ModelRun(**values)


# -- a model project written with the low-level writers ---------------------------

def write_model_project(path: str) -> None:
    """Datasets, references, ROIs and a fitted PLS model of three bands (low-level writers)."""
    from enginir.io.project_store import (
        create_project,
        save_reference_correction,
        save_dataset_ref,
        save_preprocessing_recipe,
        save_signal_domain_contract,
        save_target_assignment,
        save_target_definition,
        save_training_roi,
    )
    wavelengths = np.array([1000.0, 1100.0, 1200.0], dtype=np.float32)
    create_project(path)
    save_dataset_ref(
        path,
        "dataset_train",
        {
            "source_path": "/data/train.h5",
            "source_format": "hdf5",
            "source_modality": "nir_hsi",
            "cube_shape": (4, 4, 3),
            "n_bands": 3,
            "wavelengths": wavelengths,
            "loader_metadata": {"instrument": "alpha"},
            "import_timestamp": "2026-04-02T10:00:00+00:00",
        },
    )
    save_reference_correction(
        path,
        "dataset_train",
        "cal_train",
        {
            "mode": "dark_white_reference",
            "output_signal_unit": "Absorbance",
            "black_level": 0.0,
            "white_level": 100.0,
            "created_at": "2026-04-02T10:01:00+00:00",
        },
    )
    save_target_definition(
        path,
        "target_moisture",
        {
            "name": "moisture",
            "unit": "% w/w",
            "value_type": "float",
            "missing_value_policy": "allowed",
            "created_at": "2026-04-02T10:02:00+00:00",
        },
    )
    save_preprocessing_recipe(
        path,
        "recipe_none",
        recipe_payload(sg_edges="interp"),
    )
    save_preprocessing_recipe(
        path,
        "recipe_msc",
        recipe_payload(msc=True, sg_edges="interp"),
    )
    save_signal_domain_contract(
        path,
        "signal_domain_1",
        {
            "input_format": "hdf5",
            "source_modality": "nir_hsi",
            "required_signal_unit": "Absorbance",
            "required_correction_mode": "dark_white_reference",
            "requires_dark_ref": True,
            "requires_white_ref": True,
            "corrected_input_allowed": False,
            "n_bands": 3,
            "wavelengths": wavelengths,
            "wavelength_tolerance_nm": 0.0,
            "band_mapping_policy": "exact",
            "output_dtype": "float32",
            "created_at": "2026-04-02T10:04:00+00:00",
        },
    )

    roi_payloads = {
        "roi_1": (np.array([[10, 20, 30], [12, 22, 32]], dtype=np.uint16), 1.0),
        "roi_2": (np.array([[14, 24, 34], [16, 26, 36]], dtype=np.uint16), 2.0),
        "roi_3": (np.array([[18, 28, 38], [20, 30, 40]], dtype=np.uint16), 3.0),
        "roi_4": (np.array([[22, 32, 42], [24, 34, 44]], dtype=np.uint16), 4.0),
    }
    for index, (roi_id, (spectra, value)) in enumerate(roi_payloads.items(), start=1):
        save_training_roi(
            path,
            roi_id,
            {
                "dataset_id": "dataset_train",
                "reference_correction_id": "cal_train",
                "name": roi_id.upper(),
                "source_file": "/data/train.h5",
                "job": "calibration",
                "n_pixels": int(spectra.shape[0]),
                "created_at": f"2026-04-02T10:0{4 + index}:00+00:00",
                "spatial_bounds": (index, index, 0, 1),
                "wavelengths": wavelengths,
                "provenance": roi_provenance(
                    dataset_id="dataset_train",
                    reference_correction_id="cal_train",
                    created_at=f"2026-04-02T10:0{4 + index}:00+00:00"
                ),
            },
            spectra,
        )
        save_target_assignment(
            path,
            "target_moisture",
            roi_id,
            {"value": value, "assigned_at": f"2026-04-02T10:1{index}:00+00:00"},
        )


def write_prediction_cube(path: str, *, wavelengths: np.ndarray) -> None:
    """A 2 x 2 float32 scan of three bands to predict with :func:`write_model_project`."""
    from enginir.io import save_hdf5

    cube = np.array(
        [
            [[11.0, 21.0, 31.0], [13.0, 23.0, 33.0]],
            [[15.0, 25.0, 35.0], [17.0, 27.0, 37.0]],
        ],
        dtype=np.float32,
    )
    save_hdf5(
        path,
        cube,
        wavelengths,
        meta={"source_modality": "nir_hsi", "instrument": "alpha"},
    )


__all__ = [
    "T0", "assignment", "dataset", "model_run", "prediction_model_payload",
    "projection_payload", "recipe_payload", "reference_correction", "roi",
    "roi_provenance", "sample_set", "signal_domain", "target", "write_model_project", "write_prediction_cube",
]
