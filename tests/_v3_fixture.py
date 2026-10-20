"""Helpers for building one deterministic end-to-end v3 project fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from enginir.io import load_spectra
from enginir.io.project_store import load_project_store
from enginir.preprocessing import PreprocessingRecipe
from enginir.project import (
    CalibrationSpec,
    DatasetRef,
    SignalDomainContract,
    TargetAssignment,
    TargetDefinition,
    TrainingROI,
)
from enginir.viewer._state import ProjectState

REPO_ROOT = Path(__file__).resolve().parents[1]
KEILE_EIFF_PATH = REPO_ROOT / "data" / "Keile_5%ohne_Folie.eiff"
KEILE_FRAME_SLICE = slice(0, 32)
KEILE_TARGET_ID = "target_random_y"
KEILE_DATASET_ID = "dataset_keile_subset"
KEILE_CALIBRATION_ID = "calibration_keile_subset"
KEILE_RECIPE_ID = "recipe_keile_subset"
KEILE_SIGNAL_DOMAIN_ID = "signal_domain_keile_subset"
KEILE_SAMPLE_SET_ID = "sample_set_keile_subset"
KEILE_MODEL_RUN_ID = "model_run_keile_subset"
KEILE_PREDICTION_RUN_ID = "prediction_run_keile_subset"


def build_keile_v3_project(project_path: Path) -> dict[str, object]:
    """Create one deterministic v3 project fixture from the Keile EIFF cube."""
    data_path = str(KEILE_EIFF_PATH.resolve())
    spectral_data = load_spectra(data_path, frames=KEILE_FRAME_SLICE)
    cube = np.asarray(spectral_data.X, dtype=np.uint16)
    wavelengths = np.asarray(spectral_data.wavelengths, dtype=np.float32)
    meta = dict(spectral_data.meta)

    state = ProjectState()
    state.new_project(str(project_path))
    state.register_dataset(
        DatasetRef(
            dataset_id=KEILE_DATASET_ID,
            source_path=data_path,
            source_format="eiff",
            source_modality="nir_hsi",
            cube_shape=tuple(int(v) for v in cube.shape),
            n_bands=int(wavelengths.shape[0]),
            wavelengths=wavelengths,
            loader_metadata={
                **meta,
                "frame_slice": {
                    "start": KEILE_FRAME_SLICE.start,
                    "stop": KEILE_FRAME_SLICE.stop,
                    "step": KEILE_FRAME_SLICE.step,
                },
            },
            import_timestamp="2026-04-02T12:00:00+00:00",
        )
    )
    state.set_active_calibration(
        CalibrationSpec(
            calibration_id=KEILE_CALIBRATION_ID,
            dataset_id=KEILE_DATASET_ID,
            mode="dark_white_reference",
            dark_ref_source=None,
            white_ref_source=None,
            dark_ref_data=None,
            white_ref_data=None,
            black_level=128.0,
            white_level=3500.0,
            output_signal_unit="Reflectance",
            created_at="2026-04-02T12:01:00+00:00",
        )
    )
    state.define_target(
        TargetDefinition(
            target_id=KEILE_TARGET_ID,
            name="random_y",
            unit="a.u.",
            value_type="float",
            missing_value_policy="allowed",
            created_at="2026-04-02T12:02:00+00:00",
        )
    )
    recipe = PreprocessingRecipe(
        filter="sg_smooth",
        sg_window=11,
        sg_polyorder=2,
        snv=True,
        msc=False,
        emsc=False,
        detrend=False,
        detrend_order=1,
    )
    state.register_recipe(KEILE_RECIPE_ID, recipe.to_dict())

    rng = np.random.default_rng(20260402)
    roi_ids: list[str] = []
    for idx in range(7):
        frame_start = int(rng.integers(0, cube.shape[0] - 4))
        frame_stop = frame_start + int(rng.integers(2, 5))
        row_start = int(rng.integers(0, cube.shape[1] - 20))
        row_stop = row_start + int(rng.integers(12, 21))
        spectra = (
            cube[frame_start:frame_stop, row_start:row_stop, :]
            .reshape(-1, cube.shape[2])
            .astype(np.uint16)
        )
        roi_id = f"roi_keile_{idx + 1}"
        roi_ids.append(roi_id)
        state.add_roi(
            TrainingROI(
                roi_id=roi_id,
                dataset_id=KEILE_DATASET_ID,
                calibration_id=KEILE_CALIBRATION_ID,
                name=f"keile_roi_{idx + 1}",
                wavelengths=wavelengths,
                n_pixels=int(spectra.shape[0]),
                spatial_bounds=(frame_start, frame_stop - 1, row_start, row_stop - 1),
                source_file=data_path,
                created_at=f"2026-04-02T12:{10 + idx:02d}:00+00:00",
                provenance={
                    "dataset_id": KEILE_DATASET_ID,
                    "calibration_id": KEILE_CALIBRATION_ID,
                    "signal_unit": "DN",
                    "preview_recipe_summary": "none",
                    "created_at": f"2026-04-02T12:{10 + idx:02d}:00+00:00",
                    **recipe.to_dict(),
                },
            ),
            spectra,
        )
        state.assign_target_to_roi(
            TargetAssignment(
                roi_id=roi_id,
                target_id=KEILE_TARGET_ID,
                value=float(rng.uniform(0.5, 9.5)),
                assigned_at=f"2026-04-02T12:{20 + idx:02d}:00+00:00",
            )
        )

    sample_set_data = state.build_sample_set(
        target_id=KEILE_TARGET_ID,
        preprocessing_recipe_id=KEILE_RECIPE_ID,
        roi_ids=roi_ids,
        sample_set_id=KEILE_SAMPLE_SET_ID,
    )
    state.register_signal_domain(
        SignalDomainContract(
            signal_domain_id=KEILE_SIGNAL_DOMAIN_ID,
            input_format="eiff",
            source_modality="nir_hsi",
            required_signal_unit="Reflectance",
            required_calibration_mode="dark_white_reference",
            requires_dark_ref=True,
            requires_white_ref=True,
            calibrated_input_allowed=False,
            n_bands=int(wavelengths.shape[0]),
            wavelengths=wavelengths,
            wavelength_tolerance_nm=0.0,
            band_mapping_policy="exact",
            output_dtype="float32",
            created_at="2026-04-02T12:40:00+00:00",
        )
    )
    state.fit_model_run(
        sample_set_id=sample_set_data.sample_set.sample_set_id,
        signal_domain_id=KEILE_SIGNAL_DOMAIN_ID,
        model_type="PLS",
        hyperparameters={"n_components": 3},
        cv="logo",
        model_run_id=KEILE_MODEL_RUN_ID,
        created_at="2026-04-02T12:41:00+00:00",
    )

    reopened = ProjectState()
    reopened.open_project(str(project_path))
    prediction_result = reopened.predict_dataset(
        model_run_id=KEILE_MODEL_RUN_ID,
        data_path=data_path,
        loader_kwargs={"frames": KEILE_FRAME_SLICE},
        prediction_run_id=KEILE_PREDICTION_RUN_ID,
        created_at="2026-04-02T12:42:00+00:00",
    )
    store = load_project_store(str(project_path))
    return {
        "project_path": str(project_path),
        "data_path": data_path,
        "cube_shape": tuple(int(v) for v in cube.shape),
        "wavelengths": wavelengths,
        "roi_ids": roi_ids,
        "sample_set_id": sample_set_data.sample_set.sample_set_id,
        "model_run_id": KEILE_MODEL_RUN_ID,
        "prediction_run_id": prediction_result.prediction_run.prediction_run_id,
        "store": store,
        "prediction_shape": tuple(
            int(v) for v in prediction_result.prediction_run.artifact_refs["spatial_map_shape"]
        ),
    }
