"""Synthetic scans and small projects shared by the tests.

Each builder writes an HDF5 scan whose spectra carry a known content, so a fit
on it has a known answer; none of them needs a data file outside the test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from enginir import Project
from enginir.io import load_project_store, save_hdf5

#: The labelled scan: N_SAMPLES samples of FRAMES_PER_SAMPLE frames x HEIGHT rows.
N_SAMPLES, FRAMES_PER_SAMPLE, HEIGHT, N_BANDS = 6, 3, 4, 12
CONTENT = np.linspace(1, 6, N_SAMPLES)

#: The tray: ten ROIs of TRAY_FRAMES frames between dark and white frames.
TRAY_FRAMES, TRAY_HEIGHT, TRAY_BANDS = 4, 5, 24

#: A fit specification that excludes pixels above the DModX limit.
EXCLUSION = {"exclusion": {"method": "dmodx", "level": 0.99}}


def scan_file(path, *, offset=0.0, wavelengths=None, dtype=np.uint16) -> str:
    """A scan of N_SAMPLES samples whose absorption grows with CONTENT."""
    rng = np.random.default_rng(1)
    peak = np.exp(-((np.arange(N_BANDS) - N_BANDS / 2) / 2) ** 2)
    content = np.repeat(CONTENT, FRAMES_PER_SAMPLE)[:, None, None]
    cube = 2500 - 150 * content * peak + offset + rng.normal(0, 3, (len(content), HEIGHT, N_BANDS))
    if np.dtype(dtype).kind == "f":
        cube = cube / 3500
    if wavelengths is None:
        wavelengths = np.linspace(1000, 1600, N_BANDS)
    save_hdf5(path, cube.astype(dtype), wavelengths)
    return str(path)


def labelled_project(tmp_path, **dataset_kwargs):
    """A project of :func:`scan` with target ``API`` and one ROI per sample."""
    project = Project.create(tmp_path / "project.h5")
    dataset = project.add_dataset(scan_file(tmp_path / "scan.h5"), **dataset_kwargs)
    project.add_target("API", unit="%")
    for i, value in enumerate(CONTENT):
        first = FRAMES_PER_SAMPLE * i
        dataset.add_roi(f"s{i}", frames=(first, first + FRAMES_PER_SAMPLE - 1),
                        rows=(0, HEIGHT - 1), values={"API": value})
    return project, dataset


def stored_counts(project) -> dict[str, int]:
    """How many objects of each kind the project file holds."""
    stored = load_project_store(project.path)
    return {key: len(value) for key, value in stored.items() if key != "project_meta"}


def warnings_during(action: Callable[[], Any]) -> tuple[Any, list[str]]:
    """Run ``action``; return its result and the warnings it logged."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        result = action()
    finally:
        logger.remove(sink_id)
    return result, messages


def uneven_project(tmp_path, n_rois=9, n_bands=12):
    """ROIs of 2..4 frames x 3 rows, so GroupKFold and round-robin folds differ."""
    rng = np.random.default_rng(3)
    content = np.linspace(1.0, 9.0, n_rois)
    sizes = [2 + (i % 3) for i in range(n_rois)]
    peak = np.exp(-((np.arange(n_bands) - 6) / 2.5) ** 2)
    frames = []
    for value, size in zip(content, sizes):
        frames += [2600 - 120 * value * peak + rng.normal(0, 25, (size, 3, n_bands))]
    cube = np.vstack([f[None] if f.ndim == 2 else f for f in frames]).astype(np.uint16)
    path = tmp_path / "scan.h5"
    save_hdf5(path, cube, np.linspace(1000, 1600, n_bands))
    project = Project.create(tmp_path / "uneven.h5")
    dataset = project.add_dataset(path)
    project.add_target("API", unit="%")
    start = 0
    for i, (value, size) in enumerate(zip(content, sizes)):
        dataset.add_roi(f"s{i}", frames=(start, start + size - 1), rows=(0, 2),
                        values={"API": float(value)})
        start += size
    return project


def project_with_an_odd_pixel(tmp_path, n_bands=30):
    """8 ROIs of 4 frames x 3 rows; pixel 5 of ROI s3 carries a band the others lack."""
    rng = np.random.default_rng(4)
    content = np.linspace(1.0, 8.0, 8)
    bands = np.arange(n_bands)
    peak = np.exp(-((bands - 12) / 3.0) ** 2)
    cube = np.vstack([2600 - 150 * c * peak + rng.normal(0, 8, (4, 3, n_bands))
                      for c in content])
    odd = np.exp(-((bands - 24) / 1.5) ** 2)
    cube[3 * 4 + 1, 2] += 400 * odd  # ROI s3: frame 1, row 2 -> pixel 1 * 3 + 2 = 5
    path = tmp_path / "scan.h5"
    save_hdf5(path, cube.astype(np.uint16), np.linspace(1000, 1600, n_bands))
    project = Project.create(tmp_path / "odd.h5")
    dataset = project.add_dataset(path)
    project.add_target("API", unit="%")
    for i, value in enumerate(content):
        dataset.add_roi(f"s{i}", frames=(4 * i, 4 * i + 3), rows=(0, 2),
                        values={"API": float(value)})
    return project


def tray_project(tmp_path, odd: str | None = None):
    """One tray of 10 ROIs (levels 1..10, references from frames); ROIs 8-10 for prediction.

    ``odd`` (an ROI name, e.g. ``"t10"``) gives that ROI a foreign absorption band
    the calibration ROIs lack, so its spectra lie outside the model.
    Returns ``(project, dataset, path)``.
    """
    rng = np.random.default_rng(11)
    peak = np.exp(-((np.arange(TRAY_BANDS) - 9) / 3.0) ** 2)
    foreign = np.exp(-((np.arange(TRAY_BANDS) - 19) / 1.5) ** 2)
    blocks = [np.full((2, TRAY_HEIGHT, TRAY_BANDS), 120.0)]  # dark frames
    values = np.arange(1.0, 11.0)
    for i, value in enumerate(values):
        block = (2600 - 110 * value * peak
                 + rng.normal(0, 20, (TRAY_FRAMES, TRAY_HEIGHT, TRAY_BANDS)))
        if odd == f"t{i + 1:02d}":
            block -= 600 * foreign
        blocks.append(block)
    blocks.append(np.full((2, TRAY_HEIGHT, TRAY_BANDS), 3400.0))  # white frames
    cube = np.vstack(blocks).astype(np.uint16)
    path = tmp_path / "tray.h5"
    save_hdf5(path, cube, np.linspace(1000, 1600, TRAY_BANDS))
    project = Project.create(tmp_path / "project.h5")
    n = cube.shape[0]
    dataset = project.add_dataset(path, dark_frames=(0, 1), white_frames=(n - 2, n - 1))
    project.add_target("API", unit="%")
    for i, value in enumerate(values):
        first = 2 + i * TRAY_FRAMES
        dataset.add_roi(f"t{i + 1:02d}", frames=(first, first + TRAY_FRAMES - 1),
                        rows=(0, TRAY_HEIGHT - 1), values={"API": float(value)},
                        job="prediction" if i >= 7 else "calibration")
    return project, dataset, path


def predicts_its_training_scan(model, dataset):
    """Prediction of the training scan replays the training preprocessing exactly."""
    prediction = model.predict(dataset.path)
    np.testing.assert_allclose(prediction.t2_map.reshape(-1), model.diagnostics["t2"],
                               rtol=1e-3)
    return prediction


def framed_scan(path, *, wavelengths=True, white=3500, bright=0):
    """20 frames x 6 rows x 10 bands: dark frames 0-1, white frames 18-19, counts between.

    ``bright`` makes frames 5-6 brighter than the white reference by that much;
    ``wavelengths=False`` stores band numbers instead of wavelengths.
    """
    rng = np.random.default_rng(3)
    cube = rng.integers(1500, 2500, (20, 6, 10)).astype(np.uint16)
    cube[:2], cube[-2:] = 100, white
    if bright:
        cube[5:7, :, :] = white + bright  # brighter than the white reference
    axis = np.linspace(1000, 1600, 10) if wavelengths else np.arange(10, dtype=float)
    save_hdf5(str(path), cube, axis)
    return path


#: The first real model of the plan (20 ROIs, levels 1/4/7/10): one SE chose 7 components.
EVIDENCE = {
    "rmsecv": [3.01, 2.84, 2.46, 2.01, 1.88, 1.66, 1.585, 1.49, 1.485, 1.48, 1.48, 1.47,
               1.46, 1.46, 1.459],
    "rmsecv_se": [0.27] * 14 + [0.164],
    "q2": [0.193, -0.015, 0.149, 0.23, -0.021, 0.065, -0.062, -0.031, -0.201, -0.181,
           -0.186, -0.185, -0.174, -0.194, -0.188],
    "r2x_cum": np.cumsum([0.713, 0.147, 0.048, 0.029, 0.044, 0.004, 0.003, 0.001, 0.002,
                          0.001, 0.001, 0.0, 0.001, 0.0, 0.0]),
}


def every_stored_object_project(tmp_path):
    """A project with every kind of stored object, written by the current writers.

    References from files (so ROIs store reference spectra), an already corrected
    dataset with a prediction ROI, a fitted model and one stored prediction run
    (written by the low-level service, as a script stores one). Returns the path.
    """
    from enginir import PreprocessingRecipe
    from enginir.prediction import predict_with_model_run

    scan = scan_file(tmp_path / "scan.h5")
    save_hdf5(tmp_path / "dark.h5", np.full((2, HEIGHT, N_BANDS), 100, np.uint16),
              np.linspace(1000, 1600, N_BANDS))
    save_hdf5(tmp_path / "white.h5", np.full((2, HEIGHT, N_BANDS), 3500, np.uint16),
              np.linspace(1000, 1600, N_BANDS))
    project = Project.create(tmp_path / "every.h5")
    dataset = project.add_dataset(scan, dark=tmp_path / "dark.h5", white=tmp_path / "white.h5")
    project.add_target("API", unit="%")
    for i, value in enumerate(CONTENT):
        first = FRAMES_PER_SAMPLE * i
        dataset.add_roi(f"s{i}", frames=(first, first + FRAMES_PER_SAMPLE - 1),
                        rows=(0, HEIGHT - 1), values={"API": value})
    model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=2)
    corrected = project.add_dataset(scan_file(tmp_path / "corrected.h5", dtype=np.float32),
                                    unit="Absorbance")
    corrected.add_roi("p0", frames=(0, FRAMES_PER_SAMPLE - 1), rows=(0, HEIGHT - 1),
                      values={"API": float(CONTENT[0])}, job="prediction")
    ws = project.workspace
    recipe_id = ws.sample_sets[model.record.sample_set_id].preprocessing_recipe_id
    predict_with_model_run(
        project_path=project.path, model_run=model.record,
        signal_domain=ws.signal_domains[model.record.signal_domain_id],
        recipe=PreprocessingRecipe.from_dict(ws.recipes[recipe_id]), dataset=dataset.record,
        reference_correction=dataset.reference_correction, model_registry={},
        data_path=scan, preprocessing_recipe_id=recipe_id,
        prediction_run_id="stored-prediction", persist_artifacts=True)
    return project.path
