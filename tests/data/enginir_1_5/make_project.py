"""Make the enginir-1.5 test project (run once, with engiNIR 1.6.1, before the 1.7 rename).

    uv run python -m tests.data.enginir_1_5.make_project

Builds ``project.h5`` and the data files it references in ``/tmp/enginir-1.5``
and copies them next to this script,
and ``expected_map.npy``: the prediction of ``scan.h5`` by the project's
model. The migration tests of format ``enginir-1.7`` convert this project and
check that the converted model predicts the same map. The files are committed;
this script uses the names of format ``enginir-1.5`` and does not run with
1.7 or later. Nothing here is packaged (the sdist and the wheel hold only
``src/enginir``).

The project holds everything the rename touches:

- a dataset with dark and white references from files (mode
  ``dark_white_reference``), so its ROIs store reference spectra;
- an already corrected float dataset (mode ``precalibrated_import``) with a
  prediction ROI;
- a target, calibration ROIs with values, a recipe, a calibration set, an
  input contract and a fitted PLS model;
- one stored prediction run (written by the low-level service, as a script
  would; the facade and the viewer never store predictions).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from enginir import PreprocessingRecipe, Project
from enginir.io import save_hdf5
from enginir.prediction import predict_with_model_run

HERE = Path(__file__).resolve().parent
#: Built here, then copied next to this script: the paths stored in the project
#: name this neutral folder, not the machine it was made on (data files are found
#: through ``relative_path``, next to the project).
WORK = Path("/tmp/enginir-1.5")
FILES = ("project.h5", "scan.h5", "dark.h5", "white.h5", "corrected.h5", "expected_map.npy")
N_SAMPLES, FRAMES, HEIGHT, N_BANDS = 6, 2, 3, 10
CONTENT = np.linspace(1.0, 6.0, N_SAMPLES)
WAVELENGTHS = np.linspace(1000.0, 1600.0, N_BANDS)


def _cube(offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(15)
    peak = np.exp(-((np.arange(N_BANDS) - N_BANDS / 2) / 2) ** 2)
    content = np.repeat(CONTENT, FRAMES)[:, None, None]
    return 2500 - 150 * content * peak + offset + rng.normal(0, 3, (len(content), HEIGHT, N_BANDS))


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    save_hdf5(WORK / "scan.h5", _cube().astype(np.uint16), WAVELENGTHS)
    save_hdf5(WORK / "dark.h5", np.full((2, HEIGHT, N_BANDS), 100, np.uint16), WAVELENGTHS)
    save_hdf5(WORK / "white.h5", np.full((2, HEIGHT, N_BANDS), 3500, np.uint16), WAVELENGTHS)
    save_hdf5(WORK / "corrected.h5", (_cube(20.0) / 3500).astype(np.float32), WAVELENGTHS)

    project = Project.create(WORK / "project.h5")
    scan = project.add_dataset(WORK / "scan.h5", dark=WORK / "dark.h5", white=WORK / "white.h5")
    project.add_target("API", unit="%")
    for i, value in enumerate(CONTENT):
        scan.add_roi(f"s{i}", frames=(FRAMES * i, FRAMES * i + FRAMES - 1),
                     rows=(0, HEIGHT - 1), values={"API": float(value)})
    model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=2)

    corrected = project.add_dataset(WORK / "corrected.h5", unit="Absorbance")
    corrected.add_roi("p0", frames=(0, FRAMES - 1), rows=(0, HEIGHT - 1),
                      values={"API": float(CONTENT[0])}, job="prediction")

    np.save(WORK / "expected_map.npy", model.predict(WORK / "scan.h5").map)

    # One stored prediction run, as a script stores it with the low-level service.
    ws = project.workspace
    ref = model.record
    recipe_id = ws.sample_sets[ref.sample_set_id].preprocessing_recipe_id
    dataset = scan.record
    predict_with_model_run(
        project_path=project.path, model_run=ref,
        signal_domain=ws.signal_domains[ref.signal_domain_id],
        recipe=PreprocessingRecipe.from_dict(ws.recipes[recipe_id]), dataset=dataset,
        calibration=scan.calibration,
        model_registry={}, data_path=str(WORK / "scan.h5"),
        preprocessing_recipe_id=recipe_id, prediction_run_id="stored-prediction",
        persist_artifacts=True)

    for name in FILES:
        shutil.copy2(WORK / name, HERE / name)
    shutil.rmtree(HERE / "__pycache__", ignore_errors=True)
    print("written:", ", ".join(FILES))


if __name__ == "__main__":
    main()
