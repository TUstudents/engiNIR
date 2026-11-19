"""The example project: synthetic tablet scans with a finished model (plan gui-1.6, Q1).

Generated on first use, so nothing is downloaded and the package stays
small: five push-broom scans of five tablets each. Each of the four
calibration scans holds tablets with different numbers of printed layers
(every number from 1 to 10 twice in all), so a scan and a value are not the
same thing; the fifth scan holds tablets to predict. Each scan carries its
own dark and white reference frames. The project has the target "Layers",
one sample label per tablet, and a PLS model held out by sample (two
components: layers and a varying binder). Its only open check is in the
Predict step: the prediction tablet P_T4 lies outside the model
(``predict.rois_outside``). The spectra are made up; the numbers mean nothing
beyond the demo.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ._logging import get_logger

logger = get_logger(__name__)

EXAMPLE_NAME = "engiNIR example.h5"
#: The scan to predict; its name also tells this example from the one of 1.6.0rc3
#: (one layer count per scan), which is created again.
_PREDICT_FILE = "tablets_to_predict.h5"
#: Layers of the tablets per scan; the last scan is the one to predict.
_SCANS = ((1, 4, 7, 10, 2), (3, 6, 9, 5, 8), (10, 1, 5, 8, 3), (2, 7, 4, 9, 6),
          (2, 5, 5, 8, 9))
_N_BANDS, _HEIGHT, _TABLET_FRAMES, _GAP = 64, 36, 10, 4
_TABLETS = 5


def _scan(path: Path, layers: tuple[int, ...], seed: int) -> None:
    """One scan: dark frames, five tablets on a bright tray, white frames."""
    from .io import save_hdf5

    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(1000.0, 1700.0, _N_BANDS)
    bands = np.arange(_N_BANDS)
    api = np.exp(-((bands - 22) / 5.0) ** 2) + 0.7 * np.exp(-((bands - 44) / 7.0) ** 2)
    binder = 0.4 * np.exp(-((bands - 32) / 9.0) ** 2)
    n_frames = 2 + _GAP + _TABLETS * (_TABLET_FRAMES + _GAP) + 2
    dark, white = 120.0, 3400.0
    reflectance = np.full((n_frames, _HEIGHT, _N_BANDS), 0.85)
    for tablet in range(_TABLETS):
        first = 2 + _GAP + tablet * (_TABLET_FRAMES + _GAP)
        rows = slice(8, _HEIGHT - 8)
        thickness = layers[tablet] * (1.0 + 0.04 * rng.standard_normal())
        binder_share = 0.15 * (1.0 + 0.6 * rng.uniform(-1.0, 1.0))  # an interferent
        absorbance = 0.08 + 0.025 * thickness * api + binder_share * binder \
            + 0.006 * rng.standard_normal(_N_BANDS)
        block = 10 ** -(absorbance[None, None, :]
                        + 0.004 * rng.standard_normal((_TABLET_FRAMES, _HEIGHT - 16, _N_BANDS)))
        reflectance[first:first + _TABLET_FRAMES, rows] = block * 0.9
    counts = dark + (white - dark) * reflectance \
        + rng.normal(0, 6, reflectance.shape)
    counts[:2] = dark + rng.normal(0, 3, (2, _HEIGHT, _N_BANDS))
    counts[-2:] = white + rng.normal(0, 8, (2, _HEIGHT, _N_BANDS))
    save_hdf5(str(path), np.clip(counts, 0, 65535).astype(np.uint16), wavelengths)


def create_example_project(folder: str | os.PathLike[str], *, fit: bool = True) -> Path:
    """Create the example project in ``folder`` (if it is not there yet); return its path.

    An existing example project that opens is reused; one that does not is
    created again.
    """
    from . import Project
    from .preprocessing import PreprocessingRecipe

    folder = Path(folder)
    path = folder / EXAMPLE_NAME
    if path.is_file():
        try:
            Project.open(path)
            if (folder / _PREDICT_FILE).is_file():
                return path
            logger.info("Creating the example project again; {} is an older example", path)
        except (OSError, ValueError) as exc:
            logger.warning("Creating the example project again; {} does not open: {}",
                           path, exc)
        path.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    project = Project.create(path)
    project.add_target("Layers", unit="1")
    tablet_rows = (8, _HEIGHT - 9)
    for seed, layers in enumerate(_SCANS):
        predict = seed == len(_SCANS) - 1
        scan_path = folder / (_PREDICT_FILE if predict else f"tablets_{seed + 1}.h5")
        _scan(scan_path, layers, seed)
        n_frames = 2 + _GAP + _TABLETS * (_TABLET_FRAMES + _GAP) + 2
        dataset = project.add_dataset(scan_path, dark_frames=(0, 1),
                                      white_frames=(n_frames - 2, n_frames - 1),
                                      unit="Absorbance")
        for tablet in range(_TABLETS):
            first = 2 + _GAP + tablet * (_TABLET_FRAMES + _GAP) + 2
            name = f"{'P' if predict else f'S{seed + 1}'}_T{tablet + 1}"
            roi = dataset.add_roi(
                name, frames=(first, first + _TABLET_FRAMES - 5), rows=tablet_rows,
                values=None if predict else {"Layers": float(layers[tablet])},
                job="prediction" if predict else "calibration")
            project.set_sample(roi, name)
    if fit:
        # No SNV: these synthetic spectra have no scatter to remove, and SNV would
        # bend their linear relation to the layers (thin tablets far from the model).
        project.fit("Layers", recipe=PreprocessingRecipe(), n_components=2,
                    name="PLS 2 LV",
                    fit_spec={"folds": {"group_by": "sample"}})
    logger.info("Created the example project {}", path)
    return path
