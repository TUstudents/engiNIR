"""The SIMCA reference project: synthetic blends whose models are also built in SIMCA 18.

Plan ``plans/fit-specification.md``, Phase R. :func:`build` writes a
push-broom cube of raw counts with dark and white references, an engiNIR
project with one ROI per blend (fixed ROI ids; the pixel subsample depends
only on the seed and each ROI's spectra), and :func:`tables` gives the absorbance tables that are
imported into SIMCA. The committed CSV files in ``tests/data/simca_reference``
were written by ``uv run python -m tests._simca_reference``; the tests check
that they still equal what engiNIR computes.
"""

from __future__ import annotations

import csv
import hashlib
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from enginir import Project
from enginir._workflow import make_roi, read_frames
from enginir.io import save_hdf5
from enginir.modeling._service import _signal_domain_matrix, build_sample_set
from enginir.project import SignalDomainContract, TargetAssignment

DATA_DIR = Path(__file__).parent / "data" / "simca_reference"
TARGET = "conc"
SEED = 2026
PIXELS_PER_ROI = 10
N_TRAINING, N_TEST = 14, 4
FRAMES_PER_ROI, ROWS = 6, 8
WAVELENGTHS = np.linspace(1100.0, 1694.0, 100)
_ID_NAMESPACE = uuid.UUID("6f1d3c52-5d0e-4b8e-9a51-6a1f1c2e7b10")


def _component_spectra() -> np.ndarray:
    """Absorbance of the three pure components, shape (3, n_bands)."""
    def bands(*peaks: tuple[float, float, float]) -> np.ndarray:
        return sum(h * np.exp(-0.5 * ((WAVELENGTHS - c) / w) ** 2) for c, w, h in peaks)

    return np.stack([
        bands((1210, 25, 0.30), (1450, 40, 0.45), (1530, 30, 0.20)),   # the analyte
        bands((1190, 30, 0.25), (1420, 35, 0.35), (1660, 25, 0.30)),
        bands((1300, 50, 0.20), (1500, 45, 0.25), (1600, 35, 0.35)),
    ])


def _compositions(rng: np.random.Generator) -> np.ndarray:
    """Mixture fractions of the 18 blends (rows sum to 1); column 0 is the target."""
    analyte = np.concatenate([np.linspace(0.05, 0.70, N_TRAINING),
                              np.array([0.15, 0.32, 0.48, 0.61])])
    rest = 1.0 - analyte
    share = rng.uniform(0.3, 0.7, analyte.size)
    return np.column_stack([analyte, rest * share, rest * (1 - share)])


def roi_names() -> list[str]:
    """Training blends B01..B14, then test blends T01..T04, in cube order."""
    return ([f"B{i + 1:02d}" for i in range(N_TRAINING)]
            + [f"T{i + 1:02d}" for i in range(N_TEST)])


def roi_id(name: str) -> str:
    """The fixed id of the reference ROI ``name``."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"simca-reference/{name}"))


def make_cube() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw counts ``(frames, rows, bands)`` uint16, dark and white ``(rows, bands)``, targets."""
    rng = np.random.default_rng(SEED)
    pure = _component_spectra()
    fractions = _compositions(rng)
    n_rois, n_bands = fractions.shape[0], WAVELENGTHS.size
    dark = 128.0 + rng.normal(0.0, 2.0, (ROWS, n_bands))
    white = 3500.0 + 150.0 * np.sin(np.linspace(0, np.pi, n_bands)) \
        + rng.normal(0.0, 10.0, (ROWS, n_bands))
    x = (WAVELENGTHS - WAVELENGTHS.mean()) / np.ptp(WAVELENGTHS)
    cube = np.empty((n_rois * FRAMES_PER_ROI, ROWS, n_bands))
    for k in range(n_rois):
        chemistry = fractions[k] @ pure
        baseline = rng.normal(0.05, 0.02) + rng.normal(0.0, 0.03) * x + rng.normal(0.0, 0.05) * x**2
        for f in range(FRAMES_PER_ROI):
            for r in range(ROWS):
                scatter = rng.normal(1.0, 0.04)
                offset = rng.normal(0.0, 0.015)
                absorbance = offset + scatter * (chemistry + baseline)
                reflectance = 10.0 ** -absorbance
                counts = dark[r] + reflectance * (white[r] - dark[r]) + rng.normal(0.0, 4.0, n_bands)
                cube[k * FRAMES_PER_ROI + f, r] = counts
    cube_u16 = np.clip(np.rint(cube), 0, 65535).astype(np.uint16)
    return cube_u16, dark.astype(np.float32), white.astype(np.float32), fractions[:, 0]


@dataclass(frozen=True)
class Reference:
    """The built reference project."""

    project: Project
    names: list[str]
    targets: dict[str, float]
    #: The stored raw spectra of every ROI, by name.
    raw: dict[str, np.ndarray]


def build(directory: Path) -> Reference:
    """Write the cube and the project into ``directory`` and return the project."""
    cube, dark, white, targets = make_cube()
    cube_path = directory / "reference_cube.h5"
    save_hdf5(str(cube_path), cube, WAVELENGTHS)
    project = Project.create(directory / "reference_project.h5")
    target = project.add_target(TARGET, unit="fraction")
    dataset = project.add_dataset(cube_path, dark=dark, white=white)
    names = roi_names()
    rois, assignments, raw = [], [], {}
    for k, name in enumerate(names):
        frames = (k * FRAMES_PER_ROI, (k + 1) * FRAMES_PER_ROI - 1)
        roi, spectra, refs = make_roi(dataset.ref, dataset.calibration, name=name, frames=frames,
                                      rows=(0, ROWS - 1),
                                      pixels=read_frames(str(cube_path), *frames))
        roi = replace(roi, roi_id=roi_id(name))
        rois.append((roi, spectra, refs))
        raw[name] = spectra
        assignments.append(TargetAssignment(roi_id=roi.roi_id, target_id=target.target_id,
                                            value=float(targets[k]), assigned_at=roi.created_at))
    project.save_rois(rois, assignments)
    return Reference(project, names, {name: float(t) for name, t in zip(names, targets)}, raw)


def roi_digest(spectra: np.ndarray) -> bytes:
    """SHA-256 of an ROI's stored spectra (``sample-set-model-run-contract_v3.md`` §7.1, step 1)."""
    stored = np.ascontiguousarray(spectra)
    little = stored.astype(stored.dtype.newbyteorder("<"), copy=False)
    header = f"{stored.dtype.name}:{stored.shape[0]}x{stored.shape[1]}:".encode("ascii")
    return hashlib.sha256(header + little.tobytes(order="C")).digest()


def draw(spectra: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indices of the ``n`` pixels drawn from an ROI (contract §7.1, steps 2 and 3)."""
    digest, prefix = roi_digest(spectra), int(seed).to_bytes(8, "little")
    keys = [hashlib.sha256(prefix + digest + j.to_bytes(4, "little")).digest()
            for j in range(spectra.shape[0])]
    return np.sort(np.array(sorted(range(len(keys)), key=keys.__getitem__)[:n]))


def absorbance(reference: Reference, names: list[str]) -> dict[str, np.ndarray]:
    """Absorbance of every pixel of the ROIs ``names``, converted exactly as the fit does."""
    ws = reference.project.workspace
    target_id = reference.project.target(TARGET).target_id
    data = build_sample_set(project_path=reference.project.path, rois=ws.rois,
                            target_assignments=ws.target_assignments, target_id=target_id,
                            preprocessing_recipe_id="none", roi_ids=[roi_id(n) for n in names])
    domain = SignalDomainContract.for_training_data(
        dataset=next(iter(ws.datasets.values())), calibration=next(iter(ws.calibrations.values())),
        required_signal_unit="Absorbance")
    X = _signal_domain_matrix(reference.project.path, data, ws.datasets, ws.calibrations,
                              ws.rois, domain)
    return {name: X[data.roi_slices[roi_id(name)]] for name in names}


def tables(reference: Reference) -> dict[str, tuple[list[str], list[str], np.ndarray, np.ndarray]]:
    """The SIMCA tables: ``{file name: (observation ids, roi names, conc, absorbance)}``."""
    training, test = reference.names[:N_TRAINING], reference.names[N_TRAINING:]
    spectra = absorbance(reference, reference.names)

    def pixel_table(names: list[str]):
        ids, rois, y, rows = [], [], [], []
        for name in names:
            picked = draw(reference.raw[name], PIXELS_PER_ROI, SEED)
            ids += [f"{name}_p{int(i):02d}" for i in picked]
            rois += [name] * picked.size
            y += [reference.targets[name]] * picked.size
            rows.append(spectra[name][picked])
        return ids, rois, np.array(y), np.vstack(rows)

    pixels = pixel_table(training)
    means = np.vstack([pixels[3][[r == n for r in pixels[1]]].mean(axis=0, dtype=np.float64)
                       for n in training]).astype(np.float32)
    return {
        "pixels.csv": pixels,
        "roi_means.csv": (list(training), list(training),
                          np.array([reference.targets[n] for n in training]), means),
        "pixels_test.csv": pixel_table(test),
    }


def band_names() -> list[str]:
    """Column names of the bands: the wavelength in nm as a number, e.g. ``1100``.

    SIMCA's spectroscopic projects use the variable names as the spectral
    axis, so they must be numeric.
    """
    return [f"{w:g}" for w in WAVELENGTHS]


def write_tables(reference: Reference, directory: Path) -> None:
    """Write the SIMCA tables as CSV: comma separated, decimal point, 9 significant digits."""
    for file_name, (ids, rois, y, X) in tables(reference).items():
        with open(directory / file_name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["obs_id", "roi", TARGET, *band_names()])
            for i in range(len(ids)):
                writer.writerow([ids[i], rois[i], f"{y[i]:.9g}",
                                 *(f"{v:.9g}" for v in X[i])])


def read_table(path: Path) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Read a table written by :func:`write_tables` (its header must be the current one)."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if rows[0] != ["obs_id", "roi", TARGET, *band_names()]:
        raise ValueError(f"{path.name} has an outdated header; regenerate the tables")
    body = rows[1:]
    return ([r[0] for r in body], [r[1] for r in body], np.array([float(r[2]) for r in body]),
            np.array([[float(v) for v in r[3:]] for r in body], dtype=np.float32))


if __name__ == "__main__":
    import tempfile

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        write_tables(build(Path(tmp)), out)
    print(f"wrote the SIMCA tables to {out}")
