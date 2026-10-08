# engiNIR

`engiNIR` is the universal engineering tool for near-infrared spectra and
hyperspectral imaging.

The Python package name remains `enginir`. The desktop viewer command remains
`enginir-viewer`.

The release version is defined in `pyproject.toml`.

## What engiNIR is for

engiNIR is built for real chemometric work, not one-off notebook prototypes.
It supports the full HSI workflow:

1. load raw spectral or hyperspectral data
2. attach dataset-local calibration
3. inspect spectra and spatial context
4. define ROIs and targets inside a project-backed workspace
5. build reproducible sample sets
6. fit and persist model runs
7. apply saved models to new datasets
8. inspect and persist prediction maps with provenance

The current product is strongest as a project-backed HSI workbench with a
responsive PyQt viewer, explicit scientific workspace objects, and practical
NIR preprocessing/modeling utilities.

## Release highlights

The current release tightens the stable baseline:

- v3 HDF5 scientific workspaces with first-class datasets, calibrations, ROIs,
  targets, assignments, recipes, signal-domain contracts, sample sets, model
  runs, and prediction runs
- project-first `Explore` / `Model` / `Predict` viewer workspaces
- memory-aware and redraw-aware viewer behavior for larger HSI workflows
- grouped ROI-level model fitting and project-backed prediction workflows, with
  grouped `cv="auto"` defaulting to 7-fold `GroupKFold` when possible and
  `LeaveOneGroupOut` for smaller group counts
- Chemotools-style EMSC with validated reference, interference, and wavelength
  weighting support
- a Pyright-clean library and viewer codebase, with Qt/HDF5 optional values
  validated at their boundaries instead of suppressed
- polished desktop startup with a compact theme, welcome screen, and recent
  project/dataset launcher
- responsive welcome-screen dataset/project loading with progress feedback and
  wait-cursor coverage for blocking open/attach paths
- memory-budgeted prediction chunking with larger default EIFF frame batches
  for fewer chunk loads during full-cube prediction

## Core capabilities

- **I/O**: CSV / Excel, SPA, SPC, JCAMP-DX, ENVI cubes, HDF5, and EVK EIFF
- **Preprocessing**: SNV, MSC, EMSC, Savitzky-Golay, detrending, smoothing
- **Models**: PLS, PCR, SVR, Random Forest, MLP, SIMCA, LDA
- **Validation**: grouped cross-validation, Kennard-Stone, outlier metrics,
  regression figures of merit
- **Project workspace**: persisted datasets, calibrations, ROIs, targets,
  assignments, sample sets, model runs, and prediction runs
- **Viewer**: interactive HSI exploration, ROI capture, project-backed
  modeling, and prediction map review

## Installation

Requirements:

- Python `>=3.13,<3.14`
- `uv` is the recommended environment manager

```bash
git clone https://rcpe-gitlab/pat/enginir.git
cd enginir

# runtime dependencies
uv sync

# development tools
uv sync --extra dev

# documentation tools
uv sync --extra docs

# everything
uv sync --extra dev --extra docs
```

## Quickstart: spectra and models

```python
import numpy as np
from sklearn.pipeline import Pipeline

from enginir.io import load_spectra
from enginir.models import PLSRegression
from enginir.preprocessing import EMSC, SavitzkyGolay
from enginir.validation import cross_validate, metrics_report

X, wavelengths, meta = load_spectra("data/demo_spectra.csv")
target_band = np.argmin(np.abs(wavelengths - 1450.0))
baseline_band = np.argmin(np.abs(wavelengths - 1300.0))
y = X[:, target_band] - X[:, baseline_band]

model = Pipeline([
    ("emsc", EMSC(method="mean", order=2)),
    ("sg", SavitzkyGolay(window_length=11, polyorder=2, deriv=1)),
    ("pls", PLSRegression(n_components=4)),
])
results = cross_validate(model, X, y, cv=5)

print(metrics_report(y, results["y_pred"]))
```

This example uses the bundled `data/demo_spectra.csv` file and derives a small
demonstration target from two spectral bands so the snippet is runnable after
installation. Real calibration work should replace `y` with measured reference
values from the sample metadata or an external lab table.

All estimators and transformers follow the scikit-learn API. Put stateful
preprocessing such as MSC or EMSC inside `sklearn.pipeline.Pipeline` for
cross-validation and held-out prediction.

## Quickstart: viewer

Launch the standalone viewer:

```bash
enginir-viewer
```

Open a dataset directly:

```bash
enginir-viewer data/sample.eiff
enginir-viewer data/sample.eiff --dark dark.eiff --white white.eiff
enginir-viewer data/sample.eiff --project project.h5
```

Open a project directly:

```bash
enginir-viewer project.h5
enginir-viewer --project project.h5
```

Open from Python:

```python
from enginir.viewer import open_project, open_viewer, run

viewer = open_viewer("data/sample.eiff")
project_viewer = open_project("project.h5")
attached_viewer = open_viewer("data/sample.eiff", project="project.h5")
run()  # blocking outside Jupyter
```

The viewer is built around three workspaces:

- **Explore** for spatial/spectral inspection and ROI capture
- **Model** for targets, sample sets, diagnostics, and model fitting
- **Predict** for saved-model inference and prediction-map review

## Project-backed workflow

Project files are the scientific workspace in engiNIR. They persist the real
domain objects needed for repeatable HSI work.

```python
import numpy as np

from enginir.io import create_project, load_project_store
from enginir.project import DatasetRef, TargetDefinition
from enginir.viewer import ProjectState

create_project("project.h5")
wavelengths = np.linspace(1000.0, 1700.0, 50, dtype=np.float32)

project = ProjectState()
project.open_project("project.h5")

project.register_dataset(
    DatasetRef(
        dataset_id="dataset_1",
        source_path="/data/sample.eiff",
        source_format="eiff",
        source_modality="nir_hsi",
        cube_shape=(20, 15, 50),
        n_bands=50,
        wavelengths=wavelengths.astype("float32"),
        loader_metadata={},
        import_timestamp="2026-04-22T10:00:00+00:00",
    )
)
project.define_target(
    TargetDefinition(
        target_id="target_moisture",
        name="moisture",
        unit="% w/w",
        value_type="float",
        missing_value_policy="allowed",
        created_at="2026-04-22T10:01:00+00:00",
    )
)

store = load_project_store("project.h5")
print(store["datasets"].keys(), store["targets"].keys())
```

## Documentation

Start here:

- published docs landing page: [docs/index.rst](docs/index.rst)
- viewer guide: [docs/user_guide/viewer.rst](docs/user_guide/viewer.rst)
- workspace/domain model: [docs/workspace_domain_model.md](docs/workspace_domain_model.md)
- HSI product roadmap: [docs/hsi_product_roadmap.md](docs/hsi_product_roadmap.md)
- engineering status: [docs/engineering_status.md](docs/engineering_status.md)

Build the docs locally:

```bash
uv run python -m sphinx -W -b html docs docs/_build/html
```

## Repository layout

```text
src/enginir/     Python package
docs/            Sphinx documentation and reference material
tests/           pytest suite
notebooks/       notebooks and workflow examples
plan/            engineering plans and design notes
theory/          theory notes and references
```

## Status

engiNIR is a strong baseline release for project-backed NIR/HSI work.
The original library implementation plan and test-suite compression plan are
complete. Current local verification is ruff clean, pyright clean, and
`353` pytest tests passing. The Sphinx documentation builds cleanly with
warnings treated as errors.

The product is stable enough to build on, but the roadmap still prioritizes:

- stronger model comparison and interpretation
- richer prediction analysis and uncertainty workflows
- broader chemometric workflow support

## License

engiNIR © 2026 Johannes Poms.
Please cite it as 
Licensed under CC BY 4.0.
