# engiNIR

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23235907.svg)](https://doi.org/10.5281/zenodo.23235907)
[![Documentation](https://app.readthedocs.org/projects/enginir/badge/?version=stable)](https://enginir.readthedocs.io/en/stable/)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FTUstudents%2FengiNIR%2Fmain%2Fpyproject.toml)](https://pypi.org/project/enginir/)

engiNIR turns near-infrared spectra and hyperspectral images into validated
calibration models and property predictions, with every step recorded. It is
built for pharmaceutical process analytics.

## Workflow

1. **Import** EVK EIFF, ENVI, HDF5, CSV/Excel, SPC, SPA and JCAMP-DX data.
2. **Calibrate** raw counts with dark and white reference spectra.
3. **Preprocess** with strict recipes: Savitzky-Golay, SNV, MSC, EMSC, detrend.
4. **Model** with PLS or PCR, cross-validated by region of interest.
5. **Review** diagnostics and figures of merit.
6. **Predict** properties of new spectra and whole images.

## Guarantees

- Training and prediction run through the same calibration and preprocessing
  pipeline.
- Prediction is refused when new data does not match the model's instrument
  format, number of bands, calibration or signal unit.
- One project file links data, references, recipes, models and predictions;
  saved results cannot be altered.

## Usage

engiNIR needs Python 3.12 or newer. Add it to a
[uv](https://docs.astral.sh/uv/) project:

```bash
uv add enginir
```

To work on engiNIR itself, clone the repository and run `uv sync`.

### Viewer

The desktop viewer runs the whole workflow interactively: inspect images and
spectra and draw regions (**Explore**), assign reference values, fit models and
review diagnostics (**Model**), and map predictions over new scans
(**Predict**). All results are saved to a project file.

```bash
uv run enginir-viewer                                  # welcome screen
uv run enginir-viewer scan.eiff --dark dark.eiff --white white.eiff
uv run enginir-viewer project.h5
```

### Library

The library can be used in two ways.

**Project mode** records every step in a project file, so each result is
reproducible and traceable to its data. Recommended unless you know exactly
which steps you need:

```python
from enginir import PreprocessingRecipe, Project

project = Project.create("tablets.h5")
scan = project.add_dataset("scan.eiff", dark="dark.eiff", white="white.eiff")
project.add_target("API content", unit="% w/w")
scan.add_roi("tablet 1", frames=(0, 4), rows=(0, 9), values={"API content": 1.0})
# ... more ROIs
model = project.fit("API content", recipe=PreprocessingRecipe(snv=True), n_components=2)
prediction = model.predict("new_scan.eiff")      # prediction.map: one value per pixel
```

**Function mode** uses the readers, preprocessing steps, models and
validation functions directly on arrays, as scikit-learn estimators:

```python
from sklearn.pipeline import make_pipeline
from enginir.models import PLSRegression
from enginir.preprocessing import SNV
from enginir.validation import cross_validate

model = make_pipeline(SNV(), PLSRegression(n_components=4))
cv = cross_validate(model, X, y, cv="logo", groups=samples)   # cv["rmsecv"], cv["y_pred"]
```

### Documentation

The documentation is at <https://enginir.readthedocs.io>. To build it
locally:

```bash
uv sync --extra docs
uv run python -m sphinx -b html docs docs/_build/html
```

## Citation

If you use engiNIR in your work, please cite:

> Johannes Poms. (2026). TUstudents/engiNIR: A universal engineering tool for
> near-infrared spectra and hyperspectral imaging. Zenodo.
> https://doi.org/10.5281/zenodo.23235907

## License

engiNIR © 2026 Johannes Poms and contributors, licensed under
[CC BY 4.0](LICENSE).
