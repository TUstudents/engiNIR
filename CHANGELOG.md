# Changelog

## Unreleased

### Added

- Viewer plot panels: every plot (Explore, Model, Predict) has a header with
  its title, a maximize button and an export menu. The menu saves the current
  view as PNG or SVG (300 dpi, light print style) and exports the plotted
  values as CSV. Each CSV starts with `#` provenance lines: project, data file
  or dataset, model run, sample set, recipe, components, CV scheme, and the
  frame/row/band shown.
- Model diagnostics on two pages: **Fit** (predicted vs reference,
  residuals, target distribution, training scores) and **Spectral**
  (regression coefficients, loadings). Predicted vs reference now annotates
  R², RMSECV and bias.
- Prediction distribution plot under the prediction map, following the
  selected output. The map header holds the output selector, Pan, Zoom and
  Reset View.
- **File > Export Plots...** writes every plot of the current workspace as
  PNG and CSV into one folder.
- The maximized plot of each workspace is remembered between sessions;
  **View > Reset Layout** clears it.

### Changed

- The spectral-plot selector of the Model workspace is gone: coefficients
  and loadings have their own plots.
- The scores plot is labelled as training scores.
- The RAW and Sensor Frame panels draw with pyqtgraph, and the histogram
  sets its limits from the bins. A frame step with every Explore plot shown
  takes about 39 ms instead of 73 ms on the benchmark machine. Their PNG/SVG
  exports are still drawn with matplotlib.
- Changing one Model diagnostic redraws only that plot: 46 ms instead of
  121 ms on the benchmark machine. Building the diagnostics takes 230 ms
  instead of 380 ms.

### Fixed

- Replacing a stored target value (`Project.set_value`, and editing a value in
  the viewer) was refused as a "duplicate target assignment". A new value now
  replaces the stored one.
- Re-attaching a registered data file warned "wavelength axis differs" for the
  same file. The project stores wavelengths as float32, and the file's float64
  axis differs from it by about 1e-4 nm. Axes are now compared at the stored
  precision.
- Large prediction maps are shown as a subsampled preview, and their axes
  and hover readout reported preview indices instead of the true frame and
  row. Axes, hover and CSV now use the true indices.

## 1.0.1 (2026-10-20)

### Fixed

- Cross-validation of model runs (`Project.fit` and the viewer's Model
  workspace) is reproducible. With 7 or more ROIs, `GroupKFold` received the
  random ROI ids as group labels, so which ROIs shared a fold, and with it
  RMSECV, R², bias and the cross-validated predictions, changed whenever the
  same analysis was repeated in a new project (the documented example
  reported R² between 0.87 and 0.94). Folds now follow the order of the ROIs
  in the sample set.
- The fitted model and its predictions were never affected: the final model
  is fitted on all pixels. Stored model runs are unchanged. Refitting an
  analysis with 7 or more ROIs under `cv="auto"` or an integer `cv` can
  report different, now reproducible, cross-validation metrics than under
  1.0.0. Leave-one-ROI-out (`cv="logo"`, and `"auto"` with fewer than 7 ROIs)
  is not affected.

## 1.0.0 (2026-10-20)

First public release.

### Workflow

- `enginir.Project` runs the project workflow in one call per step:
  `add_dataset`, `add_target`, `Dataset.add_roi`, `fit` and `Model.predict`,
  with datasets, ROIs, targets and models addressed by name. Calibrations,
  sample sets, signal domains, stored recipes and provenance are created
  automatically.
- One HDF5 project file (schema `viewer-v3`) holds datasets, calibrations,
  ROIs, reference values, preprocessing recipes, signal domains, sample sets,
  model runs and prediction runs, linked by stable ids.
- Desktop viewer `enginir-viewer` with the Explore, Model and Predict
  workspaces, built on the same `Project` rules.

### Library

- Readers for EVK EIFF, ENVI, HDF5, CSV/Excel, SPC, SPA and JCAMP-DX.
- Preprocessing: Savitzky-Golay smoothing and derivatives, SNV, MSC, EMSC,
  detrending and smoothing, as scikit-learn transformers and as validated
  `PreprocessingRecipe`s.
- Models: PLS and PCR for the project workflow; SIMCA, LDA, SVR, random
  forest and MLP as scikit-learn estimators.
- Validation: grouped cross-validation, Kennard-Stone, RMSEC/RMSECV/RMSEP,
  R², bias, RPD and RPIQ, leverage, Hotelling T² and Q outliers.
- Plots of spectra, predictions, scores, loadings, VIP and image maps.

### Guarantees

- Training and prediction convert and preprocess spectra with the same code;
  MSC and EMSC are fitted inside every cross-validation fold.
- Cross-validation groups by ROI, so pixels of one ROI never appear in both
  training and validation folds.
- Prediction is refused before it runs when the data cannot produce the
  model's signal (instrument format, number of bands, calibration, signal
  unit).
- Every object is validated when it is written and when a project is opened;
  stored objects cannot be overwritten with different content.
- Prediction streams through image files in bounded memory.

### Packaging

- Python 3.12 and newer; CI runs on 3.12, 3.13 and 3.14.
- Install from PyPI with `uv add enginir` or `pip install enginir`.
- Documentation on Read the Docs: <https://enginir.readthedocs.io>.
- Each GitHub release is archived on Zenodo and uploaded to PyPI by a
  workflow that checks that the tag matches the version and that CI passed
  for the commit.

### Notes for users of pre-1.0 versions

- Fitted estimators pickled with engiNIR before 1.0 cannot be loaded: their
  fitted state means something else (the PLS intercept, for example, applied
  to mean-centred spectra) and would give wrong predictions. Loading one
  raises an error; fit the model again. Pickles now record the engiNIR
  version.
- SNV divides by the sample standard deviation (n - 1), as defined by Barnes
  et al.; earlier versions used the population standard deviation, which
  scales SNV spectra by sqrt(n / (n - 1)) (0.2 % for 220 bands).

## Release candidates

1.0.0rc1 to 1.0.0rc3 were tagged but not published; 1.0.0rc4 was published
as a pre-release. The library code is the same in all release candidates and
in 1.0.0; they differ in tests, packaging and documentation only.
