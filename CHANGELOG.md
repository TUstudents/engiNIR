# Changelog

## 1.0.0rc2 (2026-10-20)

Second release candidate for 1.0.0. The library is unchanged from 1.0.0rc1.

### Tests

- The test suite isolates the viewer's configuration directory, so tests
  never write to the user's `~/.config/enginir`. On 1.0.0rc1 the
  recent-files tests ignored an `XDG_CONFIG_HOME` set by the environment and
  failed on GitHub runners.

## 1.0.0rc1 (2026-10-20, tagged, not published)

Release candidate for 1.0.0, the first public release.

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

### Notes for users of pre-1.0 versions

- Fitted estimators pickled with engiNIR before 1.0 cannot be loaded: their
  fitted state means something else (the PLS intercept, for example, applied
  to mean-centred spectra) and would give wrong predictions. Loading one
  raises an error; fit the model again. Pickles now record the engiNIR
  version.
- SNV divides by the sample standard deviation (n - 1), as defined by Barnes
  et al.; earlier versions used the population standard deviation, which
  scales SNV spectra by sqrt(n / (n - 1)) (0.2 % for 220 bands).
