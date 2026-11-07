# Changelog

## Unreleased

### Added

- Fit specification: every new model run stores how it was fitted
  (`fit_spec`: fold rule, pixel sample, exclusion settings), and
  `model.refit()` repeats it exactly. `project.fit(..., fit_spec=...)` gives
  full control; `enginir.project.FitSpec` is the typed form.
- Pixel sample: `fit_spec={"pixels": {"per_roi": n, "seed": s}}` draws the
  same number of pixels from every ROI, so every ROI has the same weight.
  The draw depends only on the seed and each ROI's spectra (not on ids,
  order or the NumPy version); a refit reports a different sample.
- Pixel exclusion by DModX: `fit_spec={"exclusion": {"method": "dmodx"}}`
  leaves out pixels above SIMCA's critical distance (level 0.99, at most 20 %
  of an ROI, one or two rounds), inside every cross-validation fold from its
  training pixels only. The run stores the excluded pixels
  (`model.excluded_pixels`), flagged ROIs and DCrit; a refit reports
  different exclusions. `"formula": "dmodx-2"` weights the residuals by the
  modelling power, SIMCA's default (`"dmodx-1"`, unweighted, is engiNIR's).
  `enginir.validation.dmodx`, `dmodx_limit` and `modelling_power` compute
  SIMCA 18's normalised DModX (unweighted or weighted), DCrit and MPowX.
- X and Y scaling: `PLSRegression(x_scaling=..., y_scaling=...)` and
  `PCR(x_scaling=...)` take `"center"`, `"uv"` or `"pareto"` (standard
  deviation with N - 1).
- `enginir.validation.RoundRobinGroups`: grouped folds dealt round-robin, the
  i-th group in fold i mod k (SIMCA's "every Nth" rule for groups).
- Sample groups: ROIs can carry a sample label (`project.set_sample(roi,
  label)`, the **Sample** column of the viewer's ROI table), and
  `fit_spec={"folds": {"group_by": "sample"}}` holds the ROIs of one physical
  sample out together; `"target"` groups ROIs with the same reference value.
  The fit stores the groups it used, so a refit repeats its folds and warns
  when labels changed. The viewer's fit dialog has **Hold out together**.
- The viewer's fit dialog has a **Fitting** group (pixels per ROI and seed,
  X and Y scaling, DModX exclusion), and the inspector shows how a model was
  fitted, with its excluded pixels and flagged ROIs.

### Changed

- Model runs deal ROIs round-robin to the folds (`cv="auto"`: 7 folds, the
  ROI at position i of the sample set in fold i mod 7) instead of
  `GroupKFold`. Cross-validated metrics of new fits can differ from fits made
  before on the same data; stored models keep their folds when refitted.
- An integer `cv` larger than the number of ROIs now leaves one ROI out
  instead of being refused; the viewer needs only 2 ROIs.
- New models centre the spectra and scale the reference values to unit
  variance (`x_scaling="center"`, `y_scaling="uv"`, the usual setting for
  spectra) instead of autoscaling the spectra. Their metrics differ from
  models fitted before on the same data; stored models keep their `scale`
  when refitted. `PLSRegression()` and `PCR()` used directly keep their
  defaults (`scale=True`).
- The fold rule `leave_one_roi_out` of the (unreleased) fit specification is
  now `leave_one_group_out`; with the default grouping it still leaves one
  ROI out. `cv_spec/n_groups` counts fold groups (the ROIs, by default).

## 1.1.0 (2026-11-04)

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
- Lookups by id: `project.roi(roi_id=)`, `target(target_id=)`,
  `model(model_run_id=)`, `dataset(dataset_id=)`; `set_value` and `fit` take
  the objects as well as names.
- `enginir._workflow.data_file_labels`: data files that share a file name are
  shown with the folder that tells them apart.
- `Model.refit()` fits a stored model again on exactly its sample set, recipe,
  signal domain, hyperparameters and cross-validation, stores the new run, and
  reports changed reference values and differing metrics.
- `save_csv(columns=)` writes text columns next to `sample_id`; `load_csv`
  returns them in `meta["sample_columns"]`.
- Projects with ROIs from several data files (trays):
  - **File > Add Data File...** adds a scan to the open project with its
    calibration: dark/white references from files, from frame ranges of the
    scan itself (`dark_frames`/`white_frames`), or the references of another
    data file (`references_from`), or detector levels typed in the dialog
    (`black_level`/`white_level`). The dialog plots the file's mean spectrum
    in counts with the mean references (and their saturated share), the
    other file's references or the detector levels; the data file and
    calibration are stored together. The data file list switches between the project's
    scans, and each shows only its own ROIs.
  - **Add ROI grid** lays out a tray of equal ROIs by number, ROI size,
    origin and spacing in pixels, with empty positions and a selectable name
    pattern, drawn live on the image (outside cells red) and stored all or
    none (`roi_grid`, `Dataset.add_rois`). Every grid starts from **Options >
    ROI Grid Defaults** (built in: 5 x 1 ROIs of 180 x 110 px, origin 150, 64,
    spacing 214, 10); **Make defaults** in the grid dialog keeps the current
    values for new grids. Each ROI records its grid position and layout.
  - The **ROI table** holds the values of every ROI for every target: type
    with Enter to the next row, clear to remove (`set_value(..., None)`),
    paste a column (checked first). It works without the data files.
  - The **Project ROIs** tab draws every stored ROI from the project file
    alone: a tile mosaic per data file, ROI mean spectra coloured by value,
    and the pixel spread of the selected ROI, converted by the same function
    as the fit. A project with ROIs opens on it.
  - **Fit model** is one dialog: target, signal unit, preprocessing
    (prefilled from Explore), model, components, validation, and the
    labelled ROIs per data file. It reuses an equal stored sample set and
    signal domain and stores new ones only with the fitted model.
  - **Locate file...** finds a moved scan again, accepting it only if its
    pixels equal the stored ROI pixels (`Dataset.locate`). The last shown
    scan of each project is remembered in the user settings, and windows on
    one project refresh each other.

### Changed

- **Build sample set** is gone from the Model workspace; **Fit model**
  builds or reuses the sample set.
- Names of ROIs, targets and models are labels and may repeat; ids are the
  only identity. The viewer works by id only. `Project.save_rois` checks ids,
  not names; the name-based facade functions still refuse names they create
  themselves. `project.rois`, `project.targets`, `project.models` and
  `Dataset.rois` raise a `KeyError` listing the ids when names repeat.
- Explore's ROI spectra export writes the ROI id as `sample_id`, with
  `roi_name` and `data_file` columns.
- **Add Data File** no longer asks for a signal unit for raw counts: the
  project keeps the counts with their references or levels, and DN,
  reflectance or absorbance are chosen in Explore and in **Fit model**. Only
  floating-point files ask what they hold.
- Explore enables only the signals the shown data file can give
  (`enginir.project.viewable_signal_units`): DN only for raw counts without
  calibration, no DN for already calibrated files. The chosen signal is kept
  when another data file is shown.
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

- The ROI grid no longer fails with "A ROI called … already exists" when scans
  share a file name (e.g. `stream_hsi.h5` in several folders): names are
  labels and may repeat.
- Profile, histogram and prediction plots fit their margins again when the
  tick labels change, not only when they are resized. A window that started
  without a data file clipped the absorbance axis labels (1.2 showed as 2).
- The Explore signal buttons now follow the signal in use. Showing a data
  file set its signal while the button stayed on **Raw DN**.
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
