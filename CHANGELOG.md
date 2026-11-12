# Changelog

## Unreleased

The terminology contract applied where 1.5.0 missed it, and the contracts
read against the code.

### Breaking

- **New project file format `enginir-1.5`; projects written by earlier
  versions do not open** (no migration, no repair; create them again).
  `/project_meta` stores `FileSchemaVersion = "enginir-1.5"`, `FileCreator`
  and `FileWriter` (`"engiNIR <version>"`) instead of `schema_version`;
  every write updates `FileWriter` and `updated_at`, and every timestamp
  must state its UTC offset. `enginir.io.PROJECT_SCHEMA_VERSION` is now
  `FILE_SCHEMA_VERSION`.
- The reader refuses a project whose core is incomplete, now including a
  model without its executable model or without the MSC/EMSC state its
  recipe needs; missing display data (a model's metrics or CV
  specification) no longer stops a project from opening.
- The ROI field `use` is renamed `job` in the project file and the API:
  `Dataset.add_roi(..., job=...)`, `Project.set_job`, `Project.save_roi_job`,
  `TrainingROI.job`, `enginir.io.project_store.save_roi_job`.
- Recipes store every field, also those without effect; a recipe given as a
  mapping must have every field (`PreprocessingRecipe(...).to_dict()` gives
  one). Recipe ids change.
- New PCR models are named in principal components (`"PCR 5 PC, SNV"`);
  PLS keeps `LV`. Stored names are labels and do not change.
- Exported CSV columns: the Model plots name the measured value
  `target_value` (was `reference`), the score plot `lv1_calibration_score`
  and `lv2_calibration_score` (was `..._training_score`), the per-ROI
  prediction table `mean_dmodxps` and `mean_t2ps` (was `mean_dmodx`,
  `mean_t2`).
- The Predict workspace names the distances of predicted spectra DModXPS and
  T²PS (map layers, hover readout, the exclusion checkbox); the Model
  workspace keeps DModX and T² for calibration spectra.
- Cancellation errors say "canceled" (US spelling); scripts that match the
  message must be updated.

### Fixed

- Prediction progress said "Calibrating" for the reference correction step;
  it now says "Reference correction".
- The Inspector showed "centre", "Fitted on: pixel" and raw reference
  correction modes (`dark_white_reference`); it now shows "center",
  "Fitted on: pixels" and the modes in words.
- Predict hints pointed to a "Predict column"; the ROI table's column is Set.
- Contracts and documentation: leftovers of the 1.5.0 rename (garbled
  sentences, wrong levels, words not used), labels matched to the viewer,
  "pixel sample" is now "pixel subsample", and the prediction interval's
  leverage is stated as used (h = 1/R + T²PS/(R − 1) over the calibration
  ROIs).
- The terminology test also checks one-word labels for British spelling.

## 1.5.0 (2026-11-11)

One vocabulary for the process analytical technology expert: everything a
person reads follows the new terminology contract
(`contracts/terminology_v3.md`), decided term by term and reviewed
independently. US English throughout.

### Breaking

- `Prediction.uncertainty_map(level)` is renamed `Prediction.interval_map(level)`;
  the old name is removed.
- Calibration ROIs are stored as `use = calibration` (the default in the
  API). A missing `use` and the 1.4 name `training` still read as
  calibration, and re-saving such an ROI is not a change. Passing
  `use="training"` to the API is refused with the rename. A project with ROIs
  stored by 1.5 does not open in engiNIR 1.4.
- Viewer labels, plot titles and the per-ROI CSV columns use the new terms
  (CSV columns `dataset`, `excluded`, `target_value`); scripts that read
  exported CSVs by column title must be updated.
- Library log and error messages use the new terms; scripts that match
  message texts must be updated.

### Added

- `contracts/terminology_v3.md`: the terms, their definitions, synonyms in
  SIMCA, The Unscrambler, PLS_Toolbox, ENVI and the standards, the internal
  names and the words not used.
- Documentation: a Glossary page.
- Viewer: the ROI table's **Set** column (Calibration or Prediction, chosen
  from a drop-down) replaces the Predict checkbox; the Project Browser shows
  each model's calibration set and input contract beneath it.
- A test that scans every viewer string for words not used, British
  spellings and capitalization (title case for buttons and menu items,
  sentence case for titles).

### Changed

- Terms in the viewer, messages and documentation:
  - Reference correction (was calibration, for dark/white references and
    detector levels), Dataset (was data file);
  - calibration ROIs, Calibration set, calibration range (was training,
    sample set);
  - Model (was model run), Input contract (was signal domain);
  - Target value (was reference value), Raw counts (DN) (was DN);
  - Scatter correction and Smoothing and derivatives (were Normalisation and
    Filtering);
  - Prediction interval (was Uncertainty), excluded pixels (was removed),
    "above DCrit" or "outside the model" (no "outlier").
- Metrics name their level: "RMSECV (per ROI)", "RMSECV (per pixel)".
- Contracts and documentation: US spelling and sentence-case headings.
- The stored names of the reference correction (`calibrations/`,
  `calibration_id`, `CalibrationSpec`) keep their names until 2.0.

## 1.4.0 (2026-11-10)

### Added

- Prediction intervals (leverage formula, ASTM E1655): `prediction.uncertainty_map(level=0.95)` gives each pixel's ± half-width q · pixel RMSEE · √(1 + h), with h from T²PS and the training ROIs as the samples; `model.predict_rois()` gives each prediction ROI's mean interval (`row.interval`), following the pixel removal. Pixels and ROIs outside the model (above the 0.99 limits) get no interval. Nothing is stored.
- Fit: `interval_coverage_roi` and `interval_coverage_pixel` metrics, the share of the run's held-out predictions inside their 0.95 intervals.
- Viewer: the Predict map's **Uncertainty** layer (± half-width at 95 %, pixels outside the model marked), the interval in the hover readout and the median interval in the summary.
- Band importance: every new PLS and PCR model stores VIP (PLS), the
  selectivity ratio (target projection, PLS and PCR) and the mean spectrum of
  the fitted pixels (`model.diagnostics`). The viewer's Spectral page shows
  VIP with lines at 1.0 and 0.8 and the mean spectrum behind it, VIP sorted
  with the top bands labelled, and the selectivity ratio; PLS runs from
  before get VIP from their stored weights.
- Band selection: `PreprocessingRecipe(bands_keep=..., bands_exclude=...)`
  keeps or leaves out inclusive ranges (nm, or band numbers without
  wavelengths) as the last recipe step, so normalization and MSC/EMSC see
  the whole spectrum and the model only the selected bands; ranges are
  normalized before the recipe id, and recipes stored before keep their ids.
  `enginir.preprocessing.BandSelection`, `output_band_indices`.
- `project.propose_bands(target, ...)` proposes keep ranges by forward
  interval PLS (one-SE stop) and reports `nested_rmsecv`, the error of the
  whole procedure with the search repeated inside every outer fold, next to
  the RMSECV with all bands. Nothing is stored.
- Viewer: the fit dialog's **Bands** row (Keep, Exclude, Water bands
  suggestions, Select on plot..., Propose...); Fit again keeps the ranges;
  the Explore preview shows left-out bands as missing.
- User guide: *Band importance and band selection*.
- Autofit: rules `f_test` (Haaland–Thomas F test over the fold groups) and
  `q2_lookahead` (SIMCA R1, but a failing component is kept when the next one
  passes), and the **noise guard**, on by default with `n_components="auto"`:
  at most one component more than those explaining ≥ 0.5 % of the spectra,
  then no trailing component that explains < 1 % of the spectra and does not
  improve the prediction. `components/rule_choice` stores the rule's number;
  `noise_guard: off` chooses as 1.3 did. The curve stores `rmsee_roi`,
  `dw_weights` (Durbin–Watson of each component's weights), `noise` and the
  structure ceiling.
- Prediction ROIs: an ROI's **use** is training or prediction. Prediction
  ROIs (Prediction only in the ROI and ROI grid dialogs, the Predict column
  of the ROI table, `use="prediction"`) never enter a fit; they are the
  tablets to predict. `model.predict_rois()` predicts them from the project
  file alone with each ROI's own calibration, with per-ROI mean, median, SD,
  5–95 %, DModX and T², an optional removal of pixels above the DModX limit
  (default: the model's own), and RMSEP and bias over ROIs with a value.
- Viewer, Predict: **Predict ROIs...** (one or more models, one or all data
  files) and **Predict map...** (a project data file, with its prediction
  ROIs); both name models and files, show the chain read-only and refuse an
  incompatible file with the reason. The per-ROI plot and table, a map colour
  scale from the training range (own limits in the ⋯ menu), ROI outlines with
  their means, and a grouped summary. The free path, dataset id and
  calibration fields of the old dialog are gone; a scan to predict is added
  with the Add data file dialog, which shows what the model was trained with.
  The per-ROI plot labels each mean and rings tablets with unusual spectra;
  the ROI table shows each model's results (mean, SD, % removed, % above
  DCrit, residual; sortable, not stored); the histogram shows the kept ROI
  pixels; map and per-ROI results always describe the same data file; the
  summary has one headline and no repeats; the status line describes the
  hovered pixel or tablet; the Controls dock sets the colour scale, the map
  overlays and the histogram source and shows DCrit.
- Fold warnings: a fit records and logs `few_groups` and `one_level_per_file`
  (a data file's ROIs share one value but are held out in different folds)
  in `cv_spec/@warnings`; the fit dialog, the readiness line and the summary
  show them.
- Viewer, Model: metrics **per ROI** and **per pixel** in separate groups,
  every Model plot names its level, a **?** button explains each plot (the
  same texts in the new user guide page *Reading the Model plots*), and the
  Components page shows four plots: RMSECV and RMSEE with every rule's reason,
  R²Y and Q² per ROI, each component's Q²ₐ, ΔR²X and roughness, and R²X and
  R²Y per pixel.

### Changed

- The application icon runs from amber to deep red, left to right, short
  to long wavelengths as in a physical spectrum (it was reversed). The
  documentation shows it as logo and favicon.
- README and documentation start page describe the workflow of 1.4:
  several scans, ROI grids, automatic components, band selection,
  applicability and prediction intervals, prediction ROIs.
- The prediction check compares restored MSC/EMSC state with the band count
  after Savitzky-Golay (before a band selection) and the model with the
  selected bands.
- Viewer, Add data file: **Detector levels** (the file's levels, else the
  instrument defaults) is preselected; references are chosen deliberately.
- Viewer, Explore: the data-file docks (Histogram, RAW, Sensor Frame,
  Spatial Profile) show only on the Data file tab. The ROI table opens with
  Project ROIs in the right-hand column and is hidden on the Data file tab
  (opened there, it takes the Sensor Frame's place); the choice is kept per
  tab, and the table shows the page's ROIs.
- Viewer, ROI mosaic: tiles in a compact grid (ROI grid positions, else scan
  order), text in reserved room that zooms with the tiles, labels without
  the shared name prefix, outlines in the value colour of the mean spectra;
  **View** offers the scan positions and a contrast stretch per data file.
- Viewer, Model, Distance page: the DModX and T² histograms show bins up to
  the 99.5th percentile and the tail as one bar (a few far pixels put every
  other pixel into one bin), the share above each limit is shown next to the
  expected 1 %, and a new plot shows each ROI's share of pixels above the
  limits, sorted, with clickable bars.
- Viewer, Model: the score plot adds the ROI-level Hotelling T² ellipses
  (the ROI means as observations, with their covariance) next to the pixel
  ellipses, and a legend names both.
- A repeated reflectance clipping warning (the same ROI converted again) is
  logged once, then at debug level.

### Fixed

- `model.predict(path)` of a data file registered in the project now uses
  the file's own calibration. Without `dark`/`white` it built a new
  calibration from the detector levels, so a scan recorded with references
  was predicted with a different calibration than its ROIs.
- Viewer: the Controls dock could hide its lower rows (the Predict buttons)
  since 1.4's sideways scrolling; the content now always fits the dock's
  height.

- Viewer on WSLg Wayland: the session could end ("xdg_surface buffer does not
  match the configured maximized state") when the docks' minimum widths added
  up to more than the screen, e.g. on Predict after a prediction. The window
  no longer turns its docks' minimums into its own minimum size, and the
  Controls dock scrolls sideways instead of requiring 1162 px. Predict names
  the model instead of showing its id.

- Viewer: ROIs with equal target values took the lowest colour of the scale
  while the colour bar showed them in the middle.
- Viewer: plots of a data file without wavelengths are labelled "Band", not
  "Wavelength (nm)".
- Viewer, Model: the summary showed "-" for the cross-validation of models
  fitted since 1.2 (it now reads the stored folds, e.g. "7 folds, round-robin
  over 20 samples") and ids where names belong (model, sample set,
  readiness, status lines).
- Viewer, Model, Spectral page: VIP and the selectivity ratio had a white
  background in the dark theme (the mean spectrum's axes), files without
  wavelengths were labelled "Wavelength" and "175 nm", the five largest VIP
  labels overlapped (now listed in one corner), and legends had white boxes.
- Viewer, Model: a few reference levels (1, 4, 7, 10) were binned into one
  flat block; they get one bar each. A unit of "1" is no longer printed
  after every statistic.
- Viewer, Model: checking whether the Explore signal can be modelled (DN for
  data with references or detector levels) logged an ERROR after every fit;
  it now only explains it in the readiness line.
- Viewer on WSLg Wayland: a drop-down list stayed on screen after choosing
  an item (a WSLg fault, microsoft/wslg#1522); the list's native window is
  now released when it closes.

### Compatibility

- Projects stay schema v3. An engiNIR before 1.4 ignores the ROI `use`
  attribute with a warning and treats prediction ROIs as training ROIs:
  a fit there includes those that have a target value. Models fitted
  before 1.4 predict with intervals when they have projection data
  (1.3 and later); the coverage metrics and the selectivity ratio need a
  refit.

## 1.3.0 (2026-11-09)

### Added

- Choosing the number of components: every PLS and PCR model stores its
  component curve (RMSECV with standard error, Q², Q²(cum), R²X, R²Y against
  the number of components, computed with the model's folds), and
  `project.fit(..., n_components="auto")` chooses the number by a rule
  (`one_se` by default, `min_rmsecv`, `q2_limit`); `model.component_curve`,
  `enginir.validation.choose_components`, `PLSRegression/PCR.predict_components`.
- Fit statistics and training diagnostics of every model: RMSEE and adjusted
  R² (per pixel and per ROI), CoeffCS, Hotelling T² and DModX of every
  training pixel with their limits.
- Viewer: Components and Distance pages, T² ellipses in the score plot,
  loadings of the first three components, CoeffCS; **Auto** components in
  the fit dialog; **Fit again...** with the selected model's settings.
- Applicability diagnostics of predictions: every predicted pixel gets
  DModXPS (distance to the model) and T²PS, with the 0.99 limits from the
  training data and a summary (share of pixels above the limits, mean
  prediction within and outside them). Model runs store the projection data
  they need (`prediction_model`: `x_mean`, `x_std`, `rotations`, `loadings`,
  `score_variances`, `s0`, limits); models fitted before have no maps until
  they are fitted again. `Prediction.dmodx_map`, `.t2_map`, `.diagnostics`,
  `.provenance`.
- Wavelength warning at prediction: when the scan and the model both have
  a real wavelength axis and they differ, the prediction logs a warning with
  the largest difference and records it in `provenance["wavelength_warning"]`
  (Predict summary: **Wavelengths**); it never blocks or changes the
  prediction. `enginir.prediction.wavelength_mismatch`.
- Viewer: **Layer** selector in the Predict map (Prediction, DModX, T²) with
  the pixels above the limit marked, and the applicability summary.
- AsLS baseline correction (asymmetric least squares, as in SIMCA):
  `enginir.preprocessing.AsLSBaseline` and `PreprocessingRecipe(asls=True,
  asls_lambda=..., asls_p=..., asls_iterations=...)`, first in the recipe
  order, as is in absorbance and flipped in reflectance and DN. Scripts
  only; the viewer shows and keeps it.
- `SavitzkyGolay(edges="drop")` and `PreprocessingRecipe.output_wavelengths`.

### Fixed

- Viewer: a new project (and an exported CSV) gets its extension (`.h5`,
  `.csv`) when the name was typed without one; the file dialog did not add
  it on some platforms, and the open dialog, filtering on `.h5`, then hid
  the project. The open dialog also offers *All files*. `Project.create`
  documents that it replaces an existing file.
- Viewer on Linux: the file dialogs' toolbar buttons (back, up, new folder,
  list or detail view) show their icons again; the theme's button padding
  left no room for them.
- Viewer theme: disabled buttons, fields and lists are greyed out (the
  style sheet's text colour had hidden the disabled state); spin boxes and
  combo boxes show their arrows; tables no longer draw a rounded box around
  their headers; check boxes and radio buttons have a visible outline in the
  dark theme and an accent fill when checked; the progress bar is visible in
  the light theme; a focused field has an accent border; the default
  button of a dialog stands out; secondary text is sized in points (85 % of
  the UI font) instead of a fixed 9 px; the light theme's selection is
  stronger and an unfocused selection is grey; slim scroll bars, themed
  tooltips, a splitter handle on hover; plot backgrounds follow the window
  colours and reference lines share one guide colour.
- Viewer: closing the viewer after a project window was open no longer ends
  the process with a segmentation fault (exit code 139). PyQt's exit cleanup
  crashed when the application-wide event filter ran during it; the filter is
  now removed before that cleanup. Project data was never affected.
- Viewer under WSL (WSLg): uses native Wayland instead of X11, so menus no
  longer fall behind the main window after a few seconds; Qt draws the title
  bar with the engiNIR icon. Other Linux sessions with both Wayland and X11
  keep X11; `QT_QPA_PLATFORM` still overrides. The window icon is set once
  for the application, so dialogs carry it too.

### Changed

- PCR computes its principal components with a full SVD (reproducible);
  results can differ from earlier fits in the last digits.
- **Breaking:** predictions are not stored. `model.predict(...)` returns an
  in-memory `Prediction` and writes nothing to the project (the scan is not
  registered); every prediction writes one log entry with its provenance
  (`checksum=True` adds the SHA-256 of the data file). The viewer keeps the
  last three predictions of the session and does not show stored runs.
  Storing a prediction is a low-level operation
  (`predict_with_model_run(..., persist_artifacts=True)`); such runs are
  listed in `project.predictions` as `StoredPrediction`.
- `ProjectState.register_prediction_run` and `finalize_prediction_result`
  were removed; `predict_dataset` no longer takes `return_arrays`.
- **Breaking:** Savitzky-Golay in recipes drops the first and last
  `sg_window // 2` bands (`sg_edges="drop"`, the new default, as SIMCA does):
  new models with a Savitzky-Golay filter have fewer bands (their
  coefficients, loadings and stored wavelengths use the shorter axis) and
  slightly different numbers. Stored recipes keep their ids and keep
  extrapolating the edges (`"interp"`), so stored models refit unchanged;
  `PreprocessingRecipe(..., sg_edges="interp")` keeps the old behaviour. A
  recipe given as a dict is the stored form (without `sg_edges` it keeps the
  edges). Projects with the new recipe fields need this version or later.
- `PreprocessingRecipe.build_steps`, `make_model_pipeline` and
  `apply_preprocessing_recipe` take `signal_unit` (required with AsLS).
- **Breaking:** the fit stores the ROI scores of the score plot
  (`diagnostics/roi_scores`, `roi_score_pixels`: the mean training score of
  each ROI's fitted pixels). The viewer only reads them, so the plot now
  works with a pixel sample and DModX exclusion (it was empty before);
  models fitted before show a placeholder until they are fitted again.

## 1.2.0 (2026-11-07)

### Added

- Fit specification: every new model run stores exactly how it was fitted
  (`fit_spec`: folds and fold groups, pixel sample, pixel exclusion), and
  `model.refit()` repeats it. `project.fit(..., fit_spec=...)` gives full
  control; `enginir.project.FitSpec` is the typed form. User guide: *Fit
  specification*, including a comparison with SIMCA 18.
- Sample groups: ROIs can carry a sample label (`project.set_sample(roi,
  label)`, the **Sample** column of the viewer's ROI table), and
  `fit_spec={"folds": {"group_by": "sample"}}` holds the ROIs of one physical
  sample out together; `"target"` groups ROIs with the same reference value.
  The fit stores the groups it used, so a refit repeats its folds and warns
  when labels changed.
- Pixel sample: `fit_spec={"pixels": {"per_roi": n, "seed": s}}` draws the
  same number of pixels from every ROI, so every ROI has the same weight.
  The draw depends only on the seed and each ROI's spectra (not on ids,
  order or the NumPy version); a refit reports a different sample.
- X and Y scaling: `PLSRegression(x_scaling=..., y_scaling=...)` and
  `PCR(x_scaling=...)` take `"center"`, `"uv"` or `"pareto"` (standard
  deviation with N - 1).
- Pixel exclusion by DModX: `fit_spec={"exclusion": {"method": "dmodx"}}`
  leaves out pixels above SIMCA's critical distance (level 0.99, at most 20 %
  of an ROI, one or two rounds), inside every cross-validation fold from its
  training pixels only. The run stores the excluded pixels
  (`model.excluded_pixels`), flagged ROIs and DCrit; a refit reports
  different exclusions. `"formula": "dmodx-2"` weights the residuals by the
  modelling power, SIMCA's default (`"dmodx-1"`, unweighted, is engiNIR's).
- `enginir.validation.dmodx`, `dmodx_limit` and `modelling_power`: SIMCA 18's
  normalised DModX (unweighted or weighted), DCrit and MPowX, verified
  against SIMCA 18.1.
- `enginir.validation.RoundRobinGroups`: grouped folds dealt round-robin, the
  i-th group in fold i mod k (SIMCA's "every Nth" rule for groups).
- Viewer: the fit dialog has **Hold out together** and a **Fitting** group
  (pixels per ROI and seed, X and Y scaling, DModX exclusion); the inspector
  shows how a model was fitted, with its excluded pixels and flagged ROIs.

### Changed

- New models are validated with ROIs dealt round-robin to the folds
  (`cv="auto"`: 7 folds, the ROI at position i of the sample set in fold
  i mod 7) instead of `GroupKFold`, and centre the spectra and scale the
  reference values to unit variance (`x_scaling="center"`,
  `y_scaling="uv"`) instead of autoscaling the spectra. Their metrics differ
  from models fitted before on the same data. Stored models keep their folds
  and their `scale` when refitted; `PLSRegression()` and `PCR()` used
  directly keep their defaults (`scale=True`).
- An integer `cv` larger than the number of ROIs now leaves one ROI out
  instead of being refused; the viewer needs only 2 ROIs.
- `cv_spec/n_groups` counts fold groups (the ROIs, unless ROIs are grouped).

### Fixed

- Opening another viewer window no longer restyles every open widget; the
  repeated restyle could crash Qt.

### Compatibility

- Projects stay schema v3. An engiNIR before 1.2 opens a project with fit
  specifications and sample labels, ignores them with a warning and still
  predicts with the stored models, but cannot refit such models or re-save
  such ROIs.

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
