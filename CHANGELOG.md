# Changelog

## 1.7.5 (2026-11-22)

Spatial pixel binning of a dataset and tidier menus. The format stays
`enginir-1.7`; projects without binned datasets are unchanged.

### Added

- **Spatial pixel binning of a dataset.** Add Dataset's **Binning** (2 × 2,
  3 × 3, 4 × 4) and `Project.add_dataset(path, binning=k)` average k × k
  blocks of frames and rows every time the file is read, before the reference
  correction. Counts stay counts (the mean is rounded and keeps the file's
  integer type), so levels, references, ROIs and models work unchanged on the
  binned pixels. Reference frames are frames of the file as stored. A file has
  one binning in a project, fixed when it is added. A model records its
  binning: a new scan is binned like the model to be predicted, a scan
  registered with another binning is refused, and a calibration set mixing
  binnings is refused. `enginir.io.bin_blocks`, `bin_rows` and
  `bin_frame_chunks` do the same on arrays. The format stays `enginir-1.7`;
  the binning is stored only for binned datasets. engiNIR 1.7.4 opens a
  project with a binned dataset and refits its models from the stored ROIs,
  but refuses its data file (the shape differs) rather than read it unbinned.

### Changed

- **Menus tidied** (plan `plans/menu-polish.md`):
  - File: Open Project is **Ctrl+O**; Open Data… and Add Dataset… left File
    (**Project > Add Dataset…**, now **Ctrl+D**); Close Window is gone (Quit
    stays); Open Recent lists projects first, then data files.
  - Edit: **Paste** (Ctrl+V) next to Copy; **Preferences…** moved here from
    Tools, which is gone.
  - View: Title Case items; **Go To** the five steps with **Ctrl+1** to
    **Ctrl+5**; "Reset View" as on the image's button.
  - Project: **Rename Dataset…** (was Name Dataset…).
  - ROI: **ROI Grid Defaults…** (from Tools), **Find ROI…** (from Edit), and
    **Delete** deletes the selected ROI on the image or the mosaic.
  - Predict: **Predict Map…** on **F7**.
  - Help: **User Guide**, **Keyboard Shortcuts** and **Report a Problem…**;
    the log entries of Tools moved here.

### Removed

- **The Band image panel** (View > Band image): it showed no more than the
  Spatial image.

## 1.7.4 (2026-11-22)

Viewer fixes and a smaller, sharper test suite. Projects, scripts and the
format (`enginir-1.7`) are unchanged.

### Fixed

- **Tooltips break into lines** of at most 72 characters. The help of each
  plot (its **?** button) and the other long tooltips were one wide line.
- **The ROI mosaic fills its frame:** an empty message line took a dark strip
  under the mosaic, so it ended above the ROI mean spectra and Selected ROI
  pixels plots.
- **Sensor frame and Spatial profile without data** show the panel colour,
  like the Spatial image beside them, not the darker window colour.

### Changed

- **Tests:** 1190 down to 694 test cases (830 to 578 test functions) with the
  same line coverage. Table-style cases are one test each; tests that detect
  nothing the others do not, and tests of layout and colours only, are gone;
  the model-run and prediction tests run without Qt (`tests/_sessions.py`);
  the kernel PLS timing test is in `benchmarks/`.
- **Viewer internals:** members nothing called are removed
  (`ProjectState.fit_model_run`, `build_sample_set`, `predict_dataset`,
  `get_target_assignment`, `ensure_model_run_artifact_refs`,
  `HSIViewer.show_project_rois`, `ViewerState.raw_roi_spectra`, `icon_names`
  and others). The viewer fits and predicts through its jobs, as before.

## 1.7.3 (2026-11-19)

Smaller and faster project files (Blosc), ROIs added from anywhere and named
after their dataset, the white level of the Helios setting, and the rest of
the 1.7.2 backlog. The format stays `enginir-1.7`.

Upgrading from 1.7.2:

- **Projects** open as before. A project saved by 1.7.3 stores its new
  arrays with Blosc and needs engiNIR 1.7.3 or newer (or `hdf5plugin`) to be
  read: update every computer that opens it.
- **Scripts** use `DatasetRecord` and `.record` (was `DatasetRef` and
  `.ref`); there are no aliases.
- **New datasets** without levels get the white level 3100 (was 3500). Do
  not mix them into one calibration set with datasets of the same camera
  added at 3500; set the level in Tools > Preferences or pass
  `white_level=` to keep 3500.

### Changed

- **`DatasetRef` is now `DatasetRecord`, and `.ref` is `.record`**
  (`dataset.record`, `model.record`, `stored_prediction.record`). "Ref"
  read like the dark and white references; it meant the stored record of a
  dataset or model. The old names are gone, with no aliases: scripts must
  use the new ones (e.g. `model.record.model_run_id`). The project file is
  unchanged.
- **Project files store their arrays with Blosc** (Zstandard, level 5,
  shuffle) instead of gzip. On a real tray's ROI pixels: 21 % smaller,
  writing 1.7 times and reading about 5 times as fast. Projects written
  before keep their gzip arrays and open as before; the format stays
  `enginir-1.7`. **A project written by 1.7.3 needs engiNIR 1.7.3 or newer**
  (or `import hdf5plugin` in Python, the Blosc plugin in HDFView or MATLAB):
  earlier engiNIR versions open it but cannot read its arrays.
  `save_hdf5` keeps gzip and offers `compression="blosc"`.
- **The default white level is 3100** (the EVK Helios white balance), not
  3500, and the viewer's Add Dataset starts from the levels set in **Tools >
  Preferences** (128 and 3100 by default). A file that states its levels
  keeps them, and stored datasets keep theirs; but a dataset added from now
  on without levels gets 3100, so do not mix it into one calibration set with
  datasets of the same camera added at 3500. Scripts can still pass
  `white_level=3500`.
- **Add Dataset's detector-level choice reads "Detector calibrated with dark
  and white reference levels"** (it was "Detector levels, no references").
- **The component curve of PLS and PCR ends at 10 components** (15
  before): A_max = max(n, min(10, G − 2, rank)). A fit asking for more
  still gets its count; Auto chooses within the curve. Stored curves keep
  their length.

### Added

- **Add Dataset has a Name field**, and `Project.add_dataset` a `name=`
  parameter: the dataset's short name, shown instead of its file name and
  used by `{name}` in ROI grid names.
- **ROI grids are named after the dataset's name.** The new placeholder
  `{name}` gives the dataset's name (Project > Name Dataset), else its file
  name, and the default patterns use it: Kestreleye scans, all called
  `stream_hsi.h5`, no longer give every grid `stream_hsi-A01`. `{folder}`
  gives the file's folder. The grid dialog lists the placeholders.
- **Add ROI and Add ROI Grid work from anywhere.** Pressed without an
  image shown (from the Settings, the ROI menu, the Checks tab's fix or the
  new buttons of the Project ROIs tab), they show a dataset first, then start
  drawing or open the grid dialog. They were disabled where a project with
  ROIs opens, and the fix only switched the step.
- **The About box cites engiNIR** (with the concept DOI) and links its
  GitHub repository.
- **The domain model page** is a short overview again: each object of a
  project with its term, its Python class and its place in the file.
- **The glossary** gives Frame and Row their own entries and adds the terms
  the viewer and the guides use: the Explore plots (spatial image, sensor
  frame, band image, spatial profile, false-color image), dark and white
  references, detector levels, the wavelength table, the preprocessing
  steps, the figures of merit, scores, loadings, coefficients, VIP, the
  component curve, check and verdict.

### Fixed

- **The Log tab shows the log level of Tools > Preferences** (INFO by
  default) instead of every DEBUG line, and follows a change at once.
- **An image opens as Reset View shows it.** It was fitted with a margin that
  depended on the view's size at that moment (up to 10 % while the page was
  not laid out yet).
- **A Kestreleye scan with a wavelength table is read without the warning
  that it has no wavelengths.**
- **The Checks table's Where and Fix columns have room** (they were fitted
  exactly to their contents) and can be widened by hand.
- **Typing an ROI grid's name pattern no longer logs an error per
  keystroke** (`{fol}` while typing `{folder}`); the grid that is made is
  still checked.

## 1.7.2 (2026-11-19)

Wavelengths for Kestreleye scans, which store none: a wavelength table gives
them, when a scan is added or later. Also a faster project window on Windows
and the welcome window's tagline. The format stays `enginir-1.7`.

### Added

- **Wavelength tables for files without wavelengths.** A Kestreleye
  `stream_hsi.h5` stores band numbers only; the wavelengths of its camera
  setting (EVK-Helios-EC-32, 176 bands, 1041.75 to 1589.5 nm, taken from an
  `.eiff` the vendor software wrote with the same setting) now ship as the
  table "Kestreleye 176 bands". `Project.add_dataset(path,
  wavelength_table=...)` and the **Wavelengths** choice of **Add Dataset...**
  (preselected when one table fits) copy them into the project;
  `Dataset.set_wavelengths` and **Project > Set Wavelengths...** give them to a
  dataset added with band numbers and to its ROIs. Models fitted on band
  numbers predict as before. Your own tables are CSV files in
  `~/.enginir/wavelength_tables/` (`enginir.io.wavelength_tables`). The
  dataset records the table's name and id in two new optional attributes;
  the format stays `enginir-1.7`.

### Fixed

- **The welcome window's tagline** read "Reference correction models for NIR spectra and
  hyperspectral images", a wrong renaming of 1.7.0; it is engiNIR's tagline again,
  "A universal engineering tool for near-infrared spectra and hyperspectral
  imaging".
- **A project window opens up to a second faster on Windows.** The Log
  tab set its fixed-width font when the window was built, which makes Qt
  load the whole Windows font database (0.94 s in a user's log, 0.25 s on a
  warm test machine); the font is now set when the Log tab is first
  selected. It is the system's fixed-width font, since the family
  "Monospace" exists on Linux only.

## 1.7.1 (2026-11-19)

A fast start that never waits for a network drive. A bugfix release: the
project format and the API are unchanged.

### Fixed

- **The welcome screen appears without the scientific libraries.**
  `import enginir` and the launcher load scikit-learn, SciPy, pandas, h5py,
  matplotlib and pyqtgraph only when they are needed; scripts keep
  `from enginir import Project` unchanged. On Linux the welcome window is
  painted after about 0.2 s instead of 1.4 s.
- **The first project opens without loading the libraries:** while the
  welcome screen waits, the viewer loads them in the background.
- **No recent file and no data file is touched on the UI thread.** The
  recent-files list (welcome screen and **Open Recent**) is drawn from the
  settings file alone and grays out missing files when a background check
  ends; the list now stores whether an entry is a project or a dataset.
  Opening a project looks for its data files in the background and lists
  them as "(checking…)" until then, so an unreachable network drive no
  longer freezes the start or the window.
- **The project window is built faster** (0.30 s instead of 0.40 s for an
  empty project on Linux): the Fit settings and the Predict page are built
  when first shown.
- **The plot help setting applies to the Model and Predict plots** also
  when they are first shown after the start (they showed their help
  buttons whatever the setting).
- **The Windows installer compiles the bytecode at install**
  (`uv tool install --compile-bytecode`), so the first start after an
  install or update does not compile about 2,200 modules.

### Added

- **Startup times in the log**, counted from the moment the operating
  system started the process: when the launcher ran, the welcome window was
  shown, the background loading ended and the first main window was shown
  (`Startup timing` lines).

## 1.7.0 (2026-11-19)

The project format `enginir-1.7`: "calibration" now means only model
calibration, the dark/white step is the reference correction everywhere in
the file and the API, and the DModX and T² limits follow SIMCA's default
level 0.95. A new check names prediction ROIs outside the model, and viewer
windows are deleted when they close. Everything of 1.7.0rc2 and 1.7.0rc1
(below) is part of 1.7.0.

Upgrading from 1.5.1 to 1.6.1:

- **Projects** must be converted once with `enginir-migrate SOURCE TARGET`
  (opening one names the command). The conversion writes a new file, keeps
  every dataset, ROI, value, model and id, and adds the 0.95 limits; the
  converted model predicts as before.
- **Scripts** must use the new names (the tables under 1.7.0rc1 and below);
  the old ones are gone, with no aliases.
- **Projects made with 1.7.0rc1 or rc2** must be created again: their
  models lack the 0.95 limits that `enginir-1.7` requires, and they are not
  converted. (A third release candidate was not released; its changes are
  the ones below.)

Changes since 1.7.0rc2:

### Changed

- **The limits follow SIMCA's default significance level, 0.95,** instead
  of 0.99:
  - the DModX exclusion of a fit (`fit_spec` `exclusion/level`, default
    0.95; a model keeps the level it was fitted with);
  - the prediction exclusion default (Preferences, Prediction settings);
  - the limit beyond which a pixel is outside the model (no prediction
    interval; counted outside in the summaries), including the ROI-level
    T² limit of the mean interval;
  - the per-ROI shares the fit stores (`roi_share_above_dcrit`,
    `roi_share_above_t2crit`), the Model page's Distance plots and texts.

  About 5 % of normal pixels now lie above DCrit (1 % before).
- Format `enginir-1.7`: `prediction_model/@dcrit_095` and `@t2_crit_095`
  are the core limits checked at open; the 0.99 pair is kept as display
  data.
- Public keys renamed (no aliases): `Prediction.diagnostics` and the
  `predict_rois` result use `dcrit_095` and `t2_crit_095` instead of
  `dcrit_099` and `t2_crit_099`.
- The `model.distance` check asks for attention above three times the
  expected share, 15 % at 0.95 (5 % before, at 0.99).
- `enginir-migrate` adds the 0.95 limits to converted models (computed from
  the stored number of pixels, bands and components) and keeps their 0.99
  per-ROI shares as `roi_share_above_*_099`; a converted model gets the new
  check and the Predict ring after a refit.
- The Predict page's ring now comes from the library
  (`enginir.prediction.unusual_rois`), per model; it compared every model
  against the first model's calibration shares before.

### Added

- `predict.rois_outside`: the check names the prediction ROIs outside the
  newest model (more pixels above DCrit than 90 % of its calibration ROIs),
  with **Go to ROI**; with none outside, the Predict step says that all lie
  within the model. `Project.checks(predict=False)` skips the prediction it
  needs; the predictions are cached. In the viewer it arrives from a
  background job after the other checks.
- `enginir.prediction.unusual_rois` and `enginir.validation.LIMIT_LEVEL`.
- The example project opens with one open check: its prediction tablet
  P_T4 lies outside the model.

## 1.7.0rc2 (2026-11-19)

The second release candidate of 1.7.0: everything of 1.7.0rc1 (below), and
viewer windows are deleted when they close.

### Fixed

- A closed viewer window is deleted. A window opened by **Open Data...** used
  to keep its data in memory after it was closed, until the window that
  opened it closed too; now its memory is released when it closes.
- A closed window no longer reloads its project after every change in
  another window on the same project.
- Closed windows are no longer freed at random moments, which could crash
  the viewer while it drew another window. A background job that finishes
  after its window closed calls nothing.
- The welcome window is deleted after it hands over to the first viewer
  window.

## 1.7.0rc1 (2026-11-18)

The project format changes to `enginir-1.7`: in the file and the Python API,
"calibration" no longer names the dark/white reference correction, only
model calibration. **Projects of engiNIR 1.5.1 to 1.6.1 (format
`enginir-1.5`) must be converted once** with `enginir-migrate SOURCE
TARGET`. Opening one names the command. The conversion writes a new file,
keeps every dataset, ROI, value, model and id, and the converted model
predicts exactly as before. **Scripts must use the new names below;** the
old ones are gone (no aliases).

### Changed

- Project format `enginir-1.7` (`FileSchemaVersion`). Stored names:

  | `enginir-1.5` | `enginir-1.7` |
  |---|---|
  | `datasets/{id}/calibrations/` | `datasets/{id}/reference_corrections/` |
  | `@active_calibration_id` | `@active_reference_correction_id` |
  | `@calibration_id` (ROIs, prediction runs) | `@reference_correction_id` |
  | `rois/{id}/calibration_refs/` | `rois/{id}/reference_spectra/` |
  | `@required_calibration_mode` | `@required_correction_mode` |
  | `@calibrated_input_allowed` | `@corrected_input_allowed` |
  | mode `precalibrated_import` | `already_corrected` |

- Python API (`enginir`, `enginir.project`, `enginir.io`,
  `enginir.prediction`):

  | Before | Now |
  |---|---|
  | `CalibrationSpec` | `ReferenceCorrection` |
  | `CalibrationStats` | `ReferenceCorrectionStats` |
  | `CALIBRATION_MODES` | `CORRECTION_MODES` |
  | `calibration_id`, `active_calibration_id(s)` (fields, keywords) | `reference_correction_id`, `active_reference_correction_id(s)` |
  | `calibration=` (keyword of the prediction and conversion functions) | `reference_correction=` |
  | `Dataset.calibration` | `Dataset.reference_correction` |
  | `Project.save_calibration` | `Project.save_reference_correction` |
  | `workspace.calibrations` | `workspace.reference_corrections` |
  | `make_calibration`, `resolve_calibration`, `resolve_prediction_calibration` | `make_reference_correction`, `resolve_reference_correction`, `resolve_prediction_reference_correction` |
  | `check_calibration_signal_domain_compatibility` | `check_reference_correction_compatibility` |
  | `save_calibration_spec`, `iter_roi_calibration_refs` (`enginir.io.project_store`) | `save_reference_correction`, `iter_roi_reference_spectra` |
  | `SignalDomainContract.required_calibration_mode`, `.calibrated_input_allowed` | `.required_correction_mode`, `.corrected_input_allowed` |
  | mode value `"precalibrated_import"` | `"already_corrected"` |
  | progress phase `"calibrating"` | `"correcting"` |
  | prediction provenance `calibration_id`, `calibration_mode` | `reference_correction_id`, `correction_mode` |

- Library messages say "reference correction" where they meant it
  (e.g. "Reference correction '…' has no dark/white reference").
- The contract `dataset-calibration-contract_v3.md` is renamed
  `dataset-reference-correction-contract_v3.md`.

### Added

- `enginir-migrate SOURCE TARGET` and `enginir.io.migrate_project`: convert
  an `enginir-1.5` project into a new `enginir-1.7` file (user guide, I/O,
  "Projects from engiNIR 1.5 and 1.6"). The source is never changed; the
  result is read with the strict reader before it is kept.
- A test that no stored name and no public name says "calibration" for the
  reference correction (gap G36, closed).

### Fixed

- Opening a project checks that each stored prediction names the
  preprocessing recipe and input contract of its model; a prediction run
  naming another one is refused, as the project-schema contract now says.

## 1.6.1 (2026-11-18)

A bugfix release. It fixes seven findings of an external review of 1.6.0
(among them a replaced data file predicted into a wrong map, a lost DModX
check for small models and wavenumber axes shown as nm), brings the
contracts in line with the 1.6 viewer, and adds a Regulatory context page
to the documentation. The test suite is reorganized and CI now also runs
on macOS. Projects open as they are; a dataset added with a wavenumber
axis before 1.6.1 is stored with those values and must be added again to
be predicted.

### Changed

- The contracts describe the 1.6 window: the workflow strip as the
  navigation, the Fit settings and **Load Model Settings** instead of the
  fit dialog and **Fit Again...**, the Settings dock (Prediction settings
  on Predict), the project browser's groups and `Properties`, the Model
  workspace's regions and fit summary, and Tools > ROI Grid Defaults. Every
  contract says it is active for engiNIR 1.5 and 1.6; the project schema
  names the stored-model `algorithm` default as its one exception.
- The fit summary names the model's signal and preprocessing and the number
  of datasets of its ROIs; the hidden Readiness box is gone (the Checks tab
  and the Fit settings say what blocks a fit).
- The Explore dock "RAW" is now **Band image** ("Band image, band 32
  (1352.4 nm), raw counts (DN)").
- The ROI spectra export is **Export ROI Spectra (CSV)...** on its button
  and in the File menu, as in the ROI menu.
- A recipe without steps reads "no preprocessing" in the Predict chain and
  status line (was "none").
- The README follows the 1.6 viewer, with new screenshots of the example
  project; its guarantees are two guiding principles, and it shows how to
  install the viewer on macOS and Linux.

- The test suite is organized by topic: 44 files instead of 79, with the same
  line coverage. The files named after the work that produced them
  (`test_polish.py`, `test_review_fixes.py`) are split up by topic; project
  objects, synthetic scans, EIFF files and Qt waits come from shared builders
  (`tests/_project_payloads.py`, `tests/_scans.py`, `tests/_eiff.py`,
  `tests/_qt.py`) and `qapp`, `tray` and the example window from
  `conftest.py`, so no test module imports another. Tests that only checked
  stored attributes, an end-to-end test whose data file is not in the
  repository, and duplicates are gone; the timing benchmarks moved to
  `benchmarks/`. Every test now runs with HOME in a temporary folder, so no
  test can write the user's real settings. After each test the windows it
  opened are destroyed at once; before, Python's garbage collector freed
  them at random moments, which crashed CI runs now and then while Qt
  painted another window. CI also runs the tests on macOS.

### Added

- Documentation: **Regulatory context**, what engiNIR guarantees and which
  evidence it supplies for ICH Q2(R2), Q12 and Q14, USP <1039> and <1119>
  and the FDA guidance, and what it does not cover (21 CFR Part 11, system
  validation, a validation report).

### Fixed

Found by an external review of 1.6.0:

- `Model.predict` on a registered path trusted the stored shape and axis: a
  replaced file of another shape was predicted into a wrong map. The file is
  now compared with its registration (format, shape, modality, wavelength
  axis) and refused on a mismatch; the runtime also refuses blocks that do
  not fit the dataset's shape.
- Models with 16 or fewer ROIs lost the DModX check (their per-ROI shares are
  stored as HDF5 attributes), so their verdict could read Good with many
  pixels above the limit.
- `Project.save_calibration` accepted an id of another dataset's reference
  correction (the project then did not open) and replaced a stored reference
  correction with different content; both are refused now.
- A project whose model lacks a core member (coefficients, intercept,
  wavelengths, target names, projection data) opened and failed only at
  prediction; it is refused at open, as the schema contract says.
- A wavenumber axis (JCAMP, SPA, SPC) entered a project as nm values
  (5000 cm⁻¹ shown as 5000 nm); wavenumber and micrometer axes are
  converted to nm when a dataset is added and when the viewer shows a file.
- Predict Map in the viewer loaded the whole cube before the chunked
  prediction, and a new ENVI file was read whole for its header; the map is
  streamed now and ENVI headers are read without the pixels.
- With `cube.img` and `cube.raw` next to one header, prediction provenance
  (and its checksum) named `.raw` while the reader used `.img`.
- A project opened without a dataset showed its ROIs in raw counts (DN);
  Explore now starts with the working signal of the ROIs' reference
  corrections (Absorbance for dark/white references), as with a dataset.

## 1.6.0 (2026-11-17)

The guided viewer: one window for every step with a workflow strip, checks
and verdicts that say what a project needs next, fit settings in a dock, a
light theme and preferences. Everything of 1.6.0rc3 (below) is part of
1.6.0; the library API is unchanged apart from the additions listed there.
Upgrading from 1.5.x needs nothing: projects open as they are.

Changes since 1.6.0rc3:

### Changed

- **Example project:** it opens without open checks. Each calibration scan
  holds tablets with different numbers of layers (a scan is not a value),
  each tablet is its own sample, a binder content varies from tablet to
  tablet, and the model "PLS 2 LV" (no SNV: the synthetic spectra have no
  scatter) is rated good. An example project of 1.6.0rc3 is created again
  when it is opened.
- **Project tree:** items show names, sorted (ROIs by dataset, then name:
  "T2" before "T10"), instead of names with ids; ids are in the tooltip and
  in Properties. Reference corrections show their dataset and what they are
  ("tablets_1.h5: dark/white references").
- **Dataset page:** the Sensor Frame and the Spatial Profile moved from
  the right-hand docks, where they narrowed the settings, into the page
  beside the Spatial image. The three share the scan's rows (zoom, pan and
  the selected row line up); the Spatial Profile is upright, and the Sensor
  Frame shows wavelengths when the spectral axis is nm. View > Sensor Frame
  and Spatial Profile show or hide them.
- **Predicted vs target value:** the statistics sit in the corner (top left
  or bottom right) where they cover no point, on a background; the axis
  grows when no corner is free.

### Fixed

- **"Above white" warning:** the share of values above white counted the
  frames the white reference was taken from, half of whose values lie above
  their own mean, so every scan with embedded references warned (1.2 % for
  two white frames of 80). Those frames are left out now.
- **Leaving Values by the Model or Predict tab:** the tall ROI table stayed
  in front (only the workflow strip put it back). Now Model brings Models to
  the front and Predict brings Checks, each with the page's height.
- **Preferences > Data and paths:** the Projects folder was not used, and
  the welcome page's dialogs ignored the Datasets folder. Every project and
  dataset file dialog now starts in its folder (unless it is missing). The
  page shows the drive mapping applied to stored paths instead of "nothing
  to set here".

## 1.6.0rc3 (2026-11-16)

Release candidate of the guided viewer: one window for every step with a
workflow strip, checks and verdicts that say what a project needs next,
fit settings in a dock, a light theme and preferences. For trying before
1.6.0; the library API is unchanged apart from the additions below.

### Added

- **Checks and verdicts.** `Project.checks()` lists, per step of the
  workflow, what is done and what needs attention: dataset files that are
  not found, ROIs without a value, a value that differs within its sample,
  an optimistic cross-validation, a model whose ROIs or values changed
  since the fit, calibration pixels far from the model. Each check names
  what it refers to and, where there is one, the fix. `Model.verdict()`
  gives a model one level (good, usable with care, not usable) with its
  reasons. Both read the project and change nothing.
- **A guided viewer.** The window is a desktop program in one layout for
  every step: menus hold every action (no toolbar); the **workflow strip**
  Data → ROIs → Values → Model → Predict shows the order, where you are
  and what is done; open checks show as a number in a round badge. The
  Model page's fit summary starts with the verdict of the shown model;
  the bottom panel has **Checks** (every check with its fix; passed checks
  fold into one row),
  the **ROI table**, **Models** (figures of merit and verdict per model)
  and the **Log**. Built for 1920 × 1080, complete at 1536 × 864.
- **Fit settings in a dock.** Every fit option sits in the Settings dock on
  the right instead of a dialog: **Fit Model** (F5) fits with what it
  shows and stays in view under the settings. The settings are in parts
  that fold to a one-line summary (Data, Preprocessing, Bands, Model,
  Validation, Fitting), show the selected model's settings until they are
  edited, and a line says when they changed; **Load Model Settings** shows
  the model's settings again. View and Prediction settings sit in the same
  place for their steps.
- **Said once.** A model's fold warning is in the Checks, not also in the
  fit summary and Properties; the figures of merit are on the Model page,
  not also in Properties; the View settings have no Project buttons (File
  menu). No bar above the pages repeats the first check.
- **ROI table:** the bottom panel takes about 40 % of the window while the
  ROI table is in front, and its rows are compact.
- **Example project:** five synthetic tablet scans with a fitted model,
  created on first use (welcome screen, File > Open Example Project).
- **Preferences** (Tools > Preferences): theme including follow the system,
  interface scale for small screens, plot help, start folders,
  defaults of the Fit and Prediction settings, export resolution, threads
  of a fit, log file level, the keyboard shortcuts.
- **Dataset names:** Project > Name Dataset... gives a dataset a short name
  shown instead of its file name (`Dataset.set_label`; stored as display
  data, projects without it open as before).
- Empty pages invite the next action; values above the white reference
  are counted and shown under Signal; plot titles that do not fit end in
  an ellipsis with the whole title as tooltip.

### Changed

- **One number format:** figures of merit (RMSECV, RMSEE, bias, R², Q²,
  RPD) have four significant digits everywhere: the verdict sentence of
  `Model.verdict()` (before: RMSECV with three, Q² and RPD with two
  decimals), the Models table, the fit summary and the plots.
- A dimensionless target unit (`1`) is no longer printed in the viewer
  ("Layers", not "Layers (1)").
- **Status strip:** it names what the step works on; frame, row and band
  only on the Dataset page, the model in the Model step.
- **Viewer details:** the project tree counts its objects and has icons;
  Properties is a table; ROI names on the image have a plate behind them;
  an empty plot shows its message without axes; plot headers use line
  icons; the ROI mosaic puts its blocks beside each other in a wide view
  and no longer cuts the labels of narrow tiles; the workflow strip names
  the project and the model shown.
- **New viewer palette, dark and light.** All colours come from one table
  of named colours: the orange of the engiNIR icon as the accent (current
  tab, default button, selection, image crosshair, ROI outlines), cool
  colours for data, viridis for values and a heat scale from the icon's
  bands for distances and uncertainty. Square, compact controls. Text meets
  WCAG AA contrast in both themes (tested). Predictions outside the
  calibration range are violet (below) and pink (above).

### Fixed

- **Closing the viewer while it loads:** a background job still reading the
  project file (ROI images, spectra) when the program ended could crash it
  or make it hang. engiNIR now lets running jobs finish, for up to 10
  seconds, before it exits.
- **Model step after opening a project:** a project whose models of one
  target use different preprocessing could show no model until one was
  picked in the project tree. The Model step now shows the selected model,
  else the active one, else the latest fitted, of the selected target.
- **Project ROIs, mean spectra:** after the shown bands changed (a band
  selection in Explore, or a fit with bands), the plot could pair the new
  band axis with spectra still computed for the old one and log "Plot
  table columns differ in length". It now uses the axis of its spectra.
- Files without wavelengths no longer offer or show "nm": the spectral axis
  is the band number.
- Opening a dataset from the Project tree shows its image; Predict names the
  prediction ROIs waiting and the model a prediction would use, instead of
  "none".
- A refit reported values as changed that had not changed, when they are
  not exact in 32-bit precision (for example 12/7): the comparison with the
  values the model was fitted with now allows for the stored precision.

## 1.5.2 (2026-11-13)

The viewer on Windows for users without Python knowledge: one PowerShell
line installs it, and it starts from the Start menu like any desktop
program. CI now tests every change on Windows, including the install on a
clean machine. Also a fix of the 1.5.1 kernel engine (NaN Y scores).

### Added

- **The viewer on Windows like a desktop program.** `enginir-viewer` is now a
  GUI script: on Windows it starts without a console window, and its
  messages (`--help`, the new `--version`, a failed start) appear in a
  message box. `enginir-viewer-console` is the same launcher with a console.
  `enginir-viewer --install-shortcut [--desktop]` adds "engiNIR Viewer" with
  the engiNIR icon to the Start menu (and the desktop), with the viewer's
  taskbar id, so a pinned window starts engiNIR; `--remove-shortcut` removes
  them.
- **One-line install on Windows** for users without Python knowledge:
  `install-enginir-for-windows.ps1`, attached to each release, installs uv
  if needed, installs engiNIR as a uv tool on its own Python 3.13, puts it
  on the `PATH` and adds the Start menu shortcut; no administrator rights.
  The installation guide starts with it.
- CI runs the tests on Windows as well.

### Changed

- On Windows, `enginir-viewer` typed in a terminal returns at once and
  prints nothing (it is a GUI program now); use `enginir-viewer-console` to
  see messages in the terminal. Linux and macOS are unchanged.

### Fixed

- Kernel PLS gave NaN Y scores (and a `RuntimeWarning`) for components the
  data no longer support; they are zero, as with NIPALS (1.5.1 regression).
- The default log file falls back to the system temporary folder when the
  home folder is not writable (it was a hard-coded `/tmp`, `C:\tmp` on
  Windows).
- Logging works in a process without a console (`sys.stderr` is `None`):
  the log file records everything, nothing is printed.

## 1.5.1 (2026-11-13)

A minimum, strict project format, the bug fixes of the 1.5.x review and
demo project, projects that move between Windows and Linux, and a kernel
engine that fits PLS and PCR about ten times faster with a quarter of the
memory. **Projects written by 1.5.0 do not open** (create them again): see
Breaking. Also: the terminology contract applied where 1.5.0 missed it,
and the contracts read against the code.

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
- `PLSRegression` and `PCR` no longer take `scale`; scaling is `x_scaling`
  (and `y_scaling` for PLS) only. Their defaults are now X centered and Y
  unit variance (as for model runs); before, `scale=True` autoscaled X too.
- A stored model must have its fit specification; models fitted before 1.2
  are not read. GroupKFold remains an optional fold rule. The legacy reading
  rules (`eiff_raw`, `band_mapping_policy = reject`, absent noise guard) are
  removed, and a request that names `components` without `noise_guard`
  now gets the default (on with `"auto"`).
- Undefined values are written as NaN, empty strings or empty arrays instead
  of being left out (e.g. RMSEE (per ROI) when R - 1 - A < 1; an ROI without a
  sample label; `artifact_refs/exclusion` without DModX exclusion). The viewer
  shows a missing value as "not available" instead of asking to refit.
- Every model stores the software it was fitted with (`software`: engiNIR,
  numpy, scipy, scikit-learn), shown in the Inspector; a refit whose metrics
  differ names the versions that differ.
- Stored predictions (low-level writers only) are staged under `/staging/`
  and moved to `/predictions` when finalized: a crash during a write leaves
  a project that opens, and `enginir.io.discard_staged_predictions` removes
  the leftover. `artifact_write_status` and the second `spatial_map` array
  are gone; `initialize_prediction_run_artifacts` no longer takes
  `spatial_map_shape`, and `PredictionArtifactWriter.write_chunk` only
  `row_start` and `y_pred_chunk`.
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

### Added

- Kernel engine for PLS and PCR (plan `plans/kernel-pls.md`): fits on sums
  (XᵀX per fold group, built in one chunked pass), kernel PLS (Dayal and
  MacGregor) and PCA from XᵀX. A fit in a project with a recipe without
  MSC/EMSC computes the cross-validation, the component curve and the final
  model from the sums, and one more pass the per-pixel diagnostics; the band
  proposal works on the same sums. On 396 000 pixels × 176 bands (SNV, 7
  folds, curve to 15) a fit took 43 s and 1.7 GB with NIPALS and takes 4.5 s
  and 0.45 GB. Results equal NIPALS and the full SVD within floating-point
  precision (tested to 1e-8 on the model and its diagnostics).
- `hyperparameters/algorithm`: `kernel` or `nipals` (PLS), `gram` or `svd`
  (PCR, chosen by a conditioning guard). Refits and Fit Again use the stored
  algorithm; a 1.5 model without one reads `nipals` or `svd`
  (`stored_hyperparameters`). The Inspector shows it. NIPALS and the full
  SVD are kept until 2.0.
- `ENGINIR_BLAS_THREADS` sets the BLAS threads of the engine's passes (by
  default half the logical CPUs: OpenBLAS with every thread can collapse on
  these products when other work shares the machine).

- A project moves between computers and platforms: a dataset's data file
  is looked for next to the project (new dataset field `relative_path`), at
  its stored path, and through the Windows ↔ WSL drive mapping (`Z:\…` ↔
  `/mnt/z/…`, root from `/etc/wsl.conf`). A file found elsewhere is checked
  (format, shape, wavelengths, the pixels of the smallest ROI) and the
  project is not changed; the viewer reads it from there
  (`Project.find_data_file`, `Project.data_file`). After one **Locate
  file...**, the viewer offers the same folder change for the other missing
  files.
- Stored paths are read by their own syntax: a Windows path read on Linux
  no longer turns into a path inside the current folder.

### Fixed

- Windows: the taskbar shows the engiNIR icon; the viewer sets its own app id
  instead of being grouped under python.exe.
- A project of an earlier format (for example `viewer-v3`) is refused with the
  message naming its format, in the viewer and in `load_spectra`; it was read
  as a dataset and failed with "has no dataset 'spectra'".
- A project that cannot be created (for example in a folder you cannot write
  to) is reported in a dialog with the reason; the viewer logged a traceback
  and showed nothing.
- Refusals for a target without values name the target, not its id
  (`plan_sample_set` and `resolve_sample_set` take an optional `target_name`).
- A canceled band proposal or prediction is logged as info; the log said the
  job failed and showed a traceback.

- Wording: the fit dialog's groups are "Smoothing and derivatives" and
  "Scatter correction" (were "Filter", "Normalization", "Norm"); "fitted"
  replaces "trained"; DModX exclusion says "pixels above DCrit"; the Explore
  and Project ROIs axes say "Raw counts (DN)"; PCR plots label their
  components PC1, PC2, … (the score and loading plot titles name
  "components"); the welcome screen says "engiNIR"; the fold warning says
  "dataset".
- The dataset selector showed "Dataset: none" while a file opened without a
  project was shown; it names the file.
- The notes of the Distance histograms overlapped in narrow panels; they are
  one block inside the plot, and an undefined limit is not drawn.
- The terminology test also scans library messages.
- **Fit Again** preset the signal unit shown in Explore instead of the
  model's, so OK could fit a different model; it now repeats the model's.
- The pixel-subsample seed field took only 32-bit values; a stored seed
  above 2³¹ − 1 was cut down silently and a refit drew other pixels.
- The ROI table lost its ROI names off the left edge after Predict ROIs
  added result columns: the Dataset column now shows the data file's name
  without its folder (full label as tooltip), so it stays narrow and the ROI
  names next to it stay in view; rows stay grouped by data file.
- The Model summary kept the previous model's fold warnings when the next
  model's metrics looked the same, and cut off long values such as the
  validation scheme; it now updates the warnings and wraps.
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
- `PLSRegression` fits by kernel PLS and `PCR` by PCA from XᵀX by default
  (`algorithm="nipals"` or `"svd"` for the earlier fits). The models agree
  to floating-point precision, not bit for bit.
- `fit_model_run`: with a recipe without MSC/EMSC, the result's
  `sample_set_data.X` has no columns (the raw spectra are released after the
  reference correction and stay in the project file; `pixel_indices` tells
  the drawn pixels), and the recipe is applied ROI by ROI.

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
