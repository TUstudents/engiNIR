# Headless Workflow Contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.0; v3 is the default and only supported project schema
- Scope: the Qt-free public API surface for project-backed scientific
  workflows — creation, persistence, hydration, sample building, model
  fitting, and prediction from scripts and Jupyter notebooks

## Goal

engiNIR is a library first. The viewer is one client of the scientific
services, not their owner. This contract freezes the boundary that makes the
complete v3 workflow — create project, register data, persist ROIs, define
targets, build sample sets, fit models, predict — usable without any Qt
dependency, so notebook and script workflows are first-class and cannot be
broken by viewer refactors.

## Problem This Contract Solves

Without a headless contract, the system cannot answer reliably:

- which operations must work without a running QApplication
- whether scientific validity rules live in reusable services or only inside
  viewer widgets and state objects
- whether progress and cancellation callbacks require Qt types
- whether a project written from a notebook is identical to one written from
  the viewer

The failure mode is drift: workflow logic accumulating in Qt-side classes
until the library can no longer run its own scientific workflow headless.

## Core Decisions

- the packages `enginir.io`, `enginir.project`, `enginir.modeling`,
  `enginir.prediction`, `enginir.preprocessing`, `enginir.models`, and
  `enginir.validation` must not import Qt, directly or transitively
- `enginir.viewer` is the only Qt boundary; importing the top-level `enginir`
  package must not pull in Qt
- every scientific validity rule must be enforceable without Qt: typed model
  validation, relationship validation, and service preflights are the
  authorities; viewer widgets may only add convenience and presentation
- a project file written headless and a project file written through the
  viewer are the same v3 project; there are no viewer-only persisted fields
- service callbacks (progress, cancellation) use plain Python callables and
  value objects, never Qt signal/slot types

## The Headless Surface

The v3 headless workflow composes a project facade over four layers. Function
lists name the current public entry points; additions are allowed, removals or
Qt-contamination are contract violations.

### 0. Project facade (`enginir.Project`)

`Project` is the primary user entry point. It runs the complete workflow with
the objects a scientist works with — datasets, ROIs, targets, models,
predictions — and creates the bookkeeping objects (calibrations, sample sets,
signal domains, recipes as stored objects, provenance) itself. Its rules are
defined in [Project Facade Rules](#project-facade-rules). The layers below
remain public for advanced use; the facade adds no rule they do not enforce
and writes nothing they would reject.

### 1. Persistence (`enginir.io`)

- `create_project` initializes an empty v3 workspace
- `load_project_store` is the canonical metadata-first reader
- object writers persist first-class objects (`save_dataset_ref`,
  `save_calibration_spec`, `save_training_roi`, `save_roi_sample`,
  `save_target_definition`,
  `save_target_assignment`, `save_preprocessing_recipe`,
  `save_signal_domain_contract`, `save_sample_set`, `save_model_run`,
  `save_prediction_run`)
- lazy loaders hydrate heavy arrays on demand (`load_roi_spectra`,
  `load_roi_spectra_batch`, `load_roi_spectra_matrix`,
  `iter_roi_calibration_refs`, `load_model_run_artifact_refs`,
  `load_model_run_prediction_artifact`, `load_prediction_run_artifact_refs`,
  `load_prediction_spatial_map_preview`)
- staged prediction artifact writes (`initialize_prediction_run_artifacts`,
  `open_prediction_run_artifact_writer`,
  `finalize_prediction_run_artifacts`, `discard_prediction_run_artifacts`)
- spectral file loading (`load_spectra` and the per-format loaders)

### 2. Typed domain (`enginir.project`)

- the typed first-class objects and their validating constructors
- `hydrate_project_store` builds a validated `ProjectWorkspace` from a raw
  store
- `validate_project_relationships` enforces the id graph
- the signal-domain checks (`check_signal_domain_compatibility`,
  `check_calibration_signal_domain_compatibility`) and
  `transform_spectra_to_signal_domain`, which converts raw spectra under a
  calibration context identically for training and prediction

### 3. Modeling service (`enginir.modeling`)

- `build_sample_set` constructs labeled pixel/ROI arrays under the frozen v3
  semantics
- `fit_model_run` fits, cross-validates ROI-grouped, aggregates, and returns
  the persisted-shape `ModelRun` plus in-memory results
- `aggregate_sample_set_predictions` exposes the ROI aggregation used for
  primary metrics

### 4. Prediction service (`enginir.prediction`)

- `preflight_prediction` runs the compatibility checks and resolves runtime
  state
- `predict_with_model_run` executes the chunked runtime contract
- `build_dataset_ref_from_spectral_data` creates prediction dataset identity

## Project Facade Rules

### Entry points

- `Project.create(path)` creates an empty project; `Project.open(path)` opens
  an existing one and validates it completely (`hydrate_project_store`)
- `project.add_dataset(path, *, dark=None, white=None, unit="Absorbance")`
  registers a dataset with its calibration and returns a dataset handle
- `project.add_target(name, *, unit)` defines a target
- `dataset.add_roi(name, *, frames, rows, values=None)` stores an ROI cut from
  the dataset, with its reference values (`{target name: value}`)
- `project.fit(target, *, recipe, model, name=None, cv=None, fit_spec=None,
  **hyperparameters)` builds the sample set, fits and cross-validates, and
  returns a model handle. `fit_spec` (a `FitSpec` or a mapping with the keys
  of `sample-set-model-run-contract_v3.md` §7) gives full control; keys it
  leaves out take their defaults. `cv` is the short form of the fold rule
  (`"auto"`, an integer fold count, `"logo"`; `None` means `"auto"` unless a
  `fit_spec` gives the folds) and is refused together with a `fit_spec`
  that sets the folds. The complete specification is stored with
  the run
- `model.refit(*, name=None)` fits the model again on its stored sample set,
  recipe, signal domain, hyperparameters and fit specification and stores
  the new run (`sample-set-model-run-contract_v3.md`, Refitting a stored model)
- `project.fit(..., n_components="auto")` chooses the number of components
  from the component curve with `fit_spec={"components": {"rule": ...,
  "max": ...}}` (default rule `one_se`); every PLS/PCR model has the curve
  (`model.component_curve`: a dict of arrays and the suggestions, or `None`
  for a model fitted before 1.3) and the training diagnostics
  (`sample-set-model-run-contract_v3.md` §7.4, §7.5)
- `project.set_sample(roi, label)` sets the sample label of an ROI (name or
  object); `None` or an empty or blank text removes it
  (`roi-target-contract_v3.md`). The typed, id-based form is
  `project.save_roi_sample(roi_id, label)`; the viewer's `ProjectState`
  writes labels only through it. `project.fit(..., fit_spec={"folds":
  {"group_by": "sample"}})` holds the ROIs of one sample out together
  (`sample-set-model-run-contract_v3.md` §4.1); `model.refit()` uses the
  groups stored with the run and warns when current labels or values would
  group differently
- `model.predict(path, *, dark=None, white=None, checksum=False)` predicts a
  data file and returns an in-memory `Prediction` (`enginir._workflow.Prediction`: `map`, `y_pred`,
  `dmodx_map`, `t2_map`, `diagnostics` with limits and summary,
  `provenance`, and `uncertainty_map(level=0.95)`, the pixel prediction
  interval's half-width as an image, NaN outside the model, `None` without
  the interval inputs; `prediction-runtime-contract_v3.md`, Prediction
  Intervals); it writes nothing to the project (the file is not
  registered, the prediction not stored) and logs one entry
  (`prediction-runtime-contract_v3.md`, Result Handling). Persisting a
  prediction is a low-level operation (`enginir.io.project_store`)
- `model.predict_rois(rois=None, *, exclude_dmodx=None)` predicts the
  prediction ROIs (all, or the given ones) from the project file alone and
  returns per-ROI results (`prediction-runtime-contract_v3.md`, Prediction
  ROIs), each with its mean interval's half-width at 0.95 (`interval`);
  `exclude_dmodx` a level, `False` for none, `None` the model's own
  setting. `roi_prediction_statistics(...)` recalculates them for another
  exclusion without predicting again. `project.set_use(roi, use)` marks an
  ROI; `scan.add_roi(..., use="prediction")` creates one
- read access: `project.datasets`, `project.rois`, `project.targets`,
  `project.models`, `project.predictions`, keyed by name (datasets by path,
  predictions by id; `project.predictions` lists only predictions persisted
  by low-level writers, as `StoredPrediction` handles). When names of a kind repeat, its name-keyed mapping is
  an error listing the repeated names and their ids instead of silently
  keeping one; `project.workspace` holds every collection keyed by id;
  handles expose their typed object (`.ref`) and the essentials (`model.metrics`,
  `model.diagnostics`)

### Band proposal

- `project.propose_bands(target, *, recipe=None, model="PLS",
  n_intervals=20, max_steps=None, fit_spec=None, progress=None,
  cancel=None)` proposes band ranges by forward interval PLS (iPLS; PCR
  with `model="PCR"`) and returns a `BandProposal` (`ranges`, the RMSECV
  after each step with its standard error, `nested_rmsecv`,
  `all_bands_rmsecv`, the intervals and the chosen number of components per
  step). It writes nothing to the project and logs one entry
- the search works on the recipe's axis after Savitzky-Golay (a band
  selection already in `recipe` is ignored), split into `n_intervals`
  contiguous intervals as `numpy.array_split` of its band indices (the
  first K mod n intervals get one band more); 2 ≤ `n_intervals` ≤ K, and
  `max_steps=None` means `n_intervals`
- a candidate is the recipe with `bands_keep` = the chosen intervals plus
  one more; preprocessing runs on the whole spectrum after Savitzky-Golay,
  as in a fit, so stateful preprocessing is fitted once per fold and
  reused for every candidate. Its number of components is chosen by the
  component rule of `fit_spec` (default `one_se`) with
  Amax = min(the `fit_spec` maximum, or the default of
  `sample-set-model-run-contract_v3.md` §7.4 with the fold groups and rank
  limit of the current training part, the candidate's band count); a
  `fixed` number is capped at that Amax with a warning. The RMSECV and SE
  of a candidate are the §7.4 values at that number
- forward: step 0 is the mean predictor (the training-fold mean of y,
  its RMSECV and SE by §7.4); each step adds the interval whose candidate
  has the lowest RMSECV (ties: the lower interval index). The search stops
  when that RMSECV is not lower than the current RMSECV minus the current
  SE (the one-SE idea), when `max_steps` intervals are chosen, or when
  every interval is chosen. Each proposed range is [λ of its first band,
  λ of its last band] of a run of adjacent chosen intervals
- the RMSECV of every candidate uses the run's fold scheme (`fit_spec`
  folds and fold groups, ROI-grouped; `sample-set-model-run-contract_v3.md`
  §4.1) on ROI means, with stateful preprocessing fitted inside the folds,
  like a fit; the pixel sample of `fit_spec` is applied once before the
  search; no DModX exclusion is used in the search, the outer refits or
  `all_bands_rmsecv`, so `all_bands_rmsecv` can differ from a run's RMSECV.
  `propose_bands` refuses fewer than 3 fold groups
- nested validation: for each outer fold of that scheme, the whole search
  runs on the outer training ROIs only, with inner folds built from them
  by the same rule (k = min(F, inner fold groups)); the chosen ranges are
  fitted on those ROIs with the number of components of the inner
  search's last step and predict the held-out ROIs; `nested_rmsecv` is the
  RMSECV of these held-out predictions. It estimates the error of the
  whole procedure (selection included) and is reported next to
  `all_bands_rmsecv`, the RMSECV of the recipe without a band selection.
  The proposed `ranges` come from the search on all ROIs
- the proposal only proposes: a fit with the ranges is an ordinary fit
  with `bands_keep`, and a refit never searches again
- cost: about `n_intervals` × steps × (outer folds + 1) × inner folds × the
  component curve's fits; `progress(done, total)` reports candidate
  evaluations and `cancel()` returning true stops with a cancellation error

### Names and ids

- every object has a generated stable id; ids are its only identity and what
  the project file links by (`project-schema-contract_v3.md` §2–3). Names
  are display labels: mutable, and they may repeat within a project, for
  example ROIs from two data files with the same file name
- addressing objects by name is a convenience of this facade (`Project`,
  `Dataset`, `Model`) for scripts and interactive work. It is not a
  reproducible reference: a renamed or reused name refers to another object
- every facade function that takes a name also takes the object itself
  (`TrainingROI`, `TargetDefinition`, `Model`) or its id as a keyword
  (`roi_id=`, `target_id=`, `model_run_id=`). A string argument is always a
  name and an id keyword always an id, so neither is taken for the other.
  Scripts that must be reproducible use objects or ids
- looking up a name that several objects share is an error that lists their
  ids; it never picks one
- the facade refuses a name it is asked to create when that name already
  exists for that kind (`Dataset.add_roi`, `Dataset.add_rois`,
  `Project.add_target`, `Project.fit(name=)`), so a script can find what it
  created by name. The typed, id-based layer (`Project.save_*`) and the
  viewer do not check names (`viewer-workspace-ui-contract_v3.md`,
  Identity in the viewer)
- the name-based calls released in 1.0 stay supported
- `recipe` is a `PreprocessingRecipe` or a mapping; a mapping is the
  persisted form (`project-schema-contract_v3.md` §4), where an absent
  `sg_edges` means `interp`, an absent `asls` means no AsLS and absent
  `bands_keep`/`bands_exclude` select every band
- `fit` names a model `"{model} {n} LV, {recipe summary}"` (e.g.
  `"PLS 2 LV, SNV"`; the summary names AsLS, marks a Savitzky-Golay
  filter that keeps its edges and ends with `bands` when a band selection
  is set) unless `name` is given, adding ` (2)`, ` (3)`, … when the
  name exists

### Derived objects

- **Dataset**: identity comes from the absolute file path, compared
  independent of slash style; adding a path that is already registered
  returns the existing dataset after checking that the file still has the
  registered format, shape and (if the file states one) modality; wavelength
  values are metadata, not a gate (`dataset-calibration-contract_v3.md`)
- **Calibration**: built from the arguments exactly as the viewer builds it:
  - `unit="DN"`: mode `raw_dn`, no references
  - floating-point data: mode `precalibrated_import` (the data are already
    reflectance or absorbance)
  - `dark`/`white` given (files or arrays; 3-D cubes are averaged over frames):
    mode `dark_white_reference` with the reference spectra
  - otherwise: mode `dark_white_reference` with the file's `black_level` and
    `white_level` metadata, else the instrument levels
    `DEFAULT_BLACK_LEVEL = 128` and `DEFAULT_WHITE_LEVEL = 3500` (EVK)

  A calibration with the same content as an existing one of that dataset is
  reused (`CalibrationSpec.same_content_as`), never duplicated; the one used
  becomes the dataset's active calibration
- **ROI**: `frames` and `rows` are inclusive `(first, last)` index ranges, the
  `spatial_bounds` convention; spectra, dark/white reference pixels and
  provenance are taken from the dataset and its active calibration
- **Sample set**: `fit` (and `propose_bands`) builds it from every training
  ROI with a value for the target (prediction ROIs are never included; with
  only prediction ROIs valued, the fit is refused saying how many were left
  out)
  (`build_sample_set`) and reuses an existing sample set with the same ROIs,
  target and recipe
- **Recipe**: stored under an id derived from its content, so identical
  recipes are stored once
- **Signal domain**: derived with `SignalDomainContract.for_training_data`
  from the training datasets and calibrations; `fit` refuses ROIs whose
  datasets or calibrations would need different signal domains
- **Prediction**: `predict` resolves the data file and its calibration as
  `add_dataset` does, with the model's signal unit as `unit`, but in memory
  only (a registered file reuses its stored calibration read-only), then
  runs `preflight_prediction` and `predict_with_model_run` without
  persisting; nothing is written to the project

### Writes and failures

- every call validates and computes everything it can before its first write;
  a call that fails there leaves the project file and the `Project` unchanged
- writes happen in dependency order (dataset, calibration, recipe, sample
  set, signal domain, model run; for low-level persistence also prediction
  run), each one atomic; a failure
  during writing can leave complete, valid objects that nothing references
  yet, never a broken reference
- after a successful call, the `Project` in memory equals what
  `Project.open` would read from the file, without the heavy artifact arrays,
  which are read on demand; single-object writes update memory without
  rereading the file

## Callback Contract

Required behavior:

- prediction progress is reported through a plain callable receiving
  `PredictionProgressEvent` value objects, as defined by
  `prediction-runtime-contract_v3.md`
- cancellation is requested through a plain boolean-returning callable;
  cancellation raises `PredictionCancelledError` and discards staged artifacts
- no service signature may require a Qt type; the viewer adapts callables to
  signals on its side of the boundary (`_jobs.py`), never the reverse

## Notebook Rules

Required behavior:

- metadata-first project inspection must work in a notebook without loading
  ROI spectra or prediction arrays (`load_project_store` defaults)
- heavy arrays are pulled explicitly through the lazy loaders when needed
- plotting uses the matplotlib functions in `enginir.visualization`; no
  workflow step requires the viewer for visual inspection
- long-running fits and predictions must be usable synchronously (blocking
  call with callbacks); background threading is a caller decision, not a
  service requirement

## Viewer Relationship

- `enginir.viewer.open_viewer` and `enginir.viewer.open_project` are Qt entry
  points and are not part of the headless surface
- `ProjectState` is a Qt-side wrapper around `Project`: it delegates every
  workflow operation to it and adds only Qt signals and presentation state; it
  holds no workflow rule of its own
- workflow rules defined by the other v3 contracts (attachment validation,
  eager writes, placeholder rules) bind headless callers exactly as they bind
  the viewer

## Required Tests

- importing the headless packages must not import Qt (asserted in a clean
  interpreter)
- the end-to-end v3 workflow — create, persist, hydrate, build, fit, predict —
  must have test coverage that never constructs a QApplication
- the facade rules (names, derived objects, failures) each have a test, and
  the documented workflow example runs as a test

## Out Of Scope For v3

- a stable plugin or extension API
- remote/async execution services
- notebook widgets or IPython display integrations
- CLI workflow tooling beyond the existing viewer launcher

## Acceptance Criteria

This contract is implemented correctly when:

- the complete scientific workflow runs from a plain Python script with Qt
  absent
- the headless packages import without Qt in a clean interpreter
- projects produced headless and through the viewer are indistinguishable v3
  files
- progress and cancellation work with plain callables
- no scientific validity rule exists only in a viewer widget or `ProjectState`
- a complete workflow (create, add dataset, add target, add ROIs, fit,
  predict) takes one call per step through `Project`, with no hand-built ids,
  timestamps, provenance or bookkeeping objects
