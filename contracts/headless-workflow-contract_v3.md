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
  `save_calibration_spec`, `save_training_roi`, `save_target_definition`,
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
- `project.fit(target, *, recipe, model, name=None, cv="auto", **hyperparameters)`
  builds the sample set, fits and cross-validates, and returns a model handle
- `model.predict(path, *, dark=None, white=None)` predicts a dataset and
  returns a prediction handle
- read access: `project.datasets`, `project.rois`, `project.targets`,
  `project.models`, `project.predictions`, keyed by name (datasets by path);
  handles expose their typed object (`.ref`) and the essentials (`model.metrics`,
  `model.diagnostics`, `prediction.map`, `prediction.y_pred`)

### Names

- users address ROIs, targets and models by name; every object keeps a
  generated stable id, which is what the project file links by
- names are unique per kind within a project; creating a second ROI, target
  or model with an existing name is an error
- projects written by earlier tools may hold duplicate names; opening them is
  allowed, and looking up an ambiguous name is an error that lists the ids
- `fit` names a model `"{model} {n} LV, {recipe summary}"` (e.g.
  `"PLS 2 LV, SNV"`) unless `name` is given, adding ` (2)`, ` (3)`, … when the
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
- **Sample set**: `fit` builds it from every ROI with a value for the target
  (`build_sample_set`) and reuses an existing sample set with the same ROIs,
  target and recipe
- **Recipe**: stored under an id derived from its content, so identical
  recipes are stored once
- **Signal domain**: derived with `SignalDomainContract.for_training_data`
  from the training datasets and calibrations; `fit` refuses ROIs whose
  datasets or calibrations would need different signal domains
- **Prediction**: `predict` resolves the dataset as `add_dataset` does, with
  the model's signal unit as `unit`, then runs `preflight_prediction` and
  `predict_with_model_run` and stores the prediction run

### Writes and failures

- every call validates and computes everything it can before its first write;
  a call that fails there leaves the project file and the `Project` unchanged
- writes happen in dependency order (dataset, calibration, recipe, sample
  set, signal domain, model run, prediction run), each one atomic; a failure
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
