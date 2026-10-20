# Project Dataset Attachment Contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.0; v3 is the default and only supported project schema
- Scope: viewer/project workflow states, dataset attachment, project-only behavior, mismatch validation, eager-write semantics

## Goal

Define the concrete workflow contract for:

- how a project is created
- what a project-only window is allowed to do
- when a dataset is considered attached to a project
- where mismatch validation must occur
- which actions are forbidden without a real dataset context

This contract exists to prevent project corruption caused by placeholder viewer
state being treated as real scientific context.

## Core Decisions

- a project file is a scientific workspace, not a viewer session snapshot
- a viewer window may open a project without an attached dataset
- project-only mode is a valid inspection mode, but not a valid ROI-creation mode
- dataset-derived mutations require an explicit attached dataset context
- placeholder cubes are UI bootstraps only and must never be promoted to project data
- project/dataset attach validation must live in one shared attach operation
- v3 project writes are eager; there is no deferred unsaved project graph

## Problem This Contract Solves

Without an explicit project/dataset workflow contract, the system cannot answer
reliably:

- whether the current window has a real dataset behind the project
- whether ROI creation is allowed
- whether a project being attached to a dataset-backed window is compatible
- whether API startup and in-window startup enforce the same mismatch rules
- whether `Save Project` is meaningful

The result is repeated workflow bugs:

- ROIs being created from placeholder cubes
- mismatch warnings firing in one open path but not another
- project state mutating before the scientific context is valid

## Domain Summary

There are three distinct concepts:

1. `ProjectWorkspace`
   The persisted v3 scientific workspace stored in the project file.
2. `ViewerDatasetContext`
   The currently loaded real dataset in one viewer window.
3. `ProjectAttachment`
   The binding between the active project workspace and the current viewer dataset context.

The viewer must not infer scientific validity from the mere existence of a
`ViewerState.cube`.

## Viewer Workflow States

v3 defines three valid window states:

### 1. Dataset-only

- a real dataset is loaded
- no project is attached

Allowed:

- inspect the dataset
- configure signal/preprocessing/axes
- create a new project and attach it
- open an existing project and attach it after validation

Forbidden:

- persist project-owned ROIs until a project is attached

### 2. Project-only

- a project is open
- no real dataset is attached to the current window

Allowed:

- inspect existing project objects
- inspect ROIs already stored in the project
- inspect targets, sample sets, model runs, and predictions
- export persisted project-backed data that does not depend on the current viewer cube
- attach a real dataset to the current window

Forbidden:

- create ROIs
- synthesize `DatasetRef` from placeholder viewer state
- synthesize `CalibrationSpec` from placeholder viewer state
- perform any dataset-derived mutation

### 3. Project-attached + dataset-attached

- a project is open
- a real dataset is attached to the current window

Allowed:

- all project-backed workflow actions
- ROI creation
- dataset-local calibration creation or reuse
- sample-set build
- model fit
- prediction against the attached or explicitly loaded dataset

## Placeholder Viewer Rule

Some startup paths may use a placeholder cube to bootstrap the UI shell.

Required invariant:

- placeholder viewer state must never be treated as a scientific dataset

Therefore:

- placeholder cube shape
- placeholder wavelengths
- placeholder metadata
- empty `_cube_path`

must never be used to create:

- `DatasetRef`
- `CalibrationSpec`
- `TrainingROI`
- `SampleSet` built from current viewer data

## ViewerDatasetContext

A dataset is considered attached only when all of the following are true:

- it was loaded from a real user-selected or API-supplied dataset source
- it has a non-empty canonical `source_path`
- it has a real wavelength axis
- it has a real `cube_shape`
- its source format and modality are known

Recommended runtime representation:

- `dataset_id`
- `source_path`
- `source_format`
- `source_modality`
- `cube_shape`
- `wavelengths`
- `is_real_dataset`

Required invariant:

- no dataset-derived mutation may run unless `is_real_dataset = true`

## Project Creation Contract

### 1. Create empty project

Creating a project file must create an empty scientific workspace only.

Required behavior:

- create all top-level v3 groups
- write `project_meta`
- do not create any dataset implicitly
- do not create any calibration implicitly
- do not create any ROI implicitly

### 2. Attach dataset later

Attaching a dataset to a project is a separate operation from creating the project.

Required behavior:

- resolve or create `DatasetRef` from the real attached dataset
- resolve or create dataset-local active calibration
- mark the current window as dataset-attached

## Project Attach Contract

Attaching a project to a dataset-backed window must go through one shared
operation used by:

- menu-driven `Open Project...`
- welcome-screen startup
- CLI startup
- public Python API startup

Required invariant:

- mismatch validation semantics must not depend on which open path was used

This means attach validation must not live only inside one UI button handler.

## Mismatch Validation Contract

When a project is attached to a dataset-backed window, the system must validate
whether existing project ROIs are compatible with the currently attached dataset.

Minimum v3 comparison basis:

- canonical `source_path`

Recommended additional comparison basis:

- `source_format`
- `source_modality`
- `cube_shape`
- wavelength axis

v3 result categories:

### Safe

- all ROI source datasets represented in the project match the attached dataset

Behavior:

- attach without warning

### Warning

- one or more persisted ROI datasets do not match the attached dataset

Behavior:

- interactive UI flows must warn the user and require confirmation
- non-interactive API/CLI flows must either:
  - raise, or
  - opt into an explicit `allow_mismatch` policy

### Project-only open

- no dataset is attached to the current window

Behavior:

- allow project open
- keep dataset-derived mutations disabled

## Dataset-Derived Mutation Contract

A dataset-derived mutation is any action that creates or mutates project objects
using the current viewer dataset context.

Examples:

- create ROI
- create dataset-local calibration defaults from current viewer state
- build a sample set from current viewer selections
- fit a model from current viewer-derived sample-building inputs
- predict against the current viewer dataset

Required invariant:

- dataset-derived mutations require `Project-attached + dataset-attached`

Project-only mode must reject these actions explicitly.

## ROI Creation Contract

Creating a `TrainingROI` requires:

- active project
- real attached dataset context
- resolved dataset-local calibration
- non-empty canonical dataset source path

Required invariants:

- every new ROI must reference an existing `DatasetRef`
- every new ROI must reference an existing `CalibrationSpec`
- `source_file` must not be empty
- ROI provenance must come from real dataset context, not placeholder viewer state

## Save Semantics

v3 project object writes are eager.

Required interpretation:

- creating or mutating a first-class project object writes immediately
- there is no pending unsaved project graph in memory

Resolved for v3:

- `Save Project` is removed from the UI; no save action exists or is required
  to persist ROIs, targets, sample sets, models, or predictions
- reintroducing a save-like action requires a real transaction model defined
  in a future contract revision, not a placeholder button over eager writes

## Public API Requirements

### `open_project(path)`

Required behavior:

- open a project-only window
- do not synthesize dataset context
- disable dataset-derived mutations until a real dataset is attached

### `open_viewer(data_path, project=...)`

Required behavior:

- load the real dataset first
- attach the project through the shared attach-validation path
- enforce the same mismatch policy as interactive attach

### In-window `Open Data...` from project-only mode

Required behavior:

- attach the dataset to the same project window
- do not silently fork the user into a separate dataset-only child window

## Error Handling Requirements

UI-triggered contract violations must be user-visible and non-destructive.

Required behavior:

- show a clear warning/error dialog
- do not persist partial objects before validation completes
- do not leave orphan `DatasetRef`, `CalibrationSpec`, `SignalDomainContract`, or `PreprocessingRecipe` objects behind on failure

Orphan definition for this rule:

- an orphan is an object persisted by the failed action that no other
  persisted object references and that is not a content-stable shared value
  object
- content-stable idempotent objects — preprocessing recipes and
  content-deduplicated calibrations — are exempt: identical content resolves
  to the same id, so a pre-existing or re-creatable value object left behind
  by a failed downstream step is not corruption
- the current implementation satisfies this by ordering: signal domains,
  sample sets, model runs, and prediction finalization are persisted only
  after their producing computation succeeds

## Required Tests

v3 requires regression coverage for:

- project-only startup rejects ROI creation
- placeholder viewer state cannot create `DatasetRef`
- `open_viewer(data_path, project=...)` enforces the same mismatch policy as in-window attach
- in-window dataset attach from project-only mode reuses the same window
- failed dataset-derived actions are side-effect free
- `Save Project` semantics match the documented behavior

## Non-Goals For v3

This contract does not define:

- multi-dataset active editing in one window
- transactional undo/redo of project writes
- multi-user locking semantics
- full project session restore

## Summary

The key v3 workflow rule is:

- a project may be open without a dataset, but dataset-derived mutations are forbidden until a real dataset is attached

This is the boundary that prevents placeholder viewer state from becoming
scientific project state and ensures that all project attach paths obey the
same validation rules.
