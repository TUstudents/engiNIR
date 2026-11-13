# Project dataset attachment contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.5; the file format `enginir-1.5` is the
  only format engiNIR reads and writes (`project-schema-contract_v3.md`)
- Scope: viewer/project workflow states, dataset attachment, datasets of a project, project-only behavior, mismatch validation, eager-write semantics

## Goal

Define the concrete workflow contract for:

- how a project is created
- what a project-only window is allowed to do
- when a dataset is considered attached to a project
- where mismatch validation must occur
- which actions are forbidden without a real dataset context

This contract exists to prevent project corruption caused by placeholder viewer
state being treated as real scientific context.

## Core decisions

- a project file is a scientific workspace, not a viewer session snapshot
- a viewer window may open a project without an attached dataset
- project-only mode is a valid inspection mode, but not a valid ROI-creation mode
- dataset-derived mutations require an explicit attached dataset context
- placeholder cubes are UI bootstraps only and must never be promoted to project data
- project/dataset attach validation must live in one shared attach operation
- v3 project writes are eager; there is no deferred unsaved project graph
- a project collects ROIs from several datasets; one window shows one
  dataset of its project at a time and switches between them (see Datasets Of A Project)
- stored ROIs carry their raw pixels and references, so inspecting them,
  editing their target values, building calibration sets and fitting models need
  only the project file

## Problem this contract solves

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

## Domain summary

There are three distinct concepts:

1. `ProjectWorkspace`
   The persisted v3 scientific workspace stored in the project file.
2. `ViewerDatasetContext`
   The currently loaded real dataset in one viewer window: the *shown dataset*. In a project window it is one of the project's registered datasets.
3. `ProjectAttachment`
   The binding between the active project workspace and the current viewer dataset context.

The viewer must not infer scientific validity from the mere existence of a
`ViewerState.cube`.

## Viewer workflow states

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
- inspect targets, calibration sets, models, and predictions
- export persisted project-backed data that does not depend on the current viewer cube
- attach a real dataset to the current window
- show the stored ROIs from the project file (the Project ROIs view of the
  viewer workspace UI contract)
- create, edit and remove target assignments of stored ROIs
- build calibration sets from stored ROIs and fit models; both read only the
  project file

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
- ROI creation on the shown dataset
- dataset-local reference correction creation or reuse
- calibration-set build
- model fit
- prediction against the attached or explicitly loaded dataset
- switching the shown dataset to another registered dataset, or adding
  a new dataset

## Placeholder viewer rule

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

## Project creation contract

### 1. Create empty project

Creating a project file must create an empty scientific workspace only.

Required behavior:

- create all top-level v3 groups
- write `project_meta` with `FileSchemaVersion = "enginir-1.5"`,
  `FileCreator`, `FileWriter` and the timestamps
  (`project-schema-contract_v3.md`, Object identity rules §1)
- do not create any dataset implicitly
- do not create any reference correction implicitly
- do not create any ROI implicitly

### 2. Attach dataset later

Attaching a dataset to a project is a separate operation from creating the project.

Required behavior:

- resolve or create `DatasetRef` from the real attached dataset
- resolve or create dataset-local active reference correction
- mark the current window as dataset-attached

## Project attach contract

Attaching a project to a dataset-backed window must go through one shared
operation used by:

- menu-driven `Open Project...`
- welcome-screen startup
- CLI startup
- public Python API startup

Required invariant:

- mismatch validation semantics must not depend on which open path was used

This means attach validation must not live only inside one UI button handler.

## Mismatch validation contract

Validation is per dataset. When a dataset is attached to a project window
(at startup, by Open Data, or by switching the shown dataset), it is compared
with its own registration, never with the datasets of other ROIs.

Comparison basis:

- canonical `source_path`, which decides whether the file is already
  registered
- `source_format`
- `source_modality`
- `cube_shape`
- wavelength axis

v3 result categories:

### Safe

- the file is registered and matches its `DatasetRef`, or
- the file is not registered yet; it is added to the project as a new dataset (see Datasets Of A Project)

Behavior:

- attach without warning
- ROIs of other datasets are never a mismatch

### Warning

- the file at a registered `source_path` no longer matches its `DatasetRef`
  (format, modality, shape or wavelength axis differ)

Behavior:

- interactive UI flows must warn the user, name the differing fields, and
  require confirmation; ROIs are named by their labels, not only by id
- non-interactive API/CLI flows must either:
  - raise, or
  - opt into an explicit `allow_mismatch` policy

### Project-only open

- no dataset is attached to the current window

Behavior:

- allow project open
- keep dataset-derived mutations disabled

## Datasets of a project

A project registers every dataset its ROIs come from (`DatasetRef`), each
with its own active reference correction (`dataset-calibration-contract_v3.md`).

### Adding a dataset

`Add Dataset...`, and `Open Data...` in a project window, add a dataset
to the project in the same window.

Required behavior:

- the user chooses the reference correction together with the file:
  - raw counts
  - already reflectance or absorbance
  - dark/white references, each from another file or from a frame range of
    this file
  - the references of another dataset of the project
  - detector levels (black and white) without references: the file's own
    levels, else the instrument defaults; for counts this is the preselected
    choice, so references are always chosen deliberately
- the `DatasetRef` and its `CalibrationSpec` are written together, before any
  ROI is drawn; if either fails, neither is written
- the added file becomes the shown dataset
- a dataset is never opened in a separate window without the project

### Switching the shown dataset

Required behavior:

- choosing a registered dataset loads it from its `source_path` and makes
  its active reference correction current
- the image shows only the ROIs of the shown dataset
- loading runs off the UI thread with visible progress

### Missing and moved files

A dataset whose data file is not found is a normal state, not an error.

A project moves between computers and platforms (the same share is
`/mnt/z/…` in WSL and `Z:\…` on Windows). The data file is therefore
looked for, in this order, at:

1. its `relative_path`, joined to the folder of the project file
2. its stored `source_path`
3. its stored `source_path` mapped between a Windows drive and its WSL
   mount: `<X>:\rest` ↔ `<root><x>/rest`, where `<root>` is the WSL
   automount root (`[automount] root` in `/etc/wsl.conf`, default `/mnt/`)

A stored path is read by its own syntax, whatever the platform reading it:
a drive letter or a `\\` prefix makes it a Windows path, a leading `/` a
POSIX path.

Required behavior:

- the first candidate that exists and is this dataset's data file is used
  for the session; a candidate that is the stored `source_path` itself is
  used as before, without a further check
- any other candidate must have the dataset's format, shape, modality (if
  the file states one) and wavelength axis, and the pixels of the dataset's
  smallest ROI must equal its stored raw pixels; a candidate that fails is
  skipped and logged, and the next one is tried
- finding a file never writes the project: `source_path` and
  `relative_path` stay as stored, and only `Locate File...` changes them
- a registered dataset is recognized from any of its candidate paths (a
  file opened at `Z:\…` is the dataset registered at `/mnt/z/…`), so it is
  never registered twice
- the viewer shows the path the file is read from (the tooltip of the
  dataset list); how it was found ("next to the project", "via drive Z:")
  and the stored path are recorded in the log at info level only
- when no candidate is found, it is marked as not found
- its ROIs, their target assignments, calibration sets and models remain fully
  usable, because they need only the project file
- `Locate File...` accepts a new path only if:
  - the file matches the `DatasetRef` (format, modality, shape, wavelength
    axis), and
  - for every ROI of that dataset, the stored raw pixels equal the file's
    pixels at the ROI's `spatial_bounds`
- an accepted path updates `DatasetRef.source_path` and its
  `relative_path` (an identity exception of `project-schema-contract_v3.md`
  §5) and is logged
- after a successful `Locate File...`, the viewer offers the same folder
  change for the other data files that are not found; each is checked as
  above and written only if it passes

### Reopening a project

Required behavior:

- a project with ROIs opens on the stored ROIs (Project ROIs view); no dataset is loaded until the user shows one
- a project without ROIs opens on the Dataset tab with `Add Dataset...`
- the last shown dataset of a project is UI state, kept in the user
  settings by `project_id`, never in the project file

### Several windows on one project

Required behavior:

- a window that shows a project re-reads its project mirror after another
  window of the same process wrote to that project
- no action opens a second window on the same project implicitly

## Dataset-derived mutation contract

A dataset-derived mutation is any action that creates or mutates project objects
using the current viewer dataset context.

Examples:

- create ROI
- create dataset-local reference correction defaults from current viewer state
- predict against the current viewer dataset

Not dataset-derived: building a calibration set from stored ROIs, fitting a model
on it, and editing target assignments. They read only the project file and
are allowed in project-only mode.

Required invariant:

- dataset-derived mutations require `Project-attached + dataset-attached`

Project-only mode must reject these actions explicitly.

## ROI creation contract

Creating a `TrainingROI` requires:

- active project
- real attached dataset context
- resolved dataset-local reference correction
- non-empty canonical dataset source path

Required invariants:

- every new ROI must reference an existing `DatasetRef`
- every new ROI must reference an existing `CalibrationSpec`
- `source_file` must not be empty
- ROI provenance must come from real dataset context, not placeholder viewer state

## Save semantics

v3 project object writes are eager.

Required interpretation:

- creating or mutating a first-class project object writes immediately
- there is no pending unsaved project graph in memory

Resolved for v3:

- `Save Project` is removed from the UI; no save action exists or is required
  to persist ROIs, targets, calibration sets, models, or predictions
- reintroducing a save-like action requires a real transaction model defined
  in a future contract revision, not a placeholder button over eager writes

## Public API requirements

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

### In-window `Open Data...` in a project window

Required behavior, from project-only mode and from a window that already
shows a dataset:

- add the dataset to the project (or show it, if it is registered) in the
  same window
- do not fork the user into a separate dataset-only child window

## Error handling requirements

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
  content-deduplicated reference corrections — are exempt: identical content resolves
  to the same id, so a pre-existing or re-creatable value object left behind
  by a failed downstream step is not corruption
- the current implementation satisfies this by ordering: input contracts,
  calibration sets, models, and prediction finalization are persisted only
  after their producing computation succeeds

## Required tests

v3 requires regression coverage for:

- project-only startup rejects ROI creation
- placeholder viewer state cannot create `DatasetRef`
- `open_viewer(data_path, project=...)` enforces the same mismatch policy as in-window attach
- in-window dataset attach from project-only mode reuses the same window
- failed dataset-derived actions are side-effect free
- `Save Project` semantics match the documented behavior
- a project window shows only the ROIs of the shown dataset
- attaching a dataset never warns about ROIs of other datasets, and warns
  when a registered file no longer matches its `DatasetRef`
- `Add Dataset...` writes the dataset and its reference correction together or not
  at all
- `Open Data...` in a project window keeps the project
- `Locate File...` refuses a file whose pixels differ from the stored ROI
  pixels
- target assignments, calibration sets and fits work with every dataset
  missing

## Non-goals for v3

This contract does not define:

- showing several datasets at once in one window (one shown dataset at a
  time is in scope)
- transactional undo/redo of project writes
- multi-user locking semantics
- full project session restore

## Summary

The key v3 workflow rules are:

- a project may be open without a dataset, but dataset-derived mutations are forbidden until a real dataset is attached
- a project window shows one registered dataset at a time and draws only
  its ROIs; everything built from stored ROIs works without any dataset

This is the boundary that prevents placeholder viewer state from becoming
scientific project state and ensures that all project attach paths obey the
same validation rules.
