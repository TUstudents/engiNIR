# ROI target contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.5; the file format `enginir-1.5` is the
  only format engiNIR reads and writes (`project-schema-contract_v3.md`)
- Scope: persisted calibration ROIs, target definitions, target assignments, ROI provenance, modeling selection semantics

## Goal

Define the concrete contract for:

- what a persisted calibration ROI is
- what target definitions are
- how target values attach to ROIs
- how ROI data and target data are used for modeling
- what is in scope for v3 and what is intentionally deferred

This contract exists to prevent the domain model from collapsing into a single
anonymous `y_value` field tied too closely to one viewer dialog.

## Core decisions

- calibration ROIs are project-owned once created
- raw ROI spectra are persisted as the scientific truth
- ROI provenance records extraction context, but does not replace raw spectra
- target definitions are explicit project objects
- each ROI may carry one or more target assignments
- v3 modeling may fit one selected target at a time, but the domain contract is multi-target

## Problem this contract solves

The workflow requires:

1. draw ROIs from corrected raw datasets
2. persist ROIs into the project
3. assign one or more `y` values to each ROI
4. choose one target for a model
5. fit and validate a model reproducibly

Without an ROI/target contract, the system cannot answer reliably:

- what an ROI actually represents
- which dataset an ROI came from
- what target values are available in the project
- whether a model was fit to moisture, protein, or some other response
- whether missing or partial target assignments are allowed

## Contract summary

There are three separate but linked concepts:

1. `TrainingROI`
   The persisted scientific object extracted from one dataset region.
2. `TargetDefinition`
   The named response variable definition used by the project.
3. `TargetAssignment`
   The value of one target for one ROI.

The model never points to a raw anonymous `y_value`.
It points to:

- a selected `TargetDefinition`
- a `SampleSet` built from ROIs

## Domain objects

## 1. TrainingROI

Represents one persisted calibration ROI in the project.

Required fields:

- `roi_id`
- `dataset_id`
- `name`
- `spatial_bounds`
- `n_pixels`
- `wavelengths`
- `source_file`
- `created_at`
- `calibration_id`
- `provenance`
- `job`: `calibration` or `prediction` (`terminology_v3.md`, Calibration
  and Prediction). A `prediction` ROI marks a sample to be predicted (e.g. a
  tablet of a production tray): it never enters a calibration set, whatever
  target values it has; a value it has serves only the residual and RMSEP (per ROI) of
  a prediction (`prediction-runtime-contract_v3.md`, Prediction ROIs). Any
  other stored value, and a missing `job`, is refused on read with a logged
  error

Optional fields:

- `sample`: label of the physical sample the ROI was taken from (e.g. the
  blend or tablet), a non-empty string; absent when the ROI is its own
  sample

Required invariants:

- every ROI belongs to exactly one source dataset
- every ROI belongs to exactly one project
- every ROI references exactly one applied reference correction context
- `n_pixels` must match the stored spectra shape
- `len(wavelengths)` must match the spectra band dimension
- if the reference correction context uses full-sensor dark/white reference arrays, the ROI must persist ROI-local dark/white reference snapshots with the same `(n_pixels, n_bands)` shape and flattened pixel order as the raw ROI spectra
- ROI identity is stable inside the project and must not depend only on its display name

v3 rule:

- `roi_id` is the real persistent identity
- `name` is a mutable user-facing label and may be changed without affecting identity
- `sample` is a mutable label like `name`: setting, changing or removing it
  never changes the ROI's identity, spectra or other fields. It only says
  which ROIs belong to one physical sample, for fold groups
  (`sample-set-model-run-contract_v3.md` §4.1). Every writer removes
  surrounding spaces and stores an empty result as no label; labels are
  then compared as exact Unicode strings (case and inner spaces matter, no
  further normalization). Because labels that differ only in case would
  split one physical sample into two groups, a fit warns when two labels of
  its calibration set are equal after case folding. A stored model keeps the
  fold groups it used, so later label edits do not change how it refits.
  Only `save_roi_sample` sets, changes or removes the label; re-saving an
  ROI never changes it
- `job` is set when the ROI is created (the ROI and ROI grid dialogs,
  `Dataset.add_roi(..., job=...)`); afterwards it is mutable like `sample`:
  only `save_roi_job` changes it, in place;
  it never changes the ROI's identity, spectra or other fields. Switching an
  ROI to `prediction` leaves stored calibration sets and models that contain
  it unchanged (they refit as they were fitted); new calibration sets leave it
  out
- raw ROI spectra remain the scientific truth; ROI-local reference snapshots preserve the sensor-row reference correction context needed to transform those raw spectra reproducibly
- ROI spectra persist in their source dtype: uint16 for Raw counts (DN) sources,
  float32 for already corrected floating sources; float spectra must never be
  truncated to integers, and already corrected ROIs carry no dark/white reference
  snapshots because their reference correction mode stores no reference context
- the spectra storage location is canonical and derived as
  `/rois/{roi_id}/spectra`; it is not separately persisted or user-editable

## 2. ROI provenance

ROI provenance records extraction-time scientific context.

Required provenance fields:

- `dataset_id`
- `calibration_id`
- `signal_unit`
- `preview_recipe_summary`
- `created_at`

Required provenance invariants:

- `dataset_id` equals the owning ROI's `dataset_id`
- `calibration_id` equals the owning ROI's `calibration_id`
- `signal_unit` is one of `DN`, `Reflectance`, or `Absorbance`
- `preview_recipe_summary` and `created_at` are non-empty

Recommended provenance fields:

- `viewer_state_snapshot`
- `user_label`
- `notes`

v3 interpretation:

- provenance is an audit trail
- provenance is informational for later review
- provenance does not define the modeling preprocessing recipe
- provenance is a declared structured field; additional preview and audit keys
  are allowed inside it and are not schema fields of the owning ROI

## 3. TargetDefinition

Represents one named target variable in the project.

Required fields:

- `target_id`
- `name`
- `unit`
- `value_type`
- `missing_value_policy`
- `created_at`

Allowed `value_type` values in v3:

- `float`

Allowed `missing_value_policy` values in v3:

- `allowed`
- `forbidden_for_modeling`

`missing_value_policy` semantics in v3:

- target assignments must carry finite float values, so no persisted
  missing-value state exists for the policy to police
- the field therefore records declared intent and has no executable effect in
  v3; calibration-set building excludes ROIs without an assignment for the selected
  target regardless of the policy value
- viewer-created targets default to `allowed`
- a future contract must define enforcement semantics before
  `forbidden_for_modeling` changes runtime behavior

Required invariants:

- target identity is stable inside the project
- target `name` is a user-facing label; identity is `target_id`. The UI offers
  the existing targets by name, so a new target does not repeat a name by
  accident, but a repeated name is a label, not an error
- target definitions are project-owned objects, not implicit table columns

v3 scope:

- numeric scalar regression targets only

Out of scope for v3:

- categorical class labels
- vector-valued targets
- hierarchical target schemas

## 4. TargetAssignment

Represents the value of one target for one ROI.

Required fields:

- `roi_id`
- `target_id`
- `value`
- `assigned_at`

Required invariants:

- each `(roi_id, target_id)` pair has at most one active assignment in v3
- `value` must conform to the `TargetDefinition.value_type`
- missing assignment is different from assigned null semantics

v3 interpretation:

- an ROI may have zero, one, or many target assignments
- a model chooses one target definition at a time

## Separation of responsibilities

### TrainingROI owns

- source dataset linkage
- spatial bounds
- extracted raw spectra
- extraction provenance

### TargetDefinition owns

- what a response variable means
- its display name and unit
- its value semantics

### TargetAssignment owns

- the actual value for one ROI/target pair

### Model owns

- which target definition is being modeled
- which ROIs are included
- which fitting/grouping semantics are used

## Workflow semantics

## 1. Persist ROI

When a calibration ROI is saved into the project:

- the current window must have a real attached dataset context
- a `TrainingROI` object is created
- raw spectra are persisted
- provenance is persisted
- no target assignment is required at creation time

v3 rule:

- unlabeled ROIs are allowed in the project
- unlabeled ROIs are excluded from a model for any target they do not have
- project-only mode may inspect existing ROIs, but must not create new ones
- newly created ROIs must not be derived from placeholder viewer state

### ROI grid

Many samples in one dataset (a tray) may be created as a grid:

- the grid is explicit geometry in image pixels: the number of rows and
  columns, the ROI size (width along frames, height along rows), the origin
  (frame and row of the first ROI's top-left pixel) and the spacing between
  neighboring ROIs; all ROIs of a grid have the same size and never overlap
- the user also chooses the empty positions and a name pattern (selectable
  presets or a custom pattern) and the order
- every created ROI lies inside the image; positions marked empty may lie
  outside it
- a new grid starts from the user's defaults (Options > ROI Grid Defaults,
  kept in the user settings; built in: 5 columns x 1 row, ROI 180 x 110,
  origin 150, 64, spacing 214, 10); changes in the grid dialog apply to that
  grid only, unless the user makes them the defaults from the dialog
  (`Make defaults`)
- each ROI's provenance records its grid position (e.g. `B4`) and the grid
  geometry
- every cell becomes one `TrainingROI` with its own raw spectra, reference
  snapshot and provenance, exactly as a single ROI
- the grid is validated completely before anything is written, and written
  completely or not at all; a failure leaves no ROI of the grid behind
- names are labels and may repeat ROIs of other datasets, for example when
  several datasets share a file name; the default pattern starts with the
  dataset name. Only names repeated within one grid are refused, because
  such a pattern cannot tell its own cells apart

## 2. Define target

When the user creates a new response variable such as `moisture` or `protein`,
the project must create a `TargetDefinition`.

Examples:

- `name = "moisture"`, `unit = "% w/w"`, `value_type = "float"`
- `name = "protein"`, `unit = "% w/w"`, `value_type = "float"`

## 3. Assign targets to ROI

When the user edits target values for an ROI:

- one `TargetAssignment` is created or updated per `(roi_id, target_id)`
- removing a value deletes the `TargetAssignment` of that pair; the ROI is
  then unlabeled for that target, which is different from any stored value

Examples:

- ROI `cheese_A` may have:
  - moisture = 41.2
  - fat = 23.8
  - protein = 19.4

## 4. Build modeling inputs

When a model is created:

1. choose one `TargetDefinition`
2. choose included ROIs
3. filter to ROIs that have an assignment for that target and whose `job`
   is `calibration`
4. build `X` from pixel-level raw ROI spectra
5. build `Y` by repeating the selected ROI-level target assignment across those ROI pixels
6. build grouping metadata from `roi_id`

v3 rule:

- one model uses one selected target definition
- multi-target projects are allowed
- multi-target model fitting is out of scope for v3
- pixel-level spectral diversity is preserved in `X`
- validation and grouping semantics remain ROI-level

## 5. Review and edit targets

The viewer edits target values in an ROI table:

- one row per ROI, showing its dataset, name and pixel count
- a "Set" column, Calibration or Prediction chosen from a drop-down, switched in
  place (`save_roi_job`)
- a "Sample" column with the ROI's sample label; typing a label sets it
  when the cell is left, clearing the cell removes it, and pasting a column
  fills consecutive rows (labels are text, so any text is accepted;
  leading and trailing spaces are removed and an empty result removes the
  label)
- one column per target definition
- an empty cell is a missing assignment, never zero
- typing a value creates or updates the assignment when the cell is left;
  clearing a cell removes it
- pasting a column of values fills consecutive rows; values that are not
  finite numbers are refused and nothing of that paste is written
- the table works in project-only mode, without any dataset

The contract requirement is:

- target definitions and assignments must remain explicit project objects

The UI must not imply:

- that the project has only one universal anonymous `y` column

## Compatibility rules

Modeling for a selected target must block or warn when:

- no ROIs are included
- included ROIs do not have assignments for the selected target
- target values are not valid for the target definition
- wavelength compatibility across included ROIs is broken
- ROI provenance shows incompatible source/reference correction context for the intended calibration-set building rules
- grouping metadata cannot preserve ROI-level validation semantics

## v3 scope rules

v3 supports:

- project-owned persisted ROIs
- explicit target definitions
- one or more target assignments per ROI
- numeric scalar regression targets
- one selected target per model
- pixel-level spectra as the data points of the fit
- ROI-level target semantics
- ROI-grouped validation semantics

v3 rejects:

- implicit anonymous `y_value` as the core project contract
- modeling multiple targets simultaneously in one model
- category labels as first-class target values
- mixed incompatible wavelength axes inside one model

## Persistence requirements

`TrainingROI`, `TargetDefinition`, and `TargetAssignment` must be persisted inside
the project file.

v3 persistence decision:

- ROI identity travels with the project
- target definitions travel with the project
- target assignments travel with the project
- no external target registry is required for contract compliance

## Required persisted ROI information

- `roi_id`
- `dataset_id`
- `name`
- `spatial_bounds`
- `n_pixels`
- `wavelengths`
- `source_file`
- `created_at`
- `calibration_id`
- provenance
- `sample`, when set
- raw spectra storage
- ROI-local dark/white reference storage when the active reference correction has embedded full-sensor references

Required creation-time invariants for newly created ROIs:

- `dataset_id` must reference a real attached dataset context
- `calibration_id` must reference the active reference correction for that dataset
- `source_file` must be non-empty
- ROI identity must not depend on the display name

## Required persisted target-definition information

- `target_id`
- `name`
- `unit`
- `value_type`
- `missing_value_policy`
- `created_at`

## Required persisted target-assignment information

- `roi_id`
- `target_id`
- `value`
- `assigned_at`

## v3 HDF5 shape basis

v3 requires project-internal groups shaped like:

```text
/rois/{roi_id}/
  /calibration_refs/
    dark
    white
/targets/{target_id}/
/target_assignments/{target_id}/{roi_id}
```

Minimum expectation:

- ROI metadata and spectra are persisted inside `/rois/`
- ROI-local reference correction reference snapshots, when present, are persisted inside `/rois/{roi_id}/calibration_refs/`
- target definitions are persisted independently from ROIs
- target assignments are persisted independently from both ROI identity and target identity

v3 note:

- a single `@y_value` stored on an ROI is not a target representation; the
  project reader rejects ROI groups that carry one

## Required UI behavior

The UI must make these distinctions legible:

- saving an ROI creates a project-owned ROI object
- defining a target creates a named project-level response variable
- assigning a target value links an ROI to that target definition
- modeling chooses one target definition explicitly

The UI must not imply:

- that ROI names are sufficient identity
- that one built-in `y_value` field is the whole target model
- that provenance and modeling target are the same concept

## Out of scope for v3

These are intentionally deferred:

- classification labels as a target type
- multi-output models
- target inheritance or grouping
- assay-version tracking
- multiple active assignments for the same `(roi_id, target_id)` pair
- full target history/audit beyond latest assignment timestamp

## Open questions after this decision

These remain open after v3 contract definition.

### 1. ROI label editing

Decision:

- ROI identity is defined only by `roi_id`
- ROI `name` is a mutable label and may be changed

Out of scope for v3:

- alias history for renamed ROI labels

### 2. Target editing UX

Decision:

- the domain model is multi-target
- the v3 UI edits all targets in the ROI table, one column per target
  (Workflow Semantics §5)

Out of scope for v3:

- importing target values from external files (CSV, spreadsheets)
- undo of target edits

### 3. Fitting-unit interaction

Decision:

- the domain may support multiple fitting-unit styles later
- v3 modeling uses pixel-level spectra with ROI-level target semantics
- v3 validation groups pixels by `roi_id`; folds may join the ROIs
  of one physical sample (ROI `sample` label) or of one target value
  (`sample-set-model-run-contract_v3.md` §4.1), never split an ROI

Out of scope for v3:

- `roi_mean` as the primary fitting contract
- `roi_feature` modeling
- exposing a user-facing fitting-unit selector in the first modeling UI

## Acceptance criteria

This contract is implemented correctly when:

- persisting a calibration ROI creates a stable ROI identity inside the project
- ROI provenance records source dataset and reference correction context
- the project can define more than one named target variable
- one ROI can carry assignments for more than one target
- a model chooses one target definition explicitly
- modeling inputs preserve pixel-level spectral diversity while keeping target semantics ROI-level
- unlabeled ROIs remain valid project objects but are excluded from modeling for targets they do not have
- targets are represented by explicit definitions and assignments, never by a
  single anonymous `y_value`
- target values of all targets can be edited, removed and pasted in the ROI
  table, also without any dataset
- the sample label can be set, changed and removed in the ROI table and by
  script without changing the ROI's identity, spectra or other fields
- an ROI grid is written completely or not at all
