# ROI Target Contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.0; v3 is the default and only supported project schema
- Scope: persisted training ROIs, target definitions, target assignments, ROI provenance, modeling selection semantics

## Goal

Define the concrete contract for:

- what a persisted training ROI is
- what target definitions are
- how target values attach to ROIs
- how ROI data and target data are used for modeling
- what is in scope for v3 and what is intentionally deferred

This contract exists to prevent the domain model from collapsing into a single
anonymous `y_value` field tied too closely to one viewer dialog.

## Core Decisions

- training ROIs are project-owned once created
- raw ROI spectra are persisted as the scientific truth
- ROI provenance records extraction context, but does not replace raw spectra
- target definitions are explicit project objects
- each ROI may carry one or more target assignments
- v3 modeling may fit one selected target at a time, but the domain contract is multi-target

## Problem This Contract Solves

The workflow requires:

1. draw ROIs from calibrated raw datasets
2. persist ROIs into the project
3. assign one or more `y` values to each ROI
4. choose one target for a model run
5. fit and validate a model reproducibly

Without an ROI/target contract, the system cannot answer reliably:

- what an ROI actually represents
- which dataset an ROI came from
- what target values are available in the project
- whether a model was fit to moisture, protein, or some other response
- whether missing or partial target assignments are allowed

## Contract Summary

There are three separate but linked concepts:

1. `TrainingROI`
   The persisted scientific object extracted from one dataset region.
2. `TargetDefinition`
   The named response variable definition used by the project.
3. `TargetAssignment`
   The value of one target for one ROI.

The model run never points to a raw anonymous `y_value`.
It points to:

- a selected `TargetDefinition`
- a `SampleSet` built from ROIs

## Domain Objects

## 1. TrainingROI

Represents one persisted training ROI in the project.

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

Required invariants:

- every ROI belongs to exactly one source dataset
- every ROI belongs to exactly one project
- every ROI references exactly one applied calibration context
- `n_pixels` must match the stored spectra shape
- `len(wavelengths)` must match the spectra band dimension
- if the calibration context uses full-sensor dark/white reference arrays, the ROI must persist ROI-local dark/white reference snapshots with the same `(n_pixels, n_bands)` shape and flattened pixel order as the raw ROI spectra
- ROI identity is stable inside the project and must not depend only on its display name

v3 rule:

- `roi_id` is the real persistent identity
- `name` is a mutable user-facing label and may be changed without affecting identity
- raw ROI spectra remain the scientific truth; ROI-local reference snapshots preserve the sensor-row calibration context needed to transform those raw spectra reproducibly
- ROI spectra persist in their source dtype: uint16 for raw DN sources,
  float32 for precalibrated floating sources; float spectra must never be
  truncated to integers, and precalibrated ROIs carry no dark/white reference
  snapshots because their calibration mode stores no reference context
- the spectra storage location is canonical and derived as
  `/rois/{roi_id}/spectra`; it is not separately persisted or user-editable

## 2. ROI Provenance

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
  v3; sample-set building excludes ROIs without an assignment for the selected
  target regardless of the policy value
- viewer-created targets default to `allowed`
- a future contract must define enforcement semantics before
  `forbidden_for_modeling` changes runtime behavior

Required invariants:

- target identity is stable inside the project
- target `name` is user-facing and should be unique within one project
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
- a model run chooses one target definition at a time

## Separation Of Responsibilities

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

### Model run owns

- which target definition is being modeled
- which ROIs are included
- which observation/grouping semantics are used

## Workflow Semantics

## 1. Persist ROI

When a training ROI is saved into the project:

- the current window must have a real attached dataset context
- a `TrainingROI` object is created
- raw spectra are persisted
- provenance is persisted
- no target assignment is required at creation time

v3 rule:

- unlabeled ROIs are allowed in the project
- unlabeled ROIs are excluded from a model run for any target they do not have
- project-only mode may inspect existing ROIs, but must not create new ones
- newly created ROIs must not be derived from placeholder viewer state

## 2. Define target

When the user creates a new response variable such as `moisture` or `protein`,
the project must create a `TargetDefinition`.

Examples:

- `name = "moisture"`, `unit = "% w/w"`, `value_type = "float"`
- `name = "protein"`, `unit = "% w/w"`, `value_type = "float"`

## 3. Assign targets to ROI

When the user edits target values for an ROI:

- one `TargetAssignment` is created or updated per `(roi_id, target_id)`

Examples:

- ROI `cheese_A` may have:
  - moisture = 41.2
  - fat = 23.8
  - protein = 19.4

## 4. Build modeling inputs

When a model run is created:

1. choose one `TargetDefinition`
2. choose included ROIs
3. filter to ROIs that have an assignment for that target
4. build `X` from pixel-level raw ROI spectra
5. build `Y` by repeating the selected ROI-level target assignment across those ROI pixels
6. build grouping metadata from `roi_id`

v3 rule:

- one model run uses one selected target definition
- multi-target projects are allowed
- multi-target model fitting is out of scope for v3
- pixel-level spectral diversity is preserved in `X`
- validation and grouping semantics remain ROI-level

## 5. Review and edit targets

The UI may expose one target at a time for editing or many targets in a grid.
That is a UI decision.

The contract requirement is:

- target definitions and assignments must remain explicit project objects

The UI must not imply:

- that the project has only one universal anonymous `y` column

## Compatibility Rules

Modeling for a selected target must block or warn when:

- no ROIs are included
- included ROIs do not have assignments for the selected target
- target values are not valid for the target definition
- wavelength compatibility across included ROIs is broken
- ROI provenance shows incompatible source/calibration context for the intended sample-building rules
- grouping metadata cannot preserve ROI-level validation semantics

## v3 Scope Rules

v3 supports:

- project-owned persisted ROIs
- explicit target definitions
- one or more target assignments per ROI
- numeric scalar regression targets
- one selected target per model run
- pixel-level spectra as modeling observations
- ROI-level target semantics
- ROI-grouped validation semantics

v3 rejects:

- implicit anonymous `y_value` as the core project contract
- modeling multiple targets simultaneously in one model run
- category labels as first-class target values
- mixed incompatible wavelength axes inside one model run

## Persistence Requirements

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
- raw spectra storage
- ROI-local dark/white reference storage when the active calibration has embedded full-sensor references

Required creation-time invariants for newly created ROIs:

- `dataset_id` must reference a real attached dataset context
- `calibration_id` must reference the active calibration for that dataset
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

## v3 HDF5 Shape Basis

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
- ROI-local calibration reference snapshots, when present, are persisted inside `/rois/{roi_id}/calibration_refs/`
- target definitions are persisted independently from ROIs
- target assignments are persisted independently from both ROI identity and target identity

v3 note:

- a single `@y_value` stored on an ROI is not a target representation; the
  project reader rejects ROI groups that carry one

## Required UI Behavior

The UI must make these distinctions legible:

- saving an ROI creates a project-owned ROI object
- defining a target creates a named project-level response variable
- assigning a target value links an ROI to that target definition
- modeling chooses one target definition explicitly

The UI must not imply:

- that ROI names are sufficient identity
- that one built-in `y_value` field is the whole target model
- that provenance and modeling target are the same concept

## Out Of Scope For v3

These are intentionally deferred:

- classification labels as a target type
- multi-output model runs
- target inheritance or grouping
- assay-version tracking
- multiple active assignments for the same `(roi_id, target_id)` pair
- full target history/audit beyond latest assignment timestamp

## Open Questions After This Decision

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
- the v3 UI edits one selected target at a time

Out of scope for v3:

- multi-column target editing
- simultaneous side-by-side editing of many targets
- a dedicated multi-target spreadsheet editor

### 3. Sample-unit interaction

Decision:

- the domain may support multiple sample-unit styles later
- v3 modeling uses pixel-level spectral observations with ROI-level target semantics
- v3 validation groups observations by `roi_id`

Out of scope for v3:

- `roi_mean` as the primary fitting contract
- `roi_feature` modeling
- exposing a user-facing sample-unit selector in the first modeling UI

## Acceptance Criteria

This contract is implemented correctly when:

- persisting a training ROI creates a stable ROI identity inside the project
- ROI provenance records source dataset and calibration context
- the project can define more than one named target variable
- one ROI can carry assignments for more than one target
- a model run chooses one target definition explicitly
- modeling inputs preserve pixel-level spectral diversity while keeping target semantics ROI-level
- unlabeled ROIs remain valid project objects but are excluded from modeling for targets they do not have
- targets are represented by explicit definitions and assignments, never by a
  single anonymous `y_value`
