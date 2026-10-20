# Project Schema Contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.0; v3 is the default and only supported project schema
- Scope: project-level persistence structure, object ids, relationships, required HDF5 groups, v3 compatibility rules

## Goal

Define the concrete project-level schema contract that ties together:

- dataset identity and calibration
- persisted ROIs
- target definitions and assignments
- preprocessing recipes
- sample sets
- model runs
- prediction runs

This contract exists so the project file becomes the persistent scientific
workspace, not just a loose container of ROI blobs and model artifacts.

## Core Decisions

- the project file is the authoritative persistence boundary for v3
- all first-class scientific objects are persisted inside the project file
- all first-class objects have stable ids
- references between objects must use ids, not user-facing names
- schema version is explicit and required
- metadata-only project hydration must be possible without loading heavy ROI or
  model/prediction artifact arrays into RAM
- the project schema must reflect the frozen contracts already defined in:
  - `dataset-calibration-contract_v3.md`
  - `project-dataset-attachment-contract_v3.md`
  - `roi-target-contract_v3.md`
  - `sample-set-model-run-contract_v3.md`
  - `prediction-runtime-contract_v3.md`

## Problem This Contract Solves

Without a project schema contract, the system cannot reliably answer:

- which dataset a ROI came from
- which calibration context was used for ROI extraction
- which target a model was fit against
- which preprocessing recipe and signal-domain contract a model requires
- which prediction run used which model and which dataset-local calibration

The result would be a project file that stores data, but does not preserve the
scientific relationships needed for reproducibility.

## Project Definition

A v3 project is a versioned file that persists the minimum scientific objects
required for:

- ROI-centric data collection
- explicit target assignment
- reproducible model fitting
- reproducible prediction

The project file must be sufficient to reconstruct:

- dataset references used by the project
- active calibration records for those datasets
- persisted ROIs and their provenance
- target definitions and assignments
- preprocessing recipes
- sample sets
- model runs
- prediction runs

v3 workflow note:

- the project file may exist as an empty scientific workspace before any
  dataset is attached to a viewer window
- creating a project must not implicitly create a dataset, calibration, or ROI
- viewer workflow rules for project-only versus dataset-attached mode are
  defined in `project-dataset-attachment-contract_v3.md`

## First-Class Objects In v3

The following are first-class persisted objects:

- `ProjectMeta`
- `DatasetRef`
- `CalibrationSpec`
- `TrainingROI`
- `TargetDefinition`
- `TargetAssignment`
- `PreprocessingRecipe`
- `SignalDomainContract`
- `SampleSet`
- `ModelRun`
- `PredictionRun`

These objects are persisted independently and connected by ids.

## Empty Project Rule

An empty project is valid in v3.

That means a newly created project may legally contain:

- `project_meta`
- empty top-level object groups
- no datasets
- no calibrations
- no ROIs
- no targets
- no sample sets
- no model runs
- no predictions

Required invariant:

- project creation initializes workspace structure only
- project creation must not infer scientific data from placeholder viewer state

## Object Identity Rules

### 1. Project identity

Required fields:

- `project_id`
- `schema_version`
- `created_at`
- `updated_at`

Required invariant:

- every project file declares exactly one schema version

### 2. Stable ids

Every first-class object must have a stable id persisted in the project.

Examples:

- `dataset_id`
- `calibration_id`
- `roi_id`
- `target_id`
- `recipe_id`
- `signal_domain_id`
- `sample_set_id`
- `model_run_id`
- `prediction_run_id`

Required invariant:

- object references inside the project must use ids, not user-facing labels

### 3. Labels versus ids

User-facing names remain mutable and non-authoritative.

Examples:

- ROI `name`
- target `name`
- model display name

Required invariant:

- changing a user-facing label must not change object identity

### 4. Preprocessing recipe identity and schema

A persisted preprocessing recipe is an executable scientific value object.
Its stable `recipe_id` is the `{recipe_id}` path component under `/recipes`;
the id is not duplicated inside the recipe payload.

Every persisted recipe requires exactly these current fields:

- `filter`: one of `none`, `sg_smooth`, `sg_d1`, or `sg_d2`
- `sg_window`: odd integer greater than or equal to 3
- `sg_polyorder`: integer greater than or equal to 1 and less than `sg_window`
- `snv`: boolean
- `msc`: boolean
- `emsc`: boolean
- `detrend`: boolean
- `detrend_order`: non-negative integer

All eight fields are required even when a switch is false or `filter` is
`none`. Python defaults are construction conveniences, not missing-field
defaults for persisted projects. The execution order is Savitzky-Golay,
SNV, MSC, EMSC, then detrending for the enabled steps.

Recipe timestamps and arbitrary additional recipe parameters are not part of
the v3 recipe schema. They follow the project extra-field policy: warn and
discard. A future executable parameter requires a new explicit contract rather
than being silently retained.

Recipe identity is content-stable. Once `/recipes/{recipe_id}` exists, saving
the same normalized content is idempotent, but changing any executable field
under that id is forbidden. Changed preprocessing requires a new `recipe_id` so
existing sample sets, model runs, and prediction runs cannot be reinterpreted.

### 5. Content immutability of stable ids

The recipe rule generalizes: a stable id must not silently change its
scientific content. Writers enforce, per object:

- re-saving an existing id with identical persisted content is an idempotent
  no-op and always allowed
- re-saving an existing id with changed content is rejected with a logged
  error, except for the mutable label fields (`TrainingROI.name`,
  `TargetDefinition.name`, `ModelRun.name`), which may change freely without
  affecting identity; `ModelRun.name` is optional
- changed scientific content requires a new id

Explicit exceptions, each already governed by its own contract rule:

- `CalibrationSpec`: replacing a dataset's single active calibration
  overwrites in place by design (`dataset-calibration-contract_v3.md`)
- `TargetAssignment`: updating the value of an existing `(roi_id, target_id)`
  pair is the documented editing model (`roi-target-contract_v3.md`)
- prediction staging: `initialize`/`write`/`finalize`/`discard` mutate the
  staged prediction group as part of the lifecycle status contract; the
  one-shot prediction writer is immutable like other objects

## Required Object Relationships

These references are mandatory in v3.

### Dataset and calibration

- `CalibrationSpec.dataset_id -> DatasetRef.dataset_id`

v3 rule:

- one dataset has exactly one active calibration spec in v3
- identical executable calibration content for the same dataset should resolve to
  one calibration id instead of repeated duplicate records

v3 identity rule:

- calibration ids must be unique across the whole project, not only within one
  dataset subtree, because ROIs and prediction runs reference calibrations by
  `calibration_id` alone
- the reader rejects a project where two dataset subtrees persist the same
  calibration id
- a persisted dataset `active_calibration_id` must reference an existing
  calibration owned by that dataset; the reader validates this during
  hydration

### ROI relationships

- `TrainingROI.dataset_id -> DatasetRef.dataset_id`
- `TrainingROI.calibration_id -> CalibrationSpec.calibration_id`

### Target relationships

- `TargetAssignment.roi_id -> TrainingROI.roi_id`
- `TargetAssignment.target_id -> TargetDefinition.target_id`

### Sample-set relationships

- `SampleSet.roi_ids -> TrainingROI.roi_id[]`
- `SampleSet.target_id -> TargetDefinition.target_id`
- `SampleSet.preprocessing_recipe_id -> PreprocessingRecipe.recipe_id`
- `SampleSet.dataset_ids -> DatasetRef.dataset_id[]`

v3 coverage invariant:

- `dataset_ids` must contain the `dataset_id` of every included ROI that
  resolves in the project; validation rejects a sample set whose live ROI
  references a dataset missing from `dataset_ids`

### Model-run relationships

- `ModelRun.sample_set_id -> SampleSet.sample_set_id`
- `ModelRun.signal_domain_id -> SignalDomainContract.signal_domain_id`

v3 note:

- the selected target and recipe are reached through the sample set

### Prediction-run relationships

- `PredictionRun.dataset_id -> DatasetRef.dataset_id`
- `PredictionRun.calibration_id -> CalibrationSpec.calibration_id`
- `PredictionRun.model_run_id -> ModelRun.model_run_id`
- `PredictionRun.signal_domain_id -> SignalDomainContract.signal_domain_id`
- `PredictionRun.preprocessing_recipe_id -> PreprocessingRecipe.recipe_id`

## Required Top-Level Project Structure

v3 requires these top-level groups:

```text
/
  project_meta/
  datasets/
  targets/
  target_assignments/
  recipes/
  signal_domains/
  rois/
  sample_sets/
  model_runs/
  predictions/
```

Required invariant:

- all top-level groups exist even if some are empty

## Hydration And Memory Rules

The schema must support two distinct read patterns in v3:

1. metadata hydration
2. heavy array hydration

Required v3 rule:

- project/browser/inspector/workspace open must be metadata-first by default

That means the canonical reader must support loading:

- ROI metadata without materializing `/rois/{roi_id}/spectra`
- ROI metadata without materializing `/rois/{roi_id}/calibration_refs/*`
- model/prediction metadata without materializing nested `artifact_refs` arrays

Examples of heavy arrays that may remain disk-backed until explicitly needed:

- ROI `spectra`
- ROI-local dark/white `calibration_refs`
- model CV outputs
- fitted model numeric arrays
- prediction `y_pred`
- prediction `spatial_map`
- prediction `preview_map` or other display-oriented map pyramids

Required invariant:

- metadata-only inspection or mutation must not require those arrays to be read
  eagerly

Allowed v3 implementation pattern:

- one metadata-first reader plus explicit lazy-load helpers for heavy arrays
- large prediction artifact helpers that can read bounded slices/chunks without
  hydrating the complete output array
- integrity checks over stored array shape and dtype during metadata-first
  open, using HDF5 dataset metadata only; the reader must reject ROI groups
  whose stored spectra, wavelengths, or calibration-reference shapes
  contradict their metadata, without materializing the arrays

## v3 HDF5 Shape Basis

The minimum object layout is:

```text
project.h5
|
+-- /project_meta/
|   +-- @project_id
|   +-- @schema_version
|   +-- @created_at
|   +-- @updated_at
|
+-- /datasets/{dataset_id}/
|   +-- @source_path
|   +-- @source_format
|   +-- @source_modality
|   +-- @cube_shape
|   +-- @n_bands
|   +-- @import_timestamp
|   +-- wavelengths
|   +-- /loader_metadata/
|   +-- /calibrations/{calibration_id}/
|       +-- @mode
|       +-- @output_signal_unit
|       +-- @black_level
|       +-- @white_level
|       +-- @dark_ref_source
|       +-- @white_ref_source
|       +-- dark_ref_data
|       +-- white_ref_data
|       +-- @created_at
|
+-- /rois/{roi_id}/
|   +-- @dataset_id
|   +-- @calibration_id
|   +-- @name
|   +-- @source_file
|   +-- @n_pixels
|   +-- @created_at
|   +-- @spatial_bounds
|   +-- wavelengths
|   +-- spectra
|   +-- /calibration_refs/
|       +-- dark
|       +-- white
|   +-- /provenance/
|
+-- /targets/{target_id}/
|   +-- @name
|   +-- @unit
|   +-- @value_type
|   +-- @missing_value_policy
|   +-- @created_at
|
+-- /target_assignments/{target_id}/{roi_id}
|   +-- @value
|   +-- @assigned_at
|
+-- /recipes/{recipe_id}/
|   +-- @filter
|   +-- @sg_window
|   +-- @sg_polyorder
|   +-- @snv
|   +-- @msc
|   +-- @emsc
|   +-- @detrend
|   +-- @detrend_order
|
+-- /signal_domains/{signal_domain_id}/
|   +-- signal-domain fields / attrs
|
+-- /sample_sets/{sample_set_id}/
|   +-- @target_id
|   +-- @observation_unit
|   +-- @target_level
|   +-- @grouping_unit
|   +-- @validation_unit
|   +-- @weighting_policy
|   +-- @reporting_level
|   +-- @roi_aggregation
|   +-- @preprocessing_recipe_id
|   +-- roi_ids
|   +-- dataset_ids
|   +-- @created_at
|
+-- /model_runs/{model_run_id}/
|   +-- @sample_set_id
|   +-- @signal_domain_id
|   +-- @model_type
|   +-- @name                optional display name
|   +-- hyperparameters
|   +-- cv_spec
|   +-- metrics
|   +-- fitted artifacts
|   +-- @created_at
|
+-- /predictions/{prediction_run_id}/
    +-- @dataset_id
    +-- @calibration_id
    +-- @model_run_id
    +-- @signal_domain_id
    +-- @preprocessing_recipe_id
    +-- @created_at
    +-- /artifact_refs/
        +-- @artifact_write_status (absent or `complete`; `writing` only while staged)
        +-- @source_path
        +-- @output_dtype
        +-- @prediction_shape
        +-- @spatial_shape
        +-- @spatial_map_shape
        +-- @output_axis
        +-- @output_index
        +-- @output_names
        +-- @chunk_size
        +-- @n_chunks
        +-- @completed_chunks
        +-- @clip_low_count
        +-- @clip_high_count
        +-- y_pred
        +-- spatial_map (optional legacy/materialized view)
        +-- preview_map (optional)
```

Prediction runtime metadata (source path, dtype, shapes, output-axis
selection, chunking counters, clipping telemetry) is persisted as attributes on
`/artifact_refs/`, alongside the artifact datasets. There is no separate
`/runtime/` subgroup in v3. The complete attribute table is documented in
`docs/hdf5_format.md`; the attributes shown above are the representative
required set.

`y_pred` is the canonical prediction data array. `spatial_map_shape` supplies
the lossless spatial interpretation. A separate full `spatial_map` dataset is
not required when it would only duplicate `y_pred`; readers must accept both
the canonical single-array layout and existing dual-array files.

## Required Schema Semantics

### 1. Project metadata

`/project_meta/` must store:

- project identity
- schema version
- timestamps

This is mandatory.

### 2. Dataset-local calibration storage

Calibration must live under the dataset that owns it.

v3 requirement:

- calibration is not a project-global free-floating object

Therefore:

- `/calibration/` at project root is not the target v3 contract shape

### 3. Explicit target storage

Target definitions and assignments must be independent objects.

v3 requirement:

- the schema must not depend on a single `@y_value` stored directly on each ROI as the core target model

### 4. Explicit sample-set semantics

Sample-set semantics must be stored directly.

v3 requirement:

- the schema must persist `observation_unit`, `target_level`, `grouping_unit`, `validation_unit`, `weighting_policy`, `reporting_level`, and `roi_aggregation`

### 5. Model runs are not just fitted arrays

Model artifacts alone are not enough.

v3 requirement:

- a model run must persist the references and semantics needed to interpret those artifacts
- reading a model run for navigation or summary display must not require eager
  hydration of all fitted numeric artifacts

### 6. Prediction provenance must be explicit

A prediction map is not meaningful without provenance.

v3 requirement:

- every prediction run must reference:
  - prediction dataset
  - applied calibration
  - model run
  - signal-domain contract
  - preprocessing recipe
- reading a prediction run for navigation or summary display must not require
  eager hydration of `y_pred` or `spatial_map`
- a completed prediction run must store runtime/artifact metadata sufficient to
  interpret output shape, dtype, target/unit, source path, and chunking strategy
  without reading the output datasets
- a completed multi-output prediction run must store output-axis metadata
  sufficient for the UI to select a target/output explicitly
- wavelength metadata may be stored for diagnostics, but completed prediction
  runs do not need wavelength proof to be considered valid
- a completed calibrated prediction run must store reflectance clipping telemetry
  when any values were clipped
- failed or partial prediction output must not be indistinguishable from a
  finalized prediction run

### 7. Prediction lifecycle status

The staged prediction write path persists an explicit lifecycle marker at
`/predictions/{prediction_run_id}/artifact_refs/@artifact_write_status`.

Allowed values:

- `writing` — artifact datasets are initialized and chunk writes are in
  progress; the run is not a valid finalized project state
- `complete` — finalization succeeded after all required artifact writes

Required semantics:

- an absent `artifact_write_status` attribute means `complete`; writers that
  persist an already-complete prediction run in one operation may omit it
- the staged write path sets `writing` at artifact initialization and sets
  `complete` only during finalization
- a failed or cancelled staged run is discarded by deleting the prediction
  group; v3 persists no `failed` or `cancelled` status value
- the reader must reject a project containing a prediction run marked
  `writing` with a logged error
- the reader must reject unknown `artifact_write_status` values with a logged
  error; unknown status values are not extras to warn about, because they
  change whether the run may be interpreted as finalized

## Required Serialization Rules

### Arrays

Persist as datasets:

- wavelengths
- spectra
- dark/white references
- ROI ids where array form is useful
- dataset ids where array form is useful
- fitted numeric artifacts
- prediction vectors/maps/previews

v3 hydration rule:

- the existence of these datasets does not imply they must be loaded during
  metadata-first project open

### Scalars and small metadata

Persist as attributes where practical:

- ids
- timestamps
- enums
- units
- modes
- simple numeric metadata

### Structured metadata

Persist in dedicated subgroups or JSON-encoded fields when structure is non-trivial.

Examples:

- `loader_metadata`
- `hyperparameters`
- `cv_spec`
- `metrics`

v3 requirement:

- structure must be reconstructable without relying on implicit Python objects or pickled state

## Schema Validation And Extra-Field Policy

The reader always interprets a project using the newest schema implemented by
the running engiNIR version.

Required read behavior:

1. scan the project structure before hydrating scientific objects
2. require the current `schema_version`
3. require every current top-level group and required object field
4. validate current enum, shape, dtype, identity, and relationship invariants
5. log and raise for a missing, malformed, or incompatible required field
6. log unknown extra groups, datasets, and attributes as warnings
7. ignore unknown extras rather than interpreting, copying, merging, or
   promoting them into current objects or artifact metadata

The scan applies at each schema object boundary: the root, project metadata,
and every identified dataset, calibration, ROI, target, assignment, recipe,
signal-domain contract, sample set, model run, and prediction run. A field that
the current object schema does not declare is an extra at that boundary.

Declared structured fields are values, not additional object-schema
boundaries. Their internal keys remain extensible where their owning contract
permits it. Examples include `loader_metadata`, `hyperparameters`, `cv_spec`,
`metrics`, `artifact_refs`, and ROI provenance preview details. Their contents
must not be promoted to fields on the owning object.

Extra-field tolerance is not a migration mechanism. In particular:

- an old file is accepted only when it already satisfies the complete newest
  schema and merely contains additional fields
- old field names are not aliases for current field names
- missing current fields are not inferred from legacy fields or defaults
- legacy objects are not transformed into current objects during project open
- current writes emit only the newest schema and do not preserve unknown extras
- warning messages identify the project location and extra field names; repeated
  validation of the same location should be coalesced to avoid log flooding

This policy allows harmless additions such as a root `/models` group to
coexist with the current graph while keeping `/model_runs` exclusively
authoritative.

The executable model of a ModelRun is stored exclusively in
`artifact_refs/prediction_model`. When the referenced recipe enables `MSC` or
`EMSC`, the payload must contain the required `preprocessing_state`; a payload
missing that state fails prediction preflight rather than refitting on
prediction data.

Calibration content hashes are derived comparison aids, not persisted
scientific fields in v3. Calibration identity is determined from the canonical
executable fields; implementations may compute hashes in memory to accelerate
comparison but must not make a stored hash authoritative over that content.

## Required Invariants

- schema version is explicit
- every persisted id is unique within its object namespace
- all inter-object references resolve to an existing object
- every ROI references an existing dataset and calibration
- every target assignment references an existing ROI and target
- every sample set references existing ROIs, a target, and a recipe
- every model run references an existing sample set and signal-domain contract
- every prediction run references an existing dataset, calibration, model run, signal-domain contract, and recipe
- every completed prediction run with multiple outputs stores explicit
  output-axis metadata
- every persisted prediction run carries an `artifact_write_status` that is
  absent or `complete`; a `writing` marker or an unknown status value is
  rejected on read
- wavelength metadata differences do not invalidate an otherwise completed
  prediction run

## Out Of Scope For v3

These are intentionally deferred:

- schema migration from pre-v3 project formats
- multiple active calibration variants per dataset
- project-external object registries as a requirement
- object-level history/version timelines
- distributed or split project storage
- non-HDF5 backends

## Open Questions After This Decision

These remain open after the v3 project-schema contract.

### 1. Exact encoding of structured metadata

Resolved for v3:

- structured fields (`loader_metadata`, `hyperparameters`, `cv_spec`,
  `metrics`, `artifact_refs`, ROI `provenance`) persist as nested HDF5 groups:
  scalars and short strings become attributes, arrays become datasets, nested
  mappings become subgroups; `None` uses the writer's sentinel encoding
- no JSON blobs and no pickled Python state are used for these fields
- `hyperparameters` holds constructor keyword arguments for the normalized
  `model_type` backend, so the estimator is reconstructable as
  `Backend(**hyperparameters)`
- `cv_spec` persists the ROI-grouped CV `type` (`GroupKFold` or
  `LeaveOneGroupOut`), the requested strategy, the fold and group counts, and
  the frozen grouping/validation/reporting semantics
- `metrics` persists ROI-aggregated primary metrics and `pixel_*`-prefixed
  secondary per-pixel diagnostics
- the concrete key tables are documented in `docs/hdf5_format.md` and must be
  kept in sync with the modeling service

### 2. Update semantics

Resolved for v3:

- updates follow the content-immutability rule in Object Identity Rules §5:
  idempotent re-saves and label-only changes overwrite in place; changed
  scientific content requires a new id
- calibration replacement, target-assignment value updates, and prediction
  staging are the documented overwrite-in-place exceptions
- no object is append-only in v3; history/version timelines remain out of
  scope

### 3. Compatibility policy

Resolved for v3:

- no runtime compatibility layer is part of the project reader
- pre-v3 project shapes fail loudly with explicit errors
- an otherwise valid newest-schema project may contain extra fields; the reader
  warns and ignores them rather than interpreting or merging them

## Acceptance Criteria

This contract is implemented correctly when:

- the project file declares a schema version explicitly
- dataset identity and calibration live inside the project file
- persisted ROIs are keyed by stable `roi_id`
- targets and target assignments are first-class persisted objects
- sample sets are first-class persisted objects
- model runs and prediction runs store explicit references to the objects they depend on
- calibration is dataset-owned and targets are explicit objects, not a
  project-global calibration or one anonymous ROI `y_value`
- duplicate explicit prediction calibrations can be detected by persisted
  calibration content identity
- prediction runtime metadata records output-axis and clipping telemetry needed
  to interpret disk-backed artifacts without loading them
