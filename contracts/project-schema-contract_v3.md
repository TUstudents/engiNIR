# Project schema contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.0; v3 is the default and only supported project schema
- Scope: project-level persistence structure, object ids, relationships, required HDF5 groups, v3 compatibility rules

## Goal

Define the concrete project-level schema contract that ties together:

- dataset identity and reference correction
- persisted ROIs
- target definitions and assignments
- preprocessing recipes
- calibration sets
- models
- prediction runs

This contract exists so the project file becomes the persistent scientific
workspace, not just a loose container of ROI blobs and model artifacts.

## Core decisions

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

## Problem this contract solves

Without a project schema contract, the system cannot reliably answer:

- which dataset a ROI came from
- which reference correction context was used for ROI extraction
- which target a model was fit against
- which preprocessing recipe and input contract a model requires
- which prediction run used which model and which dataset-local reference correction

The result would be a project file that stores data, but does not preserve the
scientific relationships needed for reproducibility.

## Project definition

A v3 project is a versioned file that persists the minimum scientific objects
required for:

- ROI-centric data collection
- explicit target assignment
- reproducible model fitting
- reproducible prediction

The project file must be sufficient to reconstruct:

- dataset references used by the project
- active reference correction records for those datasets
- persisted ROIs and their provenance
- target definitions and assignments
- preprocessing recipes
- calibration sets
- models
- prediction runs

v3 workflow note:

- the project file may exist as an empty scientific workspace before any
  dataset is attached to a viewer window
- creating a project must not implicitly create a dataset, reference correction, or ROI
- viewer workflow rules for project-only versus dataset-attached mode are
  defined in `project-dataset-attachment-contract_v3.md`

## First-class objects in v3

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

## Empty project rule

An empty project is valid in v3.

That means a newly created project may legally contain:

- `project_meta`
- empty top-level object groups
- no datasets
- no reference corrections
- no ROIs
- no targets
- no calibration sets
- no models
- no predictions

Required invariant:

- project creation initializes workspace structure only
- project creation must not infer scientific data from placeholder viewer state

## Object identity rules

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

Every persisted recipe requires these eight fields:

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
defaults for persisted projects.

Two groups of optional fields were added by the release that introduced
SIMCA preprocessing (planned 1.3). They are written only when they change
what the recipe executes, so every recipe stored before keeps its content
and its id:

- `sg_edges`: `drop` or `interp`, how a Savitzky-Golay filter treats the
  first and last ⌊`sg_window`/2⌋ bands. `drop` removes them (the spectra get
  K − 2⌊`sg_window`/2⌋ bands, as in SIMCA); `interp` keeps them, extrapolated
  by the polynomial of the first and last window. Written only when `filter`
  is a Savitzky-Golay filter and the value is `drop`; a stored recipe without
  the field means `interp`. New recipe objects default to `drop`; a recipe
  given as a mapping is the persisted form, where an absent field means
  `interp`. Without a Savitzky-Golay filter the field has no effect: it is
  normalized to `drop` (the default) and not written, so `from_dict` yields
  `interp` only for a Savitzky-Golay recipe without the field.
- `asls` (boolean), `asls_lambda` (positive real), `asls_p` (real,
  0 < p < 0.5), `asls_iterations` (integer ≥ 1): asymmetric least-squares
  baseline correction. Written, all four, only when `asls` is true; a stored
  recipe without them means `asls = false`. With `asls` false the other
  three have no effect and are normalized to their defaults (10000, 0.001,
  10). `asls_lambda` and `asls_p` are finite and normalized to float
  (float64 attributes), `asls_iterations` to int, before the id is computed,
  so `10000` and `10000.0` give the same recipe and id.

A third group was added by engiNIR 1.4 (band selection), again written
only when set:

- `bands_keep`, `bands_exclude`: lists of inclusive `[low, high]` ranges on
  the recipe's axis after the Savitzky-Golay step (wavelengths in nm; for
  data without wavelengths, the band numbers of the dataset, i.e. the
  input index 0 … K − 1, so dropping edges does not renumber). A band is
  selected when it lies in a keep range (or no keep list is given) and in
  no exclude range. λ is the axis value as stored (float32 for an input contract) converted to float64; a band lies in a range when
  low − τ ≤ λ ≤ high + τ, with τ = 10⁻³ times the smallest band spacing of
  the axis (10⁻⁶ for a single band), so a typed bound equal to a stored
  float32 wavelength matches it. Normalized before the id is computed,
  independent of any axis: each pair ordered (low ≤ high), the ranges
  sorted by low, a range merged with the previous one when its low ≤ the
  previous high (exact float64 comparison; ranges one band apart stay
  separate), values finite and float; an empty list is the same as an
  absent field and is not written. Stored as float64 datasets of shape
  (n, 2). A recipe without them selects every band.

Reading: an explicit `sg_edges = interp` equals an absent field; `sg_edges`
without a Savitzky-Golay filter and AsLS parameters with `asls = false` are
accepted and normalized; an `sg_edges` value other than `drop` or `interp`
is rejected; `asls = true` with any of the three parameters missing is a
missing-field error, never filled from defaults. "The same content" below
compares the normalized stored recipe with the normalized new one.

The execution order is AsLS, Savitzky-Golay, SNV, MSC, EMSC, detrending,
then band selection for the enabled steps. AsLS fits, for every spectrum z, the
baseline b that minimizes Σᵢ wᵢ(zᵢ − bᵢ)² + λ Σᵢ ((Δ²b)ᵢ)², i.e. solves
(W + λDᵀD) b = W z, with Δ² (D) the second difference over the band index
(unit spacing; wavelength values are not used) and weights from the
previous solve (start: all 1). Exactly `asls_iterations` solves are done
(no early stop); the step returns z − b. When the signal unit is
`Absorbance` (peaks point up, the baseline lies under the spectrum) the
weights are p where zᵢ > bᵢ and 1 − p otherwise; for `Reflectance` and `DN`
(absorption bands are dips, the baseline lies over the spectrum) the step
is AsLS of −z, negated: weights p where zᵢ < bᵢ and 1 − p otherwise. A
recipe with AsLS cannot be applied without a signal unit. The signal unit
is the model's (its input contract), so fitting and prediction use the same
direction. λ is applied as given, not rescaled with the band count; the
stiffness grows roughly with K⁴, so a camera with more bands needs a larger
λ. AsLS runs on the full input axis (before Savitzky-Golay); input and
output are float32, the solver works in float64. AsLS and Savitzky-Golay
act on each spectrum alone (stateless).

Band selection is the last step: SNV, MSC and EMSC references and Detrend
see every band after Savitzky-Golay, and the selection keeps the selected
columns (stateless, no effect on the MSC or EMSC state). A range that
selects no band of the axis is logged as a warning, once per range and
axis in a session (fit, prediction or replay); a recipe that selects no
band is refused when it is applied, and a fit refuses a given (fixed)
number of components larger than the selected band count. The selected
axis may have gaps.

A recipe with `sg_edges = drop` or a band selection changes the number of
bands: its output axis is the input axis without the dropped and the
unselected bands (`output_wavelengths`, with the input band indices
`output_band_indices`).
SNV, MSC and EMSC references and Detrend use the axis after Savitzky-Golay
(fewer bands only with `sg_edges = drop`); the model (coefficients,
loadings, `x_mean`, `x_std`, `rotations`, stored `wavelengths` of
`prediction_model` and `diagnostics`), K in the DModX, s0, DCrit and DModXPS
formulas (as SIMCA's K = 86 of 100 bands for a 15-point window),
preprocessed ROI spectrum summaries and plots, and prediction use the
output axis after band selection; the input contract keeps the input bands.

Projects with these fields need the release that introduced them or later:
an older version would discard them as unknown fields and replay a
different recipe (with `drop` its band check fails; the band ranges are
datasets, which 1.3 skips without a message, but its band check fails;
without, AsLS would be skipped silently).

Recipe timestamps and other additional recipe parameters are not part of
the v3 recipe schema. They follow the project extra-field policy: warn and
discard. A future executable parameter requires a new explicit contract rather
than being silently retained.

Recipe identity is content-stable. Once `/recipes/{recipe_id}` exists, saving
the same normalized content is idempotent, but changing any executable field
under that id is forbidden. Changed preprocessing requires a new `recipe_id` so
existing calibration sets, models, and prediction runs cannot be reinterpreted.

### 5. Content immutability of stable ids

The recipe rule generalizes: a stable id must not silently change its
scientific content. Writers enforce, per object:

- re-saving an existing id with identical persisted content is an idempotent
  no-op and always allowed
- re-saving an existing id with changed content is rejected with a logged
  error, except for the mutable label fields (`TrainingROI.name`,
  `TargetDefinition.name`, `ModelRun.name`), which may change freely without
  affecting identity; `ModelRun.name` is optional
- `TrainingROI.sample` (optional) is also a label, but it is set, changed or
  removed only by its own writer (`save_roi_sample`), in place; re-saving an
  ROI never changes or removes it (a re-save whose `sample` differs from the
  stored one is changed content)
- `TrainingROI.use` (`calibration` | `prediction`; a missing `use` and
  `training` read `calibration`) is set when the ROI is created; afterwards
  only `save_roi_use` changes it, in place; a re-save whose `use` differs
  from the stored one is changed content, where a missing `use`, `training`
  and `calibration` count as the same content (an older ROI re-saves
  unchanged)
- changed scientific content requires a new id

Explicit exceptions, each already governed by its own contract rule:

- `CalibrationSpec`: replacing a dataset's single active reference correction
  overwrites in place by design (`dataset-calibration-contract_v3.md`)
- `TargetAssignment`: updating the value of an existing `(roi_id, target_id)`
  pair is the documented editing model (`roi-target-contract_v3.md`)
- prediction staging: `initialize`/`write`/`finalize`/`discard` mutate the
  staged prediction group as part of the lifecycle status contract; the
  one-shot prediction writer is immutable like other objects
- `DatasetRef.source_path`: the path is a location, not identity. It may be
  updated after `Locate File...` has verified the file against the dataset
  and against the stored raw pixels of its ROIs
  (`project-dataset-attachment-contract_v3.md`, Missing and moved files)
- `TargetAssignment` removal: deleting the assignment of a `(roi_id,
  target_id)` pair returns it to "no value" (`roi-target-contract_v3.md`)

## Required object relationships

These references are mandatory in v3.

### Dataset and reference correction

- `CalibrationSpec.dataset_id -> DatasetRef.dataset_id`

v3 rule:

- one dataset has exactly one active reference correction spec in v3
- identical executable reference correction content for the same dataset should resolve to
  one reference correction id instead of repeated duplicate records

v3 identity rule:

- reference correction ids must be unique across the whole project, not only within one
  dataset subtree, because ROIs and prediction runs refer to reference corrections by
  `calibration_id` alone
- the reader rejects a project where two dataset subtrees persist the same
  reference correction id
- a persisted dataset `active_calibration_id` must reference an existing
  reference correction owned by that dataset; the reader validates this during
  hydration

### ROI relationships

- `TrainingROI.dataset_id -> DatasetRef.dataset_id`
- `TrainingROI.calibration_id -> CalibrationSpec.calibration_id`

### Target relationships

- `TargetAssignment.roi_id -> TrainingROI.roi_id`
- `TargetAssignment.target_id -> TargetDefinition.target_id`

### Calibration-set relationships

- `SampleSet.roi_ids -> TrainingROI.roi_id[]`
- `SampleSet.target_id -> TargetDefinition.target_id`
- `SampleSet.preprocessing_recipe_id -> PreprocessingRecipe.recipe_id`
- `SampleSet.dataset_ids -> DatasetRef.dataset_id[]`

v3 coverage invariant:

- `dataset_ids` must contain the `dataset_id` of every included ROI that
  resolves in the project; validation rejects a calibration set whose live ROI
  references a dataset missing from `dataset_ids`

### Model relationships

- `ModelRun.sample_set_id -> SampleSet.sample_set_id`
- `ModelRun.signal_domain_id -> SignalDomainContract.signal_domain_id`
- `ModelRun.fit_spec/folds/groups` has one entry per ROI of its calibration set's
  `roi_ids`, and its values are well-formed fold-group indices
  (`sample-set-model-run-contract_v3.md` §7); a run that breaks this is
  rejected when the project is read

v3 note:

- the selected target and recipe are reached through the calibration set

### Prediction-run relationships

- `PredictionRun.dataset_id -> DatasetRef.dataset_id`
- `PredictionRun.calibration_id -> CalibrationSpec.calibration_id`
- `PredictionRun.model_run_id -> ModelRun.model_run_id`
- `PredictionRun.signal_domain_id -> SignalDomainContract.signal_domain_id`
- `PredictionRun.preprocessing_recipe_id -> PreprocessingRecipe.recipe_id`

## Required top-level project structure

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

## Hydration and memory rules

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
  whose stored spectra, wavelengths, or reference correction-reference shapes
  contradict their metadata, without materializing the arrays

## v3 HDF5 shape basis

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
|   +-- @sample              optional sample label (mutable)
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
|   +-- @sg_edges, @asls, @asls_lambda, @asls_p, @asls_iterations (optional, §4)
|   +-- bands_keep, bands_exclude (optional float64 (n, 2) datasets, §4)
|
+-- /signal_domains/{signal_domain_id}/
|   +-- input-contract fields / attrs
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
|   +-- fit_spec            absent only for runs before engiNIR 1.2
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

## Required schema semantics

### 1. Project metadata

`/project_meta/` must store:

- project identity
- schema version
- timestamps

This is mandatory.

### 2. Dataset-local reference correction storage

Reference correction must live under the dataset that owns it.

v3 requirement:

- reference correction is not a project-global free-floating object

Therefore:

- `/calibration/` at project root is not the target v3 contract shape

### 3. Explicit target storage

Target definitions and assignments must be independent objects.

v3 requirement:

- the schema must not depend on a single `@y_value` stored directly on each ROI as the core target model

### 4. Explicit calibration-set semantics

Calibration-set semantics must be stored directly.

v3 requirement:

- the schema must persist `observation_unit`, `target_level`, `grouping_unit`, `validation_unit`, `weighting_policy`, `reporting_level`, and `roi_aggregation`

### 5. Models are not just fitted arrays

Model artifacts alone are not enough.

v3 requirement:

- a model must persist the references and semantics needed to interpret those artifacts
- reading a model for navigation or summary display must not require eager
  hydration of all fitted numeric artifacts

### 6. Prediction provenance must be explicit

A prediction map is not meaningful without provenance.

v3 requirement:

- every prediction run must reference:
  - prediction dataset
  - applied reference correction
  - model
  - input contract
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
- a completed corrected prediction run must store reflectance clipping telemetry
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
- a failed or canceled staged run is discarded by deleting the prediction
  group; v3 persists no `failed` or `cancelled` status value
- the reader must reject a project containing a prediction run marked
  `writing` with a logged error
- the reader must reject unknown `artifact_write_status` values with a logged
  error; unknown status values are not extras to warn about, because they
  change whether the run may be interpreted as finalized

## Required serialization rules

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

## Schema validation and extra-field policy

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
and every identified dataset, reference correction, ROI, target, assignment, recipe,
input contract, calibration set, model, and prediction run. A field that
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

Reference correction content hashes are derived comparison aids, not persisted
scientific fields in v3. Reference correction identity is determined from the canonical
executable fields; implementations may compute hashes in memory to accelerate
comparison but must not make a stored hash authoritative over that content.

## Required invariants

- schema version is explicit
- every persisted id is unique within its object namespace
- all inter-object references resolve to an existing object
- every ROI references an existing dataset and reference correction
- every target assignment references an existing ROI and target
- every calibration set references existing ROIs, a target, and a recipe
- every model references an existing calibration set and input contract
- every prediction run references an existing dataset, reference correction, model, input contract, and recipe
- every completed prediction run with multiple outputs stores explicit
  output-axis metadata
- every persisted prediction run carries an `artifact_write_status` that is
  absent or `complete`; a `writing` marker or an unknown status value is
  rejected on read
- wavelength metadata differences do not invalidate an otherwise completed
  prediction run

## Out of scope for v3

These are intentionally deferred:

- schema migration from pre-v3 project formats
- multiple active reference correction variants per dataset
- project-external object registries as a requirement
- object-level history/version timelines
- distributed or split project storage
- non-HDF5 backends

## Open questions after this decision

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
- `cv_spec` persists what ran: the ROI-grouped CV `type`
  (`RoundRobinGroups`, `GroupKFold` or `LeaveOneGroupOut`), the requested
  strategy, the fold and group counts, and the frozen
  grouping/validation/reporting semantics
- `/predictions/{prediction_run_id}/` is written only by the low-level
  writers when a script asks for it; the facade and the viewer do not store
  predictions or register predicted scans as datasets
  (`prediction-runtime-contract_v3.md`, Result handling)
- `fit_spec` persists what was requested: fold rule and fold groups (with
  the group index of every ROI as used), pixel subsample (count, seed,
  checksum) and DModX exclusion
  (`sample-set-model-run-contract_v3.md` §7); a run without it is a legacy
  run with the defined legacy specification
- scaling of X and Y is a constructor argument of the backend and lives in
  `hyperparameters` (`x_scaling`, `y_scaling`; legacy runs: `scale`)
- `metrics` persists ROI-aggregated primary metrics and `pixel_*`-prefixed
  secondary per-pixel diagnostics
- the concrete key tables are documented in `docs/hdf5_format.md` and must be
  kept in sync with the modeling service

### 2. Update semantics

Resolved for v3:

- updates follow the content-immutability rule in Object identity rules §5:
  idempotent re-saves and label-only changes overwrite in place; changed
  scientific content requires a new id
- reference correction replacement, target-assignment value updates, and prediction
  staging are the documented overwrite-in-place exceptions
- an ROI's `sample` label is a label-only change: it is set, changed or
  removed in place by `save_roi_sample`, never the ROI's other content
- no object is append-only in v3; history/version timelines remain out of
  scope

### 3. Compatibility policy

Resolved for v3:

- no runtime compatibility layer is part of the project reader
- pre-v3 project shapes fail loudly with explicit errors
- an otherwise valid newest-schema project may contain extra fields; the reader
  warns and ignores them rather than interpreting or merging them
- `fit_spec` is added within v3 without a schema-version change: an
  engiNIR before 1.2 ignores it with a warning; such a reader still predicts
  correctly with the stored model artifacts, but refitting a run with a
  `fit_spec` is not supported there (it fails on the unknown `x_scaling`
  and `y_scaling` hyperparameters, or ignores the fit specification), and it
  cannot rename such a run: it would save the run without its `fit_spec`,
  which the content-immutability rule (§5) refuses as changed content
- the ROI `sample` attribute is added within v3 without a schema-version
  change: an engiNIR before 1.2 ignores it with a warning (its folds are
  per ROI); re-saving such an ROI there is refused as changed content, as
  above

## Acceptance criteria

This contract is implemented correctly when:

- the project file declares a schema version explicitly
- dataset identity and reference correction live inside the project file
- persisted ROIs are keyed by stable `roi_id`
- targets and target assignments are first-class persisted objects
- calibration sets are first-class persisted objects
- models and prediction runs store explicit references to the objects they depend on
- reference correction is dataset-owned and targets are explicit objects, not a
  project-global reference correction or one anonymous ROI `y_value`
- duplicate explicit prediction reference corrections can be detected by persisted
  reference correction content identity
- prediction runtime metadata records output-axis and clipping telemetry needed
  to interpret disk-backed artifacts without loading them
