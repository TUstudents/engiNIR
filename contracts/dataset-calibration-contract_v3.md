# Dataset Calibration Contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.0; v3 is the default and only supported project schema
- Scope: raw `.eiff` import, calibration ownership, signal-domain compatibility, prediction-time reuse

## Goal

Define the concrete contract for:

- dataset identity at raw `.eiff` import time
- calibration ownership
- signal conversion to reflectance or absorbance
- how saved models express their required input signal domain
- how prediction datasets are checked and processed against that requirement

This contract exists to prevent ambiguity between:

- dataset-owned calibration state
- model-owned preprocessing
- prediction-time pipeline reuse

## Core Decision

Calibration ownership belongs to the dataset at EIFF import time.

That means:

- calibration is created when a raw dataset is imported
- calibration is attached to the dataset reference, not to the project as a free-floating global
- a model does not own the training dataset's dark/white arrays as universal assets
- a model owns the required signal-domain contract that future datasets must satisfy
- a prediction run records which dataset-local calibration context was actually applied

## Problem This Contract Solves

The scientist workflow begins with:

1. load raw `.eiff`
2. choose dark and white references
3. convert to reflectance or absorbance
4. inspect and draw ROIs
5. fit a model
6. apply the model to a new raw `.eiff`

Without a calibration contract, the system cannot answer reliably:

- what exactly the training spectra represented
- whether a prediction dataset is compatible
- whether a model expects raw DN, reflectance, or absorbance
- whether a prediction used dataset-local calibration or stale training-session state

## Contract Summary

There are three separate but linked objects:

1. `DatasetRef`
   Owns dataset identity.
2. `CalibrationSpec`
   Owned by the dataset.
3. `SignalDomainContract`
   Owned by the model run.

The model run references a signal-domain contract.
The prediction run references both:

- the model run
- the calibration spec actually applied to the prediction dataset

Memory-aware v3 rule:

- calibration/signal-domain checks should operate on metadata whenever possible
  and must not require eager loading of unrelated heavy model/prediction
  artifact arrays
- prediction execution, progress, chunking, practical memory discipline, and
  artifact handoff
  are governed by `prediction-runtime-contract_v3.md`

Workflow note:

- viewer/project attachment rules are defined in
  `project-dataset-attachment-contract_v3.md`
- `DatasetRef` and `CalibrationSpec` may only be created from real attached
  dataset context, never from placeholder viewer bootstrap state

## Domain Objects

## 1. DatasetRef

Represents one imported raw dataset.

Required fields:

- `dataset_id`
- `source_path`
- `source_format`
- `source_modality`
- `cube_shape`
- `n_bands`
- `wavelengths`
- `loader_metadata`
- `import_timestamp`

Required invariants:

- one imported raw dataset maps to one dataset id
- `source_format` for this contract version is at least `eiff`
- `cube_shape` must use the canonical viewer shape
- `len(wavelengths) == n_bands`
- `source_path` must be non-empty for newly imported viewer datasets
- placeholder viewer cubes must not be promoted into `DatasetRef`

`source_format` vocabulary in v3:

- a lowercase file-format key inferred from the source extension; known values
  are `eiff`, `hdf5`, `envi`, and `csv`; unknown extensions persist as their
  bare lowercase suffix
- compatibility comparisons normalize case and whitespace and treat legacy
  `eiff_raw` as `eiff`

`source_modality` vocabulary in v3:

- an opaque acquisition-modality token; compatibility comparisons use exact
  string equality with no normalization
- writers propagate `source_modality` from loader metadata when present and
  default to `NIR` otherwise
- tokens must remain stable within one project, because signal domains copy
  the training dataset's modality and prediction compares them verbatim

Recommended future fields:

- `sensor_name`
- `acquisition_id`
- `content_hash`
- `wavelength_hash`

## 2. CalibrationSpec

Represents the calibration context attached to one dataset.

Required fields:

- `calibration_id`
- `dataset_id`
- `mode`
- `output_signal_unit`
- `created_at`

Declared conditional fields (always part of the schema, nullable, populated
only when the mode's executable-context rules below require or permit them):

- `dark_ref_source`
- `white_ref_source`
- `dark_ref_data`
- `white_ref_data`
- `black_level`
- `white_level`

Allowed `mode` values in v3:

- `raw_dn`
- `dark_white_reference`
- `precalibrated_import`

Allowed `output_signal_unit` values in v3:

- `DN`
- `Reflectance`
- `Absorbance`

String comparison rule:

- persisted and user-entered signal-unit values must be normalized to canonical
  v3 values before compatibility checks
- casing differences such as `reflectance` versus `Reflectance` must not change
  compatibility semantics

Required invariants:

- every `CalibrationSpec` belongs to exactly one `DatasetRef`
- in v3, a dataset has exactly one active `CalibrationSpec`
- each ROI extraction and prediction run must reference exactly one applied calibration spec
- `output_signal_unit` must be explicit
- `dark_white_reference` mode requires compatible dark and white references or explicit fallback levels
- `Absorbance` is derived from calibrated reflectance, never directly from raw DN
- `CalibrationSpec` must be created only for a real dataset context, not for placeholder viewer state

Executable calibration-context rule in v3:

- `raw_dn` stores no dark/white reference context and must use
  `output_signal_unit = DN`
- `precalibrated_import` stores no dark/white reference context and must use
  `output_signal_unit = Reflectance` or `Absorbance`
- `dark_white_reference` must use exactly one executable context:
  - embedded `dark_ref_data` + `white_ref_data`
  - or scalar `black_level` + `white_level`
- `dark_white_reference` must not mix embedded arrays with scalar fallback levels
- `dark_ref_source` and `white_ref_source` are provenance-only fields in v3; if
  present, they must be paired and accompany embedded reference arrays, not a
  scalar-level calibration

v3 rule:

- the UI may expose calibration edits, but every committed calibration state must become an explicit `CalibrationSpec`, not mutable anonymous widget state
- replacing calibration in v3 updates the dataset's single active calibration spec rather than keeping multiple active variants
- before writing an explicit prediction calibration, the executable content must
  be compared against existing dataset-local calibrations and reused when it
  matches

Calibration content identity in v3:

- compare `mode`, canonical `output_signal_unit`, scalar levels, reference
  presence, reference provenance, and reference data/content hashes
- identical scalar `black_level` and `white_level` calibrations for the same
  dataset must not create duplicate `dark_white_reference` records
- identical embedded dark/white references for the same dataset must not create
  duplicate `dark_white_reference` records

Reflectance clipping rule:

- `dark_white_reference` conversion produces reflectance clipped to `[1e-6,
  1.0]` in v3
- the lower bound avoids invalid absorbance/log inputs; the upper bound caps
  over-bright pixels
- prediction runtime must count, log, and surface low-side and high-side clipped
  values
- training-side conversion (ROI spectra transformed into the model's signal
  domain during sample-set fitting) must also count clipped values and log a
  warning with the totals when any value was clipped; persisting training-side
  clip telemetry is not required in v3
- future contract versions may move the clip range into explicit
  `CalibrationSpec` fields, but v3 implementations must not clip silently

## 3. SignalDomainContract

Represents the input signal domain a saved model requires before preprocessing.

This is model-owned, not dataset-owned.

Required fields:

- `signal_domain_id`
- `input_format`
- `source_modality`
- `required_signal_unit`
- `required_calibration_mode`
- `requires_dark_ref`
- `requires_white_ref`
- `calibrated_input_allowed`
- `n_bands`
- `wavelengths`
- `wavelength_tolerance_nm`
- `band_mapping_policy`
- `output_dtype`
- `created_at`

Allowed `band_mapping_policy` values in v3:

- `exact`
- `reject`

v3 band-mapping rule:

- no interpolation, regridding, or automatic band remapping is supported
- both `exact` and `reject` assume band count and band order are already aligned
  by the import/modeling workflow
- `band_mapping_policy` is a strict compatibility declaration in v3, not an
  executable interpolation strategy
- `exact` and `reject` have identical runtime behavior in v3; current writers
  emit `exact`, readers accept both, and the distinction between the two
  values is reserved for a future remapping contract
- defer interpolation/regridding to a later contract version

Required invariants:

- every saved model run references exactly one signal-domain contract
- a prediction dataset must satisfy the signal-domain contract before model preprocessing is applied
- `required_signal_unit` is the signal state the estimator expects at fit and predict time
- `SignalDomainContract` does not own raw dark/white arrays
- `output_dtype` must be canonical `float32` in v3
- non-`float32` output dtypes are rejected during prediction preflight unless a
  future backend contract explicitly expands this field

Wavelength metadata rule:

- wavelengths are retained for display, diagnostics, and rough import/load
  sanity checks
- wavelength values are not repeatedly compared as a prediction-time hard gate
  after a dataset/model have been accepted
- saved model artifacts may persist their own wavelength axis for diagnostics or
  legacy compatibility, but prediction does not fail solely because that
  metadata differs numerically

Executable field semantics in v3:

- a prediction calibration satisfies the contract's calibration requirement
  when its `mode` equals `required_calibration_mode`, or when its `mode` is
  `precalibrated_import` and `calibrated_input_allowed` is true
- `calibrated_input_allowed` therefore permits an already-calibrated import to
  substitute for the trained calibration mode, provided the calibration can
  still produce `required_signal_unit`; current writers set it true exactly
  when the training calibration mode was `precalibrated_import`
- `requires_dark_ref` and `requires_white_ref` are additional preflight gates
  for non-`precalibrated_import` prediction calibrations: when set, the
  calibration must carry the corresponding reference context (embedded
  reference arrays, reference source provenance, or scalar levels); both
  checks are skipped when the prediction calibration is `precalibrated_import`
- `wavelength_tolerance_nm` is persisted diagnostic metadata with no
  executable gate in v3; current writers emit `0.0`; a future
  wavelength-compatibility contract must define its use before it gains
  runtime effect

## Separation Of Responsibilities

### Dataset owns

- raw source identity
- imported wavelengths
- calibration specs created from that dataset

### CalibrationSpec owns

- how that dataset was transformed from raw sensor values into a defined signal unit

### Model run owns

- the required input signal domain
- the preprocessing recipe to apply after calibration
- fitting and validation provenance

### Prediction run owns

- which dataset was predicted
- which calibration spec was applied to that dataset
- which model run was used
- which preprocessing recipe was applied

## Workflow Semantics

## 1. Import raw `.eiff`

On raw `.eiff` import, the system must create a `DatasetRef`.

Minimum outputs:

- `dataset_id`
- canonical cube representation
- wavelengths
- source metadata

Required v3 rule:

- import/attach paths in UI, CLI, and public API must use the same
  project-attachment validation semantics when binding a dataset-backed window
  to an existing project
- metadata-only attachment checks must avoid eager loading of ROI spectra or
  unrelated heavy prediction/model artifact arrays

## 2. Select calibration

When the user chooses dark, white, and output signal unit, the system must create
or update an explicit `CalibrationSpec` for that dataset.

The active viewer state must reference that calibration spec.

## 3. Draw ROIs

When a training ROI is persisted, the ROI provenance must record:

- `dataset_id`
- `calibration_id`
- signal unit used at extraction time
- preview context as informational provenance if shown in the UI

v3 storage rule:

- raw ROI spectra are persisted as truth
- calibration and signal unit are stored as provenance, not as a replacement for raw ROI values

## 4. Fit model

When a model is fit, the system must persist:

- the training dataset calibration context reference
- the model's `SignalDomainContract`
- the `PreprocessingRecipe`

Required fitted-preprocessing behavior:

- if the saved preprocessing recipe contains stateful fitted steps such as
  `MSC` or `EMSC`, the persisted model pipeline must also persist the fitted
  preprocessing state required to reproduce prediction after project reload
- if that fitted preprocessing state is not yet persisted, reopened-project
  prediction must reject those recipe variants explicitly rather than silently
  refitting them on prediction data

The fit pipeline is:

1. start from raw ROI spectra
2. interpret them in the saved dataset/calibration context
3. transform into the required signal domain
4. apply the preprocessing recipe
5. fit the estimator

## 5. Predict on new raw `.eiff`

When a new dataset is imported for prediction:

1. create `DatasetRef`
2. create/select `CalibrationSpec` for that dataset
3. if dataset-associated calibration defaults already exist, reuse them by default
4. compare the dataset + calibration state to the model's `SignalDomainContract`
5. if compatible, initialize disk-backed prediction artifacts
6. load and process bounded chunks
7. for each chunk, convert to the required signal unit
8. for each chunk, apply the saved preprocessing recipe
9. for each chunk, run prediction and write outputs incrementally
10. persist/finalize a `PredictionRun` with both model and applied calibration references

v3 default behavior:

- when a prediction dataset already has associated calibration defaults, the system reuses them by default
- when no dataset-associated defaults exist, the user must select or define the required calibration explicitly
- the user may override them explicitly
- explicit overrides are content-deduplicated against existing dataset-local
  calibrations before a new calibration id is created
- the system must never silently reuse training-dataset dark/white references as prediction defaults
- stateful fitted preprocessing must either be reproduced from persisted
  training-side state or be blocked explicitly on reopened prediction runs
- prediction must expose visible progress for preflight, loading, calibration,
  preprocessing, predicting, writing, and finalization phases
- large prediction datasets must not require complete in-memory copies of raw
  input, calibrated signal, preprocessed signal, full `y_pred`, and full
  `spatial_map` at the same time
- full-sensor dark/white references must be applied in the active chunk context
  rather than tiled over the complete prediction cube
- flattened prediction spectra may use `DatasetRef.cube_shape` to align
  calibration references when an explicit spatial shape is not supplied
- reflectance clipping counts must be preserved in prediction runtime metadata
  or an equivalent persisted run summary

## Compatibility Rules

Prediction must be blocked or warned when any of these fail:

- source format incompatible with model contract
- source modality incompatible with model contract
- required signal unit cannot be produced
- required calibration mode is not satisfied
- required dark reference is missing
- required white reference is missing
- `n_bands` mismatch
- band mapping policy would be violated
- output dtype other than canonical `float32`

## v3 Compatibility Policy

For v3, prediction compatibility should be strict about executable contracts and
pragmatic about wavelength metadata.

Required behavior:

- same band count
- same required signal unit
- same expected calibration mode

Rejected in v3:

- automatic band interpolation
- automatic wavelength remapping
- silent use of training dark/white references on new datasets

## Persistence Requirements

`DatasetRef` and `CalibrationSpec` must be persisted inside the project file.

v3 persistence decision:

- dataset identity travels with the project
- calibration provenance travels with the project
- no external registry is required for contract compliance

This contract therefore requires the following information to be serializable
inside the project file.

## Required persisted dataset information

- `dataset_id`
- `source_path`
- `source_format`
- `source_modality`
- `cube_shape`
- `n_bands`
- `wavelengths`
- `loader_metadata`
- `import_timestamp`

## Required persisted calibration information

- `calibration_id`
- `dataset_id`
- `mode`
- `output_signal_unit`
- `created_at`
- `black_level` / `white_level` (nullable, per the executable-context rules)
- dark reference presence and source (nullable, per the executable-context rules)
- white reference presence and source (nullable, per the executable-context rules)

## Required persisted model signal-domain information

- `signal_domain_id`
- `input_format`
- `source_modality`
- `required_signal_unit`
- `required_calibration_mode`
- `requires_dark_ref`
- `requires_white_ref`
- `calibrated_input_allowed`
- `n_bands`
- `wavelengths`
- `wavelength_tolerance_nm`
- `band_mapping_policy`
- `output_dtype`
- `created_at`

## Required persisted prediction provenance

- `prediction_run_id`
- `prediction_dataset_id`
- `applied_calibration_id`
- `model_run_id`
- `signal_domain_id`
- `preprocessing_recipe_id`
- artifact metadata sufficient to interpret disk-backed prediction outputs
  without hydrating the full arrays
- runtime metadata sufficient to audit chunking/progress behavior, including
  chunk axis, chunk size, output dtype, and spatial shape where available
- output-axis metadata sufficient to select a target/output when the model emits
  more than one prediction value per spectrum
- reflectance clipping telemetry when calibration clipped values

## v3 HDF5 Shape Basis

v3 requires project-internal groups shaped like:

```text
/datasets/{dataset_id}/
/datasets/{dataset_id}/calibrations/{calibration_id}/
/model_runs/{model_run_id}/
/model_runs/{model_run_id}/artifact_refs/prediction_model/
/predictions/{prediction_run_id}/
  /artifact_refs/
    y_pred
    spatial_map
    preview_map
```

The canonical object layout and field tables are owned by
`project-schema-contract_v3.md` and documented concretely in
`docs/hdf5_format.md`; the field lists in this contract must stay consistent
with them.

v3 note:

- calibrations live only under their owning dataset; a root `/calibration`
  group is rejected by the project reader
- extra fields, including a root `/models` group, are ignored with a
  warning; canonical project model runs remain exclusively under
  `/model_runs`, and no legacy field is treated as an alias or migration source

## Required UI Behavior

The UI must make these distinctions legible:

- importing a raw dataset creates dataset identity
- choosing dark/white defines dataset calibration
- reusing dataset-associated calibration defaults is a dataset-local action
- fitting a model freezes a required signal domain
- predicting on a new dataset uses that dataset's own calibration context, not whatever happened to be active during training

The UI must not imply:

- that model preprocessing alone is enough to reproduce the pipeline
- that calibration is a global project setting independent of dataset
- that training dark/white references are silently reused for prediction
- that stateful preprocessing can be recreated from prediction data without
  persisted training-side state

## Out of Scope For v3

These are intentionally deferred:

- calibration transfer between instruments
- wavelength interpolation contracts
- multi-sensor compatibility contracts
- migration of pre-v3 project calibration data
- non-EIFF raw import semantics beyond the same abstract shape

## Open Questions After This Decision

These remain open even with calibration ownership fixed.

### 1. Prediction-time calibration UX

Decision:

- when a new prediction dataset is imported, the system reuses dataset-associated defaults if they already exist

Still to specify:

- how the UI exposes override versus reuse
- whether reuse is automatic on import or suggested as the preferred default selection

### 2. Dataset and calibration storage location

Decision:

- `DatasetRef` and `CalibrationSpec` are persisted inside the project file

Still to specify:

- whether any optional external cache or registry is allowed as a secondary optimization
- how project-internal dataset records map to imported files that are later moved or renamed

### 3. Multiple calibration specs per dataset

Decision:

- one dataset has only one active calibration spec in v3

Out of scope for v3:

- calibration history
- multiple saved calibration variants
- named calibration presets per dataset

Possible later-version extension:

- later contract versions may support multiple calibration variants, but this is explicitly not part of v3

## Acceptance Criteria

This contract is implemented correctly when:

- importing raw `.eiff` creates a stable dataset reference
- calibration is explicit and attached to that dataset
- dataset identity and calibration records are stored inside the project file
- ROI provenance records the applied dataset calibration context
- saved models declare the required input signal domain
- prediction on a new dataset records the dataset-local calibration context actually used
- prediction is blocked when the new dataset cannot satisfy the saved model's signal-domain contract
- repeated explicit calibration choices with identical executable content reuse
  an existing dataset-local calibration id
- `float32` is the only supported v3 prediction output dtype
- reflectance clipping telemetry is logged and persisted or surfaced
- model artifact wavelength metadata is not a prediction-time hard failure when
  band count/source/calibration/recipe checks pass
