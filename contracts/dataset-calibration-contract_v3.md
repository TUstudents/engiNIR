# Dataset reference correction contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.5; the file format `enginir-1.5` is the
  only format engiNIR reads and writes (`project-schema-contract_v3.md`)
- Scope: raw `.eiff` import, reference correction ownership, input-contract compatibility, prediction-time reuse

## Goal

Define the concrete contract for:

- dataset identity at raw `.eiff` import time
- reference correction ownership
- signal conversion to reflectance or absorbance
- how saved models express their required input contract
- how prediction datasets are checked and processed against that requirement

This contract exists to prevent ambiguity between:

- dataset-owned reference correction state
- model-owned preprocessing
- prediction-time pipeline reuse

## Core decision

Reference correction ownership belongs to the dataset at EIFF import time.

That means:

- reference correction is created when a raw dataset is imported
- reference correction is attached to the dataset reference, not to the project as a free-floating global
- a model does not own the calibration dataset's dark/white arrays as universal assets
- a model owns the required input contract that future datasets must satisfy
- a prediction run records which dataset-local reference correction context was actually applied

## Problem this contract solves

The scientist workflow begins with:

1. load raw `.eiff`
2. choose dark and white references
3. convert to reflectance or absorbance
4. inspect and draw ROIs
5. fit a model
6. apply the model to a new raw `.eiff`

Without a reference correction contract, the system cannot answer reliably:

- what exactly the calibration spectra represented
- whether a prediction dataset is compatible
- whether a model expects Raw counts (DN), reflectance, or absorbance
- whether a prediction used dataset-local reference correction or stale calibration-session state

## Contract summary

There are three separate but linked objects:

1. `DatasetRef`
   Owns dataset identity.
2. `CalibrationSpec`
   Owned by the dataset.
3. `SignalDomainContract`
   Owned by the model.

The model references an input contract.
The prediction run references both:

- the model
- the reference correction spec actually applied to the prediction dataset

Memory-aware v3 rule:

- reference correction/input-contract checks should operate on metadata whenever possible
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

## Domain objects

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
- compatibility comparisons normalize case and whitespace; there are no
  aliases for earlier values

`source_modality` vocabulary in v3:

- an opaque acquisition-modality token; compatibility comparisons use exact
  string equality with no normalization
- writers propagate `source_modality` from loader metadata when present and
  default to `NIR` otherwise
- tokens must remain stable within one project, because input contracts copy
  the calibration dataset's modality and prediction compares them verbatim

Recommended future fields:

- `sensor_name`
- `acquisition_id`
- `content_hash`
- `wavelength_hash`

## 2. CalibrationSpec

Represents the reference correction context attached to one dataset.

Required fields:

- `calibration_id`
- `dataset_id`
- `mode`
- `output_signal_unit`
- `created_at`

Declared conditional fields (always written; populated only when the mode's
executable-context rules below require or permit them, otherwise "not
defined": NaN, an empty string or an empty array,
`project-schema-contract_v3.md`, Core and display data):

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
- each ROI extraction and prediction run must reference exactly one applied reference correction spec
- `output_signal_unit` must be explicit
- `dark_white_reference` mode requires compatible dark and white references or explicit fallback levels
- `Absorbance` is derived from corrected reflectance, never directly from Raw counts (DN)
- `CalibrationSpec` must be created only for a real dataset context, not for placeholder viewer state

Executable reference correction-context rule in v3:

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
  scalar-level reference correction

v3 rule:

- the UI may expose reference correction edits, but every committed reference correction state must become an explicit `CalibrationSpec`, not mutable anonymous widget state
- replacing reference correction in v3 updates the dataset's single active reference correction spec rather than keeping multiple active variants
- a prediction does not write its reference correction (in-memory by default,
  `prediction-runtime-contract_v3.md` §Result handling); before low-level
  persistence writes an explicit prediction reference correction, the executable
  content must be compared against existing dataset-local reference corrections and
  reused when it matches

Reference correction content identity in v3:

- compare `mode`, canonical `output_signal_unit`, scalar levels, reference
  presence, reference provenance, and reference data/content hashes
- identical scalar `black_level` and `white_level` reference corrections for the same
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
- calibration-side conversion (ROI spectra transformed into the model's input contract during model fitting) must also count clipped values and log a
  warning with the totals when any value was clipped; persisting calibration-side
  clip telemetry is not required in v3
- future contract versions may move the clip range into explicit
  `CalibrationSpec` fields, but v3 implementations must not clip silently

## 3. SignalDomainContract

Represents the input contract a saved model requires before preprocessing.

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

Allowed `band_mapping_policy` value in v3:

- `exact`

v3 band-mapping rule:

- no interpolation, regridding, or automatic band remapping is supported
- `exact` assumes band count and band order are already aligned by the
  import/modeling workflow
- `band_mapping_policy` is a strict compatibility declaration in v3, not an
  executable interpolation strategy
- the field is kept, with its one value, for a future remapping contract
  (calibration transfer); any other value is refused on read
- defer interpolation/regridding to a later contract version

Required invariants:

- every saved model references exactly one input contract
- a prediction dataset must satisfy the input contract before model preprocessing is applied
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
- saved model artifacts persist their own wavelength axis for diagnostics, but
  prediction does not fail solely because that metadata differs numerically

Executable field semantics in v3:

- a prediction reference correction satisfies the contract's reference correction requirement
  when its `mode` equals `required_calibration_mode`, or when its `mode` is
  `precalibrated_import` and `calibrated_input_allowed` is true
- `calibrated_input_allowed` therefore permits an already-corrected import to
  substitute for the reference correction mode the model was fitted with, provided the reference correction can
  still produce `required_signal_unit`; current writers set it true exactly
  when the calibration reference correction mode was `precalibrated_import`
- `requires_dark_ref` and `requires_white_ref` are additional preflight gates
  for non-`precalibrated_import` prediction reference corrections: when set, the
  reference correction must carry the corresponding reference context (embedded
  reference arrays, reference source provenance, or scalar levels); both
  checks are skipped when the prediction reference correction is `precalibrated_import`
- `wavelength_tolerance_nm` is the tolerance of the prediction-time
  wavelength warning (`prediction-runtime-contract_v3.md`, Input contract at runtime); it never gates prediction; current writers emit `0.0`
  (any difference above float noise warns)

## Separation of responsibilities

### Dataset owns

- raw source identity
- imported wavelengths
- reference correction specs created from that dataset

### CalibrationSpec owns

- how that dataset was transformed from raw sensor values into a defined signal unit

### Model owns

- the required input contract
- the preprocessing recipe to apply after reference correction
- fitting and validation provenance

### Prediction run owns

- which dataset was predicted
- which reference correction spec was applied to that dataset
- which model was used
- which preprocessing recipe was applied

## Workflow semantics

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

## 2. Select reference correction

When the user chooses how a dataset is corrected (dark/white references,
detector levels, or raw counts), the system must create or update an explicit
`CalibrationSpec` for that dataset.

Signal units of a reference correction:

- the project always stores the spectra of every ROI as recorded. Raw
  counts are stored with what converts them: the ROI's reference snapshot,
  or the reference correction's scalar `black_level`/`white_level`. Raw
  counts (DN), reflectance and absorbance are views of those counts,
  computed when needed, never a stored replacement; a `precalibrated_import`
  ROI stores the corrected values it was imported with
- for `dark_white_reference`, `output_signal_unit` is the working signal: the
  signal Explore starts with and the one `Project.fit` fits in when no other
  is chosen. It is not asked when a dataset is added; it is `Absorbance`.
  Reflectance and absorbance can both be produced from the counts at any
  time; the shown signal is chosen in Explore and in the fit dialog
- for `raw_dn`, `output_signal_unit` is `DN`: without references or levels,
  the counts give no other signal
- for `precalibrated_import`, `output_signal_unit` states what the file holds
  (`Reflectance` or `Absorbance`); this is the only case in which the UI asks
  for it, and Raw counts (DN) are unavailable because there are no counts
- `viewable_signal_units` gives these signals per reference correction; Explore offers
  only them for the shown dataset

The active viewer state must reference that reference correction spec.

The reference correction is chosen when a dataset is added to a project
(`project-dataset-attachment-contract_v3.md`, Adding a dataset). The dark and
white references of `dark_white_reference` come from one of these sources:

- **another file**: averaged over all its frames, giving one reference
  spectrum per row and band
- **a frame range of the dataset itself**, such as a white tile or
  shutter-closed frames scanned in the same image: averaged over those frames
- **another dataset of the project**: its reference arrays are copied
  into a new `CalibrationSpec` of the new dataset; reference corrections stay
  dataset-local

Required behavior:

- `dark_ref_source` and `white_ref_source` record the source, for example
  `white.eiff` or `tray03.eiff frames 0-20`; for copied references they keep
  the original source
- before a reference is accepted, the UI shows its mean spectrum and the share
  of saturated values
- scalar `black_level`/`white_level` remain the fallback without references

Replacing the active reference correction of a dataset that already has ROIs
applies to ROIs drawn afterwards only:

- existing ROIs keep the reference correction and the reference snapshot they were
  drawn with
- the UI must say so before the change, with the number of affected ROIs

## 3. Draw ROIs

When a calibration ROI is persisted, the ROI provenance must record:

- `dataset_id`
- `calibration_id`
- signal unit used at extraction time
- preview context as informational provenance if shown in the UI

v3 storage rule:

- raw ROI spectra are persisted as truth
- reference correction and signal unit are stored as provenance, not as a replacement for raw ROI values

## 4. Fit model

When a model is fit, the system must persist:

- the reference corrections (`calibration_id`) of the calibration datasets,
  through the calibration set's ROIs
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
2. interpret them in the saved dataset/reference correction context
3. transform into the required input contract
4. apply the preprocessing recipe
5. fit the estimator

## 5. Predict on new raw `.eiff`

When a new dataset is predicted (nothing is written to the project by
default; `prediction-runtime-contract_v3.md` §Result handling):

1. describe the file as a `DatasetRef` in memory (a registered file uses its
   stored one)
2. create/select the `CalibrationSpec` in memory (a registered file reuses its
   stored active reference correction read-only)
3. if dataset-associated reference correction defaults already exist, reuse them by default
4. compare the dataset + reference correction state to the model's `SignalDomainContract`
5. if compatible, allocate the in-memory result maps
6. load and process bounded chunks
7. for each chunk, convert to the required signal unit
8. for each chunk, apply the saved preprocessing recipe
9. for each chunk, run prediction and the applicability diagnostics and
   write the outputs into the result maps
10. return the result and write one log entry; only low-level persistence
    writes the `DatasetRef`, the `CalibrationSpec` and a `PredictionRun`
    with both model and applied reference correction references

v3 default behavior:

- when a prediction dataset already has associated reference correction defaults, the system reuses them by default
- when no dataset-associated defaults exist, the user must select or define the required reference correction explicitly
- the user may override them explicitly
- with low-level persistence, explicit overrides are content-deduplicated
  against existing dataset-local reference corrections before a new reference correction id is
  created
- the system must never silently reuse calibration-dataset dark/white references as prediction defaults
- stateful fitted preprocessing must either be reproduced from persisted
  calibration-side state or be blocked explicitly on reopened prediction runs
- prediction must expose visible progress for preflight, loading, reference correction,
  preprocessing, predicting, writing, and finalization phases
- large prediction datasets must not require complete in-memory copies of raw
  input, corrected signal, preprocessed signal, full `y_pred`, and full
  `spatial_map` at the same time
- full-sensor dark/white references must be applied in the active chunk context
  rather than tiled over the complete prediction cube
- flattened prediction spectra may use `DatasetRef.cube_shape` to align
  reference correction references when an explicit spatial shape is not supplied
- reflectance clipping counts must be preserved in prediction runtime metadata
  or an equivalent persisted run summary

## Compatibility rules

Prediction must be blocked or warned when any of these fail:

- source format incompatible with model contract
- source modality incompatible with model contract
- required signal unit cannot be produced
- required reference correction mode is not satisfied
- required dark reference is missing
- required white reference is missing
- `n_bands` mismatch
- band mapping policy would be violated
- output dtype other than canonical `float32`

## v3 compatibility policy

For v3, prediction compatibility should be strict about executable contracts and
pragmatic about wavelength metadata.

Required behavior:

- same band count
- same required signal unit
- same expected reference correction mode

Rejected in v3:

- automatic band interpolation
- automatic wavelength remapping
- silent use of calibration dark/white references on new datasets

## Persistence requirements

`DatasetRef` and `CalibrationSpec` must be persisted inside the project file.

v3 persistence decision:

- dataset identity travels with the project
- reference correction provenance travels with the project
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

## Required persisted reference correction information

- `calibration_id`
- `dataset_id`
- `mode`
- `output_signal_unit`
- `created_at`
- `black_level` / `white_level` (nullable, per the executable-context rules)
- dark reference presence and source (nullable, per the executable-context rules)
- white reference presence and source (nullable, per the executable-context rules)

## Required persisted model input-contract information

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
- reflectance clipping telemetry when reference correction clipped values

## v3 HDF5 shape basis

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

- reference corrections live only under their owning dataset; a root `/calibration`
  group is rejected by the project reader
- extra fields, including a root `/models` group, are ignored with a
  warning; canonical project models remain exclusively under
  `/model_runs`, and no legacy field is treated as an alias or migration source

## Required UI behavior

The UI must make these distinctions legible:

- importing a raw dataset creates dataset identity
- choosing dark/white defines dataset reference correction
- reusing dataset-associated reference correction defaults is a dataset-local action
- fitting a model freezes a required input contract
- predicting on a new dataset uses that dataset's own reference correction context, not whatever happened to be active during calibration

The UI must not imply:

- that model preprocessing alone is enough to reproduce the pipeline
- that reference correction is a global project setting independent of dataset
- that calibration dark/white references are silently reused for prediction
- that stateful preprocessing can be recreated from prediction data without
  persisted calibration-side state

## Out of scope for v3

These are intentionally deferred:

- calibration transfer between instruments
- wavelength interpolation contracts
- multi-sensor compatibility contracts
- migration of pre-v3 project calibration data
- non-EIFF raw import semantics beyond the same abstract shape

## Open questions after this decision

These remain open even with reference correction ownership fixed.

### 1. Prediction-time reference correction UX

Decision:

- when a new prediction dataset is imported, the system reuses dataset-associated defaults if they already exist

Still to specify:

- how the UI exposes override versus reuse
- whether reuse is automatic on import or suggested as the preferred default selection

### 2. Dataset and reference correction storage location

Decision:

- `DatasetRef` and `CalibrationSpec` are persisted inside the project file

Resolved:

- a moved or renamed file is mapped back with `Locate File...`. It is
  verified against the `DatasetRef` and the stored raw ROI pixels, then
  `source_path` is updated (`project-dataset-attachment-contract_v3.md`).

Still to specify:

- whether any optional external cache or registry is allowed as a secondary optimization

### 3. Multiple reference correction specs per dataset

Decision:

- one dataset has only one active reference correction spec in v3

Out of scope for v3:

- reference correction history
- multiple saved reference correction variants
- named reference correction presets per dataset

Possible later-version extension:

- later contract versions may support multiple reference correction variants, but this is explicitly not part of v3

## Acceptance criteria

This contract is implemented correctly when:

- importing raw `.eiff` creates a stable dataset reference
- reference correction is explicit and attached to that dataset
- dataset identity and reference correction records are stored inside the project file
- ROI provenance records the applied dataset reference correction context
- saved models declare the required input contract
- prediction on a new dataset records the dataset-local reference correction context actually used
- prediction is blocked when the new dataset cannot satisfy the saved model's input contract
- repeated explicit reference correction choices with identical executable content reuse
  an existing dataset-local reference correction id
- `float32` is the only supported v3 prediction output dtype
- reflectance clipping telemetry is logged and persisted or surfaced
- model artifact wavelength metadata is not a prediction-time hard failure when
  band count/source/reference correction/recipe checks pass
