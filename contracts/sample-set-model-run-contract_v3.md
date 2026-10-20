# Sample Set Model Run Contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.0; v3 is the default and only supported project schema
- Scope: sample-building, observation semantics, grouped validation, model-run identity, v3 PLS/PCR fitting behavior

## Goal

Define the concrete contract for:

- how project ROIs and targets become modeling inputs
- what the observation unit is in v3
- how grouping and validation work
- what a model run must store to be reproducible

This contract exists to prevent two common errors:

- collapsing ROI spectra to one mean and losing useful spectral diversity
- treating ROI pixels as fully independent samples and inflating confidence through pseudo-replication

## Core Decisions

- v3 preserves pixel-level spectra as modeling observations
- v3 targets remain ROI-level
- v3 grouping and validation are ROI-level
- one model run uses one selected target definition
- one model run uses one preprocessing recipe
- working spectral arrays should default to `float32` unless a concrete backend
  contract requires higher precision
- v3 may support PLS and PCR, but both obey the same sample-set contract

## Problem This Contract Solves

For your workflow, a training ROI carries:

- many pixel spectra
- one or more ROI-level target assignments

PLS benefits from diverse spectra within an ROI.
But if those pixels are treated as independent scientific samples, the model
will overstate performance and become statistically misleading.

The contract must therefore separate:

- observation unit
- target level
- grouping unit
- validation unit

## Contract Summary

There are two core objects:

1. `SampleSet`
   The reproducible description of how `X`, `Y`, and grouping are built.
2. `ModelRun`
   The reproducible description of how a model was fit and validated from one sample set.

## Domain Objects

## 1. SampleSet

Represents one modeling dataset built from project objects.

Required fields:

- `sample_set_id`
- `roi_ids`
- `target_id`
- `observation_unit`
- `target_level`
- `grouping_unit`
- `validation_unit`
- `weighting_policy`
- `reporting_level`
- `roi_aggregation`
- `preprocessing_recipe_id`
- `dataset_ids`
- `created_at`

Allowed `observation_unit` values in v3:

- `pixel`

Allowed `target_level` values in v3:

- `roi`

Allowed `grouping_unit` values in v3:

- `roi`

Allowed `validation_unit` values in v3:

- `roi`

Allowed `weighting_policy` values in v3:

- `per_pixel_unweighted`

Allowed `reporting_level` values in v3:

- `roi`

Allowed `roi_aggregation` values in v3:

- `mean`

Required invariants:

- one sample set references exactly one selected `target_id`
- one sample set references exactly one preprocessing recipe
- `observation_unit = pixel` in v3
- `target_level = roi` in v3
- `grouping_unit = roi` in v3
- `validation_unit = roi` in v3
- `weighting_policy = per_pixel_unweighted` in v3
- `reporting_level = roi` in v3
- `roi_aggregation = mean` in v3
- all included ROIs must have compatible wavelength axes
- all included ROIs must have valid assignments for the selected target
- `dataset_ids` must contain the `dataset_id` of every included ROI that still
  exists in the project; service-built sample sets record exactly the ordered
  unique dataset ids of the included ROIs at build time
- a persisted `dataset_ids` entry without a matching live ROI may remain when
  included ROIs were later deleted (the stale sample-set policy)

## 2. ModelRun

Represents one fitted and validated model.

Required fields:

- `model_run_id`
- `sample_set_id`
- `model_type`
- `hyperparameters`
- `cv_spec`
- `metrics`
- `artifact_refs`
- `created_at`

Optional fields:

- `name`: display name, a mutable label

Allowed `model_type` values in v3:

- `PLS`
- `PCR`

Required invariants:

- one model run references exactly one sample set
- one model run references exactly one target definition through that sample set
- one model run references exactly one preprocessing recipe through that sample set
- model metrics must be interpreted under ROI-grouped validation rules

## Modeling Semantics

## 1. Build `X`

`X` is built from pixel spectra extracted from included ROIs.

For each included ROI:

- load its raw spectra
- apply the selected preprocessing recipe after the required signal-domain conversion
- append each pixel spectrum to `X`

Memory-aware v3 rules:

- multi-ROI sample-set construction should use one batched/single-file-open
  path rather than reopening the project once per ROI
- raw ROI spectra remain the scientific truth and should stay in their stored
  raw dtype until the transformation boundary
- the transformed working matrix may be `float32`

v3 meaning:

- pixel-level spectral diversity is preserved
- no ROI mean collapse is used for the primary fitting contract

## 2. Build `Y`

`Y` is built from ROI-level target assignments.

For each included ROI:

- look up the assignment for the selected target
- repeat that ROI-level value across the ROI's pixel observations

v3 meaning:

- observations are pixel-level
- response semantics are ROI-level

## 3. Build `groups`

`groups` is built from `roi_id`.

For each included ROI:

- repeat its `roi_id` for every pixel observation coming from that ROI

This is not optional in v3.

## 4. Validation semantics

Cross-validation must be ROI-grouped.

Allowed v3 behavior:

- `GroupKFold`
- `LeaveOneGroupOut`

Required invariant:

- no pixel from a held-out ROI may appear in the training fold for that validation split

Required fold assignment:

- folds depend only on the recorded sample set: the splitter labels each ROI
  with its position in the sample set's `roi_ids`, never with the `roi_id`
  value itself
- `roi_id` values are random, and `GroupKFold` deals groups of equal size to
  folds in label order; labelling by id would make the cross-validated
  metrics of a repeated analysis (same data, new ids) differ

## 5. Reporting semantics

Because observation unit and target level differ, reporting must be explicit.

Required v3 rule:

- validation semantics are ROI-level even though predictions are generated for pixels

Recommended v3 behavior:

- aggregate held-out pixel predictions back to ROI-level for primary performance reporting
- allow pixel-level diagnostic views as secondary analysis

Primary reported metrics should therefore be interpreted as ROI-grouped metrics, not naive per-pixel metrics.

v3 reporting decision:

- held-out pixel predictions are aggregated back to ROI-level by arithmetic mean
- primary validation metrics are computed from one aggregated prediction per held-out ROI
- pixel-level prediction distributions remain secondary diagnostics only

## Weighting Rule

v3 must not silently bias the fit toward larger ROIs without an explicit decision.

v3 weighting decision:

- fitting uses `per_pixel_unweighted`
- larger ROIs therefore contribute more pixel observations to `X`
- this weighting policy must be explicit in the saved sample set and model run context
- validation and reporting must still remain ROI-grouped

Out of scope for v3:

- equal-ROI weighting
- configurable ROI weighting strategies
- per-pixel uncertainty weighting

## Blocking Conditions

Model fitting for a sample set must block or warn when:

- no ROIs are selected
- selected ROIs do not have assignments for the chosen target
- ROI wavelength axes are incompatible
- ROI calibration or signal-domain requirements are incompatible with the chosen recipe/model contract
- grouping metadata cannot be constructed by ROI
- the chosen CV strategy would violate ROI grouping
- the current workflow is in project-only mode without a real attached dataset
  for any action that depends on current viewer dataset context

## Persistence Requirements

`SampleSet` and `ModelRun` must be persisted inside the project file.

## Required persisted sample-set information

- `sample_set_id`
- included `roi_ids`
- selected `target_id`
- `observation_unit`
- `target_level`
- `grouping_unit`
- `validation_unit`
- `weighting_policy`
- `reporting_level`
- `roi_aggregation`
- `preprocessing_recipe_id`
- `dataset_ids`
- `created_at`

## Required persisted model-run information

- `model_run_id`
- `sample_set_id`
- `model_type`
- hyperparameters
- CV specification
- metrics
- references to fitted artifacts
- `created_at`

Memory-aware persistence rules:

- persisted ROI-level diagnostics and prediction-like numeric arrays may be
  stored as `float32`
- references to fitted artifacts may point to heavy numeric arrays, but those
  arrays must remain lazily hydratable during project navigation and
  summary-only UI flows

Stateful preprocessing persistence rules:

- `artifact_refs/prediction_model/preprocessing_state` is required when the
  referenced recipe enables `MSC` or `EMSC`
- MSC state contains the fitted training reference spectrum
- EMSC state contains its fitted reference spectrum, wavelength weights,
  optional interference basis, method, and polynomial order
- these arrays and parameters are prediction-critical model artifacts, not
  mutable recipe fields
- reopening reconstructs fitted transformers from this state and must never
  call `fit()` on prediction data
- missing or malformed required state blocks prediction during preflight

## v3 HDF5 Shape Basis

v3 requires project-internal groups shaped like:

```text
/sample_sets/{sample_set_id}/
/model_runs/{model_run_id}/
```

Minimum expectation:

- the sample set stores the observation/target/grouping semantics explicitly
- the model run stores metrics and artifacts explicitly
- no model run relies on an implicit “current table column” or “current viewer state” as its target contract

## Required UI Behavior

The UI must make these distinctions legible:

- fitting uses one selected target definition
- pixel spectra are used as observations
- ROI grouping is used for validation
- weighting policy is `per_pixel_unweighted`
- primary reporting level is ROI
- ROI prediction aggregation uses arithmetic mean
- reported performance is not naive independent-pixel CV

The UI must not imply:

- that `roi_mean` is the default v3 fitting contract
- that each pixel is an independent scientific sample with its own independent target
- that one ROI with many pixels automatically represents many independent reference samples

## Out Of Scope For v3

These are intentionally deferred:

- `roi_mean` as the primary model-fitting contract
- `roi_feature` sample sets
- multi-target model runs
- user-selectable sample-unit switching in the first modeling UI
- equal-ROI weighting
- configurable weighting strategies
- mixed observation-unit comparisons in one comparison table

## Open Questions After This Decision

### 1. Weighting implementation detail

Decision:

- v3 uses `per_pixel_unweighted`

Out of scope for v3:

- equal-ROI weighting
- selectable weighting strategies

### 2. ROI-level reporting aggregation

Decision:

- held-out pixel predictions are aggregated to ROI-level by arithmetic mean for primary reporting

Out of scope for v3:

- median aggregation
- selectable aggregation strategies
- robust aggregation policy tuning in the UI

## Acceptance Criteria

This contract is implemented correctly when:

- a sample set explicitly records `observation_unit = pixel`
- a sample set explicitly records `weighting_policy = per_pixel_unweighted`
- a sample set explicitly records `reporting_level = roi`
- a sample set explicitly records `roi_aggregation = mean`
- the selected target is explicit and project-defined
- `Y` is derived from ROI-level assignments
- validation groups observations by `roi_id`
- a model run stores its sample-set and CV semantics explicitly
- reported model performance cannot be mistaken for naive independent-pixel validation
