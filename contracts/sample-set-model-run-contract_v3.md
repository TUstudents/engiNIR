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
- `fit_spec`: the fit specification (§7). Every run written by engiNIR 1.2
  or later has one; a run without it was fitted under the legacy
  specification (§7, Legacy runs)

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

- round-robin folds over ROIs (`fit_spec` fold rule `round_robin`, the
  default for new runs)
- `LeaveOneGroupOut` (fold rule `leave_one_roi_out`)
- `GroupKFold` (fold rule `group_kfold`), kept only so that legacy runs
  refit identically; new runs do not use it unless it is requested
  explicitly

Required invariant:

- no pixel from a held-out ROI may appear in the training fold for that validation split

Required fold assignment:

- folds depend only on the recorded sample set: the splitter labels each ROI
  with its position in the sample set's `roi_ids`, never with the `roi_id`
  value itself
- `roi_id` values are random, and `GroupKFold` deals groups of equal size to
  folds in label order; labelling by id would make the cross-validated
  metrics of a repeated analysis (same data, new ids) differ

Round-robin rule:

- with *R* ROIs in the sample set and the requested fold count *F*
  (default 7), the number of folds is *k* = min(*F*, *R*)
- the ROI at position *i* (0-based) of the sample set's `roi_ids` is held
  out in fold *i* mod *k*
- with *R* ≤ *F* every fold holds out exactly one ROI (leave one ROI out)
- this is SIMCA's "assign every Nth observation to the same group" applied
  to ROIs (SIMCA 18 help, Cross validation group examples, No. 1); SIMCA
  numbers its groups differently, but the partition into folds is the same

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

## 6. Refitting a stored model

A stored model run can be fitted again on exactly what it was fitted on:

- the refit uses the run's sample set (its `roi_ids` and preprocessing
  recipe), signal domain, model type, hyperparameters and fit
  specification (the legacy specification for a run without one), never the
  set of ROIs that have values now
- the refit draws the pixel sample again from the stored seed and compares
  its checksum with the stored one; a different checksum (the stored
  spectra of an ROI changed) is logged as a warning naming the run, the
  refit proceeds, and the new run stores its own checksum
- with DModX exclusion, the refit compares its excluded pixels with the
  stored ones and logs differing ROIs as a warning (borderline pixels can
  flip with floating-point differences between platforms)
- it is refused when an ROI of the sample set was deleted
- because folds depend only on the sample set, unchanged ROIs and values give
  the same metrics; reference values that changed since the fit are
  reported, and differing metrics are reported
- the refit is stored as a new model run of the same sample set and signal
  domain; the original run is never changed

## 7. Fit specification

A model run records how it was fitted in `fit_spec`. Everything that changes
the numbers is stored, in `fit_spec` or in `hyperparameters`; nothing
depends on names, random ids or mapping order, so a repeated analysis (same
spectra and values, new ids) gives the same numbers.

| Key | Type and allowed values | Default for new runs |
|---|---|---|
| `folds/rule` | `round_robin`, `leave_one_roi_out`, `group_kfold` | `round_robin` |
| `folds/n_folds` | integer ≥ 2, the requested *F* | `7` |
| `pixels/per_roi` | `all`, or an integer from 1 to the pixel count of the smallest ROI of the sample set | `all` |
| `pixels/seed` | integer, 0 ≤ seed < 2⁶⁴ | `0` |
| `pixels/checksum` | SHA-256 (hex) of the drawn pixel indices (§7.1), written at fit time; empty string for `all` | – |
| `exclusion/method` | `none`, `dmodx` | `none` |
| `exclusion/level` | probability of the DCrit limit, 0.5 < level < 1 | `0.99` |
| `exclusion/rounds` | `1` or `2` | `1` |
| `exclusion/max_fraction` | largest share of an ROI's drawn pixels that may be excluded over all rounds, 0 < value < 1 | `0.2` |
| `exclusion/formula` | version of the DModX definition used, currently `dmodx-1` (§7.2) | `dmodx-1` |

Every key is written for every new run, also when it has no effect (e.g.
`n_folds` with `leave_one_roi_out`, `level` with `method = none`), so that
a stored specification compares equal to the same specification
rebuilt from its keys.

Scaling of X and Y is a model setting, not part of `fit_spec`: it is stored
in `hyperparameters` as `x_scaling` and `y_scaling` (each `none`, `center`,
`uv` or `pareto`; defaults for new runs: `x_scaling = center`,
`y_scaling = uv`, the usual setting for spectra), so the estimator stays
reconstructable as `Backend(**hyperparameters)`. PCR has `x_scaling` only.
Centring and scaling are learnt from the training pixels of each fold, like
every other fitted step; standard deviations use N − 1.

The short `cv` argument of the facade maps to the fold rule: `"auto"` →
`round_robin` with *F* = 7, an integer *F* → `round_robin` with that *F*,
`"logo"` → `leave_one_roi_out`. Two consequences for new runs: `cv="auto"`
uses round-robin folds instead of `GroupKFold`, so cross-validated metrics
can differ from a run fitted before engiNIR 1.2 on the same data; and an
integer *F* larger than the number of ROIs gives leave one ROI out instead
of being refused.

`cv_spec` records what ran: `type` `RoundRobinGroups` (also when *k* = *R*),
`LeaveOneGroupOut` or `GroupKFold`; `n_splits` = *k*; `requested` = the short
`cv` value, or `"fit_spec"` when the folds came from a `fit_spec`.

Requests that cannot be met block the fit before any spectra are read (see
Blocking Conditions).

### 7.1 Pixel sample

- `per_roi = all` uses every stored pixel of every ROI
- an integer *n* takes *n* pixels of every ROI, chosen by this algorithm,
  which uses only SHA-256 and so gives the same pixels on every platform and
  in every version:
  1. the ROI digest *d* is SHA-256 of the ROI's stored spectra as stored
     (raw dtype, little-endian, row-major), preceded by the ASCII text
     `"{dtype}:{n_pixels}x{n_bands}:"` (e.g. `"uint16:48x100:"`)
  2. the key of pixel *j* (0-based row of the stored spectra) is
     SHA-256 of `seed` as 8-byte little-endian unsigned, then *d*, then *j*
     as 4-byte little-endian unsigned
  3. the *n* pixels with the smallest keys (compared as bytes) are taken;
     their indices are sorted ascending
- the checksum is SHA-256 over, for each ROI in sample-set order, its digest
  *d* followed by its drawn indices as 4-byte little-endian unsigned
- the choice for one ROI depends only on its own spectra and the seed:
  adding or removing an ROI, or new ids for the same spectra, never change
  it
- the drawn pixels are used for everything: every fold, the final model,
  pixel diagnostics and the ROI means of the reported metrics
- every used pixel has the weight 1 (`per_pixel_unweighted`)

### 7.2 Pixel exclusion by DModX

With `exclusion/method = dmodx`, outlying pixels are left out of a model.
For one model (a fold's model or the final model):

1. fit the complete pipeline (stateful preprocessing, scaling, estimator)
   on the drawn pixels of its training ROIs
2. compute the normalised DModX of every training pixel and DCrit at
   `exclusion/level`, from that fit
3. exclude the pixels above DCrit, largest DModX first, so that an ROI
   never loses more than ⌊`max_fraction` × its drawn pixels⌋ in total over
   all rounds
4. refit the complete pipeline on the remaining pixels; with `rounds = 2`,
   repeat steps 2–4 once on the refitted model

Leakage rule: in cross-validation the exclusion runs inside every fold, on
that fold's training pixels only. Held-out pixels are never excluded: the
held-out ROI is predicted from all its drawn pixels, as a new scan is.

The final model applies the same procedure to all ROIs. The run stores the
union of the pixels excluded from the final model over all rounds, per ROI
(`artifact_refs/exclusion`), and the DCrit of its last round. An ROI with
more pixels above DCrit than `max_fraction` allows is flagged there; ROIs
are never excluded automatically.

Pixels of one ROI are correlated and *N* is large, so DCrit approaches a
fixed value and about `1 − level` of normal pixels lie above it in every
round; the default level 0.99 keeps that loss near 1 % per round.

Definition `dmodx-1` (SIMCA 18 help, pages
`15-Appendix A Statistics_files/AbsDistWorkset.htm`, `NormDistWorkset.htm`
and `Critical_distance_to_the_model.htm`). With *N* training pixels, *K*
bands, *A* components, *A₀* = 1 when X is centred (else 0) and *eᵢₖ* the X
residuals in scaled units:

- absolute: sᵢ = √(Σₖ eᵢₖ² / (K − A)) · v, with v = √(N / (N − A − A₀))
- pooled: s₀ = √(ΣᵢΣₖ eᵢₖ² / ((N − A − A₀)(K − A)))
- normalised DModX = sᵢ / s₀
- DCrit = √F⁻¹(level; DF_obs, DF_mod) with DF_mod = √((N − A₀ − A)(K − A))
  (at least 1); M = min(K, 100, DF_mod), CORR = N / (N − A₀ − A);
  DF_obs = (M + √(K − DF_mod) − A) / CORR if K > DF_mod, else
  (M − A) / CORR (at least 1)
- N − A₀ − A must be at least 1, otherwise exclusion blocks the fit

The help calls *v* only "a correction factor slightly larger than one"; the
value above is engiNIR's. Residuals are not weighted by the modelling power
(SIMCA's default weighting). If the SIMCA reference exports (plan
`fit-specification.md`, Phase R) show a different *v* or the weighting is
added, that is a new `exclusion/formula` version; stored runs keep theirs.

### 7.3 Legacy runs

A run without `fit_spec` is a legacy run. Reading and refitting it uses:

- folds from `cv_spec/requested`, with each ROI labelled by its position in
  the sample set's `roi_ids`: `"auto"` → scikit-learn `GroupKFold(7)`
  (without shuffling), or `LeaveOneGroupOut` with fewer than 7 ROIs; an
  integer *F* → `GroupKFold(F)`, refused with fewer than *F* ROIs;
  `"logo"` → `LeaveOneGroupOut`
- all pixels, no exclusion
- the stored `hyperparameters/scale` (absent means `true`): `true` scales X
  and Y (PLS, standard deviation with N − 1) or X (PCR, scikit-learn
  `StandardScaler`, standard deviation with N) to unit variance; `false`
  only centres

`GroupKFold` deals ROIs to folds by pixel count, so the folds of a legacy
run depend on the ROI sizes and on scikit-learn's algorithm. The refit
reproduces the metrics of runs written by engiNIR 1.0.1 to 1.1.x as long as
that algorithm is unchanged. Runs written by 1.0.0 labelled ROIs by their
random ids (CHANGELOG 1.0.1); their refit uses positions and may differ,
and this is reported like any metric difference.

## Weighting Rule

v3 must not silently bias the fit toward larger ROIs without an explicit decision.

v3 weighting decision:

- fitting uses `per_pixel_unweighted`: every used pixel has the weight 1
- with `pixels/per_roi = all`, larger ROIs contribute more pixel
  observations to `X`
- with an integer `pixels/per_roi`, every ROI contributes the same number of
  drawn pixels (§7.1); DModX exclusion can lower an ROI's count by at most
  `max_fraction`
- the sample set records the weighting policy; the model run records the
  pixel sample (`fit_spec/pixels`) and the excluded pixels
- validation and reporting must still remain ROI-grouped

Out of scope for v3:

- weights other than 1 (per-ROI or per-pixel weights)
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
- a `fit_spec` value is outside its allowed values (§7)
- `pixels/per_roi` exceeds the pixel count of the smallest ROI
- fewer than 2 ROIs, or a requested fold count below 2
- DModX exclusion with N − A₀ − A < 1 or K − A < 1

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
- CV specification (what ran: fold type, fold and group counts)
- fit specification (what was requested, §7; absent only for legacy runs)
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
- weighting policy is `per_pixel_unweighted`, and how many pixels per ROI
  were used
- which fold rule ran, and how many pixels were excluded by DModX
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
- weights other than 1
- mixed observation-unit comparisons in one comparison table

## Open Questions After This Decision

### 1. Weighting implementation detail

Decision:

- v3 uses `per_pixel_unweighted`

- equal ROI influence comes from equal pixel samples (§7.1), not from
  weights

Out of scope for v3:

- weights other than 1

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
- a model run written by engiNIR 1.2 or later stores its complete fit
  specification, and a refit reproduces its pixel sample, folds and metrics
- DModX exclusion never sees held-out pixels
- reported model performance cannot be mistaken for naive independent-pixel validation
