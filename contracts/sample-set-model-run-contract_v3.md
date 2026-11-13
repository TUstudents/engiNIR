# Calibration set and model contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.5; the file format `enginir-1.5` is the
  only format engiNIR reads and writes (`project-schema-contract_v3.md`)
- Scope: sample-building, fitting semantics, grouped validation, model identity, v3 PLS/PCR fitting behavior

## Goal

Define the concrete contract for:

- how project ROIs and targets become modeling inputs
- what the fitting unit is in v3
- how grouping and validation work
- what a model must store to be reproducible

This contract exists to prevent two common errors:

- collapsing ROI spectra to one mean and losing useful spectral diversity
- treating ROI pixels as fully independent samples and inflating confidence through pseudo-replication

## Core decisions

- v3 preserves pixel-level spectra as the data points of the fit
- v3 targets remain ROI-level
- v3 grouping and validation are ROI-level
- one model uses one selected target definition
- one model uses one preprocessing recipe
- working spectral arrays should default to `float32` unless a concrete backend
  contract requires higher precision
- v3 may support PLS and PCR, but both obey the same calibration-set contract

## Problem this contract solves

For your workflow, a calibration ROI carries:

- many pixel spectra
- one or more ROI-level target assignments

PLS benefits from diverse spectra within an ROI.
But if those pixels are treated as independent scientific samples, the model
will overstate performance and become statistically misleading.

The contract must therefore separate:

- fitting unit
- target level
- grouping unit
- validation unit

## Contract summary

There are two core objects:

1. `SampleSet`
   The reproducible description of how `X`, `Y`, and grouping are built.
2. `ModelRun`
   The reproducible description of how a model was fit and validated from one calibration set.

## Domain objects

## 1. SampleSet

Represents one modeling dataset built from project objects.

A new calibration set takes every ROI with a value for its target, except ROIs
whose `job` is `prediction` (`roi-target-contract_v3.md`): those are never
calibration data. A stored calibration set keeps its `roi_ids`; refitting it uses
them as stored, and logs a warning when one of them has since been switched
to `prediction`.

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

- one calibration set references exactly one selected `target_id`
- one calibration set references exactly one preprocessing recipe
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
  exists in the project; service-built calibration sets record exactly the ordered
  unique dataset ids of the included ROIs at build time
- a persisted `dataset_ids` entry without a matching live ROI may remain when
  included ROIs were later deleted (the stale calibration-set policy)

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
- `fit_spec`: the fit specification (§7)
- `software`: the software that fitted the model (§7)
- `created_at`

Optional fields:

- `name`: display name, a mutable label

Allowed `model_type` values in v3:

- `PLS`
- `PCR`

Required invariants:

- one model references exactly one calibration set
- one model references exactly one target definition through that calibration set
- one model references exactly one preprocessing recipe through that calibration set
- model metrics must be interpreted under ROI-grouped validation rules

## Modeling semantics

## 1. Build `X`

`X` is built from pixel spectra extracted from included ROIs.

For each included ROI:

- load its raw spectra
- apply the selected preprocessing recipe after the required signal conversion
- append each pixel spectrum to `X`

A recipe without MSC or EMSC acts on each spectrum alone and is applied
once before cross-validation. A recipe with MSC or EMSC is fitted and
applied inside every fold (all its steps, through the fold pipeline);
either way the result equals applying the whole recipe per fold. AsLS
takes its direction from the input contract's signal unit
(`project-schema-contract_v3.md`, Object identity rules §4). A recipe with `sg_edges = drop` or a
band selection gives `X` the recipe's output axis (fewer bands than the
input contract, with gaps after a selection); the
model's coefficients, loadings, projection data (`x_mean`, `x_std`,
`rotations`, `loadings`) and the stored wavelengths of `prediction_model`
and `diagnostics` use that axis, and K in §7.2 and §7.5 is its band count.
MSC/EMSC state uses the axis after Savitzky-Golay, before the selection.

Memory-aware v3 rules:

- multi-ROI calibration-set construction should use one batched/single-file-open
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
- repeat that ROI-level value across the ROI's pixels

v3 meaning:

- the fit uses pixel-level spectra
- response semantics are ROI-level

## 3. Build `groups`

`groups` is built from `roi_id`.

For each included ROI:

- repeat its `roi_id` for every pixel coming from that ROI

This is not optional in v3. The ROI is the smallest unit that is never
split between calibration and held-out data (`grouping_unit = roi`,
`validation_unit = roi`). Folds are built from fold groups of whole ROIs
(§4.1), which may join several ROIs of one physical sample.

## 4. Validation semantics

Cross-validation must be ROI-grouped.

Allowed v3 behavior:

- round-robin folds over fold groups (`fit_spec` fold rule `round_robin`,
  the default for new runs)
- leave one fold group out (fold rule `leave_one_group_out`; with
  `group_by = roi` this is leave one ROI out)
- `GroupKFold` (fold rule `group_kfold`), an optional rule a user requests
  explicitly; it requires `group_by = roi`. Each ROI is labeled by its
  position in the calibration set's `roi_ids`; scikit-learn's
  `GroupKFold(F)` (without shuffling) deals the ROIs to folds by pixel
  count, so the folds depend on the ROI sizes and on scikit-learn's
  algorithm; with fewer ROIs than *F* it is `LeaveOneGroupOut`. The fit
  dialog does not offer it; round-robin is the default

Required invariant:

- no pixel from a held-out ROI may appear in the calibration fold for that validation split
- no ROI of a held-out fold group may appear in the calibration fold for that
  validation split

Required fold assignment:

- folds depend only on the recorded calibration set: the splitter labels each ROI
  with the index of its fold group (§4.1), which depends only on positions
  in `roi_ids`, the equality of sample labels and the target values, never
  on the `roi_id` value itself
- `roi_id` values are random, and `GroupKFold` deals groups of equal size to
  folds in label order; labeling by id would make the cross-validated
  metrics of a repeated analysis (same data, new ids) differ

### 4.1 Fold groups

`fit_spec/folds/group_by` decides which ROIs are held out together:

- `roi` (default): every ROI is its own group, for ROIs that are
  independent samples
- `sample`: ROIs with the same sample label (roi-target contract, ROI
  `sample`) form one group, for several ROIs of one physical sample (e.g.
  three regions of the same blend); an ROI without a label is its own group
- `target`: ROIs with the same value of the calibration set's target form one
  group, for one sample per level without labels; different samples at the
  same value are then held out together, which is conservative. Values are
  compared as the stored float64 numbers (−0.0 equals 0.0): values that only
  print alike (e.g. 0.3 and 0.30000000000000004) form two groups

Group order, which decides round-robin membership:

- `roi` and `sample`: by the position (in the calibration set's `roi_ids`) of
  the group's first ROI
- `target`: by the target value, ascending

Label text never changes the numbers: a sample label only decides which
ROIs are equal, and group order comes from positions or values. Renaming
every label consistently gives the same folds.

The fit stores the group index of every ROI as used
(`fit_spec/folds/groups`, §7), so later label or value edits do not change
how a stored run is refitted (§6).

Round-robin rule:

- with *G* fold groups and the requested fold count *F* (default 7), the
  number of folds is *k* = min(*F*, *G*)
- the group at position *j* (0-based, in group order) is held out in fold
  *j* mod *k*
- with *G* ≤ *F* every fold holds out exactly one group (leave one group
  out)
- with `group_by = roi` the groups are the ROIs in calibration-set order: this is
  SIMCA's "assign every Nth observation to the same group" applied to ROIs
  (SIMCA 18 help, Cross validation group examples, No. 1); SIMCA numbers its
  groups differently, but the partition into folds is the same when SIMCA's
  observations are ROI means in calibration-set order, or pixels reordered so
  that every Nth observation comes from the same ROIs (as in the SIMCA
  reference export; verified against SIMCA 18.1). On pixels in ROI order
  SIMCA's every-Nth rule puts pixels of every ROI into every group
- with `group_by = target` round-robin deals the levels like venetian
  blinds (lowest level to fold 0, the next to fold 1, …), so the levels of
  every fold are spread over the range of the target; leave one group out
  with `group_by = target` holds out one level at a time, so the lowest and
  highest levels are predicted by extrapolation

Comparison with SIMCA's CV-groups dialog (verified against SIMCA 18.1
exports):

- grouping by an observation ID is leave one group out with that ID as the
  group: `leave_one_group_out` with `group_by = roi` (ID = ROI, verified) or,
  by the same mechanism, `group_by = sample` (ID = sample name)
- grouping by a variable sorts the observations by the value and cuts them
  into equal blocks. It is deliberately not offered (decided 2026-11-07): it
  can split one ROI or one sample across folds (leakage) and leaves the
  lowest and highest values to be predicted by extrapolation. When the
  blocks happen to coincide with the levels (every level has the same pixel
  count and *G* equals the number of levels), it equals
  `leave_one_group_out` with `group_by = target`

## 5. Reporting semantics

Because fitting unit and target level differ, reporting must be explicit.

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

Fit statistics of the final model, stored in
`metrics`, with *A* components, *N* the pixels the final model was fitted on
(drawn pixels minus excluded ones) and *R* ROIs:

- `pixel_rmsee` = √(Σᵢ (ŷᵢ − yᵢ)² / (N − 1 − A)) over those pixels (ŷ the
  final model's fitted values), as SIMCA computes RMSEE (over its observations, here pixels); NaN when
  N − 1 − A < 1
- `rmsee` = √(Σᵣ (ŷᵣ − yᵣ)² / (R − 1 − A)), ŷᵣ the mean fitted value of an
  ROI's pixels the model was fitted on; NaN when R − 1 − A < 1
- `pixel_r2_adj` = 1 − (RSS / (N − 1 − A)) / (TSS / (N − 1)) and `r2_adj`
  the same over ROI means (each NaN when its denominator is below 1)

Interval coverage (PLS and PCR): how often the
run's own cross-validated predictions fall inside the 0.95 prediction
intervals of `prediction-runtime-contract_v3.md` (Prediction intervals),
which use the fit error RMSEE (per ROI for the mean interval, per pixel for the pixel interval). A coverage well below 0.95 says the interval
is too narrow for new data. Each held-out prediction gets the interval
around itself, with the leverage of its ROI or pixel in the final model, and
counts as covered when |ŷ_cv − y| ≤ half-width. This is an approximation:
the held-out fold's model is not used, and the in-sample leverage and RMSEE (of either level)
are smaller than a held-out ROI's or pixel's would be, so the counted
intervals are, if anything, too narrow and the coverage errs low.

- `interval_coverage_roi`: over the ROIs with a cross-validated mean
  prediction and a mean interval (their `roi_scores` as t̄, R′ and the
  ROI-level limit as in Prediction intervals; ROIs above the limit are left
  out), s = `rmsee`. ŷ_cv is the stored ROI mean of the held-out pixel
  predictions over all drawn pixels (as `rmsecv`), while t̄ is the
  `roi_scores` row over the fitted pixels
- `interval_coverage_pixel`: over the drawn pixels with a finite held-out
  prediction and a pixel interval (their calibration T² and DModX, as
  `roi_share_above_dcrit`, against the stored 0.99 limits; pixels above
  either are left out; pixels excluded by DModX are counted like the
  others), s = `pixel_rmsee`. It is pixel-weighted, and its count is not an
  effective sample size (pixels of one ROI are not independent)
- each also stores its count, `interval_coverage_roi_n` and
  `interval_coverage_pixel_n`; a coverage without any interval is NaN with
  the count 0
- they are reported, never used to correct the interval

## 6. Refitting a stored model

A stored model can be fitted again on exactly what it was fitted on:

- the refit uses the run's calibration set (its `roi_ids` and preprocessing
  recipe), input contract, model type, hyperparameters and fit
  specification, never the set of ROIs that have values now
- the refit draws the pixel subsample again from the stored seed and compares
  its checksum with the stored one; a different checksum (the stored
  spectra of an ROI changed) is logged as a warning naming the run, the
  refit proceeds, and the new run stores its own checksum
- the refit uses the stored number of components
  (`hyperparameters/n_components`), never chooses it again; when the
  stored rule (`fit_spec/components/rule`, not `fixed`) would now choose a
  different number, this is logged as a warning naming the run
- the refit uses the stored fold groups (`fit_spec/folds/groups`), not the
  current sample labels or target values. The groups computed from the
  current labels or values are compared with the stored array element by
  element; any difference (also a changed order of target levels) is logged
  as a warning naming the run, and the new run stores the groups it used
  (the stored ones). When the current groups cannot be computed (a missing
  assignment), there is no comparison. With `group_by = target` a changed
  value is thus reported twice, as drift and as a changed target value
  (below); that is intended
- with DModX exclusion, the refit compares its excluded pixels with the
  stored ones and logs differing ROIs as a warning (borderline pixels can
  flip with floating-point differences between platforms)
- it is refused when an ROI of the calibration set was deleted
- because folds depend only on the calibration set, unchanged ROIs and values give
  the same metrics; target values that changed since the fit are
  reported, and differing metrics are reported
- the refit compares the stored `software` versions with the running ones;
  when its metrics differ, the warning names every version that differs
  (e.g. "fitted with scikit-learn 1.7.2, refitted with 1.8.0")
- the refit is stored as a new model of the same calibration set and input contract; the original run is never changed

## 7. Fit specification

A model records how it was fitted in `fit_spec`. Everything that changes
the numbers is stored, in `fit_spec` or in `hyperparameters`; nothing
depends on names, random ids or mapping order, so a repeated analysis (same
spectra and values, new ids) gives the same numbers.

| Key | Type and allowed values | Default for new runs |
|---|---|---|
| `folds/rule` | `round_robin`, `leave_one_group_out`, `group_kfold` | `round_robin` |
| `folds/n_folds` | integer ≥ 2, the requested *F* | `7` |
| `folds/group_by` | `roi`, `sample`, `target` (§4.1) | `roi` |
| `folds/groups` | one integer per ROI of the calibration set, in `roi_ids` order: the index of its fold group in group order (§4.1); a fit output like `checksum` | – |
| `components/rule` | `fixed` (the number was given), `one_se`, `min_rmsecv`, `q2_limit`, `f_test`, `q2_lookahead` (§7.4) | `fixed` with a number; `one_se` with `n_components="auto"` |
| `components/noise_guard` | `on`, `off`: `on` applies the structure ceiling and the noise drop after an autofit rule (§7.4), `off` gives the rule's plain choice; no effect with `fixed` | `on` with `"auto"`, `off` with a number |
| `components/rule_choice` | the rule's number before the ceiling and the noise drop (§7.4); equal to `selected` with `fixed` or `noise_guard = off`; a fit output (a value in a request is ignored and replaced by the fit), and on read 1 ≤ `selected` ≤ `rule_choice` ≤ `max` | – |
| `components/max` | integer ≥ 1, the largest number of components of the curve (*A*max) | max(*n*, min(15, max(1, *G* − 2), *M*)), *n* the given number (0 for `"auto"`), *M* the rank limit (§7.4) |
| `components/selected` | the number of components of the run (equal to `hyperparameters/n_components`); a fit output | – |
| `pixels/per_roi` | `all`, or an integer from 1 to the pixel count of the smallest ROI of the calibration set | `all` |
| `pixels/seed` | integer, 0 ≤ seed < 2⁶³ (a signed 64-bit HDF5 attribute) | `0` |
| `pixels/checksum` | SHA-256 (hex) of the drawn pixel indices (§7.1), written at fit time; empty string for `all` | – |
| `exclusion/method` | `none`, `dmodx` | `none` |
| `exclusion/level` | probability of the DCrit limit, 0.5 < level < 1 | `0.99` |
| `exclusion/rounds` | `1` or `2` | `1` |
| `exclusion/max_fraction` | largest share of an ROI's drawn pixels that may be excluded over all rounds, 0 < value < 1 | `0.2` |
| `exclusion/formula` | DModX definition (§7.2): `dmodx-1` (unweighted) or `dmodx-2` (weighted by the modeling power, SIMCA's default) | `dmodx-1` |

`cv_spec/@warnings` is required as well: the codes of §7.4, comma-separated,
an empty string when there are none.

Every key is written for every run, also when it has no effect (e.g.
`n_folds` with `leave_one_group_out`, `level` with `method = none`), so that
a stored specification compares equal to the same specification
rebuilt from its keys.

`n_components` may be requested as `"auto"`: the fit computes the component
curve (§7.4), chooses the number with `components/rule` and stores the chosen
integer in `hyperparameters/n_components`, so the stored model stays
`Backend(**hyperparameters)`. A number given by the user is stored as is,
with `components/rule = fixed`; the curve is computed in both cases.
`components/selected` is written by the fit and refused in a request, like
`folds/groups`. A stored `selected` must equal
`hyperparameters/n_components`, or the project is refused when read.

`folds/groups` is written by the fit; a fit request (`project.fit`) that
contains it is refused. A stored `groups` is well-formed when it has one
entry per ROI of `roi_ids` and its values are exactly 0 … *G* − 1 with every
index used; with `group_by = roi` it equals 0 … *R* − 1, and with `roi` or
`sample` the indices first appear in increasing order. A project holding a
run whose `groups` breaks this is refused when it is read, and a refit
never uses such groups.

Scaling of X and Y is a model setting, not part of `fit_spec`: it is stored
in `hyperparameters` as `x_scaling` and `y_scaling` (each `center`, `uv`
or `pareto`; defaults for new runs: `x_scaling = center`, `y_scaling = uv`,
the usual setting for spectra), so the estimator stays reconstructable as
`Backend(**hyperparameters)`. PCR has `x_scaling` only. Every scaling
center: scikit-learn's PLS and PCA always center, and uncentred models are
not used for spectra, so SIMCA's `none` is not offered (decided 2026-11-07;
it was listed in Phase 0). `uv` divides by the standard deviation with
N − 1, `pareto` by its square root. A fit that gives no scaling gets the
defaults. There is no other way to request scaling (the `scale` switch of
earlier versions is removed).
Centering and scaling are learnt from the calibration pixels of each fold, like
every other fitted step; standard deviations use N − 1.

The numerical algorithm is a model setting too, stored in `hyperparameters`
as `algorithm` (plan `plans/kernel-pls.md`, decided 2026-11-13):

- PLS: `kernel` (improved kernel PLS on the cross-product matrix XᵀX,
  Dayal and MacGregor 1997) or `nipals` (scikit-learn's NIPALS). New fits
  use `kernel`; a script may request `nipals`.
- PCR: `gram` (a symmetric eigendecomposition of the centered, scaled XᵀX)
  or `svd` (a full SVD of X). A new fit requests `auto`, which uses `gram`
  when σ_Amax / σ₁ ≥ 10⁻⁶ (σ the singular values of the centered, scaled X
  of the whole calibration set, *A*max the largest number of components the
  fit computes) and `svd` otherwise, with the reason logged; the run stores
  the algorithm it used, never `auto`.
- A stored model without `algorithm` (models of 1.5 projects) reads
  `nipals` (PLS) or `svd` (PCR). This is the one compatibility rule of the
  key; `nipals` and `svd` are removed in 2.0.
- A refit and "Fit Again" use the stored algorithm, so a refit reproduces
  the model exactly.

The algorithm does not change what a model means: the definitions of §5,
§7.2, §7.4 and §7.5 hold for every algorithm, and two algorithms give the
same model within floating-point precision (tested to 10⁻⁸ relative on
coefficients, weights, loadings, scores and every stored metric). Rule
choices that differ only by that noise may fall differently; a refit
reports that as any other difference (§7.4). The weights and loadings of
every algorithm follow scikit-learn's sign convention (PLS: the largest
absolute element of each X weight vector is positive; PCR: that of each
loading vector), so stored plots do not flip between algorithms.

The short `cv` argument of the facade maps to the fold rule: `"auto"` →
`round_robin` with *F* = 7, an integer *F* → `round_robin` with that *F*,
`"logo"` → `leave_one_group_out`, each with `group_by = roi`. An integer *F*
larger than the number of ROIs gives leave one ROI out instead of being
refused.

`cv_spec` records what ran: `type` `RoundRobinGroups` (also when *k* = *G*),
`LeaveOneGroupOut` or `GroupKFold`; `n_splits` = *k*; `n_groups` = *G*, the
number of fold groups (= the number of ROIs with `group_by = roi`); `requested` = the short
`cv` value, or `"fit_spec"` when the folds came from a `fit_spec`.

`software` records the versions the fit ran with, as attributes: `enginir`,
`numpy`, `scipy` and `scikit-learn`, each the installed version from the
package metadata. Every fit and refit writes it; it is part of the model's
content (a re-save with other versions is changed content), and a refit is
a new model with its own versions. A model's numbers can depend on these
versions (§6).

Requests that cannot be met block the fit before any spectra are read (see
Blocking Conditions).

### 7.1 Pixel subsample

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
- the checksum is SHA-256 over, for each ROI in calibration-set order, its digest
  *d* followed by its drawn indices as 4-byte little-endian unsigned
- the choice for one ROI depends only on its own spectra and the seed:
  adding or removing an ROI, or new ids for the same spectra, never change
  it
- the drawn pixels are used for everything: every fold, the final model,
  pixel diagnostics and the ROI means of the reported metrics
- every used pixel has the weight 1 (`per_pixel_unweighted`)

### 7.2 Pixel exclusion by DModX

With `exclusion/method = dmodx`, pixels above DCrit are excluded from a model.
For one model (a fold's model or the final model):

1. fit the complete pipeline (stateful preprocessing, scaling, estimator)
   on the drawn pixels of its calibration ROIs
2. compute the normalized DModX of every calibration pixel and DCrit at
   `exclusion/level`, from that fit
3. exclude the pixels above DCrit, largest DModX first, so that an ROI
   never loses more than ⌊`max_fraction` × its drawn pixels⌋ in total over
   all rounds
4. refit the complete pipeline on the remaining pixels; with `rounds = 2`,
   repeat steps 2–4 once on the refitted model

Leakage rule: in cross-validation the exclusion runs inside every fold, on
that fold's calibration pixels only. Held-out pixels are never excluded: the
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
and `Critical_distance_to_the_model.htm`). With *N* calibration pixels, *K*
bands, *A* components, *A₀* = 1 when X is centered (else 0) and *eᵢₖ* the X
residuals in scaled units:

- absolute: sᵢ = √(Σₖ eᵢₖ² / (K − A)) · v, with v = √(N / (N − A − A₀))
- pooled: s₀ = √(ΣᵢΣₖ eᵢₖ² / ((N − A − A₀)(K − A)))
- normalized DModX = sᵢ / s₀
- DCrit = √F⁻¹(level; DF_obs, DF_mod) with DF_mod = √((N − A₀ − A)(K − A))
  (at least 1); M = min(K, 100, DF_mod), CORR = N / (N − A₀ − A);
  DF_obs = (M + √(K − DF_mod) − A) / CORR if K > DF_mod, else
  (M − A) / CORR (at least 1)
- N − A₀ − A must be at least 1, otherwise exclusion blocks the fit

The help calls *v* only "a correction factor slightly larger than one"; the
SIMCA 18.1 reference exports (plan `fit-specification.md`, Phase R) confirm
the value above, and DModX, DCrit and absolute DCrit (normalized DCrit × s₀)
match them to float32 precision. Residuals are not weighted by the modeling
power in `dmodx-1`, the default: weighting turns down the bands the model
does not describe, where an unexpected component (a contaminant, glare)
often shows, and makes the exclusion depend on how well the model fits each
band.

Definition `dmodx-2` (SIMCA 18's default, help page
`15-Appendix A Statistics_files/MPow_weighted_distance_to_the_model.htm`
and `Modeling_power_MPowX.htm`): as `dmodx-1`, with every residual
multiplied by the modeling power of its band, eᵢₖ · MPowₖ, in sᵢ and in s₀:

- MPowₖ = max(0, 1 − SVₖ / SV0ₖ), with SVₖ = √(Σᵢ eᵢₖ² / (N − A − A₀)) the
  residual SD and SV0ₖ = √(Σᵢ (xᵢₖ − x̄ₖ)² / (N − 1)) the initial SD of band
  *k*, both in the scaled units of the model (SV0ₖ = 1 for `uv` X)
- MPow is computed from the fit whose pixels are being judged (every
  exclusion round recomputes it)
- DCrit is the same as for `dmodx-1`

Both definitions reproduce SIMCA 18.1 (plan Phase R: MPowX with the clip
at 0, weighted and unweighted DModX); a run keeps the definition it was
fitted with.

### 7.3 Removed

Earlier versions read and refitted models without a fit specification
(legacy runs). The `enginir-1.5` format has none: every model has a
complete `fit_spec` (`project-schema-contract_v3.md`, Compatibility
policy).

### 7.4 Component curve

Every PLS or PCR run stores, for *a* = 1 … *A*max (`components/max`),
how cross-validated prediction and fit change with the number of
components (`artifact_refs/component_curve`). The curve uses the run's
folds, fold groups, pixel subsample, scaling and preprocessing; DModX pixel
exclusion is not applied to it (the exclusion depends on *A*; the chosen
model is fitted with exclusion as specified). Without exclusion the curve's
RMSECV (per ROI) at the run's *A* is the run's RMSECV (per ROI) (one cross-validation serves
both); with exclusion the run's cross-validation is a second one and the
two differ. RSS(*b*), R²Y and R²X come from a fit on all drawn pixels with
*A*max components, without exclusion.

The rank limit *M* is the smallest of min(n − 1, K) over the calibration
pixels of every fold and of all drawn pixels (n pixels, K bands after
preprocessing): every fold must be able to fit *A*max components.

Definitions, at ROI level (*R* ROIs, ŷᵣ the mean of an ROI's out-of-fold
pixel predictions, ȳ the mean target value, *SS* = Σᵣ (yᵣ − ȳ)²):

- PRESS(*a*) = Σᵣ (ŷᵣ(*a*) − yᵣ)², RMSECV (per ROI)(*a*) = √(PRESS(*a*) / *R*)
- SE(*a*), the standard error of RMSECV (per ROI)(*a*): s(eᵣ²) / (2 RMSECV (per ROI)(*a*) √*R*),
  with eᵣ = ŷᵣ(*a*) − yᵣ and s the standard deviation with *R* − 1 (delta
  method)
- Q²(cum)(*a*) = 1 − PRESS(*a*) / *SS*
- Q²ₐ = 1 − PRESS(*a*) / RSS(*a* − 1), RSS(*b*) the ROI-level residual sum of
  squares of the model fitted on all ROIs with *b* components (RSS(0) =
  *SS*)
- R²Y(cum)(*a*) = 1 − RSSₚ(*a*) / SSₚ over the drawn pixels (as SIMCA; never
  decreases with *a*), and `r2y_cum_roi` = 1 − RSS(*a*) / *SS* over ROI means
  (it can decrease)
- R²X(cum)(*a*) = 1 − ‖X − Tₐ Pₐᵀ‖² / ‖X‖², X the scaled, centered drawn
  pixels after preprocessing

Every fold is refitted with all *a* components, so Q² is stricter than
SIMCA's component-wise cross-validation (plan `fit-specification.md`,
Phase R), and Q²(cum) is not 1 − Πₐ (1 − Q²ₐ) as in SIMCA: both are kept as
defined above. Guards: SE(*a*) = 0 when RMSECV (per ROI)(*a*) < 10⁻¹⁰; Q²ₐ is NaN when
RSS(*a* − 1) < 10⁻¹⁰, and Q²(cum), R²Y(cum) are NaN when *SS* < 10⁻¹⁰, each
with a logged warning.

Rules (`components/rule`), each choosing a number from 1 to *A*max:

- `min_rmsecv`: the smallest *a* with the lowest RMSECV (per ROI)
- `one_se` (default): the smallest *a* with RMSECV (per ROI)(*a*) ≤ RMSECV (per ROI)(*a*\*) +
  SE(*a*\*), *a*\* the `min_rmsecv` choice
- `q2_limit`: the largest *a* such that Q²_b > *L* for every *b* ≤ *a*, with
  *L* = 0.05 when *G* ≤ 100 and 0 otherwise, *G* the number of fold groups
  (the independent samples; SIMCA's Autofit rule R1); 1 when Q²₁ ≤ *L*; a
  NaN Q² fails the test
- `f_test` (Haaland and Thomas 1988): the smallest *a* with
  PRESS(*a*) / PRESS(*a*\*) < F(0.75; *G*, *G*), *a*\* the `min_rmsecv`
  choice and *G* the number of fold groups (the independent samples, not
  pixels); *a*\* always qualifies; when PRESS(*a*\*) < 10⁻¹⁰ the choice is
  *a*\* (with a logged warning)
- `q2_lookahead`: as `q2_limit`, but a failing component is kept when the
  next one passes. With "fails" meaning Q²_b ≤ *L* or NaN: *s* = the
  smallest *b* that fails and is either *A*max or followed by a failing
  *b* + 1; the choice is max(1, *s* − 1), or *A*max when there is no such
  *b*

Per component, from the same curve:

- ΔR²X(*a*) = R²X(cum)(*a*) − R²X(cum)(*a* − 1), R²X(cum)(0) = 0: the share of
  the spectra (scaled, centered drawn pixels) component *a* explains
- `noise`(*a*): ΔR²X(*a*) < 0.01 and Q²ₐ ≤ *L* (*L* as for `q2_limit`; a NaN
  Q² counts as ≤ *L*): the component explains under 1 % of the spectra and
  does not improve the prediction (SIMCA's rule N4 combined with R1)
- the structure ceiling *C* = min(*A*max, 1 + #{*a* ≤ *A*max : ΔR²X(*a*) ≥
  0.005}): one more than the number of components, anywhere up to *A*max,
  that each explain at least 0.5 % of the spectra. The cap at *G* − 2 comes
  from `components/max` (its default); a maximum the user gives overrides it
- ΔR²X, `noise` and *C* are computed in float64 from the stored `r2x_cum`
  and `q2`, at fit time and when read, so they agree with what is stored
- `rmsee_roi`(*a*) = √(RSS(*a*) / (*R* − *a* − 1)) over ROI means, NaN when
  *R* − *a* − 1 < 1; equal to `metrics/rmsee` at the run's *A* when no
  pixels are excluded (RSS comes from the *A*max fit without exclusion)
- `dw_weights`(*a*): the Durbin–Watson statistic
  Σₖ₌₁ᴷ⁻¹ (wₖ₊₁ − wₖ)² / Σₖ₌₁ᴷ wₖ² of component *a*'s weight vector over
  the *K* bands of the model axis in order (across a band-selection gap the
  neighbors are the kept bands), from the fit with *A*max components (PLS:
  the column of `x_weights_`, in the model's scaled space; PCR: the
  principal component); NaN when *K* < 2 or Σ wₖ² < 10⁻¹⁰. Near 0 for a
  smooth, spectrum-like vector, near 2 for white noise; a diagnostic, not a
  rule

The curve stores `rmsee_roi`, `dw_weights`, `noise` and *C*
(`@structure_ceiling`); ΔR²X is derived from `r2x_cum`.

With `n_components="auto"` the run's number is chosen in this order: *r*,
the rule's choice (`components/rule_choice`); with `noise_guard = on`,
*s* = min(*r*, *C*), then while *s* > 1 and `noise`(*s*), *s* − 1; with
`noise_guard = off`, *r*. A given number is used as given. The guard is on
by default because a curve from optimistic folds keeps falling slowly with
components that model noise. The ceiling belongs to the guard, so
`noise_guard = off` gives the rule's plain choice.

Ties are decided by exact comparison; values within floating-point noise of
each other can choose differently on another platform, which a refit
reports. The curve stores the suggestion of every rule, so the user sees
where they disagree.

**Fold warnings.** A fit records, in `cv_spec/@warnings` (codes separated by
commas, empty when none), conditions that make the cross-validation
optimistic; it warns in the log and does not refuse:

- `few_groups`: fewer than 7 fold groups
- `one_level_per_file`: some dataset (the ROIs' `dataset_id`) has at least
  2 ROIs of the calibration set, they share one value of the calibration set's target
  (compared as in §4.1, exact float64), and they are held out in more than
  one fold: held-out ROIs then have ROIs of their own dataset in calibration,
  so the model can learn the dataset instead of the target; giving each
  such dataset's ROIs one sample label holds them out together

### 7.5 Calibration diagnostics

Every PLS or PCR model also stores, from the final model (*N* the pixels
it was fitted on):

- `coef_cs`: the coefficients in the model's own scaled units (SIMCA's
  CoeffCS), bₖ · sₓₖ / s_y with sₓₖ the model's X divisor (`x_std_`; 1 for
  centered X) and s_y its Y divisor (`y_std_`; for PCR, which does not scale
  Y, the standard deviation of y with N − 1)
- one value per drawn pixel, in the order of the calibration set's ROIs and then
  the drawn indices: Hotelling's T² = Σₐ tₐ² / var(tₐ) (variance over the
  N fitted pixels, N − 1) and the normalized, unweighted DModX (`dmodx-1`,
  §7.2, s₀ and v from the N fitted pixels). Pixels excluded by DModX get the
  same formulas from the final model (`artifact_refs/exclusion` says which
  they are)
- T²Crit = A(N − 1)/(N − A) · F(level; A, N − A) at 0.95 and 0.99 (SIMCA's
  F form; `enginir.validation.outliers.hotelling_t2` uses the Beta form for
  calibration data points, a different number); T²Crit for the first two
  components (the score-plot ellipse), NaN when A < 2; DModX and DCrit at
  0.99, NaN when K − A < 1 or N − A₀ − A < 1 (`project-schema-contract_v3.md`, Core and display data)
- ROI scores for the score plot (model-workspace-diagnostics contract §2):
  `roi_scores`, per ROI (in `cv_outputs/roi_ids` order) the mean calibration
  score of each component over that ROI's fitted pixels (the drawn pixels
  with a pixel subsample, the kept pixels with DModX exclusion), and
  `roi_score_pixels`, the number of those pixels. An ROI with no fitted
  pixel has NaN scores and 0. The fit computes them because only the fit
  knows which row belongs to which ROI
- per ROI the share of its drawn pixels above DCrit and above T²Crit (0.99)
- band importance, on the model's axis:
  `mean_spectrum`, the mean preprocessed spectrum of the N fitted pixels
  (the model's `x_mean` in the units of X; stored with the diagnostics,
  which plots load without the prediction model); for PLS `vip`,
  VIPⱼ = √(K Σₐ SSₐ (wⱼₐ/‖wₐ‖)² / Σₐ SSₐ) with w the X weights W
  (`x_weights_`, not the rotations) and SSₐ = tₐᵀtₐ Σₖ qₖₐ², N the pixels
  of the final fit (after exclusion) (Chong and Jun; mean of VIP² is 1;
  summed over targets for several); for PLS and PCR with one target
  `selectivity_ratio` (Rajalahti et al.): in the model's scaled, centered X
  space (Z, rows the N fitted pixels) the target-projection weight
  w = b / ‖b‖ with b = coef · x_std (the regression vector in that space),
  t = Z w, p = Zᵀt / tᵀt, and per band SRⱼ = pⱼ² tᵀt / (Σᵢ Zᵢⱼ² − pⱼ² tᵀt),
  i.e. Σᵢ (tᵢ pⱼ)² / Σᵢ (Zᵢⱼ − tᵢ pⱼ)² (explained over residual variance;
  computed in one chunked pass). A band whose residual is below 10⁻¹⁰ times
  its sum of squares (at least 10⁻¹⁰), or with zero variance, gets NaN with
  a warning. Both are computed from the final model without a refit

PCR computes its principal components deterministically, with the full SVD
of X (`svd`) or the symmetric eigendecomposition of XᵀX (`gram`), so a fit
is reproducible and its first *a* components are those of an *a*-component
fit, for both algorithms. Likewise the first *a* components of a PLS fit
are the *a*-component PLS model for both `kernel` and `nipals`.

The final model's `prediction_model` artifact also stores its projection
data (`x_mean`, `x_std`, `rotations`, `loadings`, `score_variances`, `s0`,
`n_train`, `n_components`, `t2_crit_099`, `dcrit_099`), the same values the
calibration diagnostics above use, so predictions can compute DModXPS and T²PS
(`prediction-runtime-contract_v3.md`, Applicability diagnostics).

## Weighting rule

v3 must not silently bias the fit toward larger ROIs without an explicit decision.

v3 weighting decision:

- fitting uses `per_pixel_unweighted`: every used pixel has the weight 1
- with `pixels/per_roi = all`, larger ROIs contribute more pixels to `X`
- with an integer `pixels/per_roi`, every ROI contributes the same number of
  drawn pixels (§7.1); DModX exclusion can lower an ROI's count by at most
  `max_fraction`
- the calibration set records the weighting policy; the model records the
  pixel subsample (`fit_spec/pixels`) and the excluded pixels
- fold groups do not change weights: a sample with more ROIs contributes
  more pixels and more held-out ROI means than a sample with fewer
  (equal pixels per ROI, not per sample)
- validation and reporting must still remain ROI-grouped

Out of scope for v3:

- weights other than 1 (per-ROI, per-sample or per-pixel weights)
- metrics reported per sample (they stay per ROI)
- per-pixel uncertainty weighting

## Blocking conditions

Model fitting for a calibration set must block or warn when:

- no ROIs are selected
- selected ROIs do not have assignments for the chosen target
- ROI wavelength axes are incompatible
- ROI reference correction or input-contract requirements are incompatible with the chosen recipe/model contract
- grouping metadata cannot be constructed by ROI
- the chosen CV strategy would violate ROI grouping
- the current workflow is in project-only mode without a real attached dataset
  for any action that depends on current viewer dataset context
- a `fit_spec` value is outside its allowed values (§7)
- `pixels/per_roi` exceeds the pixel count of the smallest ROI
- fewer than 2 ROIs, fewer than 2 fold groups, or a requested fold count
  below 2
- fold rule `group_kfold` with `group_by` other than `roi`
- a fit request that contains `folds/groups` or `components/selected`;
  stored groups that are not well-formed (§7)
- `components/max` or a given number of components above the rank limit
  *M* (§7.4); a rule other than `fixed` with a given number (except a refit,
  whose number equals the stored `selected`)
- `group_by = sample` when no ROI of the calibration set has a sample label:
  warn (the folds equal `group_by = roi`), do not block
- DModX exclusion with N − A₀ − A < 1 or K − A < 1

## Persistence requirements

`SampleSet` and `ModelRun` must be persisted inside the project file.

## Required persisted calibration-set information

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

## Required persisted model information

- `model_run_id`
- `sample_set_id`
- `model_type`
- hyperparameters
- CV specification (what ran: fold type, fold and group counts)
- fit specification (what was requested, §7)
- software versions of the fit (§7)
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
- MSC state contains the fitted calibration reference spectrum
- EMSC state contains its fitted reference spectrum, wavelength weights,
  optional interference basis, method, and polynomial order
- these arrays and parameters are prediction-critical model artifacts, not
  mutable recipe fields
- reopening reconstructs fitted transformers from this state and must never
  call `fit()` on prediction data
- missing or malformed required state is core: the reader refuses the
  project, and preflight refuses an in-memory model without it

## v3 HDF5 shape basis

v3 requires project-internal groups shaped like:

```text
/sample_sets/{sample_set_id}/
/model_runs/{model_run_id}/
```

Minimum expectation:

- the calibration set stores the fitting/target/grouping semantics explicitly
- the model stores metrics and artifacts explicitly
- no model relies on an implicit “current table column” or “current viewer state” as its target contract

## Required UI behavior

The UI must make these distinctions legible:

- fitting uses one selected target definition
- pixel spectra are the data points of the fit
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

## Out of scope for v3

These are intentionally deferred:

- `roi_mean` as the primary model-fitting contract
- `roi_feature` calibration sets
- multi-target models
- user-selectable fitting-unit switching in the first modeling UI
- weights other than 1
- mixed fitting-unit comparisons in one comparison table

## Open questions after this decision

### 1. Weighting implementation detail

Decision:

- v3 uses `per_pixel_unweighted`

- equal ROI influence comes from equal pixel subsamples (§7.1), not from
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

## Acceptance criteria

This contract is implemented correctly when:

- a calibration set explicitly records `observation_unit = pixel`
- a calibration set explicitly records `weighting_policy = per_pixel_unweighted`
- a calibration set explicitly records `reporting_level = roi`
- a calibration set explicitly records `roi_aggregation = mean`
- the selected target is explicit and project-defined
- `Y` is derived from ROI-level assignments
- validation groups pixels by `roi_id`
- fold groups never split a group: no ROI of a held-out fold group is in
  the calibration data of that fold, under every `group_by` and fold rule
- a refit uses the stored fold groups and warns when the current ones differ
- a model stores its calibration-set and CV semantics explicitly
- every model stores its complete fit specification and its software
  versions, and a refit reproduces its pixel subsample, folds and metrics
- DModX exclusion never sees held-out pixels
- reported model performance cannot be mistaken for naive independent-pixel validation
