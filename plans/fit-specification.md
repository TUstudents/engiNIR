# Plan: fit specification

Status: Phases 0–6, R and S done (2026-11-07); Phase 7 open

## Goal

A model run states completely how it was fitted, so that a refit gives the
same numbers and the numbers can be compared with Sartorius SIMCA 18. A
stored **fit specification** replaces the current `cv_spec` and gives
control over the fit:

- folds: ROIs dealt round-robin to the folds
- pixel sampling: the same number of pixels from every ROI, drawn with a
  stored seed
- scaling of X and Y
- automatic exclusion of outlying pixels by DModX

The SIMCA statistics (R²X, Q², Q²(cum), RMSEE, Autofit, T² limits,
DModXPS) and the SIMCA filters (Savitzky-Golay without edges, AsLS) are a
second plan built on this one (see Follow-up).

## Decisions taken (2026-11-04)

- One ROI is one cross-validation group (the "sample" of SIMCA). No
  sample label above the ROI. *Superseded 2026-11-07 by Phase S:* ROIs
  may be grouped into samples; the ROI stays the smallest unit that is
  never split.
- Folds: ROI at position *i* of the sample set's `roi_ids` goes to fold
  *i* mod *k*, with *k* = min(7, number of ROIs). With fewer than 7 ROIs
  this is leave-one-ROI-out. This is SIMCA's "every Nth" applied to groups.
- Pixels per ROI: *n* is chosen by the user, at most the pixel count of
  the smallest ROI of the sample set. Pixels are drawn without replacement
  with a stored seed.
- Reported metrics come from the drawn pixels (the final model and every
  fold use only drawn pixels; ROI means are means over drawn pixels).
- Automatic pixel exclusion uses DModX only. Whole ROIs are never excluded
  automatically; they are flagged.

## Contract changes (written in Phase 0)

- `sample-set-model-run-contract_v3.md`: optional `fit_spec` on a model run;
  round-robin rule in §4; refit with the stored specification and checksum
  (§6); new §7 Fit specification (keys, short `cv` mapping, blocking), §7.1
  Pixel sample, §7.2 Pixel exclusion by DModX with SIMCA's formulas, §7.3
  Legacy runs; Weighting Rule: weights stay 1, equal ROI influence comes
  from equal pixel samples (no new weighting policy).
- `cv_spec` stays as the record of what ran (new type `RoundRobinGroups`);
  `fit_spec` records what was requested.
- Scaling of X and Y is a backend constructor argument and lives in
  `hyperparameters` (`x_scaling`, `y_scaling`), so `Backend(**hyperparameters)`
  still reconstructs the estimator; legacy runs keep `scale`.
- `project-schema-contract_v3.md`: `fit_spec` in the layout and the
  structured-metadata rules; compatibility note for readers before 1.2.
- `headless-workflow-contract_v3.md`: `project.fit(..., fit_spec=None)`.
- `docs/hdf5_format.md`: `fit_spec/`, scaling attributes, `cv_spec` type,
  metrics `n_pixels`/`n_excluded_pixels`, `artifact_refs/exclusion`.
- `docs/contract_coverage.md`: fit-specification rows and gap `G22`.

## Pixel drawing

Contract §7.1 (revised after the contract review): the pixels of an ROI
are chosen by SHA-256 keys of (seed, SHA-256 of the ROI's stored spectra,
pixel index), the *n* smallest keys win. No NumPy generator is involved, so
the choice is the same on every platform and version, and it depends on
the ROI's spectra, never on its random id. The checksum covers the ROI
digests and the drawn indices.

## Pixel exclusion by DModX

Per fold, on that fold's training pixels only:

1. fit the model on the drawn pixels
2. compute normalised DModX of every training pixel and DCrit at the
   chosen level (SIMCA formulas, `15-Appendix A`: s_i, s0, DF_mod, DF_obs)
3. exclude pixels above DCrit, at most `max_fraction` of each ROI (the
   worst ones first)
4. refit (`rounds` = 2 repeats 2–4 once)

The final model does the same on all ROIs. The run stores, per ROI, the
indices of the excluded pixels of the final model. An ROI that would lose
more than `max_fraction` is flagged in the run (diagnostic list), not
dropped.

## Phases

### Phase 0: contracts and ledger
Contract text above, `hdf5_format.md`, coverage ledger entries. No code.

Contract review: when the text is written, an independent review agent
(fresh context, read-only) reviews the contract changes before the phase is
committed. It gets the diff, the full contracts and this plan, and checks:

- consistency: no contract, `hdf5_format.md`, the ledger or `AGENTS.md`
  (frozen semantics) contradicts the new text; every changed rule is
  mapped in `docs/contract_coverage.md`
- completeness: every key of the fit specification has a type, allowed
  values, a default and a persistence rule; legacy runs are covered
- reproducibility: every input that changes the numbers is stored;
  nothing depends on names, random ids or dict order
- leakage: no step (pixel drawing, scaling, stateful filters, exclusion)
  uses held-out pixels
- testability: every rule can be checked by a test that fails when the
  rule is broken
- SIMCA claims: every formula attributed to SIMCA cites its help page

Its findings are reported to the user with the fixes made; the phase is
committed only after that. Later phases that touch a contract (5, and the
follow-up plan) repeat the review.

### Phase R: SIMCA reference project (parallel to Phases 1–5)

A small project whose numbers are computed by SIMCA 18 and by engiNIR, so
"close to SIMCA" is tested, not claimed.

Data (`tests/data/simca_reference/`, small enough for the repository):

- a synthetic push-broom cube, generated by a committed, seeded script:
  14 blends of three component spectra (known concentrations), with
  scatter (offset and multiplicative), a curved baseline and noise;
  about 100 bands; uint16 counts with black and white references, as
  every engiNIR project stores raw counts. SIMCA 18 has no raw-to-absorbance
  conversion: engiNIR converts, and the exported tables already hold
  absorbance. The comparison starts at that table; the conversion itself
  is not part of it (it is covered by engiNIR's own tests)
- an engiNIR project on it: one ROI per blend, target "conc"
- 14 ROIs give 7 folds of 2 ROIs; few pixels per ROI keep SIMCA's
  N small

Exports from engiNIR for SIMCA (CSV, one row per observation):

- `pixels.csv`: 10 drawn pixels per ROI (140 observations; seed stored),
  columns: observation ID (`roi-name_pixel-index`), secondary ID `roi`,
  `conc`, one column per band (absorbance). Values are written with 9
  significant digits (exact for float32), and a test checks that the table
  equals the array engiNIR fits on, so both programs start from the same
  numbers
- `roi_means.csv`: one row per ROI (14 observations), same columns
- `pixels_test.csv`: 4 ROIs left out, for predictions

The two tables cover both sides of SIMCA's N = 100 limit in Autofit rule R1.

Models built in SIMCA 18 (instructions in `README.md` next to the data):

| Model | Data | Filter | Scaling | CV | Components |
|---|---|---|---|---|---|
| M1 | pixels | none | Ctr X, UV Y | 7 groups, same `roi` value in one group | 3 |
| M2 | pixels | SNV | Ctr X, UV Y | as M1 | 3 |
| M3 | pixels | SG 1st derivative, 15 points, quadratic | Ctr X, UV Y | as M1 | 3 |
| M4 | roi_means | none | Ctr X, UV Y | 7 groups, every Nth | Autofit |
| M5 | pixels | AsLS correction (λ 10000, p 0.001) | Ctr X, UV Y | as M1 | 3 |

Exports from SIMCA per model (copy as CSV from the lists):

- CV-group list (fixes how SIMCA deals ROIs to the 7 groups)
- summary of fit list: R2X, R2X(cum), R2Y, R2Y(cum), Q2, Q2(cum),
  significance, limit
- scores t, loadings p and weights w, VIP
- coefficients: scaled and centred, and unscaled
- observed, fitted and cross-validated Y; RMSEE, RMSEcv
- T2Range with T2Crit(95 %, 99 %); DModX (normalised, both with and
  without modelling-power weighting) with DCrit
- for the prediction set: predicted Y, RMSEP, T2PS, DModXPS
- for M2, M3, M5: the filtered dataset (shows SNV, edge handling, AsLS)

The SIMCA exports are a one-off verification and are not committed
(decided 2026-11-04; `simca_exports/` is git-ignored). The repository keeps
the input tables, the export scripts, and the findings below; engiNIR's own
tests check the confirmed formulas without SIMCA output, so a new SIMCA
version can be re-verified by rerunning the scripts. The undocumented points
(DModX factor *v*, AsLS iterations, scaling inside CV rounds) were decided
by these exports.

Steps: generate data and project, export the tables (engiNIR side, done by
the agent); build M1–M5 and export (done by the user in SIMCA 18); compare
and record the findings (done).

The SIMCA side is scripted (`simca_export.py`, SIMCA's Python API): it
builds M1–M5 with 4 components (M4 Autofit) and exports at full precision.
CV groups cannot be set from Python; the script imports the pixels
interleaved so that SIMCA's default every-7th groups equal the round-robin
ROI folds. `simca_export_active.py` exports models configured by hand (CV
dialog).

#### Findings from the exports (SIMCA 18.1, M1–M5, 2026-11-04)

Matched to float32 precision (relative difference about 1e-7) by engiNIR
or by the formula given:

- PLS fit: scores, X weights, loadings, unscaled coefficients, fitted Y,
  VIP, R2X, R2Y; RMSEE = √(SS_res / (N − 1 − A))
- T2Range; T2Crit = A(N − 1)/(N − A) · F(level; A, N − A)
- DModX normalised, unweighted and MPow-weighted, with
  v = √(N / (N − A − A₀)) exactly as in contract §7.2 `dmodx-1`
- DCrit (normalised) with the DF_obs/DF_mod rule of §7.2; absolute DCrit
  = normalised DCrit × the unweighted s₀
- MPowX = max(0, 1 − SV_k / SV0_k), SV_k = √(Σᵢ eᵢₖ² / (N − A − A₀)),
  SV0_k = √(Σᵢ xᵢₖ² / (N − 1)) (centred); clipping at 0 shows only for
  N = 14
- filters: SNV = engiNIR `SNV`; derivative = engiNIR `SavitzkyGolay(15, 2,
  1)` on the interior, SIMCA sets the first and last 7 bands missing and
  drops them from the model (K = 86); AsLS = Eilers AsLS (second
  differences, λ 10000, p 0.001, 10 iterations), corrected = spectrum −
  baseline

- prediction set: YPredPS, RMSEP (with N), tPS, T2RangePS; DModXPS
  normalised = √(Σₖ (MPowₖ eₖ)² / (K − A)) / s₀ (weighted s₀), without
  the factor v of the workset DModX. SIMCA applies a model's filter to the
  prediction set by itself
- a filtered model is built from the filtered dataset (X) and the
  unfiltered one (Y), observations matched by ID
- all five models agree within 1e-5 relative (most quantities about 1e-7),
  M2, M3 and M5 also after engiNIR's SNV, `SavitzkyGolay` (interior) and
  the reference AsLS

Different (needs a decision):

- **Cross-validation.** SIMCA validates component by component on the
  full model's deflated residuals: centring once on all data; for
  component *a*, E and f are the full model's residuals after *a* − 1
  components; each fold fits one NIPALS component on the training rows
  and predicts the held-out rows' f. Q2ₐ = 1 − PRESSₐ / SS(f), Q2cum =
  1 − Π PRESSₐ/SSₐ, YPredcv = ȳ + Σₐ held-out predictions, RMSEcv =
  √(PRESS / N). Reproduced exactly. engiNIR refits all *a* components per
  fold (stricter: SIMCA's deflation uses a model that saw the held-out
  rows), so its Q2 and RMSECV differ (M1, A = 4: RMSECV 0.0216 vs 0.0245).
- **Grouped CV assignment** (hand-configured exports, 2026-11-07, 140
  pixels, 14 ROIs of 10):
  - by an observation ID (`roi`): one CV group per value, i.e. leave one
    ROI out; the number of groups cannot be set
  - by a variable (`conc`) with G groups: the observations are sorted by
    the value (stable: ties keep data order) and cut into G consecutive
    blocks of N/G observations (140/7 = 20, 140/5 = 28; rounding for
    uneven N/G not seen). Reproduced for all 140 pixels with G = 7 and
    G = 5. With G = 7 every block is two neighbouring ROIs; with G = 5 the
    ROIs B03, B06, B09 and B12 are split across two groups, i.e. pixels of
    one sample end up on both sides of a fold (leakage)
  - neighbouring concentrations in one group leave the lowest and highest
    concentrations to be predicted by extrapolation; round-robin
    (SIMCA's default every-Nth on data in concentration order) spreads
    them over all groups
  - engiNIR's `round_robin` equals SIMCA's default every-Nth grouping;
    `leave_one_roi_out` equals grouping by the ROI ID. Decided
    2026-11-07: SIMCA's value-sorted blocks are not offered (leakage,
    extrapolation); stated in contract §4 and the validation user guide

### Phase 1: `FitSpec` and persistence
Typed `FitSpec` in `project/_models.py` (validated, `to_dict`/`from_dict`),
stored as the `fit_spec` group of a model run; legacy reading of runs
without it; `cv_spec` kept readable. `Project.fit(fit_spec=)`,
`Model.refit()` uses the stored specification. Tests: round trip, legacy
run refits to the same metrics.

### Phase 2: round-robin folds
`cross_validate` gains the round-robin rule over group positions;
`fit_model_run` uses it for new fits. Tests: fold membership for 3, 7 and
20 ROIs; never a split ROI; legacy rule unchanged.

### Phase 3: pixel drawing
Drawing per ROI, checksum, blocking condition for *n* > smallest ROI.
Tests: same seed gives the same pixels and metrics; adding an ROI leaves
the others' pixels unchanged; a changed checksum is reported on refit;
memory stays float32 without extra full copies.

### Phase 4: scaling of X and Y (done 2026-11-07)
`PLSRegression` and `PCR` take `x_scaling` and `y_scaling` (centring and
scaling done by engiNIR, scikit-learn called with `scale=False`; variances
with N − 1). `none` dropped from the contract: scikit-learn always
centres. Legacy runs without `scale` refit with `scale=True`. Tests: `uv`/`uv` equals today's results; `center` X gives
the textbook PLS1 on centred data; coefficients in original units agree
across scalings.

### Phase 5: DModX and pixel exclusion (done 2026-11-07)
`dmodx(...)` and `dmodx_limit(...)` in `validation/outliers.py` with SIMCA's
formulas (normalised; see open decision 3), then exclusion inside folds
and in the final model, stored excluded indices and flags. Tests: limit
against hand-computed F quantiles; an injected bright pixel is excluded;
exclusion never sees test-fold pixels (fault injection: exclusion on all
pixels must change the test).

### Phase S: sample groups (decided and done 2026-11-07)

**Why.** Folds are built from single ROIs. When several ROIs come from one
physical sample (three regions of the same blend or tablet), holding out
one of them while its siblings stay in training leaks the sample, and the
cross-validated error is too optimistic. The independent unit is then the
sample, not the ROI. SIMCA gives this by grouping by an ID (leave one
sample out) or, when the counts happen to line up, by grouping by the
concentration; engiNIR has no rule for it (worked example in the
conversation of 2026-11-07: 5 levels × 3 ROIs).

**What.** A fold groups key in the fit specification:

| `folds/group_by` | Group | Use |
|---|---|---|
| `roi` (default) | each ROI | ROIs are independent samples (today's behaviour) |
| `sample` | ROIs with the same sample label; an ROI without a label is its own group | several ROIs per physical sample |
| `target` | ROIs with the same value of the sample set's target | one sample per concentration level, no labels needed |

Folds are then built from groups with the existing rules: round-robin over
groups (group at position *j* → fold *j* mod *k*, *k* = min(*F*, number of
groups)) or leave one group out. An ROI is never split; a group is never
split.

Group order (decides round-robin membership):

- `roi` and `sample`: by the position of the group's first ROI in the
  sample set's `roi_ids`, as today
- `target`: by the target value, ascending; round-robin then deals the
  levels like venetian blinds (lowest level to fold 0, next to fold 1, …),
  so every fold spans the concentration range. This equals SIMCA's
  default every-Nth on ROI means in concentration order

Equal target values: exact equality of the stored float64 values
(−0.0 equals 0.0; values that only print alike form two groups).

**Sample labels.** A mutable, optional ROI attribute `sample` (string,
like the ROI name a label, not an identity). Set in the viewer's ROI table
(new "Sample" column, through the typed `Project.save_roi_sample(roi_id,
label)`) and by `project.set_sample(roi, label)`; only that writer changes
the label, re-saving an ROI never does. Labels do not change stored objects'
content rules: an ROI's spectra and id are untouched.

**Reproducibility.** The fit stores the group of every ROI as used
(`fit_spec/folds/groups`: one integer group index per ROI in sample-set
order; a fit output like the checksum, refused in a request). `Model.refit()` uses the stored groups and warns, like the
pixel checksum, when the current labels or target values would group
differently.

**Names.** `leave_one_roi_out` becomes `leave_one_group_out` (with
`group_by = roi` it is the same rule). Fit specifications are not released
yet (CHANGELOG: Unreleased), so the rename needs no reader for old
values; legacy runs (§7.3) keep their own mapping.

**Unchanged.** Pixel drawing per ROI; metrics reported per ROI (ROI
means); DModX exclusion inside folds; `SampleSet` fields `grouping_unit =
roi` and `validation_unit = roi` keep their meaning "smallest unit never
split" (contract text to say so). Per-sample reporting is out of scope.

**Weighting.** A sample with more ROIs carries more pixels and more
held-out ROI means; this is accepted and stated (equal pixels per ROI, not
per sample). Equal weight per sample would need drawing per sample: not
in this phase.

**Blocking conditions.** Fewer than 2 groups; `group_by = target` with a
missing target value (already blocked by sample-set rules); `sample`
grouping where no ROI has a label (warn: equals `roi`, not blocked).

**SIMCA comparison** (5 levels × 3 ROIs, stated in the user guide):

| engiNIR | SIMCA |
|---|---|
| `group_by = sample`, leave one group out | grouping by an ID holding the sample name |
| `group_by = target`, leave one group out | grouping by `conc` when N/G equals the pixels per level (here G = 5) |
| `group_by = target`, round-robin *F* | no equivalent; never splits a level |

**Steps.**

1. Contract changes with an independent review agent (as in Phase 0):
   sample-set-model-run §4 and §7 (`group_by`, group order, stored
   groups, rename), roi-target (the `sample` attribute, its mutability),
   project-schema (ROI attribute, `fit_spec/folds/group_by`,
   `fit_spec/folds/groups`), headless-workflow (`set_sample`, refit
   warning); `docs/hdf5_format.md`; ledger gap G23.
2. Persistence: ROI `sample` attribute (optional; older files read as no
   label), `FitSpec.group_by` and stored groups; round trip.
3. Fold building: `_splitter` over groups; `RoundRobinGroups` reused on
   group labels; leave-one-group-out.
4. Refit: stored groups used, warning on drift.
5. Viewer: "Sample" column in the ROI table; "Group folds by" in the
   Fitting section (Phase 6).
6. Docs: user guide (when to group by sample; the 5 × 3 example; the SIMCA
   table above), API reference, CHANGELOG.

**Tests.** Membership for 5 levels × 3 ROIs under each `group_by` and
rule (no group split, levels spread for `target` round-robin);
unlabelled ROIs form their own groups; a sibling ROI is never on both
sides of a fold (fault injection: `group_by` ignored must fail);
refit reproduces metrics with changed labels and warns; old project files
without labels open unchanged.

### Phase 6: viewer (done 2026-11-07)
The fit dialog gets a "Fitting" section: folds, pixels per ROI (spin box,
maximum = smallest ROI of the plan), seed, X and Y scaling, pixel
exclusion. Diagnostics show the specification of a run and the excluded
pixels per ROI; flagged ROIs are listed. Viewer docs kept in sync.

### Phase 7: docs and release notes
User guide section "Fit specification", API reference, CHANGELOG, all
four gates.

## Open decisions

1. **Default scaling for new fits.** Taken in the contract: `center` X,
   `uv` Y (the user's SIMCA practice). This changes the numbers of new fits
   made with the default, so it goes in the CHANGELOG; old runs keep theirs.
2. **Exclusion and equal weighting.** Settled for Phase 5: excluded pixels
   are not replaced (an ROI keeps at least 1 − `max_fraction` of its
   pixels); replacement can be a later formula version.
3. **DModX weighting.** Settled: *v* is confirmed by the SIMCA exports;
   the unweighted `dmodx-1` is the default, the MPow-weighted `dmodx-2`
   (SIMCA's default) an option (2026-11-07).
4. **Sample groups (Phase S).** Settled: labels are an optional ROI
   attribute written only by `save_roi_sample`; `target` grouping works with
   both fold rules.

## Follow-up plan: SIMCA statistics and filters

- R²X, R²Y per component and cumulative; Q² and Q²(cum) (product form with
  SIMCA's truncation); RMSEE (N − 1 − A); R²adj; VIP; CoeffCS
- Autofit rules R1, R2, N3, N4
- Hotelling T² with SIMCA's limit A(N−1)/(N−A)·F and the score-plot ellipse;
  DModXPS and DModXPS+ for predictions
- Savitzky-Golay without edge bands; AsLS correction and smoothing (SIMCA
  defaults λ = 10000, p = 0.001)
- filter-once mode for stateful filters (SIMCA filters the dataset once)
- reference tests against SIMCA 18 exports (CV-group list, summary of fit,
  scores, coefficients, T², DModX)
