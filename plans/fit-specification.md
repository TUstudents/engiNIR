# Plan: fit specification

Status: Phase 0 written and reviewed, Phase R tables written; SIMCA exports pending (2026-11-04)

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
  sample label above the ROI.
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

Tests (`tests/test_simca_reference.py`): each comparison is skipped with a
reason until the engiNIR feature exists, and switched on by the phase that
adds it (scaling: Phase 4; DModX: Phase 5; the rest: follow-up plan).
Tolerances are stated per quantity; signs of components are aligned before
comparing. The undocumented points (DModX factor *v*, AsLS iterations,
scaling inside CV rounds) are decided by these exports.

Steps: generate data and project, export the tables (engiNIR side, done by
the agent); build M1–M5 and export (done by the user in SIMCA 18); commit
the exports and the tests.

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

### Phase 4: scaling of X and Y
`PLSRegression` and `PCR` take `x_scaling` and `y_scaling` (centring and
scaling done by engiNIR, scikit-learn called with `scale=False`; variances
with N − 1). Tests: `uv`/`uv` equals today's results; `center` X gives
the textbook PLS1 on centred data; coefficients in original units agree
across scalings.

### Phase 5: DModX and pixel exclusion
`dmodx(...)` and `dmodx_limit(...)` in `validation/outliers.py` with SIMCA's
formulas (normalised; see open decision 3), then exclusion inside folds
and in the final model, stored excluded indices and flags. Tests: limit
against hand-computed F quantiles; an injected bright pixel is excluded;
exclusion never sees test-fold pixels (fault injection: exclusion on all
pixels must change the test).

### Phase 6: viewer
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
2. **Exclusion and equal weighting.** After exclusion, ROIs have fewer than
   *n* pixels. Accept that (simple), or replace excluded pixels with
   unused pixels of the same ROI that pass the DModX limit (keeps every ROI
   at *n*).
3. **DModX weighting.** SIMCA weights residuals by the modelling power by
   default; the correction factor *v* is not defined in the help. Use the
   unweighted normalised DModX now and add the weighted form when a SIMCA
   export confirms *v*, or wait for the export.

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
