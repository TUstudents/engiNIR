# Plan: choosing the number of components

Status: decided (2026-11-07), not started

## Goal

Users choose `n_components` by hand today, with no curve to guide them. The
number of components is the decision that most often makes an NIR model
over- or under-fit. Every new PLS/PCR run should show how cross-validated
prediction changes with the number of components, suggest a number by a
documented rule, and let the user fit with it, in scripts and in the viewer.

The physical reading stays with the user: the first components of spectra
are often concentration and scatter (particle size, surface roughness), and
the weights and loadings of each component should show it.

## Background (discussion 2026-11-07)

- **The curve.** For *a* = 1 … *A*max: R²Y(cum) (fit, only rises), Q²(cum)
  and RMSECV (cross-validated, rise then level off or fall), R²X(cum) (how
  much of the spectra is described). A widening gap between R² and Q² marks
  components that fit noise.
- **Rules** (weakest to most robust): lowest RMSECV (overfits); the
  smallest *a* within one standard error of the lowest RMSECV ("one-SE");
  a minimum gain per component (SIMCA's Autofit: Q²ₐ above 0.05 for
  N ≤ 100, above 0 otherwise); the van der Voet randomization test; and in
  the end, interpretable weights and loadings.
- **HSI.** The effective sample size is the number of ROIs (or samples),
  not pixels; folds must be grouped (ROI or sample, Phase S of the
  fit-specification plan), otherwise extra components learn each ROI's
  peculiarities and RMSECV keeps falling. Q² and RMSECV are per ROI (ROI
  means). Preprocessing changes the number (scatter correction or a
  derivative removes physical components): compare preprocessings each at
  its own suggested number. R² > 0.95 is a fit statistic, not a criterion; an
  RMSECV far below the reference method's error signals overfitting.

## Decisions (taken with the user 2026-11-07)

1. **Default rule.** One-SE on ROI-level RMSECV. Shown next to it: lowest
   RMSECV and the SIMCA-like Q² limit. Wold's R is not offered (not needed).
2. **Curve for every fit.** Every PLS/PCR
   fit stores the curve up to *A*max (the cross-validation then fits *A*max
   components per fold instead of *A*; PLS gives the predictions for every
   *a* ≤ *A*max from one fit, so the cost grows with *A*max, not with the
   number of candidates).
3. ***A*max.** min(15, *G* − 2, the rank limit), *G* the number
   of fold groups; a user value up to the rank limit.
4. **With DModX exclusion.** The exclusion depends on *A*. In the first
   version: the curve is computed without exclusion and says so; the
   chosen model is then fitted with exclusion at the chosen *A*.
   Alternative: exclusion at every *a* (cost × *A*max).
5. **Q² definition.** engiNIR refits every fold (stricter than SIMCA's
   component-wise CV, fit-specification plan Phase R): Q²(cum) = 1 − PRESS /
   SS(y), Q²ₐ = 1 − PRESSₐ / SSₐ₋₁ (SSₐ₋₁: residual sum of squares of the
   fitted model with *a* − 1 components, SS₀ = SS(y)). The numbers differ from
   SIMCA's; the docs say why.

## Contract changes (Phase 0, with an independent review)

- `sample-set-model-run-contract_v3.md`
  - §7: `n_components` may be requested as `"auto"`; the run stores the
    chosen integer in `hyperparameters/n_components` (the model stays
    `Backend(**hyperparameters)`) and the request in a new
    `fit_spec/components` part: `max`, `rule` (`one_se`, `min_rmsecv`,
    `q2_limit`, or `fixed` when the user gave a number),
    `selected`. Every key written.
  - new §7.4 "Component curve": the definitions above (ROI-level RMSECV and
    its standard error, Q²(cum), Q²ₐ, R²X(cum), R²Y(cum)), the rules with
    their exact formulas and tie handling, and that the curve uses the same
    folds, groups, pixel sample and scaling as the run.
  - §6 refit: a refit uses the stored `n_components` (never re-chooses), and
    reports when the stored rule would now choose differently.
  - §5 reporting: RMSEE and R²adj (Phase 2b) next to the primary metrics,
    with their degrees of freedom; CoeffCS, training T² with T²Crit and
    training DModX with DCrit as stored diagnostics.
- `docs/hdf5_format.md`: `fit_spec/components/`, the new metrics and
  diagnostics of Phase 2b, and
  `artifact_refs/component_curve/` (`n_components`, `rmsecv`, `rmsecv_se`,
  `q2`, `q2_cum`, `r2x_cum`, `r2y_cum`, `suggested` per rule).
- Legacy runs: no curve, `components/rule = fixed`.
- Ledger: new gap G24.

## Phases

### Phase 1: predictions for every number of components

`PLSRegression.predict_components(X, n)` and the same for `PCR`
(coefficients of the first *n* components from one fit:
B = W(PᵀW)⁻¹Qᵀ with the first *n* columns; PCR: regression on the first *n*
scores). Tests: equal to a fresh fit with *n* components, for every scaling
and PLS1/PLS2.

### Phase 2: the curve in the fit

`cross_validate` gains an option returning out-of-fold predictions for
*a* = 1 … *A*max; `fit_model_run` computes the curve at ROI level with the
run's folds and groups, stores `artifact_refs/component_curve` and the
summary in `metrics` (`rmsecv` at the chosen *A* stays the primary
metric). Tests: the curve at *A* equals the run's RMSECV; folds and groups
are those of the run; R²Y(cum) never decreases.

### Phase 2b: SIMCA statistics of the fitted model (added 2026-11-07)

From the SIMCA follow-up list of the fit-specification plan; computed in
the same fit, shown on the pages of Phase 4.

- **RMSEE** = √(Σ res² / (N − 1 − A)) and **R²adj**: at pixel level (N
  pixels, as SIMCA computes them, so the numbers compare) and at ROI level
  (R ROI means; reported only when R − 1 − A ≥ 1). Stored in `metrics`.
- **CoeffCS**: coefficients for scaled and centred X and scaled Y (SIMCA's
  default coefficient view), so band importance compares across scalings.
  Stored with the diagnostics.
- **Training T²** per pixel and per ROI mean, with T²Crit = A(N − 1)/(N − A)
  · F(level; A, N − A) at 0.95 and 0.99, and the T² ellipse of the score plot
  (components 1 and 2, T²Crit for A = 2).
- **Training DModX** per pixel (unweighted `dmodx-1`, normalised) with
  DCrit at 0.99, also when no pixel exclusion was requested; the share of
  pixels above DCrit per ROI.

Tests: RMSEE, R²adj, T², T²Crit, DModX and DCrit against the formulas
verified with SIMCA 18.1 (fit-specification plan, Phase R) on the SIMCA
reference blends; CoeffCS equals the raw coefficients rescaled.

### Phase 3: rules and `n_components="auto"`

`enginir.validation.choose_components(curve, rule)` (public, documented
formulas); `project.fit(..., n_components="auto", fit_spec={"components":
{"rule": ..., "max": ...}})`. Tests: every rule on constructed curves (ties,
flat curves, monotone curves); a synthetic data set with a known number of
independent sources (e.g. 2 concentrations + scatter + baseline) where the
one-SE rule finds it; fault injection (rule ignored, SE wrong).

### Phase R: the rules on the SIMCA reference blends

Before choosing the default: apply every rule to M1–M5 of the reference
project (pixels and ROI means) and to one real data set from the user;
record where the rules disagree. Compare with SIMCA's Autofit (M4 chose 2).

Results on the reference blends (2026-11-07; 18 ROIs with values, 7
round-robin folds, A_max 15; RMSECV per ROI):

| Preprocessing | Pixels per ROI | one_se | min_rmsecv | q2_limit |
|---|---|---|---|---|
| none | 10 (seed 2026) | 5 | 5 | 5 |
| none | all (48) | 7 | 8 | 6 |
| SNV | 10 | 3 | 3 | 3 |
| SNV | all | 3 | 3 | 3 |
| SG 1st derivative, 15 points | 10 | 2 | 2 | 2 |
| SG 1st derivative, 15 points | all | 4 | 5 | 2 |

- SNV: all rules choose 3, as SIMCA's export showed (M2: Q² best at 3).
- The rules disagree where the curve has a long flat tail (derivative, all
  pixels: RMSECV 0.0020 ± noise from 4 to 15 components); one_se stops
  before the tail, min_rmsecv walks into it, q2_limit stops earliest
  because a small gain fails the limit.
- More pixels per ROI make extra components look better (the pixel noise
  averages out within an ROI); the default one_se is a reasonable middle.
- Pending: one real data set from the user (not available during the
  unattended run).

### Phase 4: viewer

- Model diagnostics: a **Components** page with RMSECV (error bars) and
  R²Y(cum)/Q²(cum) against *a*, the suggestion of each rule marked, and the
  run's *A* highlighted; clicking *a* offers "Fit with *a* components" (same
  target, recipe and fit specification).
- The Spectral page shows weights and loadings of the first components (not
  only LV1), so the physical meaning can be checked, and CoeffCS next to the
  raw coefficients.
- The score plot gets the T² ellipse (0.95 and 0.99); a **Distance** panel
  shows training DModX (pixels or ROI means) with DCrit at 0.99 and T² with
  T²Crit; the metrics table shows RMSEE and R²adj.
- Fit dialog: Components = a number or **Auto** (rule, maximum).
- Viewer docs, contract and architecture notes in sync.

### Phase 5: docs and release notes

User guide section "Choosing the number of components" (the curve, the
rules, HSI notes, preprocessing comparison), API reference, CHANGELOG,
four gates.

## Out of scope

- Choosing the preprocessing automatically (needs nested cross-validation
  to stay honest; separate plan).
- SIMCA's component-wise cross-validation as an option (reproduced in the
  SIMCA handover; deferred 2026-11-07).
- Autofit's other rules (R2, N3, N4), DModXPS+ (deferred 2026-11-07).
- SIMCA preprocessing (AsLS, Savitzky-Golay without edges): plan
  `simca-preprocessing.md`.
- OPLS.
