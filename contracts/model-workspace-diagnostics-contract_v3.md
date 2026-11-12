# Model workspace diagnostics contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.5; the file format `enginir-1.5` is the
  only format engiNIR reads and writes (`project-schema-contract_v3.md`)
- Scope: `Model` workspace visual hierarchy, required diagnostics, PLS/PCR plot contract, persistence expectations for reopening saved models

## Goal

Define the concrete diagnostics contract for the viewer `Model` workspace so
that fitting is not just a form submission followed by a status line.

The `Model` workspace must answer these questions immediately:

- what target is being modeled
- whether the current calibration set is valid and ready to fit
- whether the fitted model is good enough
- where the model fails
- what the latent variables are doing
- which wavelengths drive the model

This contract exists to prevent the `Model` workspace from regressing into:

- controls without diagnostics
- text-only summaries with no scientific plots
- diagnostics that require refitting because they were not persisted
- PLS-specific views that do not generalize to PCR

## Dependencies

This contract depends on:

- `viewer-workspace-ui-contract_v3.md`
- `sample-set-model-run-contract_v3.md`
- `dataset-calibration-contract_v3.md`

If this contract conflicts with a looser UI convenience, scientific
interpretability wins.

## Core decisions

- the `Model` workspace is a diagnostics workspace, not only a fit-control form
- one model has one primary diagnostics view and multiple secondary diagnostics
- `Predicted vs target value` is the required primary plot after fit
- diagnostics must render for reopened saved `ModelRun`s without refitting
- the diagnostics shell is model-agnostic first, with PLS/PCR-specific detail plots layered underneath
- ROI-grouped validation semantics from `sample-set-model-run-contract_v3.md` remain authoritative

## Layout contract

The `Model` workspace must expose three functional regions:

1. left control area
2. center primary diagnostics area
3. right summary and secondary diagnostics area

Required meanings:

### Left control area

Must contain:

- target selection / definition entry point
- target assignment entry point (the ROI table)
- fit-model entry point: one dialog with target, signal unit, preprocessing,
  model, components and CV. It builds or reuses the calibration set (all ROIs with
  a value for the target, from all datasets), so there is no separate
  calibration-set build step. The preprocessing defaults to the Explore preview,
  but is shown and chosen in the dialog, never taken silently.
- fit preflight / readiness state, including the number of labeled ROIs per
  dataset and the ROIs left out (for example by a different wavelength
  axis)
- fit hyperparameters and CV controls

### Center primary diagnostics area

Must be dominated by model-quality feedback, not by the Explore plots.

Required v3 primary plots:

- `Predicted vs target value`
- `Residuals`

Default emphasis after a successful fit:

- `Predicted vs target value`

The center area is a set of named pages of plot panels. It follows the Plot
Panel Policy in `viewer-workspace-ui-contract_v3.md`: every diagnostic can be
maximized and exported as PNG, SVG and CSV.

Required v3 pages, in this order:

1. `Fit`
   - `Predicted vs target value` in the first slot
   - `Residuals`
   - target distribution
   - latent scores, with the Hotelling T² ellipses at 0.95 and 0.99 of the
     first two components: of the calibration pixels
     (the pixels the model is fitted on), and of the ROIs (computed from their
     mean scores: centered on their mean, shaped by their
     covariance, limit `2(n−1)/(n−2)·F(level; 2, n−2)` for `n` ROIs, at
     least 4), with a legend naming both
2. `Spectral`
   - regression coefficients
   - scaled coefficients (CoeffCS)
   - loadings of the first (up to) three components
   - VIP (PLS): VIP over
     wavelength with lines at 1.0 and 0.8 and the model's mean spectrum
     shaded behind it, and VIP sorted from the largest value (bands as bars)
   - selectivity ratio (PLS and PCR) over wavelength with
     the mean spectrum behind it
   Plots over wavelength leave a gap where a band selection leaves out
   bands
3. `Components` (`sample-set-model-run-contract_v3.md` §7.4),
   2 × 2, every plot naming its level:
   - **RMSECV (per ROI)** with its standard error, RMSEE (per ROI) as a
     second line, the model's number marked, and every rule's
     suggestion listed with its reason (e.g. "Q² limit: 1 (Q²₂ = −0.02 ≤
     0.05)"); with the noise guard, the rule's number and the guarded
     number both
   - **R²Y and Q² per ROI**: R²Y(cum) of the ROI means and Q²(cum), the same
     level, so they are comparable; the gap between them shaded, and drawn
     as it is when Q² exceeds R²Y (possible: PRESS can be below the RSS of a
     fit made on pixels)
   - **per component**: Q²ₐ (bars, per ROI, with the R1 limit *L* as a
     line), ΔR²X (bars, per pixel) and the Durbin–Watson value of the
     weights (line on its own axis); the legend names the
     level of each; noise components drawn gray
   - **per pixel**: R²X(cum) and R²Y(cum) over the drawn pixels, with a note
     that single pixels are noisy, so R²Y per pixel stays below the ROI value
4. `Distance` (`sample-set-model-run-contract_v3.md` §7.5)
   - the calibration pixels' normalized DModX with DCrit (0.99) and the share
     above it next to the 1 % a 0.99 limit expects; bins up to the 99.5th
     percentile and the pixels above it as one overflow bar (count and
     largest value stated), so a long tail does not hide the distribution
   - their Hotelling T² with T²Crit (0.99), drawn the same way
   - per ROI: the share of its calibration pixels above DCrit (bars, colored
     by target value) and above T²Crit (markers), sorted from the largest
     DModX share, with the expected 1 % as a line; a bar selects its ROI

Every model stores every diagnostic of these pages; there are no "refit"
placeholders. A value that is not defined in a case (NaN, e.g. DCrit when
K − A < 1 or RMSEE (per ROI) when R − 1 − A < 1) or missing is shown as "–"
in its place, with the reason where it is known; the pages never fail on
missing display data (`project-schema-contract_v3.md`, Core and display
data).

The `Fit` page is shown after a fit and when a saved run is opened. The page
labels are not contract-frozen; their order and content are. New diagnostics
add slots or pages and must not displace `Predicted vs target value` from the
first slot of the first page.

### Right summary / secondary diagnostics area

Must contain:

- fit summary
- calibration-set summary

Required v3 secondary diagnostics, shown as panels on the center pages:

- target distribution
- latent scores
- spectral diagnostics

### Levels: per ROI and per pixel

Every metric, axis, legend entry, CSV column and summary row names its
level:

- **per ROI**: computed over ROI means (the target, the folds and the
  reported model quality): R², Q², RMSECV (per ROI), RMSEE (per ROI), bias, RPD, the ROI T²
  ellipses, the per-ROI shares above the distance limits
- **per pixel**: computed over single spectra (the pixels the model is fitted on):
  R²X, R²Y over pixels, `pixel_*` metrics, DModX, T² and their limits

A number answering "how good is the model" is per ROI by default. A plot may
draw both levels only when its legend names the level of every element.

### Plot help

Every Model diagnostics plot has a help text (what is drawn, at which
level, how to read it, what a bad sign looks like), opened from the plot
header (Plot panel policy); the user guide carries the same text.

## Required fit summary

The `Model` workspace must display the following current-run summary fields:

- target name
- target unit
- calibration set id or label
- model type
- number of components
- CV specification
- input-contract / preprocessing context
- ROI count, and the number of datasets they come from
- pixel count
- readiness state

If no fitted run exists yet, the summary must still show the current intended
fit context and block reasons.

## Required metrics

v3 primary reported metrics must reflect ROI-grouped validation.

Required displayed metrics, in a *Per ROI* group:

- `R²`
- `RMSECV`
- `bias`

with `Q²`, `RMSEE`, `R²adj` and `RPD` when the run has them, and, for runs
that store them, a *Per pixel* group with the `pixel_*` metrics (`R²`,
`RMSECV`, `RMSEE`, `RPD`) and the pixel count. The `Predicted vs
target value` statistics name their level (per ROI). The fit summary shows the
fold warnings of the run (`cv_spec/@warnings`) in plain words, and its long
values (the validation scheme, names) wrap instead of being cut off.

Metric definitions in v3:

- all primary metrics are computed from one aggregated cross-validated
  prediction per held-out ROI (arithmetic mean of that ROI's held-out pixel
  predictions), per `sample-set-model-run-contract_v3.md`
- `bias = mean(y_pred - y_ref)` over those ROI-level pairs, so a positive bias
  means over-prediction; this matches the residual definition below and
  `enginir.validation.metrics.bias`
- `R²` and `RMSECV` are the coefficient of determination and root mean square
  error of cross-validation over the same ROI-level pairs
- the persisted metric keys are documented in `docs/hdf5_format.md`;
  `pixel_*`-prefixed variants are secondary per-pixel diagnostics and must be
  labeled as such wherever shown

Optional but recommended:

- `RPD`
- held-out ROI count

Forbidden v3 behavior:

- showing naive per-pixel validation metrics as the primary fit quality signal
- mixing ROI-level and pixel-level metrics without explicit labeling

## Required primary plot: predicted vs target value

The primary diagnostics plot must show:

- x-axis: target value of the ROI
- y-axis: cross-validated predicted ROI value
- one point per held-out ROI
- a 1:1 reference line

Required annotations:

- `R²`
- `RMSECV`
- `bias`

Recommended behavior:

- point selection should map back to the originating ROI in the project browser
- selected ROI should propagate to the inspector

## Required primary plot: residuals

The residual diagnostics plot must show:

- x-axis: target value or predicted value of the ROI
- y-axis: residual (`y_pred - y_ref`)
- one point per held-out ROI
- a zero reference line

Purpose:

- expose bias trends
- expose heteroscedasticity
- expose nonlinearity not obvious in the `Predicted vs target value` plot

## Secondary diagnostics contract

### 1. Target distribution

Required content:

- ROI-level target distribution for the active calibration set

Purpose:

- reveal a narrow calibration range
- reveal imbalance or sparse target coverage

### 2. Latent scores

Required for PLS and PCR:

- scores plot in latent space
- default view: LV1 vs LV2

Required behavior:

- one point per ROI, not one point per pixel
- the ROI-level score is the arithmetic mean of the calibration scores of that
  ROI's fitted pixels (the drawn pixels with a pixel subsample, the kept pixels
  with DModX exclusion), computed and stored by the fit (`diagnostics/
  roi_scores`, sample-set-model-run contract §7.5); the viewer never
  reconstructs it from pixel counts
- an ROI without fitted pixels has no point, and the plot says how many
  ROIs that are
- ROI selection should be possible or at least inspectable
- the plot must label the scores as calibration scores, so that they are not
  mistaken for cross-validated scores

### 3. Spectral diagnostics

Required v3 spectral diagnostic families:

- regression coefficients
- loadings

Stored by every model (`sample-set-model-run-contract_v3.md` §7.5):

- VIP (PLS)
- selectivity ratio (PLS and PCR)

What changes VIP and the selectivity ratio, and the docs must say so: the
number of components, the preprocessing (derivatives move importance to
the flanks of a band) and X scaling (the selectivity ratio only through the
fitted regression vector).

Required axis:

- x-axis: wavelength when available, otherwise band index

Purpose:

- explain what spectral regions drive the model
- support scientific interpretation, not only fit acceptance

## PLS-specific expectations

For `PLS`, the diagnostics contract must support these persisted quantities when
available:

- regression coefficients
- x scores
- x loadings
- x weights
- intercept
- component count
- spectral axis used for plotting

The UI must expose at minimum:

- `Predicted vs target value`
- `Residuals`
- `Scores`
- `Coefficients`

Recommended:

- `Loadings`
- `VIP`

## PCR expectations

`PCR` reuses the same diagnostics shell.

Required minimum plots:

- `Predicted vs target value`
- `Residuals`
- `Scores`
- `Coefficients` or reconstructed regression weights if exposed by the runtime artifact

Recommended:

- `Loadings`

The UI must not special-case PLS so heavily that PCR becomes second-class or
empty.

## Persistence contract

Diagnostics must reopen from a persisted `ModelRun` without requiring refit.

Required persisted or reconstructable information:

- ROI-level target values
- ROI-level cross-validated predictions
- ROI-level residuals
- primary ROI-level metrics
- component count
- model type
- target identity
- calibration-set identity
- preprocessing / input-contract identity
- spectral axis for coefficient/loadings plots

Required when supported by the fitted backend artifact:

- latent scores
- coefficients
- loadings

Required as well (stored by every model): VIP (PLS), the selectivity
ratio, the component curve and the calibration distances.

Forbidden behavior:

- refitting the model just to render diagnostics for a saved `ModelRun`

## Selection and coordination

The `Model` workspace must coordinate with the persistent browser/inspector.

Required behavior:

- selecting a `ModelRun` in the browser updates the diagnostics workspace
- selecting a plotted ROI point should update browser/inspector selection when feasible
- browser selection remains authoritative when an explicit object is selected

## Empty and preflight states

If the user is not ready to fit, the `Model` workspace must still provide useful
diagnostic readiness information.

Required blocking/preflight states include:

- no target selected
- no ROIs with a value for the selected target
- invalid CV grouping
- reference correction or input-contract mismatch

Required UI behavior:

- show the blocking reason in the fit summary / readiness region
- do not launch a background fit job for obviously invalid runs

## Non-goals for v3

This contract does not require:

- per-pixel diagnostic plots as the primary modeling view
- variable selection tooling
- multi-target modeling in one workspace
- editable plot compositions by end users: users can maximize and export
  panels, but cannot change which plot occupies which slot
- advanced dashboards of spectra outside the model

These may be added later, but the v3 shell must remain simple and reliable.

## Acceptance criteria

v3 is satisfied when:

- the `Model` workspace has a true diagnostics-first primary area
- `Predicted vs target value` and `Residuals` are present after fit
- PLS and PCR both show meaningful secondary diagnostics
- reopened `ModelRun`s render diagnostics without refit
- ROI-grouped validation semantics remain explicit in the UI
- fit readiness is visible before the user starts a background job
- every diagnostic is a plot panel on the required pages and can be maximized
  and exported as PNG, SVG and CSV
