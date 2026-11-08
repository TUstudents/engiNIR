# Model Workspace Diagnostics Contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.0; v3 is the default and only supported project schema
- Scope: `Model` workspace visual hierarchy, required diagnostics, PLS/PCR plot contract, persistence expectations for reopening saved model runs

## Goal

Define the concrete diagnostics contract for the viewer `Model` workspace so
that fitting is not just a form submission followed by a status line.

The `Model` workspace must answer these questions immediately:

- what target is being modeled
- whether the current sample set is valid and ready to fit
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

## Core Decisions

- the `Model` workspace is a diagnostics workspace, not only a fit-control form
- one model run has one primary diagnostics view and multiple secondary diagnostics
- `Predicted vs Reference` is the required primary plot after fit
- diagnostics must render for reopened saved `ModelRun`s without refitting
- the diagnostics shell is model-agnostic first, with PLS/PCR-specific detail plots layered underneath
- ROI-grouped validation semantics from `sample-set-model-run-contract_v3.md` remain authoritative

## Layout Contract

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
  model, components and CV. It builds or reuses the sample set (all ROIs with
  a value for the target, from all data files), so there is no separate
  sample-set build step. The preprocessing defaults to the Explore preview,
  but is shown and chosen in the dialog, never taken silently.
- fit preflight / readiness state, including the number of labeled ROIs per
  data file and the ROIs left out (for example by a different wavelength
  axis)
- fit hyperparameters and CV controls

### Center primary diagnostics area

Must be dominated by model-quality feedback, not by the Explore plots.

Required v3 primary plots:

- `Predicted vs Reference`
- `Residuals`

Default emphasis after a successful fit:

- `Predicted vs Reference`

The center area is a set of named pages of plot panels. It follows the Plot
Panel Policy in `viewer-workspace-ui-contract_v3.md`: every diagnostic can be
maximized and exported as PNG, SVG and CSV.

Required v3 pages, in this order:

1. `Fit`
   - `Predicted vs Reference` in the first slot
   - `Residuals`
   - target distribution
   - latent scores, with the Hotelling T² ellipses at 0.95 and 0.99 of the
     first two components (engiNIR 1.3 and later)
2. `Spectral`
   - regression coefficients
   - scaled coefficients (CoeffCS, 1.3 and later)
   - loadings of the first (up to) three components
3. `Components` (1.3 and later; `sample-set-model-run-contract_v3.md` §7.4)
   - RMSECV with its standard error against the number of components, the
     model's number and every rule's suggestion marked
   - R²Y(cum), Q²(cum) and R²X(cum) against the number of components
4. `Distance` (1.3 and later; §7.5)
   - the training pixels' normalised DModX with DCrit (0.99) and the share
     above it
   - their Hotelling T² with T²Crit (0.99) and the share above it

Runs fitted before engiNIR 1.3 show "refit" placeholders on the new pages.
The fit summary shows RMSEE and R²adj when the run has them.

The `Fit` page is shown after a fit and when a saved run is opened. The page
labels are not contract-frozen; their order and content are. New diagnostics
add slots or pages and must not displace `Predicted vs Reference` from the
first slot of the first page.

### Right summary / secondary diagnostics area

Must contain:

- fit summary
- sample-set summary

Required v3 secondary diagnostics, shown as panels on the center pages:

- target distribution
- latent scores
- spectral diagnostics

## Required Fit Summary

The `Model` workspace must display the following current-run summary fields:

- target name
- target unit
- sample set id or label
- model type
- number of components
- CV specification
- signal-domain / preprocessing context
- ROI count, and the number of data files they come from
- pixel count
- readiness state

If no fitted run exists yet, the summary must still show the current intended
fit context and block reasons.

## Required Metrics

v3 primary reported metrics must reflect ROI-grouped validation.

Required displayed metrics:

- `R²`
- `RMSECV`
- `bias`

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

## Required Primary Plot: Predicted vs Reference

The primary diagnostics plot must show:

- x-axis: reference ROI value
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

## Required Primary Plot: Residuals

The residual diagnostics plot must show:

- x-axis: reference or predicted ROI value
- y-axis: residual (`y_pred - y_ref`)
- one point per held-out ROI
- a zero reference line

Purpose:

- expose bias trends
- expose heteroscedasticity
- expose nonlinearity not obvious in the `Predicted vs Reference` plot

## Secondary Diagnostics Contract

### 1. Target distribution

Required content:

- ROI-level target distribution for the active sample set

Purpose:

- reveal narrow calibration ranges
- reveal imbalance or sparse target coverage

### 2. Latent scores

Required for PLS and PCR:

- scores plot in latent space
- default view: LV1 vs LV2

Required behavior:

- one point per ROI, not one point per pixel
- the ROI-level score is the arithmetic mean of the training scores of that
  ROI's fitted pixels (the drawn pixels with a pixel sample, the kept pixels
  with DModX exclusion), computed and stored by the fit (`diagnostics/
  roi_scores`, sample-set-model-run contract §7.5); the viewer never
  reconstructs it from pixel counts
- a run without stored ROI scores (fitted before this rule) shows an
  explicit placeholder asking to fit the model again; an ROI without fitted
  pixels has no point, and the plot says how many ROIs that are
- ROI selection should be possible or at least inspectable
- the plot must label the scores as training scores, so that they are not
  mistaken for cross-validated scores

### 3. Spectral diagnostics

Required v3 spectral diagnostic families:

- regression coefficients
- loadings

Optional but recommended in v3:

- VIP

Required axis:

- x-axis: wavelength when available, otherwise band index

Purpose:

- explain what spectral regions drive the model
- support scientific interpretation, not only fit acceptance

## PLS-Specific Expectations

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

- `Predicted vs Reference`
- `Residuals`
- `Scores`
- `Coefficients`

Recommended:

- `Loadings`
- `VIP`

## PCR Expectations

`PCR` reuses the same diagnostics shell.

Required minimum plots:

- `Predicted vs Reference`
- `Residuals`
- `Scores`
- `Coefficients` or reconstructed regression weights if exposed by the runtime artifact

Recommended:

- `Loadings`

The UI must not special-case PLS so heavily that PCR becomes second-class or
empty.

## Persistence Contract

Diagnostics must reopen from a persisted `ModelRun` without requiring refit.

Required persisted or reconstructable information:

- ROI-level reference values
- ROI-level cross-validated predictions
- ROI-level residuals
- primary ROI-level metrics
- component count
- model type
- target identity
- sample-set identity
- preprocessing / signal-domain identity
- spectral axis for coefficient/loadings plots

Required when supported by the fitted backend artifact:

- latent scores
- coefficients
- loadings

Optional in v3:

- VIP, if computable from persisted state without refit

Forbidden behavior:

- refitting the model just to render diagnostics for a saved `ModelRun`

## Selection and Coordination

The `Model` workspace must coordinate with the persistent browser/inspector.

Required behavior:

- selecting a `ModelRun` in the browser updates the diagnostics workspace
- selecting a plotted ROI point should update browser/inspector selection when feasible
- browser selection remains authoritative when an explicit object is selected

## Empty and Preflight States

If the user is not ready to fit, the `Model` workspace must still provide useful
diagnostic readiness information.

Required blocking/preflight states include:

- no target selected
- no labeled ROIs for the selected target
- no sample set built
- invalid CV grouping
- calibration or signal-domain mismatch

Required UI behavior:

- show the blocking reason in the fit summary / readiness region
- do not launch a background fit job for obviously invalid runs

## Non-Goals for v3

This contract does not require:

- per-pixel diagnostic plots as the primary modeling view
- variable selection tooling
- multi-target modeling in one workspace
- editable plot compositions by end users: users can maximize and export
  panels, but cannot change which plot occupies which slot
- advanced outlier dashboards

These may be added later, but the v3 shell must remain simple and reliable.

## Acceptance Criteria

v3 is satisfied when:

- the `Model` workspace has a true diagnostics-first primary area
- `Predicted vs Reference` and `Residuals` are present after fit
- PLS and PCR both show meaningful secondary diagnostics
- reopened `ModelRun`s render diagnostics without refit
- ROI-grouped validation semantics remain explicit in the UI
- fit readiness is visible before the user starts a background job
- every diagnostic is a plot panel on the required pages and can be maximized
  and exported as PNG, SVG and CSV
