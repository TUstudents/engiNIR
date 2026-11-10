# Plan: prediction uncertainty (leverage formula)

Status: implemented (2026-11-10). The user's answers of 2026-11-10
replace the proposals; see Decisions.

## Goal

Every prediction gets an interval, ŷ ± q · s, that says how precise it is,
and no interval where the model does not apply. The method is the leverage
formula of the NIR standards (ASTM E1655, ISO 12099): the error of the
calibration, inflated by how far the new spectrum lies from the centre of
the training data in the model plane. It is cheap (no refits), works chunk
by chunk, and builds on what prediction diagnostics already compute: the
leverage of a pixel follows directly from its T²PS.

## The formula

For a model with A components fitted on N pixels from R ROIs:

- **Leverage of a pixel:** h = 1/R + T²PS / (R − 1), with the R training
  ROIs as the independent samples (user, 2026-11-10). With orthogonal
  scores and score variances over N − 1 (as stored), T²PS / (N − 1) equals
  tᵀ(TᵀT)⁻¹t, the leverage term of the standards; with N pixels it vanishes
  (N = 396 000 on the user's PLS 7 LV gave ±4.437 everywhere), with R = 20
  ROIs the interval runs from ±4.59 (5th percentile) to ±6.31. No new
  per-pixel computation: h comes from the T² map.
- **Standard error of a pixel prediction:** s · √(1 + h), with s
  the model's RMSEE at the level of the interval (below).
- **Interval:** ŷ ± q · s · √(1 + h), q the Student t quantile at the chosen level with
  R − A − 1 degrees of freedom (the ROIs are the independent samples; for
  R − A − 1 < 1 no interval).
- **Outside the model:** pixels above DCrit or T²Crit (0.99) get no interval
  (NaN). The leverage formula assumes the new spectrum resembles the
  training spectra; beyond the limits that assumption fails, and a number
  would be misleading.

## Pixels and means are different questions

The reported errors of engiNIR are errors of ROI means (the contract
reports metrics per ROI). A single pixel scatters more than a mean.
Therefore two intervals, with the same formula and different inputs:

| Interval | s | Leverage | Used for |
|---|---|---|---|
| Pixel | `pixel_rmsee` (fit, N − A − 1) | pixel T²PS, R ROIs as samples | the Uncertainty map layer |
| Mean (ROI, object) | `rmsee` (ROI level, R − A − 1) | the mean spectrum's scores against the ROI-mean training scores (`roi_scores`, R′ ROIs with a fitted pixel): h = 1/R′ + (t̄ − m)ᵀ C⁻¹ (t̄ − m) / (R′ − 1), C the covariance of the ROI scores | `RoiStatistics` of Predict ROIs (library); per-object results once segmentation exists |

ROI-mean scores are not orthogonal, so the mean interval needs their mean m
and covariance C (A × A), computed from the stored `roi_scores` (already
stored since the ROI-scores change).

## Decisions

1. **Leverage formula only** (user, 2026-11-08). Conformal, bootstrap or
   jack-knife intervals are not part of this plan. SIMCA computes its
   prediction intervals (YPredPS confidence intervals) by jack-knifing, a
   different number; the docs say so.
2. **s is RMSEE, as in ASTM E1655** (user, 2026-11-10): `pixel_rmsee`
   (N − A − 1 degrees of freedom) for pixel intervals, `rmsee` (ROI level,
   R − A − 1) for mean intervals. Both are already stored as metrics. Because
   RMSEE is a fit error, the empirical coverage on the run's own
   cross-validation (below) is what tells whether the interval is honest for
   new data.
3. **Level 0.95 by default**, an argument of the functions that compute
   intervals; the viewer uses 0.95 and says so.
4. **No interval outside the model** (user, 2026-11-10): a pixel above
   DCrit or T²Crit (0.99), or a mean above the ROI-level T²Crit (0.99), gets
   NaN, counted as "no interval: outside the model".
5. **Nothing new per pixel is stored**: the half-width is computed from T²PS
   and DModXPS on demand (`Prediction.uncertainty_map(level)`).
6. **Viewer: the Uncertainty map layer only** (user, 2026-11-10). The mean
   intervals of prediction ROIs are in the library (`RoiStatistics`), not in
   the ROI table; the coverage is stored as metrics, not shown on the Fit
   page.
7. **Independent review** of the contract changes before the code (user,
   2026-11-10).

## Stored with the model

Nothing new for the intervals themselves: `pixel_rmsee`, `rmsee` and
`n_rois` are metrics, `n_train`, `n_components` and the 0.99 limits are in
`prediction_model`, and the ROI score mean and covariance are computed from
the stored `roi_scores` (runs of 1.3 with ROI scores). Only the coverage
metrics are new.

## Validation that the intervals mean something

The leverage formula is an approximation (constant error, small X error,
roughly normal residuals). The fit therefore reports its **empirical
coverage**: the share of held-out ROI predictions (from the run's own
cross-validation) that fall inside their mean intervals, and the same for
held-out pixel predictions. Stored as metrics (`interval_coverage_roi`,
`interval_coverage_pixel`) and shown next to RMSECV, so a user sees at once
whether "95 %" is about 95 % for this model. No automatic correction.

## Contracts (with an independent review)

- `prediction-runtime-contract_v3.md`, Applicability Diagnostics: the
  intervals, the formula, the degrees of freedom, no interval outside the
  limits, the summary fields; the uncertainty map is derived, not stored.
- `sample-set-model-run-contract_v3.md` §7.5: the stored inputs and the
  coverage metrics.
- `headless-workflow-contract_v3.md`: `model.predict(..., level=)`,
  `prediction.uncertainty_map`, `prediction_interval_of_mean`.
- `viewer-workspace-ui-contract_v3.md`: the Uncertainty layer.
- `docs/hdf5_format.md`, gap register (new gap).

## Phases

1. **Contracts and review.**
2. **Model side:** store `rmsecv`, `pixel_rmsecv`, `n_rois`, ROI score mean
   and inverse covariance; compute the empirical coverage from the run's
   cross-validation. Tests: the stored values against an independent
   computation; coverage counted correctly.
3. **Prediction side:** `uncertainty_map` from the T² map, NaN outside the
   limits; `prediction_interval_of_mean(spectra or scores)`; summary fields
   (median half-width, share without interval); `level`. Tests: the formula
   against an independent dense computation (h = diag of X(XᵀX)⁻¹Xᵀ on the
   scores); h of a training pixel equals its training leverage; chunked
   equals unchunked; pixels above DCrit have no interval (fault-injected);
   on synthetic data with known noise the held-out coverage is near the
   nominal level (tolerance from the binomial spread).
4. **Viewer:** an **Uncertainty** layer (± half-width at 0.95) next to
   Prediction, DModX and T², the coverage in the Fit page summary, the
   interval summary in the Predict summary.
5. **Docs** (user guide: what the interval means, pixel vs mean, why none
   outside the limits, how it differs from SIMCA's jack-knife intervals),
   API reference, CHANGELOG, four gates.

## Out of scope

- Conformal, bootstrap, jack-knife or ensemble intervals.
- Intervals for classification.
- Per-object intervals in the viewer (needs object segmentation; the
  library function is ready for it).
- Heteroscedastic error models (error depending on the predicted value).
