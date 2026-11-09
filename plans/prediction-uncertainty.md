# Plan: prediction uncertainty (leverage formula)

Status: proposed (2026-11-08), not started; decisions marked *proposed* need
the user's confirmation

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

- **Leverage of a pixel:** h = 1/N + T²PS / (N − 1). With orthogonal scores
  and score variances over N − 1 (as stored), T²PS / (N − 1) equals
  tᵀ(TᵀT)⁻¹t, the leverage term of the standards. No new per-pixel
  computation: h comes from the T² map.
- **Standard error of a pixel prediction:** s = s_cv · √(1 + h), with s_cv
  the model's cross-validated error at the level of the interval (below).
- **Interval:** ŷ ± q · s, q the Student t quantile at the chosen level with
  R − A − 1 degrees of freedom (the ROIs are the independent samples; for
  R − A − 1 < 1 no interval).
- **Outside the model:** pixels above DCrit or T²Crit (0.99) get no interval
  (NaN). The leverage formula assumes the new spectrum resembles the
  training spectra; beyond the limits that assumption fails, and a number
  would be misleading.

## Pixels and means are different questions

The cross-validated errors of engiNIR are errors of ROI means (the contract
reports metrics per ROI). A single pixel scatters more than a mean.
Therefore two intervals, with the same formula and different inputs:

| Interval | s_cv | Leverage | Used for |
|---|---|---|---|
| Pixel | `pixel_rmsecv` (held-out pixel predictions) | pixel T²PS, N pixels | the Uncertainty map layer |
| Mean (ROI, object) | `rmsecv` (ROI level) | the mean spectrum's scores against the ROI-mean training scores (`roi_scores`, R ROIs): h = 1/R + (t̄ − m)ᵀ C⁻¹ (t̄ − m) / (R − 1), C the covariance of the ROI scores | `prediction_interval_of_mean(...)`; per-object results once segmentation exists |

ROI-mean scores are not orthogonal, so the mean interval needs their mean m
and covariance C (A × A, stored with the model; `roi_scores` are already
stored since the ROI-scores change).

## Decisions

1. **Leverage formula only** (user, 2026-11-08). Conformal, bootstrap or
   jack-knife intervals are not part of this plan. SIMCA computes its
   prediction intervals (YPredPS confidence intervals) by jack-knifing, a
   different number; the docs say so.
2. *Proposed:* **s_cv is the cross-validated error** (RMSECV, at pixel or ROI
   level), not RMSEE as in ASTM E1655: it is the honest, slightly more
   conservative estimate and the one engiNIR reports. It includes bias.
3. *Proposed:* **level 0.95 by default**, `model.predict(..., level=0.95)`;
   the viewer uses 0.95 and shows it.
4. *Proposed:* **no interval outside DCrit or T²Crit** (NaN, counted in the
   summary as "no interval: outside the model").
5. *Proposed:* **the map is not stored per pixel**: the half-width is
   computed from the T² map on demand (`prediction.uncertainty_map`), so a
   prediction holds no extra full-size array until it is asked for.
6. **No backward compatibility:** models without the stored inputs (below)
   predict without intervals until refitted.

## Stored with the model (`prediction_model`)

- `rmsecv`, `pixel_rmsecv` (copies of the metrics, so prediction needs only
  the prediction model), `n_rois`
- `roi_score_mean` (A,), `roi_score_cov_inv` (A × A) for the mean interval
  (pseudo-inverse with a warning if the ROI scores are rank-deficient,
  e.g. R ≤ A)
- `n_train` and `score_variances` are already stored

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
