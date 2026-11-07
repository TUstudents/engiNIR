# Plan: prediction diagnostics (DModXPS and T²PS maps), predictions not stored

Status: implemented (2026-11-07, branch feature/weekend-plans); the check on the SIMCA reference test set is pending (data not available)

## Goal

A prediction of a new scan shows, next to the predicted map, where the model
applies: the distance of every pixel to the model (DModXPS) and its
Hotelling T² in the model plane (T²PS), with the pixels above the critical
limits marked. Predictions are diagnostics of a model on new data, not
project content: the project file holds the calibration (spectra, recipe,
provenance, fitted models), so predictions and their maps are not stored.

## Decisions (taken with the user 2026-11-07)

- **Predictions are not stored by default.** The viewer never writes a
  prediction into the project. The prediction-run layout
  (`/predictions/{id}/`) and its writers stay in `enginir.io.project_store`
  for scripts that explicitly want to persist one; nothing calls them by
  default.
- **Predicted scans are not registered** as datasets in the project (no
  dataset or calibration is written by a prediction).
- **Trace:** one log entry per prediction (model run id and name, scan path,
  file size and SHA-256, time, summary values); nothing in the project.
- **Export:** the plot panels' existing export menu (PNG, SVG, data as CSV);
  no new export code.
- **DModXPS is unweighted** (residuals not weighted by the modelling power,
  like `dmodx-1`); SIMCA weights by default, the docs say so.
- **Limit level 0.99** for DCrit and T²Crit (as for the training exclusion).
- **No backward compatibility:** models fitted before get the maps after a
  refit; old projects keep their `/predictions/` group, which the viewer
  ignores.

## Computation (contract §: prediction runtime)

Unchanged principle: predictions run in chunks so that loading the raw
counts, calibrating, converting and preprocessing fit a memory budget. The
new values are computed in the same single pass; nothing needs the whole
image, because the limits come from the training data.

Per chunk, after the stored preprocessing:

1. scaled X = (X − x_mean) / x_std (the model's stored scaling)
2. scores T = scaled X · R (R the rotations, W(PᵀW)⁻¹ for PLS; the PCA
   components for PCR)
3. prediction from the coefficients (unchanged)
4. T²PS = Σₐ tₐ² / var(tₐ), var(tₐ) the training score variance (N − 1)
5. residual E = scaled X − T Pᵀ (in place in the scaled-X buffer where
   possible); DModXPS = √(Σₖ e² / (K − A)) / s₀, without the factor v, s₀ the
   training pooled residual SD (verified against SIMCA 18.1's DModXPS)
6. accumulate the summary: pixels above DCrit and above T²Crit (count and
   share), mean prediction inside and outside the limits, histograms

Limits (from the training N, K, A, level 0.99): DCrit as in
`enginir.validation.dmodx_limit`; T²Crit = A(N − 1)/(N − A) · F(0.99; A,
N − A). Notes for the docs: N is large and pixels are correlated, so about
1 % of normal pixels exceed each limit; the share above the limit matters
more than single pixels.

Memory: the residual reuses the scaled-X buffer or adds one chunk-sized
array (counted in the chunk-size budget). The outputs are 2-D maps (three
float32 values per pixel), kept in memory; for images whose maps exceed a
threshold, in a memory-mapped scratch file outside the project, deleted with
the result.

## What the model run stores (new runs)

In `artifact_refs/prediction_model` (model content, needed to apply the
model): `x_mean`, `x_std` (scaling), `rotations`, `loadings`,
`score_variances`, `s0` (training pooled residual SD, unweighted), `n_train`
(N), and `dcrit_099`, `t2_crit_099` for the record. `coef_`, `intercept_` and
the preprocessing state stay. A run without these fields predicts as before
but shows no DModXPS/T²PS (the viewer says "refit to get the maps").

## Contract changes (Phase 0, with an independent review)

- `prediction-runtime-contract_v3.md`: rewrite the persistence parts:
  prediction is computed in bounded-memory chunks; results stay in memory
  or a scratch file; persisting (`/predictions/{id}/`) is an explicit
  low-level operation, never a default; predicted scans are not registered;
  the log entry; DModXPS/T²PS definitions and limits; the summary values.
- `project-schema-contract_v3.md`: predictions and predicted datasets are no
  longer written by the facade or the viewer; the `/predictions/` layout
  remains for the low-level writers.
- `headless-workflow-contract_v3.md`: `model.predict(path, *, dark=None,
  white=None)` returns an in-memory result (prediction, DModXPS and T²PS
  maps, limits, summary; save helpers via the low-level writers if wanted)
  and writes nothing to the project.
- `viewer-workspace-ui-contract_v3.md`: the Predict workspace shows the
  three layers with the limit overlay and summary; no prediction is stored;
  the viewer's only direct project writer (the prediction worker storing
  its dataset and calibration) goes away, so the viewer writes only through
  `Project.save_*` without exception.
- `sample-set-model-run-contract_v3.md` / `docs/hdf5_format.md`: the new
  `prediction_model` fields.
- Ledger: new gap G25.

## Phases

### Phase 1: projection data in the model run
Store the fields above in new runs (PLS and PCR, also with MSC/EMSC
pipelines and DModX exclusion: the final model's values). Tests: the stored
values reproduce the training scores, residuals and s₀ of the fitted model;
round trip.

### Phase 2: DModXPS and T²PS in the chunked prediction
`predict_with_model_run` computes the three maps and the summary in the
chunk loop; the chunk-size budget counts the residual. Tests: chunked equals
unchunked; on the SIMCA reference test set (pixels_test) DModXPS and T²PS
equal an independent implementation of the formulas (the formulas were
verified against SIMCA 18.1 in the fit-specification plan, Phase R; the
exports themselves are not committed); an injected odd pixel exceeds DCrit;
memory stays within the budget (existing memory tests extended).

### Phase 3: predictions not stored, scans not registered
The facade and the viewer stop writing predictions, datasets and
calibrations for predictions; the log entry is written; the low-level
writers stay and keep their tests. Tests: a prediction leaves the project
file byte-identical (or at least without new groups); the viewer write test
(`test_viewer_writes_projects_only_through_project`) loses its exception;
fault injection: a write during prediction fails the test.

### Phase 4: viewer
Predict workspace: layer choice (prediction, DModXPS, T²PS) as plot panels
with the limit overlay (pixels above 0.99 marked), the summary (share
outside, mean prediction inside/outside), export through the panel menu.
"Refit to get the maps" for runs without projection data. Viewer docs in
sync.

### Phase 5: docs and release notes
User guide: reading DModXPS and T²PS maps, what a share above the limit
means with correlated pixels, why predictions are not stored and how to keep
one (low-level writers); API reference; CHANGELOG (behaviour change: the
viewer and `model.predict` no longer store predictions or register scans).

## Out of scope

- Per-object summaries (needs segmentation; separate plan).
- Weighted DModXPS (decided against; a later option if needed).
- Prediction uncertainty (standard errors, intervals).
