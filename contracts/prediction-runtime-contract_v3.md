# Prediction runtime contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.5; the file format `enginir-1.5` is the
  only format engiNIR reads and writes (`project-schema-contract_v3.md`)
- Scope: prediction execution, progress reporting, memory behavior, artifact
  persistence, and interactive map handoff

## Goal

Prediction on large HSI datasets must behave like a scalable scientific
service, not like a one-shot array transform hidden behind a worker thread.

This contract defines:

- how prediction is executed in chunks
- how progress is reported to the viewer
- which arrays may be held in memory at once
- what a prediction returns, and how the optional low-level prediction
  artifacts are written and reopened
- what the interactive prediction map must be able to inspect

## Relationship to other contracts

This contract complements:

- `dataset-calibration-contract_v3.md`
  - dataset-local reference correction ownership
  - input-contract compatibility
  - prediction dataset reuse and override rules
- `sample-set-model-run-contract_v3.md`
  - model input-contract and preprocessing semantics
- `project-schema-contract_v3.md`
  - persisted prediction-run layout and disk-backed artifact rules
- `viewer-workspace-ui-contract_v3.md`
  - visible progress, action availability, and map interaction behavior

When this contract conflicts with older one-shot prediction notes, this
contract is authoritative for runtime execution.

## Runtime pipeline

Prediction runtime is split into explicit phases:

1. resolve model and prediction dataset request
2. inspect dataset metadata and dimensions
3. resolve or create the scan's reference correction (in memory; not registered in the
   project, §Result handling)
4. validate input-contract compatibility
5. validate preprocessing replay and predictor shape
6. allocate the result maps (in memory, `float32`, one value per pixel)
7. load one bounded data chunk
8. transform chunk into the required input contract
9. apply saved preprocessing to the chunk
10. run model inference for the chunk, and compute the applicability
    diagnostics (DModXPS, T²PS) from the same preprocessed chunk
11. write the chunk's values into the result maps; accumulate the summary
12. update progress
13. finalize the result (metadata, limits, summary) and write the log entry
14. render or refresh the maps from the result

The service may combine adjacent phases internally, but progress/status
reporting must preserve these semantic boundaries.

## Preflight contract

Prediction preflight is a required service boundary, not only a UI convenience.

Required behavior:

- run the same compatibility checks before worker submission and inside the
  worker before chunk processing
- normalize enum-like strings before comparison, including signal units,
  reference correction modes, band policies, and output dtypes
- reject unsupported `output_dtype` values before artifact initialization
- reject missing stateful preprocessing calibration state before artifact
  initialization
- reject incompatible in-memory predictors before inference
- treat wavelength values as import/load metadata, not as a repeated
  prediction-time compatibility gate
- report failures as user-visible readiness errors and log them before raising

Forbidden behavior:

- discovering preflight-detectable incompatibilities only after expensive I/O or
  after partial artifact writes
- allowing the UI preflight and service preflight to implement diverging
  compatibility rules
- failing prediction solely because dataset or saved-artifact wavelength values
  differ when band count and source/reference correction/model contracts match

## Predictor contract

The runtime must never bypass the fitted preprocessing recipe.

Allowed predictor forms:

- a fitted `Pipeline`-like object that exposes `named_steps` and contains the
  preprocessing/model steps required for prediction
- an approved stateless adapter that is explicitly documented to replay the
  saved preprocessing recipe before estimator inference

Required behavior:

- an in-memory predictor registered for a model must be checked at runtime
  before inference
- a bare estimator must be rejected unless wrapped in an approved recipe-aware
  adapter
- saved-artifact prediction must apply the input contract and saved
  preprocessing recipe exactly once before estimator inference
- the executable model is read only from the ModelRun's
  `artifact_refs/prediction_model`; runtime must not discover models by
  scanning an extra root `/models` group
- saved-artifact prediction for `MSC` or `EMSC` must reconstruct those fitted
  transformers from `prediction_model/preprocessing_state`

Forbidden behavior:

- calling `predict()` on a bare estimator with input-contract-ready spectra when
  the model has a non-empty preprocessing recipe
- silently accepting an object because it has a `predict()` method

## Input contract at runtime

v3 prediction uses pragmatic input-contract compatibility.

Required behavior:

- prediction compatibility is based on source format, source modality, band
  count, reference correction capability, preprocessing recipe replayability, predictor
  availability, and output dtype
- wavelength arrays remain useful metadata for display, diagnostics, and rough
  import/load sanity checks
- prediction runtime must not repeatedly compare full wavelength axes once the
  dataset has been accepted into the project, apart from the wavelength warning (G18): preflight compares the scan's wavelength axis
with the model's (the input contract's) once; when both are real axes (not
empty and not the band-index fallback 0, 1, …, n − 1 that loaders use for
files without wavelengths) and differ by more than
`max(wavelength_tolerance_nm, 1e-6)` nm at any band, prediction logs a
warning with the largest difference and records it in the result
(`provenance["wavelength_warning"]`; the viewer shows it in the Predict
summary). It never refuses or changes the prediction
- `output_signal_unit`, `required_signal_unit`, and related enum strings must be
  compared after canonical normalization
- if flat spectra are supplied without an explicit `spatial_shape`, the alignment of
  the dark/white references must infer shape from `DatasetRef.cube_shape` when that
  metadata is available

v3 dtype rule:

- `output_dtype` is fixed to `float32` in v3
- the field records the supported runtime dtype and is not a free backend
  selection knob
- any value other than canonical `float32` must be rejected during preflight

v3 band-mapping rule:

- no interpolation, regridding, or automatic band remapping is supported
- `exact` means the band count and band order are accepted as already
  aligned by the import/modeling workflow
- the scan's band count is compared with the input contract (the recipe's
  input); the saved model's coefficients and projection data are compared
  with the recipe's output band count (fewer bands with `sg_edges = drop` or
  a band selection), any restored MSC/EMSC state with the band count after
  the Savitzky-Golay step (the references see every band before the
  selection), and preflight refuses a mismatch
- a future contract version must introduce a separate executable remapping
  contract before `band_mapping_policy` can imply interpolation

Forbidden behavior:

- treating differently cased but semantically identical signal-unit strings as
  incompatible
- labeling a strict exact-axis check as interpolation support
- rejecting otherwise compatible prediction runs only because wavelength
  metadata differs numerically

## Chunking contract

Required behavior:

- prediction must support chunked processing for image-like datasets
- a chunk is a bounded set of frames or flat spectra rows
- chunk size must be configurable by service defaults and overridable by tests
- the default image chunk size is 512 frames; callers and tests may override it
- the prediction service must keep a chunkable source handle and the active
  result maps (or, for low-level persistence, the artifact writer) open
  across the chunk loop rather than reopening either file
  for every chunk
- processing order must preserve the persisted spatial output order
- chunked output must be numerically equivalent to one-shot output for
  stateless preprocessing recipes within normal floating-point tolerance

Allowed behavior:

- small datasets may be processed in one chunk
- non-image spectral matrices may be chunked by row
- point-spectrum formats without an image cube (CSV, SPC, SPA, JCAMP) may be
  read completely and then processed in bounded chunks
- chunks may be read in any order (e.g. a reversed EIFF frame selection is
  read in file order) as long as each is written to its own output position

Forbidden behavior:

- requiring a full `float32` copy of the prediction cube before prediction can
  begin
- requiring simultaneous in-memory copies of raw input, corrected signal,
  preprocessed signal, full `y_pred`, and full `spatial_map`
- tiling full-sensor dark/white references across all frames when a chunk-local
  broadcast or slice is sufficient
- silently falling back to unchunked execution for large image cubes

## Memory contract

Prediction memory is budgeted by phase.

Required behavior:

- raw input chunks should remain in the source dtype until the transformation
  boundary
- `float32` is the working dtype for corrected/preprocessed/predicted numeric
  arrays unless a future contract explicitly requires otherwise
- at steady state, prediction should hold only:
  - one raw chunk
  - one corrected/preprocessed working chunk, or an explicitly reused buffer
  - one chunk of model outputs
  - small model/reference correction/preprocessing metadata
- the result maps are 2-D `float32` arrays held in memory (one value per
  pixel and layer: prediction, DModXPS, T²PS; each frames × rows, the size
  of one band image of the scan)
- the applicability diagnostics are computed from the preprocessed chunk in
  row blocks of at most 4096 spectra, in `float64` (scaled spectra,
  residuals and scores of one block); no further chunk-sized copy is held
- prediction maps persisted by the low-level writers (§Result handling) are
  disk-backed; metadata-first project open must not hydrate them

Recommended behavior:

- reuse working buffers across chunks when possible
- prepare reusable dark/white denominators once per prediction job
- expose chunk-size decisions in debug logs
- keep a small preview/downsample cache separate from full-resolution artifact
  storage
- run explicit cleanup after large jobs so worker/viewer caches do not retain
  raw chunks or generated arrays longer than needed

Forbidden behavior:

- storing duplicate full-size prediction arrays in project state when one
  disk-backed artifact can satisfy both map rendering and export
- merging large prediction artifacts into long-lived in-memory state unless the
  user explicitly asks to inspect or export them

## Preprocessing contract

Chunked prediction is valid only when preprocessing can be replayed without
leaking prediction data into fitted calibration state.

Required behavior:

- stateless per-spectrum preprocessing may be applied independently per chunk
- stateful preprocessing requires persisted calibration-side transformer state
- if the required calibration-side state is missing, prediction must fail before
  processing chunks
- failures must be user-visible and logged before raising

Forbidden behavior:

- fitting `MSC`, `EMSC`, or future stateful transforms on prediction chunks as a
  substitute for missing calibration-side state
- producing partial prediction artifacts after a preflight-detectable
  preprocessing incompatibility

## Reference correction and clipping contract

Signal conversion must be reproducible and diagnosable.

Required behavior:

- `dark_white_reference` reflectance conversion must use the clip policy
  declared by the reference correction contract
- v3 reflectance clipping is `[1e-6, 1.0]` unless a future reference correction field
  explicitly overrides it
- the runtime must count low-side and high-side clipped values per chunk and in
  aggregate
- non-zero clipping counts must be logged and surfaced in prediction run
  metadata or user-visible run summary
- a registered scan reuses its stored reference correction read-only; a scan that is
  not registered gets an in-memory reference correction that is never written
  (§Result handling); content deduplication applies to reference corrections written
  by low-level persistence

Forbidden behavior:

- silently clipping over-bright or below-dark pixels without telemetry
- writing duplicate dataset-local `dark_white_reference` reference corrections (low-level
  persistence) every time the user chooses identical scalar levels or
  identical reference arrays

## Progress event contract

Prediction must emit structured progress events.

Minimum event fields:

- `phase`
- `message`
- `completed`
- `total`
- `unit`
- `indeterminate`

Allowed `phase` values:

- `preflight`
- `loading`
- `calibrating` (applying the reference correction; shown as "Reference
  correction")
- `preprocessing`
- `predicting`
- `writing`
- `finalizing`
- `rendering`

Required behavior:

- progress events must be safe to emit from the worker thread and consume on the
  UI thread
- the first event must be emitted before expensive I/O starts
- long chunked phases must emit at least one event per chunk
- the final event must identify the completed prediction (and, for
  low-level persistence, the saved prediction run)
- errors must preserve the last phase in user-visible status

Allowed behavior:

- unknown totals may use `indeterminate=true`
- percentage display may be omitted until `total` is known

Forbidden behavior:

- a prediction job that only reports `Running prediction...` for all phases
- requiring users to inspect logs to know whether the job is loading,
  predicting, writing, or rendering

## Applicability diagnostics

Every PLS and PCR model stores projection data (`prediction_model`:
`x_mean`, `x_std`, `rotations`, `loadings`, `score_variances`, `s0`,
`n_train`, `n_components`; `sample-set-model-run-contract_v3.md` §7.5), so
every predicted pixel also
gets, from its preprocessed spectrum x (K bands after preprocessing, A
components, N kept calibration pixels, i.e. after DModX exclusion):

- the diagnostics always use the saved `prediction_model` and the saved or
  restored recipe steps; a model fitted in the current session is saved by
  the fit, and the viewer predicts with the saved model
  (`prefer_saved_model=True`) so that it gets the maps; an explicit
  in-memory predictor alone (low-level API) gives no diagnostics
- `x_mean` is the effective center in the model's input space (for PCR the
  PCA mean is folded in: `x_mean_` + `pca_.mean_` · `x_std_`); for PCR
  `rotations` = `loadings` = the PCA components

- scaled spectrum z = (x − `x_mean`) / `x_std`, scores t = z · `rotations`
- **T²PS** = Σₐ tₐ² / `score_variances`ₐ
- **DModXPS** = √(Σₖ eₖ² / (K − A)) / `s0`, e = z − t · `loadings`ᵀ, without the
  factor v of the calibration DModX (as SIMCA's DModXPS) and without
  modeling-power weighting
- limits from the calibration data at 0.99: T²Crit = A(N − 1)/(N − A) ·
  F(0.99; A, N − A) and DCrit (`sample-set-model-run-contract_v3.md` §7.2),
  stored with the model as `t2_crit_099`, `dcrit_099`
- summary over the scan: pixels above each limit (count, share), and the
  mean prediction of the pixels within both limits and of the others
  (single-output models; the diagnostics are per pixel, not per output)
- pixels with a non-finite statistic are left out of the counts, shares and
  means; shares use the finite pixels as denominator
- `t2_crit_099` is absent when N − A < 1, and `s0`/`dcrit_099` when
  K − A < 1; a missing limit means no pixel is counted above it.
  `score_variances` may be NaN (a component without variance), and then
  T²PS is NaN; with `s0` below `1e-10` DModXPS is not computed (NaN)
- the arithmetic is `float64` per row block; results are `float32`, and
  tests compare with an independent `float64` computation at rtol `1e-5`

Nothing of the diagnostics depends on the whole
image, so they are computed in the single chunked pass.

## Prediction intervals

A prediction has an interval ŷ ± q · s · √(1 + h) that says how precise it
is: the leverage formula of ASTM E1655 and ISO 12099. Only this formula is
used (no jack-knife, bootstrap or conformal intervals; SIMCA's YPredPS
confidence intervals are jack-knifed, a different number). Pixels and ROI
means have different errors, so there are two intervals with the same
formula and different inputs. Intervals exist for regression runs of PLS
and PCR with projection data and a single target; other models have none.
They are computed on demand and not stored: the pixel interval at a chosen
`level` (`interval_map(level)`, default 0.95), the mean interval of
prediction ROIs at 0.95. A `level` outside (0, 1) raises.

Inputs, all stored with the model (an input that is missing or not
finite counts as missing, and a run without one has no interval of that
kind, and says so):

- *A* `n_components`, T²Crit `t2_crit_099` and DCrit `dcrit_099` from
  `prediction_model` (with *N* `n_train`, the kept calibration pixels, behind
  the score variances)
- *R* the metric `n_rois`, the calibration ROIs
- s for pixels: the metric `pixel_rmsee`, √(RSS / (N − A − 1)) over the
  fitted pixels; s for means: the metric `rmsee`, √(RSS / (R − A − 1)) over
  the ROI means of the fitted values (`sample-set-model-run-contract_v3.md`
  §5, §7.5). RMSEE (per ROI and per pixel) is a fit error, as in ASTM E1655; the coverage metrics
  (§5 of that contract) say how well it holds on held-out data. A run whose
  ROI means are not all fitted (R′ < R, e.g. every pixel of a ROI excluded
  by DModX) can have a NaN `rmsee` and then has no mean interval
- `roi_scores` (§7.5 of that contract) for the mean interval
- q the two-sided Student t quantile, t(1 − (1 − level)/2; R − A − 1). For
  both intervals q uses R − A − 1, also when s = `pixel_rmsee` (computed
  with N − A − 1): the ROIs are the independent samples, pixels of one ROI
  are not, and the smaller degrees of freedom widen the pixel interval
  (the pixel leverage counts the ROIs too).
  With R − A − 1 < 1 there is no interval

**Pixel interval** (the Prediction interval map): an interval for the target value of a sample measured as that single pixel (calibration pixels carry
their ROI's target value). h = 1/R + T²PS / (R − 1), from the pixel's
T²PS (Applicability diagnostics). With the score variances over the N kept
pixels (N − 1) and scores centered and orthogonal on them, T²PS / (N − 1) is
tᵀ(TᵀT)⁻¹t, so the standards' leverage would be 1/N + T²PS / (N − 1). The
ROIs replace the pixels as the independent samples (decided 2026-11-10): the
target values, and with them the uncertainty of the model's
coefficients, come from R ROIs, as the degrees of freedom of q already
assume. With N pixels the leverage would vanish (N = 396 000 gives at most
5·10⁻⁵ on a real model) and every pixel would get the same interval. This
departs from the letter of ASTM E1655, which assumes independent samples.
The half-width
q · s · √(1 + h) is NaN when the prediction or T²PS is not finite, when
T²PS > T²Crit, or when DModXPS > DCrit (a missing limit is not checked; a
NaN DModXPS with DCrit present is not checked either, as in Applicability
Diagnostics). A score variance below `1e-10` is stored as NaN and makes
every T²PS NaN, so such a run has no pixel interval. The leverage formula
assumes the spectrum resembles the
calibration spectra; beyond the limits a number would mislead. Reported with
the map: the median half-width over the pixels that have one and the share
of finite predictions without one ("outside the model").

**Mean interval** (a prediction ROI): with the calibration ROI scores
(`roi_scores`, the rows with a fitted pixel, R′ of them), their mean m and
covariance C (R′ − 1), and t̄ the mean score over the ROI's kept pixels
(the same pixels as its reported mean). The mean interval treats the model
as if it had been fitted on the R′ calibration ROI means (the ROIs are the
independent samples); this is a working assumption, not a property of the
pixel fit.

- d² = (t̄ − m)ᵀ C⁻¹ (t̄ − m), C⁻¹ the pseudo-inverse with a warning when C
  is singular (rank below A, e.g. collinear ROI scores)
- h̄ = 1/R′ + d² / (R′ − 1); half-width q · s · √(1 + h̄)
- NaN without a kept pixel, with R′ − A < 1, or when d² is above the
  ROI-level limit A(R′ − 1)/(R′ − A) · F(0.99; A, R′ − A) (outside the
  model at the ROI level). DModX is judged per pixel (the exclusion and the
  shares), not for the mean
- the mean interval follows the pixel exclusion, like the mean: the
  pixels' scores (n × A) are kept in memory per ROI, so a recalculated
  exclusion needs no new prediction

For a linear model on per-pixel preprocessing, the ROI mean of the pixel
predictions equals the prediction of the mean preprocessed spectrum, whose
scores are t̄; `roi_scores` are the same kind of mean for the calibration ROIs.
The arithmetic is `float64`; half-widths are returned as `float32` for maps
and `float` for ROIs, and tests compare with an independent dense
computation (the hat matrix of the centered calibration scores: for pixels its
diagonal term scaled by (N − 1)/(R − 1)).

## Prediction ROIs

ROIs whose `job` is `prediction` (`roi-target-contract_v3.md`) mark the
samples to predict (tablets on a tray). A prediction of them reads only the
project file: each ROI's stored pixels are converted with the ROI's own
reference correction and stored references by `roi_to_signal_domain` (as model
fitting does; never another dataset's references), checked and preprocessed
by the same preflight and recipe steps as a scan, predicted, and given
DModXPS and T²PS per pixel. Several models may predict the same ROIs; the
results are per model (v3 models have one output).

Per ROI, compatibility is checked as for a scan: an ROI whose reference correction
cannot produce the model's signal unit, or whose band count differs, is
refused with its reason and the other ROIs are predicted; a wavelength axis
that differs from the model's warns once per dataset (G18).

**Map and ROIs give one result.** A map prediction of a dataset (the
scan, with the dataset's active reference correction) is for display; the per-ROI
results shown with it are always computed from the ROIs' stored pixels as
above, so an ROI has the same numbers whichever way it was predicted.

Per ROI and model, over its pixels:

- a pixel is *valid* when its prediction is finite; non-finite pixels are
  neither kept nor counted as excluded
- with the exclusion below, a valid pixel is *excluded* when its DModXPS is
  strictly above the limit; a pixel without a DModXPS (NaN) is kept
- reported: the number of valid pixels and of excluded pixels; over the kept
  pixels the mean (the reported value, as model fitting aggregates ROI
  predictions by the mean), median, standard deviation (n − 1; NaN for one
  pixel), and 5 % and 95 % percentiles (linear interpolation, the
  `numpy.percentile` default), and the mean DModXPS and T²PS; over the valid
  pixels, the share above DCrit and above T²Crit at 0.99 (the exclusion does
  not change these shares). An ROI with no kept pixel has no value
- if the ROI has a value for the model's target: the residual (mean −
  value). Per model, over the predicted ROIs (the chosen dataset or all)
  that have a value and a kept pixel: RMSEP (per ROI) = √(mean residual²) and the bias
  (mean residual), with their number *n*

**Pixel exclusion by DModX** (optional): pixels whose DModXPS is strictly
above the limit at a chosen level are excluded. The limit is
`dmodx_limit(n_train, K, A, level, centred=True)` with the run's stored
`n_train`, *K* the band count on the model's axis (after a band selection)
and *A* its components; at 0.99 it is the stored DCrit. DModXPS is
unweighted and compared with this unweighted DCrit, as in Applicability
Diagnostics. The default comes from the run's fit specification: on at
`exclusion/level` when `exclusion/method = dmodx`, off otherwise; its
`formula`, `rounds` and `max_fraction` do not carry over (no cap per ROI).
Changing the exclusion recalculates from the predicted pixels without
predicting again.

For comparison, the 10th, 50th and 90th percentiles of the calibration ROIs'
stored shares of pixels above DCrit (`roi_share_above_dcrit`, §7.5 of the
model-run contract) are kept with the result, at level 0.99 only (the
calibration share uses calibration DModX and the drawn pixels); a prediction ROI
whose share is above the 90th percentile is marked as unusual. Unavailable
for runs without it.

Per ROI the half-width of the mean interval at 0.95 (Prediction intervals)
is part of the per-ROI result in the library (NaN where there is none); the
viewer does not show it.

A prediction of ROIs writes one log entry: the project file, the models
(id and name), the ROI ids, the exclusion and its level, and the summary
(*n*, RMSEP (per ROI) and bias where available). The per-ROI results are in memory
like every prediction and are not stored (Result handling).

## Result handling

Predictions are applications of a model to new scans, not calibration
content: by default nothing of a prediction is written to the project file.

Required behavior:

- `model.predict(...)` and the viewer return an in-memory result
  (`Prediction`, `enginir._workflow`): the prediction map and vector, the DModXPS and
  T²PS maps (`dmodx_map`, `t2_map`), the limits and summary
  (`diagnostics`), and the provenance (model id and name, dataset,
  reference correction settings, clipping counts, time)
- a prediction does not register the predicted scan or its reference correction. In
  the viewer the user registers a scan to predict beforehand, as a project
  dataset with the Add dataset dialog (`viewer-workspace-ui-contract_v3.md`
  §3); that is an explicit user action, not part of the prediction
- every prediction writes one log entry with its provenance and summary:
  model id and name, dataset path(s) (for ENVI the data and header
  files), size and modification time of each, time, the summary values; a
  SHA-256 of the dataset is added only on request (`checksum=True`),
  because it reads the whole file once more. The same record is on the
  result (the library logs nothing until `enable_logging()`)
- results are exported through the plot panels (PNG, SVG, CSV)
- persisting a prediction is an explicit low-level operation
  (`enginir.io.project_store` writers, below); neither the facade nor the
  viewer calls it

Low-level persistence (`/predictions/{prediction_run_id}/`), when a script
asks for it, keeps the following rules.

Required behavior:

- the caller writes the scan's `DatasetRef` and its `CalibrationSpec` first
  (dependency order; `project-schema-contract_v3.md` referential
  integrity), e.g. with `Project.add_dataset`, then calls
  `predict_with_model_run(..., persist_artifacts=True)`; a run whose
  dataset or reference correction is missing makes the project fail to open
  (relationship validation)

- stage the run under `/staging/{prediction_run_id}/`
  (`project-schema-contract_v3.md` §7): initialize its artifact datasets
  before chunk writes, and write chunks directly into them
- store enough artifact metadata to interpret shape, dtype, target, and spatial
  orientation without loading the arrays
- for multi-output predictions, store output-axis metadata sufficient to select
  one target/output without guessing
- finalize metadata only after all required artifact writes succeed, then
  move the group to `/predictions/{prediction_run_id}/`
- failed prediction runs must not masquerade as completed predictions

Multi-output provenance in v3:

- service-fitted v3 models always produce single-output predictions,
  because one model models one selected target through its calibration set
- the output-axis metadata and explicit output-selection rules keep
  prediction artifacts with more than one output interpretable (e.g. from
  in-memory multi-target predictors); they are not a v3 multi-target modeling
  feature

Recommended artifact metadata:

- `spatial_shape`
- `prediction_shape`
- `output_dtype`
- `target_id`
- `target_name`
- `target_unit`
- `output_axis`
- `output_index`
- `output_name`
- `output_names`
- `source_path`
- `chunk_axis`
- `chunk_size`
- `n_chunks`
- `completed_chunks`
- `clip_low_count`
- `clip_high_count`
- `created_at`

Allowed artifact datasets:

- `/predictions/{prediction_run_id}/artifact_refs/y_pred`
- derived preview datasets such as `preview_map` when full maps are too large
  for immediate UI rendering

`y_pred` is the prediction array. For image datasets, `spatial_map_shape`
defines its spatial reshape; no second full `spatial_map` copy is stored.

Forbidden behavior:

- requiring complete prediction output to return to the UI thread before it can
  be saved

## Staged writes

The staged artifact write path follows `project-schema-contract_v3.md` §7:

- the run is written under `/staging/{prediction_run_id}/` and moved to
  `/predictions/{prediction_run_id}/` only after all required artifact
  writes and metadata succeeded; a run in `/predictions` is complete by
  construction, so there is no status attribute
- a failure or cancellation between initialization and finalization deletes
  the staging group
- a staging group left by a crash never stops the project from opening; the
  reader logs one warning per stale group, and the next staged write or
  `discard_staged_predictions(path)` deletes it

Forbidden behavior:

- persisting a failed or canceled run in `/predictions`
- reading anything from `/staging/` outside the staging writers

## Interactive map handoff

Prediction map visualization is driven by the in-memory result of the
session (a preview subsampled with the same step rule as before for large
maps).

Required behavior:

- full-resolution map orientation must match the main spatial image
- map axes are `Frame` and `Row`
- map rendering must include a color scale/legend
- multi-output predictions must expose an explicit output/target selector before
  display
- a layer selector shows the prediction, DModXPS or T²PS; on the diagnostics
  layers the pixels above the 0.99 limit are marked and their share is shown
- with the interval inputs, a Prediction interval layer shows the pixel
  half-width at 0.95 (Prediction intervals); pixels without an interval are
  marked as outside the model, and the median half-width and the share
  without an interval are shown
- clicking the map must expose at least frame, row, and predicted value
- map interaction must not require loading unrelated model diagnostics or ROI
  spectra
- unchanged map payloads must be no-op redraws

Recommended behavior:

- use the same interactive image technology as the main spatial canvas where
  practical
- support pan/zoom, value probe, crosshair, and linked navigation to the
  workspace image
- use a downsampled preview for very large maps, then load full-resolution
  chunks on demand

Forbidden behavior:

- rendering the prediction map transposed relative to the main spatial image
- silently choosing output or target `0` when a prediction artifact has multiple
  outputs
- rendering a numeric prediction map without a colorbar or equivalent legend

## Acceptance tests

Runtime tests must cover:

- chunked prediction equals one-shot prediction on representative stateless
  recipes
- in-memory bare estimators are rejected when a recipe-aware predictor is
  required
- unsupported `output_dtype` values are rejected during preflight
- `MSC`, `EMSC`, and future stateful recipes without persisted calibration state
  fail during preflight before chunk processing
- `band_mapping_policy` values do not imply interpolation in v3 and wavelength
  metadata differences are not prediction-time hard failures
- reflectance clipping logs low/high clipped counts and reports them in the
  result (and persists them with low-level persistence)
- multi-output predictions require explicit output selection and never silently
  display only output `0`
- a registered scan reuses its stored active reference correction read-only; a new
  scan gets an in-memory reference correction
- signal-unit comparisons are case-normalized
- the alignment of the dark/white references can infer spatial shape from
  `DatasetRef.cube_shape` for flattened inputs
- saved-artifact prediction is not blocked by model artifact wavelength metadata
  when band count/source/reference correction/recipe checks pass
- progress events include preflight, loading, prediction, writing, and final
  completion phases
- large prediction artifacts are not hydrated during metadata-first project
  open
- a prediction by the facade or the viewer leaves the project file without new
  datasets, reference corrections or prediction runs, and writes one log entry
- DModXPS and T²PS equal their formulas; chunked equals unchunked; an odd
  pixel exceeds DCrit
- a model fitted in the same viewer session gets the maps
- prediction finalization leaves no completed `PredictionRun` after a chunk
  write failure
- a write interrupted between staging and finalizing leaves a project that
  opens, with one warning and no run in `/predictions`; the next staged
  write or `discard_staged_predictions` removes the stale group
- a canceled staged prediction run is discarded and leaves no staging or
  prediction group behind
- full-sensor dark/white references are applied chunk-locally without tiling
  across the full prediction cube
- interactive map payload reports row/frame orientation and value lookup
- repeated map refreshes do not redraw unchanged payloads
