# Terminology contract v3

## Status

- Version: `v3`
- Status: settled with the user term by term (2026-11-11,
  `plans/terminology.md`), reviewed independently; applied in engiNIR 1.5.0
- Scope: the words engiNIR shows to people: viewer text, documentation,
  contracts, log and error messages, exported plot and table labels. API
  identifiers, stored keys, CSV metadata keys and file names keep their
  names (Internal names)

## Goal

engiNIR is written for the process analytical technology (PAT) expert who
builds chemometric models on spectra (NIR, Raman, IR, UV-Vis), with
hyperspectral imaging (HSI) as its niche. That reader knows SIMCA, The
Unscrambler, PLS_Toolbox and ENVI, and the standards ASTM E1655, ASTM E2617,
ISO 12099, USP <1039> and ICH Q2(R2)/Q14. Every term engiNIR uses must mean
what that reader expects, or say clearly where it differs. This contract
fixes the terms, so that the viewer, the documentation and the API keep one
vocabulary for years.

## Relationship to other contracts

- This contract owns the words; the other contracts own the behavior. A
  contract that uses a term defined here uses it with this meaning.
- Where code, docs or another contract disagree with this contract about a
  word, this contract wins.

## General rules

1. **Language:** US English (color, normalization, analyze, center,
   behavior).
2. **Capitalization:** title case for menu items and buttons ("Export
   Plots...", "Reset View", "Fit Model..."); sentence case for window, panel
   and plot titles, labels, messages, documentation and the headings of
   contracts and docs ("Prediction per ROI", "Words not used in
   user-visible text"). Abbreviations keep their case (ROI, DModX, PLS, LV,
   PC, SNV, T²).
3. **Numbers and units:** SI style (ISO 80000): a space between number and
   unit ("95 %", "1450 nm"; no space before the degree sign, "45°"), a
   decimal point, and a thin space as thousands separator in text
   ("396 000 pixels"). Machine-readable exports (CSV values) have no
   thousands separator.
4. **One word per concept:** a term defined here is used for nothing else, and
   its concept is called by no other word in user-visible text. Two
   exceptions, read by context: "row" also names a table or CSV row, and
   "prediction" alone names a computed result (the ROI role always appears
   as "prediction ROI" or in the Set column).
5. **Internal names may differ:** API, file and code identifiers may keep
   their names (Internal names); user-visible text uses the terms below, and
   the documentation states each mapping once.

## Terms

Each term lists its definition, its use, the corresponding words in other
tools and standards (for the glossary; *related* means similar but not the
same quantity), and the words it replaces.

### Data

**Dataset**
: One imported measurement: a hyperspectral cube or a file of point
  spectra, with its own reference correction. Datasets are registered in a
  project; a dataset may be a frame range of a file.
: Use: "Dataset" ("Add Dataset..."); never "data set" in two words. "Image"
  only where the spatial cube is meant (image view, prediction map of a
  dataset). "File" only for the file on disk ("Locate File...", "the
  dataset's file was moved").
: Elsewhere: SIMCA *dataset*; The Unscrambler *data set*; PLS_Toolbox
  *DataSet object*; ENVI *image* or *file*.
: Replaces: "data file".

**Frame**, **Row**
: The two spatial axes of a push-broom cube `(frames, rows, bands)`. A frame
  is one line read by the camera as the sample moves (the scan direction);
  a row is a position along the slit. As an axis value, both are
  capitalized with their number ("Frame 1046 | Row 124").
: Elsewhere: ENVI *lines* (= frames) and *samples* (= rows). ENVI's
  *samples* is never used: it collides with Sample.

**Band**
: One spectral channel, with its wavelength (nm) when known.
: Elsewhere: SIMCA *X variable*; The Unscrambler *variable*.
: Replaces: "variable" for a band ("latent variable" stays).

**Pixel**, **spectrum**
: One spectrum: *pixel* wherever the image matters ("per pixel", "pixel
  map", "396 000 pixels"), *spectrum* for point data and in general text. A
  model is fitted on pixels or on ROI means; the Inspector says "Fitted on:
  pixels" or "Fitted on: ROI means".
: Elsewhere: SIMCA *observation*; The Unscrambler and PLS_Toolbox *sample*
  (one spectrum of the data matrix, not a physical sample).
: Replaces: "observation" in user-visible text (glossary only).

**ROI**
: Region of interest: a marked area of a dataset, upper case, plural ROIs.
  The reporting unit: target values, fold groups and the metrics per ROI
  are based on the mean of its pixels.
: Elsewhere: ENVI *ROI*.

**Sample**
: The physical sample only: the object (tablet, blend, batch portion) whose
  ROIs share a sample label and can be kept together in cross-validation.
  Never a synonym for ROI, pixel or spectrum.
: Elsewhere: in The Unscrambler and PLS_Toolbox *sample* means one spectrum
  of the data matrix (here: pixel or spectrum); the glossary warns about it.

### Signal

**Raw counts (DN)**
: The uncorrected detector values, before reference correction. Always both
  words.
: Elsewhere: ENVI *DN* (digital number); camera vendors *raw*.
: Replaces: "Raw DN", "DN" alone.

**Reference correction**
: Converting raw counts to reflectance with dark and white references or
  with detector levels ("Reference correction: dark/white references",
  "Reference correction: detector levels 128 / 3500").
: Elsewhere: HSI vendors *reflectance calibration*; *related*: ENVI *Flat
  Field Correction*, *Empirical Line* (ENVI's *Radiometric Calibration*
  converts DN to radiance, a different step).
: Replaces: "calibration" in this meaning (checked in review: text alone
  cannot tell the meanings apart).

**Reflectance**
: Reference-corrected signal R, nominally between 0 and 1 (noise can give
  values slightly outside).

**Absorbance**
: log10(1/R) computed from reflectance. The label is "Absorbance"; the
  documentation states the formula and that for diffuse reflectance it is
  the apparent (pseudo-)absorbance.

### Preprocessing

**Preprocessing recipe**
: The ordered chain of preprocessing steps of a model, stored with it and
  identified by its content ("SG 1st der. + SNV").
: Elsewhere: SIMCA *spectral filters* / *preprocessing*; The Unscrambler
  *transformations*, *pretreatments*; PLS_Toolbox *preprocessing*.

**Scatter correction**
: The group of SNV, MSC, EMSC and Detrend (Rinnan et al., 2009).
: Replaces: "Normalisation" / "Normalization" as the group name.
  "Normalization" remains for scaling to a unit norm and for the normalized
  DModX (SIMCA's term).

**Smoothing and derivatives**
: The Savitzky-Golay group: smoothing, 1st and 2nd derivative.
: Replaces: "Filtering".

### Modeling

**Target**
: The quantity a model predicts (active ingredient content, layer count;
  later class labels), with an optional unit.
: Elsewhere: ASTM E1655 *property*, *constituent*; SIMCA and The Unscrambler
  *Y variable*, *response*; PLS_Toolbox *y block*.

**Target value**
: The measured value of a target for an ROI, from the reference method.
: Elsewhere: ASTM E1655 *reference value*; SIMCA *observed* Y (in
  observed-versus-predicted plots).
: Replaces: "value", "reference" and "reference value" for this meaning in
  user-visible text (the measured axis of a plot is "Target value").
  "Reference" stays for dark/white references, reference correction and the
  reference method.

**Calibration**
: Building a model from the calibration set (fitting and cross-validating
  it), as in ASTM E1655. The ROI role, the calibration set and the
  calibration range share this one meaning. The GUI verbs stay "Fit
  Model..." and "Fit Again...": fitting is the step, calibration the
  activity.

**Calibration** and **Prediction** (ROI set)
: The role of an ROI. *Calibration* ROIs build models; *prediction* ROIs are
  predicted and never enter a model, whatever target values they have. The
  RMSEP comes from prediction ROIs that have target values. Shown in the ROI
  table's column "Set"; in text always "calibration ROI(s)" or "prediction
  ROI(s)".
: Stored as the ROI field `job` = `calibration` or `prediction`, required
  (`roi-target-contract_v3.md`).
: Elsewhere: SIMCA *prediction set* for prediction ROIs; prediction ROIs
  with target values form an external *test set* (they give the RMSEP).
: Replaces: "training" in every user-visible meaning ("training range" →
  "calibration range", "training ROIs" → "calibration ROIs", "training
  scores" → "calibration scores").

**Calibration set**
: The frozen list of calibration ROIs, with their target values and the
  preprocessing recipe, that one model was fitted on.
: Elsewhere: SIMCA *workset*; USP <1039> and The Unscrambler *calibration
  set*.
: Replaces: "sample set" in user-visible text.

**Calibration range**
: The range (minimum to maximum) of the target values in the calibration
  set. The default color scale of a prediction map; predictions outside it
  get their own colors.

**Fold group**
: The unit cross-validation holds out together: by ROI (default), by sample
  label, or by target value ("Fold groups: by sample").

**Pixel subsample**
: The pixels a fit draws from every ROI: the same number from each, chosen
  by a seed, so that every ROI weighs the same
  (`sample-set-model-run-contract_v3.md` §7.1). Without one, a fit uses
  every pixel.
: Replaces: "pixel sample" ("sample" is the physical sample only).

**Model**
: One fitted model with its id, never overwritten; its name describes it
  ("PLS 7 LV, SNV").
: Replaces: "model run" in user-visible text.

**Input contract**
: What a dataset must provide for a model to predict it: input format,
  signal unit, and the reference correction it requires (dark/white
  references or detector levels). Shown under its model. The band axis is
  not part of it: a differing wavelength axis only warns (G18).
: Replaces: "signal domain" in user-visible text.

**LV**, **PC**, **components**
: Model size: *LV* (latent variables) for PLS, *PC* (principal components)
  for PCR ("PLS 7 LV", "PCR 5 PC"); *components* as the general word (the
  Components page, number of components).
: Elsewhere: SIMCA *components*; The Unscrambler *factors*.

### Diagnostics and results

**Metrics per ROI**, **metrics per pixel**
: Every metric (RMSEE, RMSECV, RMSEP, R², Q², bias, RPD, RPIQ) names its
  level, always: "RMSECV (per ROI)", "RMSECV (per pixel)". Per ROI is the
  reported level (ROI means); per pixel is secondary.
: Internal: `rmsecv`, `pixel_rmsecv`.

**DModX**, **DCrit**
: The normalized distance of a spectrum to the model (its residual, scaled
  by the calibration residual s0) and its critical limit. *DModX* is the
  distance of a calibration spectrum (the fit and the Model workspace);
  *DModXPS* the distance of a predicted spectrum, always (the Predict
  workspace, prediction results and their exports), so the two are never
  taken for each other. Tooltips and the first mention on a page say
  "distance to the model".
: Elsewhere: SIMCA *DModX*, *DModXPS*, *DCrit* (the formulas follow SIMCA);
  *related*: PLS_Toolbox *Q residuals*, The Unscrambler *F-residuals*
  (different scaling and limits).

**Hotelling's T²**, **T²**, **T²Crit**
: The distance of a spectrum within the model plane and its critical limit.
  "Hotelling's T²" in titles and first mentions, "T²" in compact labels;
  tooltips say "distance within the model". *T²PS* is the T² of a predicted
  spectrum, always, as DModXPS.
: Elsewhere: the same name in SIMCA, The Unscrambler and PLS_Toolbox.
  *Related*: leverage (see Prediction interval).

**Above DCrit**, **above T²Crit**, **outside the model**
: A pixel or ROI whose DModX or T² exceeds the 0.99 limit is "above DCrit"
  or "above T²Crit"; above either, it is "outside the model". A limit
  exceedance describes the spectrum; it is not a judgement that the
  spectrum is faulty.
: Replaces: "outlier".

**Excluded pixels**
: Pixels left out by a DModX limit: from the fit (the fit specification's
  exclusion, stored with the model) or from the statistics of a prediction
  ROI (the Predict page's limit, session only). Said with its context:
  "excluded from the fit", "excluded from the ROI statistics".
: Replaces: "removed".

**Prediction interval**
: ŷ ± q · s · √(1 + h), the leverage formula of ASTM E1655 with the R
  calibration ROIs as the independent samples
  (`prediction-runtime-contract_v3.md`, Prediction intervals): for a pixel
  h = 1/R + T²PS/(R − 1); for the mean of a prediction ROI
  h̄ = 1/R′ + d²/(R′ − 1), with d² the Mahalanobis distance of its mean
  scores from the calibration ROIs' mean scores. The standard's leverage
  over N calibration spectra, h = 1/N + T²/(N − 1), would vanish for the
  pixels of an image. Shown as "±" with its level ("Prediction interval
  (±, 95 %)"). API `prediction.interval_map(level)`.
: Elsewhere: SIMCA *YPredPS confidence interval* (jack-knifed, a different
  number); The Unscrambler *Y-deviation* (a different estimate).
: Replaces: "uncertainty".

**Prediction map**
: The spatial image of the predicted values of a dataset ("Prediction map",
  "Predict Map...", API `prediction.map`).
: Elsewhere: NIR chemical imaging *prediction image*, *chemical image*.

## Internal names

API, file and code names that stay as they are, and the term each stands
for:

| Internal name | Term |
|---|---|
| `DatasetRef`, `add_dataset`, `dataset_id` | Dataset |
| `CalibrationSpec`, `calibration_id`, `/calibrations/`, `dataset-calibration-contract_v3.md` | Reference correction |
| `SignalDomainContract`, `signal_domain_id` | Input contract |
| `SampleSet`, `sample_set_id`, `sample-set-model-run-contract_v3.md` | Calibration set |
| `ModelRun`, `model_run_id` | Model |
| `job` (ROI field), `TrainingROI.job` | Calibration and Prediction (ROI set) |
| `PreprocessingRecipe`, `recipe_id` | Preprocessing recipe |
| `raw_dn` (signal unit) | Raw counts (DN) |
| `pixel_*` metric keys | Metrics per pixel |

The stored "calibration" names of the reference correction are to be renamed
in 2.0 (`plans/terminology.md`, Deferred to 2.0); until then they are internal
names like the others.

Renamed in 1.5.0:

- ROI `use`: calibration ROIs store `calibration` (see Calibration and
  Prediction).

Renamed in 1.5.x (format `enginir-1.5`, no reading of earlier files):

- ROI `use` → `job` (`TrainingROI.job`, `add_roi(job=...)`, `set_job`,
  `save_roi_job`).
- `Prediction.uncertainty_map(level)` → `Prediction.interval_map(level)`
  (breaking: the old name is removed).

## Words not used in user-visible text

| Not used | Use instead |
|---|---|
| data file | Dataset (the object), file (on disk) |
| data set (two words) | Dataset |
| calibration (for dark/white correction) | Reference correction |
| training (any meaning) | calibration, Calibration set |
| sample set | Calibration set |
| model run | Model |
| signal domain | Input contract |
| normalisation / normalization (as the SNV group) | Scatter correction |
| filtering (as the SG group) | Smoothing and derivatives |
| uncertainty (for the interval) | Prediction interval |
| Raw DN, DN (alone) | Raw counts (DN) |
| observation | pixel, spectrum; "Fitted on: …" |
| variable (for a band) | Band |
| reference value (for a target value) | Target value |
| outlier | above DCrit, above T²Crit, outside the model |
| removed (pixels) | excluded |
| pixel sample | pixel subsample |
| a metric without its level ("RMSECV" alone) | "RMSECV (per ROI)", "RMSECV (per pixel)" |
| British spellings (colour, normalise, analyse, centre, behaviour) | US spellings |

## Enforcement

- `docs/glossary.rst` lists every term with its definition, synonyms in other
  tools and the internal name; the user guide links terms with `:term:`.
- A test scans the viewer's user-visible strings: the string and f-string
  arguments of the Qt calls that show text (labels, buttons, menu items,
  tooltips, titles, table headers, messages). Text in backticks and
  `snake_case` or `CamelCase` tokens count as identifiers and are exempt;
  justified exceptions are listed in a reviewed allow-list. The test fails
  with the file, line and replacement for every word in "Words not used"
  and every British spelling.
- Library log and error messages follow the same terms and are checked by a
  second scan (the first argument of `log_and_raise`, the logger calls and
  `raise …Error(...)` outside the viewer; text in backticks is exempt).
  Documentation and contracts follow the same terms and are checked in
  review. Words that text alone cannot judge
  (calibration for reference correction) are checked in review.

## Acceptance tests

- The viewer's user-visible strings contain no word from "Words not used"
  (the scan above, fault-injected with one banned word).
- Menu items and buttons are in title case, titles and labels in sentence
  case (the same scan, for strings passed to buttons, actions and titles),
  with an allow-list of abbreviations (ROI, PLS, PCR, LV, PC, SNV, MSC,
  EMSC, SG, DModX, DCrit, T², RMSE…).
- Every ROI stores `job` = `calibration` or `prediction`; a missing `job`
  or any other value is refused on read.
- `Prediction.interval_map(level)` returns what `uncertainty_map` returned;
  `uncertainty_map` no longer exists.
- The glossary has an entry for every term of this contract.
