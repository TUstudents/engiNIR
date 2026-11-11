# Plan: terminology standard

Status: decisions settled with the user (2026-11-11), one question each;
`contracts/terminology_v3.md` written, reviewed independently, and every
review finding decided by the user (R1–R16 below). Audience: the
process analytical technology (PAT) expert using chemometric modeling on
spectra (NIR, Raman, IR, UV-Vis), with hyperspectral imaging (HSI) as
engiNIR's niche.

## Decisions

| # | Concept | Term | Notes |
|---|---|---|---|
| 1 | Converting raw counts with dark/white references or detector levels | **Reference correction** | Replaces "Calibration" in GUI and docs; "calibration" then means only the model (ASTM E1655). API `CalibrationSpec` stays internal, mapped once in the docs |
| 2 | ROI role | **Calibration** / **Prediction** | Column "Set"; predicted ROIs with target values give the RMSEP |
| 2b | Stored ROI role value | **`calibration`** | A stored `training` is read as `calibration` (user, rule C) |
| 3 | Quantity modelled | **Target** | Kept |
| 4 | One imported measurement | **Dataset** | As the API; "image" only where the spatial cube is meant |
| 5 | Spatial axes of a push-broom cube | **Frame**, **Row** | ENVI's "samples" never appears |
| 6 | One spectrum | **Pixel** (imaging), **spectrum** (general) | "Observation" only as SIMCA's synonym in the glossary |
| 7 | Sample | **Physical sample only** | The label grouping ROIs of one object; never a synonym for ROI, pixel or spectrum |
| 8 | Marked area | **ROI** | Upper case, plural ROIs |
| 9 | Raw detector signal | **Raw counts (DN)** | Both words, always |
| 10 | Absorbance from reflectance | **Absorbance** | Kept; the docs say log10(1/R) |
| 11 | SNV, MSC, EMSC, Detrend | **Scatter correction** | Replaces "Normalisation" |
| 12 | Savitzky-Golay smoothing and derivatives | **Smoothing and derivatives** | Replaces "Filtering" |
| 13 | A model's chain of preprocessing steps | **Preprocessing recipe** | API `PreprocessingRecipe` matches |
| 14 | Model size | **LV** (PLS), **PC** (PCR); *components* in general | "PLS 7 LV", "PCR 5 PC" |
| 15 | Distance to the model | **DModX**, **DCrit** | Short meaning "distance to the model" in tooltips and first mentions; Q and F residuals in the glossary as related, not identical |
| 16 | Distance within the model | **Hotelling's T²**, short **T²**, **T²Crit** | |
| 17 | ± q·s·√(1+h) per pixel | **Prediction interval** | Layer "Prediction interval (±, 95 %)" |
| 17b | API | **`interval_map(level)`** | Breaking: `uncertainty_map` removed |
| 18 | Spatial image of predictions | **Prediction map** | Kept (GUI, API `.map`) |
| 19 | Spelling | **US English** | color, normalization, analyze |
| 20 | Measured values of a target | **Target value** | Replaces "values", "reference" in GUI text |
| 21 | Model run, sample set | **Model**, **Calibration set** | API `ModelRun`, `SampleSet` stay internal |
| 22 | Signal domain | **Input contract** | Shown under its model, not as its own browser node |
| 23 | Capitalization | **Title case for menus and buttons**, sentence case for titles and text | Abbreviations keep their case |
| 24 | Numbers with units | **SI style** | "95 %", "1450 nm", thin space as thousands separator, decimal point |
| 25 | Enforcement | **Contract + glossary + test** | `contracts/terminology_v3.md`, docs Glossary with `:term:` links, a test scanning user-visible viewer strings for banned words |
| 26 | Release of the breaking changes | **1.5.0** | User's choice over 2.0.0 |
| A | One spectral channel | **Band** | Its value in nm is the wavelength; "variable" only in the glossary |
| B | Reference-corrected signal | **Reflectance**, 0 to 1 | |
| — | "training" | **Not used at all** | "Calibration range", "calibration ROIs" |
| — | Measured axis of plots | **Target value** | "Reference value" only in the glossary |
| C | 1.4 files with `use = training` | **Read as `calibration`** | No error, no migration needed |
| D | Scope | **Everything a person reads** | Viewer, docs, contracts, logs and errors, export labels; API identifiers and stored keys keep their names |
| E | Capitalization in the test | **Tested** | With an allow-list for abbreviations |

## Review decisions (2026-11-11)

| # | Finding | Decision |
|---|---|---|
| R1 | Storage of the calibration role | **Write `use = calibration` explicitly**; a missing `use` and `training` read as calibration and count as the same content for the identity rule; new projects no longer open in 1.4 |
| R2 | "observation" in statistics | **"Fitted on: pixels / ROI means"**; "observation" glossary only |
| R3 | "reference" | Dark/white references, reference correction and reference method **stay**; only "reference value" banned |
| R4 | "data file" | **"File" for the file on disk**; "data file" banned as a phrase |
| R5 | Contract file names | **Kept** as identifiers |
| R6 | Test scope | **Viewer tested, library reviewed** |
| R7 | "row" for tables | **Allowed by context** |
| R8 | "prediction" role vs result | **Acceptable**; the role always appears as "prediction ROI" |
| R9 | Silent additions kept | "data set" ban, apparent absorbance, plain CSV numbers, "calibration" checked in review, class labels in Target, DModXPS and first-mention rules, prediction/chemical image synonyms, standards list in the Goal |
| R10 | Silent additions removed | "scan" ban, ROI spelled out per page, "recipe"/"filter" bans, sentence-case column headers, "new terms first", GUM note, Breeze/MIA synonyms, TargetAssignment/TrainingROI internal names, the transition clause |
| R11 | Headings | **Sentence case everywhere**, contracts and docs too |
| R12 | Metric names | **Always qualified**: "RMSECV (per ROI)", "RMSECV (per pixel)" |
| R13 | "outlier" | **Not used**: "above DCrit", "above T²Crit", "outside the model" |
| R14 | Excluded vs removed | **"Excluded" for both**, with context |
| R15 | Cross-validation unit | **Fold group** |
| R16 | Calibration activity and range | **Defined as proposed**; GUI verbs stay "Fit Model...", "Fit Again..." |

Fixed directly (factual errors found by the review): the input contract's
fields (no band axis), ENVI's radiometric calibration (a different step),
the SIMCA workset and test set mapping, reflectance "nominally" 0 to 1,
"normalization" kept for unit norm and normalized DModX, missing bans
("variable" for a band, "reference value"), the abbreviation allow-list,
"API content" → "active ingredient content", the unit example.

## Deferred to 2.0

**Rename the stored "calibration" names** (user, 2026-11-11): in the project
file, "calibration" means the reference correction, which clashes with model
calibration. A huge refactor, so it goes on the list for 2.0. Measured on
2026-11-11:

- file: `datasets/{id}/calibrations/{id}/`, `@active_calibration_id`, the
  ROI's `@calibration_id` and `calibration_refs/`, the input contract's
  `@required_calibration_mode` and `@calibrated_input_allowed`, the
  prediction run's `@calibration_id`, the mode value `precalibrated_import`
- code: about 1 000 occurrences in 29 source files (`CalibrationSpec` 70,
  `calibration_id` 110), about 700 in tests; `calibration_scores` and
  similar already mean model calibration and stay

Open, to decide then: rename the file keys only or the API too; the new name
(`reference_correction`, `correction`, ...); whether related names follow
(`calibration_refs`, the input contract attributes, `precalibrated_import`,
`raw_dn`); and how v3 files are handled (schema v4 with a migration tool, a
compatibility reader, or refusal).

## Next steps

1. Write `contracts/terminology_v3.md` from the table: definition, source
   standard or tool, synonyms in SIMCA / The Unscrambler / PLS_Toolbox /
   ENVI, banned words, and the API names that stay internal.
2. Independent review against the standards and SIMCA's vocabulary.
3. Glossary page and the enforcing test.
4. Apply: viewer strings, docs, the two breaking changes (ROI role value,
   `interval_map`), contracts that use the old words.
5. Then the GUI polish plan (context bar, next-step hint, empty states,
   compact Controls dock) in the agreed words.
