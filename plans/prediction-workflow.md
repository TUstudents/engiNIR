# Plan: prediction on project data files, with prediction ROIs

Status: implemented (2026-11-10, branch feature/band-selection-vip); contracts reviewed independently

## Why

The first prediction in the walkthrough (model "PLS 7 LV, SNV" on the
layer-5 tray of `Teflon_10p`) showed three problems.

1. **The Run Prediction dialog follows none of the project's rules.**
   - It lists model ids instead of names.
   - It asks for a free path and an optional "dataset id".
   - It shows the calibration as raw enum values (`dark_white_reference`)
     with an **Output unit** choice. That choice only labels the
     calibration: prediction always converts the scan to the model's signal
     unit and replays the model's recipe
     (`transform_spectra_to_signal_domain`, `recipe.build_steps`). So
     "Reflectance" for an absorbance model is misleading, not wrong.
   - It has none of the Add data file dialog's checks: no mean spectrum
     against the references or levels, no reference frames, no reuse of
     another file's references.
2. **The map answers the wrong question.** The colour scale runs from −68
   to 25, set by the substrate, which the model never saw. The wells are
   squeezed into two shades of green. The question is "what does each tablet
   contain, and can I trust it", and the page gives no per-tablet number.
3. **The contracts forbid what is needed.** Predict may not register the
   scan or draw ROIs (`viewer-workspace-ui-contract_v3.md` §3, "Forbidden
   controls in Predict"; `prediction-runtime-contract_v3.md`, Result
   Handling). So statistics over tablets are impossible today.

## Proposal

### A. One data-file path for training and prediction

The scan to predict is a **project data file**, added with the same **Add
data file** dialog (`AddDataFileDialog`) as training scans: same
calibration choices, same plot, same checks. Nothing is predicted from a
loose path in the viewer any more.

- **Run Prediction** asks for two things only:
  - the **model**, by name, with its target, signal unit and recipe shown
    read-only;
  - the **data file**, from the project's list, or **Add data file...**,
    which opens the Add data file dialog.
- The dialog shows the chain the prediction will apply, all taken from the
  model and none editable: calibration of the file → the model's signal unit
  → the model's recipe (bands, filter, normalisation, band selection) → the
  model.
- **Compatibility is checked before running**, with the reason shown:
  - the calibration must be able to produce the model's signal unit (no
    raw-count file for an absorbance model);
  - the band count must match;
  - the wavelengths must match, or a warning is shown (G18 stays a
    warning).
- Training and prediction then share one chain by construction: the same
  `CalibrationSpec`, the same `transform_spectra_to_signal_domain`, and the
  same recipe.
- **Scripts:** `model.predict(path)` keeps working for files outside the
  project; the facade already resolves calibration as `add_dataset` does.

### B. Prediction ROIs: statistics on the tablets only

ROIs on a prediction scan are ordinary project ROIs, drawn in Explore with
**Add ROI** or **Add ROI grid** on that data file. A prediction summarises
every ROI of its data file:

- per ROI, statistics of its **pixels** (decision 4):
  - mean, median, SD and 5–95 % of the pixel predictions;
  - the mean of the pixels' DModX and T²;
  - the number of pixels, and the share above the DModX and T² limits;
- **pixel exclusion by DModX** (optional, off by default): pixels above the
  DModX limit (level chosen, default 0.99) are left out of every ROI
  statistic, so glare, exposed substrate and direct reflections do not shift
  the result. The statistics are recalculated from the remaining pixels, and
  each ROI states how many pixels were removed. Switching the exclusion or
  the level recalculates from the prediction in memory, without predicting
  again. An ROI with no pixel left shows no value. The map can show the
  removed pixels greyed;
- the removed share is shown next to what the model's own training ROIs had
  (stored per ROI by the fit), e.g. "12 % removed (training ROIs 2–7 %)";
  no hard "in the model" flag;
- if the ROI has a reference value for the model's target, the residual.
  Over all such ROIs of the scan, RMSEP and bias form an independent test of
  the model.

**ROIs and fitting.** Prediction-only ROIs never enter a fit (decision 3).

### C. The Predict page, redrawn

- **Map:** the colour scale is the target's training range (the model's
  reference values, for example 1–10), with values outside drawn in their
  own colours; **your own limits** can be set in the ⋯ menu (for one scale
  over a series of trays). The colour bar and the CSV record the scale. With
  the exclusion on, removed pixels are greyed. ROI outlines are drawn on the
  map, labelled with their mean.
- **Per ROI** plot, replacing the pixel histogram as the main second plot:
  - one bar per ROI with its mean and 5–95 % range;
  - the removed share as a marker on a second axis, with the training ROIs'
    range as a band;
  - the reference value as a marker when the ROI has one;
  - clicking a bar selects the ROI.
- **Pixel histogram:** kept, over the ROIs' pixels only (after the exclusion
  when it is on), or over all pixels when the scan has no ROIs.
- **ROI statistics controls** in the Predict controls: **Remove pixels above
  the DModX limit** with its level; the per-ROI plot, histogram and summary
  follow at once.
- **Summary panel:**
  - names instead of ids;
  - word-wrapped values, so nothing is cut off ("Predictio", "Calibrat");
  - three groups: *Chain* (model, signal unit, recipe), *Scan* (file,
    calibration, clipping), and *Applicability* (share inside the limits,
    per scan and per ROI).
- **Small fixes:**
  - "every 2nd frame";
  - the preview shape is no longer shown as if it were the map shape.

### D. What stays

- **Predictions are not stored** (G18 and Result Handling unchanged):
  - the prediction, its map and the per-ROI statistics stay in memory and are
    exported through the plot panels;
  - only the data file, its calibration and its ROIs are project content,
    as for training scans.
- **The library is unchanged:** chunked prediction, the applicability
  diagnostics and the wavelength warning stay as they are; the per-ROI
  statistics are added on top of them.

## Decisions

Taken with the user (2026-11-10):

1. **A prediction scan is registered as a project data file** with the same
   **Add data file** dialog as training scans (A). Scripts keep
   `model.predict(path)` for files outside the project.
2. **Prediction ROIs are ordinary ROIs**, drawn in Explore with **Add ROI**
   or **Add ROI grid** (B).
3. **"Prediction only" is a field of the ROI**, set in the ROI and the ROI
   grid dialogs (`TrainingROI` gains a `use` of `training` or
   `prediction`, default `training`). A prediction-only ROI never enters a
   sample set, whatever values it has. It is shown as such in the ROI table,
   the mosaic and the Project ROIs spectra, and can be switched in the ROI
   table. If it has a reference value for the model's target, the value
   serves only the residual and the RMSEP of the prediction (adopted
   with the start of the implementation).
4. **Per ROI: statistics of its pixels**, with an option to remove pixels
   above the DModX limit (glare, substrate, direct reflections) and
   recalculate. No ROI-level limits and no hard "in the model" flag; the
   removed share is shown next to the training ROIs'.
5. **Colour scale: the training range**, and the user's own limits in the ⋯
   menu (no percentile scale).

Taken with the user after walking through the whole workflow (2026-11-10):

6. **The Prediction only checkbox remembers the last choice per data file**
   (user settings): a tray with prediction ROIs starts ticked, a new tray
   unticked.
7. **Two actions in Predict:**
   - **Predict ROIs**: the model on the prediction-only ROIs, from the
     project file alone (every ROI stores its raw pixels and references, as
     the Project ROIs tab uses them); for one data file or all of them, one
     table for a series of trays. No data file needs to be reachable.
   - **Predict map**: the model on the shown data file, every pixel, with
     the ROIs drawn on the map and their statistics.
8. **Several models side by side** in Predict ROIs: the table has one group
   of columns per model, so models can be compared on the same tablets.
9. **The pixel exclusion defaults to the model's own**: a model fitted with
   DModX exclusion starts with removal on at its level; a model fitted
   without starts with removal off.
10. **Per-ROI results are not stored**: no prediction record; the CSV and
    image exports keep them. The Result Handling rule stays as it is.
11. **The Add data file dialog shows what the model was trained with**
    (e.g. "model trained with: detector levels 128 / 3500") when it is
    opened from Predict; it never reuses training references silently
    (dataset-calibration contract).

The per-ROI results export as a table, the CSV of the per-ROI plot: ROI name
and id, data file, pixels, removed pixels, mean, median, SD, 5–95 %, mean
DModX and T², shares above the limits, reference, residual; the metadata
records the exclusion and its level.

## Review outcome (2026-11-10)

The independent review led to: one per-ROI result for map and ROI
predictions (always from the ROIs' stored pixels); only the exclusion's
method and level carry over from the fit (DModXPS is unweighted, no cap per
ROI); tight definitions of valid, removed and kept pixels, percentiles and
RMSEP scope; the training share shown at 0.99 only; per-ROI compatibility
and logging; `use` set at creation and then only by `save_roi_use`.
Accepted: engiNIR 1.3 and earlier ignore `use` (an unknown attribute) and
would train on prediction ROIs; the schema version is not raised, so older
versions still open these projects.

## Contract changes (Phase 0, with an independent review)

- `viewer-workspace-ui-contract_v3.md` §3 Predict:
  - the data file comes from the project, or is added with the Add data file
    dialog;
  - the chain is shown read-only;
  - per-ROI results, map colour scale and masking, and summary groups;
  - "ROI drawing" stays forbidden *in Predict*; ROIs are drawn in Explore.
- `prediction-runtime-contract_v3.md`:
  - per-ROI prediction statistics (definitions, the DModX pixel exclusion,
    RMSEP over ROIs with values);
  - Result Handling: the scan *is* a project data file in the viewer (added
    explicitly by the user, not by the prediction); the prediction itself is
    still not stored.
- `roi-target-contract_v3.md` and `sample-set-model-run-contract_v3.md`:
  the ROI's `use` (`training` | `prediction`); sample sets take only
  `training` ROIs; ROIs stored before read as `training`.
- `project-schema-contract_v3.md` / `docs/hdf5_format.md`: the `use`
  attribute of a stored ROI.
- `docs/contract_coverage.md`.

## Implementation notes (2026-11-10)

- The per-ROI plot compares each ROI's share of pixels above DCrit (0.99)
  with the training ROIs' share, like for like; the removed count is in the
  table.
- Found and fixed on the way: `model.predict(path)` of a registered file
  built a new detector-level calibration instead of the file's own, and the
  Controls dock (sideways scroll, 948ed1e) could hide its lower rows.

## Phases

0. **Contracts**, reviewed independently.
1. **Library:**
   - `roi_prediction_summary(prediction, rois, exclude_above=None)`:
     per-ROI statistics from a `Prediction`, using the same pixel indexing
     as the ROI's stored geometry; `exclude_above` a DModX level (its limit
     from the model) leaves those pixels out;
   - RMSEP and bias over ROIs with values;
   - `Model.predict` accepts a registered data file and returns the summary
     for its ROIs;
   - the ROI's `use`, stored and hydrated (absent reads `training`);
     sample sets and `resolve_sample_set` take only training ROIs.
1b. **Library, ROI prediction:** `predict_rois(model, roi_ids)` converts
   each ROI's stored pixels with its own calibration (as the fit does, via
   `roi_to_signal_domain`), replays the recipe, predicts and computes DModX
   and T² per pixel; the per-ROI statistics follow; several models share the
   conversion where their signal domains agree.
2. **Dialogs:**
   - the ROI and ROI grid dialogs get a **Prediction only** checkbox; the
     ROI table a column to switch it, and the mosaic and spectra mark it;
   - Run Prediction rebuilt: model by name, project data file or **Add data
     file...**, the chain shown read-only;
   - compatibility reasons shown before running;
   - remove the free path, the "dataset id" field and the calibration enums.
3. **Predict page:**
   - map colour scale and masking;
   - ROI outlines on the map;
   - the per-ROI plot, histogram over ROI pixels, and the summary groups;
   - small fixes.
4. **Tests:**
   - the per-ROI statistics against a hand computation;
   - the exclusion removes exactly the pixels above the limit and the
     statistics are recalculated (a glare pixel injected into an ROI shifts
     the mean without the exclusion, not with it);
   - RMSEP;
   - the chain shown equals the model's;
   - a raw-count file refused for an absorbance model;
   - a prediction-only ROI with a value never enters a sample set or a fit;
   - fault injection for each.
5. **Docs:**
   - user guide: a prediction section in the viewer guide and the workflow
     ("Predict tablets on a new tray"), and the help texts of the new plots
     (?);
   - changelog.

## Out of scope

- Prediction intervals (`plans/prediction-uncertainty.md`; its per-pixel
  interval would add one column to the per-ROI statistics later).
- Storing predictions.
- Interpolating scans with a different wavelength axis.
