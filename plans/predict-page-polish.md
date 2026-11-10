# Plan: Predict page polish

Status: implemented (2026-11-10). Follows `plans/prediction-workflow.md`
(implemented, commits 4f80ebe and bcc35c6).

## Why

On the user's project (5 prediction ROIs on the `…_152927` scan) the Predict
page works, but:

- the per-ROI numbers are in the summary panel; they belong in the ROI table
  (the user asked for this already);
- the grey band in the per-ROI plot (the training ROIs' 10–90 % share of
  pixels above DCrit) runs from about 2 % to 70 % on this model. It looks
  like a background and cannot tell an unusual tablet from a normal one;
- the summary panel repeats the data file three times, the model twice, and
  has three status lines;
- the window's status line shows Explore's data file and signal (`Dataset:
  none | Signal: DN | Frame: -`), and the map hover readout is a small label
  under the map;
- the Controls dock has room, while colour limits are hidden in the ⋯ menu.

The Inspector stays as it is: it is the authoritative view of every project
item.

## Decisions

1. **ROI table carries the prediction results.** While there are ROI results
   in the session, the table gets, per model, the columns
   `<model>: mean`, `SD`, `% removed`, `% above DCrit` and `residual` (when
   the ROI has a reference). They:
   - are read-only, sortable, and empty for ROIs that were not predicted;
   - follow the pixel removal at once;
   - are never stored and go with a new project or new results, like the
     predictions.

   The reference value is the target's own column, which is already there.
   The per-ROI list in the summary is removed. The ROI table stays visible on
   Predict.
2. **Per-ROI plot shows predictions only.** It keeps the mean dot with its
   value, the 5–95 % bar and the reference ×. The right axis, the diamonds
   and the band are removed. A tablet whose share above DCrit is above the
   training ROIs' 90th percentile gets a hollow ring around its dot, with a
   legend entry "unusual spectra (share above DCrit > training 90 %)". The
   shares are in the ROI table and the CSV (the CSV is unchanged).
3. **Summary panel without repeats:**
   - **Headline** (one wrapped line): `Map of <file> with <model> — not
     stored`, or `<n> ROIs of <scope> with <models> — not stored` without a
     map. The "Ready" and "Service" lines go; progress stays in the Controls
     dock.
   - **Prediction ROIs:** scope, RMSEP and bias with *n* per model, and
     refused ROIs.
   - **Chain:** signal and recipe, target, bands, input. The model is in the
     headline.
   - **Scan:** calibration, map shape, map range, clipping. The data file is
     in the headline.
   - **Applicability:** as now.

   The text the Inspector receives for the Predict page follows the same
   groups.
4. **Status line on Predict:**
   - **Without hovering:** `Project | Map: <file> | Signal: Absorbance (SNV)
     | Model: <name> | Target: <name>`.
   - **Hovering the map:** `Frame f | Row r | Layer 6.12 (above training
     range) | DModX 1.8 (DCrit 1.4: removed) | T² 3.2 | ROI 03`. The shown
     frame and row are the true ones, also on a preview map. ROI is the
     prediction ROI under the cursor.
   - **Hovering a dot in the per-ROI plot:** `ROI 03 | <model> 6.05 ± 0.93 |
     6 % removed | 12 % above DCrit | ref 4`.

   The map's own hover label goes. Explore and Model keep their status line.
5. **Controls dock: Predict gets map and display options.**
   - **Run:** Predict ROIs..., Predict map..., Cancel, progress (as now). The
     two repeated text lines go.
   - **Pixel removal:** the checkbox and level (as now), plus the resulting
     DCrit value per model ("DCrit 1.42").
   - **Colour scale:** *Training range* / *Own* with min and max fields. The ⋯
     menu "Colour limits..." stays as a shortcut and sets the same fields.
   - **Map overlays:** ROI outlines, Greyed removed pixels (both on by
     default).
   - **Histogram:** *ROI pixels* / *Whole map* (default: ROI pixels when
     there are ROI results).

   The settings are session state; nothing is written to the project.

## Implementation

1. **State.** `ProjectState` gets the session-only `roi_results` (roi id → per
   model `RoiStatistics`, with model names) and a `roi_results_changed`
   signal, cleared on a project switch. `PredictionView` publishes into it
   after every recompute. The ROI table reads it, so the viewer still has
   one owner per fact and no workflow rule moves.
2. **ROI table.**
   - `RoiTableModel` appends the result columns after the targets. They are
     read-only, display formatted text, and sort by `Qt.UserRole` numeric
     values.
   - It reloads on `roi_results_changed` with
     `beginResetModel`/`endResetModel`, but only when the column set changes;
     otherwise it uses `dataChanged`, so selection and scroll stay.
   - Column widths are kept.
3. **Per-ROI plot.** Remove the twin axis. Add the ring flag from
   `share_above_dcrit` against the training 90th percentile (from
   `training_share_range[2]`), and its legend entry. Update the help text.
4. **Summary.**
   - `workspace_summary` gains `headline` and drops the `status`,
     `readiness` and `service` keys.
   - The viewer removes the Shown, Data file and Model rows and the per-ROI
     value rows (`roi_value_rows` goes).
   - The Inspector text is updated to the same groups.
5. **Status line.**
   - `_update_status_strip` asks the current workspace for its status text.
     Predict supplies `prediction_status_text()`.
   - `PredictionMapWidget` emits `hovered(text)` instead of setting
     `_cursor_label`. The text includes the ROI under the cursor from the
     ROI overlays and the DModX/T² values from the payload layers. The
     per-ROI plot's hover (`PlotPanel` with `RoiPoints`) gives the ROI line.
     An empty text restores the idle line.
6. **Controls dock.** Add the colour scale, overlay and histogram rows to the
   Predict controls. The map widget reads the overlay and colour settings
   from its payload extras (already the pattern for `colour_range` and
   `removed_limit`). The histogram source goes into `histogram_payload`.
   The `_WidthScrollArea` keeps every row reachable
   (`test_every_control_row_fits_in_the_controls_dock`).
7. **Contract and docs.**
   - Viewer contract §3: the ROI table result columns, the per-ROI plot
     without the share axis, the ring flag, the summary groups, the status
     line on Predict, and the Predict controls.
   - `docs/user_guide/viewer.rst` (Predict, ROI table);
     `docs/viewer_architecture.md` (`roi_results`, hover signal);
     `docs/api/viewer.rst`; CHANGELOG.
   - `contract_coverage.md` rows.

## Tests

- **ROI table:**
  - the result columns appear after Predict ROIs, with the mean of the
    statistics;
  - they change with the pixel removal without a re-prediction;
  - they disappear with a new project;
  - they are not editable;
  - sorting by mean orders by value.
- **Per-ROI plot:** no twin axis. A tablet above the training 90 % is ringed,
  one below is not (check that the test fails with the ring removed).
- **Summary:** each of data file and model appears once; no "Ready" or
  "Service" text in the summary.
- **Status line:**
  - hovering a map pixel inside an ROI gives frame, row, value, DModX with
    the removed state, and the ROI name, with true frame and row on a
    subsampled map;
  - leaving the map restores the idle line;
  - on Explore the line is unchanged.
- **Controls:**
  - own colour limits set from the dock reach the map and its CSV;
  - turning off outlines removes the rectangles;
  - the histogram switch changes the payload's source;
  - all rows fit (existing test).

## Not in this plan

- Storing ROI prediction results in the project (decided against, decision 5
  of `prediction-workflow.md`).
- Spectra of a hovered or clicked pixel on Predict.
- A different pixel-level DModX limit. The wide training share (2–95 %)
  says the 0.99 pixel limit is tight for this data. That is a model question
  for the Distance page, not a Predict display question.
