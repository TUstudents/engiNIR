# Plan: library support for inline prediction (engiNIR-PRINT)

Status: pieces 1 to 3 built on branch `print-support` (2026-11-23); piece 4
waits for the status window's design in engiNIR-PRINT. The application itself is
engiNIR-PRINT, a separate repository (`../engiNIR-PRINT/design/architecture.md`);
this plan is only what engiNIR adds for it. The rule for the split: what the
viewer or a script would need too belongs in engiNIR; what makes sense only for
printing belongs in engiNIR-PRINT.

## Context

Kestreleye (EVK Helios EC-32 over GenICam) moves the stage and writes one
`stream_hsi.h5` per printed layer. engiNIR-PRINT predicts every layer with a
model of an engiNIR project, per tablet, and serves the results over OPC UA.
`Model.predict(path)` already does the prediction (compatibility check first,
chunked reading, the model's binning, DModXPS/T²PS, intervals, one provenance
log entry). Four pieces are missing.

## 1. The model's wavelength axis for band-number files

A Kestreleye stream stores band numbers only. `Model.predict` cannot say "use
the model's axis", so every layer would log the differing-axis warning (G18).
The viewer's Predict step has the same gap.

- `Model.predict`: a file whose axis is band numbers (`is_band_numbers`) is
  given the axis of the model's input contract when the band counts match;
  the provenance says so. Optionally `wavelength_table=` as in `add_dataset`.
- A different band count still refuses (as now).
- Contract: prediction-runtime contract, files without wavelengths.

## 2. Selectable reference correction in `predict`

`Model.predict` passes only `dark=`/`white=` on; `_resolve_dataset` already
takes `black_level`/`white_level`, `dark_frames`/`white_frames` and
`references_from`.

- `Model.predict(path, *, dark, white, dark_frames, white_frames,
  black_level, white_level)` with the same rules and refusals as
  `Project.add_dataset` (levels exclude references).
- Default unchanged. engiNIR-PRINT's default is the detector levels.

## 3. Per-tablet statistics on a new file from a layout

Per-ROI statistics (`roi_prediction_statistics`: mean, SD, DModX share,
exclusion at a level) exist only for ROIs stored in a project. The line needs
them for a layout on a file that is never registered.

- A layout: named rectangles `(frames, rows)` in the file's pixels (binned
  pixels for a binned model), as ROIs are drawn; made from an ROI grid
  (`GridCell` layout) or listed in a file.
- `Prediction.roi_statistics(layout, *, exclude_dmodx=None)` returns the same
  `RoiPredictionSummary` as for stored prediction ROIs, computed by the same
  function, so a tablet's value is the same in the viewer and on the line.
- A layout cell outside the scan refuses, naming it.

## 4. A public widget API for engiNIR-PRINT's status window

The status window reuses viewer parts, which are private today
(`PredictionMapWidget`, plot panels, theme, colour maps). Make a short list
public and documented in `enginir.viewer` (API docs, viewer contract), so an
engiNIR release cannot break engiNIR-PRINT silently. The list follows from the
window's design in engiNIR-PRINT.

## 5. The public reading surface (added 2026-11-24)

engiNIR-PRINT uses public names only (an AST check there). Added:
`Model.check_file` (the refusal of `predict` without predicting, for checking
a job template), `Model.id`, `Model.input_contract` (a copy; the binning a
layout is drawn in), `enginir.read_header`/`read_frames`, and `py.typed`.

## Later (not in this plan)

- Tablet detection on a scan (image segmentation), as an alternative to a
  layout.
- The frozen method identity (project, model id, project checksum): part of the
  v2 method file. Until then engiNIR-PRINT records the project's hash and the
  model id with every result.

## Tests

Each piece fault-injected: a band-number file predicted with the model's axis
(and refused with another band count); `predict` with each reference choice
equals `add_dataset` + `predict_rois` on the same pixels; layout statistics
equal the statistics of the same rectangles stored as prediction ROIs.
