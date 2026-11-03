# Plan: projects with ROIs from several data files

Branch: `feature/multi-file-workflow`, based on `feature/plot-panels` (it uses
the plot panels and plot pages). It merges after that branch.
Status: phases 0–2 done
Date: 2026-11-02

## Goal

One project collects ROIs from many data files. The user can:

- add data files to the project in one window, each with its own references;
- draw single ROIs or a whole tray grid on a data file;
- look at all stored ROIs from the project file alone, without the data files;
- type the reference values per ROI;
- fit a model on the ROIs of all files;
- reopen the project later and find everything again.

## How the user works (from the design discussion)

- References come in four ways, all of which must work:
  - separate dark/white files;
  - a white tile (and dark frames) inside the same image;
  - one reference set for a whole session of files;
  - files that are already reflectance.
- One ROI is one physical sample, with no replicates. ROI-level targets and
  cross-validation stay as they are.
- Values are typed per ROI. No CSV import is needed; column paste is.
- A data file holds many samples (tray or grid), so a grid ROI tool is
  needed.

## What the code does today

- The loader's Open Dataset and Create/Open Project buttons work.
- References can only be given at startup, with the loader checkboxes.
  `_attach_dataset_path` takes no references from inside the window.
- A data file is registered in the project only when its first ROI is drawn
  (`EDAView._ensure_current_dataset_context`).
- An ROI stores, in the project:
  - its raw pixels, in frame × row order. `make_roi` reshapes
    `pixels[:, r0:r1 + 1]`, so a patch can be rebuilt exactly;
  - the dark/white reference spectra of those pixels;
  - its provenance and calibration id.
- The fit reads only the project file (`build_sample_set`,
  `_signal_domain_matrix`).
- File > Open Data in a window that already shows a data file opens a second
  window without the project. Opening the project there shows a false
  "Dataset mismatch" warning, because ROIs come from another file.
- Every window draws all ROIs of the project on its image
  (`EDAView._on_rois_changed`). Other windows on the same project are not
  refreshed.
- Values are entered in the new-ROI dialog or with **Assign target**, one ROI
  at a time. A value cannot be removed: there is no delete for a
  `TargetAssignment`.
- The model's preprocessing is the Explore preview at the moment of **Build
  sample set**, a hidden coupling. The fit is a second step.

## Design

### Explore: two tabs in the image area

```
Explore ▸ [ Data file: tray03.eiff ▾ ] [ Project ROIs (124) ]
```

**Data file tab:**

- The image of one data file of the project, chosen in the ▾ list or the
  browser's Datasets section and loaded from disk in the background.
- Only that file's ROI rectangles are drawn, with name and current target
  value.
- Add ROI and Add ROI grid work here.
- The Sensor, Spatial, Spectral and RAW plots work on the file's pixels.
- A missing file shows "File not found" with **Locate file...**.

**Project ROIs tab** (needs only the project file):

- The *ROI mosaic*: one tile per ROI, redrawn from its stored pixels, grouped
  by data file in grid order.
  - Tiles keep their true relative size within a file.
  - They use the false colour of the Spatial Image (raw bands near
    1650/1300/1050 nm), with one contrast stretch for all tiles.
  - Each tile shows the ROI name and the current target value.
  - Filters: data file, has value / missing value; sort by name or value.
- Plots below the mosaic:
  - `explore.roi_means`: the mean spectrum of every shown ROI, coloured by
    the current target value. It uses the Explore signal unit and
    preprocessing, and converts each ROI with its own stored references
    through `transform_spectra_to_signal_domain`, as the fit does.
  - `explore.roi_pixels`: the pixel spectra of the selected ROI, as the
    median with a 5–95 % band.
- The file-based plots (Sensor, Spatial, RAW) are hidden in this tab.

Linking between the tabs:

- Clicking a tile, a table row or a Model point selects the ROI everywhere.
- Double-clicking a tile opens the Data file tab with that file and ROI.
- New ROIs appear as tiles at once.

### Adding a data file

**Add data file...** (Explore and File menu) opens a dialog with:

- **Data file**.
- **Calibration**, one of:
  - **Raw counts** (`raw_dn`);
  - **Already reflectance/absorbance** (`precalibrated_import`);
  - **Dark/white references** (`dark_white_reference`). Dark and white are
    each taken from either:
    - another file, averaged over all its frames;
    - a frame range of this file, such as a white tile or shutter-closed
      frames.

    The source is recorded in `dark_ref_source` / `white_ref_source`, for
    example `tray03.eiff frames 0-20`. A white region must cover all rows.
  - **Same references as...**: an earlier data file of the project, for one
    reference set per session.
  - Default levels (black 128, white 3500) remain the fallback.

The file and its calibration are registered at once and shown. Replacing
the calibration later applies to ROIs drawn afterwards; existing ROIs keep
their own (decision D1).

In a project window, File > Open Data does the same. A data file is never
opened in a second window without the project.

### ROI grid tool

**Add ROI grid...**:

1. Drag one rectangle around the tray.
2. Set:
   - rows × columns;
   - inset in % of each cell;
   - name pattern (default `{file}-{row letter}{column:02d}`, e.g.
     `tray03-A01`);
   - order, row by row or column by column.
3. A live preview shows the cells. Clicking a cell skips it.
4. **Create** stores all ROIs in one validated write: all or nothing. Names
   are unique in the project.

### ROI table

A dock, opened from Explore and Model:

- One row per ROI: data file, name, frames/rows, pixel count.
- One column per target.
- It opens filtered to the shown data file, in grid order.
- **Editing:**
  - Type a value and press Enter to move to the next row. The value is
    stored when the cell is left.
  - Deleting a value removes the assignment, which needs a new store
    operation.
  - Pasting a column fills rows in order.
- Clicking a row selects the ROI everywhere.
- The value field in the single-ROI dialog stays.

### One Fit model dialog

**Fit model...** sets target, signal unit, preprocessing (default: the
Explore preview, editable in the dialog), model, components and CV.

- It builds or reuses the sample set (all ROIs with a value for the target,
  from all files), as `Project.fit` already does, then fits.
- The separate **Build sample set** button goes.
- The readiness line counts values per data file, e.g. "38 of 40 ROIs of
  tray03 have API values".

### Reopening a project

- **Which tab opens:**
  - With ROIs: it opens on **Project ROIs**, with no data file loaded, which
    is instant.
  - Without ROIs: **Data file** with **Add data file...**.
- The last shown data file loads when the Data file tab is opened. It is
  stored in the user settings by project id, not in the project.
- **Missing files:**
  - A missing file is marked in the list.
  - **Locate file...** checks the chosen file by comparing the stored raw
    pixels of that file's ROIs with the file's pixels at the same bounds,
    plus its shape and wavelengths.
  - It then updates `source_path`. This needs a contract exception: the path
    is a location, not identity.

### Saving feedback

- After each write, the status bar says, for example, "ROI 'tray03-A01' saved
  to project" or "40 ROIs saved".
- Windows that show the same project refresh when it changes.

## Decisions

| Id | Decision | Default |
|---|---|---|
| D1 | Changing a data file's calibration after it has ROIs | **Revised in phase 1:** allowed, as the calibration contract already permits. It applies to ROIs drawn afterwards; existing ROIs keep their calibration and reference snapshot. The UI says so first, with the number of ROIs. A lock would have restricted existing, consistent behaviour. |
| D2 | Leave-one-data-file-out validation | Later, as an extra check. Its own contract decision, because CV grouping is per ROI. |
| D3 | Second windows on one project | No new project windows from Open Data. Windows that do show the same project refresh. |
| D4 | Mosaic tile size and colour | True relative size within a file; false colour from raw counts. |
| D5 | PCA, T² and Q of the ROI means in the Project ROIs tab | Later, with the LV curve and T²/Q plans. |
| D6 | Name pattern for grid ROIs | `{file}-{row letter}{column:02d}`, editable. |

## Phases

Each phase ends with all four gates clean (pytest, ruff, pyright, Sphinx `-W
-E` into a fresh directory). It also gets fault-injection checks for its new
tests and a screenshot check of the changed views.

### Phase 0: baseline

- Gates on the branch.
- Opt-in benchmark project (`ENGINIR_PERF=1`): 10 data files × 40 ROIs ×
  about 2,000 pixels × 224 bands. Record:
  - project open time;
  - time to the first visible tile;
  - time to all mean spectra.

**Done (2026-11-02).** The gates on the branch are clean: pytest 531 passed,
3 skipped; ruff, pyright and Sphinx clean.

`tests/test_multi_file_performance.py` (opt-in) builds the benchmark project.

- Content: 10 HDF5 data files of 320 × 250 × 224, each with a 5 × 8 tray of
  ROIs of 40 × 50 px with an API value per ROI. That makes 400 ROIs and
  800,000 pixels.
- Building it with the `Project` facade takes 21 s. The project file is
  170 MB.
- Machine: WSL2, 10 cores, offscreen Qt.

Baseline (two runs):

| Step | Time |
|---|---|
| Open project (`ProjectState.open_project`) | 0.78–0.79 s |
| Open project window, project-only (`viewer.open_project`) | 2.35–2.38 s |
| Read one ROI's pixels from the project | 5–6 ms |
| Read all 400 ROIs' pixels | 1.74–1.86 s |
| Raw mean spectra of all ROIs, in memory | 0.09 s |
| Fit PLS 5 LV, auto CV (7 folds + final) | 41–47 s |

What this means for the plan:

- **First tile.** Reading one ROI takes 6 ms, so a first tile under 1 s
  holds easily. All 400 ROIs take about 2 s, which is fine in the
  background.
- **Opening.** Opening the project is already under 1 s. The project-only
  window takes 2.4 s, so the phase 9 target refers to `ProjectState` and the
  first tile, not to building the window.
- **Finding: the fit is the slow step at this size.** A profile shows where
  the 47 s go:
  - 39 s are scikit-learn's PLS on 800,000 × 224 pixels, 8 times. Its
    deflation `np.outer` alone is 10.6 s, and it centres and scales in
    float64.
  - 4.3 s are reading the ROI spectra twice (`build_sample_set` is called
    twice, both through `load_roi_spectra_matrix`).

  This is library work, not part of this plan. It should be a follow-up:
  read the spectra once, and use a float32 PLS with in-place deflation. The
  phase 7 fit dialog must run the fit as a background job with progress
  and cancel, which it already does for the current fit.

### Phase 1: contracts and ledger

- `project-dataset-attachment-contract_v3.md`:
  - one project window with several data files;
  - Add data file with calibration, registered at once;
  - references from a frame range of the same file, and "same references
    as";
  - calibration fixed once the file has ROIs (D1);
  - only the shown file's ROIs drawn;
  - mismatch checked per data file against its own registration;
  - Open Data in a project window adds a file (D3);
  - reopen behaviour;
  - Locate file with the pixel check.
- `project-schema-contract_v3.md`: `source_path` may be updated after the
  pixel check, as an identity exception next to the existing ones in §5.
- `roi-target-contract_v3.md`:
  - the ROI table edits all targets, which closes the open question "Target
    editing UX";
  - removing a value deletes the assignment;
  - the grid tool writes all or nothing.
- `viewer-workspace-ui-contract_v3.md`:
  - Explore has the Data file and Project ROIs tabs;
  - Project ROIs is allowed in project-only mode;
  - the mosaic and the ROI plots follow the Plot Panel Policy.
- `model-workspace-diagnostics-contract_v3.md`: the left control area has
  one fit entry point that builds the sample set.
- `docs/contract_coverage.md`: rows and gap `G20`.

**Done (2026-11-02).** Contract changes:

- **`project-dataset-attachment-contract_v3.md`:**
  - the core decisions include several data files per project;
  - project-only mode may show stored ROIs, edit target values, build sample
    sets and fit; these are explicitly not dataset-derived;
  - mismatch validation is per data file against its own registration;
  - new section *Data Files Of A Project*: adding with calibration (written
    together), switching, missing and moved files with the pixel check,
    reopening, several windows;
  - `Open Data...` keeps the project from every state;
  - new required tests; the non-goal narrowed to "several files shown at
    once".
- **`dataset-calibration-contract_v3.md`:**
  - the three reference sources (file, frame range, another data file), with
    their source fields;
  - mean spectrum and saturation shown before acceptance;
  - replacing a calibration affects only later ROIs (D1);
  - open question 2 (moved files) resolved.
- **`project-schema-contract_v3.md` §5:** two new identity exceptions,
  `DatasetRef.source_path` after `Locate file...` and `TargetAssignment`
  removal.
- **`roi-target-contract_v3.md`:**
  - ROI grid, all or nothing;
  - removing a value deletes the assignment;
  - the ROI table (one column per target, column paste all-or-nothing, works
    without data files) closes the open question "Target editing UX";
  - CSV import listed as out of scope;
  - new acceptance criteria.
- **`viewer-workspace-ui-contract_v3.md`:**
  - Explore has the `Data file` and `Project ROIs` tabs, with their plots,
    controls and behaviour;
  - the Model controls are the ROI table and one fit dialog;
  - readiness is counted per data file.
- **`model-workspace-diagnostics-contract_v3.md`:** one fit entry point that
  builds the sample set, an explicit preprocessing choice, readiness per data
  file, and the data-file count in the summary.
- **Ledger:** seven rows and gap `G20` (open, P1). `AGENTS.md` names the gap
  range `G01`–`G20`.

Like the plot-panel branch, the contracts change ahead of the code on this
branch. They merge together with the implementation phases.

### Phase 2: quick fixes

These are useful on their own:

- Draw only the shown data file's ROIs (`EDAView._on_rois_changed`).
- Mismatch check per data file (`_attachment.confirm_project_dataset_binding`).
  ROIs of other registered files are not a mismatch, and ROIs are named, not
  shown as ids.
- Status-bar "saved" messages for ROI, value and target writes.
- Tests:
  - a two-file project shows only the file's own rectangles;
  - no warning for ROIs of other files, but a warning when the attached file
    differs from its registration.

**Done (2026-11-03).**

- **Overlays:** `EDAView.refresh_roi_overlays` draws only the ROIs of the
  shown file (`shown_dataset_id`, the project dataset at the shown path).
  Attaching a file in a project window redraws them.
- **Mismatch:** `_attachment.confirm_project_dataset_binding` compares a
  file only with the `DatasetRef` at its own path.
  - An unregistered file is not a mismatch.
  - A changed registered file names the differing fields and its ROIs
    ("Data file changed").
- **Saved messages:** `ProjectState.saved` reports ROI, value, target and
  removal writes. The status bar shows them for 8 s (`_saved_label`).

**Bug found while testing.** Re-attaching a registered file warned
"wavelength axis differs" for the same file. The project stores wavelengths
as float32; a freshly loaded axis is float64, and the two differ by about
1e-4 nm near 1700 nm, above the 1e-6 nm tolerance. The bug predates this
branch. `wavelength_axes_match` now compares both axes at float32, the stored
precision; a real 0.01 nm shift still differs.

**Process fix.** That warning blocked the test run for hours: under offscreen
Qt a modal dialog waits forever, and the 60 s pytest timeout does not
interrupt Qt's event loop. `tests/conftest.py` now turns any unexpected
`QMessageBox.warning/critical/information/question` into an immediate test
failure; tests that expect a dialog patch it themselves. The whole suite
passed with the guard, so no test relied on an unpatched dialog.

Tests:

- `tests/test_multi_file_workflow.py`:
  - two-file overlays;
  - a project-only window drawing the ROIs of the file it attaches;
  - saved messages.
- `test_project_open.py`:
  - another file attaches without a warning;
  - a changed registered file uses the warning gate (Cancel and Yes).
- The float32 wavelength test.

Seven injected faults were each caught: all ROIs drawn, new file warns, shape
not compared, no saved message, attach without redraw, ROIs not named, float64
wavelength comparison.

Still open in this area: `Open Data...` in a window that already shows a data
file opens a second window without the project. That is phase 3.

### Phase 3: data files and references (library first)

- Library, Qt-free (`_workflow.py`, `project/`):
  - reference from a frame range: `reference_from_frames(path_or_cube,
    frames)` gives the mean over frames as `(height, n_bands)`;
  - `Project.add_dataset(..., dark=..., white=..., dark_frames=...,
    white_frames=..., references_from=...)`;
  - the notice before replacing the calibration of a data file with ROIs (D1).
- Viewer:
  - Add data file dialog;
  - data file list (tab header combo and browser);
  - loading on switch in the background (`_jobs`), restoring the file's
    active calibration into `ViewerState`;
  - Open Data in a project window goes through this path.
- Tests:
  - headless: references from frames equal the mean of those frames;
  - the source is recorded;
  - "same references as" copies the arrays into a new calibration of the new
    file;
  - a replaced calibration leaves existing ROIs on their own calibration;
  - viewer: two files added, switching shows each with its own calibration
    and rectangles.

### Phase 4: ROI grid tool

- Library:
  - `grid_bounds(frames, rows, n_rows, n_cols, inset, skip, order)`, pure
    and tested;
  - `Project.add_rois(...)`, validate-all-then-write.
- Viewer: grid dialog with live preview on the canvas, click-to-skip, name
  pattern.
- Tests:
  - bounds and inset;
  - skip;
  - order;
  - names;
  - a failure in one ROI writes nothing;
  - duplicate names refused.

### Phase 5: ROI table

- Store and facade: remove a target assignment.
- Viewer: dock with a table model over `ProjectState`. Cell edit assigns or
  removes the value; Enter moves down; column paste; filter by file;
  selection linking.
- Tests:
  - typing, removing and pasting are stored and reload;
  - empty cells stay missing, not zero;
  - selection reaches the canvas and the browser.

### Phase 6: Project ROIs tab

- Store: `load_roi_band_images(path, roi_ids, band_indices)` reads only the
  three colour bands per ROI and rebuilds patches from `spatial_bounds`.
- Viewer:
  - the mosaic as a pyqtgraph scene, tiles filled progressively by a
    background job, with filters and sort;
  - `explore.roi_means` and `explore.roi_pixels` as registered specs;
  - means computed in the background and cached per (ROI, signal unit,
    recipe);
  - double-click opens the Data file tab;
  - the mosaic is a `PlotFrame`: PNG/SVG of the tiles, CSV of the ROI list
    with values.
- Tests:
  - a tile equals the stored pixels reshaped by `spatial_bounds`;
  - the mean spectrum of an ROI equals the fit's conversion of its spectra
    (same function);
  - filters;
  - the tab works with the data file deleted from disk.

### Phase 7: Fit dialog

- One Fit model dialog that builds or reuses the sample set
  (`Project.fit` rules), with the readiness counts per file.
- Remove Build sample set from the UI.
- Tests: refitting with the same settings reuses the sample set, recipe and
  signal domain; a different recipe creates a new sample set.

### Phase 8: reopen and relink

- Start tab by project content, last file per project id in the user
  settings, missing-file state.
- **Locate file...** with the pixel comparison, then the `source_path`
  update.
- Refresh other windows on project change.
- Tests:
  - reopen shows Project ROIs without any data file;
  - a moved file is accepted only if its pixels match;
  - a different file with the same shape is refused.

### Phase 9: docs and performance

- User guide: the multi-file workflow from the loader to the fit, as one
  walkthrough.
- `hsi_viewer.md`, `viewer_architecture.md` and API reference updated;
  CHANGELOG.
- Re-run the phase 0 benchmark. Targets:
  - project open stays metadata-first, under 1 s for the benchmark project;
  - the first tile appears in under 1 s;
  - the UI stays responsive while means are computed.

## Risks

- **Memory:** the mean spectra of hundreds of ROIs need all their pixels.
  Read them one ROI at a time in the background and keep only the means.
- **Wavelengths:** data files with different wavelength axes cannot share a
  sample set. The fit dialog must say which ROIs are left out and why.
- **Reference-region quality:** a white tile that is saturated or not
  uniform gives bad reflectance. The dialog shows the reference's mean
  spectrum and the share of saturated pixels before accepting it.
- **Existing projects** stay valid: every new field and action is additive.
  The files from 2026-11-02 that fail on an empty `source_modality` are an
  older format and out of scope here.
