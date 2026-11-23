# Backlog

Status: open points left after the review of every plan (2026-11-22;
updated 2026-11-23). Not
planned; each item gets its decisions before work starts. Pixel binning has
its own plan (`plans/pixel-binning.md`) and the 2.0 items are in
`plans/v2.0.md`. The plans of finished work were deleted; `git log` has them
(last commit that contained them: `ab26a81`).

## Features

1. **Click a point of the component curve to fit with that many components**
   (from the component-selection plan). Fit Again opens the settings with
   another number, but the curve is not clickable.
2. **Store the wavelength table's camera as `sensor_name`** (reserved in the
   v3 contracts for calibration transfer in 2.0).
3. **Change the reference correction of a registered dataset from the viewer.**
   The library replaces it; the viewer asks only when a new ROI would need a
   new one. Check that it is still wanted.
4. **Leave-one-data-file-out validation** (needs its own contract decision,
   because cross-validation groups are per ROI).
5. **PCA, T² and Q of the ROI means on the Project ROIs tab.**
6. **Kernel leftovers:** MSC and EMSC recipes on the sums (they keep the
   pipeline path), one diagnostics pass instead of three for the final model,
   and fit progress naming its phase.
7. **Glossary entries** for Kennard-Stone, hold-out and ROI grid (and check
   "dark and white reference").
8. **Packaging:** a signed Windows installer, opening `.h5` by double-click,
   a Linux `.desktop` entry.

## Housekeeping

9. **`save_dataset_ref`** (23 places) names a `DatasetRecord`, not a
   "dataset ref"; it is public API (`enginir.io`, user guide), so a rename
   needs an alias and a deprecation, not a cleanup.
10. **Tests:**
    - five near-twin project builders remain;
    - mutation survivors worth a test: `_exclusion` residual helpers and
      budget comparisons, `_components` adjusted R² and RMSEE denominators,
      `wavelength_tables` header parsing, `metrics._validated_pair`;
    - coverage holes: `_inspector` 63 %, `_canvas` 73 %, `spc_io` 72 %;
    - slow: `test_closing_a_project_viewer_exits_without_a_crash` 9.6 s;
    - about 1000 references to private widget attributes in the viewer tests
      (a design change, not a cleanup).

## Needs a real machine

11. **Windows novice walkthrough:** one-line install, Start menu, taskbar pin
    and icon, update, uninstall.
12. **Windows startup measurement:** cold and warm, with the `Startup timing`
    log lines (per warm-up step since `plans/cold-start.md`). First result
    (2026-11-22, 1.7.4): warm 0.8 s to the welcome window, warm-up 3.7 s;
    cold 19.3 s and 31.3 s. Per step on a second machine, fresh install:
    `plans/cold-start.md` (SciPy 12 of 20 s; half small files, half native).
    A start after a reboot on the work laptop is not recorded yet.
13. **Rare flakiness, not reproduced since:** a hang in garbage collection, a
    hang of the two-window tests, a Qt `commitData` warning when editing the
    ROI table, a stall of the full run. A fault in any later CI run reopens
    `G37`.

## Done since the review

- **Kestreleye wavelength tables** (was 9): built in 1.7.2 (Add Dataset's
  Wavelengths, Set Wavelengths, `wavelength_table=`); the item was stale.
- **Dead code** (was 10): removed before 1.7.5.
- **`contracts/coverage.md`** (was 11): every cited name checked against
  1.7.6 (2026-11-23); seven replaced by their successors. A cited test,
  `test_on_the_training_scan_the_statistics_are_the_training_ones`, had been
  deleted with the test reduction although it alone pinned T²PS = T² and
  DModXPS = DModX / v on the training scan; restored.
- **`docs/api/io.rst`** (was 12): `file_format`, `EnginirIOError` and
  `FILE_SCHEMA_VERSION` documented.
- **Citations of deleted plans** (was 13): dropped from code, tests, docs and
  contracts (dates and versions kept); the CHANGELOG keeps them as history.
- **Stale names** (was 14): "precalibrated" is "already corrected" (the
  migration keeps the old stored value's name); the `phase5`/`phase6` test
  helpers and file names are renamed.
