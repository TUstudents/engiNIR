# Backlog

Status: open points left after the review of every plan (2026-11-22). Not
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
9. **Kestreleye wavelength-table import** (the 176-band table is extracted;
   the import feature is not built).

## Housekeeping

10. **Dead code:** `_WidthScrollArea` in `viewer/_viewer.py`, and
    `title_elided` with `_ElidedLabel.elided` in `viewer/_plot_panel.py`.
    The one-line test accessors (`row_checks`, `fix_button`, ...) were kept on
    purpose and stay.
11. **`contracts/coverage.md`:** seven cited private names no longer exist
    (`_atomic_replacement_group`, `_touch_project_updated_at`,
    `_validate_v3_sample_set`, `_preview_geometry`, `_place_roi_table`,
    `_require_new_or_unchanged`, `validate_project_relationships.
    _validate_mapping_keys`); the header says "audited 2026-11-18, 1.7".
    Likely successors: `replacement_group`, `_open_project`,
    `_check_sample_set`, `_preview_step`, `_sync_roi_table`. A full re-audit
    against 1.7.1 to 1.7.4 is not done.
12. **`docs/api/io.rst`:** `EnginirIOError`, `file_format` and
    `FILE_SCHEMA_VERSION` are exported but undocumented (add after the
    binning work lands in that file).
13. **Citations of deleted plans:** about 38 source comments, and some
    contracts, docs and CHANGELOG lines, cite `plans/*.md` files that no
    longer exist. Reword them to the contract or drop the citation; until
    then they point to history in git.
14. **Stale names:** "precalibrated" in nine comments and docstrings (the
    stored value is `already_corrected`); `save_dataset_ref` in 17 places; 36
    test names and ids with `phase5` or `phase6`.
15. **Tests:**
    - five near-twin project builders remain;
    - mutation survivors worth a test: `_exclusion` residual helpers and
      budget comparisons, `_components` adjusted R² and RMSEE denominators,
      `wavelength_tables` header parsing, `metrics._validated_pair`;
    - coverage holes: `_inspector` 63 %, `_canvas` 73 %, `spc_io` 72 %;
    - slow: `test_closing_a_project_viewer_exits_without_a_crash` 9.6 s;
    - about 1000 references to private widget attributes in the viewer tests
      (a design change, not a cleanup).

## Needs a real machine

16. **Windows novice walkthrough:** one-line install, Start menu, taskbar pin
    and icon, update, uninstall.
17. **Windows startup measurement:** cold and warm, with the `Startup timing`
    log lines; no result is recorded.
18. **Rare flakiness, not reproduced since:** a hang in garbage collection, a
    hang of the two-window tests, a Qt `commitData` warning when editing the
    ROI table, a stall of the full run. A fault in any later CI run reopens
    `G37`.
