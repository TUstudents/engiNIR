# Test review (first pass, 2026-11-20)

Status: findings only; nothing deleted or changed yet.

## Method

- 830 test functions (1190 passed, 3 skipped with parametrization).
- Branch coverage of `src/enginir`: 87.4 %. By package: io 90.9, modeling 95.2,
  models 97.0, prediction 95.7, preprocessing 96.9, project 95.5, validation 91.3,
  viewer 87.5, visualization 97.0.
- Per-test coverage contexts: 399 of 804 test functions cover no line that
  another test does not also cover. This is a candidate list, not a verdict: tests
  that pin different values on shared lines (contract rules, edge cases) appear in
  it too (e.g. `test_the_rules_choose_as_the_contract_says`).
- Mutation sample (flip comparison / arithmetic / and-or / constant, run the
  module's own test file): metrics 29/36 killed, _checks 30/35, scatter 35/40,
  wavelength_tables 27/40, _components 22/40, _exclusion 8/25. The last two
  are probably overstated, since other files (model runs, predictions) also
  exercise them; rerun with the wider test set before acting.
- `vulture`: production members with no caller in `src`.

## Findings

1. The suite is mostly behaviour tests with independent oracles (textbook PLS1,
   scikit-learn, hand-computed metrics, WCAG contrast, stored files read back).
   The "half is theatre" hypothesis is not supported by the files read so far.
2. Dead production code (no caller in `src`, tests only or nothing):
   `has_pixel_interval` (prediction/_uncertainty), `icon_names` (viewer/_icons),
   `set_user_limits` and `_array_payload_fingerprint` (viewer/_prediction),
   `set_shared` (viewer/_data_files), `roi_dark_white_reference_spectra`
   (viewer/_state). Test-only hooks living in production modules: `row_checks`,
   `passed_folded`, `fix_button`, `grid_preview_count`, `check_text`, `use_file`,
   `set_choice`, `set_levels`, `set_unit`, `title_elided`, `loading_snapshot`.
3. Stale names: `test_dataset_ref_rejects_inconsistent_band_metadata`
   (DatasetRef is DatasetRecord since 1.7.3).
4. Mutation survivors worth a test or a deletion: `_exclusion` residual helpers
   (lines 36-49) and budget/limit comparisons; `_components` adjusted-R2 and
   RMSEE denominators; `wavelength_tables` header parsing and the validity
   conditions on line 112; `metrics` `_validated_pair` boolean conditions.
5. Coverage holes: viewer subprocess code (`_warmup` 0 %, `_windows_shell`
   24 %, `__main__` 40 %, `_startup_clock` 46 %) is exercised only in child
   processes, so coverage cannot see it; `_inspector` 63 %, `_canvas` 73 %,
   `spc_io` 72 %.
6. Slow tests: `test_closing_a_project_viewer_exits_without_a_crash` 9.6 s;
   `test_bands` proposals 4 s each.

## Core packages (read, second pass)

Read in full or by outline: io, preprocessing, models, validation, modeling
(exclusion, components, fit spec, bands, kernel PLS), project (types, file),
prediction (runs, diagnostics, ROIs), workflow, guidance, wavelength tables,
format migration. Verdict: keep almost all; they check against textbooks,
scikit-learn, SIMCA numbers, stored files read back, and the contracts.

Candidates (all small):

- Delete or rewrite, no behaviour pinned:
  - `test_sample_set_semantics_are_fixed` (test_model_runs): restates the
    `SAMPLE_SET_SEMANTICS` dict.
  - `test_project_state_fit_model_run_uses_roi_local_reference_snapshots`
    (76 lines): its name promises ROI-local reference snapshots, its only
    assertion is the model's id. Assert that the fit used the ROI's snapshot
    (or drop it; `test_reference_spectra_calibrate_and_travel_with_each_roi`
    in test_workflow covers part of it). Fault-inject first.
  - `test_regression_wrappers_are_sklearn_regressors`,
    `test_random_forest_importances_cover_every_wavelength` (test_models):
    shape and type bookkeeping.
  - `test_all_format_errors_share_one_value_error_base` (test_io): class
    hierarchy only; keep one line in a loader test if the contract says
    callers may catch ValueError.
- Move: `test_a_fit_on_statistics_is_fast_and_reads_the_pixels_a_few_times`
  (test_kernel_pls) is a wall-clock benchmark with a 30 s bound in the unit
  suite; AGENTS.md puts timing in `benchmarks/` (`ENGINIR_PERF=1`).
- Implementation-coupled (keep the guarded behaviour, change the method):
  - `test_hdf5_prediction_reuses_source_and_artifact_file_handles` counts
    exactly five `h5py.File` opens of the project; any extra legitimate open
    fails it. Assert a bound that scales with chunks instead.
  - `test_fit_model_run_preprocesses_stateless_recipe_once_before_cv`
    monkeypatches two internals; the behaviour (once per pixel, before the
    folds) can be seen from a counting recipe step.
- Layering: test_model_runs, test_prediction_runs, test_prediction_diagnostics
  and test_project_file test the core services through the viewer's
  `ProjectState` and the `qapp` fixture (31 uses in test_prediction_runs). The
  rules are Qt-free; these tests slow down and tie core behaviour to the
  viewer. Port them to `Project` / the service functions where `ProjectState`
  adds nothing.
- Duplication: the contract checks of test_project_types (roughly 700 lines of
  rejections) and the writer checks of test_project_file overlap
  (`test_writers_refuse_payloads_that_hydration_would_reject`,
  `test_child_object_writers_validate_too`). Merge by rule, not by file.
- Stale names: `test_dataset_ref_rejects_...`, the `save_dataset_ref` writer,
  "phase5/phase6" file names and ids in test_model_runs and
  test_prediction_runs.
- Real gaps to test: the five groups of mutation survivors above.

## Viewer tests (read, third pass)

Read by outline (all 20 files) and in full for the suspicious ones. They are
behaviour tests of what a user sees and does (menus follow buttons, a fix
starts drawing, a write refreshes the other window, a mosaic label is not
cut). Many were written against a real bug (WSLg width, Windows font,
size-dependent padding) and carry the story in their docstring.

Candidates:

- Constants and layout pinned, no behaviour:
  - `test_the_shell_builds_its_workspaces_and_docks_of_fixed_width`
    (test_window, 70 lines, 36 asserts): widths 260 and 360, dock features,
    splitter orientation. Restates the layout code; any redesign breaks it.
    Keep the few that guard a bug (fixed widths per screen class), drop the rest.
  - `test_constructor_sets_midpoints_and_default_units` (test_viewer_state):
    default attribute values.
- Mirror tests: `test_current_frame_image_default_levels_and_absorbance`
  (test_viewer_state) recomputes `(raw - black) / (white - black)` and
  `-log10` from the same formulas the code uses; compare with a known value
  (a pixel at half scale gives 0.5, absorbance 0.301) instead.
- No assertion: `test_run_is_noop_in_jupyter` (test_viewer_app) passes unless
  it crashes; assert that no application started.
- Weak assertion: `test_about_box_has_no_log_path` asserts `"Log" not in text`;
  the later about-box test already checks the content.
- Mock-heavy: `test_launcher_returns_expected_exit_codes` and the nine-patch
  `test_recommended_qt_platform_linux_wayland_cases` (the second is a proper
  table test: keep). The launcher test patches the open call it is meant to
  exercise; a failed open returning 1 is worth one real-path test.
- Reach into private state: test_welcome_window sets `_dark_path` and
  `_white_path` directly instead of using the browse buttons.
- Coupling: about 1000 references to private widget attributes (`window._x`)
  across the viewer tests (test_window alone has 308). Renaming a widget breaks
  tests in several files. A small test API on the viewer (what the user can
  press and read) would cut that, but it is a design change, not a cleanup.
- Pixel/font layout tests (`test_every_control_row_fits_in_the_controls_dock`,
  `test_mosaic_labels_are_measured_in_the_actual_font`, the Checks columns)
  have each broken on Windows once and skip on some platforms. They guard real
  bugs; keep, but mark their fragility and run them in the Windows job only if
  they stay flaky.
- Legacy path: 17 tests build `HSIViewer(cube, wavelengths)` directly (a
  viewer without a project); `_open.py` builds it too, so it is live.
- Slow: `test_closing_a_project_viewer_exits_without_a_crash` 9.6 s.

## Done: dead code (2026-11-20)

Removed (no caller in `src`, tests, docs): `has_pixel_interval`, `icon_names`,
`set_user_limits`, `_array_payload_fingerprint`, `set_shared`,
`roi_dark_white_reference_spectra` and `roi_reference_spectra` (its only caller),
both `service_status_snapshot`, `ProjectState.get_target_assignment` (tests use
`target_assignments[...]`; `docs/api/viewer.rst` updated). Four gates green.

Kept on purpose: the one-line accessors the viewer tests read through
(`row_checks`, `fix_button`, `badge_text`, `set_choice`, `use_frames`, `tile_count`,
...). They are the tests' only way to see widget state without private
attributes; removing them would push tests onto `_x` attributes.

Done (port): `tests/_sessions.py` (`CoreSession`, `fit_stored_model`) runs fits and
predictions through the library services without Qt; test_model_runs,
test_prediction_runs and test_prediction_diagnostics use it (no `qapp`, 243 tests
in 12 s), viewer tests fit through `fit_stored_model`. Removed from
`ProjectState`: `build_sample_set`, `fit_model_run`, `predict_dataset`,
`ensure_model_run_artifact_refs` and two helpers they alone used.

Was open: `ProjectState.predict_dataset` (40 lines of workflow rule, used only by
tests), `ensure_model_run_artifact_refs` and `raw_roi_spectra` (tests only).
They go with the port of the core tests off `ProjectState`.

## Done: weak tests (2026-11-20)

Rewritten, each fault-injected where a fault was easy to inject:
- the ROI-snapshot fit test now asserts the model's centre from hand-computed
  absorbances (a reversed snapshot fails it);
- the handle-count test compares opens for 1 and 3 chunks instead of the
  literal 5 (an extra open per chunk fails it);
- the state level test uses raw counts at black, halfway and white (log base
  change fails it); the launcher test goes through the real opening code (an
  exit code of 0 on error fails it); the Jupyter test asserts no app starts;
  the welcome test drives the Browse buttons; the about test compares with
  `about_text`; the shell test keeps the dock-width behaviour only; wrapper
  and importance tests assert that the models learn.
Deleted: `test_sample_set_semantics_are_fixed` (fields asserted in the
build-sample-set test), `test_constructor_sets_midpoints_and_default_units`.
Moved: the kernel PLS timing test to `benchmarks/test_kernel_pls_performance.py`
(`kernel_project` now in `tests/_scans.py`). Renamed the stale dataset-ref test.
Kept on purpose: the format-error hierarchy test (callers catch ValueError),
the preprocess-once test (documents plans/kernel-pls.md D7).

## Done: duplicates (2026-11-20)

The overlap was smaller than estimated: by message, 4 immutability tests and 2
pairs of rejection tests. Merged `test_a_stored_reference_correction_is_never_changed`,
`test_stable_object_ids_are_immutable_except_labels` and
`test_resaving_unchanged_objects_is_a_no_op` into
`test_stored_objects_never_change_except_their_labels` (disabling the overwrite check
fails it); dropped the overwrite tail of the atomic-write test; removed
`test_hydration_rejects_dangling_or_foreign_active_calibration_id` (the validator test
`test_validate_project_relationships_checks_active_calibration_ids` fails when the
ownership check is skipped) and `test_a_project_of_an_earlier_format_is_refused_with_one_message`
(covered by the "refused as a project" test). `contracts/coverage.md` cites the new
names; every test name cited in contracts and docs exists. The rejection tests of
test_project_types and the writer tests of test_project_file check different
objects and fields: kept.

## Done: viewer consolidation (2026-11-20)

Read test_window and test_explore in full and the rest by outline plus a
similarity scan of all test bodies (pairs above 0.7). Viewer tests overlap little;
the duplication was in setup code:
- test_window: the 170-line `_create_reopenable_model_project` is gone
  (`write_model_project` serves); its page assertions went into the
  navigation test; the Go-to-ROI click is no longer repeated; the canvas test
  moved to test_canvas. 1411 to 1126 lines.
- test_explore: dropped the fake-canvas button test (real drawing is tested
  in test_canvas and test_multi_file_workflow), the delete test the
  sample-set one contains, and folded the cancel test into the replace test.
- test_project_open: two identical-flow tests became one parametrized test.
- New `boxes` fixture (tests/conftest.py) records message-box texts; 11 hand-made
  monkeypatches in 7 files use it.
Not done: five near-twin low-level project builders remain
(`test_model_runs._create_model_project`, `_create_phase5_project`,
`test_project_open._save_project_roi`, `test_project_file._seed_dataset_context`,
`write_model_project`); merging them changes ids and values the tests assert.

## Done: count reduction (2026-11-20)

1190 passed (1179 after the first consolidation) -> 689 passed. 830 -> 573 test functions.
How: (1) table-style parametrizations became one test each (`tests/_tables.check_all`):
the 93-case core-field test and 20 others, about 150 cases. (2) Per-test coverage
contexts, then a greedy cover: 349 of 794 tests add no line beyond a minimal set.
(3) A mutation run of 900 random mutants over all of src, each run against the kept
tests first and then the redundant ones: 580 killed by the kept set, 33 only by
redundant tests (23 distinct tests, kept), 286 survive both. (4) Not deleted: static and
subprocess tests the line coverage cannot see (terminology, theme, compression,
startup, launcher, logging), tests with an independent oracle (scikit-learn, SciPy,
SIMCA, textbook), tests cited by a contract, tests fault-injected earlier. 249 deleted.
Check: mutation sample on six modules, same seed as before: metrics, _checks,
wavelength_tables and _exclusion kill as many mutants as before; _components and
scatter lost 8 and 4 mutants, found with the pre-deletion tests; four deleted tests
that killed them were restored (delta-method standard error, guard fields, training
statistics, SNV). Line coverage 87 % as before. Lesson: line coverage cannot see
value-pinning tests; the 900-mutant sample missed these four, so more of this kind
may be gone.

## PR review (agent, 2026-11-20)

A read-only review of PR 22 found 17 deleted tests that pinned frozen rules or values
no other test pins (ROI storage dtype and uint16 range, float32 working arrays,
per-ROI primary metrics, grouped cross-validation, training vs prediction
conversion, the bundled Kestreleye table and its id, the loaders' positive paths,
chunk preflight, enum case); all restored (708 passed). `check_all` now fails on an
empty table and lets pytest skips through; two stale doc references fixed.
`CoreSession` copies the library glue of the removed ProjectState methods, so
`finalize_model_run_result` is exercised only through the viewer tests.

## Next

Review the remaining modules by reading; rerun mutation with the full related
test set; then delete the dead code with its tests in one reviewable change.
