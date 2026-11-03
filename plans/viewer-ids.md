# Plan: ids only in the viewer, names as labels

Status: phases 0–4 done (2026-11-03)

## Goal

Stable ids are the only identity of project objects. The viewer (the
`enginir-viewer` tool and the `enginir.viewer` module) selects, links, stores
and exports objects by id only. Names are display labels and may repeat.
Addressing objects by name is a convenience of the library facade
(`Project`, `Dataset`, `Model`) for scripts.

Trigger: the user's scans are all called `stream_hsi.h5` in different
folders. Grids on two of them give the same ROI names, and the id-based
`Project.save_rois` refuses them although the ROIs have different ids.

## What the contracts already say

- `project-schema-contract_v3.md` §2–3: every object has a stable id;
  references inside the project use ids; names are mutable and
  non-authoritative.
- `roi-target-contract_v3.md`: `roi_id` is the real persistent identity;
  the name is a mutable label; the UI must not imply that names are
  identity.
- `sample-set-model-run-contract_v3.md`: the sample set (`roi_ids`) and the
  model run are the reproducible record; folds depend only on positions in
  the sample set.

What contradicts the goal:

- `headless-workflow-contract_v3.md`, "Names": names unique per kind; a
  duplicate is an error.
- `roi-target-contract_v3.md`, ROI grid: "names stay unique in the
  project".

## Current state of the code

- The viewer already works by id: selections, the target combo, the ROI
  table, the fit dialog and every write go through ids, and it calls none
  of the name-resolving facade functions.
- The id-based storage layer is not name-free: `Project.save_rois` (used by
  the viewer's ROI grid) checks names against the project and within the
  batch.
- Data files are shown by file name only, so `stream_hsi.h5` repeats
  identically in the data file list, the ROI table, the Project ROIs tab,
  the fit dialog and the browser.
- Explore's **Export csv** (ROI spectra) labels rows by name only; the
  Model and Project ROIs CSVs already carry `roi_name` and `roi_id`.
- A stored model is traceable by ids, but a fit cannot be repeated on
  exactly a stored sample set: `Project.fit` always takes every ROI that has
  a value now.

## Decisions (defaults proposed; to confirm)

| Id | Question | Default |
|---|---|---|
| N1 | Do the name-based facade functions (`Dataset.add_roi`, `Dataset.add_rois`, `add_target`, `fit(name=)`) still refuse a name they create themselves? | **Yes.** A script that creates "A01" and later looks it up would otherwise break itself. Projects with repeated names (from the viewer) stay fully usable from scripts by id or object. |
| N2 | Refit exactly a stored sample set (`model.refit()` / `fit(sample_set=…)`)? | **Separate step** after this plan (phase 6), because it extends the reproducibility API rather than the identity rule. |
| N3 | Label of a data file whose file name repeats | `file name (parent folder)`, e.g. `stream_hsi.h5 (API 10% - nozzle 0.6 mm…)`; the folder is shortened in the middle when long, the full path is the tooltip. |
| N4 | Name-based facade calls (released in 1.0.x) | Kept as the script convenience and documented as not reproducible under renames; ids and objects accepted alongside. No deprecation. |

## Phases

### Phase 0: stalled check run and the uncommitted testing fixes

- The last full pytest run stalled after about 30 tests. Find the test with
  a per-test timeout and the stack dump (`faulthandler_timeout`), fix it.
  A leftover pytest process from the phase 8 fault runs was still alive and
  is now stopped; check whether it caused the stall.
- Then all four gates on the uncommitted fixes of the testing round:
  - the Add Data File preview and editable detector levels;
  - no unit for count data, and Explore signals following the shown file;
  - the clipped absorbance axis labels;
  - the new ROI grid with Options defaults.
- Commit them on request, before this plan's changes.

**Done (2026-11-03).** Two full runs with a 90 s stall dump
(`faulthandler_timeout`) passed without stalling: 611 passed, 5 skipped, in
88–91 s. Ruff, pyright and Sphinx (`-W -E`, fresh directory) are clean. The
stall did not recur, so its cause is not proven. When it happened, the viewer
and a pytest process left over from the phase 8 fault runs (itself hung on the
two-window tests) were running at the same time; both are stopped now. The
multi-window tests remain the suspect if it returns.

### Phase 1: contracts and ledger

- `headless-workflow-contract_v3.md`, "Names": names are labels and may
  repeat; addressing by name is a facade convenience for scripts; an
  ambiguous lookup is an error listing the ids; reproducible scripts use ids
  or objects; the facade's own creation by name refuses an existing name
  (N1).
- `roi-target-contract_v3.md`: drop "names stay unique" from the ROI grid.
- `viewer-workspace-ui-contract_v3.md`, new section *Identity in the
  viewer*: by id only; names are display text; repeated names are shown
  with their data file; data files with repeated file names are labelled
  with their folder (N3); every export carries the ids.
- `docs/contract_coverage.md`: new rows, and the guard test of phase 4.

**Done (2026-11-03).** Contract changes:

- `headless-workflow-contract_v3.md`, *Names and ids* (was *Names*): names
  are labels and may repeat; name addressing is the facade's convenience for
  scripts, not a reproducible reference; every name-taking function also
  takes the object or an id keyword (`roi_id=`, `target_id=`,
  `model_run_id=`), and a string is always a name; ambiguous lookups list the
  ids and never pick one; the facade refuses names it creates itself when
  they exist (N1); the 1.0 name calls stay supported (N4). Name-keyed read
  mappings (`project.rois`, ...) are an error listing the repeats when names
  repeat; `project.workspace` is keyed by id.
- `roi-target-contract_v3.md`: grid names may repeat other data files' ROIs;
  only repeats within one grid are refused. Target names are labels; the UI
  offers existing targets so a name is not repeated by accident.
- `viewer-workspace-ui-contract_v3.md`, new *Identity In The Viewer*: ids for
  every selection, link, write and export; no name-based facade calls;
  repeated names stored without asking; data files labelled
  `file name (folder)` when file names repeat (N3); the data file label next
  to repeatable ROI names; ROI ids in every ROI export; a guard test.
- Ledger: the facade row is *Partial*, a new viewer identity row is a *Gap*,
  both under the new `G21` (open); `AGENTS.md` names the register `G01`–`G21`.

### Phase 2: library

- `Project.save_roi`/`save_rois`/`save_target` (the typed, id-based layer)
  check ids and content only, no name uniqueness. Batch checks keep
  repeated ids out.
- Name uniqueness moves to the name-based creators only:
  `Dataset.add_roi`, `Dataset.add_rois`, `Project.add_target`,
  `Project.fit(name=)`.
- `set_value`, `remove_assignment`, `fit`, `Project.roi`-style lookups
  accept the object (`TrainingROI`, `TargetDefinition`) or its id next to
  the name (N4). A string is resolved as a name; ids are passed as
  `roi_id=`/`target_id=` keywords or as objects, so a name can never be
  taken for an id or the reverse.
- Tests: repeated names stored through the typed layer; lookups of a
  repeated name fail with the ids; id/object forms; the facade still
  refuses its own duplicates.

**Done (2026-11-03).**

- Lookups: `Project.roi(name)` or `roi(roi_id=)`, likewise `target(target_id=)`,
  `model(model_run_id=)` and `dataset(path)` or `dataset(dataset_id=)`. Exactly
  one must be given (`TypeError` otherwise); an unknown id is a `KeyError`; a
  string is always a name.
- `set_value` and `fit` take the objects too, checked to belong to the
  project (an ROI of another project is refused). The ids are passed through
  lookups (`project.roi(roi_id=...)`) rather than more keywords on every
  call, which keeps one way per call.
- `project.rois`, `targets`, `models` and `Dataset.rois` raise a `KeyError`
  listing the repeated names and ids instead of keeping one silently.
- `Project.save_rois` checks that ROI ids are new and distinct, not names.
  The name refusal moved to `Dataset.add_rois`, next to `Dataset.add_roi`,
  `add_target` and `fit(name=)`.
- Tests in `test_workflow.py` and `test_roi_grid.py`; six injected faults
  caught (name views keeping one, id taken as a name, names checked in the
  typed layer, grid facade without the name check, a foreign object accepted,
  an ambiguous name picking one).

### Phase 3: data file labels

- One rule, `data_file_labels(datasets) -> {dataset_id: label}` (library,
  Qt-free): the file name when unique in the project, otherwise
  `file name (folder)` (N3), shortened, full path as tooltip.
- Used by the data file list, the ROI table "Data file" column and filter,
  the Project ROIs blocks and filter, the fit dialog's per-file counts, the
  browser, the not-found and Locate messages.
- `{file}` in grid name patterns stays the file stem (a label, may repeat).

**Done (2026-11-03).**

- `enginir._workflow.data_file_labels(datasets)`: the file name when it is
  unique (case-insensitive); otherwise `file name (folders)` with as many
  parent folders as tell the repeats apart; folders over 40 characters are
  shortened in the middle unless that makes two labels equal.
- `ProjectState.data_file_labels` (recomputed on every sync) and
  `data_file_label(dataset_id)`; used by the data file list (full path as
  tooltip), the ROI table column and filter (path tooltip), the Project ROIs
  blocks and filter, the fit dialog's per-file counts, the browser
  (`label [id]`, path tooltip), the Add Data File "same references as" list,
  the status strip and the replace-calibration question.
- Tests in `test_data_file_labels.py`; six injected faults caught.

### Phase 4: viewer by ids only

- Grid dialog: remove the "already used" refusal added on 2026-11-03;
  names may repeat. A note says how many names repeat existing ones.
- Explore **Export csv**: add the `roi_id` (and the data file label) to the
  rows.
- Guard test, like `test_viewer_writes_projects_only_through_project`: an
  AST scan of `enginir.viewer` fails on calls of the name-resolving facade
  (`Project.roi`, `.target`, `.model`, `.dataset`, `set_value`, `fit`,
  `add_target`, `Dataset.add_roi`, `Dataset.add_rois`, `Model.predict`).
- Check every place that shows ROI names where two ROIs may share a name
  (ROI table, mosaic tiles, browser, Model plot hover, inspector): the data
  file is shown next to it, or selection is by id anyway.
- Tests: a project with two `stream_hsi.h5` trays and the same grid names:
  both grids stored, table and tiles distinguishable, fit uses both,
  selection links the right ROI, exports carry distinct ids.

**Done (2026-11-03).**

- The grid dialog allows names that repeat ROIs of the project and says how
  many repeat; only repeats within one grid are refused (`roi_grid`).
- Explore's **Export csv**: `sample_id` holds the `roi_id`, with `roi_name`
  and `data_file` text columns. `save_csv(columns=)` writes text columns and
  `load_csv` returns them in `meta["sample_columns"]` instead of failing.
- Inspector: data file label for ROIs, `name [id]` for assignments; browser
  ROIs carry the data file label as tooltip; browser datasets show
  `label [id]` with the path as tooltip.
- Guard test `test_the_viewer_never_addresses_objects_by_name`: an AST scan
  of `enginir.viewer` for name-based functions and views on the library
  `Project` (`ProjectState.project`, `_require_project(...)`, a local
  `project`). A first, broader version flagged `_eda.py` where the local
  `project` was the `ProjectState` (id-keyed, so no bug); the local is now
  `project_state`, so `project` always means the library object.
- End-to-end test with two `stream_hsi.h5` trays and the same grid names;
  five injected faults caught (export by name, repeats refused again,
  `load_csv` without text columns, a name lookup and a name-keyed view in the
  viewer).
- Gap `G21` closed.

### Phase 5: docs

- User guide: names are labels; scripts that must be reproducible use ids
  or objects; the data file label rule.
- `workflows.rst` example with ids; API reference for the new keywords and
  `data_file_labels`; `viewer_architecture.md` identity rule; CHANGELOG
  (Changed: names may repeat; Added: id/object forms, data file labels).

### Phase 6 (separate, N2): refit a stored sample set

- `Model.refit()` and `Project.fit(sample_set=…)` fit exactly the stored
  `roi_ids` with the stored recipe and signal domain, and report whether the
  metrics match the stored run.

## Risks

- **Scripts relying on unique names** keep working: the facade still
  refuses its own duplicates, and lookups of a repeated name fail loudly
  with the ids instead of picking one.
- **Existing projects** stay valid: nothing in the file format changes.
- **Ambiguity between names and ids** in one string argument is avoided by
  keyword arguments for ids (phase 2).
