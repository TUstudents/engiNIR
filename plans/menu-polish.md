# Plan: menu polish

Status: decided (user, 2026-11-22). Phases 1 to 3 built on branch
`menu-polish`; phases 4 to 6 not built.

## View (decided)

- **V1. Title Case** for every item: "View Settings", "Spectral Profile",
  "Sensor Frame", "Spatial Profile", "ROI Table", "Histogram", ...
- **V2. Band image is removed** from the menu and from the code: it shows no
  more than the Spatial image. Histogram stays as it is (enabled on the Data
  file page only).
- **V3. View > Go To** submenu: Data, ROIs, Values, Model, Predict, with
  Ctrl+1 to Ctrl+5, the same as the workflow strip.
- **V4. "Reset View"** for the menu item and the image's button.

## File (decided)

- **F1. Adding a data file lives in Project only:** Project > Add Dataset…;
  File loses Open Data… and Add Dataset….
- **F2. Shortcuts:** Ctrl+O Open Project, Ctrl+N New Project, Ctrl+D Add
  Dataset (Project menu).
- **F3. Open Recent is sorted:** recent projects on top, recent data files
  below (paths shown as today).
- **F4. Export ROI Spectra (CSV)… stays in both File and ROI.**
- **F5. Close Window goes; Quit (Ctrl+Q) stays.** With Open Data… gone (F1),
  the viewer has one window: a second one only came from opening a data file
  without a project. A window a script opened still closes by its title bar.

## Edit (decided)

- **E1. Copy stays** (Ctrl+C): the focused table's or text's selection.
- **E2. Edit > Paste** (Ctrl+V): pastes into the focused table, as Ctrl+V does
  in the ROI table now.
- **E3. No Undo in this plan:** it needs its own plan (every stored write
  needs a reverse write).
- **E4. Find ROI… moves to the ROI menu** (Ctrl+F kept).

## Project and ROI (decided)

- **P1. Project > Target Values… and ROI > ROI Table both stay**, both with
  Ctrl+T (they open the same ROI table).
- **P2. "Rename Dataset…"** replaces "Name Dataset…" (same dialog).
- **R1. ROI Grid Defaults… moves from Tools to the ROI menu**, under Add ROI
  Grid….
- **R2. Delete ROI gets the Delete key** (it still asks first).
- **P3. Project > Remove Dataset…** removes a dataset that has no ROIs, with
  its reference corrections; the data file on disk is never touched. A dataset
  with ROIs is refused, naming them, so no model's record loses data. It is the
  way to add a file again with other references or another binning (the
  binning plan's D2 still holds: a dataset's binning never changes). A
  project-schema contract change: removing a stored object is new.
- With E4: Find ROI… (Ctrl+F) in the ROI menu.
- Found while reading: the "No project open" warning of Target Values says
  "Use Project > New or Project > Open first"; New and Open are in File.

## Model, Predict, Help (decided)

- **M1. "Load Model Settings" keeps its name.**
- **M2. Model > Rename Model… and Delete Model…** (Delete: see the decisions
  below; it is new, stored objects are never removed today).
- **PR1. Predict Map… gets F7** (F5 Fit Model, F6 Predict ROIs).
- **H1. Help gets User Guide** (the online documentation of this version),
  **Report a Problem** (GitHub issues, with a note to attach the log file) and
  **Keyboard Shortcuts** (the list of every menu shortcut).

## Tools and logs (decided)

- **L1. Logs live in Help:** Help > View Log and Log File… (copy the path or
  open its folder); Tools loses Open Log File; View keeps the Log toggle like
  every panel.
- **T1. Preferences… moves to Edit (Ctrl+,) and the Tools menu goes**
  (ROI Grid Defaults went to ROI, R1).

## Retire Model (decided, replaces Delete Model)

- **D1. Model > Retire Model…** instead of Delete Model (direction ICH Q14: a
  model is a record, never removed). Its calibration set, input contract and
  recipe stay (nothing shared can break); its stored predictions stay with it.
- **D2. Retiring drops the model's per-pixel diagnostics** (scores, T² and
  DModX of every calibration pixel, about 95 % of a model's size) and keeps the
  predictive core (coefficients, rotations, loadings, scaling, limits) and the
  metrics. A contract exception: retiring is the one change of a stored model's
  content.
- M2 is now Rename Model… and Retire Model…; the earlier "delete its stored
  predictions too" no longer applies.
- **D3. A retired model stays retired:** no reactivation; Refit gives the same
  model again (same calibration set, same metrics) with its diagnostics, as a
  new model.
- **D4. The reason is optional;** stored with the date and the engiNIR version.
- **D5. Retired models appear in the Project tree under "Retired models"**,
  with their reason and metrics in Properties; not in the Models panel or in
  Predict's choices.
- Lineage (`derived_from` for refits and Fit Again) and an audit trail get their
  own plan.

## The menu after the decisions

| Menu | Items (shortcut) |
|---|---|
| File | New Project (Ctrl+N), Open Project (Ctrl+O), Open Example Project, Open Recent ▸ (projects, then data files), — Export ROI Spectra (CSV)…, Export Plots…, — Quit (Ctrl+Q) |
| Edit | Copy (Ctrl+C), Paste (Ctrl+V), — Preferences… (Ctrl+,) |
| View | Project, Properties, View Settings, Log, Histogram, ROI Table, Spectral Profile, Sensor Frame, Spatial Profile, — Go To ▸ Data … Predict (Ctrl+1…5), — Theme ▸, — Reset View, Reset Layout |
| Project | Add Dataset… (Ctrl+D), Rename Dataset…, Set Wavelengths…, Locate Moved Datasets…, Remove Dataset…, — Targets…, Target Values… (Ctrl+T) |
| ROI | Add ROI (Ctrl+R), Add ROI Grid… (Ctrl+G), ROI Grid Defaults…, Delete ROI (Del), Find ROI… (Ctrl+F), — ROI Table (Ctrl+T), Export ROI Spectra (CSV)… |
| Model | Fit Model (F5), Load Model Settings, Propose Bands…, Select Bands on Plot…, — Rename Model…, Retire Model… |
| Predict | Predict ROIs… (F6), Predict Map… (F7), Cancel Prediction |
| Help | User Guide, Keyboard Shortcuts, Report a Problem, — View Log, Log File…, — About engiNIR |

## Build plan

### Phase 1: the menu itself (viewer only)

Where: `HSIViewer` menu construction (`viewer/_viewer.py`, the File menu near
`_open_data`, the Edit/Project/ROI/Model/Predict menus in the action set-up,
the Tools and Help menus), `viewer/_recent.py` (entries already carry a kind).

- F1, F2, F5: File without Open Data…, Add Dataset… and Close Window; Ctrl+O
  Open Project; Ctrl+D Add Dataset (Project). `_open_data` and
  `_open_data_path` lose their menu; the welcome window keeps its own
  Open Dataset.
- F3: Open Recent lists project entries first, then data files, each group by
  recency.
- E1, E2: Edit has Copy and a new Paste (the focused widget's `paste`, or the
  ROI table's `paste_clipboard`); E4 moves Find ROI… to ROI; T1 moves
  Preferences… to Edit and removes the Tools menu; R1 moves ROI Grid
  Defaults… to ROI; L1 removes Tools > Open Log File (Help keeps View Log and
  Log File…).
- V1: Title Case for every View item (the dock titles stay sentence case,
  as the terminology contract says for titles). V3: View > Go To with Ctrl+1…5
  calling `show_step`. V4: "Reset View" in the menu.
- P1: **one** QAction for Target Values… and ROI Table, added to both menus.
  Two actions with the same Ctrl+T would be an ambiguous shortcut in Qt, and
  neither would fire.
- P2: "Rename Dataset…" (the `name_dataset` dialog, its title too).
- R2: Delete ROI on the Delete key, with `Qt.ShortcutContext` so that Delete
  in the ROI table's cells still clears a value instead of deleting an ROI.
- PR1: Predict Map… on F7.
- The "No project open" warning of Target Values says "File > New Project or
  File > Open Project".
- Test (behavior, not layout): every menu shortcut is unique (an ambiguous one
  never fires), and each moved action still reaches its target (extend
  `test_every_workspace_action_is_a_menu_item_and_follows_its_button`).

### Phase 2: Band image removed (V2)

`DC3Panel` (`viewer/_panels.py`), its dock (`_dc3_dock`, "raw" in
`_data_file_docks`), its View toggle, its plot frame and export id
(`explore.raw`), the layout settings that name it (a stored layout with it
must still load), its tests (`test_image_panels_pick_the_clicked_pixel...`,
the `explore.raw` table of `test_every_explore_plot_exports_what_it_shows`)
and its mentions in `docs/user_guide/viewer.rst` and
`docs/viewer_architecture.md`. Check first that nothing else uses its
frame and row picks.

### Phase 3: Help (H1)

- User Guide: opens `https://enginir.readthedocs.io/en/stable/` (one URL
  constant next to `GITHUB_URL`).
- Report a Problem: opens `GITHUB_URL + "/issues/new"` after a short note
  naming the log file to attach (Help > Log File… copies its path).
- Keyboard Shortcuts: a dialog listing every menu action with a shortcut,
  read from the menus (so it never goes stale).

### Phase 4: Rename Model and Remove Dataset (library first)

- `Model.rename(name)`: the name is already a mutable field of the model
  writer; refuses an empty or taken name, like other names. Menu: Rename
  Model… for the selected model.
- `Project.remove_dataset(dataset_id)` and a store function that deletes
  `/datasets/{id}` with its reference corrections in one write. Refused when
  the dataset has ROIs (named), or a calibration set or stored prediction
  names it. The data file is never touched. Menu: Remove Dataset… with a
  confirmation naming the file.
- Contracts: project-schema §5 (removal of a dataset without ROIs is the
  first removal of a stored object), the attachment contract.
- Tests: refusals with ROIs, with a calibration set, with a stored
  prediction; a removed file can be added again with another binning;
  nothing written when refused (fault-injected).

### Phase 5: Retire Model (D1 to D5)

- `ModelRun` gets `retired_at` and `retired_reason` (empty when active;
  stored only when retired, so active models keep their content).
- `Model.retire(reason="")` and a store function that, in one write, sets
  both fields and deletes the model's **per-pixel** arrays (every array of
  `artifact_refs` whose length is the number of calibration pixels: scores,
  T², DModX; per-band and per-ROI arrays, the prediction model and the
  metrics stay). Refuses a model that is already retired.
- A retired model: not in `Project.models`' default lists, the Models panel or
  Predict's choices; `Model.predict` and `predict_rois` refuse it; its checks
  are not run; `refit` works and gives a new active model (D3).
- Project tree: a "Retired models" section; Properties show the reason, the
  date, the engiNIR version and the metrics; the Model page shows a retired
  model's metrics and per-band plots only.
- Contracts: project-schema §5 (retiring is the one change of a stored
  model's content), sample-set-model-run, model-workspace-diagnostics.
- Tests: the file shrinks by the per-pixel arrays, coefficients and metrics
  are unchanged, a retired model predicts nothing, refit equals the original
  metrics, an active model's content is unchanged (fault-injected).

### Phase 6: docs and release notes

`docs/user_guide/viewer.rst` (the menus), `docs/api/viewer.rst`,
`docs/viewer_architecture.md`, the glossary for "retired model",
`contracts/coverage.md`, CHANGELOG.

### Order and size

Phases 1 to 3 are viewer-only and can ship together. Phases 4 and 5 change
the library and the contracts; they can go in the same release or the next.

## Built (phases 1 to 3)

- As decided, with two changes found while building:
  - The settings dock's View item is "Settings": the dock is View settings,
    Fit settings or Prediction settings depending on the step, so one fixed
    "View Settings" would be wrong on the Model and Predict pages.
  - P1: two actions with Ctrl+T would be ambiguous in Qt and neither would
    fire. ROI > ROI Table holds Ctrl+T; Project > Target Values… shows
    "Ctrl+T" after a tab (Qt's shortcut column) and opens the same table.
- Delete deletes the selected ROI only with the focus on the image or the
  mosaic (`WidgetWithChildrenShortcut`), so Delete in the ROI table's cells is
  untouched.
- Keyboard Shortcuts and Preferences > Keyboard share one table, read from
  the menus (`HSIViewer.menu_shortcuts`).
- Test: `test_every_menu_shortcut_is_unique_and_go_to_opens_each_step`
  (fails with a duplicate Ctrl+T).
