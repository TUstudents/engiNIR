# Plan: the ROI workflow and the domain model doc (backlog items 2, 4, 5, 8)

Status: decided (user, 2026-11-19): R1 to R4 below. Not started; release
not decided (1.7.3 is implemented on `release-1.7.3` and not released yet).

## Decisions (user, 2026-11-19)

**R1. `{name}`: grid ROIs are named after the dataset's name (item 4).** A new
placeholder `{name}` gives the dataset's name (Project > Name Dataset,
`DatasetRef.label`), or its file name when it has none. The default name
patterns use `{name}` instead of `{file}`; `{file}` and `{folder}` stay.
Why: Kestreleye files are all named `stream_hsi.h5`, so `{file}` named every
ROI `stream_hsi-A01`.

**R2. The placeholders are listed in the grid dialog (item 2).** A line under
the name pattern lists `{name}`, `{file}`, `{folder}`, `{row}`, `{column}` and
`{n}` (today only a tooltip). `{folder}` is the data file's folder name, in
full (for a Kestreleye scan e.g. `API 10% - nozzle 0.6 mm_lay-10_API-5_
20261104_154120`). No live example.

**R3. Add ROI and Add ROI Grid reachable from anywhere (item 8).**
- Pressing either without a shown image (button, ROI menu, the check's fix)
  switches to the Data file tab, shows the last shown dataset (or asks which
  dataset when there is none), then starts drawing or opens the grid dialog.
- The Project ROIs tab gets Add ROI and Add ROI Grid buttons for each
  dataset group: they show that dataset and start the action.

**R4. `docs/workspace_domain_model.md` becomes a short overview (item 5).**
Each object (dataset, reference correction, ROI, target and value, recipe,
input contract, calibration set, model, prediction) with its term of the
terminology contract, its Python class and its HDF5 group, linking the
contracts and `docs/hdf5_format.md` for the detail. The outdated parts go
(the format name, the ROI fields, the removed `spatial_map`, the internal
names without their terms).

## Where it is built today

- Grid names: `viewer/_grid_settings.NAME_PATTERNS` (`{file}-{row}{column:02d}`,
  `{file}-{n:03d}`, ...), `GridDefaults.name_pattern`; the tooltip in
  `viewer/_roi_grid.py`; the names are formatted by `Project.roi_grid` /
  `roi_grid` in `_workflow.py` (`file=` the file stem); the grid dialog gets
  `file=pathlib.Path(cube_path).stem` from `EDAView.open_roi_grid_dialog`.
  Stored grid defaults with `{file}` stay valid.
- Add ROI / Add ROI Grid: `EDAView._btn_roi_training` (checkable, draws on the
  Spatial image) and `_btn_roi_grid`, enabled only with a shown dataset
  (`refresh_attachment_state`); the ROI menu presses them
  (`HSIViewer._menu_button_action`); the check's fix `add_roi` only shows the
  Data step (`HSIViewer.run_fix`). A project with ROIs opens on the Project
  ROIs tab (`_choose_explore_tab`); the last shown dataset is in the user
  settings (`_project_settings`).
- The Project ROIs tab groups its tiles by dataset (`viewer/_project_rois.py`,
  `RoiMosaic`).

## Phases

1. `{name}` and `{folder}` in `roi_grid` and the grid dialog; the defaults use
   `{name}`; the listed placeholders. Tests: names from a dataset with and
   without a name, `{folder}`, an old pattern with `{file}`.
2. Add ROI / Add ROI Grid from anywhere: one viewer action each that shows a
   dataset first (the last one, or a choice) and then starts; the buttons,
   the ROI menu and the fix use it; buttons on the Project ROIs groups.
   Tests: from the Project ROIs tab both start on the right dataset; the
   fix starts drawing; fault-injected.
3. The domain model overview (R4); the viewer guide for 1 and 2.

## Out of scope

A live example of the name in the grid dialog (R2: list only).
