# Public viewer widgets for inline prediction (1.8.1)

Status: built 2026-11-24 for 1.8.1 (branch `print-widgets`). Piece 4 of `plans/print-support.md`, from the
engiNIR-PRINT request of 2026-11-24, narrowed by the user's decisions below.

## Decisions (2026-11-24)

- **Grids stay where they are.** engiNIR keeps `roi_grid`, `GridCell` and
  `Prediction.roi_statistics(layout)` / `roi_prediction(layout)` as in 1.8.0.
  No grid object, grid id, `roi_grids()` listing or project format change.
- **A production layout is values in the consumer's recipe**, not a reference
  into the project. engiNIR-PRINT has its own grid tool, stores the
  `roi_grid` parameters in its job recipe, and shifts a layout by changing
  `origin`. No shift function in engiNIR, and no reading back of a dataset's
  grid layout.
- **Several grids per dataset stay allowed** (as today).
- **One release**, 1.8.1.
- Not in 1.8.1: automatic grid offsets (image recognition), exporting
  `Model` and `Prediction` by name, opening scans without HDF5 file locking
  (only if a test on the Kestreleye PC shows a problem).

## What 1.8.1 adds

All additions; nothing existing changes its behavior.

### 1. `enginir.false_color_image` (core, no Qt)

```python
false_color_image(path, *, bands=None, binning=1) -> np.ndarray
# float32 in [0, 1], (rows, frames, 3): what FalseColorCanvas.set_image takes
```

- The same image as the viewer's dataset view: bands nearest 1650/1300/1050 nm,
  2-98 % stretch (`stretch_rgb`, shared with `ViewerState.raw_false_colour_rgb`).
  `bands=` (three band indices) for band-number files; without either it
  refuses.
- Read in blocks of 256 frames, keeping three bands; every frame, so the
  image keeps the scan's pixels and cells need no conversion. Measured on a
  2100 x 312 x 176 uint16 Blosc stream: 0.17 s (1.8 s with the first
  imports), so no band-selective reader and no frame subsampling
  (the request's `max_frames` and uint8 output dropped).
- Opens the file read-only (`"r"`), like every reader today.

### 2. Theme (`enginir.viewer`)

- `apply_theme(app, *, dark=None)`: `None` uses the user's saved preference
  (`load_preference`). Unlike `apply_qt_theme`, an explicit `dark` from a
  consumer does not overwrite the viewer user's preference.
- `theme_tokens()`, `value_colormap()`, `heat_colormap()`: public names over
  `_theme`.

### 3. `PredictionMapWidget` (`enginir.viewer`)

```python
widget.show_maps(prediction, *, dmodx=None, t2=None, unit="",
                 dcrit=None, t2crit=None, color_range=None)
widget.clear(empty_reason="")
widget.hovered        # existing signal (str)
```

- `show_maps` builds the internal payload (`spatial_map`, `layers`,
  `diagnostics` with `dcrit_095` / `t2_crit_095`, `target_unit`, preview step)
  from plain arrays, so a caller without a `Prediction` object (one reading
  its own records) can show a layer. The layer selector and the limit overlay
  work as now; the interval layer shows "not available".
- `colour_range` (added): one color scale for the maps of several scans.
  `target` (requested) dropped: the widget shows the unit only.
- `header_widgets()` (layer selector, pan, zoom, reset) is public: an
  application places it above the map. `update_data(payload)` stays internal.

### 4. `FalseColorCanvas` (`enginir.viewer`)

Public: `set_image`, `reset_view`, and new

```python
canvas.show_cells(cells, *, colors=None, labels=True)   # GridCell outlines
canvas.clear_cells()
canvas.cell_at(frame, row)   # the cell name at a pixel, "" for none
canvas.cell_hovered   # signal(str): cell name under the cursor, "" when none
```

- `show_cells` generalizes `set_grid_preview`; the grid dialog's preview
  becomes a caller of it (planned cells, skipped cells dashed, outside cells
  red), so there is one drawing path.
- `colors` maps cell names to colors (default: the theme's series color).
- The ROI drawing, selection and click signals stay private.

### 5. Exports, docs, contract

- `enginir.viewer.__all__` and `_LAZY` gain `apply_theme`, `theme_tokens`,
  `value_colormap`, `heat_colormap`, `PredictionMapWidget`,
  `FalseColorCanvas`; `enginir.__all__` gains `false_color_image`.
  `import enginir` stays Qt-free.
- `docs/api/viewer.rst` (a section per name), `docs/api/` for
  `false_color_image`, `docs/user_guide/viewer.rst` (a short "embedding the
  viewer's widgets" part), `docs/viewer_architecture.md`.
- Viewer contract (`contracts/viewer-workspace-ui-contract_v3.md`): the
  public widget surface and that it follows semver; coverage row in
  `contracts/coverage.md`.
- CHANGELOG 1.8.1, version in `pyproject.toml`.

## Tests (offscreen Qt), each fault checked by injecting it

- Every new public name imports from `enginir.viewer` / `enginir`, and
  `import enginir` still loads no Qt (existing `test_headless_api.py`).
- `show_maps` with arrays shows the selected layer (Prediction, DModXPS with
  pixels above `dcrit` overlaid); `clear` empties it.
- `false_color_image` of a band-number HDF5 stream: shape, uint8, equals the
  viewer's image of the same file; refuses without wavelengths or `bands`;
  opens the file with mode `"r"`.
- `show_cells` draws one outline per cell; `cell_hovered` names the cell under
  a point and `""` outside; the grid dialog's preview is unchanged.
- No tests of colors, sizes or texts.

## Open

- Version and CHANGELOG set for 1.8.1; commit, tag, push and the GitHub release
  when the user asks.
- The viewer on Windows: the grid dialog's preview now draws through
  `_draw_cell`; check it looks unchanged.
