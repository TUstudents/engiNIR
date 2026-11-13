# Viewer architecture

Developer reference for the `src/enginir/viewer/` module.

---

## File inventory

| File | Responsibility |
|------|----------------|
| `__init__.py` | Public API: `open_viewer()`, `open_project()` |
| `__main__.py` | CLI entry point `enginir-viewer` |
| `_app.py` | QApplication singleton, platform choice (Wayland on WSLg, X11 on other mixed sessions), application icon, the application-wide event filter (dialog centering; on WSLg Wayland, combo popups release their native window after hiding, because WSLg keeps an unmapped popup on screen, microsoft/wslg#1522) |
| `_theme.py` | Shared Qt stylesheet (Fusion + light palette) |
| `_state.py` | `ViewerState` (UI) + `ProjectState` (Qt adapter over `enginir.Project`) |
| `_canvas.py` | `FalseColourCanvas` (pyqtgraph image surface, crosshair, ROI draw/selection) |
| `_panels.py` | Exploration dock widgets: Histogram, Spatial and Spectrum (matplotlib); RAW and Sensor Frame (pyqtgraph `_ImagePanel`) |
| `_viewer.py` | `HSIViewer(QMainWindow)` shell, menus, docks, status strip, workspace host |
| `_welcome.py` | Launcher window with recent files list and startup routing |
| `_recent.py` | MRU file list (`~/.config/enginir/recent.json`) |
| `_eda.py` | `EDAView` — signal, preprocessing, project, and ROI controls |
| `_attachment.py` | Project/dataset binding validation |
| `_modeling.py` | `ModelingView` — target, target values (ROI table), the fit dialog (components or Auto with a rule; Fitting group: pixel subsample, scaling, DModX exclusion → `FitSpec` and hyperparameters) and Fit again (dialog preset from a stored run; the preset recipe's AsLS fields and `sg_edges` are kept, as the dialog has no controls for them); `_component_payload` adds curve, loadings, CoeffCS, VIP, selectivity ratio, mean spectrum and calibration distances to the diagnostics payload; the fit dialog's Bands row (keep/exclude text, water-band menu, Select on plot, Propose) feeds the recipe and, after OK, the Explore preview (`ViewerState.bands_keep/bands_exclude`); `_roi_score_payload` passes the ROI scores the fit stored (`diagnostics/roi_scores`) and never rebuilds them from pixel counts |
| `_model_diagnostics.py` | Model workspace diagnostics surface (Spectral page: coefficients, CoeffCS, loadings, VIP, VIP sorted, selectivity ratio; lines over wavelength break at band-selection gaps) |
| `_band_ranges.py` | `BandRangeDialog` (drag keep/exclude ranges on the mean spectrum of the target's ROIs, loaded in a job, with the selected model's VIP or coefficients as a guide) and `ProposeBandsDialog` (runs `Project.propose_bands` in a job, progress through a polled counter, cancel flag) |
| `_prediction.py` | `PredictionView` (Predict ROIs and Predict map jobs, the ROI predictions of the session and their statistics for the current DModXPS exclusion, recalculated without predicting again) and `PredictionMapWidget` (color scale from the calibration range or user limits, ROI outlines, excluded pixels grayed, the Prediction interval layer of the 95 % pixel prediction intervals) |
| `_predict_dialogs.py` | `PredictRoisDialog`, `PredictMapDialog` (models by name, the chain read-only, refusal with the reason), the `predict.roi_results` plot spec |
| `_browser.py` | Read-only workspace object tree |
| `_inspector.py` | Selection-driven metadata display; `fit_summary` describes how a model was fitted (reads the stored exclusion arrays on demand) |
| `_jobs.py` | `BackgroundWorker`, `_CallbackRelay`, `start_background_job`, `BusyStatusWidget` |
| `_loading.py` | Shared wait-cursor/status/progress helpers for startup and open flows |
| `_data_helpers.py` | Dataset-shape and spectral array helpers shared by viewer workflows |
| `_project_rois.py` | `ProjectRoisView` (Explore's Project ROIs tab), `RoiMosaic` (stored ROI pixels as tiles, a `PlotSource`; `mosaic_layout` places them in a compact grid from `grid_cells` or at their scan positions, with room for text drawn in mosaic units; outlines in the value color; stretch shared or per dataset), specs `explore.roi_means`/`explore.roi_pixels` fed by `enginir.modeling.roi_spectrum_summaries`. Tiles arrive in background batches of 40 and are drawn incrementally with the stretch of the first batch, then re-stretched once together; target and value changes update the labels in place (coalesced), without a new layout |
| `_roi_table.py` | `RoiTableModel`/`RoiTableWidget`: sample labels and target values of all ROIs (a Sample column, one column per target), Enter-to-next-row editing, column paste, dataset and missing-value filters; labels are written through `ProjectState.set_roi_samples` → `Project.save_roi_sample` |
| `_roi_grid.py` | `RoiGridDialog`: a grid of equal ROIs by number, size, origin and spacing (`roi_grid`), empty-position buttons, a live preview on the canvas (outside cells red); `GridDefaultsDialog` (Options > ROI Grid Defaults) and the shared `GridGeometryForm` |
| `_grid_settings.py` | `GridDefaults` every new grid starts with, in the user settings (`roi_grid.json`) |
| `_data_files.py` | `AddDataFileDialog` and `DataFileRequest`: the reference correction of a dataset added to a project (reference files, frame ranges, shared references, levels, raw) |
| `_ui_helpers.py` | Shared plot styling and figure-layout helpers |
| `_plot_panel.py` | `PlotSpec`, `PlotTable`, `PlotPanel`: one plot with a header (title, maximize, PNG/SVG/CSV export) |
| `_plot_grid.py` | `PlotGrid`, `PlotPage`, `PlotSlot`: paged plot grids with lazy pages and in-place maximize |
| `_plot_registry.py` | Registry of all plots and of the plot pages of each workspace |
| `_layout_settings.py` | Maximized plot per workspace in the user settings (`plot_layout.json`) |
| `_project_settings.py` | Last shown dataset per `project_id` in the user settings (`project_ui.json`) |

---

## Module dependencies

Arrows show import direction (A → B means A imports from B).

```
__init__  → _app, _viewer, _attachment, _recent, io
__main__  → _app, __init__, _welcome, _loading, io.project_store
_viewer   → _canvas, _panels, _eda, _modeling, _prediction, _state,
            _browser, _inspector, _model_diagnostics, _attachment, _loading, _theme
_eda      → _state, _modeling (NewROIDialog), _attachment, _loading, io
_modeling → _state, _jobs, io.project_store, modeling, prediction, project
_prediction → _state, _jobs, _predict_dialogs, io, prediction, project
_browser  → _state
_inspector → _state, io.project_store, project
_panels   → _state
_canvas   → (no viewer imports — standalone pyqtgraph widget)
_jobs     → (no viewer imports — standalone Qt threading)
_loading  → _jobs
_state    → io, project, preprocessing, prediction
```

Key: `_canvas.py` and `_jobs.py` still have the loosest coupling in the viewer
package and remain the easiest parts to test or reuse independently.

---

## State management

### ViewerState (UI / EDA state)

Owns the loaded hyperspectral cube and all interactive settings:

- `cube` (ndarray, uint16 or float32), `wavelengths`, `meta`
- Current position: `frame`, `row`, `band` (with setters that emit signals)
- Signal unit: `y_unit` ("DN" / "Reflectance" / "Absorbance")
- Preprocessing: `preprocess_filter`, `sg_window`, `sg_polyorder`, `preprocess_snv/msc/emsc/detrend`
- Dark/white references, reference correction levels
- Image cache: `_image_cache` (invalidated by any setter that affects the image)

**Signals:** `frame_changed(int)`, `row_changed(int)`, `band_changed(int)`, `units_changed()`

**Key methods:**
- `current_frame_image()` → fully processed `float32` `(height, n_bands)` for current frame
- `get_spectrum(row)` → (n_bands,) at row in current frame
- `get_spatial_profile(band)` → (height,) at band in current frame
- `raw_false_colour_rgb()` → raw `float32` preview image for the central spatial canvas
- `has_dataset` → explicit project-only / empty-dataset guard used by the shell

### ProjectState (Qt adapter)

Wraps the `enginir.Project` of the open file: every write goes through
`Project.save_*`, which validates and stores the object; `ProjectState`
mirrors the project's collections for the widgets and emits a Qt signal per
change. It holds no workflow rule: Explore and Predict build datasets,
reference corrections and ROIs with the shared functions of `enginir._workflow`
(`resolve_dataset`, `make_calibration`, `resolve_calibration`,
`resolve_prediction_calibration`, `make_roi`).

Runtime collections include:

- `project_path`, `project_meta`
- Collections: `datasets`, `calibrations`, `rois`, `targets`, `target_assignments`, `recipes`, `signal_domains`, `sample_sets`, `model_runs`
- `predictions`: the in-memory `PredictionResult`s of this session (newest last, at most `SESSION_PREDICTIONS` = 3, cleared on a project switch); stored `/predictions` runs are not shown
- `roi_results`: the per-ROI prediction statistics of this session, one `RoiPredictionSummary` per model (`set_roi_results`, signal `roi_results_changed`); `PredictionView` sets them after every recompute and the ROI table shows them as read-only columns; cleared on a project switch, never stored
- `model_registry` (in-memory fitted model objects)

**Signals:** `project_loaded()`, `datasets_changed()`, `calibrations_changed()`, `rois_changed()`, `targets_changed()`, `target_assignments_changed()`, `recipes_changed()`, `signal_domains_changed()`, `sample_sets_changed()`, `model_runs_changed()`, `predictions_changed()`, `model_fitted(str)`, `active_model_changed(str)`, `prediction_completed(str)`

**Key methods:**
- `new_project(path)`, `open_project(path)` — create/load HDF5 project
- `reload()` — re-read the project; every write (`_written`) schedules it in the other `ProjectState`s of the process on the same project file, so several windows stay in step
- `locate_data_file(dataset_id, path)` — `Dataset.locate`: accept a moved file only if its pixels equal the stored ROI pixels, then update `source_path` and `relative_path`
- `find_data_file(dataset_id)`, `data_file(dataset_id)` — `Project.find_data_file`: where a data file is this session (next to the project, stored path, drive mapping), never written; the viewer opens, predicts and marks "(not found)" through it
- `dataset_id_of(path)` — the dataset a path belongs to at any of its candidate paths (`is_data_file_of`)
- `moved_data_files(old, new)` — missing data files that exist after the folder change of one Locate (`moved_path`), for `HSIViewer._locate_moved_data_files`
- `register_dataset(...)`, `set_active_calibration(...)`, `add_roi(...)`, `define_target(...)`, `assign_target_to_roi(...)` — store through `Project` + emit
- `build_sample_set(...)`, `fit_model_run(...)` — high-level workflows
- `predict_dataset(...)`, `add_prediction(result)` — predict in memory (`prefer_saved_model=True`, so a model fitted in the session gets the DModXPS and T²PS maps) and keep the result for the session; nothing is written
- `ensure_model_run_artifact_refs(...)` — lazy-load heavy persisted arrays only when a workflow actually needs them

---

## Identity

The viewer refers to every project object by id: browser payloads, the ROI
table rows, the Project ROIs tiles, canvas ROI items, Model plot selections
and every write carry ids. It reaches the library's `Project` only through
`ProjectState` and the typed, id-based methods (`save_*`, `remove_*`,
`reload`); the name-based functions and name-keyed views (`Project.roi`,
`.target`, `.model`, `set_value`, `fit`, `project.rois`, …) are for scripts,
and `tests/test_headless_api.py::test_the_viewer_never_addresses_objects_by_name`
fails if the viewer calls them. Names are display text and may repeat; datasets are shown by `ProjectState.data_file_label(dataset_id)`.

## Signal wiring

Wired in `_viewer.py:_wire_signals()`:

**Panel → State:**
```
RAW panel (`DC3Panel`).frame_pick     → ViewerState.set_frame
RAW panel (`DC3Panel`).row_pick       → ViewerState.set_row
Sensor Frame panel (`SensorPanel`).row_pick    → ViewerState.set_row
Sensor Frame panel (`SensorPanel`).band_pick   → ViewerState.set_band
SpatialProfilePanel.row_pick  → ViewerState.set_row
SpectrumPanel.band_pick → ViewerState.set_band
```

**Canvas → State:**
```
FalseColourCanvas.clicked → HSIViewer._on_canvas_click → ViewerState.set_frame + set_row
FalseColourCanvas.roi_selected → ProjectBrowser.select_payload
```

**State → UI:**
```
ViewerState.frame_changed → canvas crosshair update, status strip
ViewerState.row_changed   → canvas crosshair update, status strip
ViewerState.band_changed  → status strip
ViewerState.units_changed → status strip
```

**ProjectState / service state → UI (coalesced shell refresh):**
```
project_loaded, rois_changed, targets_changed, target_assignments_changed,
sample_sets_changed, model_runs_changed, predictions_changed, active_model_changed
ModelingView._service_status.status_changed
PredictionView._service_status.status_changed
```

`HSIViewer.schedule_workspace_content_refresh()` coalesces repeated refresh
requests into one event-loop update, and only the active workspace page redraws
its heavy widgets.

---

## Background jobs

Pattern: `_jobs.py:start_background_job(owner, fn, on_success, on_error, on_finished)`

1. Creates `_CallbackRelay` (parented to `owner` to prevent GC)
2. Creates `QThread` + `BackgroundWorker`
3. Moves worker to thread
4. Wires: `thread.started → worker.run`, `worker.succeeded → relay.handle_success`, `worker.failed → relay.handle_error`, `worker.finished → thread.quit → thread.deleteLater`, `thread.finished → relay.handle_finished`
5. Starts thread
6. Returns `(thread, worker, relay)` — caller must store all three

**Used by:** `_modeling.py` (fit model), `_prediction.py` (predict dataset), `_loading.py` (launcher startup jobs)

**Error contract:** `BackgroundWorker.run()` catches `Exception`, logs via `logger.error`, emits `failed(str)`. Callers show `QMessageBox` in `on_error` callback.

Both Modeling and Prediction expose `BusyStatusWidget` state in two places:

- the control-panel workflow group
- the primary workspace summary page in `_viewer.py`

Sample-set build and model fit currently use indeterminate progress. Prediction
uses structured service events and switches to determinate chunk progress when
the total is known; the visible phase distinguishes preflight, loading,
reference correction, preprocessing, inference, writing, finalization, and rendering.

Startup and file-open feedback uses `_loading.py`:

- `ViewerBusyContext` centralizes wait cursor, disabled controls, and optional status-bar/status-widget text.
- The welcome screen shows progress in the same `_jobs.BusyStatusWidget` as the workspaces: Qt's native busy bar, or a determinate bar once a `LoadingProgressEvent` carries a total.
- Payloads are read with `_jobs.start_background_job(...)` before Qt widget construction.

The welcome launcher uses this for dataset/project reads so the UI remains
responsive between selecting a file and the main shell appearing. Main-window
open/attach paths use the same busy context for status-bar feedback and wait
cursor coverage around blocking work.

---

## Payload protocol

`ProjectBrowser` items store a dict via `Qt.ItemDataRole.UserRole`:

```python
{"kind": "section",            "id": "datasets",    "label": "Datasets"}
{"kind": "dataset",            "id": "<uuid>"}
{"kind": "calibration",        "id": "<uuid>"}
{"kind": "roi",                "id": "<uuid>"}
{"kind": "target",             "id": "<uuid>"}
{"kind": "target_assignment",  "target_id": "...", "roi_id": "..."}
{"kind": "sample_set",         "id": "<uuid>"}
{"kind": "signal_domain",      "id": "<uuid>"}
{"kind": "model_run",          "id": "<uuid>"}
{"kind": "prediction",         "id": "<uuid>"}
```

Consumers:
- `_inspector.py:show_item()` dispatches via an explicit handler map
- `_viewer.py:_on_project_browser_selection_changed()` routes to canvas, modeling view, prediction view
- `_canvas.py:select_roi()` highlights the ROI overlay

---

## Preprocessing pipeline

`ViewerState.current_frame_image()` applies three stages in order:

```
1. Signal unit:  raw DN  →  Reflectance  →  Absorbance
   - DN: raw float32 passthrough
   - Reflectance: _to_reflectance() (dark/white ref or default levels)
   - Absorbance: -log10(reflectance)

2. Filtering (mutually exclusive):
   - none / sg_smooth / sg_d1 / sg_d2
   - SavitzkyGolay(window, polyorder, deriv)

3. Scatter correction (stackable, in order):
   - SNV → MSC → EMSC → Detrend
```

Result is cached in `_image_cache`, invalidated by any setter that changes
frame, signal unit, or preprocessing parameters.

## Performance rules

### Memory-aware data flow

- Project open is metadata-first: the default `load_project_store(...)` call keeps ROI spectra and heavy model/prediction artifacts on disk.
- Working spectral arrays in viewer/modeling/prediction paths default to `float32`.
- ROI sample-set construction bulk-loads ROI spectra from one HDF5 file handle instead of reopening the project per ROI.
- Prediction and model artifact arrays are loaded through bounded or explicit on-demand helpers.

### UI responsiveness

- `HSIViewer` only refreshes the active workspace's heavy widget (`ModelDiagnosticsWidget` or `PredictionMapWidget`).
- Repeated shell refresh requests are coalesced through `QTimer.singleShot(0, ...)`.
- Explore dock panels defer rebuilds while hidden, but the default visible Explore panels still perform one initial build so guides and labels exist immediately after viewer construction.
- `HistogramPanel` updates its existing bar artists in place instead of rebuilding the axes on every frame/unit change.
- `ModelDiagnosticsWidget` and `PredictionMapWidget` skip redraws for unchanged payload objects.
- Each plot panel has its own canvas. Changing one plot redraws only that plot. A page draws when it is first shown, and a panel draws only while visible.
- Figure layout is recomputed for a new payload or a new canvas size (`maybe_tight_layout(...)` for in-place panels), not on every redraw.
- `tests/test_plot_performance.py` (opt-in, `ENGINIR_PERF=1`) times the Model diagnostics against the single-figure baseline.
- Welcome-screen startup loads dataset/project payloads in a worker thread and only constructs Qt widgets on the GUI thread after loading completes.

### Window size on Wayland

- `HSIViewer._release_minimum_size()` sets the main window layout to
  `SetNoConstraint`: Qt would otherwise make the sum of the docks' minimum
  widths the window's minimum, and a compositor that configures a smaller
  maximized size (WSLg) ends the connection when the buffer is larger. The
  Controls dock content sits in a horizontal scroll area for the same reason.
  `tests/test_project_rois.py` checks that no workspace sets a minimum size.

### Process exit

- At interpreter exit PyQt deletes the remaining top-level Qt objects while walking sip's wrapper table. Python code that runs during those deletions can reallocate the table and crash the process (a sip bug, unfixed in PyQt6-sip 13.13). `_app._remove_dialog_centering` therefore removes the application-wide event filter (dialog centering and the WSLg combo-popup release) in an `atexit` hook that runs before PyQt's cleanup. Do not add another Python event filter on the `QApplication` without removing it the same way; `tests/test_viewer_exit.py` checks that a closed project viewer exits cleanly.

---

## Plot panels

Workspace plots follow the Plot panel policy of
`contracts/viewer-workspace-ui-contract_v3.md`. The Model diagnostics are the
first workspace built this way:

- a `PlotSpec` per plot has `render(fig, payload, colors)` and
  `table(payload) -> PlotTable`
- a `PlotPanel` shows one spec under a header (title, maximize, PNG/SVG/CSV
  export)
- a `PlotGrid` lays out the panels on pages (`PlotPage`/`PlotSlot`)

Plots that update their artists in place (the Explore panels in
`_panels.py` and the pyqtgraph `FalseColourCanvas`) are not specs. They
implement `PlotSource` (`plot_table()`, `print_figure()`) and are shown in a
`PlotFrame`:

- The RAW and Sensor Frame panels (`_ImagePanel`) draw with pyqtgraph and
  build their print figure by redrawing the shown image with matplotlib.
- The other `_panels.py` panels build their print figure with `print_copy` (a pickled copy of
  the screen figure, recoloured).
- The canvas redraws its current view with matplotlib.
- `PredictionMapWidget` is a source too. Its header controls (layer, output,
  pan, zoom, reset) are added to its frame. The layers are `MAP_LAYERS`
  (Prediction, DModXPS, T²PS, Prediction interval); a diagnostics layer marks the pixels
  above its 0.99 limit and is disabled for a model without projection data.
  Prediction interval is `PredictionResult.half_width(INTERVAL_LEVEL)` (computed in
  `_map_previews`, with `interval_summary` for the median and the pixels
  without an interval), on its own scale, with the pixels outside the model
  in the marking color; it is disabled for a model without interval inputs.
  It emits `displayed_map_changed` (values, quantity, limit) and
  `hovered(text)` (`describe_pixel`: true frame and row, prediction with its
  ± half-width, DModX against DCrit and the removal limit, T², ROI), which the window shows in its status line;
  `PlotPanel.roi_hovered` does the same for the per-ROI plot
  (`PredictionView.roi_hover_text`). On Predict the idle status line is
  `PredictionView.status_text()`. The `predict.histogram`
  spec draws `PredictionView.histogram_payload(shown)`: the kept pixels of
  the prediction ROIs per model when there are ROI results (following the
  exclusion), else the shown map, with its limit.
- `PlotPageWidget` lays out frames that are not specs in slots, with the same
  maximize as `PlotGrid`. The Predict page uses it for the map and the
  histogram.
- The dock frames give their header to the dock as its title bar.
  `HSIViewer` handles maximize for the Explore page and for the docks.
- The data-file docks (Histogram, RAW, Sensor Frame, Spatial Profile) are
  shown only on Explore's Dataset tab. `_sync_data_file_docks()` runs on
  every workspace and Explore tab change; the user's choice per dock lives in
  `_explore_plot_visibility` and is recorded from `isHidden()` (a dock behind
  another dock's tab is not visible but still shown). The ROI table follows
  the Explore tab the same way (`_sync_roi_table()`, choice per tab in
  `_roi_table_visibility`: shown on Project ROIs, hidden on Dataset);
  Model and Predict leave it as it is. `_place_roi_table()` tabs a docked
  table with the Sensor Frame on the Dataset tab. The table's dataset
  filter follows the page (`_roi_table_data_file()`).

`ModelDiagnosticsWidget` wraps the grid of the `"Model"` workspace, with the
`Fit` and `Spectral` pages. It hands the same diagnostics payload to every
plot, and `ModelingView.workspace_diagnostics_provenance()` supplies the CSV
provenance lines.

Drawing rules:

- `render` draws with the `colors` it is given: the screen theme on screen,
  `print_colors()` for exports. It never sets an axes title, because the
  header shows the title.
- When the payload has no data for its plot, `render` draws an empty-state
  message with `draw_placeholder`, and `table` returns an empty `PlotTable`.
  That disables export.
- `render` returns `RoiPoints` for clickable ROI points. The panel turns a
  click into `roi_selected`.
- A panel draws only while visible, and only for a new payload object. A
  page is built the first time it is shown.

Workspace export and layout state:

- `HSIViewer.workspace_plot_frames()` lists the plots of the current
  workspace. For Model it calls `PlotGrid.all_panels()`, which builds unopened
  pages without switching to them.
- `export_workspace_plots(folder)` writes `<plot_id>.png`/`.csv` for each
  plot that shows data.
- `_layout_settings.py` keeps the maximized plot of each workspace in
  `plot_layout.json` in the config directory (Explore page, Model grid,
  Predict page; not floating docks). `reset_layout()` clears it.

## Adding a new plot

1. Write `render` and `table` for the payload the workspace already builds,
   and wrap them in a `PlotSpec` with a unique `plot_id` (`"<workspace>.<name>"`).
2. Register it with `register_plot(...)` and add a `PlotSlot` to the
   workspace's pages in the module that owns the workspace (for example
   `_model_diagnostics.py`).
3. Add a payload with data for the workspace to `_PAYLOADS` in
   `tests/test_plot_registry.py`, if the workspace is new. That test then
   checks the new plot:
   - the empty state
   - PNG, SVG and CSV export
   - no title inside the plot
   - every CSV value is drawn
4. Use `style_axes(ax, colors)` and `draw_placeholder(ax, message, colors)`,
   not local theme constants.
