# Viewer Architecture

Developer reference for the `src/enginir/viewer/` module.

---

## File inventory

| File | Responsibility |
|------|----------------|
| `__init__.py` | Public API: `open_viewer()`, `open_project()` |
| `__main__.py` | CLI entry point `enginir-viewer` |
| `_app.py` | QApplication singleton, Wayland workaround |
| `_theme.py` | Shared Qt stylesheet (Fusion + light palette) |
| `_state.py` | `ViewerState` (UI) + `ProjectState` (Qt adapter over `enginir.Project`) |
| `_canvas.py` | `FalseColourCanvas` (pyqtgraph image surface, crosshair, ROI draw/selection) |
| `_panels.py` | Exploration dock widgets (Histogram, RAW, Sensor, Spatial, Spectrum) |
| `_viewer.py` | `HSIViewer(QMainWindow)` shell, menus, docks, status strip, workspace host |
| `_welcome.py` | Launcher window with recent files list and startup routing |
| `_recent.py` | MRU file list (`~/.config/enginir/recent.json`) |
| `_eda.py` | `EDAView` — signal, preprocessing, project, and ROI controls |
| `_attachment.py` | Project/dataset binding validation |
| `_modeling.py` | `ModelingView` — target, sample set, and model fit workflows |
| `_model_diagnostics.py` | Model workspace diagnostics surface |
| `_prediction.py` | `PredictionView` and `PredictionMapWidget` |
| `_browser.py` | Read-only workspace object tree |
| `_inspector.py` | Selection-driven metadata display |
| `_jobs.py` | `BackgroundWorker`, `_CallbackRelay`, `start_background_job`, `BusyStatusWidget` |
| `_loading.py` | Shared wait-cursor/status/progress helpers for startup and open flows |
| `_data_helpers.py` | Dataset-shape and spectral array helpers shared by viewer workflows |
| `_ui_helpers.py` | Shared plot styling and figure-layout helpers |

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
_prediction → _state, _jobs, io, prediction, project
_browser  → _state
_inspector → _state
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
- Dark/white references, calibration levels
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
calibrations and ROIs with the shared functions of `enginir._workflow`
(`resolve_dataset`, `make_calibration`, `resolve_calibration`,
`resolve_prediction_calibration`, `make_roi`).

Runtime collections include:

- `project_path`, `project_meta`
- Collections: `datasets`, `calibrations`, `rois`, `targets`, `target_assignments`, `recipes`, `signal_domains`, `sample_sets`, `model_runs`, `predictions`
- `model_registry` (in-memory fitted model objects)

**Signals:** `project_loaded()`, `datasets_changed()`, `calibrations_changed()`, `rois_changed()`, `targets_changed()`, `target_assignments_changed()`, `recipes_changed()`, `signal_domains_changed()`, `sample_sets_changed()`, `model_runs_changed()`, `predictions_changed()`, `model_fitted(str)`, `active_model_changed(str)`, `prediction_completed(str)`

**Key methods:**
- `new_project(path)`, `open_project(path)` — create/load HDF5 project
- `register_dataset(...)`, `set_active_calibration(...)`, `add_roi(...)`, `define_target(...)`, `assign_target_to_roi(...)` — store through `Project` + emit
- `build_sample_set(...)`, `fit_model_run(...)`, `predict_dataset(...)` — high-level workflows
- `ensure_model_run_artifact_refs(...)`, `ensure_prediction_artifact_refs(...)` — lazy-load heavy persisted arrays only when a workflow actually needs them

---

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

**Used by:** `_modeling.py` (build sample set, fit model), `_prediction.py` (predict dataset), `_loading.py` (launcher startup jobs)

**Error contract:** `BackgroundWorker.run()` catches `Exception`, logs via `logger.error`, emits `failed(str)`. Callers show `QMessageBox` in `on_error` callback.

Both Modeling and Prediction expose `BusyStatusWidget` state in two places:

- the control-panel workflow group
- the primary workspace summary page in `_viewer.py`

Sample-set build and model fit currently use indeterminate progress. Prediction
uses structured service events and switches to determinate chunk progress when
the total is known; the visible phase distinguishes preflight, loading,
calibration, preprocessing, inference, writing, finalization, and rendering.

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
- Figure layout uses a size-aware `maybe_tight_layout(...)` helper, so `tight_layout()` is only recomputed when the canvas size changes.
- Welcome-screen startup loads dataset/project payloads in a worker thread and only constructs Qt widgets on the GUI thread after loading completes.

---

## Adding a new panel

Follow the pattern in `_model_diagnostics.py`:

1. Create `_my_panel.py` inheriting from `QWidget`
2. Add a matplotlib `Figure` + `FigureCanvasQTAgg`
3. Implement `update_data(payload, *, empty_reason="")` and `clear()`
4. Use `panel_colors()`, `style_axes()`, and `maybe_tight_layout()` from
   `_ui_helpers.py` rather than defining local theme constants
5. Add logger: `from enginir._logging import get_logger; logger = get_logger(__name__)`
6. Wire in `_viewer.py`: import, instantiate, add to layout, connect signals
7. Add to `__all__` in your module

New viewer surfaces should follow the lightweight widget patterns used in
`_model_diagnostics.py` and `_prediction.py` rather than introducing a new
ABC layer.
