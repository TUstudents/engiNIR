# HSI Viewer Reference

## 1. Overview

The `enginir.viewer` module provides the engiNIR desktop viewer for
hyperspectral image (HSI) cubes. It is built on PyQt6 with a
[pyqtgraph](https://pyqtgraph.readthedocs.io) false-color canvas plus
[matplotlib](https://matplotlib.org)-backed diagnostic panels docked around it.

### Architecture

```
enginir.viewer
├── __init__.py      open_viewer()/open_project() — public startup entry points
├── __main__.py      enginir-viewer CLI; dataset-first and project-first startup
├── _app.py          QApplication singleton, ensure_app(), run()
├── _welcome.py      WelcomeWindow — launcher screen for dataset/project startup
├── _recent.py       load/save recent files list (pure I/O, no Qt)
├── _loading.py      shared wait-cursor, status, progress, and startup-job helpers
├── _canvas.py       FalseColourCanvas (pyqtgraph ImageItem + crosshair + ROI overlays)
├── _state.py        ViewerState(QObject), ProjectState
├── _eda.py          EDAView — dataset, calibration, project, and ROI controls
├── _modeling.py     ModelingView — target, sample-set, and model-fit workflows
├── _prediction.py   PredictionView — saved-model prediction workflows
├── _model_diagnostics.py  ModelDiagnosticsWidget — Model workspace diagnostics surface
├── _browser.py      ProjectBrowser — workspace object tree
├── _inspector.py    ProjectInspector — selection-driven metadata surface
├── _panels.py       HistogramPanel, RAW panel (`DC3Panel`), Sensor Frame panel (`SensorPanel`),
│                    SpatialProfilePanel, SpectrumPanel
├── _plot_panel.py   PlotFrame/PlotPanel — plot header (title, maximize, PNG/SVG/CSV export)
├── _plot_grid.py    PlotGrid/PlotPageWidget — plot pages with slots and in-place maximize
├── _plot_registry.py  registry of the plot specs and pages
├── _layout_settings.py  maximized plots in the user settings
└── _viewer.py       HSIViewer(QMainWindow) — shell, menus, docks, log, child windows
```

All panels read from and write to `ViewerState` through typed setters and
Qt signals.  No panel holds its own copy of the data.  Preprocessing is
computed inside `ViewerState.current_frame_image()` so every panel always
displays the same processed signal.

Every Explore plot is a plot frame (`_plot_panel.PlotFrame`). Its header
carries the title, a maximize control and an export menu (PNG, SVG, CSV,
copy). In the docks, the header is the dock's title bar and adds a close
button.

- **Spatial Image** and **Spectral Profile** maximize within the Explore page.
- A **docked plot** maximizes by floating over the main window. Restoring it,
  or closing it, puts it back in its dock.
- The `Spatial Image` header also has the `Reset View` control, which resets
  zoom and pan to the full image bounds.

`File > Export Plots...` writes every plot of the current workspace that shows
data as `<plot_id>.png` and `<plot_id>.csv` into one folder. The maximized
plot of each workspace page is remembered in `plot_layout.json` in the config
directory. `Reset Layout` clears it.

Exports show the current view in the print colors. Each CSV holds the drawn
values:

- histogram bins
- the RAW band and the sensor frame as matrices
- the profile and the spectrum as columns
- the false-colour image as one row per pixel

### Panel layout

```
┌────────────────────────────────────────────────────────────────────┐
│ Project Browser / Inspector │ Workspace content host  │ Sensor /  │
│ left dock stack             │ Explore/Model/Predict   │ Spatial   │
│                             │ primary page            │ right docks│
├─────────────────────────────┼─────────────────────────┼───────────┤
│ Histogram / RAW (collapsed) │                         │           │
├─────────────────────────────┴─────────────────────────┴───────────┤
│ Controls dock — Explore / Model / Predict tabs                     │
├────────────────────────────────────────────────────────────────────┤
│ Log dock (collapsed by default)                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Installation / requirements

The viewer is included in the `enginir` package.  The required dependencies
(`pyqt6>=6.10.2`, `pyqtgraph>=0.13`, `matplotlib>=3.7`) are listed in
`pyproject.toml` and installed automatically with:

```bash
uv sync
```

---

## 3. Launch modes

### Standalone launcher

After `uv sync`, the `enginir-viewer` command is available system-wide:

```bash
enginir-viewer                              # welcome screen (recent files)
enginir-viewer data/sample.eiff             # open dataset directly
enginir-viewer data/sample.eiff --dark dark.eiff --white white.eiff
enginir-viewer data/sample.eiff --project project.h5
enginir-viewer project.h5                   # open project directly
enginir-viewer --project project.h5
```

Launched without arguments, the welcome screen offers **Open Dataset...**,
**Open Project...**, and **Create Project...**, and also routes recent project
files into project-first startup automatically. Dataset/project reads from the
welcome screen run on a background Qt worker; the launcher disables startup
controls, shows an indeterminate progress bar, and switches to a wait cursor
until the main shell is ready.

### Python API

```python
from enginir.viewer import open_project, open_viewer, run

viewer = open_viewer('data/sample.eiff')
project_viewer = open_project('project.h5')
attached_viewer = open_viewer('data/sample.eiff', project='project.h5')
run()   # blocking event loop; omit in Jupyter
```

In a Jupyter notebook the Qt event loop is integrated automatically
(`ip.enable_gui("qt")`).  The viewer window opens immediately and stays
alive while you continue running cells.

### Module invocation

```bash
uv run python -m enginir.viewer data/sample.eiff
uv run python -m enginir.viewer data/sample.eiff --dark dark.eiff --white white.eiff
uv run python -m enginir.viewer data/sample.eiff --project project.h5
uv run python -m enginir.viewer project.h5
uv run python -m enginir.viewer --project project.h5
```

---

## 4. Panel reference

### HistogramPanel (left dock, top)

Displays a 256-bin histogram of all pixel intensities in the **current
frame** after the full preprocessing pipeline has been applied.

- Updates on: `frame_changed`, `units_changed`
- No emitted signals

### RAW panel (internal class `DC3Panel`, left dock, bottom)

Drawn with pyqtgraph, like the Sensor Frame, so that frame and band scrubbing
only swaps the image or moves the crosshair. Exports redraw the shown view
with matplotlib. Displays a raw cross-section of the cube at the selected band:
`cube[:, :, band].T` — spatial rows on the Y-axis, scan frames on the
X-axis.  Yellow crosshairs mark the active `(frame, row)`.

Note: RAW always shows raw DN (no preprocessing) to provide a stable
spatial navigation reference.

- Updates on: `frame_changed` and `row_changed` (crosshair only),
  `band_changed` (new band image)
- **Click** → emits `frame_pick(int)`, `row_pick(int)`

### Sensor Frame panel (internal class `SensorPanel`, right dock, top)

Displays the processed current frame: `current_frame_image()` — height ×
n_bands.  Yellow crosshairs mark the active `(row, band)`; drawn with pyqtgraph.

- Updates on: `frame_changed`, `units_changed` (full rebuild)
- **Click** → emits `row_pick(int)`, `band_pick(int)`

### SpatialProfilePanel (right dock, middle)

Plots processed `intensity vs. row` for the selected band.  A red vertical
marker shows the active row and can be dragged.

- Updates on: `frame_changed`, `band_changed`, `units_changed` (full rebuild);
  `row_changed` (marker only)
- **Drag marker** → emits `row_pick(int)`

### SpectrumPanel (bottom dock)

Plots processed `intensity vs. wavelength (or band)` for the selected row.
A red vertical marker shows the active band and can be dragged.

- Updates on: `frame_changed`, `row_changed`, `units_changed` (full rebuild);
  `band_changed` (marker only)
- **Drag marker** → emits `band_pick(int)`

---

## 5. Crosshair navigation

All panels share a single `(frame, row, band)` cursor stored in
`ViewerState`.  Picking a new position in any panel propagates through
`ViewerState` signals and all subscribed panels update immediately.

| Action | Panel | Sets |
|--------|-------|------|
| Click | RAW | `frame`, `row` |
| Click | Sensor Frame | `row`, `band` |
| Drag red marker | Spatial Profile | `row` |
| Drag red marker | Spectral Profile | `band` |

---

## 6. Controls panel

The controls area is a tabbed workspace shell:

- `Explore` for dataset inspection, calibration context, and ROI persistence
- `Model` for target editing, sample-set build, and model fitting
- `Predict` for saved-model prediction runs

The shell also keeps two persistent navigation panes:

- `Project Browser` on the left for project objects
- browser sections include datasets, calibrations, ROIs, targets, target
  assignments, sample sets, signal domains, model runs, and predictions
- `Inspector` below the browser in the left dock stack for selection-driven details

By default, `Histogram`, `RAW`, and `Log` stay collapsed until opened
explicitly from the View menu.

### Explore workspace

The `Explore` tab contains the project, signal, filtering, normalization,
axis, and ROI controls, and it keeps the `Spatial Image` canvas as the primary
center content.

While `Explore` is active, `Sensor Frame`, `Spatial Profile`, and
`Spectral Profile` remain available as the active exploration plots.

When no dataset is attached, Explore keeps the shell open in explicit
project-only mode: the canvas and plot docks show `No dataset attached`
empty states and ROI creation stays disabled until a real dataset is attached.

### Project group

| Button | Action |
|---|---|
| **New** | Create a new empty HDF5 project file |
| **Open** | Load an existing project (warns if ROI source files don't match the current cube) |

The status label below the buttons shows the project filename and ROI count.
The project group also shows an autosave hint because project-backed changes
are persisted eagerly as they are made.
All persisted training ROIs in the open project are drawn as magenta rectangles
on the false-color canvas.

### ROI group

| Control | Action |
|---|---|
| **Add ROI** | Enter one-shot ROI draw mode on `Spatial Image` |
| **Delete ROI** | Remove the currently selected ROI |
| **Export CSV** | Export all persisted training ROI spectra to one CSV file |

The ROI creation flow is centred on `Spatial Image`:

1. click **Add ROI**
2. drag a rectangle on `Spatial Image`
3. release to open the persisted ROI dialog
4. save the ROI and immediately show it as an overlay
5. auto-select the new ROI in the canvas, project browser, and inspector

Target values are typed afterwards in the ROI table (`_roi_table.RoiTableWidget`),
or in the ROI dialog for the selected target.

This means ROI labels are display-only and do not need to be unique: the
viewer refers to ROIs, targets, models and data files by id only
(`viewer-workspace-ui-contract_v3.md`, Identity in the viewer). Data files are
shown with `enginir._workflow.data_file_labels`: the file name, plus the
folder that tells repeated file names apart.
Viewer-generated `target_id`, `sample_set_id`, and `model_run_id` values also
use canonical UUID strings unless a caller provides explicit ids.

Persisted training ROIs also capture the current dataset and calibration
context so later sample-set build and model-fit steps can construct the saved
signal-domain contract from project state instead of relying on mutable UI-only
context.

### Model workspace

The `Model` tab contains the target, sample-set, and fit workflow plus a
readiness summary that tells the user the next expected action.

Its primary center content is a diagnostics-first model page showing:

- fit summary cards for active target, selected/default sample set, active
  model run, components, CV, ROI count, pixel count, `R²`, `RMSECV`, and bias
- the `Fit` page: `Predicted vs Reference` (with `R²`, `RMSECV` and bias),
  `Residuals`, `Target Distribution` and `Training Scores (LV1 vs LV2)`
- the `Spectral` page: `Regression Coefficients` and `Latent Loadings (LV1)`

Every plot is a panel with a header. The header holds the title, a maximize
control (also a header double-click; `Esc` restores) and an export menu with
PNG, SVG, CSV and copy to the clipboard.

When no fitted run is available yet, the readiness message remains visible
above the empty diagnostics placeholders so the blocking condition is still
obvious.

### Modeling group

| Control | Action |
|---|---|
| **Target...** | Select one existing target or define a new target |
| **Target values...** | Open the ROI table on the selected target: type, clear or paste the values of all ROIs |
| **Fit model** | One dialog: target, signal unit, preprocessing (prefilled from the Explore preview), model, components, validation, and the labelled ROIs per data file; stores the grouped v3 `ModelRun` |

The Modeling section operates from project state. The fit dialog finds the
sample set of all ROIs with a value for the target (`resolve_sample_set`,
the rule `Project.fit` uses): an equal stored sample set, recipe and
`SignalDomainContract` are reused, new ones are stored only together with
the fitted model. A refused or failed fit writes nothing. Model fitting uses
ROI-grouped CV with ROI-level primary reporting.

### Predict workspace

The `Predict` tab contains the saved-model prediction workflow plus a
readiness summary that reflects whether a model is available and whether the
user is ready to run or rerun prediction.

Its primary center content is a dedicated prediction summary page showing:

- current prediction result headline
- readiness state
- current service state
- result cards for prediction id, model run, dataset, calibration, and target
- a compact compatibility summary derived from the saved signal-domain contract
- the persisted prediction map, in a plot frame whose header holds the
  output selector, `Pan`, `Zoom` and `Reset View`
- `Prediction Distribution`, the histogram of the shown map values, which
  follows the selected output

A large map is shown as a preview that keeps every n-th frame and row
(`map_step` in the payload, the step of
`load_prediction_spatial_map_preview`). The axes, the hover readout and the
CSV use the true frame and row indices.

### Prediction group

| Control | Action |
|---|---|
| **Predict...** | Load a new dataset and run prediction against one saved model run |

Prediction reuses the scan's registration and active calibration when it is
already in the project, is refused before it runs when the scan cannot produce
the model's signal (saved `SignalDomainContract`), and stores a
`PredictionRun`.

Outside `Explore`, the Explore plot toggles are disabled so `Model` and
`Predict` keep their own primary-content emphasis instead of immediately
reopening the exploration plot set.

### Signal group

Selects the physical quantity displayed in all panels.

| Radio button | Formula | Notes |
|---|---|---|
| **Raw DN** | raw uint16 value | Always available |
| **Reflectance** | `clip((raw − dark) / (white − dark), 1e-6, 1)` | Uses default levels when no refs loaded |
| **Absorbance** | `−log10(Reflectance)` | Uses default levels when no refs loaded |

When no dark / white reference files are loaded, the viewer uses the EVK
factory default levels (black = 128, white = 3500) instead of falling back
to DN.  A small label below the radio buttons shows `"default: black=128
white=3500"` as a reminder.

**Programmatic control:**

```python
viewer.state.set_y_unit('Reflectance')   # 'DN' | 'Reflectance' | 'Absorbance'
```

### Filtering group

Applies a Savitzky-Golay filter to every spectrum in the current frame.
Options are mutually exclusive.

| Radio button | `deriv` | Effect |
|---|---|---|
| **None** | — | No filtering |
| **SG Smooth** | 0 | Polynomial smoothing, no derivative |
| **SG 1st deriv** | 1 | First spectral derivative |
| **SG 2nd deriv** | 2 | Second spectral derivative |

The **Window** and **Poly** spinboxes (enabled only when a SG mode is
selected) control `window_length` and `polyorder`.  `polyorder` is
automatically clamped to `window_length − 1`.

**Programmatic control:**

```python
viewer.state.set_preprocess_filter('sg_d1')   # 'none'|'sg_smooth'|'sg_d1'|'sg_d2'
viewer.state.set_sg_window(11)
viewer.state.set_sg_polyorder(2)
```

### Normalization group

Stackable scatter-correction and baseline steps applied after filtering,
in the order shown.

| Checkbox | Class | Description |
|---|---|---|
| **SNV** | `SNV` | Standard Normal Variate — mean-center + scale by std |
| **MSC** | `MSC` | Multiplicative Scatter Correction — reference = frame mean |
| **EMSC** | `EMSC` | Extended MSC — includes polynomial baseline terms |
| **Detrend** | `Detrend` | Subtract polynomial baseline; order set by adjacent spinbox |

Multiple checkboxes can be active simultaneously.  The pipeline is always
applied in the order SNV → MSC → EMSC → Detrend.

**Programmatic control:**

```python
viewer.state.set_preprocess_snv(True)
viewer.state.set_preprocess_detrend(True)
viewer.state.set_detrend_order(2)        # quadratic baseline removal
```

### Axis group

Controls the axis units for the profile and spectrum panels.

| Control | Options |
|---|---|
| **Spectral** radio | `nm` (wavelength axis from file) / `band` (integer index) |
| **Spatial** radio | `pixel` (row index) / `mm` (row × pixel size) |
| **px (mm)** spinbox | Physical pixel pitch in mm; enabled in `mm` mode |

Axis changes do not invalidate the image cache — only axis labels update.

**Programmatic control:**

```python
viewer.state.set_x_spectral_unit('nm')     # 'nm' | 'band'
viewer.state.set_x_spatial_unit('mm')      # 'pixel' | 'mm'
viewer.state.set_pixel_size_mm(0.25)
```

### ROI group

| Button | Action |
|---|---|
| **Add ROI** | Enter one-shot ROI draw mode on `Spatial Image` |
| **Delete ROI** | Remove the currently selected ROI |
| **Export CSV** | Save all persisted training ROI spectra to CSV (calls `export_csv`) |

---

## 7. Preprocessing pipeline

All preprocessing is applied inside `ViewerState.current_frame_image()` in
a fixed order:

```
raw uint16
    ↓  Signal unit  (DN / Reflectance / Absorbance)
    ↓  Filtering    (SG smooth / 1st / 2nd derivative)
    ↓  Normalization (SNV → MSC → EMSC → Detrend)
    → (height, n_bands) float32   ← used by all panels
```

The result is cached.  The cache is invalidated whenever frame, signal
unit, references, or any preprocessing parameter changes.  Axis-only
changes (nm↔band, pixel↔mm) do not invalidate the cache.

---

## 8. ROI workflow

### Collecting training ROIs (HDF5 project)

Training ROIs are stored in the v3 HDF5 project workspace. The project does
not embed the source cube itself, but it does persist the extracted raw ROI
spectra plus dataset/calibration provenance, explicit targets, target
assignments, preprocessing recipes, sample sets, model runs, and prediction
runs.

1. Click **Project → New** to create a project file.
2. Open a hyperspectral cube in the viewer or attach the project during startup.
3. Use **Target...** in the `Model` workspace to select one active target or define a new one.
4. Click **Add ROI** in the ROI group box.
5. Draw a rectangle on the `Spatial Image` canvas. A dialog prompts for:
   - **Name** — display label for this ROI
   - **target value** — optional value for the currently selected target
6. On save, the ROI is drawn immediately as an overlay and auto-selected in
   the canvas, project browser, and inspector.
7. Repeat for each ROI.
8. Each ROI is written immediately with a stable `roi_id`, the current
   dataset/calibration references, and raw spectra. If a target value is
   provided, it is persisted separately as a `TargetAssignment`. The status
   bar confirms every write (`ProjectState.saved`).

Explore's central area has two tabs: **Data file** (the image and its plots)
and **Project ROIs** (`_project_rois.ProjectRoisView`). The second tab reads
stored ROI pixels with `project_store.load_roi_band_images`, and ROI spectra
with `modeling.roi_spectrum_summaries`. The latter converts each ROI with
`modeling.roi_to_signal_domain`, the function the fit uses. On the benchmark
project (10 trays, 400 ROIs of 2,000 pixels) the first tile appears 0.7 s
after the window and all 400 within about 4 s, with the event loop free
between batches.

**Add ROI grid** (`EDAView.open_roi_grid_dialog`, `_roi_grid.RoiGridDialog`)
lays out equal ROIs by number, size, origin and spacing with
`enginir._workflow.roi_grid`, starting from **Options > ROI Grid Defaults**
(`_grid_settings.GridDefaults`, user settings) every time.
It previews them on the canvas (`FalseColourCanvas.set_grid_preview`) and
stores them through `ProjectState.add_rois` → `Project.save_rois`, all or none.

A project may hold ROIs from several data files. The canvas draws only the
ROIs whose `dataset_id` is the project dataset of the shown file
(`EDAView.refresh_roi_overlays`, `EDAView.shown_dataset_id`). Attach
validation is per data file: an unregistered file is a new data file, and only
a registered file that no longer matches its `DatasetRef` raises the warning
gate (`_attachment.confirm_project_dataset_binding`).

A data file whose `source_path` no longer exists is listed as "(not found)".
Showing it offers **Locate file...** (`HSIViewer.locate_data_file` →
`ProjectState.locate_data_file` → `Dataset.locate`). The chosen file must pass
`enginir._workflow.check_data_file_location`: same format, shape, modality and
wavelengths, and for every ROI the file's pixels at its `spatial_bounds`
equal the stored raw pixels. Only then `project_store.relocate_dataset_ref`
updates `source_path`. The last shown data file is kept per `project_id` in
the user settings (`_project_settings.py`) and loaded when the Data file tab
shows nothing. A write in one window makes the other windows of the process
on the same project re-read it (`ProjectState.reload`).

```python
# Inspect the canonical v3 project graph later
from enginir.io import load_project_store

store = load_project_store('project.h5')
for roi_id, roi_meta in store['rois'].items():
    print(roi_id, roi_meta['dataset_id'], roi_meta['calibration_id'])
```

Project graph loading is metadata-first by default. Spectra are loaded lazily
via `TrainingROI.load_spectra(project_path)` to avoid loading large datasets
into memory at project open time.

### Re-opening a project

Click **Project → Open**.  If any ROI was drawn from a different cube than
the one currently loaded, a warning dialog asks whether to proceed.  You can
open projects regardless of whether the source cube is loaded.

### Using ROI data for modelling

```python
from enginir import PreprocessingRecipe, Project

project = Project.open('project.h5')
model = project.fit('moisture', recipe=PreprocessingRecipe(snv=True),
                    n_components=6, cv='logo')
print(model.metrics['rmsecv'])
```

v3 modeling semantics are fixed:

- pixel observations
- ROI-grouped validation
- ROI-level primary reporting
- mean aggregation of held-out pixel predictions back to ROI level

### CSV export

The viewer exports the persisted project-backed training ROI spectra:

```python
viewer.export_csv('data/roi_train.csv')
```

---

## 9. Dark / white reference loading

In a project, references belong to each data file. **Add Data File...**
(`HSIViewer.add_data_file`, dialog `_data_files.AddDataFileDialog`) takes
them from reference files, from a frame range of the data file, or from
another data file of the project, and writes the file and its calibration
through `Project.add_dataset`. Showing a data file (`HSIViewer.show_data_file`,
loaded in the background) restores its calibration into `ViewerState`
(`set_calibration`, then the signal unit). The startup options below remain
for a quick look without a project.

Pass reference files at launch time:

```python
viewer = open_viewer(
    'data/sample.eiff',
    dark='data/dark.eiff',
    white='data/white.eiff',
)
viewer.state.set_y_unit('Reflectance')
```

References can also be set programmatically after launch:

```python
import numpy as np
from enginir.io import load_spectra

dark_cube, _, _ = load_spectra('data/dark.eiff')
white_cube, _, _ = load_spectra('data/white.eiff')

viewer.state.set_dark_ref(dark_cube.mean(axis=0).astype(float))
viewer.state.set_white_ref(white_cube.mean(axis=0).astype(float))
viewer.state.set_y_unit('Absorbance')
```

When references are loaded, the per-pixel formula
`clip((raw − dark) / (white − dark), 1e-6, 1)` is used instead of the
default scalar levels.

---

## 10. Log panel

A **Log** dock widget is automatically added at the bottom of the viewer
window.  It captures all `enginir` log messages at DEBUG level and above
in real time, displayed in a scrollable monospace text area with a
1000-line ring buffer.

The log panel shows:

- **INFO**: file loaded, model fitted, project saved
- **WARNING**: skipped corrupt chunk, fallback calibration
- **DEBUG**: preprocessing steps applied, cache invalidation, crosshair moves

The panel can be hidden via the viewer's **View** menu. To enable debug logging
outside the viewer (e.g. in a script or notebook):

```python
import enginir
enginir.enable_logging("DEBUG")  # DEBUG+ to stderr
```

See the [Logging user guide](user_guide/logging) for the full reference.

---

## 11. Extending the viewer

To add a plot to a workspace, write a `PlotSpec` (`render`, `table`), register
it and give it a slot on the workspace's pages. It then gets the header,
maximize and export, and `tests/test_plot_registry.py` checks it. The steps
are in `docs/viewer_architecture.md` under *Adding a new plot*.

For a surface that is not a plot, compose a normal Qt widget, connect it to
`ViewerState` / `ProjectState` signals, and dock it from `HSIViewer`.
`_BasePanelWidget` is a private helper for the built-in matplotlib panels, not
the public extension contract.

```python
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class MyPanel(QWidget):
    def __init__(self, state):
        super().__init__()
        self._state = state
        self._label = QLabel()
        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        state.frame_changed.connect(self._refresh)
        state.units_changed.connect(self._refresh)
        self._refresh()

    def _refresh(self):
        self._label.setText(f"Frame {self._state.frame}")
```

Keep custom surfaces behind project/viewer state methods rather than reading
raw arrays directly from child widgets.

---

## 12. Limitations

- **Axis convention:** `ViewerState` expects cubes shaped
  `(n_frames, height, n_bands)` — the push-broom layout from EVK EIFF files
  where axis 0 is the scan direction (frames), axis 1 is the spatial
  cross-track (rows), and axis 2 is the spectral axis (bands).  ENVI cubes
  shaped `(rows, cols, n_bands)` will load without error but the
  spatial/temporal axes will be misinterpreted (rows treated as frames, cols
  as spatial height).  To use ENVI data in the viewer, transpose to match
  the expected layout first.
- **Single processed snapshot:** The viewer computes preprocessing
  on-the-fly from the raw cube.  Only one set of preprocessing settings is
  active at a time.
