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
└── _viewer.py       HSIViewer(QMainWindow) — shell, menus, docks, log, child windows
```

All panels read from and write to `ViewerState` through typed setters and
Qt signals.  No panel holds its own copy of the data.  Preprocessing is
computed inside `ViewerState.current_frame_image()` so every panel always
displays the same processed signal.

The central canvas has a `Spatial Image` header with a `Reset View`
control that resets zoom and pan to the full image bounds.

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

Displays a raw cross-section of the cube at the selected band:
`cube[:, :, band].T` — spatial rows on the Y-axis, scan frames on the
X-axis.  Yellow crosshairs mark the active `(frame, row)`.

Note: RAW always shows raw DN (no preprocessing) to provide a stable
spatial navigation reference.

- Updates on: `frame_changed` (vline only), `row_changed` (hline only),
  `band_changed` (full rebuild)
- **Click** → emits `frame_pick(int)`, `row_pick(int)`

### Sensor Frame panel (internal class `SensorPanel`, right dock, top)

Displays the processed current frame: `current_frame_image()` — height ×
n_bands.  Cyan crosshairs mark the active `(row, band)`.

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

Target assignment remains a separate modeling action.

This means ROI labels are display-only and do not need to be unique.
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
- `Predicted vs Reference`, `Residuals`, and `Target Distribution`
- `Scores (LV1 vs LV2)`
- a spectral diagnostics plot controlled from the Model diagnostics side panel's
  `Spectral plot` selector

When no fitted run is available yet, the readiness message remains visible
above the empty diagnostics placeholders so the blocking condition is still
obvious.

### Modeling group

| Control | Action |
|---|---|
| **Target...** | Select one existing target or define a new target |
| **Assign target** | Assign the active target to one ROI |
| **Build sample set** | Persist one v3 sample set from labeled ROIs using the active preprocessing recipe |
| **Fit model** | Persist one `SignalDomainContract` and grouped v3 `ModelRun` |
| **Spectral plot** | Switch the diagnostics spectral plot between regression coefficients and LV1 loadings in the Model diagnostics side panel |

The Modeling section operates from project state. Sample sets are built from
persisted ROIs plus `TargetAssignment`s, and model fitting uses ROI-grouped CV
with ROI-level primary reporting.

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
- the persisted prediction map surface

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
   provided, it is persisted separately as a `TargetAssignment`.

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

To add a custom viewer surface, compose a normal Qt widget, connect it to
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
