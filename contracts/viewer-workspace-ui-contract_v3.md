# Viewer Workspace UI Contract v3

## Status

- Version: `v3`
- Status: active contract for engiNIR 1.0; v3 is the default and only supported project schema
- Scope: viewer workspace structure, phase-specific panels and controls, global status/progress behavior, project-only vs dataset-attached behavior

## Goal

Define the concrete UI contract for the v3 viewer workspace so that:

- phase navigation is intentional rather than accidental
- each workflow phase exposes the right plots, diagrams, and controls
- project-only and dataset-attached states are visible and enforced consistently
- long-running services show progress while the UI remains responsive
- future UI refactors do not reintroduce ambiguous mixed workflows

This contract exists because the viewer is now a scientific workspace UI, not
just a generic image shell with extra buttons.

## Core Decisions

- the viewer has three first-class workflow workspaces:
  - `Explore`
  - `Model`
  - `Predict`
- v3 primary navigation is tab-based
- workspace structure is contract-level behavior, not a styling detail
- each workspace has one primary task and one primary content region
- global context must remain visible regardless of workspace
- a project object browser is always available
- project-only mode is valid for inspection, but not for dataset-derived actions
- long-running service actions must never block the UI thread
- heavy workspace redraws must be scoped to visible surfaces only

## Problem This Contract Solves

Without an explicit workspace UI contract, the system cannot answer reliably:

- which controls belong to exploration vs modeling vs prediction
- which plots must be visible in each phase
- whether a project-only window may expose ROI/model/prediction actions
- where service status and progress belong
- whether a layout change is a harmless redesign or a behavior regression

The result is recurring UI ambiguity:

- unrelated controls appearing together in one dock
- invalid workflows being discoverable only through backend failures
- missing progress feedback during fitting and prediction
- project-only vs dataset-attached state being unclear

## Domain Summary

The UI is built around five distinct concepts:

1. `Global Viewer Context`
   State that must remain visible across all workspaces.
2. `Project Browser`
   Persistent navigation and inspection of project-owned objects.
3. `Explore Workspace`
   Dataset inspection, signal selection, preprocessing preview, ROI creation/export.
4. `Model Workspace`
   Target selection, sample-set build, signal-domain choice, model fit.
5. `Predict Workspace`
   Saved model selection, dataset choice, compatibility-aware prediction.

The UI contract is separate from the persistence/modeling contracts, but must
honor them.

## Dependencies

This contract depends on:

- `project-dataset-attachment-contract_v3.md`
- `dataset-calibration-contract_v3.md`
- `roi-target-contract_v3.md`
- `sample-set-model-run-contract_v3.md`
- `prediction-runtime-contract_v3.md`

If those contracts and this document conflict, workflow validity wins over
legacy UI convenience.

## Top-Level Workspace Model

v3 requires exactly three explicit user-visible workspaces:

1. `Explore`
2. `Model`
3. `Predict`

Required implementation form:

- top-level tabs

Accepted labels:

- `Explore`, `Model`, `Predict`

Alternative labels are not part of the v3 contract unless a future contract
revision replaces them.

Forbidden implementation forms:

- burying all phase controls in one undifferentiated controls dock without an
  explicit active-workspace concept
- exposing `Model` and `Predict` as merely optional button clusters inside the
  `Explore` control area
- using free-form docking or layout presets as the primary navigation model
- making the active workspace implicit

Rationale:

- tabs provide the clearest mental model for phase-based scientific work
- the user should always know which phase they are in
- free-form dock arrangements are too weak as the primary contract because they
  do not define task hierarchy

## Global Viewer Context

The following must remain visible in all workspaces:

- current project identity or `none`
- current dataset identity or `none`
- current signal unit
- current cursor position:
  - frame
  - row
  - band

v3 accepted presentation:

- one compact global status strip

If the viewer is in project-only mode, that strip must make the missing dataset
state obvious.

## Global Frame Layout

v3 requires one stable high-level frame shared by all tabs:

1. top status strip
2. left project browser
3. center primary workspace area
4. right contextual inspector / secondary views
5. bottom log area, collapsed or hidden by default

This is a behavioral contract, not a pixel-perfect styling rule.

The user must not need to rediscover the basic window anatomy when switching
workspaces.

## Project Browser

v3 requires a persistent project browser in the left column or left sidebar.

Minimum required object groups:

- datasets
- calibrations
- ROIs
- targets
- sample sets
- model runs
- prediction runs

Minimum required behavior:

- selecting an object updates the current inspector/context view
- the browser works in project-only mode
- the browser does not itself create scientific objects implicitly

Forbidden behavior:

- hiding the project object graph behind modal dialogs only
- requiring users to remember ids without a persistent object list

## Window Modes

The UI must reflect the workflow states defined by
`project-dataset-attachment-contract_v3.md`:

### Dataset-only

Allowed:

- Explore inspection
- project creation/open-and-attach workflows

Forbidden:

- persisted ROI/model/prediction actions before a project is attached

### Project-only

Allowed:

- inspect project-owned ROIs, targets, sample sets, model runs, predictions
- navigate across `Explore`, `Model`, and `Predict` workspaces if those views
  are inspection-safe

Required UI behavior:

- dataset-derived controls must be disabled or hidden
- the reason must be visible in the UI, not only in a backend error

Required message shape:

- a short actionable hint that names the missing precondition (an attached
  dataset) and the blocked action class
- reference phrasing, currently used by the implementation:
  `Attach a dataset to draw ROIs or run dataset-backed actions.`
- the exact wording is not contract-frozen; the presence, placement, and
  actionability of the hint are

### Project-attached + dataset-attached

Allowed:

- full workflow

Required UI behavior:

- all workspace controls may be enabled subject to their own local preconditions

## Identity In The Viewer

Stable ids are the only identity of project objects
(`project-schema-contract_v3.md` §2–3); names are labels. In the viewer, the
`enginir-viewer` tool and the `enginir.viewer` module:

- every selection, link, write and export refers to objects by id: the
  browser, the ROI table, the Project ROIs tab, the canvas, the Model plots,
  the fit dialog and the Predict workspace carry ids, never names
- the viewer never resolves a name to an object; it does not call the
  name-based facade functions (`Project.roi`, `.target`, `.model`,
  `set_value`, `fit`, `Dataset.add_roi`, …), which exist for scripts
  (`headless-workflow-contract_v3.md`, Names and ids)
- names are display text and may repeat; the viewer creates and stores ROIs
  whose names repeat other ROIs' names without asking
- a data file is shown by its file name; when several data files of the
  project share a file name, each is shown as `file name (folder)`, with the
  full path as tooltip. This label is used wherever a data file is shown: the
  data file list, the ROI table, the Project ROIs tab, the fit dialog, the
  browser and messages
- where ROI names may repeat, each ROI is told apart by more than its name:
  the ROI table shows its data file label, the mosaic groups ROIs by data
  file, the browser shows the id (data file label as tooltip), the inspector
  shows id and data file, and Model plots select ROIs by id
- every export of ROI data carries the ROI id next to its name, including
  Explore's ROI spectra export

A test scans `enginir.viewer` and fails on calls of the name-based facade
functions.

## Workspace Contracts

Each workspace has:

- one primary task
- one primary action area
- one preferred visual emphasis

The goal is not to expose every control everywhere. The goal is to expose the
right controls and views at the right phase.

## 1. Explore Workspace

Purpose:

- inspect the data files of the project
- choose signal/preprocessing/axis settings
- create and review ROIs, including all ROIs stored in the project

Primary task:

- move from raw dataset context to validated ROI selection

Primary content region, as two tabs:

1. `Data file`: the spatial image of the shown data file, with the ROI
   rectangles of that data file only
2. `Project ROIs`: all ROIs stored in the project, redrawn from the project
   file without any data file (the ROI mosaic)

Secondary content region:

- in `Data file`: sensor frame, spectral profile, spatial profile, histogram,
  DC3 / band slice
- in `Project ROIs`: the mean spectra of the shown ROIs and the pixel spectra
  of the selected ROI, both in the current signal unit and preprocessing
  preview, converted with each ROI's own stored references

Required visual elements:

- spatial image
- histogram
- DC3/band-slice view
- sensor-frame view
- spatial profile
- spectral profile
- ROI mosaic: one tile per stored ROI, grouped by data file, with name and
  current target value; tiles in a compact grid (ROI grid positions, else
  scan order) or at their scan positions; text in mosaic units in reserved
  room, so it never covers a tile; outlines in the value colour of the ROI
  mean spectra; contrast stretched over all tiles or per data file
- ROI mean spectra coloured by the current target value
- pixel spectra of the selected ROI

Required controls:

- project open/new
- add data file (with its calibration) and choose the shown data file
- signal unit selection
- filtering/preprocessing preview controls
- axis controls
- ROI creation, single and as a grid (the grid dialog opens with the Options
  defaults and shows the cells on the image, outside cells in red); both dialogs have a **Prediction only**
  checkbox that sets the ROIs' `use` and remembers the last choice per data
  file (user settings); prediction ROIs are marked in the ROI table (a Use
  column), the browser, the mosaic (dashed outline) and the ROI spectra
  (dashed lines, still coloured by value); counts of ROIs with a value name
  training and prediction ROIs separately
- ROI table (sample labels and target values)
- ROI export

Required behavior:

- a project with ROIs opens on `Project ROIs`; that tab works in project-only
  mode and with every data file missing
- the docks that show the data file at the cursor (histogram, DC3/band
  slice, sensor frame, spatial profile) are shown only in `Data file`; in
  `Project ROIs`, Model and Predict they are hidden, and returning to
  `Data file` restores the ones the user had shown
- the ROI table is shown with `Project ROIs` (the right dock column alone)
  and not in `Data file`; opened there, it takes the sensor frame's place as
  a tab. The user's choice per tab is kept; Model and Predict leave it as it
  is. It shows the ROIs of the page (the shown data file in `Data file`, the
  tab's data file filter in `Project ROIs`, all data files in Model and
  Predict)
- selecting an ROI in the mosaic, the ROI table, the browser or a Model plot
  selects it everywhere; opening an ROI from the mosaic shows its data file in
  `Data file` with the ROI highlighted
- mosaic tiles and ROI spectra load off the UI thread; the first tiles appear
  before all ROIs are read
- the mosaic and the ROI spectra are plot panels (Plot Panel Policy)
- the signal selection (DN, Reflectance, Absorbance) shows the signal in use
  and enables only the signals the shown data file's calibration can give
  (`dataset-calibration-contract_v3.md`, Signal units of a calibration); a
  disabled signal says why. A signal the user chose is kept when another data
  file is shown, as long as that file can give it; otherwise the file's
  working signal is shown

Required browser emphasis:

- ROIs
- datasets
- calibrations

Forbidden controls in Explore:

- model fit controls
- prediction dataset/model selection controls

v3 note:

- ROI creation belongs here, not in Modeling

## 2. Model Workspace

Purpose:

- define targets
- assign target values
- fit models (the fit builds or reuses the sample set)

Primary task:

- move from labeled ROIs to one validated model run

Primary content region:

- model summaries and diagnostics

Secondary content region:

- target coverage
- sample-set summary
- retained ROI context

Required visible information:

- selected target
- current sample set or `none`
- current active model or `none`
- service status for sample-set build / fit

Required controls:

- target selection/creation
- target assignment and sample label editing (the ROI table)
- fit model: one dialog with target, signal unit, preprocessing, model,
  components (a number, or **Auto** with a rule, the noise guard (on by
  default) and a maximum: `sample-set-model-run-contract_v3.md` §7.4; the
  dialog and the Model readiness line warn, before fitting, when the chosen
  grouping gives fewer than 7 fold groups or a data file's ROIs share one
  reference value while they are held out in different folds (the fold
  warnings of §7.4), cross-validation and which
  ROIs are held out together (each
  ROI, same sample label, same target value;
  `sample-set-model-run-contract_v3.md` §4.1), and a Fitting group: pixels
  per ROI with seed (at most the smallest training ROI with a value), X and Y scaling
  (Y for PLS only), and DModX pixel exclusion with level, rounds, the
  largest share per ROI and unweighted or modelling-power weighted
  residuals (§7); it builds or reuses the sample set of all training ROIs
  with a value for the target, from all data files; the counts, fold
  warnings and readiness consider training ROIs only
- **Fit again...** opens the fit dialog with every choice of the selected
  model (target, preprocessing, model, components, scaling, folds, pixel
  sample, exclusion), e.g. to fit with another number of components; the
  stored recipe's AsLS fields and `sg_edges` are kept (the dialog has no
  controls for them and cannot add AsLS; a recipe with AsLS is labelled
  "AsLS")
- Savitzky-Golay in the fit dialog and in Explore drops the edge bands
  (`sg_edges = drop`, the default of new recipes); in the Explore preview
  of preprocessed spectra (spectrum, sensor frame, histogram) the dropped
  bands are missing (blank), so band indices stay those of the data file;
  the Project ROIs spectra are drawn on the recipe's output axis
- band selection (engiNIR 1.4): the fit dialog has **Keep**
  and **Exclude** band ranges (`project-schema-contract_v3.md` §4) as
  editable text (`1100-1350, 1500-2300`, in nm, or band numbers without
  wavelengths) and a **Select on plot...** dialog: the mean spectrum of the
  sample set's ROIs on the recipe's axis after Savitzky-Golay, with the
  selected model's VIP (PLS) or coefficients as a guide when one is
  selected, where dragged regions become keep or exclude ranges. The
  common NIR water bands (1400-1500 and 1900-2000 nm) are offered as
  suggestions and added to Exclude only when clicked, never automatically.
  **Fit again...** keeps the stored ranges. In the Explore preview,
  unselected bands are missing (blank) like dropped Savitzky-Golay edges
- **Propose bands...** (PLS and PCR, engiNIR 1.4) runs the forward iPLS proposal with nested validation
  (`headless-workflow-contract_v3.md`, Band proposal) on the chosen
  target and recipe, with its cost estimate, progress and cancel; it shows
  the proposed ranges, the RMSECV over the steps, the nested RMSECV and the
  RMSECV with all bands, and can put the ranges into the fit dialog as keep
  ranges. The proposal is not stored
- the inspector shows how a model run was fitted: folds and fold groups,
  pixel sample, scaling, exclusion settings, excluded pixels per ROI,
  flagged ROIs and DCrit

Required browser emphasis:

- targets
- sample sets
- model runs

Required preconditions surfaced in UI before running:

- project open
- at least one ROI
- selected target
- enough labeled ROIs for the chosen validation strategy, with the count of
  labeled ROIs per data file
- compatible signal-domain/calibration context

Forbidden controls in Model:

- ROI drawing controls
- raw dataset import/open controls that are unrelated to model preparation
- prediction execution controls

v3 note:

- Explore plots may remain accessible as secondary context, but must not remain
  the dominant visual focus of this workspace

## 3. Predict Workspace

Purpose:

- choose saved model runs, by name
- choose prediction ROIs (Predict ROIs) or a project data file (Predict map)
- run prediction and read the per-ROI results

Primary task:

- predict the tablets (prediction ROIs) of one or all data files with one or
  more models, and inspect the per-ROI results and the map

Primary content region:

- prediction result visualization

Secondary content region:

- model/dataset/calibration compatibility summary
- prediction metadata and run summary

Required visible information:

- active or selected model run
- the latest results of this session or `none`: the per-ROI table of the
  last Predict ROIs and the map of the last Predict map (predictions are
  not stored; earlier stored predictions in a project are not shown)
- service status for prediction

Required controls:

- **Predict ROIs**: one or more models (by name) on the prediction ROIs of
  one data file or of all data files (`prediction-runtime-contract_v3.md`,
  Prediction ROIs); needs only the project file
- **Predict map**: one model on one project data file, every pixel, with
  the per-ROI results of that file's prediction ROIs
- the data file is a project data file; **Add data file...** opens the Add
  data file dialog (calibration chosen there, never reused from training
  silently; the dialog shows what the model was trained with). There is no
  free path, dataset id or calibration field in the prediction dialog
- the dialog shows the chain read-only, from the model: the data file's
  calibration, the model's signal unit, its recipe and the model; an
  incompatible file is refused before running, with the reason
- **Remove pixels above the DModX limit** with its level, defaulting to the
  model's fit specification; the per-ROI results follow without predicting
  again

Required prediction-map interaction:

- map axes must match the main spatial image orientation: `Frame` on x and
  `Row` on y
- a color scale/legend must be visible for numeric prediction values; the
  scale is the target's range in the model's training values, values outside
  in their own colours, or limits the user sets (⋯ menu); the colour bar and
  the CSV record the scale
- the prediction ROIs of the shown data file are drawn on the map with their
  mean; with the exclusion on, removed pixels are greyed
- a per-ROI plot (a plot panel): each ROI's mean with its 5–95 % range, the
  share of pixels above DCrit (0.99) against the training ROIs' range (like for
  like; the removed count is in the table), the reference value when
  present; a bar selects its ROI; with several models, one bar per ROI and
  model, grouped by ROI and coloured by model. Its CSV is the per-ROI table
  in long format, one row per ROI and model (ROI name and id, data file,
  model name and id, pixels, removed, mean, median, SD, 5 %, 95 %, mean
  DModXPS and T²PS, shares above the limits, reference, residual), with the
  exclusion and its level, and RMSEP, bias and *n* per model, as metadata
- the pixel histogram shows the kept pixels of the prediction ROIs (all
  pixels of the map when the data file has no prediction ROI)
- the summary names the model and data file (never ids) and groups the
  chain, the scan and the applicability
- multi-output prediction artifacts must expose an explicit target/output
  selector before the map is rendered
- a layer selector shows the prediction, the distance to the model (DModXPS)
  or Hotelling T² (T²PS); on these two the pixels above the 0.99 limit are
  marked and the share above is shown (`prediction-runtime-contract_v3.md`,
  Applicability Diagnostics)
- the Predict workspace writes nothing to the project: predictions and
  per-ROI results are not stored; the plot panels export them. A scan to
  predict is a project data file the user added beforehand
- pan and zoom must be available for the prediction map
- the map is a plot panel (see Plot Panel Policy). Its CSV export is the
  displayed output as a matrix, with one CSV row per map row and one column
  per frame.
- a subsampled preview labels its axes, hover readout and CSV with the true
  frame and row indices, and the CSV records the subsampling step
- clicking or hovering the map must expose frame, row, and predicted value
- map interaction must not hydrate unrelated heavy artifacts
- repeated map refreshes for unchanged payloads must be no-ops

Recommended prediction-map interaction:

- link prediction-map frame/row selection to the main spatial image crosshair
- show ROI overlays when ROI geometry exists for the displayed dataset
- use a downsampled preview for very large maps and load full-resolution data on
  demand

Required browser emphasis:

- model runs
- prediction runs
- datasets

Required preconditions surfaced in UI before running:

- project open
- at least one model run
- compatible dataset/calibration/signal-domain contract
- supported prediction output dtype
- recipe-aware predictor or saved artifact path capable of replaying the recipe
- persisted training-side preprocessing state when the recipe contains `MSC`,
  `EMSC`, or future stateful transforms
- explicit output selection when a model emits multiple prediction targets

Forbidden controls in Predict:

- ROI drawing (prediction ROIs are drawn in Explore)
- target assignment editing
- model fit controls

## Plot And Diagram Policy

v3 requires that the visible scientific panels match the active workspace.

This does not require destroying and recreating widgets on every switch.
It does require that the visible hierarchy changes by workspace.

### Explore

Must prioritize:

- spatial image
- sensor frame
- spectral profile
- spatial profile
- histogram
- DC3/band slice

### Model

Must prioritize:

- model diagnostics
- target/sample-set/model summaries
- fit readiness / validation context

Minimum acceptable v3 state:

- if full diagnostics are not yet implemented, the workspace must still give
  visual priority to model-related summaries and statuses over Explore plots

### Predict

Must prioritize:

- prediction status
- model/dataset/calibration context
- prediction result visualization

Minimum acceptable v3 state:

- if full result dashboards are not yet implemented, the workspace must still
  give visual priority to prediction-related summaries and statuses over
  Explore plots

## Plot Panel Policy

This section applies to every scientific plot in the viewer. That includes the
Explore plots and the spatial image, the Model diagnostics, the prediction map,
and every plot added later.

### Panel

Each plot is shown in its own plot panel with a header. The header contains:

- the plot title (the title is not drawn a second time inside the plot)
- a help control (**?**) when the plot has a help text; required for the
  Model diagnostics (`model-workspace-diagnostics-contract_v3.md`)
- a maximize/restore control
- an export menu

### Maximize

- In a grid of panels, maximize makes the panel fill the workspace's primary
  content region in place.
- In an Explore dock, maximize enlarges the dock to the main-window area.
- Restore uses the same control, a double-click on the header, or `Esc`. A
  docked panel returns to its dock position.
- Maximize and restore must not change:
  - the active workspace
  - the selected objects
  - the data shown by other panels
- Restoring must not redraw panels whose payload is unchanged.

### Export

Every plot panel offers these exports:

- **PNG** at 300 dpi or more, and **SVG**:
  - The export contains the same data and the current axis limits as the
    panel.
  - It is rendered in a print style (light background, dark text), whatever
    the screen theme.
- **CSV** of the plotted data, exactly as drawn: one row per point, bin or
  band, and a frame × row matrix for images and maps.
  - Column names carry units where the plot has units.
  - ROI rows carry both the ROI name and its stable id.
  - The file starts with `# key: value` provenance lines:
    - always: the plot id, the enginir version, the export time (ISO 8601) and
      the project id
    - when they apply: the dataset id, the signal unit, and the sample-set,
      recipe, model-run and prediction-run ids
    - for model plots also: the component count and the CV scheme

`File > Export Plots...` writes PNG and CSV of every plot of the active
workspace that shows data into one folder:

- Files are named by plot id, and plots without data are listed as skipped.
- Pages that were not opened yet are included.
- Replacing files, or writing more than 1,000,000 values in total, needs
  confirmation first.

Required behavior:

- export of an empty plot is disabled, with the reason visible
- an export of more than 1,000,000 values asks for confirmation before writing
- export failures are user-visible errors and are logged
- export never writes into the project file and never changes project objects

### Layout

- In `Model` and `Predict`, the primary content region is a fixed grid of plot
  panels. The grid may be organised in named pages, shown as sub-tabs inside
  the workspace.
- Panels snap into the slots their page defines. Free-floating plot windows
  are not the layout model, and the top-level tabs remain the primary
  navigation.
- Users can maximize and export panels. They cannot change which plot occupies
  which slot (see the model diagnostics contract, non-goals).
- The maximized panel of each workspace page is UI state. It is stored in
  the user settings, never in the project file, and `Reset layout` restores
  the defaults.
- A maximized dock and the active page are not stored: a dock maximize is
  transient, and a new model run opens the `Fit` page.

### Redraw

- A page builds its panels the first time it is shown.
- A panel redraws only while visible, and only when its payload changed.
- Changing one plot (for example its mode or output selection) must not redraw
  the other panels.

### Extensibility

A new plot is added as a plot-panel definition. It must provide:

- its rendering
- its empty state
- PNG/SVG export
- CSV export

A test that iterates over all registered plots verifies this.

## Long-Running Service Contract

The following actions are long-running service actions in v3:

- fit model (including building its sample set)
- run prediction

Required behavior:

- start work off the UI thread
- keep the UI responsive
- show visible in-progress status
- show an indeterminate or determinate progress animation
- prevent duplicate concurrent submission of the same action from the same view
- surface failures as user-visible errors

Required visible status shape:

- a phase-local service status line
- a visible busy indicator while the action runs

Recommended v3 pattern:

- phase-local indeterminate progress bar
- short status text such as:
  - `Building sample set...`
  - `Fitting model...`
  - `Running prediction...`

Additional prediction requirement:

- prediction should use determinate progress when chunk totals are known
- prediction status must expose at least these phases:
  - `Preflight`
  - `Loading`
  - `Calibrating`
  - `Preprocessing`
  - `Predicting`
  - `Writing`
  - `Finalizing`
  - `Rendering`
- prediction failures must preserve the failed phase in the user-visible error
  or service status

Forbidden behavior:

- freezing the whole window during model fit or prediction
- silent background failure
- requiring users to inspect logs to learn whether the action is still running
- showing only a single undifferentiated `Running prediction...` status for a
  long chunked prediction job

v3 performance rules for long-running actions:

- the primary workspace summary page must mirror the same busy/progress state as
  the local workflow controls
- repeated status refreshes should be coalesced rather than redrawing the same
  workspace page immediately for every signal

## Action Availability Contract

Each workspace must expose only actions that are valid in the current window mode.

Required invariant:

- invalid actions must be either:
  - disabled with an explanatory hint, or
  - hidden when their presence would imply an invalid workflow

Forbidden invariant:

- a clearly invalid action remaining fully enabled and failing only after a long
  backend path

## Preflight Visibility Contract

Before launching a long-running action, the UI must surface the relevant
readiness state in the active workspace.

Examples:

- number of labeled ROIs for the selected target
- whether the selected validation strategy is possible
- whether the chosen calibration can satisfy the requested signal unit
- whether a model run exists and is compatible for prediction
- whether prediction can replay the preprocessing recipe before a worker is
  launched
- whether the selected calibration will clip reflectance values, after the run
  has enough data to report counts

v3 rule:

- readiness may be shown as short status lines, warnings, or inline validation
- it must not be hidden only in backend exceptions or logs
- preflight-detectable prediction failures must be shown before job submission,
  including bare-estimator predictors, unsupported dtypes, missing stateful
  preprocessing state, and executable dataset/calibration/model incompatibilities

## Redraw And Visibility Contract

v3 requires explicit redraw discipline for heavy UI surfaces.

Required behavior:

- only the active workspace may redraw its heavy primary widget
- hidden Explore dock panels should defer expensive rebuilds until shown again
- repeated shell refresh requests should be coalesced into one event-loop refresh
- unchanged diagnostics/map payloads should be treated as no-op redraws
- Predict map rendering must prefer preview or sliced artifact reads over
  hydrating complete large prediction arrays

Allowed v3 behavior:

- default-visible Explore panels may perform one initial build before the main
  window is shown so guides and labels exist immediately after viewer
  construction
- small prediction maps may be fully materialized for interaction when they fit
  the configured preview threshold

Forbidden behavior:

- redrawing Model diagnostics and Predict map surfaces while those workspaces
  are hidden
- recomputing hidden Explore plots on every frame/row/band/unit change
- forcing full-figure redraws for identical payloads

## Navigation Contract

Switching workspaces must preserve:

- current project
- current dataset attachment
- current cursor/frame/row/band
- selected target where applicable
- active model where applicable

Switching workspaces must not:

- implicitly create datasets, calibrations, ROIs, sample sets, models, or predictions
- silently reset the current scientific context

## Simplicity Principle

v3 favors progressive disclosure over maximal simultaneous visibility.

Required interpretation:

- power comes from clear phase separation and persistent context
- power does not require showing every control and every plot at once

Therefore:

- one active workspace must have a clear primary focus
- secondary panels are allowed
- “everything visible at once” is not a valid success criterion

## v3 Minimum Acceptance Criteria

The UI satisfies this contract only if all of the following are true:

- there is an explicit visible active-workspace concept
- the active workspace is one of `Explore`, `Model`, or `Predict`
- top-level navigation is tab-based
- there is a persistent project browser
- project-only vs dataset-attached state is visible
- dataset-derived actions are disabled or blocked before invalid execution
- fitting and prediction show non-blocking in-progress feedback
- prediction preflight blocks recipe/predictor/dtype/source/calibration
  incompatibilities before launching a long-running job
- multi-output prediction maps require an explicit output selection and never
  silently show only the first output
- prediction run summaries surface calibration clipping telemetry when clipping
  occurred
- the global status strip remains visible across all workspaces
- Explore, Model, and Predict controls are not collapsed into one
  undifferentiated phase-less control area
- the active workspace has a clear primary content region
- every scientific plot is a plot panel that can be maximized and restored and
  exported as PNG, SVG and CSV, with the CSV provenance header

## Out Of Scope For v3

- final visual styling/theme decisions
- exact typography and iconography
- advanced docking persistence rules
- multi-window coordination semantics beyond current project/dataset attachment rules
- a final dedicated modeling-diagnostics dashboard
- a final dedicated prediction-results dashboard

Those may evolve, but they must not violate the workspace/action rules above.
