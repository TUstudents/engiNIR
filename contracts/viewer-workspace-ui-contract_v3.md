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

## Workspace Contracts

Each workspace has:

- one primary task
- one primary action area
- one preferred visual emphasis

The goal is not to expose every control everywhere. The goal is to expose the
right controls and views at the right phase.

## 1. Explore Workspace

Purpose:

- inspect the current dataset
- choose signal/preprocessing/axis settings
- create and review ROIs

Primary task:

- move from raw dataset context to validated ROI selection

Primary content region:

- spatial image

Secondary content region:

- sensor frame
- spectral profile
- spatial profile
- histogram
- DC3 / band slice

Required visual elements:

- spatial image
- histogram
- DC3/band-slice view
- sensor-frame view
- spatial profile
- spectral profile

Required controls:

- project open/new
- signal unit selection
- filtering/preprocessing preview controls
- axis controls
- ROI creation
- ROI export

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
- build sample sets
- fit models

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
- target assignment editing
- build sample set
- fit model

Required browser emphasis:

- targets
- sample sets
- model runs

Required preconditions surfaced in UI before running:

- project open
- at least one ROI
- selected target
- enough labeled ROIs for the chosen validation strategy
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

- choose a saved model run
- choose a prediction dataset
- choose or reuse calibration
- run prediction

Primary task:

- run one saved model against one dataset and inspect the result

Primary content region:

- prediction result visualization

Secondary content region:

- model/dataset/calibration compatibility summary
- prediction metadata and run summary

Required visible information:

- active or selected model run
- latest prediction run or `none`
- service status for prediction

Required controls:

- prediction dataset selection
- model-run selection
- calibration choice
- prediction execution

Required prediction-map interaction:

- map axes must match the main spatial image orientation: `Frame` on x and
  `Row` on y
- a color scale/legend must be visible for numeric prediction values
- multi-output prediction artifacts must expose an explicit target/output
  selector before the map is rendered
- pan and zoom must be available for the prediction map
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

- ROI drawing
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

## Long-Running Service Contract

The following actions are long-running service actions in v3:

- build sample set
- fit model
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

## Out Of Scope For v3

- final visual styling/theme decisions
- exact typography and iconography
- advanced docking persistence rules
- multi-window coordination semantics beyond current project/dataset attachment rules
- a final dedicated modeling-diagnostics dashboard
- a final dedicated prediction-results dashboard

Those may evolve, but they must not violate the workspace/action rules above.
