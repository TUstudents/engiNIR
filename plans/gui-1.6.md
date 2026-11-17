# Plan: engiNIR 1.6, the viewer as a guided professional desktop program

Status: decided (user, 2026-11-16): decisions D1–D9 and the proposed
answers to Q1–Q6 are accepted. Branch `gui-1.6`. Phase 1 done (library:
`enginir._guidance`; the verdict thresholds accepted by the user, Q3).
Phase 2 done (theme tokens in `viewer/_theme.py`; the icon set moves to
Phase 3, where the menus need it).
Phases 3 and 4 done together (4105128), Phase 5 (2948c81), Phase 6 (preferences
in `preferences.json` in the viewer's settings folder like its other settings,
not `QSettings`: one place, isolated in tests).
Phase 7 (89cae59). Phase 8 (docs, contract, screenshots) done; the release is
not made (user, 2026-11-16: everything but the release).
Input: `plans/gui-review-1.6.md` (22 findings on the 1.5.2 viewer) and the
mockup (local, not committed: `review/gui-1.6/mockup2.html` and the
`desk_*.png` screenshots). The mockup shows layout, density and palette;
it is not a specification of every control.

## Goal

One viewer for two users. A novice who installed engiNIR from the Start
menu is guided from scans to a prediction and always knows the next step.
A PAT scientist sees every option without opening a dialog, and can trust
and trace every number. The program looks and behaves like professional
desktop software: menu bar, dock panels, dense controls, a preferences
dialog, a consistent palette taken from the program icon, in a dark and a
light theme.

## Decisions (from the brainstorm, user 2026-11-13 to 2026-11-16)

- **D1 Desktop look, not a web app.** Menu bar, dock panels with title
  bars, a project tree, property grid, tabbed pages, table panels,
  segmented status bar; compact controls with square corners.
- **D2 No toolbar.** Every action is a menu item (with a shortcut). The
  primary action of a workspace is the button at the foot of its settings
  dock. The workflow strip is navigation, not a toolbar.
- **D3 One interface, no novice/expert mode.** Guidance is a layer
  (workflow strip, verdict bar, Checks panel) that can be switched off in
  Preferences; all options stay visible in the settings dock.
- **D4 Screen sizes.** Designed for 1920 × 1080. Minimum 1536 × 864
  without loss of function (narrower docks, lower bottom panel, the
  Advanced group collapsed). Smaller screens use the interface scale
  (Preferences, e.g. 90 %), not a third layout.
- **D5 The bottom Controls dock goes away.** Its controls move into the
  right settings dock of each workspace.
- **D6 Guidance comes from the library.** Checks, verdicts and their fixes
  are computed by `Project`, not by the viewer (the viewer holds no
  workflow rule). Scripts and later reports use the same checks.
- **D7 Palette from the icon.** Accent: the icon's orange. Warm colours
  belong to the program and to check states; data use a cool scale
  (blue to green), so plots never clash with the interface. The icon's
  yellow-to-red bands are the heat scale for maps.
- **D8 Both themes are first class.** Dark and light share one set of
  named colour tokens; every widget and plot reads tokens, never literals.
- **D9 Configuration in one Preferences dialog** (Tools › Preferences),
  stored with `QSettings`; project-specific settings stay in the project.

## The window

```
Title bar      icon, "<project> – engiNIR <version>"
Menu bar       File  Edit  View  Project  ROI  Model  Predict  Tools  Help
Workflow strip Data › ROIs › Values › Model › Predict › Report   (state per step)
┌ Project ────┬ Pages (tabs) ─────────────────────┬ Settings ──────┐
│ tree        │ verdict / next-step bar           │ groups of the  │
│             │ plots or image                    │ workspace      │
├ Properties ─┤                                   │                │
│ key : value ├ Checks │ ROI table │ Models │ Log ┤ [primary action]│
└─────────────┴───────────────────────────────────┴────────────────┘
Status bar     saved state │ scan │ signal │ model │ job and progress
```

- **Workflow strip:** six steps. A step is done (tick), needs attention
  (warning with a count) or not started (number). Clicking a step opens
  its workspace. Data, ROIs and Values share the Explore workspace with a
  different page and settings group in front; Report opens the report
  page (1.7; in 1.6 the step exports plots and is marked "coming").
- **Left dock:** Project tree (scans with their labels, reference
  corrections, ROIs, targets, models, predictions; state icons) above
  Properties (the selected object, read-only key/value grid with copy).
- **Centre:** tabbed pages per workspace; above them one bar with the
  verdict or next step and its fix button; below them the bottom panel
  with tabs Checks, ROI table, Models, Log.
- **Right dock, "settings":** collapsible groups with every option of the
  workspace, and the primary action at the foot.
  - *View settings* (Data, ROIs, Values): Scan (scan, reference, band),
    Signal (unit, detector levels), Smoothing and derivatives, Scatter
    correction, Axes, ROI (use, names, sample, add, grid, delete, export).
  - *Fit settings* (Model): Target, Signal, Smoothing and derivatives,
    Scatter correction, Bands, Model (type, components, auto rule, drop
    noise), Validation (folds, hold out together), Advanced (scaling,
    pixels per ROI, DModX exclusion, algorithm). Foot: Reset to model,
    Fit again…, Fit model. This replaces the modal fit dialog.
  - *Prediction settings* (Predict): Models, What to predict,
    Applicability, Map display. Foot: Predict map…, Predict ROIs.
- **Docks** have fixed positions and can be resized and collapsed (Q6);
  View › Reset layout
  restores the default for the current screen size. The layout is saved
  per screen-size class.

### Menus (every action lives here)

| Menu | Items |
|---|---|
| File | New project, Open project, Open example project, Recent, Add scan, Export plots, Export ROI table, Exit |
| Edit | Copy (table cells, properties), Paste (ROI table), Delete, Find ROI |
| View | Project, Properties, Settings, bottom panel tabs, Theme, Interface scale, Reset layout, Full-screen plot |
| Project | Reference corrections, Locate scans, Targets, Target values, Scan labels |
| ROI | Add ROI, ROI grid, Delete ROI, Mark as calibration / prediction, Set sample |
| Model | Fit model, Fit again, Propose bands, Compare models (1.7), Rename, Delete |
| Predict | Predict ROIs, Predict map, Cancel, Discard staged predictions |
| Tools | Preferences, Open log file, Open log folder, Install Start menu shortcut (Windows) |
| Help | User guide, What does this plot mean, Keyboard shortcuts, About |

## Guidance

- **Checks (library).** `Project.checks()` returns a list of check items:
  `id`, `step` (data, rois, values, model, predict), `severity`
  (ok, attention, blocking), a sentence, the objects it refers to, and
  an optional `fix` (a named action with arguments, e.g.
  `refit(model, group_by="sample")`, `select_roi(id)`). First checks, all
  found in the review:
  - validation optimistic (exists today as a warning text);
  - values differ within one sample;
  - reflectance clipped above 1 beyond a share of pixels;
  - pixels above the DModX limit far beyond the expected share, by scan;
  - scans without wavelengths (information);
  - ROIs without a value; too few ROIs or samples for the fold rule;
  - a model fitted before its ROIs or values changed (stale);
  - predicted ROIs outside the model.
- **Verdict (library).** `Model.verdict()` gives one of good / usable with
  care / not usable, a sentence, and the evidence items behind it
  (metrics, rule results, checks). The thresholds are written into the
  model-workspace-diagnostics contract.
- **Viewer.** The workflow strip counts open checks per step; the bar
  above the pages shows the top item of the current step with its fix
  button; the Checks tab lists all of them (check, where, fix). A fix
  button calls the same viewer action as the menu item, pre-filled.
- **Empty states:** a page without content shows one sentence and the
  button of the next action, never empty axes.
- **Welcome:** Open example project, New project (scans, references,
  target name and unit in one short dialog), Open project, recent
  projects by name with scans/models and folder.
- **Explanations:** the "?" of every plot opens three sentences on how to
  read it and a link to the user guide.

## Theme

- Tokens (both themes): window, panel, header, input, hover, selection,
  border (two levels), text (three levels), accent and on-accent, ok,
  attention, error, plot background and grid, value scale (5 stops and a
  continuous version), heat scale.
- Dark: neutral graphite (`#1c1d20` window, `#25272b` panels), accent
  `#f0912e`. Light: `#ececee` window, `#f8f8f9` panels, accent `#d2640f`.
  Check states: green, yellow, red, always with an icon.
- Value scale "engiNIR cool" (blue → cyan → green), darker stops in the
  light theme; map scale "engiNIR heat" (the icon's bands). Both
  perceptually ordered and checked for colour-blind separation; shapes or
  dashes as a second cue where series must be told apart.
- One icon set (line icons, monochrome, tinted by token), one type scale
  (12 px base, 11 px secondary, semibold headers), one spacing unit
  (4 px), control height 23–24 px.
- pyqtgraph and matplotlib plots read the same tokens; exports use the
  light palette unless the user chooses otherwise.
- Contrast: text on every surface at least 4.5 : 1 in both themes
  (tested).

## Preferences (Tools › Preferences)

Appearance (theme, interface scale, density, colour scales, guidance
switches) · Data and paths (default folders, drive mappings, recent list)
· Fitting defaults (model, components rule, folds, hold-out, scaling) ·
Prediction defaults (DModX level, interval, colour limits) · Plots and
export (format, size, palette for export) · Performance (BLAS threads,
memory limit for chunks) · Logging (level, file, open) · Keyboard.
Defaults for fits and predictions only pre-fill the settings docks; a
project never depends on a machine's preferences.

## Phases

One branch `gui-1.6`, one commit per phase, one draft PR; all four gates
plus the Windows jobs per phase.

1. **Checks and verdict in the library.** `Project.checks()`,
   `Model.verdict()`, fix descriptors; contract text
   (model-workspace-diagnostics, headless-workflow), coverage map, API
   docs. Tests per check with an injected fault. No viewer change.
2. **Theme tokens.** Token table for both themes; QSS generated from it;
   plot colours from tokens; icon set; contrast test. The current layout
   keeps working with the new palette.
3. **Window shell.** Menu bar with all actions, workflow strip, left dock
   (tree, Properties), bottom panel tabs, status bar; layout classes for
   1920 × 1080 and 1536 × 864; saved layout. The Controls dock still
   exists, hosted in the right dock.
4. **Settings docks.** View settings, Fit settings (replacing the fit
   dialog; one preprocessing recipe shared by preview and fit, "Reset to
   model" loads a model's settings), Prediction settings. The Controls
   dock is removed.
5. **Guidance in the viewer.** Strip states, verdict bar, Checks tab with
   fixes, empty states, plot explanations, the welcome screen and the
   example project.
6. **Preferences.** Dialog, `QSettings`, interface scale, defaults that
   pre-fill the docks.
7. **Review defects** not already covered (review part A: axis without
   wavelengths, ROI names cut off, truncated titles, scan labels, recent
   list, "Prediction ROIs: none").
8. **Docs and release.** User guide rewritten around the workflow with
   new screenshots (both themes), `viewer_architecture.md`, API docs,
   viewer-workspace-ui contract; walk-through on the demo project by the
   user at both screen sizes; release 1.6.0.

Tests for the layout: offscreen renders of each workspace at both sizes
and in both themes, with checks that no label is elided and no widget
overlaps (the faults of the review), next to the existing behaviour tests.

## Not in 1.6

Model comparison table with side-by-side plots, HTML/PDF report, model
lifecycle (draft, validated, locked), batch fits, command palette,
provenance view. The Models tab in 1.6 is a plain sortable list; the
Report step is visible but marked as coming. These follow in 1.7, with
the method file of 2.0 in mind.

## Answers to the open questions (user, 2026-11-16: as proposed)

- **Q1 Example project:** a small synthetic example, generated on first
  use (a few MB, no download). The user's demo (110 MB plus scans) is not
  shipped.
- **Q2 The Fit settings dock replaces the fit dialog.** A fit starts from
  what the dock shows; "Reset to model" loads a model's settings, and a
  mark shows "changed since the last fit".
- **Q3 Verdict thresholds:** a first set is proposed in Phase 1 and
  reviewed by the user before it goes into the contract.
- **Q4 Scope:** all eight phases in 1.6, with release candidate
  `1.6.0rc1` after Phase 4.
- **Q5 Review part A defects:** folded into 1.6 (Phase 7); no 1.5.3.
- **Q6 Docks:** fixed positions, resizable and collapsible, not floating.

## Changes after the first walkthrough (user, 2026-11-16)

- The workflow strip has arrows between its steps, no step numbers, and the
  open checks of a step in a round badge.
- No Report step: the report is not defined yet (a 2.0 feature with the
  method file). Exporting plots stays a menu item.
- No guide bar: it repeated the Checks tab and cost height. The verdict is
  the first line of the Model page's fit summary.
- The ROI table gets about 40 % of the window while it is the front tab.
