# GUI review for the 1.6 concept (2026-11-13)

Status: review done; input for the GUI concept (plan to follow). No decisions yet.

Method: engiNIR 1.5.2 on a copy of the user's demo project (five tablet
scans, one per layer count 1, 4, 5, 7 and 10; 20 calibration and 5
prediction ROIs; target "Layers"; one model, PLS 6 LV with SNV). Screenshots
were taken offscreen, at 1600 × 1000, with the dark theme. They are kept locally in
`review/gui-1.6/` and not committed, because they show local data paths.
Two perspectives: a PAT scientist who builds and validates a method, and a
novice who installed the viewer from the Start menu.

## What works well (keep)

- **The diagnostics are rich and correct.** Fit, Spectral, Components and
  Distance cover what a PAT expert expects from SIMCA. These include five
  rules for choosing the number of components, DModX and T² per pixel and per ROI, and
  metrics per ROI and per pixel side by side.
- **The library already warns the right way.** "Validation likely optimistic: the ROIs
  of one dataset share one value but are held out in different folds…" is
  exactly what the demo project needs (each scan is one layer count). A
  readiness text exists ("Ready: model … is available for prediction").
- **Chains are explained in the dialogs.** The Predict ROIs dialog says "the dataset's
  reference correction → Absorbance → SNV → PLS 6 LV, SNV; fitted with
  detector levels 128 / 3500". The Fit dialog lists the calibration set per
  file.
- **The ROI mosaic and the grid tool.** The outline colour follows the target value, prediction ROIs
  are dashed, and the grid has a live preview of the names (A1…A5) and the coverage.

## Findings

### A. Defects and quick fixes (could go into 1.5.3)

1. **The wavelength axis claims nm without wavelengths.** These scans have no
   `wavelengths` dataset, so band indices 0–175 are used (logged as a warning). The
   Spectral profile still labels its axis "Wavelength (nm)" and the Axis control
   has "nm" selected. The ROI mean spectra say "Band". Without wavelengths:
   label "Band", disable "nm" with a tooltip.
2. **The ROI table cuts off the ROI names.** The table is wider than its dock and
   scrolls to the target column, so names show as "0p_L01_01" instead of
   "API_10p_L01_01". The ROI column should stay visible, or use the short
   names (the mosaic shows "L01_01").
3. **Plot titles are truncated** on every page ("Predicted vs target value, per R",
   "Each component: Q²ₐ, ΔR²X, rou"). Wrap them, shorten them, or put the detail in the
   tooltip.
4. **The Model controls overlap.** The readiness text is cut off and overlaps the
   "Fit Again…" button.
5. **Opening a scan doesn't show it.** `show_data_file` attaches the scan, but the
   centre stays on "Project ROIs"; the image is one tab away.
6. **Predict says "Prediction ROIs: none"** while the project has 5 prediction
   ROIs. It means "no results yet". The status bar says "Model: none" although
   a model exists.
7. **Recent files cut the end of the path.** In the welcome screen's list, the part that tells the files apart
   (the scan folder, the file name) is cut off.
8. **Clipped reflectance is reported only in the log.** Loading clipped thousands of values above 1 per
   ROI ("clipped … 3644 above 1"). That points to a white reference problem
   the user should see once, in the GUI, per dataset.

### B. Structure and layout

9. **The workspace switch is hidden.** Explore / Model / Predict are tabs in the
   bottom "Controls" dock, below a second row of tabs (Dataset / Project ROIs)
   in the centre. The main navigation of the program looks like a
   settings panel.
10. **The same information appears three or four times.** The model summary is in Fit summary,
    Inspector and the Model controls. The validation warning is in the summary
    panel, Readiness, the Inspector and the controls. That costs space and makes
    the screen look busy.
11. **Empty states are empty plots.** Predict before a prediction shows three
    empty 0–1 axes and four panels of "-". "Selected ROI pixels" is an empty
    axis. The bottom dock in Model is mostly empty.
12. **Identical file names.** All scans are `stream_hsi.h5`. Headers and
    selectors show "stream_hsi.h5 (API 10% - nozzle 0.6…l-5_20261104_151955)",
    and the distinguishing part is in the middle. Datasets need a short label
    the user can edit ("Layer 1"), with the file as detail.
13. **Two sets of preprocessing.** Explore has Signal, Smoothing, Scatter
    correction; the Fit dialog has its own, "starting from the Explore
    preview". In the demo, Explore shows Absorbance while the fit dialog
    proposes "Raw counts (DN)" without SNV, although the existing model uses SNV.
    It's unclear which one counts.

### C. Guidance (novice)

14. **There is no "where am I / what next".** The welcome screen offers Open Dataset,
    Create Project and Open Project equally, plus dark and white references
    up front. There is no demo project and no first step. Wording like "persisted scientific
    workspace", "Attach a dataset" and "DModXPS limit" doesn't help.
15. **Warnings without an action.** The validation warning says "give them one
    sample label". The fix is in the Fit dialog ("Hold out together: Sample";
    the Sample column already holds L1, L4, …). The warning should offer it
    directly.
16. **Diagnostics without a verdict.** The Distance page shows that 18.7 % of
    pixels exceed the DModX limit (1 % expected), and that the five L01 ROIs
    account for most of it (40–90 % each). The Components page shows five rules
    that disagree (4, 6, 6, 7, 15). An expert reads this, but a novice needs one sentence:
    "The model does not describe the L01 tablets well (one scan: check its
    reference)" and "Recommended: 6 components (3 of 5 rules); you chose 6".
17. **Data checks.** In the demo, ROI L04_05 has the value 5 while the other
    L4 ROIs have 4. A check like "values differ within one sample" would catch
    typing errors before a fit.
18. **Jargon in dialogs.** Examples: "one SE (recommended)", "drop noise", "DModX rounds",
    "Unweighted", names like `API_10p_{n:02d}`, "Target: Layers (1)" for a
    dimensionless unit. All are right for experts. Novices need a short explanation
    next to them, or a tooltip.

### D. Workflow (PAT scientist)

19. **No model comparison.** Models are a list in the browser. Choosing a
    recipe or a component count means fitting, then remembering the numbers. A table of
    models (recipe, LV, RMSECV, Q², bands, validation, date) with side-by-side plots
    is missing.
20. **No stale or provenance markers.** Nothing says when a model was fitted
    before ROIs or values changed.
21. **No report.** Plots export one by one (PNG, SVG, CSV); there is no
    validation report per model.
22. **No lifecycle.** A model can't be marked validated or locked, so it can
    be refitted at any time.

## What this means for the concept

- **The step bar fits naturally.** The project already computes readiness. A
  `Project.readiness()` with check items (A8, C15, C17, D20) can drive a
  step bar: Data → References → ROIs → Targets → Model → Validate → Predict.
  Each item names its fix.
- **Navigation:** move the workspace switch to the top as the main
  navigation; keep the Controls dock for the controls of the current
  workspace (B9).
- **One summary place per workspace:** the right-hand panel. The Inspector shows
  the selected object only (B10).
- **Empty states teach:** one sentence and one button instead of empty axes
  (B11, C14).
- **Verdict lines:** a verdict line above each diagnostics page, computed by the library
  (C16), with the detail plots below for experts.
- **Actionable warnings:** each warning carries its fix as a button (C15).
- **Dataset labels** (B12) and **one preprocessing truth** (B13): Explore
  previews a recipe, and the fit dialog takes that recipe, not a copy.

## Open questions (from the brainstorm)

1. Who comes first in 1.6: novices (guidance, demo project, tour) or PAT experts
   (comparison, provenance, reports)?
2. Lifecycle states in 1.6, or with the method file in 2.0?
3. Reports: HTML only, or also PDF?
4. Guardrails: always warn, or block a few dangerous cases?
5. Part A as a 1.5.3 bug-fix release first, or folded into 1.6?
