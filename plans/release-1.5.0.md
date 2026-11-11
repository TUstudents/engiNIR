# Plan: engiNIR 1.5.0, the terminology applied

Status: implemented (2026-11-11), phases 1–4 on `terminology/contract`; release
pending. Scope, branches, contract depth and the open
points O1–O3 decided by the user.

## Goal

Bring every word a person reads (viewer, library messages, documentation,
contracts, export labels) to `contracts/terminology_v3.md`, with the two
behavior changes it requires, and a test that keeps the viewer from
drifting back. Nothing else: the GUI polish (context bar, next-step hint,
empty and loading states, compact Controls dock) is 1.6.0 with its own plan.

## Decisions (user, 2026-11-11)

1. **Scope:** terminology only; the GUI polish is 1.6.0.
2. **Branches:** one branch (`terminology/contract`), one commit per phase,
   one draft PR opened from the start, so CI runs on every push; `main`
   stays on 1.4.0 until the whole change is consistent, then one merge and
   the release (changed from phased PRs, user, 2026-11-11).
3. **Contracts and docs:** fully, in 1.5.0: US spelling, sentence-case
   headings, the new terms; internal names kept where they name code.

## Inventory (2026-11-11)

String literals in `src/enginir/viewer` containing an old word (approximate;
one string may hold several):

| Old | Strings | New |
|---|---|---|
| data file | 90 | Dataset (the object), file (on disk) |
| colour, grey | 51 | color, gray |
| training | 44 | calibration |
| removed | 26 | excluded |
| RMSECV / RMSEE / RMSEP without level | 34 | "… (per ROI)", "… (per pixel)" |
| sample set | 23 | Calibration set |
| model run | 19 | Model |
| reference value, Reference (axis) | 22 | Target value |
| Uncertainty | 12 | Prediction interval |
| Calibration (reference correction) | 10 | Reference correction |
| observation | 3 | "Fitted on: pixels / ROI means" |
| Signal domain(s) | 3 | Input contract |
| Raw DN, Normalisation, Filtering | 3 | Raw counts (DN), Scatter correction, Smoothing and derivatives |

Library: about 80 log and error messages with an old word. Docs and
contracts: 31 files (contracts alone use the old model words about 400
times, British spelling and title-case headings throughout).

## Phases

### Phase 1: the contract

Commits 524be8c (contract, decisions) and this plan, on
`terminology/contract`; pushed as the draft PR.

### Phase 2: behavior and glossary

1. **ROI role stored as `calibration`** (terminology contract, Calibration
   and Prediction):
   - `ROI_USES = ("calibration", "prediction")`; new calibration ROIs store
     `use = calibration`; `save_roi_use` writes the value instead of
     deleting the attribute;
   - reading: a missing `use` and `training` read as `calibration`;
   - identity rule: a missing `use`, `training` and `calibration` compare as
     the same content, so re-saving an older ROI is not a change;
   - `roi-target-contract_v3.md` (the `use` attribute, "Any other stored
     value is rejected", the ROI table column "Set"),
     `project-schema-contract_v3.md` §5, `docs/hdf5_format.md`;
   - the API (`scan.add_roi(..., use=...)`, `project.set_use`) accepts
     `calibration` and `prediction`; `use="training"` raises a `ValueError`
     naming the rename (O1);
   - tests: new ROI stores `calibration`; missing and `training` read as
     calibration; re-saving an ROI stored without `use` is not refused
     (fault-injected: the comparison without the normalization fails);
     `use="training"` refused by the API; prediction ROIs unchanged.
2. **`Prediction.interval_map(level)`** replaces `uncertainty_map`
   (removed); prediction-runtime and headless contracts, user guide, tests.
3. **Glossary** `docs/glossary.rst`: every term of the contract with its
   definition, synonyms in other tools and internal name; `:term:` targets.
### Phase 3: the viewer

All user-visible viewer text, following the inventory. Notable changes:

- **Menus and buttons** in title case ("Add Dataset...", "Fit Again...",
  "Add ROI Grid", "Export CSV"); titles in sentence case ("Prediction map").
- **Explore:** "Dataset:" selector; Signal "Raw counts (DN)", "Reflectance",
  "Absorbance"; groups "Smoothing and derivatives", "Scatter correction".
- **ROI table:** column "Set" showing Calibration or Prediction, edited with
  a two-item drop-down (double-click or F2), sortable by text (O2);
  "Target value" headers; result columns "% excluded".
- **Project Browser:** "Datasets", "Reference corrections", "ROIs",
  "Targets", "Target values", "Models" (each with its calibration set and,
  as the contract requires, its input contract beneath it), "Predictions".
- **Model:** metric labels "RMSECV (per ROI)", "RMSECV (per pixel)" etc.;
  Inspector "Fitted on: pixels / ROI means"; axis "Target value";
  "calibration scores".
- **Predict:** layer "Prediction interval (±, 95 %)"; color scale
  "Calibration range" / "Own limits"; "Exclude pixels above the DModX
  limit"; "Gray excluded pixels"; "above DCrit", "outside the model".
- **Exports:** plot titles and CSV column titles follow (CSV values and
  metadata keys unchanged).
- **The guard test** `tests/test_terminology.py`:
  - an AST scan of `src/enginir/viewer/*.py`: string and f-string
    arguments of the Qt calls that show text (`setText`, `setToolTip`,
    `setStatusTip`, `setWindowTitle`, `setTitle`, `setPlaceholderText`,
    `addItem(s)`, `addAction`, `addMenu`, `addRow`,
    `setHorizontalHeaderLabels`, `showMessage`, `QMessageBox.*`, the
    constructors of `QLabel`, `QPushButton`, `QCheckBox`, `QRadioButton`,
    `QGroupBox`, `QAction`) and plot titles and axis labels;
  - whole-word, case-insensitive matches of "Words not used" and British
    spellings; backticked text, `snake_case` and `CamelCase` tokens exempt;
  - capitalization: title case for buttons and menu items, sentence case
    for titles and labels, with the abbreviation allow-list;
  - a reviewed allow-list file for justified exceptions;
  - fault-injected: one banned word in a temporary module fails the test.
  It is added in this phase, passing from the start (O3).
- Tests that assert old strings are updated.

### Phase 4: library, docs, contracts, release

- About 80 library log and error messages.
- User guide, API reference prose, architecture docs, README, AGENTS.md:
  the new terms, US spelling, sentence-case headings; glossary links.
- All contracts: US spelling, sentence-case headings, the new terms
  (internal names kept where they name code); `contract_coverage.md` gets the
  terminology rows and a gap entry.
- CHANGELOG 1.5.0 with a **Breaking** section: `interval_map`; projects with
  new calibration ROIs do not open in 1.4; changed labels and CSV column
  titles (scripts that read exported CSVs by column title must be updated).
- Version 1.5.0, the four gates, release as in AGENTS.md.

## Open points, decided (user, 2026-11-11)

- **O1** `use="training"` in the API is refused with a `ValueError` naming
  the rename; stored `training` still reads as calibration.
- **O2** The ROI table's role column becomes "Set": a text cell
  (Calibration / Prediction) with a two-item drop-down.
- **O3** The guard test is added in Phase 3, with the string fixes, and
  passes from the start; Phase 2 has no guard.

## Risks

- Scripts that read exported CSVs by column title break (Breaking section).
- Many tests assert exact viewer strings; Phase 3 updates them, which makes
  that commit large but mechanical.
- `docs/examples` output is checked verbatim; changed messages need updated
  expected output.
