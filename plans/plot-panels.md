# Plan: plot panels for the viewer

Branch: `feature/plot-panels`
Status: phases 0–3 done
Date: 2026-11-02

## Goal

Give every plot in the viewer the same frame before we add new diagnostics:

- every plot fits the workspace layout, and a new plot can be added without
  squeezing the others;
- plots snap into a fixed grid, and any plot can be maximized in place;
- every plot can be saved as PNG (and SVG) and its data exported as CSV.

This plan only migrates the existing plots. The RMSECV-vs-LV curve, the
T²/Q influence plot, VIP and the T²/Q prediction maps are later plans that
add new panels to this frame.

## Current state

| Plot | Where | Engine | Maximize | PNG | CSV |
|---|---|---|---|---|---|
| Histogram, RAW, Sensor Frame, Spatial Profile | Explore docks (`_panels.py`) | matplotlib | no | no | no |
| Spectral profile | Explore centre, fixed height (`_viewer.py:285`) | matplotlib | no | no | no |
| False-colour image | Explore centre (`_canvas.py`) | pyqtgraph | no | no | no |
| Pred vs Ref, Residuals, Target distribution, Scores, Coefficients/Loadings | one figure, 3×2 gridspec (`_model_diagnostics.py:30`) | matplotlib | no | no | no |
| Prediction map | `PredictionMapWidget` (`_prediction.py:313`) | matplotlib + toolbar | no | toolbar | no |

The only CSV export is ROI spectra (`_eda.py:338`).

## Design decisions

1. **`PlotPanel`** wraps one plot. Its header replaces the in-figure title:
   title, a maximize button and a menu with *Save PNG*, *Save SVG*,
   *Export CSV*, *Copy image* and *Reset view*. This costs no space compared with today.
2. **`PlotSpec`** defines a plot by its `id`, `title`, `engine`
   (`"mpl"` or `"pg"`), `render(target, payload, colors)` and
   `table(payload) -> PlotTable`. PNG and CSV come from the same payload, so
   the CSV holds exactly what is plotted.
3. **`PlotGrid`** lays out the panels of one workspace page in fixed slots
   (1×1, 1×2, 2×2, 2×3, with spans). Maximize fills the grid area in place.
   Double-click on the header or Esc restores it. There are no floating windows,
   and the tabs stay the primary navigation (UI contract).
4. **Pages.** The Model centre area becomes sub-tabs with one grid each.
   Initial pages: *Fit* (Pred vs Ref | Residuals / Target distribution |
   Scores) and *Spectral* (Coefficients | Loadings). New plots get new slots
   or pages later.
5. **Lazy and visibility-aware rendering.** A page builds its panels the
   first time it is shown, and a panel redraws only when visible and its
   payload changed. The benchmark (Agg, 1400×900) shows a full redraw costs
   the same as one figure (78 ms), and changing one plot drops from 94 to
   36 ms.
6. **Export format.**
   - PNG at 300 dpi and SVG, rendered offscreen in a light print style,
     whatever the screen theme.
   - CSV is UTF-8, comma-separated, with `# key: value` provenance lines first:
     enginir version, export time, project, dataset, model run id,
     sample set, recipe, components, CV scheme, plot id.
   - ROI rows carry both the ROI name and the id. Units go in the column names.
   - Maps export as a frame × row matrix CSV; more than 1e6 cells asks for
     confirmation.
7. **Registry.** All specs are registered in one place. A test iterates over
   the registry, so a plot without export cannot be added.
8. **Explore docks** keep their positions. Their contents become `PlotPanel`s
   (header, export). There, maximize floats the dock at main-window size,
   and restore re-docks it.
9. **Layout state** (active page, maximized panel) is screen state. It is
   stored in QSettings, never in the project file. *Reset layout* clears it.

## Open decisions

Each of these must be settled before the phase it affects:

- **D1 (phase 6): can users swap which plot occupies a slot?** The default is no:
  curated pages only. A per-slot selector would revise the non-goal "editable
  plot compositions by end users" in the model diagnostics contract.
- **D2 (phase 5): does the prediction map move to pyqtgraph in this branch?** The default is no:
  migrate it as matplotlib and move it with the T²/Q map work.
- **D3 (phase 2): print style or screen style for PNG?** The default is the print style.

## Phases

Every phase ends with all four gates clean:

```bash
uv run pytest
uv run ruff check src/enginir tests docs/examples
uv run pyright src/enginir
uv run python -m sphinx -W -E -b html docs /tmp/docs
```

### Phase 0: baseline

- Run the four gates on the branch and note the results.
- Add `tests/test_plot_performance.py` (marked slow, skipped by default unless
  `ENGINIR_PERF=1`). It times building and redrawing the Model diagnostics
  in an offscreen Qt widget at 1400×900, with full redraw and one-plot change.
  Record the current numbers in this plan.

**Done (2026-11-02).** The gates on the branch: pytest 495 passed, 1 skipped;
ruff, pyright and Sphinx clean.

Baseline from `tests/test_plot_performance.py`:

- Setup: `ModelDiagnosticsWidget` at 1400×900, 60 ROIs, 224 bands, offscreen
  Qt. Each step includes the paint pass.
- Software: Python 3.13, matplotlib 3.11.2, PyQt6 6.11.0, WSL2, 10 cores.
- Each value is the median of 15 repeats. Three runs agreed within 2 %.

| Step | Median |
|---|---|
| Build and first draw | 380 ms |
| Redraw all plots (new payload) | 187 ms |
| Change one plot (spectral mode) | 121 ms |

Changing one plot costs about two thirds of a full redraw, because the whole
figure is repainted. Phase 7 must beat 121 ms for that step and stay within
10 % of 187 ms for a full redraw.

### Phase 1: contracts first

- `contracts/viewer-workspace-ui-contract_v3.md`: add a *Plot panel policy*
  covering:
  - the header and title;
  - in-place maximize and restore;
  - PNG/SVG/CSV export and the CSV provenance header;
  - lazy pages;
  - layout state stored in QSettings, not the project.
- `contracts/model-workspace-diagnostics-contract_v3.md`: the centre area is
  paged grids of panels; the required plots are unchanged. Record D1.
- `docs/contract_coverage.md`: add rows for the new rules, marked open.

**Done (2026-11-02), with D1 = no (curated pages) and D3 = print style.**

- The UI contract has a new *Plot Panel Policy* section and two related
  changes:
  - The prediction map is now a plot panel.
  - The acceptance criteria include plot panels.
- The model diagnostics contract changes in four places:
  - The center area becomes the `Fit` and `Spectral` pages.
  - The secondary diagnostics move from the right area to those pages.
  - The scores plot must be labelled as training scores.
  - The non-goal note on slot swapping is refined.
- The ledger has five new rows and gap `G19`.

The contracts change ahead of the code, on this branch only. They merge
together with phases 2–7.

### Phase 2: core components

New files:

- `src/enginir/viewer/_plot_panel.py`
  - `PlotTable` holds columns (name, unit, array) and metadata. Its `to_csv(path, provenance)`
    uses the `csv` module and checks that all columns have equal length.
  - `PlotSpec` (frozen dataclass).
  - `PlotPanel(QWidget)` holds the header and the canvas. Its methods are
    `update_data(payload, *, empty_reason="")`, `clear()`,
    `export_png(path, dpi=300)`, `export_svg(path)`, `export_csv(path)`, and
    the signals `maximize_requested` and `restore_requested`.
  - For matplotlib, the panel exports by rendering the spec into a fresh
    offscreen `Figure` with print colours. For pyqtgraph, it uses
    `pyqtgraph.exporters.ImageExporter`/`SVGExporter`.
  - Errors are logged and raised with `log_and_raise` and shown in a
    `QMessageBox` by the menu handler.
- `src/enginir/viewer/_plot_grid.py`
  - `PlotGrid(QWidget)` takes slots `(spec_id, row, col, rowspan, colspan)`.
    It maximizes by hiding the sibling panels and spanning the panel over
    the grid, and restores with Esc or double-click.
  - It builds panels lazily and redraws only visible, changed panels; this
    reuses the existing `maybe_tight_layout`.
- `src/enginir/viewer/_plot_registry.py` holds `PLOT_SPECS: dict[str, PlotSpec]`
  and the page definitions per workspace.

Tests (`tests/test_plot_panels.py`):

- For every registered spec: the empty state renders, PNG and SVG files are written and
  non-empty, and the CSV has the provenance lines and matches the plotted
  arrays. Check this fails when a spec's `table` drops a column.
- Maximize and restore keep the other panels' payloads and do not redraw
  them.
- A hidden page does not render until shown.

**Done (2026-11-02).** All four gates are clean (pytest 509 passed, 2 skipped).
`tests/test_plot_panels.py` has 14 tests. Eight injected faults each made a
test fail:

- CSV precision
- screen colours in the export
- axis limits not copied
- export of an empty plot enabled
- drawing while hidden
- restore redrawing all panels
- all pages built at start
- no confirmation for a large export

The CSV-precision and screen-colour faults first went unnoticed. The tests
were tightened until they failed.

Deviations from the description above:

- `render(fig, payload, colors)` draws into the whole figure and returns the
  clickable ROI points (`RoiPoints`). The panel does the ROI picking that
  `_model_diagnostics.py` does today.
- Only the matplotlib engine exists so far. The pyqtgraph canvas gets its
  header in phase 4, as an adapter around the existing widget, not as a spec.
- *Reset view* is not in the menu yet. No panel has pan or zoom until the
  prediction map (phase 5).
- Tight layout runs for a new payload or a new size, not when the same
  payload is redrawn (for example after a theme change).
- `Esc` is a `QShortcut` on the page, so it also works when the canvas has the
  focus. The canvas would otherwise take the key.

### Phase 3: migrate Model diagnostics

- Split `ModelDiagnosticsWidget` into specs: `model.pred_vs_ref`,
  `model.residuals`, `model.target_distribution`, `model.scores`,
  `model.coefficients` and `model.loadings`. Reuse `plot_prediction`.
- The scores title says *training* scores.
- Keep ROI picking: each ROI-level spec registers its points, and a click
  emits `roi_selected` (behaviour as `_model_diagnostics.py:314-349`).
- The *Spectral Plot* combo in the right panel goes, because coefficients and loadings get
  their own panels. Check the spectral wording in the contract (line 216 onwards).
- Remove the *Diagnostic Graphs* group box frame to give the grid the space.
- Adapt `test_panels.py::test_model_diagnostics_show_empty_states_in_the_plots`
  and `test_modeling_view.py::test_reopened_model_run_renders_every_diagnostic_and_selects_rois`.

**Done (2026-11-02).**

What changed:

- `ModelDiagnosticsWidget` is now a `PlotGrid` with six registered specs:
  - `Fit` page: Pred vs Ref | Residuals / Target distribution | Training scores
  - `Spectral` page: Coefficients / Loadings
- The *Spectral Plot* combo and the *Diagnostic Graphs* group box are gone.
- Pred vs Ref now annotates R², RMSECV and bias, as the contract requires.
  Before, it annotated only R².
- The payload carries ROI names.
- `ModelingView.workspace_diagnostics_provenance()` supplies the CSV header.

New `tests/test_plot_registry.py` checks every registered plot:

- the empty state
- PNG, SVG and CSV export
- no title inside the plot
- every CSV value is a drawn value

Six injected faults were caught, with each check rerun after its fix:

- residual sign in the CSV
- histogram bins differing between plot and CSV
- wrong score column
- payload copies defeating the no-op
- training label dropped
- drawing while hidden

**Finding.** The G17 navigation test never showed the viewer. With panels
that draw only while visible, it passed without drawing anything. It now
shows the viewer and checks that the Model tab draws.

Timings (same benchmark; "change one plot" is now one panel getting new data):

| Step | Baseline | Now |
|---|---|---|
| Build and first draw | 380 ms | 219 ms |
| Redraw all plots | 187 ms | 167 ms |
| Change one plot | 121 ms | 46 ms |

"Redraw all" now covers the four visible `Fit` plots. The `Spectral` page draws
when it is opened.

An offscreen screenshot check led to four fixes:

- The themed header buttons made the header about 36 px tall. They are now
  20 px, so the header takes 24 px.
- The *Maximize* standard icon was invisible in the dark theme. The buttons
  now use the `⤢` / `⤡` glyphs.
- Margins stayed at the grid size after maximize. Tight layout now also runs
  on canvas resize.
- The annotation unit read "0.41 (%)". It now reads "0.41 %".

### Phase 4: migrate Explore

- Wrap `HistogramPanel`, `DC3Panel`, `SensorPanel`, `SpatialProfilePanel`
  and the spectrum panel in `PlotPanel` headers through specs
  (`explore.histogram`, `explore.raw`, `explore.sensor`, `explore.spatial`,
  `explore.spectrum`). Keep the in-place artist updates of `HistogramPanel`
  and the deferred rebuild of hidden docks.
- False-colour canvas: give it a header with *Save PNG* (pyqtgraph exporter)
  and *Export CSV* (the displayed band image as a matrix). ROI drawing and the
  crosshair are unchanged.
- Dock maximize as in decision 8.

### Phase 5: migrate Predict

- `PredictionMapWidget` becomes spec `predict.map` (matplotlib, D2). The
  header menu replaces `NavigationToolbar`, but pan and zoom must stay (a
  pyqtgraph-style view box or a minimal pan/zoom toggle in the header).
- Output selector for multi-output maps stays in the header.
- Add `predict.histogram` (distribution of predicted values). It is cheap and
  useful, and it is the first new panel through the frame.

### Phase 6: export all and layout state

- *File → Export plots…* writes every panel of the active workspace (PNG and
  CSV) to a chosen folder, named `<page>_<plot id>.<ext>`. It renders hidden
  pages offscreen without showing them.
- Store the active page and maximized panel in QSettings; *Reset layout*
  clears them.
- If D1 is yes: add the per-slot plot selector here.

### Phase 7: performance check and docs

- Rerun `tests/test_plot_performance.py`. One-plot change must be faster than the
  phase 0 baseline, and full redraw no slower than 10 % over it. Record the numbers.
- `docs/viewer_architecture.md`: new *Plot panels* section, and replace
  *Adding a new panel* with "add a `PlotSpec` and register it".
- `docs/user_guide/viewer.rst`: maximize, pages, export.
- `docs/api/viewer.rst`: only if a public symbol changes (none planned).
- `docs/contract_coverage.md`: mark the phase 1 rows covered.
- `CHANGELOG.md`: *Unreleased* entry.

## Risks

- **Qt overhead per canvas.** Agg numbers exclude Qt's buffer copy; phase 0/7
  measure real widgets.
- **Pan/zoom on the prediction map** must survive losing `NavigationToolbar`.
- **Tests that reach into `ModelDiagnosticsWidget` internals** will break in
  phase 3. Rewrite them against behaviour (rendered plots, ROI selection),
  not attributes.
- **Large map CSVs**, handled by the size confirmation in decision 6.

## Follow-up plans (not in this branch)

1. RMSECV-vs-LV curve with a suggested count (new `model.lv_curve` spec,
   nested-component CV, model run artifact and contract change).
2. Scaling option for PLS/PCR (mean-centring vs autoscaling).
3. T²/Q influence plot per ROI (`model.influence`) and leverage.
4. VIP (`model.vip`).
5. Per-pixel T²/Q maps and an out-of-domain mask in Predict (pyqtgraph,
   prediction runtime contract change).
