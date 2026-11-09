# Plan: band selection and VIP plots

Status: decided (2026-11-09), not started

## Goal

Users see which bands a model relies on (VIP for PLS, selectivity ratio for
PLS and PCR) and can leave bands out of a model: by hand (noisy detector
ends, a saturated water band) or on the proposal of a selection algorithm
whose own effect is validated honestly.

Today `PLSRegression.vip_scores()` and `enginir.visualization.plot_vip`
exist and every PLS run stores the weights VIP needs, but the viewer shows
no VIP, PCR has no importance measure besides coefficients, and a recipe
cannot leave out a single band. The shrinking-axis plumbing built for
Savitzky-Golay edge dropping (`PreprocessingRecipe.output_wavelengths`,
carried through model, diagnostics, prediction checks and plots) is the base
for band selection.

## Decisions (taken with the user 2026-11-09)

1. **Band selection is the last recipe step**, after all preprocessing:
   AsLS → Savitzky-Golay → SNV → MSC → EMSC → Detrend → **band selection**.
   Normalisation, MSC/EMSC references and Detrend see the whole spectrum; the
   model sees only the selected bands.
2. **Ranges in nm**, inclusive, resolved against the recipe's output axis
   (after Savitzky-Golay edge dropping); files without wavelengths (band-index
   axis) use band numbers.
3. **User input as keep and exclude ranges**: an optional keep list (default:
   the whole axis) minus an optional exclude list (e.g. the water band
   1400–1500 nm). The viewer offers both, with the common NIR water bands as
   suggestions, never applied automatically.
4. **VIP plot** in the viewer (PLS) with lines at 1.0 and 0.8, the model's
   mean spectrum shaded behind it, and a sorted view.
5. **Selectivity ratio** (Rajalahti) for PLS and PCR, as the interpretation
   measure that does not depend on the weights of one algorithm.
6. **Algorithms only propose**: forward interval PLS (iPLS) returns ranges
   the user accepts into a normal recipe; a refit never searches again.
7. **Nested validation** of every selection: the search runs inside each
   outer ROI-grouped fold, and the outer RMSECV of the whole procedure is
   reported next to the RMSECV with all bands.
8. **Jack-knife intervals** on coefficients and VIP, and the Martens
   uncertainty test as a selection method, follow with the jack-knife work
   (prediction-uncertainty plan follow-up), not in this plan.

## Design

### Recipe fields

- `bands_keep`: list of `[low, high]` nm ranges, or absent (whole axis)
- `bands_exclude`: list of `[low, high]` nm ranges, or absent
- Written only when set, so every stored recipe keeps its id
  (project-schema contract §4, as for `sg_edges` and AsLS). Normalised
  before the id is computed: ranges sorted, overlaps merged, floats.
- Resolution: a band is selected when its wavelength lies in a keep range
  (or no keep list is given) and in no exclude range. A range that selects
  nothing on the axis is logged; a recipe that selects no band, or fewer than
  the model's components need, is refused.
- `output_wavelengths` returns the selected bands; the axis may have gaps.
  Plots over wavelength draw a gap (no line across an excluded range).
- The selection is a column index applied after Detrend; stateless; no
  effect on MSC/EMSC state (fitted on the whole spectrum).

### Through the system

Reuses the edge-dropping plumbing: coefficients, loadings, projection data,
VIP, stored `prediction_model/wavelengths` and `diagnostics/wavelengths` on
the selected axis; prediction checks the model's band count against the
recipe's output; the scan keeps all its bands (the signal domain is
unchanged). Explore shows excluded bands as missing in preprocessed views,
like dropped Savitzky-Golay edges.

### VIP and selectivity ratio

- VIP is computed from the stored weights (no refit); stored with the
  diagnostics for new runs so plots and exports read it directly.
- Selectivity ratio per band: target projection of the model (one
  predictive component along the regression vector), SR = explained /
  residual variance of the band over the fitted pixels. Stored for PLS and
  PCR.
- Viewer, Spectral page: **VIP** (PLS; the free slot) and **Selectivity
  ratio**, each over wavelength with the mean spectrum behind and a sorted
  view; CSV export through the panels.
- The docs state what changes VIP: number of components, preprocessing
  (derivatives move importance to band flanks), X scaling.

### iPLS proposal (forward, nested)

- `propose_bands(project, target, recipe, model="PLS", n_intervals=20,
  max_steps=…, fit_spec=…)` on the recipe's output axis before selection.
- Forward: start empty, add the interval that lowers RMSECV most, stop when
  no interval lowers it by more than its standard error (the one-SE idea of
  component selection), with the number of components chosen by the
  component rule at every step.
- Nested: the outer folds are the run's own fold scheme (ROI or sample
  groups); inside each outer training part the same search runs with inner
  folds; the outer held-out ROIs give the procedure's RMSECV. Then the
  search on all data gives the proposed ranges.
- Result (not stored in the project, like predictions): proposed ranges,
  the RMSECV curve over the steps, the nested RMSECV, and the RMSECV with
  all bands; logged once. The viewer shows it and can put the ranges into
  the fit dialog as keep ranges.
- Cost is bounded (intervals × steps × outer × inner folds of small PLS
  fits on the training matrix); the dialog shows the estimate and progress
  and can be cancelled.

## Contracts (with an independent review)

- `project-schema-contract_v3.md` §4: the two fields, normalisation,
  resolution, order, ids.
- `sample-set-model-run-contract_v3.md`: the selected axis for X and the
  model; VIP and selectivity ratio in §7.5.
- `prediction-runtime-contract_v3.md`: band count of the model = selected
  bands.
- `headless-workflow-contract_v3.md`: `propose_bands`.
- `viewer-workspace-ui-contract_v3.md`: range selection, VIP and selectivity
  ratio plots, the proposal dialog.
- `docs/hdf5_format.md`, gap register.

## Phases

1. **Contracts and review.**
2. **VIP and selectivity ratio**: stored with new runs; viewer plots
   (Spectral page); tests against independent computations (VIP: mean of
   VIP² = 1, equals `vip_scores`; SR on a synthetic set with one informative
   band); fault injection.
3. **Band ranges in the recipe**: fields, normalisation, resolution after
   Detrend, ids of stored recipes unchanged, output axis with gaps through
   model and prediction; tests (a water-band exclusion leaves the expected
   bands; Detrend/SNV computed on the whole spectrum, fault-injected; a
   model on selected bands predicts its training scan identically; stored
   recipes keep their ids; an empty selection is refused).
4. **Viewer range selection**: keep and exclude ranges in the fit dialog by
   dragging on the mean spectrum, with VIP or coefficients as guide and the
   water bands as suggestions; Explore shows excluded bands as missing; Fit
   again keeps the ranges.
5. **iPLS proposal**: library function and viewer dialog; tests (a synthetic
   set whose information lies in two known intervals finds them; the nested
   RMSECV is not lower than the outer RMSECV of the chosen ranges evaluated
   without nesting on the same data, i.e. the optimism is visible;
   fault injection: searching on all data inside the outer loop must fail
   the leakage test).
6. **Docs**: user guide (reading VIP and SR, cutting bands, the water bands,
   why selections need nested validation), API reference, CHANGELOG, four
   gates.

## Out of scope

- CARS, genetic algorithms and other stochastic searches.
- VIP elimination as an automatic method (the VIP plot supports manual use).
- Jack-knife intervals and the Martens test (follow the jack-knife work).
- Choosing bands for a multispectral filter design (a later report built on
  the iPLS result).
