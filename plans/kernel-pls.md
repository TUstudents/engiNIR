# Plan: kernel PLS engine and faster cross-validation

Status: Phases 0–4 implemented on branch `kernel-pls` (2026-11-13), in review
as a pull request. The demo fit (20 ROIs, 396 000 × 176, SNV, 7 folds, curve to 15)
takes 4.5 s and 0.45 GB (traced) against 42.9 s and 1.7 GB with NIPALS on
the same idle machine; equal RMSECV. Deviations from the plan:

- MSC/EMSC recipes keep the pipeline path (with the kernel engine): the
  per-fold pass on statistics is not built.
- The final model is fitted again from the sums (microseconds), not cut
  from the A_max fit; the result is the same.
- The diagnostics of the final model keep their three passes
  (`project`, `band_importance`, the fitted values) instead of one.
- The viewer's fit progress does not name the phase.
- Phase 0 had no independent review (no agents were asked for).

Tolerances found: the pipeline predicts held-out pixels in float32 (the
working dtype), so cross-validated values agree to about 1e-7 (tests: 1e-5);
scikit-learn's PCA keeps float32 input in float32, so PCR is compared with
an SVD of float64 spectra.

## Goal

Fitting a model on an HSI calibration set takes too long and needs far more
memory than the data. Every further feature (component curve, DModX
exclusion, band proposal, OPLS, calibration transfer) multiplies that cost.
This plan adds a kernel engine on sufficient statistics (XᵀX and sums) for
PLS and PCR, with all diagnostics and no needless copies or type
conversions. Every result the user sees stays the same within
floating-point noise. NIPALS stays available until 2.0. It lays the base
for OPLS, EPO and OSC in 2.0.

## Measurements (2026-11-13)

**The current fit.** Copy of the demo project `Teflon_demo_v2.h5`: 20
calibration ROIs, 396 000 pixels × 176 bands, PLS, SNV, 7 round-robin
folds, component curve to 15 components.

| | Current |
|---|---|
| Fit time | 81 s |
| Peak memory | 2.1 GB traced (2.4 GB resident) for a 279 MB float32 matrix |
| PLS fits on the full pixel matrix | 9: 7 folds at A_max, the curve's full fit, the final model |
| Time in scikit-learn NIPALS | 66 s (82 %) |

Where the time and memory go today:

- every fold fit copies its rows of X (float32, fancy indexing), then makes
  a float64 copy for scaling (up to 558 MB); NIPALS then runs about 15
  matrix–vector passes over it per component
- the final model is fitted twice: the curve's full fit at A_max already
  holds it (its first A components), and the run fits it again
- `project()` runs twice (curve and diagnostics), each a pass over all
  pixels with a deflation step per component (9 s)
- `band_importance` and `fitted.predict` are two more passes
- the raw `uint16` matrix (139 MB) stays alive to the end (`SampleSetData`
  in `ModelRunResult`); the float32 signal matrix and the preprocessed
  matrix coexist for a moment

**BLAS threads.** On this machine (WSL, 10 vCPUs; 4 cores busy with
another job during the measurement), OpenBLAS with 10 threads drops to
about 2 GFLOP/s on the tall, narrow product XᵀX, against 170–190 GFLOP/s
with 4–6 threads. This also slows the current code. Re-measure on an idle
machine before the release notes quote times.

**Prototype on the same data** (4 BLAS threads):

| Step | Time | Memory |
|---|---|---|
| Per-ROI sums (XᵀX, Σx, n) in one pass, float64 accumulated from 16 384-row chunks | 0.39 s | 56 MB peak, sums 5 MB |
| Kernel PLS (Dayal–MacGregor algorithm 2) to 15 components | 1.8 ms | – |
| 7-fold CV to 15 components, from the sums | 11 ms | – |
| Coefficients against the current fit (6 components) | equal to 7.7·10⁻¹² relative; weights of equal sign | – |

**PCR** (measured 2026-11-11, 592 800 real tablet spectra × 220 bands):
eigendecomposition of XᵀX against the full SVD: components 1–15 within
3·10⁻⁶ °, coefficients within 2·10⁻¹¹ relative; 0.17–0.5 s against 13 s
per fit.

**Existing work.** [ikpls](https://github.com/sm00thix/ikpls) (Engstrøm et
al., JOSS 2024, doi:10.21105/joss.06533) implements IKPLS and fast
partition-based cross-validation (arXiv 2401.13185); a 2026 follow-up
(arXiv 2607.16138) speeds up the R and Q steps.

## Decisions

- **D1. Own engine; ikpls as a test oracle.** The algorithm is short and the
  speed comes from BLAS. engiNIR controls its numerics (fold groups by ROI,
  DModX exclusion inside folds, SIMCA scaling, float32 inputs). ikpls is a
  test-only dependency (`dev` extra). No JAX or GPU.
- **D2. NIPALS stays until 2.0 (user, 2026-11-13).** PLS `hyperparameters`
  gain `algorithm`: `nipals` or `kernel`. New fits write `kernel`; a
  script may ask for `nipals`; Fit Again and a refit use the stored value.
  A model without the key (every model of a 1.5 project) reads `nipals`,
  so its refit reproduces it exactly. This is the one compatibility rule
  between 1.5 and 1.6. NIPALS is removed in 2.0.
- **D3. Equivalence, not bitwise identity.** Kernel and NIPALS give the
  same model in exact arithmetic. Tests require agreement to 10⁻⁸ relative
  on coefficients, weights, loadings, scores and every stored metric. Rule
  ties that differ by floating-point noise may fall differently; contract
  §7.4 already says a refit reports that.
- **D4. Release.** Speed and memory only, no new results: a 1.x release
  (1.6). OPLS, EPO and OSC go to 2.0 with their own plan.
- **D5. PCR from XᵀX, with a conditioning guard (2026-11-11).** PCR
  `hyperparameters` gain `algorithm`: `svd` or `gram`; a model without the
  key reads `svd`. A new fit uses `gram` when σ_Amax/σ₁ ≥ 10⁻⁶ (A_max the
  largest number of components the fit computes), otherwise the full SVD
  with a logged reason; the stored value is the algorithm used, so a refit
  runs the same one. scikit-learn's PCA sign convention is kept. The SVD
  path stays until 2.0, like NIPALS.
- **D6. Sums per fold group (user, 2026-11-13).** XᵀX, Σx, n and the y
  sums are kept per fold group; n and Σx also per ROI (the ROI mean
  spectra). This is the coarsest unit that serves the fit (folds never
  split a group) and the band proposal (its nested folds split the outer
  training groups again). Without sample labels every ROI is its own group,
  so it equals per-ROI sums; with sample labels it is smaller; it never
  exceeds per-ROI sums. The code keys blocks by a block unit with a
  ROI → block map, so per-ROI blocks (for weights per ROI in 2.0) are a
  change of map, not of engine.
- **D7. Memory and threads (user, 2026-11-13).**
  - The preprocessed float32 matrix is kept in memory for the pixel pass
    (279 MB here) rather than re-read from the project file.
  - The raw `uint16` matrix is released right after the reference
    correction, and the preprocessed matrix is filled ROI by ROI, so only
    one full matrix exists at a time.
  - The engine's matrix products run with a capped number of BLAS threads
    (`threadpoolctl`, already installed with scikit-learn), by default half
    the logical CPUs, configurable with an environment variable
    (`ENGINIR_BLAS_THREADS`).
  - The diagnostics keep storing the per-pixel training scores
    (`x_scores_`, `scores_`); the pixel pass computes them anyway.

## The interface

Four layers, each with one rule.

### 1. Sufficient statistics (`enginir/models/_gram.py`, private)

`GramStats`, built in one streaming pass over the preprocessed float32 rows:

- per block (fold group): n, Σx, XᵀX (K × K), Σy, Σy² (and XᵀY for PLS2)
- per ROI: n and Σx (its mean spectrum) and its y value
- float64 only inside one chunk (default 16 384 rows) and in the sums; a
  fixed shift (the mean of the first chunk) is subtracted before
  accumulating, so raw counts around 3000 do not cancel when centering
- blocks add up: the calibration part of a fold is the sum of its
  training blocks (summed, not total minus held-out, for precision)
- methods: `combine(blocks)`, `centered()`, `scaled(x_scaling, y_scaling)`
  (uv and pareto from the diagonal, N − 1 as contract §7 requires),
  `select(columns)` (a band subset for the band proposal)

Size: blocks × K² × 8 B (5 MB here; 79 MB for 320 groups at K = 176).

### 2. Engine functions on the statistics (`_gram.py`)

- `kernel_pls(stats, A, x_scaling, y_scaling)`: improved kernel PLS,
  PLS1 and PLS2 (w the dominant eigenvector of XᵀYYᵀX for PLS2). Returns
  W, P, Q, R = W(PᵀW)⁻¹, the means and the scale factors. Signs as
  scikit-learn's NIPALS.
- `gram_pca(stats, A, x_scaling)`: symmetric eigendecomposition of the
  centered, scaled XᵀX with the guard of D5; signs as scikit-learn's PCA.
- The model with a < A components is the first a columns: nothing is
  refitted for smaller models, including the final model.

### 3. Everything the fit reports, from the statistics (no pixel pass)

- cross-validation: each fold's statistics are the sum of its training
  blocks; one engine call per fold to A_max
- held-out ROI predictions for every a ≤ A_max: the prediction of the
  ROI's mean spectrum (the model is linear in the preprocessed spectrum,
  §5); this gives the component curve's RMSECV (per ROI), its standard
  error, Q², `cv_outputs` and `rmsecv`
- pixel RMSECV and bias: per held-out ROI,
  Σ(y − xb − b₀)² = n y² − 2y(sᵀb + n b₀) + bᵀGb + 2b₀ sᵀb + n b₀²,
  summed per block (bᵀGb adds up over blocks)
- R²Y (per pixel and per ROI), R²X (trace formulas on XᵀX, R and P),
  RMSEE (per ROI), R²adj, weight roughness and the component rules
- the final model: the A_max factors cut to the chosen A
- VIP (W and TᵀT = RᵀXᵀXR), selectivity ratio (target projection from
  XᵀX), score variances and ROI scores (from the ROI mean spectra)

### 4. One pass over the pixels for what is per pixel

Per chunk, scaled once in float64, one product `z @ [R | P]` gives the
scores T and Pᵀz. From them:

- residual sum per pixel: ‖z‖² − 2 tᵀ(Pᵀz) + tᵀ(PᵀP)t (no deflation loop)
- T², DModX, the fitted value, the stored training scores
- the held-out prediction of each pixel with its own fold's coefficients
  (chunks follow the ROIs, so a chunk uses one fold vector)

Outputs are float32 vectors of length N (and the N × A scores). They feed
the pixel interval coverage, the shares above the limits and the stored
`t2`, `dmodx` and scores.

### Cases with more pixel passes

- **DModX exclusion** (inside every fold, §7.2): one pass per round per
  fold over that fold's training pixels for their residuals; excluded
  pixels are subtracted from their block's sums (a rank update) instead of
  a refit.
- **MSC and EMSC** (fitted inside every fold): one pass per fold that
  preprocesses with the fold's state and builds its statistics, and the
  held-out ROIs' mean spectra after that preprocessing. No fancy-index or
  float64 copies.
- **NIPALS and SVD** (D2, D5): the current code path, unchanged, used when
  the stored or requested algorithm says so. The tests compare the two.

### Prediction

Unchanged: it uses the stored coefficients and projection data, chunk by
chunk, for every algorithm.

### Expected result

| | Now | Kernel engine |
|---|---|---|
| Fit, 396 000 × 176, 7 folds, A_max 15 | 81 s | about 4–5 s (mostly loading, reference correction, SNV) |
| Peak memory | 2.1 GB | about 0.45 GB (the float32 matrix plus small arrays) |
| Full-size copies of X | 9 float64 and 7 float32 fold copies | none |

## Contract changes (Phase 0, with an independent review)

- `sample-set-model-run-contract_v3.md`
  - §7: PLS `hyperparameters` gain `algorithm` (`nipals`, `kernel`;
    missing reads `nipals`; new fits write `kernel`; a refit uses the
    stored value). PCR gains `algorithm` (`svd`, `gram`; missing reads
    `svd`; new fits use `gram` under the guard of D5 and store the
    algorithm used). §7.5's "deterministic full SVD" sentence covers both
    PCR algorithms, with scikit-learn's sign convention, and the first *a*
    components are those of an *a*-component fit in both.
  - §7: the algorithm does not change what a model means; §5, §7.2, §7.4
    and §7.5 hold for every algorithm to floating-point precision.
- `project-schema-contract_v3.md` §5: no change (a new key in a declared
  structured field).
- `docs/hdf5_format.md`: `hyperparameters/@algorithm`.
- `docs/contract_coverage.md`: new gap G33 (kernel engine), closed by
  Phases 1–3b.
- No change to the prediction runtime contract.

## Phases

### Phase 0: contracts

The contract changes above, reviewed independently. Gap G33.

### Phase 1: statistics and kernel engine behind `PLSRegression`

`_gram.py` (`GramStats`, `kernel_pls`), `PLSRegression(algorithm=...)`
with `fit_stats(stats)`; `fit(X, y)` builds the statistics and calls it.
The same public attributes as today. The BLAS thread cap (D7).

Tests:

- equal to NIPALS within 10⁻⁸ relative for every scaling (center, uv,
  pareto), PLS1 and PLS2, float32 and float64 input, K > N and N ≫ K:
  coefficients, W, P, Q, R, T and predictions; weight signs equal
- equal to ikpls (test oracle)
- the SIMCA reference-blend parity tests pass with `kernel`
- raw counts with an offset of 3000 and a small signal stay within
  tolerance; fault-injected: no shift before accumulating fails
- statistics summed from blocks equal the statistics of the joined rows
- memory: a fit of 400 000 × 176 float32 peaks below a quarter of today's
  float64 copy (tracemalloc)
- a model without `algorithm` refits with NIPALS and reproduces its
  metrics exactly; a `kernel` model refits with kernel and reproduces them

### Phase 2: the fit on statistics

`fit_model_run` builds `GramStats` per fold group, runs the CV and the
curve from them (layer 3), takes the final model from the A_max fit, and
runs the pixel pass (layer 4). The raw matrix is released early and the
preprocessed matrix filled ROI by ROI (D7). DModX exclusion and MSC/EMSC
as above.

Tests:

- for random calibration sets, every fold rule (round-robin,
  leave-one-group-out, GroupKFold) and every `group_by`, with and without
  a pixel subsample, with and without DModX exclusion, with SNV and with
  MSC/EMSC: the component curve, `rmsecv`, `pixel_rmsecv`, `cv_outputs`,
  the fold warnings, the stored diagnostics (T², DModX, scores, VIP,
  selectivity ratio, limits, shares) and both coverage metrics equal the
  NIPALS path within 10⁻⁸ relative
- fault-injected: the centering correction left out, a held-out block
  left in its fold's statistics (leakage), the final model refitted
  instead of cut
- a benchmark at demo-project size in the style of
  `test_plot_performance.py`, with a generous bound; measured times in the
  plan's status line

### Phase 3: band proposal on statistics

`propose_bands` uses `GramStats.select` and the nested folds' block sums.
Tests: the same ranges, steps, RMSECV (per ROI) and nested RMSECV as the
current implementation on the test sets. Benchmark before and after.

### Phase 3b: PCR from XᵀX (D5)

`PCR(algorithm=...)` with `fit_stats`, `gram_pca`, the guard, and the fit
of Phase 2.

Tests:

- against the SVD for components 1 … 15 on the SIMCA reference blends and
  on synthetic spectra like the measured cases (raw counts, absorbance,
  SNV, SNV + derivative): coefficients within 10⁻⁸ relative, loadings
  within 10⁻⁶ °, signs equal
- a request whose A_max approaches the rank limit uses the SVD and logs
  why; fault-injected: without the guard the comparison fails
- a PCR model without `algorithm` refits with the SVD and reproduces its
  metrics exactly; a `gram` model refits with `gram`

### Phase 4: viewer, docs and release

- No new control in the viewer. The Inspector's "how a model was fitted"
  lines show the algorithm; Fit Again keeps it; the fit's progress names
  the phase.
- User guide: a short "Fitting speed" section; API reference for
  `algorithm`; `docs/hdf5_format.md`; CHANGELOG (1.6) with times measured
  on an idle machine.
- Four gates.

## Later (2.0, separate plan)

- NIPALS and the full SVD removed (D2, D5).
- **OPLS** as model type `OPLS` (predictive and orthogonal components from
  XᵀX); needs the model-run and diagnostics contracts.
- **EPO and OSC** as stateful recipe steps, stored with the model.
- **Calibration transfer**: PDS and DS need fast refits on transfer
  samples.
- **Weighted fits** (equal weight per ROI or sample): weights scale the
  blocks; per-ROI weights need per-ROI blocks (a change of the block map,
  D6).
- SIMCA and the LDA dimension reduction on `gram_pca`.

## Out of scope

- GPU or JAX back ends.
- Changing any definition of the fit (folds, metrics, DModX, the component
  rules).
- Faster prediction (already chunked; dominated by I/O and preprocessing).
- SVR, random forest and MLP: they keep the pipeline path.
