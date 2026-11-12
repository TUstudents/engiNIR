# Plan: kernel PLS engine and faster cross-validation

Status: proposed (2026-11-11); D5 decided with the user (2026-11-11), D1–D4 to confirm

## Goal

Fitting a model on an HSI calibration set takes too long. A realistic fit
is dominated by scikit-learn's NIPALS PLS, and every further feature
(component curve, DModX exclusion, band proposal, OPLS, calibration
transfer) multiplies that cost. This plan replaces the numerical engine of
PLS (and later PCR) with a kernel algorithm on XᵀX. Every stored model
stays reproducible and every result the user sees stays the same within
floating-point noise. It lays the base for OPLS, EPO and OSC in 2.0.

## Background (measurements 2026-11-11)

**A real fit.** Copy of the Teflon project: 20 calibration ROIs, 396 000
pixels, 176 bands, PLS 7 LV, SNV, 7 round-robin folds, component curve up
to 15. Profiled with cProfile:

| Part | Time | Share |
|---|---|---|
| `fit_model_run` (total) | 46.7 s | 100 % |
| scikit-learn NIPALS (9 fits: 7 folds, 1 curve fit, final model) | 34.4 s | 73 % |
| engiNIR's `PLSRegression.fit` wrapper beyond that (float64 copy of X, scaling) | 3.7 s | 8 % |
| `training_diagnostics` | 2.4 s | 5 % |
| loading spectra, reference correction, SNV, saving | ≈ 2.5 s | 5 % |

The fit already gets the predictions for every number of components
from one fit at Amax per fold (`AllComponents`), so the cost is NIPALS
itself, not the number of candidates.

**A prototype.** Improved kernel PLS (Dayal and MacGregor 1997, algorithm 2)
in numpy, on 396 000 × 176 synthetic spectra with 15 components:

| | scikit-learn NIPALS | Kernel PLS |
|---|---|---|
| One fit | 5.2 s | 0.13 s (40×) |
| 7-fold cross-validation | 27.5 s | 1.1 s (25×), fold XᵀX by subtraction |
| Coefficients | | equal to 3.6·10⁻¹⁰ (coefficients up to 0.016) |

**Why it works.** A calibration set has N ≫ K: many pixels, few bands. One
pass over the pixels builds XᵀX (K × K) and Xᵀy; every component, fold and
band candidate then works on those small matrices.

**Per-ROI blocks.** engiNIR's folds never split an ROI (contract §3–4). So
the sufficient statistics of every ROI can be computed once:

- nᵣ (pixels), sᵣ = Σx, Gᵣ = XᵣᵀXᵣ, gᵣ = Xᵣᵀyᵣ, Σy, Σy²

Here yᵣ is the ROI's target value repeated over its pixels, so Xᵣᵀyᵣ =
yᵣ sᵣ. Then:

- the statistics of any fold's calibration part, under any fold rule and
  `group_by`, are sums of ROI blocks. Centering and scaling of that part
  come from those sums (Engstrøm and Jensen 2025).
- the held-out ROI prediction (the ROI mean of its pixel predictions,
  §5) equals the prediction of the ROI's mean spectrum sᵣ/nᵣ, because the
  model is linear on per-pixel preprocessing. So RMSECV (per ROI) and the
  component curve need no pass over the held-out pixels.
- the per-pixel sums of squared residuals of a held-out ROI are
  Σ(y − xb − b₀)² = nᵣy² − 2y(sᵣb + nᵣb₀) + bᵀGᵣb + 2b₀sᵣb + nᵣb₀². So
  RMSECV (per pixel) also needs no pixel pass.

Memory: 20 ROIs × 176² × 8 B ≈ 5 MB of blocks, instead of the float64 copy
of X (≈ 560 MB) that every fit makes today.

**Existing work.** [ikpls](https://github.com/sm00thix/ikpls) (Engstrøm et
al., JOSS 2024, doi:10.21105/joss.06533) implements IKPLS and the fast
partition-based cross-validation (arXiv 2401.13185) in numpy and JAX. A
2026 follow-up (arXiv 2607.16138) speeds up the R and Q steps by about 2×.

## Decisions (proposed, to confirm with the user)

- **D1. Own engine; ikpls as a test oracle.** The algorithm is short. The
  speed comes from BLAS, not the library. engiNIR must control its numerics,
  because a refit must reproduce a stored model: fold groups by ROI, DModX
  exclusion inside folds, SIMCA scaling, float32 inputs. ikpls becomes a
  test-only dependency (`dev` extra) to cross-check results. JAX or GPU is
  not needed at these sizes.
- **D2. The engine is recorded with the model.** A new hyperparameter
  `algorithm`: `nipals` or `kernel`, written for every new run, so the
  model stays `Backend(**hyperparameters)`. An absent key reads `nipals`
  (every model before this release), so legacy refits use NIPALS and
  reproduce exactly. New runs use `kernel`. "Fit Again..." keeps the stored
  algorithm. A user can request `nipals` in `project.fit` for comparison;
  the viewer offers no choice.
- **D3. Equivalence, not bitwise identity.** Kernel and NIPALS give the
  same model in exact arithmetic. Tests require agreement to a relative
  10⁻⁸ on coefficients, weights, loadings, scores and every stored metric.
  Ties between component rules that differ by floating-point noise can fall
  differently; contract §7.4 already says a refit reports that.
- **D4. Release split.** Phases 1–3 change speed and memory, not results.
  They go into a 1.x release (1.6). The contract change is the `algorithm`
  key, for PLS and PCR (D5). OPLS, EPO and OSC
  add a model type and recipe steps: 2.0, in their own plan, with
  calibration transfer and the method file.
- **D5. PCR from XᵀX, with a conditioning guard (decided 2026-11-11).**
  An eigendecomposition of XᵀX squares the condition number, so only
  components whose singular value is far below the first lose accuracy.
  Measured on 592 800 real tablet spectra × 220 bands (raw counts,
  absorbance, + SNV, + SG 1st derivative): over components 1–15 the worst
  angle to the SVD's components is 3·10⁻⁶ °, and the PCR coefficients for
  A = 1 … 15 agree to 2·10⁻¹¹ relative. σ₁/σ₁₅ ≤ 160, while σ₁/σ_K reaches
  7.8·10⁸ only in the trailing directions that SNV and the derivative
  remove. Time: 0.17–0.5 s against 13 s for the full SVD, per fit; today
  every fold, the curve fit and the final model each do a full SVD.
  - PCR `hyperparameters` gain `algorithm`: `svd` or `gram`. An absent key
    reads `svd`, so every stored PCR model refits as before.
  - A new fit uses `gram` when σ_Amax/σ₁ ≥ 10⁻⁶ (Amax the largest number of
    components the fit computes), otherwise the full SVD with a logged
    reason. The stored value is the algorithm used, so a refit runs the same
    one.
  - No separate path for the pre-1.2 `scale` form: in the `gram` engine
    `scale=True` means `uv` (standard deviation with N − 1). This is a
    uniform factor on every band, so components, scores up to that factor,
    coefficients and predictions are unchanged. Stored pre-1.2 runs have no
    `algorithm` key and read `svd` like every run before this release.
  - Loadings and scores keep scikit-learn's PCA sign convention, so stored
    plots do not flip between engines.
  - Nearly equal eigenvalues (λ_A ≈ λ_A+1) make the split between
    components A and A + 1 arbitrary for every algorithm. The guard does not
    address this; a refit reproduces it because it runs the stored
    algorithm.

## Contract changes (Phase 0, with an independent review)

- `sample-set-model-run-contract_v3.md`
  - §7 (scaling paragraph and the hyperparameters note): PLS
    `hyperparameters` gain `algorithm` (`nipals`, `kernel`). An absent key
    reads `nipals`. New runs write `kernel`. The refit (§6) uses the stored
    value. PCR `hyperparameters` gain `algorithm` (`svd`, `gram`): absent
    reads `svd`, new runs use `gram` under the conditioning guard of D5 and
    store the algorithm used. §7.5's "deterministic full SVD" sentence is
    rewritten: PCR's components come from a deterministic full SVD of X
    (`svd`) or a symmetric eigendecomposition of XᵀX (`gram`), with
    scikit-learn's sign convention, and the first *a* components are those
    of an *a*-component fit in both.
  - §7 states that the algorithm does not change what a model means: the
    definitions of §5, §7.2, §7.4 and §7.5 hold for both engines, to
    floating-point precision.
- `project-schema-contract_v3.md` §5: no change (a new key in a declared
  structured field).
- `docs/hdf5_format.md`: `hyperparameters/@algorithm`.
- `docs/contract_coverage.md`: new gap G31 (kernel engine), closed by
  Phases 1–3.
- No change to the prediction runtime contract: prediction uses the stored
  coefficients and projection data, which keep their shapes and meaning.

## Design

### The engine (`enginir/models/_kernel.py`, private)

- `GramStats`: n, Σx, XᵀX, Xᵀy, Σy, ΣyᵀY in float64, built from float32
  row blocks (default 65 536 rows). Before accumulating, a fixed shift is
  subtracted: the mean of the first block. This keeps raw counts around
  3000 from cancelling when centering later. Methods: `+` and `−` (blocks
  combine and subtract), `centered()`, `scaled(x_scaling, y_scaling)` (uv
  and pareto from the diagonal, with N − 1 as contract §7 requires),
  `select(columns)` (a band subset).
- `ikpls(XtX, XtY, A)`: algorithm 2, PLS1 and PLS2. For PLS2, w is the
  dominant eigenvector of XᵀYYᵀX. Returns W, P, Q and R = W(PᵀW)⁻¹ for
  1 … A.
- Sign convention: as scikit-learn's NIPALS (tested), so stored weights,
  loadings and score plots do not flip between engines.
- Scores T = Xs R and `x_scores_` are computed in one blocked pass after
  the fit, only where a stored artifact or a diagnostic needs them (the
  final model). The stored `x_scores_` keep their meaning.

### `PLSRegression`

- New parameter `algorithm="kernel"`. The scikit-learn style default for
  library users becomes `kernel`. The project layer reads an absent stored
  key as `nipals` (D2).
- The same attributes as today: `coef_`, `intercept_`, `x_mean_`,
  `x_std_`, `y_mean_`, `y_std_`, `x_weights_`, `x_loadings_`,
  `y_loadings_`, `x_rotations_`, `x_scores_`, `y_scores_`. `y_scores_`
  are computed on demand.
- `fit_gram(stats)`: a fit from sufficient statistics, used by the CV path.
  `fit(X, y)` = build `GramStats` from X, then `fit_gram`.

### The cross-validation path (`modeling`)

- **Stateless recipe** (no MSC or EMSC, the usual case) and **no DModX
  exclusion**:
  1. One pass over the preprocessed drawn pixels builds a `GramStats` per
     ROI (and the mean spectrum of every ROI).
  2. Per fold, the calibration statistics are the total minus the held-out
     ROIs' blocks. `ikpls` runs to Amax.
  3. Held-out ROI predictions for every a ≤ Amax come from the ROI mean
     spectra. Pixel-level PRESS comes from the block identity above. That
     gives the component curve, `rmsecv`, `pixel_rmsecv` and `cv_outputs`.
  4. One pass over the held-out pixels, each pixel predicted once by its
     fold's model, gives the per-pixel held-out predictions that
     `interval_coverage_pixel` needs.
- **MSC or EMSC** (stateful, fitted per fold): per fold, the calibration
  pixels are preprocessed with the fold's state and a `GramStats` is
  built. This is a full pass per fold as today, still with the kernel
  engine's speed. The held-out ROIs are preprocessed with the fold state
  and predicted from their mean spectra.
- **DModX exclusion** (inside every fold, §7.2): the pixel DModX of the
  fold's calibration pixels needs their residuals, which means a pass over
  them per round. Excluded pixels are subtracted from their ROI blocks.
  Correct first, fast later: the first version keeps a pixel pass per
  round.
- The pipeline path (`cross_validate` with cloned estimators) stays for the
  arrays-only API, for SVR, random forest and MLP, and as the reference the
  tests compare the fast path with.

### Band proposal (iPLS)

A candidate band set is `GramStats.select(columns)`. All candidates, steps
and nested folds reuse the per-ROI blocks without a pass over pixels. The
proposal's rules (§ Band proposal of the headless contract) do not change.

## Phases

### Phase 0: contracts

The contract changes above, reviewed independently. Ledger gap G31.

### Phase 1: kernel engine behind `PLSRegression`

`_kernel.py` (`GramStats`, `ikpls`), and `PLSRegression(algorithm=...)` with
`fit_gram`. The fit records `algorithm`; an absent key reads `nipals`.

Tests:

- Equal to scikit-learn NIPALS within 10⁻⁸ relative for every scaling
  (center, uv, pareto), PLS1 and PLS2, float32 and float64 input, K > N and
  N ≫ K. Compares coefficients, W, P, Q, R, T and predictions. Weight
  signs match.
- Equal to ikpls (test oracle).
- SIMCA reference blends: the existing parity tests pass with `kernel`.
- Raw counts with an offset of 3000 and a small signal: the kernel
  coefficients stay within tolerance of NIPALS. Fault-injected: no shift
  before accumulating, and the test fails.
- Memory: a fit of 400 000 × 176 float32 peaks below a quarter of the
  current float64 copy (tracemalloc).
- A legacy run (no `algorithm`) refits with NIPALS and reproduces its
  metrics exactly. A new run refits with kernel and reproduces its metrics
  exactly.

### Phase 2: per-ROI cross-validation

`GramStats` per ROI, the stateless and stateful paths above, and the
held-out pixel pass. `fit_model_run` chooses the path; the result does not
depend on it.

Tests:

- For random calibration sets, every fold rule (round-robin,
  leave-one-group-out, GroupKFold for legacy) and every `group_by`, with
  and without a pixel subsample: the component curve, `rmsecv`,
  `pixel_rmsecv`, `cv_outputs`, the fold warnings and both coverage
  metrics equal the pipeline path within 10⁻⁸ relative.
- MSC and EMSC recipes: equal to the pipeline path.
- Fault-injected: the fold correction for centering left out, and a
  held-out ROI left in its own calibration statistics (leakage); each must
  fail.
- A realistic benchmark (Teflon size) in the style of
  `test_plot_performance.py`, with a generous bound and the measured times
  in the plan's status line.

### Phase 3: band proposal on Gram blocks

`propose_bands` uses `GramStats.select`. Tests: the same ranges, steps,
RMSECV (per ROI) and nested RMSECV (per ROI) as the current implementation
on the test sets. Benchmark before and after.

### Phase 3b: PCR from XᵀX (D5)

`PCR(algorithm=...)` with `fit_gram`. A symmetric eigendecomposition of the
centered and scaled XᵀX, the conditioning guard (σ_Amax/σ₁ ≥ 10⁻⁶, else the
full SVD with a logged reason) and scikit-learn's sign convention. PCR gets
the per-ROI cross-validation of Phase 2.

Tests:

- Against the SVD for components 1 … 15 on the SIMCA reference blends and
  on synthetic spectra like the measured cases (raw counts, absorbance,
  SNV, SNV + derivative): coefficients within 10⁻⁸ relative, loadings
  within 10⁻⁶ ° of the SVD's, signs equal.
- A request whose Amax approaches the rank limit of SNV or derivative
  spectra uses the SVD and logs why. Fault-injected: without the guard, the
  comparison with the SVD fails.
- `scale=True` without `x_scaling` gives the same predictions and
  coefficients as `x_scaling="uv"`.
- A stored PCR run without `algorithm` refits with the SVD and reproduces
  its metrics exactly; a `gram` run refits with `gram` and reproduces its
  metrics exactly.

### Phase 4: viewer, docs and release

- The viewer needs no new control. The Inspector's "how a model was
  fitted" lines show the algorithm. Fit Again keeps it. Fitting progress
  says which phase runs.
- User guide: a short "Fitting speed" section. API reference for
  `algorithm`. `docs/hdf5_format.md`. CHANGELOG (1.6).
- Four gates. Measured times before and after in the CHANGELOG.

## Later (2.0, separate plan)

These are built on this engine:

- **OPLS** as model type `OPLS`: one predictive component and orthogonal
  components from XᵀX. Predictions equal PLS with the same total number of
  components for one target; the value is interpretation. Needs the
  model-run and diagnostics contracts: predictive and orthogonal scores and
  loadings, and a Components page split into "1 + n".
- **EPO and OSC** as recipe steps, stateful like MSC and EMSC, stored with
  the model (EPO for the foil, moisture or temperature effect).
- **Calibration transfer** (plan to come): PDS and DS need fast refits on
  transfer samples.
- Weighted fits (equal weight per ROI or sample, out of scope in v3) become
  cheap: weights scale the ROI blocks.

## Out of scope

- GPU or JAX back ends.
- Changing any definition of the fit (folds, metrics, DModX, the component
  rules). This plan only changes how fast they are computed.
- Faster prediction. Prediction is already chunked and dominated by I/O and
  preprocessing; measure separately if needed.
- SVR, random forest and MLP. They keep the pipeline path.
