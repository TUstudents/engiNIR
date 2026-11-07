# Plan: SIMCA preprocessing (AsLS baseline correction, Savitzky-Golay without edges)

Status: decided (2026-11-07), not started

## Goal

Two preprocessing behaviours SIMCA 18 has and engiNIR lacks, so that models
built in both programs start from the same spectra:

- **AsLS baseline correction** (asymmetric least squares, Eilers): spectrum
  − fitted baseline.
- **Savitzky-Golay without edge bands**: SIMCA sets the first and last
  ⌊window/2⌋ bands of a Savitzky-Golay filter to missing and leaves them out
  of the model (K = 86 of 100 bands for a 15-point window); engiNIR
  extrapolates them today (`scipy.signal.savgol_filter`, `mode="interp"`).

Both were verified against SIMCA 18.1 exports (fit-specification plan,
Phase R): AsLS with second differences, weights p and 1 − p, 10 iterations,
λ 10000, p 0.001 matched to float32 precision; the Savitzky-Golay interior
matched, SIMCA dropping the edges.

## Decisions (taken with the user 2026-11-07)

1. **AsLS is first** in the fixed order and does **baseline correction
   only** (no AsLS smoothing): AsLS → Savitzky-Golay → SNV → MSC → EMSC →
   Detrend. Savitzky-Golay remains the step for smoothing or derivatives.
2. **Direction by signal unit.** AsLS assumes peaks point up. In absorbance
   it is used as is (p small: the baseline lies under the spectrum). In
   reflectance, where absorption bands are dips, it is flipped (weights
   1 − p: the baseline lies over the spectrum). Raw DN is treated like
   reflectance (absorption lowers the counts); to be confirmed by the user.
3. **AsLS has no viewer controls.** Its options exist only in scripts
   (`PreprocessingRecipe(asls=True, asls_lambda=..., asls_p=...,
   asls_iterations=...)`). The viewer shows such a recipe (label "AsLS")
   and keeps it when refitting a stored model, but its fit dialog and
   Explore cannot add it.
4. **Existing recipes keep their ids.** New fields are left out of
   `to_dict()` (and so of the id) while they are at their values for stored
   recipes; `from_dict` accepts recipes without them.
5. **Savitzky-Golay drops the edges by default, smoothing and derivatives
   alike: a breaking change.** New recipes with any Savitzky-Golay filter
   (`sg_smooth`, `sg_d1`, `sg_d2`) drop ⌊window/2⌋ bands at each end. A stored
   recipe without the field keeps extrapolating, so stored models refit
   unchanged; the new default is written explicitly (`sg_edges = "drop"`)
   and therefore gets a new id. The CHANGELOG lists it under breaking
   changes; the release that ships it should say so in its version (see
   below).

## Design

### AsLS

- Recipe fields: `asls` (bool, default False), `asls_lambda` (default
  10000), `asls_p` (default 0.001; note the SIMCA Python API defaults to
  0.01), `asls_iterations` (default 10). Scripts only.
- Algorithm: for every spectrum z, solve (W + λDᵀD) b = W z with D the
  second-difference matrix, starting with W = I, then wᵢ = p where zᵢ > bᵢ
  and 1 − p otherwise (flipped in reflectance and DN: wᵢ = 1 − p where
  zᵢ > bᵢ, p otherwise); `asls_iterations` solves. The result is z − b.
- λ depends on the number of bands (a camera with more bands needs a larger
  λ for the same stiffness, roughly with the fourth power of the band
  count): the docs say so, and give an example of checking the baseline in a
  plot.
- Cost: the system is pentadiagonal; a banded solver
  (`scipy.linalg.solveh_banded`) is O(K) per solve, 10 solves per spectrum.
  For 10⁶ pixels × 200 bands about 2·10⁹ operations: seconds to a few
  minutes, done chunk by chunk in prediction like the other steps. Measure;
  solving a batch of spectra per call is the first optimisation if needed.
- Stateless per spectrum, so it is applied once before cross-validation
  like SNV, never refitted per fold.
- The signal unit is known at fit time (signal domain) and at prediction
  (the model's signal domain), so the direction is fixed per model.

### Savitzky-Golay without edges

- Recipe field `sg_edges` = `"drop"` (default for new recipes) |
  `"interp"` (stored recipes, or chosen in scripts).
- The step returns K − 2⌊window/2⌋ bands and the wavelength axis shrinks with
  it: later steps (EMSC and Detrend use the trimmed axis), the model's stored
  `wavelengths`, coefficients, loadings and plots use the trimmed axis, and
  prediction trims before the model. A step that changes the number of bands
  is new in engiNIR: the recipe reports its output axis
  (`recipe.output_wavelengths(wavelengths)`), and every check that compares
  model and data wavelengths or band counts uses it.
- `enginir.preprocessing.SavitzkyGolay` gets `edges="interp" | "drop"`.
  Proposed default for the transformer used directly: `"interp"` (it keeps
  the input shape, as scikit-learn pipelines expect); recipes set `"drop"`.
- The viewer's fit dialog and Explore preview show SG with dropped edges by
  default (no new control; scripts can choose `interp`).

### Versioning

The new SG default changes the numbers of new models with SG
preprocessing and changes the number of bands of their spectra. engiNIR
follows semantic versioning in spirit: this is a breaking change for scripts
and should be released as **2.0.0**, or as 1.x with a prominent breaking-
change note; decided at release time.

## Contract changes (with an independent review)

- `sample-set-model-run-contract_v3.md` / preprocessing parts: the new
  recipe fields, the fixed order, AsLS as stateless, the shrinking wavelength
  axis with `sg_edges = drop` and what it means for the model's wavelengths
  and prediction.
- `docs/hdf5_format.md`: recipe fields.
- `prediction-runtime-contract_v3.md`: the trimmed axis in the band checks.
- Ledger: new gap G26.

## Phases

1. **AsLS** in `enginir.preprocessing` (`AsLSBaseline` transformer,
   scikit-learn API, `direction="up" | "down"`) with the banded solver. Tests:
   equal to a dense reference solver; a synthetic spectrum with a known
   curved baseline and peaks: the baseline removed, the peaks kept within
   tolerance, in absorbance and (flipped) in reflectance; constant and
   linear baselines removed; float32 in, float32 out; memory per chunk.
2. **Recipe fields and order**, ids kept stable, the direction from the
   signal unit, AsLS labels in the viewer. Tests: recipe round trip; stored
   recipes load with the same id; the order; fault injection (reflectance not
   flipped must fail).
3. **Savitzky-Golay `drop`** as the recipe default, with the output axis
   through recipe, model, prediction and plots. Tests: K − 2⌊w/2⌋ bands;
   interior equal to `interp`; a stored recipe without the field keeps
   extrapolating and refits unchanged; prediction of a new scan uses the
   trimmed axis; fault injection (model applied to untrimmed spectra fails
   the check).
4. **Docs**: user guide (preprocessing: AsLS with an example of choosing λ
   and p, edge handling, comparison with SIMCA), API reference, CHANGELOG
   with the breaking change, four gates.

## Out of scope

- SIMCA's "filter once" for stateful filters (MSC/EMSC fitted once on all
  data): not offered, it leaks held-out samples into the reference.
- Other SIMCA filters (OSC, wavelets, EWMA).
