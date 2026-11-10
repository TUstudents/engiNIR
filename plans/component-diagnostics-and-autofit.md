# Plan: pixel and ROI metrics kept apart, explained plots, better autofit

Status: implemented (2026-11-09, branch feature/band-selection-vip); contracts reviewed independently. From the walkthrough of the first real model
(project `Teflon_10p`, 20 ROIs from 4 scans, levels 1/4/7/10, 396,000 pixels,
176 bands without wavelengths).

## Why

The walkthrough showed three problems with the Model workspace.

1. **Pixel and ROI metrics are mixed without saying so.** The Components page
   plots R²Y(cum) over the 396,000 pixels (0.67, as SIMCA) next to Q²(cum)
   over the 20 ROI means (0.81). Q² then looks better than R², which reads as
   a bug. Elsewhere it is similar: the Fit page's R², RMSECV and bias are per
   ROI, RMSEE and the T² ellipses were per pixel, the run also stores
   `pixel_r2`, `pixel_rmsecv`, `pixel_rpd` that no plot shows, and no plot
   says which level it uses.
2. **The rules are hard to follow.** "Q² limit: 1" is correct (Q²₂ = −0.02
   ≤ 0.05) but nothing on the page shows the per-component Q² it depends on.
3. **Autofit chooses too many components.** The default `one_se` rule chose
   7. Components 6 and up explain under 1 % of the spectra each, the
   regression vector becomes wiggly, and the user's experience is that more
   than about 5 latent variables fit noise on such data.

## Evidence from the first model

| a | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| RMSECV (ROI) | 3.01 | 2.84 | 2.46 | 2.01 | 1.88 | 1.66 | 1.59 | 1.49 |
| Q²ₐ (ROI) | 0.19 | −0.02 | 0.15 | 0.23 | −0.02 | 0.07 | −0.06 | −0.03 |
| ΔR²X (pixels) | 0.713 | 0.147 | 0.048 | 0.029 | 0.044 | 0.004 | 0.003 | 0.001 |
| Durbin–Watson of wₐ | 0.005 | 0.020 | 0.025 | 0.013 | 0.012 | 0.050 | 0.110 | – |

What the rules choose on this curve (G = 20 fold groups):

| rule | choice | why |
|---|---|---|
| `min_rmsecv` | 15 | RMSECV keeps creeping down (1.46 at 15) |
| `one_se` (default) | 7 | first a with RMSECV ≤ 1.459 + 0.164; 6 misses by 0.04 |
| `q2_limit` (SIMCA R1) | 1 | stops at the first Q²ₐ ≤ 0.05, i.e. at the scatter component LV2 |
| Haaland–Thomas F (α = 0.25) | 6 | first a with PRESS(a)/PRESS(min) < F(0.75; 20, 20) = 1.36 |
| R1 with one look-ahead | 6 | LV2 and LV5 kept because LV3 and LV6 are significant; stops at LV7, LV8 |
| noise components (ΔR²X < 1 % and Q²ₐ ≤ 0.05) | 7 and up | each adds < 1 % of the spectra and no prediction |
| Durbin–Watson of the weights | 5 | roughness rises ×4 at LV6 and ×9 at LV7 over LV5 |

Two observations:

- The spectral criteria (ΔR²X, Durbin–Watson) agree with the user: the
  structure ends at 5, LV6 and later are mostly noise and fine detail.
- The curve itself is optimistic: folds hold out single ROIs while the other
  ROIs of the same scan stay in training, and every scan holds one level.
  Grouping by scan changes the curve more than any rule does. A rule cannot
  repair an optimistic curve; the plots must make the validation visible.

## Proposal A: one vocabulary for the two levels

- Every metric, axis, legend entry, CSV column and summary row says its
  level: **per ROI** (the ROI means; the target, the folds and the reported
  model quality) or **per pixel** (single spectra; the observations of the
  fit, R²X, the pixel T² and DModX limits).
- Default to ROI level wherever a number answers "how good is the model".
  Pixel numbers stay where they answer "how noisy is a single spectrum" or
  where SIMCA defines them on observations (R²Y pixels, RMSEE pixels,
  T²/DModX limits).
- Fit summary: two groups, *Per ROI* (R², Q², RMSECV, RMSEE, bias, RPD) and
  *Per pixel* (R², RMSECV, RPD, n). The stored `pixel_*` metrics finally show.
- Text on every plot that mixes levels is not allowed; when both are useful
  they are drawn in the same plot only with the level in the legend (as the
  score plot's "T² 95 % pixels" / "T² 95 % ROIs").

## Proposal B: explanations in the plots

- `PlotSpec` gets a short `help` text (two to five sentences: what is drawn,
  at which level, how to read it, what a bad sign looks like). The plot
  header shows a **?** button that opens it; the same text goes into the
  user guide (one source, tested to match).
- Rule annotations say why: "one SE: 7 (RMSECV ≤ 1.62)", "Q² limit: 1
  (Q²₂ = −0.02 ≤ 0.05)", "noise from 7 (ΔR²X < 1 %, Q²ₐ ≤ 0.05)".

## Proposal C: the Components page, redrawn (2 × 2)

1. **RMSECV per ROI** with its SE, every rule's choice marked and explained,
   the model's number highlighted; RMSEE per ROI as a second line, so the
   gap (over-fit) is visible.
2. **R² and Q² per ROI**: R²Y(cum) of the ROI means and Q²(cum), the same
   level, so Q² ≤ R²Y always; the area between them shaded.
3. **Per component**: bars of Q²ₐ with the R1 limit line, and ΔR²X as a
   second bar series. Noise components (both small) drawn grey. This is
   where "Q² limit: 1" and "noise from 7" become visible.
4. **Per pixel**: R²X(cum) and R²Y(cum) over the pixels, with a note that
   single pixels are noisy, so R²Y per pixel stays below the ROI value.

Optional fifth plot (decision 4): the Durbin–Watson of each component's
weight vector, as a measure of how noise-like it is.

## Proposal D: better autofit

Keep the existing rules (they are stored in runs and documented) and add:

1. **`noise_guard`**, applied after any rule, on by default: drop trailing
   components that are both spectrally negligible and predictively useless,
   i.e. ΔR²X(a) < 1 % and Q²ₐ ≤ L (L as in R1). This is SIMCA's N4 idea
   (a component that explains under 1 % is not significant), combined with
   R1 so a small but predictive component survives. On the first model it
   turns `one_se`'s 7 into 6.
2. **`f_test`** (Haaland and Thomas 1988, *Anal. Chem.* 60, 1193): the
   smallest a whose PRESS is not significantly larger than the minimum,
   PRESS(a)/PRESS(a*) < F(1 − α; G, G), α = 0.25, with G the number of fold
   groups, not pixels. A standard in NIR calibration software; it chooses
   fewer components than `min_rmsecv` and does not need the SE estimate,
   which is unstable with few groups. First model: 6.
3. **`q2_lookahead`**: SIMCA R1, but an insignificant component is kept when
   the next one is significant (the "undecided" logic SIMCA applies to PCA).
   It removes the stop at a scatter component (LV2 here). First model: 6.
4. **A ceiling from the data**: Aₘₐₓ for autofit at most G − 2 and at most
   the number of components with ΔR²X ≥ 0.5 % plus one. With 4 scans held
   out as groups, G = 4 caps autofit at 2.
5. **Optional `durbin_watson`** (Rutledge and Barros 2002): stop before the
   first component whose weight vector's Durbin–Watson value jumps above a
   multiple of the previous ones. Spectrum-specific and independent of the
   folds, but its threshold is empirical; offer it as a diagnostic first,
   as a rule only after testing on the SIMCA reference blends and this
   project.

Recommended default: **`one_se` + `noise_guard`** (the current default made
safer, small change for existing users), with `f_test` and `q2_lookahead`
offered in the dialog and all suggestions shown on the curve. Whatever the
rule, the dialog warns when the fold groups are fewer than 7 or when every
group holds a single reference level (the scan confound seen here).

## Decisions (taken with the user 2026-11-09, as recommended)

1. **ROI level by default**; pixel metrics in their own *Per pixel* group and
   labelled as such wherever they appear (A).
2. **A ? button with a help text in every plot header**, the same text as the
   user guide (B).
3. **Components page 2 × 2 as in C**; the Durbin–Watson value of each
   component's weights is drawn in the per-component plot (its own axis),
   not as a fifth plot.
4. **Autofit**: default `one_se` + `noise_guard` (D1); `f_test` (D2) and
   `q2_lookahead` (D3) offered; the data ceiling (D4) applies to every
   autofit rule; Durbin–Watson only as a diagnostic (D5).
5. **Warn, do not refuse**, when every fold group holds a single reference
   level or there are fewer than 7 fold groups: the fit dialog and the
   readiness line say so, the run stores the condition, and the fit goes on.

## Deviations agreed in the contract review (2026-11-09)

- The structure ceiling belongs to `noise_guard`: `on` applies the ceiling
  and then drops trailing noise components; `off` reproduces the 1.3 choice
  (so refits of 1.3 runs choose as before). Decision 4's "ceiling for every
  autofit rule" holds for the default (guard on). The cap at G − 2 comes
  from `components/max`.
- Decision 5's "every fold group holds a single reference level" is
  implemented as the condition that matters: a data file whose ROIs share
  one value and are held out in more than one fold (`one_level_per_file`).

## Phases

0. **Contracts** (with an independent review): model-workspace diagnostics
   (levels, help texts, Components page), sample-set-model-run §7.4 (new
   rules and `noise_guard`, the ceiling; stored as rule names so old runs
   keep their meaning), `docs/contract_coverage.md`.
1. **Library**: ΔR²X per component and Durbin–Watson of the weights in the
   curve; `noise_guard`, `f_test`, `q2_lookahead`, the ceiling in
   `choose_components`; tests with hand-computed curves, each rule checked
   to fail when its condition is inverted; the SIMCA reference blends
   (Phase R of `component-selection.md`) and this project as fixtures for
   the expected choices.
2. **Viewer, levels**: metric labels and CSV columns with their level, the
   Fit summary in two groups, Fit page statistics box per ROI with a pixel
   line.
3. **Viewer, Components page**: the four plots of C, rule annotations with
   reasons, the dialog's rule list and warnings.
4. **Help texts**: `PlotSpec.help`, the **?** button, user guide sections
   generated from or checked against the same texts.
5. **Docs and changelog**: `user_guide/fit_specification.rst` (rules with
   the worked example above), `user_guide/viewer.rst`, API pages.

## Out of scope

- Changing how the curve is computed (fold refits stay; SIMCA's
  component-wise CV leaks held-out rows, see the SIMCA reference notes).
- Faster PLS (kernel algorithm): separate plan; it speeds every rule up.
- Automatic grouping by scan: the user decides the sample labels; the plan
  only warns.
