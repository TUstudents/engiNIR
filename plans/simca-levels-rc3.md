# Plan: SIMCA's 0.95 limits, and predicted ROIs outside the model (engiNIR 1.7.0rc3)

Status: decided (user, 2026-11-18/19): D1 to D9 below. Not started.

## Goal

1. **Use SIMCA's default significance level, 0.95, for the DModX and T²
   limits** wherever engiNIR uses 0.99 today: the fit's DModX exclusion, the
   prediction exclusion, the limit beyond which a pixel is outside the model
   (no prediction interval, counted outside in summaries), and the
   per-ROI shares the fit stores. The format `enginir-1.7` (not final yet)
   takes `dcrit_095` and `t2_crit_095` as its core limits, and the migration
   tool adds them to older projects.
2. **Add the check "predicted ROIs outside the model"** (G34), with the
   Predict page's rule at the new level, moved into the library.
3. **Record that clipped reflectance is not a check.**

With these, G34 closes. It ships as `1.7.0rc3`, so the final 1.7.0 has
the format change, the window lifetime and the 0.95 limits together.

## Not a check: clipped reflectance (decided by the user, 2026-11-18)

Reflectance above the white reference is light reflected directly from the
surface (specular reflection, glare), not a fault of the data or the
reference correction. It is irrelevant to the model, and the DModX outlier
exclusion removes those pixels anyway. The conversion keeps counting and
logging clipped values as the dataset-reference-correction contract
requires; nothing is stored for them and no check reads them.

## What exists (measured 2026-11-19)

- **Every DModX limit uses 0.99.** The 0.99 level appears in 67 lines of
  library code in 17 files, plus 8 contracts, 9 docs and 13 test files. It
  covers:
  - the limits stored with every model (`prediction_model/@dcrit_099` and
    `@t2_crit_099`, core fields checked at open by
    `project_store._check_executable_model`; the same in `diagnostics`);
  - the per-ROI shares (`roi_share_above_dcrit`, `roi_share_above_t2crit`);
  - the fit exclusion default (`FitSpec.exclusion_level = 0.99`);
  - the prediction exclusion default (preference `dmodx_level`, the
    Prediction settings);
  - the Model page's level box;
  - the interval map, which blanks pixels above DCrit or T²Crit at 0.99
    (`prediction/_uncertainty.py`, including the ROI-level T² limit);
  - the prediction summaries (`Prediction.diagnostics` keys `dcrit_099` and
    `t2_crit_099`, the counts above them);
  - `predict_rois` (`dcrit_099`, `training_share_range` at 0.99).
- **0.95 exists only** for the prediction intervals themselves
  (`interval_map(level=0.95)`) and the stored `t2_crit_095` and
  `t2_crit_2_095` (diagnostics, plot lines).
- **Guidance:** `model.distance` needs attention above a 5 % pixel-weighted
  share above DCrit, which is five times the 1 % expected at 0.99
  (`DMODX_SHARE_ATTENTION`).
- **Prediction ROIs:** `predict_rois` needs only the project file. The
  Predict page rings an "unusual" ROI when its share above DCrit exceeds
  the 90th percentile of the calibration ROIs, and that rule lives in the
  viewer (`viewer/_predict_dialogs.py`), against `AGENTS.md`.
- **The DModX definition stays `dmodx-1`** (unweighted). SIMCA's default is
  MPow-weighted, and engiNIR keeps unweighted by contract decision: weighting
  turns down the bands where an unexpected component shows. Only the levels
  follow SIMCA.

## Decisions

**D1. Release (user).** Part of engiNIR 1.7.0, released as **`1.7.0rc3`**
after rc2. Branch `simca-levels` from `main` once rc2 is merged, one commit
per phase, a draft PR.

**D2. The level (user).** 0.95, SIMCA's default significance for DModX and
T², replaces 0.99 for:
- the fit's DModX exclusion default;
- the prediction exclusion default;
- the outside-the-model limit (no interval; counted outside in the
  summaries; the ROI-level T² limit of the mean interval);
- the per-ROI shares the fit stores.

A model fitted with an explicit level keeps it (its stored `fit_spec`). A
refit uses the stored level, so an old model refits at 0.99.

**D3. Core limits of format `enginir-1.7` (user).**
`prediction_model/@dcrit_095` and `@t2_crit_095` become the core limits,
checked at open, in place of `dcrit_099` and `t2_crit_099`. Every new model
also stores `dcrit_099` and `t2_crit_099` as display data (the Model page
draws both levels, as SIMCA does). The public API names follow: the keys of
`Prediction.diagnostics` and the `predict_rois` fields become `dcrit_095` and
`t2_crit_095`, with no aliases.

**D4. The stored ROI shares (user).** The fit stores
`roi_share_above_dcrit` and `roi_share_above_t2crit` at **0.95** instead of
0.99, under the same names; their meaning is documented as the 0.95 level.

**D5. The migration tool adds the 0.95 limits (user).** Converting a model
of an earlier file, `enginir-migrate`:
- computes `dcrit_095 = dmodx_limit(N, K, A, 0.95)` and
  `t2_crit_095 = A(N − 1)/(N − A) · F(0.95; A, N − A)` from the stored
  `n_train`, band count and components;
- writes them to `prediction_model` and `diagnostics`, and keeps
  `dcrit_099` and `t2_crit_099` as display data;
- renames the old per-ROI shares (0.99) to `roi_share_above_dcrit_099` and
  `roi_share_above_t2crit_099`, so no reader takes them for 0.95 shares.

A converted model shows the Distance page at 0.99 values until it is refit.
It has no 0.95 shares, so the outside-the-model check and the ring stay off
for it.

**D6. "Outside the model" for prediction ROIs (user): the Predict page's
rule at 0.95, moved into the library.** A prediction ROI is outside when its
share of pixels above DCrit (0.95) exceeds the 90th percentile of the
calibration ROIs' stored shares (D4). The rule is one library function,
`prediction.unusual_rois`, used by the check and by the viewer's ring. The
viewer then holds no rule.

**D7. Model and cost of the check (user).** The check uses the newest model,
as `predict.rois_ready` does. Results are cached per process by
`(model_run_id, roi_id)`, which is sound because neither changes under its
id. `Project.checks(predict=False)` skips the check. The viewer shows the
other checks at once with `predict=False` and adds this one from a
background job.

**D8. The `model.distance` threshold (user).** At 0.95 about 5 % of normal
pixels lie above DCrit by construction, so today's 5 % line would fire on
almost every model. The rule becomes **three times the expected share**:
**15 %** at 0.95 (`DMODX_SHARE_ATTENTION` derived as 3 × (1 − level)). On
the user's Teflon/API layer project, 15–22 % of training pixels were above
DCrit at 0.99 (tray-to-tray noise, `dmodx-limit-finding`). At 0.95 that
share will be higher, so the check will ask for attention there; that is
intended.

**D9. Projects written by 1.7.0rc1 and rc2 (user): no backward
compatibility.** They say `enginir-1.7` but have `dcrit_099` as their core
limit. The reader refuses them as missing a core field (`dcrit_095`), with
the message it gives for any missing core field. `enginir-migrate` does not
convert them; they must be created again. They were release candidates.

## Phases

### Phase 1: the 0.95 limits in the library and the format

- `modeling/_components.py`:
  - store `dcrit_095` and `t2_crit_095` in `prediction_model` (core) and
    `diagnostics`, with `dcrit_099` and `t2_crit_099` as display data;
  - compute the per-ROI shares at 0.95 (D4).
- `project_store._check_executable_model`: core `dcrit_095` and
  `t2_crit_095` in place of the 0.99 pair.
- `FitSpec.exclusion_level` default 0.95.
- `prediction/_diagnostics.py`, `_service.py`, `_uncertainty.py`,
  `_rois.py`: the 0.95 limits for the maps, the summaries, the intervals and
  the ROI-level T² limit. Public keys `dcrit_095` and `t2_crit_095`.
- `_guidance.py`: `DMODX_SHARE_ATTENTION` per D8.
- Contracts:
  - project-schema (core fields, the display pair);
  - sample-set-model-run §7.2 (default level), §7.5 (shares at 0.95);
  - prediction-runtime (diagnostics, intervals, Prediction ROIs);
  - headless-workflow (Guidance threshold);
  - `docs/hdf5_format.md`.
- Tests:
  - DCrit and T²Crit at 0.95 equal the formulas (the same formulas matched
    SIMCA 18.1 at both levels in Phase R of `plans/fit-specification.md`;
    the exports are not committed);
  - the shares at 0.95 equal an independent count;
  - a model without `dcrit_095` is refused at open;
  - fault-injected: the 0.99 level left in the exclusion default, the
    shares, the intervals and the core check, each one at a time.

### Phase 2: migration

- `enginir-migrate` adds the 0.95 limits and renames the old shares (D5)
  for `enginir-1.5` files. `enginir-1.7` files from rc1 and rc2 are not
  converted (D9).
- Tests on the committed `enginir-1.5` project:
  - the added limits equal the formulas from its `n_train`, K and A;
  - the old shares are kept under the `_099` names;
  - the converted model predicts its stored map;
  - an rc-written `enginir-1.7` file (made in the test by removing
    `dcrit_095`) is refused by the reader, naming `dcrit_095`, and by
    `enginir-migrate`;
  - fault-injected: a limit computed with the wrong degrees of freedom.

### Phase 3: predicted ROIs outside the model

- `prediction.unusual_rois(summary)` (D6), the check `predict.rois_outside`
  (step `predict`, `attention`, fix `select_roi`), "all within the model"
  in `predict.rois_ready` when none is outside, the cache and
  `predict=False` (D7).
- Contract: a row in the Guidance table. The sentence "Not yet checks"
  becomes "Not a check: reflectance clipped by the reference correction
  (specular reflection; removed by the DModX exclusion; decided
  2026-11-18)".
- Tests:
  - an odd prediction ROI is named and a normal one is not;
  - a second call makes no prediction (the cache);
  - `predict=False` makes none;
  - a converted model without 0.95 shares gives no check;
  - fault-injected: the 50th percentile instead of the 90th, and the oldest
    model.

### Phase 4: viewer

- The Predict page's ring uses `unusual_rois`; its own comparison is
  removed.
- The level defaults move to 0.95: the preference `dmodx_level`, the
  Prediction settings and the Model page's level box. The DCrit labels say
  0.95. The Distance page draws both levels.
- `refresh_checks` adds the prediction check from a background job, and
  drops a result for a project that changed meanwhile.
- Tests:
  - the defaults are 0.95;
  - the ring marks exactly `unusual_rois`;
  - the panel shows the checks at once and the prediction check after the
    job;
  - a stale result is ignored;
  - the `select_roi` fix works.

### Phase 5: docs and release

- User guide: the fit specification (exclusion level), the viewer (Model
  Distance, Predict ring, Prediction settings), prediction (intervals and
  limits) and the Guidance check. `docs/viewer_architecture.md` for the
  background check. `contracts/coverage.md`: G34 closed, with the clipping
  decision.
- CHANGELOG `1.7.0rc3`:
  - Changed: the 0.95 defaults; the core limits of `enginir-1.7`; the
    renamed public keys;
  - Added: the check;
  - Migration: `enginir-migrate` adds the 0.95 limits to `enginir-1.5`
    projects. Projects made with 1.7.0rc1 or rc2 must be created again (no
    conversion).
- All four gates and the full CI run. Then version `1.7.0rc3`, tag on
  `main`, and a GitHub pre-release.

## Acceptance

- New models store and use 0.95 limits, and pass the formula and SIMCA
  checks. Each place where 0.99 is put back makes its fault-injected test
  fail.
- `enginir-1.5` projects convert, open and predict their stored maps.
  Projects of rc1 and rc2 are refused with the missing core field named.
- The check names an odd prediction ROI in `Project.checks()` and in the
  viewer, and the ring and the check agree.
- No rule remains in the viewer for the ring.

## Risks

- **More pixels excluded and flagged.** At 0.95, the exclusion removes
  about 5 % of normal pixels per round instead of 1 %, and more pixels are
  outside the model in prediction maps. This is SIMCA's behavior; the user
  guide says so. On data with tray-to-tray noise
  (`dmodx-limit-finding`) the shares rise further; D8's threshold has to be
  checked on that project.
- **Size of the change.** About 150 lines of code, contract and doc text,
  and several public keys. One phase per commit, each passing the four
  gates; the fault-injected tests guard each place.
- **rc projects.** Users of rc1 and rc2 must create their projects again
  (D9). The rc3 release notes say so at the top.

## Out of scope

- The DModX definition (stays `dmodx-1`, by contract).
- A T² rule for prediction ROIs (the check uses the DModX share).
- Any clipping check.
