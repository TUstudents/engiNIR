# Plan: ROI scores stored by the fit (score plot with pixel sample or exclusion)

Status: implemented (2026-11-08, branch feature/weekend-plans)

## Problem

The Model workspace's score plot shows one point per ROI, the mean of that
ROI's training scores. The viewer rebuilt these means itself
(`ModelingView._extract_roi_scores`): it split the model's stored score rows
(`x_scores_` / `scores_`) into blocks of each ROI's full pixel count. The
rows are the pixels the final model was fitted on, so with a pixel sample
(`fit_spec.pixels.per_roi`) or DModX exclusion there are fewer rows and the
plot was empty; with more rows than pixels it averaged misaligned blocks
(only a warning). The viewer was reconstructing a workflow fact (which row
belongs to which ROI) that only the fit knows.

## Decisions

- The fit computes and stores the ROI scores: `diagnostics/roi_scores`
  (ROIs × components), the mean training score of each ROI's **fitted**
  pixels (the drawn pixels with a pixel sample, the kept pixels with
  exclusion), in `cv_outputs/roi_ids` order, with `roi_score_pixels` (the
  number of pixels behind each mean). An ROI whose pixels were all excluded
  has NaN scores and 0 pixels; the plot says how many ROIs have no point.
- Breaking, no backward compatibility (user, 2026-11-08): the viewer reads
  only `roi_scores`. Runs without it show a placeholder ("fit the model
  again"); the pixel-count reconstruction is removed.

## Phases

1. Contracts: model-workspace-diagnostics §2 Latent scores; sample-set-model-run
   §7.5; `docs/hdf5_format.md`; gap register.
2. Library: `training_diagnostics` stores `roi_scores`, `roi_score_pixels`,
   reordered like the per-ROI shares.
3. Viewer: payload from `roi_scores`; placeholder without it; count of ROIs
   without fitted pixels.
4. Tests (pixel sample, exclusion, both; independent means; placeholder;
   fault injection: all pixels instead of fitted, no reorder), docs,
   CHANGELOG, four gates, commit.
