"""ROI scores of the score plot, stored by the fit (model-workspace-diagnostics contract §2,
sample-set-model-run contract §7.5)."""

from __future__ import annotations

import numpy as np

from enginir.viewer._modeling import _roi_score_payload
from tests.test_exclusion import EXCLUSION, _project_with_an_odd_pixel

PIXELS = 12  # per ROI in the fixture


def _expected(model, drawn: int) -> tuple[np.ndarray, np.ndarray]:
    """ROI means of the final model's own training scores (rows: ROIs in order, drawn pixels,
    excluded rows left out), written out independently of the library."""
    diagnostics = model.diagnostics
    scores = np.asarray(diagnostics["x_scores_"], dtype=np.float64)
    excluded = model.excluded_pixels
    roi_ids = [str(r) for r in np.asarray(diagnostics["roi_ids"])]
    labels = []
    for index, roi_id in enumerate(roi_ids):
        kept = drawn - len(excluded.get(roi_id, ()))
        labels += [index] * kept
    labels = np.asarray(labels)
    assert labels.size == scores.shape[0]
    counts = np.bincount(labels, minlength=len(roi_ids))
    means = np.stack([np.bincount(labels, weights=scores[:, c], minlength=len(roi_ids))
                      for c in range(scores.shape[1])], axis=1) / counts[:, None]
    return means, counts


def test_roi_scores_with_exclusion_are_the_means_of_the_kept_pixels(tmp_path):
    project = _project_with_an_odd_pixel(tmp_path)
    model = project.fit("API", n_components=2, fit_spec=EXCLUSION)
    assert model.metrics["n_excluded_pixels"] > 0
    means, counts = _expected(model, PIXELS)
    np.testing.assert_allclose(model.diagnostics["roi_scores"], means, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(model.diagnostics["roi_score_pixels"], counts)
    payload = _roi_score_payload(model.diagnostics)
    assert payload["scores_roi"].shape == (8, 2) and payload["scores_without_pixels"] == 0


def test_roi_scores_with_a_pixel_sample_and_exclusion(tmp_path):
    project = _project_with_an_odd_pixel(tmp_path)
    for spec, drawn in (({"pixels": {"per_roi": 5, "seed": 1}}, 5),
                        ({"pixels": {"per_roi": 8, "seed": 2}, **EXCLUSION}, 8)):
        model = project.fit("API", n_components=2, fit_spec=spec)
        means, counts = _expected(model, drawn)
        np.testing.assert_allclose(model.diagnostics["roi_scores"], means, rtol=1e-4,
                                   atol=1e-5)
        np.testing.assert_array_equal(model.diagnostics["roi_score_pixels"], counts)


def test_the_viewer_shows_missing_roi_scores_as_not_available():
    payload = _roi_score_payload({"x_scores_": np.ones((10, 2))})
    assert "scores_roi" not in payload and "not available" in payload["scores_message"]
    nan_row = np.array([[1.0, 2.0], [np.nan, np.nan]], dtype=np.float32)
    assert _roi_score_payload({"roi_scores": nan_row})["scores_without_pixels"] == 1
