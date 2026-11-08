"""Prediction-time wavelength warning (G18: warn only, never a gate)."""

from __future__ import annotations

import numpy as np
import pytest

from enginir.prediction import wavelength_mismatch
from tests.test_fit_spec import _warnings_during
from tests.test_workflow import N_BANDS, _labelled_project, _scan

AXIS = np.linspace(1000, 1600, N_BANDS)


@pytest.mark.parametrize(("scan", "model", "tolerance", "warns"), [
    (AXIS + 2.0, AXIS, 0.0, True),
    (AXIS + 2.0, AXIS, 2.5, False),             # within the signal domain's tolerance
    (AXIS + 1e-7, AXIS, 0.0, False),            # float noise
    (AXIS, AXIS, 0.0, False),
    (np.arange(N_BANDS, dtype=float), AXIS, 0.0, False),   # band-index fallback: no axis
    (AXIS, np.arange(N_BANDS, dtype=float), 0.0, False),
    (np.array([]), AXIS, 0.0, False),
    (AXIS[:-1], AXIS, 0.0, False),              # band counts are checked elsewhere
])
def test_axes_are_compared_only_when_both_are_real(scan, model, tolerance, warns):
    assert (wavelength_mismatch(scan, model, tolerance) is not None) is warns


def test_a_shifted_scan_warns_and_predicts_exactly_as_before(tmp_path):
    project, scan = _labelled_project(tmp_path)
    model = project.fit("API", n_components=2)
    same = model.predict(scan.path)
    shifted_path = _scan(tmp_path / "shifted.h5", wavelengths=AXIS + 5.0)
    shifted, messages = _warnings_during(lambda: model.predict(shifted_path))
    assert same.provenance["wavelength_warning"] is None
    warning = shifted.provenance["wavelength_warning"]
    assert warning is not None and "up to 5 nm" in warning
    assert sum("predicted anyway" in m for m in messages) == 1, "logged once"
    np.testing.assert_array_equal(shifted.map, same.map)


def test_the_viewer_summary_shows_the_warning():
    from enginir.viewer._app import ensure_app
    from enginir.viewer._prediction import PredictionView
    from enginir.viewer._state import ProjectState
    from tests.test_prediction_view import _seed_model_run, _session_result

    ensure_app()
    state = ProjectState()
    _seed_model_run(state)
    view = PredictionView(state)
    result = _session_result("p", np.ones((2, 3), dtype=np.float32))
    result.provenance["wavelength_warning"] = "wavelengths differ from the model's by up to 5 nm"
    state.add_prediction(result)
    assert view.workspace_summary()["wavelengths"].startswith("Warning: wavelengths differ")
    result.provenance["wavelength_warning"] = None
    view._invalidate_prediction_workspace_cache()
    assert view.workspace_summary()["wavelengths"] == "match the model"
