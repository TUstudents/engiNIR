"""Tools > Preferences (plan gui-1.6, Phase 6): stored per user, never in a project."""

from __future__ import annotations

import json
import os

import pytest

from enginir.viewer import _preferences as prefs
from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


def test_preferences_round_trip_and_tolerate_a_damaged_file():
    assert prefs.load() == prefs.Preferences()
    changed = prefs.Preferences(theme="light", n_components=3, hold_out="sample",
                                export_dpi=150, blas_threads=2)
    prefs.save(changed)
    assert prefs.load() == changed
    path = prefs._path()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["n_components"] = "many"   # unreadable value: its default, the rest kept
    data["unknown"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = prefs.load()
    assert loaded.n_components == prefs.Preferences().n_components and loaded.theme == "light"
    path.write_text("{not json", encoding="utf-8")
    assert prefs.load() == prefs.Preferences()


def test_the_dialog_shows_and_returns_every_preference(qapp):
    values = prefs.Preferences(theme="system", interface_scale=0.9, guide_bar=False,
                               raise_checks=False, plot_help=False, project_folder="/p",
                               dataset_folder="/d", model_type="PCR", n_components=4,
                               auto_components=True, hold_out="target", x_scaling="uv",
                               y_scaling="pareto", exclude_dmodx=True, dmodx_level=0.95,
                               export_dpi=600, blas_threads=3, log_level="DEBUG")
    dialog = prefs.PreferencesDialog(values, shortcuts=[("Fit Model", "F5")])
    assert dialog.values() == values
    dialog.set_values(prefs.Preferences())
    assert dialog.values() == prefs.Preferences()
    assert [dialog.pages.item(i).text() for i in range(dialog.pages.count())] \
        == list(prefs.PreferencesDialog.PAGES)


def test_threads_and_scale_act_through_the_environment(monkeypatch):
    from enginir.models._gram import BLAS_THREADS_ENV, blas_thread_count

    monkeypatch.delenv(BLAS_THREADS_ENV, raising=False)
    monkeypatch.delenv("_ENGINIR_BLAS_FROM_PREFERENCES", raising=False)
    prefs.apply_environment(prefs.Preferences(blas_threads=3))
    assert blas_thread_count() == 3
    prefs.apply_environment(prefs.Preferences(blas_threads=0))
    assert BLAS_THREADS_ENV not in os.environ, "automatic again"

    monkeypatch.delenv("QT_SCALE_FACTOR", raising=False)
    prefs.save(prefs.Preferences(interface_scale=0.9))
    prefs.startup_scale()
    assert os.environ["QT_SCALE_FACTOR"] == "0.9"
    monkeypatch.setenv("QT_SCALE_FACTOR", "2")
    prefs.startup_scale()
    assert os.environ["QT_SCALE_FACTOR"] == "2", "the user's own setting wins"


def test_the_window_applies_preferences(qapp, tmp_path, monkeypatch):
    import numpy as np

    from enginir import Project
    from enginir.io import save_hdf5
    from enginir.viewer import open_project

    prefs.save(prefs.Preferences(n_components=3, hold_out="sample", exclude_dmodx=True,
                                 dmodx_level=0.95, guide_bar=False, plot_help=False))
    cube = np.full((6, 4, 8), 2000, dtype=np.uint16)
    save_hdf5(str(tmp_path / "scan.h5"), cube, np.linspace(1000, 1600, 8))
    project = Project.create(tmp_path / "p.h5")
    project.add_dataset(tmp_path / "scan.h5")
    window = open_project(project.path)
    try:
        form = window._modeling_view.fit_form
        assert form is not None and form._components_spin.value() == 3
        assert form._group_by_combo.currentData() == "sample"
        assert window._prediction_view._exclusion_check.isChecked()
        assert window._prediction_view._exclusion_level.value() == pytest.approx(0.95)
        window.refresh_checks()
        assert window._guide_bar.isHidden()
        helps = [frame._help_button for frame in window.findChildren(type(window._canvas_frame))
                 if getattr(frame, "_help_button", None) is not None]
        assert all(button.isHidden() for button in helps)

        accepted = prefs.Preferences(guide_bar=True)
        monkeypatch.setattr(prefs.PreferencesDialog, "exec", lambda self: 1)
        monkeypatch.setattr(prefs.PreferencesDialog, "values", lambda self: accepted)
        assert window.edit_preferences()
        assert prefs.load() == accepted
        window.refresh_checks()
        assert not window._guide_bar.isHidden()
    finally:
        window.close()
