"""Per-user settings: Tools > Preferences and recent files, never stored in a project."""

from __future__ import annotations

from importlib import reload
import json
import os
from pathlib import Path
import sys
import pytest
from enginir.viewer import _preferences as prefs


# -- Tools > Preferences (plan gui-1.6, Phase 6): stored per user, never in a project ----


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
    values = prefs.Preferences(theme="system", interface_scale=0.9,
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
                                 dmodx_level=0.95, plot_help=False))
    cube = np.full((6, 4, 8), 2000, dtype=np.uint16)
    save_hdf5(str(tmp_path / "scan.h5"), cube, np.linspace(1000, 1600, 8))
    project = Project.create(tmp_path / "p.h5")
    project.add_dataset(tmp_path / "scan.h5")
    window = open_project(project.path)
    try:
        # The Fit settings and the Predict page get the defaults when first shown (D8).
        window.show_step("model")
        form = window._modeling_view.fit_form
        assert form is not None and form._components_spin.value() == 3
        assert form._group_by_combo.currentData() == "sample"
        assert window._prediction_view._exclusion_check.isChecked()
        assert window._prediction_view._exclusion_level.value() == pytest.approx(0.95)
        helps = [frame._help_button for frame in window.findChildren(type(window._canvas_frame))
                 if getattr(frame, "_help_button", None) is not None]
        assert all(button.isHidden() for button in helps)

        accepted = prefs.Preferences(plot_help=True)
        monkeypatch.setattr(prefs.PreferencesDialog, "exec", lambda self: 1)
        monkeypatch.setattr(prefs.PreferencesDialog, "values", lambda self: accepted)
        assert window.edit_preferences()
        assert prefs.load() == accepted
        assert not any(button.isHidden() for button in helps)
    finally:
        window.close()


def test_file_dialogs_open_in_the_preferred_folders(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    from enginir.viewer._data_files import _ReferenceSource
    from enginir.viewer._welcome import WelcomeWindow

    projects, datasets = tmp_path / "projects", tmp_path / "datasets"
    projects.mkdir()
    datasets.mkdir()
    opened: list[tuple[str, str]] = []

    def record(_parent, title, directory="", _filter=""):
        opened.append((title, directory))
        return "", ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(record))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(record))

    def ask_all() -> dict[str, str]:
        opened.clear()
        welcome = WelcomeWindow()
        try:
            welcome._on_open_project_clicked()
            welcome._on_new_project_clicked()
            welcome._on_open_clicked()
            welcome._on_browse_dark()
        finally:
            welcome.close()
        _ReferenceSource("White", 10, (0, 1), lambda *_: None)._browse()
        return dict(opened)

    prefs.save(prefs.Preferences(project_folder=str(projects), dataset_folder=str(datasets)))
    assert ask_all() == {"Open project": str(projects), "Create Project": str(projects),
                         "Open dataset": str(datasets), "Dark reference": str(datasets),
                         "White reference": str(datasets)}
    prefs.save(prefs.Preferences(project_folder=str(tmp_path / "gone"),
                                 dataset_folder=str(datasets)))
    assert ask_all()["Open project"] == "", "a folder that is not there: Qt's default"


def test_the_drive_mapping_shown_is_the_one_applied(qapp, monkeypatch):
    from pathlib import PureWindowsPath

    from enginir.project._data_paths import drive_mapped

    monkeypatch.setattr("enginir.project._data_paths.wsl_mount_root", lambda: "/wsl/")
    dialog = prefs.PreferencesDialog(prefs.Preferences())
    assert dialog.drive_mapping.text() == prefs.drive_mapping_text(), "this platform's rule"
    shown = prefs.drive_mapping_text(windows=False)
    applied = drive_mapped(PureWindowsPath("Z:\\"), windows=False, wsl_root="/wsl/")
    assert applied is not None and f"→  {applied}" in shown, (shown, applied)
    assert "Z:\\" in prefs.drive_mapping_text(windows=True), "Windows maps to the drive"


# -- Focused tests for recent-file persistence -----------------------------------------


@pytest.fixture()
def _no_xdg_config_home(monkeypatch, tmp_path):
    """HOME in ``tmp_path`` and no XDG_CONFIG_HOME, for the recent-files tests.

    The tests redirect HOME; an XDG_CONFIG_HOME set by the environment (as on
    CI runners) would take precedence and point at the real config directory.
    HOME is redirected here too, so nothing reaches the real one.
    """
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.mark.usefixtures("_no_xdg_config_home")
@pytest.mark.parametrize(
    ("platform_name", "env_name", "env_value", "expected"),
    [
        ("linux", "HOME", "home", Path("home/.config/enginir")),
        ("win32", "APPDATA", "appdata", Path("appdata/enginir")),
        ("darwin", "HOME", "home", Path("home/Library/Application Support/enginir")),
    ],
)
def test_config_dir_uses_platform_specific_location(
    monkeypatch,
    tmp_path,
    platform_name,
    env_name,
    env_value,
    expected,
):
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setenv(env_name, str(tmp_path / env_value))
    import enginir.viewer._recent as recent

    reload(recent)

    assert recent.config_dir() == tmp_path / expected


@pytest.mark.usefixtures("_no_xdg_config_home")
def test_load_returns_empty_list_for_malformed_json(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    import enginir.viewer._recent as recent

    reload(recent)
    cfg = recent.config_dir()
    cfg.mkdir(parents=True)
    (cfg / "recent.json").write_text("not json{{", encoding="utf-8")

    assert recent.load() == []


@pytest.mark.usefixtures("_no_xdg_config_home")
def test_add_keeps_most_recent_order_without_duplicates_and_caps_length(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    import enginir.viewer._recent as recent

    reload(recent)
    recent.add("/a.eiff")
    recent.add("/b.eiff")
    recent.add("/a.eiff")
    for index in range(15):
        recent.add(f"/file_{index}.eiff")

    loaded = recent.load()

    assert len(loaded) == 10
    assert loaded[0].endswith("/file_14.eiff")
    assert len(set(loaded)) == len(loaded)
    assert not any(path.endswith("/b.eiff") for path in loaded)
