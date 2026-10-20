"""Tests for enginir.viewer._app."""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_ensure_app_returns_singleton_and_applies_theme(qapp):
    from enginir.viewer._app import ensure_app

    app1 = ensure_app()
    app2 = ensure_app()

    assert app1 is app2
    assert isinstance(app1, QApplication)
    assert app1.styleSheet() != ""
    assert bool(app1.property("_enginir_theme_applied")) is True


def test_ensure_app_uses_saved_theme_preference(qapp, monkeypatch):
    import enginir.viewer._app as appmod

    called: list[bool] = []
    monkeypatch.setattr(appmod, "load_preference", lambda: False)
    monkeypatch.setattr(appmod, "apply_qt_theme", lambda app, *, dark: called.append(dark) or app)

    app = appmod.ensure_app()

    assert app is qapp
    assert called == [False]


def test_apply_qt_theme_persists_panel_colors_and_preference(qapp, tmp_path, monkeypatch):
    import enginir.viewer._theme as thememod

    monkeypatch.setattr(thememod, "config_dir", lambda: tmp_path)
    thememod.apply_qt_theme(qapp, dark=False)

    assert thememod.is_dark() is False
    assert thememod.panel_colors()["fg"] == "#2e3742"
    assert thememod.load_preference() is False


@pytest.mark.parametrize(
    ("explicit_platform", "wayland", "display", "expected"),
    [
        (None, "wayland-0", ":0", "xcb"),
        ("offscreen", "wayland-0", ":0", None),
        (None, "wayland-0", None, None),
    ],
)
def test_recommended_qt_platform_linux_wayland_cases(
    monkeypatch, explicit_platform, wayland, display, expected
):
    import enginir.viewer._app as appmod

    if explicit_platform is None:
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    else:
        monkeypatch.setenv("QT_QPA_PLATFORM", explicit_platform)
    if wayland is None:
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    else:
        monkeypatch.setenv("WAYLAND_DISPLAY", wayland)
    if display is None:
        monkeypatch.delenv("DISPLAY", raising=False)
    else:
        monkeypatch.setenv("DISPLAY", display)
    monkeypatch.setattr(appmod.sys, "platform", "linux")

    assert appmod._recommended_qt_platform() == expected


def test_run_is_noop_in_jupyter(qapp, monkeypatch):
    import builtins
    from enginir.viewer._app import run

    class FakeIP:
        def enable_gui(self, backend):
            pass

    monkeypatch.setattr(builtins, "get_ipython", lambda: FakeIP(), raising=False)
    run()


def test_launcher_returns_expected_exit_codes(qapp, monkeypatch):
    import enginir.viewer.__main__ as launcher

    monkeypatch.setattr(
        "enginir.viewer.open_viewer",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )
    monkeypatch.setattr("enginir.viewer.__main__.run", lambda: None, raising=False)
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.critical", lambda *args, **kwargs: None)
    assert launcher._main(["/tmp/missing.eiff"]) == 1

    monkeypatch.setattr("enginir.viewer.open_project", lambda path: object())
    monkeypatch.setattr("enginir.viewer._app.run", lambda: None)
    assert launcher._main(["--project", "/tmp/project.h5"]) == 0
