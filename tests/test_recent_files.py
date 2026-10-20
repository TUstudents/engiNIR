"""Focused tests for recent-file persistence."""

from __future__ import annotations

import sys
from importlib import reload
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_xdg_config_home(monkeypatch):
    # The tests redirect HOME; an XDG_CONFIG_HOME set by the environment (as on
    # CI runners) would take precedence and point at the real config directory.
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)


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


def test_load_returns_empty_list_for_malformed_json(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    import enginir.viewer._recent as recent

    reload(recent)
    cfg = recent.config_dir()
    cfg.mkdir(parents=True)
    (cfg / "recent.json").write_text("not json{{", encoding="utf-8")

    assert recent.load() == []


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
