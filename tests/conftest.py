"""Shared pytest configuration and fixtures."""

import os
import sys
import tempfile

import pytest
from loguru import logger

# Use offscreen Qt platform so tests pass in headless CI environments.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Viewer tests enable file logging; keep it out of the user's home directory.
os.environ.setdefault(
    "ENGINIR_LOG_FILE", os.path.join(tempfile.mkdtemp(prefix="enginir-tests-"), "enginir.log")
)
# The viewer stores recent files and the theme in the user's config directory;
# point it at a temporary one for the whole session (module-scoped Qt fixtures
# start before function fixtures), and per test in _isolated_user_config.
_SESSION_CONFIG = tempfile.mkdtemp(prefix="enginir-tests-config-")
os.environ["XDG_CONFIG_HOME"] = _SESSION_CONFIG
os.environ["APPDATA"] = _SESSION_CONFIG


@pytest.fixture(autouse=True)
def _isolated_user_config(tmp_path, monkeypatch):
    """Keep the viewer's recent files and theme out of the user's real config."""
    config = tmp_path / "user_config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))  # Linux
    monkeypatch.setenv("APPDATA", str(config))          # Windows
    if sys.platform == "darwin":
        monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture(autouse=True)
def _enginir_records_enabled():
    """Let tests observe enginir log records (the library is muted by default)."""
    logger.enable("enginir")
    yield
