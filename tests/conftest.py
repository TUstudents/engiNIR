"""Shared pytest configuration and fixtures."""

import os
import tempfile

import pytest
from loguru import logger

# Use offscreen Qt platform so tests pass in headless CI environments.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Viewer tests enable file logging; keep it out of the user's home directory.
os.environ.setdefault(
    "ENGINIR_LOG_FILE", os.path.join(tempfile.mkdtemp(prefix="enginir-tests-"), "enginir.log")
)


@pytest.fixture(autouse=True)
def _enginir_records_enabled():
    """Let tests observe enginir log records (the library is muted by default)."""
    logger.enable("enginir")
    yield
