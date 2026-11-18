"""Projects of an earlier format: the committed enginir-1.5 project (tests/data/enginir_1_5)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from enginir import Project

OLD = Path(__file__).parent / "data" / "enginir_1_5"


def copy_old_project(tmp_path) -> Path:
    """A copy of the enginir-1.5 project with its data files; returns the project path."""
    folder = tmp_path / "old"
    shutil.copytree(OLD, folder, ignore=shutil.ignore_patterns("*.py", "__pycache__"))
    return folder / "project.h5"


def test_an_old_project_is_refused_with_the_command_that_converts_it(tmp_path):
    path = copy_old_project(tmp_path)
    with pytest.raises(ValueError, match="format 'enginir-1.5'.*enginir-migrate") as error:
        Project.open(path)
    assert str(path) in str(error.value)
