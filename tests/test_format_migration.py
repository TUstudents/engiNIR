"""Projects of an earlier format: the committed enginir-1.5 project (tests/data/enginir_1_5)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from enginir import Project

OLD = Path(__file__).parent / "data" / "enginir_1_5"


def copy_old_project(tmp_path) -> Path:
    """A copy of the enginir-1.5 project with its data files; returns the project path."""
    folder = tmp_path / "old"
    shutil.copytree(OLD, folder, ignore=shutil.ignore_patterns("*.py", "__pycache__"))
    return folder / "project.h5"


def test_the_old_project_predicts_its_stored_map(tmp_path):
    path = copy_old_project(tmp_path)
    project = Project.open(path)
    (model,) = project.models.values()
    prediction = model.predict(path.parent / "scan.h5")
    np.testing.assert_array_equal(prediction.map, np.load(path.parent / "expected_map.npy"))
