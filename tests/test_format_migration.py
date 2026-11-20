"""Projects of an earlier format: the committed enginir-1.5 project (tests/data/enginir_1_5)."""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import pytest

from enginir import Project
from enginir.io import load_project_store, migrate_project
from enginir.io import _migrate

OLD = Path(__file__).parent / "data" / "enginir_1_5"


def copy_old_project(tmp_path) -> Path:
    """A copy of the enginir-1.5 project with its data files; returns the project path."""
    folder = tmp_path / "old"
    shutil.copytree(OLD, folder, ignore=shutil.ignore_patterns("*.py", "__pycache__"))
    return folder / "project.h5"


def test_only_an_enginir_1_5_project_is_converted(tmp_path):
    current = Project.create(tmp_path / "current.h5").path
    with pytest.raises(ValueError, match="format 'enginir-1.7'; only 'enginir-1.5'"):
        migrate_project(current, tmp_path / "new.h5")
    older = tmp_path / "older.h5"
    with h5py.File(older, "w") as f:
        f.create_group("project_meta").attrs["schema_version"] = "viewer-v3"
    with pytest.raises(ValueError, match="format 'viewer-v3'"):
        migrate_project(older, tmp_path / "new.h5")
    with pytest.raises(FileNotFoundError):
        migrate_project(tmp_path / "missing.h5", tmp_path / "new.h5")
    assert not (tmp_path / "new.h5").exists()


def test_a_missed_rename_is_caught_by_the_reader_and_leaves_nothing(tmp_path, monkeypatch):
    """Fault injection: forgetting one stored name must fail the conversion, not pass it."""
    source = copy_old_project(tmp_path)
    target = source.with_name("converted.h5")
    attributes = dict(_migrate.RENAMED_ATTRIBUTES)
    del attributes["calibration_id"]
    monkeypatch.setattr(_migrate, "RENAMED_ATTRIBUTES", attributes)
    with pytest.raises(ValueError):
        migrate_project(source, target)
    assert not target.exists()
    assert not target.with_name(target.name + ".partial").exists()


def test_the_command_converts_and_reports_errors(tmp_path, capsys):
    source = copy_old_project(tmp_path)
    target = source.with_name("converted.h5")
    assert _migrate.main([str(source), str(target)]) == 0
    assert "Converted" in capsys.readouterr().out and target.exists()
    assert _migrate.main([str(source), str(target)]) == 1
    assert "exists" in capsys.readouterr().err


def test_the_viewer_opens_an_old_project_as_a_project_and_shows_the_command(tmp_path):
    """The viewer recognizes the file as a project; reading it gives the conversion message."""
    from enginir.viewer import _is_project_file

    path = copy_old_project(tmp_path)
    assert _is_project_file(path)
    with pytest.raises(ValueError, match="enginir-migrate"):
        load_project_store(path)
