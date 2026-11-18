"""Projects of an earlier format: the committed enginir-1.5 project (tests/data/enginir_1_5)."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from enginir import Project
from enginir.io import FILE_SCHEMA_VERSION, load_project_store, migrate_project
from enginir.io import _migrate

OLD = Path(__file__).parent / "data" / "enginir_1_5"


def copy_old_project(tmp_path) -> Path:
    """A copy of the enginir-1.5 project with its data files; returns the project path."""
    folder = tmp_path / "old"
    shutil.copytree(OLD, folder, ignore=shutil.ignore_patterns("*.py", "__pycache__"))
    return folder / "project.h5"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content(path: Path, *, renamed: bool) -> dict[str, object]:
    """Every group path and attribute of a project, with the old names mapped to the new ones.

    Only the format metadata (FileSchemaVersion, FileWriter, updated_at) is left out.
    """
    groups = dict(_migrate.RENAMED_GROUPS) if renamed else {}
    attributes = dict(_migrate.RENAMED_ATTRIBUTES) if renamed else {}
    values = dict(_migrate.RENAMED_VALUES) if renamed else {}
    content: dict[str, object] = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            new_name = "/".join(groups.get(part, part) for part in name.split("/"))
            if not isinstance(obj, h5py.Dataset):
                content[new_name] = "group"
            elif obj.dtype.kind in "OSU":  # strings: compare the values, not object pointers
                content[new_name] = repr(obj.asstr()[()].tolist())
            else:
                content[new_name] = (obj.dtype.str, obj.shape, obj[()].tobytes())
            for key, value in obj.attrs.items():
                if name == "project_meta" and key in ("FileSchemaVersion", "FileWriter", "updated_at"):
                    continue
                value = value.decode() if isinstance(value, bytes) else value
                if isinstance(value, str):
                    value = values.get(value, value)
                elif isinstance(value, np.ndarray):
                    value = (value.dtype.str, value.tobytes())
                else:  # numbers by repr, so NaN equals NaN
                    value = repr(value)
                content[f"{new_name}@{attributes.get(key, key)}"] = value
        f.visititems(visit)
    return content


def test_an_old_project_is_refused_with_the_command_that_converts_it(tmp_path):
    path = copy_old_project(tmp_path)
    with pytest.raises(ValueError, match="format 'enginir-1.5'.*enginir-migrate") as error:
        Project.open(path)
    message = str(error.value)
    assert str(path) in message
    # the suggested target is a new file next to the source
    assert str(path.with_name("project (enginir-1.7).h5")) in message


def test_the_converted_project_keeps_everything_and_predicts_the_same_map(tmp_path):
    source = copy_old_project(tmp_path)
    before = _digest(source)
    target = migrate_project(source, source.with_name("converted.h5"))

    assert _digest(source) == before, "the source is never changed"
    assert _content(target, renamed=False) == _content(source, renamed=True)
    with h5py.File(target, "r") as f:
        meta = f["project_meta"].attrs
        assert meta["FileSchemaVersion"] == FILE_SCHEMA_VERSION
        assert meta["FileCreator"] == "engiNIR 1.6.1"
        names = []
        f.visit(names.append)
        stored = " ".join(names + [k for n in names for k in f[n].attrs]
                          + [str(v) for n in names for v in f[n].attrs.values()])
    assert not re.search(r"calibrations|calibration_id|calibration_refs|precalibrated|"
                         r"required_calibration|calibrated_input", stored)

    project = Project.open(target)
    (model,) = project.models.values()
    prediction = model.predict(source.parent / "scan.h5")
    # The stored model is byte-identical (checked above); the expected map was computed on
    # another machine, and other BLAS builds (macOS, Windows) differ in the last float32 digits.
    np.testing.assert_allclose(prediction.map, np.load(source.parent / "expected_map.npy"),
                               rtol=0, atol=1e-3)
    store = load_project_store(target)
    modes = {c.get("mode") for d in store["datasets"].values()
             for c in d["reference_corrections"].values()}
    assert modes == {"dark_white_reference", "already_corrected"}
    assert len(store["predictions"]) == 1


def test_an_existing_target_or_the_source_itself_is_refused(tmp_path):
    source = copy_old_project(tmp_path)
    taken = source.with_name("taken.h5")
    taken.write_bytes(b"keep")
    for target in (taken, source):
        with pytest.raises(FileExistsError, match="exists"):
            migrate_project(source, target)
    assert taken.read_bytes() == b"keep"


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
