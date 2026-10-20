"""Contract tests for the headless (Qt-free) workflow API surface."""

from __future__ import annotations

import subprocess
import sys

_HEADLESS_IMPORT_PROBE = """
import sys

import enginir
import enginir.io
import enginir.project
import enginir.modeling
import enginir.prediction
import enginir.preprocessing
import enginir.models
import enginir.validation

qt_modules = sorted(name for name in sys.modules if name.startswith("PyQt"))
if qt_modules:
    print("Qt modules imported:", qt_modules)
    raise SystemExit(1)
"""


def test_headless_packages_do_not_import_qt():
    result = subprocess.run(
        [sys.executable, "-c", _HEADLESS_IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


def test_viewer_writes_projects_only_through_project():
    """ProjectState delegates to enginir.Project; only the prediction worker writes the
    job's new dataset and calibration itself, through the validated writers."""
    import ast
    from pathlib import Path

    import enginir.io.project_store as store

    writers = {name for name in dir(store) if name.startswith(("save_", "remove_"))}
    viewer = Path(__file__).parents[1] / "src" / "enginir" / "viewer"
    found: dict[str, set[str]] = {}
    for path in viewer.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and "project_store" in node.module:
                used = {alias.name for alias in node.names} & writers
                if used:
                    found.setdefault(path.name, set()).update(used)

    assert found == {"_prediction.py": {"save_dataset_ref", "save_calibration_spec"}}
