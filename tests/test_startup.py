"""A fast start: what the launcher loads before the welcome window (plan startup-speed-1.7.1)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

#: Not loaded before the welcome window is shown (D2): the scientific stack and the parts of
#: engiNIR that need it. Each would cost seconds on a cold Windows start.
HEAVY = ("sklearn", "scipy", "pandas", "h5py", "matplotlib", "pyqtgraph",
         "enginir._workflow", "enginir.io", "enginir.project", "enginir.models",
         "enginir.modeling", "enginir.prediction", "enginir.validation", "enginir.viewer._viewer")

_LAUNCHER = textwrap.dedent("""
    import json, sys
    import enginir.viewer.__main__
    from enginir.viewer._app import enable_viewer_logging, ensure_app
    enable_viewer_logging()
    ensure_app()
    from enginir.viewer._welcome import WelcomeWindow
    from PyQt6.QtWidgets import QApplication
    window = WelcomeWindow()
    window.show()
    QApplication.processEvents()
    print(json.dumps(sorted(sys.modules)))
""")


def _modules_at_welcome(tmp_path) -> set[str]:
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "HOME": str(home),
           "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
           "APPDATA": str(home / "AppData"), "ENGINIR_LOG_FILE": str(tmp_path / "viewer.log")}
    result = subprocess.run([sys.executable, "-c", _LAUNCHER], env=env, capture_output=True,
                            text=True, timeout=60, check=True)
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_the_welcome_window_shows_without_the_scientific_stack(tmp_path):
    modules = _modules_at_welcome(tmp_path)
    assert "PyQt6.QtWidgets" in modules, "the launcher did reach the welcome window"
    loaded = [name for name in HEAVY if name in modules]
    assert loaded == [], f"loaded before the welcome window: {loaded}"


@pytest.mark.parametrize("package", ["enginir", "enginir.viewer"])
def test_every_public_name_still_resolves(package):
    """The names load on first use; each one in __all__ still exists."""
    import importlib

    module = importlib.import_module(package)
    for name in module.__all__:
        assert getattr(module, name) is not None, name
    with pytest.raises(AttributeError):
        module.__getattr__("no_such_name")
