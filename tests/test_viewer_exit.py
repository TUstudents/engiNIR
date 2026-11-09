"""The viewer process exits cleanly after a project window was open.

PyQt deletes the remaining top-level Qt objects at interpreter exit while
walking sip's wrapper table; Python code running during those deletions (such
as an application-wide event filter) can reallocate the table mid-walk and
crash the process with a segmentation fault. The crash is intermittent, so
the session runs several times.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

_SESSION = textwrap.dedent("""
    import sys
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication
    import enginir
    from enginir.viewer import open_project
    from enginir.viewer._app import ensure_app, run

    app = ensure_app()
    enginir.Project.create(sys.argv[1])
    open_project(sys.argv[1])
    QTimer.singleShot(300, lambda: [w.close() for w in QApplication.topLevelWidgets()])
    run()
""")


def test_closing_a_project_viewer_exits_without_a_crash(tmp_path):
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen",
           "ENGINIR_LOG_FILE": str(tmp_path / "viewer.log")}
    for run in range(4):
        result = subprocess.run(
            [sys.executable, "-c", _SESSION, str(tmp_path / f"p{run}.h5")],
            env=env, capture_output=True, text=True, timeout=50)
        assert result.returncode == 0, (
            f"run {run} ended with {result.returncode}:\n{result.stderr[-2000:]}")
