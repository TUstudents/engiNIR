"""The example scripts included in the documentation run as shown."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).parents[1] / "docs" / "examples"


def test_project_workflow_example_predicts_the_regions(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(EXAMPLES / "project_workflow.py"))

    tablet_means = namespace["prediction"].map.reshape(8, 5, 10).mean(axis=(1, 2))
    np.testing.assert_allclose(tablet_means, np.linspace(1, 9, 8), atol=1.0)
    assert namespace["model"].metrics["r2"] > 0.9
    assert "PLS 2 LV, SNV" in capsys.readouterr().out
