"""Every registered plot meets the Plot Panel Policy.

For each plot of the viewer: the empty state is visible and cannot be
exported; with data, PNG, SVG and CSV are written; the title is only in the
header; every number in the CSV is a value the plot draws.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from enginir.viewer._app import ensure_app


@pytest.fixture(scope="module")
def qapp():
    return ensure_app()


def _model_payload() -> dict:
    rng = np.random.default_rng(7)
    n_rois, n_bands = 12, 40
    y_true = rng.normal(10.0, 2.0, n_rois).astype(np.float32)
    return {
        "target_unit": "%",
        "y_true_roi": y_true,
        "y_pred_roi": (y_true + rng.normal(0.0, 0.4, n_rois)).astype(np.float32),
        "roi_ids": np.array([f"roi-{i}" for i in range(n_rois)], dtype=object),
        "roi_names": tuple(f"Tablet {i}" for i in range(n_rois)),
        "scores_roi": rng.normal(size=(n_rois, 3)),
        "spectral_axis": np.linspace(1000.0, 1700.0, n_bands, dtype=np.float32),
        "coefficients": rng.normal(size=n_bands).astype(np.float32),
        "loadings": rng.normal(size=n_bands).astype(np.float32),
    }


#: A payload with data for every plot, per workspace.
_PAYLOADS = {"Model": _model_payload}


def _registry():
    import enginir.viewer._model_diagnostics  # noqa: F401 (registers the Model plots)
    from enginir.viewer._plot_registry import PLOT_SPECS, WORKSPACE_PAGES

    return PLOT_SPECS, WORKSPACE_PAGES


def _registered_plots() -> list[tuple[str, str]]:
    specs, pages = _registry()
    return [(workspace, plot_id) for workspace, workspace_pages in pages.items()
            for page in workspace_pages for plot_id in page.plot_ids]


def test_every_registered_plot_is_on_a_page():
    specs, _pages = _registry()
    assert specs, "the viewer registers its plots"
    assert sorted(plot_id for _w, plot_id in _registered_plots()) == sorted(specs)


def _drawn_values(fig) -> np.ndarray:
    values: list[np.ndarray] = []
    for ax in fig.axes:
        for collection in ax.collections:
            values.append(np.ravel(collection.get_offsets()))
            if collection.get_array() is not None:
                values.append(np.ravel(collection.get_array()))
        for line in ax.lines:
            values += [np.ravel(line.get_xdata()), np.ravel(line.get_ydata())]
        for patch in ax.patches:
            if hasattr(patch, "get_data"):  # StepPatch (histograms)
                data = patch.get_data()
                values += [np.ravel(data.values), np.ravel(data.edges)]
    return np.concatenate(values).astype(float) if values else np.array([])


def _csv_numbers(path: Path) -> np.ndarray:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if not line.startswith("# ")]
    header, *rows = list(csv.reader(lines))
    numbers = []
    for index, name in enumerate(header):
        if name in ("roi_name", "roi_id"):
            continue
        numbers += [float(row[index]) for row in rows]
    return np.array(numbers)


@pytest.mark.parametrize("workspace, plot_id", _registered_plots())
def test_registered_plot_meets_the_plot_panel_policy(qapp, tmp_path, workspace, plot_id):
    from enginir.viewer._plot_panel import PlotPanel

    specs, _pages = _registry()
    panel = PlotPanel(specs[plot_id])
    panel.resize(500, 350)
    panel.show()

    panel.update_data(None, empty_reason="Nothing fitted yet.")
    assert "Nothing fitted yet." in [t.get_text() for ax in panel.figure.axes for t in ax.texts]
    assert panel.export_reason()

    panel.update_data(_PAYLOADS[workspace](), provenance={"project_id": "p"})
    assert not panel.export_reason(), "a full payload is exportable"
    assert all(not ax.get_title() for ax in panel.figure.axes), "title only in the header"
    for name in ("plot.png", "plot.svg", "plot.csv"):
        (panel.export_csv if name.endswith(".csv") else panel.export_image)(tmp_path / name)
        assert (tmp_path / name).stat().st_size > 0

    drawn = _drawn_values(panel.figure)
    exported = _csv_numbers(tmp_path / "plot.csv")
    assert exported.size
    missing = [value for value in exported
               if not np.isclose(drawn, value, rtol=1e-6, atol=1e-9).any()]
    assert not missing, f"CSV values that the plot does not draw: {missing[:5]}"
