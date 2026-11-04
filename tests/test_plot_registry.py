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


def _predict_payload() -> dict:
    values = np.random.default_rng(3).normal(12.0, 1.5, (40, 25)).astype(np.float32)
    values[0, 0] = np.nan  # pixels without a prediction are left out
    return {"values": values, "target_unit": "%", "output": "moisture", "map_step": 2,
            "provenance": {"output": "moisture"}}


#: A payload with data for every plot, by the workspace prefix of its id.
def _explore_payload() -> dict:
    from enginir.modeling import RoiSpectrumSummary

    rng = np.random.default_rng(5)
    wavelengths = np.linspace(1000.0, 1700.0, 9)
    summaries = {}
    for i in range(3):
        base = rng.normal(0.3 + 0.1 * i, 0.01, 9).astype(np.float32)
        summaries[f"roi-{i}"] = RoiSpectrumSummary(f"roi-{i}", base, base - 0.02, base,
                                                   base + 0.02)
    return {"wavelengths": wavelengths, "signal_label": "Absorbance",
            "names": {k: f"T{i}" for i, k in enumerate(summaries)},
            "means": {k: s.mean for k, s in summaries.items()},
            "values": {"roi-0": 1.0, "roi-1": 3.0, "roi-2": None},
            "value_label": "API (%)", "selected": summaries["roi-1"], "selected_name": "T1"}


_PAYLOADS = {"model": _model_payload, "predict": _predict_payload, "explore": _explore_payload}


def _registry():
    import enginir.viewer._model_diagnostics  # noqa: F401 (registers the Model plots)
    import enginir.viewer._prediction  # noqa: F401 (registers the Predict plots)
    import enginir.viewer._project_rois  # noqa: F401 (registers the Project ROIs plots)
    from enginir.viewer._plot_registry import PLOT_SPECS, WORKSPACE_PAGES

    return PLOT_SPECS, WORKSPACE_PAGES


def _registered_plots() -> list[tuple[str, str]]:
    specs, _pages = _registry()
    return [(plot_id.split(".")[0], plot_id) for plot_id in sorted(specs)]


def test_every_registered_plot_has_a_test_payload_and_model_plots_are_on_pages():
    specs, pages = _registry()
    assert {"model.pred_vs_ref", "predict.histogram"} <= set(specs)
    assert all(workspace in _PAYLOADS for workspace, _plot_id in _registered_plots())
    model_plots = {plot_id for page in pages["Model"] for plot_id in page.plot_ids}
    assert model_plots == {plot_id for plot_id in specs if plot_id.startswith("model.")}


def _drawn_values(fig) -> np.ndarray:
    values: list[np.ndarray] = []
    for ax in fig.axes:
        for collection in ax.collections:
            values.append(np.ravel(collection.get_offsets()))
            values += [np.ravel(path.vertices) for path in collection.get_paths()]
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
