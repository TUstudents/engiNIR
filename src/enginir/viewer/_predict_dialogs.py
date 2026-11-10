"""Predict ROIs and Predict map: dialogs and the per-ROI results plot.

Prediction ROIs (ROI ``use = prediction``) are predicted from the project file
alone (:func:`enginir.prediction.predict_rois`); a map prediction predicts a
registered data file with its own calibration. Both dialogs name models and
data files, show the chain the prediction applies read-only (the data file's
calibration, the model's signal unit and recipe, the model), and refuse an
incompatible choice before running (viewer-workspace contract §3,
prediction-runtime contract, Prediction ROIs).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._plot_panel import PlotColumn, PlotSpec, PlotTable, RoiPoints, draw_placeholder
from ._theme import secondary_text_style
from ._ui_helpers import style_axes

#: "All data files" in the Predict ROIs dialog.
ALL_FILES = "__all__"


def _show_whole_items(combo: QComboBox) -> None:
    """Size the combo and its open list to its longest item (data file folders are long)."""
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    view = combo.view()
    if view is not None:
        width = max((combo.fontMetrics().horizontalAdvance(combo.itemText(i))
                     for i in range(combo.count())), default=0)
        view.setMinimumWidth(width + 40)


@dataclass(frozen=True)
class ModelChoice:
    """A model offered for prediction: its id, name and the chain it applies."""

    model_run_id: str
    name: str
    chain: str


class PredictRoisDialog(QDialog):
    """Choose models and the data file(s) whose prediction ROIs are predicted."""

    def __init__(self, models: Sequence[ModelChoice], files: Sequence[tuple[str, str, int]],
                 *, default_model_id: str | None = None,
                 default_file: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Predict ROIs")
        self._models = list(models)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Models (tick one or more; each gives its own columns):"))
        self._model_list = QListWidget()
        for model in self._models:
            item = QListWidgetItem(model.name)
            item.setData(Qt.ItemDataRole.UserRole, model.model_run_id)
            item.setToolTip(model.chain)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if model.model_run_id == default_model_id
                               else Qt.CheckState.Unchecked)
            self._model_list.addItem(item)
        self._model_list.itemChanged.connect(self._update)
        self._model_list.currentRowChanged.connect(self._update)
        layout.addWidget(self._model_list)
        form = QFormLayout()
        self._file_combo = QComboBox()
        total = sum(count for _id, _label, count in files)
        self._file_combo.addItem(f"All data files ({total} prediction ROIs)", ALL_FILES)
        for dataset_id, label, count in files:
            self._file_combo.addItem(f"{label} ({count} prediction ROIs)", dataset_id)
        self._file_combo.setCurrentIndex(max(0, self._file_combo.findData(default_file)))
        self._file_combo.currentIndexChanged.connect(self._update)
        _show_whole_items(self._file_combo)
        form.addRow("Prediction ROIs of:", self._file_combo)
        layout.addLayout(form)
        self._chain = QLabel("")
        self._chain.setWordWrap(True)
        self._chain.setStyleSheet(secondary_text_style())
        layout.addWidget(self._chain)
        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                         | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._counts = {dataset_id: count for dataset_id, _label, count in files}
        self._counts[ALL_FILES] = total
        self._update()

    @property
    def model_ids(self) -> list[str]:
        """The ticked models, in list order."""
        items = [self._model_list.item(row) for row in range(self._model_list.count())]
        return [str(item.data(Qt.ItemDataRole.UserRole)) for item in items
                if item is not None and item.checkState() == Qt.CheckState.Checked]

    @property
    def dataset_id(self) -> str | None:
        """The chosen data file, or ``None`` for all."""
        value = self._file_combo.currentData()
        return None if value == ALL_FILES else str(value)

    def _update(self, *_args: Any) -> None:
        chosen = [model for model in self._models if model.model_run_id in self.model_ids]
        lines = [f"<b>{model.name}</b>: {model.chain}" for model in chosen]
        count = self._counts.get(self._file_combo.currentData() or ALL_FILES, 0)
        if not count:
            lines.append("No prediction ROIs here: tick Prediction only when adding ROIs, "
                         "or the Predict column of the ROI table.")
        self._chain.setText("<br>".join(lines) or "Tick at least one model.")
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(bool(chosen) and count > 0)


class PredictMapDialog(QDialog):
    """Choose one model and one project data file to predict pixel by pixel."""

    def __init__(self, models: Sequence[ModelChoice], files: Sequence[tuple[str, str]], *,
                 check: Callable[[str, str], str | None],
                 add_data_file: Callable[[str], str | None] | None = None,
                 file_label: Callable[[str], str] = str,
                 default_model_id: str | None = None, default_file: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Predict map")
        self._models = {model.model_run_id: model for model in models}
        self._check, self._add_data_file, self._file_label = check, add_data_file, file_label
        form = QFormLayout(self)
        self._model_combo = QComboBox()
        for model in models:
            self._model_combo.addItem(model.name, model.model_run_id)
        self._model_combo.setCurrentIndex(max(0, self._model_combo.findData(default_model_id)))
        self._model_combo.currentIndexChanged.connect(self._update)
        form.addRow("Model:", self._model_combo)
        self._file_combo = QComboBox()
        for dataset_id, label in files:
            self._file_combo.addItem(label, dataset_id)
        self._file_combo.setCurrentIndex(max(0, self._file_combo.findData(default_file)))
        self._file_combo.currentIndexChanged.connect(self._update)
        _show_whole_items(self._file_combo)
        add = QPushButton("Add data file...")
        add.setToolTip("Register a new scan with the Add data file dialog (its calibration is "
                       "chosen there, as for training scans)")
        add.clicked.connect(self._add)
        add.setEnabled(add_data_file is not None)
        row = QHBoxLayout()
        row.addWidget(self._file_combo, 1)
        row.addWidget(add)
        form.addRow("Data file:", row)
        self._chain = QLabel("")
        self._chain.setWordWrap(True)
        self._chain.setStyleSheet(secondary_text_style())
        form.addRow("Chain:", self._chain)
        self._reason = QLabel("")
        self._reason.setWordWrap(True)
        form.addRow(self._reason)
        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                         | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        form.addRow(self._buttons)
        self._update()

    @property
    def model_id(self) -> str:
        return str(self._model_combo.currentData())

    @property
    def dataset_id(self) -> str | None:
        value = self._file_combo.currentData()
        return None if value is None else str(value)

    def _add(self) -> None:
        if self._add_data_file is None:
            return
        dataset_id = self._add_data_file(self.model_id)
        if dataset_id is None:
            return
        if self._file_combo.findData(dataset_id) < 0:
            self._file_combo.addItem(self._file_label(dataset_id), dataset_id)
            _show_whole_items(self._file_combo)
        self._file_combo.setCurrentIndex(self._file_combo.findData(dataset_id))

    def _update(self, *_args: Any) -> None:
        model = self._models.get(self.model_id)
        self._chain.setText("" if model is None else model.chain)
        reason = None
        if self.dataset_id is None:
            reason = "Add a data file first."
        elif model is not None:
            reason = self._check(model.model_run_id, self.dataset_id)
        self._reason.setText("" if reason is None else f"⚠ Cannot predict: {reason}")
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(model is not None and reason is None)


# ----------------------------------------------------------------------
# The per-ROI results plot (viewer-workspace contract §3)
# ----------------------------------------------------------------------

_MODEL_COLORS = ("#7c8cff", "#e3a33b", "#3fb27f", "#e05d6f", "#b07cff", "#4cc3d9")


def _rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get("roi_rows") or [])


def _render_roi_results(fig: Figure, payload: Any, colors: Mapping[str, str]
                        ) -> Sequence[RoiPoints] | None:
    ax = fig.add_subplot(111)
    style_axes(ax, colors)
    rows = _rows(payload)
    if not rows:
        draw_placeholder(ax, payload.get("message") or "Predict ROIs to see one result per "
                         "tablet (Predict > Predict ROIs...).", colors)
        return None
    from ._project_rois import _short_names

    models = list(dict.fromkeys(row["model"] for row in rows))
    rois = list(dict.fromkeys(row["roi_id"] for row in rows))
    names = _short_names({row["roi_id"]: row["roi"] for row in rows}, {})
    width = 0.8 / max(len(models), 1)
    points: list[RoiPoints] = []
    for m, model in enumerate(models):
        color = _MODEL_COLORS[m % len(_MODEL_COLORS)]
        mine = [row for row in rows if row["model"] == model]
        x = np.array([rois.index(row["roi_id"]) + (m - (len(models) - 1) / 2) * width
                      for row in mine])
        mean = np.array([row["mean"] for row in mine], dtype=float)
        low = np.array([row["p05"] for row in mine], dtype=float)
        high = np.array([row["p95"] for row in mine], dtype=float)
        ax.vlines(x, low, high, colors=color, linewidth=max(2.0, 18 * width), alpha=0.35)
        ax.scatter(x, mean, color=color, s=28, zorder=3, label=f"{model} (mean, 5–95 %)")
        if len(rows) <= 40:  # the reported value, readable at the dot
            for xi, value in zip(x, mean, strict=True):
                if np.isfinite(value):
                    ax.annotate(f"{value:.3g}", (xi, value), xytext=(5, 0),
                                textcoords="offset points", va="center", fontsize=7,
                                color=color)
        refs = [(xi, row["reference"]) for xi, row in zip(x, mine, strict=True)
                if row["reference"] is not None]
        if refs and m == 0:
            rx, ry = zip(*refs, strict=True)
            ax.scatter(rx, ry, marker="x", color=colors["fg"], s=40, zorder=4,
                       label="reference value")
        points.append(RoiPoints(ax, x, mean, tuple(row["roi_id"] for row in mine)))
    ax.set_xticks(range(len(rois)))
    labels = [names[roi_id] for roi_id in rois]
    long_labels = max(len(label) for label in labels) > 6
    ax.set_xticklabels(labels, rotation=60 if long_labels else 0,
                       ha="right" if long_labels else "center", fontsize=7)
    unit = str(payload.get("target_unit", ""))
    ax.set_ylabel(f"Prediction{f' ({unit})' if unit else ''}", color=colors["fg"])
    note = payload.get("exclusion_text", "")
    ax.set_xlabel(f"ROI ({note})" if note else "ROI", color=colors["fg"], fontsize=8)
    handles, labels_ = ax.get_legend_handles_labels()
    # Unusual spectra: a tablet with a larger share of pixels above DCrit than 90 % of the
    # model's training ROIs is ringed (the shares themselves are in the ROI table).
    band = payload.get("training_share")
    if band is not None:
        high = float(band[-1])
        ring_x, ring_y = [], []
        for m, model in enumerate(models):
            mine = [row for row in rows if row["model"] == model]
            for xi, row in zip(points[m].x, mine, strict=True):
                if np.isfinite(row["share_above_dcrit"]) and row["share_above_dcrit"] > high \
                        and np.isfinite(row["mean"]):
                    ring_x.append(xi)
                    ring_y.append(row["mean"])
        if ring_x:
            ax.scatter(ring_x, ring_y, s=150, facecolors="none", edgecolors=colors["fg"],
                       linewidths=1.4, zorder=4)
            handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                                  markeredgecolor=colors["fg"], markersize=9))
            labels_.append(f"unusual spectra (> {100 * high:.0f} % of pixels above DCrit, "
                           "training 90 %)")
    # Headroom above the bars for the legend, so it never hides a tablet.
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.12 * len(handles) * (high - low))
    legend = ax.legend(handles, labels_, fontsize=7, loc="upper left", frameon=True,
                       labelcolor=colors["fg"], framealpha=0.85)
    legend.get_frame().set_facecolor(colors.get("ax_bg", colors["bg"]))
    legend.get_frame().set_edgecolor("none")
    ax.set_xlim(-0.6, len(rois) - 0.4)
    return points


def _table_roi_results(payload: Any) -> PlotTable:
    rows = _rows(payload)
    if not rows:
        return PlotTable()
    unit = str(payload.get("target_unit", ""))

    def column(key: str, unit_: str = "") -> PlotColumn:
        return PlotColumn(key, [row[key] for row in rows], unit_)

    return PlotTable(
        columns=(column("roi"), column("roi_id"), column("data_file"), column("model"),
                 column("model_run_id"), column("pixels"), column("removed"),
                 column("mean", unit), column("median", unit), column("sd", unit),
                 column("p05", unit), column("p95", unit), column("mean_dmodx"),
                 column("mean_t2"), column("share_above_dcrit"), column("share_above_t2crit"),
                 PlotColumn("reference", [np.nan if row["reference"] is None
                                          else row["reference"] for row in rows], unit),
                 PlotColumn("residual", [np.nan if row["residual"] is None
                                         else row["residual"] for row in rows], unit)),
        metadata={"exclusion": payload.get("exclusion_text", "none") or "none",
                  **{f"rmsep {name}": text for name, text in
                     (payload.get("rmsep") or {}).items()}})


ROI_RESULTS = PlotSpec(
    "predict.roi_results", "Prediction per ROI", _render_roi_results, _table_roi_results,
    empty_text="No ROI predictions yet.",
    help=("Each prediction ROI (a tablet), predicted from its stored pixels: the dot is the "
          "mean of its pixel predictions (the reported value, written next to it), the bar "
          "the 5–95 % range of its pixels; one colour per model; × the reference value when "
          "the ROI has one. A ring marks unusual spectra: a larger share of the ROI's pixels "
          "above the DModX limit (0.99) than 90 % of the model's training ROIs have. The "
          "shares, SD and removed pixels are in the ROI table. Remove pixels above the DModX "
          "limit (Predict controls) to leave glare and substrate out of the statistics."))

__all__ = ["ALL_FILES", "ROI_RESULTS", "ModelChoice", "PredictMapDialog", "PredictRoisDialog"]
