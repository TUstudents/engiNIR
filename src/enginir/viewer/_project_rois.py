"""Project ROIs tab: every stored ROI, redrawn from the project file alone.

The ROI mosaic shows each ROI's stored pixels as a tile at its position in
its data file (true size and place), one block per data file, with the ROI
name and the current target's value. Below it, two plot panels:

- ``explore.roi_means``: the mean spectrum of every shown ROI, coloured by
  the current target value;
- ``explore.roi_pixels``: the median and 5–95 % band of the selected ROI's
  pixels.

Both use the Explore signal unit and preprocessing and convert each ROI with
its own stored references, through the same function as model fitting
(:func:`enginir.modeling.roi_spectrum_summaries`). Tiles and spectra load in
background jobs; nothing needs a data file.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pyqtgraph as pg
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pyqtgraph.GraphicsScene import GraphicsScene
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QGraphicsRectItem, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .._checks import log_and_raise
from .._logging import get_logger
from ..io.project_store import load_roi_band_images
from ..modeling import roi_spectrum_summaries
from ..project import TrainingROI
from ._jobs import start_background_job
from ._plot_grid import PlotPage, PlotPageWidget, PlotSlot
from ._plot_panel import (
    PlotColumn,
    PlotFrame,
    PlotPanel,
    PlotSpec,
    PlotTable,
    draw_placeholder,
)
from ._plot_registry import register_plot
from ._state import ProjectState, ViewerState
from ._theme import panel_colors, print_colors, secondary_text_style
from ._ui_helpers import style_axes

logger = get_logger(__name__)

#: Raw bands of the false colour, as in the Spatial Image.
_RGB_NM = (1650.0, 1300.0, 1050.0)
_TILE_BATCH = 40
_ALL_FILES = "__all__"
_NO_VALUE_COLOR = "#9aa3ad"


def rgb_band_indices(wavelengths: np.ndarray) -> list[int]:
    """Bands nearest the Spatial Image's false-colour wavelengths."""
    axis = np.asarray(wavelengths, dtype=float)
    if axis.size == 0:
        return [0, 0, 0]
    return [int(np.argmin(np.abs(axis - nm))) for nm in _RGB_NM]


def mosaic_layout(rois: list[TrainingROI], file_names: Mapping[str, str]
                  ) -> tuple[dict[str, tuple[float, float, float, float]], list[tuple[str, float]]]:
    """Place each ROI at its frame/row position, one block per data file, stacked down.

    Returns ``{roi_id: (x, y, width, height)}`` in mosaic coordinates and the
    ``(file name, y)`` of each block's header.
    """
    by_file: dict[str, list[TrainingROI]] = {}
    for roi in rois:
        by_file.setdefault(roi.dataset_id, []).append(roi)
    places: dict[str, tuple[float, float, float, float]] = {}
    headers: list[tuple[str, float]] = []
    y = 0.0
    for dataset_id in sorted(by_file, key=lambda d: file_names.get(d, d).lower()):
        members = by_file[dataset_id]
        top = min(roi.spatial_bounds[2] for roi in members)
        bottom = max(roi.spatial_bounds[3] for roi in members)
        header = max(4.0, 0.12 * (bottom - top + 1))
        headers.append((file_names.get(dataset_id, dataset_id), y))
        y += header
        for roi in members:
            f0, f1, r0, r1 = roi.spatial_bounds
            places[roi.roi_id] = (float(f0), y + r0 - top, float(f1 - f0 + 1), float(r1 - r0 + 1))
        y += bottom - top + 1 + header
    return places, headers


def _short_name(name: str, file_name: str) -> str:
    """The ROI name without the data file prefix the grid tool adds (``tray03-B04`` → ``B04``);
    the block header already names the file."""
    stem = os.path.splitext(file_name)[0]
    for separator in ("-", "_", " "):
        if stem and name.startswith(stem + separator) and len(name) > len(stem) + 1:
            return name[len(stem) + 1:]
    return name


#: Pixels sampled for the shared stretch; enough for stable 2/98 % percentiles.
_STRETCH_SAMPLE = 200_000


def stretch_limits(patches: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The 2 and 98 % levels per channel of all patches (from at most ~200,000 pixels)."""
    pooled = np.concatenate([p.reshape(-1, p.shape[-1]) for p in patches.values()])
    step = max(1, pooled.shape[0] // _STRETCH_SAMPLE)
    low, high = np.percentile(pooled[::step].astype(np.float32), [2, 98], axis=0)
    return low, np.maximum(high - low, 1e-9)


def _stretched(patch: np.ndarray, limits: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    low, scale = limits
    return np.clip((patch.astype(np.float32) - low) / scale, 0.0, 1.0)


def stretch_to_rgb(patches: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """2–98 % contrast stretch per channel, shared by all patches."""
    if not patches:
        return {}
    limits = stretch_limits(patches)
    return {roi_id: _stretched(p, limits) for roi_id, p in patches.items()}


class RoiMosaic(QWidget):
    """The ROI tiles; a :class:`~enginir.viewer._plot_panel.PlotSource`."""

    roi_selected = pyqtSignal(str)
    roi_opened = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot_widget = pg.PlotWidget()
        plot = self._plot_widget.getPlotItem()
        if plot is None:
            log_and_raise("PyQtGraph plot item unavailable for the ROI mosaic",
                          error=RuntimeError)
        self._plot = plot
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._view = self._plot.getViewBox()
        self._view.invertY(True)
        self._view.setAspectLocked(True)
        scene = self._plot_widget.scene()
        if not isinstance(scene, GraphicsScene):
            log_and_raise("PyQtGraph scene unavailable for the ROI mosaic", error=RuntimeError)
        scene.sigMouseClicked.connect(self._on_click)
        self._message = QLabel("")
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setStyleSheet(secondary_text_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_widget, 1)
        layout.addWidget(self._message)
        self._rois: dict[str, TrainingROI] = {}
        self._labels: dict[str, str] = {}
        self._roi_files: dict[str, str] = {}
        self._places: dict[str, tuple[float, float, float, float]] = {}
        self._headers: list[tuple[str, float]] = []
        self._patches: dict[str, np.ndarray] = {}
        self._images: dict[str, pg.ImageItem] = {}
        self._label_items: dict[str, pg.TextItem] = {}
        # Fixed from the first tiles while they load, recomputed once all are in.
        self._limits: tuple[np.ndarray, np.ndarray] | None = None
        self._selection: QGraphicsRectItem | None = None
        self._selected: str | None = None
        self.restyle_theme()

    # -- content --------------------------------------------------------

    def set_rois(self, rois: list[TrainingROI], file_names: Mapping[str, str],
                 labels: Mapping[str, str]) -> None:
        """Lay out ``rois`` (tiles stay empty until :meth:`add_patches`)."""
        self._plot.clear()
        self._images.clear()
        self._label_items.clear()
        self._limits = None
        self._selection = None
        self._patches = {k: v for k, v in self._patches.items() if k in {r.roi_id for r in rois}}
        self._rois = {roi.roi_id: roi for roi in rois}
        self._labels = dict(labels)
        self._places, self._headers = mosaic_layout(rois, file_names)
        self._roi_files = {roi.roi_id: file_names.get(roi.dataset_id, roi.dataset_id)
                           for roi in rois}
        colors = panel_colors()
        for name, y in self._headers:
            header = pg.TextItem(name, color=colors["fg"], anchor=(0, 0))
            header.setPos(0, y)
            self._plot.addItem(header)
        for roi_id, (x, y, w, h) in self._places.items():
            outline = QGraphicsRectItem(x, y, w, h)
            pen = pg.mkPen(color=(120, 120, 120), width=1)
            pen.setCosmetic(True)
            outline.setPen(pen)
            self._plot.addItem(outline)
            label = pg.TextItem(self._labels.get(roi_id, ""), color=colors["fg"], anchor=(0, 0))
            font = label.textItem.font()
            font.setPointSizeF(7.0)
            label.setFont(font)
            label.setPos(x, y + h)
            self._plot.addItem(label)
            self._label_items[roi_id] = label
        self._message.setText("" if rois else "No ROIs stored in the project yet.")
        self._redraw_images()
        self._view.autoRange(padding=0.02)
        self.select_roi(self._selected)


    def set_labels(self, labels: Mapping[str, str]) -> None:
        """Change the tile labels (e.g. another target) without laying the tiles out again."""
        self._labels = dict(labels)
        for roi_id, item in self._label_items.items():
            text = self._labels.get(roi_id, "")
            if item.toPlainText() != text:
                item.setText(text)

    def shown_roi_ids(self) -> set[str]:
        return set(self._rois)

    def add_patches(self, patches: Mapping[str, np.ndarray]) -> None:
        """Raw band patches ``(frames, rows, 3)`` of some ROIs.

        Only the new tiles are drawn, with the stretch of the first tiles;
        when the last tile arrives, all are stretched once more together.
        """
        new = {k: v for k, v in patches.items() if k in self._rois}
        self._patches.update(new)
        if self._limits is None and self._patches:
            self._limits = stretch_limits(self._patches)
        if not self.missing_patches() and len(self._patches) > len(new):
            self._limits = stretch_limits(self._patches)
            for roi_id, item in self._images.items():
                item.setImage(np.transpose(_stretched(self._patches[roi_id], self._limits),
                                           (1, 0, 2)), levels=(0.0, 1.0))
        for roi_id in new:
            self._draw_tile(roi_id)

    def missing_patches(self) -> list[str]:
        return [roi_id for roi_id in self._rois if roi_id not in self._patches]

    def _redraw_images(self) -> None:
        for item in self._images.values():
            self._plot.removeItem(item)
        self._images.clear()
        self._limits = stretch_limits(self._patches) if self._patches else None
        for roi_id in self._patches:
            self._draw_tile(roi_id)

    def _draw_tile(self, roi_id: str) -> None:
        if self._limits is None:
            return
        old = self._images.pop(roi_id, None)
        if old is not None:
            self._plot.removeItem(old)
        x, y, w, h = self._places[roi_id]
        rgb = _stretched(self._patches[roi_id], self._limits)
        item = pg.ImageItem(np.transpose(rgb, (1, 0, 2)), axisOrder="row-major",
                            levels=(0.0, 1.0))
        item.setRect(QRectF(x, y, w, h))
        item.setZValue(-1)
        self._plot.addItem(item)
        self._images[roi_id] = item

    # -- selection ------------------------------------------------------

    def select_roi(self, roi_id: str | None) -> None:
        if self._selection is not None:
            self._plot.removeItem(self._selection)
            self._selection = None
        self._selected = roi_id
        if roi_id is None or roi_id not in self._places:
            return
        x, y, w, h = self._places[roi_id]
        self._selection = QGraphicsRectItem(x, y, w, h)
        pen = pg.mkPen(color=(255, 255, 0), width=2)
        pen.setCosmetic(True)
        self._selection.setPen(pen)
        self._plot.addItem(self._selection)

    @property
    def selected_roi(self) -> str | None:
        return self._selected

    def roi_at(self, x: float, y: float) -> str | None:
        for roi_id, (rx, ry, w, h) in self._places.items():
            if rx <= x < rx + w and ry <= y < ry + h:
                return roi_id
        return None

    def _on_click(self, event: Any) -> None:
        scene_pos = event.scenePos()
        if not self._view.sceneBoundingRect().contains(scene_pos):
            return
        point = self._view.mapSceneToView(scene_pos)
        roi_id = self.roi_at(point.x(), point.y())
        if roi_id is None:
            return
        self.select_roi(roi_id)
        if event.double():
            self.roi_opened.emit(roi_id)
        else:
            self.roi_selected.emit(roi_id)

    def tile_count(self) -> int:
        return len(self._images)

    def restyle_theme(self) -> None:
        self._plot_widget.setBackground(panel_colors()["bg"])

    # -- plot source ------------------------------------------------------

    def plot_table(self) -> PlotTable | None:
        if not self._rois:
            return None
        rois = sorted(self._rois.values(), key=lambda r: (self._places[r.roi_id][1],
                                                          self._places[r.roi_id][0]))
        return PlotTable(columns=(
            PlotColumn("data_file", [self._roi_files[r.roi_id] for r in rois]),
            PlotColumn("roi_name", [r.name for r in rois]),
            PlotColumn("roi_id", [r.roi_id for r in rois]),
            PlotColumn("frame_first", [r.spatial_bounds[0] for r in rois]),
            PlotColumn("frame_last", [r.spatial_bounds[1] for r in rois]),
            PlotColumn("row_first", [r.spatial_bounds[2] for r in rois]),
            PlotColumn("row_last", [r.spatial_bounds[3] for r in rois]),
            PlotColumn("pixels", [r.n_pixels for r in rois]),
            PlotColumn("label", [self._labels.get(r.roi_id, "") for r in rois]),
        ))

    def print_figure(self) -> Figure:
        colors = print_colors()
        fig = Figure(figsize=(max(self.width(), 300) / 100, max(self.height(), 200) / 100),
                     facecolor=colors["bg"])
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        style_axes(ax, colors)
        ax.set_xticks([])
        ax.set_yticks([])
        for roi_id, rgb in stretch_to_rgb(self._patches).items():
            x, y, w, h = self._places[roi_id]
            ax.imshow(np.transpose(rgb, (1, 0, 2)), extent=(x, x + w, y + h, y),
                      interpolation="nearest")
        for roi_id, (x, y, w, h) in self._places.items():
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#888888",
                                   linewidth=0.5))
            ax.text(x, y + h, self._labels.get(roi_id, ""), fontsize=5, va="top",
                    color=colors["fg"])
        for name, y in self._headers:
            ax.text(0, y, name, fontsize=7, va="top", color=colors["fg"])
        (x0, x1), (y0, y1) = self._view.viewRange()
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_aspect("equal")
        fig.tight_layout()
        return fig


# ----------------------------------------------------------------------
# Spectra plots
# ----------------------------------------------------------------------


def _x_label(payload: Mapping[str, Any]) -> str:
    return "Wavelength (nm)"


def _render_means(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = fig.add_subplot(111)
    style_axes(ax, colors)
    means: dict[str, np.ndarray] = payload.get("means", {})
    if not means:
        draw_placeholder(ax, payload.get("message") or "No ROI spectra in this signal unit.",
                         colors)
        return
    values: Mapping[str, float | None] = payload.get("values", {})
    known = [v for v in values.values() if v is not None]
    norm = Normalize(min(known), max(known)) if known else None
    cmap = colormaps["viridis"]
    x = np.asarray(payload["wavelengths"])
    for roi_id, mean in means.items():
        value = values.get(roi_id)
        color = cmap(norm(value)) if value is not None and norm is not None else _NO_VALUE_COLOR
        ax.plot(x, mean, color=color, linewidth=0.8, alpha=0.85)
    if norm is not None:
        mappable = fig.colorbar(colormaps_mappable(norm, cmap), ax=ax, fraction=0.05, pad=0.02)
        mappable.set_label(payload.get("value_label", ""), color=colors["fg"], fontsize=8)
        mappable.ax.tick_params(colors=colors["fg"], labelsize=7)
    ax.set_xlabel(_x_label(payload), color=colors["fg"])
    ax.set_ylabel(payload.get("signal_label", ""), color=colors["fg"])


def colormaps_mappable(norm: Normalize, cmap: Any) -> Any:
    from matplotlib.cm import ScalarMappable

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    return mappable


def _table_means(payload: Any) -> PlotTable:
    means: dict[str, np.ndarray] = payload.get("means", {})
    if not means:
        return PlotTable()
    names: Mapping[str, str] = payload.get("names", {})
    return PlotTable(columns=(PlotColumn("wavelength", payload["wavelengths"], "nm"),
                              *(PlotColumn(names.get(roi_id, roi_id), mean)
                                for roi_id, mean in means.items())),
                     metadata={"signal": payload.get("signal_label", ""),
                               "colour": payload.get("value_label", "")})


def _render_pixels(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = fig.add_subplot(111)
    style_axes(ax, colors)
    summary = payload.get("selected")
    if summary is None:
        draw_placeholder(ax, "Select an ROI to see the spread of its pixels.", colors)
        return
    x = np.asarray(payload["wavelengths"])
    ax.fill_between(x, summary.p05, summary.p95, color=colors["accent"], alpha=0.25,
                    linewidth=0)
    ax.plot(x, summary.median, color=colors["accent"], linewidth=1.2)
    ax.set_xlabel(_x_label(payload), color=colors["fg"])
    ax.set_ylabel(payload.get("signal_label", ""), color=colors["fg"])
    ax.text(0.01, 0.97, payload.get("selected_name", ""), transform=ax.transAxes, va="top",
            fontsize=8, color=colors["fg"])


def _table_pixels(payload: Any) -> PlotTable:
    summary = payload.get("selected")
    if summary is None:
        return PlotTable()
    return PlotTable(columns=(PlotColumn("wavelength", payload["wavelengths"], "nm"),
                              PlotColumn("p05", summary.p05), PlotColumn("median", summary.median),
                              PlotColumn("p95", summary.p95)),
                     metadata={"roi": payload.get("selected_name", ""),
                               "signal": payload.get("signal_label", "")})


ROI_MEANS = register_plot(PlotSpec("explore.roi_means", "ROI Mean Spectra", _render_means,
                                   _table_means, empty_text="No ROIs stored yet."))
ROI_PIXELS = register_plot(PlotSpec("explore.roi_pixels", "Selected ROI Pixels", _render_pixels,
                                    _table_pixels, empty_text="No ROIs stored yet."))


class ProjectRoisView(QWidget):
    """The Project ROIs tab: data file filter, mosaic and the two ROI spectra plots."""

    roi_selected = pyqtSignal(str)
    roi_opened = pyqtSignal(str)

    def __init__(self, state: ViewerState, project_state: ProjectState,
                 current_target_id: Callable[[], str | None],
                 provenance: Callable[[], Mapping[str, Any]],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.project_state = project_state
        self._current_target_id = current_target_id
        self._file_filter = QComboBox()
        self._file_filter.currentIndexChanged.connect(self._schedule_rebuild)
        self._status = QLabel("")
        self._status.setStyleSheet(secondary_text_style())
        self.mosaic = RoiMosaic()
        self.mosaic.roi_selected.connect(self._on_tile_selected)
        self.mosaic.roi_opened.connect(self.roi_opened.emit)
        self.mosaic_frame = PlotFrame("explore.roi_mosaic", "ROI Mosaic", self.mosaic,
                                      self.mosaic, provenance=provenance)
        self.means = PlotPanel(ROI_MEANS)
        self.pixels = PlotPanel(ROI_PIXELS)
        self.page = PlotPageWidget(
            PlotPage("Project ROIs", (PlotSlot("explore.roi_mosaic", 0, 0, row_span=2,
                                               column_span=2),
                                      PlotSlot("explore.roi_means", 0, 2),
                                      PlotSlot("explore.roi_pixels", 1, 2))),
            {"explore.roi_mosaic": self.mosaic_frame, "explore.roi_means": self.means,
             "explore.roi_pixels": self.pixels})
        top = QHBoxLayout()
        top.addWidget(QLabel("Data file:"))
        top.addWidget(self._file_filter, 1)
        top.addWidget(self._status, 2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.page, 1)

        self._summaries: dict[str, Any] = {}
        self._skipped: dict[str, str] = {}
        self._tile_job: Any = None
        self._spectra_job: Any = None
        self._spectra_stale = True
        self._rebuild_pending = True
        self._generation = 0
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(150)
        self._rebuild_timer.timeout.connect(self.rebuild)
        # Target and value changes come in bursts (opening, pasting): one label update per pass.
        self._labels_timer = QTimer(self)
        self._labels_timer.setSingleShot(True)
        self._labels_timer.setInterval(0)
        self._labels_timer.timeout.connect(self._apply_labels)
        for signal in (project_state.project_loaded, project_state.rois_changed,
                       project_state.datasets_changed):
            signal.connect(self._refresh_files)
        project_state.target_assignments_changed.connect(self._refresh_labels)
        project_state.targets_changed.connect(self._refresh_labels)
        state.units_changed.connect(self._spectra_settings_changed)
        self._refresh_files()

    # -- public ---------------------------------------------------------

    def select_roi(self, roi_id: str | None) -> None:
        self.mosaic.select_roi(roi_id)
        self._update_pixels_plot()

    def current_target_changed(self) -> None:
        self._refresh_labels()

    def busy(self) -> bool:
        return self._tile_job is not None or self._spectra_job is not None

    # -- building -------------------------------------------------------

    def _shown_rois(self) -> list[TrainingROI]:
        dataset = self._file_filter.currentData()
        return [roi for roi in self.project_state.rois.values()
                if dataset in (None, _ALL_FILES) or roi.dataset_id == dataset]

    def _file_names(self) -> dict[str, str]:
        return {dataset_id: os.path.basename(ref.source_path)
                for dataset_id, ref in self.project_state.datasets.items()}

    def _labels(self) -> dict[str, str]:
        state = self.project_state
        target_id = self._current_target_id()
        labels: dict[str, str] = {}
        names = self._file_names()
        for roi in state.rois.values():
            assignment = state.target_assignments.get((roi.roi_id, target_id or ""))
            labels[roi.roi_id] = _short_name(roi.name, names.get(roi.dataset_id, "")) + (
                "" if assignment is None else f" {assignment.value:.3g}")
        return labels

    def _refresh_files(self, *_args: Any) -> None:
        current = self._file_filter.currentData()
        self._file_filter.blockSignals(True)
        self._file_filter.clear()
        self._file_filter.addItem("All data files", _ALL_FILES)
        for dataset_id, name in sorted(self._file_names().items(), key=lambda i: i[1].lower()):
            self._file_filter.addItem(name, dataset_id)
        self._file_filter.setCurrentIndex(max(0, self._file_filter.findData(current)))
        self._file_filter.blockSignals(False)
        self._schedule_rebuild()

    def _schedule_rebuild(self, *_args: Any) -> None:
        if not self.isVisible():
            self._rebuild_pending = True  # nothing is read while the tab is hidden
            return
        self._rebuild_timer.start()

    def showEvent(self, a0: Any) -> None:  # noqa: N802 (Qt API)
        super().showEvent(a0)
        if self._rebuild_pending:
            self._rebuild_pending = False
            self._rebuild_timer.start()

    def rebuild(self) -> None:
        """Lay out the shown ROIs and (re)load their tiles and spectra in the background."""
        self._generation += 1
        rois = self._shown_rois()
        self.mosaic.set_rois(rois, self._file_names(), self._labels())
        self._spectra_stale = True
        self._load_next_tiles()
        self._compute_spectra()

    def _refresh_labels(self, *_args: Any) -> None:
        self._labels_timer.start()

    def _apply_labels(self) -> None:
        if {roi.roi_id for roi in self._shown_rois()} != self.mosaic.shown_roi_ids():
            self._schedule_rebuild()
            return
        self.mosaic.set_labels(self._labels())
        self._update_means_plot()

    def _spectra_settings_changed(self, *_args: Any) -> None:
        self._spectra_stale = True
        self._schedule_rebuild()

    def _load_next_tiles(self) -> None:
        if self._tile_job is not None or self.project_state.project_path is None:
            return
        missing = self.mosaic.missing_patches()[:_TILE_BATCH]
        if not missing:
            self._update_status()
            return
        rois = {roi_id: self.project_state.rois[roi_id] for roi_id in missing}
        path = str(self.project_state.project_path)
        generation = self._generation

        def load() -> dict[str, np.ndarray]:
            patches: dict[str, np.ndarray] = {}
            for roi_id, roi in rois.items():  # bands can differ between data files
                bands = rgb_band_indices(roi.wavelengths)
                patches.update(load_roi_band_images(path, {roi_id: roi}, bands))
            return patches

        def done() -> None:
            self._tile_job = None
            self._load_next_tiles()

        def loaded(patches: dict[str, np.ndarray]) -> None:
            if generation == self._generation:
                self.mosaic.add_patches(patches)
            else:
                self.mosaic.add_patches({k: v for k, v in patches.items()
                                         if k in self.mosaic.missing_patches()})

        self._tile_job = start_background_job(owner=self, fn=load, on_success=loaded,
                                              on_error=self._job_failed, on_finished=done)
        self._update_status()

    def _compute_spectra(self) -> None:
        if self._spectra_job is not None or self.project_state.project_path is None:
            return
        rois = self._shown_rois()
        if not rois:
            self._summaries, self._skipped = {}, {}
            self._spectra_stale = False
            self._update_means_plot()
            return
        state, project = self.state, self.project_state
        path, unit, recipe = str(project.project_path), str(state.y_unit), state.recipe
        datasets, calibrations = dict(project.datasets), dict(project.calibrations)
        self._spectra_stale = False

        def compute() -> Any:
            return roi_spectrum_summaries(path, rois, datasets=datasets,
                                          calibrations=calibrations, signal_unit=unit,
                                          recipe=recipe)

        def done() -> None:
            self._spectra_job = None
            if self._spectra_stale:
                self._compute_spectra()

        def computed(result: Any) -> None:
            self._summaries, self._skipped = result
            described = recipe.describe()
            self._signal_label = unit if described == "none" else f"{unit}, {described}"
            self._update_means_plot()

        self._signal_label = unit
        self._spectra_job = start_background_job(owner=self, fn=compute, on_success=computed,
                                                 on_error=self._job_failed, on_finished=done)
        self._update_status()

    def _job_failed(self, message: str) -> None:
        logger.error("Project ROIs could not be loaded: {}", message)
        self._status.setText(f"Could not load ROIs: {message}")

    # -- plots ----------------------------------------------------------

    def _payload_base(self) -> dict[str, Any]:
        rois = self.project_state.rois
        first = next(iter(self._summaries), None)
        wavelengths = rois[first].wavelengths if first in rois else np.array([])
        return {"wavelengths": np.asarray(wavelengths), "signal_label":
                getattr(self, "_signal_label", ""),
                "names": {roi_id: rois[roi_id].name for roi_id in self._summaries
                          if roi_id in rois}}

    def _update_means_plot(self) -> None:
        state = self.project_state
        target_id = self._current_target_id()
        target = state.targets.get(target_id or "")
        payload = self._payload_base()
        payload["means"] = {roi_id: s.mean for roi_id, s in self._summaries.items()}
        payload["values"] = {
            roi_id: (state.target_assignments[(roi_id, target_id)].value
                     if target_id and (roi_id, target_id) in state.target_assignments else None)
            for roi_id in self._summaries}
        payload["value_label"] = "" if target is None else (
            f"{target.name} ({target.unit})" if target.unit else target.name)
        if not self._summaries and self._skipped:
            payload["message"] = (f"None of the shown ROIs can be shown as {self.state.y_unit}"
                                  " (for example raw counts without references).")
        self.means.update_data(payload)
        self._update_pixels_plot()
        self._update_status()

    def _update_pixels_plot(self) -> None:
        selected = self.mosaic.selected_roi
        payload = self._payload_base()
        if selected is not None and selected in self._summaries:
            payload["selected"] = self._summaries[selected]
            payload["selected_name"] = self.project_state.rois[selected].name
        self.pixels.update_data(payload)

    def _update_status(self) -> None:
        shown = len(self._shown_rois())
        parts = [f"{shown} ROIs"]
        loading = self.mosaic.missing_patches()
        if loading:
            parts.append(f"loading {len(loading)} tiles")
        if self._spectra_job is not None:
            parts.append("computing spectra")
        if self._skipped:
            parts.append(f"{len(self._skipped)} not convertible to {self.state.y_unit}")
        self._status.setText(" | ".join(parts))

    def _on_tile_selected(self, roi_id: str) -> None:
        self._update_pixels_plot()
        self.roi_selected.emit(roi_id)


__all__ = ["ProjectRoisView", "RoiMosaic", "mosaic_layout", "rgb_band_indices",
           "stretch_to_rgb"]
