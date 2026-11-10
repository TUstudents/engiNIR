"""Project ROIs tab: every stored ROI, redrawn from the project file alone.

The ROI mosaic shows each ROI's stored pixels as a tile (true size), one
block per data file, in a compact grid or at its place in the scan, with the
ROI name and the current target's value; the outline has the value's colour.
Beside it, two plot panels:

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
from dataclasses import dataclass
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
from PyQt6.QtGui import QAction, QActionGroup, QColor, QFont, QFontMetricsF
from PyQt6.QtWidgets import (
    QComboBox,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

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
from ._theme import GUIDE_COLOR, panel_colors, print_colors, secondary_text_style
from ._ui_helpers import spectral_axis_label, style_axes

logger = get_logger(__name__)

#: Raw bands of the false colour, as in the Spatial Image.
_RGB_NM = (1650.0, 1300.0, 1050.0)
_TILE_BATCH = 40
_ALL_FILES = "__all__"


def rgb_band_indices(wavelengths: np.ndarray) -> list[int]:
    """Bands nearest the Spatial Image's false-colour wavelengths."""
    axis = np.asarray(wavelengths, dtype=float)
    if axis.size == 0:
        return [0, 0, 0]
    return [int(np.argmin(np.abs(axis - nm))) for nm in _RGB_NM]


#: Layouts of the mosaic: a compact grid, or each ROI at its place in the scan.
GRID, SCAN = "grid", "scan"
#: Contrast stretch of the tiles: one for all ROIs, or one per data file.
SHARED, PER_FILE = "shared", "per_file"


@dataclass(frozen=True)
class MosaicLayout:
    """Where the mosaic draws each tile and header, in mosaic units (pixels of the scans).

    ``places`` maps an ROI id to ``(x, y, width, height)``; ``headers`` holds
    ``(dataset_id, name, x, y, width)`` per data file block; ``unit`` is the
    label text height, so text scales with the tiles and has room reserved.
    """

    places: dict[str, tuple[float, float, float, float]]
    headers: list[tuple[str, str, float, float, float]]
    unit: float


def _position_cell(position: Any) -> tuple[int, int] | None:
    """``"B4"`` → ``(1, 3)`` (row, column from 0), the inverse of the ROI grid's names."""
    text = str(position or "").strip().upper()
    letters = "".join(ch for ch in text if ch.isalpha())
    digits = text[len(letters):]
    if not letters or not text.startswith(letters) or not digits.isdigit() or int(digits) < 1:
        return None
    row = 0
    for ch in letters:
        row = row * 26 + (ord(ch) - ord("A") + 1)
    return row - 1, int(digits) - 1


def grid_cells(members: list[TrainingROI]) -> dict[str, tuple[int, int]]:
    """Row and column of each ROI of one data file in the compact grid.

    The ROI grid's positions when every ROI has a distinct one (empty wells
    keep their place, so B4 stays B4); otherwise rows in scan order: ROIs whose
    rows overlap the first ROI of a row join it, ordered along the frames.
    """
    cells = {roi.roi_id: _position_cell((roi.provenance or {}).get("grid_position"))
             for roi in members}
    if all(cell is not None for cell in cells.values()) \
            and len(set(cells.values())) == len(cells):
        found = [cell for cell in cells.values() if cell is not None]
        top, left = min(r for r, _c in found), min(c for _r, c in found)
        return {roi_id: (cell[0] - top, cell[1] - left)
                for roi_id, cell in cells.items() if cell is not None}
    rows: list[list[TrainingROI]] = []
    for roi in sorted(members, key=lambda r: (r.spatial_bounds[2], r.spatial_bounds[0])):
        if rows:
            first = rows[-1][0].spatial_bounds
            if roi.spatial_bounds[2] <= (first[2] + first[3]) / 2:
                rows[-1].append(roi)
                continue
        rows.append([roi])
    return {roi.roi_id: (i, j) for i, row in enumerate(rows)
            for j, roi in enumerate(sorted(row, key=lambda r: r.spatial_bounds[0]))}


def mosaic_layout(rois: list[TrainingROI], file_names: Mapping[str, str],
                  mode: str = GRID) -> MosaicLayout:
    """Lay out the ROIs, one block per data file, stacked down with a header each.

    ``GRID``: equal cells in the rows and columns of :func:`grid_cells`; each
    tile keeps its size in pixels. ``SCAN``: each ROI at its frame (x) and row
    (y) in its data file. Room for the header above a block and for the labels
    below each tile is reserved in the same units, so text never covers tiles.
    """
    by_file: dict[str, list[TrainingROI]] = {}
    for roi in rois:
        by_file.setdefault(roi.dataset_id, []).append(roi)
    if not rois:
        return MosaicLayout({}, [], 1.0)
    cell_w = max(float(r.spatial_bounds[1] - r.spatial_bounds[0] + 1) for r in rois)
    cell_h = max(float(r.spatial_bounds[3] - r.spatial_bounds[2] + 1) for r in rois)
    unit = 0.16 * max(cell_h, cell_w / 2.5)
    label_band, header_band, gap = 1.6 * unit, 2.0 * unit, 0.6 * unit
    column_gap = max(1.5 * unit, 0.08 * cell_w)
    places: dict[str, tuple[float, float, float, float]] = {}
    blocks: list[tuple[str, float, float]] = []  # dataset id, header y, block width
    y = 0.0
    for dataset_id in sorted(by_file, key=lambda d: file_names.get(d, d).lower()):
        members = by_file[dataset_id]
        header_y = y
        y += header_band
        if mode == SCAN:
            top = min(roi.spatial_bounds[2] for roi in members)
            bottom = max(roi.spatial_bounds[3] for roi in members)
            for roi in members:
                f0, f1, r0, r1 = roi.spatial_bounds
                places[roi.roi_id] = (float(f0), y + r0 - top, float(f1 - f0 + 1),
                                      float(r1 - r0 + 1))
            width = max(float(roi.spatial_bounds[1] + 1) for roi in members)
            y += bottom - top + 1 + label_band
        else:
            cells = grid_cells(members)
            n_rows = 1 + max(i for i, _j in cells.values())
            n_columns = 1 + max(j for _i, j in cells.values())
            for roi in members:
                i, j = cells[roi.roi_id]
                f0, f1, r0, r1 = roi.spatial_bounds
                places[roi.roi_id] = (j * (cell_w + column_gap), y + i * (cell_h + label_band),
                                      float(f1 - f0 + 1), float(r1 - r0 + 1))
            width = n_columns * (cell_w + column_gap) - column_gap
            y += n_rows * (cell_h + label_band)
        blocks.append((dataset_id, header_y, width))
        y += gap
    widest = max(width for _d, _y, width in blocks)
    headers = [(dataset_id, file_names.get(dataset_id, dataset_id), 0.0, header_y,
                max(width, widest)) for dataset_id, header_y, width in blocks]
    return MosaicLayout(places, headers, unit)


def _short_names(names: Mapping[str, str], file_names: Mapping[str, str]) -> dict[str, str]:
    """ROI names without the file name the grid tool adds (``tray03-B04`` → ``B04``;
    ``file_names`` maps an ROI id to its data file name) and without a prefix all
    names share, up to a separator (``API_10p_L1_01``, ``API_10p_L4_01`` → ``L1_01``,
    ``L4_01``)."""
    short: dict[str, str] = {}
    for roi_id, name in names.items():
        stem = os.path.splitext(file_names.get(roi_id, ""))[0]
        for separator in ("-", "_", " "):
            if stem and name.startswith(stem + separator) and len(name) > len(stem) + 1:
                name = name[len(stem) + 1:]
                break
        short[roi_id] = name
    if len(short) > 1:
        prefix = os.path.commonprefix(list(short.values()))
        cut = max(prefix.rfind(separator) for separator in ("-", "_", " ")) + 1
        if cut > 0 and all(len(name) > cut for name in short.values()):
            short = {roi_id: name[cut:] for roi_id, name in short.items()}
    return short


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


def stretch_to_rgb(patches: Mapping[str, np.ndarray],
                   groups: Mapping[str, str] | None = None) -> dict[str, np.ndarray]:
    """2–98 % contrast stretch per channel, shared by all patches or by each group
    (``groups`` maps an ROI id to its data file)."""
    limits = _group_limits(patches, groups)
    return {roi_id: _stretched(p, limits[(groups or {}).get(roi_id, "")])
            for roi_id, p in patches.items()}


def _group_limits(patches: Mapping[str, np.ndarray], groups: Mapping[str, str] | None
                  ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    members: dict[str, dict[str, np.ndarray]] = {}
    for roi_id, patch in patches.items():
        members.setdefault((groups or {}).get(roi_id, ""), {})[roi_id] = patch
    return {key: stretch_limits(group) for key, group in members.items()}


def _value_color(value: float | None, norm: Normalize | None) -> tuple[int, int, int]:
    if value is None or norm is None:
        return _rgb255(GUIDE_COLOR)
    r, g, b, _a = colormaps["viridis"](norm(value))
    return int(r * 255), int(g * 255), int(b * 255)


def _rgb255(color: str) -> tuple[int, int, int]:
    q = QColor(color)
    return q.red(), q.green(), q.blue()


class RoiMosaic(QWidget):
    """The ROI tiles; a :class:`~enginir.viewer._plot_panel.PlotSource`.

    Tiles are outlined in the colour of their target value (the scale of the
    ROI mean spectra; grey without a value). Text is drawn in mosaic units in
    the room :func:`mosaic_layout` reserves, so it scales with the tiles.
    :meth:`view_button` switches the layout (:data:`GRID`, :data:`SCAN`) and
    the stretch (:data:`SHARED`, :data:`PER_FILE`).
    """

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
        self.layout_mode = GRID
        self.stretch_mode = SHARED
        self._view_button = self._make_view_button()
        self._rois: dict[str, TrainingROI] = {}
        self._file_names: dict[str, str] = {}
        self._labels: dict[str, str] = {}
        self._values: dict[str, float | None] = {}
        self._roi_files: dict[str, str] = {}
        self._layout = MosaicLayout({}, [], 1.0)
        self._patches: dict[str, np.ndarray] = {}
        self._images: dict[str, pg.ImageItem] = {}
        self._outlines: dict[str, QGraphicsRectItem] = {}
        self._label_items: dict[str, QGraphicsSimpleTextItem] = {}
        # Fixed from the first tiles while they load, recomputed once all are in.
        self._limits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._selection: QGraphicsRectItem | None = None
        self._selected: str | None = None
        self.restyle_theme()

    # -- view options -----------------------------------------------------

    def view_button(self) -> QToolButton:
        """The header button with the layout and stretch choices."""
        return self._view_button

    def _make_view_button(self) -> QToolButton:
        button = QToolButton()
        button.setText("View")
        button.setToolTip("Layout of the tiles and their contrast stretch")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(button)
        self._mode_actions: dict[str, QAction] = {}
        for options, setter in (
                (((GRID, "Compact grid"), (SCAN, "Positions in the scan")), self.set_layout_mode),
                (((SHARED, "Stretch: all ROIs together"),
                  (PER_FILE, "Stretch: each data file")), self.set_stretch_mode)):
            group = QActionGroup(menu)
            for key, text in options:
                action = QAction(text, menu)
                action.setCheckable(True)
                action.triggered.connect(lambda _checked=False, key=key, setter=setter:
                                         setter(key))
                group.addAction(action)
                menu.addAction(action)
                self._mode_actions[key] = action
            menu.addSeparator()
        self._mode_actions[GRID].setChecked(True)
        self._mode_actions[SHARED].setChecked(True)
        button.setMenu(menu)
        return button

    def set_layout_mode(self, mode: str) -> None:
        """Lay the tiles out as a compact grid (:data:`GRID`) or at their scan places."""
        if mode not in (GRID, SCAN):
            log_and_raise("Unknown mosaic layout {!r}", mode)
        self._mode_actions[mode].setChecked(True)
        if mode != self.layout_mode:
            self.layout_mode = mode
            self._lay_out()

    def set_stretch_mode(self, mode: str) -> None:
        """Stretch the tiles' contrast together (:data:`SHARED`) or per data file."""
        if mode not in (SHARED, PER_FILE):
            log_and_raise("Unknown mosaic stretch {!r}", mode)
        self._mode_actions[mode].setChecked(True)
        if mode != self.stretch_mode:
            self.stretch_mode = mode
            self._redraw_images()

    # -- content --------------------------------------------------------

    def set_rois(self, rois: list[TrainingROI], file_names: Mapping[str, str],
                 labels: Mapping[str, str],
                 values: Mapping[str, float | None] | None = None) -> None:
        """Lay out ``rois`` (tiles stay empty until :meth:`add_patches`)."""
        self._patches = {k: v for k, v in self._patches.items() if k in {r.roi_id for r in rois}}
        self._rois = {roi.roi_id: roi for roi in rois}
        self._file_names = dict(file_names)
        self._labels = dict(labels)
        self._values = dict(values or {})
        self._roi_files = {roi.roi_id: file_names.get(roi.dataset_id, roi.dataset_id)
                           for roi in rois}
        self._lay_out()

    def _lay_out(self) -> None:
        self._plot.clear()
        self._images.clear()
        self._outlines.clear()
        self._label_items.clear()
        self._limits = {}
        self._selection = None
        self._layout = mosaic_layout(list(self._rois.values()), self._file_names,
                                     self.layout_mode)
        unit = self._layout.unit
        for _dataset_id, name, x, y, width in self._layout.headers:
            header = self._text(name, x, y + 0.2 * unit, 1.25 * unit, width, bold=True)
            header.setToolTip(name)
        for roi_id, (x, y, w, h) in self._layout.places.items():
            outline = QGraphicsRectItem(x, y, w, h)
            self._plot.addItem(outline)
            self._outlines[roi_id] = outline
            self._label_items[roi_id] = self._text(self._labels.get(roi_id, ""), x,
                                                   y + h + 0.15 * unit, unit, w)
        self._color_outlines()
        self._message.setText("" if self._rois else "No ROIs stored in the project yet.")
        self._redraw_images()
        self._view.autoRange(padding=0.02)
        self.select_roi(self._selected)

    def _text(self, text: str, x: float, y: float, height: float, width: float,
              bold: bool = False) -> QGraphicsSimpleTextItem:
        """Text ``height`` mosaic units tall, elided to ``width``; it zooms with the tiles."""
        item = QGraphicsSimpleTextItem()
        font = QFont(self.font())
        font.setBold(bold)
        item.setFont(font)
        metrics = QFontMetricsF(font)
        scale = height / max(metrics.height(), 1.0)
        item.setText(metrics.elidedText(text, Qt.TextElideMode.ElideMiddle,
                                        max(width / scale, 1.0)))
        item.setBrush(QColor(panel_colors()["fg"]))
        item.setScale(scale)
        item.setPos(x, y)
        item.setData(0, text)
        self._plot.addItem(item)
        return item

    def set_labels(self, labels: Mapping[str, str],
                   values: Mapping[str, float | None] | None = None) -> None:
        """Change labels and values (e.g. another target) without laying the tiles out again."""
        self._labels = dict(labels)
        if values is not None:
            self._values = dict(values)
        unit = self._layout.unit
        for roi_id, item in self._label_items.items():
            text = self._labels.get(roi_id, "")
            if item.data(0) != text:
                metrics = QFontMetricsF(item.font())
                width = self._layout.places[roi_id][2]
                item.setText(metrics.elidedText(text, Qt.TextElideMode.ElideMiddle,
                                                max(width / (unit / max(metrics.height(), 1.0)),
                                                    1.0)))
                item.setData(0, text)
        self._color_outlines()

    def label_text(self, roi_id: str) -> str:
        """The full label of a tile (as given, before eliding)."""
        item = self._label_items.get(roi_id)
        return "" if item is None else str(item.data(0))

    def outline_color(self, roi_id: str) -> tuple[int, int, int]:
        color = self._outlines[roi_id].pen().color()
        return color.red(), color.green(), color.blue()

    def _color_outlines(self) -> None:
        norm = value_norm(self._values.get(roi_id) for roi_id in self._outlines)
        for roi_id, outline in self._outlines.items():
            value = self._values.get(roi_id)
            pen = pg.mkPen(color=_value_color(value, norm), width=1 if value is None else 2.5)
            roi = self._rois.get(roi_id)
            if roi is not None and roi.use == "prediction":  # a sample to predict, not training
                pen.setStyle(Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            outline.setPen(pen)

    def shown_roi_ids(self) -> set[str]:
        return set(self._rois)

    def add_patches(self, patches: Mapping[str, np.ndarray]) -> None:
        """Raw band patches ``(frames, rows, 3)`` of some ROIs.

        Only the new tiles are drawn, with the stretch of the first tiles of
        their group; when the last tile arrives, all are stretched once more.
        """
        new = {k: v for k, v in patches.items() if k in self._rois}
        self._patches.update(new)
        if not self.missing_patches() and len(self._patches) > len(new):
            self._redraw_images()
            return
        missing = {self._group(roi_id) for roi_id in new} - set(self._limits)
        if missing:
            self._limits.update({key: limits for key, limits in self._all_limits().items()
                                 if key in missing})
        for roi_id in new:
            self._draw_tile(roi_id)

    def missing_patches(self) -> list[str]:
        return [roi_id for roi_id in self._rois if roi_id not in self._patches]

    def _group(self, roi_id: str) -> str:
        return self._rois[roi_id].dataset_id if self.stretch_mode == PER_FILE else ""

    def _all_limits(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return _group_limits(self._patches, {roi_id: self._group(roi_id)
                                             for roi_id in self._patches})

    def _redraw_images(self) -> None:
        for item in self._images.values():
            self._plot.removeItem(item)
        self._images.clear()
        self._limits = self._all_limits() if self._patches else {}
        for roi_id in self._patches:
            self._draw_tile(roi_id)

    def _draw_tile(self, roi_id: str) -> None:
        limits = self._limits.get(self._group(roi_id))
        if limits is None or roi_id not in self._layout.places:
            return
        old = self._images.pop(roi_id, None)
        if old is not None:
            self._plot.removeItem(old)
        x, y, w, h = self._layout.places[roi_id]
        rgb = _stretched(self._patches[roi_id], limits)
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
        if roi_id is None or roi_id not in self._layout.places:
            return
        x, y, w, h = self._layout.places[roi_id]
        margin = 0.15 * self._layout.unit  # outside the value outline, which stays visible
        self._selection = QGraphicsRectItem(x - margin, y - margin, w + 2 * margin,
                                            h + 2 * margin)
        pen = pg.mkPen(color=panel_colors()["fg"], width=2.5)
        pen.setCosmetic(True)
        self._selection.setPen(pen)
        self._plot.addItem(self._selection)

    @property
    def selected_roi(self) -> str | None:
        return self._selected

    @property
    def places(self) -> dict[str, tuple[float, float, float, float]]:
        return dict(self._layout.places)

    def header_names(self) -> list[str]:
        """The data file names of the blocks, top to bottom."""
        return [name for _dataset_id, name, _x, _y, _w in self._layout.headers]

    def roi_at(self, x: float, y: float) -> str | None:
        for roi_id, (rx, ry, w, h) in self._layout.places.items():
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
        if hasattr(self, "_layout"):
            self._lay_out()

    # -- plot source ------------------------------------------------------

    def plot_table(self) -> PlotTable | None:
        if not self._rois:
            return None
        places = self._layout.places
        rois = sorted(self._rois.values(), key=lambda r: (places[r.roi_id][1],
                                                          places[r.roi_id][0]))
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
            PlotColumn("value", [self._values.get(r.roi_id) for r in rois]),
        ), metadata={"layout": self.layout_mode, "stretch": self.stretch_mode})

    def print_figure(self) -> Figure:
        colors = print_colors()
        fig = Figure(figsize=(max(self.width(), 300) / 100, max(self.height(), 200) / 100),
                     facecolor=colors["bg"])
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        style_axes(ax, colors)
        ax.set_xticks([])
        ax.set_yticks([])
        places = self._layout.places
        groups = {roi_id: self._group(roi_id) for roi_id in self._patches}
        for roi_id, rgb in stretch_to_rgb(self._patches, groups).items():
            x, y, w, h = places[roi_id]
            ax.imshow(np.transpose(rgb, (1, 0, 2)), extent=(x, x + w, y + h, y),
                      interpolation="nearest")
        norm = value_norm(self._values.get(roi_id) for roi_id in places)
        for roi_id, (x, y, w, h) in places.items():
            value = self._values.get(roi_id)
            color = tuple(c / 255 for c in _value_color(value, norm))
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color,
                                   linewidth=0.5 if value is None else 1.2))
            ax.text(x, y + h, self._labels.get(roi_id, ""), fontsize=5, va="top",
                    color=colors["fg"], clip_on=True)
        for _dataset_id, name, x, y, _width in self._layout.headers:
            ax.text(x, y, name, fontsize=7, va="top", color=colors["fg"])
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
    return payload.get("axis_label", "Wavelength (nm)")


def _x_unit(payload: Mapping[str, Any]) -> str:
    return payload.get("axis_unit", "nm")


def value_norm(values: Any) -> Normalize | None:
    """Colour scale of the target values; equal values get the middle colour, not the lowest."""
    known = [float(v) for v in values if v is not None]
    if not known:
        return None
    low, high = min(known), max(known)
    return Normalize(low - 1, high + 1) if low == high else Normalize(low, high)


def _render_means(fig: Figure, payload: Any, colors: Mapping[str, str]) -> None:
    ax = fig.add_subplot(111)
    style_axes(ax, colors)
    means: dict[str, np.ndarray] = payload.get("means", {})
    if not means:
        draw_placeholder(ax, payload.get("message") or "No ROI spectra in this signal unit.",
                         colors)
        return
    values: Mapping[str, float | None] = payload.get("values", {})
    norm = value_norm(values.values())
    cmap = colormaps["viridis"]
    x = np.asarray(payload["wavelengths"])
    predicted = set(payload.get("prediction_ids", ()))
    for roi_id, mean in means.items():
        value = values.get(roi_id)
        color = cmap(norm(value)) if value is not None and norm is not None else GUIDE_COLOR
        ax.plot(x, mean, color=color, linewidth=0.8, alpha=0.85,
                linestyle="--" if roi_id in predicted else "-")
    if predicted & set(means):
        ax.text(0.99, 0.02, "dashed: prediction ROIs", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7, color=colors["fg"])
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
    return PlotTable(columns=(PlotColumn("wavelength", payload["wavelengths"], _x_unit(payload)),
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
    return PlotTable(columns=(PlotColumn("wavelength", payload["wavelengths"], _x_unit(payload)),
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
    #: The data file filter of this tab changed.
    data_file_filter_changed = pyqtSignal()

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
        self._file_filter.currentIndexChanged.connect(self.data_file_filter_changed.emit)
        self._status = QLabel("")
        self._status.setStyleSheet(secondary_text_style())
        self.mosaic = RoiMosaic()
        self.mosaic.roi_selected.connect(self._on_tile_selected)
        self.mosaic.roi_opened.connect(self.roi_opened.emit)
        self.mosaic_frame = PlotFrame("explore.roi_mosaic", "ROI Mosaic", self.mosaic,
                                      self.mosaic, provenance=provenance)
        self.mosaic_frame.add_header_widget(self.mosaic.view_button())
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

    def shown_dataset_id(self) -> str | None:
        """The data file this tab is filtered to, or ``None`` for all data files."""
        dataset = self._file_filter.currentData()
        return None if dataset in (None, _ALL_FILES) else str(dataset)

    def busy(self) -> bool:
        return self._tile_job is not None or self._spectra_job is not None

    # -- building -------------------------------------------------------

    def _shown_rois(self) -> list[TrainingROI]:
        dataset = self._file_filter.currentData()
        return [roi for roi in self.project_state.rois.values()
                if dataset in (None, _ALL_FILES) or roi.dataset_id == dataset]

    def _file_names(self) -> dict[str, str]:
        return dict(self.project_state.data_file_labels)

    def _values(self) -> dict[str, float | None]:
        """The current target's value of every ROI (``None`` without one)."""
        state = self.project_state
        target_id = self._current_target_id() or ""
        return {roi_id: (None if (assignment := state.target_assignments.get(
                    (roi_id, target_id))) is None else float(assignment.value))
                for roi_id in state.rois}

    def _labels(self) -> dict[str, str]:
        """Short ROI names (without what all shown ROIs share) and the current value."""
        state = self.project_state
        values = self._values()
        files = self._file_names()
        shown = self._shown_rois() or list(state.rois.values())
        short = _short_names({roi.roi_id: roi.name for roi in shown},
                             {roi.roi_id: files.get(roi.dataset_id, "") for roi in shown})
        labels: dict[str, str] = {}
        for roi in state.rois.values():
            value = values.get(roi.roi_id)
            labels[roi.roi_id] = short.get(roi.roi_id, roi.name) + (
                "" if value is None else f"  {value:.3g}")
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
        self.mosaic.set_rois(rois, self._file_names(), self._labels(), self._values())
        self._spectra_stale = True
        self._load_next_tiles()
        self._compute_spectra()

    def _refresh_labels(self, *_args: Any) -> None:
        self._labels_timer.start()

    def _apply_labels(self) -> None:
        if {roi.roi_id for roi in self._shown_rois()} != self.mosaic.shown_roi_ids():
            self._schedule_rebuild()
            return
        self.mosaic.set_labels(self._labels(), self._values())
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
        wavelengths = (self.state.recipe.output_wavelengths(rois[first].wavelengths)
                       if first in rois else np.array([]))
        label, unit = spectral_axis_label(rois[first].wavelengths if first in rois else [])
        return {"wavelengths": np.asarray(wavelengths), "axis_label": label, "axis_unit": unit,
                "signal_label":
                getattr(self, "_signal_label", ""),
                "names": {roi_id: rois[roi_id].name for roi_id in self._summaries
                          if roi_id in rois}}

    def _update_means_plot(self) -> None:
        state = self.project_state
        target_id = self._current_target_id()
        target = state.targets.get(target_id or "")
        payload = self._payload_base()
        payload["means"] = {roi_id: s.mean for roi_id, s in self._summaries.items()}
        payload["prediction_ids"] = [roi_id for roi_id in self._summaries
                                     if roi_id in state.rois
                                     and state.rois[roi_id].use == "prediction"]
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


__all__ = ["GRID", "PER_FILE", "SCAN", "SHARED", "MosaicLayout", "ProjectRoisView", "RoiMosaic",
           "grid_cells", "mosaic_layout", "rgb_band_indices", "stretch_to_rgb"]
