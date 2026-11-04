"""Five matplotlib panel widgets for the HSI viewer.

Each widget is a QWidget containing a matplotlib Figure embedded via
FigureCanvasQTAgg. Panels are driven by ``ViewerState``; the raw-band panel
intentionally uses explicit raw-data accessors to show unprocessed cube data.

Each panel is a :class:`~enginir.viewer._plot_panel.PlotSource`: the viewer
shows it in a plot frame whose header carries the title (``plot_title``) and
exports the panel's current view (Plot Panel Policy). Panels update their
artists in place, so the CSV is read from the drawn artists.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from pyqtgraph.GraphicsScene import GraphicsScene
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QShowEvent
from PyQt6.QtWidgets import QLabel, QSizePolicy, QStackedLayout, QVBoxLayout, QWidget

from .._checks import log_and_raise
from .._logging import get_logger
from ._plot_panel import PlotColumn, PlotTable, print_copy
from ._state import ViewerState
from ._theme import panel_colors, print_colors
from ._ui_helpers import LayoutState, maybe_tight_layout, style_axes

logger = get_logger(__name__)

__all__ = [
    "DC3Panel",
    "HistogramPanel",
    "SensorPanel",
    "SpatialProfilePanel",
    "SpectrumPanel",
]


# ------------------------------------------------------------------
# Base panel
# ------------------------------------------------------------------


class _PanelBase(QWidget):
    """Shared behaviour of the Explore panels: header title, CSV provenance, deferral.

    Hidden dock panels defer expensive rebuilds until they are shown again.
    Panels that are visible by default in the Explore workspace can opt out of
    deferring their very first hidden build so guide lines and labels exist
    immediately after viewer construction.
    """

    #: The header title changed (e.g. the RAW panel's band).
    title_changed = pyqtSignal(str)
    plot_title = ""

    def __init__(self, state: ViewerState, *, defer_initial_hidden_refresh: bool = True) -> None:
        super().__init__()
        self._state = state
        self._pending_visible_refresh: Callable[[], None] | None = None
        self._defer_initial_hidden_refresh = defer_initial_hidden_refresh
        self._has_drawn_once = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)

    def plot_table(self) -> PlotTable | None:
        """The values drawn now, or ``None`` when the panel shows no data."""
        return None

    def print_figure(self) -> Figure:
        """The current view in the print colors."""
        raise NotImplementedError

    def restyle_theme(self) -> None:
        """Apply the current theme."""

    def _set_title(self, title: str) -> None:
        if title != self.plot_title:
            self.plot_title = title
            self.title_changed.emit(title)

    def _metadata(self, *keys: str) -> dict[str, object]:
        """CSV provenance of the current cursor position and signal unit."""
        state = self._state
        values: dict[str, object] = {"signal_unit": state.y_unit_effective}
        if "frame" in keys:
            values["frame"] = state.frame
        if "row" in keys:
            values["row"] = state.row
        if "band" in keys:
            values["band"] = state.band
            values["wavelength_nm"] = f"{float(state.wavelengths[state.band]):.6g}"
        return values

    def _defer_until_visible(self, callback: Callable[[], None]) -> bool:
        """Return True when the caller should defer work until the panel is shown."""
        if self.isVisible():
            return False
        if not self._defer_initial_hidden_refresh and not self._has_drawn_once:
            return False
        self._pending_visible_refresh = callback
        return True

    def showEvent(self, a0: QShowEvent | None) -> None:
        super().showEvent(a0)
        pending = self._pending_visible_refresh
        self._pending_visible_refresh = None
        if pending is not None:
            pending()


class _BasePanelWidget(_PanelBase):
    """Explore panel drawn with matplotlib (line plots and the histogram)."""

    def __init__(self, state: ViewerState, *, defer_initial_hidden_refresh: bool = True) -> None:
        super().__init__(state, defer_initial_hidden_refresh=defer_initial_hidden_refresh)
        self._fig = Figure(figsize=(4, 3), facecolor=panel_colors()["bg"])
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._last_layout_size: LayoutState | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self._canvas.mpl_connect("resize_event", self._on_canvas_resize)

    def _style_ax(self) -> None:
        """Apply the current theme styling to the current axes."""
        style_axes(self._ax)

    def _on_canvas_resize(self, _event: object) -> None:
        # Margins follow the new size (e.g. a maximized dock); the canvas redraws itself.
        if self._has_drawn_once:
            self._last_layout_size = maybe_tight_layout(
                self._fig, self._canvas, pad=1.2, last_size=self._last_layout_size)

    @property
    def figure(self) -> Figure:
        return self._fig

    def print_figure(self) -> Figure:
        """The current view in the print colors."""
        return print_copy(self._fig)

    def _show_empty(self, message: str) -> None:
        colors = panel_colors()
        self._ax.clear()
        self._style_ax()
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.text(
            0.5,
            0.5,
            message,
            color=colors["fg"],
            ha="center",
            va="center",
            transform=self._ax.transAxes,
            fontsize=9,
        )
        self._draw_idle()

    def restyle_theme(self) -> None:
        """Update the panel figure background for the current theme."""
        self._fig.set_facecolor(panel_colors()["bg"])
        self._last_layout_size = None

    def _draw_idle(self) -> None:
        """Draw only when visible; otherwise defer one refresh until show."""
        self._last_layout_size = maybe_tight_layout(
            self._fig,
            self._canvas,
            pad=1.2,
            last_size=self._last_layout_size,
        )
        self._has_drawn_once = True
        if self.isVisible():
            self._canvas.draw_idle()
        else:
            self._pending_visible_refresh = self._draw_idle


class _ImagePanel(_PanelBase):
    """Explore image panel drawn with pyqtgraph, for fast frame/band scrubbing.

    Pixel ``(row, column)`` is centred on integer coordinates, as in the
    matplotlib panels it replaces; the crosshair marks the selected pixel
    centres and a click picks the nearest pixel. Exports redraw the shown view
    with matplotlib in the print colors.
    """

    _CMAP = "inferno"

    def __init__(self, state: ViewerState, *, x_label: str, y_label: str,
                 defer_initial_hidden_refresh: bool = True) -> None:
        super().__init__(state, defer_initial_hidden_refresh=defer_initial_hidden_refresh)
        self._x_label, self._y_label = x_label, y_label
        self._data: np.ndarray | None = None
        self._crosshair = (0.0, 0.0)
        self._levels: tuple[float, float] = (0.0, 1.0)

        self._plot_widget = pg.PlotWidget()
        plot = self._plot_widget.getPlotItem()
        if plot is None:
            log_and_raise("PyQtGraph plot item unavailable for the image panel",
                          error=RuntimeError)
        self._plot = plot
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._view = self._plot.getViewBox()
        self._view.invertY(True)
        self._view.setMouseEnabled(x=False, y=False)
        self._image = pg.ImageItem(axisOrder="row-major")
        colormap = pg.colormap.get(self._CMAP, source="matplotlib")
        if not isinstance(colormap, pg.ColorMap):
            log_and_raise("Colormap '{}' unavailable for the image panel", self._CMAP,
                          error=RuntimeError)
        self._image.setColorMap(colormap)
        self._plot.addItem(self._image)
        pen = pg.mkPen("yellow", width=1)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        self._plot.addItem(self._vline)
        self._plot.addItem(self._hline)
        scene = self._plot_widget.scene()
        if not isinstance(scene, GraphicsScene):
            log_and_raise("PyQtGraph scene unavailable for the image panel", error=RuntimeError)
        scene.sigMouseClicked.connect(self._on_scene_click)

        self._message = QLabel("")
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._plot_widget)
        self._stack.addWidget(self._message)
        self.restyle_theme()

    # -- drawing --------------------------------------------------------

    def _set_image(self, data: np.ndarray, crosshair: tuple[float, float]) -> None:
        """Show ``data`` (rows x columns) with its own value range and the crosshair."""
        new_shape = self._data is None or self._data.shape != data.shape
        self._data = data
        finite = data[np.isfinite(data)] if data.size else data
        self._levels = ((float(finite.min()), float(finite.max())) if finite.size
                        else (0.0, 1.0))
        self._image.setImage(data, levels=self._levels, autoLevels=False)
        self._image.setRect(QRectF(-0.5, -0.5, data.shape[1], data.shape[0]))
        self._set_crosshair(*crosshair)
        if new_shape:
            self._view.autoRange(padding=0.02)
        self._stack.setCurrentWidget(self._plot_widget)
        self._has_drawn_once = True

    def _set_crosshair(self, x: float, y: float) -> None:
        self._crosshair = (float(x), float(y))
        self._vline.setPos(x)
        self._hline.setPos(y)

    def _show_empty(self, message: str) -> None:
        self._data = None
        self._image.clear()
        self._message.setText(message)
        self._stack.setCurrentWidget(self._message)
        self._has_drawn_once = True

    def restyle_theme(self) -> None:
        colors = panel_colors()
        self._plot_widget.setBackground(colors["bg"])
        self._message.setStyleSheet(f"color: {colors['fg']};")
        axis_color = QColor(colors["fg"])
        axis_color.setAlpha(160)
        for name, label in (("bottom", self._x_label), ("left", self._y_label)):
            axis = self._plot.getAxis(name)
            axis.setPen(pg.mkPen(axis_color))
            axis.setTextPen(pg.mkPen(colors["fg"]))
            self._plot.setLabel(name, label, color=colors["fg"])

    # -- picking --------------------------------------------------------

    def _on_scene_click(self, event: object) -> None:
        if self._data is None or not self._state.has_dataset:
            return
        scene_pos = getattr(event, "scenePos", None)
        if scene_pos is None or not self._view.sceneBoundingRect().contains(scene_pos()):
            return
        point = self._view.mapSceneToView(scene_pos())
        rows, columns = self._data.shape
        column = int(np.clip(round(point.x()), 0, columns - 1))
        row = int(np.clip(round(point.y()), 0, rows - 1))
        self._picked(column, row)

    def _picked(self, column: int, row: int) -> None:
        """A pixel was clicked; subclasses emit their pick signals."""

    # -- export ---------------------------------------------------------

    def print_figure(self) -> Figure:
        """The shown view (image, crosshair, zoom) redrawn in the print colors."""
        colors = print_colors()
        fig = Figure(figsize=(max(self.width(), 200) / 100, max(self.height(), 150) / 100),
                     facecolor=colors["bg"])
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        style_axes(ax, colors)
        if self._data is not None:
            ax.imshow(self._data, aspect="auto", origin="upper", interpolation="nearest",
                      cmap=self._CMAP, vmin=self._levels[0], vmax=self._levels[1])
            ax.axvline(self._crosshair[0], color="yellow", linewidth=0.8)
            ax.axhline(self._crosshair[1], color="yellow", linewidth=0.8)
            (x0, x1), (y0, y1) = self._view.viewRange()
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)  # rows grow downwards, as on screen
        ax.set_xlabel(self._x_label, color=colors["fg"])
        ax.set_ylabel(self._y_label, color=colors["fg"])
        fig.tight_layout()
        return fig


# ------------------------------------------------------------------
# Histogram panel
# ------------------------------------------------------------------


class HistogramPanel(_BasePanelWidget):
    """256-bin histogram of the current frame image (top-left dock)."""

    plot_title = "Histogram"

    def __init__(self, state: ViewerState) -> None:
        super().__init__(state)
        self._stairs = None
        self._build()
        state.frame_changed.connect(self._on_update)
        state.units_changed.connect(self._on_update)

    def _build(self) -> None:
        if self._defer_until_visible(self._build):
            return
        if not self._state.has_dataset:
            self._stairs = None
            self._show_empty("No dataset attached")
            return
        colors = panel_colors()
        counts, edges = np.histogram(self._state.current_frame_image(), bins=256)
        self._ax.clear()
        self._style_ax()
        # One filled step artist, updated in place on every frame change (a bar per bin
        # was four times slower to redraw while scrubbing through frames).
        self._stairs = self._ax.stairs(counts, edges, fill=True, color=colors["accent"])
        unit = self._state.y_unit_effective
        self._ax.set_xlabel(unit, color=colors["fg"], fontsize=8)
        self._ax.set_ylabel("Count", color=colors["fg"], fontsize=8)
        self._draw_idle()

    def _on_update(self, *_) -> None:
        if self._defer_until_visible(self._on_update):
            return
        if not self._state.has_dataset:
            self._build()
            return
        if self._stairs is None:
            self._build()
            return
        colors = panel_colors()
        counts, edges = np.histogram(self._state.current_frame_image(), bins=256)
        self._stairs.set_data(counts, edges)
        unit = self._state.y_unit_effective
        self._ax.set_xlabel(unit, color=colors["fg"], fontsize=8)
        # Limits straight from the bins: relim() walks the 256-step path and
        # cost more than half of this update.
        span = float(edges[-1] - edges[0]) or 1.0
        self._ax.set_xlim(float(edges[0]) - 0.05 * span, float(edges[-1]) + 0.05 * span)
        self._ax.set_ylim(0.0, 1.05 * max(float(counts.max()), 1.0))
        self._draw_idle()

    def restyle_theme(self) -> None:
        super().restyle_theme()
        self._build()

    def plot_table(self) -> PlotTable | None:
        if self._stairs is None:
            return None
        values, edges, _baseline = self._stairs.get_data()
        unit = self._state.y_unit_effective
        return PlotTable(columns=(PlotColumn("bin_start", edges[:-1], unit),
                                  PlotColumn("bin_end", edges[1:], unit),
                                  PlotColumn("count", values)),
                         metadata=self._metadata("frame"))


# ------------------------------------------------------------------
# Raw panel
# ------------------------------------------------------------------


class DC3Panel(_ImagePanel):
    """Raw band view: cube[:, :, band].T — height × n_frames with yellow crosshairs.

    Emits
    -----
    frame_pick(int): user clicked a new frame position
    row_pick(int):   user clicked a new row position
    """

    frame_pick = pyqtSignal(int)
    row_pick = pyqtSignal(int)
    plot_title = "RAW"

    def __init__(self, state: ViewerState) -> None:
        super().__init__(state, x_label="Frame", y_label="Row")
        self._build()
        state.frame_changed.connect(self._on_cursor_changed)
        state.row_changed.connect(self._on_cursor_changed)
        state.band_changed.connect(self._build)

    def _build(self, *_) -> None:
        if self._defer_until_visible(self._build):
            return
        if not self._state.has_dataset:
            self._set_title("RAW")
            self._show_empty("No dataset attached")
            return
        band = self._state.band
        self._set_image(self._state.raw_band_image(band), (self._state.frame, self._state.row))
        self._set_title(f"RAW  band {band} ({float(self._state.wavelengths[band]):.1f} nm)")

    def _on_cursor_changed(self, *_) -> None:
        # Only the crosshair moves; that is cheap enough to do while hidden.
        if self._data is not None:
            self._set_crosshair(self._state.frame, self._state.row)

    def _picked(self, column: int, row: int) -> None:
        self.frame_pick.emit(column)
        self.row_pick.emit(row)

    def plot_table(self) -> PlotTable | None:
        if self._data is None:
            return None
        metadata = self._metadata("band")
        metadata["signal_unit"] = "DN (raw)"
        return PlotTable(matrix=self._data, row_label="row", column_label="frame",
                         metadata=metadata)


# ------------------------------------------------------------------
# Sensor panel
# ------------------------------------------------------------------


class SensorPanel(_ImagePanel):
    """Sensor view: current_frame_image() — height × n_bands with crosshairs.

    Emits
    -----
    row_pick(int):  user clicked a new row
    band_pick(int): user clicked a new band
    """

    row_pick = pyqtSignal(int)
    band_pick = pyqtSignal(int)
    plot_title = "Sensor Frame"

    def __init__(self, state: ViewerState) -> None:
        super().__init__(state, x_label="Band", y_label="Cross-track",
                         defer_initial_hidden_refresh=False)
        self._build()
        state.frame_changed.connect(self._build)
        state.units_changed.connect(self._build)
        state.row_changed.connect(self._on_cursor_changed)
        state.band_changed.connect(self._on_cursor_changed)

    def _build(self, *_) -> None:
        """Show the processed current frame; a frame or unit change only swaps the image."""
        if self._defer_until_visible(self._build):
            return
        if not self._state.has_dataset:
            self._show_empty("No dataset attached")
            return
        self._set_image(self._state.current_frame_image(), (self._state.band, self._state.row))

    def _on_cursor_changed(self, *_) -> None:
        if self._data is not None:
            self._set_crosshair(self._state.band, self._state.row)

    def _picked(self, column: int, row: int) -> None:
        self.band_pick.emit(column)
        self.row_pick.emit(row)

    def plot_table(self) -> PlotTable | None:
        if self._data is None:
            return None
        return PlotTable(matrix=self._data, row_label="row", column_label="band",
                         metadata=self._metadata("frame"))


# ------------------------------------------------------------------
# Spatial profile panel
# ------------------------------------------------------------------


class SpatialProfilePanel(_BasePanelWidget):
    """Spatial profile: get_spatial_profile(band) with a draggable row marker.

    Emits
    -----
    row_pick(int): user dragged or clicked the marker to a new row
    """

    row_pick = pyqtSignal(int)
    plot_title = "Spatial Profile"

    def __init__(self, state: ViewerState) -> None:
        super().__init__(state, defer_initial_hidden_refresh=False)
        self._line = None
        self._marker = None
        self._drag_cid: int | None = None

        self._build()

        state.frame_changed.connect(self._refresh_profile)
        state.band_changed.connect(self._refresh_profile)
        state.units_changed.connect(self._refresh_profile)
        state.row_changed.connect(self._on_row_changed)

        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("button_release_event", self._on_release)

    def _build(self, *_) -> None:
        if self._defer_until_visible(self._build):
            return
        if not self._state.has_dataset:
            self._line = None
            self._marker = None
            self._show_empty("No dataset attached")
            return
        colors = panel_colors()
        self._ax.clear()
        self._style_ax()

        x = self._state.spatial_x_axis()
        y = self._state.get_spatial_profile(self._state.band)
        (self._line,) = self._ax.plot(x, y, color=colors["accent"], linewidth=1)

        xr = x[self._state.row]
        self._marker = self._ax.axvline(xr, color="red", linewidth=1.5)

        xlabel = "Row (mm)" if self._state.x_spatial_unit == "mm" else "Row (pixel)"
        self._ax.set_xlabel(xlabel, color=colors["fg"], fontsize=8)
        self._ax.set_ylabel(self._state.y_unit_effective, color=colors["fg"], fontsize=8)
        self._draw_idle()

    def _refresh_profile(self, *_) -> None:
        if self._defer_until_visible(self._refresh_profile):
            return
        if self._line is None or self._marker is None:
            self._build()
            return
        colors = panel_colors()
        x = self._state.spatial_x_axis()
        y = self._state.get_spatial_profile(self._state.band)
        self._line.set_data(x, y)
        self._line.set_color(colors["accent"])
        self._marker.set_xdata([x[self._state.row], x[self._state.row]])
        xlabel = "Row (mm)" if self._state.x_spatial_unit == "mm" else "Row (pixel)"
        self._ax.set_xlabel(xlabel, color=colors["fg"], fontsize=8)
        self._ax.set_ylabel(self._state.y_unit_effective, color=colors["fg"], fontsize=8)
        self._ax.relim()
        self._ax.autoscale_view()
        self._draw_idle()

    def restyle_theme(self) -> None:
        super().restyle_theme()
        self._refresh_profile()

    def plot_table(self) -> PlotTable | None:
        if self._line is None:
            return None
        x, y = self._line.get_data()
        position = ("row_mm", "mm") if self._state.x_spatial_unit == "mm" else ("row", "")
        return PlotTable(columns=(PlotColumn(position[0], np.asarray(x), position[1]),
                                  PlotColumn("value", np.asarray(y),
                                             self._state.y_unit_effective)),
                         metadata=self._metadata("frame", "band"))

    def _on_row_changed(self, r: int) -> None:
        if self._marker is not None:
            x = self._state.spatial_x_axis()
            self._marker.set_xdata([x[r], x[r]])
            self._draw_idle()

    def _on_press(self, event) -> None:
        if not self._state.has_dataset:
            return
        if event.inaxes != self._ax:
            return
        if self._drag_cid is not None:
            self._canvas.mpl_disconnect(self._drag_cid)
        self._drag_cid = self._canvas.mpl_connect("motion_notify_event", self._on_drag)

    def _on_drag(self, event) -> None:
        if event.inaxes != self._ax or event.xdata is None:
            return
        x = self._state.spatial_x_axis()
        r = int(np.argmin(np.abs(x - event.xdata)))
        r = int(np.clip(r, 0, self._state.height - 1))
        self.row_pick.emit(r)

    def _on_release(self, _event) -> None:
        if self._drag_cid is not None:
            self._canvas.mpl_disconnect(self._drag_cid)
            self._drag_cid = None


# ------------------------------------------------------------------
# Spectrum panel
# ------------------------------------------------------------------


class SpectrumPanel(_BasePanelWidget):
    """Spectrum: get_spectrum(row) with a draggable band marker.

    Emits
    -----
    band_pick(int): user dragged or clicked the marker to a new band
    """

    band_pick = pyqtSignal(int)
    plot_title = "Spectral Profile"

    def __init__(self, state: ViewerState) -> None:
        super().__init__(state, defer_initial_hidden_refresh=False)
        self._line = None
        self._marker = None
        self._drag_cid: int | None = None

        self._build()

        state.frame_changed.connect(self._refresh_spectrum)
        state.row_changed.connect(self._refresh_spectrum)
        state.units_changed.connect(self._refresh_spectrum)
        state.band_changed.connect(self._on_band_changed)

        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("button_release_event", self._on_release)

    def _build(self, *_) -> None:
        if self._defer_until_visible(self._build):
            return
        if not self._state.has_dataset:
            self._line = None
            self._marker = None
            self._show_empty("No dataset attached")
            return
        colors = panel_colors()
        self._ax.clear()
        self._style_ax()

        x = self._state.spectral_x_axis()
        y = self._state.get_spectrum(self._state.row)
        (self._line,) = self._ax.plot(x, y, color=colors["accent"], linewidth=1)

        xb = x[self._state.band]
        self._marker = self._ax.axvline(xb, color="red", linewidth=1.5)

        xlabel = "Wavelength (nm)" if self._state.x_spectral_unit == "nm" else "Band"
        self._ax.set_xlabel(xlabel, color=colors["fg"], fontsize=8)
        self._ax.set_ylabel(self._state.y_unit_effective, color=colors["fg"], fontsize=8)
        self._draw_idle()

    def _refresh_spectrum(self, *_) -> None:
        if self._defer_until_visible(self._refresh_spectrum):
            return
        if self._line is None or self._marker is None:
            self._build()
            return
        colors = panel_colors()
        x = self._state.spectral_x_axis()
        y = self._state.get_spectrum(self._state.row)
        self._line.set_data(x, y)
        self._line.set_color(colors["accent"])
        self._marker.set_xdata([x[self._state.band], x[self._state.band]])
        xlabel = "Wavelength (nm)" if self._state.x_spectral_unit == "nm" else "Band"
        self._ax.set_xlabel(xlabel, color=colors["fg"], fontsize=8)
        self._ax.set_ylabel(self._state.y_unit_effective, color=colors["fg"], fontsize=8)
        self._ax.relim()
        self._ax.autoscale_view()
        self._draw_idle()

    def restyle_theme(self) -> None:
        super().restyle_theme()
        self._refresh_spectrum()

    def plot_table(self) -> PlotTable | None:
        if self._line is None:
            return None
        x, y = self._line.get_data()
        position = (("wavelength", "nm") if self._state.x_spectral_unit == "nm"
                    else ("band", ""))
        return PlotTable(columns=(PlotColumn(position[0], np.asarray(x), position[1]),
                                  PlotColumn("value", np.asarray(y),
                                             self._state.y_unit_effective)),
                         metadata=self._metadata("frame", "row"))

    def _on_band_changed(self, b: int) -> None:
        if self._marker is not None:
            x = self._state.spectral_x_axis()
            self._marker.set_xdata([x[b], x[b]])
            self._draw_idle()

    def _on_press(self, event) -> None:
        if not self._state.has_dataset:
            return
        if event.inaxes != self._ax:
            return
        if self._drag_cid is not None:
            self._canvas.mpl_disconnect(self._drag_cid)
        self._drag_cid = self._canvas.mpl_connect("motion_notify_event", self._on_drag)

    def _on_drag(self, event) -> None:
        if event.inaxes != self._ax or event.xdata is None:
            return
        x = self._state.spectral_x_axis()
        b = int(np.argmin(np.abs(x - event.xdata)))
        b = int(np.clip(b, 0, self._state.n_bands - 1))
        self.band_pick.emit(b)

    def _on_release(self, _event) -> None:
        if self._drag_cid is not None:
            self._canvas.mpl_disconnect(self._drag_cid)
            self._drag_cid = None
