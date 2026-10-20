"""Five matplotlib panel widgets for the HSI viewer.

Each widget is a QWidget containing a matplotlib Figure embedded via
FigureCanvasQTAgg. Panels are driven by ``ViewerState``; the raw-band panel
intentionally uses explicit raw-data accessors to show unprocessed cube data.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QShowEvent

from .._logging import get_logger
from ._state import ViewerState
from ._theme import panel_colors
from ._ui_helpers import maybe_tight_layout, style_axes

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


class _BasePanelWidget(QWidget):
    """QWidget with an embedded matplotlib figure on a dark background.

    Hidden dock panels defer expensive rebuilds until they are shown again.
    Panels that are visible by default in the Explore workspace can opt out of
    deferring their very first hidden build so guide lines and labels exist
    immediately after viewer construction.
    """

    def __init__(self, state: ViewerState, *, defer_initial_hidden_refresh: bool = True) -> None:
        super().__init__()
        self._state = state
        self._fig = Figure(figsize=(4, 3), facecolor=panel_colors()["bg"])
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._pending_visible_refresh: Callable[[], None] | None = None
        self._last_layout_size: tuple[int, int] | None = None
        self._defer_initial_hidden_refresh = defer_initial_hidden_refresh
        self._has_drawn_once = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)

    def _style_ax(self) -> None:
        """Apply the current theme styling to the current axes."""
        style_axes(self._ax)

    def _show_empty(self, title: str, message: str) -> None:
        colors = panel_colors()
        self._ax.clear()
        self._style_ax()
        self._ax.set_title(title, color=colors["fg"], fontsize=9)
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

    def _defer_until_visible(self, callback: Callable[[], None]) -> bool:
        """Return True when the caller should defer work until the panel is shown."""
        if self.isVisible():
            return False
        if not self._defer_initial_hidden_refresh and not self._has_drawn_once:
            return False
        self._pending_visible_refresh = callback
        return True

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

    def showEvent(self, a0: QShowEvent | None) -> None:
        super().showEvent(a0)
        pending = self._pending_visible_refresh
        self._pending_visible_refresh = None
        if pending is not None:
            pending()


# ------------------------------------------------------------------
# Histogram panel
# ------------------------------------------------------------------


class HistogramPanel(_BasePanelWidget):
    """256-bin histogram of the current frame image (top-left dock)."""

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
            self._show_empty("Histogram", "No dataset attached")
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
        self._ax.set_title("Histogram", color=colors["fg"], fontsize=9)
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
        self._ax.relim()
        self._ax.autoscale_view()
        self._draw_idle()

    def restyle_theme(self) -> None:
        super().restyle_theme()
        self._build()


# ------------------------------------------------------------------
# Raw panel
# ------------------------------------------------------------------


class DC3Panel(_BasePanelWidget):
    """Raw band view: cube[:, :, band].T — height × n_frames with yellow crosshairs.

    Emits
    -----
    frame_pick(int): user clicked a new frame position
    row_pick(int):   user clicked a new row position
    """

    frame_pick = pyqtSignal(int)
    row_pick = pyqtSignal(int)

    def __init__(self, state: ViewerState) -> None:
        super().__init__(state)
        self._img_handle = None
        self._vline = None
        self._hline = None

        self._build()

        state.frame_changed.connect(self._on_frame_changed)
        state.row_changed.connect(self._on_row_changed)
        state.band_changed.connect(self._on_band_changed)
        self._canvas.mpl_connect("button_press_event", self._on_click)

    def _build(self, *_) -> None:
        if self._defer_until_visible(self._build):
            return
        if not self._state.has_dataset:
            self._img_handle = None
            self._vline = None
            self._hline = None
            self._show_empty("RAW", "No dataset attached")
            return
        colors = panel_colors()
        self._ax.clear()
        self._style_ax()

        b = self._state.band
        data = self._state.raw_band_image(b)
        self._img_handle = self._ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            cmap="inferno",
            interpolation="nearest",
        )
        self._vline = self._ax.axvline(self._state.frame, color="yellow", linewidth=0.8, alpha=0.9)
        self._hline = self._ax.axhline(self._state.row, color="yellow", linewidth=0.8, alpha=0.9)
        wl = self._state.wavelengths[b]
        self._ax.set_title(f"RAW  band {b} ({wl:.1f} nm)", color=colors["fg"], fontsize=9)
        self._ax.set_xlabel("Frame", color=colors["fg"], fontsize=8)
        self._ax.set_ylabel("Row", color=colors["fg"], fontsize=8)

        self._draw_idle()

    def _refresh_band_view(self) -> None:
        if self._defer_until_visible(self._refresh_band_view):
            return
        if self._img_handle is None or self._vline is None or self._hline is None:
            self._build()
            return
        colors = panel_colors()
        b = self._state.band
        data = self._state.raw_band_image(b)
        self._img_handle.set_data(data)
        self._img_handle.set_clim(vmin=float(np.min(data)), vmax=float(np.max(data)))
        self._vline.set_xdata([self._state.frame, self._state.frame])
        self._hline.set_ydata([self._state.row, self._state.row])
        wl = self._state.wavelengths[b]
        self._ax.set_title(f"RAW  band {b} ({wl:.1f} nm)", color=colors["fg"], fontsize=9)
        self._draw_idle()

    def restyle_theme(self) -> None:
        super().restyle_theme()
        self._refresh_band_view()

    def _on_frame_changed(self, f: int) -> None:
        if self._vline is not None:
            self._vline.set_xdata([f, f])
            self._draw_idle()

    def _on_row_changed(self, r: int) -> None:
        if self._hline is not None:
            self._hline.set_ydata([r, r])
            self._draw_idle()

    def _on_band_changed(self, _: int) -> None:
        self._refresh_band_view()

    def _on_click(self, event) -> None:
        if not self._state.has_dataset:
            return
        if event.inaxes != self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        f = int(np.clip(round(event.xdata), 0, self._state.n_frames - 1))
        r = int(np.clip(round(event.ydata), 0, self._state.height - 1))
        self.frame_pick.emit(f)
        self.row_pick.emit(r)


# ------------------------------------------------------------------
# Sensor panel
# ------------------------------------------------------------------


class SensorPanel(_BasePanelWidget):
    """Sensor view: current_frame_image() — height × n_bands with crosshairs.

    Emits
    -----
    row_pick(int):  user clicked a new row
    band_pick(int): user clicked a new band
    """

    row_pick = pyqtSignal(int)
    band_pick = pyqtSignal(int)

    def __init__(self, state: ViewerState) -> None:
        super().__init__(state, defer_initial_hidden_refresh=False)
        self._img_handle = None
        self._vline = None
        self._hline = None

        self._build()

        state.frame_changed.connect(self._refresh_frame_view)
        state.row_changed.connect(self._on_row_changed)
        state.band_changed.connect(self._on_band_changed)
        state.units_changed.connect(self._refresh_frame_view)
        self._canvas.mpl_connect("button_press_event", self._on_click)

    def _build(self, *_) -> None:
        if self._defer_until_visible(self._build):
            return
        if not self._state.has_dataset:
            self._img_handle = None
            self._vline = None
            self._hline = None
            self._show_empty("Sensor Frame", "No dataset attached")
            return
        colors = panel_colors()
        self._ax.clear()
        self._style_ax()

        img = self._state.current_frame_image()  # (height, n_bands)
        self._img_handle = self._ax.imshow(
            img,
            aspect="auto",
            origin="upper",
            cmap="inferno",
            interpolation="nearest",
        )
        self._vline = self._ax.axvline(self._state.band, color="yellow", linewidth=0.8, alpha=0.9)
        self._hline = self._ax.axhline(self._state.row, color="yellow", linewidth=0.8, alpha=0.9)
        self._ax.set_xlabel("Band", color=colors["fg"], fontsize=8)
        self._ax.xaxis.label.set_rotation(0)
        self._ax.xaxis.label.set_verticalalignment("center")
        self._ax.xaxis.label.set_horizontalalignment("left")
        self._ax.xaxis.set_label_coords(-0.13, -0.13)
        self._ax.set_ylabel("Cross-track", color=colors["fg"], fontsize=8)

        self._draw_idle()

    def _refresh_frame_view(self, *_) -> None:
        """Update the sensor image in place for a frame or signal-unit change.

        Frame scrubbing and unit toggles change only the image data and its
        value range; the axes, labels, and crosshair geometry are unchanged.
        A full rebuild here regressed Explore responsiveness because frame
        changes fire on every canvas click and DC3 drag.
        """
        if self._defer_until_visible(self._refresh_frame_view):
            return
        if self._img_handle is None or self._vline is None or self._hline is None:
            self._build()
            return
        img = self._state.current_frame_image()  # (height, n_bands)
        self._img_handle.set_data(img)
        self._img_handle.set_clim(vmin=float(np.min(img)), vmax=float(np.max(img)))
        self._draw_idle()

    def restyle_theme(self) -> None:
        super().restyle_theme()
        self._build()

    def _on_row_changed(self, r: int) -> None:
        if self._hline is not None:
            self._hline.set_ydata([r, r])
            self._draw_idle()

    def _on_band_changed(self, b: int) -> None:
        if self._vline is not None:
            self._vline.set_xdata([b, b])
            self._draw_idle()

    def _on_click(self, event) -> None:
        if not self._state.has_dataset:
            return
        if event.inaxes != self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        b = int(np.clip(round(event.xdata), 0, self._state.n_bands - 1))
        r = int(np.clip(round(event.ydata), 0, self._state.height - 1))
        self.band_pick.emit(b)
        self.row_pick.emit(r)


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
            self._show_empty("Spatial Profile", "No dataset attached")
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
            self._show_empty("Spectrum", "No dataset attached")
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
        self._ax.set_title("")
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
        self._ax.set_title("")
        self._ax.relim()
        self._ax.autoscale_view()
        self._draw_idle()

    def restyle_theme(self) -> None:
        super().restyle_theme()
        self._refresh_spectrum()

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
