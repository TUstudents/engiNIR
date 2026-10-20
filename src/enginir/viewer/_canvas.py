"""FalseColourCanvas — pyqtgraph-based central image widget.

Wraps a pyqtgraph ``ImageItem`` in a plot with visible axes.
Provides pan/zoom, crosshair overlay, ROI interaction, and click-to-navigate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from pyqtgraph.GraphicsScene import GraphicsScene
from pyqtgraph.graphicsItems.PlotItem import PlotItem
from pyqtgraph.graphicsItems.ViewBox.ViewBox import ViewBox
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeyEvent
from PyQt6.QtWidgets import QGraphicsSceneMouseEvent

from .._checks import log_and_raise
from .._logging import get_logger
from ._theme import panel_colors

# Use numpy (row, col) axis convention for ImageItem data.
pg.setConfigOption("imageAxisOrder", "row-major")

__all__ = ["FalseColourCanvas"]
logger = get_logger(__name__)


@dataclass(slots=True)
class _RoiOverlay:
    rect: pg.RectROI
    text: pg.TextItem
    bounds: tuple[int, int, int, int]


class FalseColourCanvas(QWidget):
    """Central image widget showing a false-colour RGB composite.

    Signals
    -------
    clicked(int, int)
        Emitted on left-click inside the image.  Arguments: ``(frame, row)``,
        i.e. ``(column_index, row_index)`` in image coordinates.
    """

    clicked = pyqtSignal(int, int)
    roi_draw_finished = pyqtSignal(int, int, int, int)
    roi_draw_canceled = pyqtSignal()
    roi_selected = pyqtSignal(str)
    roi_selection_cleared = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._gw = pg.PlotWidget()
        plot = self._gw.getPlotItem()
        if not isinstance(plot, PlotItem):
            log_and_raise("PyQtGraph plot item unavailable for false-colour canvas",
                          error=RuntimeError)
        self._plot = plot
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.showAxis("bottom")
        self._plot.showAxis("left")
        self._plot.hideAxis("top")
        self._plot.hideAxis("right")

        self._view: ViewBox = self._plot.getViewBox()
        self._view.invertY(True)
        self._view.setAspectLocked(False)

        self._image_item = pg.ImageItem()
        self._plot.addItem(self._image_item)

        # Crosshair: vertical line marks frame (x), horizontal marks row (y).
        pen_y = pg.mkPen("yellow", width=1)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pen_y)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pen_y)
        self._plot.addItem(self._vline)
        self._plot.addItem(self._hline)
        self._vline.setVisible(False)
        self._hline.setVisible(False)

        self._roi_items: list[pg.GraphicsObject] = []
        self._roi_item_map: dict[str, _RoiOverlay] = {}
        # Stored as (height, width) to match the row-major ImageItem convention.
        self._image_shape: tuple[int, int] | None = None
        self._interaction_mode = "navigate"
        self._roi_draw_start: tuple[int, int] | None = None
        self._roi_preview_item: pg.RectROI | None = None
        self._ignore_next_click = False
        self._selected_roi_id: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._gw)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.restyle_theme()

        # Connect scene click (after layout so scene exists).
        scene = self._gw.scene()
        if not isinstance(scene, GraphicsScene):
            log_and_raise("PyQtGraph scene unavailable for false-colour canvas", error=RuntimeError)
        self._scene = scene
        self._scene.sigMouseClicked.connect(self._on_click)
        self._scene.installEventFilter(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, rgb: np.ndarray) -> None:
        """Display an RGB image.

        Parameters
        ----------
        rgb : ndarray, shape (height, width, 3), float32 in [0, 1].
        """
        rgb_arr = np.asarray(rgb)
        if rgb_arr.size == 0:
            self.clear_image()
            return
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
            log_and_raise("FalseColourCanvas.set_image expects RGB shape (height, width, 3), "
                          "got {}", rgb_arr.shape)
        img = (np.clip(rgb_arr, 0.0, 1.0) * 255).astype(np.uint8)
        self._image_item.setImage(img)
        self._image_shape = (int(img.shape[0]), int(img.shape[1]))
        self._view.autoRange()

    def clear_image(self) -> None:
        """Clear the current image and disable image-space interaction."""
        self._image_item.clear()
        self._image_shape = None
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._clear_roi_preview()

    def set_crosshair(self, frame: int, row: int) -> None:
        """Move crosshair to *(frame, row)* and make it visible."""
        self._vline.setPos(frame)
        self._hline.setPos(row)
        self._vline.setVisible(True)
        self._hline.setVisible(True)

    def reset_view(self) -> None:
        """Reset zoom and pan so the full image is visible again."""
        self._view.autoRange(padding=0.02)

    def enter_roi_draw_mode(self) -> None:
        """Enter one-shot ROI drawing mode on the spatial image."""
        self._interaction_mode = "draw_roi"
        self._roi_draw_start = None
        self._clear_roi_preview()
        self._view.setMouseEnabled(x=False, y=False)
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def cancel_roi_draw_mode(self) -> None:
        """Cancel the current ROI drawing gesture and restore navigation."""
        if self._interaction_mode != "draw_roi":
            return
        self._roi_draw_start = None
        self._clear_roi_preview()
        self._interaction_mode = "navigate"
        self._view.setMouseEnabled(x=True, y=True)
        self._ignore_next_click = True
        self.roi_draw_canceled.emit()

    def add_roi_rect(
        self,
        f0: int,
        f1: int,
        r0: int,
        r1: int,
        label: str,
        roi_id: str | None = None,
    ) -> None:
        """Draw a non-interactive magenta rectangle with a text label."""
        is_selected = roi_id is not None and roi_id == self._selected_roi_id
        pen = self._roi_pen(selected=is_selected)
        rect = pg.RectROI(
            pos=(f0, r0),
            size=(f1 - f0 + 1, r1 - r0 + 1),
            movable=False,
            resizable=False,
            pen=pen,
        )
        for handle in rect.getHandles():
            rect.removeHandle(handle)
        text = pg.TextItem(text=label, color=self._roi_color(selected=is_selected), anchor=(0, 0))
        text.setPos(f0, r0)
        self._plot.addItem(rect)
        self._plot.addItem(text)
        items = [rect, text]
        self._roi_items.extend(items)
        if roi_id is not None:
            self._roi_item_map[roi_id] = _RoiOverlay(
                rect=rect,
                text=text,
                bounds=(f0, f1, r0, r1),
            )

    def clear_roi_rects(self) -> None:
        """Remove all ROI overlays."""
        for item in self._roi_items:
            self._plot.removeItem(item)
        self._roi_items = []
        self._roi_item_map = {}

    def select_roi(self, roi_id: str | None) -> None:
        """Select one persisted ROI overlay by id."""
        self._selected_roi_id = roi_id if roi_id in self._roi_item_map else None
        for existing_roi_id, entry in self._roi_item_map.items():
            selected = existing_roi_id == self._selected_roi_id
            rect = entry.rect
            text = entry.text
            rect.setPen(self._roi_pen(selected=selected))
            text.setColor(self._roi_color(selected=selected))

    def restyle_theme(self) -> None:
        """Apply the current app theme to the pyqtgraph canvas chrome."""
        colors = panel_colors()
        self._gw.setBackground(colors["bg"])
        self._plot.setLabel("bottom", "Frame", color=colors["fg"])
        self._plot.setLabel("left", "Row", color=colors["fg"])
        axis_color = QColor(colors["fg"])
        axis_color.setAlpha(160)
        axis_pen = pg.mkPen(axis_color)
        text_pen = pg.mkPen(colors["fg"])
        self._plot.getAxis("bottom").setPen(axis_pen)
        self._plot.getAxis("bottom").setTextPen(text_pen)
        self._plot.getAxis("left").setPen(axis_pen)
        self._plot.getAxis("left").setTextPen(text_pen)

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if (
            a0 is not None
            and self._interaction_mode == "draw_roi"
            and a0.key() == Qt.Key.Key_Escape
        ):
            self.cancel_roi_draw_mode()
            a0.accept()
            return
        super().keyPressEvent(a0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is None:
            return super().eventFilter(a0, a1)
        if a0 is self._scene and self._interaction_mode == "draw_roi":
            event_type = a1.type()
            if event_type == QEvent.Type.GraphicsSceneMousePress:
                if not isinstance(a1, QGraphicsSceneMouseEvent):
                    return False
                if a1.button() == Qt.MouseButton.RightButton:
                    self.cancel_roi_draw_mode()
                    a1.accept()
                    return True
                if a1.button() == Qt.MouseButton.LeftButton:
                    image_point = self._scene_pos_to_image(a1.scenePos())
                    if image_point is None:
                        return False
                    self._begin_roi_draw(*image_point)
                    a1.accept()
                    return True
            elif event_type == QEvent.Type.GraphicsSceneMouseMove:
                if not isinstance(a1, QGraphicsSceneMouseEvent):
                    return False
                if self._roi_draw_start is not None:
                    image_point = self._scene_pos_to_image(a1.scenePos())
                    if image_point is not None:
                        self._update_roi_draw_preview(*image_point)
                        a1.accept()
                        return True
            elif event_type == QEvent.Type.GraphicsSceneMouseRelease:
                if not isinstance(a1, QGraphicsSceneMouseEvent):
                    return False
                if a1.button() == Qt.MouseButton.LeftButton and self._roi_draw_start is not None:
                    image_point = self._scene_pos_to_image(a1.scenePos())
                    if image_point is not None:
                        self._finish_roi_draw(*image_point)
                    else:
                        self.cancel_roi_draw_mode()
                    a1.accept()
                    return True
        return super().eventFilter(a0, a1)

    def _on_click(self, event) -> None:
        if self._ignore_next_click:
            self._ignore_next_click = False
            return
        if self._interaction_mode != "navigate":
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        image_point = self._scene_pos_to_image(event.scenePos())
        if image_point is None:
            return
        frame, row = image_point
        roi_id = self._roi_id_at_point(frame, row)
        if roi_id is not None:
            self.select_roi(roi_id)
            self.roi_selected.emit(roi_id)
            return
        if self._selected_roi_id is not None:
            self.select_roi(None)
            self.roi_selection_cleared.emit()
        self.clicked.emit(frame, row)

    def _scene_pos_to_image(self, pos) -> tuple[int, int] | None:
        if self._image_shape is None:
            return None
        if not self._view.sceneBoundingRect().contains(pos):
            return None
        mouse_point = self._view.mapSceneToView(pos)
        frame = int(round(mouse_point.x()))
        row = int(round(mouse_point.y()))
        return self._clamp_point(frame, row)

    def _clamp_point(self, frame: int, row: int) -> tuple[int, int]:
        if self._image_shape is None:
            return 0, 0
        height, width = self._image_shape
        frame = int(np.clip(frame, 0, width - 1))
        row = int(np.clip(row, 0, height - 1))
        return frame, row

    def _begin_roi_draw(self, frame: int, row: int) -> None:
        frame, row = self._clamp_point(frame, row)
        self._roi_draw_start = (frame, row)
        self._set_roi_preview(frame, frame, row, row)

    def _update_roi_draw_preview(self, frame: int, row: int) -> None:
        if self._roi_draw_start is None:
            return
        frame, row = self._clamp_point(frame, row)
        f0, f1, r0, r1 = self._normalized_bounds(
            self._roi_draw_start[0],
            frame,
            self._roi_draw_start[1],
            row,
        )
        self._set_roi_preview(f0, f1, r0, r1)

    def _finish_roi_draw(self, frame: int, row: int) -> None:
        if self._roi_draw_start is None:
            return
        frame, row = self._clamp_point(frame, row)
        f0, f1, r0, r1 = self._normalized_bounds(
            self._roi_draw_start[0],
            frame,
            self._roi_draw_start[1],
            row,
        )
        width = f1 - f0 + 1
        height = r1 - r0 + 1
        if width < 2 or height < 2:
            self.cancel_roi_draw_mode()
            return
        self._roi_draw_start = None
        self._clear_roi_preview()
        self._interaction_mode = "navigate"
        self._view.setMouseEnabled(x=True, y=True)
        self._ignore_next_click = True
        self.roi_draw_finished.emit(f0, f1, r0, r1)

    def _normalized_bounds(self, f0: int, f1: int, r0: int, r1: int) -> tuple[int, int, int, int]:
        left, right = sorted((int(f0), int(f1)))
        top, bottom = sorted((int(r0), int(r1)))
        left, top = self._clamp_point(left, top)
        right, bottom = self._clamp_point(right, bottom)
        return left, right, top, bottom

    def _set_roi_preview(self, f0: int, f1: int, r0: int, r1: int) -> None:
        left, right, top, bottom = int(f0), int(f1), int(r0), int(r1)
        width = max(1, right - left + 1)
        height = max(1, bottom - top + 1)
        if self._roi_preview_item is None:
            self._roi_preview_item = pg.RectROI(
                pos=(left, top),
                size=(width, height),
                movable=False,
                resizable=False,
                pen=pg.mkPen(color=(255, 255, 0), width=1.5),
            )
            for handle in self._roi_preview_item.getHandles():
                self._roi_preview_item.removeHandle(handle)
            self._plot.addItem(self._roi_preview_item)
            return
        self._roi_preview_item.setPos((left, top))
        self._roi_preview_item.setSize((width, height))

    def _clear_roi_preview(self) -> None:
        if self._roi_preview_item is None:
            return
        self._plot.removeItem(self._roi_preview_item)
        self._roi_preview_item = None

    def _roi_id_at_point(self, frame: int, row: int) -> str | None:
        for roi_id, entry in reversed(list(self._roi_item_map.items())):
            f0, f1, r0, r1 = entry.bounds
            if f0 <= frame <= f1 and r0 <= row <= r1:
                return roi_id
        return None

    def _roi_pen(self, *, selected: bool) -> pg.QtGui.QPen:
        return pg.mkPen(color=self._roi_color(selected=selected), width=2.0 if selected else 1.5)

    def _roi_color(self, *, selected: bool) -> tuple[int, int, int]:
        return (255, 255, 0) if selected else (255, 0, 255)
