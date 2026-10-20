"""Tests for enginir.viewer._canvas — pyqtgraph FalseColourCanvas."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QWidget


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestFalseColourCanvas:
    def test_is_qwidget(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        assert isinstance(canvas, QWidget)

    def test_has_clicked_signal(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        assert hasattr(canvas, "clicked")

    def test_set_image(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        rgb = np.random.default_rng(0).random((10, 20, 3)).astype(np.float32)
        canvas.set_image(rgb)
        assert canvas._image_item.image is not None
        # row-major: stored as (rows, cols, channels)
        assert canvas._image_item.image.shape[:2] == (10, 20)

    def test_set_image_rejects_non_rgb_shape(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        with pytest.raises(ValueError, match="expects RGB shape"):
            canvas.set_image(np.zeros((10, 20), dtype=np.float32))

    def test_spatial_canvas_exposes_axes_and_labels(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()

        bottom_axis = canvas._plot.getAxis("bottom")
        left_axis = canvas._plot.getAxis("left")

        assert bottom_axis.isVisible()
        assert left_axis.isVisible()
        assert bottom_axis.label.toPlainText().strip() == "Frame"
        assert left_axis.label.toPlainText().strip() == "Row"

    def test_set_crosshair_visible(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        canvas.set_crosshair(5, 3)
        assert canvas._vline.isVisible()
        assert canvas._hline.isVisible()

    def test_set_crosshair_position(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        canvas.set_crosshair(7, 4)
        assert canvas._vline.value() == 7
        assert canvas._hline.value() == 4

    def test_crosshair_hidden_initially(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        assert not canvas._vline.isVisible()
        assert not canvas._hline.isVisible()

    def test_reset_view_restores_full_image_range(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        rgb = np.random.default_rng(0).random((10, 20, 3)).astype(np.float32)
        canvas.set_image(rgb)
        canvas._view.setRange(xRange=(2, 4), yRange=(3, 5), padding=0.0)
        before = canvas._view.viewRange()
        canvas.reset_view()
        after = canvas._view.viewRange()
        assert after[0][1] - after[0][0] > before[0][1] - before[0][0]
        assert after[1][1] - after[1][0] > before[1][1] - before[1][0]

    def test_enter_roi_draw_mode_disables_pan_and_zoom(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        canvas.enter_roi_draw_mode()

        assert canvas._interaction_mode == "draw_roi"
        assert tuple(canvas._view.state["mouseEnabled"]) == (False, False)

    def test_escape_cancels_roi_draw_mode_and_restores_navigation(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        canceled = []
        canvas.roi_draw_canceled.connect(lambda: canceled.append(True))

        canvas.enter_roi_draw_mode()
        canvas._begin_roi_draw(2, 3)
        assert canvas._roi_preview_item is not None

        canvas.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        )

        assert canvas._interaction_mode == "navigate"
        assert canvas._roi_preview_item is None
        assert tuple(canvas._view.state["mouseEnabled"]) == (True, True)
        assert canceled == [True]

    def test_finish_roi_draw_emits_normalized_clamped_bounds(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        rgb = np.random.default_rng(0).random((10, 20, 3)).astype(np.float32)
        canvas.set_image(rgb)
        bounds = []
        canvas.roi_draw_finished.connect(lambda f0, f1, r0, r1: bounds.append((f0, f1, r0, r1)))

        canvas.enter_roi_draw_mode()
        canvas._begin_roi_draw(18, 8)
        canvas._update_roi_draw_preview(3, 1)
        canvas._finish_roi_draw(-4, 99)

        assert bounds == [(0, 18, 8, 9)]
        assert canvas._interaction_mode == "navigate"
        assert canvas._roi_preview_item is None
        assert tuple(canvas._view.state["mouseEnabled"]) == (True, True)

    def test_finish_roi_draw_rejects_single_pixel_roi(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        rgb = np.random.default_rng(0).random((10, 20, 3)).astype(np.float32)
        canvas.set_image(rgb)
        bounds = []
        canceled = []
        canvas.roi_draw_finished.connect(lambda f0, f1, r0, r1: bounds.append((f0, f1, r0, r1)))
        canvas.roi_draw_canceled.connect(lambda: canceled.append(True))

        canvas.enter_roi_draw_mode()
        canvas._begin_roi_draw(5, 6)
        canvas._finish_roi_draw(5, 6)

        assert bounds == []
        assert canceled == [True]
        assert canvas._interaction_mode == "navigate"
        assert canvas._roi_preview_item is None
        assert tuple(canvas._view.state["mouseEnabled"]) == (True, True)

    @pytest.mark.parametrize("finish_point", [(5, 7), (6, 6)])
    def test_finish_roi_draw_requires_at_least_2_by_2(self, qapp, finish_point):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        rgb = np.random.default_rng(0).random((10, 20, 3)).astype(np.float32)
        canvas.set_image(rgb)
        bounds = []
        canceled = []
        canvas.roi_draw_finished.connect(lambda f0, f1, r0, r1: bounds.append((f0, f1, r0, r1)))
        canvas.roi_draw_canceled.connect(lambda: canceled.append(True))

        canvas.enter_roi_draw_mode()
        canvas._begin_roi_draw(5, 6)
        canvas._finish_roi_draw(*finish_point)

        assert bounds == []
        assert canceled == [True]
        assert canvas._interaction_mode == "navigate"
        assert canvas._roi_preview_item is None
        assert tuple(canvas._view.state["mouseEnabled"]) == (True, True)

    def test_select_roi_updates_selected_id_and_overlay_style(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        canvas.add_roi_rect(1, 3, 2, 4, "roi_a", "roi_1")
        canvas.select_roi("roi_1")

        assert canvas._selected_roi_id == "roi_1"
        rect = canvas._roi_item_map["roi_1"].rect
        pen = rect.pen
        assert pen.color().name().lower() == "#ffff00"

    def test_roi_id_at_point_returns_matching_roi(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        canvas.add_roi_rect(1, 3, 2, 4, "roi_a", "roi_1")
        canvas.add_roi_rect(6, 8, 1, 2, "roi_b", "roi_2")

        assert canvas._roi_id_at_point(2, 3) == "roi_1"
        assert canvas._roi_id_at_point(7, 2) == "roi_2"
        assert canvas._roi_id_at_point(0, 0) is None

    def test_add_roi_rect_registers_overlay_items(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        rgb = np.zeros((20, 30, 3), dtype=np.float32)
        canvas.set_image(rgb)

        canvas.add_roi_rect(2, 5, 1, 4, "grain")

        assert len(canvas._roi_items) > 0

    def test_clear_roi_rects_removes_registered_overlays(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        rgb = np.zeros((20, 30, 3), dtype=np.float32)
        canvas.set_image(rgb)

        canvas.add_roi_rect(0, 3, 0, 3, "A")
        canvas.add_roi_rect(5, 8, 1, 4, "B")
        canvas.clear_roi_rects()

        assert canvas._roi_items == []

    def test_add_roi_rect_tracks_items_by_roi_id(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()
        rgb = np.zeros((20, 30, 3), dtype=np.float32)
        canvas.set_image(rgb)

        canvas.add_roi_rect(0, 3, 0, 2, "grain", "roi_a")
        canvas.add_roi_rect(5, 8, 0, 2, "grain", "roi_b")

        assert set(canvas._roi_item_map) == {"roi_a", "roi_b"}
        assert canvas._roi_item_map["roi_a"] != canvas._roi_item_map["roi_b"]

    def test_clear_roi_rects_is_idempotent_when_empty(self, qapp):
        from enginir.viewer._canvas import FalseColourCanvas

        canvas = FalseColourCanvas()

        canvas.clear_roi_rects()
        canvas.clear_roi_rects()

        assert canvas._roi_items == []
