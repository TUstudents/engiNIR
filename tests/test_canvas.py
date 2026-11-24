"""The spatial image canvas (``enginir.viewer._canvas.FalseColorCanvas``)."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent

from enginir.viewer._canvas import FalseColorCanvas
from enginir.viewer._theme import tokens


@pytest.fixture()
def canvas(qapp):
    canvas = FalseColorCanvas()
    canvas.set_image(np.random.default_rng(0).random((10, 20, 3)).astype(np.float32))
    return canvas


def _navigates(canvas) -> bool:
    return (canvas._interaction_mode == "navigate" and canvas._roi_preview_item is None
            and tuple(canvas._view.state["mouseEnabled"]) == (True, True))


def test_the_image_is_rows_by_frames_with_named_axes(canvas):
    assert canvas._image_item.image.shape[:2] == (10, 20)
    assert canvas._plot.getAxis("bottom").label.toPlainText().strip() == "Frame"
    assert canvas._plot.getAxis("left").label.toPlainText().strip() == "Row"
    with pytest.raises(ValueError, match="expects RGB shape"):
        canvas.set_image(np.zeros((10, 20), dtype=np.float32))


def test_drawing_an_roi_emits_clamped_bounds_and_escape_cancels(canvas):
    bounds, canceled = [], []
    canvas.roi_draw_finished.connect(lambda *b: bounds.append(b))
    canvas.roi_draw_canceled.connect(lambda: canceled.append(True))

    canvas.enter_roi_draw_mode()
    assert canvas._interaction_mode == "draw_roi"
    assert tuple(canvas._view.state["mouseEnabled"]) == (False, False)
    canvas._begin_roi_draw(18, 8)
    canvas._update_roi_draw_preview(3, 1)
    canvas._finish_roi_draw(-4, 99)
    assert bounds == [(0, 18, 8, 9)] and _navigates(canvas)

    canvas.enter_roi_draw_mode()
    canvas._begin_roi_draw(2, 3)
    assert canvas._roi_preview_item is not None
    canvas.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert canceled == [True] and _navigates(canvas)


@pytest.mark.parametrize("finish_point", [(5, 6), (5, 7), (6, 6)])
def test_an_roi_smaller_than_2_by_2_is_canceled(canvas, finish_point):
    bounds, canceled = [], []
    canvas.roi_draw_finished.connect(lambda *b: bounds.append(b))
    canvas.roi_draw_canceled.connect(lambda: canceled.append(True))
    canvas.enter_roi_draw_mode()
    canvas._begin_roi_draw(5, 6)
    canvas._finish_roi_draw(*finish_point)
    assert bounds == [] and canceled == [True] and _navigates(canvas)


def test_roi_overlays_are_tracked_selected_hit_and_cleared(canvas):
    canvas.clear_roi_rects()  # nothing to clear: no error
    canvas.add_roi_rect(1, 3, 2, 4, "roi_a", "roi_1")
    canvas.add_roi_rect(6, 8, 1, 2, "roi_b", "roi_2")
    assert set(canvas._roi_item_map) == {"roi_1", "roi_2"}

    canvas.select_roi("roi_1")
    assert canvas._selected_roi_id == "roi_1"
    assert canvas._roi_item_map["roi_1"].rect.pen.color().name().lower() == tokens()["attention"]
    assert canvas._roi_item_map["roi_2"].rect.pen.color().name().lower() == tokens()["accent"]

    assert canvas._roi_id_at_point(2, 3) == "roi_1"
    assert canvas._roi_id_at_point(7, 2) == "roi_2"
    assert canvas._roi_id_at_point(0, 0) is None

    canvas.clear_roi_rects()
    assert canvas._roi_items == []


def test_an_image_opens_fitted_as_reset_view_fits_it(qapp):
    """Shown while its view is small (a page not laid out yet), an image used to get
    pyqtgraph's size-dependent padding (up to 10 %); Reset View uses 2 %."""
    from enginir.viewer._canvas import FalseColorCanvas

    canvas = FalseColorCanvas()
    canvas.resize(60, 60)
    canvas.show()
    qapp.processEvents()
    try:
        canvas.set_image(np.random.default_rng(0).random((40, 80, 3)))
        canvas.resize(900, 600)
        qapp.processEvents()
        shown = np.array(canvas.view_box.viewRange())
        canvas.reset_view()
        np.testing.assert_allclose(shown, np.array(canvas.view_box.viewRange()), atol=1e-6)
    finally:
        canvas.close()
