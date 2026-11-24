"""The public widgets of ``enginir.viewer`` and ``enginir.false_color_image`` for other
applications (viewer contract, Public widgets)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from PyQt6.QtCore import QPointF

import enginir
from enginir import GridCell, false_color_image, roi_grid
from enginir.io import save_hdf5
from enginir.viewer import (
    FalseColorCanvas,
    PredictionMapWidget,
    ViewerState,
    apply_theme,
    heat_colormap,
    theme_tokens,
    value_colormap,
)

from ._scans import framed_scan


def _long_scan(path, *, wavelengths=True) -> tuple[str, np.ndarray, np.ndarray]:
    """300 frames (more than one read block) x 7 rows x 12 bands of uint16 counts."""
    cube = np.random.default_rng(5).integers(1000, 3000, (300, 7, 12)).astype(np.uint16)
    axis = np.linspace(1000, 1700, 12) if wavelengths else np.arange(12, dtype=float)
    save_hdf5(str(path), cube, axis)
    return str(path), cube, axis


def test_the_public_names_are_exported():
    import enginir.viewer as viewer

    for name in ("apply_theme", "theme_tokens", "value_colormap", "heat_colormap",
                 "PredictionMapWidget", "FalseColorCanvas"):
        assert name in viewer.__all__ and getattr(viewer, name) is not None
    assert "false_color_image" in enginir.__all__


# -- false_color_image -----------------------------------------------------------------

def test_the_false_color_image_is_the_viewers_image_of_the_scan(tmp_path):
    path, cube, axis = _long_scan(tmp_path / "scan.h5")
    image = false_color_image(path)
    assert image.shape == (7, 300, 3) and image.dtype == np.float32
    np.testing.assert_array_equal(image, ViewerState(cube, axis).raw_false_colour_rgb())


def test_a_scan_without_wavelengths_needs_its_bands(tmp_path):
    path, cube, _axis = _long_scan(tmp_path / "stream_hsi.h5", wavelengths=False)
    with pytest.raises(ValueError, match="no wavelengths"):
        false_color_image(path)
    with pytest.raises(ValueError, match="three band indices"):
        false_color_image(path, bands=(1, 2, 12))
    image = false_color_image(path, bands=(10, 6, 1))
    by_wavelength = ViewerState(cube, np.array([0, 1050, 0, 0, 0, 0, 1300, 0, 0, 0, 1650, 0.]))
    np.testing.assert_array_equal(image, by_wavelength.raw_false_colour_rgb())


def test_the_scan_is_opened_read_only(tmp_path, monkeypatch):
    path = framed_scan(tmp_path / "scan.h5")
    modes = []
    real_file = h5py.File

    def recording(name, mode="r", *args, **kwargs):
        modes.append(mode)
        return real_file(name, mode, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", recording)
    false_color_image(path)
    assert modes and set(modes) == {"r"}


# -- Theme ------------------------------------------------------------------------------

def test_a_chosen_theme_keeps_the_users_saved_choice(qapp):
    from enginir.viewer._theme import _save_preference, is_dark, load_preference

    _save_preference(False)
    apply_theme(qapp, dark=True)
    assert is_dark() and load_preference() is False
    apply_theme(qapp)
    assert not is_dark()
    assert theme_tokens()["accent"].startswith("#")
    assert value_colormap().N == 256 and heat_colormap() is not None


# -- PredictionMapWidget.show_maps --------------------------------------------------------

def test_maps_given_as_arrays_show_each_layer_with_its_limit(qapp):
    rng = np.random.default_rng(2)
    prediction = rng.normal(5, 1, (30, 8)).astype(np.float32)   # (frames, rows)
    dmodx = rng.random((30, 8)).astype(np.float32)
    widget = PredictionMapWidget()
    widget.show_maps(prediction, dmodx=dmodx, unit="%", dcrit=0.9, color_range=(3.0, 7.0))

    table = widget.plot_table()
    assert table is not None
    np.testing.assert_array_equal(table.matrix, prediction.T)
    assert table.metadata["colour_scale"].startswith("3 .. 7")

    widget.set_layer("DModXPS")
    table = widget.plot_table()
    assert table is not None
    np.testing.assert_array_equal(table.matrix, dmodx.T)
    assert table.metadata["limit_095"] == "0.9"
    assert f"above DCrit 0.9: {int((dmodx > 0.9).sum())} pixels" in widget._stats_label.text()

    widget.set_layer("T²PS")   # not given: the layer says why, no map
    assert widget.plot_table() is None

    widget.set_layer("Prediction")
    widget.clear(empty_reason="No layer yet")
    assert widget.plot_table() is None
    with pytest.raises(ValueError, match="T²PS map has shape"):
        widget.show_maps(prediction, t2=np.zeros((8, 30)))


# -- FalseColorCanvas.show_cells ---------------------------------------------------------

def test_cells_are_outlined_named_and_hovered(qapp):
    cells = roi_grid(n_rows=2, n_columns=3, size=(10, 4), origin=(5, 2), spacing=(5, 2))
    canvas = FalseColorCanvas()
    canvas.resize(900, 400)
    canvas.show()
    hovered: list[str] = []
    canvas.cell_hovered.connect(hovered.append)
    try:
        canvas.set_image(np.random.default_rng(0).random((20, 60, 3)).astype(np.float32))
        qapp.processEvents()
        canvas.show_cells(cells, colors={"A01": "#ff0000"})
        assert [canvas.cell_at(c.frames[0], c.rows[1]) for c in cells] == [c.name for c in cells]
        assert canvas.cell_at(cells[0].frames[1] + 1, cells[0].rows[0]) == ""   # the gap

        def move_to(frame: float, row: float) -> None:
            canvas._scene.sigMouseMoved.emit(canvas.view_box.mapViewToScene(QPointF(frame, row)))

        move_to(cells[4].frames[0] + 0.5, cells[4].rows[0] + 0.5)
        assert hovered == [cells[4].name]
        move_to(cells[4].frames[1] + 0.9, cells[4].rows[1] + 0.9)   # its last pixel
        assert hovered == [cells[4].name]
        move_to(0.5, 0.5)
        assert hovered == [cells[4].name, ""]

        canvas.show_cells([GridCell(row=0, column=0, frames=(0, 3), rows=(0, 3), name="only")],
                          labels=False)
        assert canvas.cell_at(cells[0].frames[0], cells[0].rows[0]) == ""
        canvas.clear_cells()
        assert canvas.cell_at(1, 1) == ""
    finally:
        canvas.close()
