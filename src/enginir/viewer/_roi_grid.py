"""ROI grid dialogs: a tray of equal ROIs laid out by number, size, origin and spacing.

The cells come from :func:`enginir._workflow.roi_grid`, the same rule scripts
use. :class:`RoiGridDialog` starts from the defaults of Options > ROI Grid
Defaults (:class:`GridDefaultsDialog`); its changes apply to that grid only.
Every change is reported through :attr:`RoiGridDialog.preview_changed`, so
the image shows the cells while the dialog is open. Clicking a cell button
marks an empty position.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .._logging import get_logger
from .._workflow import GridCell, roi_grid
from ._grid_settings import NAME_PATTERNS, GridDefaults, save_grid_defaults
from ._theme import secondary_text_style

logger = get_logger(__name__)

DEFAULT_NAME_PATTERN = NAME_PATTERNS[0]
_MAX_CELLS_PER_SIDE = 50
_MAX_PIXELS = 100_000


class GridGeometryForm(QWidget):
    """Rows and columns, ROI size, origin and spacing (pixels), names and order."""

    changed = pyqtSignal()

    def __init__(self, values: GridDefaults, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spins: dict[str, QSpinBox] = {}
        for name, low, high in (("n_rows", 1, _MAX_CELLS_PER_SIDE),
                                ("n_columns", 1, _MAX_CELLS_PER_SIDE),
                                ("width", 1, _MAX_PIXELS), ("height", 1, _MAX_PIXELS),
                                ("origin_x", 0, _MAX_PIXELS), ("origin_y", 0, _MAX_PIXELS),
                                ("spacing_x", 0, _MAX_PIXELS), ("spacing_y", 0, _MAX_PIXELS)):
            spin = QSpinBox()
            spin.setRange(low, high)
            spin.valueChanged.connect(self.changed)
            self._spins[name] = spin
        for name in ("width", "height", "origin_x", "origin_y", "spacing_x", "spacing_y"):
            self._spins[name].setSuffix(" px")
        self._pattern = QComboBox()
        self._pattern.setEditable(True)
        self._pattern.addItems(list(NAME_PATTERNS))
        self._pattern.setToolTip("{file}: dataset file name, {row}: A, B, ..., {column}: 1, 2, "
                                 "..., {n}: running number. Pick one or type your own.")
        self._pattern.currentTextChanged.connect(self.changed)
        self._order = QComboBox()
        self._order.addItem("Row by row", "rows")
        self._order.addItem("Column by column", "columns")
        self._order.currentIndexChanged.connect(self.changed)

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QLabel("x (frames)"), 0, 1)
        grid.addWidget(QLabel("y (rows)"), 0, 2)
        for row, (label, x, y) in enumerate((
                ("Number:", "n_columns", "n_rows"),
                ("ROI size:", "width", "height"),
                ("Origin (first ROI):", "origin_x", "origin_y"),
                ("Spacing between ROIs:", "spacing_x", "spacing_y")), start=1):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(self._spins[x], row, 1)
            grid.addWidget(self._spins[y], row, 2)
        self._extent = QLabel("")
        self._extent.setStyleSheet(secondary_text_style())
        grid.addWidget(self._extent, 5, 0, 1, 3)
        grid.addWidget(QLabel("Names:"), 6, 0)
        grid.addWidget(self._pattern, 6, 1, 1, 2)
        grid.addWidget(QLabel("Order:"), 7, 0)
        grid.addWidget(self._order, 7, 1, 1, 2)
        self.changed.connect(self._update_extent)
        self.set_values(values)

    def values(self) -> GridDefaults:
        spins = {name: spin.value() for name, spin in self._spins.items()}
        return GridDefaults(**spins, name_pattern=self._pattern.currentText(),
                            order=str(self._order.currentData()))

    def set_values(self, values: GridDefaults) -> None:
        for name, spin in self._spins.items():
            spin.blockSignals(True)
            spin.setValue(int(getattr(values, name)))
            spin.blockSignals(False)
        self._pattern.blockSignals(True)
        self._pattern.setCurrentText(values.name_pattern)
        self._pattern.blockSignals(False)
        self._order.blockSignals(True)
        self._order.setCurrentIndex(max(0, self._order.findData(values.order)))
        self._order.blockSignals(False)
        self.changed.emit()

    def _update_extent(self) -> None:
        v = self.values()
        pitch_x, pitch_y = v.width + v.spacing_x, v.height + v.spacing_y
        last_x = v.origin_x + (v.n_columns - 1) * pitch_x + v.width - 1
        last_y = v.origin_y + (v.n_rows - 1) * pitch_y + v.height - 1
        self._extent.setText(f"Pitch {pitch_x} x {pitch_y} px; the grid covers frames "
                             f"{v.origin_x}–{last_x} and rows {v.origin_y}–{last_y}.")


class GridDefaultsDialog(QDialog):
    """Options > ROI Grid Defaults: the geometry every new ROI grid starts with."""

    def __init__(self, values: GridDefaults, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ROI grid defaults")
        self.form = GridGeometryForm(values)
        note = QLabel("Every new ROI grid starts from these values. Changes made in the grid "
                      "dialog apply to that grid only.")
        note.setWordWrap(True)
        note.setStyleSheet(secondary_text_style())
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel
                                   | QDialogButtonBox.StandardButton.RestoreDefaults)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        restore = buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        if restore is not None:
            restore.clicked.connect(lambda: self.form.set_values(GridDefaults()))
        layout = QVBoxLayout(self)
        layout.addWidget(self.form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        save_grid_defaults(self.form.values())
        logger.info("ROI grid defaults saved: {}", self.form.values())
        self.accept()


def prediction_only_check(checked: bool) -> QCheckBox:
    """The Prediction only checkbox of the ROI dialogs (roi-target contract, ``job``)."""
    check = QCheckBox("Prediction only (a sample to predict; never used to fit a model)")
    check.setChecked(checked)
    check.setToolTip("Prediction ROIs mark tablets to predict in Predict > Predict ROIs. They "
                     "never enter a model fit, even with a target value; a target value then "
                     "only gives the prediction's residual. The choice is remembered for this "
                     "dataset.")
    return check


class RoiGridDialog(QDialog):
    """Lay out a tray of ROIs on the shown image and choose the empty positions."""

    #: Cells to create, skipped cells, and cells outside the image, after every change.
    preview_changed = pyqtSignal(object, object, object)

    def __init__(self, image_size: tuple[int, int], *, file: str, defaults: GridDefaults,
                 taken_names: Iterable[str] = (), prediction: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add ROI grid")
        self._image_size, self._file = image_size, file
        #: Names of the project's ROIs. Names are labels: repeating one is allowed and only
        #: mentioned, because ROIs are told apart by id and data file.
        self._taken = set(taken_names)
        self._skip: set[tuple[int, int]] = set()
        self._buttons: dict[tuple[int, int], QToolButton] = {}
        self._shape: tuple[int, int] | None = None
        self.cells: list[GridCell] = []

        self.form = GridGeometryForm(defaults)
        self.form.changed.connect(self._update)
        select_all = QPushButton("Select All")
        select_all.clicked.connect(lambda: self._select_all(True))
        select_none = QPushButton("Select None")
        select_none.clicked.connect(lambda: self._select_all(False))
        self._make_defaults = QPushButton("Make Defaults")
        self._make_defaults.setToolTip("Use this geometry and these names for every new ROI grid "
                                       "(Options > ROI Grid Defaults)")
        self._make_defaults.clicked.connect(self.make_defaults)
        select_row = QHBoxLayout()
        hint = QLabel("Click a cell to mark an empty position.")
        hint.setStyleSheet(secondary_text_style())
        select_row.addWidget(hint, 1)
        select_row.addWidget(select_all)
        select_row.addWidget(select_none)
        self._cell_area = QWidget()
        self._cell_grid = QGridLayout(self._cell_area)
        self._cell_grid.setSpacing(2)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self.prediction_check = prediction_only_check(prediction)
        self._buttons_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                             | QDialogButtonBox.StandardButton.Cancel)
        self._buttons_box.accepted.connect(self.accept)
        self._buttons_box.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Image: {image_size[0]} frames x {image_size[1]} rows"))
        form_box = QFormLayout()
        form_box.addRow(self.form)
        layout.addLayout(form_box)
        defaults_row = QHBoxLayout()
        self._defaults_note = QLabel("")
        self._defaults_note.setStyleSheet(secondary_text_style())
        defaults_row.addWidget(self._defaults_note, 1)
        defaults_row.addWidget(self._make_defaults)
        layout.addLayout(defaults_row)
        layout.addLayout(select_row)
        layout.addWidget(self._cell_area)
        layout.addWidget(self._status)
        layout.addWidget(self.prediction_check)
        layout.addWidget(self._buttons_box)
        self._update()

    @property
    def job(self) -> str:
        """``"prediction"`` when Prediction only is ticked, else ``"calibration"``."""
        return "prediction" if self.prediction_check.isChecked() else "calibration"

    # -- settings (also used by tests and scripts) ----------------------

    def set_values(self, values: GridDefaults) -> None:
        self.form.set_values(values)

    def set_grid(self, **changes: Any) -> None:
        """Change some values, e.g. ``set_grid(n_rows=2, width=30)``."""
        self.form.set_values(replace(self.form.values(), **changes))

    def make_defaults(self) -> None:
        """Store the dialog's geometry and names as the defaults of every new ROI grid."""
        save_grid_defaults(self.form.values())
        logger.info("ROI grid defaults saved from the grid dialog: {}", self.form.values())
        self._defaults_note.setText("Saved as the defaults for new grids.")

    def redraw_preview(self) -> None:
        """Report the current cells again (e.g. once the image is connected)."""
        self._update()

    def toggle_cell(self, row: int, column: int) -> None:
        """Mark a position empty, or include it again."""
        self._buttons[(row, column)].click()

    @property
    def status_text(self) -> str:
        return self._status.text()

    # -- internals ------------------------------------------------------

    def _select_all(self, included: bool) -> None:
        self._skip = set() if included else set(self._buttons)
        for position, button in self._buttons.items():
            button.blockSignals(True)
            button.setChecked(included)
            button.blockSignals(False)
        self._update()

    def _rebuild_cells(self, n_rows: int, n_columns: int) -> None:
        for button in self._buttons.values():
            self._cell_grid.removeWidget(button)
            button.deleteLater()
        self._buttons.clear()
        self._skip = {rc for rc in self._skip if rc[0] < n_rows and rc[1] < n_columns}
        for r in range(n_rows):
            for c in range(n_columns):
                button = QToolButton()
                button.setCheckable(True)
                button.setChecked((r, c) not in self._skip)
                button.setMinimumWidth(36)
                button.toggled.connect(
                    lambda checked, rc=(r, c): self._on_cell_toggled(rc, checked))
                self._cell_grid.addWidget(button, r, c)
                self._buttons[(r, c)] = button
        self._shape = (n_rows, n_columns)

    def _on_cell_toggled(self, position: tuple[int, int], included: bool) -> None:
        if included:
            self._skip.discard(position)
        else:
            self._skip.add(position)
        self._update()

    def _update(self, *_args: Any) -> None:
        v = self.form.values()
        self._defaults_note.setText("")
        if self._shape != (v.n_rows, v.n_columns):
            self._rebuild_cells(v.n_rows, v.n_columns)
        ok = self._buttons_box.button(QDialogButtonBox.StandardButton.Ok)
        geometry: dict[str, Any] = dict(
            n_rows=v.n_rows, n_columns=v.n_columns, size=(v.width, v.height),
            origin=(v.origin_x, v.origin_y), spacing=(v.spacing_x, v.spacing_y))
        try:
            planned = roi_grid(**geometry, skip=self._skip, order=v.order,
                               name_pattern=v.name_pattern, file=self._file)
        except ValueError as exc:
            self._show([], [], [], str(exc), ok)
            return
        n_frames, height = self._image_size
        inside = [c for c in planned if c.frames[1] < n_frames and c.rows[1] < height]
        outside = [c for c in planned if c not in inside]
        skipped = [c for c in roi_grid(**geometry, name_pattern="{row}{column}")
                   if (c.row, c.column) in self._skip]
        names = {(cell.row, cell.column): cell.name for cell in planned}
        for (row, column), button in self._buttons.items():
            # Positions fit a button; the full ROI name is the tooltip.
            button.setText(f"{GridCell(row, column, (0, 0), (0, 0), '').position}")
            button.setToolTip(names.get((row, column), "empty"))
        if outside:
            message = (f"{len(outside)} ROI(s) lie outside the image (red), e.g. "
                       f"{outside[0].position}: move the origin, shrink the grid or mark "
                       "them empty.")
            self._show([], skipped, outside, message, ok, inside=inside)
            return
        message = f"{len(planned)} ROIs will be created" + (
            f", {len(self._skip)} position(s) empty" if self._skip else "")
        repeated = [cell.name for cell in planned if cell.name in self._taken]
        if repeated:
            message += (f". {len(repeated)} name(s) repeat ROIs of the project, e.g. "
                        f"'{repeated[0]}'; that is allowed (ROIs are told apart by id and "
                        "dataset).")
        self._show(planned, skipped, [], message, ok)

    def _show(self, cells: list[GridCell], skipped: list[GridCell], outside: list[GridCell],
              message: str, ok: Any, *, inside: list[GridCell] | None = None) -> None:
        self.cells = cells
        self._status.setText(message)
        if ok is not None:
            ok.setEnabled(bool(cells))
        self.preview_changed.emit(list(inside if inside is not None else cells), skipped,
                                  outside)


__all__ = ["DEFAULT_NAME_PATTERN", "GridDefaultsDialog", "GridGeometryForm", "RoiGridDialog"]
