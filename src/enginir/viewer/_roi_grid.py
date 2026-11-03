"""ROI grid dialog: one rectangle around a tray becomes one ROI per sample cell.

The cells come from :func:`enginir._workflow.roi_grid`, the same rule scripts
use. Every change of the grid is reported through :attr:`RoiGridDialog.
preview_changed`, so the image can show the cells while the dialog is open.
Clicking a cell button skips that position (an empty place in the tray).
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .._logging import get_logger
from .._workflow import GridCell, roi_grid
from ._theme import secondary_text_style

logger = get_logger(__name__)

DEFAULT_NAME_PATTERN = "{file}-{row}{column:02d}"
_MAX_CELLS_PER_SIDE = 26


class RoiGridDialog(QDialog):
    """Choose rows, columns, inset, cells to skip and names for a tray of ROIs."""

    #: The cells to create and the skipped positions, after every change.
    preview_changed = pyqtSignal(object, object)

    def __init__(self, frames: tuple[int, int], rows: tuple[int, int], *, file: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add ROI grid")
        self._frames, self._rows, self._file = frames, rows, file
        self._skip: set[tuple[int, int]] = set()
        self._buttons: dict[tuple[int, int], QToolButton] = {}
        self.cells: list[GridCell] = []

        self._n_rows = QSpinBox()
        self._n_columns = QSpinBox()
        for spin, value, span in ((self._n_rows, 1, rows), (self._n_columns, 1, frames)):
            spin.setRange(1, max(1, min(_MAX_CELLS_PER_SIDE, span[1] - span[0] + 1)))
            spin.setValue(value)
            spin.valueChanged.connect(self._rebuild_cells)
        self._inset = QDoubleSpinBox()
        self._inset.setRange(0.0, 45.0)
        self._inset.setSingleStep(5.0)
        self._inset.setSuffix(" %")
        self._inset.setValue(10.0)
        self._inset.setToolTip("Left out on every side of a cell, to keep edges and gaps out")
        self._inset.valueChanged.connect(self._update)
        self._pattern = QLineEdit(DEFAULT_NAME_PATTERN)
        self._pattern.setToolTip("{file}, {row} (A, B, ...), {column} (1, 2, ...), "
                                 "{n} (running number)")
        self._pattern.textChanged.connect(self._update)
        self._order = QComboBox()
        self._order.addItem("Row by row", "rows")
        self._order.addItem("Column by column", "columns")
        self._order.currentIndexChanged.connect(self._update)

        form = QFormLayout()
        form.addRow("Rows:", self._n_rows)
        form.addRow("Columns:", self._n_columns)
        form.addRow("Inset:", self._inset)
        form.addRow("Names:", self._pattern)
        form.addRow("Order:", self._order)
        hint = QLabel("Click a cell to skip an empty position.")
        hint.setStyleSheet(secondary_text_style())
        self._cell_area = QWidget()
        self._cell_grid = QGridLayout(self._cell_area)
        self._cell_grid.setSpacing(2)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._buttons_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                             | QDialogButtonBox.StandardButton.Cancel)
        self._buttons_box.accepted.connect(self.accept)
        self._buttons_box.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Tray: frames {frames[0]}–{frames[1]}, "
                                f"rows {rows[0]}–{rows[1]}"))
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(self._cell_area)
        layout.addWidget(self._status)
        layout.addWidget(self._buttons_box)
        self._rebuild_cells()

    # -- settings (also used by tests and scripts) ----------------------

    def set_grid(self, n_rows: int, n_columns: int, *, inset_percent: float | None = None,
                 pattern: str | None = None, order: str | None = None) -> None:
        self._n_rows.setValue(n_rows)
        self._n_columns.setValue(n_columns)
        if inset_percent is not None:
            self._inset.setValue(inset_percent)
        if pattern is not None:
            self._pattern.setText(pattern)
        if order is not None:
            self._order.setCurrentIndex(self._order.findData(order))

    def toggle_cell(self, row: int, column: int) -> None:
        """Skip a position, or include it again."""
        self._buttons[(row, column)].click()

    @property
    def status_text(self) -> str:
        return self._status.text()

    # -- internals ------------------------------------------------------

    def _rebuild_cells(self, *_args: Any) -> None:
        for button in self._buttons.values():
            self._cell_grid.removeWidget(button)
            button.deleteLater()
        self._buttons.clear()
        n_rows, n_columns = self._n_rows.value(), self._n_columns.value()
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
        self._update()

    def _on_cell_toggled(self, position: tuple[int, int], included: bool) -> None:
        if included:
            self._skip.discard(position)
        else:
            self._skip.add(position)
        self._update()

    def _update(self, *_args: Any) -> None:
        ok = self._buttons_box.button(QDialogButtonBox.StandardButton.Ok)
        try:
            self.cells = roi_grid(
                self._frames, self._rows, n_rows=self._n_rows.value(),
                n_columns=self._n_columns.value(), inset=self._inset.value() / 100.0,
                skip=self._skip, order=str(self._order.currentData()),
                name_pattern=self._pattern.text(), file=self._file)
        except ValueError as exc:
            self.cells = []
            self._status.setText(str(exc))
            if ok is not None:
                ok.setEnabled(False)
            self.preview_changed.emit([], [])
            return
        names = {(cell.row, cell.column): cell.name for cell in self.cells}
        for (row, column), button in self._buttons.items():
            # Positions fit a button; the full ROI name is the tooltip.
            button.setText(f"{chr(ord('A') + row)}{column + 1}")
            button.setToolTip(names.get((row, column), "skipped"))
        skipped = roi_grid(self._frames, self._rows, n_rows=self._n_rows.value(),
                           n_columns=self._n_columns.value(), inset=self._inset.value() / 100.0,
                           name_pattern="{row}{column}") if self._skip else []
        skipped = [cell for cell in skipped if (cell.row, cell.column) in self._skip]
        self._status.setText(f"{len(self.cells)} ROIs will be created"
                             + (f", {len(self._skip)} position(s) skipped" if self._skip else ""))
        if ok is not None:
            ok.setEnabled(bool(self.cells))
        self.preview_changed.emit(list(self.cells), skipped)


__all__ = ["DEFAULT_NAME_PATTERN", "RoiGridDialog"]
