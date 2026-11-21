"""The bottom panels of the main window: Checks and Models (plan gui-1.6, Phases 3 and 5).

Both only show what the library computes: the checks are
:meth:`enginir.Project.checks`, the verdicts :meth:`Model.verdict`; neither
panel judges anything itself (the viewer holds no workflow rule).
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .._guidance import Check, Fix, Verdict
from .._logging import get_logger
from ._icons import icon
from ._ui_helpers import wrap_tooltip

logger = get_logger(__name__)

#: How each workflow step is named in the "Where" column.
STEP_LABELS = {"data": "Data", "rois": "ROIs", "values": "Values", "model": "Model",
               "predict": "Predict"}
_SEVERITY_ICONS = {"ok": ("circle-check", "ok"), "info": ("info-circle", "text_secondary"),
                   "attention": ("alert-triangle", "attention"),
                   "blocking": ("circle-x", "error")}
#: From this many passed checks on, they fold into one row.
_FOLD_PASSED_FROM = 2
_VERDICT_TEXT = {"good": "Good", "usable_with_care": "Usable with care",
                 "not_usable": "Not usable"}
_VERDICT_ICONS = {"good": ("circle-check", "ok"), "usable_with_care": ("alert-triangle",
                                                                       "attention"),
                  "not_usable": ("circle-x", "error")}


#: Room added to the Where and Fix columns of the Checks table beyond their contents, and
#: their least widths (they were fitted exactly and looked cramped).
_COLUMN_MARGIN = 28
_MIN_WIDTHS = {2: 150, 3: 140}


def _table(columns: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setWordWrap(False)
    table.setShowGrid(False)
    vertical = table.verticalHeader()
    if vertical is not None:
        vertical.setVisible(False)
        vertical.setDefaultSectionSize(24)
    return table


def _fix_button(label: str) -> QPushButton:
    """A small outlined button in a table row; wide enough for its label (the style
    sheet's padding is not in the size hint a table column is fitted to)."""
    button = QPushButton(label)
    button.setObjectName("FixButton")
    button.setFlat(True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setMinimumWidth(button.fontMetrics().horizontalAdvance(label) + 22)
    return button


class ChecksPanel(QWidget):
    """Every check of the project: severity, sentence, where, and a fix button."""

    #: A fix button was pressed.
    fix_requested = pyqtSignal(object)
    #: A check was double-clicked (to show what it refers to).
    check_activated = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._checks: list[Check] = []
        self._names: dict[str, str] = {}
        self._rows: list[Check | None] = []
        self._fold_row: int | None = None
        self._show_passed = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = _table(["", "Check", "Where", "Fix"])
        self.table.setObjectName("ChecksTable")
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            # Where and Fix: their contents plus a margin (_fit_columns), and wider by hand.
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(0, 28)
        self.table.cellDoubleClicked.connect(self._activated)
        layout.addWidget(self.table)

    def checks(self) -> list[Check]:
        """Every check of the panel, open ones first, then in workflow order."""
        return list(self._checks)

    def row_checks(self) -> list[Check | None]:
        """The check of every table row (``None``: the row of the folded passed checks)."""
        return list(self._rows)

    def open_count(self) -> int:
        """The number of checks that ask for an action."""
        return sum(1 for check in self._checks if check.is_open)

    def set_checks(self, checks: list[Check], names: dict[str, str] | None = None) -> None:
        """Show ``checks``; ``names`` maps model ids to names for the Where column.

        Two or more passed checks fold into one row ("4 checks passed") that
        unfolds with its button, so the open ones are not pushed out of a
        low panel.
        """
        rank = {"blocking": 0, "attention": 1, "info": 2, "ok": 3}
        self._checks = sorted(checks, key=lambda check: rank.get(check.severity, 4))
        self._names = dict(names or {})
        self._fill()

    def passed_folded(self) -> bool:
        """Whether the passed checks are shown as one row."""
        return self._fold_row is not None and not self._show_passed

    def set_passed_shown(self, shown: bool) -> None:
        """Unfold (or fold) the passed checks."""
        if shown != self._show_passed:
            self._show_passed = shown
            self._fill()

    def _fill(self) -> None:
        passed = [check for check in self._checks if check.severity == "ok"]
        others = [check for check in self._checks if check.severity != "ok"]
        fold = len(passed) >= _FOLD_PASSED_FROM
        #: The check of every table row; ``None`` for the "passed" row.
        self._rows: list[Check | None] = [*others, *([None] if fold else []),
                                          *(passed if self._show_passed or not fold else [])]
        self._fold_row = len(others) if fold else None
        self.table.setRowCount(len(self._rows))
        for row, check in enumerate(self._rows):
            if check is None:
                self._fill_fold_row(row, len(passed))
                continue
            name, color = _SEVERITY_ICONS.get(check.severity, ("info-circle", None))
            mark = QTableWidgetItem(icon(name, color), "")
            mark.setToolTip(wrap_tooltip(check.severity.capitalize()))
            self.table.setItem(row, 0, mark)
            text = QTableWidgetItem(check.message)
            text.setToolTip(wrap_tooltip(check.message))
            self.table.setItem(row, 1, text)
            where = STEP_LABELS.get(check.step, check.step)
            if check.model_run_id in self._names:
                where += f" › {self._names[check.model_run_id]}"
            self.table.setItem(row, 2, QTableWidgetItem(where))
            self.table.setCellWidget(row, 3, self._fix_button(check.fix))
        self._fit_columns()

    def _fit_columns(self) -> None:
        """Where and Fix: their contents plus a margin, at least a fixed width; never
        narrower than the user has dragged them (the table refills on every change)."""
        for column, least in _MIN_WIDTHS.items():
            current = self.table.columnWidth(column)
            self.table.resizeColumnToContents(column)
            needed = max(least, self.table.columnWidth(column) + _COLUMN_MARGIN)
            self.table.setColumnWidth(column, max(needed, current))

    def _fill_fold_row(self, row: int, count: int) -> None:
        name, color = _SEVERITY_ICONS["ok"]
        self.table.setItem(row, 0, QTableWidgetItem(icon(name, color), ""))
        self.table.setItem(row, 1, QTableWidgetItem(f"{count} checks passed"))
        self.table.setItem(row, 2, QTableWidgetItem(""))
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(4, 0, 4, 0)
        button = _fix_button("Hide" if self._show_passed else "Show")
        button.clicked.connect(lambda _checked=False: self.set_passed_shown(
            not self._show_passed))
        layout.addWidget(button)
        layout.addStretch(1)
        self.table.setCellWidget(row, 3, holder)

    def _fix_button(self, fix: Fix | None) -> QWidget:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(4, 0, 4, 0)
        if fix is not None:
            button = _fix_button(fix.label)
            button.clicked.connect(lambda _checked=False, fix=fix: self.fix_requested.emit(fix))
            layout.addWidget(button)
        layout.addStretch(1)
        return holder

    def fix_button(self, row: int) -> QPushButton | None:
        """The fix button of ``row`` (for tests), or ``None``."""
        holder = self.table.cellWidget(row, 3)
        return None if holder is None else holder.findChild(QPushButton)

    def _activated(self, row: int, _column: int) -> None:
        if not 0 <= row < len(self._rows):
            return
        check = self._rows[row]
        if check is None:  # the "passed" row unfolds
            self.set_passed_shown(not self._show_passed)
        else:
            self.check_activated.emit(check)


class ModelsPanel(QWidget):
    """The fitted models of the project with their figures of merit and verdict."""

    #: The model run id of a double-clicked row.
    model_selected = pyqtSignal(str)

    COLUMNS = ["Model", "Type", "Components", "RMSECV (per ROI)", "Q²", "RPD", "Validation",
               "Verdict", "Fitted"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ids: list[str] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = _table(self.COLUMNS)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._activated)
        self._empty = QLabel("No model yet. Fit one in the Model step "
                             "(Model > Fit Model...).")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setObjectName("EmptyState")
        layout.addWidget(self._empty)
        layout.addWidget(self.table)

    def set_models(self, rows: list[dict[str, Any]]) -> None:
        """Show one row per model: ``id``, ``name``, ``type``, ``components``, ``metrics``,
        ``validation``, ``verdict`` (a :class:`Verdict` or ``None``) and ``fitted``."""
        self.table.setSortingEnabled(False)
        self._ids = [row["id"] for row in rows]
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            metrics = row.get("metrics", {})
            verdict: Verdict | None = row.get("verdict")
            cells: list[QTableWidgetItem] = [
                QTableWidgetItem(row["name"]),
                QTableWidgetItem(row.get("type", "")),
                _number_item(row.get("components")),
                _number_item(metrics.get("rmsecv"), 4),
                _number_item(metrics.get("r2"), 4),
                _number_item(metrics.get("rpd"), 4),
                QTableWidgetItem(row.get("validation", "")),
                QTableWidgetItem(""),
                QTableWidgetItem(row.get("fitted", "")),
            ]
            if verdict is not None:
                name, color = _VERDICT_ICONS[verdict.level]
                cells[7] = QTableWidgetItem(icon(name, color), _VERDICT_TEXT[verdict.level])
                cells[7].setToolTip(wrap_tooltip(verdict.summary))
            cells[0].setData(Qt.ItemDataRole.UserRole, row["id"])
            for column, cell in enumerate(cells):
                self.table.setItem(index, column, cell)
        self.table.setSortingEnabled(True)
        self._empty.setVisible(not rows)
        self.table.setVisible(bool(rows))

    def model_ids(self) -> list[str]:
        """The ids of the models shown, in the order given."""
        return list(self._ids)

    def _activated(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.model_selected.emit(str(item.data(Qt.ItemDataRole.UserRole)))


def _number_item(value: Any, digits: int | None = None) -> QTableWidgetItem:
    item = QTableWidgetItem()
    try:
        number = float(value)
    except (TypeError, ValueError):
        item.setText("–")
        return item
    # A number, so the column sorts by value; figures of merit with ``digits``
    # significant digits (the rule of ``figure_text``).
    item.setData(Qt.ItemDataRole.DisplayRole, int(number) if digits is None else
                 float(f"{number:.{digits}g}"))
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


def verdict_mark(level: str) -> tuple[Any, str]:
    """The icon and the words of a verdict level (``good``: a tick and "Good")."""
    name, color = _VERDICT_ICONS[level]
    return icon(name, color), _VERDICT_TEXT[level]
