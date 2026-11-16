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

logger = get_logger(__name__)

#: How each workflow step is named in the "Where" column.
STEP_LABELS = {"data": "Data", "rois": "ROIs", "values": "Values", "model": "Model",
               "predict": "Predict"}
_SEVERITY_ICONS = {"ok": ("circle-check", "ok"), "info": ("info-circle", "text_secondary"),
                   "attention": ("alert-triangle", "attention"),
                   "blocking": ("circle-x", "error")}
_VERDICT_TEXT = {"good": "Good", "usable_with_care": "Usable with care",
                 "not_usable": "Not usable"}
_VERDICT_ICONS = {"good": ("circle-check", "ok"), "usable_with_care": ("alert-triangle",
                                                                       "attention"),
                  "not_usable": ("circle-x", "error")}


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


class ChecksPanel(QWidget):
    """Every check of the project: severity, sentence, where, and a fix button."""

    #: A fix button was pressed.
    fix_requested = pyqtSignal(object)
    #: A check was double-clicked (to show what it refers to).
    check_activated = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._checks: list[Check] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = _table(["", "Check", "Where", "Fix"])
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            self.table.setColumnWidth(0, 28)
        self.table.cellDoubleClicked.connect(self._activated)
        layout.addWidget(self.table)

    def checks(self) -> list[Check]:
        """The checks shown, open ones first, then in workflow order."""
        return list(self._checks)

    def open_count(self) -> int:
        """The number of checks that ask for an action."""
        return sum(1 for check in self._checks if check.is_open)

    def set_checks(self, checks: list[Check], names: dict[str, str] | None = None) -> None:
        """Show ``checks``; ``names`` maps model ids to names for the Where column."""
        names = names or {}
        rank = {"blocking": 0, "attention": 1, "info": 2, "ok": 3}
        self._checks = sorted(checks, key=lambda check: rank.get(check.severity, 4))
        self.table.setRowCount(len(self._checks))
        for row, check in enumerate(self._checks):
            name, color = _SEVERITY_ICONS.get(check.severity, ("info-circle", None))
            mark = QTableWidgetItem(icon(name, color), "")
            mark.setToolTip(check.severity.capitalize())
            self.table.setItem(row, 0, mark)
            text = QTableWidgetItem(check.message)
            text.setToolTip(check.message)
            self.table.setItem(row, 1, text)
            where = STEP_LABELS.get(check.step, check.step)
            if check.model_run_id in names:
                where += f" › {names[check.model_run_id]}"
            self.table.setItem(row, 2, QTableWidgetItem(where))
            self.table.setCellWidget(row, 3, self._fix_button(check.fix))

    def _fix_button(self, fix: Fix | None) -> QWidget:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(4, 0, 4, 0)
        if fix is not None:
            button = QPushButton(fix.label)
            button.setObjectName("FixButton")
            button.setFlat(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, fix=fix: self.fix_requested.emit(fix))
            layout.addWidget(button)
        layout.addStretch(1)
        return holder

    def fix_button(self, row: int) -> QPushButton | None:
        """The fix button of ``row`` (for tests), or ``None``."""
        holder = self.table.cellWidget(row, 3)
        return None if holder is None else holder.findChild(QPushButton)

    def _activated(self, row: int, _column: int) -> None:
        if 0 <= row < len(self._checks):
            self.check_activated.emit(self._checks[row])


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
                _number_item(metrics.get("rmsecv"), 3),
                _number_item(metrics.get("r2"), 3),
                _number_item(metrics.get("rpd"), 2),
                QTableWidgetItem(row.get("validation", "")),
                QTableWidgetItem(""),
                QTableWidgetItem(row.get("fitted", "")),
            ]
            if verdict is not None:
                name, color = _VERDICT_ICONS[verdict.level]
                cells[7] = QTableWidgetItem(icon(name, color), _VERDICT_TEXT[verdict.level])
                cells[7].setToolTip(verdict.summary)
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
    item.setData(Qt.ItemDataRole.DisplayRole, int(number) if digits is None else
                 round(number, digits))
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item
