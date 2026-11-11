"""ROI table: the target values of all ROIs, one column per target (ROI/target contract).

One row per ROI with its data file, name and pixel count, then one column per
target, then, while the session has per-ROI prediction results, read-only
columns per model (mean, SD, % removed, % above DCrit, residual; viewer
contract §3). A header click sorts the rows. Values are edited in place:

- typing a value and pressing Enter stores it and moves to the next row;
- clearing a cell removes the value (the ROI becomes unlabeled for it);
- pasting a column (``Ctrl+V``) fills consecutive rows from the current cell;
  every pasted value is checked first, and an invalid one writes nothing.

Every write goes through :meth:`ProjectState.set_roi_values`. The table needs
only the project file, never the data files.
"""

from __future__ import annotations

import math
import re
from typing import Any

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QGuiApplication, QKeyEvent, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemDelegate,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .._logging import get_logger
from ._state import ProjectState
from ._theme import secondary_text_style

logger = get_logger(__name__)

_FIXED = ("Dataset", "ROI", "Pixels", "Set", "Sample")
_SAMPLE_COLUMN = _FIXED.index("Sample")
#: The ROI's set, Calibration or Prediction, chosen from a drop-down (roi-target contract,
#: ``use``; terminology contract, Calibration and Prediction).
_USE_COLUMN = _FIXED.index("Set")
#: Stored ``use`` -> the label of the Set column, in the drop-down's order.
SET_LABELS = {"calibration": "Calibration", "prediction": "Prediction"}
_ALL_FILES = "__all__"
#: Per-ROI prediction result columns per model: (key, header, decimals as format).
_RESULT_COLUMNS = (("mean", "mean", ".3g"), ("sd", "SD", ".2g"),
                   ("removed", "% excluded", ".0f"), ("dcrit", "% above DCrit", ".0f"),
                   ("residual", "residual", ".3g"))
ModelIndex = QModelIndex | QPersistentModelIndex


def _natural_key(text: str) -> list[Any]:
    """Sort ``A2`` before ``A10``."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_value(text: str) -> float | None:
    """A cell's text as a value: empty means "no value"; otherwise a finite number.

    A decimal comma is accepted. Raises :class:`ValueError` for anything else.
    """
    text = text.strip()
    if not text:
        return None
    value = float(text.replace(",", ".")) if text.count(",") == 1 and "." not in text \
        else float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text!r} is not a finite number")
    return value


class RoiTableModel(QAbstractTableModel):
    """Rows: ROIs (filtered by data file); columns: data file, ROI, pixels, sample, targets."""

    #: A value could not be stored; the message says why.
    edit_failed = pyqtSignal(str)

    def __init__(self, project_state: ProjectState, parent: Any = None) -> None:
        super().__init__(parent)
        self.project_state = project_state
        self._dataset_filter: str | None = None
        self._missing_target: str | None = None
        self._roi_ids: list[str] = []
        self._target_ids: list[str] = []
        #: (model run id, model name, key) per result column.
        self._result_columns: list[tuple[str, str, str]] = []
        self._results: dict[tuple[str, str], Any] = {}  # (model run id, roi id) -> statistics
        self._sort: tuple[int, Qt.SortOrder] | None = None
        for signal in (project_state.project_loaded, project_state.rois_changed,
                       project_state.targets_changed, project_state.target_assignments_changed,
                       project_state.datasets_changed, project_state.roi_results_changed):
            signal.connect(self.reload)
        self.reload()

    # -- content ----------------------------------------------------------

    def set_dataset_filter(self, dataset_id: str | None) -> None:
        """Show only the ROIs of one data file (``None``: all)."""
        self._dataset_filter = dataset_id
        self.reload()

    def set_missing_filter(self, target_id: str | None) -> None:
        """Show only ROIs without a value for ``target_id`` (``None``: all)."""
        self._missing_target = target_id
        self.reload()

    def reload(self, *_args: Any) -> None:
        """Re-read the project; unchanged rows and columns only refresh their cells.

        Keeping the layout when only values change keeps the current cell, so
        Enter can move on to the next row after a value was stored.
        """
        state = self.project_state
        target_ids = sorted(state.targets, key=lambda t: state.targets[t].name.lower())
        rois = [roi for roi in state.rois.values()
                if self._dataset_filter is None or roi.dataset_id == self._dataset_filter]
        if self._missing_target is not None:
            rois = [roi for roi in rois
                    if (roi.roi_id, self._missing_target) not in state.target_assignments]
        rois.sort(key=lambda roi: (_natural_key(self._file_name(roi.dataset_id)),
                                   _natural_key(roi.name or roi.roi_id)))
        roi_ids = [roi.roi_id for roi in rois]
        self._results = {(summary.model_run_id, row.roi_id): row
                         for summary in state.roi_results for row in summary.rois}
        result_columns = [(summary.model_run_id, summary.model_name, key)
                          for summary in state.roi_results for key, _h, _f in _RESULT_COLUMNS]
        if (set(roi_ids) == set(self._roi_ids) and target_ids == self._target_ids
                and result_columns == self._result_columns):
            # Same rows: keep their order (an edit never moves the row being edited).
            if roi_ids:
                self.dataChanged.emit(self.index(0, 0),
                                      self.index(len(roi_ids) - 1, self.columnCount() - 1))
            return
        self.beginResetModel()
        self._target_ids, self._result_columns = target_ids, result_columns
        self._roi_ids = self._sorted(roi_ids)
        self.endResetModel()

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        """Sort the rows by a column (numbers by value; empty cells last); a negative column
        restores the data file and name order."""
        self._sort = (column, order) if column >= 0 else None
        if self._sort is None:
            self._roi_ids = []  # the next reload rebuilds the default order
            self.reload()
            return
        self.layoutAboutToBeChanged.emit()
        self._roi_ids = self._sorted(self._roi_ids)
        self.layoutChanged.emit()

    def _sorted(self, roi_ids: list[str]) -> list[str]:
        if self._sort is None or not 0 <= self._sort[0] < self.columnCount():
            return roi_ids
        column, order = self._sort
        values = {roi_id: self._sort_value(roi_id, column) for roi_id in roi_ids}
        filled = sorted((r for r in roi_ids if values[r] is not None), key=lambda r: values[r],
                        reverse=order == Qt.SortOrder.DescendingOrder)
        return filled + [r for r in roi_ids if values[r] is None]  # empty cells last

    def _sort_value(self, roi_id: str, column: int) -> Any:
        """A cell's sort key: a float for numbers, a natural key for text, None if empty."""
        roi = self.project_state.rois[roi_id]
        if column == _USE_COLUMN:
            return SET_LABELS[roi.use]
        if column == 2:
            return float(roi.n_pixels)
        number = self._number(roi_id, column)
        if number is not None or column >= len(_FIXED):
            return number
        text = self._text(roi, column)
        return _natural_key(text) if text else None

    def _number(self, roi_id: str, column: int) -> float | None:
        """The value of a target or result cell, if any."""
        target_id = self.target_id(column)
        if target_id is not None:
            assignment = self.project_state.target_assignments.get((roi_id, target_id))
            return None if assignment is None else float(assignment.value)
        result = self.result_column(column)
        if result is None:
            return None
        model_run_id, _name, key = result
        row = self._results.get((model_run_id, roi_id))
        if row is None:
            return None
        value = {"mean": row.mean, "sd": row.sd,
                 "removed": 100.0 * row.n_removed / max(row.n_pixels, 1),
                 "dcrit": 100.0 * row.share_above_dcrit, "residual": row.residual}[key]
        return None if value is None or not math.isfinite(value) else float(value)

    def result_column(self, column: int) -> tuple[str, str, str] | None:
        """(model run id, model name, key) of a prediction result column, else ``None``."""
        index = column - len(_FIXED) - len(self._target_ids)
        return self._result_columns[index] if 0 <= index < len(self._result_columns) else None

    def roi_id(self, row: int) -> str:
        return self._roi_ids[row]

    def row_of(self, roi_id: str) -> int | None:
        return self._roi_ids.index(roi_id) if roi_id in self._roi_ids else None

    def target_id(self, column: int) -> str | None:
        index = column - len(_FIXED)
        return self._target_ids[index] if 0 <= index < len(self._target_ids) else None

    def column_of(self, target_id: str) -> int | None:
        return (len(_FIXED) + self._target_ids.index(target_id)
                if target_id in self._target_ids else None)

    def _file_name(self, dataset_id: str) -> str:
        return self.project_state.data_file_label(dataset_id)

    # -- Qt model -----------------------------------------------------------

    def rowCount(self, parent: ModelIndex = QModelIndex()) -> int:  # noqa: N802 (Qt API)
        return 0 if parent.isValid() else len(self._roi_ids)

    def columnCount(self, parent: ModelIndex = QModelIndex()) -> int:  # noqa: N802 (Qt API)
        return 0 if parent.isValid() else (len(_FIXED) + len(self._target_ids)
                                           + len(self._result_columns))

    def headerData(self, section: int, orientation: Qt.Orientation,  # noqa: N802 (Qt API)
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        if section < len(_FIXED):
            return _FIXED[section]
        result = self.result_column(section)
        if result is not None:
            header = next(h for key, h, _f in _RESULT_COLUMNS if key == result[2])
            return f"{result[1]}: {header}"
        target = self.project_state.targets.get(self.target_id(section) or "")
        if target is None:
            return None
        return f"{target.name} ({target.unit})" if target.unit else target.name

    def data(self, index: ModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        state = self.project_state
        roi = state.rois[self._roi_ids[index.row()]]
        column = index.column()
        if column == _USE_COLUMN:
            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
                return SET_LABELS[roi.use]
            if role == Qt.ItemDataRole.ToolTipRole:
                return ("Prediction ROI: predicted in Predict > Predict ROIs, never used to fit "
                        "a model (double-click to change)" if roi.use == "prediction"
                        else "Calibration ROI: used to fit models (double-click to change)")
            return None
        result = self.result_column(column)
        if result is not None:
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            if role == Qt.ItemDataRole.ToolTipRole:
                return (f"Predicted by '{result[1]}' in this session (not stored); "
                        "follows the pixel exclusion of Predict")
            if role != Qt.ItemDataRole.DisplayRole:
                return None
            number = self._number(roi.roi_id, column)
            fmt = next(f for key, _h, f in _RESULT_COLUMNS if key == result[2])
            return "" if number is None else format(number, fmt)
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole,
                        Qt.ItemDataRole.TextAlignmentRole):
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return (Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    if column == 2 or column > _SAMPLE_COLUMN else None)
        if column < len(_FIXED):
            return self._text(roi, column)
        assignment = state.target_assignments.get((roi.roi_id, self.target_id(column) or ""))
        return "" if assignment is None else f"{assignment.value:g}"

    def _text(self, roi: Any, column: int) -> str:
        if column == 0:
            return self._file_name(roi.dataset_id)
        if column == 1:
            return roi.name or roi.roi_id
        if column == 2:
            return str(roi.n_pixels)
        if column == _SAMPLE_COLUMN:
            return roi.sample or ""
        return ""

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        base = super().flags(index)
        editable = (self.target_id(index.column()) is not None
                    or index.column() in (_SAMPLE_COLUMN, _USE_COLUMN))
        return base | Qt.ItemFlag.ItemIsEditable if editable else base

    def setData(self, index: ModelIndex, value: Any,  # noqa: N802 (Qt API)
                role: int = Qt.ItemDataRole.EditRole) -> bool:
        if role == Qt.ItemDataRole.EditRole and index.column() == _USE_COLUMN:
            roi_id = self._roi_ids[index.row()]
            by_label = {label.lower(): use for use, label in SET_LABELS.items()}
            use = by_label.get(str(value).strip().lower(), str(value))
            if use not in SET_LABELS:
                self.edit_failed.emit(f"{value!r} is not a set: choose Calibration or "
                                      "Prediction.")
                return False
            if use != self.project_state.rois[roi_id].use:
                try:
                    self.project_state.set_roi_uses([(roi_id, use)])
                except (OSError, ValueError, KeyError) as exc:
                    logger.error("ROI use not stored: {}", exc)
                    self.edit_failed.emit(str(exc))
                    return False
            return True
        if role == Qt.ItemDataRole.EditRole and index.column() == _SAMPLE_COLUMN:
            roi_id = self._roi_ids[index.row()]
            label = str(value).strip() or None
            if label != self.project_state.rois[roi_id].sample:
                self.write_samples([(roi_id, label)])
            return True
        target_id = self.target_id(index.column())
        if role != Qt.ItemDataRole.EditRole or target_id is None:
            return False
        roi_id = self._roi_ids[index.row()]
        try:
            parsed = parse_value(str(value))
        except ValueError as exc:
            logger.warning("Target value not stored: {}", exc)
            self.edit_failed.emit(f"{value!r} is not a number; the value was not changed.")
            return False
        current = self.project_state.target_assignments.get((roi_id, target_id))
        if (current is None and parsed is None) or (
                current is not None and parsed is not None and current.value == parsed):
            return True  # unchanged: no write
        self.write_values([(roi_id, target_id, parsed)])
        return True

    def write_values(self, changes: list[tuple[str, str, float | None]]) -> None:
        """Store values (``None`` removes); the table reloads from the project afterwards."""
        try:
            self.project_state.set_roi_values(changes)
        except (OSError, ValueError, KeyError) as exc:
            logger.error("Target values not stored: {}", exc)
            self.edit_failed.emit(str(exc))


    def write_samples(self, changes: list[tuple[str, str | None]]) -> None:
        """Store sample labels (``None`` removes); the table reloads afterwards."""
        try:
            self.project_state.set_roi_samples(changes)
        except (OSError, ValueError, KeyError) as exc:
            logger.error("Sample labels not stored: {}", exc)
            self.edit_failed.emit(str(exc))


class _SetDelegate(QStyledItemDelegate):
    """The Set column's editor: a drop-down with Calibration and Prediction."""

    def createEditor(self, parent: QWidget | None, option: Any,  # noqa: N802 (Qt API)
                     index: QModelIndex) -> QWidget:
        combo = QComboBox(parent)
        for use, label in SET_LABELS.items():
            combo.addItem(label, use)
        return combo

    def setEditorData(self, editor: QWidget | None, index: QModelIndex) -> None:  # noqa: N802
        if isinstance(editor, QComboBox):
            editor.setCurrentIndex(max(0, editor.findText(str(index.data()))))

    def setModelData(self, editor: QWidget | None, model: Any,  # noqa: N802 (Qt API)
                     index: QModelIndex) -> None:
        if isinstance(editor, QComboBox) and model is not None:
            model.setData(index, editor.currentData(), Qt.ItemDataRole.EditRole)


class _RoiTableView(QTableView):
    """Enter stores the value and edits the row below; Ctrl+V pastes a column."""

    paste_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._next_edit: QPersistentModelIndex | None = None
        self._edit_timer = QTimer(self)
        self._edit_timer.setSingleShot(True)
        self._edit_timer.timeout.connect(self._edit_next)

    def keyPressEvent(self, e: QKeyEvent | None) -> None:  # noqa: N802 (Qt API)
        if e is None:
            return
        if e.matches(QKeySequence.StandardKey.Paste):
            self.paste_requested.emit()
            return
        super().keyPressEvent(e)

    def closeEditor(self, editor: QWidget | None,  # noqa: N802 (Qt API)
                    hint: QAbstractItemDelegate.EndEditHint) -> None:
        """Enter in a cell editor (``SubmitModelCache``) continues in the row below."""
        current = self.currentIndex()
        if hint != QAbstractItemDelegate.EndEditHint.SubmitModelCache:
            super().closeEditor(editor, hint)
            return
        super().closeEditor(editor, QAbstractItemDelegate.EndEditHint.NoHint)
        below = current.siblingAtRow(current.row() + 1)
        if below.isValid():
            self.setCurrentIndex(below)
            # Open the next editor once the closed one is gone; the timer is a
            # child of the view, so it never fires after the view was deleted.
            self._next_edit = QPersistentModelIndex(below)
            self._edit_timer.start(0)

    def _edit_next(self) -> None:
        if self._next_edit is not None and self._next_edit.isValid():
            self.edit(QModelIndex(self._next_edit))
        self._next_edit = None


class RoiTableWidget(QWidget):
    """The ROI table with its data file filter, for a dock."""

    #: The user selected an ROI in the table.
    roi_selected = pyqtSignal(str)

    def __init__(self, project_state: ProjectState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_state = project_state
        self.model = RoiTableModel(project_state, self)
        self.model.edit_failed.connect(self._show_error)
        self._file_filter = QComboBox()
        self._file_filter.currentIndexChanged.connect(self._on_filter_changed)
        self._missing = QCheckBox("Only ROIs without a value for")
        self._missing_target = QComboBox()
        self._missing.toggled.connect(self._on_filter_changed)
        self._missing_target.currentIndexChanged.connect(self._on_filter_changed)
        self._count = QLabel("")
        self._count.setStyleSheet(secondary_text_style())
        self.view = _RoiTableView()
        self.view.setModel(self.model)
        self.view.setItemDelegateForColumn(_USE_COLUMN, _SetDelegate(self.view))
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.view.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                  | QAbstractItemView.EditTrigger.EditKeyPressed
                                  | QAbstractItemView.EditTrigger.AnyKeyPressed)
        header = self.view.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # A header click sorts (numbers by value); the indicator starts unset, so the rows
        # keep their data file and name order until then.
        if header is not None:
            header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.view.setSortingEnabled(True)
        self.view.paste_requested.connect(self.paste_clipboard)
        selection = self.view.selectionModel()
        if selection is not None:
            selection.currentRowChanged.connect(self._on_current_row_changed)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Dataset:"))
        filters.addWidget(self._file_filter, 1)
        filters.addWidget(self._missing)
        filters.addWidget(self._missing_target)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(filters)
        layout.addWidget(self.view, 1)
        layout.addWidget(self._count)
        for signal in (project_state.project_loaded, project_state.datasets_changed,
                       project_state.targets_changed, project_state.rois_changed,
                       project_state.target_assignments_changed):
            signal.connect(self._refresh_choices)
        self._refresh_choices()

    # -- filters --------------------------------------------------------

    def show_data_file(self, dataset_id: str | None) -> None:
        """Filter to one data file (``None``: all files)."""
        index = self._file_filter.findData(dataset_id if dataset_id is not None else _ALL_FILES)
        self._file_filter.setCurrentIndex(max(0, index))

    def focus_target(self, target_id: str | None) -> None:
        """Make ``target_id``'s column current, e.g. the target selected in Model."""
        column = self.model.column_of(target_id) if target_id is not None else None
        if column is not None and self.model.rowCount() > 0:
            self.view.setCurrentIndex(self.model.index(0, column))

    def _refresh_choices(self, *_args: Any) -> None:
        state = self.project_state
        current_file = self._file_filter.currentData()
        current_target = self._missing_target.currentData()
        for combo in (self._file_filter, self._missing_target):
            combo.blockSignals(True)
            combo.clear()
        self._file_filter.addItem("All datasets", _ALL_FILES)
        for dataset_id, dataset in sorted(state.datasets.items(),
                                          key=lambda item: item[1].source_path.lower()):
            self._file_filter.addItem(state.data_file_label(dataset_id), dataset_id)
            self._file_filter.setItemData(self._file_filter.count() - 1, dataset.source_path,
                                          Qt.ItemDataRole.ToolTipRole)
        for target_id, target in sorted(state.targets.items(),
                                        key=lambda item: item[1].name.lower()):
            self._missing_target.addItem(target.name, target_id)
        self._file_filter.setCurrentIndex(max(0, self._file_filter.findData(current_file)))
        self._missing_target.setCurrentIndex(
            max(0, self._missing_target.findData(current_target)))
        for combo in (self._file_filter, self._missing_target):
            combo.blockSignals(False)
        self._on_filter_changed()

    def _on_filter_changed(self, *_args: Any) -> None:
        dataset = self._file_filter.currentData()
        self.model.set_dataset_filter(None if dataset in (None, _ALL_FILES) else str(dataset))
        target = self._missing_target.currentData() if self._missing.isChecked() else None
        self.model.set_missing_filter(None if target is None else str(target))
        self._update_count()

    def _update_count(self) -> None:
        state = self.project_state
        rows = self.model.rowCount()
        parts = [f"{rows} ROIs"]
        for target_id in sorted(state.targets, key=lambda t: state.targets[t].name.lower()):
            done = sum((self.model.roi_id(row), target_id) in state.target_assignments
                       for row in range(rows))
            parts.append(f"{state.targets[target_id].name}: {done} of {rows}")
        self._count.setText(" | ".join(parts))

    # -- selection ------------------------------------------------------

    def select_roi(self, roi_id: str | None) -> None:
        """Make the row of ``roi_id`` current (without emitting ``roi_selected``)."""
        row = None if roi_id is None else self.model.row_of(roi_id)
        if row is None:
            return
        column = max(self.view.currentIndex().column(), 1)
        selection = self.view.selectionModel()
        if selection is not None:
            selection.blockSignals(True)
        self.view.setCurrentIndex(self.model.index(row, column))
        if selection is not None:
            selection.blockSignals(False)

    def _on_current_row_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        if current.isValid():
            self.roi_selected.emit(self.model.roi_id(current.row()))

    # -- paste ------------------------------------------------------------

    def paste_clipboard(self, text: str | None = None) -> bool:
        """Fill the current target or Sample column downwards from the clipboard (or ``text``).

        One value per line (the first tab-separated field); an empty line
        removes a value. Every value is checked first: an invalid one, or more
        lines than rows below, writes nothing. Sample labels are text, so any
        text is accepted.
        """
        if text is None:
            clipboard = QGuiApplication.clipboard()
            text = clipboard.text() if clipboard is not None else ""
        current = self.view.currentIndex()
        is_sample = current.isValid() and current.column() == _SAMPLE_COLUMN
        target_id = self.model.target_id(current.column()) if current.isValid() else None
        if target_id is None and not is_sample:
            self._show_error("Select a cell in a target or the Sample column, then paste.")
            return False
        lines = text.splitlines()
        while lines and not lines[-1].strip():
            lines.pop()  # spreadsheets end a copied column with a newline
        if current.row() + len(lines) > self.model.rowCount():
            self._show_error(f"{len(lines)} values do not fit below this row "
                             f"({self.model.rowCount() - current.row()} rows); nothing pasted.")
            return False
        if is_sample:
            samples = [(self.model.roi_id(current.row() + offset),
                        line.split("\t", 1)[0].strip() or None)
                       for offset, line in enumerate(lines)]
            if samples:
                self.model.write_samples(samples)
            return True
        if target_id is None:  # not reached: a sample paste returned above
            return False
        changes: list[tuple[str, str, float | None]] = []
        for offset, line in enumerate(lines):
            field = line.split("\t", 1)[0]
            try:
                value = parse_value(field)
            except ValueError:
                self._show_error(f"Line {offset + 1}: {field!r} is not a number; "
                                 "nothing pasted.")
                return False
            changes.append((self.model.roi_id(current.row() + offset), target_id, value))
        if changes:
            self.model.write_values(changes)
        return True

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "ROI table", message)


__all__ = ["RoiTableModel", "RoiTableWidget", "parse_value"]
