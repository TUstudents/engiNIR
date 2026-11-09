"""Paged grids of plot panels with in-place maximize (Plot Panel Policy).

A :class:`PlotGrid` shows the pages of one workspace as sub-tabs. Each page
places its panels in fixed slots. A page builds its panels the first time it
is shown; maximize moves one panel over the whole page, and restore (the
same control, a double-click on the header or ``Esc``) puts it back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QGridLayout, QStackedLayout, QTabWidget, QVBoxLayout, QWidget

from .._checks import log_and_raise
from .._logging import get_logger
from ._plot_panel import PlotFrame, PlotPanel, PlotSpec

logger = get_logger(__name__)


@dataclass(frozen=True)
class PlotSlot:
    """Position of one plot in a page grid."""

    plot_id: str
    row: int
    column: int
    row_span: int = 1
    column_span: int = 1

    def cells(self) -> set[tuple[int, int]]:
        return {(row, column)
                for row in range(self.row, self.row + self.row_span)
                for column in range(self.column, self.column + self.column_span)}


@dataclass(frozen=True)
class PlotPage:
    """A named page of plot slots; slots must not overlap or repeat a plot."""

    name: str
    slots: tuple[PlotSlot, ...]

    def __post_init__(self) -> None:
        if not self.slots:
            log_and_raise("Plot page '{}' has no slots", self.name)
        seen: set[tuple[int, int]] = set()
        for slot in self.slots:
            if slot.row < 0 or slot.column < 0 or slot.row_span < 1 or slot.column_span < 1:
                log_and_raise("Plot page '{}': slot of '{}' has an invalid position",
                              self.name, slot.plot_id)
            if seen & slot.cells():
                log_and_raise("Plot page '{}': slot of '{}' overlaps another slot",
                              self.name, slot.plot_id)
            seen |= slot.cells()
        plot_ids = [slot.plot_id for slot in self.slots]
        if len(set(plot_ids)) != len(plot_ids):
            log_and_raise("Plot page '{}' shows a plot twice: {}", self.name, plot_ids)

    @property
    def plot_ids(self) -> tuple[str, ...]:
        return tuple(slot.plot_id for slot in self.slots)


class PlotPageWidget(QWidget):
    """One page: plot frames in their slots, plus a layer that holds a maximized frame.

    ``frames`` maps each slot's plot id to its frame (a :class:`PlotPanel` or
    any other :class:`PlotFrame`).
    """

    #: The maximized plot id, or ``None`` after restore.
    maximized_changed = pyqtSignal(object)

    def __init__(self, page: PlotPage, frames: Mapping[str, PlotFrame],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        missing = [plot_id for plot_id in page.plot_ids if plot_id not in frames]
        if missing:
            log_and_raise("Plot page '{}' has no frames for {}", page.name, missing)
        self.page = page
        self.panels: dict[str, PlotFrame] = {}
        self.maximized: str | None = None

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(6)
        self._max_host = QWidget()
        self._max_layout = QVBoxLayout(self._max_host)
        self._max_layout.setContentsMargins(0, 0, 0, 0)
        self._stack = QStackedLayout(self)
        self._stack.addWidget(self._grid_host)
        self._stack.addWidget(self._max_host)

        for slot in page.slots:
            panel = frames[slot.plot_id]
            panel.maximize_toggled.connect(
                lambda plot_id=slot.plot_id: self.toggle_maximized(plot_id))
            self.panels[slot.plot_id] = panel
            self._place(slot, panel)
        rows = max(slot.row + slot.row_span for slot in page.slots)
        columns = max(slot.column + slot.column_span for slot in page.slots)
        for row in range(rows):
            self._grid.setRowStretch(row, 1)
        for column in range(columns):
            self._grid.setColumnStretch(column, 1)

        self._escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._escape.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._escape.activated.connect(self.restore)
        self._escape.setEnabled(False)

    def _place(self, slot: PlotSlot, panel: PlotFrame) -> None:
        self._grid.addWidget(panel, slot.row, slot.column, slot.row_span, slot.column_span)

    def toggle_maximized(self, plot_id: str) -> None:
        if self.maximized == plot_id:
            self.restore()
        else:
            self.maximize(plot_id)

    def maximize(self, plot_id: str) -> None:
        if plot_id not in self.panels:
            log_and_raise("Plot page '{}' has no plot '{}'", self.page.name, plot_id)
        self.restore()
        panel = self.panels[plot_id]
        self._grid.removeWidget(panel)
        self._max_layout.addWidget(panel)
        self._stack.setCurrentWidget(self._max_host)
        panel.set_maximized(True)
        self.maximized = plot_id
        self._escape.setEnabled(True)
        self.maximized_changed.emit(plot_id)

    def restore(self) -> None:
        if self.maximized is None:
            return
        panel = self.panels[self.maximized]
        slot = next(slot for slot in self.page.slots if slot.plot_id == self.maximized)
        self._max_layout.removeWidget(panel)
        self._place(slot, panel)
        self._stack.setCurrentWidget(self._grid_host)
        panel.set_maximized(False)
        self.maximized = None
        self._escape.setEnabled(False)
        self.maximized_changed.emit(None)


class PlotGrid(QWidget):
    """The plot pages of one workspace, as sub-tabs (one page: no tab bar)."""

    roi_selected = pyqtSignal(str)
    #: A plot was maximized (its id) or restored (``None``).
    maximized_changed = pyqtSignal(object)

    def __init__(self, pages: Sequence[PlotPage], specs: Mapping[str, PlotSpec],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if not pages:
            log_and_raise("A plot grid needs at least one page")
        names = [page.name for page in pages]
        if len(set(names)) != len(names):
            log_and_raise("Plot page names repeat: {}", names)
        for page in pages:
            unknown = [plot_id for plot_id in page.plot_ids if plot_id not in specs]
            if unknown:
                log_and_raise("Plot page '{}' names unknown plots: {}", page.name, unknown)
        self._pages = tuple(pages)
        self._specs = specs
        self._built: dict[str, PlotPageWidget] = {}
        self._payloads: dict[str, Any] = {}
        self._empty_reason = ""
        self._provenance: dict[str, Any] = {}

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        tab_bar = self._tabs.tabBar()
        if tab_bar is not None:
            tab_bar.setAutoHide(True)
            tab_bar.setDrawBase(False)  # the theme draws no base; Qt's would be a white line
        for page in self._pages:
            self._tabs.addTab(QWidget(), page.name)
        self._tabs.currentChanged.connect(self._ensure_page)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabs)
        self._ensure_page(0)

    @property
    def page_names(self) -> tuple[str, ...]:
        return tuple(page.name for page in self._pages)

    @property
    def current_page(self) -> str:
        return self._pages[self._tabs.currentIndex()].name

    def set_current_page(self, name: str) -> None:
        self._tabs.setCurrentIndex(self._page_index(name))

    def panel(self, plot_id: str) -> PlotPanel | None:
        """The panel of ``plot_id``, or ``None`` while its page is not built."""
        for page in self._built.values():
            panel = page.panels.get(plot_id)
            if isinstance(panel, PlotPanel):
                return panel
        return None

    def all_panels(self) -> dict[str, PlotPanel]:
        """Every panel of every page, building pages that were not shown yet.

        The current page stays current; panels of hidden pages do not draw
        until shown, but they export (an export draws its own figure).
        """
        current = self._tabs.currentIndex()
        for index in range(len(self._pages)):
            self._ensure_page(index)
        blocked = self._tabs.blockSignals(True)
        self._tabs.setCurrentIndex(current)
        self._tabs.blockSignals(blocked)
        return {plot_id: panel for page in self._built.values()
                for plot_id, panel in page.panels.items() if isinstance(panel, PlotPanel)}

    def maximized_plot(self) -> str | None:
        page = self._built.get(self.current_page)
        return None if page is None else page.maximized

    def maximize(self, plot_id: str) -> None:
        """Show the page of ``plot_id`` with that plot maximized."""
        page = next((page for page in self._pages if plot_id in page.plot_ids), None)
        if page is None:
            log_and_raise("No plot page shows '{}'", plot_id)
        self.set_current_page(page.name)
        self._built[page.name].maximize(plot_id)

    def restore(self) -> None:
        self.restore_page(self.current_page)

    def restore_page(self, name: str) -> None:
        """Restore the maximized plot of page ``name``, if it was built."""
        page = self._built.get(name)
        if page is not None:
            page.restore()

    @property
    def plot_ids(self) -> tuple[str, ...]:
        return tuple(plot_id for page in self._pages for plot_id in page.plot_ids)

    def update_data(
        self,
        payloads: Mapping[str, Any],
        *,
        empty_reason: str = "",
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Give each plot its payload (missing ids show the empty state).

        Panels that are hidden draw when shown; an unchanged payload is a no-op.
        """
        self._payloads = dict(payloads)
        self._empty_reason = empty_reason
        if provenance is not None:
            self._provenance = dict(provenance)
        for page in self._built.values():
            self._feed(page)

    def restyle_theme(self) -> None:
        for page in self._built.values():
            for panel in page.panels.values():
                if isinstance(panel, PlotPanel):
                    panel.restyle_theme()

    def _feed(self, page: PlotPageWidget) -> None:
        for plot_id, panel in page.panels.items():
            if not isinstance(panel, PlotPanel):
                continue
            panel.update_data(self._payloads.get(plot_id), empty_reason=self._empty_reason,
                              provenance=self._provenance)

    def _page_index(self, name: str) -> int:
        for index, page in enumerate(self._pages):
            if page.name == name:
                return index
        log_and_raise("Unknown plot page '{}'; pages are {}", name, list(self.page_names),
                      depth=2)

    def _ensure_page(self, index: int) -> None:
        if index < 0:
            return
        page = self._pages[index]
        if page.name in self._built:
            return
        widget = PlotPageWidget(page, {plot_id: PlotPanel(self._specs[plot_id])
                                       for plot_id in page.plot_ids})
        for panel in widget.panels.values():
            if isinstance(panel, PlotPanel):
                panel.roi_selected.connect(self.roi_selected.emit)
        widget.maximized_changed.connect(self.maximized_changed.emit)
        self._built[page.name] = widget
        self._feed(widget)
        placeholder = self._tabs.widget(index)
        blocked = self._tabs.blockSignals(True)
        self._tabs.removeTab(index)
        self._tabs.insertTab(index, widget, page.name)
        self._tabs.setCurrentIndex(index)
        self._tabs.blockSignals(blocked)
        if placeholder is not None:
            placeholder.deleteLater()
        logger.debug("Built plot page '{}' with {} panels", page.name, len(widget.panels))


__all__ = ["PlotGrid", "PlotPage", "PlotPageWidget", "PlotSlot"]
