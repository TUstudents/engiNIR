"""Plot panels: one plot with a header for maximize and export.

Implements the Plot Panel Policy of ``viewer-workspace-ui-contract_v3.md``.
A plot is defined by a :class:`PlotSpec`: a function that draws a payload
into a matplotlib figure and a function that returns the plotted values as a
:class:`PlotTable`. A :class:`PlotPanel` shows one spec with a header (title,
maximize, export menu). The screen and the exported PNG/SVG are drawn by the
same function, and the CSV comes from the same payload, so an export always
contains what the panel shows.
"""

from __future__ import annotations

import csv
import io
import pickle
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QGuiApplication, QImage, QMouseEvent, QShowEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from .._checks import log_and_raise
from .._logging import get_logger
from ._theme import panel_colors, print_colors
from ._ui_helpers import style_axes

logger = get_logger(__name__)

#: Exports with more values than this ask for confirmation (Plot Panel Policy).
LARGE_EXPORT_VALUES = 1_000_000
_IMAGE_DPI = 300
_IMAGE_SUFFIXES = (".png", ".svg")
_NO_DATA = "No data to export."
_PICK_RADIUS_PX = 12.0
_LAYOUT_PAD = 1.0
_HEADER_BUTTON = 20  # px; the header costs about as much height as an in-plot title


@dataclass(frozen=True)
class PlotColumn:
    """One CSV column: a name, its values and an optional unit."""

    name: str
    values: Any
    unit: str = ""

    @property
    def header(self) -> str:
        return f"{self.name} ({self.unit})" if self.unit else self.name


@dataclass(frozen=True)
class PlotTable:
    """The plotted values of one plot, as written to CSV.

    Either ``columns`` (one row per point, bin or band) or ``matrix`` (an image
    or map, one CSV row per ``row_label`` and one column per ``column_label``).
    ``row_index`` and ``column_index`` label the matrix rows and columns
    (default ``0, 1, ...``), e.g. the true frames of a subsampled map.
    ``metadata`` adds plot-specific provenance lines to the CSV header.
    """

    columns: tuple[PlotColumn, ...] = ()
    matrix: Any = None
    row_label: str = "row"
    column_label: str = "frame"
    row_index: Any = None
    column_index: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.matrix is not None:
            if self.columns:
                log_and_raise("A plot table has either columns or a matrix, not both")
            if np.ndim(self.matrix) != 2:
                log_and_raise("A plot table matrix must be 2-D, got {} dimensions",
                              np.ndim(self.matrix))
            n_rows, n_columns = np.shape(self.matrix)
            for name, index, n in (("row_index", self.row_index, n_rows),
                                   ("column_index", self.column_index, n_columns)):
                if index is not None and len(np.ravel(index)) != n:
                    log_and_raise("Plot table {} has {} labels for {} matrix entries",
                                  name, len(np.ravel(index)), n)
            return
        lengths = {column.header: len(np.ravel(column.values)) for column in self.columns}
        if len(set(lengths.values())) > 1:
            log_and_raise("Plot table columns differ in length: {}", lengths)

    @property
    def n_values(self) -> int:
        """Number of values the CSV holds."""
        if self.matrix is not None:
            return int(np.size(self.matrix))
        if not self.columns:
            return 0
        return len(self.columns) * len(np.ravel(self.columns[0].values))

    @property
    def is_empty(self) -> bool:
        return self.n_values == 0

    def write_csv(self, path: str | Path, provenance: Mapping[str, Any]) -> None:
        """Write ``# key: value`` provenance lines, then the table."""
        with open(path, "w", encoding="utf-8", newline="") as handle:
            for key, value in {**provenance, **self.metadata}.items():
                text = " ".join(str(value).split())  # one line per entry
                handle.write(f"# {key}: {text}\n")
            writer = csv.writer(handle)
            if self.matrix is not None:
                matrix = np.asarray(self.matrix)
                rows = (range(matrix.shape[0]) if self.row_index is None
                        else np.ravel(self.row_index).tolist())
                columns = (range(matrix.shape[1]) if self.column_index is None
                           else np.ravel(self.column_index).tolist())
                writer.writerow([f"{self.row_label}\\{self.column_label}", *columns])
                for index, row in zip(rows, matrix, strict=True):
                    writer.writerow([index, *_format_values(row)])
                return
            writer.writerow([column.header for column in self.columns])
            columns = [_format_values(np.ravel(column.values)) for column in self.columns]
            writer.writerows(zip(*columns, strict=True))


def _format_values(values: np.ndarray) -> list[str]:
    """Text for CSV cells; floats round-trip at their own precision."""
    array = np.asarray(values)
    if array.dtype == np.float32:
        return [format(float(value), ".9g") for value in array]
    if np.issubdtype(array.dtype, np.floating):
        return [repr(float(value)) for value in array]
    return [str(value) for value in array.tolist()]


@dataclass(frozen=True)
class RoiPoints:
    """Points of one axes that select an ROI when clicked."""

    ax: Axes
    x: np.ndarray
    y: np.ndarray
    roi_ids: tuple[str, ...]


#: Draws a payload into an empty figure with the given colors. It returns the
#: clickable ROI points, if any. It must not set axes titles: the panel header
#: shows the title.
RenderFn = Callable[[Figure, Any, Mapping[str, str]], Sequence[RoiPoints] | None]


@dataclass(frozen=True)
class PlotSpec:
    """Definition of one plot.

    ``render`` is called only with a payload that is not ``None``; it draws an
    empty-state message itself when the payload holds no data for this plot.
    ``table`` returns the plotted values; an empty table disables export.
    """

    plot_id: str
    title: str
    render: RenderFn
    table: Callable[[Any], PlotTable]
    empty_text: str = "No data yet."
    #: How to read the plot (what, at which level, a bad sign); shown by the header's ?.
    help: str = ""


def draw_placeholder(ax: Axes, message: str, colors: Mapping[str, str]) -> None:
    """Write an empty-state message in the middle of ``ax``."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            color=colors["fg"], fontsize=8, wrap=True)


class _PlotHeader(QWidget):
    double_clicked = pyqtSignal()

    def mouseDoubleClickEvent(self, a0: QMouseEvent | None) -> None:  # noqa: N802 (Qt API)
        self.double_clicked.emit()
        if a0 is not None:
            a0.accept()


class PlotSource(Protocol):
    """What a plot frame needs from the plot it shows."""

    def plot_table(self) -> PlotTable | None:
        """The plotted values, or ``None`` when nothing is plotted."""
        ...

    def print_figure(self) -> Figure:
        """A new figure of the current view in :func:`print_colors`."""
        ...


class PlotFrame(QWidget):
    """A plot under a header: title, maximize/restore and the export menu.

    The header works for any plot that implements :class:`PlotSource`; a
    :class:`PlotPanel` is the frame of a :class:`PlotSpec`. ``provenance``
    callables supply the CSV provenance lines at export time.
    """

    maximize_toggled = pyqtSignal()

    def __init__(
        self,
        plot_id: str,
        title: str,
        body: QWidget,
        source: PlotSource | None = None,
        *,
        provenance: Callable[[], Mapping[str, Any]] | None = None,
        help_text: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.plot_id = plot_id
        self.help_text = help_text
        self._source = source
        self._provenance_provider = provenance
        self._provenance: dict[str, Any] = {}
        self._maximized = False

        self._header = _PlotHeader(self)
        self._header.double_clicked.connect(self.maximize_toggled.emit)
        self._header_layout = QHBoxLayout(self._header)
        self._header_layout.setContentsMargins(6, 0, 2, 0)
        self._header_layout.setSpacing(2)
        self._title = QLabel(title)
        self._title.setStyleSheet("font-weight: 600;")
        self._max_button = self._header_button("")
        self._max_button.clicked.connect(self.maximize_toggled.emit)
        self._menu = QMenu(self)
        self._menu.setToolTipsVisible(True)
        self._menu.aboutToShow.connect(self._update_export_actions)
        self._png_action = self._add_action("Save PNG...", lambda: self._save_image(".png"))
        self._svg_action = self._add_action("Save SVG...", lambda: self._save_image(".svg"))
        self._csv_action = self._add_action("Export CSV...", self._save_csv)
        self._copy_action = self._add_action("Copy image", self._copy_image)
        menu_button = self._header_button("⋯")
        menu_button.setToolTip("Export")
        menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu_button.setMenu(self._menu)
        self._header.setFixedHeight(_HEADER_BUTTON + 4)
        self._header_layout.addWidget(self._title, 1)
        self._help_button: QToolButton | None = None
        if help_text:
            self._help_button = self._header_button("?")
            self._help_button.setToolTip(help_text)
            self._help_button.clicked.connect(self.show_help)
            self._header_layout.addWidget(self._help_button)
        self._header_layout.addWidget(self._max_button)
        self._header_layout.addWidget(menu_button)
        self.set_maximized(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(body, 1)
        title_changed = getattr(source, "title_changed", None)
        if title_changed is not None:  # e.g. the RAW panel's band, the map's unit
            title_changed.connect(self.set_title)
        self._update_export_actions()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def show_help(self) -> None:
        """Show the help text next to the ? button (a tooltip that stays until moved away)."""
        if self._help_button is None:
            return
        anchor = self._help_button.mapToGlobal(self._help_button.rect().bottomLeft())
        QToolTip.showText(anchor, self.help_text, self._help_button)

    @property
    def title(self) -> str:
        return self._title.text()

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def take_header(self) -> QWidget:
        """Remove the header from the frame, e.g. to use it as a dock title bar."""
        layout = self.layout()
        if layout is not None:
            layout.removeWidget(self._header)
        return self._header

    def add_menu_action(self, text: str, slot: Callable[[], None]) -> QAction:
        """Add a plot-specific entry to the ⋯ menu, after the exports."""
        self._menu.addSeparator()
        return self._add_action(text, slot)

    def add_header_widget(self, widget: QWidget) -> None:
        """Add a control to the header, left of the maximize button."""
        self._header_layout.insertWidget(self._header_layout.indexOf(self._max_button), widget)

    def add_close_button(self, close: Callable[[], None]) -> None:
        """Add a close control (for a frame used as a dock title bar)."""
        button = self._header_button("×")
        button.setToolTip("Close")
        button.clicked.connect(lambda _checked=False: close())
        self._header_layout.addWidget(button)

    def _header_button(self, text: str) -> QToolButton:
        button = QToolButton()
        button.setAutoRaise(True)
        button.setText(text)
        button.setFixedSize(_HEADER_BUTTON, _HEADER_BUTTON)
        button.setStyleSheet("QToolButton { padding: 0px; margin: 0px; }"
                             "QToolButton::menu-indicator { image: none; width: 0px; }")
        return button

    @property
    def is_maximized(self) -> bool:
        return self._maximized

    def set_maximized(self, maximized: bool) -> None:
        """Show the maximize or the restore control (the owner does the layout)."""
        self._maximized = maximized
        self._max_button.setText("⤡" if maximized else "⤢")
        self._max_button.setToolTip("Restore (Esc)" if maximized else "Maximize")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def table(self) -> PlotTable | None:
        """The plotted values, or ``None`` when the frame shows no data."""
        return None if self._source is None else self._source.plot_table()

    def print_figure(self) -> Figure:
        """The current view as a new figure in the print style."""
        if self._source is None:
            log_and_raise("Plot '{}' has nothing to print", self.plot_id)
        return self._source.print_figure()

    def set_provenance(self, provenance: Mapping[str, Any]) -> None:
        self._provenance = dict(provenance)

    def export_reason(self) -> str:
        """Why export is disabled, or ``""`` when it is possible."""
        table = self.table()
        return _NO_DATA if table is None or table.is_empty else ""

    def export_image(self, path: str | Path, *, dpi: int = _IMAGE_DPI) -> None:
        """Save the plot as PNG or SVG (by suffix) in the print style."""
        path = Path(path)
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            log_and_raise("Plot images are saved as {}, not '{}'",
                          " or ".join(_IMAGE_SUFFIXES), path.suffix)
        self._require_data()
        fig = self.print_figure()
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        logger.info("Saved plot '{}' to {}", self.plot_id, path)

    def export_csv(self, path: str | Path) -> None:
        """Write the plotted values with the provenance header."""
        table = self._require_data()
        provided = {} if self._provenance_provider is None else self._provenance_provider()
        provenance = {
            "plot": self.plot_id,
            "title": self.title,
            "enginir_version": __version__,
            "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **provided,
            **self._provenance,
        }
        table.write_csv(path, provenance)
        logger.info("Exported data of plot '{}' to {}", self.plot_id, path)

    def _require_data(self) -> PlotTable:
        table = self.table()
        if table is None or table.is_empty:
            log_and_raise("Plot '{}' has no data to export", self.plot_id, depth=2)
        return table

    def _add_action(self, text: str, slot: Callable[[], None]) -> QAction:
        action = QAction(text, self)
        action.triggered.connect(lambda _checked=False: slot())
        self._menu.addAction(action)
        return action

    def _update_export_actions(self) -> None:
        reason = self.export_reason()
        for action in (self._png_action, self._svg_action, self._csv_action,
                       self._copy_action):
            action.setEnabled(not reason)
            action.setToolTip(reason)

    def _save_image(self, suffix: str) -> None:
        path = self._ask_path(suffix, "PNG image (*.png)" if suffix == ".png"
                              else "SVG image (*.svg)")
        if path is not None:
            self._run_export(lambda: self.export_image(path))

    def _save_csv(self) -> None:
        table = self.table()
        if table is not None and table.n_values > LARGE_EXPORT_VALUES:
            answer = QMessageBox.question(
                self, "Large export",
                f"This export holds {table.n_values:,} values. Write it anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        path = self._ask_path(".csv", "CSV (*.csv)")
        if path is not None:
            self._run_export(lambda: self.export_csv(path))

    def _copy_image(self) -> None:
        def copy() -> None:
            buffer = io.BytesIO()
            self._require_data()
            self.print_figure().savefig(buffer, format="png", dpi=150)
            clipboard = QGuiApplication.clipboard()
            if clipboard is not None:
                clipboard.setImage(QImage.fromData(buffer.getvalue(), "PNG"))
        self._run_export(copy)

    def _ask_path(self, suffix: str, file_filter: str) -> Path | None:
        name, _ = QFileDialog.getSaveFileName(
            self, f"Export {self.title}", f"{self.plot_id}{suffix}", file_filter)
        if not name:
            return None
        path = Path(name)
        return path if path.suffix.lower() == suffix else path.with_name(path.name + suffix)

    def _run_export(self, export: Callable[[], None]) -> None:
        try:
            export()
        except (OSError, ValueError) as exc:
            logger.error("Export of plot '{}' failed: {}", self.plot_id, exc)
            QMessageBox.warning(self, "Export failed", str(exc))


class PlotPanel(PlotFrame):
    """The frame of one :class:`PlotSpec`: draws its payload and exports it."""

    roi_selected = pyqtSignal(str)
    #: The ROI point under the cursor (its id), or ``""`` when there is none.
    roi_hovered = pyqtSignal(str)

    def __init__(self, spec: PlotSpec, parent: QWidget | None = None) -> None:
        self.spec = spec
        self._payload: Any = None
        self._empty_reason = ""
        self._rendered: tuple[int, str] | None = None
        self._pending = True
        self._layout_size: tuple[int, int] | None = None
        self._roi_points: list[RoiPoints] = []
        self._table: PlotTable | None = None

        self._fig = Figure(figsize=(6, 4), facecolor=panel_colors()["bg"])
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setMinimumSize(120, 90)
        self._canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self._canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self._canvas.mpl_connect("resize_event", self._on_canvas_resize)
        self._hovered_roi = ""
        super().__init__(spec.plot_id, spec.title, self._canvas, help_text=spec.help,
                         parent=parent)

    # ------------------------------------------------------------------
    # Data and drawing
    # ------------------------------------------------------------------

    @property
    def payload(self) -> Any:
        return self._payload

    @property
    def figure(self) -> Figure:
        return self._fig

    def update_data(
        self,
        payload: Any,
        *,
        empty_reason: str = "",
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Show ``payload``; ``None`` shows the empty state with ``empty_reason``.

        The same payload object with the same reason is a no-op. A hidden
        panel draws when it is shown.
        """
        if provenance is not None:
            self.set_provenance(provenance)
        if self._rendered == (id(payload), empty_reason) and payload is self._payload:
            return
        self._payload = payload
        self._empty_reason = empty_reason
        self._table = None
        self._pending = True
        self._update_export_actions()
        if self.isVisible():
            self._render()

    def clear(self, *, empty_reason: str = "") -> None:
        """Show the empty state."""
        self.update_data(None, empty_reason=empty_reason)

    def restyle_theme(self) -> None:
        """Redraw with the current theme colors."""
        self._rendered = None
        self._pending = True
        if self.isVisible():
            self._render()

    def showEvent(self, a0: QShowEvent | None) -> None:  # noqa: N802 (Qt API)
        super().showEvent(a0)
        if self._pending:
            self._render()

    def _render(self) -> None:
        colors = panel_colors()
        new_content = self._rendered is None or self._rendered[0] != id(self._payload)
        self._fig.clear()
        self._fig.set_facecolor(colors["bg"])
        self._roi_points = list(self._draw(self._fig, colors))
        size = (int(self._canvas.width()), int(self._canvas.height()))
        if new_content or size != self._layout_size:
            _tight_layout(self._fig)
            self._layout_size = size
        self._canvas.draw_idle()
        self._rendered = (id(self._payload), self._empty_reason)
        self._pending = False

    def _on_canvas_resize(self, _event: Any) -> None:
        # Margins follow the new size (e.g. after maximize); the canvas redraws itself.
        size = (int(self._canvas.width()), int(self._canvas.height()))
        if self._rendered is not None and size != self._layout_size and min(size) > 0:
            _tight_layout(self._fig)
            self._layout_size = size

    def _draw(self, fig: Figure, colors: Mapping[str, str]) -> Sequence[RoiPoints]:
        if self._payload is None:
            ax = fig.add_subplot(111)
            style_axes(ax, colors)
            ax.set_xticks([])
            ax.set_yticks([])
            draw_placeholder(ax, self._empty_reason or self.spec.empty_text, colors)
            return ()
        return self.spec.render(fig, self._payload, colors) or ()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def table(self) -> PlotTable | None:
        if self._payload is None:
            return None
        if self._table is None:
            self._table = self.spec.table(self._payload)
        return self._table

    def print_figure(self) -> Figure:
        fig = Figure(figsize=tuple(self._fig.get_size_inches()), facecolor=print_colors()["bg"])
        FigureCanvasAgg(fig)
        self._draw(fig, print_colors())
        # The export shows the current view: copy the axis limits of the screen.
        if len(fig.axes) == len(self._fig.axes):
            for shown, printed in zip(self._fig.axes, fig.axes, strict=True):
                printed.set_xlim(shown.get_xlim())
                printed.set_ylim(shown.get_ylim())
        _tight_layout(fig)
        return fig

    # ------------------------------------------------------------------
    # ROI picking
    # ------------------------------------------------------------------

    def _roi_at(self, event: Any) -> str:
        """The id of the ROI point within the pick radius of a mouse event, or ``""``."""
        if event.x is None or event.y is None:
            return ""
        for points in self._roi_points:
            if event.inaxes is not points.ax or points.x.size == 0:
                continue
            positions = points.ax.transData.transform(np.column_stack([points.x, points.y]))
            distances = np.linalg.norm(positions - np.array([event.x, event.y]), axis=1)
            nearest = int(np.nanargmin(distances)) if np.isfinite(distances).any() else -1
            if nearest >= 0 and distances[nearest] <= _PICK_RADIUS_PX:
                return points.roi_ids[nearest]
        return ""

    def _on_canvas_click(self, event: Any) -> None:
        roi_id = self._roi_at(event)
        if roi_id:
            self.roi_selected.emit(roi_id)

    def _on_canvas_motion(self, event: Any) -> None:
        roi_id = self._roi_at(event)
        if roi_id != self._hovered_roi:
            self._hovered_roi = roi_id
            self.roi_hovered.emit(roi_id)


def print_copy(fig: Figure) -> Figure:
    """A copy of a screen figure in the print colors, for figures drawn in place.

    Background, axes, labels and texts take :func:`print_colors`; artists drawn
    in the screen accent color take the print accent. Data colors (images,
    colormaps, crosshairs) are kept.
    """
    screen, printed = panel_colors(), print_colors()
    copy = pickle.loads(pickle.dumps(fig))
    FigureCanvasAgg(copy)
    copy.set_facecolor(printed["bg"])
    accent = to_hex(screen["accent"])
    for ax in copy.axes:
        style_axes(ax, printed)
        for text in ax.texts:
            text.set_color(printed["fg"])
        for line in ax.lines:
            if to_hex(line.get_color()) == accent:
                line.set_color(printed["accent"])
        for patch in ax.patches:
            if to_hex(patch.get_facecolor()) == accent:
                patch.set_facecolor(printed["accent"])
    _tight_layout(copy)
    return copy


def _tight_layout(fig: Figure) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied.*",
                                category=UserWarning)
        fig.tight_layout(pad=_LAYOUT_PAD)


__all__ = [
    "LARGE_EXPORT_VALUES",
    "PlotColumn",
    "PlotFrame",
    "PlotPanel",
    "PlotSource",
    "PlotSpec",
    "PlotTable",
    "RoiPoints",
    "draw_placeholder",
    "print_copy",
]
