"""Small shared UI helpers for viewer widgets."""

from __future__ import annotations

import platform
import subprocess
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QPoint, QRect, QSize, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .._logging import get_logger
from ._theme import panel_colors

logger = get_logger(__name__)


#: File-dialog filters of project files: saving offers only the project
#: extensions; opening also lists every file, so a project saved without an
#: extension is not hidden.
PROJECT_SAVE_FILTER = "engiNIR project (*.h5 *.hdf5)"
PROJECT_OPEN_FILTER = "engiNIR project (*.h5 *.hdf5);;All files (*)"


def is_band_index_axis(wavelengths: Any) -> bool:
    """Whether ``wavelengths`` is the axis given to a file without wavelengths: 0, 1, ..., n-1."""
    import numpy as np

    values = np.asarray(wavelengths, dtype=float).ravel()
    return values.size > 1 and bool(np.array_equal(values, np.arange(values.size)))


def spectral_axis_label(wavelengths: Any) -> tuple[str, str]:
    """Axis label and CSV unit of a spectral axis: band numbers or wavelengths in nm."""
    return ("Band", "band") if is_band_index_axis(wavelengths) else ("Wavelength (nm)", "nm")


def ask_save_path(parent: QWidget | None, title: str, file_filter: str, suffix: str,
                  accepted: tuple[str, ...] = (), directory: str = "") -> str | None:
    """Ask for a file to write; add ``suffix`` when the name has none of the accepted ones.

    File dialogs on some platforms return the typed name without the
    extension of the chosen filter, and an open dialog filtering on that
    extension would then hide the file. When the suffix is added and that
    file exists, the user confirms replacing it (the dialog only checked the
    typed name). The dialog opens in ``directory`` ("": Qt's default).
    Returns ``None`` when cancelled.
    """
    name, _ = QFileDialog.getSaveFileName(parent, title, directory, file_filter)
    if not name:
        return None
    path = Path(name)
    if path.suffix.lower() in {suffix, *accepted}:
        return str(path)
    path = path.with_name(path.name + suffix)
    if path.exists():
        answer = QMessageBox.question(
            parent, title, f"'{path.name}' already exists.\nDo you want to replace it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return None
    return str(path)


def with_unit(name: str, unit: str | None) -> str:
    """``"API (%)"``; the name alone for a dimensionless unit (``1`` or empty).

    The same rule as the library's verdict sentence (model-workspace-diagnostics
    contract): a dimensionless unit is not printed.
    """
    unit = (unit or "").strip()
    return name if unit in ("", "1") else f"{name} ({unit})"


def style_axes(ax: Any, colors: Mapping[str, str] | None = None) -> None:
    """Apply the shared enginir plot styling to one matplotlib axes.

    ``colors`` defaults to the current theme (:func:`panel_colors`).
    """
    colors = panel_colors() if colors is None else colors
    ax.set_facecolor(colors["ax_bg"])
    for spine in ax.spines.values():
        spine.set_color(colors["fg"])
    ax.tick_params(colors=colors["fg"], labelsize=8)
    ax.xaxis.label.set_color(colors["fg"])
    ax.yaxis.label.set_color(colors["fg"])
    ax.title.set_color(colors["fg"])


#: What a figure's margins were last fitted to: the canvas size and the label space.
LayoutState = tuple[tuple[int, int], tuple[Any, ...]]


def _label_space(fig: Any) -> tuple[Any, ...]:
    """What the margins depend on besides the size: axis titles and the widest tick labels.

    Reading the tick positions does not draw the figure, so this stays cheap
    enough for every frame step.
    """
    space: list[Any] = []
    for ax in fig.get_axes():
        widths = []
        for axis, ticks in ((ax.yaxis, ax.get_yticks()), (ax.xaxis, ax.get_xticks())):
            labels = axis.get_major_formatter().format_ticks(list(ticks)) if len(ticks) else []
            widths.append(max((len(str(label)) for label in labels), default=0))
        space.append((ax.get_ylabel(), ax.get_xlabel(), ax.get_title(), *widths))
    return tuple(space)


def maybe_tight_layout(
    fig: Any,
    canvas: Any,
    *,
    pad: float,
    last_size: LayoutState | None,
) -> LayoutState | None:
    """Fit the margins (tight_layout) when the canvas size or the label space changed.

    Frame steps keep both, so they skip the layout; a new signal unit or
    data with wider tick labels (e.g. from an empty placeholder to
    absorbance) fits the margins again, so no label is clipped.
    """
    size = (int(canvas.width()), int(canvas.height()))
    if size == (0, 0) and last_size is not None:
        return last_size
    state = (size, _label_space(fig))
    if state == last_size and size != (0, 0):
        return last_size
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Tight layout not applied.*",
            category=UserWarning,
        )
        fig.tight_layout(pad=pad)
    return state


def reveal_in_file_manager(path: Path) -> bool:
    """Open the folder of ``path`` in the file manager; return ``False`` on failure.

    Under WSL the folder opens in Windows Explorer with the file selected,
    because the Linux desktop opener usually cannot reach Windows there.
    """
    if "microsoft" in platform.uname().release.lower():
        try:
            windows_path = subprocess.run(
                ["wslpath", "-w", str(path)], capture_output=True, text=True, check=True
            ).stdout.strip()
            # explorer.exe exits with 1 even on success, so it is not waited for or checked.
            subprocess.Popen(["explorer.exe", f"/select,{windows_path}"])
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning("Cannot open {} in Windows Explorer: {}", path, exc)
            return False
        return True
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))


__all__ = [
    "is_band_index_axis",
    "maybe_tight_layout",
    "reveal_in_file_manager",
    "spectral_axis_label",
    "style_axes",
    "with_unit",
]


class FlowLayout(QLayout):
    """A row of widgets that wraps onto further lines when the width runs out.

    For the setting rows of the Fit settings dock (plan gui-1.6, Phase 4): one
    line in a wide window, two or three in the narrow dock.
    """

    def __init__(self, parent: QWidget | None = None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, a0: QLayoutItem | None) -> None:  # noqa: N802 (Qt API)
        if a0 is not None:
            self._items.append(a0)

    def addStretch(self, _stretch: int = 0) -> None:  # noqa: N802 (QBoxLayout's API)
        """Accepted for drop-in use with ``QHBoxLayout`` code; a flow has no stretch."""

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 (Qt API)
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 (Qt API)
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 (Qt API)
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt API)
        return True

    def heightForWidth(self, a0: int) -> int:  # noqa: N802 (Qt API)
        return self._place(QRect(0, 0, a0, 0), move=False)

    def setGeometry(self, a0: QRect) -> None:  # noqa: N802 (Qt API)
        super().setGeometry(a0)
        self._place(a0, move=True)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        # The widest item, not the one-line width: a flow asks for no more than it
        # needs to wrap, and takes the width its parent gives (heightForWidth).
        width = max((item.sizeHint().width() for item in self._items), default=0)
        height = max((item.sizeHint().height() for item in self._items), default=0)
        return QSize(width, height)

    def minimumSize(self) -> QSize:  # noqa: N802 (Qt API)
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _place(self, rect: QRect, *, move: bool) -> int:
        x, y, line = rect.x(), rect.y(), 0
        space = self.spacing()
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            if x > rect.x() and x + hint.width() > rect.right() + 1:
                x, y, line = rect.x(), y + line + space, 0
            if move:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + space
            line = max(line, hint.height())
        return y + line - rect.y()


class CollapsibleSection(QWidget):
    """A titled part of a settings form that folds to its header.

    The header shows a one-line summary of the folded settings (``set_summary``),
    so a folded section still says what a fit would use.
    """

    #: Emitted when the section is folded or unfolded. Arg: expanded.
    toggled = pyqtSignal(bool)

    def __init__(self, title: str, content: QWidget, *, expanded: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._content = content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self.header = QToolButton()
        self.header.setObjectName("SectionHeader")
        self.header.setText(title)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setCheckable(True)
        self.header.setAutoRaise(True)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.toggled.connect(self._on_toggled)
        header.addWidget(self.header, 1)
        layout.addLayout(header)
        self._summary = QLabel("")
        self._summary.setObjectName("SectionSummary")
        self._summary.setWordWrap(True)
        self._summary.setContentsMargins(18, 0, 0, 2)
        layout.addWidget(self._summary)
        layout.addWidget(content)
        self._summary_text = ""
        self.header.setChecked(expanded)
        self._on_toggled(expanded)

    @property
    def title(self) -> str:
        return self.header.text()

    @property
    def expanded(self) -> bool:
        return self.header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(expanded)

    def set_summary(self, text: str) -> None:
        """What the folded section holds, in a few words (shown only when folded)."""
        self._summary_text = text
        self._summary.setText(text)
        self._summary.setVisible(bool(text) and not self.expanded)

    @property
    def summary(self) -> str:
        return self._summary_text

    def _on_toggled(self, expanded: bool) -> None:
        self.header.setArrowType(Qt.ArrowType.DownArrow if expanded
                                 else Qt.ArrowType.RightArrow)
        self._content.setVisible(expanded)
        self._summary.setVisible(bool(self._summary_text) and not expanded)
        self.toggled.emit(expanded)
