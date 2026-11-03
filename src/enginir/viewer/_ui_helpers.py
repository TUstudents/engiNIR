"""Small shared UI helpers for viewer widgets."""

from __future__ import annotations

import platform
import subprocess
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices

from .._logging import get_logger
from ._theme import panel_colors

logger = get_logger(__name__)


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
    "maybe_tight_layout",
    "reveal_in_file_manager",
    "style_axes",
]
