"""Small shared UI helpers for viewer widgets."""

from __future__ import annotations

import platform
import subprocess
import warnings
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices

from .._logging import get_logger
from ._theme import panel_colors

logger = get_logger(__name__)


def style_axes(ax: Any) -> None:
    """Apply the shared enginir plot styling to one matplotlib axes."""
    colors = panel_colors()
    ax.set_facecolor(colors["ax_bg"])
    for spine in ax.spines.values():
        spine.set_color(colors["fg"])
    ax.tick_params(colors=colors["fg"], labelsize=8)
    ax.xaxis.label.set_color(colors["fg"])
    ax.yaxis.label.set_color(colors["fg"])
    ax.title.set_color(colors["fg"])


def maybe_tight_layout(
    fig: Any,
    canvas: Any,
    *,
    pad: float,
    last_size: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Apply tight_layout only when the canvas size changed."""
    size = (int(canvas.width()), int(canvas.height()))
    if size == last_size and size != (0, 0):
        return last_size
    if size == (0, 0) and last_size is not None:
        return last_size
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Tight layout not applied.*",
            category=UserWarning,
        )
        fig.tight_layout(pad=pad)
    return size


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
