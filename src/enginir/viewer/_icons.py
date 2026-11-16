"""Line icons of the viewer (Tabler Icons, MIT licence, ``_resources/icons/LICENSE``).

The SVGs draw with ``currentColor``; :func:`icon` renders them in a theme
colour (``text_secondary`` by default), so menus, the workflow strip and the
status bar use one monochrome set that follows the dark and light theme.
"""

from __future__ import annotations

import importlib.resources
from functools import lru_cache

from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from .._logging import get_logger
from ._theme import is_dark, tokens

logger = get_logger(__name__)

_SIZES = (16, 20, 24, 32)


@lru_cache(maxsize=1)
def _sources() -> dict[str, str]:
    """The SVG text of every bundled icon, by name."""
    try:
        folder = importlib.resources.files("enginir._resources").joinpath("icons")
        return {entry.name.removesuffix(".svg"): entry.read_text(encoding="utf-8")
                for entry in folder.iterdir() if entry.name.endswith(".svg")}
    except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
        logger.debug("No viewer icons: {}", exc)
        return {}


def icon_names() -> list[str]:
    """The names of the bundled icons."""
    return sorted(_sources())


def icon(name: str, color: str | None = None) -> QIcon:
    """The icon ``name`` in ``color`` (a token name or a colour; default ``text_secondary``).

    An unknown name gives an empty icon (logged), never an error.
    """
    return _icon(name, color or "text_secondary", is_dark())


@lru_cache(maxsize=512)
def _icon(name: str, color: str, _dark: bool) -> QIcon:
    source = _sources().get(name)
    if source is None:
        logger.debug("Unknown viewer icon '{}'", name)
        return QIcon()
    fill = tokens().get(color, color)
    renderer = QSvgRenderer(QByteArray(source.replace("currentColor", fill).encode("utf-8")))
    result = QIcon()
    for size in _SIZES:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter, QRectF(0, 0, size, size))
        painter.end()
        result.addPixmap(pixmap)
    return result
