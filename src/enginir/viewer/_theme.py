"""Shared Qt theme engine for the viewer shell widgets.

The viewer uses a compact custom Fusion palette rather than a Material base
theme. That keeps widget spacing, button contrast, and dock density under our
control instead of fighting large upstream defaults with overrides.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from .._logging import get_logger
from ._recent import config_dir

logger = get_logger(__name__)

__all__ = [
    "apply_qt_theme",
    "is_dark",
    "load_preference",
    "panel_colors",
    "secondary_text_style",
]


_THEME_PROPERTY = "_enginir_theme_applied"
_THEME_MODE_PROPERTY = "_enginir_theme_dark"
_PREFERENCE_FILE = "theme.json"
_current_dark = True

_PANEL_COLORS = {
    True: {
        "bg": "#1f2328",
        "ax_bg": "#272d34",
        "fg": "#f3f6fb",
        "accent": "#7c8cff",
    },
    False: {
        "bg": "#f3f5f8",
        "ax_bg": "#ffffff",
        "fg": "#2e3742",
        "accent": "#4660d9",
    },
}

_SURFACE_COLORS = {
    True: {
        "window": "#1b1f24",
        "surface": "#232831",
        "surface_alt": "#2b313b",
        "surface_hover": "#343b46",
        "surface_pressed": "#3d4654",
        "border": "#566172",
        "border_subtle": "#3d4652",
        "text": "#f3f6fb",
        "muted": "#b3bdca",
        "accent": "#7c8cff",
        "accent_active": "#98a6ff",
        "selection": "#3f4fbe",
    },
    False: {
        "window": "#eef2f7",
        "surface": "#ffffff",
        "surface_alt": "#f6f8fc",
        "surface_hover": "#edf2fb",
        "surface_pressed": "#dfe7f7",
        "border": "#bcc6d6",
        "border_subtle": "#d6dde9",
        "text": "#2f3944",
        "muted": "#65707d",
        "accent": "#4660d9",
        "accent_active": "#5b73e5",
        "selection": "#dfe6ff",
    },
}


def _preference_path() -> Path:
    return _storage_dir() / _PREFERENCE_FILE


def _storage_dir() -> Path:
    preferred = config_dir()
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        if os.access(preferred, os.W_OK):
            return preferred
        raise OSError(f"{preferred} is not writable")
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "enginir"
        fallback.mkdir(parents=True, exist_ok=True)
        logger.debug("Falling back to temporary theme storage {}: {}", fallback, exc)
        return fallback


def apply_qt_theme(app: QApplication, *, dark: bool = True) -> QApplication:
    """Apply the shared compact Fusion theme to the current application."""
    global _current_dark
    app.setStyle("Fusion")
    app.setPalette(_build_palette(dark))
    app.setStyleSheet(_app_stylesheet(dark))
    _current_dark = dark
    app.setProperty(_THEME_PROPERTY, True)
    app.setProperty(_THEME_MODE_PROPERTY, dark)
    _save_preference(dark)
    logger.info("Applied {} viewer theme", "dark" if dark else "light")
    return app


def is_dark() -> bool:
    """Return the current application theme preference."""
    return _current_dark


def load_preference() -> bool:
    """Return the persisted dark/light preference, defaulting to dark."""
    path = _preference_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return True
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("Could not load theme preference from {}: {}", path, exc)
        return True
    dark = data.get("dark")
    return True if not isinstance(dark, bool) else dark


def _save_preference(dark: bool) -> None:
    """Persist the current dark/light preference."""
    path = _preference_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dark": dark}), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not persist theme preference to {}: {}", path, exc)


def panel_colors() -> dict[str, str]:
    """Return synchronized plot/canvas colors for the current theme."""
    return dict(_PANEL_COLORS[_current_dark])


def secondary_text_style() -> str:
    """Return a shared compact secondary-label style fragment."""
    return "font-size: 9px;"


def _build_palette(dark: bool) -> QPalette:
    colors = _SURFACE_COLORS[dark]
    palette = QPalette()
    window = QColor(colors["window"])
    surface = QColor(colors["surface"])
    surface_alt = QColor(colors["surface_alt"])
    text = QColor(colors["text"])
    muted = QColor(colors["muted"])
    accent = QColor(colors["accent"])
    selection = QColor(colors["selection"])

    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, surface)
    palette.setColor(QPalette.ColorRole.AlternateBase, surface_alt)
    palette.setColor(QPalette.ColorRole.ToolTipBase, surface_alt)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, surface_alt)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, selection)
    palette.setColor(QPalette.ColorRole.HighlightedText, text if not dark else QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Link, accent)
    palette.setColor(QPalette.ColorRole.PlaceholderText, muted)

    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, muted)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, muted)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, muted)
    return palette


def _app_stylesheet(dark: bool) -> str:
    colors = _SURFACE_COLORS[dark]
    return f"""
QWidget {{
    color: {colors["text"]};
}}

QMainWindow, QDialog, QMenuBar, QStatusBar {{
    background: {colors["window"]};
}}

QGroupBox {{
    margin-top: 4px;
    padding-top: 3px;
    border: 1px solid {colors["border_subtle"]};
    border-radius: 8px;
    background: {colors["surface"]};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 2px;
    color: {colors["muted"]};
}}

QPushButton, QToolButton {{
    background: {colors["surface_alt"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: 7px;
    padding: 2px 10px;
    min-height: 22px;
    font-weight: 500;
}}

QPushButton:hover, QToolButton:hover {{
    background: {colors["surface_hover"]};
    border-color: {colors["accent"]};
}}

QPushButton:pressed, QToolButton:pressed,
QPushButton:checked, QToolButton:checked {{
    background: {colors["surface_pressed"]};
    border-color: {colors["accent_active"]};
    color: {colors["text"]};
}}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit, QListWidget,
QTreeView, QTreeWidget, QAbstractItemView {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 6px;
    padding: 2px 6px;
    selection-background-color: {colors["selection"]};
}}

QComboBox::drop-down {{
    border: 0;
    width: 18px;
}}

QMenuBar {{
    border-bottom: 1px solid {colors["border_subtle"]};
}}

QMenuBar::item {{
    padding: 4px 8px;
    margin: 1px 2px;
    border-radius: 5px;
}}

QMenuBar::item:selected, QMenu::item:selected {{
    background: {colors["surface_hover"]};
}}

QMenu {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    padding: 4px;
}}

QMenu::item {{
    padding: 4px 18px 4px 10px;
    border-radius: 4px;
}}

QTabWidget::pane {{
    border: 1px solid {colors["border"]};
    border-radius: 8px;
    top: -1px;
    background: {colors["surface"]};
}}

QTabBar::tab {{
    background: {colors["surface_alt"]};
    border: 1px solid {colors["border_subtle"]};
    border-bottom: 0;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    padding: 3px 10px;
    margin-right: 2px;
    min-height: 22px;
}}

QTabBar::tab:selected {{
    background: {colors["surface"]};
    border-color: {colors["border"]};
}}

QTabBar::tab:hover:!selected {{
    background: {colors["surface_hover"]};
}}

QDockWidget::title {{
    background: {colors["surface"]};
    padding: 3px 8px;
    border-bottom: 1px solid {colors["border_subtle"]};
}}

QStatusBar {{
    padding: 0;
}}

QCheckBox, QRadioButton {{
    spacing: 6px;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px;
    height: 14px;
}}

QHeaderView::section {{
    background: {colors["surface_alt"]};
    border: 0;
    border-bottom: 1px solid {colors["border_subtle"]};
    border-right: 1px solid {colors["border_subtle"]};
    padding: 4px 6px;
    color: {colors["muted"]};
}}
""".strip()
