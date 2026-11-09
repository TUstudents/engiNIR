"""Shared Qt theme engine for the viewer shell widgets.

The viewer uses a compact custom Fusion palette rather than a Material base
theme. That keeps widget spacing, button contrast, and dock density under our
control instead of fighting large upstream defaults with overrides.
"""

from __future__ import annotations

import importlib.resources
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
    "GUIDE_COLOR",
    "print_colors",
    "secondary_text_style",
    "theme_applied",
]


_THEME_PROPERTY = "_enginir_theme_applied"
_THEME_MODE_PROPERTY = "_enginir_theme_dark"
_PREFERENCE_FILE = "theme.json"
_current_dark = True

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
        "disabled_text": "#6f7985",
        "disabled_bg": "#20252c",
        "progress": "#4f5fd0",
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
        "selection": "#c7d3ff",
        "disabled_text": "#a3acb8",
        "disabled_bg": "#f1f4f8",
        "progress": "#9aa9ef",
    },
}


#: Plot colours follow the widget colours: the figure is the window, the axes
#: are a surface, so plots sit in the shell without a visible seam.
_PANEL_COLORS = {
    dark: {
        "bg": colors["window"],
        "ax_bg": colors["surface"],
        "fg": colors["text"],
        "accent": colors["accent"],
    }
    for dark, colors in _SURFACE_COLORS.items()
}

#: Reference lines and "no value" marks in plots (zero lines, the identity
#: line, confidence ellipses, ROIs without a value, dark references); one
#: neutral grey readable on the light and the dark plot background.
GUIDE_COLOR = "#9aa3ad"


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


def theme_applied(app: QApplication, *, dark: bool) -> bool:
    """Whether ``app`` already has the engiNIR theme in this mode."""
    return (bool(app.property(_THEME_PROPERTY))
            and bool(app.property(_THEME_MODE_PROPERTY)) == dark)


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


def print_colors() -> dict[str, str]:
    """Return the plot colors for exported images (light, whatever the screen theme)."""
    return {"bg": "#ffffff", "ax_bg": "#ffffff", "fg": "#1a1a1a",
            "accent": _PANEL_COLORS[False]["accent"]}


def secondary_text_style() -> str:
    """Return a shared secondary-label style fragment: 85 % of the UI font, in points.

    Points follow the display scaling; a fixed pixel size stays tiny on
    high-resolution screens.
    """
    app = QApplication.instance()
    size = app.font().pointSizeF() if isinstance(app, QApplication) else -1.0
    if size <= 0:
        size = 9.0
    return f"font-size: {0.85 * size:.1f}pt;"


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


def _resource_url(name: str) -> str:
    """A bundled image as a style-sheet path, or "" when it is not a plain file."""
    try:
        path = Path(str(importlib.resources.files("enginir._resources").joinpath(name)))
    except (ModuleNotFoundError, TypeError) as exc:
        logger.debug("No theme image {}: {}", name, exc)
        return ""
    return path.as_posix() if path.is_file() else ""


def _image(name: str) -> str:
    """The ``image:`` declaration for a bundled image, empty when it is unavailable."""
    url = _resource_url(name)
    return f'image: url("{url}");' if url else ""


def _app_stylesheet(dark: bool) -> str:
    colors = _SURFACE_COLORS[dark]
    mode = "dark" if dark else "light"
    up, down = _image(f"arrow-up-{mode}.svg"), _image(f"arrow-down-{mode}.svg")
    up_off, down_off = _image("arrow-up-disabled.svg"), _image("arrow-down-disabled.svg")
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

QFileDialog QToolButton {{
    padding: 2px;
}}

QPushButton:default {{
    border: 1px solid {colors["accent"]};
    font-weight: 600;
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

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {colors["accent"]};
}}

QPushButton:disabled, QToolButton:disabled, QLineEdit:disabled, QComboBox:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {{
    color: {colors["disabled_text"]};
    background: {colors["disabled_bg"]};
    border-color: {colors["border_subtle"]};
}}

QComboBox::drop-down {{
    border: 0;
    width: 18px;
}}

QComboBox::down-arrow {{
    {down}
    width: 8px;
    height: 5px;
}}

QComboBox::down-arrow:disabled {{
    {down_off}
}}

QSpinBox, QDoubleSpinBox {{
    padding-right: 18px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    width: 16px;
    border: 0;
    border-left: 1px solid {colors["border_subtle"]};
    background: transparent;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-position: top right;
    border-top-right-radius: 6px;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-position: bottom right;
    border-bottom-right-radius: 6px;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {colors["surface_hover"]};
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    {up}
    width: 8px;
    height: 5px;
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    {down}
    width: 8px;
    height: 5px;
}}

QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
QSpinBox::up-arrow:off, QDoubleSpinBox::up-arrow:off {{
    {up_off}
}}

QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled,
QSpinBox::down-arrow:off, QDoubleSpinBox::down-arrow:off {{
    {down_off}
}}

QProgressBar {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 6px;
    text-align: center;
    min-height: 16px;
}}

QProgressBar::chunk {{
    background: {colors["progress"]};
    border-radius: 5px;
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
    margin-bottom: -1px;
    padding-bottom: 4px;
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
    width: 12px;
    height: 12px;
    border: 1px solid {colors["border"]};
    background: {colors["surface"]};
}}

QCheckBox::indicator {{
    border-radius: 3px;
}}

QRadioButton::indicator {{
    border-radius: 7px;
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {colors["accent"]};
}}

QCheckBox::indicator:checked {{
    background: {colors["accent"]};
    border-color: {colors["accent"]};
    {_image("check.svg")}
}}

QCheckBox::indicator:indeterminate {{
    background: {colors["surface_pressed"]};
    border-color: {colors["accent"]};
}}

QRadioButton::indicator:checked {{
    background: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5,
        stop: 0 {colors["text"]}, stop: 0.45 {colors["text"]},
        stop: 0.5 {colors["accent"]}, stop: 1 {colors["accent"]});
    border-color: {colors["accent"]};
}}

QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {colors["border_subtle"]};
    background: {colors["disabled_bg"]};
}}

QCheckBox:disabled, QRadioButton:disabled, QLabel:disabled {{
    color: {colors["disabled_text"]};
}}

QAbstractItemView::item:selected:!active {{
    background: {colors["surface_pressed"]};
    color: {colors["text"]};
}}

QToolTip {{
    background: {colors["surface_alt"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: 4px;
    padding: 3px 6px;
}}

QSplitter::handle {{
    background: transparent;
}}

QSplitter::handle:hover {{
    background: {colors["border"]};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {colors["border_subtle"]};
    border-radius: 4px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    min-height: 24px;
}}

QScrollBar::handle:horizontal {{
    min-width: 24px;
}}

QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: {colors["border"]};
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0;
    height: 0;
}}

QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

QHeaderView {{
    border: 0;
    border-radius: 0;
    padding: 0;
    background: {colors["surface_alt"]};
}}

QTableCornerButton::section {{
    background: {colors["surface_alt"]};
    border: 0;
    border-bottom: 1px solid {colors["border_subtle"]};
    border-right: 1px solid {colors["border_subtle"]};
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
