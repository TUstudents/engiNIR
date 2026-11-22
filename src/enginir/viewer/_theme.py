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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    "series_colors",
    "tokens",
    "value_colormap",
    "heat_colormap",
    "TOKENS",
    "secondary_text_style",
    "theme_applied",
]


_THEME_PROPERTY = "_enginir_theme_applied"
_THEME_MODE_PROPERTY = "_enginir_theme_dark"
_PREFERENCE_FILE = "theme.json"
_current_dark = True

#: The colour tokens of both themes (plan gui-1.6, theme). Every widget and
#: plot colour of the viewer comes from here; no other viewer module writes a
#: colour literal (``tests/test_theme.py``). The accent is the orange of the
#: program icon; warm colours belong to the program and to check states, the
#: data scales are cool, so plots never compete with the interface.
TOKENS: dict[bool, dict[str, str]] = {
    True: {
        "window": "#1c1d20",        # main window, gaps between panels
        "panel": "#25272b",         # docks, pages, group bodies
        "header": "#2c2f34",        # dock titles, table headers, title strips
        "input": "#17181a",         # editable fields, lists, tables
        "hover": "#33363d",
        "pressed": "#3c4048",
        "selection": "#4a3a26",     # selected rows and items (warm, from the accent)
        "border": "#4a4e58",
        "border_subtle": "#363940",
        "text": "#e4e6ea",
        "text_secondary": "#a6abb5",
        "text_hint": "#868c97",
        "accent": "#f0912e",
        "accent_hover": "#f5a54f",
        "on_accent": "#1c1206",
        "ok": "#4fbf87",
        "attention": "#ecc94b",
        "error": "#f07068",
        "disabled_text": "#6b707a",
        "disabled_bg": "#222427",
        "plot_bg": "#1e1f23",
        "plot_grid": "#2e3036",
        "guide": "#8b919c",
    },
    False: {
        "window": "#ececee",
        "panel": "#f8f8f9",
        "header": "#e3e4e8",
        "input": "#ffffff",
        "hover": "#dfe1e6",
        "pressed": "#d3d6dd",
        "selection": "#f6d9bd",
        "border": "#b9bdc6",
        "border_subtle": "#d4d7de",
        "text": "#1e2227",
        "text_secondary": "#4f5560",
        "text_hint": "#6c727d",
        "accent": "#b8520b",
        "accent_hover": "#cf6114",
        "on_accent": "#ffffff",
        "ok": "#1b7a50",
        "attention": "#8a6400",
        "error": "#b8332b",
        "disabled_text": "#9aa0aa",
        "disabled_bg": "#eeeff2",
        "plot_bg": "#ffffff",
        "plot_grid": "#e6e8ec",
        "guide": "#7d838e",
    },
}

#: Categorical data colours (models, curves, series), cool first; one row per
#: theme, the light row darker so it reads on white.
SERIES: dict[bool, tuple[str, ...]] = {
    True: ("#7aa2ff", "#2fc0c4", "#8fd16a", "#b48cff", "#f07fae", "#d9b26a"),
    False: ("#2f5bd3", "#11888c", "#4c8f2a", "#7a4fd6", "#c43f7a", "#9a7428"),
}

#: The colours of the icon's bands, yellow to red; the middle of the heat scale.
ICON_BANDS = ("#f2bc12", "#ee9d14", "#e27714", "#cf4c1c", "#b52c22")
#: Heat scale for maps of distances and uncertainty and the sensor frame:
#: dark to the icon's red, orange and yellow to pale; ordered by lightness.
HEAT_STOPS = ("#1a0f12", "#5c1a1b", ICON_BANDS[4], ICON_BANDS[3], ICON_BANDS[2],
              ICON_BANDS[1], ICON_BANDS[0], "#fbe7a0")
#: The value scale (target values, predictions) is viridis, perceptually
#: uniform and readable with colour-vision deficiencies, without its dark end
#: on dark backgrounds and without its yellow end on white.
VALUE_RANGE = {True: (0.12, 1.0), False: (0.0, 0.88)}

#: Backwards-compatible view of the tokens under the keys of engiNIR 1.5.
_SURFACE_COLORS = {
    dark: {**tokens, "surface": tokens["panel"], "surface_alt": tokens["header"],
           "surface_hover": tokens["hover"], "surface_pressed": tokens["pressed"],
           "muted": tokens["text_secondary"], "accent_active": tokens["accent_hover"],
           "progress": tokens["accent"]}
    for dark, tokens in TOKENS.items()
}


def _plot_colors(dark: bool) -> dict[str, str]:
    tokens = TOKENS[dark]
    return {
        "bg": tokens["panel"],
        "ax_bg": tokens["plot_bg"],
        "fg": tokens["text"],
        "grid": tokens["plot_grid"],
        "guide": tokens["guide"],
        # The primary data colour: the first series colour, not the UI accent.
        "accent": SERIES[dark][0],
        "highlight": tokens["accent"],
        "ok": tokens["ok"],
        "attention": tokens["attention"],
        "error": tokens["error"],
        **{f"series{index}": color for index, color in enumerate(SERIES[dark])},
        "mode": "dark" if dark else "light",
    }


#: Plot colours: the figure is a panel, the axes the plot background.
_PANEL_COLORS = {dark: _plot_colors(dark) for dark in TOKENS}

#: Reference lines and "no value" marks in plots (zero lines, the identity
#: line, confidence ellipses, ROIs without a value, dark references); one
#: neutral grey readable on the light and the dark plot background.
GUIDE_COLOR = "#8b919c"


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
    # Before the palette: widgets that redraw their icons and colours on the palette
    # change (plot headers, project tree, Properties) must read the new theme.
    _current_dark = dark
    app.setPalette(_build_palette(dark))
    app.setStyleSheet(_app_stylesheet(dark))
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
    return {**_PANEL_COLORS[False], "bg": "#ffffff", "ax_bg": "#ffffff", "fg": "#1a1a1a"}


def series_colors() -> tuple[str, ...]:
    """The categorical data colours of the current theme (models, curves)."""
    return SERIES[_current_dark]


def tokens() -> dict[str, str]:
    """The colour tokens of the current theme."""
    return dict(TOKENS[_current_dark])


def value_colormap(colors: Mapping[str, str] | None = None) -> Any:
    """The matplotlib colour map for target values and predictions.

    For the plot colours ``colors`` (screen or print, by their ``mode``); the
    current theme by default.
    """
    import numpy as np  # not at module level: the launcher imports this module (D2)
    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap

    dark = _current_dark if colors is None else colors.get("mode", "dark") == "dark"
    low, high = VALUE_RANGE[dark]
    return ListedColormap(colormaps["viridis"](np.linspace(low, high, 256)),
                          name="enginir_values")


def heat_colormap() -> Any:
    """The matplotlib colour map built from the icon's bands (distances, uncertainty)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("enginir_heat", HEAT_STOPS)


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
    t = TOKENS[dark]
    palette = QPalette()
    roles = QPalette.ColorRole
    for role, key in ((roles.Window, "window"), (roles.WindowText, "text"),
                      (roles.Base, "input"), (roles.AlternateBase, "panel"),
                      (roles.ToolTipBase, "header"), (roles.ToolTipText, "text"),
                      (roles.Text, "text"), (roles.Button, "header"),
                      (roles.ButtonText, "text"), (roles.BrightText, "on_accent"),
                      (roles.Highlight, "selection"), (roles.HighlightedText, "text"),
                      (roles.Link, "accent"), (roles.PlaceholderText, "text_hint"),
                      (roles.Mid, "border"), (roles.Midlight, "border_subtle"),
                      (roles.Dark, "window"), (roles.Light, "hover")):
        palette.setColor(role, QColor(t[key]))
    for role in (roles.Text, roles.ButtonText, roles.WindowText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(t["disabled_text"]))
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
    """The style sheet of the whole application, generated from :data:`TOKENS`.

    Desktop proportions (plan gui-1.6, D1): square corners (3 px on controls),
    controls 22–24 px high, dock and table headers on the header colour, the
    current tab marked by an accent line.
    """
    c = TOKENS[dark]
    mode = "dark" if dark else "light"
    up, down = _image(f"arrow-up-{mode}.svg"), _image(f"arrow-down-{mode}.svg")
    up_off, down_off = _image("arrow-up-disabled.svg"), _image("arrow-down-disabled.svg")
    return f"""
QWidget {{
    color: {c["text"]};
}}

QMainWindow, QDialog, QStatusBar {{
    background: {c["window"]};
}}

QMainWindow::separator {{
    background: {c["border_subtle"]};
    width: 1px;
    height: 1px;
}}

QMenuBar {{
    background: {c["panel"]};
    border-bottom: 1px solid {c["border_subtle"]};
}}

QMenuBar::item {{
    padding: 4px 9px;
    background: transparent;
}}

QMenuBar::item:selected, QMenu::item:selected {{
    background: {c["hover"]};
}}

QMenu {{
    background: {c["panel"]};
    border: 1px solid {c["border"]};
    padding: 3px 0;
}}

QMenu::item {{
    padding: 4px 24px 4px 12px;
}}

QMenu::separator {{
    height: 1px;
    background: {c["border_subtle"]};
    margin: 3px 6px;
}}

QGroupBox {{
    margin-top: 6px;
    padding-top: 4px;
    border: 1px solid {c["border_subtle"]};
    border-radius: 3px;
    background: {c["panel"]};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 3px;
    color: {c["text_secondary"]};
    font-weight: 600;
}}

QPushButton, QToolButton {{
    background: {c["header"]};
    color: {c["text"]};
    border: 1px solid {c["border"]};
    border-radius: 3px;
    padding: 2px 10px;
    min-height: 20px;
}}

QToolButton {{
    padding: 2px 4px;
}}

QFileDialog QToolButton {{
    padding: 2px;
}}

QPushButton:default {{
    background: {c["accent"]};
    border-color: {c["accent"]};
    color: {c["on_accent"]};
}}

QPushButton:default:hover {{
    background: {c["accent_hover"]};
    border-color: {c["accent_hover"]};
}}

QPushButton:hover, QToolButton:hover {{
    background: {c["hover"]};
    border-color: {c["accent"]};
}}

QPushButton:pressed, QToolButton:pressed,
QPushButton:checked, QToolButton:checked {{
    background: {c["pressed"]};
    border-color: {c["accent"]};
    color: {c["text"]};
}}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit, QListWidget,
QTreeView, QTreeWidget, QAbstractItemView {{
    background: {c["input"]};
    border: 1px solid {c["border"]};
    border-radius: 3px;
    padding: 1px 5px;
    selection-background-color: {c["selection"]};
    selection-color: {c["text"]};
}}

QAbstractItemView {{
    padding: 0;
    gridline-color: {c["border_subtle"]};
    alternate-background-color: {c["panel"]};
}}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {c["accent"]};
}}

QPushButton:disabled, QToolButton:disabled, QLineEdit:disabled, QComboBox:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {{
    color: {c["disabled_text"]};
    background: {c["disabled_bg"]};
    border-color: {c["border_subtle"]};
}}

QComboBox {{
    min-height: 20px;
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

QComboBox QAbstractItemView {{
    background: {c["panel"]};
    border: 1px solid {c["border"]};
}}

QSpinBox, QDoubleSpinBox {{
    padding-right: 18px;
    min-height: 20px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    width: 16px;
    border: 0;
    border-left: 1px solid {c["border_subtle"]};
    background: transparent;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-position: top right;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-position: bottom right;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {c["hover"]};
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
    background: {c["input"]};
    border: 1px solid {c["border"]};
    border-radius: 3px;
    text-align: center;
    min-height: 14px;
}}

QProgressBar::chunk {{
    background: {c["accent"]};
}}

QTabWidget::pane {{
    border: 1px solid {c["border_subtle"]};
    top: -1px;
    background: {c["panel"]};
}}

QTabBar::tab {{
    background: transparent;
    color: {c["text_secondary"]};
    border: 1px solid transparent;
    border-bottom: 0;
    padding: 4px 12px;
    margin-right: 1px;
    min-height: 20px;
}}

QTabBar::tab:selected {{
    background: {c["panel"]};
    color: {c["text"]};
    border-color: {c["border_subtle"]};
    border-top: 2px solid {c["accent"]};
    margin-bottom: -1px;
}}

QTabBar::tab:hover:!selected {{
    background: {c["hover"]};
    color: {c["text"]};
}}

QDockWidget {{
    titlebar-close-icon: none;
}}

QDockWidget::title {{
    background: {c["header"]};
    padding: 4px 8px;
    border-bottom: 1px solid {c["border_subtle"]};
}}

QStatusBar {{
    background: {c["header"]};
    border-top: 1px solid {c["border_subtle"]};
    padding: 0;
}}

QStatusBar::item {{
    border: 0;
    border-right: 1px solid {c["border_subtle"]};
}}

QCheckBox, QRadioButton {{
    spacing: 6px;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 12px;
    height: 12px;
    border: 1px solid {c["border"]};
    background: {c["input"]};
}}

QCheckBox::indicator {{
    border-radius: 2px;
}}

QRadioButton::indicator {{
    border-radius: 7px;
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {c["accent"]};
}}

QCheckBox::indicator:checked {{
    background: {c["accent"]};
    border-color: {c["accent"]};
    {_image("check.svg")}
}}

QCheckBox::indicator:indeterminate {{
    background: {c["pressed"]};
    border-color: {c["accent"]};
}}

QRadioButton::indicator:checked {{
    background: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5,
        stop: 0 {c["on_accent"]}, stop: 0.35 {c["on_accent"]},
        stop: 0.45 {c["accent"]}, stop: 1 {c["accent"]});
    border-color: {c["accent"]};
}}

QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {c["border_subtle"]};
    background: {c["disabled_bg"]};
}}

QCheckBox:disabled, QRadioButton:disabled, QLabel:disabled {{
    color: {c["disabled_text"]};
}}

QAbstractItemView::item:hover {{
    background: {c["hover"]};
}}

QAbstractItemView::item:selected {{
    background: {c["selection"]};
    color: {c["text"]};
}}

QAbstractItemView::item:selected:!active {{
    background: {c["pressed"]};
    color: {c["text"]};
}}

QToolTip {{
    background: {c["header"]};
    color: {c["text"]};
    border: 1px solid {c["border"]};
    padding: 3px 6px;
}}

QSplitter::handle {{
    background: {c["border_subtle"]};
}}

QSplitter::handle:hover {{
    background: {c["accent"]};
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
    background: {c["border"]};
    border-radius: 3px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    min-height: 24px;
}}

QScrollBar::handle:horizontal {{
    min-width: 24px;
}}

QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: {c["text_hint"]};
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
    background: {c["header"]};
}}

QTableCornerButton::section {{
    background: {c["header"]};
    border: 0;
    border-bottom: 1px solid {c["border_subtle"]};
    border-right: 1px solid {c["border_subtle"]};
}}

QToolBar#WorkflowBar {{
    background: {c["window"]};
    border: 0;
    border-bottom: 1px solid {c["border_subtle"]};
    padding: 0;
    spacing: 0;
}}

QLabel#PropertiesTitle {{
    font-weight: 600;
}}

QLabel#WorkflowContext {{
    color: {c["text_secondary"]};
    padding-right: 6px;
}}

QTreeWidget#ProjectBrowser::item {{
    padding: 2px 0;
}}

QTableWidget#ChecksTable::item {{
    padding-right: 8px;
}}

QLabel#WorkflowLabel {{
    color: {c["text_hint"]};
}}

QToolButton#WorkflowStep {{
    background: transparent;
    border: 0;
    border-radius: 0;
    padding: 5px 12px;
    color: {c["text_secondary"]};
}}

QToolButton#WorkflowStep[badge="true"] {{
    padding-right: 34px;
}}

QLabel#WorkflowArrow {{
    color: {c["text_hint"]};
    padding: 0 2px;
}}

QLabel#StepBadge {{
    background: {c["attention"]};
    color: {c["window"]};
    border-radius: 8px;
    min-width: 16px;
    min-height: 16px;
    max-height: 16px;
    padding: 0 0px;
    font-size: 8pt;
    font-weight: 700;
}}

QLabel#StepBadge[level="blocked"] {{
    background: {c["error"]};
}}

QToolButton#WorkflowStep:hover {{
    background: {c["hover"]};
    color: {c["text"]};
}}

QToolButton#WorkflowStep:checked {{
    background: {c["panel"]};
    color: {c["text"]};
    font-weight: 600;
    border-bottom: 2px solid {c["accent"]};
}}

QPushButton#FixButton {{
    background: transparent;
    border: 1px solid {c["accent"]};
    border-radius: 3px;
    color: {c["accent"]};
    padding: 0 8px;
    min-height: 18px;
    max-height: 18px;
}}

QPushButton#FixButton:hover {{
    background: {c["hover"]};
}}

QLabel#ChangedMark {{
    color: {c["attention"]};
}}

QLabel#ChangedMark[changed="false"] {{
    color: {c["text_secondary"]};
}}

QWidget#FitFooter {{
    border-top: 1px solid {c["border_subtle"]};
}}

QLabel#EmptyState {{
    color: {c["text_secondary"]};
}}

QLabel#EmptyTitle {{
    font-size: 15pt;
    font-weight: 600;
}}

QLabel#VerdictText {{
    font-weight: 600;
}}

QToolButton#PrimaryAction {{
    background: {c["accent"]};
    border-color: {c["accent"]};
    color: {c["on_accent"]};
    font-weight: 600;
    padding: 4px 10px;
}}

QToolButton#PrimaryAction:hover {{
    background: {c["accent_hover"]};
    border-color: {c["accent_hover"]};
}}

QToolButton#PrimaryAction:disabled {{
    background: {c["header"]};
    border-color: {c["border"]};
    color: {c["text_hint"]};
}}

QToolButton#SectionHeader {{
    background: transparent;
    border: 0;
    border-bottom: 1px solid {c["border_subtle"]};
    border-radius: 0;
    padding: 3px 2px;
    font-weight: 600;
    text-align: left;
}}

QToolButton#SectionHeader:hover {{
    background: {c["hover"]};
}}

QLabel#SectionSummary {{
    color: {c["text_secondary"]};
}}

QHeaderView::section {{
    background: {c["header"]};
    border: 0;
    border-bottom: 1px solid {c["border_subtle"]};
    border-right: 1px solid {c["border_subtle"]};
    padding: 3px 6px;
    color: {c["text_secondary"]};
    font-weight: 600;
}}
""".strip()
