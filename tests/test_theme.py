"""The viewer theme (plan gui-1.6, Phase 2): one token table, readable in both themes.

Every colour of the viewer comes from ``enginir.viewer._theme``; text keeps
WCAG AA contrast on every surface; exports take the print colours.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from enginir.viewer import _theme as theme

VIEWER = Path(theme.__file__).parent
_HEX = re.compile(r"""["']#[0-9a-fA-F]{3,8}["']|mkPen\(\s*["'](?:yellow|red|green|blue|white|"""
                  r"""black|magenta|cyan)["']|["'](?:viridis|magma|inferno|cividis|plasma)["']""")


def _luminance(color: str) -> float:
    rgb = [int(color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(foreground: str, background: str) -> float:
    high, low = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


SURFACES = ("window", "panel", "header", "input")


@pytest.mark.parametrize("dark", [True, False], ids=["dark", "light"])
def test_text_is_readable_on_every_surface(dark):
    tokens = theme.TOKENS[dark]
    for surface in SURFACES:
        for key, minimum in (("text", 7.0), ("text_secondary", 4.5), ("text_hint", 3.0),
                             ("ok", 3.0), ("attention", 3.0), ("error", 3.0),
                             ("accent", 3.0)):
            ratio = contrast(tokens[key], tokens[surface])
            assert ratio >= minimum, f"{key} on {surface}: {ratio:.2f} < {minimum}"
    assert contrast(tokens["on_accent"], tokens["accent"]) >= 4.5
    assert contrast(tokens["text"], tokens["selection"]) >= 7.0
    for color in theme.SERIES[dark]:
        assert contrast(color, tokens["plot_bg"]) >= 3.0, color


def test_an_export_swaps_every_screen_series_colour_for_its_print_colour(monkeypatch):
    from matplotlib.colors import to_hex
    from matplotlib.figure import Figure

    from enginir.viewer._plot_panel import print_copy

    monkeypatch.setattr(theme, "_current_dark", True)
    screen, printed = theme.panel_colors(), theme.print_colors()
    fig = Figure()
    ax = fig.add_subplot(111)
    for index in range(len(theme.SERIES[True])):
        ax.plot([0, 1], [index, index], color=screen[f"series{index}"])
    copy = print_copy(fig)
    shown = [to_hex(line.get_color()) for line in copy.axes[0].lines]
    assert shown == [to_hex(printed[f"series{index}"])
                     for index in range(len(theme.SERIES[True]))]


# -- A new theme reaches icons and Properties ------------------------------------------


def test_icons_and_properties_take_the_colours_of_a_new_theme(qapp, window):
    from PyQt6.QtWidgets import QLabel

    from enginir.viewer._theme import TOKENS, is_dark

    was_dark = is_dark()
    body = window._inspector.body_label
    body.setText("Target: Layers")
    section = window._project_browser.topLevelItem(0)

    def tree_icon():
        return section.icon(0).pixmap(16, 16).toImage()

    try:
        window._set_theme_mode(True)
        qapp.processEvents()
        dark_icon = tree_icon()
        assert TOKENS[True]["text_secondary"] in QLabel.text(body)
        window._set_theme_mode(False)
        qapp.processEvents()
        assert TOKENS[False]["text_secondary"] in QLabel.text(body), "the keys' colour"
        assert tree_icon() != dark_icon, "the icon is drawn again in the light theme's colour"
    finally:
        window._set_theme_mode(was_dark)
