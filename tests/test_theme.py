"""The viewer theme (plan gui-1.6, Phase 2): one token table, readable in both themes.

Every colour of the viewer comes from ``enginir.viewer._theme``; text keeps
WCAG AA contrast on every surface; exports take the print colours.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
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


def test_both_themes_have_the_same_tokens_and_six_series_colours():
    assert set(theme.TOKENS[True]) == set(theme.TOKENS[False])
    assert len(theme.SERIES[True]) == len(theme.SERIES[False]) == 6
    assert set(theme.panel_colors()) == set(theme.print_colors())


def test_no_viewer_module_writes_its_own_colour():
    found = []
    for path in sorted(VIEWER.glob("*.py")):
        if path.name == "_theme.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _HEX.search(line):
                found.append(f"{path.name}:{number}: {line.strip()}")
    assert not found, "colours belong in _theme.TOKENS:\n" + "\n".join(found)


@pytest.mark.parametrize("dark", [True, False], ids=["dark", "light"])
def test_the_style_sheet_uses_only_tokens(dark):
    sheet = theme._app_stylesheet(dark)
    colours = set(re.findall(r"#[0-9a-fA-F]{6}\b", sheet))
    assert colours and colours <= set(theme.TOKENS[dark].values())
    assert theme.TOKENS[dark]["accent"] in sheet


def test_value_and_heat_scales_are_ordered_by_lightness():
    def lightness(cmap):
        return np.array([_luminance("#%02x%02x%02x" % tuple(int(255 * c) for c in cmap(x)[:3]))
                         for x in np.linspace(0, 1, 32)])

    for colors in (theme.panel_colors(), theme.print_colors()):
        assert np.all(np.diff(lightness(theme.value_colormap(colors))) > 0)
    assert np.all(np.diff(lightness(theme.heat_colormap())) > 0)
    # Light: no pale yellow end on white; dark: no near-black end on the dark plot.
    assert lightness(theme.value_colormap(theme.print_colors()))[-1] < 0.7
    theme._current_dark = True
    try:
        assert lightness(theme.value_colormap())[0] > 0.02
    finally:
        theme._current_dark = True


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
