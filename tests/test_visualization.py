"""Representative behavior tests for enginir.visualization plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from enginir.visualization import (  # noqa: E402
    plot_image_map,
    plot_leverage,
    plot_loadings,
    plot_prediction,
    plot_scores,
    plot_spectra,
    plot_vip,
)

RNG = np.random.default_rng(0)
N, P = 20, 80

X = RNG.random((N, P))
WL = np.linspace(1000, 2500, P)
Y = RNG.random(N)
SCORES = RNG.random((N, 5))
LOADINGS = RNG.random((P, 5))
VIP = RNG.random(P) * 2
LEVERAGE = RNG.random(N) * 0.3
Q_RES = RNG.random(N) * 5


def _close_all() -> None:
    plt.close("all")


def test_plot_spectra_returns_axes_and_adds_colorbar_for_color_by():
    ax = plot_spectra(X, WL, color_by=Y)

    assert isinstance(ax, matplotlib.axes.Axes)
    (lines,) = ax.collections  # all spectra in one collection, colored by their value
    assert len(lines.get_segments()) == N
    np.testing.assert_allclose(lines.get_segments()[3][:, 1], X[3])
    np.testing.assert_allclose(lines.get_array(), Y)
    assert len(ax.figure.axes) == 2  # the colorbar

    _close_all()


def test_plot_spectra_shows_legend_only_for_small_labeled_sets():
    labels_small = [f"s{i}" for i in range(N)]
    ax_small = plot_spectra(X, WL, labels=labels_small)
    assert ax_small.get_legend() is not None
    _close_all()

    X_big = RNG.random((25, P))
    labels_big = [f"s{i}" for i in range(25)]
    ax_big = plot_spectra(X_big, WL, labels=labels_big)
    assert ax_big.get_legend() is None
    _close_all()


def test_plot_prediction_draws_reference_line_and_r2_annotation():
    ax = plot_prediction(Y, Y, units="%")

    assert isinstance(ax, matplotlib.axes.Axes)
    assert any(line.get_linestyle() == "--" for line in ax.lines)
    assert any("R" in text.get_text() for text in ax.texts)
    assert "%" in ax.get_xlabel()
    assert "%" in ax.get_ylabel()

    _close_all()


def test_plot_loadings_uses_requested_components_and_zero_reference_line():
    _, external_ax = plt.subplots()
    ax = plot_loadings(LOADINGS, WL, components=[0, 2], ax=external_ax)

    assert ax is external_ax
    assert len(ax.lines) == 3
    assert any(np.all(np.asarray(line.get_ydata()) == 0) for line in ax.lines)

    _close_all()


def test_plot_scores_adds_colorbar_and_rejects_invalid_component_index():
    ax = plot_scores(SCORES, y=Y)

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.figure.axes) == 2

    _close_all()

    with pytest.raises(ValueError, match="must differ and be below"):
        plot_scores(SCORES, pc_x=0, pc_y=99)

    with pytest.raises(ValueError, match="y must have shape"):
        plot_scores(SCORES, y=Y[:-1])


def test_plot_scores_only_annotates_labels_for_small_sets():
    labels_small = [f"s{i}" for i in range(N)]
    ax_small = plot_scores(SCORES, labels=labels_small)
    assert len(ax_small.texts) == N
    _close_all()

    big_scores = RNG.random((40, 5))
    labels_big = [f"s{i}" for i in range(40)]
    ax_big = plot_scores(big_scores, labels=labels_big)
    assert len(ax_big.texts) == 0
    _close_all()


def test_plot_vip_draws_threshold_line_and_bars():
    ax = plot_vip(VIP, WL, threshold=0.8)

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.patches) == P
    assert any(np.allclose(np.asarray(line.get_ydata()), 0.8) for line in ax.lines)

    _close_all()


def test_plot_leverage_draws_thresholds_and_only_annotates_outliers():
    labels = [f"s{i}" for i in range(N)]
    leverage_crit = np.percentile(LEVERAGE, 80)
    q_crit = np.percentile(Q_RES, 80)
    ax = plot_leverage(
        LEVERAGE,
        Q_RES,
        leverage_crit=leverage_crit,
        q_crit=q_crit,
        labels=labels,
    )

    is_outlier = (LEVERAGE > leverage_crit) | (Q_RES > q_crit)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.lines) == 2
    assert len(ax.texts) == int(is_outlier.sum())

    _close_all()


def test_plot_image_map_adds_overlay_and_colorbar():
    y_map = RNG.random((10, 15))
    rgb = RNG.random((10, 15, 3)) * 255
    ax = plot_image_map(y_map, background_rgb=rgb, cmap="hot")

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.get_images()) == 2
    assert ax.get_images()[-1].cmap.name == "hot"
    assert len(ax.figure.axes) == 2

    _close_all()


def _close_figures():
    yield
    plt.close("all")


def test_plot_image_map_draws_nan_pixels_transparent():
    y_map = RNG.random((6, 7))
    y_map[:2] = np.nan

    ax = plot_image_map(y_map, colorbar_label="moisture (%)")

    assert np.ma.getmaskarray(ax.images[-1].get_array())[0, 0]
    assert ax.figure.axes[-1].get_ylabel() == "moisture (%)"
    with pytest.raises(ValueError, match="no finite values"):
        plot_image_map(np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="finite"):
        plot_image_map(np.full((3, 3), np.inf))


def test_plot_image_map_rescales_backgrounds_outside_unit_range():
    background = RNG.random((6, 7, 3)) * 2.0 - 0.5
    ax = plot_image_map(RNG.random((6, 7)), background_rgb=background)

    drawn = np.asarray(ax.images[0].get_array())
    assert drawn.min() == pytest.approx(0.0) and drawn.max() == pytest.approx(1.0)


def test_plot_prediction_requires_one_target():
    with pytest.raises(ValueError, match="one target"):
        plot_prediction(RNG.random((10, 2)), RNG.random((10, 2)))
    ax = plot_prediction(Y[:, np.newaxis], Y[:, np.newaxis] + 0.1, color="red")
    assert ax.get_legend() is None


def test_plot_scores_colors_class_labels_with_a_legend():
    classes = np.array(["a", "b"] * (N // 2))

    ax = plot_scores(SCORES, y=classes, component_label="LV")

    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["a", "b"]
    assert ax.get_xlabel() == "LV 1"
    with pytest.raises(ValueError, match="must differ"):
        plot_scores(SCORES, pc_x=1, pc_y=1)


def test_plot_vip_bars_have_positive_width_for_descending_wavelengths():
    ax = plot_vip(RNG.random(P) * 2, WL[::-1])

    widths = [patch.get_width() for patch in ax.patches]
    assert min(widths) > 0


def test_plots_use_physical_axis_labels_and_pass_style_keywords():
    ax = plot_spectra(X, WL, ylabel="Absorbance", color="black")
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Wavelength (nm)", "Absorbance")
    assert all(line.get_color() == "black" for line in ax.get_lines())
    assert plot_loadings(LOADINGS).get_xlabel() == "Band index"


def test_plots_do_not_modify_their_inputs():
    data = X.copy()
    y_map = RNG.random((4, 4))
    y_map_copy = y_map.copy()
    plot_spectra(data, WL, color_by=Y)
    plot_image_map(y_map)
    np.testing.assert_array_equal(data, X)
    np.testing.assert_array_equal(y_map, y_map_copy)


def test_plot_loadings_accepts_a_coefficient_vector():
    ax = plot_loadings(LOADINGS[:, 0], WL, ylabel="Regression coefficient")
    assert ax.get_ylabel() == "Regression coefficient"
    assert len(ax.lines) == 2  # the vector and the zero line
    _close_all()


def test_style_keywords_do_not_collide_with_computed_ones():
    X = np.random.default_rng(0).random((4, WL.size))
    plot_spectra(X, WL, color_by=np.arange(4.0), color="k", cmap="plasma")
    plot_spectra(X, WL, labels=list("abcd"), label="ignored")
    plot_loadings(LOADINGS, WL, label="ignored")
    plot_scores(SCORES, y=Y, cmap="plasma")
    plot_scores(SCORES, y=np.array(["a", None] * (SCORES.shape[0] // 2), dtype=object))
    plot_scores(SCORES, y=Y[:, np.newaxis])
    _close_all()


def test_unflatten_cube_turns_predictions_into_a_map():
    from enginir.utils import flatten_cube, unflatten_cube

    cube = np.random.default_rng(0).random((3, 4, WL.size))
    X_2d, spatial_shape = flatten_cube(cube)
    y_map = unflatten_cube(X_2d.sum(axis=1), spatial_shape)
    assert y_map.shape == (3, 4)
    assert isinstance(plot_image_map(y_map), matplotlib.axes.Axes)
    _close_all()


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: plot_spectra(X, WL, labels=["only-one"]), "labels has"),
        (lambda: plot_spectra(X, WL[:-1]), "wavelengths"),
        (lambda: plot_spectra(X, WL, color_by=Y[:-1]), "color_by"),
        (lambda: plot_prediction(np.array([1.0, 2.0]), np.array([1.0])), "values but"),
        (lambda: plot_prediction(np.array([1.0, np.nan]), np.array([1.0, 2.0])), "finite"),
        (lambda: plot_loadings(LOADINGS[np.newaxis], WL), "1-D or 2-D"),
        (lambda: plot_loadings(LOADINGS, WL, components=[99]), "components"),
        (lambda: plot_vip(VIP, WL[:-1]), "wavelengths"),
        (lambda: plot_leverage(np.array([0.1, 0.2]), np.array([1.0])), "values but"),
        (lambda: plot_image_map(RNG.random((10, 15, 1))), "2-D"),
        (lambda: plot_image_map(RNG.random((10, 15)), background_rgb=RNG.random((10, 15))),
         "background_rgb must"),
    ],
)
def test_plots_reject_inputs_that_do_not_fit_together(call, match):
    """Mismatches raise a clear ValueError instead of a matplotlib error or a wrong plot."""
    with pytest.raises(ValueError, match=match):
        call()
    _close_all()
