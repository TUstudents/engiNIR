"""Behaviour of the spectral preprocessing transformers.

Generic input validation (2-D, finite, numeric) lives in ``_checks`` and is
tested in ``test_checks.py``; only rules specific to a transformer are tested
here.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from enginir.preprocessing import EMSC, MSC, SNV, Detrend, SavitzkyGolay, Smoother

RNG = np.random.default_rng(42)
N, P = 20, 100
X_DUMMY = RNG.random((N, P))


def _reference_emsc(X, *, order, reference=None, interferences=None, weights=None):
    """Textbook EMSC by ``lstsq``: corrected spectra and coefficients."""
    reference = X.mean(axis=0) if reference is None else reference
    A = np.column_stack([np.vander(np.linspace(-1.0, 1.0, X.shape[1]), order + 1, increasing=True),
                         reference])
    if interferences is not None:
        A = np.column_stack([A, interferences.T])
    w = np.ones(X.shape[1]) if weights is None else weights
    coeffs = np.linalg.lstsq(A * w[:, None], (X * w).T, rcond=None)[0]
    additive = coeffs.copy()
    additive[order + 1] = 0.0
    return (X - (A @ additive).T) / coeffs[order + 1][:, None], coeffs.T


def test_snv_uses_the_sample_std_and_zeroes_constant_spectra():
    np.testing.assert_allclose(SNV().transform(np.array([[1.0, 2.0, 3.0]])), [[-1.0, 0.0, 1.0]])

    X = np.full((2, 220), 0.123, dtype=np.float32)  # float32 constant: std is rounding noise
    X[1] = np.linspace(0.0, 1.0, 220)
    out = SNV().transform(X)
    np.testing.assert_array_equal(out[0], 0.0)
    assert out[1].mean() == pytest.approx(0.0, abs=1e-6)
    assert out[1].std(ddof=1) == pytest.approx(1.0, rel=1e-5)


def test_msc_maps_its_reference_onto_itself_and_rejects_a_constant_one():
    msc = MSC().fit(X_DUMMY)
    np.testing.assert_allclose(msc.reference_, X_DUMMY.mean(axis=0))
    # a spectrum a + b * r is mapped back onto r: both scatter terms are removed
    shifted = 0.4 + 2.5 * msc.reference_
    np.testing.assert_allclose(msc.transform(shifted[None]), msc.reference_[None], rtol=1e-10)
    with pytest.raises(ValueError, match="constant"):
        MSC(reference=np.ones(P)).fit(X_DUMMY)


@pytest.mark.parametrize("transformer", [MSC(), EMSC()], ids=["MSC", "EMSC"])
def test_fitted_scatter_correction_rejects_a_different_band_count(transformer):
    transformer.fit(X_DUMMY)
    with pytest.raises(ValueError, match="features"):
        transformer.transform(RNG.random((5, P - 2)))


def test_emsc_matches_the_reference_algorithm_with_interferences_and_weights():
    interferences = RNG.random((2, P))
    weights = np.linspace(0.25, 1.0, P)
    emsc = EMSC(order=1, interferences=interferences, weights=weights).fit(X_DUMMY)
    expected, expected_coeffs = _reference_emsc(
        X_DUMMY, order=1, interferences=interferences, weights=weights
    )
    np.testing.assert_allclose(emsc.transform(X_DUMMY), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(emsc.compute_coefficients(X_DUMMY), expected_coeffs,
                               rtol=1e-12, atol=1e-12)


def test_emsc_reference_can_be_the_median_or_supplied():
    np.testing.assert_allclose(EMSC(method="median", order=0).fit(X_DUMMY).reference_,
                               np.median(X_DUMMY, axis=0))
    reference = X_DUMMY[0].copy()
    expected, _ = _reference_emsc(X_DUMMY, order=2, reference=reference)
    np.testing.assert_allclose(EMSC(reference=reference).fit(X_DUMMY).transform(X_DUMMY),
                               expected, rtol=1e-12, atol=1e-12)


def test_emsc_rejects_invalid_weights_and_rank_deficient_designs():
    with pytest.raises(ValueError, match="non-negative"):
        EMSC(weights=np.r_[-1.0, np.ones(P - 1)]).fit(X_DUMMY)
    with pytest.raises(ValueError, match="positive"):
        EMSC(weights=np.zeros(P)).fit(X_DUMMY)
    with pytest.raises(ValueError, match="interferences have"):
        EMSC(interferences=RNG.random((2, P - 1))).fit(X_DUMMY)
    with pytest.raises(ValueError, match="rank"):  # reference is a degree-1 polynomial
        EMSC(reference=np.linspace(0.0, 1.0, P)).fit(X_DUMMY)
    with pytest.raises(ValueError, match="rank"):  # interference collinear with the offset
        EMSC(interferences=np.ones((1, P))).fit(X_DUMMY)


def test_emsc_keeps_output_finite_for_a_zero_multiplicative_term():
    assert np.isfinite(EMSC(order=0).fit(X_DUMMY).transform(np.zeros((3, P)))).all()


def test_emsc_transform_replays_the_fitted_order_not_the_current_parameter():
    emsc = EMSC(order=2).fit(X_DUMMY)
    expected = emsc.transform(X_DUMMY)
    emsc.set_params(order=1)
    np.testing.assert_array_equal(emsc.transform(X_DUMMY), expected)


def test_savitzky_golay_differentiates_a_polynomial_exactly():
    x = np.arange(30.0)
    X = np.tile(0.5 * x**3, (2, 1))
    np.testing.assert_allclose(SavitzkyGolay(7, 3, deriv=1).transform(X), np.tile(1.5 * x**2, (2, 1)),
                               atol=1e-8)
    np.testing.assert_allclose(SavitzkyGolay(7, 3, deriv=2).transform(X), np.tile(3.0 * x, (2, 1)),
                               atol=1e-8)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"window_length": 4}, "odd"),
        ({"window_length": 5, "polyorder": 5}, "polyorder"),
        ({"window_length": 11, "polyorder": 2, "deriv": 3}, "deriv"),
        ({"window_length": 101}, "exceeds"),
    ],
)
def test_savitzky_golay_rejects_inconsistent_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SavitzkyGolay(**kwargs).transform(X_DUMMY)


def test_detrend_removes_polynomial_baselines_on_a_raw_wavelength_axis():
    wavelengths = np.linspace(991.0, 1677.0, 220)
    baseline = 0.3 + 2e-4 * wavelengths - 1e-7 * wavelengths**2 + 1e-11 * wavelengths**3
    np.testing.assert_allclose(Detrend(3, wavelengths).transform(np.tile(baseline, (4, 1))), 0.0,
                               atol=1e-9)
    # float32 stays accurate because the abscissa is rescaled to [-1, 1]
    X = (baseline + 0.01 * np.sin(wavelengths / 20.0))[None]
    single = Detrend(3, wavelengths).transform(X.astype(np.float32))
    np.testing.assert_allclose(single, Detrend(3, wavelengths).transform(X), atol=1e-7)
    with pytest.raises(ValueError, match="order=30"):
        Detrend(order=30).transform(RNG.random((3, 30)))


def test_moving_average_repeats_edge_values():
    out = Smoother(window_length=5).transform(np.array([[0.0, 5.0, 10.0, 15.0, 20.0]]))
    np.testing.assert_allclose(out, [[3.0, 6.0, 10.0, 14.0, 17.0]])  # edges padded 0,0 / 20,20
    with pytest.raises(ValueError, match="odd"):
        Smoother(window_length=4).transform(X_DUMMY)


def test_whittaker_matches_the_sparse_solution_and_stays_accurate_in_float32():
    X = RNG.standard_normal((4, 32))
    lam, order = 1e3, 2
    D = sp.eye(32, format="csr")
    for _ in range(order):
        D = D[1:] - D[:-1]  # type: ignore[reportIndexIssue]
    A = (sp.eye(32) + lam * D.T @ D).tocsc()
    expected = np.vstack([spla.spsolve(A, row) for row in X])
    np.testing.assert_allclose(Smoother(method="whittaker", lam=lam).transform(X), expected)

    X = np.random.default_rng(1).random((20, 200)) + np.linspace(0.0, 1.0, 200)
    for lam in (1e4, 1e8):  # condition number ~16 lam: solved in float64 blocks
        smoother = Smoother(method="whittaker", lam=lam)
        np.testing.assert_allclose(smoother.transform(X.astype(np.float32)), smoother.transform(X),
                                   atol=1e-5)


@pytest.mark.parametrize(
    "transformer",
    [SNV(), MSC(), EMSC(), Detrend(order=2, wavelengths=np.linspace(1000, 1700, 220)),
     Smoother(method="whittaker"), Smoother(), SavitzkyGolay()],
    ids=lambda t: f"{type(t).__name__}-{getattr(t, 'method', '')}",
)
def test_transformers_work_in_float32_without_full_size_upcasts(transformer):
    """Working arrays stay float32 (prediction-runtime contract, memory rules)."""
    X = np.random.default_rng(0).random((5000, 220)).astype(np.float32)
    transformer.fit(X)
    tracemalloc.start()
    out = transformer.transform(X)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert out.dtype == np.float32
    assert peak < 2.0 * X.nbytes  # a float64 copy of X alone would be 2x
