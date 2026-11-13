"""Fits on sufficient statistics: kernel PLS and PCA from XᵀX (plan ``plans/kernel-pls.md``).

A calibration set has many more pixels than bands. One pass over the rows
builds, per block of rows, the sums n, Σx, XᵀX, Σy, YᵀY and XᵀY; every fit,
fold, component count and band subset then works on those K × K matrices.

- :class:`GramStats` holds the sums per block (engiNIR's blocks are fold
  groups: folds never split one) and, per ROI, n, Σx and Σy (its mean
  spectrum and value).
- :class:`Moments` are the centered sums of a set of blocks, with the
  column scaling of contract §7 (standard deviations with N − 1).
- :func:`kernel_pls` is the improved kernel PLS of Dayal and MacGregor
  (1997, algorithm 2); :func:`gram_pca` the eigendecomposition of the
  centered, scaled XᵀX. Both follow scikit-learn's sign convention.

Rows arrive as float32; the sums are float64, built chunk by chunk with a
fixed shift (the mean of the first chunk) subtracted, so raw counts around
3000 do not cancel when centering.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._checks import log_and_raise
from .._logging import get_logger
from ._common import SCALINGS

logger = get_logger(__name__)

_EPS = 1e-10
#: Rows per chunk of a pass over the pixels (float64 chunk: 16 384 × K × 8 B).
CHUNK_ROWS = 16_384
#: Environment variable that sets the BLAS threads of the engine's passes.
BLAS_THREADS_ENV = "ENGINIR_BLAS_THREADS"
#: PCR uses ``gram`` when σ_A / σ₁ is at least this (sample-set-model-run contract §7).
GRAM_CONDITION_LIMIT = 1e-6


def blas_thread_count() -> int:
    """BLAS threads of the engine's passes: ``$ENGINIR_BLAS_THREADS``, else half the CPUs.

    OpenBLAS with every logical CPU can collapse on the tall, narrow products
    of a pass when other work shares the machine (measured 2026-11-13: about
    2 GFLOP/s with 10 threads, 170–190 with 4–6).
    """
    value = os.environ.get(BLAS_THREADS_ENV, "").strip()
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            logger.warning("{}={!r} is not a number; using the default", BLAS_THREADS_ENV, value)
    return max(1, (os.cpu_count() or 2) // 2)


#: Passes smaller than this many matrix elements run without the cap (it costs a
#: scan of the loaded libraries, and small products do not oversubscribe).
_CAP_MIN_ELEMENTS = 2_000_000
_controller: Any = None


@contextmanager
def blas_threads(n_elements: int | None = None) -> Iterator[None]:
    """Cap the BLAS threads for one pass over ``n_elements`` matrix elements.

    The cap is :func:`blas_thread_count`; passes below 2·10⁶ elements run
    uncapped.
    """
    global _controller
    if n_elements is not None and n_elements < _CAP_MIN_ELEMENTS:
        yield
        return
    if _controller is None:
        from threadpoolctl import ThreadpoolController

        _controller = ThreadpoolController()
    with _controller.limit(limits=blas_thread_count(), user_api="blas"):
        yield


def _as_2d(y: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64)
    return values.reshape(n, -1)


@dataclass
class GramStats:
    """Sums over rows per block, and n, Σx, Σy per ROI.

    Attributes
    ----------
    shift_x, shift_y:
        Subtracted from every row before summing (float64).
    n:
        Rows per block, shape ``(B,)``.
    sx, sy:
        Σ(x − shift_x), Σ(y − shift_y) per block, shapes ``(B, K)`` and ``(B, T)``.
    sxx, sxy, syy:
        Cross products of the shifted rows per block, shapes ``(B, K, K)``,
        ``(B, K, T)`` and ``(B, T, T)``.
    roi_n, roi_sx, roi_sy:
        n, Σ(x − shift_x) and Σ(y − shift_y) per ROI.
    roi_block:
        Block of every ROI, shape ``(R,)``.
    """

    shift_x: np.ndarray
    shift_y: np.ndarray
    n: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    sxx: np.ndarray
    sxy: np.ndarray
    syy: np.ndarray
    roi_n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    roi_sx: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    roi_sy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    roi_block: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def n_features(self) -> int:
        return int(self.sx.shape[1])

    @property
    def n_targets(self) -> int:
        return int(self.sy.shape[1])

    @property
    def n_blocks(self) -> int:
        return int(self.n.shape[0])

    @classmethod
    def from_ranges(cls, X: np.ndarray, Y: np.ndarray,
                    ranges: Iterable[tuple[int, int, int, int]], *, n_blocks: int,
                    n_rois: int, keep: np.ndarray | None = None) -> GramStats:
        """Sum the rows of ``X`` and ``Y`` over ``(start, stop, block, roi)`` row ranges.

        One pass, chunk by chunk, under :func:`blas_threads`. Rows where
        ``keep`` is ``False`` are left out (the DModX exclusion).
        """
        X = np.asarray(X)
        Y = _as_2d(Y, X.shape[0]) if np.ndim(Y) == 1 else np.asarray(Y, dtype=np.float64)
        ranges = list(ranges)
        k, t = X.shape[1], Y.shape[1]
        first = next(((a, b) for a, b, _blk, _roi in ranges if b > a), None)
        if first is None:
            log_and_raise("No rows to sum")
        head = X[first[0]:min(first[1], first[0] + 4096)]
        shift_x = np.asarray(head, dtype=np.float64).mean(axis=0)
        shift_y = Y[first[0]].astype(np.float64)
        stats = cls(shift_x=shift_x, shift_y=shift_y, n=np.zeros(n_blocks),
                    sx=np.zeros((n_blocks, k)), sy=np.zeros((n_blocks, t)),
                    sxx=np.zeros((n_blocks, k, k)), sxy=np.zeros((n_blocks, k, t)),
                    syy=np.zeros((n_blocks, t, t)), roi_n=np.zeros(n_rois),
                    roi_sx=np.zeros((n_rois, k)), roi_sy=np.zeros((n_rois, t)),
                    roi_block=np.full(n_rois, -1, dtype=np.int64))
        with blas_threads(sum(b - a for a, b, _blk, _roi in ranges) * k):
            for start, stop, block, roi in ranges:
                stats.roi_block[roi] = block
                for lo in range(start, stop, CHUNK_ROWS):
                    hi = min(lo + CHUNK_ROWS, stop)
                    stats._add(X[lo:hi], Y[lo:hi], block, roi,
                               None if keep is None else keep[lo:hi])
        return stats

    @classmethod
    def from_rows(cls, X: np.ndarray, y: np.ndarray) -> GramStats:
        """The sums of all rows as one block (and one ROI)."""
        X = np.asarray(X)
        return cls.from_ranges(X, y, [(0, X.shape[0], 0, 0)], n_blocks=1, n_rois=1)

    def _add(self, x: np.ndarray, y: np.ndarray, block: int, roi: int,
             keep: np.ndarray | None, sign: float = 1.0) -> None:
        if keep is not None:
            x, y = x[keep], y[keep]
        if x.shape[0] == 0:
            return
        xs = np.asarray(x, dtype=np.float64) - self.shift_x
        ys = np.asarray(y, dtype=np.float64) - self.shift_y
        n = sign * xs.shape[0]
        sx, sy = sign * xs.sum(axis=0), sign * ys.sum(axis=0)
        self.n[block] += n
        self.sx[block] += sx
        self.sy[block] += sy
        self.sxx[block] += sign * (xs.T @ xs)
        self.sxy[block] += sign * (xs.T @ ys)
        self.syy[block] += sign * (ys.T @ ys)
        self.roi_n[roi] += n
        self.roi_sx[roi] += sx
        self.roi_sy[roi] += sy

    def remove_rows(self, X: np.ndarray, Y: np.ndarray, rows: np.ndarray,
                    block_of_row: np.ndarray, roi_of_row: np.ndarray) -> None:
        """Take rows out of their blocks and ROIs (a downdate; the DModX exclusion)."""
        Y = _as_2d(Y, X.shape[0]) if np.ndim(Y) == 1 else np.asarray(Y, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.int64)
        for block in np.unique(block_of_row[rows]):
            in_block = rows[block_of_row[rows] == block]
            for roi in np.unique(roi_of_row[in_block]):
                take = in_block[roi_of_row[in_block] == roi]
                for lo in range(0, take.size, CHUNK_ROWS):
                    chunk = take[lo:lo + CHUNK_ROWS]
                    self._add(X[chunk], Y[chunk], int(block), int(roi), None, sign=-1.0)

    def copy(self) -> GramStats:
        return GramStats(**{name: np.array(getattr(self, name), copy=True)
                            for name in self.__dataclass_fields__})

    def moments(self, blocks: Iterable[int] | np.ndarray | None = None,
                columns: np.ndarray | None = None) -> Moments:
        """Centered sums of ``blocks`` (default all), on ``columns`` (default all)."""
        index = (np.arange(self.n_blocks) if blocks is None
                 else np.asarray(list(blocks) if not isinstance(blocks, np.ndarray) else blocks,
                                 dtype=np.int64))
        cols = slice(None) if columns is None else np.asarray(columns, dtype=np.int64)
        n = float(self.n[index].sum())
        if n < 1:
            log_and_raise("No rows in the selected blocks")
        sx = self.sx[index].sum(axis=0)[cols]
        sy = self.sy[index].sum(axis=0)
        sxx = self.sxx[index].sum(axis=0)
        sxy = self.sxy[index].sum(axis=0)
        if columns is not None:
            sxx = sxx[np.ix_(cols, cols)]  # type: ignore[arg-type]
            sxy = sxy[cols]
        syy = self.syy[index].sum(axis=0)
        mx, my = sx / n, sy / n
        return Moments(n=n, x_mean=self.shift_x[cols] + mx, y_mean=self.shift_y + my,
                       cxx=sxx - n * np.outer(mx, mx), cxy=sxy - n * np.outer(mx, my),
                       cyy=syy - n * np.outer(my, my))

    def roi_means(self, columns: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Mean spectrum and mean y of every ROI, shapes ``(R, K)`` and ``(R, T)``."""
        cols = slice(None) if columns is None else np.asarray(columns, dtype=np.int64)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = self.roi_sx[:, cols] / self.roi_n[:, None] + self.shift_x[cols]
            y = self.roi_sy / self.roi_n[:, None] + self.shift_y
        return x, y


@dataclass
class Moments:
    """Centered sums of a set of rows: n, the means and the centered cross products."""

    n: float
    x_mean: np.ndarray
    y_mean: np.ndarray
    cxx: np.ndarray
    cxy: np.ndarray
    cyy: np.ndarray

    def x_scale(self, scaling: str) -> np.ndarray:
        """The divisor of every column for ``scaling`` (contract §7; constant columns: 1)."""
        return _divisor(np.diag(self.cxx), self.n, scaling, "column")

    def y_scale(self, scaling: str) -> np.ndarray:
        return _divisor(np.diag(self.cyy), self.n, scaling, "target")


def _divisor(sum_of_squares: np.ndarray, n: float, scaling: str, what: str) -> np.ndarray:
    if scaling not in SCALINGS:
        log_and_raise("Unknown scaling {!r}", scaling)
    if scaling == "center":
        return np.ones_like(sum_of_squares)
    variance = np.clip(sum_of_squares, 0.0, None) / max(n - 1.0, 1.0)
    std = np.sqrt(variance)
    constant = std < _EPS
    if np.any(constant):
        logger.warning("{} constant {}(s) are only centered, not scaled ({})",
                       int(constant.sum()), what, scaling)
        std = np.where(constant, 1.0, std)
    return std if scaling == "uv" else np.sqrt(std)


def _flip(vector: np.ndarray) -> float:
    """scikit-learn's sign: the largest absolute element positive (``_svd_flip_1d``)."""
    return float(np.sign(vector[np.argmax(np.abs(vector))]) or 1.0)


@dataclass
class PLSFactors:
    """A kernel PLS fit in the scaled space Z = (X − x_mean) / x_std, Ys likewise."""

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    weights: np.ndarray      # W (K, A)
    loadings: np.ndarray     # P (K, A)
    y_loadings: np.ndarray   # Q (T, A)
    rotations: np.ndarray    # R = W (PᵀW)⁻¹ (K, A)
    score_ss: np.ndarray     # tₐᵀtₐ (A,)
    gram: np.ndarray         # ZᵀZ (K, K)
    gram_y: np.ndarray       # ZᵀYs (K, T)

    def coefficients(self, n_components: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Coefficients (K, T) and intercept (T,) in raw units with the first *a* components."""
        a = self.rotations.shape[1] if n_components is None else int(n_components)
        scaled = self.rotations[:, :a] @ self.y_loadings[:, :a].T
        coef = scaled / self.x_std[:, None] * self.y_std[None, :]
        return coef, self.y_mean - self.x_mean @ coef


def kernel_pls(moments: Moments, n_components: int, *, x_scaling: str,
               y_scaling: str) -> PLSFactors:
    """Improved kernel PLS (Dayal and MacGregor 1997, algorithm 2) on centered sums.

    Equal to NIPALS (scikit-learn's ``PLSRegression``, regression mode) in exact
    arithmetic; signs as scikit-learn's. PLS2 takes each weight vector as the
    dominant left singular vector of the deflated ZᵀYs.

    When the data support fewer components (the X or Y residual is zero,
    e.g. after SNV, which lowers the rank by one), the remaining components
    are zero, with a warning, as scikit-learn's NIPALS leaves them: their
    predictions equal those of the last supported component.
    """
    x_std = moments.x_scale(x_scaling)
    y_std = moments.y_scale(y_scaling)
    gram = moments.cxx / np.outer(x_std, x_std)
    gram_y = moments.cxy / x_std[:, None] / y_std[None, :]
    k, t = gram_y.shape
    a_max = int(n_components)
    W, P, R = np.zeros((k, a_max)), np.zeros((k, a_max)), np.zeros((k, a_max))
    Q, tt_all = np.zeros((t, a_max)), np.zeros(a_max)
    xy = gram_y.copy()
    scale = max(float(np.abs(gram_y).max()), _EPS)
    for a in range(a_max):
        if t == 1:
            w = xy[:, 0].copy()
        else:
            u, _s, _vt = np.linalg.svd(xy, full_matrices=False)
            w = u[:, 0].copy()
        norm = float(np.linalg.norm(w))
        if norm < _EPS * scale:
            logger.warning("Kernel PLS: the residual is exhausted after {} of {} components; "
                           "the rest are zero", a, a_max)
            break
        w /= norm
        w *= _flip(w)
        r = w.copy()
        for j in range(a):
            r -= (P[:, j] @ w) * R[:, j]
        gr = gram @ r
        tt = float(r @ gr)
        if tt < _EPS * max(float(np.trace(gram)), _EPS):
            logger.warning("Kernel PLS: the X residual is exhausted after {} of {} components; "
                           "the rest are zero", a, a_max)
            break
        p = gr / tt
        q = (xy.T @ r) / tt
        xy -= tt * np.outer(p, q)
        W[:, a], P[:, a], R[:, a], Q[:, a], tt_all[a] = w, p, r, q, tt
    return PLSFactors(x_mean=moments.x_mean, x_std=x_std, y_mean=moments.y_mean, y_std=y_std,
                      weights=W, loadings=P, y_loadings=Q, rotations=R, score_ss=tt_all,
                      gram=gram, gram_y=gram_y)


@dataclass
class PCAFactors:
    """A PCA of the centered, scaled X from its cross products."""

    x_mean: np.ndarray       # the column means (raw units)
    x_std: np.ndarray        # the column divisors
    components: np.ndarray   # (A, K), rows are principal axes
    eigenvalues: np.ndarray  # λ of ZᵀZ for every component (K,), descending
    total_ss: float          # trace(ZᵀZ)
    gram: np.ndarray         # ZᵀZ (K, K)
    gram_y: np.ndarray       # Zᵀ(y − ȳ) (K, T), unscaled y


def scaled_eigenvalues(moments: Moments, x_scaling: str) -> tuple[np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray]:
    """(eigenvalues descending, eigenvectors, ZᵀZ, x_std) of the centered, scaled X."""
    x_std = moments.x_scale(x_scaling)
    gram = moments.cxx / np.outer(x_std, x_std)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order], gram, x_std


def gram_condition(moments: Moments, n_components: int, x_scaling: str) -> float:
    """σ_A / σ₁ of the centered, scaled X (the PCR guard, contract §7)."""
    values = scaled_eigenvalues(moments, x_scaling)[0]
    a = min(int(n_components), values.size) - 1
    if values[0] <= 0:
        return 0.0
    return float(np.sqrt(max(values[a], 0.0) / values[0]))


def gram_pca(moments: Moments, n_components: int, *, x_scaling: str) -> PCAFactors:
    """PCA from ZᵀZ (eigendecomposition); signs as scikit-learn's PCA."""
    values, vectors, gram, x_std = scaled_eigenvalues(moments, x_scaling)
    a = int(n_components)
    components = vectors[:, :a].T.copy()
    for i in range(a):
        components[i] *= _flip(components[i])
    gram_y = moments.cxy / x_std[:, None]
    return PCAFactors(x_mean=moments.x_mean, x_std=x_std, components=components,
                      eigenvalues=np.clip(values, 0.0, None), total_ss=float(np.trace(gram)),
                      gram=gram, gram_y=gram_y)


def chunks(start: int, stop: int, size: int = CHUNK_ROWS) -> Iterator[slice]:
    """Row slices of at most ``size`` rows covering ``start … stop``."""
    for lo in range(start, stop, size):
        yield slice(lo, min(lo + size, stop))


__all__ = ["BLAS_THREADS_ENV", "CHUNK_ROWS", "GRAM_CONDITION_LIMIT", "GramStats", "Moments",
           "PCAFactors", "PLSFactors", "blas_thread_count", "blas_threads", "chunks",
           "gram_condition", "gram_pca", "kernel_pls", "scaled_eigenvalues"]
