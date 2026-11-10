"""Applicability diagnostics of predictions: DModXPS and T²PS (prediction-runtime contract).

Computed per chunk from the model's stored projection data (``x_mean``,
``x_std``, ``rotations``, ``loadings``, ``score_variances``, ``s0``) on the
preprocessed spectra, in small row blocks so that memory stays bounded.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .._logging import get_logger

logger = get_logger(__name__)

_BLOCK_ROWS = 4096
_EPS = 1e-10
_FIELDS = ("x_mean", "x_std", "rotations", "loadings", "score_variances")


@dataclass(frozen=True)
class Projection:
    """The projection data of a model (``None`` from :func:`projection_of` without them)."""

    x_mean: np.ndarray
    x_std: np.ndarray
    rotations: np.ndarray
    loadings: np.ndarray
    score_variances: np.ndarray
    s0: float | None
    t2_crit: float | None
    dcrit: float | None

    @property
    def n_components(self) -> int:
        return int(self.rotations.shape[1])


def projection_of(blob: dict[str, Any] | None) -> Projection | None:
    """The projection data of a saved ``prediction_model``, or ``None`` (runs before 1.3)."""
    if not blob or any(key not in blob for key in _FIELDS):
        return None

    def limit(key: str) -> float | None:
        return None if blob.get(key) is None else float(np.asarray(blob[key]))

    rotations = np.atleast_2d(np.asarray(blob["rotations"], dtype=np.float64))
    return Projection(
        x_mean=np.asarray(blob["x_mean"], dtype=np.float64).reshape(-1),
        x_std=np.asarray(blob["x_std"], dtype=np.float64).reshape(-1),
        rotations=rotations,
        loadings=np.atleast_2d(np.asarray(blob["loadings"], dtype=np.float64)),
        score_variances=np.asarray(blob["score_variances"], dtype=np.float64).reshape(-1),
        s0=limit("s0"),
        t2_crit=limit("t2_crit_099"),
        dcrit=limit("dcrit_099"),
    )


def chunk_diagnostics(X: np.ndarray, projection: Projection) -> tuple[np.ndarray, np.ndarray]:
    """T²PS and DModXPS of every row of a preprocessed chunk, as float32.

    T²PS = Σₐ tₐ² / var(tₐ); DModXPS = √(Σₖ eₖ² / (K − A)) / s0 (no factor v,
    no modelling-power weighting), with z = (x − x_mean) / x_std, t = z R,
    e = z − t Pᵀ. DModXPS is NaN when the model has no usable ``s0``.
    """
    n_rows, n_bands = X.shape
    a = projection.n_components
    t2 = np.empty(n_rows, dtype=np.float32)
    distance = np.empty(n_rows, dtype=np.float32)
    dof = max(n_bands - a, 1)
    s0 = projection.s0
    if s0 is not None and s0 < _EPS:
        logger.warning("Training residual SD s0 = {} is below {}; DModXPS is not computed", s0, _EPS)
        s0 = None
    for start in range(0, n_rows, _BLOCK_ROWS):
        z = (np.asarray(X[start:start + _BLOCK_ROWS], dtype=np.float64)
             - projection.x_mean) / projection.x_std
        t = z @ projection.rotations
        t2[start:start + len(z)] = (t ** 2 / projection.score_variances).sum(axis=1)
        z -= t @ projection.loadings.T
        distance[start:start + len(z)] = (np.nan if s0 is None
                                          else np.sqrt(np.einsum("ij,ij->i", z, z) / dof) / s0)
    return t2, distance


def chunk_scores(X: np.ndarray, projection: Projection) -> np.ndarray:
    """Scores t = z R of every row of a preprocessed chunk, ``(n_rows, A)`` float32."""
    scores = np.empty((X.shape[0], projection.n_components), dtype=np.float32)
    for start in range(0, X.shape[0], _BLOCK_ROWS):
        z = (np.asarray(X[start:start + _BLOCK_ROWS], dtype=np.float64)
             - projection.x_mean) / projection.x_std
        scores[start:start + len(z)] = z @ projection.rotations
    return scores


@dataclass
class DiagnosticsSummary:
    """Pixels above the limits and the mean prediction inside and outside them."""

    t2_crit: float | None
    dcrit: float | None
    n_pixels: int = 0
    above_t2: int = 0
    above_dcrit: int = 0
    _sum_inside: float = 0.0
    _n_inside: int = 0
    _sum_outside: float = 0.0
    _n_outside: int = 0
    _seen: bool = field(default=False, repr=False)

    def add(self, y: np.ndarray, t2: np.ndarray, distance: np.ndarray) -> None:
        finite = np.isfinite(t2) & np.isfinite(distance)
        over_t2 = (t2 > self.t2_crit) if self.t2_crit is not None else np.zeros_like(finite)
        over_d = (distance > self.dcrit) if self.dcrit is not None else np.zeros_like(finite)
        self.n_pixels += int(finite.sum())
        self.above_t2 += int((over_t2 & finite).sum())
        self.above_dcrit += int((over_d & finite).sum())
        if y.ndim == 1:
            inside = finite & ~over_t2 & ~over_d & np.isfinite(y)
            outside = finite & (over_t2 | over_d) & np.isfinite(y)
            self._sum_inside += float(y[inside].sum())
            self._n_inside += int(inside.sum())
            self._sum_outside += float(y[outside].sum())
            self._n_outside += int(outside.sum())
        self._seen = True

    def as_dict(self) -> dict[str, Any]:
        """The summary values (limits, counts, shares, mean predictions)."""
        n = max(self.n_pixels, 1)
        return {
            "t2_crit_099": self.t2_crit, "dcrit_099": self.dcrit,
            "n_pixels": self.n_pixels,
            "n_above_t2crit": self.above_t2, "share_above_t2crit": self.above_t2 / n,
            "n_above_dcrit": self.above_dcrit, "share_above_dcrit": self.above_dcrit / n,
            "mean_prediction_within_limits": (self._sum_inside / self._n_inside
                                              if self._n_inside else float("nan")),
            "mean_prediction_outside_limits": (self._sum_outside / self._n_outside
                                               if self._n_outside else float("nan")),
        }


def file_records(path: str, *, checksum: bool = False) -> list[dict[str, Any]]:
    """Path, size and modification time of a data file (and its ENVI header or data file).

    With ``checksum`` the SHA-256 of each file is added (one more full read).
    """
    paths = [os.path.abspath(path)]
    stem, extension = os.path.splitext(paths[0])
    if extension.lower() == ".hdr":
        paths += [stem + ext for ext in ("", ".raw", ".img", ".dat", ".bil", ".bip", ".bsq")
                  if os.path.isfile(stem + ext)][:1]
    elif os.path.isfile(stem + ".hdr"):
        paths.append(stem + ".hdr")
    records = []
    for file in paths:
        if not os.path.isfile(file):
            records.append({"path": file})
            continue
        stat = os.stat(file)
        record: dict[str, Any] = {
            "path": file, "size": int(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
        if checksum:
            digest = hashlib.sha256()
            with open(file, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(block)
            record["sha256"] = digest.hexdigest()
        records.append(record)
    return records


def log_prediction(provenance: dict[str, Any], summary: dict[str, Any] | None) -> None:
    """The one log entry of a prediction (prediction-runtime contract, Result Handling)."""
    logger.info("Prediction with model run '{}' ({}) of {}: {}", provenance.get("model_run_id"),
                provenance.get("model_name"), provenance.get("files"),
                summary if summary is not None else "no applicability diagnostics")


__all__ = ["DiagnosticsSummary", "Projection", "chunk_diagnostics", "file_records",
           "log_prediction", "projection_of"]
