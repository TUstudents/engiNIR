"""Small builders for complete current-schema project test payloads."""

from __future__ import annotations


def roi_provenance(
    *,
    dataset_id: str = "dataset_1",
    calibration_id: str = "cal_1",
    signal_unit: str = "DN",
    created_at: str = "2026-04-02T10:05:00+00:00",
) -> dict[str, str]:
    """Return the required v3 ROI provenance fields."""
    return {
        "dataset_id": dataset_id,
        "calibration_id": calibration_id,
        "signal_unit": signal_unit,
        "preview_recipe_summary": "none",
        "created_at": created_at,
    }


def recipe_payload(**overrides: object) -> dict[str, object]:
    """Return all required fields for one persisted v3 preprocessing recipe."""
    payload: dict[str, object] = {
        "filter": "none",
        "sg_window": 11,
        "sg_polyorder": 2,
        "snv": False,
        "msc": False,
        "emsc": False,
        "detrend": False,
        "detrend_order": 1,
        "sg_edges": "drop",
        "asls": False,
        "asls_lambda": 10000.0,
        "asls_p": 0.001,
        "asls_iterations": 10,
        "bands_keep": [],
        "bands_exclude": [],
    }
    payload.update(overrides)
    return payload


__all__ = ["recipe_payload", "roi_provenance"]


def projection_payload(n_bands: int, n_components: int = 1) -> dict:
    """The projection data every stored PLS/PCR model has (core of the format)."""
    import numpy as np

    components = np.zeros((n_bands, n_components))
    components[0] = 1.0
    return {"x_mean": np.zeros(n_bands), "x_std": np.ones(n_bands),
            "rotations": components, "loadings": components,
            "score_variances": np.ones(n_components), "s0": 1.0, "n_train": 10,
            "t2_crit_099": 10.0, "dcrit_099": 2.0}
