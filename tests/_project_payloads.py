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
    }
    payload.update(overrides)
    return payload


__all__ = ["recipe_payload", "roi_provenance"]
