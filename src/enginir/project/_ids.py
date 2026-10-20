"""Ids and timestamps of project objects."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def new_id() -> str:
    """Return a new random object id (a UUID4 string)."""
    return str(uuid.uuid4())


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


__all__ = ["new_id", "now_iso"]
