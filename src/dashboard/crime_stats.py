"""
Official open crime-statistics fetcher.

Honesty contract:
    - Only integrates real open official APIs.
    - First integration: data.police.uk (England, Wales, Northern Ireland).
    - Everywhere else → a declared jurisdiction gap, never a proxy or fabricated
      crime score.
    - No financial or loss metrics are computed from crime data.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date
from typing import Any, Dict, List, Optional

from .cache import cached
from .real_data import _UA, _valid_point

TTL_CRIME_STATS = 7 * 24 * 3600.0  # crime statistics are released monthly
_OHSOME_URL = "https://api.ohsome.org/v1/elements/count"

_DATA_POLICE_UK_URL = "https://data.police.uk/api/crimes-street/all-crime"
_GB_BOUNDS = {
    "lat_min": 49.9,
    "lat_max": 60.9,
    "lon_min": -8.6,
    "lon_max": 1.8,
}


def _last_complete_months(n: int = 6) -> List[str]:
    """Return the last ``n`` complete months as 'YYYY-MM' strings."""
    today = date.today()
    # The most recent *complete* month is the previous calendar month.
    year, month = today.year, today.month - 1
    if month == 0:
        year, month = year - 1, 12
    months: List[str] = []
    for _ in range(n):
        months.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return months


def _get_json(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@cached("street_crime", TTL_CRIME_STATS)
def fetch_street_crime(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch official street-crime statistics for a point.

    Coverage: data.police.uk covers England, Wales and Northern Ireland. For
    points outside that jurisdiction the function returns an honest declared
    gap. It never invents a crime score or uses a proxy source.
    """
    if not _valid_point(lat, lon):
        return {"error": "Coordinates out of range"}

    if not (
        _GB_BOUNDS["lat_min"] <= lat <= _GB_BOUNDS["lat_max"]
        and _GB_BOUNDS["lon_min"] <= lon <= _GB_BOUNDS["lon_max"]
    ):
        return {
            "jurisdiction_gap": True,
            "claim_status": "UNKNOWN",
            "reason": (
                "No official open crime-statistics source is integrated for this "
                "jurisdiction (only data.police.uk is integrated). Declared gap — "
                "not a zero-crime statement."
            ),
        }

    months = _last_complete_months(6)
    by_month: List[Dict[str, Any]] = []
    by_category: Dict[str, int] = {}
    total = 0
    errors: List[str] = []
    successful_months = 0

    for month in months:
        params = urllib.parse.urlencode(
            {"lat": lat, "lng": lon, "date": month}
        )
        url = f"{_DATA_POLICE_UK_URL}?{params}"
        try:
            records = _get_json(url, timeout=15.0)
            if not isinstance(records, list):
                errors.append(f"{month}: unexpected response type")
                continue
            count = len(records)
            total += count
            successful_months += 1
            for record in records:
                cat = (record.get("category") or "unknown").lower()
                by_category[cat] = by_category.get(cat, 0) + 1
            by_month.append({"month": month, "total": count})
        except Exception as exc:
            errors.append(f"{month}: {exc}")

    if successful_months == 0:
        return {
            "claim_status": "UNKNOWN",
            "reason": f"data.police.uk could not be reached for any month: {'; '.join(errors)}",
        }

    # Top categories by volume, capped at 6.
    top_categories = sorted(
        [{"category": k, "count": v} for k, v in by_category.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:6]

    return {
        "claim_status": "OBSERVED",
        "source": "data.police.uk — official UK police open data",
        "source_url": "https://data.police.uk/",
        "period": f"{months[-1]} to {months[0]}",
        "months_requested": len(months),
        "months_returned": successful_months,
        "latest_period": months[0],
        "monthly_points": by_month,
        "total": total,
        "by_category": top_categories,
        "coverage_note": (
            "Street-level crime recorded by UK police forces within approximately "
            "1 mile of the point. Counts reflect police-recorded incidents, not "
            "all crime."
        ),
        "limitations": errors if errors else None,
    }
