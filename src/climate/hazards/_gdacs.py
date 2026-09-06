"""
Shared GDACS event flattening for hazard modules (flood, volcanic, …).

The cyclone module keeps its own TC-specific flattener (storm semantics
differ); this helper covers the multi-hazard feeds wired in the 2026-09
gradual engine wiring (``FL`` flood alerts, ``VO`` volcanic-activity
alerts). Records carry the official alert level, validity window, affected
countries and the originating centre — monitoring context, never a
forecast.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def flatten_gdacs_event(
    feature: Dict[str, Any],
    lat: float,
    lon: float,
    event_type: str,
) -> Optional[Dict[str, Any]]:
    """Flatten one GDACS feature of ``event_type`` into the platform's
    event shape; None when the feature is of another type or malformed."""
    props = feature.get("properties") or {}
    if props.get("eventtype") != event_type:
        return None
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") or []
    if len(coords) < 2:
        return None
    elon, elat = coords[0], coords[1]
    urls = props.get("url") or {}
    sev = props.get("severitydata") or {}
    return {
        "id": f"gdacs-{event_type.lower()}-{props.get('eventid')}-{props.get('episodeid')}",
        "name": props.get("name") or props.get("eventname") or f"GDACS {event_type} event",
        "lat": float(elat),
        "lon": float(elon),
        "alert_level": props.get("episodealertlevel") or props.get("alertlevel"),
        "alert_score": props.get("episodealertscore", props.get("alertscore")),
        "from_date": props.get("fromdate"),
        "to_date": props.get("todate"),
        "countries": props.get("country") or "",
        "warning_centre": props.get("source") or "GDACS",
        "severity": {
            "value": sev.get("severity"),
            "unit": sev.get("severityunit"),
            "text": sev.get("severitytext"),
        } if sev else None,
        "report_url": urls.get("report"),
        "distance_km": round(haversine_km(lat, lon, float(elat), float(elon)), 1),
    }
