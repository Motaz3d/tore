"""
Multi-source fire-evidence layer.

Talaix does not depend on one fire source. Every contributor keeps its
identity; detections from different sensors/products are NEVER merged into
one unexplained number. When sources disagree, the disagreement is shown
with a declared interpretation note.

Terminology (kept strict everywhere):
    ACTIVE FIRE DETECTION  — a satellite hotspot detection (point, time,
                             confidence, FRP). NOT an exact fire perimeter.
    BURNED AREA            — a mapped post-fire scar (area product).
    HISTORICAL FIRE        — an active-fire detection from a past period
                             used as a validation label.

Currently integrated (real, when FIRMS_MAP_KEY is configured):
    - NASA FIRMS VIIRS S-NPP NRT (375 m)
    - NASA FIRMS MODIS NRT (1 km)

Candidate sources are documented in config/source_registry.json with their
integration status and the reason — nothing is claimed before it works.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import real_data


def _firms_entry(sensor: str, lat: float, lon: float,
                 radius_km: float, days: int) -> Dict:
    """One FIRMS product as an evidence entry (real or honestly unavailable)."""
    product = real_data.FIRMS_PRODUCTS[sensor]
    res = real_data.fetch_active_fires(lat, lon, radius_km=radius_km,
                                       days=days, sensor=sensor)
    base = {
        "id": f"firms_{sensor.lower()}",
        "source": "NASA FIRMS",
        "sensor": product["sensor"],
        "product": sensor,
        "kind": "observed" if res.get("available") else "unavailable",
        "observation_type": "active_fire_detection",
        "resolution": product["resolution"],
        "freshness": f"last {days} day(s), near-real-time",
        "source_label": product["label"],
    }
    if res.get("available"):
        base.update({
            "status": "ok",
            "count": res.get("count", 0),
            "detections": res.get("fires") or [],
            "limitations": "Hotspot detections, not fire perimeters; cloud "
                           "cover and overpass timing can miss fires.",
        })
    else:
        base.update({
            "status": "unavailable",
            "reason": res.get("error"),
            "signup": res.get("signup"),
            "detections": [],
        })
    return base


def _disagreement_note(entries: List[Dict]) -> Optional[str]:
    """
    Detect and explain inter-source disagreement (only between sources that
    actually returned data). Never resolves it silently.
    """
    ok = [e for e in entries if e.get("status") == "ok"]
    if len(ok) < 2:
        return None
    counts = {e["sensor"]: e.get("count", 0) for e in ok}
    vals = set(counts.values())
    if len(vals) <= 1:
        return ("Sources agree: " +
                ", ".join(f"{s}: {c}" for s, c in counts.items()) + ".")
    # Genuine disagreement — show it with the declared interpretation.
    return (
        "Sources disagree: " +
        ", ".join(f"{s} reported {c} detection(s)" for s, c in counts.items()) +
        ". Interpretation note: VIIRS (375 m) routinely detects smaller/cooler "
        "fires than MODIS (1 km), so a higher VIIRS count is expected and is "
        "not automatically an error. Both counts are shown; neither is "
        "discarded."
    )


def build_fire_evidence(lat: float, lon: float,
                        radius_km: float = 50.0, days: int = 5) -> Dict:
    """
    Build the transparent multi-source fire-evidence block for a location.

    Returns one entry per integrated source (with its own status), the
    disagreement note when applicable, and the strict observation-type
    labels. No key configured -> every entry is honestly unavailable.
    """
    entries: List[Dict] = []
    for sensor in ("VIIRS_SNPP_NRT", "MODIS_NRT"):
        try:
            entries.append(_firms_entry(sensor, lat, lon, radius_km, days))
        except Exception as exc:
            product = real_data.FIRMS_PRODUCTS[sensor]
            entries.append({
                "id": f"firms_{sensor.lower()}",
                "source": "NASA FIRMS",
                "sensor": product["sensor"],
                "product": sensor,
                "kind": "unavailable",
                "observation_type": "active_fire_detection",
                "resolution": product["resolution"],
                "status": "unavailable",
                "reason": str(exc),
                "detections": [],
            })

    total = sum(e.get("count", 0) for e in entries if e.get("status") == "ok")
    any_available = any(e.get("status") == "ok" for e in entries)

    return {
        "status": "ok" if any_available else "unavailable",
        "entries": entries,
        "total_detections": total if any_available else None,
        "disagreement": _disagreement_note(entries),
        "observation_types_note": (
            "ACTIVE FIRE DETECTION = satellite hotspot (point, time, "
            "confidence, FRP) — not a fire perimeter. BURNED AREA = mapped "
            "post-fire scar (no such product is currently integrated; see the "
            "source registry). HISTORICAL FIRE = a past detection used as a "
            "validation label."
        ),
        "provenance": {
            "kind": "observed" if any_available else "unavailable",
            "source": "; ".join(sorted({e["source_label"] for e in entries
                                        if e.get("source_label")})),
            "quality": "ok" if any_available else "missing",
            "limitations": None if any_available else
                "NASA FIRMS API key not configured (set FIRMS_MAP_KEY) or "
                "FIRMS unreachable.",
        },
    }
