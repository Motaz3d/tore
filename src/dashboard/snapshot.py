"""
Public wildfire-risk snapshot ("Where is wildfire risk highest right now?").

Computes a small ranking of the highest-risk areas among a *configured set
of monitored areas* (``config/monitored_areas.json``), using the same real
analysis engine and the same SQLite cache that power ``/api/analyze``:

    monitored areas (config, coordinates — no geocoding at snapshot time)
        -> TalaixRealAnalyser (real Sentinel-2 / Open-Meteo / DEM /
           WorldCover / FIRMS data, per-analysis 15-min cache)
        -> top-k ranking by the real composite risk score
        -> aggregate snapshot (30-min cache)

Honesty rules inherited from the rest of the platform:
    - Areas whose real risk score cannot be computed are dropped, never
      filled in.
    - When no valid real snapshot can be produced the endpoint reports
      ``status: unavailable`` instead of inventing values.
    - The scope of the ranking (configured monitored areas only) is part of
      the response so the UI can never imply global coverage.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .cache import cached, default_cache, TTL_ANALYSIS, TTL_SNAPSHOT
from .explain import compact_factors
from .real_analysis import TalaixRealAnalyser

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "monitored_areas.json"
)

# How long an "unavailable" answer is pinned before a rebuild is attempted.
TTL_SNAPSHOT_FAILED = 60.0

_CACHE_KEY = "risk_snapshot:current"
_build_lock = threading.Lock()


@cached("analysis", TTL_ANALYSIS)
def cached_analysis(lat: float, lon: float, name: str) -> Dict:
    """Cached full analysis for a point (15 min TTL), shared with /api/analyze."""
    analyser = TalaixRealAnalyser()
    return analyser.analyse_point(lat, lon, name=name or None)


# --------------------------------------------------------------------------
# Monitored-area configuration
# --------------------------------------------------------------------------

class MonitoredAreasConfig:
    """Validated monitored-area configuration."""

    def __init__(self, scope: str, top_k: int, areas: List[Dict]) -> None:
        self.scope = scope
        self.top_k = top_k
        self.areas = areas


def load_monitored_areas(path: Optional[str] = None) -> MonitoredAreasConfig:
    """
    Load and validate the monitored-area configuration.

    Coordinates come from the config file, so building the snapshot never
    touches the geocoder (Nominatim rate limits are respected by design).
    Areas with invalid coordinates are skipped. Raises ValueError when the
    file is unreadable or contains no usable area.
    """
    cfg_path = path or os.environ.get("HYDRASHIELD_MONITORED_AREAS") or _DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    scope = str(raw.get("scope") or "Configured monitored areas")[:300]
    try:
        top_k = int(raw.get("top_k", 5))
    except (TypeError, ValueError):
        top_k = 5
    top_k = max(1, min(top_k, 10))

    areas: List[Dict] = []
    for item in raw.get("areas") or []:
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
            name = str(item["name"]).strip()[:200]
        except (KeyError, TypeError, ValueError):
            continue
        if not name or not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        areas.append({"name": name, "lat": lat, "lon": lon})

    if not areas:
        raise ValueError(f"No valid monitored areas in {cfg_path}")
    if len(areas) > 50:
        raise ValueError("Too many monitored areas (max 50)")
    return MonitoredAreasConfig(scope=scope, top_k=top_k, areas=areas)


# --------------------------------------------------------------------------
# Source registry (display names + official URLs for the UI)
# --------------------------------------------------------------------------

_SOURCE_REGISTRY = {
    "fire_danger": (
        "Canadian FWI System",
        "https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi",
    ),
    "weather": ("Open-Meteo (weather model)", "https://open-meteo.com/"),
    "satellite": (
        "Copernicus Sentinel-2",
        "https://sentiwiki.copernicus.eu/web/s2-mission",
    ),
    "terrain": ("EU-DEM / SRTM via OpenTopoData", "https://www.opentopodata.org/"),
    "landcover": ("ESA WorldCover", "https://esa-worldcover.org/en"),
    "active_fires": ("NASA FIRMS", "https://firms.modaps.eosdis.nasa.gov/"),
}

# Components whose real data feeds the risk score / ranking context.
_RANKING_COMPONENTS = (
    "fire_danger", "weather", "satellite", "terrain", "landcover", "active_fires",
)


def _contributed(kind: Optional[str], quality: Optional[str]) -> bool:
    return kind not in (None, "unavailable") and quality != "missing"


# --------------------------------------------------------------------------
# Snapshot computation
# --------------------------------------------------------------------------

def _entry_from_analysis(area: Dict, result: Dict) -> Optional[Dict]:
    """
    Reduce a full real analysis to a snapshot entry.

    Returns None when the analysis failed or produced no real risk score —
    the area is then simply absent from the ranking.
    """
    if not isinstance(result, dict) or "error" in result:
        return None
    analysis = result.get("analysis") or {}
    risk = analysis.get("risk") or {}
    score = risk.get("baseline")
    if score is None:
        return None

    fire_danger = result.get("fire_danger") or {}
    trend = result.get("fire_danger_trend") or {}
    fires = result.get("active_fires") or {}
    satellite = result.get("satellite") or {}

    entry = {
        "name": area["name"],
        "latitude": round(float(area["lat"]), 4),
        "longitude": round(float(area["lon"]), 4),
        "risk": score,
        "risk_class": risk.get("class"),
        "fwi": fire_danger.get("fwi") if fire_danger.get("available") else None,
        "fwi_class": fire_danger.get("class") if fire_danger.get("available") else None,
        "fwi_date": fire_danger.get("date") if fire_danger.get("available") else None,
        "trend": trend.get("trend"),
        "active_fires": (
            {"count": fires.get("count", len(fires.get("fires") or [])), "days": fires.get("days")}
            if fires.get("available")
            else None
        ),
        "satellite_date": (
            (satellite.get("observation_date") or "")[:10] or None
            if "error" not in satellite
            else None
        ),
        # "Why this score?" — compact factor levels from the real inputs.
        "factors": compact_factors(result.get("risk_explanation") or {}),
        "score_disclaimer": (result.get("risk_explanation") or {}).get("disclaimer"),
        # Top evidence-linked preventive recommendation, if any rule fired.
        "top_recommendation": (
            {
                "what": (result.get("recommendations") or [{}])[0].get("what"),
                "priority": (result.get("recommendations") or [{}])[0].get("priority"),
            }
            if result.get("recommendations")
            else None
        ),
        # Full component provenance of the underlying real analysis.
        "provenance": result.get("provenance") or {},
    }
    return entry


def _collect_sources(entries: List[Dict]) -> List[Dict]:
    """Union of the sources that actually contributed to the ranked entries."""
    seen = []
    for entry in entries:
        prov = entry.get("provenance") or {}
        for component in _RANKING_COMPONENTS:
            p = prov.get(component) or {}
            if component in _SOURCE_REGISTRY and _contributed(p.get("kind"), p.get("quality")):
                if component not in seen:
                    seen.append(component)
    return [
        {"key": key, "name": _SOURCE_REGISTRY[key][0], "url": _SOURCE_REGISTRY[key][1]}
        for key in seen
    ]


def compute_snapshot(
    config_path: Optional[str] = None,
    analyse_fn: Optional[Callable[[float, float, str], Dict]] = None,
) -> Dict:
    """
    Build the risk snapshot from real analyses of the monitored areas.

    Every area is analysed with the same cached real pipeline used by
    ``/api/analyze`` (no geocoding — coordinates come from the config).
    Areas are analysed concurrently (4 workers) so a cold rebuild stays
    well inside the gunicorn timeout; warm rebuilds are fast because each
    underlying analysis is itself cached.
    """
    analyse = analyse_fn or cached_analysis
    generated_at = datetime.utcnow().isoformat() + "Z"

    try:
        cfg = load_monitored_areas(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "unavailable",
            "message": f"Monitored-area configuration problem: {exc}",
            "generated_at": generated_at,
            "entries": [],
        }

    def _run(area: Dict) -> Optional[Dict]:
        try:
            result = analyse(round(area["lat"], 4), round(area["lon"], 4), area["name"])
        except Exception:
            return None
        return _entry_from_analysis(area, result)

    entries: List[Dict] = []
    # 2 workers: each real analysis loads heavy context (satellite scenes,
    # terrain, landcover); four concurrent analyses peaked >3.5 GB and the
    # kernel OOM-killed the builder on the 4 GB production host.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for entry in pool.map(_run, cfg.areas):
            if entry is not None:
                entries.append(entry)

    if not entries:
        return {
            "status": "unavailable",
            "message": "Risk snapshot temporarily unavailable — no monitored area "
                       "could be analysed from real data right now.",
            "generated_at": generated_at,
            "scope": cfg.scope,
            "entries": [],
        }

    entries.sort(key=lambda e: e["risk"], reverse=True)
    top = entries[: cfg.top_k]
    for rank, entry in enumerate(top, start=1):
        entry["rank"] = rank

    return {
        "status": "ok",
        "scope": cfg.scope,
        "generated_at": generated_at,
        "valid_for_seconds": int(TTL_SNAPSHOT),
        "areas_considered": len(cfg.areas),
        "areas_with_data": len(entries),
        "entries": top,
        "sources": _collect_sources(top),
        "model": {
            "risk_score": "Talaix composite screening score (FWI-anchored, 0-100)",
            "note": "Screening-level score from real Earth Observation and weather "
                    "data; not a validated local fire-danger rating.",
            "disclaimer": "Composite wildfire-risk indicator (0-100) — not a "
                          "probability of fire.",
        },
        "data_policy": "Real data only; areas without a computable real risk score "
                       "are omitted, never estimated.",
    }


def get_snapshot(
    config_path: Optional[str] = None,
    analyse_fn: Optional[Callable[[float, float, str], Dict]] = None,
    build: bool = False,
) -> Dict:
    """
    Return the cached snapshot.

    Cold rebuilds are NOT run on the request path: ten real analyses can
    peak several GB and OOM-kill the gunicorn worker on the 4 GB host
    (observed in production — SIGKILL, 502). The snapshot is rebuilt
    periodically by the watch_checker (``scripts/build_risk_snapshot.py``
    passes ``build=True``); a cache miss on the request path reports an
    honest warming state (503) and the homepage retries.

    A module-level lock ensures only one thread rebuilds at a time; failed
    snapshots are pinned for only 60 s so a transient upstream outage does
    not disable the bar for half an hour.
    """
    cache = default_cache()
    hit = cache.get(_CACHE_KEY)
    if hit is not None:
        return hit
    if not build:
        return {
            "status": "unavailable",
            "message": "Risk snapshot is warming — it is rebuilt periodically "
                       "by the platform (every 30 minutes).",
            "entries": [],
        }

    with _build_lock:
        # Re-check: another thread may have built it while we waited.
        hit = cache.get(_CACHE_KEY)
        if hit is not None:
            return hit
        snapshot = compute_snapshot(config_path=config_path, analyse_fn=analyse_fn)
        ttl = TTL_SNAPSHOT if snapshot.get("status") == "ok" else TTL_SNAPSHOT_FAILED
        cache.set(_CACHE_KEY, snapshot, ttl)
        return snapshot
