"""
"Lessons from the Past" — historical intelligence for a location.

Reconstructs recent fire-danger history from REAL data:

    ERA5 reanalysis (Open-Meteo archive, real)
        -> Canadian FWI per day (real computation)
        -> Talaix risk per day (FWI-anchored, static terrain)
        -> high-risk periods (consecutive days >= High threshold)
        -> observed fire events per period (NASA FIRMS, when configured)
        -> what Talaix would have recommended (rules engine, labelled)

Labels used throughout:
    OBSERVED    — a real measurement / detection (FIRMS, ERA5 values)
    MODELLED    — computed by Talaix / the FWI system from real inputs
    RECOMMENDED — generated advice (never an observed intervention)
    UNKNOWN     — no real record exists (e.g. actual interventions taken)

Nothing is invented: without a FIRMS key the fire-observation layer is
reported as unavailable, and actual past interventions are always UNKNOWN
(Talaix has no record of them).
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from . import real_data
from .cache import cached
from .real_analysis import TalaixRealAnalyser
from .recommendations import build_recommendations

TTL_HISTORY = 6 * 3600.0
HIGH_RISK_THRESHOLD = 65.0

DANGER_VS_OCCURRENCE_NOTE = (
    "Fire danger is not fire occurrence: high danger means conditions favour "
    "fire spread IF an ignition happens — it does not guarantee ignition, "
    "and fires can occur on lower-danger days."
)


def _fwi_series_from_archive(archive: Dict) -> List[Dict]:
    """Run the FWI System over a real ERA5 archive series."""
    from ..prediction.fwi import compute_fwi_series

    times = archive.get("time") or []
    tmax = archive.get("temperature_2m_max") or []
    rh = archive.get("relative_humidity_2m_mean") or []
    wind = archive.get("wind_speed_10m_max") or []
    rain = archive.get("precipitation_sum") or []

    series_in = []
    for i, t in enumerate(times):
        if i >= len(tmax) or i >= len(rh) or i >= len(wind):
            break
        if tmax[i] is None or rh[i] is None or wind[i] is None:
            continue
        series_in.append({
            "date": t,
            "temp_c": float(tmax[i]),
            "rh_pct": float(rh[i]),
            "wind_kmh": float(wind[i]),
            "rain_mm": float(rain[i] or 0.0) if i < len(rain) and rain[i] is not None else 0.0,
        })
    if len(series_in) < 5:
        return []
    fwi_days = compute_fwi_series(series_in)
    out = []
    for d, src in zip(fwi_days, series_in):
        out.append({
            "date": d.date,
            "fwi": round(d.fwi, 1),
            "danger_class": d.danger_class,
            "temp_max_c": src["temp_c"],
            "wind_kmh": src["wind_kmh"],
            "rain_mm": src["rain_mm"],
        })
    return out


def _high_risk_periods(series: List[Dict], slope: float,
                       threshold: float = HIGH_RISK_THRESHOLD) -> List[Dict]:
    """Find consecutive-day periods with risk >= threshold (real scores)."""
    analyser = TalaixRealAnalyser()
    scored = []
    for d in series:
        risk = analyser._risk_score(fwi=d["fwi"], slope=slope, fmc=None, wind_kmh=0.0)
        scored.append({**d, "risk": risk})

    periods: List[Dict] = []
    current: List[Dict] = []
    for d in scored:
        if d["risk"] is not None and d["risk"] >= threshold:
            current.append(d)
        else:
            if current:
                periods.append(current)
                current = []
    if current:
        periods.append(current)

    out = []
    for p in periods:
        risks = [d["risk"] for d in p]
        fwis = [d["fwi"] for d in p]
        winds = [d["wind_kmh"] for d in p]
        peak = max(p, key=lambda d: d["risk"])
        out.append({
            "start": p[0]["date"],
            "end": p[-1]["date"],
            "days": len(p),
            "max_risk": round(max(risks), 1),
            "peak_date": peak["date"],
            "max_fwi": round(max(fwis), 1),
            "mean_wind_kmh": round(sum(winds) / len(winds), 1),
            "total_rain_mm": round(sum(d["rain_mm"] for d in p), 1),
        })
    return out


def _observed_fires(lat: float, lon: float, start: str, end: str) -> Dict:
    """Real FIRMS detections in a small bbox for the period, or unavailable."""
    if not os.environ.get("FIRMS_MAP_KEY"):
        return {
            "available": False,
            "reason": "NASA FIRMS API key not configured (set FIRMS_MAP_KEY)",
            "signup": "https://firms.modaps.eosdis.nasa.gov/api/area/",
        }
    from ..prediction.training import firms_fire_points_in_range

    half = 0.25  # ~25 km box around the point
    bbox = (lon - half, lat - half, lon + half, lat + half)
    try:
        points = firms_fire_points_in_range(bbox, start, end)
    except Exception as exc:
        return {"available": False, "reason": f"FIRMS retrieval failed: {exc}"}
    return {"available": True, "points": points, "bbox": list(bbox),
            "source": "NASA FIRMS VIIRS S-NPP (375 m)"}


def _fires_in_period(fires: Dict, start: str, end: str) -> Optional[int]:
    if not fires.get("available"):
        return None
    return sum(1 for p in fires.get("points") or [] if start <= p.get("date", "") <= end)


def _lesson_for_period(period: Dict, fires: Dict) -> Dict:
    """One structured lesson with strict OBSERVED/MODELLED/RECOMMENDED/UNKNOWN labels."""
    observed_count = _fires_in_period(fires, period["start"], period["end"])
    if observed_count is None:
        outcome, outcome_kind = "unknown (fire-observation layer unavailable)", "UNKNOWN"
    elif observed_count > 0:
        outcome = f"{observed_count} fire detection(s) during the period (NASA FIRMS)"
        outcome_kind = "OBSERVED"
    else:
        outcome = "no fire detected during the period (NASA FIRMS)"
        outcome_kind = "OBSERVED"

    # What Talaix would have recommended under these modelled conditions.
    pseudo_analysis = {
        "fire_danger": {"available": True, "fwi": period["max_fwi"], "class": "High"},
        "analysis": {"fuel_moisture_baseline_pct": None,
                     "risk": {"baseline": period["max_risk"]}},
        "weather": {"wind_speed_kmh": period["mean_wind_kmh"]},
        "terrain": {"slope_degrees": None},
        "landcover": {},
        "active_fires": {"available": False},
        "fire_danger_trend": {"trend": "unknown"},
    }
    would_recommend = [
        {"what": r["what"], "priority": r["priority"], "why": r["why"], "label": "RECOMMENDED"}
        for r in build_recommendations(pseudo_analysis)[:3]
    ]

    return {
        "period": {"start": period["start"], "end": period["end"], "days": period["days"]},
        "conditions": {
            "max_fwi": period["max_fwi"],
            "mean_wind_kmh": period["mean_wind_kmh"],
            "total_rain_mm": period["total_rain_mm"],
            "label": "MODELLED",
            "source": "ERA5 reanalysis via Open-Meteo archive + Canadian FWI System",
        },
        "hydrashield_score": {
            "value": period["max_risk"],
            "peak_date": period["peak_date"],
            "label": "MODELLED",
            "note": "FWI-anchored score with static terrain; no fuel-moisture "
                    "adjustment (no historical FMC).",
        },
        "observed_fire": {
            "status": outcome,
            "label": outcome_kind,
            "source": fires.get("source") if fires.get("available") else None,
        },
        "would_recommend": would_recommend,
        "interventions_recorded": {
            "status": "unknown",
            "label": "UNKNOWN",
            "note": "Talaix has no record of interventions actually taken "
                    "during this period; none are claimed.",
        },
    }


@cached("history", TTL_HISTORY)
def compute_history(lat: float, lon: float, name: str, days: int = 90) -> Dict:
    """
    Build the "Lessons from the Past" block for a location from real data.

    Cached for 6 h; the ERA5 archive itself is stable (reanalysis).
    """
    days = max(14, min(int(days), 180))
    lat, lon = float(lat), float(lon)
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {"error": "Coordinates out of range"}

    terrain = real_data.fetch_terrain(lat, lon)
    slope = terrain.get("slope_degrees") if "error" not in terrain else 0.0

    end_d = date.today()
    # ERA5 archive lags real time by several days; the API simply returns
    # what exists, so the effective window is taken from the response.
    start_d = end_d - timedelta(days=days)
    archive = real_data.fetch_weather_archive(lat, lon, start_d.isoformat(), end_d.isoformat())
    if "error" in archive or not archive.get("time"):
        return {"error": f"Historical reanalysis unavailable: {archive.get('error', 'no data')}"}

    series = _fwi_series_from_archive(archive)
    if not series:
        return {"error": "Reanalysis series too short to reconstruct fire-danger history."}

    eff_start, eff_end = series[0]["date"], series[-1]["date"]
    periods = _high_risk_periods(series, slope)
    fires = _observed_fires(lat, lon, eff_start, eff_end)

    lessons = [_lesson_for_period(p, fires) for p in periods[-5:]]

    recent = series[-14:]
    return {
        "location": {"name": name or f"{lat:.4f}, {lon:.4f}", "latitude": lat, "longitude": lon},
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window": {"start": eff_start, "end": eff_end, "days": len(series)},
        "high_risk_periods": periods,
        "lessons": lessons,
        "recent_fire_danger": recent,
        "fire_observations": {
            "available": fires.get("available", False),
            "reason": fires.get("reason"),
            "count": len(fires.get("points") or []) if fires.get("available") else None,
            "source": fires.get("source"),
        },
        "labels_note": DANGER_VS_OCCURRENCE_NOTE,
        "provenance": {
            "history": {
                "kind": "modeled",
                "source": "ERA5 reanalysis via Open-Meteo archive + Canadian FWI System",
                "acquired": f"{eff_start}..{eff_end}",
                "resolution": "daily, ~11 km (ERA5)",
                "retrieved_at": datetime.utcnow().isoformat() + "Z",
                "quality": "ok",
                "limitations": "Daily aggregates approximate noon-standard FWI "
                               "inputs; no historical fuel moisture.",
            },
            "fire_observations": {
                "kind": "observed" if fires.get("available") else "unavailable",
                "source": "NASA FIRMS VIIRS S-NPP (375 m)",
                "acquired": f"{eff_start}..{eff_end}" if fires.get("available") else None,
                "resolution": "375 m",
                "retrieved_at": datetime.utcnow().isoformat() + "Z",
                "quality": "ok" if fires.get("available") else "missing",
                "limitations": None if fires.get("available") else fires.get("reason"),
            },
        },
    }
