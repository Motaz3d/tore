"""
Historical wildfire event intelligence (Stage 3).

Derives :class:`~src.climate.events.ClimateEvent` records from **real** data:

- NASA FIRMS satellite detections (VIIRS S-NPP, 375 m; key-gated, free) —
  spatio-temporal clustering into multi-day event records,
- ERA5 reanalysis via Open-Meteo archive — the weather conditions observed
  during each event (reanalysis, declared as such),
- Canadian FWI System — modelled fire-danger context per event day,
  structurally separated from observations.

Discipline (docs/EVIDENCE_ARCHITECTURE.md):

- Without ``FIRMS_MAP_KEY`` the layer reports ``key_required`` — nothing is
  synthesised.
- Cause of ignition is always ``UNKNOWN`` here: FIRMS detections say
  nothing about cause, and no other authoritative source is consulted.
- Containment/extinguishing information is not available from these
  datasets and is reported as UNKNOWN.
- Years are never hardcoded: the caller asks for any year within the
  dataset coverage (VIIRS 2012 → present), and the response carries the
  coverage note.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from ..dashboard import real_data
from ..dashboard.cache import cached
from ..dashboard.history import DANGER_VS_OCCURRENCE_NOTE
from .events import ClimateEvent, EventStore
from .ontology import ClaimStatus, HazardType, TemporalClass
from .evidence import EvidenceRecord

TTL_FIRE_EVENTS = 24 * 3600.0          # historical windows are static
MAX_EVENTS_ENRICHED = 8                 # ERA5 condition enrichment cap per request
DEFAULT_RADIUS_KM = 50.0
VIIRS_COVERAGE_START_YEAR = 2012        # VIIRS S-NPP archive start (declared)


def _km_to_deg(km: float) -> float:
    return km / 111.0  # declared approximation, used for the query bbox


def _cluster_detections(points: List[Dict], gap_days: int = 2) -> List[List[Dict]]:
    """Group detections into event clusters: runs of detection days where
    consecutive detection days are at most ``gap_days`` apart.

    Declared limitation: distinct simultaneous fires inside the query
    radius may merge into one cluster (noted in the response).
    """

    by_day: Dict[str, List[Dict]] = {}
    for p in points:
        day = p.get("date")
        if day:
            by_day.setdefault(day, []).append(p)
    days = sorted(by_day)
    if not days:
        return []

    clusters: List[List[Dict]] = []
    current: List[Dict] = []
    prev_day: Optional[date] = None
    for day in days:
        d = date.fromisoformat(day)
        if prev_day is not None and (d - prev_day).days > gap_days and current:
            clusters.append(current)
            current = []
        current.extend(by_day[day])
        prev_day = d
    if current:
        clusters.append(current)
    return clusters


def _fwi_for_window(lat: float, lon: float, start: str, end: str) -> Dict[str, Any]:
    """ERA5 conditions + FWI for one event window (with 21-day FWI spin-up).

    Returns {"observed_daily": [...], "modelled_fwi": [...]} or {"error": …}.
    """

    from ..prediction.fwi import compute_fwi_series

    spinup_start = (date.fromisoformat(start) - timedelta(days=21)).isoformat()
    archive = real_data.fetch_weather_archive(lat, lon, spinup_start, end)
    if "error" in archive or not archive.get("time"):
        return {"error": f"ERA5 archive unavailable: {archive.get('error', 'no data')}"}

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
        return {"error": "ERA5 series too short for FWI reconstruction."}

    fwi_days = compute_fwi_series(series_in)
    observed_daily, modelled_fwi = [], []
    for d, src in zip(fwi_days, series_in):
        if src["date"] < start:      # spin-up days feed the model but are not reported
            continue
        observed_daily.append({
            "date": src["date"],
            "temp_max_c": src["temp_c"],
            "rh_mean_pct": src["rh_pct"],
            "wind_max_kmh": src["wind_kmh"],
            "rain_mm": src["rain_mm"],
        })
        modelled_fwi.append({"date": d.date, "fwi": round(d.fwi, 1), "danger_class": d.danger_class})
    return {"observed_daily": observed_daily, "modelled_fwi": modelled_fwi}


def _build_event(
    cluster: List[Dict],
    lat: float,
    lon: float,
    enrich: bool,
) -> ClimateEvent:
    """Build one ClimateEvent from a detection cluster (+ optional ERA5/FWI)."""

    days = sorted({p["date"] for p in cluster})
    start, end = days[0], days[-1]
    frps = [float(p.get("frp_mw") or 0.0) for p in cluster]
    total_frp = sum(frps) or 1.0
    c_lat = sum(p["lat"] * float(p.get("frp_mw") or 0.0) for p in cluster) / total_frp
    c_lon = sum(p["lon"] * float(p.get("frp_mw") or 0.0) for p in cluster) / total_frp
    peak = max(cluster, key=lambda p: float(p.get("frp_mw") or 0.0))

    evidence = [
        EvidenceRecord.satellite(
            "NASA FIRMS (VIIRS S-NPP, 375 m)",
            temporal=TemporalClass.HISTORICAL.value,
            dataset="FIRMS area CSV archive",
            provider_url="https://firms.modaps.eosdis.nasa.gov/",
            reference_period={"start": start, "end": end},
            method="spatio-temporal clustering of satellite fire detections "
                   "(declared: same-radius detections ≤2 days apart merge)",
            resolution="375 m",
            limitations="Thermal anomalies, not burned area; cloud/vegetation can "
                        "obscure detections; distinct simultaneous fires in the "
                        "radius may merge into one event record.",
            content_hash=None,
        ).to_dict(),
    ]

    conditions_observed: Dict[str, Any] = {}
    context_modelled: Dict[str, Any] = {}
    lessons: List[Dict[str, Any]] = []

    lessons.append({
        "text": f"{len(cluster)} satellite detection(s) over {len(days)} day(s); "
                f"peak fire radiative power {float(peak.get('frp_mw') or 0.0):.1f} MW "
                f"on {peak['date']}.",
        "basis": ClaimStatus.OBSERVED.value,
        "source": "NASA FIRMS (VIIRS S-NPP, 375 m)",
    })

    if enrich:
        env = _fwi_for_window(c_lat, c_lon, start, end)
        if "error" in env:
            conditions_observed = {"status": "unavailable", "reason": env["error"]}
        else:
            conditions_observed = {
                "daily": env["observed_daily"],
                "source": "ERA5 reanalysis via Open-Meteo archive",
                "limitations": "Reanalysis grid (~11 km), not a station measurement.",
            }
            context_modelled = {
                "fwi_daily": env["modelled_fwi"],
                "method": "Canadian FWI System (Van Wagner 1987) on ERA5 daily inputs, "
                          "21-day spin-up; no historical fuel moisture.",
            }
            evidence.append(EvidenceRecord.open_data(
                "ERA5 reanalysis via Open-Meteo archive",
                temporal=TemporalClass.HISTORICAL.value,
                reference_period={"start": start, "end": end},
                resolution="daily, ~11 km",
                limitations="Reanalysis, not a station measurement.",
            ).to_dict())
            evidence.append(EvidenceRecord.modelled(
                "Talaix / Canadian FWI System",
                method="FWI on ERA5 daily inputs (Van Wagner 1987)",
            ).to_dict())

            fwis = [d["fwi"] for d in env["modelled_fwi"]]
            if fwis:
                peak_fwi = max(env["modelled_fwi"], key=lambda d: d["fwi"])
                high_days = sum(1 for f in fwis if f >= 30.0)  # EFFIS "Very high" boundary
                lessons.append({
                    "text": f"Fire danger peaked at FWI {peak_fwi['fwi']} "
                            f"({peak_fwi['danger_class']}) on {peak_fwi['date']}; "
                            f"FWI was ≥30 (EFFIS 'Very high') on {high_days} of "
                            f"{len(fwis)} event day(s).",
                    "basis": ClaimStatus.MODELLED.value,
                    "source": "Canadian FWI System on ERA5 inputs",
                })
            daily = env["observed_daily"]
            if daily:
                wet = sum(d["rain_mm"] for d in daily)
                windy = max(daily, key=lambda d: d["wind_max_kmh"])
                lessons.append({
                    "text": f"Weather during the event: {wet:.1f} mm total rain; "
                            f"max wind {windy['wind_max_kmh']:.0f} km/h on {windy['date']}; "
                            f"max temperature {max(d['temp_max_c'] for d in daily):.1f} °C.",
                    "basis": ClaimStatus.OBSERVED.value,
                    "source": "ERA5 reanalysis via Open-Meteo archive",
                })
    else:
        conditions_observed = {
            "status": "not_enriched",
            "reason": f"ERA5 condition enrichment is limited to the "
                      f"{MAX_EVENTS_ENRICHED} largest events per request.",
        }

    lessons.append({
        "text": "Containment, suppression actions and extinguishing information: "
                "no record in the datasets used.",
        "basis": ClaimStatus.UNKNOWN.value,
        "source": None,
    })

    return ClimateEvent(
        hazard=HazardType.WILDFIRE.value,
        lat=round(c_lat, 5),
        lon=round(c_lon, 5),
        start_date=start,
        end_date=end,
        name=f"Fire event near {lat:.3f}, {lon:.3f} ({start})",
        classification=ClaimStatus.OBSERVED.value,
        severity={
            "detections": len(cluster),
            "detection_days": len(days),
            "max_frp_mw": round(max(frps), 1),
            "mean_frp_mw": round(total_frp / len(cluster), 1),
            "sensor": "VIIRS S-NPP",
            "resolution": "375 m",
        },
        conditions_observed=conditions_observed,
        context_modelled=context_modelled,
        cause={
            "status": ClaimStatus.UNKNOWN.value,
            "value": None,
            "source": None,
            "note": "Satellite detections say nothing about ignition cause; no "
                    "authoritative cause documentation in the datasets used.",
        },
        response=[],
        impacts=[],
        lessons=lessons,
        uncertainty="Event extent is inferred from detection clustering, not a "
                    "mapped perimeter; detection count is not burned area.",
        evidence=evidence,
    )


@cached("fire_events", TTL_FIRE_EVENTS)
def derive_fire_events(
    lat: float,
    lon: float,
    radius_km: float = DEFAULT_RADIUS_KM,
    year: Optional[int] = None,
    gap_days: int = 2,
) -> Dict[str, Any]:
    """Historical wildfire events near a point for a given year (real data).

    Cached 24 h — historical detection archives are static.
    """

    lat, lon = float(lat), float(lon)
    radius_km = max(5.0, min(float(radius_km), 200.0))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {"hazard": HazardType.WILDFIRE.value, "status": "error",
                "reason": "Coordinates out of range", "events": []}

    today = date.today()
    year = int(year) if year else today.year
    coverage_note = (
        "Observed events derive from NASA FIRMS VIIRS S-NPP (archive from "
        f"{VIIRS_COVERAGE_START_YEAR}); ERA5 fire-weather history reaches back "
        "to 1940 and is available via /api/history."
    )
    if year < VIIRS_COVERAGE_START_YEAR or year > today.year:
        return {
            "hazard": HazardType.WILDFIRE.value,
            "status": "unavailable",
            "reason": f"Year {year} is outside the observed-event coverage "
                      f"({VIIRS_COVERAGE_START_YEAR}–{today.year}).",
            "coverage_note": coverage_note,
            "events": [],
        }

    if not os.environ.get("FIRMS_MAP_KEY"):
        return {
            "hazard": HazardType.WILDFIRE.value,
            "status": "key_required",
            "reason": "NASA FIRMS API key not configured (set FIRMS_MAP_KEY).",
            "signup": "https://firms.modaps.eosdis.nasa.gov/api/area/",
            "fallback": "ERA5 fire-weather history (1940→present) is available "
                        "without a key via /api/history.",
            "events": [],
        }

    start = date(year, 1, 1)
    end = min(date(year, 12, 31), today - timedelta(days=1))
    half = _km_to_deg(radius_km)
    bbox = (lon - half, lat - half, lon + half, lat + half)

    from ..prediction.training import firms_fire_points_in_range  # lazy

    try:
        points = firms_fire_points_in_range(bbox, start.isoformat(), end.isoformat())
    except Exception as exc:
        return {
            "hazard": HazardType.WILDFIRE.value,
            "status": "unavailable",
            "reason": f"FIRMS retrieval failed: {exc}",
            "events": [],
        }

    clusters = _cluster_detections(points, gap_days=gap_days)
    # Enrich only the largest events with ERA5 conditions/FWI (bounded calls).
    clusters.sort(key=len, reverse=True)
    store = EventStore()
    events = []
    for rank, cluster in enumerate(clusters):
        event = _build_event(cluster, lat, lon, enrich=rank < MAX_EVENTS_ENRICHED)
        store.upsert_event(event)
        events.append(event.to_dict())

    return {
        "hazard": HazardType.WILDFIRE.value,
        "status": "ok",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "query": {
            "lat": lat, "lon": lon, "radius_km": radius_km, "year": year,
            "bbox": [round(v, 4) for v in bbox],
            "window": {"start": start.isoformat(), "end": end.isoformat()},
        },
        "event_count": len(events),
        "detection_count": len(points),
        "events": events,
        "coverage_note": coverage_note,
        "clustering_note": "Events are spatio-temporal clusters of detections "
                           "(≤2-day gaps); distinct simultaneous fires inside the "
                           "radius may merge into one record. Detection count is "
                           "not burned area.",
        "labels_note": DANGER_VS_OCCURRENCE_NOTE,
        "provenance": {
            "detections": {
                "kind": "observed",
                "claim_status": "OBSERVED",
                "source": "NASA FIRMS (VIIRS S-NPP, 375 m)",
                "acquired": f"{start.isoformat()}..{end.isoformat()}",
                "resolution": "375 m",
                "limitations": "Thermal anomalies; cloud/vegetation can obscure detections.",
            },
            "conditions": {
                "kind": "observed",
                "claim_status": "OBSERVED",
                "source": "ERA5 reanalysis via Open-Meteo archive",
                "resolution": "daily, ~11 km",
                "limitations": "Reanalysis, not a station measurement.",
            },
            "fire_danger_context": {
                "kind": "modelled",
                "claim_status": "MODELLED",
                "source": "Canadian FWI System (Van Wagner 1987) on ERA5 inputs",
                "limitations": "No historical fuel moisture; daily aggregates "
                               "approximate noon-standard inputs.",
            },
        },
    }
