"""
Smoke Intelligence — atmospheric transport guidance, NOT certainty.

Talaix answers two strictly separated questions:

    OBSERVED mode ....... "An observed fire is burning (NASA FIRMS
                           detection). Under current/forecast winds, where
                           is the smoke likely to move FROM NOW?"
    SCENARIO mode ....... "IF a fire were to occur near this location under
                           current atmospheric conditions, where could the
                           smoke move?" (clearly labelled SCENARIO/MODELLED)

These two modes are never mixed: a scenario is never presented as an
observation, and an observed-fire estimate always names its source
detection (sensor, time, location).

Method (declared screening model — not a dispersion model):
    - Transport wind: hourly 850 hPa wind (~1.5 km, a standard smoke-
      transport level) from the real Open-Meteo forecast; 10 m surface wind
      fallback with an explicit weakness note. (real_data.fetch_wind_profile)
    - Trajectory: hourly Lagrangian steps from the source position.
    - Corridor: a widening envelope around the trajectory
      (half-width = CORRIDOR_W0_KM + CORRIDOR_GROWTH_KM_H * hours), NOT a
      deterministic smoke path. Plume rise, chemistry, deposition, vertical
      wind shear and terrain channelling are NOT modelled — declared.
    - Uncertainty: from the circular variability of the hourly transport
      directions and the mean transport speed. Confidence is only ever
      "low" or "moderate" — a screening trajectory from ~11 km NWP output
      never justifies "high".

Population/facility overlay uses the real WorldPop grid (reference year
declared) and real OpenStreetMap features inside the corridor polygon.

Safety messaging is general public-health guidance only (no medical
advice), always subordinate to official emergency instructions.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from . import population as population_module
from . import real_data
from .cache import cached
from .exposure import _post_overpass

TTL_SMOKE = 1800.0  # 30 min — tracks the wind-profile freshness

#: Declared corridor geometry constants (screening envelope, not dispersion).
CORRIDOR_W0_KM = 1.5        # source + near-field uncertainty half-width
CORRIDOR_GROWTH_KM_H = 0.75  # cross-wind envelope growth per transport hour

#: Uncertainty classification thresholds (declared).
_VARIABILITY_MODERATE = 0.2   # circular variability (1 - mean resultant length)
_VARIABILITY_LOW_CONF = 0.4
_MIN_SPEED_FOR_GUIDANCE = 5.0  # km/h; below this, drift is poorly defined

_MAX_OBSERVED_FIRES = 3  # upstream-call bound for per-detection wind profiles
_DETECTION_OLD_HOURS = 48.0

TRANSPORT_DISCLAIMER = (
    "Atmospheric transport guidance, not certainty. Screening trajectory from "
    "numerical-weather-model wind fields (~11 km grid): plume rise, chemistry, "
    "deposition, vertical wind shear and terrain channelling are not modelled. "
    "The corridor is an envelope of likely transport, not a predicted smoke path."
)

SAFETY_GUIDANCE = {
    "kind": "general public-health guidance (WHO / national fire-service public advice)",
    "not_medical_advice": True,
    "points": [
        "Follow instructions from official civil-protection and fire services first — they override any model output here.",
        "During smoke episodes, general public-health guidance is to reduce prolonged outdoor exertion and keep windows closed where smoke is present.",
        "People with respiratory or cardiovascular conditions, children and the elderly are generally advised to take extra care during smoke episodes; individual medical advice must come from a health professional.",
        "Never treat an area as safe merely because a model predicts low smoke exposure there — wind shifts are common and models are uncertain.",
    ],
    "distinction_note": (
        "This section is environmental exposure information from modelled "
        "atmospheric transport. It is neither an observation of smoke at ground "
        "level nor an official emergency instruction."
    ),
}

_COMPASS8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _compass(heading_deg: float) -> str:
    return _COMPASS8[int((heading_deg % 360.0 + 22.5) // 45.0) % 8]


def _destination(lat: float, lon: float, bearing_deg: float, dist_km: float) -> Tuple[float, float]:
    """Spherical destination point from lat/lon along a bearing."""
    r = 6371.0088
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(dist_km / r)
        + math.cos(lat1) * math.sin(dist_km / r) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(dist_km / r) * math.cos(lat1),
        math.cos(dist_km / r) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _circular_stats(headings_deg: List[float]) -> Tuple[float, float]:
    """Return (mean heading, variability) for a list of directions in degrees."""
    if not headings_deg:
        return 0.0, 1.0
    sx = sum(math.cos(math.radians(h)) for h in headings_deg)
    sy = sum(math.sin(math.radians(h)) for h in headings_deg)
    n = float(len(headings_deg))
    r_len = math.hypot(sx, sy) / n  # 1 = perfectly steady, 0 = uniform spread
    mean = math.degrees(math.atan2(sy, sx)) % 360.0
    return mean, round(1.0 - r_len, 3)


def _corridor_polygon(points: List[Tuple[float, float]],
                      half_widths_km: List[float]) -> List[List[float]]:
    """
    Widening envelope around a trajectory polyline.

    Perpendicular offsets per point (left side forward, right side back with
    end caps). A screening envelope — sharp-turn self-intersections are
    possible and declared; it is never drawn as a deterministic path.
    """
    if len(points) < 2:
        return []
    left: List[Tuple[float, float]] = []
    right: List[Tuple[float, float]] = []
    n = len(points)
    for i, (lat, lon) in enumerate(points):
        if i < n - 1:
            lat2, lon2 = points[i + 1]
        else:
            lat2, lon2 = points[i]
            lat, lon = points[i - 1]
        # Local segment bearing (equirectangular approximation, fine at km scale).
        dy = (lat2 - lat) * 110.54
        dx = (lon2 - lon) * 111.32 * max(0.01, math.cos(math.radians(lat)))
        seg_bearing = math.degrees(math.atan2(dx, dy)) % 360.0
        w = half_widths_km[i]
        for bearing, out in ((seg_bearing + 90.0, left), (seg_bearing - 90.0, right)):
            out.append(_destination(points[i][0], points[i][1], bearing, w))
    polygon = left + list(reversed(right))
    if polygon:
        polygon.append(polygon[0])  # close the ring
    return [[round(p[0], 5), round(p[1], 5)] for p in polygon]


def compute_transport(steps: List[Dict], lat: float, lon: float) -> Dict:
    """
    Pure transport computation from an hourly wind-profile step list.

    ``steps`` entries need ``time``, ``transport_speed_kmh`` and
    ``transport_dir_deg`` (the direction the wind blows FROM, as meteorology
    reports it). Returns the hourly trajectory, the widening corridor
    polygon, dominant transport direction, path/displacement distances and
    the declared uncertainty classification. Deterministic — no randomness.
    """
    if not steps:
        return {"error": "No wind profile steps supplied"}

    points: List[Tuple[float, float]] = [(lat, lon)]
    trajectory: List[Dict] = [{
        "time": steps[0].get("time"),
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "hour": 0,
        "distance_from_origin_km": 0.0,
    }]
    headings: List[float] = []
    speeds: List[float] = []
    path_km = 0.0
    cur_lat, cur_lon = lat, lon

    for i, step in enumerate(steps):
        speed = float(step.get("transport_speed_kmh") or 0.0)
        dir_from = float(step.get("transport_dir_deg") or 0.0)
        heading = (dir_from + 180.0) % 360.0  # direction the air moves TO
        headings.append(heading)
        speeds.append(speed)
        move_km = speed * 1.0  # one hour per step
        cur_lat, cur_lon = _destination(cur_lat, cur_lon, heading, move_km)
        path_km += move_km
        points.append((cur_lat, cur_lon))
        trajectory.append({
            "time": step.get("time"),
            "lat": round(cur_lat, 5),
            "lon": round(cur_lon, 5),
            "hour": i + 1,
            "transport_speed_kmh": round(speed, 1),
            "transport_heading_deg": round(heading, 1),
            "distance_from_origin_km": round(path_km, 1),
        })

    half_widths = [CORRIDOR_W0_KM + CORRIDOR_GROWTH_KM_H * i for i in range(len(points))]
    polygon = _corridor_polygon(points, half_widths)

    mean_heading, variability = _circular_stats(headings)
    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    final = points[-1]
    displacement_km = _haversine_km(lat, lon, final[0], final[1])

    if variability > _VARIABILITY_LOW_CONF or mean_speed < _MIN_SPEED_FOR_GUIDANCE:
        confidence = "low"
        confidence_note = (
            "Wind direction varies strongly over the window"
            if variability > _VARIABILITY_LOW_CONF
            else "Transport winds are very light — smoke may drift, pool or recirculate"
        ) + "; treat the corridor as weak guidance."
    elif variability > _VARIABILITY_MODERATE:
        confidence = "moderate"
        confidence_note = "Some directional variability over the window; the corridor is widened accordingly."
    else:
        confidence = "moderate"
        confidence_note = (
            "Steady transport direction. 'Moderate' is the highest confidence a "
            "screening trajectory from ~11 km model winds ever receives here."
        )

    return {
        "trajectory": trajectory,
        "corridor_polygon": polygon,
        "dominant_transport_direction": _compass(mean_heading),
        "dominant_transport_heading_deg": round(mean_heading, 1),
        "mean_transport_speed_kmh": round(mean_speed, 1),
        "path_length_km": round(path_km, 1),
        "displacement_km": round(displacement_km, 1),
        "hours": len(steps),
        "direction_variability": variability,
        "corridor_model": {
            "type": "widening envelope (screening), not a deterministic path",
            "initial_half_width_km": CORRIDOR_W0_KM,
            "growth_km_per_hour": CORRIDOR_GROWTH_KM_H,
        },
        "confidence": confidence,
        "confidence_note": confidence_note,
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 6371.0088 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _now_floor_hour() -> datetime:
    now = datetime.utcnow()
    return now.replace(minute=0, second=0, microsecond=0)


def _steps_from_now(profile: Dict, hours: int) -> List[Dict]:
    """Slice the profile steps to the [now, now+hours) UTC window."""
    now = _now_floor_hour()
    out = []
    for s in profile.get("steps") or []:
        try:
            t = datetime.fromisoformat(str(s.get("time")))
        except ValueError:
            continue
        if t >= now:
            out.append(s)
        if len(out) >= hours:
            break
    return out


# ---------------------------------------------------------------------------
# Corridor overlays (real population + real OSM facilities)
# ---------------------------------------------------------------------------

def _corridor_population(polygon: List[List[float]]) -> Dict:
    """
    Estimated population inside the corridor polygon (real WorldPop grid).

    The country raster must already be available (one-time download handled
    by the population module). Honest note when it cannot be produced.
    """
    if not polygon:
        return {"available": False, "reason": "No corridor polygon"}
    clat = sum(p[0] for p in polygon) / len(polygon)
    clon = sum(p[1] for p in polygon) / len(polygon)
    cc = population_module.country_code_for(round(clat, 1), round(clon, 1))
    if "error" in cc:
        return {"available": False, "reason": cc["error"]}
    iso3 = population_module._ALPHA2_TO_ALPHA3.get(cc["country_code"])
    if iso3 is None:
        return {"available": False, "reason": f"No WorldPop mapping for '{cc['country_code']}'"}
    est = population_module.population_in_polygon(iso3, [(p[0], p[1]) for p in polygon])
    if "error" in est:
        return {"available": False, "reason": est["error"]}
    return {
        "available": True,
        "estimated_population_in_corridor": est["estimated_population"],
        "estimate_note": est["estimate_note"],
        "source": est["source"],
        "coverage_note": (
            "Estimated against the country raster of the corridor centroid; "
            "corridor portions crossing into a neighbouring country are clipped "
            "to that one raster (declared approximation)."
        ),
    }


def _simplify_polygon(polygon: List[List[float]], max_vertices: int = 24) -> List[List[float]]:
    """Evenly decimate a polygon ring for the Overpass poly filter."""
    ring = polygon[:-1] if len(polygon) > 1 and polygon[0] == polygon[-1] else polygon
    if len(ring) <= max_vertices:
        return ring
    step = len(ring) / float(max_vertices)
    return [ring[int(i * step)] for i in range(max_vertices)]


@cached("smoke_facilities", TTL_SMOKE)
def facilities_in_polygon(polygon: List[List[float]]) -> Dict:
    """
    Mapped critical facilities (hospitals, schools, fire stations) inside the
    corridor polygon via the Overpass ``poly`` filter (real OSM data).
    """
    if not polygon or len(polygon) < 4:
        return {"error": "Polygon too small"}
    ring = _simplify_polygon(polygon)
    poly_str = " ".join(f"{p[0]:.5f} {p[1]:.5f}" for p in ring)
    parts = []
    for key, val in (("amenity", "hospital"), ("amenity", "school"), ("amenity", "fire_station")):
        parts.append(f'node["{key}"="{val}"](poly:"{poly_str}");way["{key}"="{val}"](poly:"{poly_str}");')
    query = f"[out:json][timeout:25];({''.join(parts)});out center tags 60;"
    try:
        data = _post_overpass(query)
    except Exception as exc:
        return {"error": f"OpenStreetMap corridor facilities unavailable: {exc}"}

    label = {"hospital": "hospitals", "school": "schools", "fire_station": "fire_stations"}
    facilities: List[Dict] = []
    for el in data.get("elements") or []:
        tags = el.get("tags") or {}
        category = label.get(tags.get("amenity"))
        if category is None:
            continue
        center = el.get("center") or {}
        elat = el.get("lat", center.get("lat"))
        elon = el.get("lon", center.get("lon"))
        if elat is None or elon is None:
            continue
        facilities.append({
            "category": category,
            "lat": float(elat),
            "lon": float(elon),
            "name": tags.get("name"),
        })
    counts: Dict[str, int] = {}
    for f in facilities:
        counts[f["category"]] = counts.get(f["category"], 0) + 1
    return {
        "facilities": facilities,
        "counts": counts,
        "source": "OpenStreetMap (Overpass API, corridor polygon filter)",
        "note": "Mapped OSM features inside the modelled corridor; completeness varies by region.",
    }


def _corridor_overlays(polygon: List[List[float]]) -> Dict:
    """Population + facility overlays for one corridor (both honest on failure)."""
    out: Dict = {}
    pop = _corridor_population(polygon)
    out["population"] = pop
    fac = facilities_in_polygon(tuple(tuple(p) for p in polygon))
    if "error" in fac:
        out["facilities"] = {"available": False, "reason": fac["error"]}
    else:
        out["facilities"] = {
            "available": True,
            "counts": fac["counts"],
            "facilities": fac["facilities"],
            "source": fac["source"],
            "note": fac["note"],
        }
    return out


# ---------------------------------------------------------------------------
# Scenario mode (hypothetical fire — never presented as observed)
# ---------------------------------------------------------------------------

@cached("smoke_scenario", TTL_SMOKE)
def smoke_scenario(lat: float, lon: float, hours: int = 24) -> Dict:
    """
    SCENARIO smoke-transport estimate for a hypothetical fire at a location.

    "If a fire were to occur near this location under current atmospheric
    conditions..." — always labelled SCENARIO / MODELLED. No fire is claimed
    to exist.
    """
    if not real_data._valid_point(lat, lon):
        return {"error": "Coordinates out of range"}
    lat, lon = round(float(lat), 2), round(float(lon), 2)  # ~1 km, model grid is ~11 km
    hours = min(max(int(hours), 3), 48)

    profile = real_data.fetch_wind_profile(lat, lon, hours + 1)
    if "error" in profile:
        return {"error": profile["error"], "mode": "scenario"}
    steps = _steps_from_now(profile, hours)
    if len(steps) < 3:
        return {"error": "Insufficient forecast wind steps from now", "mode": "scenario"}

    transport = compute_transport(steps, lat, lon)
    if "error" in transport:
        return {"error": transport["error"], "mode": "scenario"}

    overlays = _corridor_overlays(transport["corridor_polygon"])
    return {
        "status": "ok",
        "mode": "scenario",
        "mode_label": "SCENARIO / MODELLED — no fire is observed at this location",
        "scenario": (
            "If a fire were to occur near this location under current atmospheric "
            "conditions, this is where the smoke could move."
        ),
        "location": {"latitude": lat, "longitude": lon},
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window": {
            "from": steps[0].get("time"),
            "to": steps[-1].get("time"),
            "hours": transport["hours"],
            "timezone": "UTC",
        },
        "transport": transport,
        "overlays": overlays,
        "disclaimer": TRANSPORT_DISCLAIMER,
        "safety": SAFETY_GUIDANCE,
        "provenance": {
            "kind": "modeled",
            "source": f"{profile['source']} (transport level {profile['transport_level']})",
            "resolution": "~11 km NWP grid; corridor is a screening envelope",
            "temporal": f"next {transport['hours']} h from {_now_floor_hour().isoformat()}Z",
            "quality": "ok",
            "limitations": profile.get("level_note"),
        },
    }


# ---------------------------------------------------------------------------
# Observed mode (NASA FIRMS detections — requires FIRMS_MAP_KEY)
# ---------------------------------------------------------------------------

def _detection_age_hours(acq_date: Optional[str], acq_time: Optional[str]) -> Optional[float]:
    """Age of a FIRMS detection in hours (UTC), None when unparseable."""
    if not acq_date:
        return None
    t = (acq_time or "0000").strip().zfill(4)
    try:
        dt = datetime.fromisoformat(f"{acq_date}T{t[:2]}:{t[2:4]}:00")
    except ValueError:
        return None
    return round((datetime.utcnow() - dt).total_seconds() / 3600.0, 1)


@cached("smoke_observed", TTL_SMOKE)
def smoke_observed(lat: float, lon: float, radius_km: float = 50.0,
                   days: int = 3, hours: int = 24) -> Dict:
    """
    OBSERVED-fire smoke-transport estimates near a location.

    Uses real NASA FIRMS detections (when ``FIRMS_MAP_KEY`` is configured).
    For up to ``_MAX_OBSERVED_FIRES`` nearest detections, transport is
    modelled FROM NOW under forecast winds anchored at the observed fire
    location — it is not a reconstruction of the historical plume (the
    detection age is reported). Honestly unavailable without a key or when
    no fires are detected: no fire is ever invented.
    """
    if not real_data._valid_point(lat, lon):
        return {"error": "Coordinates out of range"}
    lat, lon = float(lat), float(lon)
    radius_km = min(max(float(radius_km), 5.0), 200.0)
    days = min(max(int(days), 1), 10)
    hours = min(max(int(hours), 3), 48)

    from .fire_evidence import build_fire_evidence

    evidence = build_fire_evidence(lat, lon, radius_km=radius_km, days=days)
    if evidence.get("status") != "ok":
        return {
            "status": "unavailable",
            "mode": "observed",
            "reason": (evidence.get("provenance") or {}).get("limitations")
                      or "No observed-fire source available",
            "signup": "https://firms.modaps.eosdis.nasa.gov/api/area/",
            "note": "Without observed fire detections no observed-mode smoke "
                    "estimate is produced. Use the scenario mode for a "
                    "hypothetical fire (clearly labelled SCENARIO).",
            "provenance": evidence.get("provenance"),
        }

    detections: List[Dict] = []
    for entry in evidence.get("entries") or []:
        if entry.get("status") != "ok":
            continue
        for d in entry.get("detections") or []:
            if d.get("lat") is None or d.get("lon") is None:
                continue
            d = dict(d)
            d["distance_km"] = round(_haversine_km(lat, lon, d["lat"], d["lon"]), 1)
            d["evidence_source"] = entry.get("source_label")
            detections.append(d)
    detections.sort(key=lambda d: d["distance_km"])

    if not detections:
        return {
            "status": "ok",
            "mode": "observed",
            "fires": [],
            "count": 0,
            "note": f"No active-fire detections within {radius_km:.0f} km in the last "
                    f"{days} day(s) from the configured sensors — no smoke sources to model.",
            "provenance": evidence.get("provenance"),
        }

    fires: List[Dict] = []
    for det in detections[:_MAX_OBSERVED_FIRES]:
        # Per-detection wind profile at ~0.25 deg rounding (cache-friendly,
        # matched to the ~11 km model grid).
        plat, plon = round(det["lat"] * 4) / 4.0, round(det["lon"] * 4) / 4.0
        profile = real_data.fetch_wind_profile(plat, plon, hours + 1)
        age = _detection_age_hours(det.get("acq_date"), det.get("acq_time_utc"))
        fire_block: Dict = {
            "detection": {
                "lat": det["lat"],
                "lon": det["lon"],
                "sensor": det.get("sensor"),
                "acq_date": det.get("acq_date"),
                "acq_time_utc": det.get("acq_time_utc"),
                "age_hours": age,
                "frp_mw": det.get("frp_mw"),
                "confidence": det.get("confidence"),
                "distance_km": det["distance_km"],
                "source": det.get("evidence_source"),
                "note": "Observed satellite detection — an active-fire hotspot, "
                        "not a fire perimeter.",
            },
        }
        if age is not None and age > _DETECTION_OLD_HOURS:
            fire_block["detection"]["age_note"] = (
                f"Detection is {age:.0f} h old — the fire may no longer be active."
            )
        if "error" in profile:
            fire_block["transport"] = {"available": False, "reason": profile["error"]}
        else:
            steps = _steps_from_now(profile, hours)
            transport = compute_transport(steps, det["lat"], det["lon"]) if len(steps) >= 3 else {"error": "insufficient wind steps"}
            if "error" in transport:
                fire_block["transport"] = {"available": False, "reason": transport["error"]}
            else:
                transport["available"] = True
                transport["anchored_at"] = "observed fire location; transport modelled FROM NOW under forecast winds (not the historical plume)"
                fire_block["transport"] = transport
                fire_block["overlays"] = _corridor_overlays(transport["corridor_polygon"])
        fires.append(fire_block)

    skipped = len(detections) - len(fires)
    return {
        "status": "ok",
        "mode": "observed",
        "mode_label": "OBSERVED FIRE (satellite detection) + MODELLED ATMOSPHERIC TRANSPORT",
        "location": {"latitude": lat, "longitude": lon},
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window_days": days,
        "radius_km": radius_km,
        "fires": fires,
        "count": len(fires),
        "skipped_detections": skipped or None,
        "model_time_note": (
            "Transport is modelled from the current hour onward using forecast "
            "winds, anchored at each observed detection. It does not reconstruct "
            "where smoke went between detection and now."
        ),
        "disclaimer": TRANSPORT_DISCLAIMER,
        "safety": SAFETY_GUIDANCE,
        "provenance": {
            "kind": "observed+modeled",
            "source": "NASA FIRMS detections + Open-Meteo wind profile",
            "resolution": "detections 375 m/1 km; winds ~11 km NWP grid",
            "temporal": f"detections last {days} day(s); transport next {hours} h",
            "quality": "ok",
            "limitations": TRANSPORT_DISCLAIMER,
        },
    }
