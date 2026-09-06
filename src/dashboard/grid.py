"""
Grid-level risk computation for map display.

Computes a coarse grid of FWI-based fire-danger values over a bounding box
using batched real-data calls:

    - One Open-Meteo multi-location daily request (all cells at once)
    - One OpenTopoData batch elevation request (all cells at once)

Slope is estimated from grid-level elevation differences (declared, coarse).
Results are cached per bounding box for one hour to bound upstream call
rates.
"""

from __future__ import annotations

import math
import urllib.parse
from datetime import date
from typing import Dict, List, Optional, Tuple

from . import real_data
from .cache import cached
from .real_analysis import TalaixRealAnalyser, _clamp
from ..prediction.fwi import compute_fwi_series

TTL_GRID = 3600.0


def _cell_centers(bbox: Tuple[float, float, float, float], n: int) -> List[Tuple[float, float]]:
    """Return (lat, lon) centers of an n x n grid over bbox=(s, w, n_, e)."""
    south, west, north, east = bbox
    dlat = (north - south) / n
    dlon = (east - west) / n
    return [
        (south + (i + 0.5) * dlat, west + (j + 0.5) * dlon)
        for i in range(n)
        for j in range(n)
    ]


def _fetch_daily_multi(points: List[Tuple[float, float]]) -> Optional[List[Dict]]:
    """One Open-Meteo daily request for many points. None on failure."""
    lats = ",".join(f"{p[0]:.4f}" for p in points)
    lons = ",".join(f"{p[1]:.4f}" for p in points)
    params = urllib.parse.urlencode(
        {
            "latitude": lats,
            "longitude": lons,
            "daily": (
                "temperature_2m_max,relative_humidity_2m_min,"
                "relative_humidity_2m_mean,wind_speed_10m_mean,"
                "wind_speed_10m_max,precipitation_sum"
            ),
            "timezone": "auto",
            "past_days": 14,
            "forecast_days": 1,
        }
    )
    try:
        data = real_data._get_json(f"https://api.open-meteo.com/v1/forecast?{params}", timeout=30.0)
    except RuntimeError:
        return None
    if isinstance(data, dict):
        data = [data]
    return data


def _fetch_elevations_multi(points: List[Tuple[float, float]]) -> Optional[List[Optional[float]]]:
    """One OpenTopoData batch request for many points. None on failure."""
    strs = [f"{p[0]:.4f},{p[1]:.4f}" for p in points]
    for dataset in ("eudem25m", "srtm90m"):
        result = real_data._opentopodata_lookup(strs, dataset)
        if result is not None:
            return result
    return None


@cached("risk_grid", TTL_GRID)
def compute_risk_grid(south: float, west: float, north: float, east: float, n: int = 5) -> Dict:
    """
    Compute an n x n fire-danger grid over a bounding box.

    Returns GeoJSON FeatureCollection of square cell polygons, each with
    fwi / risk / class properties, plus grid-level provenance. Cell size is
    reported so the UI can label the resolution honestly.
    """
    # Hard limits to protect upstream services and worker time.
    n = int(_clamp(n, 2, 7))
    span_lat = abs(north - south)
    span_lon = abs(east - west)
    if span_lat <= 0 or span_lon <= 0 or span_lat > 1.5 or span_lon > 1.5:
        return {"error": "Bounding box invalid or too large (max 1.5 deg per side)"}

    bbox = (south, west, north, east)
    centers = _cell_centers(bbox, n)

    daily_multi = _fetch_daily_multi(centers)
    elevations = _fetch_elevations_multi(centers)
    if daily_multi is None and elevations is None:
        return {"error": "Grid data sources unavailable"}

    today = date.today().isoformat()
    analyser = TalaixRealAnalyser()

    cells: List[Dict] = []
    fwi_by_idx: Dict[int, Optional[float]] = {}

    for idx, (lat, lon) in enumerate(centers):
        fwi_val: Optional[float] = None
        fwi_class: Optional[str] = None
        if daily_multi is not None and idx < len(daily_multi):
            daily = (daily_multi[idx] or {}).get("daily") or {}
            times = daily.get("time") or []
            series_in = []
            for i, t in enumerate(times):
                def _v(name: str):
                    s = daily.get(name) or []
                    return s[i] if i < len(s) else None

                temp = _num_safe(_v("temperature_2m_max"))
                rh = _num_safe(_v("relative_humidity_2m_min")) or _num_safe(
                    _v("relative_humidity_2m_mean")
                )
                wind = _num_safe(_v("wind_speed_10m_mean")) or _num_safe(
                    _v("wind_speed_10m_max")
                )
                rain = _num_safe(_v("precipitation_sum")) or 0.0
                if temp is None or rh is None or wind is None:
                    continue
                series_in.append(
                    {"date": t, "temp_c": temp, "rh_pct": rh, "wind_kmh": wind, "rain_mm": rain}
                )
            if len(series_in) >= 5:
                days = compute_fwi_series(series_in)
                past = [d for d in days if d.date <= today]
                current = past[-1] if past else days[-1]
                fwi_val = round(current.fwi, 1)
                fwi_class = current.danger_class
        fwi_by_idx[idx] = fwi_val

    # Grid-scale slope from cell-to-cell elevation differences.
    dlat_m = span_lat / n * 110_540.0
    dlon_m = span_lon / n * 111_320.0 * math.cos(math.radians((south + north) / 2.0))

    def elev_at(i: int, j: int) -> Optional[float]:
        idx = i * n + j
        if elevations is None or idx >= len(elevations):
            return None
        return elevations[idx]

    features = []
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            lat, lon = centers[idx]
            slope = 0.0
            e = elev_at(i, j)
            e_w = elev_at(i, j - 1) if j > 0 else None
            e_e = elev_at(i, j + 1) if j < n - 1 else None
            e_s = elev_at(i - 1, j) if i > 0 else None
            e_n = elev_at(i + 1, j) if i < n - 1 else None
            if e is not None:
                dz_dx = ((e_e - e) / dlon_m) if e_e is not None else ((e - e_w) / dlon_m if e_w is not None else 0.0)
                dz_dy = ((e_n - e) / dlat_m) if e_n is not None else ((e - e_s) / dlat_m if e_s is not None else 0.0)
                slope = math.degrees(math.atan(math.hypot(dz_dx, dz_dy)))

            fwi_val = fwi_by_idx[idx]
            risk = analyser._risk_score(fwi=fwi_val, slope=slope, fmc=None, wind_kmh=0.0)
            risk_class = analyser._risk_class(risk)

            # Cell polygon (lon/lat).
            lat0 = south + i * (span_lat / n)
            lat1 = south + (i + 1) * (span_lat / n)
            lon0 = west + j * (span_lon / n)
            lon1 = west + (j + 1) * (span_lon / n)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]
                        ]],
                    },
                    "properties": {
                        "lat": round(lat, 4),
                        "lon": round(lon, 4),
                        "fwi": fwi_val,
                        "fwi_class": fwi_class,
                        "risk": risk,
                        "risk_class": risk_class,
                        "slope_deg": round(slope, 2),
                    },
                }
            )

    return {
        "type": "FeatureCollection",
        "features": features,
        "grid": {
            "n": n,
            "cell_size_km": round((span_lat / n) * 110.6, 2),
            "bbox": [south, west, north, east],
        },
        "provenance": {
            "fire_danger": "Derived: Canadian FWI from Open-Meteo daily model data (batched)",
            "terrain": "Observed: OpenTopoData DEM, grid-scale slope (coarse)",
            "fuel_moisture": "Not included at grid level (scene-level only)",
            "freshness": "Daily; cached 1 h",
        },
    }


def _num_safe(value) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
