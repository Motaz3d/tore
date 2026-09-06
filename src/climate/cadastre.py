"""
Cadastral real floor areas (docs/LOSS_DATA_ACQUISITION.md §2) — replaces
the declared floor-area assumption with REAL building areas where an
official cadastre is integrated.

Integrated:
- Netherlands — BAG (Basisregistratie Adressen en Gebouwen) via the PDOK
  WFS 2.0 service (open data, Kadaster attribution). The `pand` layer
  carries `oppervlakte_min`/`oppervlakte_max` per building — real floor
  areas, not assumptions.

Honesty contract: when no cadastre covers the location (or the service
fails), the caller keeps the declared assumption and says which basis was
used. The real-area result is a MEAN over sampled cadastral buildings —
never a per-building valuation.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from pyproj import Transformer

from ..dashboard.cache import cached

# Netherlands coverage (matches config/loss_estimate_benchmarks.json NL bbox).
_NL_BBOX = (3.358, 50.750, 7.227, 53.555)

_WFS_URL = "https://service.pdok.nl/lv/bag/wfs/v2_0"

_BAG_SOURCE = "Kadaster BAG via PDOK WFS (open data, attribution: Kadaster)"

TTL_CADASTRE = 24 * 3600.0  # 24 h — cadastral areas change slowly

# WGS84 -> RD New (EPSG:28992); pyproj is a declared platform dependency.
_TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)


def _in_netherlands(lat: float, lon: float) -> bool:
    return (_NL_BBOX[0] <= lon <= _NL_BBOX[2]
            and _NL_BBOX[1] <= lat <= _NL_BBOX[3])


def _rd_bbox(lat: float, lon: float, radius_m: float) -> List[float]:
    x, y = _TO_RD.transform(lon, lat)
    r = max(250.0, min(float(radius_m or 1000.0), 5000.0))
    return [x - r, y - r, x + r, y + r]


@cached("cadastre_bag_area", TTL_CADASTRE)
def _fetch_bag_area_sample(lat: float, lon: float, radius_m: float) -> Dict[str, Any]:
    """Sample BAG pand buildings around the point (RD bbox query, JSON)."""
    bbox = _rd_bbox(lat, lon, radius_m)
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "bag:pand",
        "count": "300",
        "bbox": ",".join(f"{v:.1f}" for v in bbox),
        "outputFormat": "json",
    }
    url = _WFS_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Talaix-Cadastre/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=40.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        return {"error": f"BAG WFS fetch failed: {exc}"}

    areas: List[float] = []
    for feature in data.get("features") or []:
        props = feature.get("properties") or {}
        amin = props.get("oppervlakte_min")
        amax = props.get("oppervlakte_max")
        if isinstance(amin, (int, float)) and isinstance(amax, (int, float)) and amax > 0:
            areas.append((float(amin) + float(amax)) / 2.0)
    if not areas:
        return {"error": "BAG WFS returned no buildings with floor areas here"}
    return {
        "building_count": len(areas),
        "mean_area_m2": round(sum(areas) / len(areas), 1),
    }


def real_floor_area_m2(
    lat: float, lon: float, radius_m: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Mean real floor area per building near the point, or None.

    Returns {mean_area_m2, building_count, source, method, licence_note}
    when an integrated cadastre covers the point; None otherwise — the
    caller keeps the declared assumption and labels the basis honestly.
    """
    if not _in_netherlands(lat, lon):
        return None
    sample = _fetch_bag_area_sample(lat, lon, radius_m or 1000.0)
    if "error" in sample:
        return None
    return {
        "mean_area_m2": sample["mean_area_m2"],
        "building_count": sample["building_count"],
        "source": _BAG_SOURCE,
        "licence_note": "BAG is Dutch open data (Kadaster); attribution required.",
        "method": (
            "Mean of (oppervlakte_min + oppervlakte_max)/2 over BAG pand "
            "buildings sampled within the radius; a real cadastral mean per "
            "building — not a per-building valuation."),
    }
