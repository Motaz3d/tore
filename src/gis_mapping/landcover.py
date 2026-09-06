"""
ESA WorldCover land-cover lookup and fuel-model mapping.

Reads the real ESA WorldCover 10 m land-cover product (2021, v200) from the
public COG bucket (no credentials required) and maps the dominant land-cover
class around a point to a Scott & Burgan style fuel model identifier used by
``prediction.fire_spread.FireSpreadModel``.

WorldCover classes (ESA WorldCover 2021):
    10 Tree cover | 20 Shrubland | 30 Grassland | 40 Cropland | 50 Built-up
    60 Bare/sparse | 70 Snow/ice | 80 Water | 90 Wetland | 95 Mangrove
    100 Moss/lichen
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np

from ..dashboard.cache import cached, TTL_LANDCOVER

try:
    import rasterio
    from rasterio.windows import Window
    from pyproj import Transformer

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False

_BUCKET = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
_PRODUCT = "ESA WorldCover 10m 2021 v200"

# WorldCover class -> (label, Scott & Burgan style fuel model, burnable).
# Fuel-model assignments are a documented screening approximation for the
# Mediterranean/European context; non-burnable classes map to None.
CLASS_MAP: Dict[int, Tuple[str, Optional[str], bool]] = {
    10: ("Tree cover", "TL3", True),
    20: ("Shrubland", "SH5", True),
    30: ("Grassland", "GR2", True),
    40: ("Cropland", "GR1", True),
    50: ("Built-up", "TU1", False),
    60: ("Bare/sparse vegetation", "TL1", False),
    70: ("Snow/ice", None, False),
    80: ("Permanent water", None, False),
    90: ("Herbaceous wetland", "GR2", True),
    95: ("Mangroves", None, False),
    100: ("Moss/lichen", "TU1", True),
}

_DEFAULT_FUEL = "TL3"  # fallback when the lookup fails entirely


def _tile_url(lat: float, lon: float) -> str:
    """Return the WorldCover tile URL for a point (3 deg x 3 deg tiles)."""
    lat0 = int(math.floor(lat / 3.0) * 3)
    lon0 = int(math.floor(lon / 3.0) * 3)
    lat_tag = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}"
    lon_tag = f"{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"
    return f"{_BUCKET}/ESA_WorldCover_10m_2021_v200_{lat_tag}{lon_tag}_Map.tif"


@cached("landcover", TTL_LANDCOVER)
def fetch_landcover(lat: float, lon: float, window_m: float = 500.0) -> Dict:
    """
    Look up the real land-cover class and fuel model for a point.

    Returns the dominant WorldCover class in a ~1 km box around the point,
    the class histogram, the mapped fuel model, and a burnable flag.
    Reports an honest error when the product cannot be read.
    """
    if not _HAS_RASTERIO:
        return {"error": "rasterio not installed", "source": _PRODUCT}
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {"error": "Coordinates out of range"}

    url = _tile_url(lat, lon)
    try:
        with rasterio.open(url) as ds:
            transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            row, col = ds.index(x, y)
            half = int(round(window_m / 10.0))  # 10 m pixels
            win = Window(col - half, row - half, 2 * half, 2 * half)
            win = win.intersection(Window(0, 0, ds.width, ds.height))
            arr = ds.read(1, window=win)
    except Exception as exc:
        return {"error": f"WorldCover read failed: {exc}", "source": _PRODUCT}

    if arr.size == 0:
        return {"error": "WorldCover returned no data for this location", "source": _PRODUCT}

    values, counts = np.unique(arr, return_counts=True)
    total = int(arr.size)
    histogram = {
        int(v): {
            "label": CLASS_MAP.get(int(v), ("Unknown", None, False))[0],
            "fraction": round(int(c) / total, 4),
        }
        for v, c in zip(values.tolist(), counts.tolist())
    }

    # Dominant burnable class (ignore non-burnable when picking the fuel model).
    ordered = sorted(zip(values.tolist(), counts.tolist()), key=lambda vc: -vc[1])
    dominant_class = int(ordered[0][0])
    fuel_class = None
    for v, _c in ordered:
        label, fuel, burnable = CLASS_MAP.get(int(v), ("Unknown", None, False))
        if burnable and fuel is not None:
            fuel_class = fuel
            break

    label, fuel, burnable = CLASS_MAP.get(dominant_class, ("Unknown", None, False))
    return {
        "dominant_class": dominant_class,
        "dominant_label": label,
        "dominant_fraction": histogram[dominant_class]["fraction"],
        "fuel_model": fuel_class or _DEFAULT_FUEL,
        "fuel_model_is_fallback": fuel_class is None,
        "burnable": burnable,
        "histogram": histogram,
        "resolution": "10 m",
        "source": _PRODUCT,
    }
