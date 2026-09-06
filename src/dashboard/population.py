"""
Population exposure intelligence from real gridded population data.

Primary source: **WorldPop** gridded population estimates (open data,
CC-BY 4.0). WorldPop rasters are modelled dasymetric population estimates
at ~100 m resolution with an explicit reference year — they are never
presented as exact counts.

Access path (probed 2026-08-16): ``data.worldpop.org`` serves per-country
GeoTIFFs over HTTPS. The server advertises ``Accept-Ranges: bytes`` but in
practice ignores HTTP Range requests (returns 200 with the whole file), so
true windowed remote reads are not possible. Talaix therefore
downloads a country raster **once** into a local disk cache
(``data/population/``) and performs all subsequent reads locally and
cheaply. No global download is ever triggered by a user request — only the
single country covering the analysed point, on first use, with a hard size
guard.

Source audit status: WorldPop = integrated; GHSL / GPW / Eurostat GEOSTAT =
candidates (see config/source_registry.json for the full audit trail).

Everything here is an *estimate with a reference year*: wording follows
"Estimated population exposure based on WorldPop, reference year YYYY"
rather than "there are exactly X people here".
"""

from __future__ import annotations

import math
import os
import shutil
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import real_data
from .cache import cached

try:
    import rasterio
    from rasterio.windows import Window
    from pyproj import Transformer

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False

TTL_POPULATION = 30 * 24 * 3600.0      # computed products: 30 days
TTL_POPULATION_EXPOSURE = 1800.0       # hazard overlay: 30 min (tracks risk grid)

_MAX_RASTER_BYTES = 400 * 1024 * 1024  # hard guard for one-time country download
_DEFAULT_RADIUS_KM = 3.0
_GRID_CELLS = 24                        # downsampled map grid (grid_n x grid_n)

#: WorldPop download candidates, tried in order. R2025A (Global 2) constrained
#: 100 m is the newest annual series; the Global 1 2020 UN-adjusted product is
#: the established fallback. Both are documented WorldPop products.
_WORLDPOP_BASE = "https://data.worldpop.org/GIS/Population"


def _candidate_products(iso3: str) -> List[Dict]:
    lower = iso3.lower()
    return [
        {
            "url": (
                f"{_WORLDPOP_BASE}/Global_2015_2030/R2025A/2025/{iso3}/v1/100m/"
                f"constrained/{lower}_pop_2025_CN_100m_R2025A_v1.tif"
            ),
            "product": "WorldPop Global 2 (R2025A) constrained 100 m",
            "reference_year": 2025,
            "variant": "constrained (population allocated to built-up/settled cells)",
        },
        {
            "url": f"{_WORLDPOP_BASE}/Global_2000_2020/2020/{iso3}/{lower}_ppp_2020_UNadj.tif",
            "product": "WorldPop Global 1 unconstrained 100 m (UN-adjusted)",
            "reference_year": 2020,
            "variant": "unconstrained",
        },
    ]


#: ISO 3166-1 alpha-2 -> alpha-3 (static reference table; needed because
#: Nominatim returns alpha-2 while WorldPop organises rasters by alpha-3).
_ALPHA2_TO_ALPHA3 = {
    "ad": "AND", "ae": "ARE", "af": "AFG", "ag": "ATG", "ai": "AIA", "al": "ALB",
    "am": "ARM", "ao": "AGO", "aq": "ATA", "ar": "ARG", "as": "ASM", "at": "AUT",
    "au": "AUS", "aw": "ABW", "ax": "ALA", "az": "AZE", "ba": "BIH", "bb": "BRB",
    "bd": "BGD", "be": "BEL", "bf": "BFA", "bg": "BGR", "bh": "BHR", "bi": "BDI",
    "bj": "BEN", "bl": "BLM", "bm": "BMU", "bn": "BRN", "bo": "BOL", "bq": "BES",
    "br": "BRA", "bs": "BHS", "bt": "BTN", "bv": "BVT", "bw": "BWA", "by": "BLR",
    "bz": "BLZ", "ca": "CAN", "cc": "CCK", "cd": "COD", "cf": "CAF", "cg": "COG",
    "ch": "CHE", "ci": "CIV", "ck": "COK", "cl": "CHL", "cm": "CMR", "cn": "CHN",
    "co": "COL", "cr": "CRI", "cu": "CUB", "cv": "CPV", "cw": "CUW", "cx": "CXR",
    "cy": "CYP", "cz": "CZE", "de": "DEU", "dj": "DJI", "dk": "DNK", "dm": "DMA",
    "do": "DOM", "dz": "DZA", "ec": "ECU", "ee": "EST", "eg": "EGY", "eh": "ESH",
    "er": "ERI", "es": "ESP", "et": "ETH", "fi": "FIN", "fj": "FJI", "fk": "FLK",
    "fm": "FSM", "fo": "FRO", "fr": "FRA", "ga": "GAB", "gb": "GBR", "gd": "GRD",
    "ge": "GEO", "gf": "GUF", "gg": "GGY", "gh": "GHA", "gi": "GIB", "gl": "GRL",
    "gm": "GMB", "gn": "GIN", "gp": "GLP", "gq": "GNQ", "gr": "GRC", "gs": "SGS",
    "gt": "GTM", "gu": "GUM", "gw": "GNB", "gy": "GUY", "hk": "HKG", "hm": "HMD",
    "hn": "HND", "hr": "HRV", "ht": "HTI", "hu": "HUN", "id": "IDN", "ie": "IRL",
    "il": "ISR", "im": "IMN", "in": "IND", "io": "IOT", "iq": "IRQ", "ir": "IRN",
    "is": "ISL", "it": "ITA", "je": "JEY", "jm": "JAM", "jo": "JOR", "jp": "JPN",
    "ke": "KEN", "kg": "KGZ", "kh": "KHM", "ki": "KIR", "km": "COM", "kn": "KNA",
    "kp": "PRK", "kr": "KOR", "kw": "KWT", "ky": "CYM", "kz": "KAZ", "la": "LAO",
    "lb": "LBN", "lc": "LCA", "li": "LIE", "lk": "LKA", "lr": "LBR", "ls": "LSO",
    "lt": "LTU", "lu": "LUX", "lv": "LVA", "ly": "LBY", "ma": "MAR", "mc": "MCO",
    "md": "MDA", "me": "MNE", "mf": "MAF", "mg": "MDG", "mh": "MHL", "mk": "MKD",
    "ml": "MLI", "mm": "MMR", "mn": "MNG", "mo": "MAC", "mp": "MNP", "mq": "MTQ",
    "mr": "MRT", "ms": "MSR", "mt": "MLT", "mu": "MUS", "mv": "MDV", "mw": "MWI",
    "mx": "MEX", "my": "MYS", "mz": "MOZ", "na": "NAM", "nc": "NCL", "ne": "NER",
    "nf": "NFK", "ng": "NGA", "ni": "NIC", "nl": "NLD", "no": "NOR", "np": "NPL",
    "nr": "NRU", "nu": "NIU", "nz": "NZL", "om": "OMN", "pa": "PAN", "pe": "PER",
    "pf": "PYF", "pg": "PNG", "ph": "PHL", "pk": "PAK", "pl": "POL", "pm": "SPM",
    "pn": "PCN", "pr": "PRI", "ps": "PSE", "pt": "PRT", "pw": "PLW", "py": "PRY",
    "qa": "QAT", "re": "REU", "ro": "ROU", "rs": "SRB", "ru": "RUS", "rw": "RWA",
    "sa": "SAU", "sb": "SLB", "sc": "SYC", "sd": "SDN", "se": "SWE", "sg": "SGP",
    "sh": "SHN", "si": "SVN", "sj": "SJM", "sk": "SVK", "sl": "SLE", "sm": "SMR",
    "sn": "SEN", "so": "SOM", "sr": "SUR", "ss": "SSD", "st": "STP", "sv": "SLV",
    "sx": "SXM", "sy": "SYR", "sz": "SWZ", "tc": "TCA", "td": "TCD", "tf": "ATF",
    "tg": "TGO", "th": "THA", "tj": "TJK", "tk": "TKL", "tl": "TLS", "tm": "TKM",
    "tn": "TUN", "to": "TON", "tr": "TUR", "tt": "TTO", "tv": "TUV", "tw": "TWN",
    "tz": "TZA", "ua": "UKR", "ug": "UGA", "um": "UMI", "us": "USA", "uy": "URY",
    "uz": "UZB", "va": "VAT", "vc": "VCT", "ve": "VEN", "vg": "VGB", "vi": "VIR",
    "vn": "VNM", "vu": "VUT", "wf": "WLF", "ws": "WSM", "ye": "YEM", "yt": "MYT",
    "za": "ZAF", "zm": "ZMB", "zw": "ZWE",
}


def _population_dir() -> str:
    """Local raster disk cache (one-time download per country)."""
    path = os.environ.get("HYDRASHIELD_POPULATION_DIR")
    if not path:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "population")
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


@cached("rev_geocode", 30 * 24 * 3600.0)
def country_code_for(lat: float, lon: float) -> Dict:
    """
    ISO 3166-1 alpha-2 country code for a point via Nominatim reverse
    geocoding (real data, ODbL). Coordinates are rounded to ~0.1 deg for the
    cache key; near borders this can pick the neighbouring country, which is
    declared as a limitation.
    """
    if not real_data._valid_point(lat, lon):
        return {"error": "Coordinates out of range"}
    url = (
        "https://nominatim.openstreetmap.org/reverse?format=jsonv2&zoom=5"
        f"&lat={lat:.4f}&lon={lon:.4f}"
    )
    try:
        data = real_data._get_json(url, timeout=10.0, retries=1)
    except RuntimeError as exc:
        return {"error": f"Reverse geocoding failed: {exc}"}
    code = ((data or {}).get("address") or {}).get("country_code")
    if not code:
        return {"error": "No country at this location (ocean or unmapped area)"}
    return {
        "country_code": code.lower(),
        "country": (data.get("address") or {}).get("country"),
        "source": "Nominatim (OpenStreetMap) reverse geocoding",
    }


def _ensure_raster(iso3: str) -> Dict:
    """
    Ensure the WorldPop country raster exists locally; download once if not.

    Returns ``{"path": ..., "meta": ...}`` or ``{"error": ...}``.
    """
    if not _HAS_RASTERIO:
        return {"error": "rasterio not installed"}
    dest_dir = _population_dir()
    errors = []
    for cand in _candidate_products(iso3):
        fname = cand["url"].rsplit("/", 1)[-1]
        dest = os.path.join(dest_dir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return {"path": dest, "meta": cand}
        try:
            req = urllib.request.Request(
                cand["url"], method="HEAD", headers={"User-Agent": real_data._UA}
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                size = int(resp.headers.get("Content-Length") or 0)
            if size <= 0:
                errors.append(f"{cand['product']}: no size reported")
                continue
            if size > _MAX_RASTER_BYTES:
                errors.append(
                    f"{cand['product']}: raster too large for on-demand download "
                    f"({size / 1e6:.0f} MB > {_MAX_RASTER_BYTES / 1e6:.0f} MB guard)"
                )
                continue
            tmp = dest + ".part"
            req = urllib.request.Request(cand["url"], headers={"User-Agent": real_data._UA})
            with urllib.request.urlopen(req, timeout=300.0) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1024 * 1024)
            os.replace(tmp, dest)
            return {"path": dest, "meta": cand}
        except Exception as exc:  # try the next candidate product
            errors.append(f"{cand['product']}: {exc}")
    return {
        "error": "WorldPop raster unavailable for this country: " + "; ".join(errors),
        "errors": errors,
    }


def _read_window(ds, lat: float, lon: float, radius_km: float):
    """Read the raster window covering the radius; return (array, transform)."""
    transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = ds.index(x, y)
    pixel_y = abs(ds.res[1])  # ~100 m in raster units (degrees for EPSG:4326)
    pixel_x = abs(ds.res[0])
    half_lat = radius_km / 110.54
    half_lon = radius_km / (111.32 * max(0.01, math.cos(math.radians(lat))))
    # Convert the geographic half-extent into raster pixels (raster CRS units
    # per degree approximated via the dataset resolution; WorldPop is EPSG:4326).
    if ds.crs and ds.crs.to_epsg() == 4326:
        half_rows = int(math.ceil(half_lat / pixel_y))
        half_cols = int(math.ceil(half_lon / pixel_x))
    else:
        metres_per_unit = 111320.0  # conservative fallback for projected rasters
        half_rows = int(math.ceil(radius_km * 1000.0 / (pixel_y * metres_per_unit)))
        half_cols = int(math.ceil(radius_km * 1000.0 / (pixel_x * metres_per_unit)))
    win = Window(col - half_cols, row - half_rows, 2 * half_cols + 1, 2 * half_rows + 1)
    win = win.intersection(Window(0, 0, ds.width, ds.height))
    arr = ds.read(1, window=win, masked=True)
    return arr, ds.window_transform(win)


def _pixel_coords(transform, shape) -> Tuple[np.ndarray, np.ndarray]:
    """Lat/lon arrays of pixel centres for a window (EPSG:4326 rasters)."""
    rows, cols = np.indices(shape)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    return ys, xs  # lat, lon


def _radius_mask(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray,
                 radius_km: float) -> np.ndarray:
    """Boolean mask of pixels whose centre lies within radius_km of the point."""
    dlat = np.deg2rad(lats - lat0)
    dlon = np.deg2rad(lons - lon0)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat0)) * np.cos(np.deg2rad(lats)) * np.sin(dlon / 2) ** 2
    dist_km = 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return dist_km <= radius_km


@cached("population", TTL_POPULATION)
def fetch_population(lat: float, lon: float, radius_km: float = _DEFAULT_RADIUS_KM) -> Dict:
    """
    Estimated population around a point from the real WorldPop raster.

    Returns totals, density and a downsampled cell grid for map display, all
    labelled with product, reference year, resolution and license. Honest
    error dict when no real estimate can be produced.
    """
    if not real_data._valid_point(lat, lon):
        return {"error": "Coordinates out of range"}
    radius_km = min(max(float(radius_km), 0.5), 10.0)

    cc = country_code_for(round(lat, 1), round(lon, 1))
    if "error" in cc:
        return {"error": cc["error"], "stage": "country_lookup"}
    iso3 = _ALPHA2_TO_ALPHA3.get(cc["country_code"])
    if iso3 is None:
        return {"error": f"No WorldPop raster mapping for country code '{cc['country_code']}'"}

    ensured = _ensure_raster(iso3)
    if "error" in ensured:
        return {"error": ensured["error"], "stage": "raster_download", "iso3": iso3}

    try:
        with rasterio.open(ensured["path"]) as ds:
            arr, transform = _read_window(ds, lat, lon, radius_km)
    except Exception as exc:
        return {"error": f"WorldPop read failed: {exc}", "stage": "raster_read", "iso3": iso3}
    if arr.size == 0:
        return {"error": "WorldPop raster window empty for this location", "iso3": iso3}

    values = np.asarray(arr.filled(0.0), dtype=np.float64)
    values[values < 0] = 0.0
    lats, lons = _pixel_coords(transform, values.shape)
    mask = _radius_mask(lat, lon, lats, lons, radius_km)
    total = float(values[mask].sum())
    area_km2 = math.pi * radius_km ** 2

    # Downsampled non-zero cells for the map layer (bounded payload).
    cells = []
    h, w = values.shape
    gh = max(1, h // _GRID_CELLS)
    gw = max(1, w // _GRID_CELLS)
    for r0 in range(0, h, gh):
        for c0 in range(0, w, gw):
            block = values[r0:r0 + gh, c0:c0 + gw]
            bmask = mask[r0:r0 + gh, c0:c0 + gw]
            pop = float(block[bmask].sum())
            if pop < 0.5:
                continue
            cells.append({
                "south": float(lats[r0:r0 + gh, c0:c0 + gw].min()),
                "north": float(lats[r0:r0 + gh, c0:c0 + gw].max()),
                "west": float(lons[r0:r0 + gh, c0:c0 + gw].min()),
                "east": float(lons[r0:r0 + gh, c0:c0 + gw].max()),
                "population": round(pop),
            })

    meta = ensured["meta"]
    year = meta["reference_year"]
    return {
        "status": "ok",
        "latitude": lat,
        "longitude": lon,
        "radius_km": radius_km,
        "country_code": cc["country_code"],
        "iso3": iso3,
        "estimated_population": int(round(total)),
        "mean_density_per_km2": round(total / area_km2, 1) if area_km2 else None,
        "area_km2": round(area_km2, 2),
        "grid": {"max_cells": _GRID_CELLS, "cells": cells},
        "estimate_note": (
            f"Estimated population exposure based on WorldPop, reference year {year} "
            "(modelled gridded estimates at ~100 m) — not an exact count."
        ),
        "product": meta["product"],
        "variant": meta["variant"],
        "reference_year": year,
        "resolution": "100 m (grid cells)",
        "license": "CC-BY 4.0 (WorldPop, University of Southampton)",
        "source": "WorldPop gridded population",
        "source_url": meta["url"],
        "provenance": {
            "kind": "modeled",
            "source": f"{meta['product']}, reference year {year}",
            "resolution": "100 m",
            "temporal": f"reference year {year}",
            "quality": "ok",
            "limitations": (
                "Gridded modelled estimates (dasymetric), not a census count; "
                f"variant: {meta['variant']}; country raster cached locally after "
                "a one-time download; country lookup rounded to ~0.1 deg."
            ),
        },
    }


def population_in_polygon(iso3: str, polygon: List[Tuple[float, float]]) -> Dict:
    """
    Estimated population inside a lat/lon polygon (e.g. a smoke corridor).

    Vectorised even-odd point-in-polygon over the raster pixels inside the
    polygon bounding box. Returns an honest error when the raster is not
    available locally (never downloads implicitly here).
    """
    if not _HAS_RASTERIO:
        return {"error": "rasterio not installed"}
    if not polygon or len(polygon) < 3:
        return {"error": "Polygon too small"}
    ensured = _ensure_raster(iso3)
    if "error" in ensured:
        return {"error": ensured["error"]}

    lats = np.array([p[0] for p in polygon])
    lons = np.array([p[1] for p in polygon])
    try:
        with rasterio.open(ensured["path"]) as ds:
            if ds.crs and ds.crs.to_epsg() != 4326:
                return {"error": "Unsupported raster CRS for polygon overlay"}
            south, north = float(lats.min()), float(lats.max())
            west, east = float(lons.min()), float(lons.max())
            try:
                row0, col0 = ds.index(west, north)
                row1, col1 = ds.index(east, south)
            except Exception:
                return {"error": "Polygon outside raster extent"}
            win = Window(col0, row0, max(1, col1 - col0 + 1), max(1, row1 - row0 + 1))
            win = win.intersection(Window(0, 0, ds.width, ds.height))
            arr = ds.read(1, window=win, masked=True)
            transform = ds.window_transform(win)
    except Exception as exc:
        return {"error": f"WorldPop read failed: {exc}"}
    if arr.size == 0:
        return {"error": "Polygon outside raster extent"}

    values = np.asarray(arr.filled(0.0), dtype=np.float64)
    values[values < 0] = 0.0
    plat, plon = _pixel_coords(transform, values.shape)

    # Even-odd ray casting, vectorised over pixels, one polygon edge at a time.
    inside = np.zeros(values.shape, dtype=bool)
    n = len(polygon)
    for i in range(n):
        y1, x1 = polygon[i]
        y2, x2 = polygon[(i + 1) % n]
        crosses = (plat > min(y1, y2)) & (plat <= max(y1, y2))
        if y1 != y2:
            xint = x1 + (plat - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crosses & (plon < xint)
    total = float(values[inside].sum())
    year = ensured["meta"]["reference_year"]
    return {
        "estimated_population": int(round(total)),
        "reference_year": year,
        "estimate_note": (
            f"Estimated population within the modelled area based on WorldPop, "
            f"reference year {year} (gridded estimates, not an exact count)."
        ),
        "source": f"{ensured['meta']['product']}, reference year {year}",
    }


def _density_level(density: Optional[float]) -> Optional[str]:
    """Declared density levels (people/km²) for communication only."""
    if density is None:
        return None
    if density < 25:
        return "low"
    if density < 100:
        return "moderate"
    if density < 500:
        return "high"
    return "very high"


_ELEVATED_HAZARD = {"High", "Very high", "Extreme"}


def build_population_block(analysis: Dict, radius_km: float = _DEFAULT_RADIUS_KM) -> Dict:
    """
    Population & human-exposure block for the analysis result.

    Combines the real WorldPop estimate with the *existing* OSM exposure
    block (reused, not duplicated) and the current fire-danger class. The
    combination is a declared, qualitative priority statement — population is
    never multiplied by a risk score to invent a probability.
    """
    loc = analysis.get("location") or {}
    lat, lon = loc.get("latitude"), loc.get("longitude")
    if lat is None or lon is None:
        return {"status": "unavailable", "reason": "No analysis location"}

    pop = fetch_population(float(lat), float(lon), radius_km)
    if "error" in pop:
        return {
            "status": "unavailable",
            "reason": pop["error"],
            "provenance": {
                "kind": "unavailable",
                "source": "WorldPop gridded population",
                "quality": "missing",
                "limitations": pop.get("stage"),
            },
        }

    danger = analysis.get("fire_danger") or {}
    hazard_class = danger.get("effis_class") or danger.get("class")
    density_level = _density_level(pop.get("mean_density_per_km2"))

    exposure = analysis.get("exposure") or {}
    assets = (exposure.get("vulnerable_assets") or {}) if exposure.get("status") == "ok" else {}
    buildings = ((exposure.get("exposure") or {}).get("buildings_mapped")
                 if exposure.get("status") == "ok" else None)

    # Declared qualitative combination matrix (text, never a number).
    elevated = hazard_class in _ELEVATED_HAZARD
    if elevated and density_level in ("high", "very high"):
        priority = "high"
        priority_note = (
            f"{hazard_class} wildfire hazard coincides with {density_level} "
            "population density — elevated human-exposure priority."
        )
    elif elevated:
        priority = "moderate"
        priority_note = (
            f"{hazard_class} wildfire hazard with {density_level or 'unknown'} "
            "population density — fewer people exposed, exposure still possible."
        )
    elif density_level in ("high", "very high"):
        priority = "watch"
        priority_note = (
            f"Population density is {density_level} but the current hazard class "
            f"is {hazard_class or 'unknown'} — no elevated exposure signal now."
        )
    else:
        priority = "routine"
        priority_note = (
            f"Current hazard class {hazard_class or 'unknown'}, population density "
            f"{density_level or 'unknown'} — no combined exposure signal."
        )

    exposed = pop["estimated_population"] if elevated else None
    return {
        "status": "ok",
        "radius_km": pop["radius_km"],
        "estimated_population": pop["estimated_population"],
        "mean_density_per_km2": pop["mean_density_per_km2"],
        "density_level": density_level,
        "estimate_note": pop["estimate_note"],
        "reference_year": pop["reference_year"],
        "product": pop["product"],
        "resolution": pop["resolution"],
        "license": pop["license"],
        "hazard_class": hazard_class,
        "estimated_population_in_hazard_area": exposed,
        "exposure_note": (
            f"Estimated population within {pop['radius_km']} km of the analysed point "
            f"while the area carries hazard class '{hazard_class}'. The hazard class is "
            "modelled at analysis-area scale; this is a screening overlay, not a "
            "person-level exposure assessment."
            if elevated else
            "Current hazard class is not elevated; no population exposure estimate "
            "is attached to the hazard. Population figures remain available above."
        ),
        "critical_facilities": {
            "hospitals": assets.get("hospitals"),
            "schools": assets.get("schools"),
            "fire_stations": assets.get("fire_stations"),
            "power_facilities": assets.get("power_facilities"),
            "note": "Mapped OpenStreetMap features within the OSM exposure radius "
                    "(reused from the exposure layer; completeness varies by region).",
        } if assets else None,
        "mapped_buildings": buildings,
        "human_exposure_priority": priority,
        "human_exposure_note": priority_note,
        "separate_from_score_note": (
            "Population exposure is reported separately from the composite "
            "wildfire-risk score; it is never multiplied into a probability."
        ),
        "provenance": pop["provenance"],
    }


@cached("population_exposure", TTL_POPULATION_EXPOSURE)
def population_exposure_overlay(lat: float, lon: float, radius_km: float = _DEFAULT_RADIUS_KM,
                                n: int = 5) -> Dict:
    """
    Spatial overlay: estimated population per wildfire-hazard class.

    The real risk grid (FWI-derived danger classes per cell) is intersected
    with the real WorldPop population grid over the same bounding box, so the
    "exposed population by class" figures are a genuine spatial overlay of
    two real datasets — not a scalar multiplication.
    """
    from . import grid as risk_grid_mod

    if not real_data._valid_point(lat, lon):
        return {"error": "Coordinates out of range"}
    radius_km = min(max(float(radius_km), 1.0), 10.0)
    half_lat = radius_km / 110.54
    half_lon = radius_km / (111.32 * max(0.01, math.cos(math.radians(lat))))
    south, north = lat - half_lat, lat + half_lat
    west, east = lon - half_lon, lon + half_lon

    rg = risk_grid_mod.compute_risk_grid(south, west, north, east, n)
    if "error" in rg:
        return {"error": f"Risk grid unavailable: {rg['error']}"}
    pop = fetch_population(lat, lon, radius_km)
    if "error" in pop:
        return {"error": pop["error"]}

    # Cell-class lookup from the risk grid (n x n regular cells over bbox).
    cells = rg.get("features") or []
    gn = int((rg.get("grid") or {}).get("n") or n)
    dlat = (north - south) / gn
    dlon = (east - west) / gn

    def class_at(plat: float, plon: float) -> Optional[str]:
        i = int((plat - south) / dlat)
        j = int((plon - west) / dlon)
        if not (0 <= i < gn and 0 <= j < gn):
            return None
        idx = i * gn + j
        if idx >= len(cells):
            return None
        return (cells[idx].get("properties") or {}).get("risk_class")

    by_class: Dict[str, int] = {}
    unclassified = 0
    for cell in pop["grid"]["cells"]:
        clat = (cell["south"] + cell["north"]) / 2.0
        clon = (cell["west"] + cell["east"]) / 2.0
        cls = class_at(clat, clon)
        if cls is None:
            unclassified += cell["population"]
            continue
        by_class[cls] = by_class.get(cls, 0) + int(cell["population"])

    year = pop["reference_year"]
    return {
        "status": "ok",
        "latitude": lat,
        "longitude": lon,
        "radius_km": radius_km,
        "estimated_population": pop["estimated_population"],
        "mean_density_per_km2": pop["mean_density_per_km2"],
        "population_by_hazard_class": by_class,
        "population_unclassified": unclassified or None,
        "population_grid": pop["grid"],
        "risk_grid": rg.get("grid"),
        "estimate_note": (
            f"Estimated population exposure by hazard class: real WorldPop grid "
            f"(reference year {year}) overlaid on the FWI-derived risk grid. "
            "Gridded estimates — not exact counts."
        ),
        "product": pop["product"],
        "reference_year": year,
        "resolution": f"population 100 m; hazard cells ~{rg['grid']['cell_size_km']} km",
        "provenance": {
            "population": pop["provenance"],
            "hazard": "Derived: Canadian FWI from Open-Meteo daily model data (risk grid)",
        },
    }
