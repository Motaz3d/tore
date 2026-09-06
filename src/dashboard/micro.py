"""
Micro-area intelligence.

Distinguishes what each real data layer actually resolves at a location:

    micro    (< ~100 m)  Sentinel-2 indices, ESA WorldCover
    local    (~25-90 m)  DEM-derived slope/aspect
    regional (~11 km)    weather model, FWI, soil moisture

Where a real Sentinel-2 scene grid exists, the block reports the measured
within-scene variability of vegetation moisture (NDMI) — genuine
micro-area information, computed from the actual scene pixels.

It never pretends that coarse (~11 km) model data resolves streets.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

_RESOLUTION_TABLE = [
    {"layer": "Satellite vegetation indices (NDVI/NDMI)",
     "resolution": "10 m", "scope": "micro",
     "source": "Copernicus Sentinel-2 L2A"},
    {"layer": "Land cover / burnability", "resolution": "10 m", "scope": "micro",
     "source": "ESA WorldCover"},
    {"layer": "Terrain (slope/aspect)", "resolution": "25 m (EU-DEM) / 90 m (SRTM)",
     "scope": "local", "source": "OpenTopoData DEM"},
    {"layer": "Weather / FWI / soil moisture", "resolution": "~11 km model grid",
     "scope": "regional",
     "source": "Open-Meteo (weather model + ERA5)"},
    {"layer": "Built environment (buildings, roads, facilities)",
     "resolution": "mapped features within radius", "scope": "micro",
     "source": "OpenStreetMap (completeness varies)"},
]

_COARSE_NOTE = (
    "Weather, fire danger (FWI) and soil moisture are ~11 km model-grid "
    "data: they describe the regional conditions, not a street, building or "
    "individual parcel. Micro-area statements rely on the 10–30 m satellite "
    "scene (Sentinel-2, Landsat fallback) and 10 m land cover only."
)


def _grid_stats(grid: Optional[List[List[Optional[float]]]]) -> Optional[Dict]:
    """Real statistics over the measured NDMI scene grid (None = no grid)."""
    if not grid:
        return None
    vals = [float(v) for row in grid for v in row if v is not None]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {
        "cells": n,
        "mean": round(mean, 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "range": round(max(vals) - min(vals), 4),
        "std": round(math.sqrt(var), 4),
    }


def _extent_m(bounds) -> Optional[Dict]:
    """Approximate scene extent in metres from the real grid bounds."""
    if isinstance(bounds, dict):
        west = bounds.get("lon_min")
        east = bounds.get("lon_max")
        south = bounds.get("lat_min")
        north = bounds.get("lat_max")
    elif isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        west, south, east, north = bounds
    else:
        return None
    try:
        west, south, east, north = float(west), float(south), float(east), float(north)
    except (TypeError, ValueError):
        return None
    dy = abs(north - south) * 110_540.0
    dx = abs(east - west) * 111_320.0 * math.cos(math.radians((south + north) / 2.0))
    return {"width_m": round(dx, 0), "height_m": round(dy, 0)}


def build_micro_area_block(analysis: Dict) -> Dict:
    """Build the micro-area context block from a real analysis result."""
    satellite = analysis.get("satellite") or {}
    terrain = analysis.get("terrain") or {}
    landcover = analysis.get("landcover") or {}

    ndmi_stats = None
    extent = None
    scene_available = "error" not in satellite and satellite.get("ndmi") is not None
    if scene_available:
        ndmi_stats = _grid_stats(satellite.get("ndmi_grid"))
        extent = _extent_m(satellite.get("grid_bounds"))

    # Name the sensor that actually delivered the scene — Sentinel-2 (10 m)
    # primary, Landsat C2 L2 (30 m) fallback — never a fixed label.
    sat_source = str(satellite.get("source") or "")
    sensor = "Landsat C2 L2" if "Landsat" in sat_source else "Sentinel-2"
    try:
        scene_res = float(satellite.get("resolution_m") or 10.0)
    except (TypeError, ValueError):
        scene_res = 10.0
    res_label = f"{scene_res:g} m"

    variability_note = None
    if ndmi_stats is not None:
        if ndmi_stats["range"] >= 0.3:
            variability_note = (
                f"High within-scene NDMI variability ({ndmi_stats['range']:.2f} over "
                f"{ndmi_stats['cells']} measured cells): vegetation moisture is "
                f"heterogeneous at {res_label} scale — micro-area differences are real "
                "and locally relevant."
            )
        else:
            variability_note = (
                f"Within-scene NDMI variability is low ({ndmi_stats['range']:.2f}): "
                f"vegetation moisture is fairly uniform at {res_label} scale."
            )

    return {
        "status": "ok",
        "regional_context": {
            "scope": "regional (~11 km)",
            "layers": ["weather", "FWI fire danger", "soil moisture"],
            "note": _COARSE_NOTE,
        },
        "local_context": {
            "scope": "local (25-90 m)",
            "layers": ["terrain slope/aspect (DEM)"],
            "resolution": terrain.get("resolution") if "error" not in terrain else None,
        },
        "micro_context": {
            "scope": f"micro ({res_label})",
            "layers": [f"{sensor} NDVI/NDMI", "ESA WorldCover"],
            "scene_available": scene_available,
            "scene_extent_m": extent,
            "ndmi_variability": ndmi_stats,
            "variability_note": variability_note,
            "unavailable_note": (
                None if scene_available else
                "No recent cloud-free Sentinel-2 or Landsat scene — micro-area "
                "vegetation information is unavailable, not estimated."
            ),
        },
        "land_cover_resolution": (
            landcover.get("resolution") if "error" not in landcover else None
        ),
        "resolution_table": _RESOLUTION_TABLE,
        "provenance": {
            "kind": "derived",
            "source": "Measured Sentinel-2 scene grid + declared layer resolutions",
            "limitations": _COARSE_NOTE,
        },
    }
