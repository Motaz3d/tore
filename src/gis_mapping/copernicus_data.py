"""
Copernicus / Sentinel-2 data access for Talaix — REAL data.

Queries the Element84 Earth Search public STAC catalog (Sentinel-2 Level-2A
surface reflectance, hosted on AWS Open Data, sponsored by the Copernicus
programme) and reads the actual spectral bands (B04 red, B08 NIR, B03 green,
B11 SWIR) plus the Scene Classification Layer (SCL) for cloud masking.

    - No credentials are required (public STAC + public COG bucket).
    - Bands are read as small windows around the requested point via
      range-request COG reads (rasterio), so no full scenes are downloaded.
    - Results are cached (see ``dashboard.cache``).

If no usable cloud-free scene exists for the requested period, the methods
return ``None`` / an empty dict and the caller reports the layer as
unavailable. Nothing in this module is simulated.

Official alternative (when an account exists): the Copernicus Data Space
Ecosystem STAC/OData API (free registration). The interface of this module is
kept compatible so a CDS-backed implementation can be swapped in later.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

# Defensive global timeout so a stalled upstream socket can never hang a
# web worker forever (rasterio/urllib honour this default).
socket.setdefaulttimeout(60.0)

# GDAL/rasterio HTTP tuning for COG range reads.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

from .indices import compute_ndvi, compute_ndmi, compute_ndwi

try:  # Optional at import time; required for real data access.
    import rasterio
    from rasterio.windows import Window
    from pyproj import Transformer
    from pystac_client import Client

    _HAS_STAC = True
except ImportError:  # pragma: no cover - depends on deployment extras
    _HAS_STAC = False


STAC_API_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Reflectance conversion fallbacks (overridden by per-asset raster:bands).
_FALLBACK_SCALE = 0.0001
_FALLBACK_OFFSET = -0.1
_NODATA = 0

# SCL classes considered unusable for vegetation/moisture analysis.
_SCL_INVALID = {0, 1, 3, 8, 9, 10, 11}  # no data, defective, shadow, clouds, cirrus, snow

# Analysis window: +/-600 m around the point (120 px at 10 m).
_WINDOW_M = 600.0
# Downsampled overlay grid dimension for map display.
_GRID_N = 24


@dataclass
class SatelliteObservation:
    """Container for a single real satellite observation."""

    latitude: float
    longitude: float
    timestamp: datetime
    ndvi: Optional[float] = None
    ndmi: Optional[float] = None
    ndwi: Optional[float] = None
    cloud_cover_pct: Optional[float] = None
    source: str = "Sentinel-2 L2A (Earth Search STAC)"
    processing_level: str = "Level-2A"
    product_id: Optional[str] = None
    resolution_m: Optional[float] = 10.0
    valid_pixel_fraction: Optional[float] = None
    # Coarse per-cell grids for map overlays (row-major, north to south).
    ndvi_grid: Optional[List[List[Optional[float]]]] = None
    ndmi_grid: Optional[List[List[Optional[float]]]] = None
    grid_bounds: Optional[Dict[str, float]] = None  # lat_min/lat_max/lon_min/lon_max


class CopernicusDataAccess:
    """
    Access to real Sentinel-2 Earth Observation data via public STAC.

    The public method names are unchanged from earlier versions of this
    module so downstream code (`real_data`, `real_analysis`, tests) keeps
    working; the implementation is now backed by real satellite scenes.
    """

    def __init__(self, stac_url: str = STAC_API_URL, collection: str = COLLECTION):
        self.stac_url = stac_url
        self.collection = collection
        self.user_agent = "Talaix/1.0 (Earth Observation Decision Support)"
        if not _HAS_STAC:
            # Defer the error to call sites so the module still imports in
            # minimal environments (e.g. unit tests without raster extras).
            self._catalog = None
        else:
            self._catalog = None  # opened lazily per call (thread-safety)

    # ------------------------------------------------------------------
    # STAC search
    # ------------------------------------------------------------------
    def search_sentinel2_products(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 100.0,
    ) -> List[Dict]:
        """
        Search the STAC catalog for real Sentinel-2 L2A scenes covering a
        point within a date range, ordered by ascending cloud cover.
        """
        if not _HAS_STAC:
            return []
        try:
            catalog = Client.open(self.stac_url)
            search = catalog.search(
                collections=[self.collection],
                intersects={"type": "Point", "coordinates": [lon, lat]},
                datetime=f"{start_date}/{end_date}",
                query={"eo:cloud_cover": {"lt": max_cloud_cover}},
                max_items=20,
            )
            items = list(search.items())
        except Exception:
            return []

        products = []
        for it in items:
            products.append(
                {
                    "id": it.id,
                    "title": it.id,
                    "date": (it.properties.get("datetime") or "")[:10],
                    "cloud_cover": float(it.properties.get("eo:cloud_cover") or 100.0),
                    "link": it.get_self_href(),
                    "constellation": it.properties.get("constellation", "sentinel-2"),
                    "_item": it,  # internal handle for band reads
                }
            )
        products.sort(key=lambda p: p["cloud_cover"])
        return products

    # ------------------------------------------------------------------
    # Band reads (windowed COG range requests)
    # ------------------------------------------------------------------
    @staticmethod
    def _read_window(asset, lon: float, lat: float, window_m: float) -> Tuple[np.ndarray, dict]:
        """Read a window of a COG asset around a point. Returns (array, meta)."""
        with rasterio.open(asset.href) as ds:
            transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            cx, cy = transformer.transform(lon, lat)
            col, row = ds.index(cx, cy)
            res_x = abs(ds.res[0])
            half_px = int(round(window_m / res_x))
            win = Window(col - half_px, row - half_px, 2 * half_px, 2 * half_px)
            # Clamp the window to the raster.
            win = win.intersection(Window(0, 0, ds.width, ds.height))
            arr = ds.read(1, window=win).astype("float64")

            rb = (asset.extra_fields.get("raster:bands") or [{}])[0]
            scale = float(rb.get("scale", _FALLBACK_SCALE))
            offset = float(rb.get("offset", _FALLBACK_OFFSET))
            nodata = rb.get("nodata", _NODATA)

            # Geographic bounds of the window (for map overlays).
            inv = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
            x_min, y_min = ds.transform * (win.col_off, win.row_off + win.height)
            x_max, y_max = ds.transform * (win.col_off + win.width, win.row_off)
            lon_min, lat_min = inv.transform(x_min, y_min)
            lon_max, lat_max = inv.transform(x_max, y_max)

        meta = {
            "scale": scale,
            "offset": offset,
            "nodata": nodata,
            "resolution_m": res_x,
            "bounds": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
        }
        return arr, meta

    def fetch_sentinel2_bands(
        self,
        lat: float,
        lon: float,
        date_range: Tuple[str, str],
        max_cloud_cover: float = 40.0,
    ) -> Dict[str, np.ndarray]:
        """
        Fetch REAL Sentinel-2 bands (B03/B04/B08/B11 + SCL) for a point.

        Returns reflectance arrays (scale/offset applied) for a ~1.2 km box
        around the point, plus acquisition metadata. Returns {} when no
        usable scene is available.
        """
        if not _HAS_STAC:
            return {}

        products = self.search_sentinel2_products(
            lat, lon, date_range[0], date_range[1], max_cloud_cover
        )
        if not products:
            return {}

        # Try scenes in ascending cloud-cover order until one yields data.
        for product in products[:4]:
            item = product["_item"]
            try:
                red, red_meta = self._read_window(item.assets["red"], lon, lat, _WINDOW_M)
                nir, _ = self._read_window(item.assets["nir"], lon, lat, _WINDOW_M)
                green, _ = self._read_window(item.assets["green"], lon, lat, _WINDOW_M)
                swir, _ = self._read_window(item.assets["swir16"], lon, lat, _WINDOW_M)
                scl, _ = self._read_window(item.assets["scl"], lon, lat, _WINDOW_M)
            except Exception:
                continue

            if red.size == 0 or nir.size == 0:
                continue

            scale, offset = red_meta["scale"], red_meta["offset"]

            def to_reflectance(arr: np.ndarray) -> np.ndarray:
                arr = np.where(arr == red_meta["nodata"], np.nan, arr)
                return arr * scale + offset

            return {
                "B04": to_reflectance(red),
                "B08": to_reflectance(nir),
                "B03": to_reflectance(green),
                "B11": to_reflectance(swir),
                "SCL": scl,  # classification codes, not reflectance
                "metadata": {
                    "acquisition_date": product["date"],
                    "cloud_cover": product["cloud_cover"],
                    "product_id": product["id"],
                    "product_title": product["title"],
                    "resolution_m": red_meta["resolution_m"],
                    "bounds": red_meta["bounds"],
                    "source": "Sentinel-2 Level-2A (Earth Search STAC, real)",
                },
            }
        return {}

    # ------------------------------------------------------------------
    # Indices
    # ------------------------------------------------------------------
    @staticmethod
    def _match_shapes(*arrays: np.ndarray) -> List[np.ndarray]:
        """Bring arrays to a common shape by nearest-neighbour upsampling."""
        target = max(arr.shape for arr in arrays)
        out = []
        for arr in arrays:
            if arr.shape == target:
                out.append(arr)
                continue
            fy = target[0] / arr.shape[0]
            fx = target[1] / arr.shape[1]
            idx_y = (np.arange(target[0]) / fy).astype(int).clip(0, arr.shape[0] - 1)
            idx_x = (np.arange(target[1]) / fx).astype(int).clip(0, arr.shape[1] - 1)
            out.append(arr[np.ix_(idx_y, idx_x)])
        return out

    def compute_indices_from_bands(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute NDVI/NDMI/NDWI from real bands, masked by the SCL layer."""
        required = ["B04", "B08", "B03", "B11", "SCL"]
        if not bands or not all(b in bands for b in required):
            return {}

        red, nir, green, swir, scl = self._match_shapes(
            bands["B04"], bands["B08"], bands["B03"], bands["B11"], bands["SCL"]
        )
        valid = ~np.isin(scl.astype(int), list(_SCL_INVALID))

        with np.errstate(invalid="ignore"):
            ndvi = np.where(valid, compute_ndvi(nir, red), np.nan)
            ndmi = np.where(valid, compute_ndmi(nir, swir), np.nan)
            ndwi = np.where(valid, compute_ndwi(green, nir), np.nan)

        return {"ndvi": ndvi, "ndmi": ndmi, "ndwi": ndwi, "valid_mask": valid}

    # ------------------------------------------------------------------
    # Overlay grid for maps
    # ------------------------------------------------------------------
    @staticmethod
    def _downsample_grid(arr: np.ndarray, n: int = _GRID_N) -> List[List[Optional[float]]]:
        """Block-mean an array to an n x n grid of floats (NaN -> None)."""
        h, w = arr.shape
        grid: List[List[Optional[float]]] = []
        for i in range(n):
            row: List[Optional[float]] = []
            y0, y1 = int(i * h / n), int((i + 1) * h / n)
            for j in range(n):
                x0, x1 = int(j * w / n), int((j + 1) * w / n)
                block = arr[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
                with np.errstate(invalid="ignore"):
                    val = np.nanmean(block)
                row.append(None if np.isnan(val) else round(float(val), 4))
            grid.append(row)
        return grid

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def get_latest_observation(
        self,
        lat: float,
        lon: float,
        days_back: int = 30,
        max_cloud_cover: float = 40.0,
    ) -> Optional[SatelliteObservation]:
        """
        Get the latest REAL Sentinel-2 observation for a location.

        Returns None when the STAC service is unreachable, no scene exists,
        or all scenes are too cloudy — callers report the layer unavailable.
        Results are cached for 12 h per location/period.
        """
        from ..dashboard.cache import default_cache, TTL_SATELLITE

        cache = default_cache()
        key = cache.make_key(
            "satellite_obs", round(lat, 4), round(lon, 4), days_back, max_cloud_cover
        )
        payload = cache.get(key)
        if payload is None:
            payload = self._fetch_observation(lat, lon, days_back, max_cloud_cover)
            cache.set(key, payload, TTL_SATELLITE if payload.get("ok") else 600.0)
        if not payload.get("ok"):
            return None
        return self._observation_from_payload(lat, lon, payload)

    def _fetch_observation(self, lat: float, lon: float, days_back: int, max_cloud_cover: float) -> Dict:
        """Fetch + compute; returns a JSON-serialisable payload for the cache."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        bands = self.fetch_sentinel2_bands(lat, lon, (start_date, end_date), max_cloud_cover)
        if not bands:
            return {"ok": False}

        indices = self.compute_indices_from_bands(bands)
        if not indices:
            return {"ok": False}

        valid = indices["valid_mask"]
        valid_fraction = float(np.mean(valid)) if valid.size else 0.0
        if valid_fraction < 0.05:
            # Effectively nothing usable in the window (cloud/shadow/water).
            return {"ok": False}

        with np.errstate(invalid="ignore"):
            ndvi_val = float(np.nanmean(indices["ndvi"]))
            ndmi_val = float(np.nanmean(indices["ndmi"]))
            ndwi_val = float(np.nanmean(indices["ndwi"]))

        if any(np.isnan(v) for v in (ndvi_val, ndmi_val, ndwi_val)):
            return {"ok": False}

        meta = bands["metadata"]
        return {
            "ok": True,
            "timestamp": meta["acquisition_date"],
            "ndvi": round(ndvi_val, 4),
            "ndmi": round(ndmi_val, 4),
            "ndwi": round(ndwi_val, 4),
            "cloud_cover_pct": round(float(meta["cloud_cover"]), 2),
            "valid_pixel_fraction": round(valid_fraction, 3),
            "product_id": meta["product_id"],
            "resolution_m": meta["resolution_m"],
            "source": meta["source"],
            "ndvi_grid": self._downsample_grid(indices["ndvi"]),
            "ndmi_grid": self._downsample_grid(indices["ndmi"]),
            "grid_bounds": meta["bounds"],
        }

    @staticmethod
    def _observation_from_payload(lat: float, lon: float, payload: Dict) -> SatelliteObservation:
        try:
            timestamp = datetime.strptime(payload["timestamp"], "%Y-%m-%d")
        except (KeyError, ValueError):
            timestamp = datetime.now()
        return SatelliteObservation(
            latitude=lat,
            longitude=lon,
            timestamp=timestamp,
            ndvi=payload.get("ndvi"),
            ndmi=payload.get("ndmi"),
            ndwi=payload.get("ndwi"),
            cloud_cover_pct=payload.get("cloud_cover_pct"),
            source=payload.get("source", "Sentinel-2 L2A (Earth Search STAC)"),
            product_id=payload.get("product_id"),
            resolution_m=payload.get("resolution_m"),
            valid_pixel_fraction=payload.get("valid_pixel_fraction"),
            ndvi_grid=payload.get("ndvi_grid"),
            ndmi_grid=payload.get("ndmi_grid"),
            grid_bounds=payload.get("grid_bounds"),
        )


# --------------------------------------------------------------------------
# FMC estimation helpers (single shared calibration)
# --------------------------------------------------------------------------

def _estimate_fmc_from_ndmi(ndmi_array: np.ndarray) -> np.ndarray:
    """
    Estimate Fuel Moisture Content from NDMI values.

    Single shared linear calibration (identical to
    ``FuelMoistureModel.estimate_fmc_from_ndmi``). NOTE: placeholder
    calibration — to be fitted against measured FMC data in Phase 6. Declared
    as "uncalibrated" in analysis provenance.
    """
    ndmi_array = np.asarray(ndmi_array, dtype=float)
    fmc = 120.0 * ndmi_array + 30.0
    return np.clip(fmc, 0.0, 100.0)


def _estimate_fmc_from_weather(soil_moisture_m3m3: float) -> float:
    """Estimate FMC from a soil moisture value via the capillary-transfer model."""
    sm = max(0.0, min(1.0, float(soil_moisture_m3m3)))
    rel_sat = sm / 0.45
    fmc = 100.0 * 0.35 * min(rel_sat, 1.0)
    return round(fmc, 2)


def fuse_satellite_weather_data(
    satellite_data: Dict[str, np.ndarray],
    weather_data: Dict,
) -> Dict:
    """
    Fuse satellite-derived FMC with weather-derived FMC (weighted average).

    Satellite NDMI carries spatial structure; the soil-moisture-based
    estimate carries temporal continuity. Weights are fixed and declared.
    """
    fused_data: Dict = {}
    if "ndmi" in satellite_data and weather_data.get("soil_moisture_m3m3") is not None:
        sat_fmc = _estimate_fmc_from_ndmi(satellite_data["ndmi"])
        weather_fmc = _estimate_fmc_from_weather(weather_data["soil_moisture_m3m3"])
        fused_fmc = 0.7 * sat_fmc + 0.3 * weather_fmc
        fused_data["estimated_fmc"] = fused_fmc
        fused_data["satellite_fmc"] = sat_fmc
        fused_data["weather_fmc"] = weather_fmc
    return fused_data
