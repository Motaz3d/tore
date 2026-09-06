"""
Landsat Collection 2 Level-2 data access for Talaix — REAL data.

Queries the Microsoft Planetary Computer public STAC catalog (USGS/NASA
Landsat Collection 2 Level-2 surface reflectance) and reads the actual
spectral bands (red B4, NIR B5, green B3, SWIR1 B6) plus QA_PIXEL for
cloud/cirrus/shadow/snow/fill masking, so NDVI/NDMI/NDWI can be computed at
30 m when Sentinel-2 has no usable scene.

Why Planetary Computer and not the Element84 Earth Search catalog used for
Sentinel-2: Earth Search indexes the same USGS scenes (collection
``landsat-c2-l2``), but its asset hrefs point at the requester-pays
``usgs-landsat`` S3 bucket — anonymous reads are refused (verified
2026-09-03: HTTP 403 on unsigned range GETs). Planetary Computer serves the
same Collection 2 Level-2 products from Azure Blob Storage and issues free
short-lived SAS read tokens through an unauthenticated endpoint, so the
windowed COG range reads used by :mod:`copernicus_data` work without any
credentials (verified end-to-end 2026-09-03: STAC search → SAS sign →
HTTP 206 range read).

Same honesty discipline as :mod:`copernicus_data`: if no usable cloud-free
scene exists for the requested period the methods return ``None`` / an
empty dict and the caller reports the layer unavailable. Nothing in this
module is simulated.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from .copernicus_data import (
    _FALLBACK_OFFSET,
    _FALLBACK_SCALE,
    _HAS_STAC,
    _WINDOW_M,
    CopernicusDataAccess,
    SatelliteObservation,
)

#: Planetary Computer STAC API + the USGS Landsat collection mirrored there.
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT_COLLECTION = "landsat-c2-l2"

#: Unauthenticated SAS signing endpoint (GET ?href=... → {"href": signed}).
PC_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"

#: QA_PIXEL bits that make a pixel unusable for vegetation/moisture analysis:
#: bit 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow.
#: (Water, bit 6, stays valid — same treatment as the Sentinel-2 SCL mask,
#: where the water class is not excluded.)
_QA_INVALID_BITS = 0b00111111

#: Landsat C2 L2 surface-reflectance conversion fallbacks (overridden by the
#: per-asset raster:bands fields, which PC items carry — verified 2026-09-03).
_LANDSAT_SCALE = 2.75e-05
_LANDSAT_OFFSET = -0.2

#: Asset key for each analysis band (identical names on PC and Earth Search).
_BAND_ASSETS = {"B04": "red", "B08": "nir08", "B03": "green", "B11": "swir16"}
_QA_ASSET = "qa_pixel"


class LandsatDataAccess(CopernicusDataAccess):
    """
    Access to real Landsat Collection 2 Level-2 data via public STAC.

    Reuses the windowed COG reads, index computation and caching discipline
    of :class:`CopernicusDataAccess`; only the catalog, the band mapping and
    the cloud mask differ (QA_PIXEL bit flags instead of the SCL classes).
    """

    def __init__(
        self,
        stac_url: str = PC_STAC_URL,
        collection: str = LANDSAT_COLLECTION,
        sign_url: str = PC_SIGN_URL,
    ):
        super().__init__(stac_url=stac_url, collection=collection)
        self.sign_url = sign_url

    # ------------------------------------------------------------------
    # SAS signing (PC serves unsigned hrefs that 403 without a token)
    # ------------------------------------------------------------------
    def _sign_href(self, href: str) -> Optional[str]:
        """Return a short-lived SAS-signed URL for a PC asset href."""
        try:
            url = self.sign_url + "?" + urllib.parse.urlencode({"href": href})
            req = urllib.request.Request(
                url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("href")
        except Exception:
            return None

    def _sign_item_assets(self, item) -> None:
        """Rewrite the asset hrefs we read in place with SAS-signed URLs.

        Raises if any required asset is missing or cannot be signed — the
        caller then skips the scene (an unreadable band must never degrade
        to a silently partial index).
        """
        for key in list(_BAND_ASSETS.values()) + [_QA_ASSET]:
            asset = item.assets.get(key)
            if asset is None:
                raise KeyError(f"landsat item {item.id}: missing asset '{key}'")
            signed = self._sign_href(asset.href)
            if not signed:
                raise RuntimeError(
                    f"landsat item {item.id}: SAS signing failed for '{key}'")
            asset.href = signed

    # ------------------------------------------------------------------
    # Band reads — Landsat band mapping + QA_PIXEL mask
    # ------------------------------------------------------------------
    def fetch_landsat_bands(
        self,
        lat: float,
        lon: float,
        date_range: Tuple[str, str],
        max_cloud_cover: float = 40.0,
    ) -> Dict[str, np.ndarray]:
        """
        Fetch REAL Landsat C2 L2 bands (red/NIR/green/SWIR1 + QA_PIXEL).

        Returns the same dict shape as ``fetch_sentinel2_bands`` (B04/B08/
        B03/B11 reflectance + an SCL-shaped validity layer) so the shared
        index computation applies unchanged. Returns {} when no usable
        scene is available.
        """
        if not _HAS_STAC:
            return {}

        products = self.search_sentinel2_products(
            lat, lon, date_range[0], date_range[1], max_cloud_cover
        )
        if not products:
            return {}

        for product in products[:4]:
            item = product["_item"]
            try:
                self._sign_item_assets(item)
                red, red_meta = self._read_window(
                    item.assets["red"], lon, lat, _WINDOW_M)
                nir, _ = self._read_window(
                    item.assets["nir08"], lon, lat, _WINDOW_M)
                green, _ = self._read_window(
                    item.assets["green"], lon, lat, _WINDOW_M)
                swir, _ = self._read_window(
                    item.assets["swir16"], lon, lat, _WINDOW_M)
                qa, _ = self._read_window(
                    item.assets["qa_pixel"], lon, lat, _WINDOW_M)
            except Exception:
                continue

            if red.size == 0 or nir.size == 0:
                continue

            # _read_window falls back to the Sentinel-2 scale/offset when an
            # asset carries no raster:bands; detect that fallback and apply
            # the Landsat C2 L2 conversion instead (PC items do carry
            # raster:bands — verified 2026-09-03 — so this is a guard only).
            scale = red_meta["scale"]
            offset = red_meta["offset"]
            if scale == _FALLBACK_SCALE and offset == _FALLBACK_OFFSET:
                scale, offset = _LANDSAT_SCALE, _LANDSAT_OFFSET
            nodata = red_meta["nodata"]

            def to_reflectance(arr: np.ndarray) -> np.ndarray:
                arr = np.where(arr == nodata, np.nan, arr)
                return arr * scale + offset

            # QA_PIXEL bit mask → SCL-shaped layer (4 = valid vegetation,
            # 8 = cloud) so CopernicusDataAccess.compute_indices_from_bands
            # applies its _SCL_INVALID filter unchanged.
            valid = (qa.astype(np.uint32) & _QA_INVALID_BITS) == 0
            if float(np.mean(valid)) < 0.05:
                # The window sits in this scene's fill/edge region (Landsat
                # L2SP products are partially filled) — try the next scene
                # instead of returning an empty observation.
                continue
            scl_like = np.where(valid, 4, 8).astype("float64")

            return {
                "B04": to_reflectance(red),
                "B08": to_reflectance(nir),
                "B03": to_reflectance(green),
                "B11": to_reflectance(swir),
                "SCL": scl_like,
                "metadata": {
                    "acquisition_date": product["date"],
                    "cloud_cover": product["cloud_cover"],
                    "product_id": product["id"],
                    "product_title": product["title"],
                    "resolution_m": red_meta["resolution_m"],
                    "bounds": red_meta["bounds"],
                    "source": "Landsat Collection 2 Level-2 "
                              "(Planetary Computer STAC, real)",
                },
            }
        return {}

    # ------------------------------------------------------------------
    # Main entry point — Landsat cache namespace, same payload shape
    # ------------------------------------------------------------------
    def get_latest_observation(
        self,
        lat: float,
        lon: float,
        days_back: int = 30,
        max_cloud_cover: float = 40.0,
    ) -> Optional[SatelliteObservation]:
        """
        Get the latest REAL Landsat observation for a location.

        Returns None when the STAC service is unreachable, no scene exists,
        or all scenes are too cloudy — callers report the layer unavailable.
        Cached separately from the Sentinel-2 cache namespace.
        """
        from ..dashboard.cache import default_cache, TTL_SATELLITE

        cache = default_cache()
        key = cache.make_key(
            "satellite_obs_landsat",
            round(lat, 4), round(lon, 4), days_back, max_cloud_cover,
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

        bands = self.fetch_landsat_bands(lat, lon, (start_date, end_date), max_cloud_cover)
        if not bands:
            return {"ok": False}

        indices = self.compute_indices_from_bands(bands)
        if not indices:
            return {"ok": False}

        valid = indices["valid_mask"]
        valid_fraction = float(np.mean(valid)) if valid.size else 0.0
        if valid_fraction < 0.05:
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
