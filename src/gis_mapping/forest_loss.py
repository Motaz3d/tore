"""
Hansen/UMD Global Forest Change (GFC) 2023 v1.11 tree-cover-loss screening.

Reads the public GFC tiles from Google Cloud Storage (no credentials required)
and reports tree-cover status and loss history for a ~1 km window around a
point. The layer is used for EUDR-style deforestation screening; it is
explicitly not a legal determination.

Data: Hansen/UMD/Google/USGS/NASA — https://earthenginepartners.appspot.com/science-2013-global-forest
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..dashboard.cache import TTL_LANDCOVER, cached

try:
    import rasterio
    from rasterio.windows import Window

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False

_BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11"
_PRODUCT = "Hansen/UMD Global Forest Change 2023 v1.11 (GFC)"
_PROVIDER_URL = "https://earthenginepartners.appspot.com/science-2013-global-forest"

# Lossyear band values: 0 = no loss, 1–23 = loss in 2001–2023.
_VINTAGE_FIRST_YEAR = 2000
_VINTAGE_LAST_YEAR = 2023
_CUTOFF_YEAR_CODE = 21  # loss after 2020-12-31 ⇔ 2021, 2022, 2023
_CANOPY_THRESHOLD_PCT = 30  # Global Forest Watch forest-definition threshold


def _tile_tag(lat: float, lon: float) -> str:
    """Return the Hansen GFC 10°x10° tile tag for a point."""
    lat0 = int(math.floor(lat / 10.0) * 10 + 10)
    lon0 = int(math.floor(lon / 10.0) * 10)
    lat_tag = f"{abs(lat0):02d}{'N' if lat0 >= 0 else 'S'}"
    lon_tag = f"{abs(lon0):03d}{'E' if lon0 >= 0 else 'W'}"
    return f"{lat_tag}_{lon_tag}"


def _layer_url(layer: str, tag: str) -> str:
    return f"{_BASE_URL}/Hansen_GFC-2023-v1.11_{layer}_{tag}.tif"


def _read_layer(lat: float, lon: float, layer: str, half_px: int) -> np.ndarray:
    """Windowed rasterio read of one GFC layer. Raises on failure."""
    tag = _tile_tag(lat, lon)
    url = _layer_url(layer, tag)
    with rasterio.open(url) as ds:
        row, col = ds.index(lon, lat)
        win = Window(col - half_px, row - half_px, 2 * half_px, 2 * half_px)
        win = win.intersection(Window(0, 0, ds.width, ds.height))
        return ds.read(1, window=win)


def _loss_code_to_year(code: int) -> int:
    return _VINTAGE_FIRST_YEAR + int(code)


def _round1(value: Optional[float]) -> Optional[float]:
    return round(value, 1) if value is not None else None


@cached("forest_loss", TTL_LANDCOVER)
def fetch_forest_loss(lat: float, lon: float, window_m: float = 500.0) -> Dict[str, Any]:
    """
    Screen a point for tree cover and tree-cover loss using Hansen/UMD GFC.

    Returns an honest dict with mean 2000 canopy cover, forested fraction,
    loss history, post-2020 loss flag, and declared limitations. On failure it
    returns a dict with ``error`` and ``source`` keys.
    """
    if not _HAS_RASTERIO:
        return {"error": "rasterio not installed", "source": _PRODUCT}
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {"error": "Coordinates out of range"}

    half_px = int(round(window_m / 30.0))  # ~30 m pixels
    if half_px < 1:
        half_px = 1

    try:
        treecover = _read_layer(lat, lon, "treecover2000", half_px)
    except Exception as exc:
        return {"error": f"GFC treecover2000 read failed: {exc}", "source": _PRODUCT}

    try:
        lossyear = _read_layer(lat, lon, "lossyear", half_px)
    except Exception as exc:
        return {"error": f"GFC lossyear read failed: {exc}", "source": _PRODUCT}

    if treecover.size == 0 or lossyear.size == 0:
        return {"error": "GFC returned no data for this location", "source": _PRODUCT}

    total = int(treecover.size)

    # 2000 canopy cover
    tree_cover_mean = float(np.mean(treecover))
    forested_count = int(np.sum(treecover >= _CANOPY_THRESHOLD_PCT))
    forested_fraction = round(forested_count / total, 4)

    # Loss history
    loss_mask = lossyear > 0
    loss_detected = bool(np.any(loss_mask))
    loss_pixel_count = int(np.sum(loss_mask))
    loss_pixel_fraction = round(loss_pixel_count / total, 4)

    loss_years: Dict[int, int] = {}
    if loss_detected:
        codes, counts = np.unique(lossyear[loss_mask], return_counts=True)
        for code, count in zip(codes.tolist(), counts.tolist()):
            year = _loss_code_to_year(code)
            loss_years[year] = int(count)

    latest_loss_year: Optional[int] = None
    if loss_years:
        latest_loss_year = max(loss_years.keys())

    post_cutoff_mask = lossyear >= _CUTOFF_YEAR_CODE
    loss_after_2020 = bool(np.any(post_cutoff_mask))
    post_cutoff_count = int(np.sum(post_cutoff_mask))
    loss_after_2020_pixel_fraction = round(post_cutoff_count / total, 4)

    limitations = [
        "30 m spatial resolution: misses small clearings and narrow degradation.",
        "Canopy threshold 30%: pixels below this are not counted as forested.",
        "Window fraction is a screening statistic, not a legal plot audit.",
    ]

    return {
        "tree_cover_2000_mean_pct": _round1(tree_cover_mean),
        "forested_fraction_2000": forested_fraction,
        "loss_detected": loss_detected,
        "loss_pixel_fraction": loss_pixel_fraction,
        "loss_years": loss_years,
        "latest_loss_year": latest_loss_year,
        "loss_after_2020": loss_after_2020,
        "loss_after_2020_pixel_fraction": loss_after_2020_pixel_fraction,
        "window_note": f"screened ~{int(round(window_m * 2))} m box centred on the point",
        "eudr_cutoff_date": "2020-12-31",
        "source": _PRODUCT,
        "dataset": _PRODUCT,
        "provider_url": _PROVIDER_URL,
        "resolution": "30 m",
        "vintage_note": (
            f"Covers tree-cover loss through {_VINTAGE_LAST_YEAR}; "
            f"losses in {_VINTAGE_LAST_YEAR + 1}+ are not included — declared limitation."
        ),
        "limitations": limitations,
    }
