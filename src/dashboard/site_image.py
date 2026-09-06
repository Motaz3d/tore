"""
Site-context image generator for PDF reports.

Builds a side-by-side PNG for a given lat/lon:
- left: ESA WorldCover 10m 2021 land-cover classes;
- right: Hansen/UMD GFC tree-cover loss 2001-2023, with brighter red for post-2020 loss.

Any failure (missing rasterio/PIL, network error, out-of-bounds read) returns None
so PDF builders can skip the image honestly.
"""

from __future__ import annotations

import io
import math
from typing import Dict, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import rasterio
    from rasterio.windows import Window

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

_WORLDCOVER_BUCKET = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
_GFC_BASE = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11"

# WorldCover class -> (R, G, B)
_WORLDCOVER_PALETTE: Dict[int, Tuple[int, int, int]] = {
    10: (30, 80, 30),      # Tree cover
    20: (180, 120, 60),    # Shrubland
    30: (220, 210, 80),    # Grassland
    40: (240, 180, 100),   # Cropland
    50: (200, 60, 60),     # Built-up
    60: (210, 190, 150),   # Bare / sparse vegetation
    70: (250, 250, 250),   # Snow / ice
    80: (50, 100, 200),    # Permanent water
    90: (60, 160, 150),    # Herbaceous wetland
    95: (20, 90, 80),      # Mangroves
    100: (180, 180, 160),  # Moss / lichen
}
_DEFAULT_LANDCOVER_COLOUR = (120, 120, 120)  # Unknown class

# GFC lossyear codes 21-23 are 2021-2023, i.e. post-2020.
_POST_2020_CODE = 21

_OUTPUT_WIDTH = 1000
_OUTPUT_HEIGHT = 460
_PANEL_WIDTH = 470
_PANEL_HEIGHT = 420
_PANEL_LEFT_ORIGIN = (15, 20)
_PANEL_RIGHT_ORIGIN = (515, 20)


def _worldcover_url(lat: float, lon: float) -> str:
    """Return the ESA WorldCover 3°x3° tile URL for a point."""
    lat0 = int(math.floor(lat / 3.0) * 3)
    lon0 = int(math.floor(lon / 3.0) * 3)
    lat_tag = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}"
    lon_tag = f"{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"
    return f"{_WORLDCOVER_BUCKET}/ESA_WorldCover_10m_2021_v200_{lat_tag}{lon_tag}_Map.tif"


def _gfc_tile_tag(lat: float, lon: float) -> str:
    """Return the Hansen GFC 10°x10° tile tag for a point."""
    lat0 = int(math.floor(lat / 10.0) * 10 + 10)
    lon0 = int(math.floor(lon / 10.0) * 10)
    lat_tag = f"{abs(lat0):02d}{'N' if lat0 >= 0 else 'S'}"
    lon_tag = f"{abs(lon0):03d}{'E' if lon0 >= 0 else 'W'}"
    return f"{lat_tag}_{lon_tag}"


def _gfc_url(layer: str, tag: str) -> str:
    return f"{_GFC_BASE}/Hansen_GFC-2023-v1.11_{layer}_{tag}.tif"


def _read_worldcover_window(lat: float, lon: float, window_m: float) -> Optional[np.ndarray]:
    """Read a windowed ESA WorldCover array; return None on failure."""
    if not _HAS_RASTERIO or np is None:
        return None
    try:
        from pyproj import Transformer
    except ImportError:
        return None

    url = _worldcover_url(lat, lon)
    try:
        with rasterio.Env(GDAL_HTTP_TIMEOUT=20), rasterio.open(url) as ds:
            transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            row, col = ds.index(x, y)
            half_px = int(round(window_m / 10.0))  # 10 m pixels
            win = Window(col - half_px, row - half_px, 2 * half_px, 2 * half_px)
            win = win.intersection(Window(0, 0, ds.width, ds.height))
            return ds.read(1, window=win)
    except Exception:
        return None


def _read_gfc_window(lat: float, lon: float, layer: str, window_m: float) -> Optional[np.ndarray]:
    """Read a windowed Hansen GFC layer; return None on failure."""
    if not _HAS_RASTERIO or np is None:
        return None
    tag = _gfc_tile_tag(lat, lon)
    url = _gfc_url(layer, tag)
    try:
        with rasterio.Env(GDAL_HTTP_TIMEOUT=20), rasterio.open(url) as ds:
            row, col = ds.index(lon, lat)
            half_px = int(round(window_m / 30.0))  # ~30 m pixels
            win = Window(col - half_px, row - half_px, 2 * half_px, 2 * half_px)
            win = win.intersection(Window(0, 0, ds.width, ds.height))
            return ds.read(1, window=win)
    except Exception:
        return None


def _landcover_rgb(arr: np.ndarray) -> np.ndarray:
    """Map a WorldCover class array to an RGB image."""
    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, colour in _WORLDCOVER_PALETTE.items():
        mask = arr == cls
        rgb[mask] = colour
    unknown = ~np.isin(arr, list(_WORLDCOVER_PALETTE.keys()))
    rgb[unknown] = _DEFAULT_LANDCOVER_COLOUR
    return rgb


def _loss_rgb(lossyear: np.ndarray) -> np.ndarray:
    """Map a GFC lossyear array to an RGB image."""
    h, w = lossyear.shape
    rgb = np.full((h, w, 3), 40, dtype=np.uint8)  # dark grey background
    loss_mask = lossyear > 0
    rgb[loss_mask] = [180, 60, 60]  # red for any loss
    post_mask = lossyear >= _POST_2020_CODE
    rgb[post_mask] = [255, 80, 80]  # brighter red for post-2020 loss
    return rgb


def _place_panel(canvas: np.ndarray, rgb: np.ndarray, origin: Tuple[int, int]) -> None:
    """Resize an RGB array with nearest-neighbour and paste it onto the canvas."""
    if np is None:
        return
    if rgb.size == 0:
        return
    img = Image.fromarray(rgb)
    img = img.resize((_PANEL_WIDTH, _PANEL_HEIGHT), Image.NEAREST)
    arr = np.array(img)
    x, y = origin
    canvas[y:y + _PANEL_HEIGHT, x:x + _PANEL_WIDTH] = arr


def build_site_context_png(lat: float, lon: float, window_m: float = 1000.0) -> Optional[bytes]:
    """
    Return a side-by-side site-context PNG, or None if the image cannot be built.

    The left panel shows ESA WorldCover 10m 2021 land-cover classes; the right
    panel shows Hansen/UMD GFC tree-cover loss 2001-2023, with brighter red for
    loss in 2021-2023.
    """
    if not (_HAS_RASTERIO and _HAS_PIL and np is not None):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    if window_m <= 0:
        return None

    landcover = _read_worldcover_window(lat, lon, window_m)
    lossyear = _read_gfc_window(lat, lon, "lossyear", window_m)
    if landcover is None or landcover.size == 0 or lossyear is None or lossyear.size == 0:
        return None

    canvas = np.full((_OUTPUT_HEIGHT, _OUTPUT_WIDTH, 3), 255, dtype=np.uint8)
    _place_panel(canvas, _landcover_rgb(landcover), _PANEL_LEFT_ORIGIN)
    _place_panel(canvas, _loss_rgb(lossyear), _PANEL_RIGHT_ORIGIN)

    try:
        img = Image.fromarray(canvas)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def site_context_caption() -> str:
    return (
        "Site context (~1 km): ESA WorldCover 10m 2021 land-cover classes (left); "
        "Hansen/UMD GFC tree-cover loss 2001–2023, red = post-2020 loss (right)."
    )
