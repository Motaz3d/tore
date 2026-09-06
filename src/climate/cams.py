"""
CAMS (Copernicus Atmosphere Monitoring Service) client — dust aerosol
optical depth from the Atmosphere Data Store (ADS) ``retrieve/v1`` API.

Key-gated like NASA FIRMS: the fetch path is fully wired, and it activates
when the operator configures ``CAMS_ADS_URL`` / ``CAMS_ADS_KEY`` (see
.env.example). Without credentials the module answers with an explicit
``key_required`` error — never a fabricated aerosol value.

API contract (verified 2026-09-03, unauthenticated probe):

- ``POST {ADS}/retrieve/v1/processes/{dataset}/execution`` with
  ``{"inputs": {...}}`` → job object (401 without a key — endpoint shape
  confirmed live; the old ``/api/v2/resources/`` endpoint is retired).
- ``GET {ADS}/retrieve/v1/jobs/{jobID}`` → status; on success the results
  carry a downloadable NetCDF asset URL.
- Auth: ``PRIVATE-TOKEN`` header (the cdsapi convention).

Reading: the returned NetCDF is opened with rasterio's netCDF driver
(present in the platform's GDAL build — verified 2026-09-03) and the
nearest grid cell to the analysis point is extracted per lead time.

Honesty note: the job choreography and point extraction are covered by
offline tests with mocked transport; the live ADS round-trip requires
credentials and is marked as such wherever the result surfaces.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import urllib.request
from datetime import date
from typing import Any, Dict, List, Optional

CAMS_DATASET = "cams-global-atmospheric-composition-forecasts"
CAMS_VARIABLE = "dust_aerosol_optical_depth_550nm"
CAMS_SOURCE = "CAMS — Copernicus Atmosphere Monitoring Service (ADS)"
_DEFAULT_ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
_TIMEOUT = 30.0
_JOB_WAIT_S = 180.0
_JOB_POLL_S = 5.0

# Declared screening bands for dust AOD at 550 nm (dimensionless) —
# screening labels, not a health or visibility assessment.
_AOD_BANDS = (
    (0.2, "Low dust load"),
    (0.5, "Moderate dust load"),
    (1.0, "High dust load"),
    (float("inf"), "Very high dust load"),
)


def credentials() -> Optional[Dict[str, str]]:
    url = os.environ.get("CAMS_ADS_URL") or _DEFAULT_ADS_URL
    key = os.environ.get("CAMS_ADS_KEY")
    if not key:
        return None
    return {"url": url.rstrip("/"), "key": key}


def key_required_error() -> Dict[str, str]:
    return {
        "error": "CAMS credentials not configured (CAMS_ADS_URL / CAMS_ADS_KEY)",
        "key_required": True,
        "signup": "https://ads.atmosphere.copernicus.eu/",
    }


def aod_band(aod: Optional[float]) -> Optional[str]:
    if aod is None:
        return None
    for limit, label in _AOD_BANDS:
        if aod < limit:
            return label
    return _AOD_BANDS[-1][1]


def _request(req_url: str, key: str, payload: Optional[Dict] = None) -> Dict:
    headers = {"PRIVATE-TOKEN": key, "Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(req_url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _find_asset_href(node: Any) -> Optional[str]:
    """Recursively find a downloadable asset href in a job result object."""
    if isinstance(node, dict):
        href = node.get("href")
        if isinstance(href, str) and href.startswith("http"):
            return href
        for value in node.values():
            found = _find_asset_href(value)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_asset_href(item)
            if found:
                return found
    return None


def _submit_job(creds: Dict[str, str], lat: float, lon: float,
                day: str) -> Dict[str, Any]:
    body = {
        "inputs": {
            "variable": [CAMS_VARIABLE],
            "date": [f"{day}/{day}"],
            "time": ["00:00"],
            "leadtime_hour": ["0", "24"],
            "type": ["forecast"],
            "area": [round(lat + 2, 2), round(lon - 2, 2),
                     round(lat - 2, 2), round(lon + 2, 2)],
            "format": "netcdf",
        }
    }
    return _request(
        f"{creds['url']}/retrieve/v1/processes/{CAMS_DATASET}/execution",
        creds["key"], body)


def _await_job(creds: Dict[str, str], job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = job.get("jobID") or job.get("id")
    if not job_id:
        return {"error": f"ADS returned no job id: {str(job)[:200]}"}
    deadline = time.time() + _JOB_WAIT_S
    while time.time() < deadline:
        status = _request(f"{creds['url']}/retrieve/v1/jobs/{job_id}", creds["key"])
        state = status.get("status")
        if state == "successful":
            href = _find_asset_href(status.get("results"))
            if not href:
                return {"error": "ADS job succeeded but no downloadable asset was found"}
            return {"href": href, "job_id": job_id}
        if state in ("failed", "dismissed", "deleted"):
            return {"error": f"ADS job {state}: {str(status)[:200]}"}
        time.sleep(_JOB_POLL_S)
    return {"error": f"ADS job did not finish within {_JOB_WAIT_S:.0f}s"}


def _download(href: str, key: str) -> bytes:
    req = urllib.request.Request(href, headers={"PRIVATE-TOKEN": key})
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        return resp.read()


def extract_aod_series(nc_bytes: bytes, lat: float, lon: float) -> Dict[str, Any]:
    """Nearest-grid-cell dust AOD per lead time from a CAMS NetCDF payload.

    Returns one value per band (lead time), each with its band label.
    """
    import numpy as np
    import rasterio
    from pyproj import Transformer

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(nc_bytes)
        tmp_path = tmp.name
    try:
        with rasterio.open(f"netcdf:{tmp_path}:{CAMS_VARIABLE}") as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            cx, cy = transformer.transform(lon, lat)
            col, row = src.index(cx, cy)
            row = min(max(row, 0), src.height - 1)
            col = min(max(col, 0), src.width - 1)
            values: List[Dict[str, Any]] = []
            for band in range(1, src.count + 1):
                arr = src.read(1 if src.count == 1 else band)
                v = float(arr[row, col])
                nodata = src.nodata
                values.append({
                    "band": band,
                    "description": src.descriptions[band - 1] if src.descriptions else None,
                    "aod": None if (nodata is not None and v == nodata) or np.isnan(v)
                           else round(v, 4),
                })
            return {
                "values": values,
                "grid": {"crs": str(src.crs), "resolution": src.res},
            }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def fetch_cams_dust_aod(lat: float, lon: float,
                        day: Optional[str] = None) -> Dict[str, Any]:
    """Dust AOD at 550 nm (analysis + 24 h lead) at a point via CAMS/ADS.

    ``key_required`` error without credentials; honest error dict on any
    upstream failure — never a fabricated value.
    """
    creds = credentials()
    if creds is None:
        return key_required_error()
    day = day or date.today().isoformat()
    try:
        job = _submit_job(creds, lat, lon, day)
    except Exception as exc:
        return {"error": f"CAMS ADS job submission failed: {exc}"}
    try:
        done = _await_job(creds, job)
    except Exception as exc:
        return {"error": f"CAMS ADS job polling failed: {exc}"}
    if "error" in done:
        return done
    try:
        nc = _download(done["href"], creds["key"])
        extracted = extract_aod_series(nc, lat, lon)
    except Exception as exc:
        return {"error": f"CAMS NetCDF extraction failed: {exc}"}

    values = extracted["values"]
    analysis = values[0]["aod"] if values else None
    lead24 = values[1]["aod"] if len(values) > 1 else None
    return {
        "status": "ok",
        "claim_status": "MODELLED",
        "dataset": CAMS_DATASET,
        "variable": CAMS_VARIABLE,
        "date": day,
        "leadtimes": values,
        "aod_analysis": analysis,
        "aod_lead24": lead24,
        "band_analysis": aod_band(analysis),
        "band_lead24": aod_band(lead24),
        "grid": extracted["grid"],
        "source": CAMS_SOURCE,
        "note": ("Modelled aerosol optical depth (CAMS global forecast), "
                 "nearest grid cell; screening labels declared in "
                 "src/climate/cams.py — not a health or visibility "
                 "assessment."),
    }
