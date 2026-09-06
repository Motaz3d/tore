"""
NOAA IBTrACS best-track archive — prepared-dataset loader for the cyclone
module's historical tracks.

The platform keeps a local copy of the IBTrACS ``last3years`` CSV
(~10 MB, NOAA NCEI, free, no key) under ``data/ibtracs/`` — downloaded
once, refreshed when older than 30 days. The file covers the most recent
three cyclone seasons worldwide; queries outside that window answer with
an explicit coverage note, never invented tracks.

Columns used (v04r01 CSV): SID, SEASON, BASIN, NAME, ISO_TIME, LAT, LON,
USA_WIND (kts), USA_PRES (mb), USA_SSHS. The second CSV row is a units
row and is skipped. Agency-specific columns are taken from the US agency
columns (USA_*) — no cross-agency merging is performed.
"""

from __future__ import annotations

import csv
import io
import os
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional

IBTRACS_URL = ("https://www.ncei.noaa.gov/data/"
               "international-best-track-archive-for-climate-stewardship-"
               "ibtracs/v04r01/access/csv/ibtracs.last3years.list.v04r01.csv")
IBTRACS_SOURCE = "NOAA NCEI — International Best Track Archive (IBTrACS v04r01)"

_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "ibtracs")
_LOCAL_FILE = os.path.join(_DATA_DIR, "ibtracs.last3years.list.v04r01.csv")
_MAX_AGE_S = 30 * 24 * 3600.0      # refresh monthly
_MAX_BYTES = 40 * 1024 * 1024      # hard guard (~10 MB expected)

_lock = threading.Lock()
_tracks_cache: Optional[Dict[str, Any]] = None


def _download(dest: str) -> Optional[str]:
    """Fetch the CSV to ``dest`` (bounded, atomic-ish via .part rename)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(
            IBTRACS_URL, headers={"User-Agent": "Talaix/1.0 (climate intelligence)"})
        with urllib.request.urlopen(req, timeout=120.0) as resp, open(tmp, "wb") as fh:
            remaining = _MAX_BYTES
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                remaining -= len(chunk)
                if remaining < 0:
                    raise RuntimeError("IBTrACS file exceeds the size guard")
                fh.write(chunk)
        os.replace(tmp, dest)
        return None
    except Exception as exc:  # honest: caller reports unavailable
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return str(exc)


def ensure_local_file() -> Dict[str, Any]:
    """Ensure the local CSV exists and is fresh; status dict (never raises).

    Locking lives in :func:`load_tracks` (the only caller) — this helper
    must NOT take ``_lock`` itself (a plain Lock is not re-entrant).
    """
    if os.path.exists(_LOCAL_FILE):
        age = time.time() - os.path.getmtime(_LOCAL_FILE)
        if age < _MAX_AGE_S:
            return {"ok": True, "path": _LOCAL_FILE, "refreshed": False}
    err = _download(_LOCAL_FILE)
    if err and not os.path.exists(_LOCAL_FILE):
        return {"ok": False, "reason": f"IBTrACS download failed: {err}"}
    return {"ok": True, "path": _LOCAL_FILE, "refreshed": err is None,
            "stale": err is not None}


def _parse_tracks(path: str) -> Dict[str, Any]:
    """Parse the CSV into per-storm track summaries (full points retained)."""
    storms: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("SID") == "" or row.get("SEASON", "").startswith("Year"):
                continue  # units row / blank rows
            try:
                season = int(row["SEASON"])
                lat = float(row["LAT"])
                lon = float(row["LON"])
            except (TypeError, ValueError):
                continue

            def _num(v: Any) -> Optional[float]:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None

            point = {
                "time": (row.get("ISO_TIME") or "")[:16],
                "lat": lat,
                "lon": lon,
                "wind_kt": _num(row.get("USA_WIND")),
                "pres_mb": _num(row.get("USA_PRES")),
                "sshs": _num(row.get("USA_SSHS")),
            }
            sid = row["SID"]
            storm = storms.setdefault(sid, {
                "sid": sid,
                "name": (row.get("NAME") or "").strip() or sid,
                "season": season,
                "basin": (row.get("BASIN") or "").strip(),
                "points": [],
            })
            storm["points"].append(point)

    for storm in storms.values():
        pts = storm["points"]
        winds = [p["wind_kt"] for p in pts if p["wind_kt"] is not None]
        press = [p["pres_mb"] for p in pts if p["pres_mb"] is not None]
        sshs = [p["sshs"] for p in pts if p["sshs"] is not None]
        storm["max_wind_kt"] = max(winds) if winds else None
        storm["min_pres_mb"] = min(press) if press else None
        storm["peak_sshs"] = max(sshs) if sshs else None
        storm["start"] = pts[0]["time"] if pts else None
        storm["end"] = pts[-1]["time"] if pts else None

    seasons = sorted({s["season"] for s in storms.values()})
    return {"storms": storms, "seasons": seasons}


def load_tracks() -> Dict[str, Any]:
    """Load (and cache in-process) the track archive; honest error dict."""
    global _tracks_cache
    with _lock:
        if _tracks_cache is not None:
            return _tracks_cache
        state = ensure_local_file()
        if not state.get("ok"):
            return {"error": state.get("reason", "IBTrACS file unavailable")}
        try:
            parsed = _parse_tracks(state["path"])
        except Exception as exc:
            return {"error": f"IBTrACS parse failed: {exc}"}
        parsed["source"] = IBTRACS_SOURCE
        parsed["file"] = state["path"]
        parsed["file_mtime"] = os.path.getmtime(state["path"])
        _tracks_cache = parsed
        return parsed


def reset_for_tests() -> None:
    global _tracks_cache
    with _lock:
        _tracks_cache = None


def _decimate(points: List[Dict[str, Any]], max_points: int = 40) -> List[Dict[str, Any]]:
    """Evenly thin a track for API payloads (first/last always kept)."""
    if len(points) <= max_points:
        return points
    step = (len(points) - 1) / (max_points - 1)
    return [points[round(i * step)] for i in range(max_points)]


def storms_near(
    lat: float, lon: float, year: Optional[int] = None,
    radius_km: float = 500.0, max_storms: int = 25,
) -> Dict[str, Any]:
    """IBTrACS storms with any track point within ``radius_km`` of a point.

    ``year`` filters by SEASON. Distance reported is the track's closest
    approach to the point (great-circle over all track points).
    """
    from .hazards._gdacs import haversine_km

    data = load_tracks()
    if "error" in data:
        return {"error": data["error"]}
    seasons = data["seasons"]
    if year is not None and year not in seasons:
        return {
            "error": f"Season {year} is outside the prepared IBTrACS file "
                     f"(seasons {seasons[0]}–{seasons[-1]}).",
            "coverage": {"seasons": seasons, "source": data["source"]},
        }

    out: List[Dict[str, Any]] = []
    for storm in data["storms"].values():
        if year is not None and storm["season"] != year:
            continue
        best = None
        for p in storm["points"]:
            d = haversine_km(lat, lon, p["lat"], p["lon"])
            if best is None or d < best[0]:
                best = (d, p)
        if best is None or best[0] > radius_km:
            continue
        closest_d, closest_p = best
        out.append({
            "sid": storm["sid"],
            "name": storm["name"],
            "season": storm["season"],
            "basin": storm["basin"],
            "start": storm["start"],
            "end": storm["end"],
            "max_wind_kt": storm["max_wind_kt"],
            "min_pres_mb": storm["min_pres_mb"],
            "peak_sshs": storm["peak_sshs"],
            "closest_approach_km": round(closest_d, 1),
            "closest_point": closest_p,
            "track": _decimate(storm["points"]),
        })
    out.sort(key=lambda s: s["closest_approach_km"])
    return {
        "status": "ok",
        "storms": out[:max_storms],
        "total_matching": len(out),
        "coverage": {
            "seasons": seasons,
            "file": "IBTrACS last3years (prepared local copy)",
            "source": data["source"],
        },
    }
