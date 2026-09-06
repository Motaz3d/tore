"""
Train Talaix risk models on REAL wildfire history (Phase 6).

Data strategy (all real, all declared):
    - Positive samples: real active-fire detections from NASA FIRMS
      (VIIRS S-NPP, 375 m) over a bounding box and period. Requires the free
      FIRMS_MAP_KEY.
    - Negative samples: randomly drawn points/dates inside the same bbox and
      period where FIRMS reported NO detection. This is an approximation of
      "no fire" (small fires can be missed by satellites) and is declared as
      a limitation in the model metadata.
    - Features: ERA5-based daily fire weather from the Open-Meteo archive
      (T_max, RH_mean, wind max, precipitation) plus the Canadian FWI
      computed with a 21-day spin-up ending on the sample date, plus month.

The output is a trained ``WildfireRiskModel`` artifact (joblib) plus a JSON
metadata file with feature names, sample counts, class balance, validation
metrics and explicit limitations. The artifact is NOT automatically used for
serving; promoting it is a deliberate, reviewed step.
"""

from __future__ import annotations

import json
import math
import os
import random
import urllib.parse
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..dashboard import real_data
from ..dashboard.cache import default_cache
from .fwi import compute_fwi_series
from .risk_model import WildfireRiskModel, RiskMetrics

FEATURE_NAMES = ["temp_max_c", "rh_mean_pct", "wind_max_kmh", "precip_mm", "fwi", "month"]


def _parse_firms_csv(text: str) -> List[Dict]:
    """Parse a FIRMS area-CSV response into fire-point dicts."""
    import csv
    import io

    points = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            points.append(
                {
                    "lat": float(row["latitude"]),
                    "lon": float(row["longitude"]),
                    "date": row["acq_date"],
                    "frp_mw": float(row.get("frp") or 0.0),
                }
            )
        except (KeyError, ValueError):
            continue
    return points


def _firms_fire_points(bbox: Tuple[float, float, float, float], days: int) -> List[Dict]:
    """Fetch real fire detections for a bbox from FIRMS (CSV area API)."""
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError(
            "FIRMS_MAP_KEY is not set. Register free at "
            "https://firms.modaps.eosdis.nasa.gov/api/area/"
        )
    west, south, east, north = bbox
    url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{key}/VIIRS_SNPP_NRT/{west},{south},{east},{north}/{min(days, 10)}"
    )
    text = real_data._get_text(url, timeout=60.0, retries=1)
    return _parse_firms_csv(text)


def firms_fire_points_in_range(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    source: str = "VIIRS_SNPP_NRT",
) -> List[Dict]:
    """
    Fetch real FIRMS detections for a bbox over an explicit date range.

    The FIRMS area API accepts at most 10 days per call, so the range is
    queried in consecutive windows (``.../{day_range}/{start_date}``).
    Requires FIRMS_MAP_KEY. Archive depth depends on the FIRMS collection;
    windows that return no data simply contribute no detections — nothing
    is synthesised. Results are cached per window (7 days).
    """
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError(
            "FIRMS_MAP_KEY is not set. Register free at "
            "https://firms.modaps.eosdis.nasa.gov/api/area/"
        )
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("end_date must be >= start_date")

    west, south, east, north = bbox
    cache = default_cache()
    points: List[Dict] = []
    window_start = start
    while window_start <= end:
        span = min(10, (end - window_start).days + 1)
        cache_key = cache.make_key(
            "firms_range", source, west, south, east, north, span, window_start.isoformat()
        )
        hit = cache.get(cache_key)
        if hit is None:
            url = (
                "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                f"{key}/{source}/{west},{south},{east},{north}/{span}/{window_start.isoformat()}"
            )
            try:
                text = real_data._get_text(url, timeout=60.0, retries=1)
                hit = {"points": _parse_firms_csv(text)}
            except RuntimeError:
                # Honest gap: record the window as unavailable for a short time.
                hit = {"points": [], "unavailable": True}
            cache.set(cache_key, hit, 3600 if hit.get("unavailable") else 7 * 24 * 3600)
        points.extend(hit.get("points") or [])
        window_start += timedelta(days=span)
    return points


def _features_for(lat: float, lon: float, day: str) -> Optional[List[float]]:
    """
    ERA5-based features for one point/date via the Open-Meteo archive,
    including a 21-day FWI spin-up. Cached per point/date.
    """
    cache = default_cache()
    key = cache.make_key("train_features", round(lat, 3), round(lon, 3), day)
    hit = cache.get(key)
    if hit is not None:
        return hit.get("features")

    end = datetime.strptime(day, "%Y-%m-%d").date()
    start = end - timedelta(days=21)
    try:
        data = real_data.fetch_weather_archive(lat, lon, start.isoformat(), end.isoformat())
    except Exception:
        data = {"error": "archive unavailable"}
    if "error" in data or not data.get("time"):
        cache.set(key, {"features": None}, 3600)
        return None

    times = data["time"]
    tmax = data.get("temperature_2m_max") or []
    rh = data.get("relative_humidity_2m_mean") or []
    wind = data.get("wind_speed_10m_max") or []
    rain = data.get("precipitation_sum") or []

    series_in = []
    for i, t in enumerate(times):
        if i >= len(tmax) or i >= len(rh) or i >= len(wind):
            break
        if tmax[i] is None or rh[i] is None or wind[i] is None:
            continue
        series_in.append(
            {
                "date": t,
                "temp_c": float(tmax[i]),
                "rh_pct": float(rh[i]),
                "wind_kmh": float(wind[i]),
                "rain_mm": float(rain[i] or 0.0) if i < len(rain) else 0.0,
            }
        )
    if len(series_in) < 5:
        cache.set(key, {"features": None}, 3600)
        return None

    fwi_days = compute_fwi_series(series_in)
    last = fwi_days[-1]
    i_last = len(series_in) - 1
    features = [
        series_in[i_last]["temp_c"],
        series_in[i_last]["rh_pct"],
        series_in[i_last]["wind_kmh"],
        series_in[i_last]["rain_mm"],
        float(last.fwi),
        float(int(day[5:7])),
    ]
    cache.set(key, {"features": features}, 7 * 24 * 3600)
    return features


def build_dataset(
    bbox: Tuple[float, float, float, float],
    fire_days: int = 10,
    max_positives: int = 400,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build a real fire / no-fire dataset for a bbox over the last `fire_days`.

    Returns (X, y, metadata). Raises RuntimeError when no FIRMS key is set.
    """
    fire_points = _firms_fire_points(bbox, fire_days)
    rng = random.Random(seed)
    rng.shuffle(fire_points)
    positives = fire_points[:max_positives]

    west, south, east, north = bbox
    fire_keys = {(round(p["lat"], 2), round(p["lon"], 2), p["date"]) for p in fire_points}

    X: List[List[float]] = []
    y: List[int] = []
    used_dates: List[str] = []

    for p in positives:
        feats = _features_for(p["lat"], p["lon"], p["date"])
        if feats is None:
            continue
        X.append(feats)
        y.append(1)
        used_dates.append(p["date"])

    n_neg = int(len(X) * negative_ratio)
    end = date.today()
    attempts = 0
    while len([v for v in y if v == 0]) < n_neg and attempts < n_neg * 5:
        attempts += 1
        lat = rng.uniform(south, north)
        lon = rng.uniform(west, east)
        day = (end - timedelta(days=rng.randint(1, fire_days))).isoformat()
        if (round(lat, 2), round(lon, 2), day) in fire_keys:
            continue
        feats = _features_for(lat, lon, day)
        if feats is None:
            continue
        X.append(feats)
        y.append(0)
        used_dates.append(day)

    meta = {
        "bbox": bbox,
        "fire_days": fire_days,
        "n_samples": len(X),
        "n_positive": int(sum(y)),
        "n_negative": int(len(y) - sum(y)),
        "feature_names": FEATURE_NAMES,
        "built_at": datetime.utcnow().isoformat() + "Z",
        "sources": {
            "fires": "NASA FIRMS VIIRS S-NPP NRT (375 m)",
            "weather": "ERA5 via Open-Meteo archive",
            "fwi": "Canadian FWI System, 21-day spin-up",
        },
        "limitations": (
            "Negatives are sampled where FIRMS detected no fire (undetected "
            "small fires may mislabel some negatives). Daily aggregates "
            "approximate noon-standard FWI inputs."
        ),
    }
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), meta


def train_model(
    bbox: Tuple[float, float, float, float],
    out_dir: str,
    fire_days: int = 10,
    max_positives: int = 400,
) -> Dict:
    """
    Train a WildfireRiskModel on real fire history and persist the artifact.

    Returns the training summary (metrics + metadata), which is also written
    to ``<out_dir>/wildfire_risk_model.meta.json``.
    """
    import joblib

    X, y, meta = build_dataset(bbox, fire_days=fire_days, max_positives=max_positives)
    if len(X) < 40 or len(set(y.tolist())) < 2:
        raise RuntimeError(
            f"Not enough usable samples ({len(X)} rows, classes={set(y.tolist())}). "
            "Widen the bbox, extend fire_days, or check the FIRMS key."
        )

    model = WildfireRiskModel(n_estimators=150, random_state=42)
    metrics: RiskMetrics = model.train(X, y, feature_names=FEATURE_NAMES)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "wildfire_risk_model.joblib")
    joblib.dump(model.model, model_path)

    summary = dict(meta)
    summary["metrics"] = metrics.to_dict()
    summary["model_path"] = model_path
    with open(os.path.join(out_dir, "wildfire_risk_model.meta.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
