"""
Talaix benchmark executor (docs/BENCHMARKS.md).

Executes the Benchmark Suite (``config/benchmark_suite.json``): one case per
ground-truth event (``config/ground_truth_events.json``). Each case runs the
model's OWN declared detector — the same ``_series`` machinery the hazard
modules use — on real fetched series (network at execution time, cached
through the platform cache in ``src/dashboard/real_data.py``).

Honesty contract:

- ``passed`` means the detector reproduced the expected REAL signal (defined
  from the dataset itself) inside the declared window. It is detection
  reproduction, NOT a skill score and NOT a validation claim.
- ``failed`` means the detector ran on real data and did not find the signal.
- ``key_required`` cases are never executed and never counted as failures.
- Errors (fetch failures, missing data, bugs) are captured and reported as
  ``status: "error"`` with ``executed: false`` — never raised.

Run files are immutable: every suite run writes a new
``data/evaluation/benchmark_run_<timestamp>.json`` and never overwrites.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hazards import _series

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")
_GROUND_TRUTH_PATH = os.path.join(_CONFIG_DIR, "ground_truth_events.json")
_SUITE_PATH = os.path.join(_CONFIG_DIR, "benchmark_suite.json")

# Percentile machinery shared with the hazard modules (declared there).
_HEATWAVE_Q = 90.0        # heat.py: day-of-year p90 threshold
_HEATWAVE_MIN_DAYS = 3    # heat.py: >=3 consecutive days = one spell
_DOY_WINDOW = 7           # heat.py / wind.py: ±7-day climatology pool
_BASELINE = ("1991-01-01", "2020-12-31")
_WIND_EXTREME_Q = 95.0    # declared case threshold (storm-eunice case)
_DROUGHT_Z_MAX = -0.8     # declared case threshold (iberia-drought case)
_DROUGHT_WINDOW = 90
_DRY_THRESHOLD_MM = 1.0   # drought.py: a dry day has < 1 mm
_DRY_MIN_DAYS = 10        # drought.py: >=10 consecutive dry days = one spell
_FLOOD_Q = 90.0           # declared case threshold (ahr-flood case)
_FLOOD_PRECIP_ROLLING = 7
_HISTORY_YEARS = 10       # flood.py / drought.py: ~10-year own-series context
_WINDOW_PAD_DAYS = 6      # spell detectors see a little context around windows


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


def load_ground_truth(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """All ground-truth events (config/ground_truth_events.json)."""

    with open(path or _GROUND_TRUTH_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)["events"]


def get_ground_truth_event(event_id: str) -> Optional[Dict[str, Any]]:
    for event in load_ground_truth():
        if event.get("id") == event_id:
            return event
    return None


def load_suite(path: Optional[str] = None) -> Dict[str, Any]:
    """The benchmark suite definition (config/benchmark_suite.json)."""

    with open(path or _SUITE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _evaluation_dir() -> str:
    return os.environ.get(
        "HYDRASHIELD_EVALUATION_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "evaluation"),
    )


# ---------------------------------------------------------------------------
# Fetchers (lazy import; injectable/monkeypatchable for offline tests)
# ---------------------------------------------------------------------------


def _fetch_daily_climate(lat: float, lon: float, start: str, end: str, variables) -> Dict:
    from ..dashboard import real_data as rd

    return rd.fetch_daily_climate(lat, lon, start, end, variables)


def _fetch_flood_discharge(lat: float, lon: float, start: str, end: str) -> Dict:
    from ..dashboard import real_data as rd

    return rd.fetch_flood_discharge(lat, lon, start, end)


class _DataUnavailable(RuntimeError):
    """A fetcher returned an honest error dict or an empty series."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_window(window: str) -> Tuple[date, date]:
    start_s, end_s = str(window).split("..")
    return date.fromisoformat(start_s.strip()), date.fromisoformat(end_s.strip())


def _require_series(payload: Dict, variable: str, what: str) -> Tuple[List[str], List]:
    if "error" in payload:
        raise _DataUnavailable(f"{what}: {payload['error']}")
    times = payload.get("time") or []
    values = payload.get(variable) or []
    if not times or not any(v is not None for v in values):
        raise _DataUnavailable(f"{what}: empty '{variable}' series")
    return times, values


def _overlaps(spell_start: str, spell_end: str, ws: date, we: date) -> bool:
    s = date.fromisoformat(spell_start[:10])
    e = date.fromisoformat(spell_end[:10])
    return s <= we and e >= ws


# ---------------------------------------------------------------------------
# Per-hazard case executors — the model's own detector on the case window
# ---------------------------------------------------------------------------


def _exec_heat(
    event: Dict, ws: date, we: date, fetch_daily: Callable, fetch_discharge: Callable
) -> Dict[str, Any]:
    """heat_percentile_v1 spell detector (heat.py method) on the window."""

    lat, lon = event["location"]["lat"], event["location"]["lon"]
    clim = fetch_daily(lat, lon, _BASELINE[0], _BASELINE[1], ["temperature_2m_max"])
    clim_times, clim_vals = _require_series(clim, "temperature_2m_max", "ERA5 baseline Tmax")
    recent = fetch_daily(
        lat, lon,
        (ws - timedelta(days=_WINDOW_PAD_DAYS)).isoformat(), we.isoformat(),
        ["temperature_2m_max"],
    )
    times, vals = _require_series(recent, "temperature_2m_max", "ERA5 window Tmax")

    thresholds = _series.doy_thresholds(
        times, clim_times, clim_vals, q=_HEATWAVE_Q, window_days=_DOY_WINDOW
    )
    spells = _series.detect_spells(times, vals, thresholds, min_len=_HEATWAVE_MIN_DAYS)
    overlapping = [s for s in spells if _overlaps(s["start"], s["end"], ws, we)]
    window_pairs = [(t, v, thr) for t, v, thr in zip(times, vals, thresholds)
                    if ws.isoformat() <= t <= we.isoformat() and v is not None]
    peak = max(window_pairs, key=lambda x: x[1], default=None)
    evidence = {
        "spells_detected": spells,
        "spells_overlapping_window": overlapping,
        "window_days_analysed": len(window_pairs),
        "window_peak_tmax_c": (
            {"date": peak[0], "tmax_c": round(peak[1], 1),
             "doy_p90_threshold_c": round(peak[2], 1) if peak[2] is not None else None}
            if peak else None
        ),
        "method": (
            f"Heat spell = >={_HEATWAVE_MIN_DAYS} consecutive days above the location's "
            f"own day-of-year {_HEATWAVE_Q:.0f}th percentile (±{_DOY_WINDOW}-day pool, "
            f"baseline {_BASELINE[0][:4]}–{_BASELINE[1][:4]}); pass = >=1 spell "
            "overlapping the window by >=1 day."
        ),
    }
    return {
        "passed": bool(overlapping),
        "evidence": evidence,
        "data_sources": sorted({clim.get("source"), recent.get("source")} - {None}),
    }


def _exec_wind(
    event: Dict, ws: date, we: date, fetch_daily: Callable, fetch_discharge: Callable
) -> Dict[str, Any]:
    """wind_percentile_v1 percentile machinery on each window day."""

    lat, lon = event["location"]["lat"], event["location"]["lon"]
    clim = fetch_daily(lat, lon, _BASELINE[0], _BASELINE[1], ["wind_gusts_10m_max"])
    clim_times, clim_vals = _require_series(clim, "wind_gusts_10m_max", "ERA5 baseline gusts")
    window = fetch_daily(lat, lon, ws.isoformat(), we.isoformat(), ["wind_gusts_10m_max"])
    times, vals = _require_series(window, "wind_gusts_10m_max", "ERA5 window gusts")

    per_day: List[Dict[str, Any]] = []
    for t, v in zip(times, vals):
        if v is None:
            continue
        pool = _series.doy_window_pool(clim_times, clim_vals, t, _DOY_WINDOW)
        pct = _series.percentile_rank(pool, v)
        per_day.append({
            "date": t,
            "gust_max_kmh": round(v, 1),
            "doy_percentile": pct,
            "climatology_pool_size": len(pool),
        })
    extreme_days = [d for d in per_day
                    if d["doy_percentile"] is not None
                    and d["doy_percentile"] >= _WIND_EXTREME_Q]
    evidence = {
        "per_day": per_day,
        "extreme_days": extreme_days,
        "method": (
            f"Per-day percentile of the ERA5 daily gust maximum within the "
            f"±{_DOY_WINDOW}-day day-of-year pool of the baseline "
            f"{_BASELINE[0][:4]}–{_BASELINE[1][:4]}; pass = >=1 day at "
            f">={_WIND_EXTREME_Q:.0f}th percentile in the window."
        ),
    }
    return {
        "passed": bool(extreme_days),
        "evidence": evidence,
        "data_sources": sorted({clim.get("source"), window.get("source")} - {None}),
    }


def _exec_drought(
    event: Dict, ws: date, we: date, fetch_daily: Callable, fetch_discharge: Callable
) -> Dict[str, Any]:
    """drought_anomaly_v1 deficit + dry-spell machinery at the window end."""

    lat, lon = event["location"]["lat"], event["location"]["lon"]
    hist_start = (we - timedelta(days=365 * _HISTORY_YEARS + _DROUGHT_WINDOW)).isoformat()
    climate = fetch_daily(lat, lon, hist_start, we.isoformat(), ["precipitation_sum"])
    times, pr = _require_series(climate, "precipitation_sum", "ERA5 precipitation")

    yearly = _series.window_sums_by_year(times, pr, _DROUGHT_WINDOW, years_back=_HISTORY_YEARS)
    if len(yearly) < 2 or yearly[0]["end"] != we.isoformat():
        raise _DataUnavailable(
            f"ERA5 precipitation: no complete {_DROUGHT_WINDOW}-day window ending {we}"
        )
    current = yearly[0]
    baseline = [y["sum"] for y in yearly[1:]]
    z, mean, std = _series.standardized_anomaly(current["sum"], baseline)

    window_times = [t for t in times if ws.isoformat() <= t <= we.isoformat()]
    idx = dict(zip(times, pr))
    window_vals = [idx.get(t) for t in window_times]
    dry_spells = _series.detect_spells(
        window_times, window_vals, _DRY_THRESHOLD_MM, min_len=_DRY_MIN_DAYS, above=False
    )
    evidence = {
        "window_days": _DROUGHT_WINDOW,
        "current_period": {"start": current["start"], "end": current["end"]},
        "current_sum_mm": current["sum"],
        "climatology_mean_mm": round(mean, 2) if mean is not None else None,
        "standardized_anomaly": round(z, 2) if z is not None else None,
        "baseline_years": len(baseline),
        "per_year_sums_mm": yearly,
        "dry_spells_in_window": dry_spells,
        "method": (
            f"Standardized anomaly of the {_DROUGHT_WINDOW}-day precipitation sum "
            f"ending {we} vs same-calendar windows of the prior years (sample std, "
            f"n-1; NOT full SPI); dry spell = >={_DRY_MIN_DAYS} consecutive days "
            f"<{_DRY_THRESHOLD_MM} mm. Pass = z <= {_DROUGHT_Z_MAX} AND >=1 dry spell "
            "in the window."
        ),
    }
    passed = (z is not None and z <= _DROUGHT_Z_MAX) and bool(dry_spells)
    return {
        "passed": passed,
        "evidence": evidence,
        "data_sources": [climate.get("source")] if climate.get("source") else [],
    }


def _exec_flood(
    event: Dict, ws: date, we: date, fetch_daily: Callable, fetch_discharge: Callable
) -> Dict[str, Any]:
    """flood_discharge_percentile_v1 machinery: ERA5 7-day totals + GloFAS."""

    lat, lon = event["location"]["lat"], event["location"]["lon"]
    hist_start = (ws - timedelta(days=365 * _HISTORY_YEARS)).isoformat()
    precip = fetch_daily(lat, lon, hist_start, we.isoformat(), ["precipitation_sum"])
    pr_times, pr_vals = _require_series(precip, "precipitation_sum", "ERA5 precipitation")
    discharge = fetch_discharge(lat, lon, hist_start, we.isoformat())
    dis_times, dis_vals = _require_series(discharge, "river_discharge", "GloFAS discharge")

    # Precipitation: 7-day rolling totals over the record; percentile of the
    # totals ending on window days against all totals of the record.
    r7 = _series.rolling_sums(pr_vals, _FLOOD_PRECIP_ROLLING)
    r7_valid = [s for s in r7 if s is not None]
    precip_days: List[Dict[str, Any]] = []
    for t, total in zip(pr_times, r7):
        if total is None or not (ws.isoformat() <= t <= we.isoformat()):
            continue
        precip_days.append({
            "end_date": t,
            "total_7day_mm": round(total, 1),
            "percentile_vs_record": _series.percentile_rank(r7_valid, total),
        })
    best_precip = max(precip_days, key=lambda d: d["total_7day_mm"], default=None)

    # Discharge: percentile of each window day within the location's own series.
    dis_days: List[Dict[str, Any]] = []
    for t, v in zip(dis_times, dis_vals):
        if v is None or not (ws.isoformat() <= t <= we.isoformat()):
            continue
        dis_days.append({
            "date": t,
            "discharge_m3s": round(v, 2),
            "percentile_vs_own_series": _series.percentile_rank(dis_vals, v),
        })
    best_discharge = max(
        dis_days,
        key=lambda d: d["percentile_vs_own_series"]
        if d["percentile_vs_own_series"] is not None else -1.0,
        default=None,
    )

    precip_ok = bool(
        best_precip and best_precip["percentile_vs_record"] is not None
        and best_precip["percentile_vs_record"] >= _FLOOD_Q
    )
    discharge_ok = bool(
        best_discharge and best_discharge["percentile_vs_own_series"] is not None
        and best_discharge["percentile_vs_own_series"] >= _FLOOD_Q
    )
    evidence = {
        "precipitation_7day_totals_in_window": precip_days,
        "best_7day_total": best_precip,
        "discharge_days_in_window": dis_days,
        "best_discharge_day": best_discharge,
        "discharge_series_days": sum(1 for v in dis_vals if v is not None),
        "method": (
            f"7-day precipitation totals (rolling sums) percentile-ranked against "
            f"all 7-day totals of the location's own ~{_HISTORY_YEARS}-year record; "
            f"daily GloFAS discharge percentile-ranked within the location's own "
            f"series. Pass = >=1 7-day total AND >=1 discharge day at "
            f">={_FLOOD_Q:.0f}th percentile within the window."
        ),
    }
    return {
        "passed": precip_ok and discharge_ok,
        "evidence": evidence,
        "data_sources": sorted({precip.get("source"), discharge.get("source")} - {None}),
    }


_EXECUTORS = {
    "heat": _exec_heat,
    "wind": _exec_wind,
    "drought": _exec_drought,
    "flood": _exec_flood,
}


# ---------------------------------------------------------------------------
# Case / suite execution
# ---------------------------------------------------------------------------


def run_case(
    case: Dict[str, Any],
    *,
    fetch_daily_climate: Optional[Callable] = None,
    fetch_flood_discharge: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Execute one benchmark case. Errors are captured, never raised.

    Returns ``{case_id, model_id, ground_truth_event_id, executed, status,
    evidence, window, executed_at, data_sources}`` with status one of
    ``passed | failed | key_required | error``.
    """

    fetch_daily = fetch_daily_climate or _fetch_daily_climate
    fetch_discharge = fetch_flood_discharge or _fetch_flood_discharge
    result: Dict[str, Any] = {
        "case_id": case.get("case_id"),
        "model_id": case.get("model_id"),
        "ground_truth_event_id": case.get("ground_truth_event_id"),
        "executed": False,
        "status": "error",
        "evidence": {},
        "window": None,
        "executed_at": _utcnow(),
        "data_sources": [],
    }
    try:
        if case.get("execution") == "key_required":
            result["status"] = "key_required"
            result["evidence"] = {
                "reason": "Requires FIRMS_MAP_KEY; events derive from "
                          "src/climate/fire_events.py — never fabricated.",
                "pass_criteria": case.get("pass_criteria"),
            }
            return result

        event = get_ground_truth_event(case.get("ground_truth_event_id") or "")
        if event is None:
            raise ValueError(
                f"ground-truth event '{case.get('ground_truth_event_id')}' not found"
            )
        signal = event.get("expected_signal") or {}
        window = signal.get("window")
        if not window:
            raise ValueError(f"event '{event['id']}' declares no expected_signal window")
        result["window"] = window
        ws, we = _parse_window(window)

        executor = _EXECUTORS.get(event.get("hazard"))
        if executor is None:
            raise ValueError(f"no benchmark executor for hazard '{event.get('hazard')}'")
        outcome = executor(event, ws, we, fetch_daily, fetch_discharge)
        result["executed"] = True
        result["status"] = "passed" if outcome["passed"] else "failed"
        result["evidence"] = outcome["evidence"]
        result["data_sources"] = outcome["data_sources"]
    except Exception as exc:  # captured, never raised
        result["status"] = "error"
        result["evidence"] = {"error": str(exc), "error_type": type(exc).__name__}
    return result


def run_suite(out_dir: Optional[str] = None, suite: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run every case of the suite; write an immutable run file; return the run.

    The run file is ``<out_dir>/benchmark_run_<utc timestamp>.json`` — a new
    file per run, never overwritten. ``key_required`` cases are included with
    ``executed: false`` and counted separately, never as failures.
    """

    suite = suite or load_suite()
    results = [run_case(case) for case in suite.get("cases", [])]
    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "passed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "key_required": sum(1 for r in results if r["status"] == "key_required"),
        "errors": sum(1 for r in results if r["status"] == "error"),
    }
    run = {
        "suite_version": suite.get("version"),
        "executed_at": _utcnow(),
        "results": results,
        "summary": summary,
        "note": (
            "passed = the model's own detector reproduced the expected REAL signal "
            "in the declared window on real fetched data (detection reproduction, "
            "not a skill score). key_required cases were not executed."
        ),
    }

    out_dir = out_dir or _evaluation_dir()
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_file = os.path.join(out_dir, f"benchmark_run_{stamp}.json")
    with open(run_file, "x", encoding="utf-8") as fh:  # 'x': never overwrite
        json.dump(run, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    run["run_file"] = run_file
    return run


def latest_run_summary(out_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Summary of the most recent benchmark run file, or None if none exist."""

    out_dir = out_dir or _evaluation_dir()
    try:
        names = sorted(
            n for n in os.listdir(out_dir)
            if n.startswith("benchmark_run_") and n.endswith(".json")
        )
    except OSError:
        return None
    if not names:
        return None
    path = os.path.join(out_dir, names[-1])
    try:
        with open(path, "r", encoding="utf-8") as fh:
            run = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "run_file": path,
        "suite_version": run.get("suite_version"),
        "executed_at": run.get("executed_at"),
        "summary": run.get("summary"),
    }
