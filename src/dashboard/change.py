"""
"What changed?" — temporal intelligence for a location.

Compares today's fire-danger state with 24 h ago and 7 days ago using the
real daily weather series and the FWI series already computed for the
analysis:

    real daily aggregates (Open-Meteo) + real FWI series
        -> risk on the same FWI-anchored basis for each reference day
        -> per-driver deltas (FWI, temperature, wind, rain, humidity)
        -> a generated explanation naming the drivers that actually changed

Honesty rules:
    - The comparison basis is identical for all days: slope (static) is
      included, the fuel-moisture adjustment is NOT (there is no historical
      FMC); the headline score includes it. This is declared in the block.
    - NDMI change is reported as unavailable — a single recent Sentinel-2
      scene does not provide a time series. Nothing is interpolated.
    - The explanation sentence is generated from the actual deltas via
      declared significance thresholds — no per-location hardcoded text.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Significance thresholds: a driver is named in the explanation only when
# its absolute change exceeds these values.
_SIGNIFICANCE = {
    "fwi": 3.0,
    "temp": 2.0,
    "wind": 8.0,
    "rain": 5.0,   # 7-day precipitation sum (mm)
    "rh": 10.0,
}

_CHANGE_NOTE = (
    "Comparison basis: FWI-anchored score with static terrain for all days; "
    "the fuel-moisture adjustment is excluded (no historical FMC available) "
    "but is included in the headline score."
)


def _risk_basis(fwi: Optional[float], slope: float) -> Optional[float]:
    """FWI-anchored comparison score (same basis for every reference day)."""
    if fwi is None:
        return None
    from ..dashboard.real_analysis import TalaixRealAnalyser

    return TalaixRealAnalyser._risk_score(
        fwi=fwi, slope=slope, fmc=None, wind_kmh=0.0
    )


def _day_map(daily: Dict, fwi_block: Dict) -> Dict[str, Dict]:
    """Join real daily aggregates with the FWI series on date."""
    out: Dict[str, Dict] = {}
    for d in daily.get("days") or []:
        out[d.get("date")] = {
            "temp_max_c": d.get("temp_max_c"),
            "rh_min_pct": d.get("rh_min_pct") if d.get("rh_min_pct") is not None
            else d.get("rh_mean_pct"),
            "wind_kmh": d.get("wind_mean_kmh") if d.get("wind_mean_kmh") is not None
            else d.get("wind_max_kmh"),
            "rain_mm": d.get("precipitation_mm"),
        }
    for d in fwi_block.get("series") or []:
        if d.get("date") in out:
            out[d["date"]]["fwi"] = d.get("fwi")
    return out


def _rain_sum(days: Dict[str, Dict], dates: List[str]) -> Optional[float]:
    vals = [days[d]["rain_mm"] for d in dates if d in days and days[d]["rain_mm"] is not None]
    return round(sum(vals), 1) if vals else None


def _driver(key, label, then, now, unit):
    delta = None
    direction = "stable"
    if then is not None and now is not None:
        delta = round(now - then, 1)
        if delta > 0:
            direction = "up"
        elif delta < 0:
            direction = "down"
    return {
        "key": key,
        "label": label,
        "then": then,
        "now": now,
        "unit": unit,
        "delta": delta,
        "direction": direction,
        "significant": delta is not None and abs(delta) >= _SIGNIFICANCE[key],
    }


def _explain_7d(drivers: List[Dict], risk_then: Optional[float], risk_now: Optional[float]) -> str:
    """Generate the change explanation from the actual driver deltas."""
    if risk_then is None or risk_now is None:
        return "Not enough real data for a 7-day comparison."

    parts = []
    by_key = {d["key"]: d for d in drivers}
    fwi_d = by_key.get("fwi")
    if fwi_d and fwi_d["significant"]:
        parts.append(
            f"fire-weather danger {'strengthened' if fwi_d['delta'] > 0 else 'eased'} "
            f"(FWI {fwi_d['then']} → {fwi_d['now']})"
        )
    temp = by_key.get("temp")
    if temp and temp["significant"]:
        parts.append(
            f"temperatures {'rose' if temp['delta'] > 0 else 'fell'} "
            f"({temp['then']} °C → {temp['now']} °C)"
        )
    rain = by_key.get("rain")
    if rain and rain["significant"]:
        parts.append(
            f"7-day rainfall {'decreased' if rain['delta'] < 0 else 'increased'} "
            f"({rain['then']} mm → {rain['now']} mm)"
        )
    rh = by_key.get("rh")
    if rh and rh["significant"]:
        parts.append(
            f"minimum humidity {'dropped' if rh['delta'] < 0 else 'recovered'} "
            f"({rh['then']}% → {rh['now']}%)"
        )
    wind = by_key.get("wind")
    if wind and wind["significant"]:
        parts.append(
            f"winds {'strengthened' if wind['delta'] > 0 else 'eased'} "
            f"({wind['then']} km/h → {wind['now']} km/h)"
        )

    risk_delta = round(risk_now - risk_then, 1)
    if abs(risk_delta) < 2.0 and not parts:
        return ("Fire-danger conditions were broadly stable over the last "
                "week — no driver changed significantly.")
    direction = "increased" if risk_delta > 0 else "decreased"
    if not parts:
        return (f"Risk {direction} from {risk_then} to {risk_now} over the last "
                "7 days; individual drivers changed only slightly.")
    return (f"Risk {direction} from {risk_then} to {risk_now} over the last "
            "7 days, primarily because " + "; ".join(parts) + ".")


def build_change_block(
    daily: Dict,
    fwi_block: Dict,
    slope: float,
    satellite: Optional[Dict] = None,
) -> Dict:
    """Build the "What changed?" block from real series of the analysis."""
    if not fwi_block.get("available") or not (fwi_block.get("series") or []):
        return {
            "available": False,
            "reason": "No FWI series available — temporal comparison cannot "
                      "be computed from real data.",
        }

    days = _day_map(daily, fwi_block)
    dates = sorted(d for d, v in days.items() if v.get("fwi") is not None)
    if len(dates) < 2:
        return {"available": False, "reason": "FWI series too short for a comparison."}

    today = dates[-1]
    d24h = dates[-2] if len(dates) >= 2 else None
    d7 = dates[-8] if len(dates) >= 8 else dates[0]

    def _risk_on(date):
        return _risk_basis(days[date].get("fwi"), slope)

    risk_now, risk_24h, risk_7d = _risk_on(today), _risk_on(d24h) if d24h else None, _risk_on(d7)

    now_v, then_v = days[today], days[d7]
    last7_now = [d for d in dates if d <= today][-7:]
    last7_then = [d for d in dates if d <= d7][-7:]
    drivers = [
        _driver("fwi", "FWI", then_v.get("fwi"), now_v.get("fwi"), "FWI"),
        _driver("temp", "Temperature (max)", then_v.get("temp_max_c"), now_v.get("temp_max_c"), "°C"),
        _driver("wind", "Wind (mean)", then_v.get("wind_kmh"), now_v.get("wind_kmh"), "km/h"),
        _driver("rain", "Rainfall (7-day sum)", _rain_sum(days, last7_then),
                _rain_sum(days, last7_now), "mm"),
        _driver("rh", "Humidity (min)", then_v.get("rh_min_pct"), now_v.get("rh_min_pct"), "%"),
    ]

    change_24h = None
    if d24h and risk_24h is not None and risk_now is not None:
        change_24h = {
            "date": d24h,
            "risk": risk_24h,
            "fwi": days[d24h].get("fwi"),
            "risk_delta": round(risk_now - risk_24h, 1),
            "fwi_delta": (
                round(now_v["fwi"] - days[d24h]["fwi"], 1)
                if now_v.get("fwi") is not None and days[d24h].get("fwi") is not None
                else None
            ),
        }

    ndmi_note = None
    if satellite and "error" not in (satellite or {}) and satellite.get("observation_date"):
        ndmi_note = (
            "NDMI change unavailable: only one recent cloud-free optical "
            f"satellite scene ({str(satellite['observation_date'])[:10]}); no time series "
            "is interpolated."
        )
    else:
        ndmi_note = ("NDMI change unavailable: no recent cloud-free "
                     "Sentinel-2 or Landsat scene.")

    return {
        "available": True,
        "reference_date": today,
        "risk": {
            "today": risk_now,
            "d24h_ago": risk_24h,
            "d7d_ago": risk_7d,
            "delta_24h": (round(risk_now - risk_24h, 1)
                          if risk_now is not None and risk_24h is not None else None),
            "delta_7d": (round(risk_now - risk_7d, 1)
                         if risk_now is not None and risk_7d is not None else None),
        },
        "dates": {"today": today, "d24h_ago": d24h, "d7d_ago": d7},
        "change_24h": change_24h,
        "drivers_7d": drivers,
        "ndmi_change": {"available": False, "note": ndmi_note},
        "explanation": _explain_7d(drivers, risk_7d, risk_now),
        "basis_note": _CHANGE_NOTE,
        "provenance": {
            "kind": "derived",
            "source": "Canadian FWI System + Open-Meteo daily series (real)",
            "limitations": _CHANGE_NOTE,
        },
    }
