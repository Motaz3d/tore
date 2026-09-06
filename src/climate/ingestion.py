"""
Multi-provider ingestion v1 — declared chains, validators, honest scope.

This module does NOT fetch data. It declares, for each platform variable,
which data-registry entries (``config/data_registry.json`` ids) provide it,
in which order they are tried, and how disagreements are handled — then
provides the validators any ingestion path must pass:

- :func:`validate_series` — temporal consistency of daily series
- :func:`validate_spatial` — coordinate range / containment checks
- :func:`compare_sources` — same variable from two providers, reported
  side by side (FIRMS-style: never silently merged)
- :func:`quality_score` — a DECLARED simple heuristic (high/medium/low)
  over check results; documented, not a validated metric

Scope honesty: the remaining single-provider chain (soil_moisture) is a
declared gap, not a hidden one. The discharge gap was closed 2026-09 by
wiring GEOGLOWS as the second (independent, modelled) provider; gauge
observations remain a declared candidate gap.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ProviderChain:
    """Declared provider chain for one platform variable.

    ``providers`` are data-registry ids in preference order; ``primary`` is
    the first of them; ``fallbacks`` the rest. A single-provider chain is a
    declared gap — ``single_provider_gap`` is True and the comparison note
    must say so.
    """

    variable: str
    providers: List[str]                 # config/data_registry.json ids
    primary: str
    fallbacks: List[str] = field(default_factory=list)
    comparison_note: str = ""

    def __post_init__(self) -> None:
        if self.primary not in self.providers:
            raise ValueError(
                f"chain '{self.variable}': primary '{self.primary}' not in providers")
        unknown = [f for f in self.fallbacks if f not in self.providers]
        if unknown:
            raise ValueError(
                f"chain '{self.variable}': fallbacks not in providers: {unknown}")

    @property
    def single_provider_gap(self) -> bool:
        return len(self.providers) == 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["single_provider_gap"] = self.single_provider_gap
        return d


#: The platform's actual variables and their data-registry provider ids.
#: Ids must exist in config/data_registry.json (checked by tests).
PROVIDER_CHAINS: Dict[str, ProviderChain] = {
    "weather_daily": ProviderChain(
        variable="weather_daily",
        providers=["open-meteo-forecast", "era5-archive"],
        primary="open-meteo-forecast",
        fallbacks=["era5-archive"],
        comparison_note=(
            "Forecast model output (ECMWF/ICON/DWD) vs ERA5 reanalysis: "
            "different temporal classes (FORECAST vs HISTORICAL reanalysis) "
            "— compared for plausibility, never merged into one series."),
    ),
    "terrain": ProviderChain(
        variable="terrain",
        providers=["opentopodata-eudem", "srtm"],
        primary="opentopodata-eudem",
        fallbacks=["srtm"],
        comparison_note=(
            "EU-DEM 25 m primary in Europe; SRTM 90 m global fallback. "
            "Different vintages and resolutions — the source used is always "
            "reported with the value."),
    ),
    "exposure": ProviderChain(
        variable="exposure",
        providers=["ohsome", "overpass"],
        primary="ohsome",
        fallbacks=["overpass"],
        comparison_note=(
            "Same underlying OpenStreetMap data via two access paths: "
            "ohsome (aggregation API, extract lags ~3 weeks) and Overpass "
            "(live DB; three public mirrors tried in order: overpass-api.de, "
            "overpass.kumi.systems, overpass.private.coffee). Counts may "
            "disagree due to extract lag — the path used is reported, "
            "counts are never silently averaged."),
    ),
    "fires": ProviderChain(
        variable="fires",
        providers=["firms-viirs", "firms-modis"],
        primary="firms-viirs",
        fallbacks=["firms-modis"],
        comparison_note=(
            "VIIRS 375 m and MODIS 1 km are different sensors with "
            "different detection characteristics — detections are reported "
            "per sensor, never merged into one fire list."),
    ),
    "discharge": ProviderChain(
        variable="discharge",
        providers=["glofas-openmeteo", "geoglows", "usgs-water"],
        primary="glofas-openmeteo",
        fallbacks=["geoglows", "usgs-water"],
        comparison_note=(
            "Two independent hydrological MODELS: GloFAS (Copernicus EMS/JRC "
            "via Open-Meteo, primary) and GEOGLOWS (ECMWF Streamflow Service, "
            "wired 2026-09) — reported side by side over aligned dates in the "
            "flood analysis, never merged. USGS Water Services (usgs-water, "
            "wired 2026-09) adds real GAUGE observations — US network only; "
            "outside the US the analysis says so explicitly. Models and "
            "gauges are reported per provider, never merged."),
    ),
    "soil_moisture": ProviderChain(
        variable="soil_moisture",
        providers=["era5-land"],
        primary="era5-land",
        fallbacks=[],
        comparison_note=(
            "SINGLE-PROVIDER GAP (declared): ERA5-Land modelled soil "
            "moisture only — no independent soil-moisture source is wired. "
            "Sentinel-1 SAR (sentinel-1, candidate) is the declared "
            "cross-check candidate in the data registry."),
    ),
}


def chains_payload() -> Dict[str, Any]:
    """Serialisable view for /api/v2/ingestion/chains."""
    return {
        "chains": {k: c.to_dict() for k, c in PROVIDER_CHAINS.items()},
        "single_provider_gaps": sorted(
            k for k, c in PROVIDER_CHAINS.items() if c.single_provider_gap),
        "note": ("Provider chains are declarations of source preference and "
                 "comparison discipline — sources are compared side by side "
                 "and never silently merged. Single-provider chains are "
                 "declared gaps, not hidden ones."),
    }


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _issue(kind: str, severity: str, detail: str) -> Dict[str, str]:
    return {"type": kind, "severity": severity, "detail": detail}


def validate_series(series: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Temporal-consistency checks on a daily series.

    ``series`` is a list of dicts with ``date`` (ISO YYYY-MM-DD) and
    ``value`` (None allowed — reported, not filled). Checks: parseable ISO
    dates, strictly increasing order (non-monotonic order and duplicate
    dates flagged separately), gaps > 1 day, null values and null ratio.

    Returns ``{"ok", "issues", "coverage"}`` where ``ok`` means no
    error-severity issue (gaps and nulls are warnings — real series have
    them), and ``coverage`` summarises the span honestly.
    """

    issues: List[Dict[str, str]] = []
    parsed: List[Tuple[int, Optional[_dt.date], Any]] = []
    for i, point in enumerate(series or []):
        if not isinstance(point, dict) or "date" not in point:
            issues.append(_issue(
                "invalid_point", "error",
                f"point {i} is not a dict with a 'date' key"))
            continue
        raw = point.get("date")
        try:
            day = _dt.date.fromisoformat(str(raw))
        except ValueError:
            issues.append(_issue(
                "invalid_date", "error",
                f"point {i}: 'date' is not ISO YYYY-MM-DD: {raw!r}"))
            day = None
        parsed.append((i, day, point.get("value")))

    if not parsed and not issues:
        issues.append(_issue("empty_series", "error", "series is empty"))

    days = [(i, d) for i, d, _v in parsed if d is not None]
    seen: Dict[_dt.date, int] = {}
    for i, d in days:
        if d in seen:
            issues.append(_issue(
                "duplicate_date", "error",
                f"date {d.isoformat()} appears at points {seen[d]} and {i}"))
        else:
            seen[d] = i
    for (pi, prev), (ci, cur) in zip(days, days[1:]):
        if cur <= prev:
            issues.append(_issue(
                "non_monotonic", "error",
                f"dates not increasing between points {pi} ({prev}) and "
                f"{ci} ({cur})"))
        elif (cur - prev).days > 1:
            issues.append(_issue(
                "gap", "warning",
                f"gap of {(cur - prev).days - 1} day(s) between "
                f"{prev.isoformat()} and {cur.isoformat()}"))

    total = len(parsed)
    null_count = sum(1 for _i, _d, v in parsed if v is None)
    null_ratio = (null_count / total) if total else None
    if null_count:
        issues.append(_issue(
            "null_values", "warning",
            f"{null_count} of {total} point(s) have a null value"))

    ok = not any(i["severity"] == "error" for i in issues)
    coverage = {
        "start": days[0][1].isoformat() if days else None,
        "end": days[-1][1].isoformat() if days else None,
        "point_count": total,
        "null_count": null_count,
        "null_ratio": null_ratio,
        "span_days": ((days[-1][1] - days[0][1]).days + 1)
                     if len(days) >= 2 else (1 if days else 0),
    }
    return {"ok": ok, "issues": issues, "coverage": coverage}


def validate_spatial(lat: Any, lon: Any,
                     bbox: Optional[Tuple[float, float, float, float]] = None
                     ) -> Dict[str, Any]:
    """Range checks on coordinates; optional containment in ``bbox``.

    ``bbox`` is ``(min_lat, min_lon, max_lat, max_lon)`` — the same order
    the risk grid uses.
    """

    issues: List[Dict[str, str]] = []
    try:
        latf = float(lat)
        lonf = float(lon)
    except (TypeError, ValueError):
        return {"ok": False, "issues": [_issue(
            "invalid_coordinates", "error",
            f"lat/lon must be numbers, got {lat!r}/{lon!r}")]}
    if not (-90.0 <= latf <= 90.0):
        issues.append(_issue(
            "lat_out_of_range", "error", f"lat {latf} outside [-90, 90]"))
    if not (-180.0 <= lonf <= 180.0):
        issues.append(_issue(
            "lon_out_of_range", "error", f"lon {lonf} outside [-180, 180]"))
    if bbox is not None and not issues:
        min_lat, min_lon, max_lat, max_lon = bbox
        if not (min_lat <= latf <= max_lat and min_lon <= lonf <= max_lon):
            issues.append(_issue(
                "outside_bbox", "error",
                f"({latf}, {lonf}) outside bbox {bbox}"))
    return {"ok": not issues, "issues": issues}


def compare_sources(a: Sequence[Any], b: Sequence[Any],
                    tolerance: float) -> Dict[str, Any]:
    """Compare the same variable from two providers (aligned sequences).

    Pairs where either side is None are skipped (a gap is reported, not
    treated as zero). Returns mean/max absolute deltas over comparable
    pairs, a disagreement flag (max delta above tolerance) and a note.
    Both sources are always reported — never silently merged.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    if len(a) != len(b):
        return {
            "ok": False,
            "compared": 0,
            "mean_abs_delta": None,
            "max_delta": None,
            "tolerance": tolerance,
            "disagreement": None,
            "note": (f"series lengths differ ({len(a)} vs {len(b)}) — "
                     "aligned comparison not possible; both sources "
                     "reported separately, never merged."),
        }
    deltas = [abs(float(x) - float(y))
              for x, y in zip(a, b) if x is not None and y is not None]
    if not deltas:
        return {
            "ok": True,
            "compared": 0,
            "mean_abs_delta": None,
            "max_delta": None,
            "tolerance": tolerance,
            "disagreement": None,
            "note": ("no comparable (non-null) pairs — both sources "
                     "reported separately, never merged."),
        }
    mean_delta = sum(deltas) / len(deltas)
    max_delta = max(deltas)
    disagreement = max_delta > tolerance
    note = (
        f"{len(deltas)} comparable pair(s); mean |delta| "
        f"{mean_delta:.4g}, max {max_delta:.4g} vs tolerance {tolerance}. "
        + ("DISAGREEMENT above tolerance — both sources are reported side "
           "by side; they are never silently merged."
           if disagreement else
           "Within tolerance — both sources still reported; never merged."))
    return {
        "ok": True,
        "compared": len(deltas),
        "mean_abs_delta": mean_delta,
        "max_delta": max_delta,
        "tolerance": tolerance,
        "disagreement": disagreement,
        "note": note,
    }


def quality_score(checks: Dict[str, Dict[str, Any]]) -> str:
    """Declared simple scoring over validator results: high | medium | low.

    Heuristic (documented, NOT a validated metric):
    - ``low``    — any check carries an error-severity issue or ok=False
    - ``medium`` — no errors but at least one warning-severity issue
    - ``high``   — every check clean
    """

    saw_warning = False
    for name, result in (checks or {}).items():
        if not isinstance(result, dict):
            return "low"
        if result.get("ok") is False:
            return "low"
        for issue in result.get("issues", []):
            severity = issue.get("severity") if isinstance(issue, dict) else "error"
            if severity == "error":
                return "low"
            if severity == "warning":
                saw_warning = True
    return "medium" if saw_warning else "high"
