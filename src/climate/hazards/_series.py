"""
Shared daily-series mathematics for the Stage-4 hazard modules
(flood, drought, heat, wind, coastal).

Pure functions over aligned ``(dates, values)`` daily series — no I/O, no
network. Every method here is declared in the ``method`` strings of the
payloads that use it; nothing is a black box:

- :func:`percentile_value` / :func:`percentile_rank` — linear-interpolation
  percentiles (same convention as ``numpy.percentile(..., method='linear')``).
- :func:`detect_spells` — runs of *consecutive* calendar days breaching a
  threshold (scalar or per-day), with a minimum run length.
- :func:`doy_window_pool` — day-of-year climatology pool (±N days, circular,
  excluding the target year).
- :func:`window_sums_by_year` — W-day sums ending on the same calendar day
  for each available year (drought deficit vs previous years).
- :func:`standardized_anomaly` — ``(x - mean) / sample_std`` against a
  declared baseline; NOT a full SPEI/SPI (no distribution fitting).
- :func:`antecedent_precipitation_index` — recursive API with declared decay.
- :func:`rolling_sums` — trailing W-day sums, None where the window has gaps.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_date(value: Any) -> Optional[date]:
    """Parse an ISO date (YYYY-MM-DD prefix tolerated); None on failure."""

    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def clean(values: Sequence[Any]) -> List[float]:
    """Drop None values, coerce the rest to float."""

    return [float(v) for v in values if v is not None]


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def percentile_value(values: Sequence[Any], q: float) -> Optional[float]:
    """q-th percentile (0–100) of ``values`` with linear interpolation.

    None for empty input. Method: sort, rank = q/100 * (n-1), linear
    interpolation between neighbouring order statistics.
    """

    vals = sorted(clean(values))
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        return vals[0]
    rank = (min(max(q, 0.0), 100.0) / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return vals[lo]
    frac = rank - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def percentile_rank(values: Sequence[Any], value: Any) -> Optional[float]:
    """Percentage of series values <= ``value`` (0–100, one decimal).

    None when the series is empty or the value is None.
    """

    vals = clean(values)
    if value is None or not vals:
        return None
    v = float(value)
    below = sum(1 for x in vals if x <= v)
    return round(100.0 * below / len(vals), 1)


# ---------------------------------------------------------------------------
# Spell detection
# ---------------------------------------------------------------------------


def detect_spells(
    dates: Sequence[Any],
    values: Sequence[Any],
    threshold: Any,
    min_len: int = 3,
    above: bool = True,
) -> List[Dict[str, Any]]:
    """Runs of >= ``min_len`` consecutive days breaching ``threshold``.

    ``threshold`` is either a scalar or a per-day sequence aligned with
    ``dates`` (e.g. a day-of-year climatological threshold). A run breaks on
    a non-breaching day, a missing value, or a calendar gap > 1 day. Each
    spell reports start/end, length and the peak (max for ``above=True``,
    min otherwise) with its date.
    """

    thr_list = list(threshold) if isinstance(threshold, (list, tuple)) else None
    spells: List[Dict[str, Any]] = []
    run: List[int] = []
    prev_date: Optional[date] = None

    def flush() -> None:
        nonlocal run
        if len(run) >= min_len:
            vals = [(i, float(values[i])) for i in run]
            pick = max(vals, key=lambda iv: iv[1]) if above else min(vals, key=lambda iv: iv[1])
            spells.append(
                {
                    "start": str(dates[run[0]]),
                    "end": str(dates[run[-1]]),
                    "length_days": len(run),
                    "peak_value": round(pick[1], 3),
                    "peak_date": str(dates[pick[0]]),
                }
            )
        run = []

    for i, (d, v) in enumerate(zip(dates, values)):
        cur = parse_date(d)
        if thr_list is not None:
            thr = thr_list[i] if i < len(thr_list) else None
        else:
            thr = threshold
        ok = (
            cur is not None
            and v is not None
            and thr is not None
            and (float(v) >= float(thr) if above else float(v) < float(thr))
        )
        if ok and run and prev_date is not None and (cur - prev_date).days == 1:
            run.append(i)
        elif ok:
            flush()
            run = [i]
        else:
            flush()
        if cur is not None:
            prev_date = cur
    flush()
    return spells


# ---------------------------------------------------------------------------
# Day-of-year climatology
# ---------------------------------------------------------------------------


def _doy_distance(a: date, b: date) -> int:
    """Circular distance between two days-of-year (0–182)."""

    diff = abs(a.timetuple().tm_yday - b.timetuple().tm_yday)
    return min(diff, 365 - diff)


def doy_window_pool(
    dates: Sequence[Any],
    values: Sequence[Any],
    target: Any,
    window_days: int = 7,
) -> List[float]:
    """Climatology pool for ``target``: values within ±``window_days`` of the
    target's day-of-year across all *other* years in the series.

    Method: circular day-of-year distance <= window_days, target year
    excluded. Leap-year day numbering is a declared approximation (±1 day at
    the year boundary).
    """

    t = parse_date(target)
    if t is None:
        return []
    pool: List[float] = []
    for d, v in zip(dates, values):
        if v is None:
            continue
        dd = parse_date(d)
        if dd is None or dd.year == t.year:
            continue
        if _doy_distance(dd, t) <= window_days:
            pool.append(float(v))
    return pool


def doy_thresholds(
    recent_dates: Sequence[Any],
    clim_dates: Sequence[Any],
    clim_values: Sequence[Any],
    q: float = 90.0,
    window_days: int = 7,
) -> List[Optional[float]]:
    """Per-day q-th percentile of the day-of-year climatology pool."""

    return [
        percentile_value(doy_window_pool(clim_dates, clim_values, d, window_days), q)
        for d in recent_dates
    ]


# ---------------------------------------------------------------------------
# Window sums / deficits vs previous years
# ---------------------------------------------------------------------------


def window_sums_by_year(
    dates: Sequence[Any],
    values: Sequence[Any],
    window_days: int,
    years_back: int = 10,
) -> List[Dict[str, Any]]:
    """W-day sums ending on the same calendar day as the series' last date,
    for the current and each previous year with complete data.

    A year is included only when every day of its window is present and
    non-null. Current year first. Used for drought deficits: the current
    window is compared against the same calendar window in previous years.
    """

    idx = {str(d): v for d, v in zip(dates, values)}
    if not dates:
        return []
    end = parse_date(dates[-1])
    if end is None:
        return []
    out: List[Dict[str, Any]] = []
    for y in range(end.year, end.year - years_back - 1, -1):
        try:
            e_y = end.replace(year=y)
        except ValueError:  # Feb 29 -> Feb 28 (declared approximation)
            e_y = end.replace(year=y, day=28)
        start = e_y - timedelta(days=window_days - 1)
        vals: List[float] = []
        d = start
        complete = True
        while d <= e_y:
            v = idx.get(d.isoformat())
            if v is None:
                complete = False
                break
            vals.append(float(v))
            d += timedelta(days=1)
        if complete:
            out.append(
                {
                    "year": y,
                    "start": start.isoformat(),
                    "end": e_y.isoformat(),
                    "sum": round(sum(vals), 2),
                }
            )
    return out


def standardized_anomaly(
    current: float, baseline: Sequence[float]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(z, mean, sample std) of ``current`` against ``baseline``.

    Method: (x - mean) / sample_std (n-1). z is None when the baseline has
    fewer than 5 members or zero variance — an honest "cannot standardise"
    rather than a fabricated z-score. NOT a fitted SPEI/SPI.
    """

    base = clean(baseline)
    if not base:
        return None, None, None
    mean = sum(base) / len(base)
    if len(base) < 5:
        return None, mean, None
    var = sum((b - mean) ** 2 for b in base) / (len(base) - 1)
    std = math.sqrt(var)
    if std == 0.0:
        return None, mean, 0.0
    return (float(current) - mean) / std, mean, std


# ---------------------------------------------------------------------------
# Precipitation helpers
# ---------------------------------------------------------------------------


def antecedent_precipitation_index(
    precip: Sequence[Any], decay: float = 0.85
) -> List[Optional[float]]:
    """Antecedent Precipitation Index: API_t = P_t + decay * API_{t-1}.

    Recursive; a missing day resets the recursion (honest gap handling).
    The decay constant is declared by the caller (classic Kohler–Linsley
    style decay ~0.85–0.95).
    """

    out: List[Optional[float]] = []
    api: Optional[float] = None
    for p in precip:
        if p is None:
            out.append(None)
            api = None
            continue
        api = float(p) if api is None else float(p) + decay * api
        out.append(api)
    return out


def rolling_sums(values: Sequence[Any], window: int) -> List[Optional[float]]:
    """Trailing ``window``-day sums; None until the window fills, and None
    wherever the window contains a gap."""

    out: List[Optional[float]] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
            continue
        chunk = values[i + 1 - window : i + 1]
        if any(v is None for v in chunk):
            out.append(None)
        else:
            out.append(float(sum(chunk)))
    return out
