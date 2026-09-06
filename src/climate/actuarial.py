"""
Talaix Actuarial Intelligence layer (TX-2).

Insurance and actuarial mathematics computed **from real observed data only**:
event counts returned by the hazard modules' ``events()`` APIs, the declared
``temporal_coverage()`` of each dataset, and the severity fields carried by
real event records. Nothing monetary is ever invented — where a quantity
cannot be supported by the data, it is declared ``unavailable`` with a reason
(docs/EVIDENCE_ARCHITECTURE.md honesty contract).

What this layer IS:

- Frequency estimation: annual event rate λ̂ = n/T with an exact Poisson
  (Pearson–Klugman / Garwood chi-square) confidence interval.
- Exceedance probabilities under an explicitly declared homogeneous-Poisson
  assumption: annual exceedance probability, horizon probabilities, return
  periods.
- Severity statistics in dataset-native units (never converted to money).
- Collective risk-model moments (compound Poisson) in severity-index units.
- Cross-peril account aggregates with an explicit independence caveat.
- An actuarial reference: the formulas and the insurance terminology an
  underwriter, reinsurer, broker or insurance office needs to read this
  profile.

What this layer IS NOT:

- Not a catastrophe model, not a rate-making tool, not actuarial advice.
- No ground-up loss, no monetary AAL/PML, no EP curve in currency. The
  insurance profile keeps ``loss_quantification = "not_quantified"``.

Dependency-free (stdlib ``math``/``statistics`` only) so ``import tx_core``
stays light.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

ACTUARIAL_VERSION = "1.0.0"

#: Default two-sided confidence level for the Poisson rate interval.
DEFAULT_CONFIDENCE = 0.90

#: Below this many observed events the frequency estimate is flagged
#: ``low_count`` (limited-fluctuation credibility is weak for tiny samples).
LOW_COUNT_THRESHOLD = 5

#: Screening tiers on the annual event frequency λ (events/year). Thresholds
#: are declared here so every consumer reads the same ladder; they are
#: screening conventions, not validated calibration.
FREQUENCY_TIERS: Tuple[Tuple[float, str], ...] = (
    (0.02, "very_low"),    # rarer than ~1 in 50 years
    (0.10, "low"),         # up to ~1 in 10 years
    (0.50, "moderate"),    # up to ~1 in 2 years
    (1.00, "high"),        # about annual
    (math.inf, "very_high"),
)

POISSON_ASSUMPTION = (
    "Counts are modelled as a homogeneous Poisson process: a constant annual "
    "rate with independent occurrences over the declared dataset record. "
    "Clustering, trends and seasonality are not modelled at this screening "
    "level."
)

NO_EVENTS_NOTE = (
    "Zero events in the record does NOT imply zero risk: the upper confidence "
    "bound still admits up to ~3 events over the record length (rule of "
    "three). Absence of evidence is reported, never read as evidence of "
    "absence."
)

CATALOGUE_CAVEAT = (
    "Event catalogues are detection/reporting based; observed counts are a "
    "lower bound on true occurrence (small or unreported events are missed)."
)

NON_MONETARY_NOTE = (
    "Severity statistics are in dataset-native severity units (e.g. fire "
    "radiative power MW, alert scores). They are NOT monetary losses and are "
    "never converted to currency."
)

INDEPENDENCE_CAVEAT = (
    "Cross-peril aggregates assume independence between perils for screening; "
    "real perils can be correlated (compound events), so joint probabilities "
    "are indicative only."
)

#: Numeric event fields treated as severity metrics when present (flat keys).
NUMERIC_SEVERITY_KEYS = (
    "magnitude",
    "intensity",
    "alertscore",
    "severity_score",
    "category",
    "wind_kmh",
    "wind_speed_kmh",
    "max_frp_mw",
    "mean_frp_mw",
    "detections",
)

#: Event fields treated as categorical severity labels when they are strings.
CATEGORICAL_SEVERITY_KEYS = ("severity", "alertlevel", "alert_level")


# ---------------------------------------------------------------------------
# Special functions (chi-square via the regularized incomplete gamma)
# ---------------------------------------------------------------------------

def gammainc_p(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x).

    Numerical Recipes §6.2: series expansion for ``x < a + 1``, continued
    fraction for the upper tail otherwise. Accurate to ~1e-14 — enough to
    invert chi-square distributions for confidence intervals without scipy.
    """
    if a <= 0.0:
        raise ValueError("a must be positive")
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if x == 0.0:
        return 0.0

    gln = math.lgamma(a)
    if x < a + 1.0:
        # Series representation.
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(500):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - gln)

    # Continued fraction for Q(a, x); P = 1 - Q.
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -float(i) * (float(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, 1.0 - q))


def chi2_cdf(x: float, df: float) -> float:
    """Chi-square CDF with ``df`` degrees of freedom."""
    if df <= 0.0:
        raise ValueError("df must be positive")
    if x <= 0.0:
        return 0.0
    return gammainc_p(df / 2.0, x / 2.0)


def chi2_ppf(p: float, df: float) -> float:
    """Inverse chi-square CDF (percent point function) by bisection."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    if df <= 0.0:
        raise ValueError("df must be positive")
    lo = 0.0
    hi = max(df, 1.0)
    while chi2_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            raise ArithmeticError("chi2_ppf failed to bracket")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Frequency estimation (counts → annual rate with an exact interval)
# ---------------------------------------------------------------------------

def poisson_count_ci(n: int, confidence: float = DEFAULT_CONFIDENCE) -> Tuple[float, float]:
    """Exact (Pearson–Klugman / Garwood) two-sided interval on a Poisson count.

    ``lower = ½·χ²(α/2, 2n)`` (0 when n = 0),
    ``upper = ½·χ²(1−α/2, 2(n+1))``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    alpha = 1.0 - confidence
    lower = 0.0 if n == 0 else 0.5 * chi2_ppf(alpha / 2.0, 2.0 * n)
    upper = 0.5 * chi2_ppf(1.0 - alpha / 2.0, 2.0 * (n + 1))
    return lower, upper


def frequency_tier(lambda_per_year: float) -> str:
    """Screening tier for an annual event frequency (declared ladder)."""
    for threshold, tier in FREQUENCY_TIERS:
        if lambda_per_year < threshold:
            return tier
    return FREQUENCY_TIERS[-1][1]


def frequency_estimate(
    events_count: int,
    coverage_years: float,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Annual event-rate estimate λ̂ = n/T with an exact Poisson interval."""
    if coverage_years <= 0:
        raise ValueError("coverage_years must be positive")
    lam = events_count / coverage_years
    lo_n, hi_n = poisson_count_ci(events_count, confidence)
    return {
        "lambda_per_year": round(lam, 5),
        "ci_lower": round(lo_n / coverage_years, 5),
        "ci_upper": round(hi_n / coverage_years, 5),
        "confidence": confidence,
        "method": "Poisson exact interval (Pearson–Klugman / Garwood chi-square)",
        "tier": frequency_tier(lam),
        "low_count": events_count < LOW_COUNT_THRESHOLD,
    }


def annual_exceedance_probability(lambda_per_year: float) -> float:
    """P(N ≥ 1 in one year) = 1 − e^(−λ) under the declared Poisson assumption."""
    return 1.0 - math.exp(-lambda_per_year)


def horizon_exceedance_probability(lambda_per_year: float, years: float) -> float:
    """P(N ≥ 1 within ``years`` years) = 1 − e^(−λ·T)."""
    if years <= 0:
        raise ValueError("years must be positive")
    return 1.0 - math.exp(-lambda_per_year * years)


def return_period_years(aep: float) -> Optional[float]:
    """Return period T = 1/p in years (None when the AEP is zero)."""
    if aep <= 0.0:
        return None
    return 1.0 / aep


# ---------------------------------------------------------------------------
# Severity statistics (dataset-native units, never monetary)
# ---------------------------------------------------------------------------

def _coerce_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str):
        try:
            f = float(value)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def extract_severity(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull severity metrics out of raw event records.

    Numeric metrics are collected **per key** (units are never mixed), nested
    ``severity`` dicts are accepted key-by-key as ``severity.<key>``, and
    categorical labels are counted. Returns ``{"metrics": {key: [values]},
    "labels": {label: count}}``.
    """
    metrics: Dict[str, List[float]] = {}
    labels: Dict[str, int] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        for key in NUMERIC_SEVERITY_KEYS:
            num = _coerce_number(event.get(key))
            if num is not None:
                metrics.setdefault(key, []).append(num)
        nested = event.get("severity")
        if isinstance(nested, dict):
            for key, val in nested.items():
                num = _coerce_number(val)
                if num is not None:
                    metrics.setdefault(f"severity.{key}", []).append(num)
        for key in CATEGORICAL_SEVERITY_KEYS:
            val = event.get(key)
            if isinstance(val, str) and val.strip():
                label = val.strip().lower()
                labels[label] = labels.get(label, 0) + 1

    return {"metrics": metrics, "labels": labels}


def severity_stats(values: Sequence[float]) -> Optional[Dict[str, Any]]:
    """Descriptive statistics for one severity metric (None when empty)."""
    vals = [float(v) for v in values if _coerce_number(v) is not None]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    return {
        "n": n,
        "mean": round(mean, 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "std": round(std, 4),
        "cv": round(std / mean, 4) if mean > 0 else None,
        "variance": round(var, 4),
    }


def collective_risk_moments(lambda_per_year: float, mean: float, variance: float) -> Dict[str, Any]:
    """Compound-Poisson aggregate moments in severity-index units.

    E[S] = λ·E[X];  Var(S) = λ·E[X²] = λ·(Var(X) + E[X]²).
    """
    expected = lambda_per_year * mean
    var = lambda_per_year * (variance + mean ** 2)
    std = math.sqrt(var)
    return {
        "expected_annual_index": round(expected, 4),
        "variance_annual_index": round(var, 4),
        "std_annual_index": round(std, 4),
        "cv": round(std / expected, 4) if expected > 0 else None,
        "model": "compound Poisson (collective risk model)",
        "unit": "dataset severity-index units per year (non-monetary)",
    }


# ---------------------------------------------------------------------------
# Per-peril actuarial block
# ---------------------------------------------------------------------------

def coverage_years_from(temporal_coverage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Derive the record length in years from a module's ``temporal_coverage()``.

    Returns ``{"years": float, "start": str, "end": str}`` or None when the
    coverage cannot be parsed (the caller then reports frequency as
    unavailable with the reason).
    """
    if not isinstance(temporal_coverage, dict) or not temporal_coverage:
        return None
    entry = next(iter(temporal_coverage.values()))
    if not isinstance(entry, dict):
        return None
    start_raw = str(entry.get("start") or "")[:10]
    end_raw = str(entry.get("end") or "")[:10]
    try:
        start = date.fromisoformat(start_raw)
        end = date.fromisoformat(end_raw)
    except ValueError:
        return None
    days = (end - start).days
    if days <= 0:
        return None
    return {
        "years": round(days / 365.25, 2),
        "start": start_raw,
        "end": end_raw,
    }


def build_peril_actuarial(
    *,
    hazard_id: str,
    peril_label: str,
    events_status: str,
    events_count: int,
    events: Optional[Sequence[Dict[str, Any]]] = None,
    temporal_coverage: Optional[Dict[str, Any]] = None,
    radius_km: Optional[float] = None,
    current_level: Optional[str] = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """The actuarial block for one peril — every number traced to real data.

    Honesty paths: events unavailable → the whole block is ``unavailable``
    with the events reason; coverage unparsable → frequency unavailable but
    severity statistics (if any) still reported, so nothing real is hidden.
    """
    block: Dict[str, Any] = {
        "status": "unavailable",
        "unavailable_reason": None,
        "hazard": hazard_id,
        "peril": peril_label,
        "basis": (
            f"events within {radius_km:g} km of the asset over the declared "
            "dataset record" if radius_km else
            "events over the declared dataset record"
        ),
        "assumptions": [POISSON_ASSUMPTION, CATALOGUE_CAVEAT],
        "notes": [],
        "frequency": None,
        "annual_exceedance_probability": None,
        "return_period_years": None,
        "horizon_probabilities": None,
        "severity": None,
        "severity_fit": None,
        "trend": None,
        "collective_risk": None,
    }

    if events_status != "ok":
        block["unavailable_reason"] = (
            f"No actuarial estimate: {hazard_id} event data is {events_status}."
        )
        return block

    coverage = coverage_years_from(temporal_coverage)
    extracted = extract_severity(events or [])

    # -- severity (independent of frequency: real observations are reported
    #    even when the record length is unknown) --------------------------
    severity_block: Dict[str, Any] = {"status": "unavailable", "metrics": {}, "labels": {}}
    for key, values in sorted(extracted["metrics"].items()):
        stats = severity_stats(values)
        if stats is not None:
            severity_block["metrics"][key] = stats
    severity_block["labels"] = dict(sorted(extracted["labels"].items()))
    if severity_block["metrics"] or severity_block["labels"]:
        severity_block["status"] = "ok"
    else:
        severity_block["unavailable_reason"] = (
            "Event records carry no machine-readable severity fields."
        )
    severity_block["unit_note"] = NON_MONETARY_NOTE
    block["severity"] = severity_block

    primary_key = max(
        severity_block["metrics"],
        key=lambda k: (
            severity_block["metrics"][k]["n"],
            severity_block["metrics"][k]["mean"],
        ),
        default=None,
    )

    # -- severity distribution fit (richest numeric metric, disclosed GoF) --
    if primary_key is not None:
        block["severity_fit"] = severity_distribution_fit(
            extracted["metrics"][primary_key]
        )
        block["severity_fit"]["severity_metric"] = primary_key
    else:
        block["severity_fit"] = {
            "status": "unavailable",
            "unavailable_reason": "No numeric severity metric on the event records.",
            "note": SEVERITY_FIT_NOTE,
        }

    # -- frequency trend (non-homogeneous Poisson GLM on dated events) -----
    if coverage is None:
        block["trend"] = {
            "status": "unavailable",
            "unavailable_reason": (
                "Trend estimation needs the declared dataset temporal coverage."
            ),
            "method": "Poisson GLM, log λ(t) = a + b·(t − t̄), IRLS",
            "note": TREND_BIAS_NOTE,
        }
    else:
        block["trend"] = frequency_trend(event_years_from(events or []), coverage)

    # -- frequency ---------------------------------------------------------
    if coverage is None:
        block["unavailable_reason"] = (
            "Frequency not estimable: the dataset temporal coverage is not "
            "declared or not parseable, so a rate per year cannot be formed."
        )
        block["status"] = "partial" if severity_block["status"] == "ok" else "unavailable"
        return block

    freq = frequency_estimate(events_count, coverage["years"], confidence)
    lam = freq["lambda_per_year"]
    aep = annual_exceedance_probability(lam)
    aep_hi = annual_exceedance_probability(freq["ci_upper"])

    block["frequency"] = freq
    block["coverage"] = coverage
    block["events_observed"] = events_count
    block["annual_exceedance_probability"] = round(aep, 5)
    block["annual_exceedance_probability_ci_upper"] = round(aep_hi, 5)
    block["return_period_years"] = (
        round(return_period_years(aep), 1) if aep > 0 else None
    )
    block["horizon_probabilities"] = {
        f"{h}y": round(horizon_exceedance_probability(lam, h), 5)
        for h in (5, 10, 25)
    }
    if events_count == 0:
        block["notes"].append(NO_EVENTS_NOTE)
    if freq["low_count"]:
        block["notes"].append(
            f"Only {events_count} observed event(s): the rate interval is wide "
            "and credibility is limited (limited-fluctuation view)."
        )

    # -- collective risk model (needs a numeric severity metric) -----------
    if primary_key is not None and lam > 0:
        stats = severity_block["metrics"][primary_key]
        block["collective_risk"] = collective_risk_moments(
            lam, stats["mean"], stats["variance"]
        )
        block["collective_risk"]["severity_metric"] = primary_key
        block["collective_risk"]["note"] = NON_MONETARY_NOTE
    elif primary_key is not None:
        block["collective_risk"] = {
            "status": "unavailable",
            "unavailable_reason": (
                "Zero observed events: aggregate moments are degenerate (0) "
                "and would misstate risk; the frequency upper bound is the "
                "honest screening figure."
            ),
        }
    else:
        block["collective_risk"] = {
            "status": "unavailable",
            "unavailable_reason": "No numeric severity metric on the event records.",
        }

    block["status"] = "ok"
    return block


# ---------------------------------------------------------------------------
# Cross-peril account aggregate (single asset, all perils)
# ---------------------------------------------------------------------------

def build_account_actuarial(
    peril_actuarials: Sequence[Dict[str, Any]],
    peril_levels: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Aggregate one asset's per-peril actuarial blocks into an account view.

    Quantities that cannot be supported (e.g. a joint probability when no
    peril has a rate) are declared unavailable, never invented.
    """
    quantified = [p for p in peril_actuarials if p.get("status") == "ok" and p.get("frequency")]
    total = len(peril_actuarials)

    summary: Dict[str, Any] = {
        "status": "ok" if quantified else "unavailable",
        "perils_total": total,
        "perils_quantified": len(quantified),
        "actuarial_version": ACTUARIAL_VERSION,
        "independence_caveat": INDEPENDENCE_CAVEAT,
        "assumptions": [POISSON_ASSUMPTION, CATALOGUE_CAVEAT],
        "insurability": insurability_screen(peril_actuarials, peril_levels),
    }

    if not quantified:
        summary["unavailable_reason"] = (
            "No peril produced an estimable event frequency for this asset."
        )
        summary["text"] = (
            f"Actuarial screen: 0 of {total} perils quantifiable with real "
            "event data."
        )
        return summary

    lambdas = {p["hazard"]: p["frequency"]["lambda_per_year"] for p in quantified}
    total_lambda = sum(lambdas.values())
    any_aep = 1.0
    for p in quantified:
        any_aep *= 1.0 - (p.get("annual_exceedance_probability") or 0.0)
    any_aep = 1.0 - any_aep

    dominant = max(quantified, key=lambda p: p["frequency"]["lambda_per_year"])
    levels = peril_levels or {}
    elevated = [h for h, lvl in levels.items() if str(lvl).lower() in ("high", "extreme", "very high")]

    summary.update({
        "expected_annual_events_all_perils": round(total_lambda, 5),
        "any_peril_annual_exceedance_probability": round(any_aep, 5),
        "any_peril_return_period_years": (
            round(return_period_years(any_aep), 1) if any_aep > 0 else None
        ),
        "peril_frequencies": {k: round(v, 5) for k, v in lambdas.items()},
        "dominant_peril": {
            "hazard": dominant["hazard"],
            "peril": dominant["peril"],
            "lambda_per_year": dominant["frequency"]["lambda_per_year"],
            "annual_exceedance_probability": dominant.get("annual_exceedance_probability"),
            "return_period_years": dominant.get("return_period_years"),
        },
        "elevated_current_levels": elevated,
        "significant_trends": {
            p["hazard"]: {
                "direction": p["trend"]["direction"],
                "annual_multiplier": p["trend"]["annual_multiplier"],
                "p_value": p["trend"]["p_value"],
            }
            for p in quantified
            if (p.get("trend") or {}).get("status") == "ok"
            and p["trend"].get("direction") != "no_significant_trend"
        },
    })
    summary["text"] = (
        f"Actuarial screen: {len(quantified)} of {total} perils quantifiable; "
        f"expected {summary['expected_annual_events_all_perils']:g} event(s)/yr "
        f"across quantified perils; any-peril annual exceedance probability "
        f"{summary['any_peril_annual_exceedance_probability']:.1%} "
        f"(return period {summary['any_peril_return_period_years']} yrs). "
        f"Dominant peril: {dominant['peril']}."
    )
    return summary


# ---------------------------------------------------------------------------
# Trend-adjusted frequency (non-homogeneous Poisson, log-linear GLM)
# ---------------------------------------------------------------------------

#: Minimum data for a trend estimate: enough events, a long enough record,
#: and events spread over enough distinct years — otherwise the slope is
#: noise and the honest answer is "unavailable".
TREND_MIN_EVENTS = 10
TREND_MIN_YEARS = 10
TREND_MIN_ACTIVE_YEARS = 5

TREND_BIAS_NOTE = (
    "Detection/reporting bias can mimic a physical trend: catalogues become "
    "more complete over time, so an increasing count trend may reflect better "
    "observation rather than more hazard. Trend estimates are screening "
    "indicators, never proof of a physical trend."
)

_DATE_KEYS = ("date", "fromdate", "todate", "observed_at", "start", "start_date", "end", "year")


def event_years_from(events: Sequence[Dict[str, Any]]) -> List[int]:
    """Extract occurrence years from raw event records (ints, best-effort)."""
    years: List[int] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        found = None
        for key in _DATE_KEYS:
            val = event.get(key)
            if val is None:
                continue
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if 1900 <= float(val) <= 2100:
                    found = int(val)
                    break
                continue
            text = str(val)[:4]
            if text.isdigit() and 1900 <= int(text) <= 2100:
                found = int(text)
                break
        if found is not None:
            years.append(found)
    return years


def _normal_sf(z: float) -> float:
    """Standard normal upper-tail probability P(Z > z) via erf."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _poisson_glm(counts: Sequence[int], years: Sequence[float]) -> Optional[Dict[str, float]]:
    """Log-linear Poisson GLM log λ_t = a + b·(t − t̄) by IRLS.

    Returns the intercept/slope with standard errors, or None when the
    weighted system is singular (e.g. all counts identical to zero).
    """
    n = len(counts)
    xbar = sum(years) / n
    xs = [y - xbar for y in years]
    total = sum(counts)
    a = math.log(max(total, 1) / n)
    b = 0.0
    s_w = s_wx = s_wxx = 0.0
    for _ in range(200):
        s_w = s_wx = s_wxx = s_wz = s_wxz = 0.0
        for x, y in zip(xs, counts):
            eta = max(-20.0, min(20.0, a + b * x))
            mu = math.exp(eta)
            w = mu
            z = eta + (y - mu) / mu
            s_w += w
            s_wx += w * x
            s_wxx += w * x * x
            s_wz += w * z
            s_wxz += w * x * z
        det = s_w * s_wxx - s_wx * s_wx
        if det <= 1e-300:
            return None
        a_new = (s_wxx * s_wz - s_wx * s_wxz) / det
        b_new = (s_w * s_wxz - s_wx * s_wz) / det
        if abs(a_new - a) + abs(b_new - b) < 1e-12:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    if s_w <= 0:
        return None
    det = s_w * s_wxx - s_wx * s_wx
    if det <= 1e-300:
        return None
    var_b = s_w / det
    var_a = s_wxx / det
    return {"a": a, "b": b, "se_a": math.sqrt(var_a), "se_b": math.sqrt(var_b), "xbar": xbar}


def frequency_trend(
    event_years: Sequence[int],
    coverage: Dict[str, Any],
) -> Dict[str, Any]:
    """Trend-adjusted frequency: Poisson GLM of per-year counts on calendar year.

    Direct answer to the stationarity trap: instead of only assuming a
    constant λ, this estimates whether the observed rate drifts with time and
    what the rate looks like at the *latest* record year. Unavailable
    (honestly) when the record is too thin.
    """
    out: Dict[str, Any] = {
        "status": "unavailable",
        "unavailable_reason": None,
        "method": "Poisson GLM, log λ(t) = a + b·(t − t̄), IRLS",
        "note": TREND_BIAS_NOTE,
    }
    start_year = int(str(coverage.get("start") or "")[:4] or 0)
    end_year = int(str(coverage.get("end") or "")[:4] or 0)
    span = end_year - start_year + 1
    total = len(event_years)
    if not (start_year and end_year and span >= TREND_MIN_YEARS):
        out["unavailable_reason"] = (
            f"Trend needs a record of at least {TREND_MIN_YEARS} years; "
            f"declared record spans {max(span, 0)}."
        )
        return out
    if total < TREND_MIN_EVENTS:
        out["unavailable_reason"] = (
            f"Trend needs at least {TREND_MIN_EVENTS} dated events; "
            f"{total} available."
        )
        return out

    counts = [0] * span
    for y in event_years:
        if start_year <= y <= end_year:
            counts[y - start_year] += 1
    active_years = sum(1 for c in counts if c > 0)
    if active_years < TREND_MIN_ACTIVE_YEARS:
        out["unavailable_reason"] = (
            f"Events fall in only {active_years} distinct year(s); at least "
            f"{TREND_MIN_ACTIVE_YEARS} are needed for a slope."
        )
        return out

    fit = _poisson_glm(counts, [float(start_year + i) for i in range(span)])
    if fit is None:
        out["unavailable_reason"] = "Trend fit is singular (degenerate annual counts)."
        return out

    b, se_b = fit["b"], fit["se_b"]
    z = b / se_b if se_b > 0 else 0.0
    p_value = 2.0 * _normal_sf(abs(z))
    lambda_current = math.exp(fit["a"] + b * (end_year - fit["xbar"]))
    lambda_average = total / span

    if p_value < 0.05:
        direction = "increasing" if b > 0 else "decreasing"
    else:
        direction = "no_significant_trend"

    out.update({
        "status": "ok",
        "events_dated": total,
        "record_years": span,
        "slope_per_year": round(b, 5),
        "slope_se": round(se_b, 5),
        "annual_multiplier": round(math.exp(b), 4),
        "wald_z": round(z, 3),
        "p_value": round(p_value, 5),
        "direction": direction,
        "lambda_average": round(lambda_average, 5),
        "lambda_current_year": round(lambda_current, 5),
        "trend_ratio_current_vs_average": (
            round(lambda_current / lambda_average, 3) if lambda_average > 0 else None
        ),
    })
    return out


# ---------------------------------------------------------------------------
# Severity distribution fitting (lognormal / Pareto, MLE + goodness-of-fit)
# ---------------------------------------------------------------------------

#: Minimum sample for a distribution fit — below this the fit would pretend
#: structure the data cannot support.
SEVERITY_FIT_MIN_N = 8

SEVERITY_FIT_NOTE = (
    "Indicative distribution fit on dataset-native severity units (not money). "
    "Small samples have low discrimination power; the KS statistic and AIC are "
    "disclosed so the fit can be judged, never trusted blindly."
)


def _ks_statistic(sorted_vals: Sequence[float], cdf) -> float:
    n = len(sorted_vals)
    d = 0.0
    for i, x in enumerate(sorted_vals):
        f = min(1.0, max(0.0, cdf(x)))
        d = max(d, abs((i + 1) / n - f), abs(f - i / n))
    return d


def _fit_lognormal(vals: Sequence[float]) -> Optional[Dict[str, Any]]:
    if any(v <= 0 for v in vals):
        return None
    logs = [math.log(v) for v in vals]
    n = len(vals)
    mu = sum(logs) / n
    var = sum((l - mu) ** 2 for l in logs) / n  # MLE (denominator n)
    sigma = math.sqrt(var) if var > 0 else 1e-9
    loglik = sum(
        -l - math.log(sigma) - 0.5 * math.log(2 * math.pi) - (l - mu) ** 2 / (2 * var if var > 0 else 1e-18)
        for l in logs
    )

    def cdf(x: float) -> float:
        if x <= 0:
            return 0.0
        return 0.5 * (1.0 + math.erf((math.log(x) - mu) / (sigma * math.sqrt(2.0))))

    return {
        "distribution": "lognormal",
        "parameters": {"mu": round(mu, 5), "sigma": round(sigma, 5)},
        "loglik": round(loglik, 4),
        "aic": round(2 * 2 - 2 * loglik, 4),
        "_cdf": cdf,
    }


def _fit_pareto(vals: Sequence[float]) -> Optional[Dict[str, Any]]:
    if any(v <= 0 for v in vals):
        return None
    xm = min(vals)
    n = len(vals)
    denom = sum(math.log(v / xm) for v in vals)
    if denom <= 0:
        return None  # all values equal → degenerate
    alpha = n / denom
    loglik = n * math.log(alpha) + n * alpha * math.log(xm) - (alpha + 1.0) * sum(
        math.log(v) for v in vals)

    def cdf(x: float) -> float:
        if x < xm:
            return 0.0
        return 1.0 - (xm / x) ** alpha

    return {
        "distribution": "pareto",
        "parameters": {"x_min": round(xm, 4), "alpha": round(alpha, 5)},
        "loglik": round(loglik, 4),
        "aic": round(2 * 2 - 2 * loglik, 4),
        "_cdf": cdf,
    }


def severity_distribution_fit(values: Sequence[float]) -> Dict[str, Any]:
    """Fit lognormal and Pareto by MLE with disclosed goodness-of-fit.

    The better AIC wins; both fits, their KS statistics and the small-sample
    caveat are always reported. Never fitted below ``SEVERITY_FIT_MIN_N``
    observations.
    """
    vals = sorted(v for v in (_coerce_number(v) for v in values) if v is not None and v > 0)
    out: Dict[str, Any] = {
        "status": "unavailable",
        "unavailable_reason": None,
        "note": SEVERITY_FIT_NOTE,
    }
    if len(vals) < SEVERITY_FIT_MIN_N:
        out["unavailable_reason"] = (
            f"Distribution fitting needs at least {SEVERITY_FIT_MIN_N} numeric "
            f"severity observations; {len(vals)} available."
        )
        return out

    fits: List[Dict[str, Any]] = []
    for fitter in (_fit_lognormal, _fit_pareto):
        fit = fitter(vals)
        if fit is not None:
            fit["ks_statistic"] = round(_ks_statistic(vals, fit.pop("_cdf")), 4)
            fits.append(fit)
    if not fits:
        out["unavailable_reason"] = "No candidate distribution could be fitted."
        return out

    preferred = min(fits, key=lambda f: f["aic"])
    out.update({
        "status": "ok",
        "n": len(vals),
        "preferred": preferred["distribution"],
        "fits": fits,
        "selection": "lowest AIC",
    })
    return out


# ---------------------------------------------------------------------------
# Insurability screen (composite underwriting-attention indicator)
# ---------------------------------------------------------------------------

#: Declared pressure rubric (0–100) per current hazard level and per
#: frequency tier — screening conventions, transparent in every output.
LEVEL_PRESSURE: Dict[str, int] = {
    "minimal": 5, "low": 10, "moderate": 40, "elevated": 55,
    "high": 75, "severe": 85, "very high": 90, "extreme": 95,
}

TIER_PRESSURE: Dict[str, int] = {
    "very_low": 5, "low": 20, "moderate": 45, "high": 70, "very_high": 90,
}

INSURABILITY_BANDS: Tuple[Tuple[float, str, str], ...] = (
    (25.0, "low_attention", "Routine screening sufficient on current evidence."),
    (50.0, "standard_review", "Standard underwriting review recommended."),
    (75.0, "enhanced_review", "Enhanced review: peril-specific assessment advised."),
    (101.0, "senior_referral", "Senior underwriter referral: highest physical-risk attention."),
)

INSURABILITY_NOTE = (
    "Screening indicator, not validated and not a rating: it ranks "
    "underwriting attention from observed event frequencies and current "
    "hazard levels only. Data adequacy is reported separately as confidence — "
    "sparse data lowers confidence, never raises the risk score."
)


def insurability_screen(
    peril_actuarials: Sequence[Dict[str, Any]],
    peril_levels: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Composite underwriting-attention screen for one account.

    Score = 0.6·worst frequency pressure + 0.4·worst current-level pressure
    (declared rubric). Confidence comes from data adequacy (share of perils
    with a real frequency estimate and a current level) — missing data never
    inflates the score.
    """
    total = len(peril_actuarials)
    quantified = [p for p in peril_actuarials if p.get("status") == "ok" and p.get("frequency")]
    levels = {h: str(l).lower() for h, l in (peril_levels or {}).items()}

    out: Dict[str, Any] = {
        "status": "unavailable",
        "unavailable_reason": None,
        "note": INSURABILITY_NOTE,
        "rubric": {
            "score": "0.6 × worst frequency-tier pressure + 0.4 × worst current-level pressure",
            "level_pressure": LEVEL_PRESSURE,
            "tier_pressure": TIER_PRESSURE,
            "bands": {b[1]: b[2] for b in INSURABILITY_BANDS},
        },
    }
    if not quantified and not levels:
        out["unavailable_reason"] = (
            "No peril has a frequency estimate or a current level — nothing to screen."
        )
        return out

    freq_worst = max(
        (TIER_PRESSURE.get(p["frequency"]["tier"], 0) for p in quantified),
        default=None,
    )
    level_worst = max(
        (LEVEL_PRESSURE.get(l, 0) for l in levels.values()),
        default=None,
    )
    components: Dict[str, Any] = {}
    if freq_worst is not None:
        worst_peril = max(
            quantified, key=lambda p: TIER_PRESSURE.get(p["frequency"]["tier"], 0)
        )
        components["frequency_pressure"] = {
            "score": freq_worst, "weight": 0.6, "worst_peril": worst_peril["hazard"],
        }
    if level_worst is not None:
        worst_level_hazard = max(levels, key=lambda h: LEVEL_PRESSURE.get(levels[h], 0))
        components["current_level_pressure"] = {
            "score": level_worst, "weight": 0.4,
            "worst_hazard": worst_level_hazard,
            "worst_level": peril_levels[worst_level_hazard] if peril_levels else None,
        }

    weight_sum = sum(c["weight"] for c in components.values())
    score = sum(c["score"] * c["weight"] for c in components.values()) / weight_sum

    adequacy = (len(quantified) + len(levels)) / (2 * total) if total else 0.0
    confidence = "high" if adequacy >= 0.8 else "medium" if adequacy >= 0.5 else "low"
    band = next(b for b in INSURABILITY_BANDS if score < b[0])

    out.update({
        "status": "ok",
        "attention_score": round(score, 1),
        "attention_band": band[1],
        "band_meaning": band[2],
        "components": components,
        "data_adequacy": round(adequacy, 3),
        "confidence": confidence,
    })
    return out


# ---------------------------------------------------------------------------
# Actuarial reference — formulas + glossary
# ---------------------------------------------------------------------------

GLOSSARY_CATEGORIES: Dict[str, Dict[str, str]] = {
    "underwriting": {"en": "Underwriting"},
    "pricing": {"en": "Pricing & rate-making"},
    "reserving": {"en": "Reserving & claims"},
    "reinsurance": {"en": "Reinsurance"},
    "solvency": {"en": "Solvency & regulation"},
    "catastrophe": {"en": "Catastrophe modelling"},
    "policy": {"en": "Policy terms"},
    "market": {"en": "Market & portfolio"},
}

ACTUARIAL_FORMULAS: List[Dict[str, Any]] = [
    {
        "id": "pure_premium",
        "category": "pricing",
        "name_en": "Pure premium (loss cost)",
        "formula": "PP = f × E[X]",
        "variables": {"f": "annual claim frequency", "E[X]": "expected severity per claim"},
        "use_en": "Baseline cost of risk before any loading — the starting point of rate-making.",
    },
    {
        "id": "collective_risk_mean",
        "category": "pricing",
        "name_en": "Aggregate loss mean (collective risk model)",
        "formula": "E[S] = E[N] × E[X]",
        "variables": {"E[N]": "expected claim count", "E[X]": "expected severity"},
        "use_en": "Expected total annual loss of a risk or portfolio.",
    },
    {
        "id": "collective_risk_variance",
        "category": "pricing",
        "name_en": "Aggregate loss variance",
        "formula": "Var(S) = E[N]·Var(X) + Var(N)·E[X]²",
        "variables": {"Var(N)": "frequency variance", "Var(X)": "severity variance"},
        "use_en": "Volatility of total losses; drives risk loadings and capital.",
    },
    {
        "id": "compound_poisson_variance",
        "category": "pricing",
        "name_en": "Compound Poisson variance",
        "formula": "Var(S) = λ · E[X²]",
        "variables": {"λ": "Poisson rate", "E[X²]": "second moment of severity"},
        "use_en": "Closed-form variance when counts are Poisson — used for the aggregate moments in this profile.",
    },
    {
        "id": "poisson_frequency",
        "category": "catastrophe",
        "name_en": "Poisson frequency law",
        "formula": "P(N = k) = e^(−λ) · λ^k / k!",
        "variables": {"λ": "annual event rate", "k": "number of events"},
        "use_en": "Probability of exactly k events in a year; the frequency backbone of cat analytics.",
    },
    {
        "id": "annual_exceedance",
        "category": "catastrophe",
        "name_en": "Annual exceedance probability",
        "formula": "P(N ≥ 1) = 1 − e^(−λ)",
        "variables": {"λ": "annual event rate"},
        "use_en": "Probability of at least one event in a year — the quantity this profile estimates per peril.",
    },
    {
        "id": "return_period",
        "category": "catastrophe",
        "name_en": "Return period",
        "formula": "T = 1 / p",
        "variables": {"p": "annual exceedance probability"},
        "use_en": "Average years between exceedances; a 1% AEP is the '100-year event'.",
    },
    {
        "id": "horizon_probability",
        "category": "catastrophe",
        "name_en": "Multi-year horizon probability",
        "formula": "P(N ≥ 1 in T yrs) = 1 − e^(−λ·T)",
        "variables": {"λ": "annual event rate", "T": "horizon in years"},
        "use_en": "Probability of at least one event over a policy or mortgage horizon.",
    },
    {
        "id": "loss_ratio",
        "category": "pricing",
        "name_en": "Loss ratio",
        "formula": "LR = incurred losses / earned premium",
        "variables": {"incurred losses": "paid + reserves change", "earned premium": "premium earned over the period"},
        "use_en": "Core profitability gauge of a portfolio or treaty.",
    },
    {
        "id": "combined_ratio",
        "category": "pricing",
        "name_en": "Combined ratio",
        "formula": "CR = loss ratio + expense ratio",
        "variables": {"expense ratio": "acquisition + admin expenses / premium"},
        "use_en": "Below 100% means underwriting profit; above 100% means underwriting loss.",
    },
    {
        "id": "burning_cost",
        "category": "pricing",
        "name_en": "Burning cost",
        "formula": "BC = total incurred losses / exposure units",
        "variables": {"exposure units": "e.g. sum insured or policy-years"},
        "use_en": "Historical loss rate per unit of exposure; the experience-rating anchor.",
    },
    {
        "id": "credibility",
        "category": "pricing",
        "name_en": "Credibility factor",
        "formula": "Z = n / (n + k)  (Bühlmann)  |  Z = √(n/n₀)  (limited fluctuation)",
        "variables": {"n": "observations", "k": "Bühlmann credibility constant", "n₀": "full-credibility standard"},
        "use_en": "Weight given to own experience vs portfolio rate: estimate = Z·own + (1−Z)·prior.",
    },
    {
        "id": "std_dev_principle",
        "category": "pricing",
        "name_en": "Standard-deviation premium principle",
        "formula": "P = E[S] + α · σ(S)",
        "variables": {"α": "risk-loading coefficient", "σ(S)": "aggregate loss std dev"},
        "use_en": "Adds a volatility-based risk loading to the pure premium.",
    },
    {
        "id": "xl_recovery",
        "category": "reinsurance",
        "name_en": "Excess-of-loss recovery",
        "formula": "Recovery = min( max(L − a, 0), limit )",
        "variables": {"L": "ground-up loss", "a": "attachment point", "limit": "layer limit"},
        "use_en": "What an XL treaty pays for a loss L above the attachment.",
    },
    {
        "id": "rate_on_line",
        "category": "reinsurance",
        "name_en": "Rate on line",
        "formula": "ROL = reinsurance premium / layer limit",
        "variables": {"layer limit": "max reinsurance recovery"},
        "use_en": "Price per unit of cat capacity; compared with the layer's expected loss to judge adequacy.",
    },
    {
        "id": "var_tvar",
        "category": "solvency",
        "name_en": "VaR / TVaR",
        "formula": "VaR_α = F⁻¹(α) ;  TVaR_α = E[S | S > VaR_α]",
        "variables": {"F⁻¹": "inverse loss distribution", "α": "confidence level (e.g. 99.5%)"},
        "use_en": "Tail risk measures; Solvency II sets capital at 99.5% one-year VaR.",
    },
    {
        "id": "chain_ladder",
        "category": "reserving",
        "name_en": "Chain-ladder development",
        "formula": "Ĉ_{i,ultimate} = C_{i,k} × Π f_j",
        "variables": {"C_{i,k}": "cumulative claims to development k", "f_j": "age-to-age factors"},
        "use_en": "Projects ultimate claims from development triangles; the reserving workhorse.",
    },
    {
        "id": "ep_curve",
        "category": "catastrophe",
        "name_en": "Exceedance-probability curve (OEP/AEP)",
        "formula": "EP(x) = P(L > x) ;  AAL = ∫ L dEP",
        "variables": {"L": "loss", "x": "loss threshold", "AAL": "average annual loss"},
        "use_en": "The full loss-exceedance view of a cat model; AAL is the area under it.",
    },
    {
        "id": "expected_value_principle",
        "category": "pricing",
        "name_en": "Expected-value premium principle",
        "formula": "Π = (1 + θ) · E[S]",
        "variables": {"θ": "relative safety loading", "E[S]": "expected aggregate loss"},
        "use_en": "The simplest loading rule: a proportional margin over the pure premium.",
    },
    {
        "id": "vulnerability_curve",
        "category": "catastrophe",
        "name_en": "Vulnerability / damage function",
        "formula": "D(h) = min(1, a · h^b)",
        "variables": {"h": "hazard intensity (flood depth, wind speed, FWI…)", "D": "mean damage ratio 0–1", "a, b": "calibrated curve parameters"},
        "use_en": "Translates hazard intensity into a damage ratio on the exposed value — the bridge from hazard to loss in every cat model.",
    },
    {
        "id": "negative_binomial",
        "category": "pricing",
        "name_en": "Negative-binomial frequency",
        "formula": "Var(N) = μ + μ² / r",
        "variables": {"μ": "mean count", "r": "dispersion parameter"},
        "use_en": "Over-dispersed alternative to Poisson when counts vary more than the mean (heterogeneous portfolios).",
    },
    {
        "id": "climate_conditioning",
        "category": "catastrophe",
        "name_en": "Climate conditioning of frequency",
        "formula": "λ_scenario = λ_historical × CF(SSP, horizon)",
        "variables": {"CF": "conditioning factor from a climate scenario (e.g. IPCC SSP pathways)"},
        "use_en": "Adjusts historical rates for a non-stationary climate instead of assuming the past distribution persists — the standard defence against the stationarity trap.",
    },
    {
        "id": "event_set_ep",
        "category": "catastrophe",
        "name_en": "EP curve from a stochastic event set",
        "formula": "EP(x) = Σ p_i · 1{L_i > x}",
        "variables": {"p_i": "annual probability of scenario i", "L_i": "loss of scenario i"},
        "use_en": "How vendor cat models build the exceedance curve: sum scenario probabilities above each loss threshold.",
    },
]

INSURANCE_GLOSSARY: List[Dict[str, str]] = [
    # -- underwriting -------------------------------------------------------
    {"id": "underwriting", "category": "underwriting", "term_en": "Underwriting",
     "def_en": "The process of evaluating, selecting and pricing risks to insure.",
},
    {"id": "peril", "category": "underwriting", "term_en": "Peril",
     "def_en": "The cause of loss insured against (flood, wildfire, windstorm…).",
},
    {"id": "exposure", "category": "underwriting", "term_en": "Exposure",
     "def_en": "The assets, values or lives subject to a peril; the base to which rates apply.",
},
    {"id": "vulnerability", "category": "underwriting", "term_en": "Vulnerability",
     "def_en": "How much damage an asset suffers at a given hazard intensity (damage ratio).",
},
    {"id": "insurability", "category": "underwriting", "term_en": "Insurability",
     "def_en": "Whether a risk meets the conditions to be insured: measurable, accidental, diversifiable, affordable.",
},
    {"id": "moral_hazard", "category": "underwriting", "term_en": "Moral hazard",
     "def_en": "Behaviour change after insurance that raises loss likelihood or size.",
},
    {"id": "adverse_selection", "category": "underwriting", "term_en": "Adverse selection",
     "def_en": "Tendency of higher-risk parties to seek insurance more than lower-risk ones.",
},
    {"id": "risk_appetite", "category": "underwriting", "term_en": "Risk appetite",
     "def_en": "The amount and type of risk an insurer is willing to accept.",
},
    {"id": "accumulation", "category": "underwriting", "term_en": "Accumulation",
     "def_en": "Concentration of many insured risks exposed to the same event (e.g. one flood zone).",
},
    # -- pricing ------------------------------------------------------------
    {"id": "pure_premium", "category": "pricing", "term_en": "Pure premium",
     "def_en": "Frequency × severity: the expected loss cost per exposure unit, before loadings.",
},
    {"id": "frequency", "category": "pricing", "term_en": "Frequency",
     "def_en": "Expected number of loss events per period (usually per year).",
},
    {"id": "severity", "category": "pricing", "term_en": "Severity",
     "def_en": "The size of loss given that an event occurred.",
},
    {"id": "loss_ratio", "category": "pricing", "term_en": "Loss ratio",
     "def_en": "Incurred losses divided by earned premium; core profitability measure.",
},
    {"id": "combined_ratio", "category": "pricing", "term_en": "Combined ratio",
     "def_en": "Loss ratio plus expense ratio; below 100% is an underwriting profit.",
},
    {"id": "burning_cost", "category": "pricing", "term_en": "Burning cost",
     "def_en": "Historical losses per unit of exposure; the experience-rating anchor.",
},
    {"id": "credibility", "category": "pricing", "term_en": "Credibility",
     "def_en": "Statistical weight given to an account's own experience versus the portfolio rate.",
},
    {"id": "expense_loading", "category": "pricing", "term_en": "Expense loading",
     "def_en": "Premium component covering acquisition, administration and overheads.",
},
    {"id": "risk_loading", "category": "pricing", "term_en": "Risk loading",
     "def_en": "Premium margin above expected loss compensating volatility and uncertainty.",
},
    # -- reserving ----------------------------------------------------------
    {"id": "ibnr", "category": "reserving", "term_en": "IBNR",
     "def_en": "Incurred but not reported: reserves for claims that happened but are not yet filed.",
},
    {"id": "case_reserves", "category": "reserving", "term_en": "Case reserves",
     "def_en": "Reserves set claim-by-claim for reported cases.",
},
    {"id": "loss_development", "category": "reserving", "term_en": "Loss development",
     "def_en": "How claim amounts grow over time until final settlement (triangles, chain-ladder).",
},
    {"id": "claims_triangle", "category": "reserving", "term_en": "Claims triangle",
     "def_en": "Accident-year × development-year table of cumulative claims used for reserving.",
},
    # -- reinsurance --------------------------------------------------------
    {"id": "reinsurance", "category": "reinsurance", "term_en": "Reinsurance",
     "def_en": "Insurance for insurers: ceding part of the risk to another carrier.",
},
    {"id": "cedant", "category": "reinsurance", "term_en": "Cedant",
     "def_en": "The insurer that transfers (cedes) risk to a reinsurer.",
},
    {"id": "quota_share", "category": "reinsurance", "term_en": "Quota share",
     "def_en": "Proportional treaty: reinsurer takes a fixed percentage of every risk and premium.",
},
    {"id": "excess_of_loss", "category": "reinsurance", "term_en": "Excess of loss (XL)",
     "def_en": "Non-proportional cover paying losses above an attachment point up to a limit.",
},
    {"id": "attachment_point", "category": "reinsurance", "term_en": "Attachment point",
     "def_en": "Loss level at which an XL treaty starts to pay.",
},
    {"id": "retention", "category": "reinsurance", "term_en": "Retention",
     "def_en": "The share of risk the insurer keeps for its own account.",
},
    {"id": "stop_loss", "category": "reinsurance", "term_en": "Stop loss",
     "def_en": "Cover that caps the cedant's aggregate losses over a period.",
},
    {"id": "retrocession", "category": "reinsurance", "term_en": "Retrocession",
     "def_en": "Reinsurance of a reinsurer — passing risk further up the chain.",
},
    {"id": "rate_on_line", "category": "reinsurance", "term_en": "Rate on line",
     "def_en": "Reinsurance premium divided by the layer limit; the price of cat capacity.",
},
    {"id": "reinstatement", "category": "reinsurance", "term_en": "Reinstatement",
     "def_en": "Restoring an XL layer after it is exhausted, usually for an extra premium.",
},
    {"id": "ils", "category": "reinsurance", "term_en": "ILS / cat bond",
     "def_en": "Capital-market instruments transferring insurance risk to investors.",
},
    # -- solvency -----------------------------------------------------------
    {"id": "solvency_ii", "category": "solvency", "term_en": "Solvency II",
     "def_en": "EU prudential regime: risk-based capital, governance and disclosure for insurers.",
},
    {"id": "scr", "category": "solvency", "term_en": "SCR",
     "def_en": "Solvency Capital Requirement: capital to withstand a 1-in-200 year loss over one year (99.5% VaR).",
},
    {"id": "orsa", "category": "solvency", "term_en": "ORSA",
     "def_en": "Own Risk and Solvency Assessment: the insurer's internal view of its risks and capital needs.",
},
    {"id": "var", "category": "solvency", "term_en": "Value at Risk (VaR)",
     "def_en": "Loss quantile at a confidence level (e.g. the 99.5% worst-case loss).",
},
    {"id": "tvar", "category": "solvency", "term_en": "TVaR / expected shortfall",
     "def_en": "Expected loss given the VaR threshold is breached; a tail-sensitive measure.",
},
    {"id": "risk_margin", "category": "solvency", "term_en": "Risk margin",
     "def_en": "Solvency II addition to best-estimate liabilities for the cost of holding capital.",
},
    # -- catastrophe --------------------------------------------------------
    {"id": "cat_model", "category": "catastrophe", "term_en": "Catastrophe model",
     "def_en": "Simulation framework of hazard × exposure × vulnerability × financial modules producing loss distributions.",
},
    {"id": "aal", "category": "catastrophe", "term_en": "AAL",
     "def_en": "Average annual loss: the long-run mean loss per year (area under the EP curve).",
},
    {"id": "pml", "category": "catastrophe", "term_en": "PML",
     "def_en": "Probable maximum loss: the loss at a chosen return period (e.g. 250-year).",
},
    {"id": "ep_curve", "category": "catastrophe", "term_en": "EP curve",
     "def_en": "Probability of exceeding each loss level; OEP per event, AEP across all events in a year.",
},
    {"id": "ground_up_loss", "category": "catastrophe", "term_en": "Ground-up loss",
     "def_en": "Total loss before any policy terms, deductibles or reinsurance.",
},
    {"id": "return_period", "category": "catastrophe", "term_en": "Return period",
     "def_en": "Average years between events of a given size; reciprocal of the annual exceedance probability.",
},
    {"id": "secondary_uncertainty", "category": "catastrophe", "term_en": "Secondary uncertainty",
     "def_en": "Uncertainty in loss given the event occurred (vs primary uncertainty of occurrence).",
},
    # -- policy -------------------------------------------------------------
    {"id": "sum_insured", "category": "policy", "term_en": "Sum insured",
     "def_en": "Maximum amount payable under a policy; the exposure base.",
},
    {"id": "deductible", "category": "policy", "term_en": "Deductible",
     "def_en": "The part of each loss the policyholder bears before the insurer pays.",
},
    {"id": "policy_limit", "category": "policy", "term_en": "Policy limit / sublimit",
     "def_en": "Cap on the insurer's payment, overall or for a specific peril or item.",
},
    {"id": "exclusion", "category": "policy", "term_en": "Exclusion",
     "def_en": "Perils or circumstances the policy does not cover.",
},
    {"id": "coinsurance", "category": "policy", "term_en": "Coinsurance",
     "def_en": "Sharing of risk between insurers, or a clause penalising under-insurance.",
},
    # -- market -------------------------------------------------------------
    {"id": "protection_gap", "category": "market", "term_en": "Protection gap",
     "def_en": "Difference between economic losses and insured losses for a peril or region.",
},
    {"id": "penetration", "category": "market", "term_en": "Insurance penetration",
     "def_en": "Premiums as a share of GDP; how deeply insurance reaches an economy.",
},
    {"id": "capacity", "category": "market", "term_en": "Capacity",
     "def_en": "Amount of risk the market (or one carrier) is willing and able to write.",
},
    {"id": "hard_soft_market", "category": "market", "term_en": "Hard / soft market",
     "def_en": "Market cycle: hard = rising prices and tight capacity; soft = falling prices and abundant capacity.",
},
    # -- catastrophe / climate analytics additions --------------------------
    {"id": "damage_ratio", "category": "catastrophe", "term_en": "Damage ratio (MDR)",
     "def_en": "Repair/replacement cost as a fraction of the exposed value at a given hazard intensity — the y-axis of a vulnerability curve.",
},
    {"id": "vulnerability_curve_term", "category": "catastrophe", "term_en": "Vulnerability curve",
     "def_en": "Function mapping hazard intensity (flood depth, wind speed, FWI) to the mean damage ratio of an asset class.",
},
    {"id": "exposure_value", "category": "underwriting", "term_en": "Exposure value",
     "def_en": "The monetary value at risk (building, contents, business interruption) to which damage ratios apply.",
},
    {"id": "stochastic_event_set", "category": "catastrophe", "term_en": "Stochastic event set",
     "def_en": "Thousands of simulated catastrophe scenarios with annual probabilities, the engine of vendor cat models.",
},
    {"id": "non_stationarity", "category": "catastrophe", "term_en": "Non-stationarity",
     "def_en": "When the loss distribution itself shifts over time (climate change) — historical rates alone understate future risk.",
},
    {"id": "scenario_analysis", "category": "catastrophe", "term_en": "Climate scenario analysis",
     "def_en": "Re-estimating frequencies/severities under IPCC pathways (SSP1-2.6 … SSP5-8.5) for stress tests and ORSA.",
},
    {"id": "glm", "category": "pricing", "term_en": "GLM (generalized linear model)",
     "def_en": "The rating workhorse: log-link Poisson/gamma regressions pricing frequency and severity by risk factors.",
},
    {"id": "negative_binomial_term", "category": "pricing", "term_en": "Negative binomial",
     "def_en": "Over-dispersed count distribution for claim frequency when variance exceeds the mean.",
},
    {"id": "reporting_bias", "category": "catastrophe", "term_en": "Reporting / detection bias",
     "def_en": "Apparent trends caused by better detection or reporting rather than a real physical trend — always disclosed beside trend estimates.",
},
    {"id": "cte", "category": "solvency", "term_en": "CTE (conditional tail expectation)",
     "def_en": "Alias of TVaR: the mean loss beyond the VaR threshold.",
},
]

REFERENCE_NOTE = (
    "Actuarial reference (formulas and terminology) is knowledge material, "
    "not an analysis result: it defines the quantities an underwriter meets in "
    "this profile and in practice. Quantities the platform cannot support with "
    "real data remain declared gaps — see loss_quantification."
)


def actuarial_reference() -> Dict[str, Any]:
    """The embeddable actuarial reference block (formulas + glossary)."""
    return {
        "actuarial_version": ACTUARIAL_VERSION,
        "note": REFERENCE_NOTE,
        "formula_count": len(ACTUARIAL_FORMULAS),
        "term_count": len(INSURANCE_GLOSSARY),
        "categories": GLOSSARY_CATEGORIES,
        "formulas": ACTUARIAL_FORMULAS,
        "glossary": INSURANCE_GLOSSARY,
    }
