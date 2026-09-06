"""
Talaix Insurance & Environmental Risk engine.

No Flask imports. Combines current per-peril hazard levels (via the Green
Finance Verification engine) with long-term historical event records from each
hazard module's ``events()`` API, and quantifies the actuarial layer
(``src/climate/actuarial.py``): event-frequency estimates with exact Poisson
intervals, exceedance probabilities, return periods and severity statistics —
all derived from real observed data, never invented. Monetary loss
quantification remains out of scope (``loss_quantification = not_quantified``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import actuarial
from .engine import ProductEngine
from .evidence import content_hash, utcnow_iso
from .tx_seal import issue_seal
from .verification import verify_asset

ENGINE_VERSION = "1.1.0"

INSURANCE_PERILS: Dict[str, Dict[str, Any]] = {
    "flood": {"label": "Flood (riverine / pluvial)"},
    "wildfire": {"label": "Wildfire"},
    "wind": {"label": "Windstorm"},
    "heat": {"label": "Heatwave"},
    "drought": {"label": "Drought"},
    "coastal": {"label": "Coastal / storm surge"},
}

INSURANCE_FRAMEWORKS = [
    {
        "id": "eiopa",
        "name": "EIOPA climate and natural-catastrophe stress tests",
        "aspect": "Natural-catastrophe & climate stress-test exercises for (re)insurers",
        "role": "regulatory context",
        "note": (
            "This profile's per-peril levels and event history provide the "
            "physical-evidence layer for such exercises; it is not a scenario "
            "or loss model."
        ),
    },
    {
        "id": "solvency_ii",
        "name": "Solvency II",
        "aspect": "Natural-catastrophe underwriting risk & ORSA",
        "role": "prudential context",
        "note": (
            "The profile can inform underwriting risk identification; it does "
            "not replace internal model or standard-formula calculations."
        ),
    },
    {
        "id": "protection_gap",
        "name": "Protection gap / EIOPA nat-cat dashboard",
        "aspect": "Market-level exposure and protection-gap monitoring",
        "role": "market context",
        "note": (
            "Evidence-based screening can help close data gaps on exposed "
            "assets by declaring which perils are observable."
        ),
    },
]

NOT_QUANTIFIED = (
    "Monetary loss quantification is not provided: no ground-up loss estimate, "
    "no exceedance-probability curve, no AAL/PML, and no scenario loss number. "
    "The product is a hazard-level and event-history data layer only."
)

INSURANCE_DISCLAIMER = (
    "Talaix Insurance Environmental Risk Profiles are screening-level data "
    "products for underwriters and reinsurers. This is NOT a vendor catastrophe "
    "model, NOT a rate-making or pricing tool, and NOT actuarial advice. "
    "Levels are screening indicators unless explicitly labelled validated. "
    "Event records are limited to the declared dataset coverage per peril."
)

HONESTY_CONTRACT = (
    "Unavailable data is declared, never invented: every missing current-level "
    "layer or unavailable event dataset is recorded as a declared gap with a "
    "stated reason. No loss is quantified where the data does not support it."
)


class InsuranceEngine(ProductEngine):
    """Reference implementation of the unified product-engine contract
    (``src/climate/engine.py``) — other product engines migrate to this
    shape one by one."""

    id = "insurance"
    name = "Insurance & Environmental Risk"
    engine_version = ENGINE_VERSION
    disclaimer = INSURANCE_DISCLAIMER


_ENGINE = InsuranceEngine()


def _safe_event_summary(event: Dict[str, Any]) -> Dict[str, Any]:
    """Trim an event record to a small, safe set of scalar fields."""
    summary: Dict[str, Any] = {}
    for key in ("id", "event_id", "episodeid", "eventname", "name", "title"):
        val = event.get(key)
        if val is not None:
            summary[key] = str(val)
    # Date / year: accept several common keys.
    for key in ("date", "fromdate", "todate", "year", "observed_at", "start", "end"):
        val = event.get(key)
        if val is not None:
            summary[key] = str(val)
    # Severity-like scalar fields.
    for key in ("severity", "alertlevel", "alertscore", "magnitude", "intensity", "category"):
        val = event.get(key)
        if val is not None:
            summary[key] = val
    return summary


def _events_for_peril(module: Any, hazard_id: str, lat: float, lon: float, radius_km: float) -> Tuple[Dict[str, Any], Sequence[Dict[str, Any]]]:
    """Run one hazard module's events() and map to the insurance vocabulary.

    Returns ``(events_block, raw_events)`` — the raw records feed the
    actuarial severity extraction (never the trimmed 5-event summary).
    """
    try:
        result = module.events(lat, lon, radius_km=radius_km)
    except Exception as exc:  # noqa: BLE001 — honesty path below
        return {
            "events_status": "unavailable",
            "events_count": 0,
            "events_summary": [],
            "events_reason": f"events() raised {type(exc).__name__}: {exc}",
        }, []

    status = result.get("status")
    if status in ("unavailable", "key_required"):
        return {
            "events_status": status,
            "events_count": 0,
            "events_summary": [],
            "events_reason": result.get("reason") or result.get("unavailable_reason") or f"{hazard_id} events unavailable",
        }, []

    events = result.get("events") or []
    return {
        "events_status": "ok",
        "events_count": len(events),
        "events_summary": [_safe_event_summary(e) for e in events[:5]],
        "events_reason": None,
    }, events


def _temporal_coverage(module: Any) -> Optional[Dict[str, Any]]:
    try:
        return module.temporal_coverage() or None
    except Exception:  # noqa: BLE001
        return None


def build_risk_profile(lat: float, lon: float, name: Optional[str] = None, radius_km: float = 50.0) -> Dict[str, Any]:
    """Build a single-asset insurance environmental risk profile."""
    from . import registry

    verification = verify_asset(lat, lon, name=name)
    current_by_hazard = {c["hazard"]: c for c in verification.get("hazard_checks", [])}

    perils: List[Dict[str, Any]] = []
    declared_gaps: List[Dict[str, Any]] = []
    events_available_count = 0

    for hazard_id, config in INSURANCE_PERILS.items():
        module = registry.get(hazard_id)
        current = current_by_hazard.get(hazard_id, {})

        # Current-level gap from verification
        if current.get("claim_status") == "UNKNOWN":
            declared_gaps.append({
                "type": "current_level",
                "hazard": hazard_id,
                "peril": config["label"],
                "reason": (current.get("limitations") or ["No reason provided"])[0],
            })

        # Long-term events
        if module is None:
            events_block = {
                "events_status": "unavailable",
                "events_count": 0,
                "events_summary": [],
                "events_reason": f"{hazard_id} module is not registered in this deployment.",
            }
            raw_events: Sequence[Dict[str, Any]] = []
            coverage = None
        else:
            events_block, raw_events = _events_for_peril(module, hazard_id, lat, lon, radius_km)
            coverage = _temporal_coverage(module)

        if events_block["events_status"] in ("unavailable", "key_required"):
            declared_gaps.append({
                "type": "events",
                "hazard": hazard_id,
                "peril": config["label"],
                "reason": events_block["events_reason"] or f"{hazard_id} events unavailable",
            })
        else:
            events_available_count += 1

        level = current.get("level") or {}
        peril_actuarial = actuarial.build_peril_actuarial(
            hazard_id=hazard_id,
            peril_label=config["label"],
            events_status=events_block["events_status"],
            events_count=events_block["events_count"],
            events=raw_events,
            temporal_coverage=coverage,
            radius_km=radius_km,
            current_level=(level.get("label") or None),
        )
        perils.append({
            "hazard": hazard_id,
            "peril": config["label"],
            "current_level": level.get("label") or "—",
            "claim_status": current.get("claim_status") or "UNKNOWN",
            "confidence": current.get("confidence") or "—",
            "summary": current.get("summary") or "",
            "level_basis": level.get("basis") or "",
            "evidence": current.get("evidence") or [],
            "limitations": current.get("limitations") or [],
            "events_status": events_block["events_status"],
            "events_count": events_block["events_count"],
            "events_summary": events_block["events_summary"],
            "events_reason": events_block["events_reason"],
            "temporal_coverage": coverage,
            "actuarial": peril_actuarial,
        })

    assessed_perils = [p for p in perils if p["claim_status"] != "UNKNOWN"]
    highest_levels: Dict[str, List[str]] = {}
    for p in perils:
        lvl = p["current_level"]
        if lvl and lvl != "—":
            highest_levels.setdefault(lvl, []).append(p["peril"])

    exposure_summary = (
        f"{len(assessed_perils)} of {len(perils)} perils assessed with real data; "
        f"long-term event records available for {events_available_count} perils; "
        f"{len(declared_gaps)} declared data gap{'s' if len(declared_gaps) != 1 else ''}."
    )
    if highest_levels:
        exposure_summary += " Highest current levels: " + "; ".join(
            f"{label} ({', '.join(perils_list)})" for label, perils_list in highest_levels.items()
        ) + "."

    profile_id = content_hash({
        "asset": {"lat": lat, "lon": lon, "name": name},
        "radius_km": radius_km,
        "perils": perils,
    })[:16]
    authenticity = issue_seal(
        "insurance",
        profile_id,
        {"asset": {"lat": lat, "lon": lon, "name": name}, "radius_km": radius_km, "perils": perils},
    )

    account_actuarial = actuarial.build_account_actuarial(
        [p["actuarial"] for p in perils],
        peril_levels={
            p["hazard"]: p["current_level"] for p in perils if p["current_level"] != "—"
        },
    )

    return _ENGINE.result(
        summary=exposure_summary,
        blocks={
            "profile_id": profile_id,
            "asset": {"lat": lat, "lon": lon, "name": name},
            "radius_km": radius_km,
            "perils": perils,
            "declared_gaps": declared_gaps,
            "exposure_summary": exposure_summary,
            "actuarial_summary": account_actuarial,
            "actuarial_reference": actuarial.actuarial_reference(),
            "frameworks": INSURANCE_FRAMEWORKS,
            "loss_quantification": "not_quantified",
            "loss_quantification_note": NOT_QUANTIFIED,
            "honesty_contract": HONESTY_CONTRACT,
            "authenticity": authenticity,
        },
    ).to_dict()


def _trim_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Light portfolio result shape."""
    peril_levels: Dict[str, str] = {}
    for p in profile.get("perils") or []:
        lvl = p.get("current_level")
        if lvl and lvl != "—":
            peril_levels[p["hazard"]] = lvl
    account = profile.get("actuarial_summary") or {}
    return {
        "asset": profile.get("asset"),
        "ok": True,
        "profile_id": profile.get("profile_id"),
        "exposure_summary": profile.get("exposure_summary"),
        "peril_levels": peril_levels,
        "events_available_count": sum(
            1 for p in profile.get("perils", []) if p.get("events_status") == "ok"
        ),
        "declared_gaps_count": len(profile.get("declared_gaps", [])),
        "actuarial": {
            "status": account.get("status"),
            "perils_quantified": account.get("perils_quantified", 0),
            "expected_annual_events_all_perils": account.get("expected_annual_events_all_perils"),
            "any_peril_annual_exceedance_probability": account.get("any_peril_annual_exceedance_probability"),
            "any_peril_return_period_years": account.get("any_peril_return_period_years"),
            "dominant_peril": account.get("dominant_peril"),
        },
    }


def build_portfolio_profile(assets: List[Dict[str, Any]], radius_km: float = 50.0) -> Dict[str, Any]:
    """Build an insurance profile for a portfolio of assets in isolation."""
    results: List[Dict[str, Any]] = []
    level_distribution: Dict[str, Dict[str, int]] = {}
    total_gaps = 0

    for asset in assets:
        try:
            lat = float(asset.get("lat"))
            lon = float(asset.get("lon"))
            name = asset.get("name")
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError("lat/lon out of range")
            profile = build_risk_profile(lat, lon, name=name, radius_km=radius_km)
            trimmed = _trim_profile(profile)
            results.append(trimmed)
            total_gaps += len(profile.get("declared_gaps", []))
            for p in profile.get("perils", []):
                lvl = p.get("current_level")
                if lvl and lvl != "—":
                    level_distribution.setdefault(p["hazard"], {}).setdefault(lvl, 0)
                    level_distribution[p["hazard"]][lvl] += 1
        except Exception as exc:  # noqa: BLE001 — batch isolation
            results.append({
                "asset": asset,
                "ok": False,
                "error": str(exc),
            })

    ok_count = sum(1 for r in results if r.get("ok"))
    quantified_sites = sum(
        1 for r in results
        if r.get("ok") and (r.get("actuarial") or {}).get("perils_quantified", 0) > 0
    )
    portfolio_aep = 1.0
    any_aep_known = False
    for r in results:
        aep = (r.get("actuarial") or {}).get("any_peril_annual_exceedance_probability")
        if aep is not None:
            any_aep_known = True
            portfolio_aep *= 1.0 - aep
    portfolio_aep = 1.0 - portfolio_aep if any_aep_known else None
    portfolio_id = content_hash({
        "results": [{"profile_id": r.get("profile_id"), "ok": r.get("ok")} for r in results],
        "radius_km": radius_km,
    })[:16]

    return {
        "portfolio_id": portfolio_id,
        "generated_at": utcnow_iso(),
        "engine_version": ENGINE_VERSION,
        "radius_km": radius_km,
        "frameworks": INSURANCE_FRAMEWORKS,
        "loss_quantification": "not_quantified",
        "loss_quantification_note": NOT_QUANTIFIED,
        "disclaimer": INSURANCE_DISCLAIMER,
        "honesty_contract": HONESTY_CONTRACT,
        "portfolio_summary": {
            "site_count": len(results),
            "ok_count": ok_count,
            "level_distribution": level_distribution,
            "total_declared_gaps": total_gaps,
            "actuarial": {
                "sites_with_quantified_perils": quantified_sites,
                "any_site_any_peril_aep": (
                    round(portfolio_aep, 5) if portfolio_aep is not None else None
                ),
                "independence_caveat": actuarial.INDEPENDENCE_CAVEAT,
            },
        },
        "results": results,
    }
