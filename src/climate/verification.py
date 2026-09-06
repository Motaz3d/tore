"""
Talaix Green Finance Verification engine.

Dependency-free core (no Flask imports). It consumes the hazard registry,
runs the configured physical-risk checks for a single asset, and emits a
structured verification record plus a portfolio batch runner.

Honesty contract: any unavailable hazard layer is declared as UNKNOWN with a
reason; nothing is invented or silently dropped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .evidence import content_hash, utcnow_iso
from .ontology import ClaimStatus, Confidence
from .tx_seal import issue_seal

ENGINE_VERSION = "1.0.0"

#: EU Taxonomy Climate Delegated Act Appendix A style hazard vocabulary.
#: Only hazards present in this mapping are checked. The risk_class list
#: describes whether the hazard is treated as acute, chronic, or both in the
#: context of the verification.
VERIFICATION_HAZARDS: Dict[str, Dict[str, Any]] = {
    "flood": {
        "taxonomy_label": "Riverine / pluvial flooding",
        "risk_class": ["acute"],
    },
    "coastal": {
        "taxonomy_label": "Coastal flooding & sea-level rise",
        "risk_class": ["chronic"],
    },
    "wildfire": {
        "taxonomy_label": "Wildfire",
        "risk_class": ["acute", "chronic"],
    },
    "heat": {
        "taxonomy_label": "Heat stress / heat waves",
        "risk_class": ["chronic", "acute"],
    },
    "drought": {
        "taxonomy_label": "Drought / water stress",
        "risk_class": ["chronic"],
    },
    "wind": {
        "taxonomy_label": "Storms & extreme wind",
        "risk_class": ["acute"],
    },
}

#: Frameworks and standards that give context to the report vocabulary.
FRAMEWORKS = [
    {
        "id": "eu_taxonomy",
        "name": "EU Taxonomy Climate Delegated Act",
        "aspect": "DNSH \"climate change adaptation\" physical-risk assessment",
        "role": "risk vocabulary & assessment criterion",
        "note": (
            "Maps each checked hazard to the Taxonomy's climate-related hazard "
            "classification (Appendix A) and states whether the asset-level "
            "evidence is acute, chronic, or both."
        ),
    },
    {
        "id": "eba_pillar3",
        "name": "EBA Pillar 3 ESG ITS",
        "aspect": "Template 5 physical-risk disclosure",
        "role": "banking disclosure context",
        "note": (
            "Provides the physical-risk evidence context that banks can use "
            "when disclosing climate-risk exposure under Pillar 3 ESG templates."
        ),
    },
    {
        "id": "icma_gbp",
        "name": "ICMA Green Bond Principles",
        "aspect": "Post-issuance monitoring / impact reporting",
        "role": "monitoring cadence",
        "note": (
            "Supports regular post-issuance verification of financed assets "
            "by producing reproducible, versioned physical-risk evidence."
        ),
    },
    {
        "id": "ifrs_s2_tcfd",
        "name": "IFRS S2 / TCFD",
        "aspect": "Physical-risk disclosure language",
        "role": "disclosure language",
        "note": (
            "Uses the same observed / modelled / scenario / unavailable "
            "language expected in physical-risk disclosures."
        ),
    },
]

DISCLAIMER = (
    "Talaix provides a physical-evidence layer only. This report is NOT a "
    "Second Party Opinion, NOT an EU Green Bond external review by an "
    "ESMA-registered external reviewer, and NOT investment advice. Hazard "
    "levels are screening indicators unless explicitly labelled validated."
)

HONESTY_CONTRACT = (
    "Unavailable data is declared, never invented: every missing or "
    "unsuitable hazard layer is recorded as UNKNOWN with a stated limitation."
)

MONITORING_HINT = (
    "Continuous per-hazard monitoring for verified assets is available via "
    "/api/v2/account/alerts."
)


def _risk_class_display(classes: List[str]) -> str:
    return " & ".join(c.capitalize() for c in classes)


def _unknown_check(
    hazard_id: str,
    config: Dict[str, Any],
    *,
    reason: str,
) -> Dict[str, Any]:
    """Build an honest UNKNOWN check when a hazard cannot be assessed."""
    return {
        "hazard": hazard_id,
        "taxonomy_label": config["taxonomy_label"],
        "risk_class": config["risk_class"],
        "status": "unavailable",
        "claim_status": ClaimStatus.UNKNOWN.value,
        "confidence": Confidence.LOW.value,
        "level": None,
        "summary": f"{config['taxonomy_label']} could not be assessed for this asset.",
        "evidence": [],
        "limitations": [reason or "No reason provided."],
    }


def _assess_hazard(
    module: Any,
    hazard_id: str,
    config: Dict[str, Any],
    lat: float,
    lon: float,
    name: Optional[str],
) -> Dict[str, Any]:
    """Run one hazard module and map its result to the verification vocabulary."""
    try:
        result = module.analyze(lat, lon, name=name)
    except Exception as exc:  # noqa: BLE001 — deliberately absorb; honesty path below
        return _unknown_check(
            hazard_id,
            config,
            reason=f"Analysis raised {type(exc).__name__}: {exc}",
        )

    status = result.status
    if status in ("unavailable", "key_required"):
        return _unknown_check(
            hazard_id,
            config,
            reason=result.unavailable_reason or f"{hazard_id} analysis unavailable",
        )

    level = result.level
    if level is not None and level.validated:
        claim_status = ClaimStatus.DOCUMENTED.value
        confidence = Confidence.HIGH.value
    else:
        claim_status = ClaimStatus.MODELLED.value
        confidence = Confidence.MEDIUM.value if status == "ok" else Confidence.LOW.value

    limitations: List[str] = []
    if status == "partial":
        limitations.append(
            "Analysis returned partial data; see the hazard summary and evidence "
            "for missing components."
        )
    if level is not None and not level.validated:
        limitations.append(
            "Level is a screening indicator, not a validated predictor."
        )

    return {
        "hazard": hazard_id,
        "taxonomy_label": config["taxonomy_label"],
        "risk_class": config["risk_class"],
        "status": status,
        "claim_status": claim_status,
        "confidence": confidence,
        "level": level.to_dict() if level else None,
        "summary": result.summary,
        "evidence": result.evidence,
        "limitations": limitations,
    }


def verify_asset(lat: float, lon: float, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Verify a single asset against the configured physical-risk hazards.

    Returns a dict with hazard_checks, declared_gaps, a stable verification_id,
    frameworks, disclaimer and honesty_contract.
    """
    from . import registry

    checks: List[Dict[str, Any]] = []
    declared_gaps: List[Dict[str, str]] = []

    for hazard_id, config in VERIFICATION_HAZARDS.items():
        module = registry.get(hazard_id)
        if module is None:
            check = _unknown_check(
                hazard_id,
                config,
                reason=f"{hazard_id} module is not registered in this deployment.",
            )
        else:
            available, reason = module.availability()
            if not available:
                check = _unknown_check(
                    hazard_id,
                    config,
                    reason=reason or f"{hazard_id} analysis unavailable",
                )
            else:
                check = _assess_hazard(module, hazard_id, config, lat, lon, name)

        if check["claim_status"] == ClaimStatus.UNKNOWN.value:
            declared_gaps.append({
                "hazard": hazard_id,
                "taxonomy_label": config["taxonomy_label"],
                "reason": check["limitations"][0],
            })
        checks.append(check)

    verification_id = content_hash({"checks": checks})[:16]

    assessed = [c for c in checks if c["claim_status"] != ClaimStatus.UNKNOWN.value]
    gaps_count = len(declared_gaps)
    total = len(checks)

    # Collect named levels so the summary can say which hazards drove them.
    levels_by_label: Dict[str, List[str]] = {}
    for c in assessed:
        lvl = c.get("level")
        if lvl and lvl.get("label"):
            levels_by_label.setdefault(lvl["label"], []).append(c["hazard"])

    highest_parts = [f"{label} ({', '.join(hazards)})" for label, hazards in levels_by_label.items()]

    summary = f"{len(assessed)} of {total} hazards assessed with real data"
    if gaps_count:
        summary += f", {gaps_count} declared data gap{'s' if gaps_count != 1 else ''}"
    if highest_parts:
        summary += f". Highest levels: {'; '.join(highest_parts)}."
    else:
        summary += "."

    return {
        "verification_id": verification_id,
        "asset": {"lat": lat, "lon": lon, "name": name},
        "generated_at": utcnow_iso(),
        "engine_version": ENGINE_VERSION,
        "frameworks": FRAMEWORKS,
        "hazard_checks": checks,
        "declared_gaps": declared_gaps,
        "summary": summary,
        "disclaimer": DISCLAIMER,
        "honesty_contract": HONESTY_CONTRACT,
        "monitoring_hint": MONITORING_HINT,
        "authenticity": issue_seal("verification", verification_id, {"checks": checks}),
    }


def verify_portfolio(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run verify_asset for each asset in isolation. One asset failure never
    breaks the batch.
    """
    results: List[Dict[str, Any]] = []
    for asset in assets:
        try:
            lat = float(asset.get("lat"))
            lon = float(asset.get("lon"))
            name = asset.get("name")
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError("lat/lon out of range")
            verification = verify_asset(lat, lon, name=name)
            results.append({
                "asset": {"lat": lat, "lon": lon, "name": name},
                "ok": True,
                "verification": verification,
            })
        except Exception as exc:  # noqa: BLE001 — batch isolation
            results.append({
                "asset": asset,
                "ok": False,
                "error": str(exc),
            })
    return results
