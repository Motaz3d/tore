"""
CsrdTX orchestration — build a complete CSRD assessment.

Combines applicability, version-aware ESRS structure, double
materiality (seeded with real physical-risk evidence where Talaix has
it), coverage mapping, readiness scoring and gap analysis into one
content-hashed, sealed assessment.

Never invent: perspectives and datapoints without inputs are emitted
with status UNAVAILABLE / NOT_ASSESSED and a reason — never with
fabricated scores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..evidence import content_hash, utcnow_iso
from ..tx_seal import issue_seal
from .applicability import assess_applicability, normalise_profile
from .materiality import assess_topic, hazard_exposure_seed
from .readiness import build_gap_analysis, compute_readiness
from .regulations import esrs_version, esrs_versions

ENGINE_VERSION = "1.0.0"

DATAPOINT_STATUSES = (
    "VERIFIED",
    "SUPPORTED",
    "COMPANY_DECLARED",
    "INFERRED",
    "UNAVAILABLE",
    "NOT_ASSESSED",
)

DISCLAIMER = (
    "CsrdTX produces a screening-level CSRD/ESRS assessment from declared "
    "company facts and Talaix physical-risk evidence. It is not legal "
    "advice, not assurance under ISAE 3000 / AA1000, and not the "
    "limited-assurance engagement CSRD requires from an auditor or "
    "independent assurance provider. Unavailable data is declared, never "
    "invented."
)

# Topical standards that receive a materiality entry even without inputs.
_TOPICAL_IDS = ("E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1")


def _not_assessed(topic_id: str, reason: str) -> Dict[str, Any]:
    return {
        "topic": topic_id,
        "status": "NOT_ASSESSED",
        "reason": reason,
        "impact_score": None,
        "financial_score": None,
        "combined_score": None,
        "material": None,
        "basis": "none",
        "confidence": 0.0,
        "confidence_label": "low",
    }


def _build_materiality(
    site_results: List[Dict[str, Any]],
    materiality_inputs: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """One materiality entry per topical standard.

    E1 financial materiality is seeded from verified physical hazard
    levels when available. Company-supplied inputs (``materiality_inputs``
    keyed by topic id, each with ``impact`` and/or ``financial`` dicts)
    are scored as given and labelled COMPANY_DECLARED unless they carry
    their own evidence grades.
    """
    inputs = materiality_inputs or {}
    out: List[Dict[str, Any]] = []

    for topic_id in _TOPICAL_IDS:
        topic_input = inputs.get(topic_id) or {}
        impact = topic_input.get("impact")
        financial = topic_input.get("financial")

        if topic_id == "E1" and financial is None:
            seed = hazard_exposure_seed(site_results)
            if seed is not None:
                financial = seed

        if impact is None and financial is None:
            reason = (
                "No evidence-backed hazard data for this topic and no "
                "company-supplied materiality inputs."
                if topic_id != "E1"
                else "No site hazard verification available and no "
                     "company-supplied materiality inputs."
            )
            out.append(_not_assessed(topic_id, reason))
            continue

        assessment = assess_topic(topic_id, impact=impact, financial=financial)
        assessment["status"] = "ASSESSED"
        if financial and financial.get("basis_note"):
            assessment["financial_basis"] = financial["basis_note"]
        if topic_input:
            assessment["input_origin"] = "company_declared"
        out.append(assessment)

    return out


def _build_coverage(esrs_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Disclosure coverage matrix straight from the versioned ESRS doc."""
    matrix: List[Dict[str, Any]] = []
    for topic in esrs_doc.get("topics", []):
        for req in topic.get("disclosure_requirements", []):
            status = req.get("talaix_evidence", "none")
            matrix.append({
                "topic": topic.get("id"),
                "topic_name": topic.get("name"),
                "requirement": req.get("id"),
                "requirement_name": req.get("name"),
                "coverage": (
                    "covered_by_evidence" if status == "evidence_backed"
                    else "partial" if status == "partial"
                    else "not_covered"
                ),
                "note": req.get("note"),
            })
    return matrix


def build_csrd_assessment(
    company: Dict[str, Any],
    assets: Optional[List[Dict[str, Any]]] = None,
    *,
    materiality_inputs: Optional[Dict[str, Any]] = None,
    esrs_version_id: Optional[str] = None,
    verify_sites: bool = True,
) -> Dict[str, Any]:
    """Build the full CSRD assessment.

    ``company`` — profile with at least ``name``; sizing facts
    (employees, net_turnover_eur, balance_sheet_total_eur, listed, …)
    sharpen the applicability determination.
    ``assets`` — optional site list run through the same physical
    verification engine as the sustainability evidence pack.
    ``materiality_inputs`` — optional company-declared materiality
    inputs keyed by topic id.
    ``esrs_version_id`` — pin an ESRS version; defaults to the newest
    in-force version. Pending versions may be requested explicitly and
    are labelled with their status.
    """
    profile = normalise_profile(company)
    applicability = assess_applicability(company)
    esrs_doc = esrs_version(esrs_version_id)

    site_results: List[Dict[str, Any]] = []
    if assets and verify_sites:
        from ..verification import verify_portfolio

        raw = verify_portfolio(assets)
        for r in raw:
            verification = r.get("verification") if r.get("ok") else None
            labels: Dict[str, str] = {}
            if verification:
                for c in verification.get("hazard_checks", []):
                    lvl = c.get("level") or {}
                    if lvl.get("label"):
                        labels[c.get("hazard", "unknown")] = lvl["label"]
            site_results.append({
                "asset": r.get("asset"),
                "ok": r.get("ok"),
                "verification_id": verification.get("verification_id") if verification else None,
                "hazard_levels": labels,
            })
    elif assets:
        # Caller opted out of verification: keep declared sites, honestly.
        for a in assets:
            site_results.append({"asset": a, "ok": None, "verification_id": None, "hazard_levels": {}})

    materiality = _build_materiality(site_results, materiality_inputs)
    coverage = _build_coverage(esrs_doc)
    readiness = compute_readiness(applicability, esrs_doc, profile, site_results, materiality)
    gaps = build_gap_analysis(
        applicability, esrs_doc, materiality,
        coverage_detail=readiness["detail"]["evidence_coverage"],
    )

    basis = {
        "company": profile,
        "reporting_year": applicability["reporting_year"],
        "rule_set": applicability["rule_set"]["id"],
        "esrs_version": esrs_doc["version_id"],
        "determination": applicability["determination"],
        "readiness": readiness["overall"],
        "engine_version": ENGINE_VERSION,
    }
    assessment_id = content_hash(basis)[:16]

    return {
        "assessment_id": assessment_id,
        "generated_at": utcnow_iso(),
        "engine": "CsrdTX",
        "engine_version": ENGINE_VERSION,
        "company": {
            "fields": profile,
            "declared_by_company": True,
            "verification": "Company-supplied metadata — not verified by Talaix.",
        },
        "applicability": applicability,
        "esrs": {
            "version_id": esrs_doc["version_id"],
            "name": esrs_doc.get("name"),
            "status": esrs_doc.get("status"),
            "source": esrs_doc.get("source"),
            "topics_inherited_from": esrs_doc.get("topics_inherited_from"),
            "available_versions": esrs_versions(),
        },
        "materiality": materiality,
        "coverage_matrix": coverage,
        "site_results": site_results,
        "readiness": readiness,
        "gap_analysis": gaps,
        "datapoint_statuses": list(DATAPOINT_STATUSES),
        "disclaimer": DISCLAIMER,
        "honesty_contract": (
            "Unavailable data is declared, never invented: every missing "
            "perspective, datapoint or determination is emitted with an "
            "explicit status and reason."
        ),
        "authenticity": issue_seal("csrd_assessment", assessment_id, basis),
    }
