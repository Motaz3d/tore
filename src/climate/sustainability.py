"""
Talaix Sustainability Evidence Report engine (CSRD / ESRS-oriented).

No Flask imports. Reuses the physical verification engine from
``src.climate.verification`` to produce an integrated evidence pack for a
company's sites. The pack explicitly states which disclosure areas it covers
and which it does NOT cover.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .evidence import content_hash, utcnow_iso
from .tx_seal import issue_seal
from .verification import verify_portfolio

ENGINE_VERSION = "1.0.0"

SUSTAINABILITY_FRAMEWORKS = [
    {
        "id": "csrd_esrs",
        "name": "CSRD (EU) 2022/2464 & ESRS Delegated Regulation (EU) 2023/2772",
        "aspect": "ESRS E1 climate change — physical climate risk identification for own sites",
        "role": "primary vocabulary",
        "note": (
            "Evidence layer for the environmental pillar. Double materiality, "
            "GHG inventories, transition plans and governance disclosures are "
            "NOT covered by this pack."
        ),
    },
    {
        "id": "eu_taxonomy",
        "name": "EU Taxonomy DNSH climate adaptation",
        "aspect": "Physical-risk assessment vocabulary",
        "role": "risk vocabulary",
        "note": (
            "Reuses the same Appendix A hazard classification and claim-status "
            "language as the Green Finance Verification engine."
        ),
    },
    {
        "id": "california_sb261",
        "name": "California SB 261 — climate-related financial risk reports",
        "aspect": "Climate-related financial risk disclosure",
        "role": "expansion geography",
        "note": (
            "The physical-risk evidence in this pack is the relevant layer for "
            "climate-related financial risk reporting."
        ),
    },
    {
        "id": "california_sb253",
        "name": "California SB 253 — GHG emissions disclosure (Scope 1/2/3)",
        "aspect": "GHG emissions measurement and disclosure",
        "role": "tracked framework",
        "note": (
            "Emissions measurement is NOT covered by this pack. This is a "
            "declared boundary, not a gap in the engine."
        ),
    },
    {
        "id": "china_csds",
        "name": "China Corporate Sustainability Disclosure Standards (CSDS) — Basic Standard (Trial)",
        "aspect": "Ministry of Finance sustainability disclosure standard, in effect since 2025",
        "role": "expansion geography, vocabulary alignment",
        "note": (
            "Aligned with the 2024 SSE/SZSE/BSE sustainability reporting "
            "guidelines; physical climate-risk evidence is the relevant layer."
        ),
    },
]

ESRS_COVERAGE = [
    {
        "area": "ESRS 2 — Governance & strategy",
        "ref": "ESRS 2 GOV/SBM",
        "coverage": "not_covered",
        "note": "Company-declared fields only; Talaix does not verify governance or strategy disclosures.",
    },
    {
        "area": "ESRS E1 — Physical climate risk of sites (gross risk)",
        "ref": "ESRS E1 IRO-1 / MD-P-3",
        "coverage": "covered_by_evidence",
        "note": "Covered by the site-level physical hazard verification in this pack.",
    },
    {
        "area": "ESRS E1 — GHG emissions, targets and transition plan",
        "ref": "ESRS E1 MDR-P / MDR-T",
        "coverage": "not_covered",
        "note": "Requires company GHG inventories and transition planning; not provided by Talaix.",
    },
    {
        "area": "ESRS E2 — Pollution",
        "ref": "ESRS E2",
        "coverage": "not_covered",
        "note": "No pollutant emissions or discharge data is collected.",
    },
    {
        "area": "ESRS E3 — Water & marine resources",
        "ref": "ESRS E3",
        "coverage": "partial",
        "note": "Drought / water-stress physical evidence only; consumption or discharge metrics are not covered.",
    },
    {
        "area": "ESRS E4 — Biodiversity & ecosystems",
        "ref": "ESRS E4",
        "coverage": "not_covered",
        "note": "No biodiversity or ecosystem impact assessment is performed.",
    },
    {
        "area": "ESRS E5 — Resource use & circular economy",
        "ref": "ESRS E5",
        "coverage": "not_covered",
        "note": "No resource-flow or circular-economy metrics are collected.",
    },
    {
        "area": "ESRS S1–S4 — Social standards",
        "ref": "ESRS S1/S2/S3/S4",
        "coverage": "not_covered",
        "note": "Social and workforce disclosures are outside the scope of this pack.",
    },
    {
        "area": "ESRS G1 — Business conduct",
        "ref": "ESRS G1",
        "coverage": "not_covered",
        "note": "Governance and business-conduct disclosures are not verified.",
    },
]

EVIDENCE_STANDARD = {
    "name": "Talaix Evidence Standard",
    "criteria": [
        "Every claim carries a controlled-vocabulary claim status (OBSERVED, DOCUMENTED, REPORTED, MODELLED, INFERRED, UNKNOWN).",
        "Every evidence record states source, dataset, reference period and link where applicable.",
        "Every report is content-hashed (report id) and engine-versioned.",
        "Unavailable data is declared as gaps, never invented.",
        "Company-supplied fields are labelled 'declared by the company — not verified by Talaix'.",
    ],
    "not_accreditation": (
        "This label is a published methodology. It is NOT third-party accreditation, "
        "NOT assurance under ISAE 3000 or AA1000, and NOT the limited-assurance "
        "engagement that CSRD requires from an auditor or independent assurance provider."
    ),
}

DISCLAIMER = (
    "Talaix sells evidence, not accreditation. This pack provides physical "
    "climate-risk evidence for the company's listed sites. Company-supplied "
    "profile fields are not verified by Talaix. The pack does not constitute "
    "CSRD limited assurance, ISAE 3000 / AA1000 assurance, or investment advice."
)

HONESTY_CONTRACT = (
    "Unavailable data is declared, never invented: every missing disclosure "
    "area, unavailable hazard layer or unverified company field is explicitly "
    "stated in the coverage map or declared gaps."
)


def _validate_company(company: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(company, dict):
        raise ValueError("company must be an object")
    name = (company.get("name") or "").strip()
    if not name:
        raise ValueError("company.name is required")
    return {
        "name": name,
        "sector": (company.get("sector") or "").strip() or None,
        "country": (company.get("country") or "").strip() or None,
        "website": (company.get("website") or "").strip() or None,
        "description": (company.get("description") or "").strip() or None,
    }


def _trim_site_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Light site result: asset, ok, verification_id, summary, hazard_levels, declared_gaps."""
    verification = result.get("verification") if result.get("ok") else None
    labels: Dict[str, str] = {}
    if verification:
        for c in verification.get("hazard_checks", []):
            lvl = c.get("level") or {}
            if lvl.get("label"):
                labels[c.get("hazard", "unknown")] = lvl["label"]
    return {
        "asset": result.get("asset"),
        "ok": result.get("ok"),
        "error": result.get("error") if not result.get("ok") else None,
        "verification_id": verification.get("verification_id") if verification else None,
        "summary": verification.get("summary") if verification else None,
        "hazard_levels": labels,
        "declared_gaps": verification.get("declared_gaps") if verification else [],
    }


def build_sustainability_evidence(company: Dict[str, Any], assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build an integrated CSRD/ESRS-oriented sustainability evidence report.

    ``company`` must contain at least ``name``; other fields are treated as
    declared-by-company metadata. ``assets`` is passed to ``verify_portfolio``.
    """
    validated_company = _validate_company(company)
    company_block = {
        "fields": validated_company,
        "declared_by_company": True,
        "verification": "Company-supplied metadata — not verified by Talaix.",
    }

    site_results_raw = verify_portfolio(assets)
    trimmed_results = [_trim_site_result(r) for r in site_results_raw]

    ok_count = sum(1 for r in site_results_raw if r.get("ok"))
    total_declared_gaps = sum(
        len((r.get("verification") or {}).get("declared_gaps", []))
        for r in site_results_raw if r.get("ok")
    )

    # Aggregate highest levels across sites: label -> "hazard (site name or coords)"
    highest_levels: Dict[str, List[str]] = {}
    for r in trimmed_results:
        if not r.get("ok"):
            continue
        asset = r.get("asset") or {}
        site_label = asset.get("name") or f"{asset.get('lat')}, {asset.get('lon')}"
        for hazard, level_label in (r.get("hazard_levels") or {}).items():
            highest_levels.setdefault(level_label, []).append(f"{hazard} ({site_label})")

    portfolio_summary = {
        "site_count": len(site_results_raw),
        "ok_count": ok_count,
        "total_declared_gaps": total_declared_gaps,
        "highest_levels": highest_levels,
    }

    flat_gaps: List[Dict[str, Any]] = []
    for r in trimmed_results:
        asset = r.get("asset") or {}
        site_label = asset.get("name") or f"{asset.get('lat')}, {asset.get('lon')}"
        for gap in r.get("declared_gaps") or []:
            flat_gaps.append({
                "site": site_label,
                "hazard": gap.get("hazard"),
                "taxonomy_label": gap.get("taxonomy_label"),
                "reason": gap.get("reason"),
            })

    report_basis = {
        "company": validated_company,
        "site_verification_ids": [r.get("verification_id") for r in trimmed_results],
        "coverage": [c["coverage"] for c in ESRS_COVERAGE],
    }
    report_id = content_hash(report_basis)[:16]

    return {
        "report_id": report_id,
        "generated_at": utcnow_iso(),
        "engine_version": ENGINE_VERSION,
        "company": company_block,
        "coverage_map": list(ESRS_COVERAGE),
        "frameworks": list(SUSTAINABILITY_FRAMEWORKS),
        "evidence_standard": EVIDENCE_STANDARD,
        "portfolio_summary": portfolio_summary,
        "site_results": trimmed_results,
        "declared_gaps": flat_gaps,
        "disclaimer": DISCLAIMER,
        "honesty_contract": HONESTY_CONTRACT,
        "authenticity": issue_seal("sustainability", report_id, report_basis),
    }
