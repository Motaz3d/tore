"""
Talaix Visual Report Builder — deterministic draft engine.

Produces structured report drafts from the real engine payloads used by the
Green Finance Verification, Insurance and Sustainability products. Every line
of draft text is template-composed from payload values; no AI-generated prose
is used. Edited sections are tracked so the PDF layer can mark them honestly.

No Flask imports.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .evidence import content_hash, utcnow_iso
from .insurance import build_risk_profile
from .sustainability import build_sustainability_evidence
from .tx_seal import issue_seal
from .verification import verify_asset

REPORT_KINDS = {"verification", "insurance", "sustainability"}

_ALLOWED_SECTION_KINDS = {"introduction", "body", "gaps", "conclusion"}

ENGINE_VERSION = "1.0.0"

_HONESTY_NOTE = (
    "Engine text is template-composed from the cited evidence only; "
    "edited sections are marked in the exported PDF."
)

_INTERCONNECTION_NOTE = (
    "All sections in this draft describe the same underlying engine run; "
    "section source references point to the same verification, profile or report id."
)


def _format_coord(lat: Any, lon: Any) -> str:
    try:
        return f"{float(lat):.4f}, {float(lon):.4f}"
    except (TypeError, ValueError):
        return f"{lat}, {lon}"


def _asset_name(asset: Dict[str, Any]) -> str:
    return asset.get("name") or _format_coord(asset.get("lat"), asset.get("lon"))


def _evidence_refs(hazard_id: str, evidence: List[Dict[str, Any]]) -> List[str]:
    refs = [hazard_id]
    for rec in evidence:
        eid = rec.get("evidence_id") or rec.get("id")
        if eid:
            refs.append(f"{hazard_id}:{eid}")
    return refs


# -----------------------------------------------------------------------------
# Verification draft
# -----------------------------------------------------------------------------


def _verification_sections(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    asset = payload.get("asset") or {}
    checks = payload.get("hazard_checks") or []
    gaps = payload.get("declared_gaps") or []
    vid = payload.get("verification_id", "")

    sections: List[Dict[str, Any]] = []

    # Introduction
    intro_text = (
        f"This report assesses the physical climate hazards for the asset "
        f"'{_asset_name(asset)}' at coordinates {_format_coord(asset.get('lat'), asset.get('lon'))}. "
        f"The analysis was generated at {payload.get('generated_at')} using engine version "
        f"{payload.get('engine_version')}. {len(checks)} hazards are checked against the EU Taxonomy "
        f"Climate Delegated Act Appendix A vocabulary. {payload.get('disclaimer')}"
    )
    sections.append({
        "id": "intro",
        "kind": "introduction",
        "heading": "Introduction",
        "text": intro_text,
        "why": "Built from the verification payload scope, asset metadata and disclaimer.",
        "source_refs": [vid],
        "edited": False,
    })

    # Body: one section per hazard check
    for check in checks:
        hazard_id = check.get("hazard", "unknown")
        label = check.get("taxonomy_label") or hazard_id
        level = check.get("level") or {}
        level_label = level.get("label") or "—"
        evidence = check.get("evidence") or []
        limitations = check.get("limitations") or []

        text = (
            f"{label}: current level {level_label}. Claim status: {check.get('claim_status', 'UNKNOWN')}; "
            f"confidence: {check.get('confidence', '—')}. {check.get('summary', '')} "
            f"Evidence records: {len(evidence)}."
        )
        if limitations:
            text += " Limitations: " + "; ".join(limitations)

        sections.append({
            "id": f"hazard-{hazard_id}",
            "kind": "body",
            "heading": label,
            "text": text,
            "why": (
                f"Built from the {hazard_id} hazard module result: level '{level_label}', "
                f"claim status {check.get('claim_status', 'UNKNOWN')}, {len(evidence)} evidence record(s)."
            ),
            "source_refs": _evidence_refs(hazard_id, evidence),
            "edited": False,
        })

    # Declared gaps
    if gaps:
        gaps_text = "The following data gaps were declared for this asset:\n" + "\n".join(
            f"• {g.get('taxonomy_label') or g.get('hazard')} — {g.get('reason')}"
            for g in gaps
        )
    else:
        gaps_text = "No data gaps were declared for this asset."
    sections.append({
        "id": "gaps",
        "kind": "gaps",
        "heading": "Declared data gaps",
        "text": gaps_text,
        "why": "Built from the declared_gaps list in the verification payload.",
        "source_refs": [vid],
        "edited": False,
    })

    # Conclusion
    conclusion_text = (
        f"{payload.get('summary', '')} "
        f"Honesty contract: {payload.get('honesty_contract', '')} "
        f"{payload.get('monitoring_hint', '')}"
    ).strip()
    sections.append({
        "id": "conclusion",
        "kind": "conclusion",
        "heading": "Conclusion",
        "text": conclusion_text,
        "why": "Built from the verification summary, honesty contract and monitoring hint.",
        "source_refs": [vid],
        "edited": False,
    })

    return sections


def _build_verification_draft(params: Dict[str, Any]) -> Dict[str, Any]:
    lat = float(params.get("lat"))
    lon = float(params.get("lon"))
    name = (params.get("name") or "").strip() or None
    payload = verify_asset(lat, lon, name=name)
    sections = _verification_sections(payload)
    title = f"Physical Asset Verification — {name or _format_coord(lat, lon)}"
    return {
        "kind": "verification",
        "title": title,
        "engine_version": payload.get("engine_version", ENGINE_VERSION),
        "payload_id": payload.get("verification_id", ""),
        "disclaimer": payload.get("disclaimer", ""),
        "asset": payload.get("asset"),
        "sections": sections,
    }


# -----------------------------------------------------------------------------
# Insurance draft
# -----------------------------------------------------------------------------


def _insurance_sections(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    asset = payload.get("asset") or {}
    perils = payload.get("perils") or []
    gaps = payload.get("declared_gaps") or []
    pid = payload.get("profile_id", "")

    sections: List[Dict[str, Any]] = []

    intro_text = (
        f"This insurance environmental risk profile covers the asset "
        f"'{_asset_name(asset)}' at coordinates {_format_coord(asset.get('lat'), asset.get('lon'))}. "
        f"Event history is searched within a {payload.get('radius_km', 50)} km radius. "
        f"The profile was generated at {payload.get('generated_at')} using engine version "
        f"{payload.get('engine_version')}. {payload.get('disclaimer')}"
    )
    sections.append({
        "id": "intro",
        "kind": "introduction",
        "heading": "Introduction",
        "text": intro_text,
        "why": "Built from the insurance profile payload scope, asset metadata and disclaimer.",
        "source_refs": [pid],
        "edited": False,
    })

    for peril in perils:
        hazard_id = peril.get("hazard", "unknown")
        label = peril.get("peril") or hazard_id
        summary = peril.get("summary") or ""
        events_count = peril.get("events_count", 0)
        events_status = peril.get("events_status", "unavailable")
        basis = peril.get("level_basis") or ""
        limitations = peril.get("limitations") or []

        text = (
            f"{label}: current level {peril.get('current_level', '—')}. "
            f"Claim status: {peril.get('claim_status', 'UNKNOWN')}; confidence: {peril.get('confidence', '—')}. "
            f"{summary}"
        )
        if basis:
            text += f" Level basis: {basis}."
        text += (
            f" Long-term event records: {events_count} available "
            f"(events status: {events_status})."
        )
        if limitations:
            text += " Limitations: " + "; ".join(limitations)
        text += " " + payload.get("loss_quantification_note", "")

        sections.append({
            "id": f"peril-{hazard_id}",
            "kind": "body",
            "heading": label,
            "text": text,
            "why": (
                f"Built from the {hazard_id} insurance module result: current level "
                f"'{peril.get('current_level', '—')}', claim status {peril.get('claim_status', 'UNKNOWN')}, "
                f"{events_count} event record(s)."
            ),
            "source_refs": [hazard_id, f"{hazard_id}:events"],
            "edited": False,
        })

    if gaps:
        gaps_text = "Declared data gaps:\n" + "\n".join(
            f"• {g.get('peril')} ({g.get('type')}) — {g.get('reason')}"
            for g in gaps
        )
    else:
        gaps_text = "No declared data gaps."
    sections.append({
        "id": "gaps",
        "kind": "gaps",
        "heading": "Declared data gaps",
        "text": gaps_text,
        "why": "Built from the insurance payload declared_gaps list.",
        "source_refs": [pid],
        "edited": False,
    })

    conclusion_text = (
        f"{payload.get('exposure_summary', '')} Honesty contract: "
        f"{payload.get('honesty_contract', '')}"
    ).strip()
    sections.append({
        "id": "conclusion",
        "kind": "conclusion",
        "heading": "Conclusion",
        "text": conclusion_text,
        "why": "Built from the insurance exposure summary and honesty contract.",
        "source_refs": [pid],
        "edited": False,
    })

    return sections


def _build_insurance_draft(params: Dict[str, Any]) -> Dict[str, Any]:
    lat = float(params.get("lat"))
    lon = float(params.get("lon"))
    name = (params.get("name") or "").strip() or None
    radius_km = float(params.get("radius_km", 50.0))
    payload = build_risk_profile(lat, lon, name=name, radius_km=radius_km)
    sections = _insurance_sections(payload)
    title = f"Insurance Environmental Risk Profile — {name or _format_coord(lat, lon)}"
    return {
        "kind": "insurance",
        "title": title,
        "engine_version": payload.get("engine_version", ENGINE_VERSION),
        "payload_id": payload.get("profile_id", ""),
        "disclaimer": payload.get("disclaimer", ""),
        "asset": payload.get("asset"),
        "sections": sections,
    }


# -----------------------------------------------------------------------------
# Sustainability draft
# -----------------------------------------------------------------------------


def _sustainability_sections(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    company_block = payload.get("company") or {}
    fields = (company_block.get("fields") or {})
    rid = payload.get("report_id", "")
    site_results = payload.get("site_results") or []
    coverage = payload.get("coverage_map") or []
    gaps = payload.get("declared_gaps") or []
    summary = payload.get("portfolio_summary") or {}

    sections: List[Dict[str, Any]] = []

    intro_text = (
        f"This sustainability evidence report covers the company '{fields.get('name', '—')}'. "
        f"Company metadata (sector, country, website, description) is declared by the company and "
        f"not verified by Talaix. {len(site_results)} site(s) were analysed with the physical "
        f"hazard verification engine. {payload.get('disclaimer')}"
    )
    sections.append({
        "id": "intro",
        "kind": "introduction",
        "heading": "Introduction",
        "text": intro_text,
        "why": "Built from the sustainability company block and disclaimer.",
        "source_refs": [rid],
        "edited": False,
    })

    # Coverage map section
    covered = [c for c in coverage if c.get("coverage") == "covered_by_evidence"]
    partial = [c for c in coverage if c.get("coverage") == "partial"]
    not_covered = [c for c in coverage if c.get("coverage") not in ("covered_by_evidence", "partial")]
    coverage_text = (
        f"Disclosure coverage map: {len(covered)} area(s) covered by evidence, "
        f"{len(partial)} partial, {len(not_covered)} not covered. "
        "This report provides physical climate-risk evidence for the company's sites; "
        "GHG emissions, transition plans, governance and social disclosures are outside scope."
    )
    sections.append({
        "id": "coverage-map",
        "kind": "body",
        "heading": "Disclosure coverage map",
        "text": coverage_text,
        "why": "Built from the ESRS_COVERAGE map in the sustainability payload.",
        "source_refs": [rid, "ESRS_COVERAGE"],
        "edited": False,
    })

    # Per-site sections
    for result in site_results:
        asset = result.get("asset") or {}
        site_label = asset.get("name") or _format_coord(asset.get("lat"), asset.get("lon"))
        levels = result.get("hazard_levels") or {}
        level_text = ", ".join(f"{h}: {lvl}" for h, lvl in levels.items()) or "no levels available"
        site_text = (
            f"Site '{site_label}': analysis {'successful' if result.get('ok') else 'failed'}. "
            f"Top hazard levels: {level_text}. "
            f"Declared gaps: {len(result.get('declared_gaps') or [])}."
        )
        sections.append({
            "id": f"site-{result.get('verification_id', site_label)}",
            "kind": "body",
            "heading": f"Site: {site_label}",
            "text": site_text,
            "why": "Built from the site-level verification result in the sustainability payload.",
            "source_refs": [rid, result.get("verification_id", "")],
            "edited": False,
        })

    # Gaps
    if gaps:
        gaps_text = "Declared data gaps across the portfolio:\n" + "\n".join(
            f"• {g.get('site')} — {g.get('taxonomy_label') or g.get('hazard')}: {g.get('reason')}"
            for g in gaps
        )
    else:
        gaps_text = "No declared data gaps across the portfolio."
    sections.append({
        "id": "gaps",
        "kind": "gaps",
        "heading": "Declared data gaps",
        "text": gaps_text,
        "why": "Built from the flattened declared_gaps list in the sustainability payload.",
        "source_refs": [rid],
        "edited": False,
    })

    # Conclusion
    conclusion_text = (
        f"Portfolio summary: {summary.get('site_count', 0)} site(s), "
        f"{summary.get('ok_count', 0)} with real data, "
        f"{summary.get('total_declared_gaps', 0)} declared gap(s). "
        f"Honesty contract: {payload.get('honesty_contract', '')}"
    )
    sections.append({
        "id": "conclusion",
        "kind": "conclusion",
        "heading": "Conclusion",
        "text": conclusion_text,
        "why": "Built from the sustainability portfolio summary and honesty contract.",
        "source_refs": [rid],
        "edited": False,
    })

    return sections


def _build_sustainability_draft(params: Dict[str, Any]) -> Dict[str, Any]:
    company = params.get("company")
    assets = params.get("assets")
    if not isinstance(company, dict) or not isinstance(assets, list):
        raise ValueError("sustainability params require company object and assets list")
    payload = build_sustainability_evidence(company, assets)
    sections = _sustainability_sections(payload)
    title = f"Sustainability Evidence Report — {company.get('name', 'Unknown')}"
    return {
        "kind": "sustainability",
        "title": title,
        "engine_version": payload.get("engine_version", ENGINE_VERSION),
        "payload_id": payload.get("report_id", ""),
        "disclaimer": payload.get("disclaimer", ""),
        "sections": sections,
    }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def build_draft(kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a structured report draft for the given product kind.

    ``params`` must match the target engine:
      - verification: {lat, lon, name?}
      - insurance:    {lat, lon, name?, radius_km?}
      - sustainability: {company: {...}, assets: [...]}
    """
    if kind not in REPORT_KINDS:
        raise ValueError(f"kind must be one of: {', '.join(sorted(REPORT_KINDS))}")

    if kind == "verification":
        result = _build_verification_draft(params)
    elif kind == "insurance":
        result = _build_insurance_draft(params)
    else:
        result = _build_sustainability_draft(params)

    sections = result["sections"]
    draft_basis = {
        "kind": kind,
        "params": params,
        "section_texts": [(s.get("id"), s.get("text")) for s in sections],
    }
    draft_id = content_hash(draft_basis)[:16]

    return {
        "draft_id": draft_id,
        "kind": result["kind"],
        "title": result["title"],
        "generated_at": utcnow_iso(),
        "engine_version": result["engine_version"],
        "payload_id": result["payload_id"],
        "disclaimer": result["disclaimer"],
        "asset": result.get("asset"),
        "sections": sections,
        "interconnection_note": _INTERCONNECTION_NOTE,
        "honesty_note": _HONESTY_NOTE,
        "authenticity": issue_seal("report-builder", draft_id, draft_basis),
    }


def prepare_sections(sections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Validate and clean user-submitted sections for PDF export.

    Returns (cleaned_sections, edited_count). Raises ValueError on malformed
    input or if the section count exceeds 60.
    """
    if not isinstance(sections, list):
        raise ValueError("sections must be a list")
    if len(sections) > 60:
        raise ValueError("sections cannot exceed 60")

    cleaned: List[Dict[str, Any]] = []
    edited_count = 0

    for idx, s in enumerate(sections):
        if not isinstance(s, dict):
            raise ValueError(f"section at index {idx} must be an object")

        heading = str(s.get("heading") or "").strip()
        text = str(s.get("text") or "").strip()

        if not heading and not text:
            continue  # drop fully empty slots
        if not heading:
            raise ValueError(f"section at index {idx} is missing a heading")
        if not text:
            continue  # drop empty-body slots

        if len(heading) > 200:
            raise ValueError(f"section heading at index {idx} exceeds 200 characters")
        if len(text) > 5000:
            raise ValueError(f"section text at index {idx} exceeds 5000 characters")

        kind = s.get("kind", "body")
        if kind not in _ALLOWED_SECTION_KINDS:
            raise ValueError(f"section kind at index {idx} must be one of: introduction, body, gaps, conclusion")

        edited = bool(s.get("edited"))
        ai_polished = bool(s.get("ai_polished"))
        if edited:
            edited_count += 1

        cleaned.append({
            "id": s.get("id", f"section-{idx}"),
            "kind": kind,
            "heading": heading,
            "text": text,
            "why": str(s.get("why") or ""),
            "source_refs": list(s.get("source_refs") or []),
            "edited": edited,
            "ai_polished": ai_polished,
        })

    return cleaned, edited_count
