"""
CSRD readiness score and gap analysis.

The readiness score is a weighted composite of four components, each
0–100, with weights fixed and documented here (and mirrored in
docs/CSRD_TX_ENGINE.md):

- ``applicability_clarity`` (0.20) — how definitive the scope
  determination is.
- ``evidence_coverage`` (0.35) — how much of the applicable disclosure
  structure is backed by evidence rather than declared gaps.
- ``data_completeness`` (0.15) — how complete the supplied company
  profile and site data are.
- ``materiality_readiness`` (0.30) — how many topical standards have a
  materiality assessment at usable confidence.

The score is descriptive, never promotional: every point is traceable
to a component, and every component to its inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

WEIGHTS = {
    "applicability_clarity": 0.20,
    "evidence_coverage": 0.35,
    "data_completeness": 0.15,
    "materiality_readiness": 0.30,
}

# Coverage status → fraction credit.
_COVERAGE_CREDIT = {
    "evidence_backed": 1.0,
    "partial": 0.5,
    "none": 0.0,
}

_DETERMINATION_CLARITY = {
    "in_scope": 100.0,
    "out_of_scope": 100.0,
    "potentially_in_scope": 60.0,
    "requires_legal_confirmation": 30.0,
}


def _applicability_clarity(applicability: Dict[str, Any]) -> float:
    return _DETERMINATION_CLARITY.get(applicability.get("determination"), 30.0)


def _evidence_coverage(esrs_doc: Dict[str, Any]) -> Dict[str, Any]:
    total = 0
    credit = 0.0
    per_topic: Dict[str, str] = {}
    for topic in esrs_doc.get("topics", []):
        reqs = topic.get("disclosure_requirements", [])
        if not reqs:
            continue
        topic_credit = 0.0
        for req in reqs:
            total += 1
            status = req.get("talaix_evidence", "none")
            c = _COVERAGE_CREDIT.get(status, 0.0)
            credit += c
            topic_credit += c
        mean = topic_credit / len(reqs)
        per_topic[topic["id"]] = (
            "covered" if mean >= 0.99 else "partial" if mean > 0 else "not_covered"
        )
    score = round(100.0 * credit / total, 1) if total else 0.0
    return {"score": score, "requirement_count": total, "per_topic": per_topic}


def _data_completeness(company: Dict[str, Any], site_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields = [
        "name", "country", "sector", "employees",
        "net_turnover_eur", "balance_sheet_total_eur", "listed",
    ]
    provided = sum(1 for f in fields if company.get(f) not in (None, ""))
    profile_ratio = provided / len(fields)
    if site_results:
        ok_ratio = sum(1 for r in site_results if r.get("ok")) / len(site_results)
    else:
        ok_ratio = 0.0
    # Profile facts weigh 60%, resolved site data 40%.
    score = round(100.0 * (0.6 * profile_ratio + 0.4 * ok_ratio), 1)
    return {
        "score": score,
        "profile_fields_provided": provided,
        "profile_fields_total": len(fields),
        "sites_ok_ratio": round(ok_ratio, 3),
    }


def _materiality_readiness(materiality: List[Dict[str, Any]]) -> Dict[str, Any]:
    topics = [m for m in (materiality or []) if m.get("topic") not in (None, "ESRS2")]
    if not topics:
        return {"score": 0.0, "assessed": 0, "total": 0}
    usable = sum(1 for m in topics if m.get("confidence", 0.0) >= 0.5)
    score = round(100.0 * usable / len(topics), 1)
    return {"score": score, "assessed": usable, "total": len(topics)}


def compute_readiness(
    applicability: Dict[str, Any],
    esrs_doc: Dict[str, Any],
    company: Dict[str, Any],
    site_results: List[Dict[str, Any]],
    materiality: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Weighted 0–100 readiness composite with per-component breakdown."""
    clarity = _applicability_clarity(applicability)
    coverage = _evidence_coverage(esrs_doc)
    completeness = _data_completeness(company, site_results)
    mat = _materiality_readiness(materiality)

    components = {
        "applicability_clarity": clarity,
        "evidence_coverage": coverage["score"],
        "data_completeness": completeness["score"],
        "materiality_readiness": mat["score"],
    }
    overall = round(
        sum(components[k] * WEIGHTS[k] for k in WEIGHTS), 1
    )
    return {
        "overall": overall,
        "components": components,
        "weights": dict(WEIGHTS),
        "detail": {
            "evidence_coverage": coverage,
            "data_completeness": completeness,
            "materiality_readiness": mat,
        },
    }


_ACTION_BY_TOPIC = {
    "E1": "Extend the evidence pack: add GHG inventories (Scope 1/2/3), energy mix, targets and the transition plan from company records.",
    "E2": "Collect pollutant emission and discharge data for material sites.",
    "E3": "Add water consumption and discharge metrics per site; the physical drought/water-stress layer is already evidenced.",
    "E4": "Assess site proximity to protected areas (Natura 2000 / KBA) and biodiversity dependencies.",
    "E5": "Record resource inflows, outflows and waste streams.",
    "S1": "Compile own-workforce disclosures (working conditions, equal treatment).",
    "S2": "Map value-chain worker impacts; VSME-level requests are the cap for smaller suppliers.",
    "S3": "Assess affected-community impacts for material sites.",
    "S4": "Assess consumer and end-user impacts.",
    "G1": "Document business-conduct policies (corporate culture, supplier relationships, anti-corruption).",
}


def build_gap_analysis(
    applicability: Dict[str, Any],
    esrs_doc: Dict[str, Any],
    materiality: List[Dict[str, Any]],
    coverage_detail: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Itemised gaps, each with status, reason and a recommended action.

    Only gaps that matter for *this* company are listed: topics assessed
    non-material with usable confidence are noted as covered-by-assessment,
    not as gaps.
    """
    gaps: List[Dict[str, Any]] = []
    mat_by_topic = {m.get("topic"): m for m in (materiality or [])}
    per_topic = (coverage_detail or {}).get("per_topic", {})

    if applicability.get("determination") in ("potentially_in_scope", "requires_legal_confirmation"):
        gaps.append({
            "topic": "APPLICABILITY",
            "requirement": "CSRD scope determination",
            "status": "UNRESOLVED",
            "reason": "; ".join(applicability.get("reasons") or ["Determination unresolved."]),
            "recommended_action": "Provide the missing sizing facts (employees, turnover, balance sheet) or obtain legal confirmation.",
            "priority": "high",
        })

    for topic in esrs_doc.get("topics", []):
        topic_id = topic.get("id")
        if topic_id == "ESRS2":
            continue
        mat = mat_by_topic.get(topic_id)
        if mat and not mat.get("material") and mat.get("confidence", 0) >= 0.5:
            continue  # assessed non-material with usable confidence
        for req in topic.get("disclosure_requirements", []):
            status = req.get("talaix_evidence", "none")
            if status == "evidence_backed":
                continue
            gaps.append({
                "topic": topic_id,
                "requirement": f"{req.get('id')} — {req.get('name')}",
                "status": "PARTIAL" if status == "partial" else "UNAVAILABLE",
                "reason": req.get("note") or "No evidence collected for this disclosure requirement.",
                "recommended_action": _ACTION_BY_TOPIC.get(
                    topic_id, "Collect and document the underlying data."
                ),
                "priority": "high" if (mat and mat.get("material")) else "medium",
            })
    return gaps
