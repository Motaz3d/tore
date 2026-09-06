"""
Talaix Funding Intelligence engine (docs/SUSTAINABILITY_INTELLIGENCE.md).

Matches real caller context (hazards, sector, beneficiary type, country,
solution characteristics) against the curated funding knowledge base
(``config/funding_knowledge.json`` — real programmes with official URLs).

Honesty contract (same discipline as src/climate/solutions.py):

- No invented amounts, rates, deadlines, eligibility or programme status.
  The KB carries ``not stated`` / ``not currently verified`` for volatile
  facts; the engine passes that through and lists it under
  ``not_verified``.
- Every match explains ``why_it_matches`` from the caller's own inputs —
  never generic filler.
- Funding is not free money: the funding_type (grant / loan / equity /
  guarantee / blended / technical assistance / …) is always shown.
- No match without evidence; no guarantee of funding.

Matching is a declared, deterministic screen, not a score of merit:

- gate 1 — hazard overlap (requested hazards ∩ programme hazards);
- gate 2 — beneficiary type (when given) must be listed;
- jurisdiction — EU programmes match EU member states; the global
  adaptation funds target developing countries and are excluded for EU
  members (and vice versa), with the rule stated; unknown country →
  jurisdiction reported as unverified, never assumed;
- fit_score = matched_dimensions / relevant_dimensions over the optional
  context dimensions (sector, sustainability objective, adaptation/
  mitigation, nature-based, technology, solution class link) — declared
  and explainable, like the solutions engine.

The module performs no network I/O.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_DEFAULT_FUNDING_KB = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "funding_knowledge.json"
)

DISCLAIMER = (
    "Potential funding sources only — eligibility requires verification at "
    "the official source. This is not financial advice; no funding is "
    "guaranteed."
)

#: EU member states (2026) — used only for the declared jurisdiction rule.
EU_MEMBER_STATES = frozenset({
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",
})

#: Jurisdiction classes in the KB and how they match a caller's country.
_EU_JURISDICTIONS = ("EU", "EU + associated countries", "EU regions",
                     "EU member states (national/regional programmes)",
                     "EU member states (national CAP strategic plans)",
                     "EU member states meeting the Cohesion eligibility criterion")
_GLOBAL_DEV_JURISDICTIONS = ("developing countries (UNFCCC)",
                             "developing countries and economies in transition",
                             "client countries",
                             "ADB developing member countries (Asia-Pacific)",
                             "AIIB member countries (Asia-Pacific and beyond)")


def load_funding_knowledge(path: Optional[str] = None) -> Dict:
    kb_path = path or os.environ.get("HYDRASHIELD_FUNDING_KB") or _DEFAULT_FUNDING_KB
    with open(kb_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _jurisdiction_check(programme: Dict, country: Optional[str]) -> Dict[str, Any]:
    """Declared jurisdiction rule. Returns a descriptor with `match`
    (True/False/None=unverified) and the human-readable basis."""
    jurisdiction = programme.get("jurisdiction", "")
    if not country:
        return {"match": None, "basis": "caller country unknown — "
                f"jurisdiction '{jurisdiction}' not verified"}
    code = country.strip().upper()
    if jurisdiction in _EU_JURISDICTIONS or jurisdiction.startswith("EU"):
        if code in EU_MEMBER_STATES:
            return {"match": True, "basis": f"{code} is an EU member state; "
                    f"programme jurisdiction: {jurisdiction}"}
        return {"match": False, "basis": f"{code} is not an EU member state; "
                f"programme jurisdiction: {jurisdiction}"}
    if jurisdiction in _GLOBAL_DEV_JURISDICTIONS:
        if code in EU_MEMBER_STATES:
            return {"match": False, "basis": f"programme targets {jurisdiction}; "
                    f"{code} (EU member) is not the target group"}
        return {"match": True, "basis": f"programme targets {jurisdiction}; "
                "country-level eligibility still requires verification"}
    return {"match": None, "basis": f"jurisdiction '{jurisdiction}' not classified"}


def _match_programme(programme: Dict, query: Dict) -> Optional[Dict]:
    """Evaluate one programme. Returns the match descriptor or None when a
    hard gate fails (hazard/beneficiary/jurisdiction)."""
    p_hazards = set(programme.get("hazards") or [])
    q_hazards = set(query.get("hazards") or [])
    hazard_overlap = sorted(p_hazards & q_hazards)
    if q_hazards and not hazard_overlap:
        return None

    beneficiary = query.get("beneficiary")
    if beneficiary and beneficiary not in (programme.get("beneficiary_types") or []):
        return None

    juris = _jurisdiction_check(programme, query.get("country"))
    if juris["match"] is False:
        return None

    matched: List[str] = []
    relevant: List[str] = []
    if q_hazards:
        relevant.append("hazards")
        if hazard_overlap:
            matched.append(f"hazard overlap: {', '.join(hazard_overlap)}")
    sector = query.get("sector")
    if sector:
        relevant.append("sector")
        if sector in (programme.get("sector") or []):
            matched.append(f"sector '{sector}' is a programme target sector")
    objective = query.get("objective")
    if objective:
        relevant.append("sustainability objective")
        if objective in (programme.get("sustainability_objectives") or []):
            matched.append(f"objective '{objective}' matches the programme scope")
    role = query.get("role")  # adaptation | mitigation
    if role:
        relevant.append("adaptation/mitigation")
        if programme.get(role):
            matched.append(f"programme supports {role}")
    if query.get("nature_based") is True:
        relevant.append("nature-based")
        if programme.get("nature_based"):
            matched.append("nature-based approaches are in scope")
    if query.get("technology") is True:
        relevant.append("technology")
        if programme.get("technology"):
            matched.append("technology components are in scope")
    if juris["match"] is not None:
        relevant.append("jurisdiction")
        if juris["match"]:
            matched.append(juris["basis"])

    fit_score = round(len(matched) / len(relevant), 3) if relevant else 1.0
    not_verified = [f"{field}: {programme.get(field)}"
                    for field in ("funding_amount", "funding_rate", "deadline")
                    if programme.get(field) in ("not stated", "not currently verified")]
    if juris["match"] is None:
        not_verified.append("jurisdiction: " + juris["basis"])

    return {
        "id": programme["id"],
        "name": programme["name"],
        "programme": programme.get("programme"),
        "funding_body": programme.get("funding_body"),
        "funding_type": programme.get("funding_type"),
        "jurisdiction": programme.get("jurisdiction"),
        "why_it_matches": "; ".join(matched) if matched else
            "matches only the hard gates (hazard/beneficiary); no optional "
            "context dimensions were matched",
        "fit_score": fit_score,
        "fit": {"scoring": "matched_dimensions / relevant_dimensions",
                "dimensions_matched": matched,
                "dimensions_relevant": relevant},
        "what_is_supported": programme.get("sustainability_objectives") or [],
        "who_may_apply": programme.get("beneficiary_types") or [],
        "eligibility": programme.get("eligibility"),
        "not_verified": not_verified,
        "deadline": programme.get("deadline", "not currently verified"),
        "official_url": programme.get("official_url"),
        "source": programme.get("source"),
        "date_checked": programme.get("date_checked"),
        "confidence": programme.get("confidence"),
        "limitations": programme.get("limitations"),
        "hydrashield_relevance": programme.get("hydrashield_relevance"),
        "recommended_action": programme.get("recommended_action"),
    }


def match_funding(query: Dict, knowledge_path: Optional[str] = None) -> Dict[str, Any]:
    """Match funding opportunities against caller context.

    ``query``: hazards (list), sector, beneficiary, country (ISO-2),
    objective (sustainability objective), role (adaptation|mitigation),
    nature_based (bool), technology (bool). Missing context narrows the
    explanation, never fabricates a match.
    """
    query = dict(query or {})
    if not query.get("hazards") and not query.get("sector") and \
            not query.get("objective"):
        return {
            "status": "insufficient_data",
            "message": "Provide at least hazards, a sector or a sustainability "
                       "objective — funding matching is evidence-gated.",
            "matches": [],
            "disclaimer": DISCLAIMER,
        }
    try:
        kb = load_funding_knowledge(knowledge_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unavailable",
                "message": f"Funding knowledge base unavailable: {exc}",
                "matches": [], "disclaimer": DISCLAIMER}

    matches = []
    for programme in kb.get("programmes") or []:
        result = _match_programme(programme, query)
        if result is not None:
            matches.append(result)
    matches.sort(key=lambda m: (-m["fit_score"], m["id"]))

    return {
        "status": "ok" if matches else "no_match",
        "query": {k: v for k, v in query.items() if v not in (None, [], "")},
        "matches": matches,
        "disclaimer": DISCLAIMER,
        "provenance": {
            "kind": "derived",
            "source": "Talaix funding engine: caller context + curated "
                      "funding knowledge base (config/funding_knowledge.json)",
            "limitations": "Declared screening over curated programme "
                           "records; volatile facts (amounts, rates, "
                           "deadlines) are not stated unless officially "
                           "published — verify at the official source.",
        },
    }
