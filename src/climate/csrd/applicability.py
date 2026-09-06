"""
CSRD applicability engine — deterministic scope screening.

Answers: is this company in scope of the CSRD for a given reporting
year, under which rule set, in which phase-in wave — and would it
remain in scope under the proposed Omnibus I restriction?

Determination vocabulary:

- ``in_scope`` — the rule set clearly captures the company.
- ``out_of_scope`` — clearly below every threshold.
- ``potentially_in_scope`` — some criteria met, others unknown or
  borderline; more data would settle it.
- ``requires_legal_confirmation`` — the determination hinges on legal
  facts the engine cannot verify (group structure, NFRD history,
  public-interest status). Always the answer for borderline cases:
  screening, never legal advice.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .regulations import (
    STATUS_ADOPTED_PENDING,
    STATUS_PROPOSED,
    rule_set_for_year,
    wave_calendar,
)

DETERMINATIONS = (
    "in_scope",
    "out_of_scope",
    "potentially_in_scope",
    "requires_legal_confirmation",
)

# EU-27 member states (English names + ISO-3166 alpha-2 codes).
_EU27_NAMES = {
    "austria", "belgium", "bulgaria", "croatia", "cyprus", "czechia",
    "czech republic", "denmark", "estonia", "finland", "france", "germany",
    "greece", "hungary", "ireland", "italy", "latvia", "lithuania",
    "luxembourg", "malta", "netherlands", "poland", "portugal", "romania",
    "slovakia", "slovenia", "spain", "sweden",
}
_EU27_CODES = {
    "at", "be", "bg", "hr", "cy", "cz", "dk", "ee", "fi", "fr", "de", "gr",
    "el", "hu", "ie", "it", "lv", "lt", "lu", "mt", "nl", "pl", "pt", "ro",
    "sk", "si", "es", "se",
}


def _is_eu_country(country: Optional[str]) -> Optional[bool]:
    """True/False for EU membership of ``country``; None if unknown."""
    if not country:
        return None
    c = country.strip().lower()
    if c in ("eu", "european union"):
        return True
    if c in _EU27_NAMES or c in _EU27_CODES:
        return True
    # A non-empty country that is not EU-27 is treated as non-EU.
    return False


def normalise_profile(company: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise the company dict into an applicability profile.

    Accepts the website form fields plus optional sizing facts. Missing
    numeric facts stay ``None`` — they are never guessed.
    """
    if not isinstance(company, dict):
        raise ValueError("company must be an object")
    name = (company.get("name") or "").strip()
    if not name:
        raise ValueError("company.name is required")

    def _num(key: str) -> Optional[float]:
        value = company.get(key)
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"company.{key} must be a number")

    reporting_year = company.get("reporting_year")
    if reporting_year in (None, ""):
        reporting_year = None
    else:
        try:
            reporting_year = int(reporting_year)
        except (TypeError, ValueError):
            raise ValueError("company.reporting_year must be an integer")

    return {
        "name": name,
        "country": (company.get("country") or "").strip() or None,
        "sector": (company.get("sector") or "").strip() or None,
        "website": (company.get("website") or "").strip() or None,
        "lei": (company.get("lei") or "").strip() or None,
        "eu_established": company.get("eu_established"),
        "employees": _num("employees"),
        "net_turnover_eur": _num("net_turnover_eur"),
        "balance_sheet_total_eur": _num("balance_sheet_total_eur"),
        "listed": company.get("listed"),
        "public_interest": company.get("public_interest"),
        "previously_nfrd": company.get("previously_nfrd"),
        "non_eu_eu_turnover_eur": _num("non_eu_eu_turnover_eur"),
        "has_eu_branch_or_subsidiary": company.get("has_eu_branch_or_subsidiary"),
        "reporting_year": reporting_year,
    }


def _size_criteria(profile: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the Accounting Directive size criteria.

    Each criterion is ``met`` / ``not_met`` / ``unknown``; the outcome is
    computed only over known criteria and flagged when unknowns could
    still flip it. Two models are supported:

    - default: "large" = at least ``size_criteria_required`` of the three
      criteria exceeded (the Accounting Directive two-of-three test);
    - ``employees_mandatory`` (Omnibus proposal): the employee threshold
      is a hard gate, plus at least one financial criterion.
    """
    eu = rules["eu_undertaking"]
    checks = {
        "employees": (profile["employees"], eu.get("large_min_employees")),
        "net_turnover_eur": (profile["net_turnover_eur"], eu.get("turnover_eur")),
        "balance_sheet_total_eur": (
            profile["balance_sheet_total_eur"],
            eu.get("balance_sheet_eur"),
        ),
    }
    criteria: Dict[str, str] = {}
    met_count = 0
    unknown_count = 0
    for key, (value, threshold) in checks.items():
        if value is None or threshold is None:
            criteria[key] = "unknown"
            unknown_count += 1
        elif value > threshold:
            criteria[key] = "met"
            met_count += 1
        else:
            criteria[key] = "not_met"

    if eu.get("employees_mandatory"):
        emp = criteria["employees"]
        fin_met = sum(
            1 for k in ("net_turnover_eur", "balance_sheet_total_eur")
            if criteria[k] == "met"
        )
        fin_unknown = sum(
            1 for k in ("net_turnover_eur", "balance_sheet_total_eur")
            if criteria[k] == "unknown"
        )
        if emp == "met" and fin_met >= 1:
            outcome = "large"
        elif emp == "not_met" or (emp == "met" and fin_met + fin_unknown < 1):
            outcome = "not_large"
        else:
            outcome = "undetermined"
        return {
            "criteria": criteria,
            "met_count": met_count,
            "unknown_count": unknown_count,
            "required": "employees + 1 financial (employees mandatory)",
            "outcome": outcome,
        }

    required = eu.get("size_criteria_required", 2)
    if met_count >= required:
        outcome = "large"
    elif met_count + unknown_count < required:
        outcome = "not_large"
    else:
        outcome = "undetermined"
    return {
        "criteria": criteria,
        "met_count": met_count,
        "unknown_count": unknown_count,
        "required": required,
        "outcome": outcome,
    }


def _wave_for(
    profile: Dict[str, Any],
    rules: Dict[str, Any],
    size: Dict[str, Any],
    eu_member: Optional[bool],
) -> Optional[Dict[str, Any]]:
    """Map the profile to a phase-in wave, if any."""
    calendar = {w["wave"]: w for w in wave_calendar()}
    eu_rules = rules["eu_undertaking"]

    if eu_member is False:
        non_eu = rules.get("non_eu_undertaking") or {}
        threshold = non_eu.get("eu_turnover_eur")
        eu_turnover = profile["non_eu_eu_turnover_eur"]
        if (
            threshold is not None
            and eu_turnover is not None
            and eu_turnover > threshold
            and profile["has_eu_branch_or_subsidiary"]
        ):
            return calendar.get(4)
        return None

    # EU-established.
    if (
        (profile["previously_nfrd"] or profile["public_interest"])
        and profile["employees"] is not None
        and profile["employees"] > eu_rules.get("public_interest_employees", 500)
    ):
        return calendar.get(1)
    if size["outcome"] == "large":
        return calendar.get(2)
    if (
        profile["listed"]
        and eu_rules.get("listed_sme_in_scope")
        and size["outcome"] in ("not_large", "undetermined")
    ):
        return calendar.get(3)
    return None


def _evaluate_with_rules(profile: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Core deterministic evaluation against one rule set."""
    assumptions: List[str] = []
    reasons: List[str] = []

    eu_member = _is_eu_country(profile["country"])
    if profile["eu_established"] is not None:
        eu_member = bool(profile["eu_established"])
    elif eu_member is None:
        assumptions.append(
            "Establishment unknown (no country given): the company was "
            "evaluated under both EU and non-EU rules where data allowed."
        )

    size = _size_criteria(profile, rules)
    wave = _wave_for(profile, rules, size, eu_member)

    if wave is not None:
        determination = "in_scope"
        reasons.append(
            f"Captured by wave {wave['wave']}: {wave['population']} "
            f"(first reporting year {wave['first_reporting_year']}, "
            f"first report {wave['first_report_year']})."
        )
        if profile["employees"] is None and wave["wave"] in (1, 2, 3):
            determination = "requires_legal_confirmation"
            reasons.append("Employee count is missing; wave assignment needs confirmation.")
    else:
        if eu_member is None:
            determination = "requires_legal_confirmation"
            reasons.append("Establishment (EU vs non-EU) could not be determined.")
        elif size["outcome"] == "undetermined" and eu_member:
            determination = "potentially_in_scope"
            reasons.append(
                f"Size test undetermined: {size['met_count']} of {size['required']} "
                "criteria met, one or more criteria unknown."
            )
        elif eu_member is False:
            non_eu = rules.get("non_eu_undertaking") or {}
            threshold = non_eu.get("eu_turnover_eur")
            eu_turnover = profile["non_eu_eu_turnover_eur"]
            if eu_turnover is None:
                determination = "potentially_in_scope"
                reasons.append(
                    "Non-EU company without declared EU turnover: Article 40a "
                    "scope (€150M EU turnover over two consecutive years plus an "
                    "EU branch or subsidiary) cannot be excluded."
                )
            elif (
                threshold is not None
                and eu_turnover > threshold
                and not profile["has_eu_branch_or_subsidiary"]
            ):
                determination = "potentially_in_scope"
                reasons.append(
                    "EU turnover above the Article 40a threshold, but an EU "
                    "branch or subsidiary is not confirmed."
                )
            else:
                determination = "out_of_scope"
                reasons.append("Below every threshold of the applicable rule set.")
        else:
            determination = "out_of_scope"
            reasons.append("Below every threshold of the applicable rule set.")

    return {
        "determination": determination,
        "wave": wave,
        "size_evaluation": size,
        "reasons": reasons,
        "assumptions": assumptions,
    }


def assess_applicability(company: Dict[str, Any]) -> Dict[str, Any]:
    """Assess CSRD applicability for a company profile.

    Returns the determination under the in-force rule set for the
    requested reporting year (default: the current wave-relevant year),
    plus a forward outlook under proposed rules (Omnibus I) and the
    VSME voluntary route when out of scope.
    """
    profile = normalise_profile(company)
    reporting_year = profile["reporting_year"] or 2027  # wave-2 anchor year
    profile["reporting_year"] = reporting_year

    rules = rule_set_for_year(reporting_year)
    result = _evaluate_with_rules(profile, rules)

    # Forward outlook under the proposed Omnibus restriction (never applied,
    # always labelled proposed). Reporting years outside the proposal's
    # window simply have no forward outlook.
    try:
        omnibus = rule_set_for_year(reporting_year, statuses=(STATUS_PROPOSED,))
        omnibus_eval = _evaluate_with_rules(profile, omnibus)
        forward_outlook = {
            "rule_set_id": omnibus["id"],
            "rule_set_status": omnibus["status"],
            "determination_if_adopted": omnibus_eval["determination"],
            "note": (
                "Proposed rules are reported for forward planning only and are "
                "never applied to the determination."
            ),
        }
    except ValueError:
        forward_outlook = {
            "rule_set_id": None,
            "rule_set_status": None,
            "determination_if_adopted": None,
            "note": (
                f"No proposed rule set covers reporting year {reporting_year}; "
                "the in-force determination stands."
            ),
        }

    voluntary = None
    if result["determination"] == "out_of_scope":
        voluntary = {
            "route": "vsme_voluntary",
            "status": STATUS_ADOPTED_PENDING,
            "note": (
                "Below CSRD scope. The VSME voluntary standard is the "
                "proportionate reporting route and doubles as the value-chain "
                "cap: larger partners may not request information beyond what "
                "VSME covers."
            ),
        }

    return {
        "company": profile,
        "determination": result["determination"],
        "rule_set": {
            "id": rules["id"],
            "name": rules["name"],
            "status": rules["status"],
            "source": rules.get("source"),
        },
        "reporting_year": reporting_year,
        "wave": result["wave"],
        "size_evaluation": result["size_evaluation"],
        "reasons": result["reasons"],
        "assumptions": result["assumptions"],
        "forward_outlook": forward_outlook,
        "voluntary_route": voluntary,
        "honesty_note": (
            "Screening determination from declared company facts — not legal "
            "advice. Borderline outcomes are returned as "
            "'requires_legal_confirmation' by design."
        ),
    }
