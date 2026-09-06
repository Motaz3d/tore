"""
Talaix Solutions Intelligence engine (docs/SOLUTIONS_INTELLIGENCE.md).

Matches REAL site conditions (hazards, climate, terrain, land cover, water,
mapped infrastructure, historical events) against the curated, sourced
solutions knowledge base (``config/solutions_knowledge.json`` — generic
solution CLASSES, reference knowledge, not an observation).

Honesty contract (mirrors src/dashboard/ecology.py):

    - ``why_it_fits`` quotes the real site values that drove the match —
      never generic filler.
    - No invented products, vendors or benefits; ``expected_benefit`` is
      qualitative unless the KB carries a documented quantification (none
      does today — every entry declares ``quantified: false``).
    - Every recommended solution states limitations and repeats the
      no-guarantee disclaimer.
    - When the site inputs are missing, the engine takes the honest
      ``insufficient_data`` path and lists what would sharpen the fit.

Fit scoring is deliberately simple and declared:
``fit_score = conditions_matched / conditions_relevant`` over the
applicability conditions the KB entry declares (hazard match is the gate,
not a scored condition). A condition whose site value is missing counts as
relevant but not matched, and is reported as unverified. A condition whose
site value is present and VIOLATES a hard constraint (climate zone,
elevation range, land cover, water, urban presence) excludes the solution
for that hazard.

``fit_band`` is a declared label over the score: ``high`` when every
declared condition was verified as matched (score >= 0.99), ``moderate``
for score >= 0.5, ``low`` below that, and ``hazard_match_only`` when the
entry declares no site conditions at all (relevant == 0 — nothing was
verified beyond the hazard gate).

The module performs no network I/O; site assembly lives in the API layer.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .evidence import EvidenceRecord
from .ontology import ClaimStatus, Confidence, TemporalClass

_DEFAULT_KB = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "solutions_knowledge.json"
)

GUARANTEE_DISCLAIMER = "No solution guarantees prevention of an event."

#: Declared fit-band thresholds over fit_score (see module docstring).
FIT_BANDS = {"high": 0.99, "moderate": 0.5}

#: Declared future monetary fields (docs/SOLUTIONS_INTELLIGENCE.md §5).
#: Every one is "not_quantified" until a documented value with method,
#: assumptions and source exists. Never fabricate ROI.
RESILIENCE_ECONOMICS = {
    "status": "not_quantified",
    "fields": {
        "adaptation_cost": "not_quantified",
        "avoided_loss": "not_quantified",
        "resilience_investment": "not_quantified",
        "maintenance_cost": "not_quantified",
        "business_interruption_reduction": "not_quantified",
    },
    "rule": "Any future monetary value must carry a documented source, "
            "method, assumptions, currency, year and uncertainty — "
            "otherwise the field stays 'not_quantified'.",
}

INSUFFICIENT_DATA_MESSAGE = (
    "Site data is insufficient to fit solutions reliably; see missing_inputs "
    "for what would sharpen the fit."
)

#: Urban-presence threshold on mapped OSM buildings — matches the exposure
#: layer's "moderate" boundary (src/dashboard/exposure.py).
_URBAN_BUILDINGS_THRESHOLD = 20

#: Canonical site inputs; missing ones are listed in ``insufficient_data``.
_INPUT_LABELS = {
    "hazards": "active hazard levels (call /api/v2/analyze?hazard=… per hazard)",
    "climate_zone": "climate zone (recent weather window)",
    "moisture_regime": "moisture regime (soil moisture / recent precipitation)",
    "elevation_m": "elevation (DEM)",
    "landcover_classes": "land-cover classes (ESA WorldCover)",
    "water_features_count": "mapped water features (OSM)",
    "buildings_count": "mapped buildings (OSM population/urban proxy)",
    "historical_events": "historical event record for this location",
}


def load_solutions_knowledge(path: Optional[str] = None) -> Dict:
    kb_path = path or os.environ.get("HYDRASHIELD_SOLUTIONS_KB") or _DEFAULT_KB
    with open(kb_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.1f}" if value != int(value) else str(int(value))
    return str(value)


def _check_applicability(
    entry: Dict,
    site: Dict,
) -> Tuple[bool, int, int, List[str], List[str], List[str]]:
    """Evaluate one KB entry's declared applicability conditions against the
    real site values.

    Returns ``(excluded, matched, relevant, reasons_for, unverified,
    missing)``. Conditions the entry does not declare are irrelevant.
    Missing site values never exclude — they count as unverified and feed
    the ``insufficient_data`` block. Present-but-violated hard constraints
    (climate zone, elevation range, land cover, water, urban presence)
    exclude the solution; moisture-regime mismatch only lowers the score.
    """
    app = entry.get("applicability") or {}
    matched = 0
    relevant = 0
    reasons_for: List[str] = []
    unverified: List[str] = []
    missing: List[str] = []

    def _need(key: str, label: str) -> Tuple[bool, Any]:
        value = site.get(key)
        if value is None:
            unverified.append(f"{label} not available — condition unverified")
            missing.append(key)
            return False, None
        return True, value

    zones = app.get("climate_zones")
    if zones:
        relevant += 1
        ok, zone = _need("climate_zone", "climate zone")
        if ok:
            if zone in zones:
                matched += 1
                reasons_for.append(
                    f"detected climate zone '{zone}' is within the solution's "
                    f"range ({', '.join(zones)})")
            else:
                return True, matched, relevant, reasons_for, unverified, missing

    min_e = app.get("min_elevation_m")
    max_e = app.get("max_elevation_m")
    if min_e is not None or max_e is not None:
        relevant += 1
        ok, elev = _need("elevation_m", "elevation")
        if ok:
            try:
                elev_f = float(elev)
            except (TypeError, ValueError):
                elev_f = None
            if elev_f is None:
                unverified.append("elevation not numeric — condition unverified")
            elif (min_e is not None and elev_f < min_e) or \
                    (max_e is not None and elev_f > max_e):
                return True, matched, relevant, reasons_for, unverified, missing
            else:
                matched += 1
                lo = f">= {_fmt(min_e)} m" if min_e is not None else ""
                hi = f"<= {_fmt(max_e)} m" if max_e is not None else ""
                bounds = " and ".join(p for p in (lo, hi) if p)
                reasons_for.append(
                    f"site elevation {_fmt(elev_f)} m within range ({bounds})")

    regimes = app.get("moisture_regimes")
    if regimes:
        relevant += 1
        ok, regime = _need("moisture_regime", "moisture regime")
        if ok:
            if regime in regimes:
                matched += 1
                reasons_for.append(
                    f"detected moisture regime '{regime}' fits the solution "
                    f"({', '.join(regimes)})")
            else:
                # Moisture is transient: penalise, do not exclude.
                unverified.append(
                    f"detected moisture regime '{regime}' is outside the "
                    f"preferred regimes ({', '.join(regimes)})")

    landcover_any = app.get("landcover_any")
    if landcover_any:
        relevant += 1
        ok, classes = _need("landcover_classes", "land-cover classes")
        if ok:
            present = [c for c in (classes or []) if c in landcover_any]
            if not present:
                return True, matched, relevant, reasons_for, unverified, missing
            matched += 1
            reasons_for.append(
                f"site land cover includes {', '.join(present)} (ESA "
                f"WorldCover classes in the fetch window)")

    if app.get("requires_water"):
        relevant += 1
        ok, count = _need("water_features_count", "mapped water features")
        if ok:
            try:
                n = int(count)
            except (TypeError, ValueError):
                n = 0
            if n <= 0:
                return True, matched, relevant, reasons_for, unverified, missing
            matched += 1
            reasons_for.append(
                f"{n} water features mapped within the analysis window (OSM)")

    if app.get("requires_urban"):
        relevant += 1
        ok, count = _need("buildings_count", "mapped buildings")
        if ok:
            try:
                n = int(count)
            except (TypeError, ValueError):
                n = 0
            if n < _URBAN_BUILDINGS_THRESHOLD:
                return True, matched, relevant, reasons_for, unverified, missing
            matched += 1
            reasons_for.append(
                f"{n} buildings mapped within the analysis window (OSM) — "
                f"urban context present")

    return False, matched, relevant, reasons_for, unverified, missing


def _hazard_label(hazard: Dict) -> str:
    level = hazard.get("level") or hazard.get("severity")
    if isinstance(level, dict):
        level = level.get("label")
    return f"{hazard.get('id', 'unknown')} (level: {level or 'unspecified'})"


def _evidence_for(entry: Dict) -> List[Dict[str, Any]]:
    """One EvidenceRecord per KB source: curated reference knowledge, never
    a site measurement."""
    records = []
    for src in entry.get("sources") or []:
        records.append(EvidenceRecord(
            src.get("class", "OPEN_DATA_OFFICIAL"),
            ClaimStatus.DOCUMENTED.value,
            TemporalClass.HISTORICAL.value,
            src.get("name", "unknown source"),
            link=src.get("url"),
            method="curated reference knowledge for a generic solution class; "
                   "not a site measurement",
            confidence=Confidence.MEDIUM.value,
            limitations="Reference guidance; site-specific assessment "
                        "required before implementation.",
        ).to_dict())
    return records


def _data_confidence(relevant: int, unverified: int) -> str:
    if relevant == 0:
        return Confidence.MEDIUM.value
    if unverified == 0:
        return Confidence.HIGH.value
    return (Confidence.MEDIUM.value if unverified / relevant <= 0.5
            else Confidence.LOW.value)


def _fit_band(fit_score: float, relevant: int) -> str:
    """Declared label over the score (see module docstring)."""
    if relevant == 0:
        return "hazard_match_only"
    if fit_score >= FIT_BANDS["high"]:
        return "high"
    if fit_score >= FIT_BANDS["moderate"]:
        return "moderate"
    return "low"


def _solution_output(
    entry: Dict,
    hazard: Dict,
    site: Dict,
    matched: int,
    relevant: int,
    reasons_for: List[str],
    unverified: List[str],
) -> Dict[str, Any]:
    fit_score = round(matched / relevant, 3) if relevant else 1.0

    why_parts = [f"Hazard: {_hazard_label(hazard)}."]
    events = site.get("historical_events")
    if isinstance(events, dict):
        summary = events.get(hazard.get("id"))
        if summary:
            why_parts.append(f"Historical record: {summary}.")
    if reasons_for:
        why_parts.append("Site fit: " + "; ".join(reasons_for) + ".")
    else:
        why_parts.append(
            "Site fit: hazard-driven candidate; no additional site "
            "conditions were declared or available to verify.")
    why_it_fits = " ".join(why_parts)

    expected = entry.get("expected_benefit") or {}
    return {
        "solution_id": entry["solution_id"],
        "name": entry["name"],
        "classes": entry.get("classes") or [],
        "hazards_addressed": entry.get("hazards_addressed") or [],
        "economic_sectors": entry.get("economic_sectors") or [],
        "why_it_fits": why_it_fits,
        "fit_score": fit_score,
        "fit_band": _fit_band(fit_score, relevant),
        "fit": {
            "scoring": "conditions_matched / conditions_relevant",
            "conditions_matched": matched,
            "conditions_relevant": relevant,
            "reasons_for": reasons_for,
            "unverified": unverified,
        },
        "evidence": _evidence_for(entry),
        "expected_benefit": {
            "mechanism": entry.get("mechanism"),
            "quantified": bool(expected.get("quantified", False)),
            "quantification_note": expected.get("quantification_note"),
        },
        "limitations": entry.get("limitations") or [],
        "implementation_complexity": entry.get("implementation_complexity"),
        "maintenance": entry.get("maintenance"),
        "environmental_considerations": entry.get("environmental_considerations") or [],
        "technology_maturity": entry.get("technology_maturity"),
        "cost_basis": entry.get("cost_basis", "not quantified"),
        "knowledge_confidence": entry.get("confidence", "medium"),
        "quantification_status": entry.get("quantification_status", "not_quantified"),
        "data_confidence": _data_confidence(relevant, len(unverified)),
        "sources": entry.get("sources") or [],
        "guarantee_disclaimer": GUARANTEE_DISCLAIMER,
    }


def _normalise_hazards(site: Dict) -> List[Dict]:
    hazards = site.get("hazards")
    if not isinstance(hazards, list):
        return []
    out = []
    for h in hazards:
        if isinstance(h, dict) and h.get("id"):
            out.append(h)
        elif isinstance(h, str):
            out.append({"id": h})
    return out


def infer_site_sectors(site: Dict) -> List[Dict[str, str]]:
    """Declared inference of possibly-affected economic sectors from real
    site signals (mapped land cover / OSM counts). Each entry states its
    basis — this is INFERRED context, never a measured exposure.
    """
    sectors: List[Dict[str, str]] = []

    def _add(sector: str, basis: str) -> None:
        if not any(s["sector"] == sector for s in sectors):
            sectors.append({"sector": sector, "basis": basis,
                            "status": "inferred"})

    classes = site.get("landcover_classes") or []
    if "Cropland" in classes:
        _add("agriculture", "'Cropland' present in ESA WorldCover classes")
    if "Built-up" in classes:
        _add("population_municipal",
             "'Built-up' land cover present in ESA WorldCover classes")
        _add("real_estate_construction",
             "'Built-up' land cover present in ESA WorldCover classes")
    try:
        buildings = int(site.get("buildings_count"))
    except (TypeError, ValueError):
        buildings = 0
    if buildings >= _URBAN_BUILDINGS_THRESHOLD:
        _add("population_municipal",
             f"{buildings} buildings mapped within the analysis window (OSM)")
        _add("critical_facilities",
             "urban context — critical-facility presence likely but not "
             "verified here (see /api/v2/economy for mapped facilities)")
    try:
        water = int(site.get("water_features_count"))
    except (TypeError, ValueError):
        water = 0
    if water > 0:
        _add("water_utilities",
             f"{water} water features mapped within the analysis window (OSM)")
    return sectors


def build_packages(kb: Dict, by_hazard: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    """Assemble declared solution packages (KB ``solution_packages``) from
    the per-hazard fitted recommendations.

    A package is offered when at least two of its components passed the
    hazard gate and applicability checks for this site. Components that did
    not fit are listed with the honest reason. The package explains *why
    the combination is useful* — never that prevention is guaranteed.
    """
    packages: List[Dict[str, Any]] = []
    for pkg in kb.get("solution_packages") or []:
        hazard = str(pkg.get("hazard") or "").lower()
        fitted = {s["solution_id"]: s for s in by_hazard.get(hazard, [])}
        included, excluded = [], []
        for sid in pkg.get("components") or []:
            if sid in fitted:
                included.append({
                    "solution_id": sid,
                    "name": fitted[sid]["name"],
                    "fit_band": fitted[sid]["fit_band"],
                    "fit_score": fitted[sid]["fit_score"],
                })
            else:
                excluded.append({
                    "solution_id": sid,
                    "reason": "did not pass the hazard gate or the site's "
                              "applicability conditions (see per-hazard "
                              "recommendations for detail)",
                })
        if len(included) < 2:
            continue
        packages.append({
            "package_id": pkg["package_id"],
            "hazard": hazard,
            "name": pkg.get("name"),
            "why_together": pkg.get("why_together"),
            "components": included,
            "excluded_components": excluded,
            "guarantee_disclaimer": GUARANTEE_DISCLAIMER,
        })
    return packages


def recommend_solutions(
    site: Dict,
    knowledge_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Recommend site-fitted solutions from the curated knowledge base.

    ``site`` carries whatever is available: ``lat``/``lon``, ``hazards``
    (list of ``{"id": …, "level": …}``), ``climate_zone``,
    ``moisture_regime``, ``elevation_m``, ``landcover_classes``,
    ``water_features_count``, ``buildings_count``, ``historical_events``.
    Missing inputs are reported, never invented.
    """
    site = dict(site or {})
    hazards = _normalise_hazards(site)

    missing_inputs = []
    for key, label in _INPUT_LABELS.items():
        value = hazards if key == "hazards" else site.get(key)
        if value is None or (key == "hazards" and not value):
            missing_inputs.append({"input": key, "would_sharpen": label})

    insufficient = {
        "missing_inputs": missing_inputs,
        "note": ("Recommendations are hazard-gated: only conditions with "
                 "real site values are verified; everything else is listed "
                 "as unverified rather than assumed."),
    }

    if not hazards:
        return {
            "status": "insufficient_data",
            "message": ("Active hazard levels were not available for this "
                        "location, so no solutions could be matched. " +
                        INSUFFICIENT_DATA_MESSAGE),
            "site": site,
            "recommendations_by_hazard": {},
            "insufficient_data": insufficient,
            "guarantee_disclaimer": GUARANTEE_DISCLAIMER,
            "provenance": {
                "kind": "unavailable",
                "source": "Talaix solutions engine",
                "quality": "missing",
                "limitations": "No active hazard levels were provided; "
                               "solution matching is hazard-gated.",
            },
        }

    try:
        kb = load_solutions_knowledge(knowledge_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "insufficient_data",
            "message": f"Solutions knowledge base unavailable: {exc}",
            "site": site,
            "recommendations_by_hazard": {},
            "insufficient_data": insufficient,
            "guarantee_disclaimer": GUARANTEE_DISCLAIMER,
            "provenance": {"kind": "unavailable",
                           "source": "solutions knowledge base",
                           "quality": "missing"},
        }

    entries = kb.get("solutions") or []
    by_hazard: Dict[str, List[Dict]] = {}
    for hazard in hazards:
        hid = str(hazard.get("id")).lower()
        fitted: List[Dict] = []
        for entry in entries:
            addressed = [str(h).lower()
                         for h in entry.get("hazards_addressed") or []]
            if hid not in addressed:
                continue
            excluded, matched, relevant, reasons_for, unverified, missing = \
                _check_applicability(entry, site)
            if excluded:
                continue
            fitted.append(_solution_output(
                entry, hazard, site, matched, relevant, reasons_for, unverified))
        fitted.sort(key=lambda s: (-s["fit_score"], s["solution_id"]))
        by_hazard[hid] = fitted

    return {
        "status": "ok",
        "site": site,
        "site_sectors": infer_site_sectors(site),
        "recommendations_by_hazard": by_hazard,
        "packages": build_packages(kb, by_hazard),
        "resilience_economics": RESILIENCE_ECONOMICS,
        "insufficient_data": insufficient,
        "guarantee_disclaimer": GUARANTEE_DISCLAIMER,
        "provenance": {
            "kind": "derived",
            "source": "Talaix solutions engine: real site conditions + "
                      "curated solutions knowledge base "
                      "(config/solutions_knowledge.json)",
            "quality": "reference-level solution classes; local expert "
                       "verification required before implementation",
            "limitations": "Fit scoring is a declared screening heuristic "
                           "(conditions matched / conditions relevant), not "
                           "a validated performance estimate; costs and "
                           "benefits are not quantified; inferred site "
                           "sectors are labelled context, not measured "
                           "exposure.",
        },
    }
