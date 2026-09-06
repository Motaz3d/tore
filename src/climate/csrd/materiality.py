"""
Double materiality scoring — deterministic, documented math.

A topic is material under CSRD when it is material from the **impact**
perspective (the company's effects on people and environment), the
**financial** perspective (effects of sustainability matters on the
company), or both. The engine scores both dimensions on a 0–5 scale and
applies a configurable threshold.

Formulas (docs/CSRD_TX_ENGINE.md §Math):

- ``severity = (scale + scope + irremediability) / 3`` — each 0–5.
- ``impact_score = severity`` for actual impacts;
  ``impact_score = severity × likelihood`` for potential impacts
  (likelihood in [0, 1]).
- ``financial_score = magnitude × likelihood`` — magnitude 0–5,
  likelihood in [0, 1].
- ``combined_score = max(impact_score, financial_score)`` — double
  materiality is a union, not an average: either perspective alone can
  make a topic material.
- ``material = combined_score >= threshold`` (default 2.5).
- ``confidence = mean(evidence weights of the inputs used)`` with
  A=1.0, B=0.85, C=0.7, D=0.5, E=0.3, F=0.0.

Every function is pure: same input → same output, no clocks, no I/O.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

EVIDENCE_WEIGHTS = {
    "A": 1.0,   # independently verified
    "B": 0.85,  # official / modelled from real data
    "C": 0.7,   # reliable external source
    "D": 0.5,   # company-declared
    "E": 0.3,   # inferred
    "F": 0.0,   # unavailable
}

DEFAULT_THRESHOLD = 2.5

# Physical hazard level labels → financial magnitude seed (0–5).
_HAZARD_MAGNITUDE = {
    "very high": 4.5,
    "high": 3.5,
    "moderate": 2.5,
    "medium": 2.5,
    "low": 1.5,
    "very low": 0.5,
}


def _clamp(value: float, low: float, high: float, name: str) -> float:
    v = float(value)
    if not (low <= v <= high):
        raise ValueError(f"{name} must be within [{low}, {high}], got {value}")
    return v


def _round(value: float) -> float:
    return round(value + 1e-12, 3)


def score_impact(
    scale: float,
    scope: float,
    irremediability: float,
    likelihood: float = 1.0,
    *,
    actual: bool = False,
) -> float:
    """Impact materiality score (0–5)."""
    scale = _clamp(scale, 0, 5, "scale")
    scope = _clamp(scope, 0, 5, "scope")
    irremediability = _clamp(irremediability, 0, 5, "irremediability")
    likelihood = _clamp(likelihood, 0, 1, "likelihood")
    severity = (scale + scope + irremediability) / 3.0
    return _round(severity if actual else severity * likelihood)


def score_financial(magnitude: float, likelihood: float) -> float:
    """Financial materiality score (0–5)."""
    magnitude = _clamp(magnitude, 0, 5, "magnitude")
    likelihood = _clamp(likelihood, 0, 1, "likelihood")
    return _round(magnitude * likelihood)


def _evidence_weight(grade: Optional[str]) -> float:
    if grade is None:
        return EVIDENCE_WEIGHTS["F"]
    return EVIDENCE_WEIGHTS.get(str(grade).strip().upper(), EVIDENCE_WEIGHTS["F"])


def assess_topic(
    topic_id: str,
    *,
    impact: Optional[Dict[str, Any]] = None,
    financial: Optional[Dict[str, Any]] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Assess one ESRS topic for double materiality.

    ``impact`` keys: scale, scope, irremediability (0–5), likelihood
    (0–1), actual (bool), evidence_grade (A–F).
    ``financial`` keys: magnitude (0–5), likelihood (0–1),
    evidence_grade (A–F).
    Missing perspectives score 0 and pull confidence down honestly.
    """
    threshold = _clamp(threshold, 0, 5, "threshold")
    weights: List[float] = []

    impact_score = 0.0
    if impact:
        impact_score = score_impact(
            impact.get("scale", 0),
            impact.get("scope", 0),
            impact.get("irremediability", 0),
            impact.get("likelihood", 1.0),
            actual=bool(impact.get("actual", False)),
        )
        weights.append(_evidence_weight(impact.get("evidence_grade")))

    financial_score = 0.0
    if financial:
        financial_score = score_financial(
            financial.get("magnitude", 0),
            financial.get("likelihood", 0),
        )
        weights.append(_evidence_weight(financial.get("evidence_grade")))

    combined = max(impact_score, financial_score)
    material = combined >= threshold
    impact_material = impact_score >= threshold
    financial_material = financial_score >= threshold
    if impact_material and financial_material:
        basis = "both"
    elif impact_material:
        basis = "impact"
    elif financial_material:
        basis = "financial"
    else:
        basis = "none"

    confidence = _round(sum(weights) / len(weights)) if weights else 0.0

    return {
        "topic": topic_id,
        "impact_score": _round(impact_score),
        "financial_score": _round(financial_score),
        "combined_score": _round(combined),
        "threshold": _round(threshold),
        "material": material,
        "basis": basis,
        "confidence": confidence,
        "confidence_label": (
            "high" if confidence >= 0.7 else "medium" if confidence >= 0.4 else "low"
        ),
    }


def hazard_exposure_seed(site_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Seed an E1 financial-materiality input from verified hazard levels.

    ``site_results`` are the trimmed site results from the sustainability
    evidence pack (``hazard_levels`` mapping hazard → level label).
    Returns None when no usable hazard data exists — the caller then
    declares the perspective UNAVAILABLE instead of inventing a score.
    """
    magnitudes: List[float] = []
    exposed_sites = 0
    total_sites = 0
    for result in site_results or []:
        if not result.get("ok"):
            continue
        total_sites += 1
        levels = result.get("hazard_levels") or {}
        site_max = 0.0
        for label in levels.values():
            mapped = _HAZARD_MAGNITUDE.get(str(label).strip().lower(), 0.0)
            site_max = max(site_max, mapped)
        if site_max > 0:
            magnitudes.append(site_max)
        if site_max >= _HAZARD_MAGNITUDE["moderate"]:
            exposed_sites += 1
    if not magnitudes or total_sites == 0:
        return None
    magnitude = max(magnitudes)
    # Likelihood proxy: share of sites at moderate-or-higher exposure,
    # floored so a single exposed site never yields a near-zero score.
    likelihood = min(1.0, max(0.3, exposed_sites / total_sites))
    return {
        "magnitude": magnitude,
        "likelihood": _round(likelihood),
        "evidence_grade": "B",
        "basis_note": (
            f"Seeded from physical hazard verification of {total_sites} "
            f"site(s); {exposed_sites} at moderate-or-higher exposure. "
            "Screening-level, not actuarial."
        ),
    }
