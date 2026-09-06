"""
Uncertainty Engine — the standard envelope for analytical values.

Every NEW analytical endpoint wraps its payload in :class:`AnalyticalResult`
so that a consumer can always see *what kind of value this is*
(``observed | derived | modelled | projected | unavailable``), where it came
from, how confident the platform is, and what the known limitations are.

Norms (docs/EVIDENCE_ARCHITECTURE.md, src/climate/ontology.py):

- ``unavailable`` is a first-class, respected answer — never a failure; it
  always carries a reason.
- Projections are structurally separated from observations via ``status`` —
  never blended into one field.
- ``confidence`` is a declared screening label (high/medium/low), not a
  validated probability.

Dependency-free; no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .evidence import utcnow_iso

VALID_STATUSES = frozenset({
    "observed", "derived", "modelled", "projected", "unavailable",
})
VALID_CONFIDENCE = frozenset({"high", "medium", "low"})


@dataclass
class AnalyticalResult:
    """The standard envelope around one analytical value.

    ``value`` is None exactly when ``status == "unavailable"`` — an honest
    gap always says why (``limitations`` carries the reason, echoed as
    ``unavailable_reason`` in :meth:`to_dict`).
    """

    value: Any
    status: str                          # observed|derived|modelled|projected|unavailable
    source: Optional[str] = None         # human-readable origin, e.g. "ERA5-Land via Open-Meteo"
    timestamp_utc: Optional[str] = None  # when the value was produced/acquired (UTC ISO)
    method: Optional[str] = None         # how it was produced; mandatory for modelled/projected
    confidence: str = "medium"           # high|medium|low — declared screening label
    uncertainty: Optional[str] = None    # free text: known magnitude/sources of uncertainty
    coverage: Optional[str] = None       # spatial/temporal coverage note
    limitations: Optional[str] = None    # the honesty channel

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid analytical status '{self.status}'")
        if self.confidence not in VALID_CONFIDENCE:
            raise ValueError(f"invalid confidence '{self.confidence}'")
        if self.timestamp_utc is None:
            self.timestamp_utc = utcnow_iso()
        if self.status == "unavailable" and not self.limitations:
            raise ValueError("unavailable results must carry a reason in limitations")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.status == "unavailable":
            d["unavailable_reason"] = self.limitations
        return d

    # -- constructors ------------------------------------------------------

    @classmethod
    def observed(cls, value: Any, *, source: str,
                 method: Optional[str] = None, confidence: str = "high",
                 uncertainty: Optional[str] = None, coverage: Optional[str] = None,
                 limitations: Optional[str] = None) -> "AnalyticalResult":
        """A measured value from an instrument or authority."""
        return cls(value, "observed", source, method=method,
                   confidence=confidence, uncertainty=uncertainty,
                   coverage=coverage, limitations=limitations)

    @classmethod
    def derived(cls, value: Any, *, source: str, method: str,
                confidence: str = "medium", uncertainty: Optional[str] = None,
                coverage: Optional[str] = None,
                limitations: Optional[str] = None) -> "AnalyticalResult":
        """Computed from observed inputs; the derivation method is stated."""
        return cls(value, "derived", source, method=method,
                   confidence=confidence, uncertainty=uncertainty,
                   coverage=coverage, limitations=limitations)

    @classmethod
    def modelled(cls, value: Any, *, source: str, method: str,
                 confidence: str = "medium", uncertainty: Optional[str] = None,
                 coverage: Optional[str] = None,
                 limitations: Optional[str] = None) -> "AnalyticalResult":
        """Output of a declared model with declared inputs."""
        return cls(value, "modelled", source, method=method,
                   confidence=confidence, uncertainty=uncertainty,
                   coverage=coverage, limitations=limitations)

    @classmethod
    def projected(cls, value: Any, *, source: str, method: str,
                  confidence: str = "low", uncertainty: Optional[str] = None,
                  coverage: Optional[str] = None,
                  limitations: Optional[str] = None) -> "AnalyticalResult":
        """Long-horizon scenario-conditional projection — never an observation."""
        return cls(value, "projected", source, method=method,
                   confidence=confidence, uncertainty=uncertainty,
                   coverage=coverage, limitations=limitations)

    @classmethod
    def unavailable(cls, reason: str, source: Optional[str] = None, *,
                    method: Optional[str] = None,
                    coverage: Optional[str] = None) -> "AnalyticalResult":
        """An honest 'we do not have this value': status unavailable, reason
        in ``limitations`` (echoed as ``unavailable_reason`` by to_dict)."""
        return cls(None, "unavailable", source, method=method,
                   confidence="low", coverage=coverage, limitations=reason)


def wrap_series(points: List[Dict[str, Any]], **meta: Any) -> Dict[str, Any]:
    """Wrap a daily-series payload in the envelope.

    ``points`` is a list of dicts with at least ``date`` (ISO) and ``value``
    keys — point values may legitimately be None (a gap in the source is
    reported, not filled). ``meta`` must declare ``status`` and ``source``
    (never silently defaulted); ``method``, ``confidence``, ``uncertainty``,
    ``coverage`` and ``limitations`` are optional and passed through.
    """

    status = meta.pop("status", None)
    source = meta.pop("source", None)
    if status not in VALID_STATUSES - {"unavailable"}:
        raise ValueError(
            "wrap_series requires meta['status'] in "
            "{observed, derived, modelled, projected}")
    if not source:
        raise ValueError("wrap_series requires meta['source']")
    if status in ("modelled", "projected") and not meta.get("method"):
        raise ValueError(f"status '{status}' requires meta['method']")

    points = list(points)
    for i, p in enumerate(points):
        if not isinstance(p, dict) or "date" not in p or "value" not in p:
            raise ValueError(
                f"point {i} must be a dict with 'date' and 'value' keys")

    null_count = sum(1 for p in points if p["value"] is None)
    return {
        "status": status,
        "source": source,
        "timestamp_utc": meta.pop("timestamp_utc", None) or utcnow_iso(),
        "method": meta.pop("method", None),
        "confidence": meta.pop("confidence", "medium"),
        "uncertainty": meta.pop("uncertainty", None),
        "coverage": meta.pop("coverage", None),
        "limitations": meta.pop("limitations", None),
        "points": points,
        "point_count": len(points),
        "null_count": null_count,
    }


# ---------------------------------------------------------------------------
# Evidence Confidence Profile (additive — multi-dimensional, never a score)
# ---------------------------------------------------------------------------

#: Vocabulary for every confidence dimension. Unlike the AnalyticalResult
#: screening label, ``unknown`` is first-class here: a dimension that was
#: never assessed says so.
VALID_CONFIDENCE_DIMENSION = frozenset({"high", "medium", "low", "unknown"})

_CONFIDENCE_DIMENSIONS = (
    "source_quality", "recency", "coverage", "method_transparency",
    "validation_status", "independence",
)

_NO_SINGLE_SCORE_NOTE = (
    "Dimensions are deliberately NOT collapsible to a single score: a "
    "strong source can still be stale, narrow, undocumented or "
    "unvalidated. Report and weigh each dimension on its own."
)


@dataclass
class EvidenceConfidence:
    """Multi-dimensional confidence profile for evidence behind a claim.

    Six independent dimensions, each in {high, medium, low, unknown}:

    - ``source_quality`` — standing of the source itself,
    - ``recency`` — how fresh the underlying data is,
    - ``coverage`` — how well it covers the claim's space/time,
    - ``method_transparency`` — whether the method is documented,
    - ``validation_status`` — whether the method/indicator is validated,
    - ``independence`` — agreement across independent sources.

    The profile is explicitly NOT collapsible to one number
    (``summary_note`` in :meth:`to_dict` says so); there is intentionally
    no aggregate score field.
    """

    source_quality: str = "unknown"
    recency: str = "unknown"
    coverage: str = "unknown"
    method_transparency: str = "unknown"
    validation_status: str = "unknown"
    independence: str = "unknown"
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        for dim in _CONFIDENCE_DIMENSIONS:
            value = getattr(self, dim)
            if value not in VALID_CONFIDENCE_DIMENSION:
                raise ValueError(
                    f"invalid {dim} '{value}': must be one of "
                    f"{sorted(VALID_CONFIDENCE_DIMENSION)}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["summary_note"] = _NO_SINGLE_SCORE_NOTE
        return d


def confidence_profile(source_kind: Optional[str] = None,
                       freshness_days: Optional[float] = None,
                       coverage_note: Optional[str] = None,
                       method_documented: Optional[bool] = None,
                       validation_status: Optional[str] = None,
                       independent_sources: Optional[int] = None,
                       notes: Optional[str] = None) -> EvidenceConfidence:
    """Deterministic mapping from declared facts to an EvidenceConfidence.

    Declared rules (any input left None maps to ``unknown`` — never
    guessed):

    - ``source_kind``: official_observation / satellite_product /
      reanalysis → high; official_model / commercial_api / community →
      medium; media → low; anything else → unknown.
    - ``freshness_days``: ≤ 2 → high; ≤ 30 → medium; older → low.
    - ``coverage_note`` (free text, case-insensitive): contains "global"
      → high; contains regional/europe/national/country/province/state/
      partial/limited → medium; otherwise → low.
    - ``method_documented``: True → high; False → low.
    - ``validation_status``: validated_operational → high;
      validated_screening / validation_in_progress → medium;
      not_validated / deprecated → low; anything else → unknown.
    - ``independent_sources``: ≥ 3 → high; 2 → medium; ≤ 1 → low.
    """
    sk = (source_kind or "").strip().lower()
    if sk in ("official_observation", "satellite_product", "reanalysis"):
        source_quality = "high"
    elif sk in ("official_model", "commercial_api", "community"):
        source_quality = "medium"
    elif sk == "media":
        source_quality = "low"
    else:
        source_quality = "unknown"

    if freshness_days is None:
        recency = "unknown"
    elif freshness_days <= 2:
        recency = "high"
    elif freshness_days <= 30:
        recency = "medium"
    else:
        recency = "low"

    if coverage_note is None:
        coverage = "unknown"
    else:
        cn = coverage_note.lower()
        if "global" in cn:
            coverage = "high"
        elif any(k in cn for k in ("regional", "europe", "national",
                                   "country", "province", "state",
                                   "partial", "limited")):
            coverage = "medium"
        else:
            coverage = "low"

    if method_documented is None:
        method_transparency = "unknown"
    else:
        method_transparency = "high" if method_documented else "low"

    vs = (validation_status or "").strip().lower()
    if vs == "validated_operational":
        validation = "high"
    elif vs in ("validated_screening", "validation_in_progress"):
        validation = "medium"
    elif vs in ("not_validated", "deprecated"):
        validation = "low"
    else:
        validation = "unknown"

    if independent_sources is None:
        independence = "unknown"
    elif independent_sources >= 3:
        independence = "high"
    elif independent_sources == 2:
        independence = "medium"
    else:
        independence = "low"

    return EvidenceConfidence(
        source_quality=source_quality,
        recency=recency,
        coverage=coverage,
        method_transparency=method_transparency,
        validation_status=validation,
        independence=independence,
        notes=notes,
    )
