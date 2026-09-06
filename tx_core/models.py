"""
Standard TX result models — the one envelope every TX consumer receives.

Dependency-free dataclasses (no pydantic required): they mirror the platform
honesty contract — ``UNKNOWN`` is a first-class status, every claim carries
evidence and provenance, and the envelope always stamps engine versions so a
TX analysis can be reproduced (see docs/TX_ENGINE.md).

``TxHazardResult.from_hazard_analysis`` adapts the platform's existing
:class:`src.climate.hazards.base.HazardAnalysis` into the TX envelope
without changing a single byte of the underlying analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .adapters.climate import utcnow_iso  # single shared clock


@dataclass
class TxLocation:
    """A location analysed by TX."""

    lat: float
    lon: float
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TxRequest:
    """A TX analysis request (validated before dispatch)."""

    location: TxLocation
    hazards: List[str] = field(default_factory=list)  # empty = all available
    depth: str = "standard"                            # quick | standard | deep

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location.to_dict(),
            "hazards": list(self.hazards),
            "depth": self.depth,
        }


@dataclass
class TxHazardResult:
    """One hazard's result inside a TX analysis.

    ``tx_level`` stamps the TX analysis layer the result belongs to
    (``TX_LEVELS`` in ``tx_core.engine``): 1 for hazard screening modules,
    2+ for registered product analyses (docs/TX_ENGINE.md §4).
    """

    hazard: str
    status: str                 # ok | partial | unavailable | key_required
    summary: str = ""
    level: Optional[Dict[str, Any]] = None
    blocks: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    unavailable_reason: Optional[str] = None
    tx_level: Optional[int] = None

    @classmethod
    def from_hazard_analysis(cls, analysis: Any) -> "TxHazardResult":
        """Adapt a platform ``HazardAnalysis`` (or any dict-like) unchanged."""
        return cls(
            hazard=analysis.hazard,
            status=analysis.status,
            summary=getattr(analysis, "summary", ""),
            level=getattr(analysis, "level", None),
            blocks=dict(getattr(analysis, "blocks", {}) or {}),
            evidence=list(getattr(analysis, "evidence", []) or []),
            provenance=dict(getattr(analysis, "provenance", {}) or {}),
            unavailable_reason=getattr(analysis, "unavailable_reason", None),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "hazard": self.hazard,
            "status": self.status,
            "summary": self.summary,
            "level": self.level,
            "blocks": self.blocks,
            "evidence": self.evidence,
            "provenance": self.provenance,
            "unavailable_reason": self.unavailable_reason,
            "tx_level": self.tx_level,
        }
        if self.level is not None and hasattr(self.level, "to_dict"):
            d["level"] = self.level.to_dict()
        return d


@dataclass
class TxResult:
    """The uniform TX analysis envelope.

    Guarantees (reproducibility contract):

    - ``analysis_id`` is deterministic for the same (location, hazards,
      depth, engine versions, calendar day) — re-running yields the same id.
    - ``engine_version`` / ``tx_version`` / ``tam_version`` are always
      stamped so the exact analytical contract behind the result is known.
    """

    analysis_id: str
    location: TxLocation
    depth: str
    results: List[TxHazardResult]
    engine_version: str
    tx_version: str
    tam_version: str
    generated_at: str = field(default_factory=utcnow_iso)
    status: str = "ok"           # ok | partial | unavailable
    summary: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, str]] = field(default_factory=list)
    authenticity_code: str = ""

    @property
    def status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "analysis_id": self.analysis_id,
            "location": self.location.to_dict(),
            "depth": self.depth,
            "status": self.status,
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
            "status_counts": self.status_counts,
            "evidence": list(self.evidence),
            "sources": list(self.sources),
            "engine_version": self.engine_version,
            "tx_version": self.tx_version,
            "tam_version": self.tam_version,
            "generated_at": self.generated_at,
        }
        if self.authenticity_code:
            d["authenticity"] = {
                "code": self.authenticity_code,
                "engine": "TX",
                "verify": "POST /api/v2/verify",
            }
        return d
