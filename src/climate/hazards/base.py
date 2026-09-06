"""
Hazard plugin contract for the Talaix climate platform.

Every hazard (wildfire, flood, drought, heat, wind, coastal, …) implements
:class:`HazardModule`. The registry (`src/climate/registry.py`) only ever
contains modules wired to real, documented data sources — a hazard without
a real source does not exist in the platform.

Honesty contract (docs/EVIDENCE_ARCHITECTURE.md):

- ``analyze`` returns ``status="unavailable"`` (with ``unavailable_reason``)
  when real data cannot be obtained — never fabricated numbers.
- ``events`` returns ``status="key_required"`` / ``"unavailable"`` when the
  underlying dataset needs a credential or has no coverage — with the
  reason stated.
- Scores/levels are labelled screening indicators until validated.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HazardLevel:
    """A per-hazard level summary. ``score`` may be None when the hazard
    expresses severity categorically; ``basis`` must always say what the
    level rests on and whether it is validated."""

    label: str                       # e.g. "High", "Extreme", "Moderate"
    score: Optional[float] = None
    score_max: Optional[float] = None
    basis: str = ""
    validated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HazardAnalysis:
    """Uniform analysis result across hazards."""

    hazard: str
    location: Dict[str, Any]         # {"lat": …, "lon": …, "name": …?}
    status: str                      # "ok" | "partial" | "unavailable" | "key_required"
    summary: str = ""
    level: Optional[HazardLevel] = None
    blocks: Dict[str, Any] = field(default_factory=dict)   # hazard-specific content
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    unavailable_reason: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None                   # engine-native payload (wildfire compat)

    def to_dict(self, include_raw: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "hazard": self.hazard,
            "location": self.location,
            "status": self.status,
            "summary": self.summary,
            "level": self.level.to_dict() if self.level else None,
            "blocks": self.blocks,
            "evidence": self.evidence,
            "provenance": self.provenance,
            "unavailable_reason": self.unavailable_reason,
        }
        if include_raw and self.raw is not None:
            d["raw"] = self.raw
        return d


@dataclass
class LayerSpec:
    """A map layer descriptor. Every layer carries legend, source, date,
    resolution, status and provenance (map contract, docs §16)."""

    layer_id: str
    label: str
    group: str                       # e.g. "HAZARD", "EXPOSURE", "ENVIRONMENT", "EVIDENCE"
    kind: str                        # "geojson" | "points" | "grid" | "raster"
    endpoint: Optional[str] = None   # API path template, e.g. "/api/v2/events?hazard=wildfire&…"
    legend: Optional[Dict[str, str]] = None   # label → colour
    source: Optional[str] = None
    url: Optional[str] = None        # official source URL (opened from the layer panel)
    date: Optional[str] = None
    resolution: Optional[str] = None
    status: str = "available"        # "available" | "key_required" | "unavailable"
    temporal: str = "OBSERVED"       # TemporalClass value
    default_on: bool = False
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HazardModule(ABC):
    """Base class for hazard plugins. Heavy dependencies must be imported
    lazily inside methods so the registry itself stays import-light."""

    id: str = ""
    name: str = ""
    tagline: str = ""

    # -- availability & coverage ------------------------------------------

    def availability(self) -> Tuple[bool, Optional[str]]:
        """Is core analysis available right now? (True, None) or
        (False, human-readable reason)."""
        return True, None

    def events_availability(self) -> Tuple[bool, Optional[str]]:
        """Is the historical-events layer available right now?"""
        return True, None

    def temporal_coverage(self) -> Dict[str, Dict[str, str]]:
        """Actually-available temporal coverage per dataset — the UI builds
        its year selector from this, never from hardcoded years."""
        return {}

    # -- core operations ----------------------------------------------------

    def analyze(self, lat: float, lon: float, name: Optional[str] = None, **kw: Any) -> HazardAnalysis:
        raise NotImplementedError

    def events(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
        year: Optional[int] = None,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Historical events near a point. Default: honestly unavailable."""
        return {
            "hazard": self.id,
            "status": "unavailable",
            "reason": f"Historical events are not yet implemented for {self.name}.",
            "events": [],
        }

    def map_layers(self, **kw: Any) -> List[Dict[str, Any]]:
        return []

    def sources(self) -> List[Dict[str, str]]:
        """Official data sources behind this hazard — ``[{"name", "url"}]``.

        Derived from the module's own map-layer declarations (the same
        source/URL pairs the map UI shows), de-duplicated by name. Nothing
        is invented: a source appears here only when the module declared it
        with an official URL.
        """
        seen: List[Dict[str, str]] = []
        for layer in self.map_layers():
            name = layer.get("source")
            url = layer.get("url")
            if name and url and not any(s["name"] == name for s in seen):
                seen.append({"name": name, "url": url})
        return seen

    def descriptor(self) -> Dict[str, Any]:
        """Public descriptor for /api/v2/hazards."""
        available, reason = self.availability()
        events_ok, events_reason = self.events_availability()
        return {
            "id": self.id,
            "name": self.name,
            "tagline": self.tagline,
            # A module in the registry is enabled by definition — the
            # registry only ever contains wired modules (registry._build);
            # per-capability runtime state is reported under analysis/events.
            "enabled": True,
            "analysis": {"available": available, "reason": reason},
            "events": {"available": events_ok, "reason": events_reason},
            "temporal_coverage": self.temporal_coverage(),
            "sources": self.sources(),
            "provenance": {
                "module": f"{type(self).__module__}.{type(self).__name__}",
                "sources_declared_by": (
                    "the hazard module's map-layer declarations — the same "
                    "source/URL pairs shown in the map layer panel"
                ),
                "indicator_status": (
                    "Levels are screening indicators computed from the "
                    "listed real datasets unless explicitly labelled "
                    "validated (docs/EVIDENCE_ARCHITECTURE.md)."
                ),
            },
        }
