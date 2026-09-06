"""
Talaix climate ontology — the shared vocabulary of the platform.

This module defines the typed concepts every hazard module, event record,
exposure analysis and report builds on (see ``docs/CLIMATE_HAZARDS.md`` and
``docs/EVIDENCE_ARCHITECTURE.md``):

- :class:`HazardType` — the registered climate hazards.
- :class:`ClaimStatus` — epistemic status of a claim or event
  (``OBSERVED | DOCUMENTED | REPORTED | MODELLED | INFERRED | UNKNOWN``).
- :class:`TemporalClass` — temporal nature of data
  (``OBSERVED | HISTORICAL | FORECAST | PROJECTED | SCENARIO``).
- :class:`EvidenceClass` — the five evidence classes.
- Dataclasses for the ontology concepts: ``Observation``, ``Exposure``,
  ``Impact``, ``Response``, ``SolutionRef``, ``Uncertainty``.

Norms enforced here:

- ``UNKNOWN`` is a first-class, respected answer — never a failure.
- A cause may only be ``DOCUMENTED``; otherwise it is ``UNKNOWN``.
- Projections/scenarios are structurally separated from observations via
  ``TemporalClass`` — they never share a field with observed values.

The module is dependency-free and performs no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class HazardType(str, Enum):
    """Registered climate hazards. A hazard appears here only when a real
    data source is wired in — no placeholder hazards."""

    WILDFIRE = "wildfire"
    FLOOD = "flood"
    DROUGHT = "drought"
    HEAT = "heat"
    WIND = "wind"
    COASTAL = "coastal"
    EARTHQUAKE = "earthquake"        # geophysical (not climate) — wired 2026-09
    # Expansion candidates: registered with honest unavailable states until
    # a real pipeline is wired in (docs/CLIMATE_HAZARDS.md).
    DUST = "dust"
    VOLCANIC = "volcanic"


class ClaimStatus(str, Enum):
    """Epistemic status of a claim, event, or event attribute."""

    OBSERVED = "OBSERVED"          # measured by an instrument/authority at the time/place
    DOCUMENTED = "DOCUMENTED"      # established in an authoritative report/record
    REPORTED = "REPORTED"          # credible secondary source (incl. media); never overrides OBSERVED/DOCUMENTED
    MODELLED = "MODELLED"          # declared model, declared inputs
    INFERRED = "INFERRED"          # reasoned from other evidence; method stated
    UNKNOWN = "UNKNOWN"            # no adequate evidence — a valid answer


class TemporalClass(str, Enum):
    """Temporal nature of data. Projections/scenarios are never presented
    as observations."""

    OBSERVED = "OBSERVED"          # current / current window
    HISTORICAL = "HISTORICAL"      # past record
    FORECAST = "FORECAST"          # short-horizon model
    PROJECTED = "PROJECTED"        # long-horizon climate projection
    SCENARIO = "SCENARIO"          # conditional what-if


class EvidenceClass(str, Enum):
    """The five evidence classes (see docs/EVIDENCE_ARCHITECTURE.md §1)."""

    SCIENTIFIC = "SCIENTIFIC"
    SATELLITE_EO = "SATELLITE_EO"
    OPEN_DATA_OFFICIAL = "OPEN_DATA_OFFICIAL"
    MEDIA = "MEDIA"
    MODELLED = "MODELLED"


#: Trust order used when evidence classes disagree. MEDIA is last: it never
#: overrides scientific, satellite or official evidence.
EVIDENCE_TRUST_ORDER = (
    EvidenceClass.SCIENTIFIC,
    EvidenceClass.SATELLITE_EO,
    EvidenceClass.OPEN_DATA_OFFICIAL,
    EvidenceClass.MODELLED,
    EvidenceClass.MEDIA,
)


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Ontology dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Uncertainty:
    """Explicit limits of knowledge attached to a claim or analysis."""

    note: str
    confidence: str = Confidence.MEDIUM.value
    sources_of_uncertainty: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    """A measured fact from an instrument or authority."""

    quantity: str                    # e.g. "fire_radiative_power", "river_discharge"
    value: Any
    unit: Optional[str] = None
    observed_at: Optional[str] = None        # ISO timestamp/date of the measurement
    instrument: Optional[str] = None         # e.g. "VIIRS SNPP", "ERA5 reanalysis"
    status: str = ClaimStatus.OBSERVED.value
    temporal: str = TemporalClass.OBSERVED.value
    uncertainty: Optional[Uncertainty] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class Exposure:
    """Who/what is exposed in an area. Counts come only from real mapped
    data; completeness caveats are mandatory free-text."""

    category: str                    # e.g. "buildings", "critical_facilities", "agriculture"
    count: Optional[int] = None      # None => not quantified from available data
    description: Optional[str] = None
    source: Optional[str] = None
    completeness_caveat: Optional[str] = None
    status: str = ClaimStatus.OBSERVED.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Impact:
    """A consequence of an event. Only carried when documented/reported by
    a source — never estimated silently."""

    kind: str                        # e.g. "burned_area_ha", "displacement", "damage"
    value: Any = None
    unit: Optional[str] = None
    status: str = ClaimStatus.UNKNOWN.value
    source: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Response:
    """Containment / intervention information about an event."""

    description: str
    status: str = ClaimStatus.UNKNOWN.value   # DOCUMENTED only when an authority documents it
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolutionRef:
    """Lightweight reference to a solution (full solution contract lives in
    ``src/climate/solutions.py``)."""

    solution_id: str
    name: str
    classes: List[str] = field(default_factory=list)
    hazards_addressed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: Cause discipline: a cause is only ever DOCUMENTED (authoritative source)
#: or UNKNOWN. Helpers enforce this at the type level of the event model.
CAUSE_STATUSES = frozenset({ClaimStatus.DOCUMENTED.value, ClaimStatus.UNKNOWN.value})


def validate_cause(status: str) -> str:
    """Return ``status`` if it is a legal cause status, else ``UNKNOWN``.

    Never lets REPORTED/INFERRED/MODELLED through as a cause: media or
    model output can describe an event but cannot establish its cause.
    """

    return status if status in CAUSE_STATUSES else ClaimStatus.UNKNOWN.value
