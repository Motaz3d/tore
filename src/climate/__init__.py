"""
Talaix climate-intelligence core.

Multi-hazard ontology, evidence architecture, historical event model,
hazard plugin registry, economic exposure and solutions intelligence.
See ``docs/PRODUCT_VISION.md`` and companions.

Submodules are import-light; hazard plugins import their heavy
dependencies lazily so that ``import src.climate`` never pulls in the
wildfire engine or network clients by itself.
"""

from .ontology import (  # noqa: F401
    HazardType,
    ClaimStatus,
    TemporalClass,
    EvidenceClass,
    Confidence,
    Observation,
    Exposure,
    Impact,
    Response,
    SolutionRef,
    Uncertainty,
    validate_cause,
)
from .evidence import EvidenceRecord, content_hash, upgrade_legacy_provenance  # noqa: F401
