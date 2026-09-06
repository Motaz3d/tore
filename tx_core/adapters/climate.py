"""
Adapter over ``src.climate`` — hazards, evidence, ontology and the product
engine envelope. This is the primary analytical authority tx_core consumes:
TX never re-implements a hazard; it orchestrates the platform's registered
hazard modules (``src.climate.registry``) unchanged.

All imports are lazy so the adapter is safe in minimal environments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def registry() -> Any:
    """The platform hazard registry module (imported lazily)."""
    from src.climate import registry as _registry

    return _registry


def get_hazard_module(hazard_id: str) -> Optional[Any]:
    """One registered hazard module, or ``None`` if unknown/unbuilt."""
    try:
        return registry().get(hazard_id)
    except Exception:
        return None


def hazard_ids() -> List[str]:
    try:
        return registry().ids()
    except Exception:
        return []


def hazard_descriptors() -> List[Dict[str, Any]]:
    try:
        return registry().descriptors()
    except Exception:
        return []


def all_hazard_modules() -> List[Any]:
    try:
        return registry().all_modules()
    except Exception:
        return []


def utcnow_iso() -> str:
    """The single shared clock used across the platform and TX results."""
    from src.climate.evidence import utcnow_iso as _utcnow_iso

    return _utcnow_iso()


def evidence_record(*args: Any, **kwargs: Any) -> Any:
    """Build a platform :class:`EvidenceRecord` (validated vocabulary)."""
    from src.climate.evidence import EvidenceRecord

    return EvidenceRecord(*args, **kwargs)


def ontology() -> Any:
    """The platform ontology module (HazardType, EvidenceClass, …)."""
    from src.climate import ontology as _ontology

    return _ontology


def tam_version() -> str:
    """Version of the Talaix Analytical Model envelope (``src.climate.engine``)."""
    try:
        from src.climate.engine import TAM_VERSION as _v

        return _v
    except Exception:
        return "0.0.0"
