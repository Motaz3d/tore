"""
Talaix evidence core — one typed evidence record for every claim.

Replaces the ad-hoc provenance-dict convention with a single
:class:`EvidenceRecord` used by all new modules, plus a documented alias
mapping that upgrades legacy wildfire provenance dicts
(``{kind: observed|derived|modeled|forecast|unavailable, …}``) without
breaking the working pipeline.

Norms (docs/EVIDENCE_ARCHITECTURE.md):

- Every record states source, reference period, method, and uncertainty
  where applicable.
- ``content_hash`` binds a claim to the exact source bytes it rests on,
  wherever the payload is available.
- ``UNKNOWN`` is a first-class status and records *why*.

Dependency-free; no I/O.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .ontology import ClaimStatus, Confidence, EvidenceClass, TemporalClass


def utcnow_iso() -> str:
    """UTC timestamp (ISO-8601, ``Z`` suffix) — the single clock shared by
    every engine and API layer. Import this; never re-declare it."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_utcnow_iso = utcnow_iso  # legacy private alias


def content_hash(payload: Any) -> str:
    """Stable SHA-256 of a JSON-serialisable payload.

    Used to bind a claim to the source data it rests on. Returns the hex
    digest; callers store it in ``EvidenceRecord.content_hash``.
    """

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_id(payload: Any) -> str:
    """Short stable identifier for an evidence record (16 hex chars)."""

    return content_hash(payload)[:16]


@dataclass
class EvidenceRecord:
    """One traceable record behind a claim.

    Required: ``evidence_class``, ``claim_status``, ``temporal``, ``source``.
    Everything else is optional but encouraged; ``limitations`` is the
    honesty channel.
    """

    evidence_class: str            # EvidenceClass value
    claim_status: str              # ClaimStatus value
    temporal: str                  # TemporalClass value
    source: str                    # human-readable origin, e.g. "Open-Meteo archive (ERA5)"
    dataset: Optional[str] = None  # e.g. "ERA5 daily, single level"
    provider_url: Optional[str] = None
    link: Optional[str] = None     # request URL / document URL where applicable
    location: Optional[Dict[str, float]] = None
    reference_period: Optional[Dict[str, str]] = None  # {"start": …, "end": …}
    acquired_at: str = field(default_factory=_utcnow_iso)
    method: Optional[str] = None
    resolution: Optional[str] = None
    confidence: str = Confidence.MEDIUM.value
    license: Optional[str] = None
    limitations: Optional[str] = None
    content_hash: Optional[str] = None

    def __post_init__(self) -> None:
        # Validate the three controlled vocabularies early and loudly.
        EvidenceClass(self.evidence_class)
        ClaimStatus(self.claim_status)
        TemporalClass(self.temporal)

    @property
    def id(self) -> str:
        """Stable id derived from the record's content (excluding acquired_at)."""

        basis = asdict(self)
        basis.pop("acquired_at", None)
        return evidence_id(basis)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_id"] = self.id
        return d

    # -- constructors ------------------------------------------------------

    @classmethod
    def open_data(
        cls,
        source: str,
        *,
        status: str = ClaimStatus.OBSERVED.value,
        temporal: str = TemporalClass.OBSERVED.value,
        **kw: Any,
    ) -> "EvidenceRecord":
        return cls(EvidenceClass.OPEN_DATA_OFFICIAL.value, status, temporal, source, **kw)

    @classmethod
    def satellite(
        cls,
        source: str,
        *,
        status: str = ClaimStatus.OBSERVED.value,
        temporal: str = TemporalClass.OBSERVED.value,
        **kw: Any,
    ) -> "EvidenceRecord":
        return cls(EvidenceClass.SATELLITE_EO.value, status, temporal, source, **kw)

    @classmethod
    def scientific(cls, source: str, **kw: Any) -> "EvidenceRecord":
        kw.setdefault("status", ClaimStatus.DOCUMENTED.value)
        kw.setdefault("temporal", TemporalClass.HISTORICAL.value)
        return cls(EvidenceClass.SCIENTIFIC.value, kw.pop("status"), kw.pop("temporal"), source, **kw)

    @classmethod
    def modelled(cls, source: str, *, method: str, **kw: Any) -> "EvidenceRecord":
        return cls(
            EvidenceClass.MODELLED.value,
            ClaimStatus.MODELLED.value,
            kw.pop("temporal", TemporalClass.OBSERVED.value),
            source,
            method=method,
            **kw,
        )

    @classmethod
    def media(cls, source: str, *, link: str, **kw: Any) -> "EvidenceRecord":
        """Media evidence: metadata + link only; status can be at most REPORTED."""

        status = kw.pop("status", ClaimStatus.REPORTED.value)
        if status not in (ClaimStatus.REPORTED.value, ClaimStatus.UNKNOWN.value):
            status = ClaimStatus.REPORTED.value
        return cls(
            EvidenceClass.MEDIA.value,
            status,
            kw.pop("temporal", TemporalClass.HISTORICAL.value),
            source,
            link=link,
            **kw,
        )

    @classmethod
    def unknown(cls, source: str, *, why: str, **kw: Any) -> "EvidenceRecord":
        """An honest 'we do not know' record: status UNKNOWN, reason in limitations."""

        return cls(
            kw.pop("evidence_class", EvidenceClass.OPEN_DATA_OFFICIAL.value),
            ClaimStatus.UNKNOWN.value,
            kw.pop("temporal", TemporalClass.OBSERVED.value),
            source,
            limitations=why,
            confidence=Confidence.LOW.value,
            **kw,
        )


# ---------------------------------------------------------------------------
# Legacy provenance upgrade
# ---------------------------------------------------------------------------

#: Legacy ``kind`` values (src/dashboard/real_analysis.py ``_prov``) mapped to
#: the unified vocabulary. "modeled" (US spelling) is normalised to MODELLED.
_LEGACY_KIND_MAP = {
    "observed": (ClaimStatus.OBSERVED.value, TemporalClass.OBSERVED.value),
    "derived": (ClaimStatus.INFERRED.value, TemporalClass.OBSERVED.value),
    "modeled": (ClaimStatus.MODELLED.value, TemporalClass.OBSERVED.value),
    "modelled": (ClaimStatus.MODELLED.value, TemporalClass.OBSERVED.value),
    "forecast": (ClaimStatus.MODELLED.value, TemporalClass.FORECAST.value),
    "unavailable": (ClaimStatus.UNKNOWN.value, TemporalClass.OBSERVED.value),
}


def upgrade_legacy_provenance(prov: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade one legacy provenance dict to the unified vocabulary.

    Input shape (src/dashboard/real_analysis.py)::

        {"kind": "observed|derived|modeled|forecast|unavailable",
         "source": …, "acquired": …, "resolution": …, "temporal": …,
         "retrieved_at": …, "quality": …, "limitations": …}

    Returns a new dict with all original keys preserved plus:

    - ``claim_status`` / ``temporal_class`` from the alias table,
    - ``kind`` normalised (``modeled`` → ``modelled``),
    - ``evidence_class`` inferred from the source label where possible.

    Never raises on unknown kinds: they map to ``UNKNOWN`` honestly.
    """

    out = dict(prov)
    kind = str(prov.get("kind", "unavailable")).lower()
    status, temporal = _LEGACY_KIND_MAP.get(
        kind, (ClaimStatus.UNKNOWN.value, TemporalClass.OBSERVED.value)
    )
    out["kind"] = "modelled" if kind == "modeled" else kind
    out["claim_status"] = status
    out["temporal_class"] = temporal
    out["evidence_class"] = _infer_evidence_class(str(prov.get("source", "")), status)
    return out


def upgrade_provenance_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade a whole provenance dict keyed by component (analysis payloads)."""

    upgraded: Dict[str, Any] = {}
    for component, prov in block.items():
        if isinstance(prov, dict) and "kind" in prov:
            upgraded[component] = upgrade_legacy_provenance(prov)
        else:
            upgraded[component] = prov
    return upgraded


def _infer_evidence_class(source: str, status: str) -> str:
    """Best-effort evidence class from a legacy source label."""

    s = source.lower()
    if status == ClaimStatus.MODELLED.value:
        return EvidenceClass.MODELLED.value
    if any(k in s for k in ("sentinel", "viirs", "modis", "firms", "satellite", "landsat")):
        return EvidenceClass.SATELLITE_EO.value
    if any(k in s for k in ("van wagner", "scott", "burgan", "rothermel", "literature")):
        return EvidenceClass.SCIENTIFIC.value
    return EvidenceClass.OPEN_DATA_OFFICIAL.value
