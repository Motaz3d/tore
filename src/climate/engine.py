"""
Talaix product-engine contract — one unified envelope for every product
engine (insurance, forensics, sustainability, supply chain, compound, …).

This is the product-layer mirror of the hazard plugin contract
(``hazards/base.py``): hazard modules produce ``HazardAnalysis``; product
engines combine hazard analyses into a decision-facing product. Every
product returns the same :class:`ProductResult` envelope so reports, PDFs,
admin views and user portfolios can treat all engines uniformly.

Honesty contract (docs/EVIDENCE_ARCHITECTURE.md) — identical rules:

- ``status="unavailable"`` with ``unavailable_reason`` when real data
  cannot be obtained — never fabricated numbers.
- Scores/levels are screening indicators until validated.
- The disclaimer is part of the envelope, never an afterthought.

Dependency-free; no I/O. Heavy dependencies belong inside engines and are
imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .evidence import utcnow_iso

# Version of this unified analytical-model contract itself (TAM — Talaix
# Analytical Model). Engine versions move independently (per-engine
# semver); bump this only when the envelope/contract changes.
TAM_VERSION = "1.0.0"

#: Keys every product envelope guarantees at the top level of ``to_dict()``.
ENVELOPE_KEYS = (
    "product",
    "status",
    "summary",
    "evidence",
    "disclaimer",
    "engine_version",
    "generated_at",
    "tam_version",
    "unavailable_reason",
)


@dataclass
class ProductResult:
    """Uniform result envelope across product engines.

    ``blocks`` carries the product-specific payload. ``to_dict()`` merges
    the blocks at the top level and then stamps the envelope keys, so the
    envelope always wins on a key collision — every consumer can rely on
    :data:`ENVELOPE_KEYS` being present and accurate.
    """

    product: str                     # engine id, e.g. "insurance"
    status: str = "ok"               # "ok" | "partial" | "unavailable"
    summary: str = ""
    blocks: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    disclaimer: str = ""
    engine_version: str = ""
    generated_at: str = field(default_factory=utcnow_iso)
    tam_version: str = TAM_VERSION
    unavailable_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = dict(self.blocks)
        d.update({
            "product": self.product,
            "status": self.status,
            "summary": self.summary,
            "evidence": self.evidence,
            "disclaimer": self.disclaimer,
            "engine_version": self.engine_version,
            "generated_at": self.generated_at,
            "tam_version": self.tam_version,
            "unavailable_reason": self.unavailable_reason,
        })
        return d


class ProductEngine:
    """Base class for product engines.

    Subclasses set ``id`` / ``name`` / ``engine_version`` / ``disclaimer``
    and build results through :meth:`result` (or :meth:`unavailable`), so
    the envelope is always complete and the clock is always
    :func:`~.evidence.utcnow_iso`.
    """

    id: str = ""
    name: str = ""
    engine_version: str = "0.0.0"
    disclaimer: str = ""

    def result(
        self,
        *,
        status: str = "ok",
        summary: str = "",
        blocks: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        unavailable_reason: Optional[str] = None,
    ) -> ProductResult:
        return ProductResult(
            product=self.id,
            status=status,
            summary=summary,
            blocks=dict(blocks or {}),
            evidence=list(evidence or []),
            disclaimer=self.disclaimer,
            engine_version=self.engine_version,
            unavailable_reason=unavailable_reason,
        )

    def unavailable(self, reason: str, **kw: Any) -> ProductResult:
        """The honesty path: a declared non-result with the reason stated."""
        return self.result(status="unavailable", unavailable_reason=reason, **kw)

    def descriptor(self) -> Dict[str, str]:
        """Public descriptor (registry/docs surfaces)."""
        return {
            "id": self.id,
            "name": self.name,
            "engine_version": self.engine_version,
            "tam_version": TAM_VERSION,
        }
