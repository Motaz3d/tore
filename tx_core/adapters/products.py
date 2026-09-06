"""
Adapter over the platform's location-first product engines
(``src.climate.insurance`` / ``verification`` / ``sustainability`` /
``licensing``).

TX never re-implements a product engine — this module wraps each engine's
existing public entry point behind the same narrow surface the hazard
adapters expose (``analyze`` / ``sources`` / ``descriptor``), so the TX
engine can register product analyses next to hazards under one envelope
(docs/TX_ENGINE.md §9). All platform imports are lazy (inside methods),
keeping ``import tx_core`` light.

Honesty rules (unchanged):

- Only engines with a *real public location-first entry point* are
  registered. Claim-first engines (forensics ``assess_case``, supply-chain
  ``evaluate_claim``) require a case/claim request axis and are therefore
  NOT registered as location analyses — registering them would fake a
  capability they do not have.
- A product that cannot produce real data returns ``status="unavailable"``
  with an explicit reason — derived from the product's own payload, never
  invented. Product results are TX-2 analyses (they combine hazard
  modules); the wrapper stamps ``tx_level=2``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

#: Registered location-first product engines (registry order).
PRODUCT_IDS = ("insurance", "licensing", "verification", "sustainability")

#: TX analysis level for product analyses (TX-2 STATISTICAL and up —
#: products combine hazard modules into decision-facing envelopes).
PRODUCT_TX_LEVEL = 2

_ENVELOPE_KEYS = ("status", "summary", "evidence", "unavailable_reason")


def product_ids() -> List[str]:
    return sorted(PRODUCT_IDS)


def _namespace(product_id: str, payload: Dict[str, Any], *,
               status: str, summary: str,
               unavailable_reason: Optional[str]) -> Any:
    """Adapt a product payload dict to the hazard-analysis surface
    (``TxHazardResult.from_hazard_analysis`` consumes the attributes)."""
    return SimpleNamespace(
        hazard=product_id,
        status=status,
        summary=summary,
        level=None,
        blocks={k: v for k, v in payload.items() if k not in _ENVELOPE_KEYS},
        evidence=list(payload.get("evidence") or []),
        provenance={
            "kind": "product_engine",
            "engine": product_id,
            "engine_version": payload.get("engine_version", ""),
            "tam_version": payload.get("tam_version", ""),
        },
        unavailable_reason=unavailable_reason,
    )


def _verification_status(payload: Dict[str, Any]) -> str:
    """Derive an honest status from the verification's own claim checks."""
    checks = payload.get("hazard_checks") or []
    assessed = [c for c in checks if c.get("claim_status") != "UNKNOWN"]
    if not checks or not assessed:
        return "unavailable"
    return "ok" if len(assessed) == len(checks) else "partial"


def _sustainability_status(payload: Dict[str, Any]) -> str:
    """Derive an honest status from the portfolio summary's own counts."""
    summary = payload.get("portfolio_summary") or {}
    sites = int(summary.get("site_count") or 0)
    ok = int(summary.get("ok_count") or 0)
    if sites == 0 or ok == 0:
        return "unavailable"
    return "ok" if ok == sites else "partial"


class _InsuranceProduct:
    """``src.climate.insurance.build_risk_profile`` — the platform's
    reference ProductEngine, exposed unchanged as a TX-2 analysis."""

    id = "insurance"
    tx_level = PRODUCT_TX_LEVEL

    def analyze(self, lat: float, lon: float, name: Optional[str] = None,
                **kw: Any) -> Any:
        from src.climate.insurance import build_risk_profile

        payload = build_risk_profile(float(lat), float(lon), name=name)
        return _namespace(
            self.id, payload,
            status=payload.get("status", "ok"),
            summary=payload.get("summary", ""),
            unavailable_reason=payload.get("unavailable_reason"),
        )

    def sources(self) -> List[Dict[str, str]]:
        # Data sources are carried per-peril inside the profile's evidence
        # blocks; the wrapper adds none of its own.
        return []

    def descriptor(self) -> Dict[str, Any]:
        from src.climate.insurance import InsuranceEngine

        engine = InsuranceEngine()
        return {
            "id": self.id,
            "name": engine.name,
            "kind": "product",
            "tx_level": self.tx_level,
            "engine_version": engine.engine_version,
        }


class _VerificationProduct:
    """``src.climate.verification.verify_asset`` — the location-first
    physical-evidence check (the site-level face of the forensics stack)."""

    id = "verification"
    tx_level = PRODUCT_TX_LEVEL

    def analyze(self, lat: float, lon: float, name: Optional[str] = None,
                **kw: Any) -> Any:
        from src.climate.verification import verify_asset

        payload = verify_asset(float(lat), float(lon), name=name)
        status = _verification_status(payload)
        return _namespace(
            self.id, payload,
            status=status,
            summary=payload.get("summary", ""),
            unavailable_reason=(
                None if status != "unavailable"
                else "No hazard could be assessed with real data at this location."
            ),
        )

    def sources(self) -> List[Dict[str, str]]:
        return []

    def descriptor(self) -> Dict[str, Any]:
        from src.climate.verification import ENGINE_VERSION

        return {
            "id": self.id,
            "name": "Physical Evidence Verification",
            "kind": "product",
            "tx_level": self.tx_level,
            "engine_version": ENGINE_VERSION,
        }


def _licensing_status(payload: Dict[str, Any]) -> str:
    """Derive an honest status from the dossier's own hazard screenings."""
    checks = (payload.get("evidence_base") or {}).get("hazard_exposure") or []
    assessed = [c for c in checks if c.get("status") != "unavailable"]
    if not checks or not assessed:
        return "unavailable"
    return "ok" if len(assessed) == len(checks) else "partial"


class _SustainabilityProduct:
    """``src.climate.sustainability.build_sustainability_evidence`` — run
    as a single-site screen: the company block is a *display label* (the
    place name or coordinates), never analytical input."""

    id = "sustainability"
    tx_level = PRODUCT_TX_LEVEL

    def analyze(self, lat: float, lon: float, name: Optional[str] = None,
                **kw: Any) -> Any:
        from src.climate.sustainability import build_sustainability_evidence

        label = name or f"{float(lat):.4f}, {float(lon):.4f}"
        payload = build_sustainability_evidence(
            company={"name": label},
            assets=[{"lat": float(lat), "lon": float(lon), "name": name}],
        )
        status = _sustainability_status(payload)
        return _namespace(
            self.id, payload,
            status=status,
            summary=(
                f"Sustainability evidence screen for {label} "
                f"({(payload.get('portfolio_summary') or {}).get('ok_count', 0)}"
                f"/{(payload.get('portfolio_summary') or {}).get('site_count', 0)} sites ok)."
            ),
            unavailable_reason=(
                None if status != "unavailable"
                else "No site could be verified with real data at this location."
            ),
        )

    def sources(self) -> List[Dict[str, str]]:
        return []

    def descriptor(self) -> Dict[str, Any]:
        from src.climate.sustainability import ENGINE_VERSION

        return {
            "id": self.id,
            "name": "Sustainability Evidence Screen",
            "kind": "product",
            "tx_level": self.tx_level,
            "engine_version": ENGINE_VERSION,
        }


class _LicensingProduct:
    """``src.climate.licensing.build_licensing_dossier`` — the location-first
    pre-draft environmental licensing evidence dossier. TX runs the default
    applicant-side screen; the full request axis (side/typology/permit type)
    is exposed via ``POST /api/v2/licensing/dossier``."""

    id = "licensing"
    tx_level = PRODUCT_TX_LEVEL

    def analyze(self, lat: float, lon: float, name: Optional[str] = None,
                **kw: Any) -> Any:
        from src.climate.licensing import build_licensing_dossier

        payload = build_licensing_dossier(
            lat=float(lat), lon=float(lon), name=name)
        status = _licensing_status(payload)
        return _namespace(
            self.id, payload,
            status=status,
            summary=payload.get("summary", ""),
            unavailable_reason=(
                None if status != "unavailable"
                else "No hazard could be screened with real data at this location."
            ),
        )

    def sources(self) -> List[Dict[str, str]]:
        return []

    def descriptor(self) -> Dict[str, Any]:
        from src.climate.licensing import ENGINE_VERSION

        return {
            "id": self.id,
            "name": "Environmental Licensing Dossier",
            "kind": "product",
            "tx_level": self.tx_level,
            "engine_version": ENGINE_VERSION,
        }


_PRODUCTS: Dict[str, Callable[[], Any]] = {
    "insurance": _InsuranceProduct,
    "licensing": _LicensingProduct,
    "verification": _VerificationProduct,
    "sustainability": _SustainabilityProduct,
}


def get_product_module(product_id: str) -> Optional[Any]:
    """Resolve a registered product wrapper by id (None when unknown)."""
    cls = _PRODUCTS.get(product_id)
    return cls() if cls is not None else None
