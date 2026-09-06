"""
TX Engine — the orchestrator that turns platform hazard modules into one
uniform, reproducible TX analysis.

``TXEngine`` never computes a hazard itself: it resolves the requested
hazards through the existing platform registry (or an injected fake for
tests), runs each module's ``analyze()`` unchanged, and wraps the results in
the standard :class:`~tx_core.models.TxResult` envelope with provenance and
engine versions.

Analysis levels (docs/TX_ENGINE.md §"Analysis levels") — the engine tracks
which TX layer each hazard result belongs to, but phase-1 hazards are all
``TX-1 DETERMINISTIC`` (platform hazard modules run deterministic screening
on real data). Higher levels are reserved for future statistical/spatial/ML
steps and are advertised, not faked.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ._version import TAM_VERSION, TX_VERSION
from .adapters import climate as adapters  # noqa: F401 — explicit submodule (adapter pkg does not re-export)
from .adapters import legacy_v1 as legacy_adapters  # noqa: F401
from .adapters import products as product_adapters  # noqa: F401
from .models import TxHazardResult, TxLocation, TxResult

from src.climate.tx_seal import seal_code  # noqa: E402

#: TX analysis levels (advertised; hazards are progressively upgraded).
TX_LEVELS: Dict[int, str] = {
    0: "RETRIEVAL",
    1: "DETERMINISTIC",
    2: "STATISTICAL",
    3: "SPATIAL",
    4: "PREDICTIVE",
    5: "ML",
    6: "RESEARCH",
    7: "REASONING",
    8: "DECISION_INTELLIGENCE",
}

#: Depth presets — deeper analyses may run more hazards/layers later.
DEPTHS = ("quick", "standard", "deep")

VALID_HAZARD_RE = None  # hazard ids are validated against the registry, not a regex


class TXEngine:
    """Run TX analyses over the platform's registered hazard modules.

    :param registry: optional ``callable(hazard_id) -> module|None`` used to
        resolve hazard modules; defaults to the platform registry. Tests
        inject a fake here to stay network-free.
    :param hazard_ids: optional ``callable() -> list[str]`` for the set of
        available hazard ids (defaults to the platform registry ids).
    :param products: optional ``callable(product_id) -> module|None`` used to
        resolve product engines (TX-2+ analyses); defaults to the platform
        product registry (``adapters.products``). Tests inject fakes here.
    :param product_ids: optional ``callable() -> list[str]`` for the set of
        available product ids (defaults to the platform product registry).
    :param legacy_analysis: optional ``callable(lat, lon, name) -> dict``
        backing the TX-0/TX-1 facade for the legacy v1 /api/analyze pipeline
        (defaults to the real cached pipeline via ``adapters.legacy_v1``).
        Tests inject a fake here to stay network-free.
    :param version: engine version stamped on every result.
    """

    def __init__(
        self,
        registry: Optional[Callable[[str], Optional[Any]]] = None,
        hazard_ids: Optional[Callable[[], List[str]]] = None,
        legacy_analysis: Optional[Callable[[float, float, str], Dict[str, Any]]] = None,
        products: Optional[Callable[[str], Optional[Any]]] = None,
        product_ids: Optional[Callable[[], List[str]]] = None,
        version: str = TX_VERSION,
    ) -> None:
        self._registry = registry
        self._hazard_ids = hazard_ids
        self._legacy_analysis = legacy_analysis
        self._products = products
        self._product_ids = product_ids
        self.version = version

    # -- resolution ---------------------------------------------------------

    def _resolve(self, hazard_id: str) -> Optional[Any]:
        if self._registry is not None:
            return self._registry(hazard_id)
        return adapters.get_hazard_module(hazard_id)

    def _resolve_product(self, product_id: str) -> Optional[Any]:
        if self._products is not None:
            return self._products(product_id)
        return product_adapters.get_product_module(product_id)

    def available_hazard_ids(self) -> List[str]:
        if self._hazard_ids is not None:
            return sorted(self._hazard_ids())
        return sorted(adapters.hazard_ids())

    def available_product_ids(self) -> List[str]:
        if self._product_ids is not None:
            return sorted(self._product_ids())
        return sorted(product_adapters.product_ids())

    def resolve_hazards(self, hazards: Optional[List[str]]) -> List[str]:
        """The concrete hazard ids to run (unknown ids are dropped, honestly)."""
        requested = [h.strip().lower() for h in (hazards or []) if h and h.strip()]
        available = set(self.available_hazard_ids())
        if not requested:
            return sorted(available)
        return [h for h in requested if h in available]

    def resolve_products(self, analyses: Optional[List[str]]) -> List[str]:
        """The concrete product ids to run (unknown ids are dropped, honestly).

        Unlike hazards, products never default to "all": a product analysis
        runs only when explicitly requested.
        """
        requested = [a.strip().lower() for a in (analyses or []) if a and a.strip()]
        available = set(self.available_product_ids())
        return [a for a in requested if a in available]

    # -- analysis -----------------------------------------------------------

    def analyze(
        self,
        lat: float,
        lon: float,
        hazards: Optional[List[str]] = None,
        depth: str = "standard",
        name: Optional[str] = None,
        on_hazard: Optional[Callable[[TxHazardResult, int, int], None]] = None,
        analyses: Optional[List[str]] = None,
    ) -> TxResult:
        """Run a TX analysis at (lat, lon).

        The returned envelope always stamps engine versions and a
        deterministic ``analysis_id`` (same inputs + day → same id), so any
        TX analysis can be reproduced and audited.

        :param on_hazard: optional progress callback invoked after each
            hazard completes as ``on_hazard(result, completed, total)`` —
            used by the TX job runner to expose per-hazard progress while a
            deep analysis is running. Callback errors are swallowed:
            progress bookkeeping must never break an analysis.
        :param analyses: optional product-analysis ids (TX-2+ product
            engines such as ``insurance``/``verification``/``sustainability``).
            Products run only when explicitly requested (never by default),
            land in the same ``results[]`` list stamped ``tx_level=2``, and
            unknown ids are dropped honestly.
        """
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")

        depth = (depth or "standard").lower()
        if depth not in DEPTHS:
            raise ValueError(f"Invalid depth {depth!r}; choose from {DEPTHS}")

        location = TxLocation(lat=lat, lon=lon, name=name)
        chosen = self.resolve_hazards(hazards)
        chosen_products = self.resolve_products(analyses)

        results: List[TxHazardResult] = []
        total = len(chosen) + len(chosen_products)
        completed = 0

        def _progress() -> None:
            if on_hazard is not None:
                try:
                    on_hazard(results[-1], completed, total)
                except Exception:  # noqa: BLE001 — bookkeeping never breaks analysis
                    pass

        for hid in chosen:
            module = self._resolve(hid)
            if module is None:
                results.append(
                    TxHazardResult(
                        hazard=hid,
                        status="unavailable",
                        summary=f"No registered TX module for hazard '{hid}'.",
                        unavailable_reason=(
                            "The hazard registry does not contain a module wired to "
                            "real data for this id — TX reports it honestly as unknown."
                        ),
                        tx_level=1,
                    )
                )
            else:
                try:
                    analysis = module.analyze(lat=lat, lon=lon, name=name)
                    hazard_result = TxHazardResult.from_hazard_analysis(analysis)
                    if hazard_result.tx_level is None:
                        hazard_result.tx_level = 1
                    results.append(hazard_result)
                except Exception as exc:  # noqa: BLE001 — never invent numbers
                    results.append(
                        TxHazardResult(
                            hazard=hid,
                            status="unavailable",
                            summary=f"{hid} analysis failed without producing data.",
                            unavailable_reason=str(exc),
                            tx_level=1,
                        )
                    )
            completed += 1
            _progress()

        for pid in chosen_products:
            product = self._resolve_product(pid)
            if product is None:
                results.append(
                    TxHazardResult(
                        hazard=pid,
                        status="unavailable",
                        summary=f"No registered TX product engine for '{pid}'.",
                        unavailable_reason=(
                            "The product registry does not contain a location-first "
                            "engine for this id — TX reports it honestly as unknown."
                        ),
                        tx_level=2,
                    )
                )
            else:
                try:
                    analysis = product.analyze(lat=lat, lon=lon, name=name)
                    product_result = TxHazardResult.from_hazard_analysis(analysis)
                    product_result.tx_level = getattr(
                        product, "tx_level", 2)
                    results.append(product_result)
                except Exception as exc:  # noqa: BLE001 — never invent numbers
                    results.append(
                        TxHazardResult(
                            hazard=pid,
                            status="unavailable",
                            summary=f"{pid} product analysis failed without producing data.",
                            unavailable_reason=str(exc),
                            tx_level=2,
                        )
                    )
            completed += 1
            _progress()

        overall = self._overall_status(results)
        result = TxResult(
            analysis_id=self.analysis_id(lat=lat, lon=lon, hazards=chosen,
                                         depth=depth, analyses=chosen_products),
            location=location,
            depth=depth,
            results=results,
            engine_version=self.version,
            tx_version=TX_VERSION,
            tam_version=TAM_VERSION,
            status=overall,
            summary=self._summary(overall, len(results)),
            evidence=[e for r in results for e in r.evidence],
            sources=self.sources(hazard_ids=chosen),
        )
        result.authenticity_code = seal_code({
            "analysis_id": result.analysis_id,
            "results": [r.to_dict() for r in result.results],
            "engine_version": result.engine_version,
            "tx_version": result.tx_version,
            "tam_version": result.tam_version,
        })
        return result

    # -- legacy v1 facade (TX-0/TX-1) ----------------------------------------

    def legacy_analyze(self, lat: float, lon: float,
                       name: Optional[str] = None) -> tuple:
        """TX-0/TX-1 facade for the legacy ``GET /api/analyze`` pipeline.

        Runs the existing real-data pipeline (``TalaixRealAnalyser`` through
        its shared 15-minute cache) *unchanged* — TX never re-implements it —
        and returns ``(payload, tx_meta)``:

        - ``payload``: the EXACT legacy v1 dict. It is never mutated, so the
          existing v1 wire contract stays byte-identical for every client.
        - ``tx_meta``: TX envelope metadata as a *separate* side-channel
          (deterministic ``analysis_id``, engine/tx/tam versions, depth and
          the payload's ``generated_at``) — never injected into the legacy
          payload, but available for audit/telemetry.

        Exceptions from the underlying pipeline propagate untouched so the
        route keeps its exact error semantics (502). Invalid coordinates
        raise ``ValueError`` with the same rules as :meth:`analyze`.
        """
        lat, lon = float(lat), float(lon)
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")

        name = name or f"{lat:.4f}, {lon:.4f}"
        if self._legacy_analysis is not None:
            payload = self._legacy_analysis(lat, lon, name)
        else:
            payload = legacy_adapters.cached_analysis(lat, lon, name)

        tx_meta = {
            "analysis_id": self.analysis_id(
                lat=lat, lon=lon, hazards=["wildfire"], depth="standard"
            ),
            "hazards": ["wildfire"],
            "depth": "standard",
            "engine_version": self.version,
            "tx_version": TX_VERSION,
            "tam_version": TAM_VERSION,
            "generated_at": payload.get("generated_at"),
        }
        return payload, tx_meta

    # -- introspection ------------------------------------------------------

    def hazards(self) -> List[Dict[str, Any]]:
        """Public hazard descriptors (id/name/tagline/availability/sources)."""
        if self._registry is not None:
            out = []
            for hid in self.available_hazard_ids():
                module = self._resolve(hid)
                if module is None:
                    continue
                try:
                    out.append(module.descriptor())
                except Exception:
                    out.append({"id": hid, "name": hid, "enabled": True})
            return out
        return adapters.hazard_descriptors()

    def products(self) -> List[Dict[str, Any]]:
        """Public product-engine descriptors (id/name/kind/tx_level/version).

        Unresolvable products are honestly absent from the list.
        """
        out = []
        for pid in self.available_product_ids():
            product = self._resolve_product(pid)
            if product is None:
                continue
            try:
                out.append(product.descriptor())
            except Exception:  # noqa: BLE001 — honest minimal descriptor
                out.append({"id": pid, "name": pid, "kind": "product"})
        return out

    def sources(self, hazard_ids: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """De-duplicated official data sources behind the given hazards."""
        ids = hazard_ids or self.available_hazard_ids()
        seen: List[Dict[str, str]] = []
        for hid in ids:
            module = self._resolve(hid)
            if module is None:
                continue
            try:
                for src in module.sources():
                    if src.get("name") and not any(
                        s.get("name") == src["name"] for s in seen
                    ):
                        seen.append({"name": src["name"], "url": src.get("url", "")})
            except Exception:
                continue
        return seen

    def version_info(self) -> Dict[str, str]:
        """The exact engine versions behind every TX result."""
        return {
            "tx_version": TX_VERSION,
            "engine_version": self.version,
            "tam_version": TAM_VERSION,
            "levels": TX_LEVELS,
        }

    def analysis_id(self, *, lat: float, lon: float, hazards: List[str],
                    depth: str,
                    analyses: Optional[List[str]] = None) -> str:
        """Deterministic, day-scoped analysis id: ``TX-YYYYMMDD-<hex8>``.

        Same inputs on the same UTC day → same id (reproducibility). The
        ``analyses`` key enters the basis ONLY when non-empty, so existing
        hazard-only ids stay byte-stable.
        """
        basis = {
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "hazards": sorted(hazards),
            "depth": depth,
            "engine_version": self.version,
            "tx_version": TX_VERSION,
        }
        if analyses:
            basis["analyses"] = sorted(analyses)
        digest = hashlib.sha256(
            repr(sorted(basis.items())).encode("utf-8")
        ).hexdigest()[:8]
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"TX-{day}-{digest}"

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _overall_status(results: List[TxHazardResult]) -> str:
        if not results:
            return "unavailable"
        statuses = {r.status for r in results}
        if statuses <= {"ok"}:
            return "ok"
        if statuses <= {"ok", "partial"}:
            return "partial"
        return "partial" if "ok" in statuses or "partial" in statuses else "unavailable"

    @staticmethod
    def _summary(status: str, n: int) -> str:
        if n == 0:
            return "No hazards could be analysed for this location."
        if status == "ok":
            return f"TX analysis complete: {n} hazard(s) produced real-data results."
        if status == "partial":
            return (
                f"TX analysis partially complete: {n} hazard(s) ran; some "
                "components are unavailable and reported honestly."
            )
        return f"TX analysis unavailable: none of the {n} hazard(s) produced data."


#: Convenience alias — ``TX`` is the public entry point (docs/TX_ENGINE.md).
TX = TXEngine
