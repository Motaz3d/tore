"""Shared network-free fakes for the public engine test-suite.

Verbatim copy of the fakes used by the platform monorepo's TX suite
(``tests/test_tx_core.py``), so the tests here exercise the engine through
exactly the same injected surface as the monorepo does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from tx_core.engine import TXEngine


@dataclass
class FakeLevel:
    label: str = "Moderate"
    score: Optional[float] = 0.5
    score_max: float = 1.0
    basis: str = "fake basis"
    validated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FakeHazardModule:
    def __init__(self, hazard_id: str = "flood", status: str = "ok",
                 summary: str = "real-data summary", level: Any = None,
                 evidence: Optional[List[Dict[str, Any]]] = None,
                 sources: Optional[List[Dict[str, str]]] = None,
                 unavailable_reason: Optional[str] = None,
                 raise_on_analyze: bool = False) -> None:
        self.id = hazard_id
        self._status = status
        self._summary = summary
        self._level = level or FakeLevel()
        self._evidence = evidence or [{"kind": "observed", "source": "fake-source"}]
        self._sources = sources or [{"name": "Fake Open Data", "url": "https://example.test/"}]
        self._unavailable_reason = unavailable_reason
        self._raise_on_analyze = raise_on_analyze

    def analyze(self, lat: float, lon: float, name: Optional[str] = None,
                **kw: Any) -> SimpleNamespace:
        if self._raise_on_analyze:
            raise RuntimeError("upstream exploded")
        return SimpleNamespace(
            hazard=self.id,
            status=self._status,
            summary=self._summary,
            level=self._level,
            blocks={"component": "value"},
            evidence=self._evidence,
            provenance={"kind": "observed", "source": "fake-source"},
            unavailable_reason=self._unavailable_reason,
        )

    def sources(self) -> List[Dict[str, str]]:
        return list(self._sources)

    def descriptor(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": f"Fake {self.id.title()}",
            "tagline": "fake",
            "enabled": True,
            "analysis": {"available": True, "reason": None},
            "events": {"available": False, "reason": "no events in tests"},
            "temporal_coverage": {},
            "sources": self._sources,
        }


def make_engine(modules: Dict[str, FakeHazardModule],
                ghost: Optional[str] = None) -> TXEngine:
    """An engine whose hazard ids include real fakes + an optional 'ghost'."""

    def registry(hazard_id: str) -> Any:
        return None if hazard_id == ghost else modules.get(hazard_id)

    def hazard_ids() -> List[str]:
        ids = list(modules)
        if ghost:
            ids.append(ghost)
        return ids

    return TXEngine(registry=registry, hazard_ids=hazard_ids)
