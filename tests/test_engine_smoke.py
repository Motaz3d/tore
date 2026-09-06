"""Smoke tests for the extracted open engine — the public-repo guarantee.

These tests pin the contract the open repository promises on its own,
without the platform monorepo around it:

- ``import tx_core`` is light and exposes the documented surface;
- the real hazard registry is discoverable (ten hazard modules);
- the CLI entry points work (``tx version``, ``tx hazards``);
- an end-to-end analysis with an injected fake hazard produces the
  honest envelope (no network, no fabricated numbers).

The fuller engine test-suite lives in the platform monorepo; everything
here must pass against this repository alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import tx_core
from tx_core.engine import TXEngine
from tx_core.cli import main as cli_main


@dataclass
class FakeLevel:
    label: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label}


class FakeHazard:
    """Minimal hazard module honouring the TX hazard surface."""

    descriptor = {
        "id": "fake",
        "name": "Fake hazard (test)",
        "analysis": {"available": True},
    }

    def analyze(self, lat: float, lon: float, depth: str = "standard",
                name: Optional[str] = None) -> "FakeHazard":
        self.hazard = "fake"
        self.status = "ok"
        self.summary = f"fake analysis at {lat},{lon}"
        self.level = FakeLevel()
        self.blocks: Dict[str, Any] = {}
        self.evidence: list = []
        self.provenance: Dict[str, Any] = {"kind": "test"}
        self.unavailable_reason = None
        return self

    def sources(self) -> list:
        return [{"name": "Test source", "url": "https://example.org"}]


def test_public_surface_is_exported():
    assert tx_core.__version__
    assert tx_core.TX_VERSION == tx_core.__version__
    for symbol in ("TX", "TXEngine", "TxLocation", "TxRequest", "TxResult",
                   "TxHazardResult"):
        assert hasattr(tx_core, symbol), symbol


def test_real_hazard_registry_is_discoverable():
    engine = TXEngine()
    hazard_ids = {d["id"] for d in engine.hazards()}
    expected = {
        "wildfire", "flood", "drought", "heat", "wind",
        "coastal", "cyclone", "dust", "volcanic", "earthquake",
    }
    assert expected <= hazard_ids


def test_cli_version_and_hazards(capsys):
    assert cli_main(["version"]) == 0
    out = capsys.readouterr().out
    assert '"tx_version"' in out
    assert cli_main(["hazards", "--json"]) == 0
    assert '"wildfire"' in capsys.readouterr().out


def test_analyze_with_fake_hazard_is_honest(monkeypatch):
    engine = TXEngine()
    monkeypatch.setattr(engine, "_resolve", lambda _hid: FakeHazard())
    result = engine.analyze(lat=36.8, lon=34.6, hazards=["wildfire"],
                            depth="quick", name="test point")
    assert result.status == "ok"
    assert result.results[0].hazard == "fake"
    assert result.results[0].status == "ok"
    envelope = result.to_dict()
    assert envelope["engine_version"] == tx_core.TX_VERSION
