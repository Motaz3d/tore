"""
TX Registry — the single index of hazards, data sources, datasets and models
behind every TX result.

Facade over the platform's declared registries (``config/*.json``: model,
source, data) plus the live hazard registry. Nothing is invented here: the
registry returns exactly what the platform has declared and wired.

Reproducibility use: every TX analysis can state which model version,
dataset and source version it rests on — the registry is the lookup table
for that audit trail (docs/TX_ENGINE.md §"Reproducibility").
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
)


class TXRegistry:
    """Read-only facade over the platform's JSON registries.

    :param config_dir: directory holding ``model_registry.json``,
        ``source_registry.json`` and ``data_registry.json``. Defaults to the
        platform ``config/`` directory next to this package.
    """

    def __init__(self, config_dir: Optional[str] = None) -> None:
        self.config_dir = config_dir or _CONFIG_DIR

    # -- file-backed sections ----------------------------------------------

    def _load(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.config_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def models(self) -> List[Dict[str, Any]]:
        """Declared models (version, lifecycle, methodology, …)."""
        return list(self._load("model_registry.json").get("models", []))

    def sources(self) -> List[Dict[str, Any]]:
        """Audited data-source records (integrated/candidate/rejected)."""
        return list(self._load("source_registry.json").get("sources", []))

    def datasets(self) -> List[Dict[str, Any]]:
        """Catalogued dataset records (integrated/candidate/rejected)."""
        return list(self._load("data_registry.json").get("datasets", []))

    def lifecycle_states(self) -> List[Dict[str, Any]]:
        return list(self._load("model_registry.json").get("lifecycle_states", []))

    # -- query helpers -------------------------------------------------------

    def model(self, model_id: str) -> Optional[Dict[str, Any]]:
        for m in self.models():
            if m.get("id") == model_id:
                return m
        return None

    def dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        for d in self.datasets():
            if d.get("id") == dataset_id:
                return d
        return None

    def integrated_sources(self) -> List[Dict[str, Any]]:
        return [s for s in self.sources() if s.get("status") == "integrated"]

    def summary(self) -> Dict[str, Any]:
        """A compact digest for TX consumers (CLI, API, SDK)."""
        return {
            "hazards": self.hazard_ids(),
            "models": [m.get("id") for m in self.models()],
            "datasets_integrated": sum(
                1 for d in self.datasets() if d.get("status") == "integrated"
            ),
            "sources_integrated": len(self.integrated_sources()),
            "audit_dates": {
                "sources": self._load("source_registry.json").get("audit_date"),
                "data": self._load("data_registry.json").get("audit_date"),
            },
        }

    # -- live sections (delegated to the engine/adapters) --------------------

    def hazard_ids(self) -> List[str]:
        from .adapters import climate

        return sorted(climate.hazard_ids())

    def hazard_descriptors(self) -> List[Dict[str, Any]]:
        from .adapters import climate

        return climate.hazard_descriptors()
