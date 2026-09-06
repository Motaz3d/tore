"""
Hazard registry — the single entry point for "which hazards exist?".

A module is registered only when wired to real, documented data sources
(see docs/CLIMATE_HAZARDS.md §2). The registry is populated lazily on first
access so importing :mod:`src.climate` stays light.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from .hazards.base import HazardModule

_lock = threading.Lock()
_modules: Optional[Dict[str, HazardModule]] = None


def _build() -> Dict[str, HazardModule]:
    """Instantiate the registered hazard plugins.

    Registration rule: a hazard is listed here only when it has at least
    one real, documented data source wired in. No placeholders. Modules
    that cannot be imported (e.g. a foundation not yet built in this
    checkout) are skipped — the registry only ever exposes what truly
    exists.
    """

    modules: List[HazardModule] = []

    from .hazards.wildfire import WildfireModule

    modules.append(WildfireModule())

    for import_path, cls_name in (
        (".hazards.flood", "FloodModule"),
        (".hazards.drought", "DroughtModule"),
        (".hazards.heat", "HeatModule"),
        (".hazards.wind", "WindModule"),
        (".hazards.coastal", "CoastalModule"),
        (".hazards.earthquake", "EarthquakeModule"),
        (".hazards.dust", "DustModule"),
        (".hazards.volcanic", "VolcanicModule"),
        (".hazards.cyclone", "CycloneModule"),
    ):
        try:
            mod = __import__(f"src.climate{import_path}", fromlist=[cls_name])
            modules.append(getattr(mod, cls_name)())
        except (ImportError, AttributeError):
            continue  # foundation not built yet — honestly absent

    return {m.id: m for m in modules}


def _ensure() -> Dict[str, HazardModule]:
    global _modules
    if _modules is None:
        with _lock:
            if _modules is None:
                _modules = _build()
    return _modules


def get(hazard_id: str) -> Optional[HazardModule]:
    return _ensure().get(hazard_id)


def ids() -> List[str]:
    return list(_ensure())


def all_modules() -> List[HazardModule]:
    return list(_ensure().values())


def descriptors() -> List[Dict]:
    return [m.descriptor() for m in all_modules()]


def reset_for_tests() -> None:
    """Drop the cached registry (used by tests that monkeypatch modules)."""
    global _modules
    with _lock:
        _modules = None
