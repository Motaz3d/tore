"""
Adapter over the legacy v1 wildfire pipeline (``src.dashboard.*``).

The GET /api/analyze endpoint has always been powered by
``TalaixRealAnalyser.analyse_point`` through its shared 15-minute cache
(``src.dashboard.snapshot.cached_analysis``). TX never re-implements that
pipeline — this adapter is the *only* place tx_core touches it. Imports are
lazy (inside functions) so importing ``tx_core`` stays light, exactly like
the sibling adapters.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def cached_analysis(lat: float, lon: float, name: str) -> Dict[str, Any]:
    """The shared cached real-analysis pipeline behind /api/analyze.

    Delegates unchanged to ``src.dashboard.snapshot.cached_analysis``
    (15-min TTL), so the v1 endpoint keeps its exact caching behaviour.
    """
    from src.dashboard.snapshot import cached_analysis as _cached_analysis

    return _cached_analysis(float(lat), float(lon), str(name))


def analyser_class() -> Optional[type]:
    """The legacy real-data analyser class (lazy, never instantiated here).

    Returns ``None`` when the platform module is unavailable in a given
    checkout, so the facade can report honestly instead of crashing.
    """
    try:
        from src.dashboard.real_analysis import TalaixRealAnalyser

        return TalaixRealAnalyser
    except Exception:  # noqa: BLE001 — absent platform package is honest None
        return None


def analyser_factory() -> Callable[..., Any]:
    """A factory that returns a fresh analyser instance on call."""
    cls = analyser_class()

    def _build(*args: Any, **kwargs: Any) -> Any:
        if cls is None:
            raise RuntimeError(
                "Legacy v1 analyser (TalaixRealAnalyser) unavailable in this checkout."
            )
        return cls(*args, **kwargs)

    return _build
