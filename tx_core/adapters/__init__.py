"""
TX Core adapters — the only place tx_core touches the existing platform.

Each adapter wraps a ``src.*`` package behind a narrow, stable interface.
Imports are lazy (inside functions) so importing ``tx_core`` never pulls in
heavy or network-bound dependencies. If a platform package is absent in a
given checkout, the adapter reports it honestly (``None`` / empty) instead
of crashing the engine.
"""

from __future__ import annotations

from . import climate, gis, legacy_v1, prediction  # noqa: F401

__all__ = ["climate", "gis", "legacy_v1", "prediction"]
