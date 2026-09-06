"""
TX Core — the standalone analytical authority behind every Talaix result.

``tx_core`` is the web-free engine package (see docs/TX_ENGINE.md). It
orchestrates the *existing* platform analytical modules (``src.climate``,
``src.prediction``, ``src.gis_mapping``) behind one stable TX contract, so
that the website, the API, the SDK, the CLI and the QGIS plugin all consume
exactly the same analysis.

Strangler-pattern guarantee: ``tx_core`` never re-implements analysis. It
delegates to the wired platform modules through narrow adapters
(:mod:`tx_core.adapters`), so existing behaviour is preserved and the live
site keeps running untouched while the engine is being built out.

Design rules (mirrored from the platform honesty contract):

- No fabricated numbers: a hazard that cannot produce real data is reported
  as ``status="unavailable"`` with an explicit reason.
- Every result carries provenance, evidence and engine versions so any TX
  analysis can be reproduced and audited.
- Import-light: importing ``tx_core`` pulls no heavy dependencies; all
  analytical modules are imported lazily inside the adapters.
"""

from __future__ import annotations

from ._version import TAM_VERSION, TX_VERSION, __version__
from .engine import TX, TXEngine
from .jobs import TxJob, TxJobRunner, TxJobStore, make_job_id
from .models import TxHazardResult, TxLocation, TxRequest, TxResult

__all__ = [
    "TX",
    "TXEngine",
    "TxHazardResult",
    "TxJob",
    "TxJobRunner",
    "TxJobStore",
    "TxLocation",
    "TxRequest",
    "TxResult",
    "__version__",
    "TX_VERSION",
    "TAM_VERSION",
    "make_job_id",
]
