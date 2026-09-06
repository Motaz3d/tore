"""
Source Intelligence — health history for integrated data sources.

Every ``integrated`` entry in config/data_registry.json is wired into a
live pipeline path, so its availability directly bounds what the platform
can answer. This module records periodic health checks of those sources
in the shared platform SQLite DB (``HYDRASHIELD_CACHE_DB``, same store as
the cache / watches / accounts) and derives a current health label per
dataset.

Norms:

- Checks happen ONLY when the checker runs (scripts/check_source_health.py
  on the watch_checker loop). Nothing here fabricates health data: before
  the first run the store is simply empty and ``latest_health`` reports
  that honestly (``health == "unknown"``).
- ``ok`` means the service ANSWERED within the timeout — any HTTP status
  below 500. The exact status is always kept in ``http_status`` and
  ``note``: a 4xx typically means the bare GET lacks parameters the API
  needs (POST-oriented endpoints such as ohsome/Overpass), which still
  proves reachability. ``down`` means a server error (5xx), timeout or
  network failure.
- ``degraded`` means reachable but slow (declared threshold, see
  :data:`DEGRADED_LATENCY_MS`).
- ``status_change`` marks transitions vs the previous record:
  ``new`` (first ever record), ``ok_to_down``, ``down_to_ok``, or NULL
  (no change).

The network probe lives in :func:`_http_get` — tests monkeypatch it, so
this module itself never requires network access.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

from ..dashboard.cache import default_cache
from .evidence import utcnow_iso

#: Platform UA for health probes (identifiable, honest).
USER_AGENT = "Talaix-SourceHealth/1.0 (+https://talaix.com)"

#: Per-request timeout for a probe.
TIMEOUT_SECONDS = 10.0

#: Reachable but slower than this counts as ``degraded`` (declared
#: heuristic — a screening label, not an SLO).
DEGRADED_LATENCY_MS = 5000.0

#: Vocabulary for the derived health label.
VALID_HEALTH = frozenset({"ok", "degraded", "down", "unknown"})


class SourceHealthStore:
    """SQLite-backed store for per-dataset health-check history."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or default_cache().db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10.0)

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    checked_at TEXT NOT NULL,
                    http_status INTEGER,
                    latency_ms REAL,
                    ok INTEGER NOT NULL,
                    status_change TEXT,
                    note TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_source_health_dataset "
                "ON source_health (dataset_id, id)"
            )

    def record(self, dataset_id: str, checked_at: str,
               http_status: Optional[int], latency_ms: Optional[float],
               ok: bool, status_change: Optional[str],
               note: Optional[str]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO source_health (dataset_id, checked_at,"
                " http_status, latency_ms, ok, status_change, note)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (dataset_id, checked_at, http_status, latency_ms,
                 1 if ok else 0, status_change, note),
            )

    def previous(self, dataset_id: str) -> Optional[dict]:
        """The most recent record for a dataset, or None."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT dataset_id, checked_at, http_status, latency_ms, ok,"
                " status_change, note FROM source_health"
                " WHERE dataset_id = ? ORDER BY id DESC LIMIT 1",
                (dataset_id,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def latest_per_dataset(self) -> Dict[str, dict]:
        """Latest record per dataset, keyed by dataset_id."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT h.dataset_id, h.checked_at, h.http_status,"
                " h.latency_ms, h.ok, h.status_change, h.note"
                " FROM source_health h"
                " JOIN (SELECT dataset_id, MAX(id) AS max_id"
                "       FROM source_health GROUP BY dataset_id) latest"
                "   ON latest.max_id = h.id"
            ).fetchall()
        return {r[0]: _row_to_dict(r) for r in rows}

    def recent_changes(self, limit: int = 50,
                       dataset_id: Optional[str] = None) -> List[dict]:
        """Most recent transition records (status_change IS NOT NULL)."""
        sql = (
            "SELECT dataset_id, checked_at, http_status, latency_ms, ok,"
            " status_change, note FROM source_health"
            " WHERE status_change IS NOT NULL"
        )
        params: list = []
        if dataset_id:
            sql += " AND dataset_id = ?"
            params.append(dataset_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]


def _row_to_dict(row) -> dict:
    return {
        "dataset_id": row[0],
        "checked_at": row[1],
        "http_status": row[2],
        "latency_ms": row[3],
        "ok": bool(row[4]),
        "status_change": row[5],
        "note": row[6],
    }


def _http_get(url: str) -> Tuple[Optional[int], Optional[float], bool, Optional[str]]:
    """Probe one URL: GET with the platform UA, bounded by TIMEOUT_SECONDS.

    Returns ``(http_status, latency_ms, ok, note)``. ``http_status`` is
    None when no HTTP response was received (DNS, timeout, TLS, refused).
    ``ok`` is True for any answered status below 500 — a 4xx proves the
    service is reachable even when the bare GET lacks required parameters.
    This is the ONLY network touch in the module; tests monkeypatch it.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            status = resp.status
            resp.read(1024)  # touch the body so broken connections surface
        latency_ms = (time.monotonic() - started) * 1000.0
        return status, latency_ms, status < 500, None
    except urllib.error.HTTPError as exc:
        # The server answered — even an error status is a health signal.
        latency_ms = (time.monotonic() - started) * 1000.0
        return exc.code, latency_ms, exc.code < 500, f"HTTP {exc.code}"
    except Exception as exc:  # URLError, timeout, TLS, connection reset…
        latency_ms = (time.monotonic() - started) * 1000.0
        note = str(getattr(exc, "reason", exc))[:200]
        return None, latency_ms, False, note or exc.__class__.__name__


def _probe_url(entry: dict) -> str:
    """The URL to probe for a registry entry: its API/download URL, else
    the catalogue URL root."""
    return entry.get("api_or_download_url") or entry["url"]


def check_integrated_sources(store: Optional[SourceHealthStore] = None) -> dict:
    """Probe every ``integrated`` data-registry source and record results.

    One failure never aborts the sweep: per-dataset probe errors are
    themselves recorded as ``ok=0`` rows with the error in ``note``.
    Returns a summary dict (counts + transitions), never health data for
    sources that were not actually checked.
    """
    from . import data_registry

    store = store or SourceHealthStore()
    entries = data_registry.by_status("integrated")

    checked_at = utcnow_iso()
    checked = ok_count = 0
    transitions: List[dict] = []
    for entry in entries:
        dataset_id = entry["id"]
        prev = store.previous(dataset_id)
        http_status, latency_ms, ok, note = _http_get(_probe_url(entry))
        if prev is None:
            status_change = "new"
        elif bool(prev["ok"]) != ok:
            status_change = "down_to_ok" if ok else "ok_to_down"
        else:
            status_change = None
        store.record(dataset_id, checked_at, http_status, latency_ms, ok,
                     status_change, note)
        checked += 1
        ok_count += 1 if ok else 0
        if status_change:
            transitions.append({
                "dataset_id": dataset_id,
                "status_change": status_change,
                "ok": ok,
                "http_status": http_status,
                "note": note,
            })

    return {
        "checked_at": checked_at,
        "checked": checked,
        "ok": ok_count,
        "down": checked - ok_count,
        "transitions": transitions,
    }


def _derive_health(record: dict) -> str:
    """Latest record → health label (declared rules):

    - ``ok`` and latency below DEGRADED_LATENCY_MS (or unknown) → ``ok``
    - ``ok`` but slower than the threshold → ``degraded``
    - not ``ok`` → ``down``
    """
    if not record["ok"]:
        return "down"
    latency = record.get("latency_ms")
    if latency is not None and latency >= DEGRADED_LATENCY_MS:
        return "degraded"
    return "ok"


def latest_health(dataset_id: Optional[str] = None,
                  store: Optional[SourceHealthStore] = None) -> dict:
    """Latest health per dataset + derived label + transition history.

    Before the first checker run this is honestly empty:
    ``{"datasets": {}, "changes": []}`` (per-dataset lookups report
    ``health: "unknown"``). No record is ever synthesised.
    """
    store = store or SourceHealthStore()

    if dataset_id:
        record = store.previous(dataset_id)
        datasets = {}
        if record is not None:
            record = dict(record)
            record["health"] = _derive_health(record)
            datasets[dataset_id] = record
        changes = store.recent_changes(dataset_id=dataset_id)
    else:
        datasets = {}
        for ds_id, record in store.latest_per_dataset().items():
            record = dict(record)
            record["health"] = _derive_health(record)
            datasets[ds_id] = record
        changes = store.recent_changes()

    return {"datasets": datasets, "changes": changes}
