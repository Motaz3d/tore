"""
Historical climate event model — the platform's unit of historical
intelligence (docs/CLIMATE_HAZARDS.md §1, docs/EVIDENCE_ARCHITECTURE.md §4).

A :class:`ClimateEvent` is a concrete occurrence of a hazard in space and
time, carrying:

- **observed conditions** (measured: weather, detections, discharge, …) —
  structurally separated from
- **modelled context** (FWI reconstruction, lessons, "signals our
  indicators would have shown") — never allowed to rewrite observations,
- **evidence records** across the five evidence classes,
- **cause** — DOCUMENTED only when an authoritative source establishes it,
  otherwise UNKNOWN (enforced by :func:`validate_cause`),
- **impacts** — only when documented/reported by a source,
- **lessons** — extracted strictly from the event's own data, each labelled
  with its basis (OBSERVED / MODELLED).

The :class:`EventStore` persists derived events in the platform's SQLite
database (tables ``climate_events`` + ``event_evidence``) so reports and
the API can reference stable event IDs. Years are never hardcoded: queries
filter on whatever the underlying datasets actually contain.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .ontology import ClaimStatus, HazardType, validate_cause
from .evidence import evidence_id

# Same database file convention as the platform cache (src/dashboard/cache.py).
_DEFAULT_DB = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cache", "hydrashield_cache.sqlite3"
)


# ---------------------------------------------------------------------------
# ClimateEvent
# ---------------------------------------------------------------------------


@dataclass
class ClimateEvent:
    """One historical event with separated observation / modelled context."""

    hazard: str                              # HazardType value
    lat: float
    lon: float
    start_date: str                          # ISO date
    end_date: Optional[str] = None           # ISO date; None => single day
    name: Optional[str] = None
    classification: str = ClaimStatus.OBSERVED.value
    severity: Optional[Dict[str, Any]] = None        # per-hazard, source-bound
    conditions_observed: Dict[str, Any] = field(default_factory=dict)
    context_modelled: Dict[str, Any] = field(default_factory=dict)
    exposure: Optional[Dict[str, Any]] = None
    cause: Dict[str, Any] = field(
        default_factory=lambda: {"status": ClaimStatus.UNKNOWN.value, "value": None, "source": None}
    )
    response: List[Dict[str, Any]] = field(default_factory=list)
    impacts: List[Dict[str, Any]] = field(default_factory=list)
    lessons: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        HazardType(self.hazard)  # only registered hazards can have events
        ClaimStatus(self.classification)
        # Cause discipline: DOCUMENTED or UNKNOWN — nothing else survives.
        self.cause = dict(self.cause or {})
        self.cause["status"] = validate_cause(str(self.cause.get("status", "UNKNOWN")))
        if self.cause["status"] == ClaimStatus.UNKNOWN.value:
            self.cause.setdefault("value", None)
            self.cause.setdefault("source", None)

    @property
    def duration_days(self) -> int:
        if not self.end_date or self.end_date == self.start_date:
            return 1
        try:
            from datetime import date

            y0, m0, d0 = (int(x) for x in self.start_date[:10].split("-"))
            y1, m1, d1 = (int(x) for x in self.end_date[:10].split("-"))
            return (date(y1, m1, d1) - date(y0, m0, d0)).days + 1
        except (ValueError, TypeError):
            return 1

    @property
    def year(self) -> int:
        return int(self.start_date[:4])

    @property
    def event_id(self) -> str:
        """Stable id from content (location, dates, hazard, classification)."""

        return "ev_" + evidence_id(
            {
                "hazard": self.hazard,
                "lat": round(self.lat, 4),
                "lon": round(self.lon, 4),
                "start": self.start_date,
                "end": self.end_date,
                "classification": self.classification,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_id"] = self.event_id
        d["duration_days"] = self.duration_days
        d["year"] = self.year
        return d


# ---------------------------------------------------------------------------
# EventStore — SQLite persistence
# ---------------------------------------------------------------------------


class EventStore:
    """Persists derived events + their evidence in the shared SQLite DB.

    Additive, idempotent schema (``CREATE TABLE IF NOT EXISTS``) — safe for
    the multi-process deployment. Observed payloads are stored verbatim as
    JSON; nothing rewrites them after insertion (upserts replace the whole
    record only when the same event is re-derived from the same inputs).
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or os.environ.get("HYDRASHIELD_CACHE_DB", _DEFAULT_DB)
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS climate_events (
                    event_id TEXT PRIMARY KEY,
                    hazard TEXT NOT NULL,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT,
                    year INTEGER NOT NULL,
                    classification TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_climate_events_hazard_year
                ON climate_events (hazard, year)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_evidence (
                    event_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (event_id, evidence_id)
                )
                """
            )

    # -- write ----------------------------------------------------------

    def upsert_event(self, event: ClimateEvent) -> str:
        payload = event.to_dict()
        eid = payload["event_id"]
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO climate_events
                    (event_id, hazard, lat, lon, start_date, end_date, year,
                     classification, payload, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(event_id) DO UPDATE SET
                    payload=excluded.payload,
                    classification=excluded.classification
                """,
                (
                    eid,
                    event.hazard,
                    event.lat,
                    event.lon,
                    event.start_date,
                    event.end_date,
                    event.year,
                    event.classification,
                    json.dumps(payload, default=str),
                    time.time(),
                ),
            )
            for ev in event.evidence:
                ev_id = ev.get("evidence_id") or evidence_id(ev)
                conn.execute(
                    """
                    INSERT INTO event_evidence (event_id, evidence_id, payload)
                    VALUES (?,?,?)
                    ON CONFLICT(event_id, evidence_id) DO UPDATE SET payload=excluded.payload
                    """,
                    (eid, ev_id, json.dumps(ev, default=str)),
                )
        return eid

    # -- read -----------------------------------------------------------

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM climate_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def query(
        self,
        hazard: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_km: float = 50.0,
        year: Optional[int] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Events filtered by hazard/year and (optionally) distance to a point.

        Distance filtering happens in Python (haversine) — event counts per
        location are small and this keeps the schema index-simple.
        """

        sql = "SELECT payload, lat, lon FROM climate_events"
        clauses, params = [], []
        if hazard:
            clauses.append("hazard = ?")
            params.append(hazard)
        if year is not None:
            clauses.append("year = ?")
            params.append(int(year))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY start_date DESC LIMIT ?"
        params.append(int(limit) * 4)  # over-fetch before distance filter
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        out: List[Dict[str, Any]] = []
        for payload, elat, elon in rows:
            if lat is not None and lon is not None:
                if _haversine_km(lat, lon, float(elat), float(elon)) > radius_km:
                    continue
            out.append(json.loads(payload))
            if len(out) >= limit:
                break
        return out

    def years_available(self, hazard: str) -> List[int]:
        """Years actually present in the store for a hazard (UI year selector)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT year FROM climate_events WHERE hazard = ? ORDER BY year DESC",
                (hazard,),
            ).fetchall()
        return [int(r[0]) for r in rows]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
