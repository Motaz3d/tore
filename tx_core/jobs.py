"""
Standard TX Job Object — asynchronous execution envelope for TX analyses.

Deep analyses can take minutes (the real pipelines download and process
large upstream datasets), so the web contract is job-based:

    POST /api/tx/run                  -> 202 {job_id, status: "queued", ...}
    GET  /api/tx/jobs/<job_id>        -> status + progress (poll this)
    GET  /api/tx/jobs/<job_id>/result -> the TxResult envelope when done

The job object *shape* is the stable contract; the backend behind it is
deliberately replaceable. Phase 1 ships an in-process store (thread-safe
dict with TTL eviction) and a bounded thread-pool runner — honest for a
single-process deployment. A Redis/queue backend can later implement the
same two small interfaces (``TxJobStore`` / ``TxJobRunner``) without any
change to the routes or the clients.

Reproducibility carries over from the engine: ``job_id`` is deterministic
per (request, tx_version, UTC day), so re-submitting the same analysis on
the same day is idempotent — the existing job is returned instead of
re-running the pipeline.

Honesty contract (unchanged): a failed job reports ``status="failed"``
with the real error message; nothing is fabricated, ever.
"""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._version import TX_VERSION
from .adapters.climate import utcnow_iso  # single shared clock

#: Job lifecycle states (the full, closed set).
JOB_STATES = ("queued", "running", "succeeded", "failed")

#: Defaults for the phase-1 in-process store.
DEFAULT_TTL_SECONDS = 3600
DEFAULT_MAX_JOBS = 1000


def make_job_id(
    *,
    lat: float,
    lon: float,
    hazards: Optional[List[str]] = None,
    depth: str = "standard",
    analyses: Optional[List[str]] = None,
) -> str:
    """Deterministic, day-scoped job id: ``TXJ-YYYYMMDD-<hex8>``.

    Same request on the same UTC day → same id (idempotent submission).
    ``hazards=None`` (all available) is distinct from any explicit list.
    The ``analyses`` key enters the basis ONLY when non-empty, so existing
    hazard-only job ids stay byte-stable.
    """
    basis = {
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "hazards": sorted(h.strip().lower() for h in hazards) if hazards else None,
        "depth": (depth or "standard").lower(),
        "tx_version": TX_VERSION,
    }
    if analyses:
        basis["analyses"] = sorted(a.strip().lower() for a in analyses)
    digest = hashlib.sha256(
        repr(sorted(basis.items())).encode("utf-8")
    ).hexdigest()[:8]
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"TXJ-{day}-{digest}"


@dataclass
class TxJob:
    """One TX analysis job (the standard Job Object).

    ``request`` holds the validated analysis inputs; ``result`` is the full
    :class:`~tx_core.models.TxResult` dict on success; ``error`` is the
    honest failure message otherwise. ``created_epoch`` is bookkeeping for
    TTL eviction and is never serialized.
    """

    job_id: str
    request: Dict[str, Any]
    status: str = "queued"                 # queued | running | succeeded | failed
    created_at: str = field(default_factory=utcnow_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress: Dict[str, int] = field(
        default_factory=lambda: {"completed": 0, "total": 0}
    )
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tx_version: str = TX_VERSION
    created_epoch: float = field(default_factory=time.time, repr=False)

    def to_status_dict(self) -> Dict[str, Any]:
        """The polling payload: everything except the (heavy) result body."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "request": dict(self.request),
            "progress": dict(self.progress),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "tx_version": self.tx_version,
        }


class TxJobStore:
    """Phase-1 in-process job store: thread-safe, TTL-evicted, bounded.

    The interface is the contract — a future Redis backend only needs
    ``put_if_absent`` / ``get`` / ``update`` with the same semantics.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_jobs: int = DEFAULT_MAX_JOBS,
    ) -> None:
        self._ttl = float(ttl_seconds)
        self._max = int(max_jobs)
        self._jobs: Dict[str, TxJob] = {}
        self._lock = threading.Lock()

    def put_if_absent(self, job: TxJob) -> Tuple[TxJob, bool]:
        """Atomically insert ``job`` unless its id already exists.

        Returns ``(job, created)`` — the winning job object and whether this
        call created it. Idempotent submission races resolve to one job.
        """
        with self._lock:
            self._evict_locked()
            existing = self._jobs.get(job.job_id)
            if existing is not None:
                return existing, False
            self._jobs[job.job_id] = job
            return job, True

    def get(self, job_id: str) -> Optional[TxJob]:
        with self._lock:
            self._evict_locked()
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> Optional[TxJob]:
        """Apply field updates to a job; returns the job (or None if gone)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in fields.items():
                if not hasattr(job, key):
                    raise AttributeError(f"TxJob has no field {key!r}")
                setattr(job, key, value)
            return job

    def _evict_locked(self) -> None:
        """Drop expired jobs, then oldest-first if still over capacity."""
        now = time.time()
        expired = [
            jid for jid, j in self._jobs.items() if now - j.created_epoch > self._ttl
        ]
        for jid in expired:
            del self._jobs[jid]
        if len(self._jobs) > self._max:
            by_age = sorted(self._jobs.values(), key=lambda j: j.created_epoch)
            for job in by_age[: len(self._jobs) - self._max]:
                del self._jobs[job.job_id]


class TxJobRunner:
    """Execute TX analyses as jobs behind the standard Job Object.

    :param store: the job store (defaults to a phase-1 in-process store).
    :param engine_factory: zero-arg callable returning a
        :class:`~tx_core.engine.TXEngine`. Resolved at *execution* time, so
        tests can inject a network-free engine per job.
    :param max_workers: bound on concurrently running jobs (thread pool).
    :param synchronous: when True, ``submit`` executes the job inline before
        returning (tests and single-process debugging).
    """

    def __init__(
        self,
        store: Optional[TxJobStore] = None,
        engine_factory: Optional[Callable[[], Any]] = None,
        max_workers: int = 2,
        synchronous: bool = False,
    ) -> None:
        self.store = store or TxJobStore()
        self._engine_factory = engine_factory
        self.synchronous = synchronous
        self._pool: Optional[ThreadPoolExecutor] = (
            None
            if synchronous
            else ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="tx-job"
            )
        )

    def submit(
        self,
        *,
        lat: float,
        lon: float,
        hazards: Optional[List[str]] = None,
        depth: str = "standard",
        name: Optional[str] = None,
        analyses: Optional[List[str]] = None,
    ) -> Tuple[TxJob, bool]:
        """Submit an analysis request; returns ``(job, created)``.

        Idempotent: an identical request (same UTC day) returns the existing
        job with ``created=False`` instead of re-running the pipeline.
        """
        job_id = make_job_id(lat=lat, lon=lon, hazards=hazards, depth=depth,
                             analyses=analyses)
        request = {
            "lat": float(lat),
            "lon": float(lon),
            "name": name,
            "hazards": list(hazards) if hazards else [],
            "analyses": list(analyses) if analyses else [],
            "depth": (depth or "standard").lower(),
        }
        job, created = self.store.put_if_absent(TxJob(job_id=job_id, request=request))
        if not created:
            return job, False
        if self.synchronous:
            self._execute(job.job_id)
        else:
            assert self._pool is not None
            self._pool.submit(self._execute, job.job_id)
        return job, True

    def get(self, job_id: str) -> Optional[TxJob]:
        return self.store.get(job_id)

    # -- execution ----------------------------------------------------------

    def _execute(self, job_id: str) -> None:
        job = self.store.update(job_id, status="running", started_at=utcnow_iso())
        if job is None:  # evicted before it ever ran — nothing to do
            return
        req = job.request
        try:
            engine = self._make_engine()

            def _progress(_result: Any, done: int, total: int) -> None:
                self.store.update(
                    job_id, progress={"completed": done, "total": total}
                )

            result = engine.analyze(
                lat=req["lat"],
                lon=req["lon"],
                hazards=req["hazards"] or None,
                depth=req["depth"],
                name=req.get("name"),
                on_hazard=_progress,
                analyses=req.get("analyses") or None,
            )
            self.store.update(
                job_id,
                status="succeeded",
                finished_at=utcnow_iso(),
                result=result.to_dict(),
            )
        except Exception as exc:  # noqa: BLE001 — honest failure, never fabricated
            self.store.update(
                job_id, status="failed", finished_at=utcnow_iso(), error=str(exc)
            )

    def _make_engine(self) -> Any:
        if self._engine_factory is not None:
            return self._engine_factory()
        from .engine import TXEngine  # lazy: keep tx_core import-light

        return TXEngine()
