"""
SQLite-backed TTL cache for Talaix external data.

External data sources (geocoding, weather, terrain, satellite, FIRMS) are
rate-limited and comparatively slow. This cache gives the platform:

    - Bounded upstream call rates (e.g. Nominatim's 1 req/s policy).
    - Fast repeat analyses for the same area.
    - A single, inspectable on-disk store (no extra infrastructure).

The cache stores JSON-serialisable values keyed by a namespace plus a hash of
the call arguments. Entries expire after a per-namespace TTL. It is deliberately
simple: one SQLite file, stdlib only, safe for the multi-process gunicorn
deployment (SQLite serialises writers).
"""

from __future__ import annotations

import hashlib
import functools
import json
import os
import sqlite3
import threading
import time
from typing import Any, Callable, Optional

_DEFAULT_DB = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cache", "hydrashield_cache.sqlite3"
)

# Default time-to-live per data category (seconds).
TTL_GEOCODE = 30 * 24 * 3600        # locations move rarely
TTL_TERRAIN = 30 * 24 * 3600        # terrain is static
TTL_WEATHER_CURRENT = 30 * 60       # current conditions
TTL_WEATHER_DAILY = 6 * 3600        # daily series / reanalysis
TTL_SATELLITE = 12 * 3600           # Sentinel-2 scene selection + indices
TTL_FIRES = 3 * 3600                # FIRMS active fires
TTL_ANALYSIS = 15 * 60              # full analysis result
TTL_LANDCOVER = 30 * 24 * 3600      # land cover is near-static
TTL_SNAPSHOT = 30 * 60              # public risk snapshot aggregate
TTL_CLIMATE_SERIES = 30 * 24 * 3600 # historical climate series changes slowly


class TTLCache:
    """A minimal SQLite TTL cache for JSON-serialisable values."""

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
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def make_key(namespace: str, *args: Any, **kwargs: Any) -> str:
        """Build a deterministic cache key from a namespace and arguments."""
        payload = json.dumps([args, kwargs], sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"{namespace}:{digest}"

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value, or None if missing/expired."""
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        value, expires_at = row
        if expires_at < now:
            self.delete(key)
            return None
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Store a JSON-serialisable value with a TTL."""
        now = time.time()
        blob = json.dumps(value, default=str)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at, created_at)"
                " VALUES (?, ?, ?, ?)",
                (key, blob, now + ttl_seconds, now),
            )

    def delete(self, key: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def purge_expired(self) -> int:
        """Remove expired entries; returns the number removed."""
        now = time.time()
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM cache WHERE expires_at < ?", (now,))
            return cur.rowcount

    def stats(self) -> dict:
        """Return basic cache statistics (used by the health endpoint)."""
        now = time.time()
        with self._lock, self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            live = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE expires_at >= ?", (now,)
            ).fetchone()[0]
        return {"entries_total": total, "entries_live": live, "db_path": self.db_path}


# Shared process-wide instance.
_default_cache: Optional[TTLCache] = None
_default_lock = threading.Lock()


def default_cache() -> TTLCache:
    """Return the shared cache instance (created lazily)."""
    global _default_cache
    if _default_cache is None:
        with _default_lock:
            if _default_cache is None:
                _default_cache = TTLCache()
    return _default_cache


def cached(namespace: str, ttl_seconds: float) -> Callable:
    """
    Decorator: cache the JSON-serialisable dict return value of a function.

    Failed lookups (dicts containing an ``"error"`` key) are cached for a much
    shorter time (60 s) so a transient upstream outage does not get pinned.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = default_cache()
            key = cache.make_key(namespace, *args, **kwargs)
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = func(*args, **kwargs)
            if isinstance(result, dict) and "error" in result:
                cache.set(key, result, 60.0)
            else:
                cache.set(key, result, ttl_seconds)
            return result

        return wrapper

    return decorator
