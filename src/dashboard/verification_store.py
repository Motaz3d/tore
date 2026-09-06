"""
Persistence for Green Finance Verification portfolio batch checks.

Uses the shared platform SQLite database (same file as the cache and user
accounts), with CREATE TABLE IF NOT EXISTS additive schema changes only.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .cache import default_cache


class VerificationStore:
    """SQLite-backed store for verification portfolio results."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or default_cache().db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10.0)

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verification_portfolios (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    name TEXT,
                    assets_json TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sustainability_reports (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    company_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS insurance_portfolios (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    name TEXT,
                    assets_json TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS supplychain_claims (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    supplier TEXT,
                    commodity TEXT,
                    country TEXT,
                    claim_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forensic_cases (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    case_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS academy_progress (
                    user_id INTEGER NOT NULL,
                    course_id TEXT NOT NULL,
                    module_id TEXT NOT NULL,
                    best_correct INTEGER NOT NULL DEFAULT 0,
                    best_total INTEGER NOT NULL DEFAULT 0,
                    passed INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, course_id, module_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS academy_certificates (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    course_id TEXT NOT NULL,
                    display_name TEXT,
                    score_correct INTEGER NOT NULL,
                    score_total INTEGER NOT NULL,
                    issued_at TEXT NOT NULL,
                    UNIQUE (user_id, course_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS academy_concept_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    course_id TEXT NOT NULL,
                    concept_id TEXT NOT NULL,
                    correct INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS academy_review_schedule (
                    user_id INTEGER NOT NULL,
                    course_id TEXT NOT NULL,
                    concept_id TEXT NOT NULL,
                    interval_days INTEGER NOT NULL,
                    ease REAL NOT NULL,
                    next_due_ts INTEGER NOT NULL,
                    last_result INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, course_id, concept_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tx_seals (
                    code TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    ref_id TEXT,
                    meta_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save_portfolio(
        self,
        user_id: int,
        name: Optional[str],
        assets: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> str:
        """Persist a portfolio batch. Returns the generated portfolio id."""
        portfolio_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO verification_portfolios"
                " (id, user_id, name, assets_json, results_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    portfolio_id,
                    int(user_id),
                    (name or "")[:200] or None,
                    json.dumps(assets, default=str),
                    json.dumps(results, default=str),
                    created_at,
                ),
            )
        return portfolio_id

    def get_portfolio(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Return the full stored portfolio record, or None if unknown."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, name, assets_json, results_json, created_at"
                " FROM verification_portfolios WHERE id = ?",
                (portfolio_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "portfolio_id": row[0],
            "user_id": row[1],
            "name": row[2],
            "assets": json.loads(row[3] or "[]"),
            "results": json.loads(row[4] or "[]"),
            "created_at": row[5],
        }

    def save_report(
        self,
        user_id: int,
        company: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> str:
        """Persist a sustainability evidence report. Returns the report id."""
        report_id = payload["report_id"]
        created_at = payload.get("generated_at") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO sustainability_reports"
                " (id, user_id, company_json, payload_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    report_id,
                    int(user_id),
                    json.dumps(company, default=str),
                    json.dumps(payload, default=str),
                    created_at,
                ),
            )
        return report_id

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Return the full stored sustainability report, or None if unknown."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, company_json, payload_json, created_at"
                " FROM sustainability_reports WHERE id = ?",
                (report_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "report_id": row[0],
            "user_id": row[1],
            "company": json.loads(row[2] or "{}"),
            "payload": json.loads(row[3] or "{}"),
            "created_at": row[4],
        }

    def save_insurance_portfolio(
        self,
        user_id: int,
        name: Optional[str],
        assets: List[Dict[str, Any]],
        results: Dict[str, Any],
    ) -> str:
        """Persist an insurance portfolio profile. Returns the portfolio id."""
        portfolio_id = results["portfolio_id"]
        created_at = results.get("generated_at") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO insurance_portfolios"
                " (id, user_id, name, assets_json, results_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    portfolio_id,
                    int(user_id),
                    (name or "")[:200] or None,
                    json.dumps(assets, default=str),
                    json.dumps(results, default=str),
                    created_at,
                ),
            )
        return portfolio_id

    def get_insurance_portfolio(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Return the full stored insurance portfolio record, or None if unknown."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, name, assets_json, results_json, created_at"
                " FROM insurance_portfolios WHERE id = ?",
                (portfolio_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "portfolio_id": row[0],
            "user_id": row[1],
            "name": row[2],
            "assets": json.loads(row[3] or "[]"),
            "results": json.loads(row[4] or "{}"),
            "created_at": row[5],
        }

    def save_claim(
        self,
        user_id: int,
        claim: Dict[str, Any],
    ) -> str:
        """Persist a supply-chain claim evaluation. Returns the claim id."""
        claim_id = claim["claim_id"]
        created_at = claim.get("generated_at") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO supplychain_claims"
                " (id, user_id, supplier, commodity, country, claim_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    claim_id,
                    int(user_id),
                    (claim.get("supplier") or "")[:200] or None,
                    (claim.get("commodity") or "")[:200] or None,
                    (claim.get("country") or "")[:200] or None,
                    json.dumps(claim, default=str),
                    created_at,
                ),
            )
        return claim_id

    def get_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Return the full stored supply-chain claim record, or None if unknown."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, supplier, commodity, country, claim_json, created_at"
                " FROM supplychain_claims WHERE id = ?",
                (claim_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "claim_id": row[0],
            "user_id": row[1],
            "supplier": row[2],
            "commodity": row[3],
            "country": row[4],
            "claim": json.loads(row[5] or "{}"),
            "created_at": row[6],
        }

    def save_case(
        self,
        user_id: int,
        case: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> str:
        """Persist a forensic case evaluation. Returns the case id."""
        case_id = payload["case_id"]
        created_at = payload.get("generated_at") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO forensic_cases"
                " (id, user_id, case_json, payload_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    case_id,
                    int(user_id),
                    json.dumps(case, default=str),
                    json.dumps(payload, default=str),
                    created_at,
                ),
            )
        return case_id

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Return the full stored forensic case record, or None if unknown."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, case_json, payload_json, created_at"
                " FROM forensic_cases WHERE id = ?",
                (case_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "case_id": row[0],
            "user_id": row[1],
            "case": json.loads(row[2] or "{}"),
            "payload": json.loads(row[3] or "{}"),
            "created_at": row[4],
        }

    def save_progress(
        self,
        user_id: int,
        course_id: str,
        module_id: str,
        best_correct: int,
        best_total: int,
        passed: bool,
    ) -> None:
        """Persist or update a user's best module score."""
        updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO academy_progress
                (user_id, course_id, module_id, best_correct, best_total, passed, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, course_id, module_id) DO UPDATE SET
                    best_correct = CASE
                        WHEN excluded.best_correct > academy_progress.best_correct
                        THEN excluded.best_correct ELSE academy_progress.best_correct END,
                    best_total = excluded.best_total,
                    passed = CASE
                        WHEN excluded.passed = 1 OR academy_progress.passed = 1
                        THEN 1 ELSE 0 END,
                    updated_at = excluded.updated_at
                """,
                (
                    int(user_id),
                    course_id,
                    module_id,
                    int(best_correct),
                    int(best_total),
                    1 if passed else 0,
                    updated_at,
                ),
            )

    def get_progress(self, user_id: int, course_id: str) -> List[Dict[str, Any]]:
        """Return all progress rows for a user on a course."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT course_id, module_id, best_correct, best_total, passed, updated_at"
                " FROM academy_progress WHERE user_id = ? AND course_id = ?",
                (int(user_id), course_id),
            ).fetchall()
        return [
            {
                "course_id": r[0],
                "module_id": r[1],
                "best_correct": r[2],
                "best_total": r[3],
                "passed": bool(r[4]),
                "updated_at": r[5],
            }
            for r in rows
        ]

    def save_concept_attempts(
        self,
        user_id: int,
        course_id: str,
        attempts: List[Dict[str, Any]],
    ) -> None:
        """Persist per-concept quiz attempts for a user on a course."""
        if not attempts:
            return
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO academy_concept_attempts"
                " (user_id, course_id, concept_id, correct, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        int(user_id),
                        course_id,
                        a["concept_id"],
                        1 if a.get("correct") else 0,
                        created_at,
                    )
                    for a in attempts
                ],
            )

    def get_concept_attempts(self, user_id: int, course_id: str) -> List[Dict[str, Any]]:
        """Return all concept attempts for a user on a course, ordered by id."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT concept_id, correct, created_at"
                " FROM academy_concept_attempts"
                " WHERE user_id = ? AND course_id = ?"
                " ORDER BY id",
                (int(user_id), course_id),
            ).fetchall()
        return [
            {"concept_id": r[0], "correct": bool(r[1]), "created_at": r[2]}
            for r in rows
        ]

    def get_review_schedule(self, user_id: int, course_id: str) -> List[Dict[str, Any]]:
        """Return spaced-retrieval schedule rows for a user on a course."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT concept_id, interval_days, ease, next_due_ts, last_result, updated_at"
                " FROM academy_review_schedule"
                " WHERE user_id = ? AND course_id = ?",
                (int(user_id), course_id),
            ).fetchall()
        return [
            {
                "concept_id": r[0],
                "interval_days": r[1],
                "ease": r[2],
                "next_due_ts": r[3],
                "last_result": r[4],
                "updated_at": r[5],
            }
            for r in rows
        ]

    def upsert_review_schedule(
        self,
        user_id: int,
        course_id: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        """Insert or replace spaced-retrieval schedule rows."""
        if not rows:
            return
        updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO academy_review_schedule
                (user_id, course_id, concept_id, interval_days, ease, next_due_ts, last_result, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(user_id),
                        course_id,
                        r["concept_id"],
                        int(r["interval_days"]),
                        float(r["ease"]),
                        int(r["next_due_ts"]),
                        int(r["last_result"]),
                        updated_at,
                    )
                    for r in rows
                ],
            )

    def save_certificate(
        self,
        user_id: int,
        course_id: str,
        display_name: str,
        score_correct: int,
        score_total: int,
    ) -> str:
        """Issue a certificate idempotently; returns the certificate id."""
        from ..climate.evidence import content_hash

        issued_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        cert_id = content_hash({
            "user_id": int(user_id),
            "course_id": course_id,
            "issued_at": issued_at,
        })[:16]

        with self._lock, self._connect() as conn:
            # Idempotent: if a certificate already exists for this user+course, return it.
            existing = conn.execute(
                "SELECT id FROM academy_certificates WHERE user_id = ? AND course_id = ?",
                (int(user_id), course_id),
            ).fetchone()
            if existing:
                return existing[0]

            conn.execute(
                "INSERT INTO academy_certificates"
                " (id, user_id, course_id, display_name, score_correct, score_total, issued_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cert_id,
                    int(user_id),
                    course_id,
                    (display_name or "")[:200] or None,
                    int(score_correct),
                    int(score_total),
                    issued_at,
                ),
            )
        return cert_id

    def get_certificate(self, certificate_id: str) -> Optional[Dict[str, Any]]:
        """Return a certificate by id, or None if unknown."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, course_id, display_name, score_correct, score_total, issued_at"
                " FROM academy_certificates WHERE id = ?",
                (certificate_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "certificate_id": row[0],
            "user_id": row[1],
            "course_id": row[2],
            "display_name": row[3],
            "score_correct": row[4],
            "score_total": row[5],
            "issued_at": row[6],
        }

    def get_certificate_for(self, user_id: int, course_id: str) -> Optional[Dict[str, Any]]:
        """Return a certificate for a specific user and course, if any."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, course_id, display_name, score_correct, score_total, issued_at"
                " FROM academy_certificates WHERE user_id = ? AND course_id = ?",
                (int(user_id), course_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "certificate_id": row[0],
            "user_id": row[1],
            "course_id": row[2],
            "display_name": row[3],
            "score_correct": row[4],
            "score_total": row[5],
            "issued_at": row[6],
        }

    def record_seal(
        self,
        code: str,
        kind: str,
        ref_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a TX seal in the registry. INSERT OR IGNORE is safe because
        seal codes are collision-resistant and re-issuing the same document
        should be idempotent.
        """
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO tx_seals (code, kind, ref_id, meta_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    code,
                    kind,
                    ref_id,
                    json.dumps(meta or {}, default=str),
                    created_at,
                ),
            )

    def get_seal(self, code: str) -> Optional[Dict[str, Any]]:
        """Return a seal record by code, or None if unknown."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT code, kind, ref_id, meta_json, created_at"
                " FROM tx_seals WHERE code = ?",
                (code,),
            ).fetchone()
        if row is None:
            return None
        return {
            "code": row[0],
            "kind": row[1],
            "ref_id": row[2],
            "meta_json": row[3],
            "created_at": row[4],
        }
