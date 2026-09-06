"""TX authenticity seal.

Every report and digital product issued by the Talaix TX engine carries a
branded ``TX-XXXX-XXXX-XXXX`` code.  The code is an HMAC-SHA256 digest of a
canonical JSON payload, truncated to 16 hex characters and formatted with
dashes.  It can be verified either against the platform registry (for
stored documents) or statelessly by recomputing the digest (for JSON
analysis results).
"""

from __future__ import annotations

import hmac
import hashlib
import json
import logging
import os
import platform
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SEAL_RE = re.compile(r"^TX-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}$")


def _canonical_json(payload: Any) -> str:
    """Stable JSON representation used for every content hash and seal."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _server_key() -> bytes:
    """Server-side HMAC key for TX seals.

    Mirrors the key derivation in ``src.dashboard.accounts._server_key``:
    prefer the ``HYDRASHIELD_SECRET_KEY`` environment variable; in dev mode
    fall back to a host/home-bound SHA-256 digest.  The dev fallback lets
    local development work without a configured secret, but codes generated
    on different machines will not verify against each other unless the
    environment key is set.
    """
    key = os.environ.get("HYDRASHIELD_SECRET_KEY")
    if key:
        return key.encode("utf-8")
    basis = (
        f"hydrashield-dev-token-key|{platform.node()}|{os.path.expanduser('~')}"
    )
    return hashlib.sha256(basis.encode("utf-8")).digest()


def _format_code(digest_hex: str) -> str:
    """Format a 12-char hex digest as TX-XXXX-XXXX-XXXX."""
    parts = [digest_hex[i : i + 4] for i in range(0, 12, 4)]
    return "TX-" + "-".join(parts).upper()


def seal_code(payload: Any) -> str:
    """Compute the TX seal code for ``payload``.

    The payload is canonicalised to JSON and prefixed with a domain string
    so the seal cannot be confused with other HMAC usages in the platform.
    """
    msg = "talaix-tx-seal|" + _canonical_json(payload)
    digest = hmac.new(_server_key(), msg.encode("utf-8"), hashlib.sha256).hexdigest()
    return _format_code(digest[:12])


def is_seal_format(value: Any) -> bool:
    """Return True if ``value`` looks like a TX seal code."""
    if not isinstance(value, str):
        return False
    return bool(SEAL_RE.match(value.strip().upper()))


def normalize_code(value: Any) -> Optional[str]:
    """Return normalised uppercase code, or None if malformed."""
    if not isinstance(value, str):
        return None
    candidate = value.strip().upper()
    if SEAL_RE.match(candidate):
        return candidate
    return None


def _now_iso() -> str:
    """UTC ISO-8601 timestamp with trailing Z."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def issue_seal(
    kind: str,
    ref_id: str,
    payload: Any,
    store: Optional[Any] = None,
) -> Dict[str, str]:
    """Issue a TX seal for a document product.

    Computes the code over ``{"kind": kind, "id": ref_id, "payload": payload}``
    and records it in the registry.  Registry failures are logged but never
    break product generation: the code is still returned and verification
    degrades to the stateless recomputation mode.
    """
    code = seal_code({"kind": kind, "id": ref_id, "payload": payload})
    if store is None:
        from ..dashboard.verification_store import VerificationStore

        try:
            store = VerificationStore()
        except Exception as exc:  # noqa: BLE001
            logger.warning("tx_seal: could not instantiate VerificationStore: %s", exc)
            store = None

    if store is not None:
        try:
            store.record_seal(code, kind=kind, ref_id=ref_id, meta={"kind": kind})
        except Exception as exc:  # noqa: BLE001
            logger.warning("tx_seal: registry write failed for %s/%s: %s", kind, ref_id, exc)

    return {
        "code": code,
        "kind": kind,
        "engine": "TX",
        "verify_url": f"/verify.html#{code}",
    }


def verify_seal(code: str, store: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """Registry lookup for a seal code.

    Returns a validity dict if the code is recorded, otherwise None.
    """
    normalized = normalize_code(code)
    if normalized is None:
        return None

    if store is None:
        from ..dashboard.verification_store import VerificationStore

        try:
            store = VerificationStore()
        except Exception as exc:  # noqa: BLE001
            logger.warning("tx_seal: could not instantiate VerificationStore: %s", exc)
            return None

    try:
        row = store.get_seal(normalized)
    except Exception as exc:  # noqa: BLE001
        logger.warning("tx_seal: registry read failed for %s: %s", normalized, exc)
        return None

    if row is None:
        return None

    return {
        "valid": True,
        "code": normalized,
        "kind": row.get("kind", ""),
        "ref_id": row.get("ref_id", ""),
        "issued_at": row.get("created_at", ""),
        "engine": "TX",
    }


def check_seal(payload: Any, code: str) -> bool:
    """Stateless verification: recompute the seal and compare securely."""
    normalized = normalize_code(code)
    if normalized is None:
        return False
    expected = seal_code(payload)
    return hmac.compare_digest(expected.encode("utf-8"), normalized.encode("utf-8"))
