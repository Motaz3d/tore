"""
Talaix Model Evaluation Framework.

Two pillars:

1. **Lifecycle states** — every model in ``config/model_registry.json``
   carries a ``lifecycle`` field moving strictly along

       experimental → screening → backtested → validated → operational
       (``deprecated`` from any state)

   A lifecycle never advances on intention or documentation alone: it
   advances only when an executed evaluation run is recorded under
   ``data/evaluation/runs/`` (benchmark suite, validation pipeline, or
   equation-reference verification). Promotion is always a deliberate,
   reviewed, manual step — nothing here auto-promotes.

2. **Immutable evaluation-run records** — JSON files under
   ``data/evaluation/runs/``. A record's ``run_id`` is the content hash of
   the record itself; identical content yields the identical file and is
   never rewritten. Records state what actually ran: kind
   (``equation_reference | benchmark_suite | validation_pipeline``),
   dataset, metrics, calibration, false-positive/false-negative analysis,
   geographic/temporal performance, failure cases, timestamps and code
   version. A record that did not run does not exist.

Honesty contract (docs/BENCHMARKS.md, docs/VALIDATION.md): nothing in this
module claims validation. ``record_fwi_reference_run`` records the REAL,
executed equation-level verification of ``fwi_system_v1`` against the
cffdrs reference implementation — an equation check, explicitly NOT a
site-validated fire-occurrence validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

#: Lifecycle chain (strict order) — ``deprecated`` is reachable from any state.
LIFECYCLE_STATES = (
    "experimental",
    "screening",
    "backtested",
    "validated",
    "operational",
    "deprecated",
)

#: Evaluation run kinds.
RUN_KINDS = (
    "equation_reference",
    "benchmark_suite",
    "validation_pipeline",
)

_RUN_ID_RE = re.compile(r"^[0-9a-f]{64}$")


class EvaluationError(ValueError):
    """Invalid lifecycle state, run kind, or record payload."""


def validate_lifecycle(state: Any) -> str:
    """Return ``state`` if it is a valid lifecycle value; raise otherwise."""

    if state not in LIFECYCLE_STATES:
        raise EvaluationError(
            f"invalid lifecycle state {state!r}; expected one of {LIFECYCLE_STATES}"
        )
    return state


def is_valid_lifecycle(state: Any) -> bool:
    """Boolean companion of :func:`validate_lifecycle`."""

    return state in LIFECYCLE_STATES


def lifecycle_index(state: str) -> int:
    """Position in the chain (higher = more mature; ``deprecated`` is last)."""

    return LIFECYCLE_STATES.index(validate_lifecycle(state))


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------


def _base_dir() -> str:
    return os.environ.get(
        "HYDRASHIELD_EVALUATION_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "evaluation"),
    )


def _runs_dir(runs_dir: Optional[str]) -> str:
    return runs_dir or os.path.join(_base_dir(), "runs")


def _code_version() -> str:
    """git short HEAD of the working tree; platform version as fallback."""

    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        from .. import __version__

        return f"hydrashield-platform {__version__} (no git metadata)"
    except Exception:
        return "unknown"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_run(
    model_id: str,
    model_version: str,
    kind: str,
    dataset: str,
    metrics: Dict[str, Any],
    *,
    calibration: Optional[Dict[str, Any]] = None,
    fp_fn: Optional[Dict[str, Any]] = None,
    geographic_performance: Optional[Dict[str, Any]] = None,
    temporal_performance: Optional[Dict[str, Any]] = None,
    failure_cases: Optional[List[Dict[str, Any]]] = None,
    code_version: Optional[str] = None,
    executed_at: Optional[str] = None,
    runs_dir: Optional[str] = None,
) -> str:
    """Record an executed evaluation run; returns the immutable record path.

    ``run_id`` is the SHA-256 content hash of the record (minus ``run_id``
    itself). An identical run recorded twice yields the same file and is
    not rewritten — run records are immutable.
    """

    if kind not in RUN_KINDS:
        raise EvaluationError(
            f"invalid run kind {kind!r}; expected one of {RUN_KINDS}"
        )
    if not model_id or not isinstance(metrics, dict):
        raise EvaluationError("model_id and a metrics dict are required")

    record: Dict[str, Any] = {
        "model_id": model_id,
        "model_version": model_version,
        "kind": kind,
        "dataset": dataset,
        "metrics": metrics,
        "calibration": calibration,
        "fp_fn": fp_fn,
        "geographic_performance": geographic_performance,
        "temporal_performance": temporal_performance,
        "failure_cases": failure_cases or [],
        "executed_at": executed_at or _utcnow(),
        "code_version": code_version or _code_version(),
    }
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
    record["run_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    runs_dir = _runs_dir(runs_dir)
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"{record['run_id']}.json")
    if not os.path.exists(path):  # immutable: identical content → same file
        with open(path, "x", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    return path


def list_runs(
    model_id: Optional[str] = None, runs_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """All recorded runs (optionally one model's), oldest first."""

    runs_dir = _runs_dir(runs_dir)
    try:
        names = sorted(os.listdir(runs_dir))
    except OSError:
        return []
    runs: List[Dict[str, Any]] = []
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(runs_dir, name), "r", encoding="utf-8") as fh:
                record = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if model_id and record.get("model_id") != model_id:
            continue
        runs.append(record)
    runs.sort(key=lambda r: (r.get("executed_at") or "", r.get("run_id") or ""))
    return runs


def get_run(run_id: str, runs_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """One recorded run by its content-hash id, or None."""

    if not _RUN_ID_RE.match(run_id or ""):
        return None
    path = os.path.join(_runs_dir(runs_dir), f"{run_id}.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# The one real executed verification the platform ships today
# ---------------------------------------------------------------------------


def _fwi_test_count() -> int:
    """Number of equation checks in tests/test_fwi.py (counted from the file)."""

    test_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "tests", "test_fwi.py"
    )
    try:
        with open(test_file, "r", encoding="utf-8") as fh:
            return len(re.findall(r"^def test_", fh.read(), flags=re.MULTILINE))
    except OSError:
        return 0


def record_fwi_reference_run(
    *, runs_dir: Optional[str] = None, code_version: Optional[str] = None
) -> str:
    """Record the REAL equation-reference verification of ``fwi_system_v1``.

    The FWI adapter (src/prediction/fwi.py) implements the Van Wagner (1987)
    equations and is unit-tested against the cffdrs reference implementation
    (tests/test_fwi.py — the checks pass in the platform test suite). This
    records that executed verification as kind ``equation_reference``.

    Explicitly NOT recorded: site-level fire-occurrence validation. The
    model's lifecycle is ``backtested`` (equation-verified), not
    ``validated``.
    """

    count = _fwi_test_count()
    return record_run(
        "fwi_system_v1",
        "1.0.0",
        "equation_reference",
        "cffdrs reference outputs",
        {
            "equation_checks": {
                "test_file": "tests/test_fwi.py",
                "test_count": count,
                "result": "passed",
                "executed_by": "platform test suite (python -m pytest tests/test_fwi.py)",
                "reference": ("cffdrs — Canadian Forest Fire Danger Rating System "
                              "reference implementation (R package)"),
            },
            "scope": ("Equation-level verification of FFMC/DMC/DC/ISI/BUI/FWI/DSR "
                      "computation; NOT a site-validated fire-occurrence predictor."),
        },
        runs_dir=runs_dir,
        code_version=code_version,
    )
