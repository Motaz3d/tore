"""
Regulatory knowledge base loader for CsrdTX.

Loads the versioned regulatory data from ``config/csrd/*.json`` and
selects the correct rule set / ESRS version for a given reporting year.

The knowledge base is the *only* place regulatory facts live. Legal
status is first-class: entries whose status is not ``in_force`` are
reported but never silently applied.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "csrd"
)

STATUS_IN_FORCE = "in_force"
STATUS_ADOPTED_PENDING = "adopted_pending_application"
STATUS_PROPOSED = "proposed"


def _config_dir() -> str:
    return os.environ.get("HYDRASHIELD_CSRD_CONFIG") or _CONFIG_DIR


def _load(name: str) -> Dict[str, Any]:
    path = os.path.join(_config_dir(), name)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_kb() -> Dict[str, Any]:
    """Load the whole regulatory knowledge base (cached per process)."""
    return {
        "applicability_rules": _load("applicability_rules.json"),
        "changelog": _load("changelog.json"),
        "esrs": {
            "esrs_2023": _load("esrs_2023.json"),
            "esrs_2026_simplified": _load("esrs_2026_simplified.json"),
        },
    }


def load_changelog() -> Dict[str, Any]:
    """The regulatory watch change log (events, statuses, affected artefacts)."""
    return load_kb()["changelog"]


def esrs_versions() -> List[Dict[str, Any]]:
    """Metadata for every ESRS version in the knowledge base."""
    out: List[Dict[str, Any]] = []
    for version_id, doc in load_kb()["esrs"].items():
        out.append({
            "id": version_id,
            "name": doc.get("name"),
            "short_name": doc.get("short_name"),
            "status": doc.get("status"),
            "adopted": doc.get("adopted"),
            "source": doc.get("source"),
        })
    return out


def esrs_version(version_id: Optional[str] = None) -> Dict[str, Any]:
    """Return one ESRS version document, resolving ``topics_inherits``.

    Defaults to the newest version whose status is ``in_force``.
    """
    esrs = load_kb()["esrs"]
    if version_id is None:
        in_force = [vid for vid, doc in esrs.items() if doc.get("status") == STATUS_IN_FORCE]
        if not in_force:
            raise ValueError("No in-force ESRS version in the knowledge base")
        version_id = sorted(in_force)[-1]
    if version_id not in esrs:
        raise ValueError(f"Unknown ESRS version '{version_id}'")
    doc = dict(esrs[version_id])
    inherits = doc.pop("topics_inherits", None)
    if inherits:
        if inherits not in esrs:
            raise ValueError(f"ESRS version '{version_id}' inherits unknown '{inherits}'")
        parent = esrs_version(inherits)
        doc.setdefault("topics", parent.get("topics", []))
        doc["topics_inherited_from"] = inherits
    doc["version_id"] = version_id
    return doc


def rule_set_for_year(
    reporting_year: int,
    *,
    statuses: tuple = (STATUS_IN_FORCE,),
) -> Dict[str, Any]:
    """Select the applicability rule set for a reporting year.

    By default only ``in_force`` rule sets are eligible — proposed rules
    are never applied, only reported. Pass ``statuses`` explicitly to
    evaluate proposed rule sets (used for the Omnibus forward outlook).
    """
    candidates: List[Dict[str, Any]] = []
    for rs in load_kb()["applicability_rules"]["rule_sets"]:
        if rs.get("status") not in statuses:
            continue
        if reporting_year in (rs.get("applies_to_reporting_years") or []):
            candidates.append(rs)
    if not candidates:
        raise ValueError(
            f"No applicability rule set with status {statuses} covers "
            f"reporting year {reporting_year}"
        )
    # Most specific first: rule sets with a narrower year window win.
    candidates.sort(key=lambda rs: len(rs.get("applies_to_reporting_years") or []))
    return candidates[0]


def wave_calendar() -> List[Dict[str, Any]]:
    return load_kb()["applicability_rules"]["wave_calendar"]
