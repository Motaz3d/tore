"""
Talaix Knowledge Arm loader and query helpers.

Briefs live as a config JSON registry (``config/briefs_registry.json``),
loaded in the same honest, fail-loud style as the other data registries.

No Flask imports. Public API payloads are read-only; drafts are never served.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "briefs_registry.json"
)


class BriefsRegistryError(RuntimeError):
    """Raised when the briefs registry cannot be loaded or is invalid."""


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def _load_json(path: str, label: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise BriefsRegistryError(f"{label} not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise BriefsRegistryError(f"{label} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise BriefsRegistryError(f"{label} unreadable: {exc}") from exc


def load_briefs(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the full briefs registry."""
    path = path or os.environ.get("HYDRASHIELD_BRIEFS_REGISTRY") or _DEFAULT_PATH
    data = _load_json(path, "Briefs registry")
    if not isinstance(data, dict) or "briefs" not in data:
        raise BriefsRegistryError("Briefs registry must be an object with a 'briefs' list")
    return data


# -----------------------------------------------------------------------------
# Queries
# -----------------------------------------------------------------------------


def _is_published(brief: Dict[str, Any]) -> bool:
    return brief.get("status") == "published"


def list_briefs(kind: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Return a light list of published briefs, sorted by date descending.

    Each entry contains: id, kind, title, date, summary, source_count.
    ``kind`` may be ``framework_explainer`` or ``evidence_brief``.
    """
    data = config if config is not None else load_briefs()
    result: List[Dict[str, Any]] = []
    for brief in data.get("briefs", []):
        if not _is_published(brief):
            continue
        if kind is not None and brief.get("kind") != kind:
            continue
        sources = brief.get("sources") or []
        result.append({
            "id": brief.get("id"),
            "kind": brief.get("kind"),
            "title": brief.get("title"),
            "date": brief.get("date"),
            "summary": brief.get("summary"),
            "source_count": len(sources),
        })
    result.sort(key=lambda b: b.get("date") or "", reverse=True)
    return result


def get_brief(brief_id: str, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Return one published brief dict, or None if not found / not published."""
    data = config if config is not None else load_briefs()
    for brief in data.get("briefs", []):
        if brief.get("id") == brief_id and _is_published(brief):
            return brief
    return None
