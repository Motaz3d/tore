"""
Data Observatory — loader and query helpers for config/data_registry.json.

The data registry is the platform's dataset catalogue: every entry is a
catalog record with ``status`` in {integrated, candidate, rejected}. Only
``integrated`` entries are wired into a pipeline — candidates are catalogued
for evaluation, nothing more.

The registry is loaded lazily on first access and cached in-process
(same discipline as :mod:`src.climate.registry`). Loading validates the
document and fails loud on a malformed registry — a broken catalogue must
never silently degrade to "no data": tests raise, the API returns 503.

Also hosts the loaders for the model and research registries
(config/model_registry.json, config/research_registry.json) with the same
validation discipline.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

_REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "data_registry.json"
)
_MODEL_REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "model_registry.json"
)
_RESEARCH_REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "research_registry.json"
)

#: Every entry must carry all of these keys (null allowed where genuinely
#: unknown — the loader checks presence, not truthiness).
REQUIRED_FIELDS = (
    "id", "name", "provider", "provider_class", "url", "license",
    "geographic_coverage", "temporal_coverage", "spatial_resolution",
    "temporal_resolution", "update_frequency", "variables",
    "hazard_relevance", "provenance", "quality", "access_method",
    "api_or_download_url", "commercial_use", "status", "status_note",
)

VALID_STATUSES = frozenset({"integrated", "candidate", "rejected"})
VALID_PROVIDER_CLASSES = frozenset({
    "government", "eu_copernicus", "scientific_agency", "un_agency",
    "community", "commercial",
})
VALID_ACCESS_METHODS = frozenset({
    "api", "download", "stac", "wms", "registration_required",
})
VALID_COMMERCIAL_USE = frozenset({
    "allowed", "allowed_with_attribution", "restricted", "unknown",
})

#: Registry navigation vocabulary (optional per-entry ``catalog_group``
#: label, used by the /sources page grouping and the ?catalog_group= filter).
VALID_CATALOG_GROUPS = frozenset({
    "global_portal", "national_portal", "national_service",
    "international_org", "hazard_disaster", "earth_observation", "climate",
    "environment", "socio_economic", "energy_infrastructure",
    "evidence_knowledge",
})

VALID_MODEL_STATUSES = frozenset({
    "not_validated", "validation_in_progress", "validated_screening",
    "validated_operational", "deprecated",
})


class RegistryError(RuntimeError):
    """Raised when a registry document is missing, unreadable or malformed."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_entry(entry: dict, seen_ids: set) -> None:
    if not isinstance(entry, dict):
        raise RegistryError(f"dataset entry is not an object: {entry!r:.80}")
    eid = entry.get("id") or "<no id>"
    for field in REQUIRED_FIELDS:
        if field not in entry:
            raise RegistryError(f"dataset '{eid}': missing required field '{field}'")
    if not entry["id"]:
        raise RegistryError("dataset entry with empty id")
    if entry["id"] in seen_ids:
        raise RegistryError(f"duplicate dataset id '{entry['id']}'")
    seen_ids.add(entry["id"])
    url = entry.get("url") or ""
    if not (isinstance(url, str) and url.startswith("https://")):
        raise RegistryError(f"dataset '{eid}': url must be https, got {url!r}")
    if entry["status"] not in VALID_STATUSES:
        raise RegistryError(
            f"dataset '{eid}': invalid status '{entry['status']}'")
    if entry["provider_class"] not in VALID_PROVIDER_CLASSES:
        raise RegistryError(
            f"dataset '{eid}': invalid provider_class '{entry['provider_class']}'")
    if entry["access_method"] not in VALID_ACCESS_METHODS:
        raise RegistryError(
            f"dataset '{eid}': invalid access_method '{entry['access_method']}'")
    if entry["commercial_use"] not in VALID_COMMERCIAL_USE:
        raise RegistryError(
            f"dataset '{eid}': invalid commercial_use '{entry['commercial_use']}'")
    for list_field in ("variables", "hazard_relevance"):
        if not isinstance(entry[list_field], list):
            raise RegistryError(
                f"dataset '{eid}': '{list_field}' must be a list")


def _validate_doc(doc: object) -> Dict:
    if not isinstance(doc, dict) or not isinstance(doc.get("datasets"), list):
        raise RegistryError("data registry must be an object with a 'datasets' list")
    seen: set = set()
    for entry in doc["datasets"]:
        _validate_entry(entry, seen)
    return doc


def _load_json(path: str, label: str) -> object:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise RegistryError(f"{label} not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{label} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise RegistryError(f"{label} unreadable: {exc}") from exc


# ---------------------------------------------------------------------------
# Data registry (lazy, cached)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_doc: Optional[Dict] = None


def _ensure() -> Dict:
    global _doc
    if _doc is None:
        with _lock:
            if _doc is None:
                _doc = _validate_doc(_load_json(_REGISTRY_PATH, "data registry"))
    return _doc


def all() -> List[Dict]:
    """All dataset entries (catalog records — see status discipline above)."""
    return list(_ensure()["datasets"])


def get(dataset_id: str) -> Optional[Dict]:
    for entry in _ensure()["datasets"]:
        if entry["id"] == dataset_id:
            return entry
    return None


def by_status(status: str) -> List[Dict]:
    return [e for e in _ensure()["datasets"] if e["status"] == status]


def by_hazard(hazard: str) -> List[Dict]:
    """Entries whose declared hazard_relevance includes ``hazard``."""
    hazard = hazard.strip().lower()
    return [e for e in _ensure()["datasets"] if hazard in e["hazard_relevance"]]


def by_provider_class(provider_class: str) -> List[Dict]:
    return [e for e in _ensure()["datasets"]
            if e["provider_class"] == provider_class]


def _region_bucket(coverage: str) -> str:
    """Coarse region bucket derived from the free-text geographic_coverage.

    Declared heuristic for summary() only — never used for filtering logic.
    """
    c = (coverage or "").lower()
    if "global" in c:
        return "global"
    if any(k in c for k in ("europe", "eu/", "eu-", "eu ", "european")):
        return "europe"
    if any(k in c for k in ("china", "asia", "japan", "korea", "india", "tibetan")):
        return "asia"
    if any(k in c for k in ("united states", "usgs", "us ")):
        return "north_america"
    if "australia" in c:
        return "oceania"
    return "other"


def summary() -> Dict:
    """Counts by status, provider_class and region bucket."""
    entries = _ensure()["datasets"]
    by_s: Dict[str, int] = {}
    by_c: Dict[str, int] = {}
    by_r: Dict[str, int] = {}
    for e in entries:
        by_s[e["status"]] = by_s.get(e["status"], 0) + 1
        by_c[e["provider_class"]] = by_c.get(e["provider_class"], 0) + 1
        region = _region_bucket(e.get("geographic_coverage", ""))
        by_r[region] = by_r.get(region, 0) + 1
    return {
        "total": len(entries),
        "by_status": by_s,
        "by_provider_class": by_c,
        "by_region": by_r,
    }


def reset_for_tests() -> None:
    """Drop the cached document (used by tests that swap registry files)."""
    global _doc
    with _lock:
        _doc = None


# ---------------------------------------------------------------------------
# Model registry (config/model_registry.json) — same validation discipline
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_model_doc: Optional[Dict] = None


def _validate_models(doc: object) -> Dict:
    if not isinstance(doc, dict) or not isinstance(doc.get("models"), list):
        raise RegistryError("model registry must be an object with a 'models' list")
    seen: set = set()
    for m in doc["models"]:
        mid = (m or {}).get("id") or "<no id>"
        for field in ("id", "version", "name", "methodology", "validation",
                      "limitations"):
            if field not in m:
                raise RegistryError(f"model '{mid}': missing required field '{field}'")
        if m["id"] in seen:
            raise RegistryError(f"duplicate model id '{m['id']}'")
        seen.add(m["id"])
        status = (m.get("validation") or {}).get("status")
        if status not in VALID_MODEL_STATUSES:
            raise RegistryError(
                f"model '{mid}': invalid validation status '{status}'")
    return doc


def _ensure_models() -> Dict:
    global _model_doc
    if _model_doc is None:
        with _model_lock:
            if _model_doc is None:
                _model_doc = _validate_models(
                    _load_json(_MODEL_REGISTRY_PATH, "model registry"))
    return _model_doc


def models_all() -> List[Dict]:
    return list(_ensure_models()["models"])


def models_get(model_id: str) -> Optional[Dict]:
    for m in _ensure_models()["models"]:
        if m["id"] == model_id:
            return m
    return None


# ---------------------------------------------------------------------------
# Research registry (config/research_registry.json) — same discipline
# ---------------------------------------------------------------------------

_research_lock = threading.Lock()
_research_doc: Optional[Dict] = None


def _validate_research(doc: object) -> Dict:
    if not isinstance(doc, dict) or not isinstance(doc.get("references"), list):
        raise RegistryError(
            "research registry must be an object with a 'references' list")
    stages = doc.get("pipeline_stages") or []
    seen: set = set()
    for r in doc["references"]:
        rid = (r or {}).get("id") or "<no id>"
        for field in ("id", "title", "year", "url", "pipeline_stage"):
            if field not in r:
                raise RegistryError(
                    f"research '{rid}': missing required field '{field}'")
        if r["id"] in seen:
            raise RegistryError(f"duplicate research id '{r['id']}'")
        seen.add(r["id"])
        url = r.get("url") or ""
        if not (isinstance(url, str) and url.startswith("https://")):
            raise RegistryError(
                f"research '{rid}': url must be https, got {url!r}")
        if stages and r["pipeline_stage"] not in stages:
            raise RegistryError(
                f"research '{rid}': pipeline_stage '{r['pipeline_stage']}' "
                "not in declared pipeline_stages")
    return doc


def _ensure_research() -> Dict:
    global _research_doc
    if _research_doc is None:
        with _research_lock:
            if _research_doc is None:
                _research_doc = _validate_research(
                    _load_json(_RESEARCH_REGISTRY_PATH, "research registry"))
    return _research_doc


def research_all() -> List[Dict]:
    return list(_ensure_research()["references"])


def research_get(ref_id: str) -> Optional[Dict]:
    for r in _ensure_research()["references"]:
        if r["id"] == ref_id:
            return r
    return None
