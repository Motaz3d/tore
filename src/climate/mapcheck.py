"""
Map Check — cartographic cross-verification engine.

Cross-checks what open map sources SAY (OpenStreetMap green features) against
what satellite observation SHOWS (Sentinel-2 NDVI + ESA WorldCover class), and
reports discrepancies with rule-based possible causes.

Honesty contract:
    - Verdicts are ONLY consistent / discrepancy_detected / cannot_assess.
    - A discrepancy is a signal to verify, never proof that a map is "wrong".
    - Only OPEN sources are compared (OSM + ESA WorldCover + Sentinel-2).
      Proprietary maps (Google/Apple/etc.) are NOT fetched or compared.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from ..dashboard.exposure import _post_overpass
from ..dashboard.real_data import fetch_satellite_data
from ..gis_mapping.landcover import fetch_landcover
from .evidence import EvidenceRecord, content_hash, utcnow_iso

# Verdict vocabulary — never use absolute "error" wording.
VERDICT_CONSISTENT = "consistent"
VERDICT_DISCREPANCY = "discrepancy_detected"
VERDICT_CANNOT_ASSESS = "cannot_assess"

# OpenStreetMap selectors for green / vegetated features.
GREEN_OVERPASS_SELECTORS = [
    'way["leisure"~"^(park|garden|nature_reserve)$"]',
    'relation["leisure"~"^(park|garden|nature_reserve)$"]',
    'way["landuse"~"^(forest|grass|meadow|recreation_ground|village_green)$"]',
    'relation["landuse"~"^(forest|grass|meadow|recreation_ground|village_green)$"]',
    'way["natural"~"^(wood|scrub|grassland)$"]',
    'relation["natural"~"^(wood|scrub|grassland)$"]',
]

# ESA WorldCover classes considered green / vegetated.
GREEN_WORLD_COVER_CLASSES: Set[int] = {10, 20, 30, 90, 95}

# NDVI screening threshold for green vegetation.
NDVI_GREEN_THRESHOLD = 0.35

# Edit older than this is flagged as a possible cause of discrepancy.
OUTDATED_EDIT_YEARS = 5

DISCLAIMER = (
    "This comparison uses only open sources: OpenStreetMap (Overpass API), "
    "ESA WorldCover 10 m 2021 v200, and Sentinel-2 L2A via Earth Search STAC. "
    "Proprietary maps (Google, Apple, Bing, etc.) are NOT fetched or compared "
    "because their terms do not allow automated extraction. Users may compare "
    "those sources manually. A discrepancy is not proof of an error in any "
    "single source — it is a signal to verify before relying on the map."
)

HONESTY_CONTRACT = (
    "Map Check reports only what open map data and open satellite observation "
    "show. Counts are mapped features, not ground truth. Verdicts are limited "
    "to consistent, discrepancy_detected, or cannot_assess. No absolute claim "
    "of map error is made."
)


def _current_year() -> int:
    return datetime.now(timezone.utc).year


def _parse_edit_year(timestamp: Optional[str]) -> Optional[int]:
    """Extract the edit year from an Overpass ISO timestamp, or None."""
    if not timestamp:
        return None
    try:
        # Overpass timestamps are typically "2021-08-15T10:23:45Z".
        return int(str(timestamp)[:4])
    except (ValueError, TypeError):
        return None


def _fetch_green_features(lat: float, lon: float, radius_m: int) -> Dict[str, Any]:
    """
    Query OpenStreetMap for green features around a point via Overpass.

    Returns a dict with {features, count, source, query_note}. Raises on
    total Overpass failure so the caller can degrade to cannot_assess.
    """
    selectors = "\n  ".join(
        f'{sel}(around:{radius_m},{lat},{lon});' for sel in GREEN_OVERPASS_SELECTORS
    )
    query = (
        f"[out:json][timeout:30];\n"
        f"(\n"
        f"  {selectors}\n"
        f");\n"
        f"out meta qt;"
    )

    data = _post_overpass(query, timeout=30.0)
    elements = data.get("elements") or []

    features: List[Dict[str, Any]] = []
    for el in elements:
        tags = el.get("tags") or {}
        kind = (
            tags.get("leisure") or tags.get("landuse") or tags.get("natural") or "green"
        )
        features.append({
            "type": el.get("type", "unknown"),
            "id": el.get("id"),
            "kind": kind,
            "name": tags.get("name"),
            "tags": {
                k: v for k, v in tags.items()
                if k in ("leisure", "landuse", "natural", "name")
            },
            "timestamp": el.get("timestamp"),
            "edit_year": _parse_edit_year(el.get("timestamp")),
        })

    return {
        "features": features,
        "count": len(features),
        "source": "OpenStreetMap (Overpass API)",
        "query_note": (
            "Mapped green features within the radius; OSM completeness varies "
            "by region and a missing feature does not prove the area is not green."
        ),
    }


def map_claim_green(features_result: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the map-side green claim from the OSM feature result."""
    features = features_result.get("features") or []
    edit_years = [f.get("edit_year") for f in features if f.get("edit_year")]
    feature_summaries = [
        {
            "type": f.get("type"),
            "id": f.get("id"),
            "kind": f.get("kind"),
            "name": f.get("name"),
            "edit_year": f.get("edit_year"),
        }
        for f in features
    ]
    return {
        "green_mapped": len(features) > 0,
        "feature_summaries": feature_summaries,
        "oldest_edit_year": min(edit_years) if edit_years else None,
        "newest_edit_year": max(edit_years) if edit_years else None,
        "source": features_result.get("source"),
        "query_note": features_result.get("query_note"),
    }


def satellite_observation_green(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch satellite-derived green signals for the point.

    Combines Sentinel-2 NDVI (recent cloud-free scene) with the dominant ESA
    WorldCover class. Either layer may be unavailable; the result reports
    availability honestly.
    """
    ndvi_data = fetch_satellite_data(lat, lon, days_back=30)
    landcover = fetch_landcover(lat, lon, window_m=500)

    ndvi = ndvi_data.get("ndvi")
    ndvi_error = "error" in ndvi_data
    landcover_error = "error" in landcover

    green_by_ndvi = False
    if not ndvi_error and ndvi is not None:
        try:
            green_by_ndvi = float(ndvi) >= NDVI_GREEN_THRESHOLD
        except (TypeError, ValueError):
            green_by_ndvi = False

    dominant_class = landcover.get("dominant_class")
    green_by_landcover = (
        not landcover_error
        and dominant_class in GREEN_WORLD_COVER_CLASSES
    )

    return {
        "ndvi": ndvi if not ndvi_error else None,
        "ndvi_available": not ndvi_error,
        "ndvi_error": ndvi_data.get("error") if ndvi_error else None,
        "green_by_ndvi": green_by_ndvi,
        "observation_date": ndvi_data.get("observation_date") if not ndvi_error else None,
        "ndvi_source": ndvi_data.get("source") if not ndvi_error else None,
        "landcover_class": dominant_class if not landcover_error else None,
        "landcover_label": landcover.get("dominant_label") if not landcover_error else None,
        "landcover_available": not landcover_error,
        "landcover_error": landcover.get("error") if landcover_error else None,
        "landcover_source": landcover.get("source") if not landcover_error else None,
        "green_by_landcover": green_by_landcover,
        "satellite_available": not ndvi_error or not landcover_error,
    }


def _possible_causes_map_green_satellite_not(
    map_claim: Dict[str, Any],
    satellite: Dict[str, Any],
) -> List[str]:
    causes: List[str] = []
    oldest = map_claim.get("oldest_edit_year")
    if oldest is not None and (_current_year() - oldest) > OUTDATED_EDIT_YEARS:
        causes.append(
            f"OSM feature data may be outdated (last edit {oldest})"
        )
    if not satellite.get("ndvi_available"):
        causes.append("recent Sentinel-2 observation unavailable (cloud / revisit gap)")
    else:
        causes.append("satellite revisit / cloud cover / seasonal vegetation low")
    causes.append("10–30 m resolution thresholds may miss small mapped features")
    causes.append("real land-use change since the OSM edit")
    return causes


def _possible_causes_satellite_green_map_not(
    _satellite: Dict[str, Any],
) -> List[str]:
    return [
        "OSM completeness gap — feature may simply be unmapped (very common)",
        "private or informal green space not recorded in OSM",
        "recent planting or regeneration after the OSM edit",
        "resolution / scale mismatch between 10 m pixels and OSM geometry",
    ]


def _build_osm_evidence(map_claim: Dict[str, Any]) -> Dict[str, Any]:
    return EvidenceRecord.open_data(
        source=map_claim.get("source") or "OpenStreetMap (Overpass API)",
        dataset="OpenStreetMap green-feature query",
        method="Overpass union query for leisure/landuse/natural green tags",
        limitations=map_claim.get("query_note"),
        confidence="MEDIUM",
    ).to_dict()


def _build_satellite_evidence(satellite: Dict[str, Any]) -> Dict[str, Any]:
    limitations = []
    if satellite.get("ndvi_available"):
        limitations.append(
            f"Sentinel-2 NDVI {satellite.get('ndvi')} from "
            f"{satellite.get('observation_date')} ({satellite.get('ndvi_source')})"
        )
    else:
        limitations.append(
            f"NDVI unavailable: {satellite.get('ndvi_error')}"
        )
    if satellite.get("landcover_available"):
        limitations.append(
            f"ESA WorldCover class {satellite.get('landcover_class')} "
            f"({satellite.get('landcover_label')}) — {satellite.get('landcover_source')}"
        )
    else:
        limitations.append(
            f"WorldCover unavailable: {satellite.get('landcover_error')}"
        )

    return EvidenceRecord.satellite(
        source="Sentinel-2 + ESA WorldCover",
        dataset="NDVI + WorldCover land-cover class",
        method="NDVI threshold screening and dominant-class lookup",
        limitations="; ".join(limitations),
        confidence="MEDIUM",
    ).to_dict()


def _check_map_green_vs_satellite(
    map_claim: Dict[str, Any],
    satellite: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check A: when OSM says green, does satellite observation agree?
    """
    osm_ev = _build_osm_evidence(map_claim)
    sat_ev = _build_satellite_evidence(satellite)

    if not map_claim.get("green_mapped"):
        return {
            "id": "green_mapped_vs_satellite",
            "result": VERDICT_CONSISTENT,
            "basis": "No mapped green features within the radius; nothing to contradict.",
            "map_claim": map_claim,
            "satellite_observation": satellite,
            "possible_causes": [],
            "evidence": [osm_ev, sat_ev],
        }

    if not satellite.get("satellite_available"):
        gap_reason = []
        if not satellite.get("ndvi_available"):
            gap_reason.append(f"NDVI: {satellite.get('ndvi_error')}")
        if not satellite.get("landcover_available"):
            gap_reason.append(f"WorldCover: {satellite.get('landcover_error')}")
        return {
            "id": "green_mapped_vs_satellite",
            "result": VERDICT_CANNOT_ASSESS,
            "basis": (
                "OpenStreetMap records green features but satellite observation "
                "is unavailable, so no comparison is possible."
            ),
            "map_claim": map_claim,
            "satellite_observation": satellite,
            "possible_causes": [],
            "evidence": [osm_ev, sat_ev],
            "declared_gap": "; ".join(gap_reason),
        }

    satellite_says_green = satellite.get("green_by_ndvi") or satellite.get("green_by_landcover")
    if satellite_says_green:
        basis_parts = ["Mapped green features agree with satellite observation."]
        if satellite.get("ndvi_available"):
            basis_parts.append(f"NDVI = {satellite.get('ndvi'):.3f}.")
        if satellite.get("landcover_available"):
            basis_parts.append(
                f"WorldCover = {satellite.get('landcover_label')} "
                f"(class {satellite.get('landcover_class')})."
            )
        return {
            "id": "green_mapped_vs_satellite",
            "result": VERDICT_CONSISTENT,
            "basis": " ".join(basis_parts),
            "map_claim": map_claim,
            "satellite_observation": satellite,
            "possible_causes": [],
            "evidence": [osm_ev, sat_ev],
        }

    # Map says green, satellite does not.
    basis_parts = [
        "OpenStreetMap records green features, but satellite observation does not:",
    ]
    if satellite.get("ndvi_available"):
        basis_parts.append(f"NDVI = {satellite.get('ndvi'):.3f} (threshold {NDVI_GREEN_THRESHOLD}).")
    if satellite.get("landcover_available"):
        basis_parts.append(
            f"WorldCover = {satellite.get('landcover_label')} "
            f"(class {satellite.get('landcover_class')})."
        )
    return {
        "id": "green_mapped_vs_satellite",
        "result": VERDICT_DISCREPANCY,
        "basis": " ".join(basis_parts),
        "map_claim": map_claim,
        "satellite_observation": satellite,
        "possible_causes": _possible_causes_map_green_satellite_not(map_claim, satellite),
        "evidence": [osm_ev, sat_ev],
    }


def _check_satellite_green_vs_map(
    map_claim: Dict[str, Any],
    satellite: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check B: when satellite says green, does OSM agree?
    """
    osm_ev = _build_osm_evidence(map_claim)
    sat_ev = _build_satellite_evidence(satellite)

    if not satellite.get("satellite_available"):
        gap_reason = []
        if not satellite.get("ndvi_available"):
            gap_reason.append(f"NDVI: {satellite.get('ndvi_error')}")
        if not satellite.get("landcover_available"):
            gap_reason.append(f"WorldCover: {satellite.get('landcover_error')}")
        return {
            "id": "satellite_green_vs_map",
            "result": VERDICT_CANNOT_ASSESS,
            "basis": (
                "Satellite observation is unavailable, so no comparison against "
                "mapped features is possible."
            ),
            "map_claim": map_claim,
            "satellite_observation": satellite,
            "possible_causes": [],
            "evidence": [osm_ev, sat_ev],
            "declared_gap": "; ".join(gap_reason),
        }

    satellite_says_green = satellite.get("green_by_ndvi") or satellite.get("green_by_landcover")
    if not satellite_says_green:
        return {
            "id": "satellite_green_vs_map",
            "result": VERDICT_CONSISTENT,
            "basis": "Satellite observation does not indicate green vegetation; nothing to contradict.",
            "map_claim": map_claim,
            "satellite_observation": satellite,
            "possible_causes": [],
            "evidence": [osm_ev, sat_ev],
        }

    if map_claim.get("green_mapped"):
        basis_parts = ["Satellite indicates green vegetation and OSM records green features."]
        if satellite.get("ndvi_available"):
            basis_parts.append(f"NDVI = {satellite.get('ndvi'):.3f}.")
        if satellite.get("landcover_available"):
            basis_parts.append(
                f"WorldCover = {satellite.get('landcover_label')} "
                f"(class {satellite.get('landcover_class')})."
            )
        return {
            "id": "satellite_green_vs_map",
            "result": VERDICT_CONSISTENT,
            "basis": " ".join(basis_parts),
            "map_claim": map_claim,
            "satellite_observation": satellite,
            "possible_causes": [],
            "evidence": [osm_ev, sat_ev],
        }

    # Satellite says green, map does not.
    basis_parts = [
        "Satellite observation indicates green vegetation, but OpenStreetMap records no green feature:",
    ]
    if satellite.get("ndvi_available"):
        basis_parts.append(f"NDVI = {satellite.get('ndvi'):.3f} (threshold {NDVI_GREEN_THRESHOLD}).")
    if satellite.get("landcover_available"):
        basis_parts.append(
            f"WorldCover = {satellite.get('landcover_label')} "
            f"(class {satellite.get('landcover_class')})."
        )
    return {
        "id": "satellite_green_vs_map",
        "result": VERDICT_DISCREPANCY,
        "basis": " ".join(basis_parts),
        "map_claim": map_claim,
        "satellite_observation": satellite,
        "possible_causes": _possible_causes_satellite_green_map_not(satellite),
        "evidence": [osm_ev, sat_ev],
    }


def check_map_vs_satellite(lat: float, lon: float, radius_m: int = 300) -> Dict[str, Any]:
    """
    Run the two-way map-vs-satellite green cross-check.

    Returns a top-level result dict with honest verdicts, possible causes,
    evidence records, declared gaps, recommendations, disclaimer and
    honesty contract.
    """
    lat = float(lat)
    lon = float(lon)
    radius_m = int(radius_m)

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError("lat/lon out of range")
    if not (50 <= radius_m <= 2000):
        raise ValueError("radius_m must be between 50 and 2000")

    # Fetch inputs, isolating failures so the engine can report cannot_assess.
    map_claim: Dict[str, Any] = {
        "green_mapped": False,
        "feature_summaries": [],
        "oldest_edit_year": None,
        "newest_edit_year": None,
        "source": "OpenStreetMap (Overpass API)",
        "query_note": "Overpass request failed; map claim is unknown.",
    }
    map_error: Optional[str] = None
    try:
        features_result = _fetch_green_features(lat, lon, radius_m)
        map_claim = map_claim_green(features_result)
    except Exception as exc:
        map_error = f"OpenStreetMap green-feature query failed: {exc}"
        map_claim["query_note"] = map_error

    satellite = satellite_observation_green(lat, lon)

    if map_error and not satellite.get("satellite_available"):
        # Both inputs missing: cannot assess anything.
        return {
            "check_id": content_hash({"lat": lat, "lon": lon, "radius_m": radius_m})[:16],
            "location": {"lat": lat, "lon": lon, "radius_m": radius_m},
            "generated_at": utcnow_iso(),
            "status": "unavailable",
            "checks": [],
            "discrepancies_count": 0,
            "recommendations": [
                "Retry when both OSM and satellite sources are reachable.",
                "Compare any available local map layer manually.",
            ],
            "declared_gaps": [
                {"component": "openstreetmap", "reason": map_error},
                {"component": "satellite", "reason": "Satellite observation unavailable."},
            ],
            "disclaimer": DISCLAIMER,
            "honesty_contract": HONESTY_CONTRACT,
        }

    if map_error:
        # Map missing but satellite present: only check B is possible.
        check_b = _check_satellite_green_vs_map(map_claim, satellite)
        declared_gaps = [{"component": "openstreetmap", "reason": map_error}]
        discrepancies = 1 if check_b["result"] == VERDICT_DISCREPANCY else 0
        return {
            "check_id": content_hash({"lat": lat, "lon": lon, "radius_m": radius_m})[:16],
            "location": {"lat": lat, "lon": lon, "radius_m": radius_m},
            "generated_at": utcnow_iso(),
            "status": "degraded",
            "checks": [check_b],
            "discrepancies_count": discrepancies,
            "recommendations": [
                "OpenStreetMap query failed; verify map side manually before drawing conclusions.",
                "If satellite shows green, the area may simply be unmapped in OSM.",
            ],
            "declared_gaps": declared_gaps,
            "disclaimer": DISCLAIMER,
            "honesty_contract": HONESTY_CONTRACT,
        }

    # Normal path: both inputs available (satellite may still be partially unavailable).
    check_a = _check_map_green_vs_satellite(map_claim, satellite)
    check_b = _check_satellite_green_vs_map(map_claim, satellite)
    checks = [check_a, check_b]

    discrepancies = sum(1 for c in checks if c["result"] == VERDICT_DISCREPANCY)
    declared_gaps: List[Dict[str, str]] = []
    for c in checks:
        gap = c.get("declared_gap")
        if gap:
            declared_gaps.append({"component": c["id"], "reason": gap})

    return {
        "check_id": content_hash({"lat": lat, "lon": lon, "radius_m": radius_m})[:16],
        "location": {"lat": lat, "lon": lon, "radius_m": radius_m},
        "generated_at": utcnow_iso(),
        "status": "ok",
        "checks": checks,
        "discrepancies_count": discrepancies,
        "recommendations": [
            "Cross-check any single map source before relying on it for high-stakes decisions.",
            "Compare multiple open sources (OSM, national topographic services, local cadastre).",
            "Use on-site verification for transactions, permitting or investment decisions.",
            "Discrepancies can often be reported as corrections to the OpenStreetMap community.",
        ],
        "declared_gaps": declared_gaps,
        "disclaimer": DISCLAIMER,
        "honesty_contract": HONESTY_CONTRACT,
    }
