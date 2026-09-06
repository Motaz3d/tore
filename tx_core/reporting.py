"""
TX reporting — rendering a :class:`~tx_core.models.TxResult` into standard
output shapes (JSON-ready dict, GeoJSON feature collection, markdown).

The GeoJSON export is deliberately dependency-free: it builds plain
``FeatureCollection`` structures from hazard result provenance/location —
no geopandas/rasterio required. Raster/vector heavy exports are a later TX
layer (docs/TX_ENGINE.md).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import TxResult


def result_to_dict(result: TxResult) -> Dict[str, Any]:
    """The canonical JSON-ready shape of a TX result."""
    return result.to_dict()


def result_to_geojson(
    result: TxResult, *, point_radius_km: float = 50.0
) -> Dict[str, Any]:
    """One Point feature per hazard result (location + level label).

    ``point_radius_km`` is informational only (no geometry buffering is
    performed without a real GIS dependency — honest about it).
    """
    features: List[Dict[str, Any]] = []
    for hr in result.results:
        properties = {
            "hazard": hr.hazard,
            "status": hr.status,
            "summary": hr.summary,
            "analysis_id": result.analysis_id,
            "engine_version": result.engine_version,
            "tx_version": result.tx_version,
            "depth": result.depth,
            "point_radius_km": point_radius_km,
        }
        if hr.level:
            level = hr.level.to_dict() if hasattr(hr.level, "to_dict") else hr.level
            properties["level_label"] = level.get("label")
            properties["level_score"] = level.get("score")
            properties["level_validated"] = level.get("validated", False)
            properties["level_basis"] = level.get("basis", "")
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [result.location.lon, result.location.lat],
                },
                "properties": properties,
            }
        )
    return {
        "type": "FeatureCollection",
        "generated_at": result.generated_at,
        "features": features,
    }


def result_to_markdown(result: TxResult) -> str:
    """A human-readable summary (CLI ``--format md``)."""
    lines: List[str] = [
        f"# TX Analysis {result.analysis_id}",
        "",
        f"- Location: {result.location.name or ''} "
        f"({result.location.lat:.4f}, {result.location.lon:.4f})",
        f"- Depth: {result.depth}",
        f"- Status: {result.status}",
        f"- Engine: tx-core {result.tx_version} / {result.engine_version}",
        f"- TAM envelope: {result.tam_version}",
        f"- Generated: {result.generated_at}",
        "",
        "## Hazards",
        "",
    ]
    for hr in result.results:
        lines.append(f"### {hr.hazard} — {hr.status}")
        if hr.summary:
            lines.append(f"{hr.summary}")
        if hr.level:
            level = hr.level.to_dict() if hasattr(hr.level, "to_dict") else hr.level
            lines.append(
                f"- Level: {level.get('label')} "
                f"(validated: {level.get('validated', False)})"
            )
        if hr.unavailable_reason:
            lines.append(f"- Reason: {hr.unavailable_reason}")
        lines.append("")
    if result.sources:
        lines.append("## Sources")
        lines.append("")
        for s in result.sources:
            lines.append(f"- [{s.get('name')}]({s.get('url')})")
        lines.append("")
    return "\n".join(lines)
