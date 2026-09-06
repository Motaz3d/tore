"""
Talaix Environmental Licensing Advisory engine.

Dependency-free core (no Flask imports). It builds a pre-draft environmental
licensing evidence dossier for a site: the same evidence base serves both
sides of the table — applicants (investors, banks, developers) preparing a
permit application, and authorities (governments, municipalities) screening
one.

Every dossier statement carries the platform evidence taxonomy:

- ``OBSERVED``   — Sentinel-2 visual / spectral evidence for the site.
- ``DOCUMENTED`` — ESA WorldCover land cover and framework references.
- ``REPORTED``   — recorded environmental events at or near the site.
- ``MODELLED``   — multi-hazard exposure screening (registry hazard modules).
- ``INFERRED``   — constraints & risk flags derived from the evidence base.
- ``UNKNOWN``    — declared data gaps; gaps are stated, never filled in.

Honesty contract: an unavailable layer is declared UNKNOWN with a reason;
nothing is invented or silently dropped. The dossier is advisory evidence —
NOT a legal permit, NOT an official government decision, and NOT a substitute
for the competent authority's licensing procedure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .evidence import content_hash, utcnow_iso

ENGINE_VERSION = "1.0.0"

#: Hazards screened for the dossier's multi-hazard exposure layer, with the
#: permit-facing label each hazard carries inside the dossier. Only hazards
#: present in the deployment's registry are screened; the rest are declared
#: gaps (UNKNOWN), never dropped silently.
LICENSING_HAZARDS: Dict[str, Dict[str, str]] = {
    "wildfire": {"permit_label": "Wildfire exposure"},
    "flood": {"permit_label": "Flood exposure (riverine / pluvial)"},
    "drought": {"permit_label": "Drought & water stress"},
    "heat": {"permit_label": "Extreme heat"},
    "wind": {"permit_label": "Extreme wind / storms"},
    "coastal": {"permit_label": "Coastal exposure & sea-level rise"},
    "cyclone": {"permit_label": "Tropical cyclones"},
}

#: Which side of the table the dossier is framed for. The evidence base is
#: identical for both; only the framing note changes.
APPLICANT_SIDES: Dict[str, Dict[str, str]] = {
    "applicant": {
        "label": "Investor / bank / developer",
        "framing": (
            "Pre-application evidence: the site's environmental strengths and "
            "constraints documented up front, ready to attach to a permit "
            "request to the competent authority."
        ),
    },
    "authority": {
        "label": "Government / municipality",
        "framing": (
            "Independent screening evidence: satellite-backed context to "
            "evaluate a licence application faster and more defensibly — the "
            "same engine, the other side of the table."
        ),
    },
}

#: Project typologies the dossier can be framed for. Each carries the
#: licensing-relevant note shown in the dossier (screening context only).
PROJECT_TYPOLOGIES: Dict[str, Dict[str, str]] = {
    "solar_pv": {
        "label": "Solar PV plant",
        "note": "Land-take, vegetation clearance and heat-island context are "
                "typically material for ground-mounted solar permitting.",
    },
    "wind": {
        "label": "Wind farm",
        "note": "Wind-resource siting, access infrastructure and habitat "
                "constraints are typically material for wind permitting.",
    },
    "industrial": {
        "label": "Industrial facility",
        "note": "Flood, fire and water-stress exposure typically drive "
                "environmental permit conditions for industrial sites.",
    },
    "tourism": {
        "label": "Tourism development",
        "note": "Coastal, water-stress and land-cover context is typically "
                "material for tourism developments in sensitive areas.",
    },
    "residential": {
        "label": "Residential / urban development",
        "note": "Flood, heat and land-take context typically shape urban "
                "development permits.",
    },
    "agriculture": {
        "label": "Agriculture / agri-processing",
        "note": "Drought, water availability and land-cover change are "
                "typically material for agricultural permits.",
    },
    "infrastructure": {
        "label": "Linear infrastructure (roads, grids, pipelines)",
        "note": "Cross-terrain hazard exposure and land-cover intersections "
                "typically drive route permitting.",
    },
    "other": {
        "label": "Other development",
        "note": "Generic screening context; the evidence base is "
                "typology-neutral.",
    },
}

#: Permit / consent types the dossier can be framed for.
PERMIT_TYPES: Dict[str, Dict[str, str]] = {
    "eia_screening": {
        "label": "EIA screening / scoping",
        "note": "Evidence base oriented to environmental impact screening "
                "and scoping-stage questions.",
    },
    "construction_permit": {
        "label": "Construction permit",
        "note": "Evidence base oriented to site construction consent.",
    },
    "operating_licence": {
        "label": "Operating / environmental licence",
        "note": "Evidence base oriented to operational environmental consent.",
    },
    "water_abstraction": {
        "label": "Water abstraction / discharge consent",
        "note": "Evidence base oriented to water availability, drought and "
                "water-framework context.",
    },
    "land_use_change": {
        "label": "Land-use change consent",
        "note": "Evidence base oriented to land-cover change and vegetation "
                "clearance context.",
    },
    "other": {
        "label": "Other permit / consent",
        "note": "Generic screening context; the evidence base is "
                "permit-neutral.",
    },
}

#: International frameworks referenced by the dossier (DOCUMENTED).
LICENSING_FRAMEWORKS: List[Dict[str, str]] = [
    {
        "id": "eia_directive",
        "name": "EIA Directive 2011/92/EU (as amended by 2014/52/EU)",
        "role": "environmental impact assessment context",
        "note": (
            "Provides the screening/scoping vocabulary for environmental "
            "impact assessment of public and private projects; the dossier's "
            "hazard, land-cover and event evidence maps to EIA screening "
            "information expectations."
        ),
    },
    {
        "id": "sea_directive",
        "name": "SEA Directive 2001/42/EC",
        "role": "strategic environmental assessment context",
        "note": (
            "Frames the plan/programme-level environmental assessment "
            "context for authorities screening multiple or area-wide "
            "developments."
        ),
    },
    {
        "id": "habitats_directive",
        "name": "Habitats & Birds Directives (92/43/EEC, 2009/147/EC)",
        "role": "nature conservation context",
        "note": (
            "Land-cover and vegetation evidence supports early screening "
            "for potential effects on protected habitats and species "
            "(appropriate-assessment trigger screening only)."
        ),
    },
    {
        "id": "water_framework",
        "name": "Water Framework Directive 2000/60/EC",
        "role": "water status context",
        "note": (
            "Drought and water-stress evidence provides context for "
            "abstraction and discharge consents tied to water-body status."
        ),
    },
    {
        "id": "espoo",
        "name": "Espoo Convention (UNECE)",
        "role": "transboundary context",
        "note": (
            "Referenced where a site's environmental impact may cross "
            "borders; the dossier states transboundary evidence needs as "
            "UNKNOWN unless documented by the authority."
        ),
    },
    {
        "id": "ifc_ps",
        "name": "IFC Performance Standards / Equator Principles",
        "role": "lender safeguard context",
        "note": (
            "Banks and investors typically map site environmental risk "
            "evidence to IFC PS1 (risk assessment) and PS3/PS6 (resource "
            "efficiency, biodiversity) safeguard expectations."
        ),
    },
]

DISCLAIMER = (
    "This dossier is an advisory evidence product. It is NOT a legal permit, "
    "NOT an official government decision, and NOT a substitute for the "
    "competent authority's licensing procedure. Final licensing decisions "
    "remain with the relevant authorities."
)

HONESTY_CONTRACT = (
    "Unavailable data is declared, never invented: every missing or "
    "unsuitable evidence layer is recorded as UNKNOWN with a stated reason, "
    "and every dossier statement is labelled with its evidence type."
)

#: Severity ladder for inferred constraints.
_LEVEL_SEVERITY = {"Extreme": "high", "High": "high", "Moderate": "medium"}


def _side_or_default(side: Optional[str]) -> str:
    side = (side or "applicant").strip().lower()
    return side if side in APPLICANT_SIDES else "applicant"


def _typology_or_default(typology: Optional[str]) -> str:
    typology = (typology or "other").strip().lower()
    return typology if typology in PROJECT_TYPOLOGIES else "other"


def _permit_or_default(permit_type: Optional[str]) -> str:
    permit_type = (permit_type or "eia_screening").strip().lower()
    return permit_type if permit_type in PERMIT_TYPES else "other"


def resolve_site(site: Any) -> Dict[str, Any]:
    """Resolve a site spec to ``{"lat", "lon", "name"}`` or ``{"error": …}``.

    Accepts ``{"lat": …, "lon": …, "name": …}`` or ``{"address": "…"}``
    (geocoded via the platform's Nominatim-backed resolver).
    """
    if not isinstance(site, dict):
        return {"error": "site must be an object: {lat, lon} or {address}"}
    lat, lon = site.get("lat"), site.get("lon")
    if lat is not None and lon is not None:
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            return {"error": "site.lat and site.lon must be numbers"}
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return {"error": "site coordinates out of range"}
        name = (site.get("name") or "").strip() or f"{lat:.4f}, {lon:.4f}"
        return {"lat": lat, "lon": lon, "name": name}
    address = (site.get("address") or "").strip()
    if address:
        from ..dashboard.real_data import geocode_location

        resolved = geocode_location(address)
        if "error" in resolved:
            return {"error": resolved["error"]}
        return {
            "lat": float(resolved["lat"]),
            "lon": float(resolved["lon"]),
            "name": resolved.get("name") or address,
        }
    return {"error": "site must include lat/lon or an address"}


# ---------------------------------------------------------------------------
# Evidence layers
# ---------------------------------------------------------------------------


def _landcover_layer(lat: float, lon: float) -> Dict[str, Any]:
    """DOCUMENTED — ESA WorldCover land-cover classification for the site."""
    try:
        from ..gis_mapping.landcover import fetch_landcover

        lc = fetch_landcover(lat, lon)
    except Exception as exc:  # noqa: BLE001 — declared gap, never raised
        return {
            "evidence_label": "UNKNOWN",
            "status": "unavailable",
            "reason": f"Land-cover fetch raised {type(exc).__name__}: {exc}",
        }
    if "error" in lc:
        return {
            "evidence_label": "UNKNOWN",
            "status": "unavailable",
            "reason": lc["error"],
        }
    return {
        "evidence_label": "DOCUMENTED",
        "status": "ok",
        "dominant_label": lc.get("dominant_label"),
        "dominant_fraction": lc.get("dominant_fraction"),
        "histogram": lc.get("histogram"),
        "source": lc.get("source", "ESA WorldCover"),
        "resolution": lc.get("resolution"),
    }


def _satellite_layer(lat: float, lon: float) -> Dict[str, Any]:
    """OBSERVED — Sentinel-2 spectral evidence for the site."""
    try:
        from ..dashboard.real_data import fetch_satellite_data

        sat = fetch_satellite_data(lat, lon, days_back=60)
    except Exception as exc:  # noqa: BLE001 — declared gap, never raised
        return {
            "evidence_label": "UNKNOWN",
            "status": "unavailable",
            "reason": f"Satellite fetch raised {type(exc).__name__}: {exc}",
        }
    if "error" in sat:
        return {
            "evidence_label": "UNKNOWN",
            "status": "unavailable",
            "reason": sat["error"],
        }
    return {
        "evidence_label": "OBSERVED",
        "status": "ok",
        "ndvi": sat.get("ndvi"),
        "ndmi": sat.get("ndmi"),
        "observation_date": sat.get("observation_date"),
        "source": sat.get("source", "Sentinel-2 L2A"),
        "resolution_m": sat.get("resolution_m"),
    }


def _historical_events_layer(lat: float, lon: float, radius_km: float) -> Dict[str, Any]:
    """REPORTED — recorded environmental events at or near the site.

    Recent fire detections come from the FIRMS active-fire feed; per-hazard
    historical event layers are pulled from registered hazard modules that
    implement ``events()``. Each source fails independently into a declared
    gap.
    """
    layer: Dict[str, Any] = {"evidence_label": "REPORTED", "status": "ok"}

    try:
        from ..dashboard.real_data import fetch_active_fires

        fires = fetch_active_fires(lat, lon, radius_km=radius_km, days=10)
        if "error" in fires:
            layer["recent_fire_detections"] = {
                "status": "unavailable", "reason": fires["error"],
            }
        else:
            layer["recent_fire_detections"] = {
                "status": "ok",
                "count": fires.get("count", 0),
                "radius_km": radius_km,
                "days": 10,
                "sensor": fires.get("sensor"),
            }
    except Exception as exc:  # noqa: BLE001 — declared gap, never raised
        layer["recent_fire_detections"] = {
            "status": "unavailable",
            "reason": f"Active-fires fetch raised {type(exc).__name__}: {exc}",
        }

    hazard_events: List[Dict[str, Any]] = []
    from . import registry  # lazy: keeps import light

    for hazard_id in sorted(LICENSING_HAZARDS):
        module = registry.get(hazard_id)
        if module is None:
            continue
        try:
            available, reason = module.events_availability()
        except Exception:  # noqa: BLE001 — treat as unavailable
            available, reason = False, "events availability check failed"
        if not available:
            hazard_events.append({
                "hazard": hazard_id,
                "status": "unavailable",
                "reason": reason or "Historical events unavailable.",
                "events": [],
            })
            continue
        try:
            payload = module.events(lat, lon, radius_km=radius_km)
        except Exception as exc:  # noqa: BLE001 — declared gap, never raised
            hazard_events.append({
                "hazard": hazard_id,
                "status": "unavailable",
                "reason": f"Events fetch raised {type(exc).__name__}: {exc}",
                "events": [],
            })
            continue
        events = payload.get("events") or []
        hazard_events.append({
            "hazard": hazard_id,
            "status": payload.get("status", "ok"),
            "count": len(events),
            "events": events[:25],
            "reason": payload.get("reason"),
        })
    layer["hazard_events"] = hazard_events

    fires_ok = (layer["recent_fire_detections"] or {}).get("status") == "ok"
    events_ok = any(e.get("status") == "ok" for e in hazard_events)
    if not fires_ok and not events_ok:
        layer["status"] = "unavailable"
        layer["evidence_label"] = "UNKNOWN"
    return layer


def _hazard_exposure_layer(lat: float, lon: float, name: Optional[str],
                           hazard_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    """MODELLED — multi-hazard exposure screening via the hazard registry."""
    from . import registry  # lazy: keeps import light

    requested = [h.strip().lower() for h in (hazard_ids or []) if h and h.strip()]
    chosen = [h for h in LICENSING_HAZARDS if not requested or h in requested]

    checks: List[Dict[str, Any]] = []
    for hazard_id in chosen:
        config = LICENSING_HAZARDS[hazard_id]
        module = registry.get(hazard_id)
        if module is None:
            checks.append({
                "hazard": hazard_id,
                "permit_label": config["permit_label"],
                "evidence_label": "UNKNOWN",
                "status": "unavailable",
                "level": None,
                "summary": f"{config['permit_label']} could not be screened for this site.",
                "reason": f"{hazard_id} module is not registered in this deployment.",
                "evidence": [],
            })
            continue
        try:
            available, reason = module.availability()
        except Exception:  # noqa: BLE001 — treat as unavailable
            available, reason = False, "availability check failed"
        if not available:
            checks.append({
                "hazard": hazard_id,
                "permit_label": config["permit_label"],
                "evidence_label": "UNKNOWN",
                "status": "unavailable",
                "level": None,
                "summary": f"{config['permit_label']} could not be screened for this site.",
                "reason": reason or f"{hazard_id} analysis unavailable",
                "evidence": [],
            })
            continue
        try:
            result = module.analyze(lat, lon, name=name)
        except Exception as exc:  # noqa: BLE001 — declared gap, never raised
            checks.append({
                "hazard": hazard_id,
                "permit_label": config["permit_label"],
                "evidence_label": "UNKNOWN",
                "status": "unavailable",
                "level": None,
                "summary": f"{config['permit_label']} could not be screened for this site.",
                "reason": f"Analysis raised {type(exc).__name__}: {exc}",
                "evidence": [],
            })
            continue
        level = result.level
        checks.append({
            "hazard": hazard_id,
            "permit_label": config["permit_label"],
            "evidence_label": "MODELLED",
            "status": result.status,
            "level": level.to_dict() if level else None,
            "summary": result.summary,
            "reason": result.unavailable_reason,
            "evidence": result.evidence,
        })
    return checks


# ---------------------------------------------------------------------------
# Inferred constraints & risk flags
# ---------------------------------------------------------------------------


def _derive_constraints(
    hazard_checks: List[Dict[str, Any]],
    landcover: Dict[str, Any],
    events_layer: Dict[str, Any],
    typology: str,
) -> List[Dict[str, Any]]:
    """INFERRED — site constraints & risk flags derived from the evidence base.

    Rules are deterministic and transparent: every flag states its basis and
    the evidence it was derived from. No flag is produced without evidence.
    """
    constraints: List[Dict[str, Any]] = []

    for check in hazard_checks:
        level = check.get("level") or {}
        label = level.get("label")
        severity = _LEVEL_SEVERITY.get(label)
        if not severity:
            continue
        constraints.append({
            "id": f"elevated_{check['hazard']}_exposure",
            "severity": severity,
            "title": f"Elevated {check['permit_label'].lower()}",
            "basis": (
                f"{check['permit_label']} screened at level '{label}' "
                f"({level.get('basis') or 'screening indicator'}). Permit "
                "reviewers typically expect this exposure to be addressed in "
                "the application evidence."
            ),
            "derived_from": ["hazard_exposure"],
            "evidence_label": "INFERRED",
        })

    dominant = str(landcover.get("dominant_label") or "").lower()
    if landcover.get("status") == "ok" and (
        "tree" in dominant or "forest" in dominant
    ):
        constraints.append({
            "id": "vegetation_clearance_likely",
            "severity": "medium",
            "title": "Vegetation / land-cover change likely material",
            "basis": (
                f"Dominant land cover at the site is "
                f"'{landcover.get('dominant_label')}' "
                f"(fraction {landcover.get('dominant_fraction')}). "
                "Development on vegetated land typically triggers land-cover "
                "change, clearance and habitat-screening questions in the "
                "permit procedure."
            ),
            "derived_from": ["landcover"],
            "evidence_label": "INFERRED",
        })

    fires = (events_layer or {}).get("recent_fire_detections") or {}
    if fires.get("status") == "ok" and (fires.get("count") or 0) > 0:
        constraints.append({
            "id": "recent_fire_activity",
            "severity": "medium",
            "title": "Recent fire activity recorded near the site",
            "basis": (
                f"{fires['count']} active-fire detection(s) within "
                f"{fires.get('radius_km')} km over {fires.get('days')} days "
                f"(sensor: {fires.get('sensor') or 'see evidence'}). Fire "
                "history is typically material to site safety and "
                "environmental permit conditions."
            ),
            "derived_from": ["historical_events"],
            "evidence_label": "INFERRED",
        })

    if typology == "solar_pv":
        heat = next((c for c in hazard_checks if c["hazard"] == "heat"), None)
        if heat and (heat.get("level") or {}).get("label") in _LEVEL_SEVERITY:
            constraints.append({
                "id": "solar_heat_derating",
                "severity": "medium",
                "title": "Heat exposure material to solar yield assumptions",
                "basis": (
                    "Extreme-heat screening is elevated at the site; panel "
                    "derating and cooling assumptions are typically examined "
                    "in solar permit and financing reviews."
                ),
                "derived_from": ["hazard_exposure"],
                "evidence_label": "INFERRED",
            })

    return constraints


# ---------------------------------------------------------------------------
# Dossier assembly
# ---------------------------------------------------------------------------


def _declared_gaps(
    landcover: Dict[str, Any],
    satellite: Dict[str, Any],
    events_layer: Dict[str, Any],
    hazard_checks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """UNKNOWN — every missing or unsuitable layer, declared with a reason."""
    gaps: List[Dict[str, str]] = []
    if landcover.get("status") != "ok":
        gaps.append({"layer": "landcover", "reason": landcover.get("reason") or "unavailable"})
    if satellite.get("status") != "ok":
        gaps.append({"layer": "satellite", "reason": satellite.get("reason") or "unavailable"})
    fires = (events_layer or {}).get("recent_fire_detections") or {}
    if fires.get("status") == "unavailable":
        gaps.append({"layer": "recent_fire_detections", "reason": fires.get("reason") or "unavailable"})
    for event_block in (events_layer or {}).get("hazard_events") or []:
        if event_block.get("status") == "unavailable":
            gaps.append({
                "layer": f"historical_events:{event_block.get('hazard')}",
                "reason": event_block.get("reason") or "unavailable",
            })
    for check in hazard_checks:
        if check.get("status") == "unavailable":
            gaps.append({
                "layer": f"hazard_exposure:{check.get('hazard')}",
                "reason": check.get("reason") or "unavailable",
            })
    return gaps


def build_licensing_dossier(
    site: Any = None,
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    name: Optional[str] = None,
    radius_km: float = 25.0,
    side: Optional[str] = None,
    typology: Optional[str] = None,
    permit_type: Optional[str] = None,
    project_title: Optional[str] = None,
    description: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    hazards: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a pre-draft environmental licensing evidence dossier.

    The dossier is location-first: ``site`` (``{lat, lon}`` / ``{address}``)
    or explicit ``lat``/``lon`` define the analysed point; the project
    metadata (side, typology, permit type, description) frame the evidence
    without altering it — the same base serves applicants and authorities.

    Returns either ``{"error": …}`` for an unresolvable request, or the full
    dossier dict with labelled evidence layers, inferred constraints,
    framework references, declared gaps, a stable ``dossier_id``, and the
    disclaimer. Layer failures degrade to declared gaps — never raised.
    """
    if site is None and lat is not None and lon is not None:
        site = {"lat": lat, "lon": lon, "name": name}
    resolved = resolve_site(site)
    if "error" in resolved:
        return {"error": resolved["error"]}

    lat_f, lon_f = resolved["lat"], resolved["lon"]
    name_s = (name or resolved.get("name") or f"{lat_f:.4f}, {lon_f:.4f}").strip()
    try:
        radius_f = float(radius_km)
    except (TypeError, ValueError):
        return {"error": "radius_km must be a number"}
    if not (1.0 <= radius_f <= 200.0):
        return {"error": "radius_km must be between 1 and 200"}

    side_id = _side_or_default(side)
    typology_id = _typology_or_default(typology)
    permit_id = _permit_or_default(permit_type)

    landcover = _landcover_layer(lat_f, lon_f)
    satellite = _satellite_layer(lat_f, lon_f)
    events_layer = _historical_events_layer(lat_f, lon_f, radius_f)
    hazard_checks = _hazard_exposure_layer(lat_f, lon_f, name_s, hazards)
    constraints = _derive_constraints(hazard_checks, landcover, events_layer, typology_id)
    gaps = _declared_gaps(landcover, satellite, events_layer, hazard_checks)

    assessed = [c for c in hazard_checks if c["status"] != "unavailable"]
    elevated = [c for c in constraints if c["severity"] in ("high", "medium")]
    ok_layers = sum(
        1 for layer in (landcover, satellite, events_layer)
        if layer.get("status") == "ok"
    )

    summary = (
        f"{len(assessed)} of {len(hazard_checks)} hazards screened with real data, "
        f"{ok_layers} of 3 context evidence layers available, "
        f"{len(constraints)} inferred constraint{'s' if len(constraints) != 1 else ''}"
    )
    if gaps:
        summary += f", {len(gaps)} declared data gap{'s' if len(gaps) != 1 else ''}"
    summary += "."

    request_block: Dict[str, Any] = {
        "side": side_id,
        "side_label": APPLICANT_SIDES[side_id]["label"],
        "framing": APPLICANT_SIDES[side_id]["framing"],
        "typology": typology_id,
        "typology_label": PROJECT_TYPOLOGIES[typology_id]["label"],
        "typology_note": PROJECT_TYPOLOGIES[typology_id]["note"],
        "permit_type": permit_id,
        "permit_type_label": PERMIT_TYPES[permit_id]["label"],
        "permit_type_note": PERMIT_TYPES[permit_id]["note"],
    }
    if project_title:
        request_block["project_title"] = project_title.strip()[:200]
    if description:
        request_block["description"] = description.strip()[:2000]
    if jurisdiction:
        request_block["jurisdiction"] = jurisdiction.strip()[:120]

    dossier_id = content_hash({
        "site": {"lat": round(lat_f, 6), "lon": round(lon_f, 6)},
        "radius_km": radius_f,
        "side": side_id,
        "typology": typology_id,
        "permit_type": permit_id,
        "hazards": [c["hazard"] for c in hazard_checks],
        "engine_version": ENGINE_VERSION,
    })[:16]

    return {
        "dossier_id": dossier_id,
        "generated_at": utcnow_iso(),
        "engine_version": ENGINE_VERSION,
        "request": request_block,
        "site": {
            "lat": lat_f,
            "lon": lon_f,
            "name": name_s,
            "radius_km": radius_f,
        },
        "evidence_base": {
            "landcover": landcover,
            "satellite": satellite,
            "historical_events": events_layer,
            "hazard_exposure": hazard_checks,
        },
        "constraints": constraints,
        "elevated_constraint_count": len(elevated),
        "frameworks": LICENSING_FRAMEWORKS,
        "declared_gaps": gaps,
        "summary": summary,
        "disclaimer": DISCLAIMER,
        "honesty_contract": HONESTY_CONTRACT,
    }
