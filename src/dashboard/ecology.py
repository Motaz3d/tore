"""
Environmental Solutions / Ecological Restoration layer.

Recommends vegetation and restoration approaches for a location from its
REAL detected conditions (climate signal, moisture regime, elevation,
slope, land cover, fire danger) matched against a curated, sourced species
knowledge base (``config/species_knowledge.json`` — reference knowledge,
not an observation).

Honesty contract:
    - The species knowledge base is literature-level reference data; every
      entry carries sources and a confidence flag.
    - Suitability output always states the real site values that drove it.
    - No species is "fireproof"; fire notes describe documented ecology
      (resprouting, bark insulation, flammability) with limits.
    - When the real inputs are insufficient, the block says so with the
      exact sentence:
      "Local ecological suitability could not be established from the
      available data."
    - Every recommendation is flagged for local expert verification.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

_DEFAULT_KB = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "species_knowledge.json"
)

INSUFFICIENT_DATA_MESSAGE = (
    "Local ecological suitability could not be established from the available data."
)

VERIFICATION_NOTE = (
    "Suitability is derived from regional ecological literature matched to "
    "the detected site conditions; it is not a substitute for a local "
    "ecological assessment. Verify species choice with local forestry / "
    "conservation experts before planting."
)


def load_species_knowledge(path: Optional[str] = None) -> Dict:
    kb_path = path or os.environ.get("HYDRASHIELD_SPECIES_KB") or _DEFAULT_KB
    with open(kb_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def classify_climate_zone(climate: Dict, elevation_m: Optional[float]) -> Optional[str]:
    """
    Classify the site climate signal from real recent data.

    "mediterranean" = hot, dry recent window (declared screening
    approximation from the last ~3 weeks of real daily aggregates);
    "temperate" otherwise; "mountain" as an elevation constraint handled
    separately. Returns None when there is no usable signal.
    """
    t_mean = climate.get("mean_temp_max_c")
    rain = climate.get("total_precip_mm")
    if t_mean is None and rain is None:
        return None
    hot = t_mean is not None and t_mean >= 26.0
    dry = rain is not None and rain <= 10.0
    if hot and dry:
        return "mediterranean"
    if t_mean is None or rain is None:
        return None
    return "temperate"


def classify_moisture_regime(
    fmc: Optional[float],
    soil_moisture: Optional[float],
    recent_rain_mm: Optional[float],
) -> Optional[str]:
    """dry / normal / moist from real moisture indicators (declared thresholds)."""
    signals = []
    if fmc is not None:
        signals.append("dry" if fmc < 18.0 else "moist" if fmc > 30.0 else "normal")
    if soil_moisture is not None:
        signals.append("dry" if soil_moisture < 0.20 else "moist" if soil_moisture > 0.35 else "normal")
    if recent_rain_mm is not None:
        signals.append("dry" if recent_rain_mm <= 5.0 else "moist" if recent_rain_mm > 30.0 else "normal")
    if not signals:
        return None
    if "dry" in signals:
        return "dry"
    if signals.count("moist") >= 2:
        return "moist"
    return "normal"


_DROUGHT_OK = {"high", "moderate-high"}
_REC_ORDER = {"recommended": 0, "recommended_with_caution": 1,
              "not_recommended_in_protection_zones": 2, "not_recommended": 3}


def _species_fit(
    sp: Dict,
    zone: Optional[str],
    mountain: bool,
    moisture: Optional[str],
    elevation_m: Optional[float],
) -> Optional[Dict]:
    """
    Evaluate one species against the real site conditions.

    Returns None when the species clearly does not fit (wrong region or
    above its elevation limit); otherwise a fit record with the concrete
    reasons for/against, quoting the real site values.
    """
    reasons_for: List[str] = []
    reasons_against: List[str] = []

    if zone is not None:
        if zone not in sp.get("regions", []):
            return None
        if zone in sp.get("native_in", []):
            reasons_for.append(f"native to the detected {zone} conditions")
        else:
            reasons_against.append(f"not native to the detected {zone} conditions")
    elif sp.get("native_in"):
        # No climate signal: only surface species flagged native somewhere,
        # with reduced confidence.
        reasons_against.append("no climate signal available — regional fit unverified")

    if elevation_m is not None:
        max_e = sp.get("max_elevation_m")
        if max_e is not None and elevation_m > max_e:
            return None
        if max_e is not None:
            reasons_for.append(f"site elevation {elevation_m:.0f} m within range (<= {max_e} m)")
    if mountain and "mountain" not in sp.get("regions", []):
        return None

    if moisture == "dry":
        if sp.get("drought_tolerance") in _DROUGHT_OK:
            reasons_for.append("drought-tolerant under the detected dry regime")
        else:
            if sp.get("recommendation", "").startswith("recommended"):
                reasons_against.append("limited drought tolerance under the detected dry regime")

    return {"reasons_for": reasons_for, "reasons_against": reasons_against}


def build_ecology_block(
    analysis: Dict,
    knowledge_path: Optional[str] = None,
) -> Dict:
    """Build the Environmental Solutions block from a real analysis result."""
    terrain = analysis.get("terrain") or {}
    landcover = analysis.get("landcover") or {}
    weather = analysis.get("weather") or {}
    climate = analysis.get("climate") or {}
    a = analysis.get("analysis") or {}
    fires = analysis.get("active_fires") or {}

    elevation = terrain.get("elevation_m") if "error" not in terrain else None
    slope = terrain.get("slope_degrees") if "error" not in terrain else None
    aspect = terrain.get("aspect_degrees") if "error" not in terrain else None
    fmc = a.get("fuel_moisture_baseline_pct")
    soil_moisture = weather.get("soil_moisture_m3m3")
    recent_rain = climate.get("total_precip_mm")
    landcover_label = landcover.get("dominant_label") if "error" not in landcover else None

    usable = [v is not None for v in (elevation, fmc, soil_moisture, recent_rain, landcover_label)]
    if not any(usable):
        return {
            "status": "insufficient_data",
            "message": INSUFFICIENT_DATA_MESSAGE,
            "provenance": {
                "kind": "unavailable",
                "source": "Talaix ecology engine",
                "quality": "missing",
                "limitations": "No terrain, moisture, climate or land-cover "
                               "inputs were available.",
            },
        }

    zone = classify_climate_zone(climate, elevation)
    mountain = elevation is not None and elevation >= 1500.0
    moisture = classify_moisture_regime(fmc, soil_moisture, recent_rain)

    try:
        kb = load_species_knowledge(knowledge_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "insufficient_data",
            "message": f"Species knowledge base unavailable: {exc}",
            "provenance": {"kind": "unavailable", "source": "species knowledge base",
                           "quality": "missing"},
        }

    site = {
        "climate_zone": zone,
        "mountain": mountain,
        "moisture_regime": moisture,
        "elevation_m": elevation,
        "slope_degrees": slope,
        "aspect_degrees": aspect,
        "fuel_moisture_pct": fmc,
        "soil_moisture_m3m3": soil_moisture,
        "recent_precip_mm": recent_rain,
        "land_cover": landcover_label,
        "active_fires_nearby": fires.get("count") if fires.get("available") else None,
    }

    recommended, caution, not_recommended = [], [], []
    for sp in kb.get("species") or []:
        fit = _species_fit(sp, zone, mountain, moisture, elevation)
        if fit is None:
            continue
        entry = {
            "common_name": sp["common_name"],
            "scientific_name": sp["scientific_name"],
            "native": bool(zone and zone in sp.get("native_in", [])),
            "drought_tolerance": sp.get("drought_tolerance"),
            "water_requirement": sp.get("water_requirement"),
            "moisture_tolerance": sp.get("moisture_tolerance"),
            "environmental_role": sp.get("environmental_role"),
            "fire_considerations": sp.get("fire_considerations"),
            "post_fire_recovery": sp.get("post_fire_recovery"),
            "management": sp.get("management"),
            "recommendation": sp.get("recommendation"),
            "site_fit": fit,
            "evidence": sp.get("sources") or [],
            "confidence": sp.get("confidence"),
        }
        rec = sp.get("recommendation")
        if rec == "recommended":
            recommended.append(entry)
        elif rec == "recommended_with_caution":
            caution.append(entry)
        else:
            not_recommended.append(entry)

    recommended.sort(key=lambda e: (not e["native"], e["common_name"]))
    not_recommended.sort(key=lambda e: _REC_ORDER.get(e["recommendation"], 4))

    return {
        "status": "ok",
        "site_conditions": site,
        "recommended": recommended[:6],
        "recommended_with_caution": caution[:4],
        "not_recommended": not_recommended[:4],
        "fire_note": "No vegetation is fireproof. Fire notes describe documented "
                     "ecology (bark insulation, resprouting, flammability); any "
                     "vegetation burns under sufficiently severe conditions.",
        "verification_note": VERIFICATION_NOTE,
        "provenance": {
            "kind": "derived",
            "source": "Talaix ecology engine: real site conditions + curated "
                      "species knowledge base (config/species_knowledge.json)",
            "quality": "literature-level suitability; local verification required",
            "limitations": "Climate zone is a screening classification from the "
                           "recent real weather window; soils are not measured; "
                           "local native-species lists are not exhaustive.",
        },
    }
