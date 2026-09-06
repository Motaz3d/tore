"""
Relative Ignition-Likelihood Indicator (RILI) — screening-level, unvalidated.

Talaix keeps three concepts strictly separate:

    A. WILDFIRE HAZARD ........ the composite fire-danger/risk products
                                (FWI, danger classes, 0-100 risk score) —
                                "how dangerous are the conditions?"
    B. IGNITION LIKELIHOOD ... this module — "where do the conditions and
                                human presence coincide so that an ignition
                                is relatively more likely than elsewhere?"
    C. OBSERVED FIRE .......... NASA FIRMS detections — "a fire was seen".

    HIGH FIRE DANGER ≠ FIRE WILL OCCUR
    HIGH IGNITION SUSCEPTIBILITY ≠ OBSERVED FIRE

Scientific honesty rules (do not weaken):
    - This is a RELATIVE indicator (0-100), built from declared a-priori
      weights over declared threshold functions. The weights are NOT fitted
      to observations and the output is NOT a calibrated probability.
    - It is NOT VALIDATED. ``validation_status.validated`` stays False until
      a real historical evaluation (``scripts/evaluate_ignition.py``,
      temporal split vs NASA FIRMS detections) has been executed and its
      ValidationReport exists. See docs/VALIDATION.md.
    - Natural ignition sources (lightning) are NOT included: no openly and
      legally usable lightning data source passed the source audit
      (config/source_registry.json). This is a declared gap.

Predictors (all real, all already integrated for other purposes):
    - Fire weather ......... FFMC from the Canadian FWI System computed on
                             real Open-Meteo daily data (FFMC is the FWI
                             System's fine-fuel/ignition-relevant index).
    - Human presence ....... WorldPop estimated population density + mapped
                             OSM road network (most ignitions in Europe are
                             human-caused; settlement/road proximity are
                             standard predictors in the ignition literature).
    - Fuel dryness ......... Sentinel-2 NDMI-derived or soil-moisture-derived
                             FMC (real, from the analysis), NDMI fallback.

Each component contributes through a declared piecewise threshold function;
component weights are declared constants (weather 0.5, human 0.3, fuel 0.2),
renormalised over the components that are actually available (coverage is
reported). When no component is available the indicator is unavailable —
never fabricated.
"""

from __future__ import annotations

from typing import Dict, Optional

from . import exposure as exposure_module
from . import population as population_module

INDICATOR_NAME = "Relative Ignition-Likelihood Indicator"
MODEL_VERSION = "rili-1.0.0"  # bump on any weight/threshold change (provenance)

#: Declared component weights (a priori, NOT fitted to observations).
WEIGHTS = {"fire_weather": 0.5, "human_presence": 0.3, "fuel_dryness": 0.2}

#: Declared indicator classes (screening communication only).
CLASSES = [(25.0, "low"), (45.0, "moderate"), (65.0, "elevated"), (101.0, "high")]

NOT_A_PROBABILITY_NOTE = (
    "This is a relative screening indicator built from declared thresholds and "
    "a-priori weights. It is NOT a calibrated probability of ignition and must "
    "not be quoted as one."
)

DISTINCTIONS = [
    "HIGH FIRE DANGER ≠ FIRE WILL OCCUR — dangerous conditions do not cause ignitions by themselves.",
    "HIGH IGNITION SUSCEPTIBILITY ≠ OBSERVED FIRE — the indicator ranks relative likelihood, not occurrence.",
    "Wildfire hazard, ignition likelihood and observed fires are reported separately and never merged.",
]

VALIDATION_STATUS = {
    "validated": False,
    "status": "NOT VALIDATED — no historical evaluation has been executed yet",
    "method_when_run": (
        "Temporal train/test split against NASA FIRMS historical detections with "
        "positive/negative sampling, class-imbalance handling, precision/recall/F1, "
        "ROC-AUC, PR-AUC, Brier score, calibration and reliability analysis "
        "(scripts/evaluate_ignition.py; framework: src/prediction/validation.py)."
    ),
    "promotion_rule": "The indicator is never promoted from a single event; evaluation requires a multi-day historical sample.",
}


# ---------------------------------------------------------------------------
# Declared component threshold functions (pure; individually testable)
# ---------------------------------------------------------------------------

def _ffmc_score(ffmc: float) -> float:
    """
    Fire-weather sub-score from FFMC (0-101 scale).

    Declared screening anchors on the FFMC scale (fine fuels cure and ignite
    more readily as FFMC rises; the steepest practical increase in ignition
    ease is in the mid-80s and above). Not a fitted relationship.
    """
    if ffmc < 70.0:
        return 10.0
    if ffmc < 76.0:
        return 25.0
    if ffmc < 82.0:
        return 45.0
    if ffmc < 86.0:
        return 60.0
    if ffmc < 90.0:
        return 80.0
    return 95.0


def _density_score(density_per_km2: float) -> float:
    """Human-presence sub-score from estimated population density."""
    if density_per_km2 < 1.0:
        return 5.0
    if density_per_km2 < 25.0:
        return 15.0
    if density_per_km2 < 100.0:
        return 35.0
    if density_per_km2 < 500.0:
        return 60.0
    if density_per_km2 < 2500.0:
        return 80.0
    return 95.0


def _road_score(roads_mapped: float) -> float:
    """Human-presence sub-score from the mapped road network (access)."""
    if roads_mapped < 1:
        return 5.0
    if roads_mapped < 10:
        return 25.0
    if roads_mapped < 40:
        return 50.0
    if roads_mapped < 120:
        return 75.0
    return 90.0


def _fmc_score(fmc_pct: float) -> float:
    """Fuel-dryness sub-score from fuel moisture content (%)."""
    if fmc_pct < 8.0:
        return 95.0
    if fmc_pct < 12.0:
        return 75.0
    if fmc_pct < 18.0:
        return 55.0
    if fmc_pct < 25.0:
        return 35.0
    if fmc_pct < 35.0:
        return 20.0
    return 10.0


def _ndmi_score(ndmi: float) -> float:
    """Fuel-dryness sub-score from NDMI (-1..1; higher = wetter canopy)."""
    if ndmi < 0.0:
        return 85.0
    if ndmi < 0.1:
        return 65.0
    if ndmi < 0.2:
        return 45.0
    if ndmi < 0.3:
        return 30.0
    return 15.0


def _class(score: float) -> str:
    for threshold, label in CLASSES:
        if score < threshold:
            return label
    return "high"


# ---------------------------------------------------------------------------
# Indicator assembly
# ---------------------------------------------------------------------------

def indicator_from_components(
    ffmc: Optional[float] = None,
    population_density_per_km2: Optional[float] = None,
    roads_mapped: Optional[float] = None,
    fmc_pct: Optional[float] = None,
    ndmi: Optional[float] = None,
    burnable: bool = True,
) -> Dict:
    """
    Pure indicator computation from component inputs (all optional).

    Returns the indicator, class, per-component detail and input coverage.
    Components with no real input are dropped and the declared weights are
    renormalised over what remains; with no components at all the indicator
    is ``None`` (unavailable) — nothing is invented.
    """
    components: Dict[str, Dict] = {}

    if ffmc is not None:
        components["fire_weather"] = {
            "score": round(_ffmc_score(float(ffmc)), 1),
            "weight": WEIGHTS["fire_weather"],
            "inputs": {"ffmc": round(float(ffmc), 1)},
            "basis": "FFMC from the Canadian FWI System (real Open-Meteo daily data)",
        }

    human_parts = []
    human_inputs: Dict[str, float] = {}
    if population_density_per_km2 is not None:
        human_parts.append((0.6, _density_score(float(population_density_per_km2))))
        human_inputs["population_density_per_km2"] = round(float(population_density_per_km2), 1)
    if roads_mapped is not None:
        human_parts.append((0.4, _road_score(float(roads_mapped))))
        human_inputs["roads_mapped_within_2km"] = int(roads_mapped)
    if human_parts:
        wsum = sum(w for w, _ in human_parts)
        human = sum(w * s for w, s in human_parts) / wsum
        components["human_presence"] = {
            "score": round(human, 1),
            "weight": WEIGHTS["human_presence"],
            "inputs": human_inputs,
            "basis": "WorldPop estimated density + mapped OSM roads "
                     "(human activity is the dominant ignition cause in Europe)",
        }

    if fmc_pct is not None:
        components["fuel_dryness"] = {
            "score": round(_fmc_score(float(fmc_pct)), 1),
            "weight": WEIGHTS["fuel_dryness"],
            "inputs": {"fmc_pct": round(float(fmc_pct), 1)},
            "basis": "Fuel moisture content (Sentinel-2 NDMI-derived or soil-moisture-derived)",
        }
    elif ndmi is not None:
        components["fuel_dryness"] = {
            "score": round(_ndmi_score(float(ndmi)), 1),
            "weight": WEIGHTS["fuel_dryness"],
            "inputs": {"ndmi": round(float(ndmi), 3)},
            "basis": "Sentinel-2 NDMI (canopy moisture proxy)",
        }

    if not components:
        return {
            "indicator": None,
            "class": None,
            "components": {},
            "input_coverage": [],
            "coverage_note": "No real component inputs available — indicator not computed.",
        }

    total_weight = sum(c["weight"] for c in components.values())
    score = sum(c["score"] * c["weight"] for c in components.values()) / total_weight
    score = round(max(0.0, min(100.0, score)), 1)

    coverage = list(components.keys())
    coverage_note = None
    if len(coverage) < len(WEIGHTS):
        missing = sorted(set(WEIGHTS) - set(coverage))
        coverage_note = (
            "Reduced input coverage: missing component(s) "
            + ", ".join(missing)
            + " — declared weights renormalised over available components."
        )

    out = {
        "indicator": score,
        "class": _class(score),
        "components": components,
        "input_coverage": coverage,
        "coverage_note": coverage_note,
    }
    if not burnable:
        out["landcover_note"] = (
            "Dominant land cover is largely non-burnable; ignition likelihood here "
            "mostly concerns human-started fires in built environments."
        )
    return out


def build_ignition_block(analysis: Dict) -> Dict:
    """
    Ignition-likelihood block for an analysis result.

    Gathers the real component inputs from the analysis (FWI System outputs,
    fuel moisture / NDMI, land cover) plus the cached WorldPop density and
    OSM road counts, then computes the relative indicator. Honest
    ``status: unavailable`` when no component can be computed.
    """
    loc = analysis.get("location") or {}
    lat, lon = loc.get("latitude"), loc.get("longitude")
    if lat is None or lon is None:
        return {"status": "unavailable", "reason": "No analysis location"}

    fire_danger = analysis.get("fire_danger") or {}
    ffmc = fire_danger.get("ffmc") if fire_danger.get("available") else None

    inner = analysis.get("analysis") or {}
    fmc = inner.get("fuel_moisture_baseline_pct")
    satellite = analysis.get("satellite") or {}
    ndmi = satellite.get("ndmi") if "error" not in satellite else None
    landcover = analysis.get("landcover") or {}
    burnable = landcover.get("burnable", True) if "error" not in landcover else True

    pop = population_module.fetch_population(float(lat), float(lon))
    density = pop.get("mean_density_per_km2") if "error" not in pop else None

    osm = exposure_module.fetch_osm_context(float(lat), float(lon))
    roads = (osm.get("counts") or {}).get("roads_all") if "error" not in osm else None

    result = indicator_from_components(
        ffmc=ffmc,
        population_density_per_km2=density,
        roads_mapped=roads,
        fmc_pct=fmc,
        ndmi=ndmi,
        burnable=burnable,
    )
    if result["indicator"] is None:
        return {
            "status": "unavailable",
            "reason": result["coverage_note"],
            "name": INDICATOR_NAME,
            "not_a_probability": NOT_A_PROBABILITY_NOTE,
            "validation_status": VALIDATION_STATUS,
            "provenance": {
                "kind": "unavailable",
                "source": "Talaix ignition layer (FFI/FFMC + WorldPop + OSM)",
                "quality": "missing",
                "limitations": result["coverage_note"],
            },
        }

    return {
        "status": "ok",
        "name": INDICATOR_NAME,
        "model_version": MODEL_VERSION,
        "indicator": result["indicator"],
        "class": result["class"],
        "components": result["components"],
        "weights": dict(WEIGHTS),
        "input_coverage": result["input_coverage"],
        "coverage_note": result.get("coverage_note"),
        "landcover_note": result.get("landcover_note"),
        "not_a_probability": NOT_A_PROBABILITY_NOTE,
        "distinctions": DISTINCTIONS,
        "lightning_note": (
            "Natural ignition sources (lightning) are not included: no openly and "
            "legally usable lightning dataset passed the source audit."
        ),
        "separate_from_score_note": (
            "Ignition likelihood is reported separately from the composite "
            "wildfire-risk score and is never folded into it."
        ),
        "validation_status": VALIDATION_STATUS,
        "provenance": {
            "kind": "derived",
            "source": (
                "Talaix ignition layer (declared thresholds: FWI-System FFMC + "
                "WorldPop density + OSM roads + fuel dryness)"
            ),
            "resolution": "analysis-area screening (population 100 m; weather ~11 km)",
            "temporal": "current conditions",
            "quality": "ok" if not result.get("coverage_note") else "degraded",
            "limitations": (
                "Unvalidated relative indicator with a-priori weights; not a "
                "probability; lightning ignitions not modelled."
            ),
        },
    }
