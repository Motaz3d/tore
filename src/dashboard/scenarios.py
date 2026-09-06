"""
Intervention scenario framework.

Compares CURRENT CONDITION vs INTERVENTION — but only where an actual
Talaix model can compute the effect. Everything else is explicitly
reported as "not_quantified" (no invented percentage improvements).

Model-supported scenarios:
    - hydration        (FMC increase -> spread model + risk score)
    - fuel management  (fuel-model shift + modest FMC change -> same models)
    - combined         (both)

Non-quantified interventions (monitoring, water preparedness, ecological
restoration, access works) are listed with their real mechanism but no
invented effect size.

Every modelled scenario states: baseline, intervention, changed inputs,
model assumptions, calculated result, uncertainty note, limitations.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..prediction.fire_spread import FireSpreadModel

MODELLED_LABEL = "MODELLED INTERVENTION SCENARIO — not an observed result"


def _ros(fuel_model: str, fmc: float, wind: float, slope: float) -> Optional[float]:
    try:
        ros = FireSpreadModel(fuel_model=fuel_model).compute_ros(
            fmc=fmc, wind_speed_kmh=wind, slope_degrees=slope
        )
        return round(ros.ros_reduced, 3)
    except Exception:
        return None


def _risk(fwi, slope, fmc, burnable) -> Optional[float]:
    from .real_analysis import TalaixRealAnalyser  # deferred: avoids circular import

    return TalaixRealAnalyser._risk_score(
        fwi=fwi, slope=slope, fmc=fmc, wind_kmh=0.0, burnable=burnable
    )


def _modelled_scenario(
    sid: str,
    name: str,
    description: str,
    changed_inputs: Dict,
    baseline: Dict,
    new_fmc: float,
    new_fuel_model: str,
    fwi, slope, wind, burnable,
    assumptions: List[str],
    limitations: List[str],
) -> Dict:
    new_risk = _risk(fwi, slope, new_fmc, burnable)
    new_ros = _ros(new_fuel_model, new_fmc, wind, slope)
    base_risk = baseline.get("risk")
    base_ros = baseline.get("ros")
    return {
        "id": sid,
        "name": name,
        "status": "modelled",
        "label": MODELLED_LABEL,
        "intervention": description,
        "changed_inputs": changed_inputs,
        "baseline": {"risk": base_risk, "ros_m_min": base_ros},
        "result": {
            "risk": new_risk,
            "ros_m_min": new_ros,
            "risk_delta": (round(new_risk - base_risk, 1)
                           if new_risk is not None and base_risk is not None else None),
            "ros_delta_m_min": (round(new_ros - base_ros, 3)
                                if new_ros is not None and base_ros is not None else None),
        },
        "model": "Talaix FireSpreadModel + FWI-anchored composite risk score",
        "assumptions": assumptions,
        "uncertainty": "Screening-level point estimate; no formal uncertainty "
                       "bounds are computed. Treat deltas as directional, not exact.",
        "limitations": limitations,
    }


def build_scenarios(analysis: Dict) -> List[Dict]:
    """Build intervention scenarios from the real baseline analysis."""
    a = analysis.get("analysis") or {}
    fire_danger = analysis.get("fire_danger") or {}
    landcover = analysis.get("landcover") or {}
    terrain = analysis.get("terrain") or {}
    weather = analysis.get("weather") or {}

    fwi = fire_danger.get("fwi") if fire_danger.get("available") else None
    fmc = a.get("fuel_moisture_baseline_pct")
    wind = weather.get("wind_speed_kmh") or 0.0
    slope = terrain.get("slope_degrees") or 0.0
    burnable = landcover.get("burnable", True) if "error" not in landcover else True
    fuel_model = (a.get("fire_spread") or {}).get("fuel_model") or "TL3"
    base_risk = (a.get("risk") or {}).get("baseline")
    base_ros = (a.get("fire_spread") or {}).get("ros_current_m_min")

    baseline = {"risk": base_risk, "ros": base_ros}
    scenarios: List[Dict] = []

    if fmc is not None and base_risk is not None:
        hydration_fmc = min(fmc + 20.0, 100.0)
        scenarios.append(_modelled_scenario(
            "hydration",
            "Subsurface hydration (Talaix barrier)",
            "Raise fuel moisture content by 20 percentage points via "
            "subsurface hydration of protection zones.",
            {"fuel_moisture_pct": [fmc, hydration_fmc]},
            baseline, hydration_fmc, fuel_model, fwi, slope, wind, burnable,
            assumptions=[
                "FMC increase is uniform across the protected fuel bed (declared "
                "modelling assumption).",
                "Weather, FWI, terrain and land cover unchanged.",
            ],
            limitations=[
                "Does not model spotting, suppression or fuel breaks.",
                "Effect on ignition likelihood is not modelled.",
            ],
        ))

        fuel_fmc = min(fmc + 10.0, 100.0)
        new_fuel = "TL1" if fuel_model not in ("TL1",) else "NB1"
        scenarios.append(_modelled_scenario(
            "fuel-management",
            "Fuel reduction / vegetation management",
            "Reduce fine-fuel load and continuity (thinning, removal of dry "
            "hazardous fuel) — modelled as a shift to a low-load fuel class "
            "plus a modest moisture increase.",
            {"fuel_model": [fuel_model, new_fuel],
             "fuel_moisture_pct": [fmc, fuel_fmc]},
            baseline, fuel_fmc, new_fuel, fwi, slope, wind, burnable,
            assumptions=[
                f"Fuel treatment modelled as {fuel_model} -> {new_fuel} fuel-class "
                "shift and +10 %-pts FMC (screening approximation, not a "
                "prescription).",
                "Weather, FWI, terrain and land cover unchanged.",
            ],
            limitations=[
                "Real treatment effect depends on treated area, technique and "
                "maintenance — none are modelled spatially.",
            ],
        ))

        combined_fmc = min(fmc + 30.0, 100.0)
        scenarios.append(_modelled_scenario(
            "combined",
            "Combined: fuel management + hydration",
            "Both interventions combined (additive FMC assumption).",
            {"fuel_model": [fuel_model, new_fuel],
             "fuel_moisture_pct": [fmc, combined_fmc]},
            baseline, combined_fmc, new_fuel, fwi, slope, wind, burnable,
            assumptions=[
                "Effects assumed additive (declared); interactions are not modelled.",
            ],
            limitations=["Same model limitations as the individual scenarios."],
        ))
    else:
        scenarios.append({
            "id": "hydration",
            "name": "Subsurface hydration (Talaix barrier)",
            "status": "not_quantified",
            "reason": "No real fuel-moisture baseline available — the effect "
                      "cannot be computed without fabricating inputs.",
        })

    # ---- Interventions the current models cannot quantify --------------
    for sid, name, mechanism in [
        ("monitoring", "Increased monitoring / earlier detection",
         "Shortens detection and response time; does not change physical "
         "fire behaviour in the models."),
        ("water-preparedness", "Water-resource preparedness",
         "Improves suppression logistics; effect on spread is not modelled."),
        ("ecological-restoration", "Ecological restoration / fuel-structure conversion",
         "Long-term (years) change of fuel structure and landscape moisture; "
         "outside the time horizon of the screening models."),
        ("access-works", "Access-route improvement",
         "Improves response access; no fire-behaviour effect in the models."),
    ]:
        scenarios.append({
            "id": sid,
            "name": name,
            "status": "not_quantified",
            "mechanism": mechanism,
            "note": "Recommended where the evidence supports it (see proactive "
                    "recommendations); no effect size is invented.",
        })

    return scenarios
